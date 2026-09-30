import argparse
import networkx as nx

from GIP.heuristics import InspectionPostsolve
from GIP.solver_utils import IP_to_Group
from Utils.Readers import IRIS_reader, ExperimentPicker, SimInstanceIO

from gurobipy import Model, GRB, quicksum, GurobiError
from GIP.heuristics.InspectionHeuristic import TM_solver_groups_scipy
from GIP.solver_utils.SolutionValidation import validate_solution_groups

import os
from Analysis.gurobi_trace import AnytimeTrace
# import sys
# sys.path.append("/home/adir/PycharmProjects/SteinerTreeSolver/Simulator")

heuristic_freq = 10


def RunSolver(G, S, I, vertex_poi_vis, root, sure_edges=None, Experiment_name='', TimeLim=1000, out_path='', stats_out=None):
    """stats_out: optional dict, filled with final solver stats and the anytime
    (time, incumbent, bound) trace -- see Analysis.gurobi_trace."""
    m = Model("GIP_SCF")
    m.setParam('TimeLimit', TimeLim)
    if out_path != '':
        output_path_full = os.path.join(out_path, f"SCF_{Experiment_name}_TL-{TimeLim}.log")
        m.setParam('LogFile', output_path_full)

    D = G.to_directed()
    D_edges = list(D.edges())
    num_nodes = D.number_of_nodes()

    if sure_edges is None:
        sure_edges = []

    # 1. Binary Variables (Routing)
    lb = {e: (1.0 if e in sure_edges else 0.0) for e in D_edges}
    y = m.addVars(D_edges, vtype=GRB.BINARY, lb=lb, ub=1.0, name="y")

    # 2. Continuous Variables (Single Commodity Flow)
    # Represents "cargo" or "connectivity token" flowing from root
    f = m.addVars(D_edges, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    m.update()

    # --- Objective ---
    m.setObjective(quicksum(D[u][v]['weight'] * y[(u, v)] for u, v in D.edges()), GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Routing Constraints
    m.addConstr(quicksum(y[(root, v)] for _, v in D.out_edges(root)) >= 1, name='root_outflow')

    for id, v_g in S.items():
        m.addConstr(quicksum(y[u, v] for u, v in D.in_edges(v_g)) >= 1, name=f'group_inflow_{id}')

    for i in D.nodes():
        # Conservation of routing
        m.addConstr(quicksum(y[(u, v)] for u, v in D.in_edges(i)) ==
                    quicksum(y[(u, v)] for u, v in D.out_edges(i)), name=f'node_{i}_route_balance')

    # 2. Flow Connectivity Constraints
    for u, v in D_edges:
        m.addConstr(f[u, v] <= 2*(num_nodes - 1) * y[u, v], name=f'coupling_{u}_{v}')    # Longest tour - 2(n-1) (see lemma 2 in wafr24)

    # Flow Conservation:
    for i in D.nodes():
        if i == root:
            continue

        flow_in = quicksum(f[u, i] for u, _ in D.in_edges(i))
        flow_out = quicksum(f[i, v] for _, v in D.out_edges(i))
        visited = quicksum(y[u, i] for u, _ in D.in_edges(i))

        m.addConstr(flow_in - flow_out == visited, name=f'flow_balance_{i}')

    m.update()

    # ---------------------------------------------------------
    # OPTIMIZATION PRE-COMPUTATION & STRUCTURES
    # ---------------------------------------------------------
    m._G, m._D, m._S, m._r, m._I = G, D, S, root, I
    m._vertex_poi_vis = vertex_poi_vis
    m._x = y
    m._sure_edges = {e for e in sure_edges if e in y}
    m._unc_groups = None
    m._heuristic_counter = 0
    m._Glp = G.copy()

    # Pre-compute ordered lists for fast callback access
    m._vars_list = []
    m._index_to_edge = []

    # Only map the BINARY variables for the callback heuristics
    # The flow variables handle connectivity automatically, so heuristics just need to guide integer shapes
    for (u, v), var in y.items():
        m._vars_list.append(var)
        m._index_to_edge.append((u, v))

    # Pre-cache items list for injection
    m._x_items = list(y.items())

    # ---------------------------------------------------------

    # m.Params.LazyConstraints = 1
    _trace = AnytimeTrace()
    m.optimize(_trace.wrap(cut_heuristic_callback))
    if stats_out is not None:
        stats_out.update(_trace.summary(m), formulation='SCF')

    # m.optimize()

    return edges_from_model(m, y)


def candidate_violations(model, solution_edges):
    """Return reasons why a binary route cannot be completed by the SCF model."""
    selected = set(solution_edges)
    violations = []

    unknown = selected.difference(model._x.keys())
    if unknown:
        violations.append(f"{len(unknown)} edge(s) are not routing variables")

    missing_sure = model._sure_edges.difference(selected)
    if missing_sure:
        violations.append(f"{len(missing_sure)} required edge(s) are missing")

    for v in model._D.nodes():
        in_degree = sum((u, w) in selected for u, w in model._D.in_edges(v))
        out_degree = sum((u, w) in selected for u, w in model._D.out_edges(v))
        if in_degree != out_degree:
            violations.append(
                f"route imbalance at {v}: in={in_degree}, out={out_degree}"
            )

    if not any(u == model._r for u, _ in selected):
        violations.append("no selected edge leaves the root")

    for group_id, group_vertices in model._S.items():
        if not any((u, v) in selected for u, v in model._D.in_edges(group_vertices)):
            violations.append(f"group {group_id} has no selected in-edge")

    # In the SCF formulation every selected non-root component has positive
    # demand, so it must be reachable from the root.
    H = nx.DiGraph()
    H.add_nodes_from(model._D.nodes())
    H.add_edges_from(selected.intersection(model._x.keys()))
    reachable = nx.descendants(H, model._r) | {model._r}
    selected_vertices = {v for edge in selected for v in edge}
    unreachable = selected_vertices.difference(reachable)
    if unreachable:
        violations.append(f"{len(unreachable)} selected vertex/vertices are root-disconnected")

    return violations


def bidirected_tree_candidate(model, tree_edges):
    """Build a binary-feasible, root-connected route from the heuristic tree."""
    support_edges = set(tree_edges)

    # Forced arcs must be included. Connect them to the root as well; an
    # isolated balanced component is infeasible for single-commodity flow.
    for u, v in model._sure_edges:
        support_edges.add((u, v))
        try:
            path = nx.shortest_path(model._G, model._r, u, weight="weight")
        except nx.NetworkXNoPath:
            continue
        support_edges.update(zip(path[:-1], path[1:]))

    # If all groups are visible at the root, the inspection tree can be empty,
    # while the formulation still requires positive root out-flow.
    if not support_edges:
        root_neighbors = list(model._G.neighbors(model._r))
        if root_neighbors:
            v = min(
                root_neighbors,
                key=lambda w: model._G[model._r][w]["weight"],
            )
            support_edges.add((model._r, v))

    selected = set()
    for u, v in support_edges:
        if (u, v) in model._x:
            selected.add((u, v))
        if (v, u) in model._x:
            selected.add((v, u))

    return selected


def inject_suggested_solution(model, solution_edges, where):
    # A tour is a walk and may repeat arcs. The MILP variables are binary, so
    # objective value and feasibility must be computed from its unique support.
    selected = set(solution_edges)
    unknown = selected.difference(model._x.keys())
    if unknown:
        print(f"Primal heuristic skipped: {len(unknown)} unknown edge(s)")
        return

    x_items = model._x_items
    vars_list = [var for _, var in x_items]
    vals_list = [1.0 if edge in selected else 0.0 for edge, _ in x_items]
    cand_obj = sum(model._D[u][v]["weight"] for u, v in selected)


    # Get incumbent objective (minimization assumed)
    if where == GRB.Callback.MIPNODE:
        best_inc = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
    elif where == GRB.Callback.MIPSOL:
        best_inc = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
    else:
        return

    tol = 1e-6
    if best_inc < GRB.INFINITY and cand_obj >= best_inc - tol:
        return  # not improving

    print(f"Primal heuristic - {cand_obj=}")
    try:
        # Only y is specified. The continuous commodity-flow variables remain
        # undefined and Gurobi completes them for the connected route.
        model.cbSetSolution(vars_list, vals_list)

        if where == GRB.Callback.MIPNODE:
            obj = model.cbUseSolution()
            if obj == GRB.INFINITY:
                print("Primal heuristic was not accepted as a new incumbent")
            else:
                print(f"Primal heuristic accepted - objective={obj}")

    except GurobiError as error:
        print(f"Primal heuristic injection failed: {error}")


def cut_heuristic_callback(model, where):
    if where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_NODCNT) % 50 == 0:
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status != GRB.OPTIMAL:
                return

            nodecnt = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))

            all_vals = model.cbGetNodeRel(model._vars_list)
            lp = {model._index_to_edge[i]:val for i, val in enumerate(all_vals)}


            # --- Heuristic ---
            if model._heuristic_counter % heuristic_freq == 0:
                # Generate primal heuristic solution considering LP
                for u, v in model._Glp.edges():
                    model._Glp.edges[u, v]['weight'] = max(0, (1-max(lp[u, v], lp[v, u]))) * model._G.edges[u, v]['weight']

                tree_solution_edges, _ = TM_solver_groups_scipy(model._Glp, model._r, model._I.copy(), model._vertex_poi_vis)
                solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(model._G, tree_solution_edges,
                                                                                                     start=model._r)

                # Expanding a closed walk into binary arc support can destroy
                # route balance when an oriented arc occurs more than once.
                # Keep the tour when its support is feasible; otherwise use a
                # bidirected tree, which is balanced and root-connected.
                candidate = set(solution_edges)
                violations = candidate_violations(model, candidate)
                if violations:
                    candidate = bidirected_tree_candidate(model, tree_solution_edges)
                    fallback_violations = candidate_violations(model, candidate)
                    if fallback_violations:
                        print(
                            "Primal heuristic skipped: "
                            + "; ".join(fallback_violations[:5])
                        )
                        model._heuristic_counter = 1
                        return

                inject_suggested_solution(model, candidate, where)

                model._heuristic_counter = 1
            else:
                model._heuristic_counter += 1

            # # --- Cuts ---
            # suggested_cuts = CutsOracle.generate_group_flow_cuts_directed(model._D, model._S, model._r, lp=lp,
            #                                                               # groups_subset=model._unc_groups)
            #                                                               groups_subset=None)
            #
            # # suggested_cuts = CutsOracle.generate_group_flow_cuts_directed_guided_scipy(model._D, model._S, model._r, lp=lp,
            # #                                                               groups_subset=model._unc_groups)
            #                                                               # groups_subset=None)
            #
            # for cut_edges in suggested_cuts:
            #     if len(cut_edges) > 0:    # Protection from 0 >= 1 situation
            #         model.cbCut(quicksum(model._x[e] for e in cut_edges) >= 1)

    #
    # if where == GRB.Callback.MIPSOL:
    #     tol = 1e-4
    #     x = model._x
    #
    #     xval = {e: model.cbGetSolution(var) for e, var in x.items()}
    #     selected = {e for e, v in xval.items() if v >= 1 - tol}
    #
    #     cutset, uncovered_groups = CutsOracle.directed_group_connectivity_cut(model._D, selected, model._S, model._r)
    #
    #     if cutset:
    #         model.cbLazy(quicksum(x[u, v] for u, v in cutset) >= 1)
    #
    #         # if model._heuristic_counter % heuristic_freq == 0:
    #         #     repaired_selected = CutsOracle.correction_heuristic(model._D, selected, model._S, model._r)
    #         #     inject_suggested_solution(model, repaired_selected, where)
    #
    #         # for g_cut in cutset:
    #         #     model.cbLazy(quicksum(x[u, v] for u, v in g_cut) >= 1)
    #
    #
    #     model._unc_groups = uncovered_groups


def edges_from_model(gb_model, dir_edge_to_var, eps=1e-4):
    xvals = gb_model.getAttr('X', dir_edge_to_var)  # dict: (u,v) -> value
    dir_edge_list = [e for e, x in xvals.items() if x >= 1 - eps]

    return dir_edge_list

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run Gurobi Experiment")
    # parser.add_argument("--experiment", type=str, default="Drone2000",
    #                     help="Name of the experiment to run (e.g., Crisp1000, Drone2000)")
    #
    # # Parse arguments
    # args = parser.parse_args()
    # Experiment = args.experiment

    # ---- IRIS instance ----
    # vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(Experiment)
    # G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)
    #
    # I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    # root = 0

    # ---- Load Simulated Experiment ----
    Experiment = r"/home/adir/PycharmProjects/BridgeInspectionSimulator/inspection_experiments/gip_instance_N998_K500_bridge.pkl"
    # Experiment = "instance_300x300_POIs-500_N-1000"
    #
    G, I, S, vertex_poi_vis, root, meta = (
        SimInstanceIO.load_simulated_instance(f"{Experiment}"))
    # SimInstanceIO.load_simulated_instance(f"/home/adir/Desktop/IP-results/simulated_experiments/{Experiment}.pkl"))

    for k, v in vertex_poi_vis.items():
        vertex_poi_vis[k] = set(v)

    # Filter unseen POIs
    unseen_poi = []
    for p, l in S.items():
        if len(l) == 0:
            unseen_poi.append(p)

    for p in unseen_poi: del S[p]
    I = set(S.keys())
    # ---- Solver ----
    tour_edges = RunSolver(G, S, set(I), vertex_poi_vis, root, sure_edges=[])

    tour_edges, tour_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(G, tour_edges, start=root)

    print(f"Tour: {tour_edges}")
    print(f"Tour Weight: {tour_weight}")

    validate_solution_groups(G, S, tour_edges)
