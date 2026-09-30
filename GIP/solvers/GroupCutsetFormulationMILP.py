import argparse
import networkx as nx
from GIP.heuristics import InspectionPostsolve
from GIP.solver_utils import IP_to_Group
from Utils.Readers import IRIS_reader, ExperimentPicker, SimInstanceIO
from gurobipy import Model, GRB, quicksum, GurobiError
from GIP.heuristics.InspectionHeuristic import TM_solver_groups_scipy
from GIP.solver_utils.SolutionValidation import validate_solution_groups
from GIP.seperation import CutsOracle

import os
from Analysis.gurobi_trace import AnytimeTrace

heuristic_freq = 10
# TimeLim = 1000

greedy_PH = False
use_nested_cuts = False
use_creep_flow = False
max_groups_per_iteration = 250

def RunSolver(G, S, I, vertex_poi_vis, root, sure_edges=None, Experiment_name='', TimeLim=1000, out_path='', stats_out=None):
    """stats_out: optional dict, filled with final solver stats and the anytime
    (time, incumbent, bound) trace -- see Analysis.gurobi_trace."""
    m = Model("GroupCutset")
    m.setParam('TimeLimit', TimeLim)
    if out_path != '':
        output_path_full = os.path.join(out_path, f"Cutset_{Experiment_name}_TL-{TimeLim}.log")
        m.setParam('LogFile', output_path_full)

    D = G.to_directed()
    D_edges = list(D.edges())

    if sure_edges is None:
        sure_edges = []

    lb = {e: (1.0 if e in sure_edges else 0.0) for e in D_edges}  # pin sure edges to 1

    dir_edge_to_var = m.addVars(D_edges, vtype=GRB.BINARY, lb=lb, ub=1.0, name="x")
    m.update()

    # --- Objective ---
    # Pay for both direction - encourage to converge to single direction
    m.setObjective(quicksum(D[u][v]['weight'] * dir_edge_to_var[(u, v)] for u, v in D.edges()), GRB.MINIMIZE)
    m.update()

    # --- Constraints ---

    # Root out-flow
    m.addConstr(quicksum(dir_edge_to_var[(root, v)] for _, v in D.out_edges(root)) >= 1, name='root_outflow')

    # Groups in-flow
    for id, v_g in S.items():
        m.addConstr(quicksum(dir_edge_to_var[u, v] for u, v in D.in_edges(v_g)) >= 1, name=f'group_inflow_{id}')

    # Each node flow - if flow went in, it must leave - forcing path back to root
    for i in D.nodes():
        m.addConstr(quicksum(dir_edge_to_var[(u, v)] for u, v in D.in_edges(i)) ==
                    quicksum(dir_edge_to_var[(u, v)] for u, v in D.out_edges(i)), name=f'node_{i}_flow')


    m.update()

    #___
    m._G, m._D, m._S, m._r, m._I = G, D, S, root, I
    m._vertex_poi_vis = vertex_poi_vis
    m._x = dir_edge_to_var
    m._sure_edges = {e for e in sure_edges if e in dir_edge_to_var}
    m._unc_groups = None
    m._heuristic_counter = 0
    m._Glp = G.copy()

    m._vars_list = []  # List of Gurobi Var objects
    m._index_to_edge = []  # List of (u,v) tuples matching the order above

    for edge, var in m._x.items():
        m._vars_list.append(var)
        m._index_to_edge.append(edge)

    m._x_items = list(m._x.items())

    #___

    m.Params.LazyConstraints = 1
    m.Params.PreCrush = 1
    _trace = AnytimeTrace()
    m.optimize(_trace.wrap(cut_heuristic_callback))
    if stats_out is not None:
        stats_out.update(_trace.summary(m), formulation='GroupCutset')

    return edges_from_model(m, dir_edge_to_var)



def candidate_violations(model, solution_edges):
    """Return violations of the binary cutset formulation."""
    selected = set(solution_edges)
    violations = []

    unknown = selected.difference(model._x.keys())
    if unknown:
        violations.append(f"{len(unknown)} edge(s) are not model variables")

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

    # Check the lazy group-connectivity constraints before submitting the point.
    H = nx.DiGraph()
    H.add_nodes_from(model._D.nodes())
    H.add_edges_from(selected.intersection(model._x.keys()))
    reachable = nx.descendants(H, model._r) | {model._r}
    disconnected_groups = [
        group_id
        for group_id, vertices in model._S.items()
        if not reachable.intersection(vertices)
    ]
    if disconnected_groups:
        violations.append(f"{len(disconnected_groups)} group(s) are root-disconnected")

    return violations


def bidirected_tree_candidate(model, tree_edges):
    """Convert the undirected heuristic tree to a feasible binary route."""
    support_edges = set(tree_edges)
    support_edges.update(model._sure_edges)

    # The model requires a nonempty root cycle even when all groups are visible
    # at the root or only root-disconnected forced edges were supplied.
    if not any(model._r in edge for edge in support_edges):
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
    # The heuristic returns a walk, while the model uses binary arc variables.
    # Work with its unique support for both the values and the objective.
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
        # if model.cbGet(GRB.Callback.MIPNODE_NODCNT) % 100 == 0:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return

        nodecnt = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))

        all_vals = model.cbGetNodeRel(model._vars_list)
        lp = {model._index_to_edge[i]:val for i, val in enumerate(all_vals)}

        #
        # # --- Heuristic ---
        if model._heuristic_counter % heuristic_freq == 0:
            # Generate primal heuristic solution considering LP
            for u, v in model._Glp.edges():
                edge_lp_weight = 0
                if (u, v) in lp:
                    edge_lp_weight = lp[u, v]
                if (v, u) in lp:
                    edge_lp_weight = max(edge_lp_weight, lp[v, u])

                model._Glp.edges[u, v]['weight'] = max(0, (1-edge_lp_weight)) * model._G.edges[u, v]['weight']


            tree_solution_edges, _ = TM_solver_groups_scipy(model._Glp, model._r, model._I.copy(), model._vertex_poi_vis)

            if not greedy_PH:
                solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(model._G,
                                                                                                     tree_solution_edges,
                                                                                                     start=model._r)

            if greedy_PH:
                solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy_greedy(model._G,
                                                                                                            tree_solution_edges,
                                                                                                            start=model._r)

            # Repeated directed arcs in a closed walk disappear when projected
            # onto binary variables and can thereby destroy flow balance.
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
                else:
                    inject_suggested_solution(model, candidate, where)
            else:
                inject_suggested_solution(model, candidate, where)

            model._heuristic_counter = 1
        else:
            model._heuristic_counter += 1

        # --- Cuts ---
        suggested_cuts = CutsOracle.generate_group_flow_cuts_directed(model._D, model._S, model._r, lp=lp,
                                                                      groups_subset=None, use_nested_cuts=use_nested_cuts,
                                                                      use_creep_flow=use_creep_flow,
                                                                      max_groups_per_iteration=max_groups_per_iteration)
        # groups_subset=model._unc_groups)

        for cut_edges in suggested_cuts:
            if len(cut_edges) > 0:    # Protection from 0 >= 1 situation
                model.cbCut(quicksum(model._x[e] for e in cut_edges) >= 1)


    if where == GRB.Callback.MIPSOL:
        tol = 1e-4
        x = model._x

        xval = {e: model.cbGetSolution(var) for e, var in x.items()}
        selected = {e for e, v in xval.items() if v >= 1 - tol}

        cutset, uncovered_groups = CutsOracle.directed_group_connectivity_cut(model._D, selected, model._S, model._r)

        if cutset:
            model.cbLazy(quicksum(x[u, v] for u, v in cutset) >= 1)


        model._unc_groups = uncovered_groups

        # all_vals = model.cbGetSolution(model._vars_list)
        # lp = {model._index_to_edge[i]: val for i, val in enumerate(all_vals)}
        # suggested_cuts = CutsOracle.generate_group_flow_cuts_directed(model._D, model._S, model._r, lp=lp,
        #                                                               groups_subset=None,
        #                                                               use_nested_cuts=use_nested_cuts,
        #                                                               use_creep_flow=use_creep_flow,
        #                                                               max_groups_per_iteration=max_groups_per_iteration)
        # for cut_edges in suggested_cuts:
        #     if len(cut_edges) > 0:    # Protection from 0 >= 1 situation
        #         model.cbLazy(quicksum(model._x[e] for e in cut_edges) >= 1)

def edges_from_model(gb_model, dir_edge_to_var, eps=1e-4):
    xvals = gb_model.getAttr('X', dir_edge_to_var)  # dict: (u,v) -> value
    dir_edge_list = [e for e, x in xvals.items() if x >= 1 - eps]

    return dir_edge_list

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run Gurobi Experiment")
    # parser.add_argument("--experiment", type=str, default="Drone1000",
    #                     help="Name of the experiment to run (e.g., Crisp1000, Drone2000)")
    #
    # # Parse arguments
    # args = parser.parse_args()
    # Experiment = args.experiment


    # # ---- IRIS instance ----
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
