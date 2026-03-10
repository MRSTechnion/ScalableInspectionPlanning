import argparse

from GIP.heuristics import InspectionPostsolve
from GIP.solver_utils import IP_to_Group
from Utils.Readers import IRIS_reader, ExperimentPicker

from gurobipy import Model, GRB, quicksum, GurobiError
from GIP.heuristics.InspectionHeuristic import TM_solver_groups_scipy
from GIP.solver_utils.SolutionValidation import validate_solution_groups

import os
# import sys
# sys.path.append("/home/adir/PycharmProjects/SteinerTreeSolver/Simulator")

heuristic_freq = 10


def RunSolver(G, S, I, vertex_poi_vis, root, sure_edges=None, Experiment_name='', TimeLim=1000, out_path=''):
    m = Model("GIP_SCF")
    m.setParam('TimeLimit', TimeLim)
    if out_path != '':
        output_path_full = os.path.join(out_path, f"Cutset_{Experiment_name}_TL-{TimeLim}.log")
        m.setParam('LogFile', output_path_full)

    D = G.to_directed()
    D_edges = list(D.edges())
    num_nodes = D.number_of_nodes()

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

    # 1. Routing Constraints (Same as before)
    m.addConstr(quicksum(y[(root, v)] for _, v in D.out_edges(root)) >= 1, name='root_outflow')

    for id, v_g in S.items():
        m.addConstr(quicksum(y[u, v] for u, v in D.in_edges(v_g)) >= 1, name=f'group_inflow_{id}')

    for i in D.nodes():
        # Conservation of routing (Enter == Leave)
        m.addConstr(quicksum(y[(u, v)] for u, v in D.in_edges(i)) ==
                    quicksum(y[(u, v)] for u, v in D.out_edges(i)), name=f'node_{i}_route_balance')

    # 2. Flow Connectivity Constraints (New!)

    for u, v in D_edges:
        m.addConstr(f[u, v] <= 2*(num_nodes - 1) * y[u, v], name=f'coupling_{u}_{v}')    # Longest tour - 2(n-1) (see lemma 2 in wafr24)

    # Flow Conservation:
    for i in D.nodes():
        if i == root:
            continue  # Root is the source, we don't constrain its net flow (it supplies everything)

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
    m._x = y  # Callback uses 'y' logic (binary vars)
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
    m.optimize(cut_heuristic_callback)

    # m.optimize()

    return edges_from_model(m, y)


def inject_suggested_solution(model, solution_edges, where):
    # Freeze consistent ordering once per call
    x_items = list(model._x.items())  # [((u,v), var), ...]
    vars_list = [var for key, var in x_items]

    # Build candidate values keyed by edge
    cand_sol = {key: 0 for key, _ in x_items}
    cand_obj = 0.0
    for (u, v) in solution_edges:
        if (u, v) in cand_sol:
            cand_sol[(u, v)] = 1.0
            cand_obj += model._D[u][v]["weight"]
        else:
            # Edge not in var set: ignore or handle as error
            pass

    vals_list = [cand_sol[key] for key, _ in x_items]

    # print(f"{cand_obj=}")

    # --- Explain infeasibility of heuristic, if encountered ---
    # vars_ = model.getVars()
    # cand_vec = [cand_sol[e] for e in model._y.keys()]
    # x_by_name = {v.VarName: xv for v, xv in zip(vars_, cand_vec)}
    # GurobiUtils.explain_infeasibility_of_point(model, x_by_name)


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

        # Optional: only MIPNODE gives you useful immediate feedback from cbUseSolution
        if where == GRB.Callback.MIPNODE:
            obj = model.cbUseSolution()  # may return INFINITY if rejected
            # You can log obj if you want, but keep logging minimal

    except GurobiError:
        # keep it quiet in callbacks unless you are debugging
        pass


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

                inject_suggested_solution(model, solution_edges, where)

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
    parser = argparse.ArgumentParser(description="Run Gurobi Experiment")
    parser.add_argument("--experiment", type=str, default="Drone2000",
                        help="Name of the experiment to run (e.g., Crisp1000, Drone2000)")

    # Parse arguments
    args = parser.parse_args()
    Experiment = args.experiment

    # ----- Simulated Instance -----
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"{Experiment}"))

    #
    # ---- IRIS instance ----
    vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(Experiment)
    G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)

    I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    root = 0

    # ---- Load Simulated Experiment ----
    # Experiment = "instance_300x300_POIs-500_N-1000"
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"/home/adir/Desktop/IP-results/simulated_experiments/{Experiment}.pkl"))

    # ---- Solver ----
    tour_edges = RunSolver(G, S, set(I), vertex_poi_vis, root, sure_edges=[])

    tour_edges, tour_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(G, tour_edges, start=root)

    print(f"Tour: {tour_edges}")
    print(f"Tour Weight: {tour_weight}")

    validate_solution_groups(G, S, tour_edges)
