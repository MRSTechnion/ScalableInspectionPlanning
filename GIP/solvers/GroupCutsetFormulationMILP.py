import argparse
from GIP.heuristics import InspectionPostsolve
from GIP.solver_utils import IP_to_Group
from Utils.Readers import IRIS_reader, ExperimentPicker
from gurobipy import Model, GRB, quicksum, GurobiError
from GIP.heuristics.InspectionHeuristic import TM_solver_groups_scipy
from GIP.solver_utils.SolutionValidation import validate_solution_groups
from GIP.seperation import CutsOracle
import os
# import sys
# sys.path.append("/home/adir/PycharmProjects/SteinerTreeSolver/Simulator")

heuristic_freq = 10
# TimeLim = 1000

greedy_PH = False
use_nested_cuts = False
use_creep_flow = False
max_groups_per_iteration = 250

def RunSolver(G, S, I, vertex_poi_vis, root, sure_edges=None, Experiment_name='', TimeLim=1000, out_path=''):
    m = Model("GroupCutset")
    m.setParam('TimeLimit', TimeLim)
    if out_path != '':
        output_path_full = os.path.join(out_path, f"Cutset_{Experiment_name}_TL-{TimeLim}.log")
        m.setParam('LogFile', output_path_full)

    D = G.to_directed()
    D_edges = list(D.edges())
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
    m._unc_groups = None
    m._heuristic_counter = 0
    m._Glp = G.copy()

    m._vars_list = []  # List of Gurobi Var objects
    m._index_to_edge = []  # List of (u,v) tuples matching the order above

    for edge, var in m._x.items():
        m._vars_list.append(var)
        m._index_to_edge.append(edge)

    #___

    m.Params.LazyConstraints = 1
    m.Params.PreCrush = 1
    m.optimize(cut_heuristic_callback)

    return edges_from_model(m, dir_edge_to_var)



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


    # --- Explain infeasibility of heuristic, if encountered ---
    # vars_ = model.getVars()
    # cand_vec = [cand_sol[e] for e in model._x.keys()]
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
                model._Glp.edges[u, v]['weight'] = max(0, (1-max(lp[u, v], lp[v, u]))) * model._G.edges[u, v]['weight']


            tree_solution_edges, _ = TM_solver_groups_scipy(model._Glp, model._r, model._I.copy(), model._vertex_poi_vis)

            if not greedy_PH:
                solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(model._G,
                                                                                                     tree_solution_edges,
                                                                                                     start=model._r)

            if greedy_PH:
                solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy_greedy(model._G,
                                                                                                            tree_solution_edges,
                                                                                                            start=model._r)

            inject_suggested_solution(model, solution_edges, where)

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
    parser = argparse.ArgumentParser(description="Run Gurobi Experiment")
    parser.add_argument("--experiment", type=str, default="Drone1000",
                        help="Name of the experiment to run (e.g., Crisp1000, Drone2000)")

    # Parse arguments
    args = parser.parse_args()
    Experiment = args.experiment

    # ----- Simulated Instance -----
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"{Experiment}"))

    # # ---- IRIS instance ----
    vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(Experiment)
    G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)

    I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    root = 0

    # ---- Load Simulated Experiment ----
    # Experiment = "instance_300x300_POIs-500_N-1000"
    #
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"/home/adir/Desktop/IP-results/simulated_experiments/{Experiment}.pkl"))

    # ---- Solver ----
    tour_edges = RunSolver(G, S, set(I), vertex_poi_vis, root, sure_edges=[])

    tour_edges, tour_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(G, tour_edges, start=root)

    print(f"Tour: {tour_edges}")
    print(f"Tour Weight: {tour_weight}")

    validate_solution_groups(G, S, tour_edges)
