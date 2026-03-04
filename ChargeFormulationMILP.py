import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from collections import defaultdict
from gurobipy import Model, GRB, quicksum, GurobiError

# --- Imports from your project ---
import InspectionPresolve
import InspectionPostsolve
import Postsolve
import CutsOracle
from SolutionValidation import validate_solution, validate_solution_groups
from Utils import IP_to_Group
from Readers import ExperimentPicker, IRIS_reader
from InspectionHeuristic import TM_solver_groups_scipy
import GraphGeneration, GraphDrawing
import argparse

from Readers.SimInstanceUtils import load_simulated_instance
import sys
sys.path.append("/home/adir/PycharmProjects/SteinerTreeSolver/Simulator")


heuristic_freq = 10
TimeLim = 1000


def Solve_BnB_Charge(G, S, r, vertex_poi_vis, I, sure_edges=None):
    m = Model("GIP_Charge")

    m.setParam('TimeLimit', TimeLim)
    Experiment_name = Experiment.split("/")[-1].split(".")[0]

    # m.setParam('LogFile', f"/home/adir/Desktop/IP-results/grb_logs/Augmented_Charge_{Experiment_name}_TL{TimeLim}.log")
    m.setParam('LogFile', f"/home/adir/Desktop/IP-results/grb_logs_final/Charge_{Experiment_name}_TL{TimeLim}.log")

    # Directed arc variables x(u,v)
    x = {}
    for u, v in G.edges():
        # Create vars for both directions
        x[(u, v)] = m.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, name=f"x[{u},{v}]")
        x[(v, u)] = m.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, name=f"x[{v},{u}]")
    m.update()

    # Objective
    m.setObjective(
        quicksum(G[u][v]['weight'] * (x[(u, v)] + x[(v, u)]) for u, v in G.edges()),
        GRB.MINIMIZE
    )

    # Core Charge Constraints
    y = generate_charge_constraints(G, S, r, m, x)
    m.update()

    # --- AUGMENTATION: Setup Data for Callbacks ---
    m._G = G
    m._D = G.to_directed()  # CutsOracle needs directed graph
    m._S, m._r, m._I = S, r, I
    m._vertex_poi_vis = vertex_poi_vis
    m._x = x
    m._unc_groups = None
    m._heuristic_counter = 0
    m._Glp = G.copy()

    # Pre-compute lists for fast callback access
    m._vars_list = []
    m._index_to_edge = []
    m._x_items = list(x.items())

    for (u, v), var in x.items():
        m._vars_list.append(var)
        m._index_to_edge.append((u, v))

    # Solve with Callback
    # m.optimize(cut_heuristic_callback)
    m.optimize()

    # Extract result
    chosen = []
    # If optimization was successful/feasible
    if m.SolCount > 0:
        for u, v in G.edges():
            val_uv = x[(u, v)].X
            val_vu = x[(v, u)].X
            if val_uv + val_vu >= 1.0 - 1e-6:
                chosen.append((u, v))

    return chosen


def generate_charge_constraints(G, S, source, m, x):
    n = G.number_of_nodes()
    two_minus = 2.0 - 2.0 / (2 * n - 3)

    # 1. Flow Balance
    for v in G.nodes():
        in_v = quicksum(x[(u, v)] for u in G.neighbors(v) if (u, v) in x)
        out_v = quicksum(x[(v, u)] for u in G.neighbors(v) if (v, u) in x)

        if v == source:
            m.addConstr(out_v >= 1, name=f"flow_src_out_ge1[{v}]")
            m.addConstr(in_v == out_v, name=f"flow_src_bal[{v}]")
        else:
            m.addConstr(in_v == out_v, name=f"flow_bal[{v}]")

    # 2. Charge Variables & Linking
    y = {}

    def und(a, b):
        return (a, b) if a < b else (b, a)

    for ui, vj in G.edges():
        i, j = und(ui, vj)
        y[(i, j, i)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y[{i},{j},{i}]")
        y[(i, j, j)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y[{i},{j},{j}]")
    m.update()

    for ui, vj in G.edges():
        i, j = und(ui, vj)
        if i != source and j != source:
            m.addConstr(
                y[(i, j, i)] + y[(i, j, j)] == 2.0 * (x[(i, j)] + x[(j, i)]),
                name=f"charge_link[{i},{j}]"
            )

    # 3. Voltage Constraints
    for v in G.nodes():
        if v == source: continue
        expr_y = 0
        for u in G.neighbors(v):
            if u == source: continue
            a, b = und(u, v)
            expr_y += y[(a, b, v)]

        multiplicity = quicksum(x[(u, v)] for u in G.neighbors(v) if (u, v) in x)
        m.addConstr(expr_y <= multiplicity * two_minus, name=f"y_cap_v2[{v}]")

    # 4. Group Cover
    for c, nodes in S.items():
        expr = quicksum(x[(u, v)] for v in nodes if v in G
                        for u in G.neighbors(v) if (u, v) in x)
        m.addConstr(expr >= 1, name=f"group_cover[{c}]")

    return y


def inject_suggested_solution(model, solution_edges, where):
    if not solution_edges:
        return

    # 1. Build a temporary Multigraph from the heuristic edges
    # We use MultiGraph because a Group TSP tour might use the same link twice
    temp_G = nx.MultiGraph()
    temp_G.add_edges_from(solution_edges)

    # 2. Find the Eulerian Circuit (The valid Directed Tour)
    # This automatically orients edges A-B, B-C, C-A correctly
    try:
        # If the heuristic returns a valid cycle, this will order it u->v, v->w
        if not nx.is_eulerian(temp_G):
            # Heuristic didn't return a closed cycle/tour?
            # Attempt to make it eulerian or just return (heuristic failed)
            return

        directed_tour = list(nx.eulerian_circuit(temp_G, source=model._r))
    except nx.NetworkXError:
        return

    # 3. Map Directed Tour to Model Variables
    x_items = model._x_items
    vars_list = model._vars_list

    # Identify which directed vars need to be 1.0
    active_vars_indices = set()

    cand_obj = 0.0
    valid = True

    for u, v in directed_tour:
        # Find the specific directed variable x[u,v]
        if (u, v) in model._x:
            # We need the index of this var in vars_list
            # Since model._x and vars_list are synced, we can optimize this lookup
            # But for safety here, we assume we know the var
            var_obj = model._x[(u, v)]
            # We effectively need to map var_obj back to its index in vars_list
            # Optimization: Pre-compute {var_obj: index} in RunSolver if slow
            # For now, let's rebuild the dense vector carefully.
            pass
        else:
            valid = False
            break

        cand_obj += model._G[u][v]['weight']  # Charge usually uses G for weights

    if not valid:
        return

    # 4. Filter Incumbent
    if where == GRB.Callback.MIPNODE:
        best_inc = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
    elif where == GRB.Callback.MIPSOL:
        best_inc = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
    else:
        return

    tol = 1e-6
    if best_inc < GRB.INFINITY and cand_obj >= best_inc - tol:
        return

    print(f"Primal heuristic - {cand_obj=}")

    target_directed_edges = set(directed_tour)

    vals_list = []
    for (u, v), var in model._x.items():
        if (u, v) in target_directed_edges:
            vals_list.append(1.0)
        else:
            vals_list.append(0.0)

    # 6. Inject
    try:
        model.cbSetSolution(vars_list, vals_list)
        if where == GRB.Callback.MIPNODE:
            model.cbUseSolution()
    except GurobiError:
        pass


def cut_heuristic_callback(model, where):
    if where == GRB.Callback.MIPNODE:
        # if model.cbGet(GRB.Callback.MIPNODE_NODCNT) % 100 == 0:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return

        nodecnt = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))

        all_vals = model.cbGetNodeRel(model._vars_list)
        lp = {model._index_to_edge[i]: val for i, val in enumerate(all_vals)}

        #
        # # --- Heuristic ---
        if model._heuristic_counter % heuristic_freq == 0:
            # Generate primal heuristic solution considering LP
            for u, v in model._Glp.edges():
                model._Glp.edges[u, v]['weight'] = max(0, (1 - max(lp[u, v], lp[v, u]))) * model._G.edges[u, v][
                    'weight']

            tree_solution_edges, _ = TM_solver_groups_scipy(model._Glp, model._r, model._I.copy(),
                                                            model._vertex_poi_vis)

            solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(model._G,
                                                                                                 tree_solution_edges,
                                                                                                 start=root)

            # solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy_greedy(model._G,
            #                                                                                      tree_solution_edges,
            #                                                                                      start=root)

            # tree_solution_edges = wafr24_ST_heuristic(model._Glp, model._r, model._I.copy(), model._vertex_poi_vis)
            # solution_edges, _ = Postsolve.ST_to_tour(model._G,
            #                                          tree_solution_edges,
            #                                          start=root)
            #
            inject_suggested_solution(model, solution_edges, where)

            model._heuristic_counter = 1
        else:
            model._heuristic_counter += 1


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
    # # ---- IRIS instance ----
    vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(Experiment)
    G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)

    I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    root = 0

    # ---- Load Simulated Experiment ----
    # Experiment = "instance_500x500_POIs-1500_N-200"
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"/home/adir/Desktop/IP-results/simulated_experiments/{Experiment}.pkl"))


    # ---- Solve Augmented Charge ----
    tour_edges = Solve_BnB_Charge(G, S, root, vertex_poi_vis, set(I))

    # Post-processing usually not needed for exact solution, but useful if heuristic finished last
    tour_edges, tour_weight, _, _ = Postsolve.ST_to_tour_christofides(G, tour_edges, start=root)

    print(f"Final Tour Edges: {len(tour_edges)}")
    validate_solution_groups(G, S, tour_edges)