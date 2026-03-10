import argparse


from GIP.heuristics import InspectionPostsolve

from gurobipy import Model, GRB, quicksum
from GIP.solver_utils.SolutionValidation import validate_solution_groups

from Utils.Readers.SimInstanceIO import load_simulated_instance
import os

# import sys
# sys.path.append("/home/adir/PycharmProjects/SteinerTreeSolver/Simulator")

heuristic_freq = 10

def RunSolver(G, S, I, vertex_poi_vis, root, sure_edges=None, Experiment_name='', TimeLim=1000, out_path=''):
    m = Model("GIP_Charge")
    m.setParam('TimeLimit', TimeLim)
    if out_path != '':
        output_path_full = os.path.join(out_path, f"Cutset_{Experiment_name}_TL-{TimeLim}.log")
        m.setParam('LogFile', output_path_full)

    D = G.to_directed()
    D_edges = list(D.edges())
    K = len(I)
    Ks = list(S.keys())

    # 1. Binary Variables (Routing)
    y = m.addVars(D_edges, vtype=GRB.BINARY, lb=0.0, ub=1.0, name="y")

    f = m.addVars(((k, u, v) for k in Ks for (u, v) in D_edges),
                  vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="f")

    # 2. Objective
    m.setObjective(quicksum(D[u][v]['weight'] * y[(u, v)] for u, v in D.edges()), GRB.MINIMIZE)

    m.update()

    for i in D.nodes():
        m.addConstr(quicksum(y[(u, v)] for u, v in D.in_edges(i)) ==
                    quicksum(y[(u, v)] for u, v in D.out_edges(i)), name=f'node_{i}_flow')

    for k in Ks:
        for u, v in D.edges():
            m.addConstr(f[(k, u, v)] <= y[(u, v)] , name='flow-capacity_connection')

    for k in Ks:
        if r in S[k]:
            continue

        # A. Source: Flow K must start at Root
        m.addConstr(quicksum(f[(k, root, v)] for _, v in D.out_edges(root)) == 1, name=f'src_{k}')

        # B. Transit: Flow conservation everywhere else
        for i in D.nodes():
            if i == root or i in S[k]:
                continue  # Handled by Source and Sink rules above

            # In == Out (Flow passes through)
            m.addConstr(quicksum(f[(k, u, v)] for u, v in D.in_edges(i)) ==
                        quicksum(f[(k, u, v)] for u, v in D.out_edges(i)), name=f'transit_{i}_{k}')

        # C. Consumption at group vertices
        m.addConstr(quicksum(f[(k, u, v)] for u, v in D.in_edges(S[k])) -
                    quicksum(f[(k, v, u)] for v, u in D.out_edges(S[k])) >= 1,
                    name=f'consumption_{k}')


        # Routing through group vertices - and back to root
        # for i in D.nodes():
        #     if i == r:
        #         continue
        #
        #     # In == Out (Flow passes through)
        #     m.addConstr(quicksum(f[(k, u, v)] for u, v in D.in_edges(i)) ==
        #                 quicksum(f[(k, u, v)] for u, v in D.out_edges(i)), name=f'transit_{i}_{k}')
        #
        # m.addConstr(quicksum(f[(k, u, v)] for u, v in D.in_edges(S[k])) >= 1,
        #             name=f'consumption_{k}')



    m.update()
    m.optimize()

    return edges_from_model(m, y)


def edges_from_model(gb_model, dir_edge_to_var, eps=1e-4):
    xvals = gb_model.getAttr('X', dir_edge_to_var)  # dict: (u,v) -> value
    dir_edge_list = [e for e, x in xvals.items() if x >= 1 - eps]
    print(dir_edge_list)
    return dir_edge_list

if __name__ == '__main__':
    # crisp_scale = 1

    parser = argparse.ArgumentParser(description="Run Gurobi Experiment")
    parser.add_argument("--experiment", type=str, default="Drone1000",
                        help="Name of the experiment to run (e.g., Crisp1000, Drone2000)")

    # Parse arguments
    args = parser.parse_args()
    Experiment = args.experiment

    # ----- Simulated Instance -----
    G, I, S, vertex_poi_vis, root, meta = (
        load_simulated_instance(f"{Experiment}"))


    # print(f"--- Running Experiment: {Experiment} ---")
    #
    # # ---- IRIS instance ----
    # vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(Experiment)
    # G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)
    #
    # if 'crisp' in vertex_file:
    #     # Scale up crisp weights to avoid numerical issues
    #     for u, v in G.edges():
    #         G[u][v]['weight'] *= crisp_scale
    #
    # I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    # root = 0

    # ---- Load Simulated Experiment ----
    # Experiment = "instance_100x100_POIs-100_N-500"
    # G, I, S, vertex_poi_vis, root, meta = (
    #     load_simulated_instance(f"/home/adir/Desktop/IP-results/simulated_experiments/{Experiment}.pkl"))

    # ---- Presolve ----

    sure_edges = []

    G_modified = G

    # ---- Solver ----
    tour_edges = RunSolver(G_modified, S, set(I), vertex_poi_vis, root, sure_edges=sure_edges)

    tour_edges, tour_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(G_modified, tour_edges, start=root)

    print(f"Tour: {tour_edges}")
    print(f"Tour Weight: {tour_weight}")

    validate_solution_groups(G, S, tour_edges)
