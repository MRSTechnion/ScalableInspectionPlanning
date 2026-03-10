from GIP import heuristics
from GIP.solver_utils import IP_to_Group, SolutionValidation
import argparse
from GIP.solvers import (GroupCutsetFormulationMILP, ChargeFormulationMILP, MultiCommodityFlowFormulationMILP,
                         SingleCommodityFlowFormulationMILP)
from Utils.Readers import ExperimentPicker, IRIS_reader

solver_entry = {"GroupCutset": GroupCutsetFormulationMILP.RunSolver,
                "Charge": ChargeFormulationMILP.RunSolver,
                "MCF": MultiCommodityFlowFormulationMILP.RunSolver,
                "SCF": SingleCommodityFlowFormulationMILP.RunSolver
                }

def main():
    parser = argparse.ArgumentParser(
        prog='',
        description='Graph Inspection planning solver runner.',
        epilog='')

    parser.add_argument("--solver", '-s', type=str, default="GroupCutset",
                        help="Name of the solver to use: (GroupCutset, Charge, SCF, MCF, Heuristic-perfect, Heuristic-greedy)")
    parser.add_argument("--experiment", '-e', type=str, default="Bridge1000",
                        help="Name of the experiment to run (e.g., Crisp1000, Bridge2000)")
    parser.add_argument("--OutputPath", '-o', type=str, default="./",
                        help="Path to write logs")
    parser.add_argument('--timeout', '-t', type=int, default=500)

    args = parser.parse_args()
    print(args.solver, args.experiment, args.timeout)

    if args.experiment in ['Crisp1000', 'Crisp2000', 'Bridge1000', 'Bridge2000']:
        vertex_file, edge_file, conf_file = ExperimentPicker.pick_exp(args.experiment)
        G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)

        I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
        root = 0

    # TODO - support simulations
    else:
        print("Error - experiment not found.")
        return

    SolverCall = solver_entry[args.solver]
    tour_edges = SolverCall(G, S, set(I), vertex_poi_vis, root, sure_edges=[], Experiment_name=args.experiment,
                            TimeLim=args.timeout, out_path=args.OutputPath)
    tour_edges, tour_weight, _, _ = heuristics.InspectionPostsolve.ST_to_tour_christofides_scipy(G, tour_edges,
                                                                                                 start=root)
    SolutionValidation.validate_solution_groups(G, S, tour_edges)

    print(f"Tour: {tour_edges}")
    # print(f"Tour Weight: {tour_weight}")


if __name__ == '__main__':
    main()


