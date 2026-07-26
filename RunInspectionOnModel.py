import math
import matplotlib.pyplot as plt

from InspectionSimulator.GridGraphSpanner import Bounds3D, PlannerConfig, load_object_env
from InspectionSimulator.InspectionPlanningSimulation import (
    sample_pois_on_mesh_surface,
    build_grid_motion_planning_graph,
    compute_visibility_by_distance,
    visualize_inspection_task
)

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



def env_config() -> PlannerConfig:
    """Small example config; tune this to your bridge model placement."""
    return PlannerConfig(
        obj_path=r"./assets/OBJ/water_tower.obj",
        bounds=Bounds3D(
            xmin=-40.0,
            xmax=40.0,
            ymin=-30.5,
            ymax=30.5,
            zmin=0.0,
            zmax=2.5*30,
        ),
        robot_radius=0.15,
        grid_resolution=10,
        connectivity=18,
        edge_sample_num=3,
        scale=(0.05, 0.05, 0.05),
        translation=(0, 0.25, 1.5),
        # rotation_rpy=(0.0, 0.0, 0.0),
        rotation_rpy=[math.pi / 2, 0, 0],
    )
if __name__ == '__main__':
    config = env_config()
    inspection_object = load_object_env(
        obj_path=config.obj_path,
        scale=config.scale,
        translation=config.translation,
        rotation_rpy=config.rotation_rpy,
    )
    object_mesh = inspection_object.mesh
    seed = 0
    num_pois = 100
    poi_set = sample_pois_on_mesh_surface(object_mesh=object_mesh, num_pois=num_pois, seed=seed)

    # TODO - separate mp config from environment config
    planning_artifacts = build_grid_motion_planning_graph(config)
    G = planning_artifacts.graph

    visibility_threshold_dist = 15
    poi_to_vertices, vertex_to_pois, _ = compute_visibility_by_distance(G, poi_set, visibility_threshold_dist)

    I = set(poi_to_vertices.keys())
    root = list(G.nodes())[0]
    timeout = 5

    solver = solver_entry['SCF']
    tour_edges = solver(G, poi_to_vertices, I, vertex_to_pois, root, sure_edges=[], Experiment_name='water_tower_100',
                            TimeLim=timeout, out_path='')
    print(tour_edges)

    visualize_inspection_task(object_mesh, poi_set, G, solution_edges=tour_edges)
    plt.show()

    pass