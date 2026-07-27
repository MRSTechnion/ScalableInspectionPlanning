import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from InspectionSimulator.GridGraphSpanner import Bounds3D, PlannerConfig, load_object_env
from InspectionSimulator.GridGraphSpanner import InspectionObjectConfig, GridPlannerConfig

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


def inspection_obj_config(obj_config_file) -> InspectionObjectConfig:
    with open(obj_config_file, 'r') as f:
        obj_config = json.load(f)
    return InspectionObjectConfig(
        obj_path=obj_config['obj_path'],
        bounds=Bounds3D(
            xmin=obj_config['bounds']['xmin'],
            xmax=obj_config['bounds']['xmax'],
            ymin=obj_config['bounds']['ymin'],
            ymax=obj_config['bounds']['ymax'],
            zmin=obj_config['bounds']['zmin'],
            zmax=obj_config['bounds']['zmax'],
        ),
        scale=obj_config['scale'],
        translation=obj_config['translation'],
        rotation_rpy=obj_config['rotation'],
    )

def planner_config() -> GridPlannerConfig:
    return GridPlannerConfig(
        robot_radius=5,
        grid_resolution= np.asarray([10, 10, 10], dtype=float),
        connectivity=6,
        edge_sample_num=3
    )

if __name__ == '__main__':
    obj_config = inspection_obj_config(r'./config/water_tower.json')
    planner_config = planner_config()

    t0 = time.time()
    print("--- Loading object model ---")
    inspection_object = load_object_env(
        obj_path=obj_config.obj_path,
        scale=obj_config.scale,
        translation=obj_config.translation,
        rotation_rpy=obj_config.rotation_rpy,
    )

    bounds = obj_config.bounds
    object_mesh = inspection_object.mesh
    seed = 0

    t1 = time.time()
    print(f"Took: {t1-t0} seconds")

    print("--- Sampling Points of Interest ---")
    num_pois = 100
    poi_set = sample_pois_on_mesh_surface(object_mesh=object_mesh, num_pois=num_pois, seed=seed)

    t2 = time.time()
    print(f"Took: {t2 - t1} seconds")

    print("--- Building motion planning graph ---")

    planning_artifacts = build_grid_motion_planning_graph(planner_config, inspection_object, bounds)
    G = planning_artifacts.graph
    visibility_threshold_dist = 15
    poi_to_vertices, vertex_to_pois, _ = compute_visibility_by_distance(G, poi_set, visibility_threshold_dist)

    uninspectable = [p for p in poi_to_vertices.keys() if len(poi_to_vertices[p]) == 0]
    I = set(poi_to_vertices.keys()).difference(uninspectable)
    print(f"un-inspectable: {uninspectable}; Remained to inspect: {len(I)}")

    t3 = time.time()
    print(f"Took: {t3 - t2} seconds")
    # visualize_inspection_task(object_mesh, poi_set, G, show_graph_edges=True, show_graph_nodes=True)
    # plt.show()

    print("--- Solving GIP problem ---")

    root = list(G.nodes())[0]
    timeout = 10

    solver = solver_entry['SCF']
    tour_edges = solver(G, poi_to_vertices, I, vertex_to_pois, root, sure_edges=[], Experiment_name='water_tower_100',
                            TimeLim=timeout, out_path='')
    print(tour_edges)

    visualize_inspection_task(object_mesh, poi_set, G, solution_edges=tour_edges)
    plt.show()

    pass