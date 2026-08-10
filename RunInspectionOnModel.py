import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from InspectionSimulator.SceneLoader import Bounds3D, load_object_env, InspectionObjectConfig
from InspectionSimulator.GridGraphSpanner import GridPlannerConfig, build_grid_motion_planning_graph

from InspectionSimulator.InspectionPlanningSimulation import (
    sample_pois_on_mesh_surface,
    visualize_inspection_task_pybullet,
    run_pybullet_viewer
)

from GIP.solvers import (GroupCutsetFormulationMILP, ChargeFormulationMILP, MultiCommodityFlowFormulationMILP,
                         SingleCommodityFlowFormulationMILP)

solver_entry = {"GroupCutset": GroupCutsetFormulationMILP.RunSolver,
                "Charge": ChargeFormulationMILP.RunSolver,
                "MCF": MultiCommodityFlowFormulationMILP.RunSolver,
                "SCF": SingleCommodityFlowFormulationMILP.RunSolver}


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
        visibility_threshold=obj_config['visibility_threshold'],
        robot_radius=obj_config['robot_radius'],
        edge_CC_samples=obj_config['edge_CC_samples']
    )

def planner_config() -> GridPlannerConfig:
    return GridPlannerConfig(
        grid_resolution= np.asarray([10, 10, 10], dtype=float),
        connectivity=6,
        edge_sample_num=3
    )

if __name__ == '__main__':
    obj_config = inspection_obj_config(r'./config/water_tower.json')
    planner_config = planner_config()

    t0 = time.time()
    print("--- Loading object model ---")
    inspection_object = load_object_env(obj_path=obj_config.obj_path, scale=obj_config.scale,
                                        translation=obj_config.translation, rotation_rpy=obj_config.rotation_rpy)

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
    root = 0
    planning_artifacts = build_grid_motion_planning_graph(planner_config, obj_config, inspection_object, poi_set, root)
    G = planning_artifacts.graph
    poi_to_vertices = planning_artifacts.poi_to_vertices
    vertex_to_pois = planning_artifacts.vertex_to_pois

    # --- Visualize task ---
    # client_id, mesh_body = visualize_inspection_task_pybullet(
    #     object_mesh=object_mesh,
    #     poi_set=poi_set,
    #     G=G,
    #     start_node=None,
    #     solution_edges=None,
    #     visibility_vertex=None,
    #     vertex_to_pois=None,
    #     show_graph_nodes=True,
    #     show_graph_edges=True,
    #     show_solution_visibility=False
    # )
    #
    # run_pybullet_viewer(client_id)
    # ----------


    uninspectable = [p for p in poi_to_vertices.keys() if len(poi_to_vertices[p]) == 0]
    I = set(poi_to_vertices.keys()).difference(uninspectable)
    for u in uninspectable:
        poi_to_vertices.pop(u)
    print(f"un-inspectable: {uninspectable}; Remained to inspect: {len(I)}")

    t3 = time.time()
    print(f"Took: {t3 - t2} seconds")

    # visualize_inspection_task(object_mesh, poi_set, G, show_graph_edges=True, show_graph_nodes=True)
    # plt.show()

    print("--- Solving GIP problem ---")

    timeout = 10

    solver = solver_entry['GroupCutset']
    tour_edges = solver(G, poi_to_vertices, I, vertex_to_pois, root, sure_edges=[], Experiment_name='water_tower_100',
                            TimeLim=timeout, out_path='')
    print(tour_edges)

    # --- Visualize solution ---
    client_id, mesh_body = visualize_inspection_task_pybullet(
        object_mesh=object_mesh,
        poi_set=poi_set,
        G=G,
        start_node=root,
        solution_edges=tour_edges,
        visibility_vertex=None,
        vertex_to_pois=vertex_to_pois,
        show_graph_nodes=True,
        show_graph_edges=True,
        show_solution_visibility=True
    )

    run_pybullet_viewer(client_id)
    # ----------

    pass