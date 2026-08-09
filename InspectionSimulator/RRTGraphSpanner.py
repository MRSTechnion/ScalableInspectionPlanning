from dataclasses import dataclass
import time

from InspectionSimulator.InspectionPlanningSimulation import compute_visibility, visualize_inspection_task_pybullet, run_pybullet_viewer
from RunInspectionOnModel import inspection_obj_config, load_object_env, sample_pois_on_mesh_surface
from InspectionSimulator.CollisionCheckSphere import sphere_collision_check, is_local_path_collision_free

import random

seed = 0
rng = random.Random(seed)

@dataclass
class RRTPlannerConfig:
    step_size: float

def init_tree(root):
    pass

def sample_tree_point(T):
    # Sample from free space
    # Find tree nearest neighbor
    pass

def sample_point_until_free(object_mesh, bounds, radius, tol=1e-9):
    x_rand = rng.uniform(bounds.xmin, bounds.xmax - tol)
    y_rand = rng.uniform(bounds.ymin, bounds.ymax - tol)
    z_rand = rng.uniform(bounds.zmin, bounds.zmax - tol)
    point = (x_rand, y_rand, z_rand)

    while sphere_collision_check(point, radius, object_mesh, bounds):
        x_rand = rng.uniform(bounds.xmin, bounds.xmax - tol)
        y_rand = rng.uniform(bounds.ymin, bounds.ymax - tol)
        z_rand = rng.uniform(bounds.zmin, bounds.zmax - tol)
        point = (x_rand, y_rand, z_rand)

    return point


def extend_in_direction():
    pass

def extend_tree(T, vis_set):
    pass

def build_RRT_motion_planning_graph(planner_config, obj_config, inspection_object, poi_set, root):
    T = init_tree(root)
    vis_set = compute_visibility(root)
    while poi_set.difference(vis_set) is not None:
        extend_tree(T, vis_set)

def planner_config():
    return RRTPlannerConfig(
        step_size = 3,
    )

if __name__ == '__main__':
    obj_config = inspection_obj_config(r'../config/water_tower.json')
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
    planning_artifacts = build_RRT_motion_planning_graph(planner_config, obj_config, inspection_object, poi_set, root)
    G = planning_artifacts.graph
    poi_to_vertices = planning_artifacts.poi_to_vertices
    vertex_to_pois = planning_artifacts.vertex_to_pois

    # --- Visualize task ---
    client_id, mesh_body = visualize_inspection_task_pybullet(
        object_mesh=object_mesh,
        poi_set=poi_set,
        G=G,
        start_node=None,
        solution_edges=None,
        visibility_vertex=None,
        vertex_to_pois=vertex_to_pois,
        show_graph_nodes=True,
        show_graph_edges=True,
        show_solution_visibility=False
    )

    run_pybullet_viewer(client_id)
    # ----------