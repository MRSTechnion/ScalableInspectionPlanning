from dataclasses import dataclass
import time

from InspectionSimulator.InspectionPlanningSimulation import compute_visibility, visualize_inspection_task_pybullet, run_pybullet_viewer
from RunInspectionOnModel import inspection_obj_config, load_object_env, sample_pois_on_mesh_surface
from InspectionSimulator.CollisionCheckSphere import sphere_collision_check, is_local_path_collision_free

import random
from scipy.spatial import KDTree
import networkx as nx


seed = 0
rng = random.Random(seed)

@dataclass
class RRTPlannerConfig:
    eta: float
    max_iterations: int

def init_tree(root_cfg):
    G = nx.empty_graph()
    G.add_node(0, pos=root_cfg)
    T = KDTree([root_cfg])
    return G, T

def sample_tree_point(T, object_mesh, bounds, robot_radius, conn_radius=200):
    v_rand = sample_point_until_free(object_mesh, bounds, robot_radius)
    nearest_tree_configs = T.query_ball_point(v_rand)

    i_near = nearest_tree_configs[0]
    v_near = T[i_near]
    return i_near, v_near, v_rand

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


def extend_in_direction(v_near, v_rand, eta):
    pass

def extend_tree(G, T, i, object_mesh, bounds, robot_radius, eta):
    i_near, v_near, v_rand = sample_tree_point(T, object_mesh, bounds, robot_radius)
    v_new = extend_in_direction(v_near, v_rand, eta)

    G.add_node(i, pos=v_new)
    G.add_edge((i_near, i))

def build_RRT_motion_planning_graph(planner_config, obj_config, object_mesh, poi_set, root):
    G, T = init_tree(root)
    v_sample = sample_tree_point(T, object_mesh, obj_config.bounds, obj_config.robot_radius, conn_radius=5)

    visible_set = compute_visibility(root)
    i = 0
    while not poi_set.issubset(visible_set) and i < planner_config.max_iterations:
        i += 1
        extend_tree(T, visible_set)
        for v in T:
            visible_set.add(compute_visibility(v))



def planner_config():
    return RRTPlannerConfig(
        eta = 3,
        max_iterations = 200
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
    root = (-40, -30, 0)    # TODO - make a part of the config
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