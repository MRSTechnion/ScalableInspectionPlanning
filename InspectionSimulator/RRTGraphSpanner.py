import math
from dataclasses import dataclass
import time

from rasterio.crs import defaultdict

from InspectionSimulator.InspectionPlanningSimulation import compute_vertex_vis, visualize_inspection_task_pybullet, run_pybullet_viewer
from RunInspectionOnModel import inspection_obj_config, load_object_env, sample_pois_on_mesh_surface
from InspectionSimulator.CollisionCheckSphere import sphere_collision_check, is_local_path_collision_free

import random
from scipy.spatial import KDTree
import networkx as nx
import numpy as np


seed = 0
rng = random.Random(seed)

@dataclass
class RRTPlannerConfig:
    eta: float
    max_iterations: int
    rrg_max_r: float
    edge_sample_num: int = 5
    rrg_conn_k: int = 0

@dataclass
class PlanningArtifacts:
    graph: nx.Graph
    poi_to_vertices: dict
    vertex_to_pois: dict


def init_tree(root_cfg):
    G = nx.empty_graph()
    G.add_node(0, pos=root_cfg)
    T = KDTree([root_cfg])
    return G, T

def sample_tree_point(T, object_mesh, bounds, robot_radius, conn_radius=200):
    v_rand = sample_point_until_free(object_mesh, bounds, robot_radius)
    nearest_tree_configs = T.query(v_rand)

    d, i_near = nearest_tree_configs
    v_near = T.data[i_near]
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

    return np.array(point)

def extend_in_direction(v_near, v_rand, eta):
    d_vec = np.array(v_rand - v_near)
    d = math.hypot(*d_vec)
    step_prc = min(1, eta/d)

    v_new = v_near + step_prc*d_vec
    return v_new

def extend_tree(G, T, i, object_mesh, obj_config, planner_config):
    v_new = None

    while v_new is None:
        i_near, v_near, v_rand = sample_tree_point(T, object_mesh, obj_config.bounds, obj_config.robot_radius)
        v_new_cand = extend_in_direction(v_near, v_rand, planner_config.eta)

        if is_local_path_collision_free(v_near, v_new_cand,
                                        obj_config.robot_radius,
                                        object_mesh,
                                        obj_config.bounds,
                                        planner_config.edge_sample_num):
            v_new = v_new_cand
            G.add_node(i, pos=v_new)
            G.add_edge(i_near, i)

            # update T
            # TODO - improve using a buffer
            updated_points = np.vstack([
                np.asarray(T.data),
                np.asarray(v_new, dtype=float),
            ])
            T = KDTree(updated_points)

    return v_new, T

def augment_to_rrg(G, T, planner_config, obj_config, object_mesh):
    for i, v_pos in G.nodes(data='pos'):
        cand_vertices = T.query_ball_point(v_pos, planner_config.rrg_max_r)
        points_num = min(len(cand_vertices), planner_config.rrg_conn_k)
        for j in cand_vertices[:points_num]: #TODO - biased towards lower index? fix
            if i == j or G.has_edge(i,j):
                continue
            v_ext_pos = G.nodes()[j]['pos']
            if is_local_path_collision_free(v_pos, v_ext_pos,
                                            obj_config.robot_radius,
                                            object_mesh,
                                            obj_config.bounds,
                                            planner_config.edge_sample_num):
                G.add_edge(i, j)


def build_RRT_motion_planning_graph(planner_config, obj_config, object_mesh, poi_set, root):
    G, T = init_tree(root)
    i = 0

    visible_pois = compute_vertex_vis(root, poi_set, object_mesh, obj_config.visibility_threshold)
    all_pois = {poi.poi_id for poi in poi_set.pois}

    vertex_to_pois = defaultdict(set)
    poi_to_vertices = defaultdict(set)

    vertex_to_pois[root] = visible_pois

    for poi in visible_pois:
        poi_to_vertices[poi].add(root)

    visible_set = set(visible_pois)
    while not all_pois.issubset(visible_set) and i < planner_config.max_iterations:
        i += 1
        v_new, T = extend_tree(G, T, i, object_mesh, obj_config, planner_config)

        visible_pois = compute_vertex_vis(v_new, poi_set, object_mesh.mesh, obj_config.visibility_threshold)
        vertex_to_pois[i] = visible_pois

        for poi in visible_pois:
            poi_to_vertices[poi].add(i)

        visible_set.update(visible_pois)

    print(f"visibility ratio = {len(visible_set)/len(all_pois)}")

    if planner_config.rrg_conn_k > 0:
        augment_to_rrg(G, T, planner_config, obj_config, object_mesh)

    res = PlanningArtifacts(graph=G,
                             poi_to_vertices=poi_to_vertices,
                             vertex_to_pois=vertex_to_pois)

    return res


def planner_config():
    return RRTPlannerConfig(
        eta = 2,
        max_iterations = 1000,
        rrg_conn_k = 10,
        rrg_max_r = 5
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
    root = (-39, -29, 1)    # TODO - make a part of the config
    planning_artifacts = build_RRT_motion_planning_graph(planner_config, obj_config, inspection_object, poi_set, root)
    G = planning_artifacts.graph
    poi_to_vertices = planning_artifacts.poi_to_vertices
    vertex_to_pois = planning_artifacts.vertex_to_pois

    t3 = time.time()
    print(f"Took: {t3 - t2} seconds")

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