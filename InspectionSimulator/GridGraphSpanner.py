from __future__ import annotations

from InspectionSimulator.InspectionPlanningSimulation import compute_visibility
from InspectionSimulator.SceneLoader import Bounds3D, ObstacleMesh, ArrayLike3
from InspectionSimulator.CollisionCheckSphere import sphere_collision_check, is_local_path_collision_free

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

GridPoint = Tuple[float, float, float]


@dataclass
class GridPlannerConfig:
    grid_resolution: list[float]
    connectivity: int = 6   # 6, 18, 26
    edge_sample_num: int = 10

@dataclass
class PlanningArtifacts:
    graph: nx.Graph
    poi_to_vertices: dict
    vertex_to_pois: dict


# -----------------------------
# Grid sampling
# -----------------------------

def sample_space_grid(bounds: Bounds3D, grid_resolution) -> np.ndarray:
    """Sample a regular 3D cubic grid over the workspace bounds."""
    xs = np.linspace(bounds.xmin, bounds.xmax, int(grid_resolution[0]))
    ys = np.linspace(bounds.ymin, bounds.ymax, int(grid_resolution[1]))
    zs = np.linspace(bounds.zmin, bounds.zmax, int(grid_resolution[2]))

    # xs = np.arange(bounds.xmin, bounds.xmax + 0.5 * grid_resolution, grid_resolution)
    # ys = np.arange(bounds.ymin, bounds.ymax + 0.5 * grid_resolution, grid_resolution)
    # zs = np.arange(bounds.zmin, bounds.zmax + 0.5 * grid_resolution, grid_resolution)

    grid = np.array(list(product(xs, ys, zs)), dtype=float)
    return grid


def find_free_grid_points(
    sampled_grid: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
) -> np.ndarray:
    """Return only collision-free grid points for a sphere robot."""
    free_points: List[np.ndarray] = []
    for point in sampled_grid:
        if not sphere_collision_check(point, radius, obstacle, bounds):
            free_points.append(point)
    if not free_points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(free_points)


# -----------------------------
# Local connectivity / edges
# -----------------------------

def generate_neighbor_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
    """Return 3D neighbor offsets for 6, 18, or 26 connectivity."""
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of {6, 18, 26}")

    offsets: List[Tuple[int, int, int]] = []
    for dx, dy, dz in product([-1, 0, 1], repeat=3):
        if dx == dy == dz == 0:
            continue
        non_zero = sum(v != 0 for v in (dx, dy, dz))
        if connectivity == 6 and non_zero == 1:
            offsets.append((dx, dy, dz))
        elif connectivity == 18 and non_zero <= 2:
            offsets.append((dx, dy, dz))
        elif connectivity == 26:
            offsets.append((dx, dy, dz))
    return offsets


def point_to_grid_key(point, grid_origin, grid_step):
    point = np.asarray(point, dtype=float)
    grid_origin = np.asarray(grid_origin, dtype=float)
    grid_step = np.asarray(grid_step, dtype=float)

    return tuple(
        np.rint((point - grid_origin) / grid_step)
        .astype(int)
        .tolist()
    )


def connect_free_grid_points(
    free_grid: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    grid_resolution,
    grid_step,
    connectivity: int,
    edge_sample_num: int,
) -> nx.Graph:
    """Create an undirected graph over free grid points with valid local edges."""
    graph = nx.Graph()

    neighbor_offsets = generate_neighbor_offsets(connectivity)

    grid_origin = np.array(
        [bounds.xmin, bounds.ymin, bounds.zmin],
        dtype=float,
    )

    key_to_point = {
        point_to_grid_key(point, grid_origin, grid_step): point
        for point in free_grid
    }


    point_to_idx = {}
    for idx, point in enumerate(free_grid):
        node = tuple(point.tolist())
        graph.add_node(idx, pos=node)
        point_to_idx[node] = idx

    for point in free_grid:
        point_key = point_to_grid_key(point, grid_origin, grid_step)
        node_u = tuple(point.tolist())

        for offset in neighbor_offsets:
            neighbor_key = (
                point_key[0] + offset[0],
                point_key[1] + offset[1],
                point_key[2] + offset[2],
            )
            neighbor_point = key_to_point.get(neighbor_key)
            if neighbor_point is None:
                continue

            node_v = tuple(neighbor_point.tolist())
            u = point_to_idx[tuple(point.tolist())]
            v = point_to_idx[tuple(neighbor_point.tolist())]

            if u == v or graph.has_edge(u, v):
                continue

            if is_local_path_collision_free(
                start=point,
                goal=neighbor_point,
                radius=radius,
                obstacle=obstacle,
                bounds=bounds,
                edge_sample_num=edge_sample_num,
            ):
                weight = float(np.linalg.norm(neighbor_point - point))
                graph.add_edge(point_to_idx[node_u], point_to_idx[node_v], weight=weight)

    return graph


# -----------------------------
# Pipeline manager
# -----------------------------
# @timing
def build_grid_motion_planning_graph(planner_config, scene_config, insp_object, poi_set, root) -> PlanningArtifacts:
    """Run the full grid-based planning pipeline and return all artifacts."""
    # TODO - adjust to work with a given root
    sampled_grid = sample_space_grid(scene_config.bounds, planner_config.grid_resolution)

    grid_step = np.zeros_like(planner_config.grid_resolution)
    for i in range(len(grid_step)):
        grid_values = np.unique(sampled_grid[:,i])
        grid_step[i] = grid_values[1]-grid_values[0]

    free_grid = find_free_grid_points(
        sampled_grid=sampled_grid,
        radius=scene_config.robot_radius,
        obstacle=insp_object,
        bounds=scene_config.bounds,
    )

    graph = connect_free_grid_points(
        free_grid=free_grid,
        radius=scene_config.robot_radius,
        obstacle=insp_object,
        bounds=scene_config.bounds,
        grid_resolution=planner_config.grid_resolution,
        grid_step=grid_step,
        connectivity=planner_config.connectivity,
        edge_sample_num=planner_config.edge_sample_num,
    )

    poi_to_vertices, vertex_to_pois = compute_visibility(
        graph=graph,
        poi_set=poi_set,
        object_mesh=insp_object.mesh,
        visibility_threshold=scene_config.visibility_threshold
    )

    return PlanningArtifacts(
        graph=graph,
        poi_to_vertices = poi_to_vertices,
        vertex_to_pois = vertex_to_pois
    )

def visualize_grid_sampling(
    artifacts: PlanningArtifacts,
    show_sampled_grid: bool = False,
    sampled_grid_stride: int = 10,
    node_radius: float = 0.03,
    edge_line_width: float = 1.0,
    obstacle_rgba: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    sampled_grid_rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    free_grid_rgb: Tuple[float, float, float] = (0.0, 0.8, 0.0),
    edge_rgb: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    camera_distance: Optional[float] = None,
    camera_yaw: float = 45.0,
    camera_pitch: float = -30.0,
) -> None:
    """Visualize the obstacle mesh, free nodes, and graph edges in PyBullet.

    Notes:
    - Requires `pybullet` and `pybullet_data`.
    - Debug rendering is best for small to medium graphs. For large graphs, increase
      `sampled_grid_stride` and consider visualizing only a subset of nodes/edges.
    """
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise ImportError(
            "visualize_motion_planning_graph requires pybullet. "
            "Install it with: pip install pybullet"
        ) from exc

    def _draw_cross(point: np.ndarray, color: Tuple[float, float, float], size: float) -> None:
        x, y, z = point.tolist()
        p.addUserDebugLine([x - size, y, z], [x + size, y, z], color, edge_line_width, 0)
        p.addUserDebugLine([x, y - size, z], [x, y + size, z], color, edge_line_width, 0)
        p.addUserDebugLine([x, y, z - size], [x, y, z + size], color, edge_line_width, 0)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.loadURDF("plane.urdf")

    mesh = artifacts.obstacle.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    # Export a temporary OBJ that matches the world-transformed mesh used by the planner.
    import tempfile
    import os

    tmp_dir = tempfile.mkdtemp(prefix="planner_vis_")
    tmp_obj_path = os.path.join(tmp_dir, "obstacle_world.obj")
    mesh.export(tmp_obj_path)

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=tmp_obj_path,
        meshScale=[1.0, 1.0, 1.0],
        rgbaColor=list(obstacle_rgba),
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=tmp_obj_path,
        meshScale=[1.0, 1.0, 1.0],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0.0, 0.0, 0.0],
    )

    # # Draw sampled grid, optionally strided for readability.
    # if show_sampled_grid and len(artifacts.sampled_grid) > 0:
    #     stride = max(1, int(sampled_grid_stride))
    #     for point in artifacts.sampled_grid[::stride]:
    #         _draw_cross(np.asarray(point, dtype=float), sampled_grid_rgb, size=node_radius * 0.5)

    # # Draw free graph nodes.
    # for point in artifacts.free_grid:
    #     _draw_cross(np.asarray(point, dtype=float), free_grid_rgb, size=node_radius)

    points = [list(point) for point in artifacts.free_grid]
    colors = [free_grid_rgb for _ in points]

    p.addUserDebugPoints(
        pointPositions=points,
        pointColorsRGB=colors,
        pointSize=5.0,
        lifeTime=0,
    )

    # Draw graph edges.
    for node_u, node_v in artifacts.graph.edges():
        p.addUserDebugLine(list(node_u), list(node_v), edge_rgb, edge_line_width, 0)

    # Draw world axes.
    axis_len = max(1.0, node_radius * 20.0)
    p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], 2.0, 0)
    p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], 2.0, 0)
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], 2.0, 0)

    aabb_min = vertices.min(axis=0)
    aabb_max = vertices.max(axis=0)
    center = ((aabb_min + aabb_max) / 2.0).tolist()
    size = (aabb_max - aabb_min).tolist()
    if camera_distance is None:
        camera_distance = max(2.0, 1.5 * max(size))

    p.resetDebugVisualizerCamera(
        cameraDistance=float(camera_distance),
        cameraYaw=float(camera_yaw),
        cameraPitch=float(camera_pitch),
        cameraTargetPosition=center,
    )

    print(f"PyBullet body id: {body_id}")
    print(f"Graph nodes: {artifacts.graph.number_of_nodes()}")
    print(f"Graph edges: {artifacts.graph.number_of_edges()}")
    print("Close the PyBullet window or press Ctrl+C in the terminal to exit.")

    try:
        while p.isConnected(client):
            p.stepSimulation()
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)
