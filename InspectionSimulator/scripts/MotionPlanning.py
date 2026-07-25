from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import networkx as nx
import numpy as np
import math

from GridGraphSpanner import ArrayLike3, GridPoint, ObstacleMesh, Bounds3D, PlanningArtifacts, PlannerConfig
from GridGraphSpanner import sphere_collision_check, is_local_path_collision_free, build_grid_motion_planning_graph

@dataclass
class PathPlan:
    method: str
    start: GridPoint
    goal: GridPoint
    path: List[GridPoint]
    cost: float
    graph_with_queries: nx.Graph


# -----------------------------
# Query attachment / path planning
# -----------------------------

def euclidean_distance(a: ArrayLike3, b: ArrayLike3) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def is_state_valid(
    point: ArrayLike3,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
) -> bool:
    """Return True if the robot sphere centered at point is valid/free."""
    return not sphere_collision_check(point, radius, obstacle, bounds)


def find_connectable_neighbors(
    point: ArrayLike3,
    free_grid: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    max_candidates: int = 20,
    max_connections: int = 6,
    edge_sample_num: int = 10,
) -> List[Tuple[GridPoint, float]]:
    """Find nearby graph nodes that are collision-free to connect to from point."""
    if len(free_grid) == 0:
        return []

    query = np.asarray(point, dtype=float)
    deltas = free_grid - query[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    order = np.argsort(distances)

    neighbors: List[Tuple[GridPoint, float]] = []
    for idx in order[:max_candidates]:
        candidate = free_grid[idx]
        if is_local_path_collision_free(
            start=query,
            goal=candidate,
            radius=radius,
            obstacle=obstacle,
            bounds=bounds,
            edge_sample_num=edge_sample_num,
        ):
            neighbors.append((tuple(candidate.tolist()), float(distances[idx])))
            if len(neighbors) >= max_connections:
                break
    return neighbors


def attach_query_point(
    graph: nx.Graph,
    point: ArrayLike3,
    free_grid: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    max_candidates: int = 20,
    max_connections: int = 6,
    edge_sample_num: int = 10,
    node_prefix: str = "query",
) -> Tuple[nx.Graph, GridPoint]:
    """Add a query point to a graph and connect it to nearby reachable nodes."""
    point_tuple = tuple(np.asarray(point, dtype=float).tolist())

    if not is_state_valid(point_tuple, radius, obstacle, bounds):
        raise ValueError(f"Query point {point_tuple} is invalid or in collision")

    graph_aug = graph.copy()
    graph_aug.add_node(point_tuple, pos=point_tuple, kind=node_prefix)

    neighbors = find_connectable_neighbors(
        point=point_tuple,
        free_grid=free_grid,
        radius=radius,
        obstacle=obstacle,
        bounds=bounds,
        max_candidates=max_candidates,
        max_connections=max_connections,
        edge_sample_num=edge_sample_num,
    )
    if not neighbors:
        raise ValueError(f"No reachable graph neighbors found for query point {point_tuple}")

    for neighbor, weight in neighbors:
        graph_aug.add_edge(point_tuple, neighbor, weight=weight)

    return graph_aug, point_tuple

# @timing
def plan_path(
    artifacts: PlanningArtifacts,
    config: PlannerConfig,
    start: ArrayLike3,
    goal: ArrayLike3,
    method: Literal["astar", "dijkstra"] = "astar",
    max_candidates: int = 20,
    max_connections: int = 6,
) -> PathPlan:
    """Attach start and goal to the roadmap and plan a shortest path."""

    graph_aug, start_node = attach_query_point(
        graph=artifacts.graph,
        point=start,
        free_grid=artifacts.free_grid,
        radius=config.robot_radius,
        obstacle=artifacts.obstacle,
        bounds=config.bounds,
        max_candidates=max_candidates,
        max_connections=max_connections,
        edge_sample_num=config.edge_sample_num,
        node_prefix="start",
    )
    graph_aug, goal_node = attach_query_point(
        graph=graph_aug,
        point=goal,
        free_grid=artifacts.free_grid,
        radius=config.robot_radius,
        obstacle=artifacts.obstacle,
        bounds=config.bounds,
        max_candidates=max_candidates,
        max_connections=max_connections,
        edge_sample_num=config.edge_sample_num,
        node_prefix="goal",
    )

    if method == "astar":
        path = nx.astar_path(
            graph_aug,
            source=start_node,
            target=goal_node,
            heuristic=lambda a, b: euclidean_distance(a, b),
            weight="weight",
        )
        cost = float(nx.path_weight(graph_aug, path, weight="weight"))
    elif method == "dijkstra":
        path = nx.dijkstra_path(graph_aug, source=start_node, target=goal_node, weight="weight")
        cost = float(nx.path_weight(graph_aug, path, weight="weight"))
    else:
        raise ValueError("method must be 'astar' or 'dijkstra'")

    return PathPlan(
        method=method,
        start=start_node,
        goal=goal_node,
        path=list(path),
        cost=cost,
        graph_with_queries=graph_aug,
    )


def sample_random_valid_state(
    artifacts: PlanningArtifacts,
    config: PlannerConfig,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 10_000,
) -> GridPoint:
    """Rejection-sample a random valid point inside bounds."""
    if rng is None:
        rng = np.random.default_rng()

    bounds = config.bounds
    radius = config.robot_radius
    obstacle = artifacts.obstacle

    for _ in range(max_tries):
        point = (
            float(rng.uniform(bounds.xmin + radius, bounds.xmax - radius)),
            float(rng.uniform(bounds.ymin + radius, bounds.ymax - radius)),
            float(rng.uniform(bounds.zmin + radius, bounds.zmax - radius)),
        )
        if is_state_valid(point, radius, obstacle, bounds):
            return point

    raise RuntimeError("Failed to sample a valid random state within max_tries")

# -----------------------------
# PyBullet visualization
# -----------------------------

def visualize_motion_planning_graph(
    artifacts: PlanningArtifacts,
    path_plan: Optional[PathPlan] = None,
    show_sampled_grid: bool = False,
    sampled_grid_stride: int = 10,
    show_free_grid: bool = True,
    show_graph_edges: bool = True,
    show_path: bool = True,
    show_path_waypoints: bool = True,
    node_radius: float = 0.03,
    edge_line_width: float = 1.0,
    path_line_width: float = 4.0,
    obstacle_rgba: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    sampled_grid_rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    free_grid_rgb: Tuple[float, float, float] = (0.0, 0.8, 0.0),
    edge_rgb: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    path_rgb: Tuple[float, float, float] = (0.0, 0.4, 1.0),
    start_rgb: Tuple[float, float, float] = (1.0, 0.7, 0.0),
    goal_rgb: Tuple[float, float, float] = (1.0, 0.0, 1.0),
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
    if show_graph_edges:
        graph_to_draw = path_plan.graph_with_queries if path_plan is not None else artifacts.graph
        for node_u, node_v in graph_to_draw.edges():
            p.addUserDebugLine(list(node_u), list(node_v), edge_rgb, edge_line_width, 0)

    # Draw planned path.
    if path_plan is not None and show_path and len(path_plan.path) >= 2:
        for idx in range(len(path_plan.path) - 1):
            p.addUserDebugLine(
                list(path_plan.path[idx]),
                list(path_plan.path[idx + 1]),
                path_rgb,
                path_line_width,
                0,
            )

    if path_plan is not None and show_path_waypoints:
        for point in path_plan.path:
            _draw_cross(np.asarray(point, dtype=float), path_rgb, size=node_radius * 1.25)
        _draw_cross(np.asarray(path_plan.start, dtype=float), start_rgb, size=node_radius * 1.8)
        _draw_cross(np.asarray(path_plan.goal, dtype=float), goal_rgb, size=node_radius * 1.8)

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
    if path_plan is not None:
        print(f"Path method: {path_plan.method}")
        print(f"Path waypoints: {len(path_plan.path)}")
        print(f"Path cost: {path_plan.cost:.3f}")
    print("Close the PyBullet window or press Ctrl+C in the terminal to exit.")

    try:
        while p.isConnected(client):
            p.stepSimulation()
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)


# -----------------------------
# Example usage
# -----------------------------

def example_config() -> PlannerConfig:
    """Small example config; tune this to your bridge model placement."""
    return PlannerConfig(
        obj_path=r"./OBJ/water_tower.obj",
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

if __name__ == "__main__":
    cfg = example_config()
    artifacts = build_grid_motion_planning_graph(cfg)
    start = sample_random_valid_state(artifacts, cfg)
    goal = sample_random_valid_state(artifacts, cfg)
    path_plan = plan_path(artifacts, cfg, start=start, goal=goal, method="astar")
    visualize_motion_planning_graph(
        artifacts,
        path_plan=path_plan,
        show_sampled_grid=False,
        show_free_grid=False,
        show_graph_edges=False,
        show_path=True,
        show_path_waypoints=True,
        node_radius=0.04,
        edge_line_width=1.0,
        path_line_width=4.0,
    )
