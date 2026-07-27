from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import trimesh


ArrayLike3 = Sequence[float]
GridPoint = Tuple[float, float, float]


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned workspace bounds for planning."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def contains_point(self, point: ArrayLike3) -> bool:
        x, y, z = point
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )

    def contains_sphere(self, center: ArrayLike3, radius: float) -> bool:
        x, y, z = center
        return (
            self.xmin + radius <= x <= self.xmax - radius
            and self.ymin + radius <= y <= self.ymax - radius
            and self.zmin + radius <= z <= self.zmax - radius
        )

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)


@dataclass
class ObstacleMesh:
    """Obstacle geometry represented as a trimesh mesh in world coordinates."""

    mesh: trimesh.Trimesh
    source_path: str

    @property
    def aabb_min(self) -> np.ndarray:
        return self.mesh.bounds[0]

    @property
    def aabb_max(self) -> np.ndarray:
        return self.mesh.bounds[1]

@dataclass
class InspectionObjectConfig:
    obj_path: str
    bounds: Bounds3D
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class GridPlannerConfig:
    robot_radius: float
    grid_resolution: list[float]
    connectivity: int = 6   # 6, 18, 26
    edge_sample_num: int = 10


# Legacy - both inspectionObject and GridPlanner configs
@dataclass
class PlannerConfig:
    obj_path: str
    bounds: Bounds3D
    robot_radius: float
    grid_resolution: float
    connectivity: int = 6
    edge_sample_num: int = 10
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PlanningArtifacts:
    obstacle: ObstacleMesh
    sampled_grid: np.ndarray
    free_grid: np.ndarray
    graph: nx.Graph


# -----------------------------
# Geometry loading / transforms
# -----------------------------

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll-pitch-yaw in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def load_object_env(
    obj_path: str,
    scale: ArrayLike3 = (1.0, 1.0, 1.0),
    translation: ArrayLike3 = (0.0, 0.0, 0.0),
    rotation_rpy: ArrayLike3 = (0.0, 0.0, 0.0),
) -> ObstacleMesh:
    """Load an OBJ as a single mesh and place it in world coordinates.

    The returned mesh is ready for geometric collision checks.
    """
    scene_or_mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(scene_or_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a mesh from {obj_path}, got {type(scene_or_mesh)!r}")

    mesh = scene_or_mesh.copy()

    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    rotation = rpy_to_matrix(*rotation_rpy)

    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = vertices * scale
    vertices = (rotation @ vertices.T).T
    vertices = vertices + translation
    mesh.vertices = vertices

    return ObstacleMesh(mesh=mesh, source_path=obj_path)


# -----------------------------
# Collision checking
# -----------------------------

def sphere_collision_check(
    loc: ArrayLike3,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
) -> bool:
    """Return True if a sphere at loc with radius collides.

    Collision occurs if the sphere exits bounds or intersects the obstacle mesh.
    Uses signed distance when available via trimesh.proximity.
    """
    center = np.asarray(loc, dtype=float)

    if not bounds.contains_sphere(center, radius):
        return True

    try:
        signed_distance = trimesh.proximity.signed_distance(obstacle.mesh, [center])[0]
    except Exception:
        nearest_points, distances, _ = obstacle.mesh.nearest.on_surface([center])
        nearest_dist = float(distances[0])
        is_inside = bool(obstacle.mesh.contains([center])[0])
        return is_inside or nearest_dist <= radius

    return signed_distance >= -radius


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


def point_to_grid_key(point: ArrayLike3, grid_resolution) -> Tuple[int, int, int]:
    arr = np.asarray(point, dtype=float) / np.asarray(grid_resolution, dtype=float)
    return tuple(np.round(arr).astype(int).tolist())


def is_local_path_collision_free(
    start: ArrayLike3,
    goal: ArrayLike3,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    edge_sample_num: int,
) -> bool:
    """Check straight-line local motion between two free nodes."""
    if edge_sample_num < 2:
        edge_sample_num = 2

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    for t in np.linspace(0.0, 1.0, edge_sample_num):
        point = (1.0 - t) * start + t * goal
        if sphere_collision_check(point, radius, obstacle, bounds):
            return False
    return True



def connect_free_grid_points(
    free_grid: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    grid_resolution,
    connectivity: int,
    edge_sample_num: int,
) -> nx.Graph:
    """Create an undirected graph over free grid points with valid local edges."""
    graph = nx.Graph()

    if len(free_grid) == 0:
        return graph

    neighbor_offsets = generate_neighbor_offsets(connectivity)
    key_to_point: Dict[Tuple[int, int, int], np.ndarray] = {
        point_to_grid_key(point, grid_resolution): point for point in free_grid
    }

    for idx, point in enumerate(free_grid):
        node = tuple(point.tolist())
        graph.add_node(node, pos=node, index=idx)

    for point in free_grid:
        point_key = point_to_grid_key(point, grid_resolution)
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
            if graph.has_edge(node_u, node_v) or node_u == node_v:
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
                graph.add_edge(node_u, node_v, weight=weight)

    return graph


# -----------------------------
# Pipeline manager
# -----------------------------
# @timing
def build_grid_motion_planning_graph(config, insp_object, env_bounds) -> PlanningArtifacts:
    """Run the full grid-based planning pipeline and return all artifacts."""

    sampled_grid = sample_space_grid(env_bounds, config.grid_resolution)
    free_grid = find_free_grid_points(
        sampled_grid=sampled_grid,
        radius=config.robot_radius,
        obstacle=insp_object,
        bounds=env_bounds,
    )

    graph = connect_free_grid_points(
        free_grid=free_grid,
        radius=config.robot_radius,
        obstacle=insp_object,
        bounds=env_bounds,
        grid_resolution=config.grid_resolution,
        connectivity=config.connectivity,
        edge_sample_num=config.edge_sample_num,
    )

    return PlanningArtifacts(
        obstacle=insp_object,
        sampled_grid=sampled_grid,
        free_grid=free_grid,
        graph=graph,
    )


# -----------------------------
# PyBullet visualization
# -----------------------------

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

# -----------------------------
# Example usage
# -----------------------------

def example_config() -> PlannerConfig:
    """Small example config; tune this to your bridge model placement."""
    return PlannerConfig(
        obj_path=r"../assets/OBJ/Bridge.obj",
        bounds=Bounds3D(
            xmin=-6.0,
            xmax=6.0,
            ymin=-0.75,
            ymax=0.75,
            zmin=0.0,
            zmax=2.0,
        ),
        robot_radius=0.15,
        grid_resolution=0.5,
        connectivity=18,
        edge_sample_num=3,
        scale=(0.05, 0.05, 0.05),
        translation=(0, 0.25, 1.5),
        rotation_rpy=(0.0, 0.0, 0.0),
    )

if __name__ == "__main__":
    cfg = example_config()
    artifacts = build_grid_motion_planning_graph(cfg)

    visualize_grid_sampling(artifacts, show_sampled_grid=False, sampled_grid_stride=10, node_radius=0.08,
                            edge_line_width=2.0)