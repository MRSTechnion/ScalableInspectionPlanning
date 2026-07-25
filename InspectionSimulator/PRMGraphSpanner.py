from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import math
import os
import tempfile

import networkx as nx
import numpy as np
import trimesh


ArrayLike3 = Sequence[float]
Point3D = Tuple[float, float, float]


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
class PRMPlannerConfig:
    """Configuration for 3D PRM planning with a spherical robot."""

    obj_path: str
    bounds: Bounds3D
    robot_radius: float

    # PRM sampling / connectivity.
    num_samples: int = 1500
    max_sample_attempts: Optional[int] = None
    k_neighbors: Optional[int] = 10
    connection_radius: Optional[float] = None
    edge_check_resolution: float = 0.05
    random_seed: Optional[int] = 0

    # Optional query. If both are provided, the planner will try to solve a path.
    start: Optional[Point3D] = None
    goal: Optional[Point3D] = None

    # Mesh placement, same convention as the grid planner.
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PRMPlanningArtifacts:
    obstacle: ObstacleMesh
    sampled_points: np.ndarray
    free_points: np.ndarray
    graph: nx.Graph
    start: Optional[Point3D] = None
    goal: Optional[Point3D] = None
    path: Optional[List[Point3D]] = None


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
    """Load an OBJ as a single mesh and place it in world coordinates."""
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




def _point_triangle_distance(point: np.ndarray, tri: np.ndarray) -> float:
    """Distance from a point to one triangle, based on closest-feature regions."""
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        return float(np.linalg.norm(point - projection))

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        return float(np.linalg.norm(point - projection))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * (c - b)
        return float(np.linalg.norm(point - projection))

    normal = np.cross(ab, ac)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        return min(
            float(np.linalg.norm(point - a)),
            float(np.linalg.norm(point - b)),
            float(np.linalg.norm(point - c)),
        )
    return abs(float(np.dot(point - a, normal))) / normal_norm


def _point_mesh_distance_bruteforce(mesh: trimesh.Trimesh, point: np.ndarray) -> float:
    """Slow fallback distance from point to mesh triangles; avoids optional rtree."""
    triangles = np.asarray(mesh.triangles, dtype=float)
    if len(triangles) == 0:
        return float("inf")
    return min(_point_triangle_distance(point, tri) for tri in triangles)


def _ray_intersects_triangle(
    origin: np.ndarray,
    direction: np.ndarray,
    tri: np.ndarray,
    eps: float = 1e-9,
) -> Optional[float]:
    """Moller-Trumbore ray/triangle intersection distance, or None."""
    v0, v1, v2 = tri
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if -eps < a < eps:
        return None
    f = 1.0 / a
    s = origin - v0
    u = f * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return None
    q = np.cross(s, edge1)
    v = f * float(np.dot(direction, q))
    if v < -eps or u + v > 1.0 + eps:
        return None
    t = f * float(np.dot(edge2, q))
    if t > eps:
        return t
    return None


def _point_inside_mesh_bruteforce(mesh: trimesh.Trimesh, point: np.ndarray) -> bool:
    """Slow fallback inside test by odd/even ray intersections."""
    triangles = np.asarray(mesh.triangles, dtype=float)
    if len(triangles) == 0:
        return False

    # Slightly non-axis-aligned ray lowers the chance of ambiguous edge hits.
    direction = np.array([1.0, 0.371390676, 0.529812943], dtype=float)
    direction = direction / np.linalg.norm(direction)

    hits: List[float] = []
    for tri in triangles:
        t = _ray_intersects_triangle(point, direction, tri)
        if t is not None:
            hits.append(float(t))

    if not hits:
        return False

    # Deduplicate near-identical hits caused by adjacent triangles sharing an edge.
    hits.sort()
    unique_hits = [hits[0]]
    for t in hits[1:]:
        if abs(t - unique_hits[-1]) > 1e-7:
            unique_hits.append(t)

    return (len(unique_hits) % 2) == 1

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
        return signed_distance >= -radius
    except Exception:
        # Some trimesh proximity methods require optional spatial-index packages
        # such as rtree. Keep the planner runnable with a slower pure NumPy
        # fallback when those packages are unavailable.
        nearest_dist = _point_mesh_distance_bruteforce(obstacle.mesh, center)
        if nearest_dist <= radius:
            return True
        return _point_inside_mesh_bruteforce(obstacle.mesh, center)


def is_local_path_collision_free_by_resolution(
    start: ArrayLike3,
    goal: ArrayLike3,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    edge_check_resolution: float,
) -> bool:
    """Check a straight-line local PRM edge using distance-based sampling."""
    if edge_check_resolution <= 0:
        raise ValueError("edge_check_resolution must be positive")

    start_arr = np.asarray(start, dtype=float)
    goal_arr = np.asarray(goal, dtype=float)
    edge_length = float(np.linalg.norm(goal_arr - start_arr))
    edge_sample_num = max(2, int(math.ceil(edge_length / edge_check_resolution)) + 1)

    for t in np.linspace(0.0, 1.0, edge_sample_num):
        point = (1.0 - t) * start_arr + t * goal_arr
        if sphere_collision_check(point, radius, obstacle, bounds):
            return False
    return True


# -----------------------------
# PRM sampling
# -----------------------------

def sample_space_prm(
    bounds: Bounds3D,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw raw uniform random samples inside workspace bounds."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    low = np.array([bounds.xmin, bounds.ymin, bounds.zmin], dtype=float)
    high = np.array([bounds.xmax, bounds.ymax, bounds.zmax], dtype=float)
    return rng.uniform(low=low, high=high, size=(num_samples, 3))


def find_free_prm_points(
    sampled_points: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
) -> np.ndarray:
    """Filter candidate PRM points to collision-free sphere centers."""
    free_points: List[np.ndarray] = []
    for point in sampled_points:
        if not sphere_collision_check(point, radius, obstacle, bounds):
            free_points.append(np.asarray(point, dtype=float))

    if not free_points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(free_points)


def sample_free_space_prm(
    bounds: Bounds3D,
    num_free_samples: int,
    radius: float,
    obstacle: ObstacleMesh,
    rng: np.random.Generator,
    max_sample_attempts: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample until the requested number of free PRM points is reached.

    Returns:
        sampled_points: all raw candidate points attempted.
        free_points: the first num_free_samples collision-free points found.
    """
    if num_free_samples < 0:
        raise ValueError("num_free_samples must be non-negative")

    if num_free_samples == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    if max_sample_attempts is None:
        max_sample_attempts = max(100, 20 * num_free_samples)
    if max_sample_attempts < num_free_samples:
        raise ValueError("max_sample_attempts should be at least num_samples")

    if batch_size is None:
        batch_size = min(max(100, num_free_samples), 5000)

    sampled_batches: List[np.ndarray] = []
    free_points: List[np.ndarray] = []
    attempts = 0

    while len(free_points) < num_free_samples and attempts < max_sample_attempts:
        remaining_attempts = max_sample_attempts - attempts
        current_batch_size = min(batch_size, remaining_attempts)
        candidates = sample_space_prm(bounds, current_batch_size, rng)
        sampled_batches.append(candidates)
        attempts += current_batch_size

        for point in candidates:
            if not sphere_collision_check(point, radius, obstacle, bounds):
                free_points.append(point)
                if len(free_points) >= num_free_samples:
                    break

    sampled_points = np.vstack(sampled_batches) if sampled_batches else np.empty((0, 3), dtype=float)
    free_array = np.vstack(free_points) if free_points else np.empty((0, 3), dtype=float)

    if len(free_array) < num_free_samples:
        print(
            f"Warning: requested {num_free_samples} free samples, "
            f"but only found {len(free_array)} after {attempts} attempts."
        )

    return sampled_points, free_array


# -----------------------------
# PRM graph construction
# -----------------------------

def point_to_node(point: ArrayLike3) -> Point3D:
    arr = np.asarray(point, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _neighbor_candidates_bruteforce(
    points: np.ndarray,
    index: int,
    k_neighbors: Optional[int],
    connection_radius: Optional[float],
) -> Set[int]:
    diffs = points - points[index]
    distances = np.linalg.norm(diffs, axis=1)
    candidates: Set[int] = set()

    if k_neighbors is not None and k_neighbors > 0:
        # argsort includes self at distance 0; skip it.
        nearest = np.argsort(distances)[1 : k_neighbors + 1]
        candidates.update(int(i) for i in nearest)

    if connection_radius is not None:
        radius_ids = np.where((distances <= connection_radius) & (distances > 0.0))[0]
        candidates.update(int(i) for i in radius_ids)

    return candidates


def _all_neighbor_candidates(
    points: np.ndarray,
    k_neighbors: Optional[int],
    connection_radius: Optional[float],
) -> List[Set[int]]:
    """Return candidate neighbor indices for each point.

    Uses scipy.cKDTree if available; otherwise falls back to NumPy brute force.
    """
    if len(points) == 0:
        return []

    if (k_neighbors is None or k_neighbors <= 0) and connection_radius is None:
        raise ValueError("Set k_neighbors > 0 and/or connection_radius to build PRM edges")

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return [
            _neighbor_candidates_bruteforce(points, i, k_neighbors, connection_radius)
            for i in range(len(points))
        ]

    tree = cKDTree(points)
    all_candidates: List[Set[int]] = []

    for i, point in enumerate(points):
        candidates: Set[int] = set()

        if k_neighbors is not None and k_neighbors > 0:
            query_k = min(k_neighbors + 1, len(points))
            _, nearest = tree.query(point, k=query_k)
            nearest_arr = np.atleast_1d(nearest)
            candidates.update(int(j) for j in nearest_arr if int(j) != i)

        if connection_radius is not None:
            radius_ids = tree.query_ball_point(point, r=float(connection_radius))
            candidates.update(int(j) for j in radius_ids if int(j) != i)

        all_candidates.append(candidates)

    return all_candidates


def connect_prm_points(
    free_points: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    k_neighbors: Optional[int],
    connection_radius: Optional[float],
    edge_check_resolution: float,
) -> nx.Graph:
    """Create an undirected PRM graph over collision-free samples."""
    graph = nx.Graph()

    if len(free_points) == 0:
        return graph

    for idx, point in enumerate(free_points):
        node = point_to_node(point)
        graph.add_node(node, pos=node, index=idx, kind="sample")

    all_candidates = _all_neighbor_candidates(free_points, k_neighbors, connection_radius)

    for i, candidate_ids in enumerate(all_candidates):
        node_u = point_to_node(free_points[i])
        for j in candidate_ids:
            if j <= i:
                continue

            node_v = point_to_node(free_points[j])
            if graph.has_edge(node_u, node_v):
                continue

            if is_local_path_collision_free_by_resolution(
                start=free_points[i],
                goal=free_points[j],
                radius=radius,
                obstacle=obstacle,
                bounds=bounds,
                edge_check_resolution=edge_check_resolution,
            ):
                weight = float(np.linalg.norm(free_points[j] - free_points[i]))
                graph.add_edge(node_u, node_v, weight=weight)

    return graph


def _select_neighbors_for_query_point(
    query_point: np.ndarray,
    roadmap_points: np.ndarray,
    k_neighbors: Optional[int],
    connection_radius: Optional[float],
) -> List[int]:
    if len(roadmap_points) == 0:
        return []

    if (k_neighbors is None or k_neighbors <= 0) and connection_radius is None:
        raise ValueError("Set k_neighbors > 0 and/or connection_radius to connect query points")

    distances = np.linalg.norm(roadmap_points - query_point, axis=1)
    selected: Set[int] = set()

    if k_neighbors is not None and k_neighbors > 0:
        nearest = np.argsort(distances)[: min(k_neighbors, len(roadmap_points))]
        selected.update(int(i) for i in nearest)

    if connection_radius is not None:
        radius_ids = np.where(distances <= connection_radius)[0]
        selected.update(int(i) for i in radius_ids)

    return sorted(selected, key=lambda idx: distances[idx])


def add_query_point_to_prm(
    graph: nx.Graph,
    query_point: ArrayLike3,
    query_kind: str,
    roadmap_points: np.ndarray,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    k_neighbors: Optional[int],
    connection_radius: Optional[float],
    edge_check_resolution: float,
) -> Point3D:
    """Add start or goal to the roadmap and connect it to nearby valid nodes."""
    query_arr = np.asarray(query_point, dtype=float)
    if sphere_collision_check(query_arr, radius, obstacle, bounds):
        raise ValueError(f"{query_kind} point is in collision or outside bounds: {query_point}")

    query_node = point_to_node(query_arr)
    graph.add_node(query_node, pos=query_node, index=None, kind=query_kind)

    neighbor_ids = _select_neighbors_for_query_point(
        query_arr, roadmap_points, k_neighbors, connection_radius
    )

    for idx in neighbor_ids:
        neighbor_point = roadmap_points[idx]
        neighbor_node = point_to_node(neighbor_point)

        if is_local_path_collision_free_by_resolution(
            start=query_arr,
            goal=neighbor_point,
            radius=radius,
            obstacle=obstacle,
            bounds=bounds,
            edge_check_resolution=edge_check_resolution,
        ):
            weight = float(np.linalg.norm(neighbor_point - query_arr))
            graph.add_edge(query_node, neighbor_node, weight=weight)

    return query_node


def solve_prm_path(
    graph: nx.Graph,
    start: Point3D,
    goal: Point3D,
) -> Optional[List[Point3D]]:
    """Return the shortest roadmap path, or None if no path exists."""
    try:
        return list(nx.shortest_path(graph, start, goal, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# -----------------------------
# Pipeline manager
# -----------------------------

def build_prm_motion_planning_graph(config: PRMPlannerConfig) -> PRMPlanningArtifacts:
    """Run the full PRM pipeline and return all artifacts."""
    if config.robot_radius < 0:
        raise ValueError("robot_radius must be non-negative")
    if config.num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if config.edge_check_resolution <= 0:
        raise ValueError("edge_check_resolution must be positive")

    obstacle = load_object_env(
        obj_path=config.obj_path,
        scale=config.scale,
        translation=config.translation,
        rotation_rpy=config.rotation_rpy,
    )

    rng = np.random.default_rng(config.random_seed)
    sampled_points, free_points = sample_free_space_prm(
        bounds=config.bounds,
        num_free_samples=config.num_samples,
        radius=config.robot_radius,
        obstacle=obstacle,
        rng=rng,
        max_sample_attempts=config.max_sample_attempts,
    )

    graph = connect_prm_points(
        free_points=free_points,
        radius=config.robot_radius,
        obstacle=obstacle,
        bounds=config.bounds,
        k_neighbors=config.k_neighbors,
        connection_radius=config.connection_radius,
        edge_check_resolution=config.edge_check_resolution,
    )

    start_node: Optional[Point3D] = None
    goal_node: Optional[Point3D] = None
    path: Optional[List[Point3D]] = None

    if config.start is not None:
        start_node = add_query_point_to_prm(
            graph=graph,
            query_point=config.start,
            query_kind="start",
            roadmap_points=free_points,
            radius=config.robot_radius,
            obstacle=obstacle,
            bounds=config.bounds,
            k_neighbors=config.k_neighbors,
            connection_radius=config.connection_radius,
            edge_check_resolution=config.edge_check_resolution,
        )

    if config.goal is not None:
        goal_node = add_query_point_to_prm(
            graph=graph,
            query_point=config.goal,
            query_kind="goal",
            roadmap_points=free_points,
            radius=config.robot_radius,
            obstacle=obstacle,
            bounds=config.bounds,
            k_neighbors=config.k_neighbors,
            connection_radius=config.connection_radius,
            edge_check_resolution=config.edge_check_resolution,
        )

    if start_node is not None and goal_node is not None:
        path = solve_prm_path(graph, start_node, goal_node)

    return PRMPlanningArtifacts(
        obstacle=obstacle,
        sampled_points=sampled_points,
        free_points=free_points,
        graph=graph,
        start=start_node,
        goal=goal_node,
        path=path,
    )


# -----------------------------
# PyBullet visualization
# -----------------------------

def visualize_prm(
    artifacts: PRMPlanningArtifacts,
    show_sampled_points: bool = False,
    sampled_points_stride: int = 10,
    show_roadmap_edges: bool = True,
    show_path: bool = True,
    node_point_size: float = 5.0,
    sampled_point_size: float = 2.0,
    edge_line_width: float = 1.0,
    path_line_width: float = 5.0,
    obstacle_rgba: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    sampled_rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    free_rgb: Tuple[float, float, float] = (0.0, 0.8, 0.0),
    edge_rgb: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    path_rgb: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    start_rgb: Tuple[float, float, float] = (0.0, 1.0, 1.0),
    goal_rgb: Tuple[float, float, float] = (1.0, 0.0, 1.0),
    camera_distance: Optional[float] = None,
    camera_yaw: float = 45.0,
    camera_pitch: float = -30.0,
) -> None:
    """Visualize the obstacle mesh, PRM roadmap, and optional path in PyBullet."""
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise ImportError(
            "visualize_prm requires pybullet. Install it with: pip install pybullet"
        ) from exc

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.loadURDF("plane.urdf")

    mesh = artifacts.obstacle.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)

    tmp_dir = tempfile.mkdtemp(prefix="prm_vis_")
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

    if show_sampled_points and len(artifacts.sampled_points) > 0:
        stride = max(1, int(sampled_points_stride))
        sampled = [list(point) for point in artifacts.sampled_points[::stride]]
        p.addUserDebugPoints(
            pointPositions=sampled,
            pointColorsRGB=[sampled_rgb for _ in sampled],
            pointSize=sampled_point_size,
            lifeTime=0,
        )

    if len(artifacts.free_points) > 0:
        points = [list(point) for point in artifacts.free_points]
        p.addUserDebugPoints(
            pointPositions=points,
            pointColorsRGB=[free_rgb for _ in points],
            pointSize=node_point_size,
            lifeTime=0,
        )

    if artifacts.start is not None:
        p.addUserDebugPoints(
            pointPositions=[list(artifacts.start)],
            pointColorsRGB=[start_rgb],
            pointSize=node_point_size * 2.0,
            lifeTime=0,
        )

    if artifacts.goal is not None:
        p.addUserDebugPoints(
            pointPositions=[list(artifacts.goal)],
            pointColorsRGB=[goal_rgb],
            pointSize=node_point_size * 2.0,
            lifeTime=0,
        )

    if show_roadmap_edges:
        for node_u, node_v in artifacts.graph.edges():
            p.addUserDebugLine(list(node_u), list(node_v), edge_rgb, edge_line_width, 0)

    if show_path and artifacts.path is not None and len(artifacts.path) >= 2:
        for node_u, node_v in zip(artifacts.path[:-1], artifacts.path[1:]):
            p.addUserDebugLine(list(node_u), list(node_v), path_rgb, path_line_width, 0)

    # Draw world axes.
    axis_len = 1.0
    if len(vertices) > 0:
        axis_len = max(1.0, 0.2 * float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))
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
    print(f"Raw sampled points: {len(artifacts.sampled_points)}")
    print(f"Free PRM points: {len(artifacts.free_points)}")
    print(f"Graph nodes: {artifacts.graph.number_of_nodes()}")
    print(f"Graph edges: {artifacts.graph.number_of_edges()}")
    if artifacts.path is None:
        print("Path: not found or start/goal not provided")
    else:
        path_length = nx.path_weight(artifacts.graph, artifacts.path, weight="weight")
        print(f"Path nodes: {len(artifacts.path)}")
        print(f"Path length: {path_length:.4f}")
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

def example_config() -> PRMPlannerConfig:
    """Small example config; tune start/goal to your bridge model placement."""
    return PRMPlannerConfig(
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
        num_samples=50,
        k_neighbors=2,
        connection_radius=None,
        edge_check_resolution=0.05,
        random_seed=0,
        start=(-5.5, 0.0, 0.5),
        goal=(5.5, 0.0, 0.5),
        scale=(0.05, 0.05, 0.05),
        translation=(0, 0.25, 1.5),
        rotation_rpy=(0.0, 0.0, 0.0),
    )


if __name__ == "__main__":
    cfg = example_config()
    artifacts = build_prm_motion_planning_graph(cfg)

    visualize_prm(
        artifacts,
        show_sampled_points=False,
        sampled_points_stride=10,
        show_roadmap_edges=True,
        show_path=True,
        node_point_size=5.0,
        edge_line_width=1.0,
        path_line_width=5.0,
    )