from InspectionSimulator.SceneLoader import ArrayLike3, Bounds3D, ObstacleMesh

import numpy as np
import trimesh

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

def is_local_path_collision_free(
    start: ArrayLike3,
    goal: ArrayLike3,
    radius: float,
    obstacle: ObstacleMesh,
    bounds: Bounds3D,
    edge_sample_num: int,
) -> bool:
    """Check straight-line local motion between two free nodes."""

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    for t in np.linspace(0.0, 1.0, edge_sample_num):
        point = (1.0 - t) * start + t * goal
        if sphere_collision_check(point, radius, obstacle, bounds):
            return False
    return True
