from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from dataclasses import dataclass

ArrayLike3 = Sequence[float]
GridPoint = Tuple[float, float, float]



# # Legacy - both inspectionObject and GridPlanner configs
# @dataclass
# class PlannerConfig:
#     obj_path: str
#     bounds: Bounds3D
#     robot_radius: float
#     grid_resolution: float
#     connectivity: int = 6
#     edge_sample_num: int = 10
#     scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
#     translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
#     rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


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
class InspectionObjectConfig:
    obj_path: str
    bounds: Bounds3D
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visibility_threshold: float = np.inf
    robot_radius:float = 1,
    edge_CC_samples:int = 3

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
    rotation_rpy = rpy_to_matrix(*rotation_rpy)

    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = vertices * scale
    vertices = (rotation_rpy @ vertices.T).T
    vertices = vertices + translation
    mesh.vertices = vertices

    return ObstacleMesh(mesh=mesh, source_path=obj_path)
