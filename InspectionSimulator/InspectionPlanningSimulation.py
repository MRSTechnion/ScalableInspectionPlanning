from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import networkx as nx
import numpy as np
import math
import trimesh
from rasterio.crs import defaultdict
from scipy.spatial import KDTree

import time
from typing import Optional

import pybullet as p
import pybullet_data
import trimesh

VertexId = Hashable
POIId = int

@dataclass
class POI:
    """A point of interest on the inspected structure."""

    poi_id: POIId
    xyz: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3,)


@dataclass
class POISet:
    """
    POI storage with a reusable spatial index.

    If POIs are added, removed, or moved after construction,
    ``rebuild_spatial_index()`` must be called.
    """

    pois: List[POI]

    poi_xyz: np.ndarray = field(init=False, repr=False)
    poi_tree: Optional[KDTree] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.pois = list(self.pois)
        self._validate_unique_ids()
        self.rebuild_spatial_index()

    def _validate_unique_ids(self) -> None:
        poi_ids = [poi.poi_id for poi in self.pois]

        if len(poi_ids) != len(set(poi_ids)):
            raise ValueError("POI IDs must be unique")

    def rebuild_spatial_index(self) -> None:
        """Rebuild cached coordinates and the POI KD-tree."""
        if not self.pois:
            self.poi_xyz = np.empty((0, 3), dtype=float)
            self.poi_tree = None
            return

        self.poi_xyz = np.asarray(
            [poi.xyz for poi in self.pois],
            dtype=float,
        )

        self.poi_tree = KDTree(self.poi_xyz)

    def nearby_indices(
        self,
        xyz: np.ndarray,
        radius: float,
    ) -> List[int]:
        """Return indices of POIs within ``radius`` of ``xyz``."""
        if radius < 0:
            raise ValueError("radius must be non-negative")

        if self.poi_tree is None:
            return []

        xyz = np.asarray(xyz, dtype=float).reshape(3,)
        return self.poi_tree.query_ball_point(xyz, r=radius)

    def as_array(self) -> np.ndarray:
        return self.poi_xyz.copy()

    def as_dict(self) -> Dict[POIId, np.ndarray]:
        return {
            poi.poi_id: poi.xyz.copy()
            for poi in self.pois
        }


@dataclass
class InspectionPlanningInstance:
    """
    Inspection layer built on top of a roadmap graph.
    """
    graph: nx.Graph
    poi_set: POISet
    visibility_threshold: float

    # POI -> list of inspecting vertices
    poi_to_vertices: Dict[POIId, List[VertexId]]

    # Vertex -> list of visible POIs
    vertex_to_pois: Dict[VertexId, List[POIId]]

    # Cached xyz for graph nodes
    vertex_xyz: Dict[VertexId, np.ndarray]


def sample_pois_on_mesh_surface(
    object_mesh: trimesh.Trimesh,
    num_pois: int,
    seed: Optional[int] = None,
    metadata_factory: Optional[callable] = None,
) -> POISet:
    """
    Sample random POIs on the surface of the bridge mesh.
    """
    if num_pois < 0:
        raise ValueError(f"num_pois must be non-negative, got {num_pois}")

    if num_pois == 0:
        return POISet(pois=[])

    rng = np.random.default_rng(seed)
    points, face_indices = trimesh.sample.sample_surface(object_mesh, num_pois)

    pois: List[POI] = []
    for i, (point, face_idx) in enumerate(zip(points, face_indices)):
        metadata = {"face_index": int(face_idx)}
        if metadata_factory is not None:
            extra = metadata_factory(i, np.asarray(point, dtype=float), int(face_idx))
            if extra is not None:
                metadata.update(extra)

        pois.append(POI(poi_id=i, xyz=np.asarray(point, dtype=float), metadata=metadata))

    return POISet(pois=pois)

def compute_visibility(
    graph: nx.Graph,
    poi_set: POISet,
    object_mesh: trimesh.Trimesh,
    visibility_threshold: float = np.inf,
    occlusion_tolerance: float = 1e-5,
):
    nodes = list(graph.nodes)

    # Preserve correspondence between nodes and positions explicitly.
    vertex_xyz = np.asarray(
        [graph.nodes[node]["pos"] for node in nodes],
        dtype=float,
    )

    tree = KDTree(vertex_xyz)

    poi_to_vertices = defaultdict(set)
    vertex_to_pois = defaultdict(set)

    for poi in poi_set.pois:
        # Broad phase: vertices close enough to inspect the POI.
        candidate_indices = tree.query_ball_point(
            poi.xyz,
            r=visibility_threshold,
        )

        if not candidate_indices:
            continue

        origins = vertex_xyz[candidate_indices]
        vectors = poi.xyz[None, :] - origins
        target_distances = np.linalg.norm(vectors, axis=1)

        valid = target_distances > occlusion_tolerance    # Too close for our tolerance - we throw it away
        directions = np.zeros_like(vectors)
        directions[valid] = (
            vectors[valid] / target_distances[valid, None]
        )

        # Find the first mesh intersection along every ray.
        triangle_ids, ray_ids, hit_locations = (
            object_mesh.ray.intersects_id(
                ray_origins=origins[valid],
                ray_directions=directions[valid],
                multiple_hits=False,
                return_locations=True,
            )
        )

        # Rays without an early intersection are initially considered clear.
        clear = np.ones(np.count_nonzero(valid), dtype=bool)

        hit_distances = np.linalg.norm(
            hit_locations - origins[valid][ray_ids],
            axis=1,
        )

        # The POI itself lies on the mesh and should produce an intersection.
        # Only intersections strictly before the POI represent occlusion.
        clear[ray_ids] = (
            hit_distances
            >= target_distances[valid][ray_ids] - occlusion_tolerance
        )

        valid_candidate_indices = np.asarray(candidate_indices)[valid]

        for candidate_idx, line_is_clear in zip(
            valid_candidate_indices,
            clear,
        ):
            if not line_is_clear:
                continue

            node = nodes[candidate_idx]
            poi_to_vertices[poi.poi_id].add(node)
            vertex_to_pois[node].add(poi.poi_id)

    return poi_to_vertices, vertex_to_pois

# TODO - implement a single ray casting function to use in all visibility functions
# TODO - implement a complementary single-POI visibility function, and choose between them according to sets sizes
def compute_vertex_vis(
    v,
    poi_set: POISet,
    object_mesh: trimesh.Trimesh,
    visibility_threshold: float,
    occlusion_tolerance: float = 1e-5,
):
    v_xyz = np.asarray(v, dtype=float).reshape(3,)

    candidate_indices = poi_set.nearby_indices(
        xyz=v_xyz,
        radius=visibility_threshold,
    )

    vis_set = set()

    for candidate_idx in candidate_indices:
        poi = poi_set.pois[candidate_idx]
        vector = poi_set.poi_xyz[candidate_idx] - v_xyz
        target_distance = np.linalg.norm(vector)

        if target_distance <= occlusion_tolerance:
            continue

        direction = vector / target_distance

        _, _, hit_locations = object_mesh.ray.intersects_id(
            ray_origins=v_xyz[None, :],
            ray_directions=direction[None, :],
            multiple_hits=False,
            return_locations=True,
        )

        if len(hit_locations) == 0:
            vis_set.add(poi.poi_id)
            continue

        hit_distance = np.linalg.norm(hit_locations[0] - v_xyz)

        if hit_distance >= target_distance - occlusion_tolerance:
            vis_set.add(poi.poi_id)

    return vis_set

def build_inspection_planning_instance(
    graph: nx.Graph,
    bridge_mesh: trimesh.Trimesh,
    num_pois: int,
    visibility_threshold: float,
    seed: Optional[int] = None,
) -> InspectionPlanningInstance:
    """
    Full pipeline:
    1. sample POIs on bridge surface
    2. compute visibility from graph vertices
    3. return inspection planning instance
    """


    poi_set = sample_pois_on_mesh_surface(object_mesh=bridge_mesh, num_pois=num_pois, seed=seed)

    poi_to_vertices, vertex_to_pois, vertex_xyz = compute_visibility(
        graph=graph,
        poi_set=poi_set,
        visibility_threshold=visibility_threshold,
    )

    return InspectionPlanningInstance(
        graph=graph,
        poi_set=poi_set,
        visibility_threshold=visibility_threshold,
        poi_to_vertices=poi_to_vertices,
        vertex_to_pois=vertex_to_pois,
        vertex_xyz=vertex_xyz,
    )


def attach_inspection_data_to_graph(
    instance: InspectionPlanningInstance,
    vertex_poi_attr_name: str = "visible_pois",
) -> None:
    """
    Annotate graph nodes with the list of visible POIs from each vertex.

    POIs remain external and are NOT converted to graph nodes.
    """
    for node in instance.graph.nodes:
        instance.graph.nodes[node][vertex_poi_attr_name] = list(
            instance.vertex_to_pois.get(node, [])
        )


def summarize_inspection_instance(instance: InspectionPlanningInstance) -> dict:
    """
    Small summary for quick sanity checking.
    """
    num_uncovered_pois = sum(
        1 for poi in instance.poi_set.pois
        if len(instance.poi_to_vertices.get(poi.poi_id, [])) == 0
    )
    num_covering_vertices = sum(
        1 for node in instance.graph.nodes
        if len(instance.vertex_to_pois.get(node, [])) > 0
    )

    cover_counts = [len(instance.poi_to_vertices.get(poi.poi_id, [])) for poi in instance.poi_set.pois]
    vertex_counts = [len(instance.vertex_to_pois.get(node, [])) for node in instance.graph.nodes]

    return {
        "num_vertices": instance.graph.number_of_nodes(),
        "num_edges": instance.graph.number_of_edges(),
        "num_pois": len(instance.poi_set.pois),
        "visibility_threshold": instance.visibility_threshold,
        "num_uncovered_pois": num_uncovered_pois,
        "num_covering_vertices": num_covering_vertices,
        "min_vertices_per_poi": int(min(cover_counts)) if cover_counts else 0,
        "max_vertices_per_poi": int(max(cover_counts)) if cover_counts else 0,
        "mean_vertices_per_poi": float(np.mean(cover_counts)) if cover_counts else 0.0,
        "min_pois_per_vertex": int(min(vertex_counts)) if vertex_counts else 0,
        "max_pois_per_vertex": int(max(vertex_counts)) if vertex_counts else 0,
        "mean_pois_per_vertex": float(np.mean(vertex_counts)) if vertex_counts else 0.0,
    }


def validate_route_poi_coverage(
    route_vertex_sequence: Sequence[VertexId],
    instance: InspectionPlanningInstance,
) -> Tuple[Dict[POIId, bool], List[POIId]]:
    """
    Validate which POIs are covered by a route represented as a sequence of graph vertices.

    Returns:
        poi_covered_map: poi_id -> bool
        uncovered_pois: list of POI ids not seen from any route vertex
    """
    covered_pois = set()

    for node in route_vertex_sequence:
        covered_pois.update(instance.vertex_to_pois.get(node, []))

    poi_covered_map = {
        poi.poi_id: (poi.poi_id in covered_pois)
        for poi in instance.poi_set.pois
    }
    uncovered_pois = [poi_id for poi_id, covered in poi_covered_map.items() if not covered]

    return poi_covered_map, uncovered_pois


def _set_axes_equal(ax: plt.Axes) -> None:
    """
    Make 3D plot axes approximately equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


# def visualize_inspection_instance(
#     instance: InspectionPlanningInstance,
#     bridge_mesh: Optional[trimesh.Trimesh] = None,
#     show_bridge: bool = True,
#     show_vertices: bool = True,
#     show_pois: bool = True,
#     show_visibility: bool = False,
#     only_vertices_with_visibility: bool = False,
#     max_visibility_edges: Optional[int] = None,
#     vertex_size: float = 8.0,
#     poi_size: float = 30.0,
#     title: str = "",
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Lightweight 3D visualization.
#
#     Notes:
#     - vertices are plotted using only xyz extracted from state[:3]
#     - visibility edges are optional because they can be visually dense
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#
#     if show_bridge and bridge_mesh is not None:
#         vertices = bridge_mesh.vertices
#         faces = bridge_mesh.faces
#         ax.plot_trisurf(
#             vertices[:, 0],
#             vertices[:, 1],
#             faces,
#             vertices[:, 2],
#             alpha=0.15,
#             linewidth=0.2,
#             edgecolor="gray",
#         )
#
#     if show_vertices and instance.vertex_xyz:
#         if only_vertices_with_visibility:
#             selected_nodes = [
#                 node for node, poi_ids in instance.vertex_to_pois.items()
#                 if len(poi_ids) > 0
#             ]
#         else:
#             selected_nodes = list(instance.vertex_xyz.keys())
#
#         if selected_nodes:
#             xyz = np.asarray([instance.vertex_xyz[node] for node in selected_nodes], dtype=float)
#             ax.scatter(
#                 xyz[:, 0],
#                 xyz[:, 1],
#                 xyz[:, 2],
#                 s=vertex_size,
#                 alpha=0.5,
#                 label="Roadmap vertices",
#             )
#
#     if show_pois and instance.poi_set.pois:
#         poi_xyz = instance.poi_set.as_array()
#         ax.scatter(
#             poi_xyz[:, 0],
#             poi_xyz[:, 1],
#             poi_xyz[:, 2],
#             s=poi_size,
#             alpha=0.9,
#             label="POIs",
#         )
#
#     if show_visibility and instance.poi_set.pois:
#         visibility_segments: List[Tuple[np.ndarray, np.ndarray]] = []
#         for poi in instance.poi_set.pois:
#             for node in instance.poi_to_vertices.get(poi.poi_id, []):
#                 visibility_segments.append((poi.xyz, instance.vertex_xyz[node]))
#
#         if max_visibility_edges is not None:
#             visibility_segments = visibility_segments[:max_visibility_edges]
#
#         for poi_xyz, vertex_xyz in visibility_segments:
#             ax.plot(
#                 [poi_xyz[0], vertex_xyz[0]],
#                 [poi_xyz[1], vertex_xyz[1]],
#                 [poi_xyz[2], vertex_xyz[2]],
#                 linestyle="--",
#                 linewidth=0.6,
#                 alpha=0.35,
#             )
#
#     # ax.set_xticklabels([])
#     # ax.set_yticklabels([])
#     # ax.set_zticklabels([])
#
#     ax.set_title(title)
#     # ax.set_xlabel("X")
#     # ax.set_ylabel("Y")
#     # ax.set_zlabel("Z")
#     _set_axes_equal(ax)
#
#     handles, labels = ax.get_legend_handles_labels()
#     if labels:
#         ax.legend()
#
#     plt.tight_layout()
#     return fig, ax

def visualize_inspection_task(
    object_mesh: trimesh.Trimesh,
    poi_set: set,
    G,
    start_node=None,
    solution_edges=None,
    show_graph_nodes=False,
    show_graph_edges=False,
    poi_size: float = 10.0,
    vertex_size: float = 5.0,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Lightweight 3D visualization.

    Notes:
    - vertices are plotted using only xyz extracted from state[:3]
    - visibility edges are optional because they can be visually dense
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    vertices = object_mesh.vertices
    faces = object_mesh.faces

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        faces,
        vertices[:, 2],
        alpha=0.15,
        linewidth=0.2,
        edgecolor="gray",
    )

    # Make x/y/z axes visually proportional
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ranges = maxs - mins

    ax.set_box_aspect(ranges)

    poi_xyz = poi_set.as_array()
    ax.scatter(
        poi_xyz[:, 0],
        poi_xyz[:, 1],
        poi_xyz[:, 2],
        s=poi_size,
        alpha=0.9,
        color='orange',
        label="POIs",
    )

    if show_graph_nodes:
        positions = list(nx.get_node_attributes(G, "pos").values())
        xyz = np.asarray(positions, dtype=float)

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=vertex_size,
            alpha=0.5,
            label="Roadmap vertices",
        )

    if show_graph_edges:
        positions = list(nx.get_node_attributes(G, "pos").values())
        segments = [
            [np.asarray(positions[u], dtype=float), np.asarray(positions[v], dtype=float)]
            for u, v in G.edges
        ]

        edge_collection = Line3DCollection(
            segments,
            colors="gray",
            linewidths=0.8,
            alpha=0.4,
            label="Roadmap edges",
        )
        ax.add_collection3d(edge_collection)

    if start_node is not None:
        start_pos = G.nodes[start_node]['pos']
        xyz = np.asarray(start_pos, dtype=float)

        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            s=vertex_size*3,
            alpha=1,
            label="Start vertex",
            marker='x',
            color='red'
        )

    if solution_edges is not None:
        positions = list(nx.get_node_attributes(G, "pos").values())
        segments = [
            [np.asarray(positions[u], dtype=float), np.asarray(positions[v], dtype=float)]
            for u, v in solution_edges
        ]

        segments_np = np.array(segments)

        # Extract start points (X, Y, Z)
        starts = segments_np[:, 0, :]
        X, Y, Z = starts[:, 0], starts[:, 1], starts[:, 2]

        # Extract end points and calculate direction vectors (U, V, W)
        ends = segments_np[:, 1, :]
        vectors = ends - starts
        U, V, W = vectors[:, 0], vectors[:, 1], vectors[:, 2]

        # Plot using quiver
        ax.quiver(
            X, Y, Z,
            U, V, W,
            color="green",
            linewidth=2,
            alpha=1,
            arrow_length_ratio=0.2,  # Controls the size of the arrowhead relative to the vector length
            label="solution edges"
        )

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])


    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # _set_axes_equal(ax)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    plt.tight_layout()
    return fig, ax



def visualize_inspection_task_pybullet(
    object_mesh,
    poi_set,
    G,
    start_node=None,
    solution_edges=None,
    visibility_vertex=None,
    vertex_to_pois=None,
    show_graph_nodes=False,
    show_graph_edges=False,
    show_solution_visibility=False,
    poi_size: float = 6.0,
    vertex_size: float = 4,
    mesh_color=(0.72, 0.75, 0.78, 1.0),
    edge_color=(0.25, 0.35, 0.50),
    solution_color=(0.05, 0.8, 0.2),
):
    """
    Display an inspection task in an interactive PyBullet GUI.

    Returns
    -------
    client_id:
        PyBullet physics client ID.

    Notes
    -----
    - This function opens the window but does not block.
    - Call run_pybullet_viewer(client_id) afterward to keep it open.
    - The graph node positions are read from G.nodes[node]["pos"].
    - If visibility_vertex is supplied, it is highlighted and lines are drawn
      to the POIs listed for it in vertex_to_pois.
    """
    client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(),
        physicsClientId=client_id,
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
        physicsClientId=client_id,
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS,
        1,
        physicsClientId=client_id,
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_RGB_BUFFER_PREVIEW,
        0,
        physicsClientId=client_id,
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        0,
        physicsClientId=client_id,
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        0,
        physicsClientId=client_id,
    )

    # Background color is supported in recent PyBullet versions.
    # try:
    #     p.configureDebugVisualizer(
    #         p.COV_ENABLE_GUI,
    #         0,
    #         lightPosition=[1, 1, 3],
    #         rgbBackground=background_color,
    #         physicsClientId=client_id,
    #     )
    # except TypeError:
    #     pass

    # ---------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------

    mesh_vertices = np.asarray(object_mesh.vertices, dtype=float)
    mesh_faces = np.asarray(object_mesh.faces, dtype=np.int32)

    mins = mesh_vertices.min(axis=0)
    maxs = mesh_vertices.max(axis=0)

    plane_id = p.loadURDF(
        "plane.urdf",
        basePosition=[0, 0, float(mins[2]) - 0.02],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=mesh_vertices.tolist(),
        indices=mesh_faces.reshape(-1).tolist(),
        rgbaColor=[0.65, 0.65, 0.65, 1.0],
        specularColor=[0.2, 0.2, 0.2],
        physicsClientId=client_id,
    )

    mesh_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0],
        physicsClientId=client_id,
    )

    p.changeVisualShape(
        mesh_body,
        -1,
        rgbaColor=list(mesh_color),
        specularColor=[0.05, 0.05, 0.05],
        physicsClientId=client_id,
    )

    # ---------------------------------------------------------
    # Graph positions
    # ---------------------------------------------------------

    positions = {
        node: np.asarray(pos, dtype=float)[:3]
        for node, pos in nx.get_node_attributes(G, "pos").items()
    }

    missing_positions = set(G.nodes) - set(positions)

    if missing_positions:
        raise ValueError(
            f"{len(missing_positions)} graph nodes have no 'pos' attribute. "
            f"Examples: {list(missing_positions)[:5]}"
        )

    # ---------------------------------------------------------
    # POIs
    # ---------------------------------------------------------

    poi_xyz = np.asarray(poi_set.as_array(), dtype=float)

    if len(poi_xyz) > 0:
        p.addUserDebugPoints(
            pointPositions=poi_xyz[:, :3].tolist(),
            pointColorsRGB=[[1.0, 0.5, 0.0]] * len(poi_xyz),
            pointSize=poi_size,
            physicsClientId=client_id,
        )

    # ---------------------------------------------------------
    # Roadmap vertices
    # ---------------------------------------------------------
    if show_graph_nodes and positions:
        graph_xyz = np.asarray(list(positions.values()))

        p.addUserDebugPoints(
            pointPositions=graph_xyz.tolist(),
            pointColorsRGB=[[0.0, 0.2, 0.9]] * len(graph_xyz),
            pointSize=vertex_size,
            physicsClientId=client_id,
        )

    # ---------------------------------------------------------
    # Roadmap edges
    # ---------------------------------------------------------

    if show_graph_edges:
        for u, v in G.edges:
            p.addUserDebugLine(
                positions[u].tolist(),
                positions[v].tolist(),
                lineColorRGB=[0.5, 0.5, 0.5],
                lineWidth=0.5,
                physicsClientId=client_id,
            )

    # ---------------------------------------------------------
    # Visibility from a selected roadmap vertex
    # ---------------------------------------------------------

    if visibility_vertex is not None or show_solution_visibility:

        if visibility_vertex is not None:
            vertices_to_show = [visibility_vertex]
        else:
            solution_vertices = set()
            for u, v in solution_edges:
                solution_vertices.add(u)
                solution_vertices.add(v)
            vertices_to_show = solution_vertices

        poi_positions = poi_set.as_dict()

        for v in vertices_to_show:
            visibility_pos = positions[v]
            p.addUserDebugPoints(
                pointPositions=[visibility_pos.tolist()],
                pointColorsRGB=[[0.0, 0.2, 1.0]],
                pointSize=vertex_size * 3,
                physicsClientId=client_id,
            )

            visible_poi_ids = vertex_to_pois.get(v, [])
            for poi_id in visible_poi_ids:
                p.addUserDebugLine(
                    lineFromXYZ=visibility_pos.tolist(),
                    lineToXYZ=np.asarray(poi_positions[poi_id], dtype=float).tolist(),
                    lineColorRGB=[0.5, 0.0, 0.5],
                    lineWidth=1.0,
                    physicsClientId=client_id,
                )

    # ---------------------------------------------------------
    # Start vertex
    # ---------------------------------------------------------

    if start_node is not None:
        if start_node not in positions:
            raise ValueError(
                f"Start node {start_node!r} has no 'pos' attribute."
            )

        p.addUserDebugPoints(
            pointPositions=[positions[start_node].tolist()],
            pointColorsRGB=[[1.0, 0.0, 0.0]],
            pointSize=vertex_size * 3,
            physicsClientId=client_id,
        )

        # PyBullet debug points have no marker types, so add a 3D cross.
        _add_debug_cross(
            positions[start_node],
            size=_scene_scale(mesh_vertices) * 0.015,
            color=(1.0, 0.0, 0.0),
            line_width=3.0,
            client_id=client_id,
        )

    # ---------------------------------------------------------
    # Directed solution edges
    # ---------------------------------------------------------

    if solution_edges is not None:
        solution_edges = list(solution_edges)

        arrow_size = _scene_scale(mesh_vertices) * 0.02

        for u, v in solution_edges:
            _add_debug_arrow(
                start=positions[u],
                end=positions[v],
                color=solution_color,
                line_width=3.0,
                arrow_size=arrow_size,
                client_id=client_id,
            )

    # ---------------------------------------------------------
    # Camera
    # ---------------------------------------------------------

    center = 0.5 * (mins + maxs)
    largest_extent = float(np.max(maxs - mins))

    p.resetDebugVisualizerCamera(
        cameraDistance=1.15 * largest_extent,
        cameraYaw=35,
        cameraPitch=-12,
        cameraTargetPosition=center.tolist(),
        physicsClientId=client_id,
    )

    return client_id, mesh_body


def _scene_scale(vertices):
    """Return a representative length scale for the scene."""
    ranges = np.ptp(vertices, axis=0)
    return max(float(np.linalg.norm(ranges)), 1e-6)


def _add_debug_cross(
    position,
    size,
    color,
    line_width,
    client_id,
):
    """Draw a small 3D cross centered at position."""
    position = np.asarray(position, dtype=float)

    for axis in np.eye(3):
        p.addUserDebugLine(
            (position - size * axis).tolist(),
            (position + size * axis).tolist(),
            lineColorRGB=list(color),
            lineWidth=line_width,
            physicsClientId=client_id,
        )


def _add_debug_arrow(
    start,
    end,
    color,
    line_width,
    arrow_size,
    client_id,
):
    """Draw a line with a 3D arrowhead."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-12:
        return

    direction /= length

    p.addUserDebugLine(
        start.tolist(),
        end.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )

    # Choose a reference vector that is not parallel to the arrow.
    reference = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    perpendicular = np.cross(direction, reference)
    perpendicular /= np.linalg.norm(perpendicular)

    arrow_size = min(arrow_size, 0.25 * length)
    arrow_base = end - arrow_size * direction
    arrow_width = 0.5 * arrow_size

    head_1 = arrow_base + arrow_width * perpendicular
    head_2 = arrow_base - arrow_width * perpendicular

    p.addUserDebugLine(
        end.tolist(),
        head_1.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )

    p.addUserDebugLine(
        end.tolist(),
        head_2.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )

def _scene_scale(vertices):
    """Return a representative length scale for the scene."""
    ranges = np.ptp(vertices, axis=0)
    return max(float(np.linalg.norm(ranges)), 1e-6)


def _add_debug_cross(
    position,
    size,
    color,
    line_width,
    client_id,
):
    """Draw a small 3D cross centered at position."""
    position = np.asarray(position, dtype=float)

    for axis in np.eye(3):
        p.addUserDebugLine(
            (position - size * axis).tolist(),
            (position + size * axis).tolist(),
            lineColorRGB=list(color),
            lineWidth=line_width,
            physicsClientId=client_id,
        )


def _add_debug_arrow(
    start,
    end,
    color,
    line_width,
    arrow_size,
    client_id,
):
    """Draw a line with a 3D arrowhead."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-12:
        return

    direction /= length

    p.addUserDebugLine(
        start.tolist(),
        end.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )

    # Choose a reference vector that is not parallel to the arrow.
    reference = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    perpendicular = np.cross(direction, reference)
    perpendicular /= np.linalg.norm(perpendicular)

    arrow_size = min(arrow_size, 0.25 * length)
    arrow_base = end - arrow_size * direction
    arrow_width = 0.5 * arrow_size

    head_1 = arrow_base + arrow_width * perpendicular
    head_2 = arrow_base - arrow_width * perpendicular

    p.addUserDebugLine(
        end.tolist(),
        head_1.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )

    p.addUserDebugLine(
        end.tolist(),
        head_2.tolist(),
        lineColorRGB=list(color),
        lineWidth=line_width,
        physicsClientId=client_id,
    )


def run_pybullet_viewer(client_id):
    """Keep the PyBullet viewer responsive until its window is closed."""
    try:
        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)

