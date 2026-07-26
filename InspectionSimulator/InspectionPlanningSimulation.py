from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import networkx as nx
import numpy as np
import math
import trimesh
from scipy.spatial import cKDTree

from InspectionSimulator.GridGraphSpanner import (
    load_object_env, Bounds3D, PlannerConfig, build_grid_motion_planning_graph)

VertexId = Hashable
POIId = int


@dataclass
class POI:
    """A point of interest on the bridge structure."""
    poi_id: POIId
    xyz: np.ndarray  # shape (3,)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3,)


@dataclass
class POISet:
    """
    Separate storage for POIs.

    This is intentionally external to the graph, so it can later be used for:
    - visualization
    - route validation / coverage checking
    - inspection planning objectives
    """
    pois: List[POI]

    def as_array(self) -> np.ndarray:
        if not self.pois:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([poi.xyz for poi in self.pois], dtype=float)

    def as_dict(self) -> Dict[POIId, np.ndarray]:
        return {poi.poi_id: poi.xyz.copy() for poi in self.pois}


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

# TODO - this is not considering occlusions.. very very basic, must be replaced.
def compute_visibility_by_distance(
    graph: nx.Graph,
    poi_set: POISet,
    visibility_threshold: float,
) -> Tuple[Dict[POIId, List[VertexId]], Dict[VertexId, List[POIId]], Dict[VertexId, np.ndarray]]:
    """
    Compute visibility relation:
        POI is visible from vertex iff Euclidean distance between POI xyz
        and vertex xyz is <= visibility_threshold.

    Uses a KD-tree over graph vertex xyz for efficiency.
    """
    if visibility_threshold < 0:
        raise ValueError(f"visibility_threshold must be non-negative, got {visibility_threshold}")

    nodes = list(graph.nodes)
    vertex_xyz = np.asarray(graph.nodes, dtype=float)

    if not nodes:
        return (
            {poi.poi_id: [] for poi in poi_set.pois},
            {},
            {},
        )

    xyz_array = np.asarray(nodes, dtype=float)
    tree = cKDTree(xyz_array)

    poi_to_vertices: Dict[POIId, List[VertexId]] = {poi.poi_id: [] for poi in poi_set.pois}
    vertex_to_pois: Dict[VertexId, List[POIId]] = {node: [] for node in nodes}

    for poi in poi_set.pois:
        idxs = tree.query_ball_point(poi.xyz, r=visibility_threshold)
        visible_nodes = [nodes[i] for i in idxs]

        poi_to_vertices[poi.poi_id] = visible_nodes
        for node in visible_nodes:
            vertex_to_pois[node].append(poi.poi_id)

    return poi_to_vertices, vertex_to_pois, vertex_xyz

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

    poi_to_vertices, vertex_to_pois, vertex_xyz = compute_visibility_by_distance(
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


def visualize_inspection_instance(
    instance: InspectionPlanningInstance,
    bridge_mesh: Optional[trimesh.Trimesh] = None,
    show_bridge: bool = True,
    show_vertices: bool = True,
    show_pois: bool = True,
    show_visibility: bool = False,
    only_vertices_with_visibility: bool = False,
    max_visibility_edges: Optional[int] = None,
    vertex_size: float = 8.0,
    poi_size: float = 30.0,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Lightweight 3D visualization.

    Notes:
    - vertices are plotted using only xyz extracted from state[:3]
    - visibility edges are optional because they can be visually dense
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if show_bridge and bridge_mesh is not None:
        vertices = bridge_mesh.vertices
        faces = bridge_mesh.faces
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            faces,
            vertices[:, 2],
            alpha=0.15,
            linewidth=0.2,
            edgecolor="gray",
        )

    if show_vertices and instance.vertex_xyz:
        if only_vertices_with_visibility:
            selected_nodes = [
                node for node, poi_ids in instance.vertex_to_pois.items()
                if len(poi_ids) > 0
            ]
        else:
            selected_nodes = list(instance.vertex_xyz.keys())

        if selected_nodes:
            xyz = np.asarray([instance.vertex_xyz[node] for node in selected_nodes], dtype=float)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=vertex_size,
                alpha=0.5,
                label="Roadmap vertices",
            )

    if show_pois and instance.poi_set.pois:
        poi_xyz = instance.poi_set.as_array()
        ax.scatter(
            poi_xyz[:, 0],
            poi_xyz[:, 1],
            poi_xyz[:, 2],
            s=poi_size,
            alpha=0.9,
            label="POIs",
        )

    if show_visibility and instance.poi_set.pois:
        visibility_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for poi in instance.poi_set.pois:
            for node in instance.poi_to_vertices.get(poi.poi_id, []):
                visibility_segments.append((poi.xyz, instance.vertex_xyz[node]))

        if max_visibility_edges is not None:
            visibility_segments = visibility_segments[:max_visibility_edges]

        for poi_xyz, vertex_xyz in visibility_segments:
            ax.plot(
                [poi_xyz[0], vertex_xyz[0]],
                [poi_xyz[1], vertex_xyz[1]],
                [poi_xyz[2], vertex_xyz[2]],
                linestyle="--",
                linewidth=0.6,
                alpha=0.35,
            )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title(title)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    _set_axes_equal(ax)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    plt.tight_layout()
    return fig, ax

def visualize_inspection_task(
    object_mesh: trimesh.Trimesh,
    poi_set: set,
    G,
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
    fig = plt.figure(figsize=(10, 8))
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
        selected_nodes = G.nodes()
        xyz = np.asarray(selected_nodes, dtype=float)
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=vertex_size,
            alpha=0.5,
            label="Roadmap vertices",
        )

    if show_graph_edges:
        selected_edges = G.edges()
        segments = [
            [np.asarray(u, dtype=float), np.asarray(v, dtype=float)]
            for u, v in selected_edges
        ]

        edge_collection = Line3DCollection(
            segments,
            colors="gray",
            linewidths=0.8,
            alpha=0.4,
            label="Roadmap edges",
        )
        ax.add_collection3d(edge_collection)


    if solution_edges is not None:
        segments_np = np.array(solution_edges)

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

    sol_segments = [
        [np.asarray(u, dtype=float), np.asarray(v, dtype=float)]
        for u, v in solution_edges
    ]

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    plt.tight_layout()
    return fig, ax

def env_config() -> PlannerConfig:
    """Small example config; tune this to your bridge model placement."""
    return PlannerConfig(
        obj_path=r"../assets/OBJ/water_tower.obj",
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
if __name__ == '__main__':
    config = env_config()
    inspection_object = load_object_env(
        obj_path=config.obj_path,
        scale=config.scale,
        translation=config.translation,
        rotation_rpy=config.rotation_rpy,
    )
    object_mesh = inspection_object.mesh
    seed = 0
    num_pois = 100
    poi_set = sample_pois_on_mesh_surface(object_mesh=object_mesh, num_pois=num_pois, seed=seed)

    # TODO - separate mp config from environment config
    planning_artifacts = build_grid_motion_planning_graph(config)
    G = planning_artifacts.graph

    visibility_threshold_dist = 5
    visibility_rel = compute_visibility_by_distance(G, poi_set, visibility_threshold_dist)

    visualize_inspection_task(object_mesh, poi_set, G)
    plt.show()

    pass
    #
    # artifacts = build_grid_motion_planning_graph(cfg)
    # mp_graph = artifacts.graph
    # obst_mesh = artifacts.obstacle.mesh
    #
    # num_pois = 500
    # vis_th = 1
    #
    # ip_problem = build_inspection_planning_instance(mp_graph, obst_mesh, num_pois, vis_th)
    #
    # save_path = f"./inspection_experiments/gip_instance_N{G.number_of_nodes()}_K{num_pois}_bridge.pkl"
    #
    # S = ip_problem.poi_to_vertices
    # vertex_poi_vis = ip_problem.vertex_to_pois
    # I = set(S.keys())
    #
    # meta = {
    #     # "poi_set": ip_problem.poi_set,
    #     # "visibility_threshold": ip_problem.visibility_threshold,
    # }
    #
    # save_simulated_instance(
    #     save_path,
    #     G=G,
    #     I=I,
    #     S=S,
    #     vertex_poi_vis=vertex_poi_vis,
    #     root=0,
    #     meta=meta,
    # )
    # print(f"Instance saved to {save_path}")
    #
    # visualize_inspection_instance(ip_problem, object_mesh_for_planning,
    #                               show_visibility=True,
    #                               only_vertices_with_visibility=True,
    #                               max_visibility_edges=3000
    #                               )
    #
    # plt.show()