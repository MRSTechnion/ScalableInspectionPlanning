from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, List, Tuple

import networkx as nx
import numpy as np

def load_saved_graph_npz(npz_path: str | Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)

    nodes = data["nodes"]
    nodes_full = data["nodes_full"] if "nodes_full" in data.files else None
    embeddings = data["embeddings"] if "embeddings" in data.files else None
    adj = data["adj"].item()
    n_nodes = int(data["n_nodes"])

    return {
        "nodes": nodes,
        "nodes_full": nodes_full,
        "embeddings": embeddings,
        "adj": adj,
        "n_nodes": n_nodes,
    }


def load_metadata(metadata_path: str | Path) -> Dict[str, Any]:
    with open(metadata_path, "r") as f:
        return json.load(f)


def convert_to_networkx(
    npz_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    *,
    undirected: bool = True,
    include_embedding: bool = False,
    include_state_dist: bool = False,
) -> nx.Graph | nx.DiGraph:
    payload = load_saved_graph_npz(npz_path)
    metadata = load_metadata(metadata_path) if metadata_path is not None else {}

    nodes = payload["nodes"]
    nodes_full = payload["nodes_full"]
    embeddings = payload["embeddings"]
    adj = payload["adj"]
    n_nodes = payload["n_nodes"]

    G = nx.Graph() if undirected else nx.DiGraph()

    G.graph.update(metadata)
    G.graph["source_npz"] = str(npz_path)
    G.graph["n_nodes_loaded"] = n_nodes

    for i in range(n_nodes):
        attrs = {
            "state": nodes[i].astype(np.float32),
            "state_dim": int(nodes.shape[1]),
        }
        if nodes_full is not None:
            attrs["state_full"] = nodes_full[i].astype(np.float32)
        if include_embedding and embeddings is not None:
            attrs["embedding"] = embeddings[i].astype(np.float32)

        G.add_node(i, **attrs)

    for i, nbrs in adj.items():
        for j, w in nbrs:
            w = float(w)

            edge_attrs = {"weight": w}

            if include_state_dist:
                edge_attrs["distance_state"] = float(
                    np.linalg.norm(nodes[i] - nodes[j])
                )

            # if embeddings is not None:
            #     edge_attrs["distance_embedding"] = float(
            #         np.linalg.norm(embeddings[i] - embeddings[j])
            #     )

            if undirected:
                if G.has_edge(i, j):
                    # Keep the smaller weight if duplicates disagree slightly.
                    if w < G[i][j]["weight"]:
                        G[i][j].update(edge_attrs)
                else:
                    G.add_edge(i, j, **edge_attrs)
            else:
                G.add_edge(i, j, **edge_attrs)

    return G


def edges_to_spatial_trajectory(
    G: nx.Graph | nx.DiGraph,
    edge_path: Iterable[tuple[int, int]],
) -> tuple[list[np.ndarray], list[float]]:
    """
    Convert a path given as edges [(v1, v2), (v2, v3), ...] into:
      1. a list of xyz waypoints
      2. a cumulative sum of edge weights

    Returns
    -------
    xyz_points :
        [xyz(v1), xyz(v2), ..., xyz(vk)]
    cumulative_times :
        [0.0, w(v1,v2), w(v1,v2)+w(v2,v3), ...]
        same length as xyz_points
    """
    edge_path = list(edge_path)
    if not edge_path:
        return [], []

    # Validate continuity
    for (u1, v1), (u2, v2) in zip(edge_path[:-1], edge_path[1:]):
        if v1 != u2:
            raise ValueError(
                f"Edge path is not continuous: {(u1, v1)} is followed by {(u2, v2)}"
            )

    first_u, first_v = edge_path[0]

    if first_u not in G:
        raise KeyError(f"Start node {first_u} not in graph")
    if first_v not in G:
        raise KeyError(f"Node {first_v} not in graph")

    # First waypoint
    xyz_points: List[np.ndarray] = [np.asarray(G.nodes[first_u]["state"][:3], dtype=float)]
    cumulative_times: List[float] = [0.0]

    running_time = 0.0

    for u, v in edge_path:
        if not G.has_edge(u, v):
            raise KeyError(f"Edge ({u}, {v}) not found in graph")

        w = float(G[u][v]["weight"])
        running_time += w

        xyz = np.asarray(G.nodes[v]["state"][:3], dtype=float)
        xyz_points.append(xyz)
        cumulative_times.append(running_time)

    return xyz_points, cumulative_times

if __name__ == '__main__':
    data_path = r'/home/adir/Downloads/prm_n1000_k30_d2_5_s0_gas_gws48.npz'

    G = convert_to_networkx(data_path, undirected=False)


    # -----
    solver_edges = [(5, 145), (145, 18), (18, 10)]

    xyz_points, cumulative_times = edges_to_spatial_trajectory(G, solver_edges)

    print(f"{solver_edges=}")
    for p, t in zip(xyz_points, cumulative_times):
        print(f"xyz={p}, time={t:.3f}")
