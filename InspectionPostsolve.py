from InspectionHeuristic import TM_solver_groups_scipy
from Utils import IP_to_Group
from Readers import IRIS_reader
from SolutionValidation import validate_solution_groups

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

def _build_csr_undirected(G, weight="weight", assume_int_0_to_n_minus_1=False):
    nodes = list(G.nodes())
    n = len(nodes)

    # Safe identity mapping only if nodes are exactly {0..n-1}
    if assume_int_0_to_n_minus_1 and set(nodes) == set(range(n)):
        node_to_idx = None
        idx_to_node = None
        def idx(v): return v
    else:
        node_to_idx = {v: i for i, v in enumerate(nodes)}
        idx_to_node = nodes
        def idx(v): return node_to_idx[v]

    rows, cols, data = [], [], []
    for u, v, attrs in G.edges(data=True):
        w = attrs.get(weight, 1.0)
        iu, iv = idx(u), idx(v)
        rows.append(iu); cols.append(iv); data.append(w)
        rows.append(iv); cols.append(iu); data.append(w)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return A, node_to_idx, idx_to_node


def _reconstruct_path_from_pred_row(pred_row, src_i, dst_i):
    if src_i == dst_i:
        return [src_i]
    if pred_row[dst_i] == -9999:
        return None
    path = [dst_i]
    cur = dst_i
    while cur != src_i:
        cur = int(pred_row[cur])
        if cur == -9999:
            return None
        path.append(cur)
    path.reverse()
    return path


def _greedy_matching_from_dist_matrix(odd_nodes, dist_odd):
    """
    odd_nodes: list of node labels (len=k)
    dist_odd:  (k, k) float matrix distances among odd nodes

    Returns: list of matched pairs [(u,v), ...]
    """
    k = len(odd_nodes)
    if k < 2:
        return []

    iu, iv = np.triu_indices(k, k=1)
    w = dist_odd[iu, iv]

    # sort all candidate pairs by weight ascending
    order = np.argsort(w, kind="mergesort")  # stable
    unmatched = np.ones(k, dtype=bool)
    pairs = []

    for idx in order:
        i = int(iu[idx])
        j = int(iv[idx])
        if unmatched[i] and unmatched[j]:
            unmatched[i] = False
            unmatched[j] = False
            pairs.append((odd_nodes[i], odd_nodes[j]))
            # early stop if all matched
            if not unmatched.any():
                break

    # If k is odd (shouldn't happen for odd-degree set in a connected graph),
    # one vertex may remain unmatched. We simply leave it.
    return pairs

def ST_to_tour_christofides_scipy(
    G,
    solution_edges,
    start=None,
    return_to_start=True,
    *,
    weight="weight",
    assume_int_0_to_n_minus_1=False
):
    # --- Build forest H ---
    H = nx.Graph()
    H.add_edges_from(solution_edges)

    if H.number_of_edges() == 0:
        return [], 0.0, [], []

    # --- Choose start ---
    if start is None or start not in H:
        u, _ = next(iter(H.edges()))
        start = u

    # --- Extract component ---
    tree_nodes = set(nx.node_connected_component(H, start))
    if not tree_nodes:
        return [], 0.0, [], []

    Hc = H.subgraph(tree_nodes).copy()

    # --- Odd-degree vertices ---
    odd = [v for v in Hc.nodes() if Hc.degree(v) % 2 == 1]

    # --- Build Eulerian multigraph base ---
    M = nx.MultiGraph()
    M.add_edges_from(Hc.edges(data=True))

    # --- Build CSR + run multi-source Dijkstra ---
    A, node_to_idx, idx_to_node = _build_csr_undirected(
        G, weight=weight, assume_int_0_to_n_minus_1=assume_int_0_to_n_minus_1
    )

    if node_to_idx is None:
        def to_i(v): return v
        def to_v(i): return i
    else:
        def to_i(v): return node_to_idx[v]
        def to_v(i): return idx_to_node[i]

    sources = sorted(set(tree_nodes) | set(odd) | {start})
    src_idx = np.array([to_i(s) for s in sources], dtype=int)
    src_pos = {s: k for k, s in enumerate(sources)}

    dist, pred = dijkstra(
        A,
        directed=False,
        indices=src_idx,
        return_predecessors=True
    )

    # --- Minimum-weight matching on odd vertices (instead of greedy) ---
    matching_pairs = []
    matching_edges = []

    if len(odd) >= 2:
        odd = list(odd)

        # Build complete graph on odd vertices with weights = shortest-path distances
        K = nx.Graph()
        K.add_nodes_from(odd)

        # Use the precomputed 'dist' matrix rows (sources) to fill weights
        # Ensure src_pos covers odd vertices (it does: sources includes set(odd))
        for i in range(len(odd)):
            u = odd[i]
            row = src_pos[u]
            for j in range(i + 1, len(odd)):
                v = odd[j]
                w = float(dist[row, to_i(v)])  # shortest-path distance u->v
                if np.isfinite(w):
                    K.add_edge(u, v, weight=w)

        # Minimum-weight matching (maxcardinality forces pairing as much as possible)
        mwm = nx.algorithms.matching.min_weight_matching(
            K, weight="weight"
        )

        # Normalize as a list of (u, v) tuples
        matching_pairs = [(u, v) for (u, v) in mwm]

        # Add the corresponding shortest paths into the Eulerian multigraph
        for u, v in matching_pairs:
            row = src_pos[u]
            path_idx = _reconstruct_path_from_pred_row(
                pred[row], to_i(u), to_i(v)
            )
            if path_idx is None:
                continue
            nodes = [to_v(i) for i in path_idx]
            for a, b in zip(nodes, nodes[1:]):
                M.add_edge(a, b, weight=G[a][b][weight])
                matching_edges.append((a, b))

    # --- Euler tour/path ---
    if return_to_start:
        circuit_edges = list(nx.eulerian_circuit(M, source=start))
        euler_nodes = [circuit_edges[0][0]] + [v for _, v in circuit_edges]
    else:
        path_edges = list(nx.eulerian_path(M, source=start))
        euler_nodes = [path_edges[0][0]] + [v for _, v in path_edges]

    # --- Shortcut over tree_nodes ---
    seen = set()
    ordered = []
    for n in euler_nodes:
        if n in tree_nodes and n not in seen:
            seen.add(n)
            ordered.append(n)

    if not ordered:
        return [], 0.0, matching_pairs, matching_edges

    if start in ordered:
        sidx = ordered.index(start)
        ordered = ordered[sidx:] + ordered[:sidx]

    # --- Stitch final tour using SciPy predecessors ---
    visit_sequence = ordered + ([ordered[0]] if return_to_start else [])

    tour_nodes = []
    for a, b in zip(visit_sequence, visit_sequence[1:]):
        if a == b:
            continue
        row = src_pos[a]
        path_idx = _reconstruct_path_from_pred_row(
            pred[row], to_i(a), to_i(b)
        )
        if path_idx is None:
            continue
        sp_nodes = [to_v(i) for i in path_idx]
        if not tour_nodes:
            tour_nodes.extend(sp_nodes)
        else:
            tour_nodes.extend(sp_nodes[1:])

    # --- Edges + weight ---
    tour_edges = []
    tour_weight = 0.0
    for u, v in zip(tour_nodes, tour_nodes[1:]):
        tour_edges.append((u, v))
    #     tour_weight += G[u][v][weight]

    return tour_edges, tour_weight, matching_pairs, matching_edges

def ST_to_tour_christofides_scipy_greedy(
        G,
        solution_edges,
        start=None,
        return_to_start=True,
        *,
        weight="weight",
        assume_int_0_to_n_minus_1=False
):
        # --- Build forest H ---
        H = nx.Graph()
        H.add_edges_from(solution_edges)

        if H.number_of_edges() == 0:
            return [], 0.0, [], []

        # --- Choose start ---
        if start is None or start not in H:
            u, _ = next(iter(H.edges()))
            start = u

        # --- Extract component ---
        tree_nodes = set(nx.node_connected_component(H, start))
        if not tree_nodes:
            return [], 0.0, [], []

        Hc = H.subgraph(tree_nodes).copy()

        # --- Odd-degree vertices ---
        odd = [v for v in Hc.nodes() if Hc.degree(v) % 2 == 1]

        # --- Build Eulerian multigraph base ---
        M = nx.MultiGraph()
        M.add_edges_from(Hc.edges(data=True))

        # --- Build CSR + run multi-source Dijkstra ---
        A, node_to_idx, idx_to_node = _build_csr_undirected(
            G, weight=weight, assume_int_0_to_n_minus_1=assume_int_0_to_n_minus_1
        )

        if node_to_idx is None:
            def to_i(v):
                return v

            def to_v(i):
                return i
        else:
            def to_i(v):
                return node_to_idx[v]

            def to_v(i):
                return idx_to_node[i]

        sources = sorted(set(tree_nodes) | set(odd) | {start})
        src_idx = np.array([to_i(s) for s in sources], dtype=int)
        src_pos = {s: k for k, s in enumerate(sources)}

        dist, pred = dijkstra(
            A,
            directed=False,
            indices=src_idx,
            return_predecessors=True
        )

        # --- Greedy matching on odd vertices ---
        matching_pairs = []
        matching_edges = []

        if len(odd) >= 2:
            odd = list(odd)
            odd_idx = np.array([to_i(v) for v in odd], dtype=int)

            dist_odd = np.empty((len(odd), len(odd)), dtype=np.float64)
            for i, u in enumerate(odd):
                dist_odd[i, :] = dist[src_pos[u], odd_idx]

            matching_pairs = _greedy_matching_from_dist_matrix(odd, dist_odd)

            for u, v in matching_pairs:
                row = src_pos[u]
                path_idx = _reconstruct_path_from_pred_row(
                    pred[row], to_i(u), to_i(v)
                )
                if path_idx is None:
                    continue
                nodes = [to_v(i) for i in path_idx]
                for a, b in zip(nodes, nodes[1:]):
                    M.add_edge(a, b, weight=G[a][b][weight])
                    matching_edges.append((a, b))

        # --- Euler tour/path ---
        if return_to_start:
            circuit_edges = list(nx.eulerian_circuit(M, source=start))
            euler_nodes = [circuit_edges[0][0]] + [v for _, v in circuit_edges]
        else:
            path_edges = list(nx.eulerian_path(M, source=start))
            euler_nodes = [path_edges[0][0]] + [v for _, v in path_edges]

        # --- Shortcut over tree_nodes ---
        seen = set()
        ordered = []
        for n in euler_nodes:
            if n in tree_nodes and n not in seen:
                seen.add(n)
                ordered.append(n)

        if not ordered:
            return [], 0.0, matching_pairs, matching_edges

        if start in ordered:
            sidx = ordered.index(start)
            ordered = ordered[sidx:] + ordered[:sidx]

        # --- Stitch final tour using SciPy predecessors ---
        visit_sequence = ordered + ([ordered[0]] if return_to_start else [])

        tour_nodes = []
        for a, b in zip(visit_sequence, visit_sequence[1:]):
            if a == b:
                continue
            row = src_pos[a]
            path_idx = _reconstruct_path_from_pred_row(
                pred[row], to_i(a), to_i(b)
            )
            if path_idx is None:
                continue
            sp_nodes = [to_v(i) for i in path_idx]
            if not tour_nodes:
                tour_nodes.extend(sp_nodes)
            else:
                tour_nodes.extend(sp_nodes[1:])

        # --- Edges + weight ---
        tour_edges = []
        tour_weight = 0.0

        for u, v in zip(tour_nodes, tour_nodes[1:]):
            tour_edges.append((u, v))
        #     tour_weight += G[u][v][weight]

        return tour_edges, tour_weight, matching_pairs, matching_edges

if __name__ == '__main__':
    vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n1000_g1_vertex"
    edge_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n1000_g1_edge"

    G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file)
    I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
    root = 0

    solution_edges, _ = TM_solver_groups_scipy(G, root, set(I), vertex_poi_vis)
    solution_weight = sum([G[u][v]['weight'] for u, v in solution_edges])

    tour_edges, tour_weight, _, _ = ST_to_tour_christofides_scipy(G, solution_edges, start=root)

    print(f"Tour: {tour_edges}")
    print(f"Tour Weight: {tour_weight}")
    validate_solution_groups(G, S, tour_edges, is_tour=True)