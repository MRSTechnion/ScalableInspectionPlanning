from collections import defaultdict

import networkx as nx
import time
import maxflow
import GraphGeneration

import random, math
from collections import defaultdict, deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow


def crossing_edges(edge_list, S):
    # edges with exactly one endpoint in S
    return [(i, j) for (i, j) in edge_list if (i in S) ^ (j in S)]

def crossing_directed_edges(edge_list, S, T):
    return [(i, j) for (i, j) in edge_list if (i in S) and (j in T)]


def _mincut_partition_from_residual(residual_csr, s, eps=0):
    if not sp.isspmatrix_csr(residual_csr):
        residual_csr = residual_csr.tocsr()

    n = residual_csr.shape[0]
    indptr = residual_csr.indptr
    indices = residual_csr.indices
    data = residual_csr.data

    seen = np.zeros(n, dtype=bool)
    dq = deque([s])
    seen[s] = True

    while dq:
        u = dq.popleft()
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if (not seen[v]) and data[k] > eps:
                seen[v] = True
                dq.append(v)

    S = set(np.nonzero(seen)[0].tolist())
    T = set(range(n)) - S
    return S, T

def generate_group_flow_cuts_directed_guided_scipy(
    D, pois_groups, r, lp=None, groups_subset=None, *,
    scale=10**6,
):
    """
    Same output as before: returns new_cuts.

    Assumptions:
      - D is directed
      - lp capacities are nonnegative (LP)
      - You compare cut_val to 1 (so scaling is appropriate)
    """
    # --- Build node indexing ---
    nodes = list(D.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_node = nodes
    r_i = node_to_idx[r]

    # --- Precompute out-neighbors for fast cut-edge extraction ---
    out_neighbors = {u: [] for u in D.nodes()}
    for u, v in D.edges():
        out_neighbors[u].append(v)

    # --- Build base edge arrays once (directed) ---
    base_u = []
    base_v = []
    base_cap = []

    if lp is not None:
        for u, v in D.edges():
            cuv = lp[(u, v)]
            c_int = int(round(scale * float(cuv)))
            if c_int < 0:
                c_int = 0
            base_u.append(node_to_idx[u])
            base_v.append(node_to_idx[v])
            base_cap.append(c_int)
    else:
        for u, v in D.edges():
            base_u.append(node_to_idx[u])
            base_v.append(node_to_idx[v])
            base_cap.append(scale)

    base_u = np.array(base_u, dtype=np.int32)
    base_v = np.array(base_v, dtype=np.int32)
    base_cap = np.array(base_cap, dtype=np.int64)

    # Big-M
    M_int = int(base_cap.sum() + 1)

    # --- inflow LP for group scoring ---
    inflow_lp = defaultdict(float)
    if lp is not None:
        for (u, v), val in lp.items():
            inflow_lp[v] += float(val)

    # --- choose groups ---
    if groups_subset is None:
        groups_subset = pois_groups.keys()

    group_connectivity_score = {
        g: max(inflow_lp[v] for v in pois_groups[g])
        for g in groups_subset
    }
    groups_sorted = sorted(groups_subset, key=lambda g: group_connectivity_score[g])

    max_cuts = 50
    # K = 100
    K = len(groups_sorted)    # no sorting priority

    candidate_groups = groups_sorted[K:]
    groups_subset_sampled = random.sample(candidate_groups, min(max_cuts, len(candidate_groups)))

    tol = 1e-4
    thresh_int = int(np.floor(scale * (1.0 - tol)))

    new_cuts = []

    # Capacity overrides for nesting
    base_override = {}

    # Precompute edge -> position map
    edge_pos = {
        (int(base_u[k]), int(base_v[k])): k
        for k in range(len(base_u))
    }

    def build_augmented_csr(gv_idx):
        cap = base_cap.copy()
        for (ui, vi), newc in base_override.items():
            pos = edge_pos.get((ui, vi))
            if pos is not None:
                cap[pos] = newc

        sup_i = n

        rows = base_u
        cols = base_v
        data = cap

        gv_idx = np.array(gv_idx, dtype=np.int32)
        rows2 = gv_idx
        cols2 = np.full(len(gv_idx), sup_i, dtype=np.int32)
        data2 = np.full(len(gv_idx), M_int, dtype=np.int64)

        A = sp.csr_matrix(
            (np.concatenate([data, data2]),
             (np.concatenate([rows, rows2]),
              np.concatenate([cols, cols2]))),
            shape=(n + 1, n + 1),
            dtype=np.int64
        )
        return A, sup_i

    def cut_edges_from_partition(S_idx, T_idx):
        S_nodes = {idx_to_node[i] for i in S_idx if i < n}
        T_nodes = {idx_to_node[i] for i in T_idx if i < n}
        cut = []
        for u in S_nodes:
            for v in out_neighbors.get(u, ()):
                if v in T_nodes:
                    cut.append((u, v))
        return cut

    # ---- main loop ----
    for gid in groups_subset_sampled:

        gv = pois_groups[gid]
        gv_idx = [node_to_idx[v] for v in gv]

        group_overrides = set()

        while True:
            A_aug, sup_i = build_augmented_csr(gv_idx)

            mf = maximum_flow(A_aug, r_i, sup_i)
            cut_val_int = int(mf.flow_value)

            F = mf.flow
            if not sp.isspmatrix_csr(F):
                F = F.tocsr()

            A_csr = A_aug.tocsr()
            residual = (A_csr - F) + F.transpose()

            if cut_val_int <= thresh_int:
                S_idx, T_idx = _mincut_partition_from_residual(residual, r_i, eps=0)
                cut_edges = cut_edges_from_partition(S_idx, T_idx)

                new_cuts.append(cut_edges)
            else:
                break

            # Nested capacity updates
            for u, v in cut_edges:
                ui, vi = node_to_idx[u], node_to_idx[v]
                key = (ui, vi)
                base_override[key] = scale
                group_overrides.add(key)

        # cleanup
        for key in group_overrides:
            base_override.pop(key, None)

    return new_cuts

def generate_group_flow_cuts(G, pois_groups, r, lp=None):
    H = G.copy()

    if lp is not None:
        for i, j in G.edges():
            H[i][j]['capacity'] = lp[(i, j)]
    else:
        for i, j in G.edges():
            H[i][j]['capacity'] = 1

    M = sum(H[i][j]['capacity'] for i, j in H.edges()) + 1.0

    new_cuts = []
    for gid, vertices in pois_groups.items():
        # Targets that actually exist (and aren't the root)
        targets = [v for v in vertices if v in H and v != r]
        if not targets:
            continue

        # Build a supersink tied to this group, add big-M arcs
        supersink = ("__grp_sink__", gid)
        H.add_node(supersink)

        if H.is_directed():
            for v in targets:
                # allow reaching supersink from v without constraining the cut
                H.add_edge(v, supersink, capacity=M)
        else:
            for v in targets:
                H.add_edge(v, supersink, capacity=M)

        # Min cut from r to the supersink = min r-to-group cut
        cut_val, (pois_groups, Tset) = nx.minimum_cut(H, r, supersink, capacity="capacity")

        # Consider only original nodes when forming the cut over original G
        S_orig = {u for u in pois_groups if u in G}
        # crossing_edges: your helper that returns edges with one endpoint in S_orig and the other outside
        cut_edges = crossing_edges(G.edges(), S_orig)

        new_cuts.append(cut_edges)

        # Cleanup supersink for reuse
        H.remove_node(supersink)

    return new_cuts

def minimum_cut_pymaxflow(
    G, s, t, capacity="capacity",
    *, default_capacity=1, inf=None,
):
    """
    PyMaxflow-backed s-t min cut for a directed graph.

    Drop-in-ish replacement for:
        nx.minimum_cut(G, s, t, capacity="capacity")

    Returns:
        (cut_value, (S, T)) where S is the source side (reachable from s in residual after maxflow)
        and T is the sink side.

    Notes:
      - Directed edge (u->v) with capacity c is added as add_edge(u, v, c, 0).
      - Forces s in S and t in T by connecting s to SOURCE and t to SINK with INF capacity.
      - If capacities are floats, uses float graph; if all ints, uses int graph.
    """
    if s not in G or t not in G:
        raise ValueError("Both s and t must be nodes in G")

    # Deterministic node order helps repeatability
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Collect capacities + infer numeric type
    total_cap = 0.0
    all_int = True

    # NetworkX DiGraph: G.edges(data=True) yields (u, v, attrdict)
    # If MultiDiGraph, you’d need to handle keys; this wrapper assumes simple DiGraph-like.
    edges_data = []
    for u, v, data in G.edges(data=True):
        c = data.get(capacity, default_capacity)
        if c is None:
            c = default_capacity
        # Ensure numeric
        c = float(c) if isinstance(c, bool) else c  # bool -> float-ish safeguard
        if isinstance(c, float):
            if not c.is_integer():
                all_int = False
        else:
            # int-like
            pass
        if c < 0:
            raise ValueError(f"Negative capacity on edge {(u, v)}: {c}")
        total_cap += float(c)
        edges_data.append((u, v, c))

    # Choose INF if not provided
    if inf is None:
        # Big enough to never be in a min-cut unless unavoidable
        inf = total_cap + 1.0
        if not math.isfinite(inf) or inf <= 0:
            inf = 1e18

    # Build the maxflow graph (int or float)
    if all_int and float(inf).is_integer():
        g = maxflow.Graph[int](len(nodes), len(edges_data))
        g.add_nodes(len(nodes))
        INF = int(inf)
        # Add directed edges
        for u, v, c in edges_data:
            g.add_edge(node_to_idx[u], node_to_idx[v], int(c), 0)
        # Force terminals
        g.add_tedge(node_to_idx[s], INF, 0)
        g.add_tedge(node_to_idx[t], 0, INF)
        cut_val = g.maxflow()
        # Partition
        S = {n for n in nodes if g.get_segment(node_to_idx[n]) == 0}
        T = set(nodes) - S
        return cut_val, (S, T)
    else:
        g = maxflow.Graph[float](len(nodes), len(edges_data))
        g.add_nodes(len(nodes))
        INF = float(inf)
        for u, v, c in edges_data:
            g.add_edge(node_to_idx[u], node_to_idx[v], float(c), 0.0)
        g.add_tedge(node_to_idx[s], INF, 0.0)
        g.add_tedge(node_to_idx[t], 0.0, INF)
        cut_val = g.maxflow()
        S = {n for n in nodes if g.get_segment(node_to_idx[n]) == 0}
        T = set(nodes) - S
        return cut_val, (S, T)


def generate_group_flow_cuts_directed(D, pois_groups, r, lp=None, groups_subset=None,
                                      use_nested_cuts=False, use_creep_flow = False, max_groups_per_iteration=100):
    # t0 = time.time()
    H = D.copy()

    # creep-flow setup
    if use_creep_flow:
        creep_flow_val = 1e-4
        lp_val = {k: max(v, creep_flow_val) for k, v in lp.items()}
    else:
        lp_val = lp

    nx.set_edge_attributes(H, lp_val, 'capacity')

    M = 1.0

    new_cuts = []

    # Select groups
    if groups_subset is None:
        # sorted() ensures deterministic sampling if random seed is fixed
        all_groups = sorted(pois_groups.keys())
        groups_subset_sampled = random.sample(all_groups, min(len(all_groups), max_groups_per_iteration))
    else:
        if len(groups_subset) > max_groups_per_iteration:
            groups_subset_sampled = random.sample(groups_subset, max_groups_per_iteration)
        else:
            groups_subset_sampled = groups_subset

    # Use a static supersink node name to avoid adding/removing nodes repeatedly
    supersink = "__grp_sink__"
    H.add_node(supersink)

    tol = 1e-4

    for gid in groups_subset_sampled:
        # print(f"Connectivity test for group {gid}")
        group_vertices = pois_groups[gid]

        # --- STEP 1: Connect Group to Supersink ---
        added_edges = []
        for v in group_vertices:
            # Overwrite or add edge to supersink
            H.add_edge(v, supersink, capacity=M)
            added_edges.append((v, supersink))

        # --- STEP 2: Solve Cuts ---
        if not use_nested_cuts:
            # Standard single cut
            cut_val, partition = nx.minimum_cut(
                H, r, supersink, capacity="capacity",
                flow_func=nx.algorithms.flow.shortest_augmenting_path
            )

            if cut_val <= 1 - tol:
                # print(f"{gid} not connected to root")
                S, T = partition
                cut_edges = crossing_directed_edges(D.edges(), S, T)
                back_cut_edges = crossing_directed_edges(D.edges(), T, S)
                new_cuts.append(cut_edges)
                new_cuts.append(back_cut_edges)

        else:
            # Nested cuts loop
            # We track modified edges to restore them later
            modified_caps_history = {}

            while True:
                cut_val, partition = nx.minimum_cut(
                    H, r, supersink, capacity="capacity",
                    flow_func=nx.algorithms.flow.shortest_augmenting_path
                )

                if cut_val > 1 - tol:
                    break  # Stop if cut is satisfied

                S, T = partition
                cut_edges = crossing_directed_edges(D.edges(), S, T)
                back_cut_edges = crossing_directed_edges(D.edges(), T, S)
                new_cuts.append(cut_edges)
                new_cuts.append(back_cut_edges)

                # Modify capacities to force the next cut
                # We set capacity to 1 so this cut is no longer the minimum
                for u, v in cut_edges:
                    # Save original capacity if we haven't touched this edge yet
                    if (u, v) not in modified_caps_history:
                        modified_caps_history[(u, v)] = H[u][v]['capacity']

                    H[u][v]['capacity'] = 1.0

            # --- RESTORE H (Critical for Nested) ---
            # Reset the capacities of the edges we modified back to LP values
            for (u, v), original_cap in modified_caps_history.items():
                H[u][v]['capacity'] = original_cap

        # --- STEP 3: Cleanup Supersink Edges ---
        # Remove the temporary edges to the supersink so H is clean for the next group
        H.remove_edges_from(added_edges)

    # Cleanup the node itself
    H.remove_node(supersink)

    # print(f"Time - {time.time() - t0 }")
    return new_cuts


def generate_group_flow_cuts_directed_guided(D, pois_groups, r, lp=None, groups_subset=None):
    H = D.copy()

    use_back_cuts = False
    use_nested_cuts = True

    if lp is not None:
        for i, j in D.edges():
            H[i][j]['capacity'] = lp[(i, j)]
    else:
        for i, j in D.edges():
            H[i][j]['capacity'] = 1

    M = sum(H[i][j]['capacity'] for i, j in H.edges()) + 1.0
    # Chosen so no min-cut will ever contain an added edge

    new_cuts = []
    max_cuts = 50

    inflow_lp = defaultdict(float)
    for (u, v), val in lp.items():
        inflow_lp[v] += val

    if groups_subset is None:
        groups_subset = pois_groups.keys()

    group_connectivity_score = {g:max(inflow_lp[v] for v in pois_groups[g]) for g in groups_subset}

    # sort groups by score
    groups_sorted = sorted(groups_subset, key=lambda g: group_connectivity_score[g])

    # take bottom K (say 100–300)
    K = 100
    candidate_groups = groups_sorted[K:]

    # from those, run min-cut only on a small subset (say 20–50)
    groups_subset_sampled = random.sample(candidate_groups, min(max_cuts, len(candidate_groups)))

    # if groups_subset is None:
    #     groups_subset_sampled = random.sample(sorted(pois_groups.keys()), max_cuts)
    # else:
    #     if len(groups_subset) > max_cuts:
    #         groups_subset_sampled = random.sample(groups_subset, max_cuts)
    #     else:
    #         groups_subset_sampled = groups_subset


    for gid in groups_subset_sampled:
        group_vertices = pois_groups[gid]

        H_group = H.copy()

        # Build a supersink tied to this group, add big-M arcs
        supersink = ("__grp_sink__", gid)
        H_group.add_node(supersink)

        for v in group_vertices:
            H_group.add_edge(v, supersink, capacity=M)

        # --- Nested flow ---
        tol = 1e-4

        while use_nested_cuts:
            # Min cut from r to the supersink = min r-to-group cut
            cut_val, (S, T) = nx.minimum_cut(H_group, r, supersink, capacity="capacity")
            # S, T are the graph partition vertices

            if cut_val <= 1 - tol:
                cut_edges = crossing_directed_edges(D.edges(), S, T)
                new_cuts.append(cut_edges)
            else:
                break

            # Nested cuts - modify cap:
            if use_nested_cuts:
                for u, v in cut_edges:
                    H_group[u][v]['capacity'] = 1

            # --- Back-cut ---
            if use_back_cuts:
                # Min cut from supersink to r = min group-to-r cut

                cut_val, (S, T) = nx.minimum_cut(H_group, supersink, r, capacity="capacity")

                if cut_val <= 1 - tol:
                    cut_edges = crossing_directed_edges(D.edges(), S, T)
                    # cut_edges = crossing_directed_edges(D.edges(), T, S)    # flipped S and T - still directing the edges from root to terminal
                    new_cuts.append(cut_edges)
                else:
                    break

            # Nested cuts - modify cap:
                if use_nested_cuts:
                    for u, v in cut_edges:
                        H_group[u][v]['capacity'] = 1

        # Cleanup supersink for reuse
        # H.remove_node(supersink)

    return new_cuts


def generate_group_flow_cuts_sampled(G, pois_groups, r, lp=None, sampled_groups_num=20, tol=1e-4):
    H = G.copy()

    if lp is not None:
        for i, j in G.edges():
            H[i][j]['capacity'] = lp[(i, j)]
    else:
        for i, j in G.edges():
            H[i][j]['capacity'] = 1

    M = sum(H[i][j]['capacity'] for i, j in H.edges()) + 1.0

    new_cuts = []

    sampled_groups = random.sample(list(pois_groups.keys()), sampled_groups_num)
    for gid in sampled_groups:
        g_vertices = pois_groups[gid]

        # Targets that actually exist (and aren't the root)
        targets = [v for v in g_vertices if v in H and v != r]
        if not targets:
            continue

        # Build a supersink tied to this group, add big-M arcs
        supersink = ("__grp_sink__", gid)
        H.add_node(supersink)


        for v in targets:
            H.add_edge(v, supersink, capacity=M)

        # Min cut from r to the supersink = min r-to-group cut
        cut_val, (cut_S, _) = nx.minimum_cut(H, r, supersink, capacity="capacity")
        if cut_val >= 1 - tol:
            continue

        # Consider only original nodes when forming the cut over original G
        S_orig = {u for u in cut_S if u in G}
        # crossing_edges: your helper that returns edges with one endpoint in S_orig and the other outside
        cut_edges = crossing_edges(G.edges(), S_orig)

        new_cuts.append(cut_edges)

        # Cleanup supersink for reuse
        H.remove_node(supersink)

    return new_cuts


def group_connectivity_cut(G, edge_list, pois_groups, r):
    H = G.edge_subgraph(edge_list)
    try:
        root_comp = nx.node_connected_component(H, r)
    except:
        root_comp = {r}

    cut_edges = None
    for g, v_g in pois_groups.items():
        if len(root_comp.intersection(v_g)) == 0:    # If root is not connected to all terminals
            cut_edges = {tuple(sorted(e)) for e in nx.edge_boundary(G, root_comp)}

    return cut_edges


def directed_group_connectivity_cut(D, edge_list, pois_groups, r):

    H = D.edge_subgraph(edge_list)

    root_reachable = nx.descendants(H, r) | {r}
    root_unreachable = set(D.nodes()).difference(root_reachable)

    uncovered_groups = [g for g, g_v in pois_groups.items() if len(set(g_v).intersection(root_reachable)) == 0]

    cut_edges = None
    if uncovered_groups:
        cut_edges = crossing_directed_edges(D.edges(), root_reachable, root_unreachable)

    return cut_edges, uncovered_groups


def directed_group_connectivity_cut2(D, edge_list, pois_groups, r):
    H = D.edge_subgraph(edge_list).copy()

    root_reachable = nx.descendants(H, r) | {r}
    # root_unreachable = set(D.nodes()).difference(root_reachable)

    uncovered_groups = [g for g, g_v in pois_groups.items() if len(set(g_v).intersection(root_reachable)) == 0]
    new_cuts = []

    if uncovered_groups:
        M = 1
        tol = 1e-4
        sampled_groups_num = 20
        sampled_groups = random.sample(list(pois_groups.keys()), sampled_groups_num)

        for gid in sampled_groups:
            g_vertices = pois_groups[gid]

            # Targets that actually exist (and aren't the root)
            targets = [v for v in g_vertices if v in H and v != r]
            if not targets:
                continue

            # Build a supersink tied to this group, add big-M arcs
            supersink = ("__grp_sink__", gid)
            H.add_node(supersink)

            for v in targets:
                H.add_edge(v, supersink, capacity=M)

            # Min cut from r to the supersink = min r-to-group cut
            cut_val, (cut_S, _) = nx.minimum_cut(H, r, supersink, capacity="capacity")
            if cut_val >= 1 - tol:
                continue

            # Consider only original nodes when forming the cut over original G
            S_orig = {u for u in cut_S if u in D}
            # crossing_edges: your helper that returns edges with one endpoint in S_orig and the other outside
            cut_edges = crossing_edges(D.edges(), S_orig)

            new_cuts.append(cut_edges)

            # Cleanup supersink for reuse
            H.remove_node(supersink)

    return new_cuts, uncovered_groups

def correction_heuristic(D, edge_list, pois_groups, r, weight="weight"):
    """
    Repair heuristic:
    - Start from the candidate edge-induced subgraph H.
    - While there exists a POI-group with no node reachable from root r in H,
      add the *lightest* (minimum total weight) path in D from the current
      root-reachable component to *some* node in an uncovered group.

    Assumptions:
    - D is a directed graph (DiGraph) or graph supported by NetworkX shortest paths.
    - Edge weights live under attribute name `weight` (default: "weight").
      If missing, NetworkX treats them as weight 1.

    Returns:
    - corrected_edge_list: list of edges (u, v) to use as repaired solution.
    """

    # Build current solution subgraph
    H = D.edge_subgraph(edge_list).copy()

    corrected_edges = set(edge_list)
    root_reachable = nx.descendants(H, r) | {r} if r in H else {r}

    def compute_uncovered_groups(root_reach):
        return {
            g for g, g_v in pois_groups.items()
            if len(set(g_v).intersection(root_reach)) == 0
        }

    uncovered_groups = compute_uncovered_groups(root_reachable)

    # Main repair loop
    while uncovered_groups:
        # Multi-source shortest paths from any node currently reachable from root in H
        # (This finds the cheapest way to connect the current root component to new nodes.)
        sources = list(root_reachable) if root_reachable else [r]
        dist, paths = nx.multi_source_dijkstra(D, sources=sources, weight=weight)

        best = None  # (cost, group_id, target_node, path_nodes)

        # Pick the cheapest group-to-connect option overall (globally lightest addition)
        for g in uncovered_groups:
            candidates = pois_groups.get(g, [])
            for v in candidates:
                if v in root_reachable:
                    continue
                if v not in dist:
                    continue  # unreachable in D from current component
                cost = dist[v]
                path_nodes = paths[v]
                if best is None or cost < best[0]:
                    best = (cost, g, v, path_nodes)

        if best is None:
            # No uncovered group can be connected from the current root component (graph disconnected)
            # Return what we have (partial repair).
            return list(corrected_edges)

        _, g_best, v_best, path_nodes = best

        # Add the path edges to the solution
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        path_edges_back = list(zip(path_nodes[1:], path_nodes[:-1]))

        corrected_edges.update(path_edges)
        corrected_edges.update(path_edges_back)

        H.add_edges_from(path_edges)
        H.add_edges_from(path_edges_back)

        root_reachable = nx.descendants(H, r) | {r} if r in H else {r}
        uncovered_groups = compute_uncovered_groups(root_reachable)

    return list(corrected_edges)



if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)

    n_vertices = 50
    edge_prob = 0.2
    cost_range = (1, 5)
    root = 0

    k = 20
    max_pois = 8

    G = GraphGeneration.random_weighted_graph(n_vertices, edge_prob, cost_range, integer_costs=True)
    I, unseen_pois, vertex_vis, pois_groups = GraphGeneration.assign_random_pois(G, k, max_pois)

    cuts = generate_group_flow_cuts(G, pois_groups, root, lp=None)
    print(cuts)