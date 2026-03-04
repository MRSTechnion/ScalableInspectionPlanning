import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
import networkx as nx

from Readers.ExperimentPicker import pick_exp
from SolutionValidation import validate_solution_groups
from Utils import IP_to_Group
from Readers import IRIS_reader

import HeuristicSolvers
import Postsolve
import InspectionPostsolve
import time

def scipy_apsp_predecessors_undirected(G, weight="weight", assume_int_0_to_n_minus_1=False):
    """
    Returns:
      dist : (n, n) float array
      pred : (n, n) int array (SciPy uses -9999 as "no predecessor")
      node_to_idx, idx_to_node : mappings (None,None if identity mapping used)
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # ONLY use identity mapping if we are 100% sure nodes are exactly {0,1,...,n-1}
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
        rows.append(iv); cols.append(iu); data.append(w)  # undirected

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    dist, pred = dijkstra(A, directed=False, return_predecessors=True)
    return dist, pred, node_to_idx, idx_to_node


def reconstruct_path_scipy(pred_row, src_i, dst_i):
    """Reconstruct one shortest path from src_i to dst_i using a SciPy predecessor row."""
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


def TM_solver_groups_scipy(G, r, I, vis_set, *, weight="weight", assume_int_0_to_n_minus_1=False):
    # --- apply root coverage FIRST (matches original semantics, keeps bookkeeping consistent) ---
    if r in vis_set:
        I.difference_update(vis_set[r])

    # --- APSP via SciPy ---
    dist, pred, node_to_idx, idx_to_node = scipy_apsp_predecessors_undirected(
        G, weight=weight, assume_int_0_to_n_minus_1=assume_int_0_to_n_minus_1
    )

    if node_to_idx is None:
        def to_i(v): return v
        def to_v(i): return i
    else:
        def to_i(v): return node_to_idx[v]
        def to_v(i): return idx_to_node[i]

    # --- init ---
    L = {r}
    st_edges = set()

    # candidates are vis_set keys that cover at least one currently-uncovered group
    candidates = {v for v in vis_set if (vis_set[v] & I)}

    # reverse index: for each group g, which candidates can cover it?
    # TODO - vis_set already discribes that!
    group_to_candidates = defaultdict(set)
    for v in candidates:
        for g in (vis_set[v] & I):
            group_to_candidates[g].add(v)

    # cover_count[v] = |vis_set[v] ∩ I| (for pruning without scanning all candidates)
    cover_count = {v: len(vis_set[v] & I) for v in candidates}

    # best_to_vertex[u] = (best_dist_from_tree, predecessor_vertex_in_tree)
    best_to_vertex = {}
    r_i = to_i(r)
    r_row = dist[r_i]
    for u in candidates:
        if u == r:
            continue
        u_i = to_i(u)
        du = r_row[u_i]
        if np.isfinite(du) and cover_count[u] > 0:
            best_to_vertex[u] = (float(du), r)
            # best_to_vertex[u] = (float(du)/cover_count[u], r)     # Minimize cost-to-group-cover ratio

    def apply_removed_groups(removed_groups):
        """Update cover_count and return candidates that became useless (count==0)."""
        dead = set()
        for g in removed_groups:
            for v in group_to_candidates.get(g, ()):
                c = cover_count.get(v, 0)
                if c > 0:
                    c -= 1
                    cover_count[v] = c
                    if c == 0:
                        dead.add(v)
        return dead

    solution_process_copies = []
    iter_counter = 0

    # --- main loop ---
    while I and best_to_vertex:
        u, (best_d, p) = min(best_to_vertex.items(), key=lambda kv: kv[1][0])

        # path p -> u
        p_i, u_i = to_i(p), to_i(u)
        path_idx = reconstruct_path_scipy(pred[p_i], p_i, u_i)
        if path_idx is None:
            # should not happen in connected graphs, but keep safe
            best_to_vertex.pop(u, None)
            continue

        cur_route = [to_v(i) for i in path_idx]
        new_vertices = set(cur_route) - L
        L.update(new_vertices)

        # compute which groups are newly covered
        removed_groups = set()
        for v in new_vertices:
            gset = vis_set.get(v)
            if gset:
                newly = gset & I
                if newly:
                    removed_groups |= newly

        if removed_groups:
            I.difference_update(removed_groups)

        # add edges of the route in original graph terms
        st_edges.update(
            tuple(sorted((cur_route[i], cur_route[i + 1])))
            for i in range(len(cur_route) - 1)
        )

        # drop candidates that no longer cover any uncovered group
        dead = apply_removed_groups(removed_groups)
        for v in dead:
            best_to_vertex.pop(v, None)

        # relax best distances via newly added vertices
        alive = list(best_to_vertex.keys())
        for vL in new_vertices:
            row = dist[to_i(vL)]
            for v in alive:
                nd = row[to_i(v)]
                if np.isfinite(nd) and nd < best_to_vertex[v][0]:
                    best_to_vertex[v] = (float(nd), vL)

        # if iter_counter % 10 == 5:
        #     solution_process_copies.append(st_edges.copy())
        # iter_counter += 1

    return st_edges, solution_process_copies

def TM_solver_groups_directed(G, r, I, vis_set):
    solution_edges = TM_solver_groups_scipy(G, r, I, vis_set)

    H = nx.Graph()
    H.add_edges_from(solution_edges)

    solution_visit_order = nx.dfs_successors(H, r)
    directed_solution_edges = set()
    for u, suc_list in solution_visit_order.items():
        for v in suc_list:
            directed_solution_edges.add((u, v))

    return directed_solution_edges

def wafr24_ST_heuristic(G, r, I, vis_set):
    # Choose covering vertices set closest to r
    dist, pred, node_to_idx, idx_to_node = scipy_apsp_predecessors_undirected(
        G, weight="weight", assume_int_0_to_n_minus_1=False
    )

    L = set()
    uncovered_groups = I.copy()
    candidates = {v for v in vis_set if (vis_set[v] & uncovered_groups)}

    while uncovered_groups:
        _, closest_candidate = min([(dist[r, v], v) for v in candidates])
        L.add(closest_candidate)

        uncovered_groups.difference_update(vis_set[closest_candidate])
        candidates = {v for v in vis_set if (vis_set[v] & uncovered_groups)}

    # Construct a 2-approximation steiner tree for this set
    ST_approx_edges = HeuristicSolvers.MST_solver_scipy(G, L)

    # # Double it to get a tour
    # tour_edges, _ = Postsolve.ST_to_tour(G, ST_approx_edges, start=r)

    return ST_approx_edges

if __name__ == '__main__':
    for Experiment in ["Drone1000", "Drone2000", "Crisp1000", "Crisp2000"]:
        vertex_file, edge_file, conf_file = pick_exp(Experiment)

        print(f"---{Experiment}---")
        G, vertex_poi_vis = IRIS_reader.read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file)
        I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)
        root = 0

        G1 = G.copy()
        G2 = G.copy()
        G3 = G.copy()

        # TODO - revalidate the graphs are not modifyied in the heuristics

        # t0 = time.time()
        # tree_solution_edges, _ = TM_solver_groups_scipy(G1, root, set(I), vertex_poi_vis)
        # solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy_greedy(G1, tree_solution_edges,
        #                                                                                      start=root)
        # t1 = time.time()
        # print(f"--- Tree Building + Greedy Matching Heuristic:")
        # validate_solution_groups(G1, S, solution_edges, is_tour=True)
        # print(f"Calculation Time: {t1-t0}")
        #
        t0 = time.time()
        tree_solution_edges, _ = TM_solver_groups_scipy(G2, root, set(I), vertex_poi_vis)
        solution_edges, sol_weight, _, _ = InspectionPostsolve.ST_to_tour_christofides_scipy(G2, tree_solution_edges,
                                                                                             start=root)
        t1 = time.time()
        print(f"--- Tree Building + Christophides Heuristic:")
        validate_solution_groups(G2, S, solution_edges, is_tour=True)
        print(f"Calculation Time: {t1-t0}")

        # t0 = time.time()
        # solution_edges = wafr24_ST_heuristic(G3, root, set(I), vertex_poi_vis)
        # t1 = time.time()

        # print(f"--- ST Heuristic (Wafr'24):")
        # validate_solution_groups(G3, S, solution_edges, is_tour=True)
        # print(f"Calculation Time: {t1-t0}")

