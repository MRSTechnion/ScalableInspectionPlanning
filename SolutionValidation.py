import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

def validate_solution(G, r, T, solution_edges):
    # Generate solution tree
    H = G.edge_subgraph(solution_edges).copy()
    root_component_nodes = nx.node_connected_component(H, r)
    solution_tree = H.subgraph(root_component_nodes)
    print("--- Validating Solution ---")

    # Connectivity
    solution_tree_edges = [tuple(sorted(e)) for e in solution_tree.edges()]
    if not set(solution_edges).issubset(solution_tree_edges):
        print("Solution graph non-connected")
        return False
    else:
        print("Solution is connected")

    # Coverage
    if not T.issubset(solution_tree.nodes()):
        print("Solution graph not covering all terminals")
        return False
    else:
        print("Solution is covering all terminals")

    # Value
    solution_val = sum([G[u][v]["weight"] for u, v in solution_edges])
    print(f"Valid solution with value {solution_val}")
    return True



def validate_solution_groups(
    G,
    subsets_dict,
    solution_edges,
    coverage_rule="any",   # 'any' | 'all'
    weight_attr="weight",
    is_tour=False):
    """
    Validate a solution edge set for a weighted graph and subsets.

    Returns:
        bool  (True if valid, else False)
        and optionally details if return_info=True
    """
    # --- normalize edges ---
    E_sol = set()
    for u, v in solution_edges or []:
        if G.has_edge(u, v) or G.has_edge(v, u):
            E_sol.add((u, v) if G.has_edge(u, v) else (v, u))
    if not E_sol:
        print("no valid edges in graph")
        return False

    # --- build subgraph ---
    H = nx.Graph()
    for u, v in E_sol:
        w = G[u][v].get(weight_attr, 1.0)
        H.add_edge(u, v, **{weight_attr: w})

    # --- total weight ---
    total_weight = sum(G[u][v].get(weight_attr, 1.0) for u, v in E_sol)

    # --- connectivity ---
    if not nx.is_connected(H):
        print("solution subgraph not connected")
        return False

    # --- coverage ---
    incident = {v: 0 for v in G.nodes()}
    for u, v in E_sol:
        incident[u] += 1
        incident[v] += 1

    for label, subset in subsets_dict.items():
        members = [n for n in subset if n in G]
        if not members:
            continue
        touched = [incident[n] > 0 for n in members]
        covered = any(touched) if coverage_rule == "any" else all(touched)
        if not covered:
            print(f"subset '{label}' not covered")
            return False

    # --- tour ---
    if is_tour:
        tour_start = True
        first_vertex = None
        for u, v in solution_edges:
            if tour_start:
                first_vertex = u
                tour_start = False
            else:
                if next_u != u:
                    print(f"Solution is not a tour")
                    return False

            next_u = v

    # --- all checks passed ---
    print(f"Valid solution with value {total_weight}")
    return True