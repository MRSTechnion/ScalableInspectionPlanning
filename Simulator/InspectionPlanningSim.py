import math
import random
from collections import defaultdict
from typing import Tuple, Iterable, Optional, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import HeuristicSolvers
import InspectionPostsolve
import Simulator.InspectionMap as InspectionMap
import GSTDirectedFormulationMILP
import Postsolve
import SolutionValidation
from Readers.SimInstanceUtils import save_simulated_instance
from RobotDrone2D import Cspace
from GeoHelpers import *
from MapHelpers import *

from MotionPlanning import RRT, RRG


def visibility_graph(G, plan_map, max_view_distance=np.inf, fov_deg=360):

    vertex_vis = {}
    for v in G.nodes:
        x, y, tetha = G.nodes[v]['config']
        vertex_vis[v] = set(plan_map.goals_visible_from(
            int(x), int(y),
            obstacle_value=1,
            max_view_distance=max_view_distance,
            view_angle_deg=tetha,
            fov_deg=fov_deg
        ))

    I = set([p for vis in vertex_vis.values() for p in vis])    # visible_pois_set
    S = defaultdict(set)
    for p in I:
        for v, vis in vertex_vis.items():
            if p in vis:
                S[p].add(v)

    return S, vertex_vis, I

def plot_sbmp_graph(G: nx.Graph, plan_map, root=0, title: str = "RRT",
                    solution=None, sight_radius=None):
    if hasattr(plan_map, "grid"):
        grid = plan_map.grid
    else:
        grid = plan_map
    h, w = grid.shape

    # 1. Clean Background (Only Obstacles)
    fig, ax = plt.subplots(figsize=(8, 8))
    bg_idx = np.zeros_like(grid)
    bg_idx[grid == 1] = 1
    cmap = mcolors.ListedColormap(["#FFFFFF", "#8B4513"]) # White, Brown
    ax.imshow(bg_idx, origin='upper', extent=[0, w, h, 0], cmap=cmap, zorder=1)

    # 2. Plot POIs (Goals) as Circles
    for gid, (gx, gy) in plan_map.goals.items():
        poi_circ = plt.Circle((gx, gy), 0.4, color='#FFD700', ec='black', lw=0.5, zorder=4)
        ax.add_patch(poi_circ)

    # 3. Plot RRT edges (Background graph)
    try:
        for u, v in G.edges:
            x0, y0, _ = G.nodes[u]['config']
            x1, y1, _ = G.nodes[v]['config']
            ax.plot([x0, x1], [y0, y1], color='turquoise', lw=1, alpha=0.5, zorder=2)

        # 4. Plot sampled nodes (Small dots)
        xs = [G.nodes[n]['config'][0] for n in G.nodes]
        ys = [G.nodes[n]['config'][1] for n in G.nodes]
        ax.scatter(xs, ys, s=3, c='black', alpha=0.5, zorder=3)


        # 5. Plot Root (Large Green Circle)
        rx, ry, _ = G.nodes[root]['config']
        root_circ = plt.Circle((rx, ry), 0.6, color='#00AA00', ec='black', lw=1, zorder=6)
        ax.add_patch(root_circ)

    except:
        pass
    # 6. Plot Solution Path
    if solution is not None:
        for u, v in solution:
            x0, y0, _ = G.nodes[u]['config']
            x1, y1, _ = G.nodes[v]['config']
            ax.plot([x0, x1], [y0, y1], color='darkgreen', lw=2, alpha=1.0, zorder=5)

    ax.set_title(title)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

def display_solution(G: nx.Graph, plan_map, solution, I, root=0, title="", sight_radius=None):

    # Extract grid and dimensions whether it's a GameMap or a plain array
    if hasattr(plan_map, "grid"):  # GameMap
        grid = plan_map.grid
    else:
        grid = plan_map
    h, w = grid.shape

    # Create color map: 0=white, 1=brown
    cmap = mcolors.ListedColormap(["white", "brown"])
    grid_plot = np.zeros_like(grid)
    grid_plot[grid == 1] = 1  # obstacles

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_plot, origin='upper', extent=[0, w, h, 0], cmap=cmap)

    # Plot goals
    x_goal, y_goal = [], []
    for i in range(h):  # i is row (y)
        for j in range(w):  # j is col (x)
            if grid[i, j] == 9:
                x_goal.append(j)  # Fix: j is x
                y_goal.append(i)  # Fix: i is y

    ax.scatter(x_goal, y_goal, s=4, c='yellow', alpha=0.8)

    # Plot RRT edges (Background graph)
    for u, v in G.edges:
        x0, y0, _ = G.nodes[u]['config']
        x1, y1, _ = G.nodes[v]['config']
        ax.plot([x0, x1], [y0, y1], color='turquoise', linewidth=0.7, alpha=1.0, zorder=2)

    # Plot sampled nodes
    xs = [G.nodes[n]['config'][0] for n in G.nodes]
    ys = [G.nodes[n]['config'][1] for n in G.nodes]
    ax.scatter(xs, ys, s=6, c='black', alpha=0.6, zorder=3)

    # Plot root
    ax.scatter(G.nodes[root]['config'][0], G.nodes[root]['config'][1], s=6, c='lime', alpha=1.0, zorder=4)

    # Plot Solution Path
    solution_nodes = set()
    if solution is not None:
        for u, v in solution:
            x0, y0, _ = G.nodes[u]['config']
            x1, y1, _ = G.nodes[v]['config']
            ax.plot([x0, x1], [y0, y1], color='darkgreen', linewidth=1.5, alpha=0.9, zorder=5)
            solution_nodes.add(u)


    # plot solution coverage
    for target_poi in I:
        px, py = plan_map.goals[target_poi]

        covering_nodes = S.get(target_poi, set()) & solution_nodes

        if not covering_nodes:
            print(f"No nodes cover POI {target_poi}!")
        else:
            for n in covering_nodes:
                nx_coord, ny_coord, _ = G.nodes[n]['config']
                ax.plot([nx_coord, px], [ny_coord, py], color='gold',
                        linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)

    plt.title(title)
    plt.xlim([0, w])
    plt.ylim([h, 0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()


def plot_pois_coverage(G: nx.Graph, plan_map, S, I, target_pois=[], title=None):
    """
    Visualizes which graph nodes 'cover' (see) a specific target POI.

    Args:
        G: The graph.
        plan_map: The map (GameMap or array).
        S: The dictionary mapping POI -> Set of Node IDs (from visibility_graph).
        target_poi: The specific key in S to visualize (e.g., a tuple (row, col)).
    """
    # Extract grid
    if hasattr(plan_map, "grid"):
        grid = plan_map.grid
    else:
        grid = plan_map
    h, w = grid.shape


    # Setup Plot
    '#FFFFFF', '#8B4513', '#00AA00', '#FFD700'

    cmap = mcolors.ListedColormap(["#00AA00", "#FFD700"])
    grid_plot = np.zeros_like(grid)
    grid_plot[grid == 1] = 1  # obstacles

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_plot, origin='upper', extent=[0, w, h, 0], cmap=cmap)

    # Plot goals
    x_goal, y_goal = [], []
    for i in range(h):  # i is row (y)
        for j in range(w):  # j is col (x)
            if grid[i, j] == 9:
                x_goal.append(j)  # Fix: j is x
                y_goal.append(i)  # Fix: i is y
    ax.scatter(x_goal, y_goal, s=4, c='yellow', alpha=0.8)

    # Plot RRT edges (Background graph)
    for u, v in G.edges:
        x0, y0, _ = G.nodes[u]['config']
        x1, y1, _ = G.nodes[v]['config']
        ax.plot([x0, x1], [y0, y1], color='turquoise', linewidth=0.7, alpha=1.0, zorder=2)

    # Plot sampled nodes
    xs = [G.nodes[n]['config'][0] for n in G.nodes]
    ys = [G.nodes[n]['config'][1] for n in G.nodes]
    ax.scatter(xs, ys, s=6, c='black', alpha=0.6, zorder=3)

    if len(target_pois) == 0:
        target_pois = I


    if title:
        plt.title(title)

    plt.xlim([0, w])
    plt.ylim([h, 0])
    plt.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    for target_poi in target_pois:
        px, py = plan_map.goals[target_poi]

        covering_nodes = S.get(target_poi, set())

        if not covering_nodes:
            print(f"No nodes cover POI {target_poi}!")
        else:
            for n in covering_nodes:
                nx_coord, ny_coord, _ = G.nodes[n]['config']
                ax.plot([nx_coord, px], [ny_coord, py], color='gold',
                        linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)



        plt.pause(5)


if __name__ == '__main__':
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    # for seed in [11]:
    #     for N in [3500, 5000, 10000, 15000]:
    #         for K in [1000, 2500, 5000, 7500]:
                # , 100, 250, 350, 500]:
                # random.seed(seed)
                # np.random.seed(seed)
                #
                # width, height = 500, 500
                #
                # plan_map = InspectionMap.GameMap(width, height)
                # plan_map.add_L_obstacles(
                #     count=50, value=1, min_len=3, max_len=30,
                #     thickness=1, padding=1, forbid=[(1, 1)],
                # )
                #
                # init_config = (1, 1, 0)
                # start_position = init_config[:-1]
                #
                # # plan_map.place_object(*start_position, 5)
                # # K = 100
                # goals = plan_map.scatter_goals(K, value=9, forbid=[start_position])
                #
                # C_space = Cspace(width, height)
                #
                #
                # # N = 500
                #
                # eta = 3
                # T = RRT(C_space, plan_map, N, eta, init_config, res=1.0, seed=seed)
                # G = RRG(T, max_deg=6, max_edge_dist=15.0)
                #
                # S, vertex_poi_vis, I = visibility_graph(G, plan_map, max_view_distance=40, fov_deg=50)
                # root = 0
                #
                # # target_pois = [82, 95]
                # # plot_pois_coverage(G, plan_map, S, I, target_pois=target_pois)
                #
                #
                #
                # save_simulated_instance(
                #     f"/home/adir/Desktop/IP-results/simulated_experiments/instance_{width}x{height}_K-{K}_N-{N}_seed-{seed}.pkl",
                #     G=G, I=I, S=S, vertex_poi_vis=vertex_poi_vis, root=root,
                #     meta={
                #         "random_seed": seed,
                #         "rrt_seed": seed,
                #         "N": N,
                #         "eta": eta,
                #         "width": width,
                #         "height": height,
                #     }
                # )
                # print(f"saved to /home/adir/Desktop/IP-results/simulated_experiments/instance_{width}x{height}_K-{K}_N-{N}_seed-{seed}.pkl")

    #
    # # --- Demonstrate solvers ---
    # root = 0
    # # sure_edges = []
    # # solution_edges = GSTDirectedFormulationMILP.RunSolver(G, S, set(I), vertex_poi_vis, root, sure_edges=sure_edges)
    #
    # solution_edges, solution_process = HeuristicSolvers.TM_solver_groups(G, root, I, vertex_poi_vis)
    # tour_edges, tour_weight, matching, matching_edges = Postsolve.ST_to_tour_christofides(G, solution_edges, start=root)
    #
    # print(f"Tour: {tour_edges}")
    # print(f"Tour Weight: {tour_weight}")
    #
    # SolutionValidation.validate_solution_groups(G, S, tour_edges)
    #
    # print(f"{len(solution_process)=}")
    #

    # ---- Map demo ----
    width, height = 50, 50

    plan_map = InspectionMap.GameMap(width, height)
    plan_map.add_L_obstacles(
        count=20, value=1, min_len=3, max_len=10,
        thickness=1, padding=1, forbid=[(1, 1)],
    )


    init_config = (1, 1, 0)
    start_position = init_config[:-1]

    # plan_map.place_object(*start_position, 5)
    K = 20
    goals = plan_map.scatter_goals(K, value=9, forbid=[start_position])

    C_space = Cspace(width, height)


    N = 1000

    eta = 3
    T = RRT(C_space, plan_map, N, eta, init_config, res=1.0, seed=seed)
    G = RRG(T, max_deg=6, max_edge_dist=15.0)

    S, vertex_poi_vis, I = visibility_graph(G, plan_map, max_view_distance=40, fov_deg=50)
    root = 0

    A = nx.empty_graph()
    plot_sbmp_graph(A, plan_map, title="")
    plot_sbmp_graph(T, plan_map, title="")
    plot_sbmp_graph(G, plan_map, title="")

    # sol_iter = 5
    # for part_sol in solution_process:
    #     plot_sbmp_graph(G, plan_map, solution=part_sol, title=f"Inspection Tree - Iteration #{sol_iter}")
    #     sol_iter += 10
    #


    # solution_edges = [(0, 4), (4, 8), (8, 9), (9, 10), (10, 29), (29, 134), (134, 343), (343, 447), (447, 439), (439, 209),
    #        (209, 257), (257, 303), (303, 396), (396, 115), (115, 280), (280, 223), (223, 416), (416, 469), (469, 473),
    #        (473, 286), (286, 239), (239, 305), (305, 357), (357, 480), (480, 437), (437, 164), (164, 94), (94, 224),
    #        (224, 103), (103, 127), (127, 161), (161, 267), (267, 193), (193, 331), (331, 425), (425, 486), (486, 472),
    #        (472, 192), (192, 183), (183, 163), (163, 370), (370, 482), (482, 141), (141, 130), (130, 491), (491, 400),
    #        (400, 354), (354, 430), (430, 187), (187, 59), (59, 48), (48, 117), (117, 0)]
    #
    # display_solution(G, plan_map, solution_edges, I, title="", sight_radius=None)

    #
    # plot_sbmp_graph(G, plan_map, solution=solution_edges, title="Inspection Tree + Matching")
    # for u, v in matching:
    #     x0, y0, _ = G.nodes[u]['config']
    #     x1, y1, _ = G.nodes[v]['config']
    #     plt.plot([x0, x1], [y0, y1], color='darkred', linewidth=0.7, alpha=1.0)
    #
    # plot_sbmp_graph(G, plan_map, solution=solution_edges, title="Inspection Tree + Matching Edges")
    # for u, v in matching_edges:
    #     x0, y0, _ = G.nodes[u]['config']
    #     x1, y1, _ = G.nodes[v]['config']
    #     plt.plot([x0, x1], [y0, y1], color='tomato', linewidth=0.7, alpha=1.0)
    #
    # plot_sbmp_graph(G, plan_map, solution=tour_edges, title="Inspection Tour Heuristic Solution")
    #
    plt.show()