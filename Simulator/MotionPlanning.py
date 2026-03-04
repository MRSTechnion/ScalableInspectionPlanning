import networkx as nx
from RobotDrone2D import Cspace
from GeoHelpers import *
from MapHelpers import *

import numpy as np
import random

def init_graph(C_space, init_config, plan_map=None):
    """
    Initialize a NetworkX graph. Store helpers on G.graph for later access.
    """
    G = nx.Graph()
    G.add_node(0, config=init_config)
    G.graph['cspace'] = C_space
    if plan_map is not None:
        G.graph['plan_map'] = plan_map
    return G


def nearest_neighbors(T, x_rand, num=1):
    """
    Return K nearest nodes to x_rand under the C-space metric.
    """
    C = T.graph['cspace']
    dists = []
    for n in T.nodes:
        d = C.distance(T.nodes[n]['config'], x_rand)
        dists.append((d, n))
    dists.sort(key=lambda p: p[0])
    if num == 1:
        return [dists[0][1]]
    return [n for _, n in dists[:num]]


def step_new_state(x_start,
                   x_target,
                   eta):
    """
    Take a step from x_start towards x_target with max step-size 'eta' in the POSitional sense,
    and interpolate heading along the shortest arc.
    """
    sx, sy, sth = x_start
    tx, ty, tth = x_target

    dx, dy = tx - sx, ty - sy
    dpos = math.hypot(dx, dy)
    if dpos < 1e-12:  # same spot; only rotate slightly
        new_theta = lerp_angle(sth, tth, min(1.0, eta))  # small rotate
        return (sx, sy, new_theta)

    step = min(eta, dpos)
    t = step / dpos
    nx = sx + dx * t
    ny = sy + dy * t
    nth = lerp_angle(sth, tth, t)
    return (nx, ny, nth)


def collision_free_vertex(x_state, plan_map):
    x, y, _ = x_state
    return is_free(plan_map, x, y)


def collision_free_edge(x_start,
                        x_end,
                        res,
                        plan_map):
    """
    Check straight-line collision in workspace (x, y) only.
    'res' is the sampling step in pixels (e.g., 1.0 good default).
    """
    (x0, y0, _), (x1, y1, _) = x_start, x_end
    for px, py in sample_line((x0, y0), (x1, y1), step=max(0.5, float(res))):
        if not is_free(plan_map, px, py):
            return False
    return True


def config_to_space(config):
    x, y, _ = config
    return (x, y)


def tune_eta(C_space, N):
    """
    Practical heuristic for RRT step-size:
    - scale with (area / N)^(1/d) with d=2 for positional moves
    - clamp to reasonable bounds
    """
    area = C_space.width * C_space.height
    base = (area / max(1, N)) ** (1.0 / 2.0)
    # soften it a bit
    eta = 0.75 * base
    # clamp
    eta = float(np.clip(eta, 2.0, 0.1 * max(C_space.width, C_space.height)))
    return eta

def RRT(Cspace: Cspace,
        plan_map,
        N,
        eta,
        init_config,
        res,
        seed=None,
        goal=None,
        goal_sample_rate=0.05) -> nx.Graph:
    """
    Build a basic RRT (no kinematics). Returns a NetworkX graph whose nodes are 3-tuples (x,y,theta).
    - If eta is None, uses tune_eta.
    - Optional goal bias via goal_sample_rate (chance to sample goal).
    """
    if seed:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if eta is None:
        eta = tune_eta(Cspace, N)

    T = init_graph(Cspace, init_config, plan_map)
    T.graph['eta'] = eta
    T.graph['res'] = res

    node_cnt = len(T.nodes)
    while node_cnt < N:
        # Sample
        if goal is not None and rng.random() < goal_sample_rate:
            rand_config = goal
        else:
            rand_config = Cspace.sample(rng)

        # Nearest
        x_near = nearest_neighbors(T, rand_config, num=1)[0]

        # Steer
        new_config = step_new_state(T.nodes[x_near]['config'], rand_config, eta)

        # Collisions
        if not collision_free_vertex(new_config, plan_map):
            continue
        if not collision_free_edge(T.nodes[x_near]['config'], new_config, res, plan_map):
            continue

        # Add to tree
        x_new = node_cnt
        T.add_node(x_new, config=new_config)
        edge_weight = Cspace.distance(T.nodes[x_near]['config'], new_config)
        T.add_edge(x_near, x_new, weight=edge_weight)

        node_cnt += 1

    return T


def RRG(T_org: nx.Graph, max_deg: int = 8, max_edge_dist: float = 25.0) -> nx.Graph:
    """
    Simple RRG pass: connect each node to up to 'max_deg' nearest other nodes within 'max_dist'
    if the straight-line edge is collision free. Uses T.graph['plan_map'] and T.graph['res'] if present.
    """
    T = T_org.copy()
    Cspace = T.graph['cspace']
    plan_map = T.graph.get('plan_map', None)
    res = float(T.graph.get('res', 1.0))


    for n in T.nodes:
        # find neighbors sorted by distance (skip self)
        cand = sorted(
            ((Cspace.distance(T.nodes[n]['config'], T.nodes[m]['config']), m) for m in T.nodes if m is not n),
            key=lambda p: p[0]
        )

        added = 0
        for d, m in cand:
            if d > max_edge_dist:
                break
            if T.has_edge(n, m):
                continue
            if plan_map is not None:
                if not (collision_free_edge(T.nodes[n]['config'], T.nodes[m]['config'], res, plan_map)):
                    continue
            edge_weight = Cspace.distance(T.nodes[n]['config'], T.nodes[m]['config'])
            T.add_edge(n, m, weight=edge_weight)
            added += 1
            if added >= max_deg:
                break
    return T