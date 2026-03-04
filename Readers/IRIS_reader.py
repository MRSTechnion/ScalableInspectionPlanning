import matplotlib.pyplot as plt
import networkx as nx
import GraphDrawing
from Utils import IP_to_Group

def read_IRIS_to_inspection_graph(vertex_file, edge_file, conf_file):

    # Initialize graph
    G = nx.Graph()

    # --- Load vertices and POIs ---
    vertex_poi_vis = {}

    with open(vertex_file, "r") as vf:
        for line in vf:
            parts = line.strip().split()
            if not parts:
                continue
            vertex_id = int(parts[0])
            pois = set([int(p) for p in parts[3:]])  # remaining entries are POIs
            vertex_poi_vis[vertex_id] = pois
            G.add_node(vertex_id)

    # --- Load edges with weights ---
    with open(edge_file, "r") as ef:
        for line in ef:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            v1, v2, weight = int(parts[0]), int(parts[1]), float(parts[-1])
            G.add_edge(v1, v2, weight=weight)

    # with open(conf_file, "r") as cf:
    #     for line in cf:
    #         parts = line.strip().split()

    return G, vertex_poi_vis

if __name__ == '__main__':
    # Paths to your files
    vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n2000_g1_vertex"
    edge_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n2000_g1_edge"

    G, vertex_poi_vis = read_IRIS_to_inspection_graph(vertex_file, edge_file)

    I, S = IP_to_Group.vis_set_to_groups(vertex_poi_vis)

    print(G)
    print(f"POIs number - {len(I)}")

    # GraphDrawing.draw_weighted_with_sets(G, S)
    # plt.show()