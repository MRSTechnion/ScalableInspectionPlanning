
def pick_exp(Experiment):
    if Experiment == "Crisp1000":
        vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n1000_g1_vertex"
        edge_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n1000_g1_edge"
        conf_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n1000_g1_conf"
    elif Experiment == "Crisp2000":
        vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n2000_g1_vertex"
        edge_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n2000_g1_edge"
        conf_file = "/home/adir/GIP_data/wafr24/base_graphs/crisp_n2000_g1_conf"
    elif Experiment == "Drone1000":
        vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n1000_g1_vertex"
        edge_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n1000_g1_edge"
        conf_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n1000_g1_conf"
    elif Experiment == "Drone2000":
        vertex_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n2000_g1_vertex"
        edge_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n2000_g1_edge"
        conf_file = "/home/adir/GIP_data/wafr24/base_graphs/drone_n2000_g1_conf"

    return vertex_file, edge_file, conf_file
