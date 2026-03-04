from collections import defaultdict

def vis_set_to_groups(vertex_vis_set):
    reverse_dict = defaultdict(list)

    for key, values in vertex_vis_set.items():
        for v in values:
            reverse_dict[int(v)].append(int(key))

    S = dict(reverse_dict)
    I = set(S.keys())

    return I, S

