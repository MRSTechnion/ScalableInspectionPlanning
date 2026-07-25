import pickle
from pathlib import Path
from collections import defaultdict

def vis_set_to_groups(vertex_vis_set):
    reverse_dict = defaultdict(list)

    for key, values in vertex_vis_set.items():
        for v in values:
            reverse_dict[int(v)].append(int(key))

    S = dict(reverse_dict)
    I = set(S.keys())

    return I, S


def save_simulated_instance(path, *, G, I, S, vertex_poi_vis, root, meta=None):
    path = Path(path)
    payload = {
        "G": G,
        "I": I,
        "S": S,
        "vertex_poi_vis": vertex_poi_vis,
        "root": root,
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
