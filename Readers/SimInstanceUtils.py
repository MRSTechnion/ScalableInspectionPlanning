from pathlib import Path
import pickle

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

def load_simulated_instance(path):
    """
    Load and return (G, I, S, vertex_poi_vis, root, meta).
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    return (
        payload["G"],
        payload["I"],
        payload["S"],
        payload["vertex_poi_vis"],
        payload["root"],
        payload.get("meta", {}),
    )