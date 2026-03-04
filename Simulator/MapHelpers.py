

def _map_in_bounds(plan_map, ix, iy):
    if hasattr(plan_map, "in_bounds"):
        return plan_map.in_bounds(ix, iy)
    h, w = plan_map.shape
    return 0 <= ix < w and 0 <= iy < h

def _map_empty_value(plan_map):
    if hasattr(plan_map, "empty_value"):
        return int(plan_map.empty_value)
    return 0  # numpy-array convention

def _map_get(plan_map, ix, iy):
    if hasattr(plan_map, "get_object"):
        return int(plan_map.get_object(ix, iy))
    return int(plan_map[iy, ix])

def is_free(plan_map, x, y):
    """
    True iff (x,y) lies inside the map and the cell equals the map's empty_value.
    - For GameMap: 0 = empty, 1 = obstacle, 5 = start, 9 = goal (treated as NOT free here).
    - For ndarray: 0 = empty; any nonzero treated as occupied.
    """
    ix, iy = int(round(x)), int(round(y))
    if not _map_in_bounds(plan_map, ix, iy):
        return False
    return _map_get(plan_map, ix, iy) == _map_empty_value(plan_map)
