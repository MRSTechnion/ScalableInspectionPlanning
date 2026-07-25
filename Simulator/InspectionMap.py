import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from typing import Iterable, Optional, Tuple, Sequence, Dict, List
import random

Coord = Tuple[int, int]

class GameMap:
    """2D grid with obstacles/goals + visibility queries.

    Values: 0=empty (white), obstacle_value=1 (brown), start_value=5 (green), goal_value=9 (yellow).
    """

    def __init__(self, width: int, height: int, default_value: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.grid = np.full((self.height, self.width), default_value, dtype=np.int32)
        self.empty_value = default_value
        self.goals: Dict[int, Coord] = {}

    # ----- basic cell ops -----
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def place_object(self, x: int, y: int, value: int = 1):
        if not self.in_bounds(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
        self.grid[y, x] = value

    def get_object(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
        return int(self.grid[y, x])

    def clear_cell(self, x: int, y: int):
        if not self.in_bounds(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
        self.grid[y, x] = self.empty_value

    # ----- batch ops -----
    def place_objects(self, coords: Iterable[Coord], value: int = 1):
        for (x, y) in coords:
            self.place_object(x, y, value)

    # ----- non-overlapping L-shaped obstacles (maze-like) -----
    def add_L_obstacles(
        self,
        count: int,
        *,
        value: int = 1,
        min_len: int = 3,
        max_len: Optional[int] = None,
        thickness: int = 1,
        padding: int = 0,
        forbid: Optional[Sequence[Coord]] = None,
        max_attempts_per_piece: int = 200,
    ) -> int:
        """Place `count` non-overlapping L-shaped obstacles (┌ ┐ └ ┘)."""

        H, W = self.height, self.width
        if max_len is None:
            max_len = max(W, H) // 3 if max(W, H) >= 6 else max(W, H)
        min_len = max(1, int(min_len))
        max_len = max(min_len, int(max_len))
        thickness = max(1, int(thickness))
        padding = max(0, int(padding))

        placed = 0
        occ = (self.grid != self.empty_value)
        forbid = list(forbid) if forbid else []

        def rect_in_bounds(x0: int, y0: int, w: int, h: int) -> bool:
            return 0 <= x0 and 0 <= y0 and x0 + w <= W and y0 + h <= H

        def l_area_free(rects):
            for (x0, y0, w, h) in rects:
                if not rect_in_bounds(x0, y0, w, h):
                    return False
            x0 = min(r[0] for r in rects)
            y0 = min(r[1] for r in rects)
            x1 = max(r[0] + r[2] for r in rects)
            y1 = max(r[1] + r[3] for r in rects)
            xp0, yp0 = max(0, x0 - padding), max(0, y0 - padding)
            xp1, yp1 = min(W, x1 + padding), min(H, y1 + padding)
            if occ[yp0:yp1, xp0:xp1].any():
                return False
            for (fx, fy) in forbid:
                for (rx, ry, rw, rh) in rects:
                    if rx <= fx < rx + rw and ry <= fy < ry + rh:
                        return False
            return True

        def place_rect(x0: int, y0: int, w: int, h: int):
            self.grid[y0:y0 + h, x0:x0 + w] = value
            occ[y0:y0 + h, x0:x0 + w] = True

        for _ in range(max(0, int(count))):
            success = False
            for _attempt in range(max_attempts_per_piece):
                Lx = int(np.random.randint(min_len, max_len + 1))
                Ly = int(np.random.randint(min_len, max_len + 1))
                orientation = int(np.random.randint(0, 4))  # 0: ┌, 1: ┐, 2: └, 3: ┘
                cx = int(np.random.randint(0, W))
                cy = int(np.random.randint(0, H))
                if orientation == 0:
                    rects = [(cx, cy, Lx, thickness), (cx, cy, thickness, Ly)]
                elif orientation == 1:
                    rects = [(cx - (Lx - 1), cy, Lx, thickness), (cx, cy, thickness, Ly)]
                elif orientation == 2:
                    rects = [(cx, cy, Lx, thickness), (cx, cy - (Ly - 1), thickness, Ly)]
                else:
                    rects = [(cx - (Lx - 1), cy, Lx, thickness), (cx, cy - (Ly - 1), thickness, Ly)]
                if not l_area_free(rects):
                    continue
                for (x0, y0, w, h) in rects:
                    place_rect(x0, y0, w, h)
                placed += 1
                success = True
                break
            if not success:
                break
        return placed

    # ----- goals with numeric IDs -----
    def scatter_goals(
        self,
        count: int,
        *,
        value: int = 9,
        forbid: Optional[Sequence[Coord]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> Dict[int, Coord]:

        mask = (self.grid == self.empty_value)
        if forbid:
            for (fx, fy) in forbid:
                if self.in_bounds(fx, fy):
                    mask[fy, fx] = False
        coords = np.argwhere(mask)  # (y, x)
        if coords.size == 0 or count <= 0:
            return {}
        k = int(min(max(0, count), coords.shape[0]))
        choose = np.random.choice(coords.shape[0], size=k, replace=False)
        sel = coords[choose]
        self.grid[sel[:, 0], sel[:, 1]] = value
        existing = sorted(self.goals.keys())
        next_id = (existing[-1] + 1) if existing else 1
        if labels is None:
            ids = list(range(next_id, next_id + k))
        else:
            if len(labels) < k:
                raise ValueError("labels length < number of placed goals")
            ids = list(labels[:k])
        placed: Dict[int, Coord] = {}
        for i, (r, c) in enumerate(sel):  # Use r, c to avoid confusion
            gid = ids[i]
            # Store as (x, y) which corresponds to (col, row)
            self.goals[gid] = (int(c), int(r))
            placed[gid] = (int(c), int(r))
        return placed

    # ----- Bresenham grid line -----
    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> List[Coord]:
        points: List[Coord] = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    @staticmethod
    def _angle_deg(dx: float, dy: float) -> float:
        """Angle in degrees from +x, increasing CCW if y up; here y grows down so it's screen-space."""
        ang = np.degrees(np.arctan2(dy, dx)) % 360.0
        return float(ang)

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def goals_visible_from(
        self,
        x: int,
        y: int,
        *,
        obstacle_value: int = 1,
        max_view_distance: Optional[float] = None,
        view_angle_deg: Optional[float] = None,
        fov_deg: float = 360.0,
    ) -> List[int]:
        """IDs of goals visible from (x,y), with optional distance and FOV constraints.

        Angles are absolute degrees from +x. This uses screen coordinates (y increases downward).
        """
        if not self.in_bounds(x, y):
            return []
        visible: List[int] = []
        for gid, (gx, gy) in self.goals.items():
            dx, dy = (gx - x), (gy - y)
            dist = float(np.hypot(dx, dy))
            if max_view_distance is not None and dist > max_view_distance:
                continue
            if view_angle_deg is not None and fov_deg < 360.0:
                ang = self._angle_deg(dx, dy)
                if self._ang_diff(ang, float(view_angle_deg) % 360.0) > fov_deg / 2.0:
                    continue
            cells = self._bresenham(x, y, gx, gy)
            mid = cells[1:-1] if len(cells) > 2 else []
            blocked = any(self.grid[yy, xx] == obstacle_value for (xx, yy) in mid)
            if not blocked:
                visible.append(gid)
        return visible

    # ----- colored rendering -----
    def show(
        self,
        *,
        start_value: int = 5,
        goal_value: int = 9,
        obstacle_value: int = 1,
        title: str = 'Game Map',
    ):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        idx = np.zeros_like(self.grid, dtype=np.int8)
        idx[self.grid == obstacle_value] = 1
        idx[self.grid == start_value] = 2
        idx[self.grid == goal_value] = 3
        cmap = ListedColormap(['#FFFFFF', '#8B4513', '#00AA00', '#FFD700'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        fig, ax = plt.subplots()
        ax.imshow(
            idx,
            cmap=cmap,
            norm=norm,
            origin='upper',
            interpolation='nearest',
            extent=(-0.5, self.width - 0.5, self.height - 0.5, -0.5),
        )
        for gid, (gx, gy) in self.goals.items():
            ax.text(gx, gy, str(gid), ha='center', va='center', fontsize=7, color='black')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # lock the axes so clicks/overlays don't change size
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_autoscale_on(False)
        plt.show()

    def show_interactive(
            self,
            *,
            obstacle_value: int = 1,
            start_marker_value: int = 5,
            goal_value: int = 9,
            title: str = 'Click to query visibility',
            max_view_distance: Optional[float] = None,
            view_angle_deg: Optional[float] = None,
            fov_deg: float = 360.0,
    ):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Create a background showing ONLY obstacles
        # 0 = empty, 1 = obstacle
        bg_idx = np.zeros_like(self.grid, dtype=np.int8)
        bg_idx[self.grid == obstacle_value] = 1

        cmap = ListedColormap(['#FFFFFF', '#8B4513'])  # White background, Brown obstacles
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        ax.imshow(
            bg_idx,
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            origin='upper',
            interpolation='nearest',
            extent=(-0.5, self.width - 0.5, self.height - 0.5, -0.5),
        )

        # 2. Draw Goals/POIs as Circles
        for gid, (gx, gy) in self.goals.items():
            circle = plt.Circle((gx, gy), 0.35, color='#FFD700', ec='black', lw=1, zorder=3)
            ax.add_patch(circle)
            ax.text(gx, gy, str(gid), ha='center', va='center', fontsize=7, color='black', zorder=4)

        # 3. Draw Start Points (Root) as Circles
        start_coords = np.argwhere(self.grid == start_marker_value)
        for sy, sx in start_coords:
            root_circle = plt.Circle((sx, sy), 0.4, color='#00AA00', ec='black', lw=1.5, zorder=3)
            ax.add_patch(root_circle)

        # UI Setup
        plt.xlim([-0.5, self.width - 0.5])
        plt.ylim([self.height - 0.5, -0.5])
        ax.set_aspect('equal')
        ax.set_title(title)

        overlays: List = []

        # Force these at the end of every plotting function
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # This ensures the map doesn't 'shrink' to make room for non-existent labels
        # ax.set_axis_off()

        def clear_overlays():
            while overlays:
                artist = overlays.pop()
                try:
                    artist.remove()
                except Exception:
                    pass
            fig.canvas.draw_idle()

        def draw_fov(x: int, y: int):
            if view_angle_deg is None and (max_view_distance is None or np.isinf(max_view_distance) or max_view_distance <= 0):
                return None
            radius = max_view_distance if (max_view_distance is not None and max_view_distance > 0) else float(np.hypot(self.width, self.height))
            if view_angle_deg is None or fov_deg >= 360.0:
                if np.isfinite(radius):
                    w = Wedge((x, y), radius, 0, 360, width=0)
                    w.set_alpha(0.1)
                    w.set_clip_on(True)
                    overlays.append(ax.add_patch(w))
                    return w
                return None
            theta1 = (view_angle_deg - fov_deg/2.0) % 360.0
            theta2 = (view_angle_deg + fov_deg/2.0) % 360.0
            w = Wedge((x, y), radius, theta1, theta2, width=0)
            w.set_alpha(0.1)
            w.set_clip_on(True)
            overlays.append(ax.add_patch(w))
            return w

        def onclick(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            if not self.in_bounds(x, y):
                return
            vis = self.goals_visible_from(
                x, y,
                obstacle_value=obstacle_value,
                max_view_distance=max_view_distance,
                view_angle_deg=view_angle_deg,
                fov_deg=fov_deg,
            )
            clear_overlays()
            overlays.append(ax.plot([x], [y], marker='o')[0])
            draw_fov(x, y)
            for gid in vis:
                gx, gy = self.goals[gid]
                overlays.append(ax.plot([x, gx], [y, gy], linestyle='--')[0])
            # ax.set_title(f"{title} Coordinates=({x},{y}), visible goals: {vis}")
            fig.canvas.draw_idle()

        def onkey(event):
            if event.key == 'r':
                clear_overlays()
                ax.set_title(title)
                fig.canvas.draw_idle()

        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

    # ----- numpy helpers -----
    def to_numpy(self) -> np.ndarray:
        return self.grid

    def copy(self) -> 'GameMap':
        gm = GameMap(self.width, self.height, self.empty_value)
        gm.grid = self.grid.copy()
        gm.goals = dict(self.goals)
        return gm


# Example usage
if __name__ == "__main__":
    random.seed(3)
    np.random.seed(3)

    # 100x100 map with 50 obstacles and 50 goals
    game_map = GameMap(20, 20)

    game_map.add_L_obstacles(
        count=10, value=1, min_len=3, max_len=10,
        thickness=1, padding=1, forbid=[(1, 1)]
    )

    start = (1, 1)
    game_map.place_object(*start, 5)
    placed = game_map.scatter_goals(10, value=9, forbid=[start])
    print("Placed goals:", placed)

    game_map.show_interactive(
        title='',
        max_view_distance=np.inf,
        view_angle_deg=30.0,
        fov_deg=360.0,
    )
