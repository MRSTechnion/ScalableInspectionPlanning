from GeoHelpers import *
import math
import random

class Cspace:
    """
    2D position (x, y) in pixel coordinates + heading theta in radians.
    Bounds: 0 <= x < width, 0 <= y < height, theta in [-pi, pi).
    """
    def __init__(self, width: int, height: int, w_theta: float = 0.5):
        self.width = int(width)
        self.height = int(height)
        self.dim = 3  # (x, y, theta)
        # Weight factor for angular distance relative to position distance (pixels).
        # If your turns are "cheap", reduce this. If turns are "expensive", increase this.
        self.w_theta = float(w_theta)

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def distance(self, a, b):
        ax, ay, ath = a
        bx, by, bth = b
        dpos = math.hypot(ax - bx, ay - by)
        dth = angle_diff(ath, bth)
        return dpos + self.w_theta * abs(dth)

    def sample(self, rng):
        x = rng.uniform(0, self.width - 1e-9)
        y = rng.uniform(0, self.height - 1e-9)
        th = rng.uniform(-math.pi, math.pi)
        return (x, y, th)
