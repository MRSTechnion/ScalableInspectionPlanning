import math

def angle_wrap(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b in [-pi, pi)."""
    return angle_wrap(a - b)

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate angles along the shortest arc."""
    diff = angle_diff(b, a)  # target - start, shortest signed
    return angle_wrap(a + t * diff)

def sample_line(p0,
                p1,
                step = 1.0):
    """Sample points along a line segment from p0 to p1 at ~'step' spacing."""
    (x0, y0), (x1, y1) = p0, p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        yield (x0, y0)
        return
    n = max(1, int(math.ceil(dist / step)))
    for i in range(n + 1):
        t = i / n
        yield (lerp(x0, x1, t), lerp(y0, y1, t))
