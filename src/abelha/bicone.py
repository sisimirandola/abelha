import numpy as np

def velocity(r, vmax, rend):
    """Somente desaceleração linear: v = vmax (1 - r/rend)."""
    v = np.zeros_like(r, dtype=float)
    m =  (r <= rend)
    v[m] = vmax * (1.0 - r[m] / rend)
    return v

