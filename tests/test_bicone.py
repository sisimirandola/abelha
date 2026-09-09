
from abelha import bicone
import numpy as np

def test_velocity():
    r = np.linspace(0, 10, 100)
    vmax = 800
    rend = 5
    v = bicone.velocity(r, vmax, rend)

    assert v[-1] == 0, "Velocity at the edge should be zero"