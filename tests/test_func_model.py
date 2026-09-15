
from abelha import modelo
import numpy as np
import matplotlib.pyplot as plt

def test_velocity():
    r = np.linspace(0, 10, 100)
    vmax = 800
    rend = 5
    v = modelo.velocity(r, vmax, rend)

    assert v[-1] == 0, "Velocity at the edge should be zero"


def test_rotate_coordinates_and_velocity():
    N = 10
    extent = 5
    a = np.linspace(-extent, extent, N)
    X, Y, Z = np.meshgrid(a, a, a, indexing="ij")

    vx = np.arange(X.size)
    vy = np.arange(X.size)
    vz = np.arange(X.size)

    inc_deg = 30.0
    pa_deg = 45.0

    Xr, Yr, Zr, vxr, vyr, vzr = modelo.rotate_coordinates_and_velocity(X, Y, Z, vx, vy, vz, inc_deg, pa_deg)

    assert vy[-1] != vyr[-1]
    assert vz[-1] != vzr[-1]

def test_normal_do_disco():
    inc_deg = 30.0
    pa_deg = 45.0

    n = modelo.normal_do_disco(inc_deg, pa_deg)
    assert n[2] > 0, "O vetor normal ao disco deve ter componente z positiva após a rotação"


def test_emissividade():
    N = 10
    extent = 5
    X, Y, Z, r, r_safe = modelo.grade_3d(N, extent)

    rend = 5
    emissivity = modelo.emissividade(r_safe, rend)

    assert emissivity[r > rend].max() < emissivity[r <= rend].min()
    assert np.all(emissivity[r <= rend] != 0)