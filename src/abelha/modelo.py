import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

def velocity_both(r, rturn, vmax, rend):
    """Aceleração linear até rturn e desaceleração linear até rend.

    (Das Sen et al. 2005, https://arxiv.org/abs/astro-ph/0505103)
    """
    v = np.zeros_like(r, dtype=float)
    k_ace = vmax / rturn             # subida: 0 -> vmax em [0, rturn]
    k_des = vmax / (rend - rturn)    # descida: vmax -> 0 em [rturn, rend]
    m_up = (r >= 0) & (r <= rturn)
    m_dn = (r > rturn) & (r <= rend)
    v[m_up] = k_ace * r[m_up]
    v[m_dn] = vmax - k_des * (r[m_dn] - rturn)
    return v


def velocity(r, vmax, rend):
    """Somente desaceleração linear: v = vmax (1 - r/rend)."""
    v = np.zeros_like(r, dtype=float)
    m =  (r <= rend)
    v[m] = vmax * (1.0 - r[m] / rend)
    return v


def grade_3d(N, extent):

    x = np.linspace(-extent, extent, N)
    y = np.linspace(-extent, extent, N)
    z = np.linspace(-extent, extent, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.where(r == 0, np.nan, r)
    return X, Y, Z, r, r_safe

def rotation_matrix(inc_deg, pa_deg):
    """
    z = linha de visada (LOS)

    O eixo do bicone inicialmente coincide com +z.

    Primeiro:
        inclinação i em torno do eixo y.

    Depois:
        PA em torno do eixo z.

    """
    i = inc_deg * np.pi / 180.0
    pa = pa_deg * np.pi / 180.0

    Ry = np.array([[np.cos(i), 0, np.sin(i)],
                   [0, 1, 0],
                   [-np.sin(i), 0, np.cos(i)]])

    Rz = np.array([[np.cos(pa), -np.sin(pa), 0],
                   [np.sin(pa), np.cos(pa), 0],
                   [0, 0, 1]])

    return Ry, Rz
 
def rotate_coordinates_and_velocity(X, Y, Z, vx, vy, vz, inc_deg, pa_deg):
    """
    v_los = componente da velocidade ao longo da linha de visada do
    observador (eixo z).
    """

    Ry, Rz = rotation_matrix(inc_deg, pa_deg)


    # 1) inclinação em torno de y
    Xr = Ry[0,0]*X + Ry[0,1]*Y + Ry[0,2]*Z
    Yr = Ry[1,0]*X + Ry[1,1]*Y + Ry[1,2]*Z
    Zr = Ry[2,0]*X + Ry[2,1]*Y + Ry[2,2]*Z

    vxr = Ry[0,0]*vx + Ry[0,1]*vy + Ry[0,2]*vz
    vyr = Ry[1,0]*vx + Ry[1,1]*vy + Ry[1,2]*vz
    vzr = Ry[2,0]*vx + Ry[2,1]*vy + Ry[2,2]*vz


    #Aplicando a rotação em torno do eixo z (PA) 

    Xrr = Rz[0, 0]*Xr + Rz[0, 1]*Yr + Rz[0, 2]*Zr
    Yrr = Rz[1, 0]*Xr + Rz[1, 1]*Yr + Rz[1, 2]*Zr
    Zrr = Rz[2, 0]*Xr + Rz[2, 1]*Yr + Rz[2, 2]*Zr

    vxrr = Rz[0, 0]*vxr + Rz[0, 1]*vyr + Rz[0, 2]*vzr
    vyrr = Rz[1, 0]*vxr + Rz[1, 1]*vyr + Rz[1, 2]*vzr
    vzrr = Rz[2, 0]*vxr + Rz[2, 1]*vyr + Rz[2, 2]*vzr

    v_los = vzrr  

    return Xrr, Yrr, Zrr, vxrr, vyrr, v_los

def normal_do_disco(i_disk_deg, pa_disk_deg):
    """Versor normal ao plano de poeira, no referencial do observador.

    O sinal e fixado para n apontar para o observador (n_z >= 0).
    """
    n = np.array([0.0, 0.0, 1.0])
 
    # rotation_matrix 
    Ry, Rz = rotation_matrix(i_disk_deg, pa_disk_deg)
 
    nxr = Ry[0, 0]*n[0] + Ry[0, 1]*n[1] + Ry[0, 2]*n[2]
    nyr = Ry[1, 0]*n[0] + Ry[1, 1]*n[1] + Ry[1, 2]*n[2]
    nzr = Ry[2, 0]*n[0] + Ry[2, 1]*n[1] + Ry[2, 2]*n[2]
 
    nxrr = Rz[0, 0]*nxr + Rz[0, 1]*nyr + Rz[0, 2]*nzr
    nyrr = Rz[1, 0]*nxr + Rz[1, 1]*nyr + Rz[1, 2]*nzr
    nzrr = Rz[2, 0]*nxr + Rz[2, 1]*nyr + Rz[2, 2]*nzr
 
    n = np.array([nxrr, nyrr, nzrr])
    if n[2] < 0:
        n = -n
    return n



def emissividade(r, rend, tau_flux=5.0):
    """Decaimento exponencial do fluxo com a distancia radial (Bae & Woo 2016)."""
    return np.exp(-tau_flux * r / rend)


def campo_v_los(v_los, Xr, Yr, Zr, mask, extent, N):

    """
    Calcula a velocidade média ao longo da linha de visada (LOS) para cada pixel
    do mapa 2D.
    """
    x_valid = Xr[mask]
    y_valid = Yr[mask]
    z_valid = Zr[mask]
    v_valid = v_los[mask]

    edges = np.linspace(-extent, extent, int(N/2))

    v_map, xedges, yedges, binnumber = binned_statistic_2d(x_valid, y_valid, v_valid, statistic='mean', bins=[edges, edges])
    print(x_valid.shape, Xr.shape)

    return v_map



def main_bicone(N, extent, v0, vmax, rend, rturn, thetain_deg, thetaout_deg, modo_velocity, inc_deg=0.0, pa_deg=0.0):

    X, Y, Z, r, r_safe = grade_3d(N, extent)

    cos_theta = Z / r_safe
    theta = np.degrees(np.arccos(cos_theta))


    theta_dobrado = np.minimum(theta, 180 - theta)
    dentro_cone = (theta_dobrado >= thetain_deg) & (theta_dobrado <= thetaout_deg)
    mask = dentro_cone & (r <= rend) & (r > 0)
 
    if modo_velocity == 'both':
        vr = velocity_both(r, rturn, vmax, rend)
    else:
        vr = velocity(r, vmax, rend)  # apenas desaceleração linear

    vx, vy, vz = vr / r_safe * X, vr / r_safe * Y, vr / r_safe * Z


    # rotação (inclinação + PA) -- posição e velocidade no referencial do observador
    Xr, Yr, Zr, vxr, vyr, v_los = rotate_coordinates_and_velocity(X, Y, Z, vx, vy, vz, inc_deg, pa_deg)

    v_map = campo_v_los(v_los, Xr, Yr, Zr, mask, extent, N)

    return v_map


def main_esfera(N, extent, v0, vmax, rend, rturn, modo_velocity, inc_deg, pa_deg):
    X, Y, Z, r, r_safe = grade_3d(N, extent)

    # isotropico -- sem filtro de angulo, todo theta tem gas
    mask = (r <= rend) & (r > 0)

    if modo_velocity == 'both':
        vr = velocity_both(r, rturn, vmax, rend)
    else:
        vr = velocity(r, vmax, rend)  # apenas desaceleração linear

    vx, vy, vz = vr / r_safe * X, vr / r_safe * Y, vr / r_safe * Z

    Xr, Yr, Zr, vxr, vyr, v_los = rotate_coordinates_and_velocity(X, Y, Z, vx, vy, vz, inc_deg, pa_deg)
    v_map = campo_v_los(v_los, Xr, Yr, Zr, mask, extent, N)

    return v_map
