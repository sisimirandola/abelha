from abelha import modelo
import numpy as np
import matplotlib.pyplot as plt


N = 100
extent = 50
vmax = 800     # km/s
rend = 40      # pc
rturn = rend/2     # pc
v0 = 600      # km/s
vmin = 100    # km/s
pa_deg = 0.0    # sem PA por enquanto -- mude aqui quando quiser ligar
modo_velocity = 'x'
thetain_deg=10.0
thetaout_deg=30.0

inclinations = [0.0, 30.0, 60.0, 90.0]

if __name__ == "__main__":
 
    # uma linha por inclinação, bicone e esfera lado a lado (cada um no
    # seu próprio eixo -- antes os dois imshow caíam no mesmo ax e o
    # segundo cobria o primeiro por cima)
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), constrained_layout=True)
 
    for i, inc in enumerate(inclinations):
 
        v_map_bicone = modelo.main_bicone(N, extent, v0, vmax, rend, rturn, thetain_deg, thetaout_deg, modo_velocity, inc, pa_deg)

        im = axes[i, 0].imshow(v_map_bicone.T, origin='lower', extent=[-extent, extent, -extent, extent], cmap='plasma', vmin=-vmax, vmax=vmax)
        axes[i, 0].set_xlabel("x [pc]"); axes[i, 0].set_ylabel("y [pc]")
        axes[i, 0].set_title(rf"Bicone, $i={inc:.0f}^\circ$")
        fig.colorbar(im, ax=axes[i, 0], label=r"$\langle v_{LOS} \rangle$ [km/s]")
 
        v_map_esfera = modelo.main_esfera(N, extent, v0, vmax, rend, rturn, modo_velocity, inc, pa_deg)

        im = axes[i, 1].imshow(v_map_esfera.T, origin='lower', extent=[-extent, extent, -extent, extent], cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_xlabel("x [pc]"); axes[i, 1].set_ylabel("y [pc]")
        axes[i, 1].set_title(rf"Esfera, $i={inc:.0f}^\circ$")
        fig.colorbar(im, ax=axes[i, 1], label=r"$\langle v_{LOS} \rangle$ [km/s]")
 
    plt.show()
