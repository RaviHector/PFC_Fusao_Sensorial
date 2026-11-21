# -*- coding: utf-8 -*-
########################################
# Projeto Final de Curso: Fusão Sensorial Aplicada a Veículos Autônomos
# de Escala Reduzida
# Autor: Ravi Hector Lopes Jericó Vieira
# Orientador: Prof. Leonardo Amaral Mozelli
# Curso: Engenharia de Controle e Automação
# Escola de Engenharia
# Universidade Federal de Minas Gerais, 2026
########################################
"""Filtro inercial com 2 estados, no eixo z."""
# --- caminho de importação
if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _d = _os.path.dirname(_os.path.abspath(__file__))
    while _d != _os.path.dirname(_d):
        if _os.path.isdir(_os.path.join(_d, "kalman")):
            if _d not in _sys.path:
                _sys.path.insert(0, _d)
            break
        _d = _os.path.dirname(_d)

import numpy as np

from kalman.config import AMOSTRAS_CALIB_GIRO, SIGMA_BIAS, SIGMA_GYRO, SIGMA_MAG


def matrices(dt, sigma_g=SIGMA_GYRO, sigma_b=SIGMA_BIAS):
    """
    F = [[1, -dt], [0, 1]]      Gamma = [[dt], [0]]
    G = [[dt, 0], [0, 1]]       Q = G * diag(sigma_g^2, sigma_b^2*dt) * G^T
                                  = [[sigma_g^2*dt^2, 0], [0, sigma_b^2*dt]]
    """
    F = np.array([[1.0, -dt], [0.0, 1.0]])
    Gamma = np.array([[dt], [0.0]])
    Q = np.array(
        [
            [sigma_g**2 * dt**2, 0.0],
            [0.0, sigma_b**2 * dt],
        ]
    )
    return F, Gamma, Q


def observation(sigma_mag=SIGMA_MAG):
    """H = [[1, 0]]   R = sigma_mag^2 — yaw medido pelo magnetômetro."""
    H = np.array([[1.0, 0.0]])
    R = sigma_mag**2
    return H, R


def initial(psi0, bias0, sigma_mag=SIGMA_MAG, sigma_g=SIGMA_GYRO, n_samples=AMOSTRAS_CALIB_GIRO):
    """psi0: wrap(rumo do magnetômetro)."""
    x0 = np.array([[psi0], [bias0]])
    R = sigma_mag**2
    P0 = np.array(
        [
            [R, 0.0],
            [0.0, (sigma_g**2) / n_samples],
        ]
    )
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter
    from kalman.utils import wrap

    dt_show = 0.02
    F, Gamma, Q = matrices(dt_show)
    H, R = observation()
    print("=== kalman.filtros.imu2 — matrizes em dt =", dt_show, "===\n")
    print("F =\n", F)
    print("\nGamma =\n", Gamma)
    print("\nH =\n", H)
    print("\nQ =\n", Q)
    print("\nR =", R)

    # Demonstração: rotação a taxa constante, com viés conhecido do
    # giroscópio; o viés estimado converge em poucos segundos.
    rng = np.random.default_rng(3)
    n_steps = 500  # 10 s
    omega_true = 0.3  # [rad/s]
    bias_true = np.deg2rad(2.0)  # [rad/s]

    t = np.arange(n_steps) * dt_show
    psi_true = wrap(omega_true * t)
    gyro_meas = omega_true + bias_true + rng.normal(0, SIGMA_GYRO, n_steps)
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in psi_true])

    kf = KalmanFilter(dim_x=2, angle_indices=[0])
    kf.x, kf.P = initial(psi0=mag_meas[0], bias0=0.0)
    kf.F, kf.Gamma, kf.Q = F, Gamma, Q

    psi_est = np.zeros(n_steps)
    bias_est = np.zeros(n_steps)
    for k in range(n_steps):
        kf.predict(u=[gyro_meas[k]])
        kf.update_scalar(mag_meas[k], H, R, is_angle=True, channel="yaw")
        psi_est[k] = kf.x[0, 0]
        bias_est[k] = kf.x[1, 0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, np.degrees(mag_meas), ".", alpha=0.3, label="magnetômetro (bruto)")
    axs[0].plot(t, np.degrees(psi_true), "k--", label="verdade")
    axs[0].plot(t, np.degrees(psi_est), label="Kalman")
    axs[0].set_ylabel("yaw [graus]")
    axs[0].legend()
    axs[0].set_title("imu2 — verificação rápida")

    axs[1].axhline(np.degrees(bias_true), color="k", ls="--", label="bias verdadeiro")
    axs[1].plot(t, np.degrees(bias_est), label="bias estimado")
    axs[1].set_ylabel("bias [graus/s]")
    axs[1].set_xlabel("t [s]")
    axs[1].legend()

    out = __file__.replace("imu2.py", "_checks/imu2.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    erro_bias_final = abs(np.degrees(bias_est[-1] - bias_true))
    print(f"\nErro final do bias estimado: {erro_bias_final:.3f} graus/s")
    print(f"Figura salva em {out}")
