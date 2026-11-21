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
"""Filtro inercial com 6 estados."""
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

from kalman.config import (AMOSTRAS_CALIB_GIRO, SIGMA_ACC_TILT, SIGMA_BIAS,
                           SIGMA_GYRO, SIGMA_MAG)


def matrices(dt, sigma_g=SIGMA_GYRO, sigma_b=SIGMA_BIAS):
    """
    F = [[I3, -dt*I3], [0, I3]]     Gamma = [[dt*I3], [0]]
    Q = diag(sigma_g^2*dt^2 * I3, sigma_b^2*dt * I3)
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    F = np.block([[I3, -dt * I3], [Z3, I3]])
    Gamma = np.vstack([dt * I3, Z3])
    Q = np.block(
        [
            [(sigma_g**2 * dt**2) * I3, Z3],
            [Z3, (sigma_b**2 * dt) * I3],
        ]
    )
    return F, Gamma, Q


def observation(mag_ok, sigma_phi=SIGMA_ACC_TILT, sigma_theta=SIGMA_ACC_TILT, sigma_mag=SIGMA_MAG):
    """
    COM magnetômetro (mag_ok=True): H = [I3 | 0_3] (3 linhas),
        y = [phi_acc, theta_acc, psi_mag]^T, R = diag(sigma_phi^2, sigma_theta^2, sigma_mag^2)
        observabilidade: 6 de 6.
    SEM magnetômetro: H = 2 linhas (roll, pitch) apenas,
        y = [phi_acc, theta_acc]^T, R = diag(sigma_phi^2, sigma_theta^2)
        observabilidade: 4 de 6; rumo e b_psi não são observáveis.
    """
    if mag_ok:
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        R = np.diag([sigma_phi**2, sigma_theta**2, sigma_mag**2])
    else:
        H = np.hstack([np.eye(2, 3), np.zeros((2, 3))])
        R = np.diag([sigma_phi**2, sigma_theta**2])
    return H, R


def initial(rpy0, bias0=(0.0, 0.0, 0.0), sigma_phi=SIGMA_ACC_TILT,
            sigma_theta=SIGMA_ACC_TILT, sigma_mag=SIGMA_MAG,
            sigma_g=SIGMA_GYRO, n_samples=AMOSTRAS_CALIB_GIRO):
    """rpy0 = (roll0, pitch0, yaw0) [rad] da primeira leitura."""
    x0 = np.array([[rpy0[0]], [rpy0[1]], [rpy0[2]], [bias0[0]], [bias0[1]], [bias0[2]]])
    var_bias = (sigma_g**2) / n_samples
    P0 = np.diag([sigma_phi**2, sigma_theta**2, sigma_mag**2, var_bias, var_bias, var_bias])
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter
    from kalman.utils import wrap

    dt_show = 0.02
    F, Gamma, Q = matrices(dt_show)
    H, R = observation(mag_ok=True)
    print("=== kalman.filtros.imu6 — matrizes em dt =", dt_show, "===\n")
    print("F =\n", F)
    print("\nGamma =\n", Gamma)
    print("\nQ =\n", Q)
    print("\nH (mag_ok=True) =\n", H)
    print("\nR (mag_ok=True) =\n", R)
    H_no_mag, R_no_mag = observation(mag_ok=False)
    print("\nH (mag_ok=False) =\n", H_no_mag)
    print("\nR (mag_ok=False) =\n", R_no_mag)

    # Demonstração: taxas angulares constantes nos 3 eixos, com
    # magnetômetro disponível.
    rng = np.random.default_rng(4)
    n_steps = 500  # 10 s
    rates_true = np.array([0.10, -0.05, 0.20])  # [rad/s] phi, theta, psi

    t = np.arange(n_steps) * dt_show
    rpy_true = np.zeros((n_steps, 3))
    rpy_true[:, 2] = wrap(rates_true[2] * t)
    rpy_true[:, 0] = rates_true[0] * t
    rpy_true[:, 1] = rates_true[1] * t

    gyro_meas = rates_true[None, :] + rng.normal(0, SIGMA_GYRO, (n_steps, 3))
    acc_meas = rpy_true[:, :2] + rng.normal(0, SIGMA_ACC_TILT, (n_steps, 2))
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in rpy_true[:, 2]])

    kf = KalmanFilter(dim_x=6, angle_indices=[0, 1, 2])
    kf.x, kf.P = initial(rpy0=(acc_meas[0, 0], acc_meas[0, 1], mag_meas[0]))
    kf.F, kf.Gamma, kf.Q = F, Gamma, Q

    est = np.zeros((n_steps, 3))
    for k in range(n_steps):
        kf.predict(u=gyro_meas[k])
        kf.update_scalar(acc_meas[k, 0], H[0], R[0, 0], is_angle=True, channel="roll")
        kf.update_scalar(acc_meas[k, 1], H[1], R[1, 1], is_angle=True, channel="pitch")
        kf.update_scalar(mag_meas[k], H[2], R[2, 2], is_angle=True, channel="yaw")
        est[k] = kf.x[0:3, 0]

    labels = ["roll (phi)", "pitch (theta)", "yaw (psi)"]
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i in range(3):
        axs[i].plot(t, np.degrees(rpy_true[:, i]), "k--", label="verdade")
        axs[i].plot(t, np.degrees(est[:, i]), label="Kalman")
        axs[i].set_ylabel(f"{labels[i]} [graus]")
        axs[i].legend()
    axs[0].set_title("imu6 — verificação rápida, com magnetômetro")
    axs[2].set_xlabel("t [s]")

    out = __file__.replace("imu6.py", "_checks/imu6.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    erro_final = np.degrees(np.abs(wrap(est[-1] - rpy_true[-1])))
    print(f"\nErro final (graus) roll/pitch/yaw: {erro_final}")
    print(f"Figura salva em {out}")
