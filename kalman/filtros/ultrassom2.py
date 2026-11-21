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
"""Filtro do ultrassom com 2 estados."""
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

from kalman.config import SIGMA_D0_US2, SIGMA_US, SIGMA_US_ACCEL


def matrices(dt, sigma_alpha=SIGMA_US_ACCEL, sigma_us=SIGMA_US):
    """
    F = [[1, dt], [0, 1]]     G = [[dt^2/2], [dt]]
    H = [[1, 0]]              Q = G * sigma_alpha^2 * G^T
    R = sigma_us^2            w = aceleração relativa [m/s^2]
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[dt**2 / 2.0], [dt]])
    H = np.array([[1.0, 0.0]])
    Q = (sigma_alpha**2) * (G @ G.T)
    R = sigma_us**2
    return F, H, Q, R


def initial(d0, sigma_us=SIGMA_US, sigma_d0dot=SIGMA_D0_US2):
    """d0 = primeira leitura, ddot0 = 0. P0 = diag(R, sigma_d0dot^2)."""
    x0 = np.array([[d0], [0.0]])
    R = sigma_us**2
    P0 = np.array([[R, 0.0], [0.0, sigma_d0dot**2]])
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter

    dt_show = 0.05
    F, H, Q, R = matrices(dt_show)
    print("=== kalman.filtros.ultrassom2 — matrizes em dt =", dt_show, "===\n")
    print("F =\n", F)
    print("\nH =\n", H)
    print("\nQ =\n", Q)
    print("\nR =", R)

    # Mesma aproximação do ultrassom1.py: o atraso some, e ddot converge
    # para -v_approach.
    rng = np.random.default_rng(2)
    n_steps = 100
    v_approach = 1.0  # [m/s]
    d0_true = 2.0  # [m]
    sigma_us = SIGMA_US

    t = np.arange(n_steps) * dt_show
    d_true = d0_true - v_approach * t
    raw = d_true + rng.normal(0, sigma_us, n_steps)

    x0, P0 = initial(raw[0])
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P = x0, P0
    kf.F, kf.Q = F, Q

    d_est = np.zeros(n_steps)
    ddot_est = np.zeros(n_steps)
    d_est[0], ddot_est[0] = kf.x[0, 0], kf.x[1, 0]
    for k in range(1, n_steps):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        d_est[k] = kf.x[0, 0]
        ddot_est[k] = kf.x[1, 0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, raw, ".", alpha=0.3, label="bruto")
    axs[0].plot(t, d_true, "k--", label="verdade")
    axs[0].plot(t, d_est, label="Kalman (2 estados)")
    axs[0].set_ylabel("d [m]")
    axs[0].legend()
    axs[0].set_title("ultrassom2 — verificação rápida (atraso eliminado)")

    axs[1].axhline(-v_approach, color="k", ls="--", label="verdade (ddot = -v_ego)")
    axs[1].plot(t, ddot_est, label="Kalman (ddot)")
    axs[1].set_ylabel("ddot [m/s]")
    axs[1].set_xlabel("t [s]")
    axs[1].legend()

    out = __file__.replace("ultrassom2.py", "_checks/ultrassom2.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    atraso_medio = float(np.mean(d_est[20:] - d_true[20:]))
    print(f"\nAtraso médio (viés), após transitório: {atraso_medio*100:.1f} cm (esperado ~0)")
    print(f"Figura salva em {out}")
