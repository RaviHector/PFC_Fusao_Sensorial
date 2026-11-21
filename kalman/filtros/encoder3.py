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
"""Filtro do encoder com 3 estados."""
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

from kalman.config import SIGMA_A0_ENC3, SIGMA_ENC, SIGMA_JERK_ENC3


def matrices(dt, sigma_jerk=SIGMA_JERK_ENC3, sigma_enc=SIGMA_ENC):
    """
    F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
    G = [[dt^3/6], [dt^2/2], [dt]]
    H = [[0, 1, 0]]
    Q = G * sigma_jerk^2 * G^T
    R = sigma_enc^2          w = jerk desconhecido [m/s^3]
    """
    F = np.array(
        [
            [1.0, dt, dt**2 / 2.0],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    G = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]])
    H = np.array([[0.0, 1.0, 0.0]])
    Q = (sigma_jerk**2) * (G @ G.T)
    R = sigma_enc**2
    return F, H, Q, R


def initial(sigma_enc=SIGMA_ENC, sigma_a0=SIGMA_A0_ENC3):
    """P0 = diag(0, R, sigma_a0^2): veículo parado, aceleração inicial incerta."""
    x0 = np.array([[0.0], [0.0], [0.0]])
    R = sigma_enc**2
    P0 = np.diag([0.0, R, sigma_a0**2])
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter

    dt_show = 0.02
    F, H, Q, R = matrices(dt_show)
    print("=== kalman.filtros.encoder3 — matrizes em dt =", dt_show, "===\n")
    print("F =\n", F)
    print("\nH =\n", H)
    print("\nQ =\n", Q)
    print("\nR =", R)

    # Demonstração: aceleração constante, só a velocidade é medida
    # (H = [0, 1, 0]); a aceleração é inferida.
    rng = np.random.default_rng(1)
    n_steps = 250
    a_true = 0.3  # [m/s^2]
    sigma_enc = SIGMA_ENC

    kf = KalmanFilter(dim_x=3)
    kf.x, kf.P = initial()
    kf.F, kf.Q = F, Q

    t = np.arange(n_steps) * dt_show
    v_true = a_true * t
    raw = v_true + rng.normal(0, sigma_enc, n_steps)
    a_est = np.zeros(n_steps)
    v_est = np.zeros(n_steps)

    for k in range(n_steps):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="v")
        v_est[k] = kf.x[1, 0]
        a_est[k] = kf.x[2, 0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, raw, ".", alpha=0.3, label="bruto (v)")
    axs[0].plot(t, v_true, "k--", label="verdade (v)")
    axs[0].plot(t, v_est, label="Kalman (v)")
    axs[0].set_ylabel("v [m/s]")
    axs[0].legend()
    axs[0].set_title("encoder3 — verificação rápida")

    axs[1].axhline(a_true, color="k", ls="--", label="verdade (a)")
    axs[1].plot(t, a_est, label="Kalman (a), inferida")
    axs[1].set_ylabel("a [m/s^2]")
    axs[1].set_xlabel("t [s]")
    axs[1].legend()

    out = __file__.replace("encoder3.py", "_checks/encoder3.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    erro_final_a = abs(a_est[-1] - a_true)
    print(f"\nErro final na aceleração inferida: {erro_final_a:.4f} m/s^2")
    print(f"Figura salva em {out}")
