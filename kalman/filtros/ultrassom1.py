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
"""Filtro do ultrassom com 1 estado."""
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

from kalman.config import SIGMA_US, SIGMA_US_WALK


def matrices(dt=None, sigma_walk=SIGMA_US_WALK, sigma_us=SIGMA_US):
    """F = [[1]] H = [[1]] Q = [[sigma_walk^2]] R = [[sigma_us^2]]"""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[sigma_walk**2]])
    R = sigma_us**2
    return F, H, Q, R


def initial(d0, sigma_us=SIGMA_US):
    """d0 = primeira leitura válida. P0 = R."""
    x0 = np.array([[d0]])
    R = sigma_us**2
    P0 = np.array([[R]])
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter

    dt_show = 0.05
    F, H, Q, R = matrices(dt_show)
    print("=== kalman.filtros.ultrassom1 — matrizes ===\n")
    print("F =\n", F)
    print("\nH =\n", H)
    print("\nQ =\n", Q)
    print("\nR =", R)

    # Demonstração: aproximação a uma parede a velocidade constante, com o
    # atraso sistemático do modelo de 1 estado.
    rng = np.random.default_rng(2)
    n_steps = 100
    v_approach = 1.0  # [m/s]
    d0_true = 2.0  # [m]
    sigma_us = SIGMA_US

    t = np.arange(n_steps) * dt_show
    d_true = d0_true - v_approach * t
    raw = d_true + rng.normal(0, sigma_us, n_steps)

    x0, P0 = initial(raw[0])
    kf = KalmanFilter(dim_x=1)
    kf.x, kf.P = x0, P0
    kf.F, kf.Q = F, Q

    d_est = np.zeros(n_steps)
    d_est[0] = kf.x[0, 0]
    for k in range(1, n_steps):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        d_est[k] = kf.x[0, 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, raw, ".", alpha=0.3, label="bruto")
    ax.plot(t, d_true, "k--", label="verdade")
    ax.plot(t, d_est, label="Kalman (1 estado)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("d [m]")
    ax.legend()
    ax.set_title("ultrassom1 — verificação rápida (atraso esperado)")

    out = __file__.replace("ultrassom1.py", "_checks/ultrassom1.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    atraso_medio = float(np.mean(d_est - d_true))
    print(f"\nAtraso médio (viés) da estimativa: {atraso_medio*100:.1f} cm (esperado > 0)")
    print(f"Figura salva em {out}")
