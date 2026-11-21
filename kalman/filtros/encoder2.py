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
"""Filtro do encoder com 2 estados."""
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

from kalman.config import SIGMA_ACC, SIGMA_ENC


def matrices(dt, sigma_acc=SIGMA_ACC, sigma_enc=SIGMA_ENC):
    """
    F = [[1, dt], [0, 1]]           G = [[dt^2/2], [dt]]
    H = [[0, 1]]                    Q = G * sigma_acc^2 * G^T
    R = sigma_enc^2                 w = aceleração desconhecida do veículo [m/s^2]

    Sem Gamma: este filtro não tem entrada de controle.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[dt**2 / 2.0], [dt]])
    H = np.array([[0.0, 1.0]])
    Q = (sigma_acc**2) * (G @ G.T)
    R = sigma_enc**2
    return F, H, Q, R


def initial(sigma_enc=SIGMA_ENC):
    """x0 = [0, 0]^T: veículo parado na origem (origem definida, não medida)."""
    x0 = np.array([[0.0], [0.0]])
    R = sigma_enc**2
    P0 = np.array([[0.0, 0.0], [0.0, R]])
    return x0, P0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kalman.core import KalmanFilter

    dt_show = 0.02
    F, H, Q, R = matrices(dt_show)
    print("=== kalman.filtros.encoder2 — matrizes em dt =", dt_show, "===\n")
    print("F =\n", F)
    print("\nH =\n", H)
    print("\nQ =\n", Q)
    print("\nR =", R)

    # Demonstração: velocidade constante ruidosa, e P[0,0] (posição)
    # crescendo sem parar.
    rng = np.random.default_rng(0)
    n_steps = 250
    v_true = 1.2  # [m/s]
    sigma_enc = SIGMA_ENC

    F, H, Q, R = matrices(dt_show)
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P = initial()
    kf.F, kf.Q = F, Q

    t = np.arange(n_steps) * dt_show
    raw = v_true + rng.normal(0, sigma_enc, n_steps)
    v_est = np.zeros(n_steps)
    p00 = np.zeros(n_steps)

    for k in range(n_steps):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="v")
        v_est[k] = kf.x[1, 0]
        p00[k] = kf.P[0, 0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, raw, ".", alpha=0.3, label="bruto")
    axs[0].axhline(v_true, color="k", ls="--", label="verdade")
    axs[0].plot(t, v_est, label="Kalman")
    axs[0].set_ylabel("v [m/s]")
    axs[0].legend()
    axs[0].set_title("encoder2 — verificação rápida")

    axs[1].plot(t, p00)
    axs[1].set_ylabel("P[0,0] (posição)")
    axs[1].set_xlabel("t [s]")
    crescente = bool(np.all(np.diff(p00) >= -1e-12))
    axs[1].set_title(f"P[0,0] monotonicamente crescente: {'OK' if crescente else 'FALHOU'}")

    out = __file__.replace("encoder2.py", "_checks/encoder2.png")
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nP[0,0] monotonicamente crescente: {'OK' if crescente else 'FALHOU'}")
    print(f"Figura salva em {out}")
