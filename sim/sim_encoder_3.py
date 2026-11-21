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
"""Simulador — Encoder 3 estados."""
# --- bootstrap de path
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

from kalman.config import DT_LOOP, SIGMA_ENC, SIGMA_JERK_ENC3
from kalman.core import KalmanFilter
from kalman.filtros import encoder3
from sim.base import (
    coverage_verdict,
    compute_nees,
    draw_error_band,
    draw_nis,
    draw_state_vs_time,
    nees_verdict_mc,
    nis_verdict,
    innovation_whiteness_verdict_mc,
    output_path,
    save_fig,
)
from sim.sim_encoder_2 import true_trajectory


def true_trajectory_stochastic(dt, duration, sigma_jerk=SIGMA_JERK_ENC3, seed=999, v0=1.0):
    """Verdade = amostra do processo assumido pelo filtro: jerk i.i.d. ~ N(0, sigma_jerk^2)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    j = rng.normal(0.0, sigma_jerk, n)
    p, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    v[0] = v0
    for k in range(1, n):
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a[k - 1] * dt**2 + j[k - 1] * dt**3 / 6.0
        v[k] = v[k - 1] + a[k - 1] * dt + 0.5 * j[k - 1] * dt**2
        a[k] = a[k - 1] + j[k - 1] * dt
    t = np.arange(n) * dt
    return t, p, v, a


def _run_one(dt, duration, sigma_jerk, seed, meas_seed, burn_in_s):
    t, p_true, v_true, a_true = true_trajectory_stochastic(dt, duration, sigma_jerk, seed)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    raw_v = v_true + rng.normal(0, SIGMA_ENC, n)

    F, H, Q, R = encoder3.matrices(dt, sigma_jerk=sigma_jerk)
    x0, P0 = encoder3.initial()
    kf = KalmanFilter(dim_x=3)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 3))
    P_hist = np.zeros((n, 3, 3))
    nis_v = np.zeros(n)
    for k in range(n):
        kf.predict()
        _, nis = kf.update_scalar(raw_v[k], H, R, channel="v")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P
        nis_v[k] = nis

    x_true = np.column_stack([p_true, v_true, a_true])
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist)
    err = x_true - x_est

    return dict(
        nees_mean=float(np.mean(nees_series[burn_in_k:])),
        nis_mean=float(np.mean(nis_v[burn_in_k:])),
        innov_v=np.array(kf.innov_history["v"])[burn_in_k:],
        err_a=err[burn_in_k:, 2], var_a=P_hist[burn_in_k:, 2, 2],
    )


def run_formulacao(dt=DT_LOOP, duration=20.0, sigma_jerk=SIGMA_JERK_ENC3, n_runs=20, burn_in_s=1.0):
    runs = [_run_one(dt, duration, sigma_jerk, seed=3000 + i, meas_seed=4000 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    ok_nees, msg_nees = nees_verdict_mc([r["nees_mean"] for r in runs], dim_x=3)
    ok_nis, msg_nis = nis_verdict(float(np.mean([r["nis_mean"] for r in runs])), label="v")
    ok_bra, _, msg_bra = innovation_whiteness_verdict_mc(
        [r["innov_v"] for r in runs], label="v")

    err_a_all = np.concatenate([r["err_a"] for r in runs])
    var_a_all = np.concatenate([r["var_a"] for r in runs])
    ok_cov_a, _, msg_cov_a = coverage_verdict(err_a_all, var_a_all, label="a")

    print(f"=== sim_encoder_3 — teste de formulacao ({n_runs} realizacoes independentes) ===")
    print(msg_nees)
    print(msg_nis)
    print(msg_bra)
    print(msg_cov_a)

    return dict(ok=ok_nees and ok_nis and ok_bra and ok_cov_a)


def run(dt_sensor=DT_LOOP, seed=100):
    """Trajetória de manobra (mesma do encoder2) — figura ilustrativa."""
    dt_fine = 1e-3
    t_fine, p_fine, v_fine, a_fine = true_trajectory(dt_fine)

    step = int(round(dt_sensor / dt_fine))
    idx = np.arange(0, len(t_fine), step)
    t = t_fine[idx]
    p_true, v_true, a_true = p_fine[idx], v_fine[idx], a_fine[idx]
    n = len(t)

    rng = np.random.default_rng(seed)
    raw_v = v_true + rng.normal(0, SIGMA_ENC, n)

    F, H, Q, R = encoder3.matrices(dt_sensor)
    x0, P0 = encoder3.initial()
    kf = KalmanFilter(dim_x=3)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 3))
    P_hist = np.zeros((n, 3, 3))
    nis_v = np.zeros(n)
    for k in range(n):
        kf.predict()
        _, nis = kf.update_scalar(raw_v[k], H, R, channel="v")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P
        nis_v[k] = nis

    erro_final_a = float(abs(x_est[-1, 2] - a_true[-1]))
    print(f"\n=== sim_encoder_3 — trajetoria de manobra (ilustrativa) ===")
    print(f"Erro final na aceleracao inferida: {erro_final_a:.4f} m/s^2")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
    axs[0].set_title("sim_encoder_3 (faixas sombreadas = manobra)")
    draw_state_vs_time(axs[0], t, v_true, x_est[:, 1], raw_t=t, raw_vals=raw_v, ylabel="v [m/s]")
    draw_state_vs_time(axs[1], t, a_true, x_est[:, 2], ylabel="a [m/s^2]")
    draw_error_band(axs[2], t, a_true, x_est[:, 2], P_hist[:, 2, 2], ylabel="erro a [m/s^2]")
    draw_nis(axs[3], t, nis_v, label="v")
    axs[4].plot(t, P_hist[:, 0, 0])
    axs[4].set_ylabel("P[0,0] (posicao)")
    axs[4].set_xlabel("t [s]")

    for ax in axs:
        ylim = ax.get_ylim()
        ax.fill_between(t, ylim[0], ylim[1], where=(a_true != 0.0),
                         color="orange", alpha=0.08, step="mid")
        ax.set_ylim(ylim)

    save_fig(fig, output_path("encoder3.png"))

    return dict(ok=True)  # ilustrativa, sem gate proprio (gate e o run_formulacao)


if __name__ == "__main__":
    r_formulacao = run_formulacao()
    r_manobra = run()
    tudo_ok = r_formulacao["ok"] and r_manobra["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
