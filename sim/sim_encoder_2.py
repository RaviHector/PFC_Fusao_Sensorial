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
"""Simulador do filtro do encoder com 2 estados."""
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

from kalman.config import DT_LOOP, SIGMA_ACC, SIGMA_ENC
from kalman.core import KalmanFilter
from kalman.filtros import encoder2
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


def true_trajectory(dt_fine=1e-3, ramp=0.3):
    """
    Trechos: constante -> acelera -> constante -> freia -> constante, com
    transições de aceleração RAMPADAS (jerk finito) em vez de degraus
    instantâneos.
    """
    v0 = 1.0
    plateaus = [(5.0, 0.0), (3.0, 0.5), (5.0, 0.0), (3.0, -0.8), (4.0, 0.0)]

    keyframe_t, keyframe_a = [0.0], [plateaus[0][1]]
    t_cursor = 0.0
    for i, (dur, a) in enumerate(plateaus):
        t_cursor += dur
        if i < len(plateaus) - 1:
            next_a = plateaus[i + 1][1]
            keyframe_t += [t_cursor - ramp, t_cursor]
            keyframe_a += [a, next_a]
        else:
            keyframe_t.append(t_cursor)
            keyframe_a.append(a)

    n = int(round(t_cursor / dt_fine)) + 1
    t = np.linspace(0.0, t_cursor, n)
    a = np.interp(t, keyframe_t, keyframe_a)
    v = v0 + np.concatenate([[0.0], np.cumsum((a[:-1] + a[1:]) / 2.0 * dt_fine)])
    p = np.concatenate([[0.0], np.cumsum((v[:-1] + v[1:]) / 2.0 * dt_fine)])
    return t, p, v, a


def true_trajectory_stochastic(dt, duration, sigma_acc=SIGMA_ACC, seed=999, v0=1.0):
    """
    Verdade = amostra literal do processo que o filtro assume: a cada dt, a
    aceleração é um sorteio i.i.d.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    a = rng.normal(0.0, sigma_acc, n)
    p, v = np.zeros(n), np.zeros(n)
    v[0] = v0
    for k in range(1, n):
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a[k - 1] * dt**2
        v[k] = v[k - 1] + a[k - 1] * dt
    t = np.arange(n) * dt
    return t, p, v, a


def _run_one(dt, duration, sigma_acc, seed, meas_seed, burn_in_s):
    """Uma realização do teste de formulação."""
    x0_ref, P0_ref = encoder2.initial()
    rng_ini = np.random.default_rng(seed + 500000)
    v0_amostrado = float(rng_ini.normal(x0_ref[1, 0], np.sqrt(P0_ref[1, 1])))

    t, p_true, v_true, _ = true_trajectory_stochastic(
        dt, duration, sigma_acc, seed, v0=v0_amostrado)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    raw_v = v_true + rng.normal(0, SIGMA_ENC, n)

    F, H, Q, R = encoder2.matrices(dt, sigma_acc=sigma_acc)
    x0, P0 = encoder2.initial()
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    nis_v = np.zeros(n)
    for k in range(n):
        kf.predict()
        _, nis = kf.update_scalar(raw_v[k], H, R, channel="v")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P
        nis_v[k] = nis

    x_true = np.column_stack([p_true, v_true])
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist)

    err = x_true - x_est
    return dict(
        nees_mean=float(np.mean(nees_series[burn_in_k:])),
        nis_mean=float(np.mean(nis_v[burn_in_k:])),
        innov_v=np.array(kf.innov_history["v"])[burn_in_k:],
        err_p=err[burn_in_k:, 0], var_p=P_hist[burn_in_k:, 0, 0],
        err_v=err[burn_in_k:, 1], var_v=P_hist[burn_in_k:, 1, 1],
    )


def run_formulacao(dt=DT_LOOP, duration=20.0, sigma_acc=SIGMA_ACC, n_runs=20, burn_in_s=1.0):
    """Teste de consistência limpo — ver docstring do módulo."""
    runs = [_run_one(dt, duration, sigma_acc, seed=1000 + i, meas_seed=2000 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    nees_means = [r["nees_mean"] for r in runs]
    ok_nees, msg_nees = nees_verdict_mc(nees_means, dim_x=2)
    ok_nis, msg_nis = nis_verdict(float(np.mean([r["nis_mean"] for r in runs])), label="v")
    ok_bra, _, msg_bra = innovation_whiteness_verdict_mc(
        [r["innov_v"] for r in runs], label="v")

    err_p_all = np.concatenate([r["err_p"] for r in runs])
    var_p_all = np.concatenate([r["var_p"] for r in runs])
    err_v_all = np.concatenate([r["err_v"] for r in runs])
    var_v_all = np.concatenate([r["var_v"] for r in runs])
    ok_cov_p, _, msg_cov_p = coverage_verdict(err_p_all, var_p_all, label="p")
    ok_cov_v, _, msg_cov_v = coverage_verdict(err_v_all, var_v_all, label="v")

    print(f"=== sim_encoder_2 — teste de formulacao ({n_runs} realizacoes independentes) ===")
    print(msg_nees)
    print(msg_nis)
    print(msg_bra)
    print(msg_cov_p)
    print(msg_cov_v)

    return dict(ok=ok_nees and ok_nis and ok_bra and ok_cov_p and ok_cov_v,
                nees_means=nees_means)


def run(dt_sensor=DT_LOOP, seed=100, burn_in_s=1.0):
    """Trajetória de manobra — figura ilustrativa + teste de não observabilidade."""
    dt_fine = 1e-3
    t_fine, p_fine, v_fine, a_fine = true_trajectory(dt_fine)

    step = int(round(dt_sensor / dt_fine))
    idx = np.arange(0, len(t_fine), step)
    t = t_fine[idx]
    p_true = p_fine[idx]
    v_true = v_fine[idx]
    a_true = a_fine[idx]
    n = len(t)

    rng = np.random.default_rng(seed)
    raw_v = v_true + rng.normal(0, SIGMA_ENC, n)

    F, H, Q, R = encoder2.matrices(dt_sensor)
    x0, P0 = encoder2.initial()
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    nis_v = np.zeros(n)
    for k in range(n):
        kf.predict()
        _, nis = kf.update_scalar(raw_v[k], H, R, channel="v")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P
        nis_v[k] = nis

    x_true = np.column_stack([p_true, v_true])
    burn_in_k = int(round(burn_in_s / dt_sensor))
    post = slice(burn_in_k, None)
    steady = (a_true == 0.0)[post]
    maneuver = ~steady

    nees_series, _ = compute_nees(x_true, x_est, P_hist)
    nees_steady = float(np.mean(nees_series[post][steady]))
    nees_maneuver = float(np.mean(nees_series[post][maneuver]))

    err_v = x_true[:, 1] - x_est[:, 1]
    _, frac_cov_v_steady, _ = coverage_verdict(err_v[post][steady], P_hist[post, 1, 1][steady], label="v, constante")
    _, frac_cov_v_man, _ = coverage_verdict(err_v[post][maneuver], P_hist[post, 1, 1][maneuver], label="v, manobra")

    p00 = P_hist[:, 0, 0]
    monotonic = bool(np.all(np.diff(p00) >= -1e-9))

    print("\n=== sim_encoder_2 — trajetoria de manobra (ilustrativa) ===")
    print(f"P[0,0] estritamente nao decrescente: {'OK' if monotonic else 'FALHOU'} (este e o unico gate aqui)")
    print(f"[informativo] NEES medio — constante: {nees_steady:.2f}  manobra: {nees_maneuver:.2f}")
    print(f"[informativo] cobertura +-3sigma v — constante: {frac_cov_v_steady*100:.1f}%  "
          f"manobra: {frac_cov_v_man*100:.1f}% (atraso de rastreio esperado, nao e defeito)")

    tudo_ok = monotonic

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(6, 1, figsize=(9, 16), sharex=True)
    axs[0].set_title("sim_encoder_2 (faixas sombreadas = manobra, ilustrativo)")
    draw_state_vs_time(axs[0], t, p_true, x_est[:, 0], ylabel="p [m]")
    draw_state_vs_time(axs[1], t, v_true, x_est[:, 1], raw_t=t, raw_vals=raw_v, ylabel="v [m/s]")
    draw_error_band(axs[2], t, p_true, x_est[:, 0], P_hist[:, 0, 0], ylabel="erro p [m]")
    draw_error_band(axs[3], t, v_true, x_est[:, 1], P_hist[:, 1, 1], ylabel="erro v [m/s]")
    draw_nis(axs[4], t, nis_v, label="v")
    axs[5].plot(t, p00)
    axs[5].set_ylabel("P[0,0]")
    axs[5].set_xlabel("t [s]")
    axs[5].set_title(f"P[0,0] monotonicamente crescente: {'OK' if monotonic else 'FALHOU'}")

    for ax in axs:
        ylim = ax.get_ylim()
        ax.fill_between(t, ylim[0], ylim[1], where=(a_true != 0.0),
                         color="orange", alpha=0.08, step="mid")
        ax.set_ylim(ylim)

    save_fig(fig, output_path("encoder2.png"))

    return dict(ok=tudo_ok, nees_steady=nees_steady, nees_maneuver=nees_maneuver)


if __name__ == "__main__":
    r_formulacao = run_formulacao()
    r_manobra = run()
    tudo_ok = r_formulacao["ok"] and r_manobra["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
