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
"""Simulador — Ultrassom 2 estados."""
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

from kalman.config import DT_LOOP, SIGMA_US, SIGMA_US_ACCEL
from kalman.core import KalmanFilter
from kalman.filtros import ultrassom1, ultrassom2
from sim.base import (
    coverage_verdict,
    innovation_whiteness_verdict_mc,
    compute_nees,
    draw_state_vs_time,
    nees_verdict_mc,
    output_path,
    save_fig,
)
from sim.sim_ultrassom_1 import run as run_ultrassom_1
from sim.sim_ultrassom_1 import true_trajectory


def true_trajectory_stochastic(dt, duration, sigma_alpha=SIGMA_US_ACCEL, seed=999, d0=2.0, ddot0=0.0):
    """Verdade = amostra do processo assumido: aceleração relativa branca."""
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    a = rng.normal(0.0, sigma_alpha, n)
    d, ddot = np.zeros(n), np.zeros(n)
    d[0], ddot[0] = d0, ddot0
    for k in range(1, n):
        d[k] = d[k - 1] + ddot[k - 1] * dt + 0.5 * a[k - 1] * dt**2
        ddot[k] = ddot[k - 1] + a[k - 1] * dt
    t = np.arange(n) * dt
    return t, d, ddot


def _run_one_formulacao(dt, duration, sigma_alpha, seed, meas_seed, burn_in_s):
    t, d_true, ddot_true = true_trajectory_stochastic(dt, duration, sigma_alpha, seed)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    raw = d_true + rng.normal(0, SIGMA_US, n)

    F, H, Q, R = ultrassom2.matrices(dt, sigma_alpha=sigma_alpha)
    x0, P0 = ultrassom2.initial(raw[0])
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    x_est[0], P_hist[0] = kf.x[:, 0], kf.P
    for k in range(1, n):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    x_true = np.column_stack([d_true, ddot_true])
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist)
    err = (x_true - x_est)[burn_in_k:]

    return dict(
        nees_mean=float(np.mean(nees_series[burn_in_k:])),
        err_d=err[:, 0], var_d=P_hist[burn_in_k:, 0, 0],
        innov_d=np.array(kf.innov_history["d"])[burn_in_k:],
    )


def run_formulacao(dt=DT_LOOP, duration=20.0, sigma_alpha=SIGMA_US_ACCEL, n_runs=20, burn_in_s=0.5):
    runs = [_run_one_formulacao(dt, duration, sigma_alpha, seed=7000 + i, meas_seed=8000 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    ok_nees, msg_nees = nees_verdict_mc([r["nees_mean"] for r in runs], dim_x=2)
    err_all = np.concatenate([r["err_d"] for r in runs])
    var_all = np.concatenate([r["var_d"] for r in runs])
    ok_cov, _, msg_cov = coverage_verdict(err_all, var_all, label="d")
    ok_bra, _, msg_bra = innovation_whiteness_verdict_mc(
        [r["innov_d"] for r in runs], label="d")

    print(f"=== sim_ultrassom_2 — teste de formulacao ({n_runs} realizacoes independentes) ===")
    print(msg_nees)
    print(msg_bra)
    print(msg_cov)

    return dict(ok=ok_nees and ok_bra and ok_cov)


def _bias_estrutural(dt_sensor, steady_from_s=1.0):
    """
    Viés SEM ruído de medição — isola o defeito estrutural do modelo do ruído
    estatístico de uma única realização.
    """
    t, d_true = true_trajectory(dt_fine=1e-3)
    step = int(round(dt_sensor / 1e-3))
    idx = np.arange(0, len(t), step)
    t, d_true = t[idx], d_true[idx]
    raw = d_true.copy()
    k0 = int(round(steady_from_s / dt_sensor))

    F, H, Q, R = ultrassom1.matrices(dt_sensor)
    x0, P0 = ultrassom1.initial(raw[0])
    kf = KalmanFilter(dim_x=1)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q
    x1 = np.zeros(len(t))
    x1[0] = kf.x[0, 0]
    for k in range(1, len(t)):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x1[k] = kf.x[0, 0]

    F, H, Q, R = ultrassom2.matrices(dt_sensor)
    x0, P0 = ultrassom2.initial(raw[0])
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q
    x2 = np.zeros(len(t))
    x2[0] = kf.x[0, 0]
    for k in range(1, len(t)):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x2[k] = kf.x[0, 0]

    bias_1 = float(np.mean((d_true - x1)[k0:]))
    bias_2 = float(np.mean((d_true - x2)[k0:]))
    return bias_1, bias_2


def run_comparacao(dt_sensor=DT_LOOP, seed=200):
    us1 = run_ultrassom_1(dt_sensor=dt_sensor, seed=seed, plot=False)
    t, d_true, raw = us1["t"], us1["d_true"], us1["raw"]
    n = len(t)

    F, H, Q, R = ultrassom2.matrices(dt_sensor)
    x0, P0 = ultrassom2.initial(raw[0])
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    x_est[0] = kf.x[:, 0]
    P_hist[0] = kf.P
    for k in range(1, n):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    err_d = d_true - x_est[:, 0]
    burn_in_k = 10
    steady_from = int(0.1 * n)
    bias_2_ruidoso = float(np.mean(err_d[steady_from:]))
    bias_1_ruidoso = us1["bias_m"]

    bias_1_estrutural, bias_2_estrutural = _bias_estrutural(dt_sensor)
    ok_bias = abs(bias_1_estrutural) >= 10 * abs(bias_2_estrutural)

    print("=== sim_ultrassom_2 — cenario A: comparacao de vies ===")
    print(f"Vies (com ruido, 1 realizacao) — 1 estado: {bias_1_ruidoso*100:+.2f} cm  "
          f"2 estados: {bias_2_ruidoso*100:+.2f} cm  [informativo, nao gate]")
    print(f"Vies ESTRUTURAL (sem ruido) — 1 estado: {bias_1_estrutural*100:+.2f} cm  "
          f"2 estados: {bias_2_estrutural*100:+.2f} cm")
    print(f"Vies estrutural 1-estado >= 10x o de 2-estados: {'OK' if ok_bias else 'FALHOU'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axs[0].set_title("sim_ultrassom_2 — cenario A: mesma trajetoria do ultrassom_1")
    draw_state_vs_time(axs[0], t, d_true, x_est[:, 0], raw_t=t, raw_vals=raw, ylabel="d [m]")
    axs[1].plot(t, err_d * 100, label="2 estados")
    axs[1].plot(t, us1["err"] * 100, alpha=0.5, label="1 estado (referencia)")
    axs[1].axhline(bias_2_estrutural * 100, color="r", ls="--",
                    label=f"vies estrutural 2-estados = {bias_2_estrutural*100:+.2f} cm")
    axs[1].set_ylabel("estimado - verdade [cm]")
    axs[1].legend(fontsize=8)
    axs[2].plot(t, x_est[:, 1], label="Kalman (ddot)")
    # A verdade de ddot é a derivada da trajetória — NÃO um literal.
    ddot_verdade = float(np.gradient(d_true, t)[len(t) // 2])
    axs[2].axhline(ddot_verdade, color="k", ls="--",
                    label=f"verdade (ddot = -v_aprox = {ddot_verdade:+.2f} m/s)")
    axs[2].set_ylabel("ddot [m/s]")
    axs[2].set_xlabel("t [s]")
    axs[2].legend(fontsize=8)

    save_fig(fig, output_path("ultrassom2_comparacao.png"))

    return dict(ok=ok_bias,
                bias_1_estrutural_cm=bias_1_estrutural * 100,
                bias_2_estrutural_cm=bias_2_estrutural * 100)


def run_perda_de_alvo(dt_sensor=DT_LOOP, seed=201, gap=(1.2, 2.2)):
    """Suprime medições em [gap[0], gap[1])s."""
    dt_fine = 1e-3
    t_fine, d_fine = true_trajectory(dt_fine)
    step = int(round(dt_sensor / dt_fine))
    idx = np.arange(0, len(t_fine), step)
    t = t_fine[idx]
    d_true = d_fine[idx]
    n = len(t)

    rng = np.random.default_rng(seed)
    raw = d_true + rng.normal(0, SIGMA_US, n)
    tem_medida = ~((t >= gap[0]) & (t < gap[1]))

    F, H, Q, R = ultrassom2.matrices(dt_sensor)
    x0, P0 = ultrassom2.initial(raw[0])
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    x_est[0], P_hist[0] = kf.x[:, 0], kf.P
    for k in range(1, n):
        kf.predict()
        if tem_medida[k]:
            kf.update_scalar(raw[k], H, R, channel="d")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    gap_idx = np.where(~tem_medida)[0]
    k_start, k_end = gap_idx[0], gap_idx[-1]
    p00_antes = P_hist[k_start - 1, 0, 0]
    p00_fim_gap = P_hist[k_end, 0, 0]
    cresceu = p00_fim_gap > p00_antes

    erro_durante = float(np.max(np.abs(d_true[k_start:k_end + 1] - x_est[k_start:k_end + 1, 0])))
    nao_divergiu = erro_durante < 1.0  # [m] margem generosa para 1s de predicao

    k_reacq = min(k_end + 10, n - 1)  # ~0.5s apos o retorno da medida
    erro_reacq = float(abs(d_true[k_reacq] - x_est[k_reacq, 0]))
    reaquiriu = erro_reacq < 0.10  # [m]

    print("\n=== sim_ultrassom_2 — cenario B: perda de alvo ===")
    print(f"P[0,0] antes do gap: {p00_antes:.5f}  ao fim do gap: {p00_fim_gap:.5f}  "
          f"cresceu: {'OK' if cresceu else 'FALHOU'}")
    print(f"Erro maximo durante o gap: {erro_durante*100:.1f} cm  "
          f"nao divergiu: {'OK' if nao_divergiu else 'FALHOU'}")
    print(f"Erro ~0.5s apos reaquisicao: {erro_reacq*100:.1f} cm  "
          f"reaquiriu: {'OK' if reaquiriu else 'FALHOU'}")

    ok = cresceu and nao_divergiu and reaquiriu

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].set_title("sim_ultrassom_2 — cenario B: perda de alvo (faixa sombreada = sem medida)")
    draw_state_vs_time(axs[0], t, d_true, x_est[:, 0],
                        raw_t=t[tem_medida], raw_vals=raw[tem_medida], ylabel="d [m]")
    axs[1].plot(t, P_hist[:, 0, 0])
    axs[1].set_ylabel("P[0,0]")
    axs[1].set_xlabel("t [s]")
    for ax in axs:
        ylim = ax.get_ylim()
        ax.fill_between(t, ylim[0], ylim[1], where=~tem_medida, color="orange", alpha=0.15, step="mid")
        ax.set_ylim(ylim)

    save_fig(fig, output_path("ultrassom2_perda_de_alvo.png"))

    return dict(ok=ok)


if __name__ == "__main__":
    r0 = run_formulacao()
    r1 = run_comparacao()
    r2 = run_perda_de_alvo()
    tudo_ok = r0["ok"] and r1["ok"] and r2["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
