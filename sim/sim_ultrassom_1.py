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
"""Simulador do filtro do ultrassom com 1 estado."""
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

from kalman.config import DT_LOOP, SIGMA_US, SIGMA_US_WALK
from kalman.core import KalmanFilter
from kalman.filtros import ultrassom1
from sim.base import (
    innovation_whiteness_verdict_mc,
    coverage_verdict,
    compute_nees,
    draw_state_vs_time,
    nees_verdict_mc,
    output_path,
    save_fig,
)


def true_trajectory(dt_fine=1e-3, d0=1.9, v_approach=0.5, duration=3.0):
    """
    d(t) = d0 - v_approach*t, aproximação a velocidade constante (ilustrativa).
    """
    n = int(round(duration / dt_fine)) + 1
    t = np.linspace(0.0, duration, n)
    d = d0 - v_approach * t
    return t, d


def true_trajectory_stochastic(dt, duration, sigma_walk=SIGMA_US_WALK, seed=999, d0=2.0):
    """Verdade = amostra literal do processo assumido: passeio aleatório."""
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    w = rng.normal(0.0, sigma_walk, n)
    d = np.empty(n)
    d[0] = d0
    for k in range(1, n):
        d[k] = d[k - 1] + w[k - 1]
    t = np.arange(n) * dt
    return t, d


def _run_one(dt, duration, sigma_walk, seed, meas_seed, burn_in_s):
    t, d_true = true_trajectory_stochastic(dt, duration, sigma_walk, seed)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    raw = d_true + rng.normal(0, SIGMA_US, n)

    F, H, Q, R = ultrassom1.matrices(dt, sigma_walk=sigma_walk)
    x0, P0 = ultrassom1.initial(raw[0])
    kf = KalmanFilter(dim_x=1)
    kf.x, kf.P, kf.F, kf.Q = x0, P0, F, Q

    x_est = np.zeros((n, 1))
    P_hist = np.zeros((n, 1, 1))
    x_est[0] = kf.x[:, 0]
    P_hist[0] = kf.P
    for k in range(1, n):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    x_true = d_true.reshape(-1, 1)
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist)
    err = (d_true - x_est[:, 0])[burn_in_k:]
    var = P_hist[burn_in_k:, 0, 0]

    return dict(nees_mean=float(np.mean(nees_series[burn_in_k:])), err=err, var=var,
                innov_d=np.array(kf.innov_history["d"])[burn_in_k:])


def run_formulacao(dt=DT_LOOP, duration=20.0, sigma_walk=SIGMA_US_WALK, n_runs=20, burn_in_s=0.5):
    runs = [_run_one(dt, duration, sigma_walk, seed=5000 + i, meas_seed=6000 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    ok_nees, msg_nees = nees_verdict_mc([r["nees_mean"] for r in runs], dim_x=1)
    err_all = np.concatenate([r["err"] for r in runs])
    var_all = np.concatenate([r["var"] for r in runs])
    ok_cov, _, msg_cov = coverage_verdict(err_all, var_all, label="d")
    # Aqui a brancura E criterio: esta trajetoria e o proprio passeio aleatorio
    # que o modelo supoe, entao a hipotese vale e a inovacao tem que ser
    # branca.
    ok_bra, _, msg_bra = innovation_whiteness_verdict_mc(
        [r["innov_d"] for r in runs], label="d")

    print(f"=== sim_ultrassom_1 — teste de formulacao ({n_runs} realizacoes independentes) ===")
    print(msg_nees)
    print(msg_bra)
    print(msg_cov)

    return dict(ok=ok_nees and ok_bra and ok_cov)


def run(dt_sensor=DT_LOOP, seed=200, plot=True):
    """Aproximação a velocidade constante — demonstra o atraso sistemático."""
    dt_fine = 1e-3
    t_fine, d_fine = true_trajectory(dt_fine)

    step = int(round(dt_sensor / dt_fine))
    idx = np.arange(0, len(t_fine), step)
    t = t_fine[idx]
    d_true = d_fine[idx]
    n = len(t)

    rng = np.random.default_rng(seed)
    raw = d_true + rng.normal(0, SIGMA_US, n)

    F, H, Q, R = ultrassom1.matrices(dt_sensor)
    x0, P0 = ultrassom1.initial(raw[0])
    kf = KalmanFilter(dim_x=1)
    kf.x, kf.P = x0, P0
    kf.F, kf.Q = F, Q

    x_est = np.zeros((n, 1))
    P_hist = np.zeros((n, 1, 1))
    x_est[0] = kf.x[:, 0]
    P_hist[0] = kf.P
    for k in range(1, n):
        kf.predict()
        kf.update_scalar(raw[k], H, R, channel="d")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    err = d_true - x_est[:, 0]

    # O MESMO filtro, alimentado com a verdade SEM ruido.
    kf_lim = KalmanFilter(dim_x=1)
    kf_lim.x, kf_lim.P = ultrassom1.initial(d_true[0])
    kf_lim.F, kf_lim.Q = F, Q
    x_lim = np.zeros(n)
    x_lim[0] = float(kf_lim.x[0, 0])
    for k in range(1, n):
        kf_lim.predict()
        kf_lim.update_scalar(d_true[k], H, R, channel="d")
        x_lim[k] = float(kf_lim.x[0, 0])
    err_lim = d_true - x_lim

    # regime permanente: exclui o transitório inicial (~10% do trajeto)
    steady_from = int(0.1 * n)
    bias_mean = float(np.mean(err[steady_from:]))
    bias_estrutural = float(np.mean(err_lim[steady_from:]))
    print("\n=== sim_ultrassom_1 — aproximacao (ilustrativa, viés esperado) ===")
    print(f"Vies medio em regime permanente: {bias_mean*100:+.1f} cm "
          f"(NEES/cobertura nao se aplicam aqui — a trajetoria viola de proposito "
          f"a hipotese de distancia constante do modelo)")
    print(f"Vies ESTRUTURAL (mesmo filtro, medida sem ruido): "
          f"{bias_estrutural*100:+.3f} cm")
    print("  -> e o atraso do modelo, nao dispersao. Depende de Q/R: com o "
          "SIGMA_US de bancada (2 mm) valia +0,004 cm e sumia no ruido.")

    # BRANCURA -- informativa, e aqui ela e o RESULTADO, nao o criterio.
    steady = slice(steady_from, None)
    _, r1_ruido, msg_bra = innovation_whiteness_verdict_mc(
        [np.array(kf.innov_history["d"])[steady]], label="d, com ruido",
        criterio=False)
    _, r1_lim, msg_bra_lim = innovation_whiteness_verdict_mc(
        [np.array(kf_lim.innov_history["d"])[steady]], label="d, sem ruido",
        criterio=False)
    print(msg_bra)
    print(msg_bra_lim)
    print(f"  -> a inovacao correlacionada e a MESMA constatacao do vies, no "
          f"tempo em vez de em metros. Reprovar seria acusar o simulador de "
          f"fazer o que lhe foi pedido.")

    result = dict(ok=True, bias_m=bias_mean, bias_estrutural_m=bias_estrutural,
                  r1_inovacao=r1_ruido, r1_inovacao_estrutural=r1_lim,
                  t=t, d_true=d_true, raw=raw, x_est=x_est, err=err,
                  err_estrutural=err_lim, P_hist=P_hist)

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axs[0].set_title("sim_ultrassom_1")
        draw_state_vs_time(axs[0], t, d_true, x_est[:, 0], raw_t=t, raw_vals=raw, ylabel="d [m]")
        axs[1].plot(t, err * 100, color="0.72", lw=0.9,
                    label="com ruido de medicao")
        axs[1].plot(t, err_lim * 100, color="C3", lw=1.8,
                    label=f"atraso estrutural = {bias_estrutural*100:+.2f} cm")
        axs[1].axhline(0.0, color="0.4", lw=0.7)
        axs[1].set_ylabel("verdade - estimado [cm]")
        axs[1].legend(fontsize=8)
        axs[1].set_title("Atraso sistematico: a curva escura e o fenomeno, "
                         "a clara e o ruido por cima dele")
        axs[2].plot(t, P_hist[:, 0, 0])
        axs[2].set_ylabel("P[0,0]")
        axs[2].set_xlabel("t [s]")

        save_fig(fig, output_path("ultrassom1.png"))

    return result


if __name__ == "__main__":
    r_formulacao = run_formulacao()
    r_aprox = run()
    tudo_ok = r_formulacao["ok"] and r_aprox["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
