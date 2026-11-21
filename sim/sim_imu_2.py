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
"""Simulador — IMU 2 estados, eixo z."""
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

from kalman.config import DT_LOOP, SIGMA_BIAS, SIGMA_GYRO, SIGMA_MAG
from kalman.core import KalmanFilter
from kalman.filtros import imu2
from kalman.utils import wrap
from sim.base import (
    coverage_verdict,
    innovation_whiteness_verdict_mc,
    compute_nees,
    draw_error_band,
    draw_state_vs_time,
    nees_verdict_mc,
    output_path,
    save_fig,
)


def true_trajectory_stochastic(dt, duration, omega_true=0.3, sigma_g=SIGMA_GYRO,
                                sigma_b=SIGMA_BIAS, seed=999, bias0=0.0):
    """
    Verdade = amostra literal do processo assumido: bias com passeio aleatório
    (mesma variância de Q[1,1]) e giro medido com ruído próprio (mesma
    variância de Q[0,0], via Gamma).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    bias = np.zeros(n)
    psi = np.zeros(n)
    bias[0] = bias0
    for k in range(1, n):
        bias[k] = bias[k - 1] + rng.normal(0, sigma_b * np.sqrt(dt))
        psi[k] = wrap(psi[k - 1] + (omega_true - bias[k - 1]) * dt)
    gyro_meas = omega_true + rng.normal(0, sigma_g, n)
    t = np.arange(n) * dt
    return t, psi, bias, gyro_meas


def _run_one(dt, duration, seed, meas_seed, burn_in_s):
    t, psi_true, bias_true, gyro_meas = true_trajectory_stochastic(dt, duration, seed=seed)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in psi_true])

    F, Gamma, Q = imu2.matrices(dt)
    H, R = imu2.observation()
    x0, P0 = imu2.initial(psi0=mag_meas[0], bias0=0.0)
    kf = KalmanFilter(dim_x=2, angle_indices=[0])
    kf.x, kf.P, kf.F, kf.Gamma, kf.Q = x0, P0, F, Gamma, Q

    x_est = np.zeros((n, 2))
    P_hist = np.zeros((n, 2, 2))
    for k in range(n):
        kf.predict(u=[gyro_meas[k]])
        kf.update_scalar(mag_meas[k], H, R, is_angle=True, channel="yaw")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    x_true = np.column_stack([psi_true, bias_true])
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist, angle_indices=[0])
    err_psi = wrap(x_true[:, 0] - x_est[:, 0])[burn_in_k:]

    return dict(
        nees_mean=float(np.mean(nees_series[burn_in_k:])),
        err_psi=err_psi, var_psi=P_hist[burn_in_k:, 0, 0],
        innov_yaw=np.array(kf.innov_history["yaw"])[burn_in_k:],
    )


def run_formulacao(dt=DT_LOOP, duration=15.0, n_runs=20, burn_in_s=0.5):
    runs = [_run_one(dt, duration, seed=9000 + i, meas_seed=9500 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    ok_nees, msg_nees = nees_verdict_mc([r["nees_mean"] for r in runs], dim_x=2)
    err_all = np.concatenate([r["err_psi"] for r in runs])
    var_all = np.concatenate([r["var_psi"] for r in runs])
    ok_cov, _, msg_cov = coverage_verdict(err_all, var_all, label="psi")
    # A inovacao do yaw ja sai com wrap() aplicado, entao a serie e
    # continua mesmo quando a trajetoria cruza +-pi.
    ok_bra, _, msg_bra = innovation_whiteness_verdict_mc(
        [r["innov_yaw"] for r in runs], label="yaw")

    print(f"=== sim_imu_2 — teste de formulacao ({n_runs} realizacoes independentes) ===")
    print(msg_nees)
    print(msg_bra)
    print(msg_cov)

    return dict(ok=ok_nees and ok_bra and ok_cov)


def run(dt=DT_LOOP, duration=45.0, omega_true=0.3, bias_true_deg=2.0, seed=100):
    """Rotação >360° x2, bias constante conhecido — convergência + wraparound."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    bias_true = np.deg2rad(bias_true_deg)

    psi_true_continuo = omega_true * t  # sem wrap — referencia continua
    psi_true = wrap(psi_true_continuo)

    rng = np.random.default_rng(seed)
    gyro_meas = omega_true + bias_true + rng.normal(0, SIGMA_GYRO, n)
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in psi_true])

    F, Gamma, Q = imu2.matrices(dt)
    H, R = imu2.observation()
    x0, P0 = imu2.initial(psi0=mag_meas[0], bias0=0.0)
    kf = KalmanFilter(dim_x=2, angle_indices=[0])
    kf.x, kf.P, kf.F, kf.Gamma, kf.Q = x0, P0, F, Gamma, Q

    psi_est = np.zeros(n)
    bias_est = np.zeros(n)
    for k in range(n):
        kf.predict(u=[gyro_meas[k]])
        kf.update_scalar(mag_meas[k], H, R, is_angle=True, channel="yaw")
        psi_est[k] = kf.x[0, 0]
        bias_est[k] = kf.x[1, 0]

    # --- convergencia do bias
    erro_bias_final = abs(bias_est[-1] - bias_true)
    convergiu = erro_bias_final < 0.2 * abs(bias_true)

    psi_est_continuo = np.unwrap(psi_est)
    # unwrap pode divergir do offset absoluto por um multiplo de 2pi;
    # realinha pelo primeiro ponto antes de comparar
    psi_est_continuo -= (psi_est_continuo[0] - psi_true_continuo[0])
    erro_continuo = np.abs(psi_est_continuo - psi_true_continuo)
    sem_salto = bool(np.all(erro_continuo < 10 * SIGMA_MAG))  # margem generosa, rad

    n_voltas = (t[-1] * omega_true) / (2 * np.pi)

    print("\n=== sim_imu_2 — rotacao >360 (ilustrativa) ===")
    print(f"Numero de voltas completas na simulacao: {n_voltas:.2f}")
    print(f"Bias verdadeiro: {np.degrees(bias_true):.2f} graus/s  "
          f"estimado ao final: {np.degrees(bias_est[-1]):.3f} graus/s  "
          f"erro: {np.degrees(erro_bias_final):.3f} graus/s")
    print(f"Convergencia do bias (erro final < 20% do valor verdadeiro): {'OK' if convergiu else 'FALHOU'}")
    print(f"Erro maximo (reconstrucao continua via unwrap): {np.degrees(np.max(erro_continuo)):.2f} graus")
    print(f"Sem salto no wraparound: {'OK' if sem_salto else 'FALHOU'}")

    tudo_ok = convergiu and sem_salto

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(9, 11))
    axs[0].plot(t, np.degrees(psi_true), "k--", label="verdade", alpha=0.6)
    axs[0].plot(t, np.degrees(psi_est), label="Kalman", linewidth=0.8)
    axs[0].set_ylabel("yaw [graus], wrapped")
    axs[0].set_title("sim_imu_2 (visao wrapped: descontinuidades sao esperadas aqui)")
    axs[0].legend(fontsize=8)

    axs[1].plot(t, np.degrees(psi_true_continuo), "k--", label="verdade continua", alpha=0.6)
    axs[1].plot(t, np.degrees(psi_est_continuo), label="Kalman (unwrap)", linewidth=0.8)
    axs[1].set_ylabel("yaw [graus], continuo")
    axs[1].set_title("mesma serie, reconstruida — sem saltos = sem degraus aqui")
    axs[1].legend(fontsize=8)

    axs[2].plot(t, np.degrees(erro_continuo))
    axs[2].set_ylabel("erro continuo [graus]")
    axs[2].set_title(f"sem salto: {'OK' if sem_salto else 'FALHOU'}")

    axs[3].axhline(np.degrees(bias_true), color="k", ls="--", label="bias verdadeiro")
    axs[3].plot(t, np.degrees(bias_est), label="bias estimado")
    axs[3].set_ylabel("bias [graus/s]")
    axs[3].set_xlabel("t [s]")
    axs[3].set_title(f"convergencia: {'OK' if convergiu else 'FALHOU'}")
    axs[3].legend(fontsize=8)

    save_fig(fig, output_path("imu2.png"))

    return dict(ok=tudo_ok)


if __name__ == "__main__":
    r_formulacao = run_formulacao()
    r_rotacao = run()
    tudo_ok = r_formulacao["ok"] and r_rotacao["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
