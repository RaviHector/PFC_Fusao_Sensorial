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
"""Simulador — IMU 6 estados."""
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

from kalman.config import DT_LOOP, SIGMA_ACC_TILT, SIGMA_BIAS, SIGMA_GYRO, SIGMA_MAG
from kalman.core import KalmanFilter
from kalman.filtros import imu6
from kalman.utils import wrap
from sim.base import (
    coverage_verdict,
    innovation_whiteness_verdict_mc,
    compute_nees,
    nees_verdict_mc,
    output_path,
    save_fig,
)


def true_trajectory_stochastic(dt, duration, rates_true=(0.10, -0.05, 0.20),
                                sigma_g=SIGMA_GYRO, sigma_b=SIGMA_BIAS, seed=999):
    """
    Verdade = amostra do processo assumido nos 3 eixos: bias com passeio
    aleatório, giro medido com ruído próprio.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    bias = np.zeros((n, 3))
    rpy = np.zeros((n, 3))
    for k in range(1, n):
        bias[k] = bias[k - 1] + rng.normal(0, sigma_b * np.sqrt(dt), 3)
        for i in range(3):
            d_angle = (rates_true[i] - bias[k - 1, i]) * dt
            rpy[k, i] = wrap(rpy[k - 1, i] + d_angle) if i == 2 else rpy[k - 1, i] + d_angle
    gyro_meas = np.array(rates_true)[None, :] + rng.normal(0, sigma_g, (n, 3))
    t = np.arange(n) * dt
    return t, rpy, bias, gyro_meas


def _run_one(dt, duration, seed, meas_seed, burn_in_s):
    t, rpy_true, bias_true, gyro_meas = true_trajectory_stochastic(dt, duration, seed=seed)
    n = len(t)

    rng = np.random.default_rng(meas_seed)
    acc_meas = rpy_true[:, :2] + rng.normal(0, SIGMA_ACC_TILT, (n, 2))
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in rpy_true[:, 2]])

    F, Gamma, Q = imu6.matrices(dt)
    H, R = imu6.observation(mag_ok=True)
    x0, P0 = imu6.initial(rpy0=(acc_meas[0, 0], acc_meas[0, 1], mag_meas[0]))
    kf = KalmanFilter(dim_x=6, angle_indices=[0, 1, 2])
    kf.x, kf.P, kf.F, kf.Gamma, kf.Q = x0, P0, F, Gamma, Q

    x_est = np.zeros((n, 6))
    P_hist = np.zeros((n, 6, 6))
    for k in range(n):
        kf.predict(u=gyro_meas[k])
        kf.update_scalar(acc_meas[k, 0], H[0], R[0, 0], is_angle=True, channel="roll")
        kf.update_scalar(acc_meas[k, 1], H[1], R[1, 1], is_angle=True, channel="pitch")
        kf.update_scalar(mag_meas[k], H[2], R[2, 2], is_angle=True, channel="yaw")
        x_est[k] = kf.x[:, 0]
        P_hist[k] = kf.P

    x_true = np.column_stack([rpy_true, bias_true])
    burn_in_k = int(round(burn_in_s / dt))
    nees_series, _ = compute_nees(x_true, x_est, P_hist, angle_indices=[0, 1, 2])
    err_yaw = wrap(x_true[:, 2] - x_est[:, 2])[burn_in_k:]

    return dict(
        nees_mean=float(np.mean(nees_series[burn_in_k:])),
        err_yaw=err_yaw, var_yaw=P_hist[burn_in_k:, 2, 2],
        innov={c: np.array(kf.innov_history[c])[burn_in_k:]
               for c in ("roll", "pitch", "yaw")},
    )


def run_formulacao(dt=DT_LOOP, duration=15.0, n_runs=20, burn_in_s=0.5):
    runs = [_run_one(dt, duration, seed=11000 + i, meas_seed=11500 + i, burn_in_s=burn_in_s)
            for i in range(n_runs)]

    ok_nees, msg_nees = nees_verdict_mc([r["nees_mean"] for r in runs], dim_x=6)
    err_all = np.concatenate([r["err_yaw"] for r in runs])
    var_all = np.concatenate([r["var_yaw"] for r in runs])
    ok_cov, _, msg_cov = coverage_verdict(err_all, var_all, label="yaw")

    # Os TRES canais, e nao so o yaw: a atualizacao e escalar e sequencial,
    # entao cada canal tem inovacao propria e um modelo errado pode aparecer em
    # um so.
    bra = [innovation_whiteness_verdict_mc([r["innov"][c] for r in runs], label=c)
           for c in ("roll", "pitch", "yaw")]
    ok_bra = all(b[0] for b in bra)

    print(f"=== sim_imu_6 — teste de formulacao ({n_runs} realizacoes independentes, com magnetometro) ===")
    print(msg_nees)
    for _, _, m in bra:
        print(m)
    print(msg_cov)

    return dict(ok=ok_nees and ok_bra and ok_cov)


def run_com_magnetometro(dt=DT_LOOP, duration=20.0, rates_true=(0.10, -0.05, 0.35),
                          bias_true_deg=(2.0, -1.5, 0.5), seed=100):
    """Correção do D.2: verdade exposta, R = variância injetada, yaw completa 360°."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    bias_true = np.deg2rad(bias_true_deg)

    rpy_true_continuo = np.outer(t, rates_true)
    rpy_true = rpy_true_continuo.copy()
    rpy_true[:, 2] = wrap(rpy_true_continuo[:, 2])

    rng = np.random.default_rng(seed)
    gyro_meas = np.array(rates_true)[None, :] + bias_true[None, :] + rng.normal(0, SIGMA_GYRO, (n, 3))
    acc_meas = rpy_true[:, :2] + rng.normal(0, SIGMA_ACC_TILT, (n, 2))
    mag_meas = np.array([wrap(p + rng.normal(0, SIGMA_MAG)) for p in rpy_true[:, 2]])

    F, Gamma, Q = imu6.matrices(dt)
    H, R = imu6.observation(mag_ok=True)
    x0, P0 = imu6.initial(rpy0=(acc_meas[0, 0], acc_meas[0, 1], mag_meas[0]))
    kf = KalmanFilter(dim_x=6, angle_indices=[0, 1, 2])
    kf.x, kf.P, kf.F, kf.Gamma, kf.Q = x0, P0, F, Gamma, Q

    x_est = np.zeros((n, 6))
    for k in range(n):
        kf.predict(u=gyro_meas[k])
        kf.update_scalar(acc_meas[k, 0], H[0], R[0, 0], is_angle=True, channel="roll")
        kf.update_scalar(acc_meas[k, 1], H[1], R[1, 1], is_angle=True, channel="pitch")
        kf.update_scalar(mag_meas[k], H[2], R[2, 2], is_angle=True, channel="yaw")
        x_est[k] = kf.x[:, 0]

    n_voltas = abs(rates_true[2]) * t[-1] / (2 * np.pi)
    erro_bias_final = np.abs(x_est[-1, 3:6] - bias_true)
    convergiu = bool(np.all(erro_bias_final < 0.25 * np.abs(bias_true)))

    yaw_est_continuo = np.unwrap(x_est[:, 2])
    yaw_est_continuo -= (yaw_est_continuo[0] - rpy_true_continuo[0, 2])
    erro_yaw_continuo = np.abs(yaw_est_continuo - rpy_true_continuo[:, 2])
    sem_salto = bool(np.all(erro_yaw_continuo < 10 * SIGMA_MAG))

    print("\n=== sim_imu_6 — cenario com magnetometro (D.2, ilustrativo) ===")
    print(f"Voltas completas em yaw: {n_voltas:.2f}")
    print(f"Bias verdadeiro [graus/s]: {bias_true_deg}")
    print(f"Bias estimado final [graus/s]: {np.degrees(x_est[-1, 3:6])}")
    print(f"Convergencia dos 3 biases (erro < 25% do valor verdadeiro): {'OK' if convergiu else 'FALHOU'}")
    print(f"Sem salto no wraparound do yaw: {'OK' if sem_salto else 'FALHOU'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["roll (phi)", "pitch (theta)", "yaw (psi)"]
    fig, axs = plt.subplots(4, 1, figsize=(9, 13))
    ax0 = axs[0]
    for i, lab in enumerate(labels):
        ax0.plot(t, np.degrees(rpy_true_continuo[:, i]), "--", alpha=0.5, label=f"verdade {lab}")
        ax0.plot(t, np.degrees(np.unwrap(x_est[:, i]) if i == 2 else x_est[:, i]), label=f"Kalman {lab}")
    axs[0].set_ylabel("angulo [graus] (continuo)")
    axs[0].set_title("sim_imu_6 — com magnetometro (verdade exposta, D.2)")
    axs[0].legend(fontsize=7, ncol=3)

    axs[1].plot(t, np.degrees(erro_yaw_continuo))
    axs[1].set_ylabel("erro yaw continuo [graus]")
    axs[1].set_title(f"sem salto: {'OK' if sem_salto else 'FALHOU'}")

    for i, lab in enumerate(["b_phi", "b_theta", "b_psi"]):
        axs[2].axhline(bias_true_deg[i], ls="--", alpha=0.5, color=f"C{i}")
        axs[2].plot(t, np.degrees(x_est[:, 3 + i]), label=f"{lab} estimado", color=f"C{i}")
    axs[2].set_ylabel("bias [graus/s]")
    axs[2].set_title(f"convergencia dos biases: {'OK' if convergiu else 'FALHOU'}")
    axs[2].legend(fontsize=7, ncol=3)

    axs[3].plot(t, np.degrees(rpy_true[:, 2]), "k--", alpha=0.5, label="verdade (wrapped)")
    axs[3].plot(t, np.degrees(x_est[:, 2]), label="Kalman (wrapped)", linewidth=0.8)
    axs[3].set_ylabel("yaw [graus], wrapped")
    axs[3].set_xlabel("t [s]")
    axs[3].legend(fontsize=8)

    save_fig(fig, output_path("imu6_com_magnetometro.png"))

    return dict(ok=convergiu and sem_salto)


def run_sem_magnetometro(dt=DT_LOOP, duration=150.0, rates_true=(0.10, -0.05, 0.35), seed=101):
    """
    Sem magnetômetro, H de 2 linhas — yaw e b_psi devem crescer sem
    limite, roll/pitch e seus biases devem continuar limitados.
    """
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt

    rng = np.random.default_rng(seed)
    gyro_meas = np.array(rates_true)[None, :] + rng.normal(0, SIGMA_GYRO, (n, 3))
    rpy_true_continuo = np.outer(t, rates_true)
    acc_meas = rpy_true_continuo[:, :2] + rng.normal(0, SIGMA_ACC_TILT, (n, 2))

    F, Gamma, Q = imu6.matrices(dt)
    H, R = imu6.observation(mag_ok=False)
    x0, P0 = imu6.initial(rpy0=(acc_meas[0, 0], acc_meas[0, 1], 0.0))
    kf = KalmanFilter(dim_x=6, angle_indices=[0, 1, 2])
    kf.x, kf.P, kf.F, kf.Gamma, kf.Q = x0, P0, F, Gamma, Q

    P_diag = np.zeros((n, 6))
    for k in range(n):
        kf.predict(u=gyro_meas[k])
        kf.update_scalar(acc_meas[k, 0], H[0], R[0, 0], is_angle=True, channel="roll")
        kf.update_scalar(acc_meas[k, 1], H[1], R[1, 1], is_angle=True, channel="pitch")
        P_diag[k] = np.diag(kf.P)

    razao_yaw = P_diag[-1, 2] / P_diag[0, 2]
    razao_bpsi = P_diag[-1, 5] / P_diag[0, 5]
    razao_roll = P_diag[-1, 0] / P_diag[0, 0]
    yaw_nao_observavel = razao_yaw > 20 and razao_bpsi > 20
    roll_limitado = razao_roll < 10

    print("\n=== sim_imu_6 — cenario SEM magnetometro (ilustrativo) ===")
    print(f"Crescimento de P[yaw,yaw] (fim/inicio): {razao_yaw:.1f}x")
    print(f"Crescimento de P[b_psi,b_psi] (fim/inicio): {razao_bpsi:.1f}x")
    print(f"Crescimento de P[roll,roll] (fim/inicio): {razao_roll:.1f}x (deve ficar limitado)")
    print(f"Yaw/b_psi nao observaveis (crescem sem limite): {'OK' if yaw_nao_observavel else 'FALHOU'}")
    print(f"Roll permanece limitado: {'OK' if roll_limitado else 'FALHOU'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, P_diag[:, 0], label="P[roll,roll]")
    ax.plot(t, P_diag[:, 2], label="P[yaw,yaw]")
    ax.plot(t, P_diag[:, 5], label="P[b_psi,b_psi]")
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("variancia (escala log)")
    ax.set_title("sim_imu_6 — sem magnetometro: yaw e b_psi crescem sem limite")
    ax.legend(fontsize=8)

    save_fig(fig, output_path("imu6_sem_magnetometro.png"))

    return dict(ok=yaw_nao_observavel and roll_limitado)


if __name__ == "__main__":
    r0 = run_formulacao()
    r1 = run_com_magnetometro()
    r2 = run_sem_magnetometro()
    tudo_ok = r0["ok"] and r1["ok"] and r2["ok"]
    print("\n=== RESULTADO:", "OK" if tudo_ok else "FALHOU", "===")
    if not tudo_ok:
        raise SystemExit(1)
