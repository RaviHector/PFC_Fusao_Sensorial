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
"""Ciclo genérico do filtro de Kalman (linear e estendido)."""
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

from collections import defaultdict

import numpy as np

from kalman.config import GATING_THRESHOLD
from kalman.utils import wrap


class KalmanFilter:
    """Filtro de Kalman linear ou estendido (EKF), com atualização escalar sequencial."""

    def __init__(self, dim_x, angle_indices=None,
                 gate_escape_after=None, gate_escape_factor=2.0):
        self.dim_x = dim_x
        self.angle_indices = list(angle_indices) if angle_indices else []

        # Escapatória do gate: None = desligada (padrão). Só o filtro fundido
        # a liga; a explicação está em `_escapatoria`.
        self.gate_escape_after = gate_escape_after
        self.gate_escape_factor = gate_escape_factor
        self._rej_seguidas = defaultdict(int)
        self.escape_count = defaultdict(int)

        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.Gamma = None  # (dim_x, dim_u); None = sem entrada de controle
        self.Q = np.eye(dim_x)
        self.f = None  # None = filtro linear; senao, f(x, u) -> x do EKF

        self.nis_history = defaultdict(list)
        # Inovação normalizada (innovation / sqrt(S)), com sinal, por canal.
        self.innov_history = defaultdict(list)
        self.reject_count = defaultdict(int)
        self.update_count = defaultdict(int)

    def predict(self, u=None):
        """Predição: x = f(x, u) (ou F x + Gamma u) e P = F P F' + Q."""
        if self.f is None:
            self.x = self.F @ self.x
            if self.Gamma is not None and u is not None:
                u = np.asarray(u, dtype=float).reshape(-1, 1)
                self.x = self.x + self.Gamma @ u
        else:
            self.x = np.asarray(self.f(self.x, u), dtype=float).reshape(self.dim_x, 1)

        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_scalar(self, z, H_row, R, is_angle=False, gate=False,
                       gate_threshold=None, channel=None, z_pred=None):
        """
        Atualização escalar sequencial: um canal por vez, S escalar, sem
        inversão de matriz.
        """
        H_row = np.asarray(H_row, dtype=float).reshape(1, self.dim_x)

        if z_pred is None:
            z_pred = float((H_row @ self.x).item())
        innovation = float(z) - float(z_pred)
        if is_angle:
            innovation = wrap(innovation)

        S = float((H_row @ self.P @ H_row.T).item()) + float(R)
        nis = innovation * innovation / S

        if channel is not None:
            self.nis_history[channel].append(nis)
            self.update_count[channel] += 1

        if gate:
            threshold = gate_threshold if gate_threshold is not None else GATING_THRESHOLD
            if nis > threshold:
                if channel is not None:
                    self.reject_count[channel] += 1
                self._escapatoria(channel)
                return False, nis
            if channel is not None:
                self._rej_seguidas[channel] = 0

        if channel is not None:
            # Só a inovação aceita entra no histórico.
            self.innov_history[channel].append(innovation / np.sqrt(S))

        K = (self.P @ H_row.T) / S  # (dim_x, 1)
        self.x = self.x + K * innovation

        for idx in self.angle_indices:
            self.x[idx, 0] = wrap(self.x[idx, 0])

        # Forma de Joseph, numericamente robusta mesmo com K não ótimo:
        # P = (I-KH)P(I-KH)' + KRK'.
        I = np.eye(self.dim_x)
        IKH = I - K @ H_row
        self.P = IKH @ self.P @ IKH.T + (K @ K.T) * float(R)
        self.P = 0.5 * (self.P + self.P.T)

        return True, nis

    def _escapatoria(self, channel):
        """Escapatória do gate, inerte quando `gate_escape_after` é None."""
        if self.gate_escape_after is None or channel is None:
            return
        n = self._rej_seguidas.get(channel, 0) + 1
        self._rej_seguidas[channel] = n
        if n >= self.gate_escape_after:
            self.P = self.P * float(self.gate_escape_factor)
            self.escape_count[channel] += 1

    def reject_rate(self, channel):
        """Fração de medidas rejeitadas pelo gate naquele canal."""
        total = self.update_count.get(channel, 0)
        if total == 0:
            return 0.0
        return self.reject_count.get(channel, 0) / total


if __name__ == "__main__":
    print("=== kalman.core - autoteste ===\n")

    # 1) wrap() nas bordas
    casos = [
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (3 * np.pi, -np.pi),
        (-3 * np.pi, -np.pi),
        (0.0, 0.0),
    ]
    ok = True
    for a, esperado in casos:
        r = wrap(a)
        if not np.isclose(r, esperado, atol=1e-9):
            ok = False
            print(f"  FALHOU wrap({a:.4f}) = {r:.4f}, esperado {esperado:.4f}")
    print(f"1) wrap() nas bordas (+-pi, +-3pi): {'OK' if ok else 'FALHOU'}")

    # inovação através da descontinuidade: estado em +179 graus, medida em -179 graus
    x_deg, z_deg = 179.0, -179.0
    innov_sem_wrap = np.deg2rad(z_deg) - np.deg2rad(x_deg)
    innov_com_wrap = wrap(np.deg2rad(z_deg) - np.deg2rad(x_deg))
    print(f"   sem wrap: inovacao = {np.degrees(innov_sem_wrap):+.1f} graus (catastrofico)")
    print(f"   com wrap: inovacao = {np.degrees(innov_com_wrap):+.1f} graus (correto, ~+2)")
    assert np.isclose(np.degrees(innov_com_wrap), 2.0, atol=1e-6)

    # 2) update_scalar sequencial == update conjunto em matriz (R diagonal)
    print("\n2) Update sequencial vs. update conjunto (R diagonal):")

    rng = np.random.default_rng(42)
    dim_x = 3
    F = np.array([[1.0, 0.02, 0.0002], [0.0, 1.0, 0.02], [0.0, 0.0, 1.0]])
    Q = 0.01 * np.eye(dim_x)
    H1 = np.array([[1.0, 0.0, 0.0]])
    H2 = np.array([[0.0, 1.0, 0.0]])
    R1, R2 = 0.5, 0.2
    x0 = np.array([[0.1], [0.2], [-0.05]])
    P0 = np.eye(dim_x) * 2.0

    # --- sequencial (kalman.core, com Joseph)
    kf = KalmanFilter(dim_x)
    kf.x, kf.P, kf.F, kf.Q = x0.copy(), P0.copy(), F.copy(), Q.copy()

    # --- conjunto (referência independente, forma simples)
    x_j, P_j = x0.copy(), P0.copy()

    max_dev_x = 0.0
    max_dev_P = 0.0
    for _ in range(200):
        z1 = float((H1 @ x0).item()) + rng.normal(0, np.sqrt(R1))
        z2 = float((H2 @ x0).item()) + rng.normal(0, np.sqrt(R2))

        kf.predict()
        kf.update_scalar(z1, H1, R1, channel="c1")
        kf.update_scalar(z2, H2, R2, channel="c2")

        x_j = F @ x_j
        P_j = F @ P_j @ F.T + Q
        H = np.vstack([H1, H2])
        R = np.diag([R1, R2])
        z = np.array([[z1], [z2]])
        S = H @ P_j @ H.T + R
        K = P_j @ H.T @ np.linalg.inv(S)
        x_j = x_j + K @ (z - H @ x_j)
        P_j = (np.eye(dim_x) - K @ H) @ P_j

        max_dev_x = max(max_dev_x, float(np.max(np.abs(kf.x - x_j))))
        max_dev_P = max(max_dev_P, float(np.max(np.abs(kf.P - P_j))))

    print(f"   desvio maximo em x: {max_dev_x:.2e}")
    print(f"   desvio maximo em P: {max_dev_P:.2e}")
    equiv_ok = max_dev_x < 1e-9 and max_dev_P < 1e-9
    print(f"   equivalencia sequencial == conjunto: {'OK' if equiv_ok else 'FALHOU'}")

    # 3) Gate rejeita outlier e mantém x/P inalterados
    print("\n3) Gating:")
    kf2 = KalmanFilter(1)
    kf2.x = np.array([[0.0]])
    kf2.P = np.array([[0.1]])
    kf2.F = np.array([[1.0]])
    kf2.Q = np.array([[0.01]])
    H = np.array([[1.0]])
    R = 0.1

    kf2.predict()
    x_antes, P_antes = kf2.x.copy(), kf2.P.copy()
    aceito, nis = kf2.update_scalar(50.0, H, R, gate=True, channel="outlier")
    print(f"   medida de outlier (z=50): aceito={aceito}, NIS={nis:.1f}")
    inalterado = np.allclose(kf2.x, x_antes) and np.allclose(kf2.P, P_antes)
    print(f"   x/P inalterados apos rejeicao: {'OK' if inalterado else 'FALHOU'}")

    # 4) Ganchos do EKF
    print("\n4) Ganchos de EKF:")

    # 4a) O caminho estendido generaliza o linear.
    F4 = np.array([[1.0, 0.05], [0.0, 1.0]])
    Q4 = np.array([[1e-4, 0.0], [0.0, 1e-3]])
    H4 = np.array([[1.0, 0.0]])
    R4 = 0.01
    rng4 = np.random.default_rng(7)
    zs = [0.3 + 0.05 * k + rng4.normal(0, 0.1) for k in range(120)]

    kf_lin = KalmanFilter(2)
    kf_lin.x, kf_lin.P, kf_lin.F, kf_lin.Q = (np.zeros((2, 1)), np.eye(2),
                                              F4.copy(), Q4.copy())
    kf_ext = KalmanFilter(2)
    kf_ext.x, kf_ext.P, kf_ext.F, kf_ext.Q = (np.zeros((2, 1)), np.eye(2),
                                              F4.copy(), Q4.copy())
    kf_ext.f = lambda x, u: F4 @ x  # f linear: o EKF tem que virar o KF

    for z in zs:
        kf_lin.predict()
        kf_lin.update_scalar(z, H4, R4, channel="c")
        kf_ext.predict()
        kf_ext.update_scalar(z, H4, R4, channel="c",
                             z_pred=float((H4 @ kf_ext.x).item()))

    dev_x = float(np.max(np.abs(kf_lin.x - kf_ext.x)))
    dev_P = float(np.max(np.abs(kf_lin.P - kf_ext.P)))
    print(f"   f linear via gancho == caminho linear: dx={dev_x:.2e} dP={dev_P:.2e}")
    ekf_equiv_ok = dev_x == 0.0 and dev_P == 0.0
    print(f"   identidade exata: {'OK' if ekf_equiv_ok else 'FALHOU'}")

    # 4b) Um caso não linear: estimar (x, y) pela distância a um farol na
    #     origem. h(x) = sqrt(x^2 + y^2) não tem H constante.
    verdade = np.array([[3.0], [4.0]])  # distancia verdadeira = 5
    kf_nl = KalmanFilter(2)
    kf_nl.x = np.array([[1.0], [1.0]])  # chute ruim de proposito
    kf_nl.P = np.eye(2) * 4.0
    kf_nl.F = np.eye(2)
    kf_nl.Q = np.eye(2) * 1e-6
    kf_nl.f = lambda x, u: x  # parado

    rng_nl = np.random.default_rng(11)
    for k in range(400):
        # o farol se move em circulo ao redor: sem isso uma unica distancia
        # deixa a posicao sobre um circulo inteiro, e nada e observavel
        ang = 2 * np.pi * k / 40.0
        farol = np.array([[6.0 * np.cos(ang)], [6.0 * np.sin(ang)]])
        d_true = float(np.linalg.norm(verdade - farol))
        z_nl = d_true + rng_nl.normal(0, 0.05)

        kf_nl.predict()
        dvec = kf_nl.x - farol
        d_est = float(np.linalg.norm(dvec))
        if d_est < 1e-6:
            continue
        H_nl = (dvec / d_est).T  # jacobiano de h
        kf_nl.update_scalar(z_nl, H_nl, 0.05**2, channel="farol", z_pred=d_est)

    erro_nl = float(np.max(np.abs(kf_nl.x - verdade)))
    print(f"   EKF nao linear (trilateracao): erro final = {erro_nl:.4f} m")
    ekf_nl_ok = erro_nl < 0.05
    print(f"   convergiu para a verdade: {'OK' if ekf_nl_ok else 'FALHOU'}")

    # Escapatória do gate
    print("\n6. ESCAPATORIA DO GATE")

    def _trava(escape_after):
        """Reproduz o travamento: P colapsado e medidas legitimas fora dele."""
        k = KalmanFilter(dim_x=1, gate_escape_after=escape_after,
                         gate_escape_factor=2.0)
        k.x = np.array([[0.0]])
        k.P = np.array([[1e-4]])     # sigma = 10 mm
        H = np.array([[1.0]])
        aceitas = 0
        for _ in range(30):
            aceito, _ = k.update_scalar(0.5, H, 1e-4, gate=True,
                                        gate_threshold=6.63, channel="us")
            if aceito:
                aceitas += 1
        return aceitas, k

    sem, k_sem = _trava(None)
    com, k_com = _trava(3)
    print(f"   sem escapatoria: {sem} de 30 medidas aceitas")
    print(f"   com escapatoria: {com} de 30 medidas aceitas")
    trava_ok = (sem == 0 and com > 0)
    print(f"   o gate TRAVA sem ela e DESTRAVA com ela: "
          f"{'OK' if trava_ok else 'FALHOU'}")

    # e o padrão tem de ser desligado
    padrao = KalmanFilter(dim_x=1)
    desligada_ok = padrao.gate_escape_after is None
    print(f"   desligada por padrao: "
          f"{'OK' if desligada_ok else 'FALHOU'}")

    # sem gate, a escapatoria nao pode agir
    k_ng = KalmanFilter(dim_x=1, gate_escape_after=1)
    k_ng.x, k_ng.P = np.array([[0.0]]), np.array([[1e-4]])
    P0_ng = float(k_ng.P[0, 0])
    k_ng.update_scalar(0.5, np.array([[1.0]]), 1e-4, channel="us")
    sem_gate_ok = float(k_ng.P[0, 0]) < P0_ng     # so a correcao normal
    print(f"   canal SEM gate nao aciona a escapatoria: "
          f"{'OK' if sem_gate_ok else 'FALHOU'}")

    # uma aceitacao zera o contador de consecutivas
    k_z = KalmanFilter(dim_x=1, gate_escape_after=3)
    k_z.x, k_z.P = np.array([[0.0]]), np.array([[1.0]])
    H1 = np.array([[1.0]])
    k_z.update_scalar(20.0, H1, 1e-4, gate=True, gate_threshold=6.63,
                      channel="us")   # rejeita (1)
    k_z.update_scalar(20.0, H1, 1e-4, gate=True, gate_threshold=6.63,
                      channel="us")   # rejeita (2)
    k_z.update_scalar(0.0, H1, 1e-4, gate=True, gate_threshold=6.63,
                      channel="us")   # ACEITA -> zera
    zera_ok = (k_z._rej_seguidas["us"] == 0 and k_z.escape_count["us"] == 0)
    print(f"   uma aceitacao zera o contador de consecutivas: "
          f"{'OK' if zera_ok else 'FALHOU'}")

    escape_ok = trava_ok and desligada_ok and sem_gate_ok and zera_ok

    print("\n=== autoteste concluido ===")

    tudo_ok = (ok and equiv_ok and inalterado and ekf_equiv_ok and ekf_nl_ok
               and escape_ok)
    if not tudo_ok:
        raise SystemExit(1)
