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
"""
Filtro fundido: EKF sobre a cinemática de Ackermann, 7 estados, com o
giroscópio e o acelerômetro como entradas da propagação.
"""
import numpy as np

from kalman.config import (ENTRE_EIXOS_L, GRAVITY, SIGMA_ACC,
                           SIGMA_ACC_MEAS_MOVEL, SIGMA_ACC_MEAS_PARADO,
                           SIGMA_BIAS_FUNDIDO, SIGMA_ENC_OPERACAO, SIGMA_GRADE,
                           SIGMA_GYRO_MOVEL, SIGMA_GYRO_PARADO, SIGMA_ZUPT)
from kalman.filtros import fusao
from kalman.filtros.fusao import (ANGLE_INDICES, CH_GYRO_Z, CH_TILT, DIM_X,  # noqa: F401
                                  ENCODER, IBGY, IBGZ, IMU, IPSI, ITHETA, IV,
                                  IX, IY, configuracoes, initial)

FORMULACOES = ("variante", "vigente")

# O nível desta formulação é o plano: com a aceleração como entrada, a mesma
# leitura do acelerômetro não pode ser também a medida de inclinação.
NIVEL = "plano"

# A pose do passo com a velocidade média (v + a dt/2), que é a discretização
# exata com a aceleração conhecida; False usa o v do início do passo.
V_MEDIA_NO_ARCO = True

# Os canais de pseudo-medida com o carro parado
CH_ZUPT = "zupt"
CH_ZARU = "zaru"


def esta_parado(sensores, em_movimento, v_encoder):
    """O carro está parado?"""
    if IMU not in sensores or em_movimento:
        return False
    if ENCODER in sensores:
        return v_encoder is not None and abs(v_encoder) < 1e-9
    return True


def canais_parado(x, sensores):
    """
    As pseudo-medidas do carro parado, no formato de `observation` (nome, H, R,
    usa_gate, z_pred).
    """
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    if IMU not in sensores:
        return []
    Hz = np.zeros(DIM_X)
    Hz[IV] = 1.0
    Ha = np.zeros(DIM_X)
    Ha[IBGZ] = 1.0
    return [(CH_ZUPT, Hz, SIGMA_ZUPT ** 2, False, float(x[IV, 0])),
            (CH_ZARU, Ha, SIGMA_GYRO_PARADO ** 2, False, float(x[IBGZ, 0]))]


def usa_giro(sensores):
    """Esta formulação só muda algo quando há IMU (giroscópio e acelerômetro)."""
    return IMU in sensores


def usa_escapatoria(sensores):
    """A escapatória do gate só nas configurações sem encoder."""
    return ENCODER not in sensores


# Propagacao
def propaga(x, dt, omega_medido, nivel=NIVEL, acel=None):
    """Um passo com as leituras da IMU como entrada."""
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    psi = float(x[IPSI, 0])
    theta = 0.0 if nivel == "plano" else float(x[ITHETA, 0])
    v = float(x[IV, 0])
    ct = np.cos(theta)
    omega = (omega_medido - float(x[IBGZ, 0])) / ct
    a_via = None if acel is None else float(acel) + GRAVITY * np.sin(theta)
    v_arco = v + 0.5 * a_via * dt if (a_via is not None and V_MEDIA_NO_ARCO) else v

    xn = x.copy()
    if abs(omega) < 1e-9:
        xn[IX, 0] = x[IX, 0] + v_arco * ct * np.cos(psi) * dt
        xn[IY, 0] = x[IY, 0] + v_arco * ct * np.sin(psi) * dt
    else:
        psi_n = psi + omega * dt
        K = v_arco * ct / omega
        xn[IX, 0] = x[IX, 0] + K * (np.sin(psi_n) - np.sin(psi))
        xn[IY, 0] = x[IY, 0] - K * (np.cos(psi_n) - np.cos(psi))
    xn[IPSI, 0] = psi + omega * dt
    if a_via is not None:
        xn[IV, 0] = v + a_via * dt
    if nivel == "plano":
        xn[ITHETA, 0] = 0.0
    return xn


def matrices(x, dt, omega_medido, em_movimento, nivel=NIVEL, acel=None,
             sigma_acc=SIGMA_ACC, sigma_grade=SIGMA_GRADE,
             sigma_bias=SIGMA_BIAS_FUNDIDO):
    """F (jacobiano do passo) e Q."""
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    F = np.zeros((DIM_X, DIM_X))
    for j in range(DIM_X):
        h = 1e-6 * max(1.0, abs(float(x[j, 0])))
        xp, xm = x.copy(), x.copy()
        xp[j, 0] += h
        xm[j, 0] -= h
        F[:, j] = ((propaga(xp, dt, omega_medido, nivel, acel)
                    - propaga(xm, dt, omega_medido, nivel, acel)) / (2 * h)).ravel()

    G_v = np.zeros((DIM_X, 1))
    G_v[IX, 0] = F[IX, IV] * dt / 2.0
    G_v[IY, 0] = F[IY, IV] * dt / 2.0
    G_v[IPSI, 0] = F[IPSI, IV] * dt / 2.0
    G_v[IV, 0] = dt
    if acel is not None:
        sa = SIGMA_ACC_MEAS_MOVEL if em_movimento else SIGMA_ACC_MEAS_PARADO
    else:
        sa = sigma_acc
    Q = (sa ** 2) * (G_v @ G_v.T)

    sg = SIGMA_GYRO_MOVEL if em_movimento else SIGMA_GYRO_PARADO
    ct = np.cos(0.0 if nivel == "plano" else float(x[ITHETA, 0]))
    G_g = np.zeros((DIM_X, 1))
    G_g[IPSI, 0] = dt / ct
    G_g[IX, 0] = F[IX, IPSI] * dt / (2.0 * ct)
    G_g[IY, 0] = F[IY, IPSI] * dt / (2.0 * ct)
    Q += (sg ** 2) * (G_g @ G_g.T)

    if nivel != "plano":
        Q[ITHETA, ITHETA] += (sigma_grade ** 2) * dt
    Q[IBGZ, IBGZ] += (sigma_bias ** 2) * dt
    Q[IBGY, IBGY] += (sigma_bias ** 2) * dt
    return F, Q


def passo(x, dt, delta, sensores, gz=None, em_movimento=True, nivel=NIVEL,
          ax=None, parado=False, suspende_giro=False):
    """O que o laço do filtro chama a cada passo de predição: (F, Q, f)."""
    if parado:
        ax = None
        if suspende_giro:
            gz = None
    if usa_giro(sensores) and gz is not None:
        a = None if ax is None else float(ax)
        F, Q = matrices(x, dt, gz, em_movimento, nivel=nivel, acel=a)
        f = (lambda oo, aa: (lambda xx, uu: propaga(xx, dt, oo, nivel=nivel,
                                                     acel=aa)))(float(gz), a)
    else:
        F, Q = fusao.matrices(x, dt, delta, nivel=nivel)
        f = (lambda dd: (lambda xx, uu: fusao.propaga(xx, dt, dd, nivel=nivel)))(float(delta))
    return F, Q, f


# Medidas
def observation(x, sensores, delta, **kw):
    """Os canais de `fusao.observation`, sem giroscópio z e inclinômetro quando há IMU."""
    kw.setdefault("nivel", NIVEL)
    kw.setdefault("sigma_enc", SIGMA_ENC_OPERACAO)
    canais = fusao.observation(x, sensores, delta, **kw)
    if usa_giro(sensores):
        canais = [c for c in canais if c[0] not in (CH_GYRO_Z, CH_TILT)]
    return canais


# Observabilidade

# estados usados no nível plano (theta e b_gy ficam fora)
ESTADOS_PLANO = (IX, IY, IPSI, IV, IBGZ)


def _posto(linhas, nivel, tol):
    if not linhas:
        return 0
    O = np.array(linhas)
    if nivel == "plano":
        O = O[:, list(ESTADOS_PLANO)]
    norms = np.linalg.norm(O, axis=1, keepdims=True)
    O = O / np.where(norms > 1e-12, norms, 1.0)
    return int(np.linalg.matrix_rank(O, tol=tol))


def observabilidade(sensores, trajetoria, nivel=NIVEL, x_parede=None,
                    tol=1e-7):
    """
    Posto da matriz de observabilidade ao longo da trajetória, como em
    `fusao.observabilidade`, com o F e o H desta formulação.
    """
    linhas = []
    Phi = np.eye(DIM_X)
    for x, dt, delta, em_mov, omega in trajetoria:
        for _, H, _, _, _ in observation(x, sensores, delta, em_movimento=em_mov,
                                         nivel=nivel, x_parede=x_parede):
            linhas.append(H @ Phi)
        F, _, _ = passo(x, dt, delta, sensores, gz=omega, em_movimento=em_mov,
                        nivel=nivel)
        Phi = F @ Phi
    dim = len(ESTADOS_PLANO) if nivel == "plano" else DIM_X
    return _posto(linhas, nivel, tol), dim


def trajetoria_ref(delta, n=60, dt=0.05, v=0.8, nivel=NIVEL):
    """Trajetória de referência para o posto: esterçamento constante, giro coerente."""
    x = np.zeros((DIM_X, 1))
    x[IV, 0] = v
    omega = v / ENTRE_EIXOS_L * np.tan(delta)
    out = []
    for _ in range(n):
        out.append((x.copy(), dt, delta, True, omega))
        x = propaga(x, dt, omega, nivel=nivel)
    return out


def _posto_generico(sensores, trajetoria, formulacao, x_parede, sem_mag,
                    nivel=NIVEL, tol=1e-7):
    """Posto numa formulação, com a opção de tirar o canal do magnetômetro."""
    from kalman.filtros.fusao import CH_MAG
    linhas = []
    Phi = np.eye(DIM_X)
    for x, dt, delta, em_mov, omega in trajetoria:
        cf = observation if formulacao == "variante" else fusao.observation
        for nome, H, _, _, _ in cf(x, sensores, delta, em_movimento=em_mov,
                                   x_parede=x_parede, nivel=nivel):
            if sem_mag and nome == CH_MAG:
                continue
            linhas.append(H @ Phi)
        if formulacao == "variante":
            F, _, _ = passo(x, dt, delta, sensores, gz=omega, em_movimento=em_mov,
                            nivel=nivel)
        else:
            F, _ = fusao.matrices(x, dt, delta, nivel=nivel)
        Phi = F @ Phi
    return _posto(linhas, nivel, tol)


def tabela_posto(sem_mag=False, nivel=NIVEL):
    """
    O posto das sete configurações nas duas formulações, em três regimes: reta
    com parede, curva de 12 graus com parede e curva sem marco.
    """
    d12 = np.deg2rad(12.0)
    tr = trajetoria_ref(0.0, n=120, v=1.0, nivel=nivel)
    tc = trajetoria_ref(d12, n=120, v=1.0, nivel=nivel)
    out = {}
    for rot, sens in configuracoes():
        vig = tuple(_posto_generico(sens, t, "vigente", xp, sem_mag, nivel)
                    for t, xp in ((tr, 6.0), (tc, 6.0), (tc, None)))
        var = tuple(_posto_generico(sens, t, "variante", xp, sem_mag, nivel)
                    for t, xp in ((tr, 6.0), (tc, 6.0), (tc, None)))
        out[rot] = (vig, var)
    return out


# Autoteste
def _autoteste():
    print("=== autoteste de kalman/filtros/fusao_giro.py (a VARIANTE oficial) ===\n")
    ok = True

    def confere(rot, cond):
        nonlocal ok
        print("  %-66s %s" % (rot, "OK" if cond else "FALHOU"))
        ok = ok and bool(cond)

    x = np.zeros((DIM_X, 1))
    x[IV, 0] = 0.8
    x[IPSI, 0] = 0.3
    dt = 0.05

    confere("em reta (omega = 0), reproduz o vigente",
            np.allclose(propaga(x, dt, 0.0), fusao.propaga(x, dt, 0.0), atol=1e-12))
    d = np.deg2rad(15.0)
    om = (float(x[IV, 0]) / ENTRE_EIXOS_L) * np.tan(d)
    confere("em curva, com o omega de Ackerman, reproduz o vigente",
            np.allclose(propaga(x, dt, om), fusao.propaga(x, dt, d), atol=1e-9))
    xb = x.copy()
    xb[IBGZ, 0] = 0.1
    pose = [IX, IY, IPSI]
    confere("o vies b_gz e descontado da entrada",
            np.allclose(propaga(xb, dt, om + 0.1)[pose], propaga(x, dt, om)[pose], atol=1e-12))

    F, Q = matrices(x, dt, om, True, nivel="arfagem")
    for j in (IPSI, IV, IBGZ):
        h = 1e-7
        xp, xm = x.copy(), x.copy()
        xp[j, 0] += h
        xm[j, 0] -= h
        num = (propaga(xp, dt, om, "arfagem") - propaga(xm, dt, om, "arfagem")) / (2 * h)
        confere("F confere com a derivada na coluna %d" % j,
                np.allclose(F[:, j], num.ravel(), atol=1e-5))
    confere("dpsi/db_gz = -dt (era ZERO no vigente)", abs(F[IPSI, IBGZ] + dt) < 1e-6)
    Fv, _ = fusao.matrices(x, dt, d)
    confere("no vigente, dpsi/db_gz e exatamente zero", abs(Fv[IPSI, IBGZ]) < 1e-15)
    confere("Q simetrica e positiva semidefinida",
            np.allclose(Q, Q.T, atol=1e-18) and np.min(np.linalg.eigvalsh(Q)) > -1e-18)

    # o canal do giroscopio z sai so quando ha IMU
    todos = (fusao.ENCODER, fusao.ULTRASSOM, fusao.IMU)
    nomes = [c[0] for c in observation(x, todos, d, x_parede=6.0)]
    confere("com IMU, o giro_z NAO e medida", CH_GYRO_Z not in nomes)
    confere("com IMU, o inclinometro NAO e medida, nem em arfagem",
            CH_TILT not in [c[0] for c in observation(x, todos, d, x_parede=6.0,
                                                      nivel="arfagem")])
    nv = [c[0] for c in fusao.observation(x, todos, d, x_parede=6.0, nivel=NIVEL)]
    confere("os outros canais sao os mesmos do vigente",
            sorted(nomes) == sorted(n for n in nv if n not in (CH_GYRO_Z, CH_TILT)))
    sem_imu = (fusao.ENCODER, fusao.ULTRASSOM)
    confere("sem IMU, os canais sao os do vigente",
            [c[0] for c in observation(x, sem_imu, d, x_parede=6.0)]
            == [c[0] for c in fusao.observation(x, sem_imu, d, x_parede=6.0)])

    # sem IMU, o passo cai na formulação de fusao.py
    F1, Q1, f1 = passo(x, dt, d, sem_imu, gz=0.7)
    F2, Q2 = fusao.matrices(x, dt, d, nivel=NIVEL)
    confere("sem IMU, o passo e o do vigente (F, Q e f)",
            np.allclose(F1, F2) and np.allclose(Q1, Q2)
            and np.allclose(f1(x, None), fusao.propaga(x, dt, d, nivel=NIVEL)))


    # a aceleração como entrada
    a = -0.39
    xa = propaga(x, dt, om, acel=a)
    confere("a aceleracao integra a velocidade: v' = v + a dt",
            abs(float(xa[IV, 0]) - (0.8 + a * dt)) < 1e-15)
    confere("a pose do passo anda com a velocidade media (V_MEDIA_NO_ARCO)",
            V_MEDIA_NO_ARCO and np.allclose(
                xa[[IX, IY]] - x[[IX, IY]],
                (propaga(x, dt, om)[[IX, IY]] - x[[IX, IY]]) * (0.8 + 0.5 * a * dt) / 0.8,
                atol=1e-12, rtol=0))
    Fa, Qa = matrices(x, dt, om, True, acel=a)
    F0, Q0 = matrices(x, dt, om, True)
    xm = x.copy()
    xm[IV, 0] += 0.5 * a * dt
    Fm, _ = matrices(xm, dt, om, True)
    confere("com a velocidade media, o jacobiano e o avaliado em v + a dt/2",
            np.allclose(Fa[:, [IPSI, IBGZ]], Fm[:, [IPSI, IBGZ]], atol=1e-9, rtol=0)
            and np.allclose(Fa[:, IV], F0[:, IV], atol=1e-9, rtol=0))
    confere("Q da velocidade = SIGMA_ACC_MEAS_MOVEL^2 dt^2 em movimento",
            abs(Qa[IV, IV] - (SIGMA_ACC_MEAS_MOVEL * dt) ** 2) < 1e-15)
    _, Qp = matrices(x, dt, om, False, acel=a)
    confere("e SIGMA_ACC_MEAS_PARADO^2 dt^2 parado",
            abs(Qp[IV, IV] - (SIGMA_ACC_MEAS_PARADO * dt) ** 2) < 1e-15)
    confere("sem aceleracao, o Q da velocidade e o do vigente (SIGMA_ACC)",
            abs(Q0[IV, IV] - (SIGMA_ACC * dt) ** 2) < 1e-15)
    _, _, fa = passo(x, dt, d, todos, gz=om, ax=a)
    confere("o passo repassa a aceleracao", np.array_equal(fa(x, None), xa))
    # a suspensão das entradas com o carro parado (opção desligada)
    _, _, fp = passo(x, dt, d, todos, gz=om, ax=a, parado=True)
    confere("parado: a entrada do acelerometro e suspensa (v nao muda)",
            float(fp(x, None)[IV, 0]) == float(x[IV, 0])
            and np.array_equal(fp(x, None), propaga(x, dt, om)))
    _, _, fg = passo(x, dt, 0.0, todos, gz=om, ax=a, parado=True, suspende_giro=True)
    confere("parado com suspende_giro: o giroscopio tambem (rumo pelo esterco)",
            np.allclose(fg(x, None), fusao.propaga(x, dt, 0.0, nivel=NIVEL)))
    _, _, fn = passo(x, dt, d, todos, gz=om, ax=a)
    confere("sem `parado` (o padrao), nada muda", np.array_equal(fn(x, None), xa))

    # ZUPT + ZARU
    confere("parado: detector parado e encoder em zero",
            esta_parado(todos, False, 0.0) and not esta_parado(todos, False, 0.3)
            and not esta_parado(todos, True, 0.0))
    confere("sem encoder, o encoder NAO e consultado (sem vazamento)",
            esta_parado((fusao.ULTRASSOM, fusao.IMU), False, 0.3))
    confere("sem IMU nao ha detector: nunca parado",
            not esta_parado(sem_imu, False, 0.0))
    cp = canais_parado(x, todos)
    confere("ZUPT mede v e ZARU mede b_gz, com os R declarados",
            [c[0] for c in cp] == [CH_ZUPT, CH_ZARU]
            and cp[0][1][IV] == 1.0 and abs(cp[0][2] - SIGMA_ZUPT ** 2) < 1e-18
            and cp[1][1][IBGZ] == 1.0 and abs(cp[1][2] - SIGMA_GYRO_PARADO ** 2) < 1e-18)
    confere("sem IMU, nenhuma pseudo-medida", canais_parado(x, sem_imu) == [])

    # o sigma do encoder de operacao, e a escapatoria so sem encoder
    r_enc = [c[2] for c in observation(x, todos, d, x_parede=6.0) if c[0] == fusao.CH_ENCODER]
    confere("o encoder entra com SIGMA_ENC_OPERACAO",
            len(r_enc) == 1 and abs(r_enc[0] - SIGMA_ENC_OPERACAO ** 2) < 1e-15)
    r_enc2 = [c[2] for c in observation(x, todos, d, x_parede=6.0, sigma_enc=0.1)
              if c[0] == fusao.CH_ENCODER]
    confere("salvo sigma_enc explicito", abs(r_enc2[0] - 0.01) < 1e-15)
    esc = {rot: usa_escapatoria(s) for rot, s in configuracoes()}
    confere("escapatoria so nas tres configuracoes sem encoder (%s)"
            % ", ".join(r for r, e in esc.items() if e),
            sum(esc.values()) == 3
            and all(e == (fusao.ENCODER not in s) for (r, s), e in
                    zip(configuracoes(), esc.values())))

    # o posto: sem medida absoluta de rumo, a pose não é observável; com o
    # giroscópio como entrada, o viés b_gz só aparece pelo rumo, que nada mede
    tr = trajetoria_ref(np.deg2rad(10.0))
    p_ei, dim = observabilidade((fusao.ENCODER, fusao.IMU), tr)
    confere("encoder+imu (sem parede) tem posto %d de %d, incompleto" % (p_ei, dim),
            p_ei < dim)
    confere("a IMU sozinha, no plano e sem magnetometro, tem posto 0",
            _posto_generico((fusao.IMU,), tr, "variante", None, sem_mag=True) == 0)

    print("\n=== RESULTADO: %s ===" % ("OK" if ok else "FALHOU"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _autoteste() else 1)
