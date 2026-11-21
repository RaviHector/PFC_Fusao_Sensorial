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
Filtro fundido com o esterçamento como entrada: EKF sobre a cinemática de
Ackermann (modelo de bicicleta), parametrizado pelas 7 configurações de
sensores.
"""
import numpy as np

from kalman.config import (ALFA_US, ELL_S, ENTRE_EIXOS_L,
                           GATING_THRESHOLD_99, SIGMA_ACC,
                           SIGMA_BIAS0, SIGMA_POS0, SIGMA_PSI0,
                           SIGMA_THETA0, SIGMA_V0,
                           SIGMA_ACC_TILT_MOVEL, SIGMA_ACC_TILT_PARADO,
                           SIGMA_BIAS_FUNDIDO, SIGMA_ENC, SIGMA_GRADE,
                           SIGMA_GYRO_MOVEL, SIGMA_GYRO_PARADO, SIGMA_MAG,
                           SIGMA_STEER, SIGMA_US)

# Nomes canonicos dos SENSORES (o que a configuracao liga ou desliga)
ENCODER = "encoder"
ULTRASSOM = "ultrassom"
IMU = "imu"

# Nomes canonicos dos CANAIS (o que aporta uma linha do jacobiano de h).
# Um sensor pode dar mais de um canal: a IMU da quatro.
CH_ENCODER = "encoder"
CH_ULTRASSOM = "ultrassom"
CH_MAG = "magnetometro"
CH_GYRO_Z = "giro_z"
CH_GYRO_Y = "giro_y"
CH_TILT = "inclinometro"

# indices do estado, para nao espalhar numeros magicos
IX, IY, IPSI, ITHETA, IV, IBGZ, IBGY = range(7)
DIM_X = 7
ANGLE_INDICES = [IPSI, ITHETA]

NIVEIS = ("plano", "arfagem")


def configuracoes():
    """
    As 7 configurações comparadas: os três sensores isolados, os três pares e
    os três juntos.
    """
    return [
        ("encoder", (ENCODER,)),
        ("ultrassom", (ULTRASSOM,)),
        ("imu", (IMU,)),
        ("encoder+ultrassom", (ENCODER, ULTRASSOM)),
        ("encoder+imu", (ENCODER, IMU)),
        ("ultrassom+imu", (ULTRASSOM, IMU)),
        ("os tres", (ENCODER, ULTRASSOM, IMU)),
    ]


# Propagacao
def propaga(x, dt, delta, nivel="arfagem", L=ENTRE_EIXOS_L):
    """f(x, u) do EKF: integra a cinemática de Ackermann por um passo `dt`."""
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    psi = float(x[IPSI, 0])
    theta = 0.0 if nivel == "plano" else float(x[ITHETA, 0])
    v = float(x[IV, 0])
    ct = np.cos(theta)
    td = np.tan(delta)

    xn = x.copy()
    if abs(td) < 1e-9:
        # reta: o arco degenera, e a integral e trivial
        xn[IX, 0] = x[IX, 0] + v * ct * np.cos(psi) * dt
        xn[IY, 0] = x[IY, 0] + v * ct * np.sin(psi) * dt
    else:
        omega = (v / L) * td / ct
        psi_n = psi + omega * dt
        K = L * ct * ct / td
        xn[IX, 0] = x[IX, 0] + K * (np.sin(psi_n) - np.sin(psi))
        xn[IY, 0] = x[IY, 0] - K * (np.cos(psi_n) - np.cos(psi))
    xn[IPSI, 0] = psi + (v / L) * td / ct * dt

    if nivel == "plano":
        xn[ITHETA, 0] = 0.0
    # theta, v, b_gz e b_gy são propagados como constantes: o modelo não sabe
    # como eles mudam, e é o Q que representa isso.
    return xn


def matrices(x, dt, delta, nivel="arfagem", L=ENTRE_EIXOS_L,
             sigma_acc=SIGMA_ACC, sigma_grade=SIGMA_GRADE,
             sigma_steer=SIGMA_STEER,
             sigma_bias=SIGMA_BIAS_FUNDIDO):
    """F (jacobiano de f no estado atual) e Q, para o passo `dt`."""
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    psi = float(x[IPSI, 0])
    theta = 0.0 if nivel == "plano" else float(x[ITHETA, 0])
    v = float(x[IV, 0])
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    td = np.tan(delta)

    F = np.eye(DIM_X)
    # psi' = psi + (v/L) td/ct dt, igual nos dois ramos
    dpsi_dv = (td / L) / ct * dt
    dpsi_dth = (v / L) * td * st / (ct * ct) * dt
    F[IPSI, IV] = dpsi_dv
    if nivel != "plano":
        F[IPSI, ITHETA] = dpsi_dth

    if abs(td) < 1e-9:
        # Ramo reto: o arco degenera e o jacobiano é o de sempre
        F[IX, IPSI] = -v * ct * sp * dt
        F[IX, IV] = ct * cp * dt
        F[IY, IPSI] = v * ct * cp * dt
        F[IY, IV] = ct * sp * dt
        if nivel != "plano":
            F[IX, ITHETA] = -v * st * cp * dt
            F[IY, ITHETA] = -v * st * sp * dt
    else:
        # Ramo do arco. Derivando x' = x + K(sin psi' - sin psi) com
        # K = L ct^2 / td, que não depende de v:
        omega = (v / L) * td / ct
        psi_n = psi + omega * dt
        cpn, spn = np.cos(psi_n), np.sin(psi_n)
        K = L * ct * ct / td
        dK_dth = -2.0 * L * ct * st / td

        F[IX, IPSI] = K * (cpn - cp)
        F[IY, IPSI] = K * (spn - sp)
        F[IX, IV] = K * cpn * dpsi_dv
        F[IY, IV] = K * spn * dpsi_dv
        if nivel != "plano":
            F[IX, ITHETA] = dK_dth * (spn - sp) + K * cpn * dpsi_dth
            F[IY, ITHETA] = -dK_dth * (cpn - cp) + K * spn * dpsi_dth

    # 1) aceleração desconhecida, propagada pela mesma integração de F.
    G_v = np.zeros((DIM_X, 1))
    G_v[IX, 0] = F[IX, IV] * dt / 2.0
    G_v[IY, 0] = F[IY, IV] * dt / 2.0
    G_v[IPSI, 0] = F[IPSI, IV] * dt / 2.0
    G_v[IV, 0] = dt
    Q = (sigma_acc**2) * (G_v @ G_v.T)

    # 2) erro de estercamento: d(psi)/d(delta) = (v/L) sec^2(delta) / cos(theta)
    G_d = np.zeros((DIM_X, 1))
    G_d[IPSI, 0] = (v / L) / (np.cos(delta) ** 2) / ct * dt
    Q += (sigma_steer**2) * (G_d @ G_d.T)

    # 3) passeios aleatorios proprios
    if nivel != "plano":
        Q[ITHETA, ITHETA] += (sigma_grade**2) * dt
    Q[IBGZ, IBGZ] += (sigma_bias**2) * dt
    Q[IBGY, IBGY] += (sigma_bias**2) * dt


    return F, 0.5 * (Q + Q.T)


# Medidas
def _canal_ultrassom(x, psi, x_parede, ell_s, alfa):
    """
    Medida prevista e jacobiano do ultrassom, com o feixe tratado como cone.
    """
    D = x_parede - float(x[IX, 0])
    d_s = D - ell_s * np.cos(psi)
    phi = max(abs(psi) - alfa, 0.0)
    cphi = np.cos(phi)
    if cphi < 1e-3:               # feixe quase paralelo à parede
        cphi = 1e-3

    z = d_s / cphi
    H = np.zeros(DIM_X)
    H[IX] = -1.0 / cphi
    H[IPSI] = ell_s * np.sin(psi) / cphi
    if phi > 0.0:
        H[IPSI] += (d_s * np.sin(phi) * np.sign(psi)) / (cphi * cphi)
    return z, H


def observation(x, sensores, delta, em_movimento=True, nivel="arfagem",
                L=ENTRE_EIXOS_L, x_parede=None, sigma_us=SIGMA_US, ell_s=ELL_S,
                alfa_us=ALFA_US, sigma_enc=SIGMA_ENC, sigma_mag=SIGMA_MAG):
    """
    Canais de medida da configuração: lista de (nome, H_row, R, usa_gate,
    z_pred).
    """
    x = np.asarray(x, dtype=float).reshape(DIM_X, 1)
    psi = float(x[IPSI, 0])
    theta = 0.0 if nivel == "plano" else float(x[ITHETA, 0])
    v = float(x[IV, 0])
    canais = []

    if ULTRASSOM in sensores and x_parede is not None:
        cp = np.cos(psi)
        # de costas para a parede, ou de lado: o feixe não a alcança
        if cp > 1e-3:
            d, H = _canal_ultrassom(x, psi, x_parede, ell_s, alfa_us)
            canais.append((CH_ULTRASSOM, H, sigma_us**2, True, d))

    if ENCODER in sensores:
        H = np.zeros(DIM_X)
        H[IV] = 1.0
        canais.append((CH_ENCODER, H, sigma_enc**2, False, v))

    if IMU in sensores:
        # magnetômetro -> guinada absoluta. É a única fonte absoluta de
        # orientação; sem ela psi só integra e deriva sem limite.
        H = np.zeros(DIM_X)
        H[IPSI] = 1.0
        canais.append((CH_MAG, H, sigma_mag**2, False, psi))

        # giroscópio z -> taxa de guinada; o R soma o erro de esterçamento,
        # com peso (v/L) sec^2(delta)
        sigma_g = SIGMA_GYRO_MOVEL if em_movimento else SIGMA_GYRO_PARADO
        peso_delta = (v / L) / (np.cos(delta) ** 2)
        r_gz = sigma_g**2 + (peso_delta * SIGMA_STEER) ** 2
        H = np.zeros(DIM_X)
        H[IV] = np.tan(delta) / L
        H[IBGZ] = 1.0
        canais.append((CH_GYRO_Z, H, r_gz, False,
                       (v / L) * np.tan(delta) + float(x[IBGZ, 0])))

        if nivel != "plano":
            # acelerômetro -> inclinação, pela direção da gravidade
            sigma_t = (SIGMA_ACC_TILT_MOVEL if em_movimento
                       else SIGMA_ACC_TILT_PARADO)
            H = np.zeros(DIM_X)
            H[ITHETA] = 1.0
            canais.append((CH_TILT, H, sigma_t**2, False, theta))

            # giroscópio y -> taxa de arfagem.
            H = np.zeros(DIM_X)
            H[IBGY] = 1.0
            canais.append((CH_GYRO_Y, H, sigma_g**2, False,
                           float(x[IBGY, 0])))

    return canais


def initial(sensores, pose0=(0.0, 0.0, 0.0), v0=0.0, nivel="arfagem",
            x_parede=None, sigma_pos0=SIGMA_POS0, sigma_psi0=SIGMA_PSI0,
            sigma_v0=SIGMA_V0, sigma_theta0=SIGMA_THETA0,
            sigma_bias0=SIGMA_BIAS0):
    """Estado e covariância iniciais."""
    x0 = np.zeros((DIM_X, 1))
    x0[IX, 0], x0[IY, 0], x0[IPSI, 0] = pose0
    x0[IV, 0] = v0

    var = np.zeros(DIM_X)
    var[IX] = sigma_pos0**2
    var[IY] = sigma_pos0**2
    var[IPSI] = sigma_psi0**2
    var[ITHETA] = sigma_theta0**2 if nivel != "plano" else 1e-12
    var[IV] = sigma_v0**2
    var[IBGZ] = sigma_bias0**2
    var[IBGY] = sigma_bias0**2
    return x0, np.diag(var)


# Observabilidade
def observabilidade(sensores, trajetoria, nivel="arfagem", L=ENTRE_EIXOS_L,
                    x_parede=None, tol=1e-7):
    """
    Posto da matriz de observabilidade da configuração, ao longo da trajetória.
    """
    linhas = []
    Phi = np.eye(DIM_X)
    for x, dt, delta, em_mov in trajetoria:
        for _, H, _, _, _ in observation(x, sensores, delta,
                                         em_movimento=em_mov, nivel=nivel,
                                         L=L, x_parede=x_parede):
            linhas.append(H @ Phi)
        F, _ = matrices(x, dt, delta, nivel=nivel, L=L)
        Phi = F @ Phi
    if not linhas:
        return 0, DIM_X
    O = np.array(linhas)
    # normaliza as linhas: sem isso o posto numérico depende das unidades, e
    # uma linha em rad/s (ordem 1e-2) some diante de uma em metros
    norms = np.linalg.norm(O, axis=1, keepdims=True)
    O = O / np.where(norms > 1e-12, norms, 1.0)
    return int(np.linalg.matrix_rank(O, tol=tol)), DIM_X


# Trajetorias de referencia para o autoteste
def _traj(delta_fn, n=120, dt=0.05, v=1.0, em_mov=True, nivel="arfagem"):
    """Gera uma trajetória propagando o próprio modelo. Usada nos testes."""
    x = np.zeros((DIM_X, 1))
    x[IV, 0] = v
    out = []
    for k in range(n):
        d = delta_fn(k)
        out.append((x.copy(), dt, d, em_mov))
        x = propaga(x, dt, d, nivel=nivel)
    return out


# Autoteste
def _autoteste():
    from kalman.config import DT_LOOP
    from kalman.core import KalmanFilter
    from kalman.utils import wrap

    print("=" * 74)
    print("AUTOTESTE  kalman/filtros/fusao.py   (Ackerman, EKF, 7 estados)")
    print("=" * 74)
    ok = True
    dt = DT_LOOP

    print("")
    print("  1. A CINEMATICA REPRODUZ GEOMETRIA CONHECIDA")

    # 1a) delta = 0 -> linha reta, e o avanco e exatamente v*t
    x = np.zeros((DIM_X, 1)); x[IV, 0] = 1.0
    for _ in range(200):
        x = propaga(x, dt, 0.0)
    erro = abs(float(x[IX, 0]) - 10.0) + abs(float(x[IY, 0]))
    print("     reta: x = %.6f m (esperado 10), y = %.2e m" % (x[IX, 0], x[IY, 0]))
    if erro > 1e-9:
        print("     FALHOU"); ok = False

    # 1b) delta constante -> círculo de raio L/tan(delta)
    delta = np.deg2rad(15.0)
    R_esperado = ENTRE_EIXOS_L / np.tan(delta)
    x = np.zeros((DIM_X, 1)); x[IV, 0] = 1.0
    pts = []
    # uma volta inteira: perimetro 2*pi*R a 1 m/s
    n = int(round(2 * np.pi * R_esperado / (1.0 * dt)))
    for _ in range(n):
        x = propaga(x, dt, delta)
        pts.append((float(x[IX, 0]), float(x[IY, 0])))
    pts = np.array(pts)
    # o centro esta em (0, R) porque a largada e na origem apontando para +X
    R_obtido = float(np.mean(np.linalg.norm(pts - np.array([0.0, R_esperado]),
                                            axis=1)))
    desvio = 100 * abs(R_obtido - R_esperado) / R_esperado
    print("     circulo a %.0f graus: R = %.4f m, esperado L/tan(d) = %.4f m  (%.2f%%)"
          % (np.degrees(delta), R_obtido, R_esperado, desvio))
    if desvio > 0.5:
        print("     FALHOU: Euler acumulando demais"); ok = False

    # 1c) fechou a volta?
    fecho = float(np.linalg.norm(pts[-1]))
    print("     erro de fechamento da volta: %.4f m em %.2f m de perimetro"
          % (fecho, 2 * np.pi * R_esperado))
    if fecho > 0.05:
        print("     FALHOU"); ok = False

    print("")
    print("  2. O JACOBIANO BATE COM A DERIVADA NUMERICA DE f")
    # Pega erro de sinal e termo esquecido no jacobiano, o defeito clássico
    # de EKF e o mais difícil de ver por leitura.
    rng = np.random.default_rng(3)
    pior = 0.0
    for _ in range(200):
        xt = np.array([[rng.uniform(-5, 5)], [rng.uniform(-5, 5)],
                       [rng.uniform(-np.pi, np.pi)], [rng.uniform(-0.3, 0.3)],
                       [rng.uniform(-1.5, 1.5)], [rng.normal(0, 0.01)],
                       [rng.normal(0, 0.01)]])
        d = rng.uniform(-0.35, 0.35)
        F, _ = matrices(xt, dt, d)
        Fn = np.zeros((DIM_X, DIM_X))
        h = 1e-6
        for j in range(DIM_X):
            xp, xm = xt.copy(), xt.copy()
            xp[j, 0] += h; xm[j, 0] -= h
            Fn[:, j] = ((propaga(xp, dt, d) - propaga(xm, dt, d)) / (2 * h)).ravel()
        pior = max(pior, float(np.max(np.abs(F - Fn))))
    print("     maior discrepancia em 200 estados aleatorios: %.2e" % pior)
    if pior > 1e-6:
        print("     FALHOU: o jacobiano nao e a derivada de f"); ok = False
    else:
        print("     jacobiano analitico == derivada numerica              OK")

    print("")
    print("  3. O JACOBIANO DE h BATE COM A DERIVADA NUMERICA")
    pior = 0.0
    for _ in range(200):
        xt = np.array([[rng.uniform(-3, 3)], [rng.uniform(-3, 3)],
                       [rng.uniform(-1.0, 1.0)], [rng.uniform(-0.3, 0.3)],
                       [rng.uniform(0.1, 1.5)], [rng.normal(0, 0.01)],
                       [rng.normal(0, 0.01)]])
        d = rng.uniform(-0.35, 0.35)
        args = dict(delta=d, x_parede=6.0)
        base = observation(xt, (ENCODER, ULTRASSOM, IMU), **args)
        for i, (nome, H, _, _, _) in enumerate(base):
            Hn = np.zeros(DIM_X)
            h = 1e-7
            for j in range(DIM_X):
                xp, xm = xt.copy(), xt.copy()
                xp[j, 0] += h; xm[j, 0] -= h
                zp = observation(xp, (ENCODER, ULTRASSOM, IMU), **args)[i][4]
                zm = observation(xm, (ENCODER, ULTRASSOM, IMU), **args)[i][4]
                Hn[j] = (zp - zm) / (2 * h)
            pior = max(pior, float(np.max(np.abs(H - Hn))))
    print("     maior discrepancia nos 6 canais: %.2e" % pior)
    if pior > 1e-5:
        print("     FALHOU"); ok = False
    else:
        print("     jacobianos de h == derivada numerica                  OK")

    print("")
    print("  4. SIMETRIA E POSITIVIDADE DE Q")
    # A tolerância é relativa, e não absoluta.
    mau = 0
    pior_rel = 0.0
    for _ in range(100):
        xt = np.array([[0.0], [0.0], [rng.uniform(-np.pi, np.pi)],
                       [rng.uniform(-0.3, 0.3)], [rng.uniform(0, 1.5)],
                       [0.0], [0.0]])
        _, Q = matrices(xt, dt, rng.uniform(-0.35, 0.35))
        ev = np.linalg.eigvalsh(Q)
        tol = DIM_X * np.finfo(float).eps * max(abs(ev.max()), 1e-300)
        if not np.allclose(Q, Q.T) or ev.min() < -tol:
            mau += 1
        if ev.max() > 0:
            pior_rel = max(pior_rel, -ev.min() / ev.max())
    if mau:
        print("     FALHOU em %d de 100 estados" % mau); ok = False
    else:
        print("     Q simetrica e PSD em 100 estados (pior autovalor")
        print("     negativo relativo: %.1e, contra eps = %.1e)          OK"
              % (pior_rel, np.finfo(float).eps))

    print("")
    print("  5. POSTO DE OBSERVABILIDADE, e agora ele e LOCAL")
    print("     %-20s %12s %12s %14s"
          % ("configuracao", "reta+parede", "curva+parede", "curva s/ marco"))
    t_reta = _traj(lambda k: 0.0)
    t_curva = _traj(lambda k: np.deg2rad(12.0))
    postos = {}
    for rotulo, sens in configuracoes():
        r_reta, _ = observabilidade(sens, t_reta, x_parede=6.0)
        r_cp, _ = observabilidade(sens, t_curva, x_parede=6.0)
        r_curva, _ = observabilidade(sens, t_curva, x_parede=None)
        postos[rotulo] = (r_reta, r_cp, r_curva)
        print("     %-20s %10d/7 %10d/7 %12d/7" % (rotulo, r_reta, r_cp, r_curva))

    # Invariante 1: `y` não é observável em configuração nenhuma, em regime
    # nenhum.
    if max(max(v) for v in postos.values()) >= DIM_X:
        print("     FALHOU: posto pleno sem sensor lateral e impossivel")
        ok = False
    else:
        print("")
        print("     posto nunca chega a 7: `y` nao e observavel em config")
        print("     nenhuma, porque nao ha sensor que olhe para o lado    OK")

    # Invariante 2: acrescentar sensor nunca reduz o posto.
    monotona = True
    for a, b in [("encoder", "encoder+ultrassom"), ("encoder", "encoder+imu"),
                 ("ultrassom", "encoder+ultrassom"), ("ultrassom", "ultrassom+imu"),
                 ("imu", "encoder+imu"), ("imu", "ultrassom+imu"),
                 ("encoder+ultrassom", "os tres"), ("encoder+imu", "os tres"),
                 ("ultrassom+imu", "os tres")]:
        for j in range(3):
            if postos[b][j] < postos[a][j]:
                print("     FALHOU: %s observa menos que %s no regime %d"
                      % (b, a, j))
                monotona = False
    if monotona:
        print("     acrescentar sensor nunca reduz o posto, nos 3 regimes OK")
    else:
        ok = False

    print("")
    print("  5b. O GIROSCOPIO SO MEDE VELOCIDADE EM CURVA")
    # O canal do giroscópio z tem H[v] = tan(delta)/L.
    for g in (0.0, 5.0, 20.0):
        _, H, _, _, _ = [c for c in observation(
            _traj(lambda k: 0.0)[0][0], (IMU,), np.deg2rad(g))
            if c[0] == CH_GYRO_Z][0]
        print("     delta = %4.1f graus  ->  H[v] = %.4f" % (g, H[IV]))
    ganho_curva = postos["imu"][2] - postos["imu"][0]
    print("     posto de `imu` sozinha: %d em reta, %d em curva  (+%d)"
          % (postos["imu"][0], postos["imu"][2], ganho_curva))
    if ganho_curva <= 0:
        print("     FALHOU: era para a curva revelar a velocidade"); ok = False
    else:
        print("     a curva revela `v` para quem so tem IMU               OK")

    print("")
    print("  6. UM CICLO COMPLETO RODA E MANTEM P SIMETRICA E FINITA")
    for rotulo, sens in configuracoes():
        x0, P0 = initial(sens, pose0=(0.0, 0.0, 0.0), v0=1.0, x_parede=6.0)
        kf = KalmanFilter(dim_x=DIM_X, angle_indices=ANGLE_INDICES)
        kf.x, kf.P = x0, P0
        kf.f = lambda xx, uu: propaga(xx, dt, uu)
        for k in range(200):
            d = np.deg2rad(10.0) * np.sin(k / 30.0)
            kf.F, kf.Q = matrices(kf.x, dt, d)
            kf.predict(u=d)
            for nome, H, R, gate, zp in observation(kf.x, sens, d,
                                                    x_parede=6.0):
                kf.update_scalar(zp, H, R, gate=gate,
                                 gate_threshold=GATING_THRESHOLD_99,
                                 channel=nome, z_pred=zp,
                                 is_angle=(nome == CH_MAG))
        if not np.allclose(kf.P, kf.P.T, atol=1e-10):
            print("     FALHOU: P assimetrica em %s" % rotulo); ok = False
        if not np.all(np.isfinite(kf.x)) or not np.all(np.isfinite(kf.P)):
            print("     FALHOU: estado nao finito em %s" % rotulo); ok = False
    print("     as 7 rodam 200 ciclos com P simetrica e estado finito  OK")

    print("")
    print("  7. NIVEL 'plano' TRAVA A ARFAGEM E OS CANAIS DELA")
    x = np.zeros((DIM_X, 1)); x[IV, 0] = 1.0; x[ITHETA, 0] = 0.3
    xn = propaga(x, dt, 0.1, nivel="plano")
    n_plano = len(observation(x, (ENCODER, ULTRASSOM, IMU), 0.1,
                              nivel="plano", x_parede=6.0))
    n_arf = len(observation(x, (ENCODER, ULTRASSOM, IMU), 0.1,
                            nivel="arfagem", x_parede=6.0))
    if abs(float(xn[ITHETA, 0])) > 1e-15 or n_plano != 4 or n_arf != 6:
        print("     FALHOU: theta=%.3e, canais plano=%d arfagem=%d"
              % (xn[ITHETA, 0], n_plano, n_arf)); ok = False
    else:
        print("     theta travado em zero; 4 canais no plano, 6 na arfagem OK")

    print("")
    print("  8. SEM PAREDE CONHECIDA, O ULTRASSOM NAO ENTRA")
    com = [c[0] for c in observation(x, (ULTRASSOM,), 0.0, x_parede=6.0)]
    sem = [c[0] for c in observation(x, (ULTRASSOM,), 0.0, x_parede=None)]
    if com != [CH_ULTRASSOM] or sem != []:
        print("     FALHOU: com parede %s, sem parede %s" % (com, sem)); ok = False
    else:
        print("     distancia a objeto de posicao desconhecida nao informa OK")

    print("")
    print("  8b. O CANAL DO ULTRASSOM: BRACO E CONE")
    # O canal do ultrassom: braço ell_s e feixe em cone.
    x8 = np.zeros((DIM_X, 1))
    x8[IV, 0] = 1.0

    def _z(psi_g, **kw):
        x8[IPSI, 0] = np.deg2rad(psi_g)
        c = observation(x8, (ULTRASSOM,), 0.0, x_parede=2.0, **kw)[0]
        return c[4], c[1]

    # (a) com psi = 0 o braço entra inteiro
    z0, _ = _z(0.0)
    if abs(z0 - (2.0 - ELL_S)) > 1e-12:
        print("     FALHOU: psi=0 previu %.4f, esperado %.4f"
              % (z0, 2.0 - ELL_S)); ok = False
    else:
        print("     psi=0: preve %.3f m, e nao 2,000 OK" % z0)

    # (b) dentro da abertura a leitura quase não muda com psi
    z10, _ = _z(10.0)
    esperado = 2.0 - ELL_S * np.cos(np.deg2rad(10.0))
    fino = 2.0 / np.cos(np.deg2rad(10.0)) - ELL_S
    if abs(z10 - esperado) > 1e-12:
        print("     FALHOU: psi=10 previu %.4f, cone preve %.4f"
              % (z10, esperado)); ok = False
    else:
        print("     psi=10 (dentro do cone): %+.1f mm contra psi=0; o raio "
              "fino daria %+.1f mm OK" % (1000 * (z10 - z0),
                                          1000 * (fino - z0)))

    # (c) FORA da abertura, o eco vem da borda do cone
    pg = 25.0
    z25, _ = _z(pg)
    p = np.deg2rad(pg)
    d_s = 2.0 - ELL_S * np.cos(p)
    esperado = d_s / np.cos(p - ALFA_US)
    if abs(z25 - esperado) > 1e-12:
        print("     FALHOU: psi=25 previu %.4f, esperado %.4f"
              % (z25, esperado)); ok = False
    else:
        print("     psi=25 (fora do cone): eco pela BORDA, %.4f m OK" % z25)

    # (d) o jacobiano bate com a diferenca finita, inclusive fora do cone
    jac_ok = True
    for pg in (0.0, 5.0, 14.0, 16.0, 25.0, -18.0):
        h = 1e-6
        za, H = _z(pg)
        zb, _ = _z(pg + np.rad2deg(h))
        num = (zb - za) / h
        if abs(num - H[IPSI]) > 1e-4:
            print("     FALHOU: dH/dpsi a %.0f graus: analitico %.6f, "
                  "numerico %.6f" % (pg, H[IPSI], num)); jac_ok = False
        # e em x
        x8[IPSI, 0] = np.deg2rad(pg)
        c1 = observation(x8, (ULTRASSOM,), 0.0, x_parede=2.0)[0]
        x8[IX, 0] = 1e-6
        c2 = observation(x8, (ULTRASSOM,), 0.0, x_parede=2.0)[0]
        x8[IX, 0] = 0.0
        if abs((c2[4] - c1[4]) / 1e-6 - c1[1][IX]) > 1e-4:
            print("     FALHOU: dH/dx a %.0f graus" % pg); jac_ok = False
    ok = ok and jac_ok
    print("     jacobiano confere com diferenca finita em 6 angulos, "
          "dentro e fora do cone: %s" % ("OK" if jac_ok else "FALHOU"))

    # (e) o jacobiano é contínuo na transição. Teste de limite, e não de
    # tolerância fixa: o salto tem de encolher junto com o passo.
    saltos = []
    for eps in (1e-3, 1e-4, 1e-5):
        _, Ha = _z(np.rad2deg(ALFA_US) - eps)
        _, Hb = _z(np.rad2deg(ALFA_US) + eps)
        saltos.append(max(abs(Ha[IPSI] - Hb[IPSI]), abs(Ha[IX] - Hb[IX])))
    cont_ok = (saltos[1] < saltos[0] / 5.0 and saltos[2] < saltos[1] / 5.0)
    ok = ok and cont_ok
    print("     jacobiano CONTINUO na abertura (alfa = %.1f graus): salto "
          "%.1e -> %.1e -> %.1e ao refinar o passo %s"
          % (np.rad2deg(ALFA_US), saltos[0], saltos[1], saltos[2],
             "OK" if cont_ok else "FALHOU"))

    # (f) a sensibilidade a psi é pequena dentro do cone
    _, H10 = _z(10.0)
    antigo = 2.0 * np.sin(np.deg2rad(10.0)) / np.cos(np.deg2rad(10.0)) ** 2
    encolheu = abs(H10[IPSI]) < 0.35 * antigo
    ok = ok and encolheu
    print("     dH/dpsi a 10 graus: %.3f contra %.3f do raio fino "
          "(%.1fx menor) %s" % (H10[IPSI], antigo, antigo / abs(H10[IPSI]),
                                "OK" if encolheu else "FALHOU"))

    print("")
    print("  9. O CANAL DO MAGNETOMETRO ATRAVESSA A DESCONTINUIDADE")
    # psi estimado em +179 graus, medida em -179: sem wrap a inovacao e -358.
    kf = KalmanFilter(dim_x=DIM_X, angle_indices=ANGLE_INDICES)
    kf.x, kf.P = initial((IMU,), pose0=(0.0, 0.0, np.deg2rad(179.0)))
    _, H, R, _, zp = observation(kf.x, (IMU,), 0.0)[0]
    kf.update_scalar(np.deg2rad(-179.0), H, R, is_angle=True, z_pred=zp,
                     channel=CH_MAG)
    psi_fim = np.degrees(wrap(float(kf.x[IPSI, 0])))
    print("     psi 179 graus + medida -179 -> %.1f graus" % psi_fim)
    if not (abs(psi_fim) > 170.0):
        print("     FALHOU: o wrap nao funcionou, psi foi para perto de zero")
        ok = False
    else:
        print("     ficou na vizinhanca de 180, e nao saltou para 0        OK")

    print("")
    print("=" * 74)
    print("=== RESULTADO: %s ===" % ("OK" if ok else "FALHOU"))
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _autoteste() else 1)
