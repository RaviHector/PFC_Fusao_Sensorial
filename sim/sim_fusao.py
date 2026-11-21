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
Simulador do filtro fundido — EKF sobre a cinematica de Ackerman, 7 estados,
com verdade conhecida.
"""
import argparse
import sys

import numpy as np

if __package__ in (None, ""):
    import os as _os

    _d = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _d not in sys.path:
        sys.path.insert(0, _d)

from kalman import filtros
from kalman.config import (ALFA_US, DT_LOOP, ELL_S, ENTRE_EIXOS_L,
                           GATE_ESCAPE_FATOR, GATE_ESCAPE_N,
                           GATING_THRESHOLD_99,
                           SIGMA_ACC, SIGMA_ACC_MEAS_MOVEL,
                           SIGMA_ACC_MEAS_PARADO, SIGMA_ACC_TILT_MOVEL,
                           SIGMA_ACC_TILT_PARADO, SIGMA_BIAS, SIGMA_ENC,
                           SIGMA_ENC_OPERACAO,
                           SIGMA_GRADE, SIGMA_GYRO_MOVEL, SIGMA_GYRO_PARADO,
                           SIGMA_MAG, SIGMA_STEER, SIGMA_US, US_MAX_RANGE)
from kalman.core import KalmanFilter
from kalman.filtros import fusao, fusao_giro
from kalman.filtros.fusao import (ANGLE_INDICES, CH_ENCODER, CH_GYRO_Y,
                                  CH_GYRO_Z, CH_MAG, CH_TILT, CH_ULTRASSOM,
                                  DIM_X, IBGY, IBGZ, IPSI, ITHETA, IV, IX, IY)
from kalman.movimento import EstadoMovimento
from sim.base import (coverage_verdict, innovation_whiteness_verdict_mc,
                      nees_verdict_mc, nis_verdict)

# Ecos espurios do ultrassom: 0,4 a 1,2% das leituras, afastadas da
# mediana por 30 a 300 mm.
TAXA_ESPURIOS = 0.008
ESPURIO_MIN, ESPURIO_MAX = 0.030, 0.300

V_MOVIMENTO_VERDADE = 0.05  # [m/s] limiar da VERDADE, nao do detector

# ruído medido dos três eixos do acelerômetro, parado e em cruzeiro
SIGMA_EIXOS_PARADO = (0.177, 0.194, 0.408)
SIGMA_EIXOS_MOVEL = (1.069, 2.971, 3.146)
G_MEDIDO = 9.911  # media de az medida, com o veiculo nivelado

VIES_GYRO = np.deg2rad(0.3)  # vies residual apos a calibracao do arranque

N_RUNS_PADRAO = 200

FORMULACAO_PADRAO = "variante"   # fusao_giro.py; "vigente" = fusao.py

SUSPENDE_PARADO = False
SUSPENDE_GIRO = False

# ZUPT + ZARU com o carro parado, na variante
ZUPT = True

# sem o canal do magnetometro, como o filtro usado nos dados reais
SEM_MAG = False

# derrapadas simuladas no encoder
DERRAPADA_ENCODER = None
GATE_ENCODER = False
N_ESCAPE_ENCODER = 12


def nivel_padrao():
    """O nivel da formulacao em uso: plano na variante, arfagem no vigente."""
    return fusao_giro.NIVEL if FORMULACAO_PADRAO == "variante" else "arfagem"

# Uma realizacao conta como DIVERGIDA quando o NEES medio dela passa deste
# limiar, que e cerca de 14 vezes a dimensao do estado.
LIMIAR_DIVERGENCIA = 100.0
# Acima desta fracao de realizacoes divergidas a configuracao e declarada
# DIVERGENTE, e a divergencia passa a ser o resultado sobre ela.
FRACAO_DIVERGENCIA_TOLERADA = 0.05


# Geracao de verdade
def gera_verdade(perfil, n_passos, dt, rng=None, sigma_steer=0.0,
                 sigma_acc=0.0, sigma_grade=0.0, nivel="arfagem",
                 substeps=10):
    """Integra a cinematica VERDADEIRA a partir de um perfil de comandos."""
    h = dt / substeps
    x = np.zeros((DIM_X, 1))
    v0, d0, th0 = perfil(0)
    x[IV, 0] = v0
    x[ITHETA, 0] = 0.0 if nivel == "plano" else th0
    x[IBGZ, 0] = VIES_GYRO
    x[IBGY, 0] = VIES_GYRO

    estocastico = rng is not None and sigma_acc > 0
    out = np.zeros((n_passos, DIM_X))
    deltas = np.zeros(n_passos)
    deltas_reais = np.zeros(n_passos)

    for k in range(n_passos):
        out[k] = x.ravel()
        v_alvo, delta_cmd, th_alvo = perfil(k)
        deltas[k] = delta_cmd

        delta_real = delta_cmd
        if sigma_steer > 0 and rng is not None:
            delta_real += rng.normal(0.0, sigma_steer)
        deltas_reais[k] = delta_real

        w_v = 0.0
        if estocastico:
            w_v = rng.normal(0.0, sigma_acc * dt)
            x[IV, 0] += 0.5 * w_v
        else:
            x[IV, 0] = v_alvo

        if nivel != "plano":
            if sigma_grade > 0 and rng is not None:
                x[ITHETA, 0] += rng.normal(0.0, sigma_grade * np.sqrt(dt))
            else:
                x[ITHETA, 0] = th_alvo
        if estocastico:
            x[IBGZ, 0] += rng.normal(0.0, SIGMA_BIAS * np.sqrt(dt))
            x[IBGY, 0] += rng.normal(0.0, SIGMA_BIAS * np.sqrt(dt))

        for _ in range(substeps):
            x = fusao.propaga(x, h, delta_real, nivel=nivel)

        if estocastico:
            x[IV, 0] += 0.5 * w_v

    return out, deltas, deltas_reais


# Os perfis de cada cenario
X_PAREDE = 6.0
D_LARGADA = 4.75  # m da parede, como nos ensaios reais

# trecho parado na largada dos cenarios realistas, como nos ensaios
T_PARADO_LARGADA = 2.0  # [s]


def perfil_frenagem(dt, v_cruzeiro=1.043, t_arranque=1.0, t_corte=4.0,
                    desacel=1.043, t_parado=T_PARADO_LARGADA):
    """Parado, arrancada, cruzeiro, corte do motor, costeio ate parar."""
    def f(k):
        t = k * dt - t_parado
        if t < 0.0:
            v = 0.0
        elif t < t_arranque:
            v = v_cruzeiro * (t / t_arranque)
        elif t < t_corte:
            v = v_cruzeiro
        else:
            v = max(0.0, v_cruzeiro - desacel * (t - t_corte))
        return v, 0.0, 0.0
    return f


def perfil_circulo(dt, v=0.8, delta=np.deg2rad(15.0), t_arranque=1.0,
                   t_parado=T_PARADO_LARGADA):
    def f(k):
        t = k * dt - t_parado
        return (v * min(1.0, max(t, 0.0) / t_arranque), delta, 0.0)
    return f


def perfil_corredor(dt, v=0.8, t_arranque=1.0, t_parado=T_PARADO_LARGADA):
    """Reta, curva a direita, reta, curva a esquerda, reta."""
    def f(k):
        t = k * dt - t_parado
        vv = v * min(1.0, max(t, 0.0) / t_arranque)
        if t < 4.0:
            d = 0.0
        elif t < 6.0:
            d = np.deg2rad(18.0)
        elif t < 10.0:
            d = 0.0
        elif t < 12.0:
            d = -np.deg2rad(18.0)
        else:
            d = 0.0
        return vv, d, 0.0
    return f


def perfil_rampa(dt, v=0.8, t_arranque=1.0, grau=8.0,
                 t_parado=T_PARADO_LARGADA):
    """Plano, subida, plano. Valida a arfagem — SO EM SIMULACAO."""
    def f(k):
        t = k * dt - t_parado
        vv = v * min(1.0, max(t, 0.0) / t_arranque)
        if t < 3.0:
            th = 0.0
        elif t < 4.0:
            th = np.deg2rad(grau) * (t - 3.0)
        elif t < 9.0:
            th = np.deg2rad(grau)
        elif t < 10.0:
            th = np.deg2rad(grau) * (10.0 - t)
        else:
            th = 0.0
        return vv, 0.0, th
    return f


# Uma realizacao
def _roda(x_true, deltas, sensores, dt, rng, x_parede=None, espurios=False,
          nivel="arfagem", chavear=True, deltas_reais=None, escapatoria=None,
          formulacao=None, sigma_enc_verdade=SIGMA_ENC):
    """Roda o filtro sobre uma verdade dada."""
    n_passos = len(x_true)
    formulacao = formulacao or FORMULACAO_PADRAO
    variante = formulacao == "variante" and fusao_giro.usa_giro(sensores)
    obs = fusao_giro.observation if formulacao == "variante" else fusao.observation
    # a escapatoria do gate e so do filtro fundido
    if escapatoria is None:
        escapatoria = (fusao_giro.usa_escapatoria(sensores)
                       if formulacao == "variante" else True)
    kf = KalmanFilter(dim_x=DIM_X, angle_indices=ANGLE_INDICES,
                      gate_escape_after=(GATE_ESCAPE_N if escapatoria
                                         else None),
                      gate_escape_factor=GATE_ESCAPE_FATOR)
    x0, P0 = fusao.initial(sensores,
                           pose0=(x_true[0, IX], x_true[0, IY], x_true[0, IPSI]),
                           v0=x_true[0, IV], nivel=nivel, x_parede=x_parede)
    kf.x = (x_true[0].reshape(-1, 1)
            + np.linalg.cholesky(P0) @ rng.standard_normal((DIM_X, 1)))
    kf.P = P0.copy()
    kf.f = None  # definido a cada passo, porque depende de delta

    det = EstadoMovimento()
    x_est = np.zeros((n_passos, DIM_X))
    P_hist = np.zeros((n_passos, DIM_X, DIM_X))
    nis_por_canal = {}
    n_rejeitados = 0
    n_us_validos = 0

    em_mov_ant = True
    z_enc_ant = None
    # derrapadas (desligadas por padrao): um rng PROPRIO, para que ligar ou
    # desligar o gate nao mude o sorteio dos outros ruidos
    excesso_enc = (None if DERRAPADA_ENCODER is None
                   else DERRAPADA_ENCODER(np.random.default_rng(
                       int(rng.integers(0, 2**31))), x_true, dt))
    rej_enc_seguidas = 0
    n_rej_enc = 0
    for k in range(n_passos):
        d_cmd = float(deltas[k])
        if k > 0:
            if variante:
                xa = x_true[k - 1]
                d_ant = float(deltas[k - 1] if deltas_reais is None
                              else deltas_reais[k - 1])
                em_ant_real = (abs(xa[IV]) > V_MOVIMENTO_VERDADE) if chavear else True
                sg_in = SIGMA_GYRO_MOVEL if em_ant_real else SIGMA_GYRO_PARADO
                gz_in = ((xa[IV] / ENTRE_EIXOS_L) * np.tan(d_ant) + xa[IBGZ]
                         + rng.normal(0.0, sg_in))
                sa_in = SIGMA_ACC_MEAS_MOVEL if em_ant_real else SIGMA_ACC_MEAS_PARADO
                ax_in = ((x_true[k][IV] - xa[IV]) / dt
                         - G_MEDIDO * np.sin(xa[ITHETA]) + rng.normal(0.0, sa_in))
                kf.F, kf.Q, kf.f = fusao_giro.passo(
                    kf.x, dt, float(deltas[k - 1]), sensores, gz=float(gz_in),
                    em_movimento=em_mov_ant, nivel=nivel, ax=float(ax_in),
                    parado=(SUSPENDE_PARADO and fusao_giro.esta_parado(
                        sensores, em_mov_ant,
                        None if z_enc_ant is None else
                        (0.0 if abs(z_enc_ant) < 3.0 * sigma_enc_verdade else z_enc_ant))),
                    suspende_giro=SUSPENDE_GIRO)
            else:
                kf.F, kf.Q = fusao.matrices(kf.x, dt, d_cmd, nivel=nivel)
                kf.f = (lambda dd: (lambda xx, uu: fusao.propaga(xx, dt, dd,
                                                                 nivel=nivel)))(d_cmd)
            kf.predict()

        xt = x_true[k]
        v_k = xt[IV]
        em_mov_real = (abs(v_k) > V_MOVIMENTO_VERDADE) if chavear else True
        sig = SIGMA_EIXOS_MOVEL if em_mov_real else SIGMA_EIXOS_PARADO

        # --- acelerometro nos tres eixos, para o detector
        z_ax = -G_MEDIDO * np.sin(xt[ITHETA]) + rng.normal(0.0, sig[0])
        z_ay = rng.normal(0.0, sig[1])
        z_az = G_MEDIDO * np.cos(xt[ITHETA]) + rng.normal(0.0, sig[2])
        norma = float(np.sqrt(z_ax**2 + z_ay**2 + z_az**2))
        em_mov = det.atualiza(norma) if chavear else True
        em_mov_ant = em_mov

        # A inclinacao MEDIDA sai da direcao da gravidade, como no veiculo.
        sig_tilt = SIGMA_ACC_TILT_MOVEL if em_mov_real else SIGMA_ACC_TILT_PARADO
        z_tilt = xt[ITHETA] + rng.normal(0.0, sig_tilt)

        sig_g = SIGMA_GYRO_MOVEL if em_mov_real else SIGMA_GYRO_PARADO
        # o giro mede a taxa REAL de guinada no corpo, que a verdade produziu
        d_real = d_cmd if deltas_reais is None else float(deltas_reais[k])
        psi_ponto_real = (xt[IV] / ENTRE_EIXOS_L) * np.tan(d_real)
        z_gz = psi_ponto_real + xt[IBGZ] + rng.normal(0.0, sig_g)
        z_gy = xt[IBGY] + rng.normal(0.0, sig_g)

        z_enc = xt[IV] + rng.normal(0.0, sigma_enc_verdade)
        if excesso_enc is not None:
            z_enc += float(excesso_enc[k])
        z_enc_ant = z_enc
        z_mag = None if SEM_MAG else xt[IPSI] + rng.normal(0.0, SIGMA_MAG)

        # --- ultrassom
        z_us = None
        if x_parede is not None:
            cp = np.cos(xt[IPSI])
            if cp > 1e-3:
                # sensor ELL_S a frente do eixo traseiro, feixe em cone
                d_s = (x_parede - xt[IX]) - ELL_S * cp
                phi = max(abs(xt[IPSI]) - ALFA_US, 0.0)
                d_real = d_s / max(np.cos(phi), 1e-3)
                # sem medida nova, so predicao
                if 0.0 < d_real <= US_MAX_RANGE:
                    z_us = d_real + rng.normal(0.0, SIGMA_US)
                    if espurios and rng.random() < TAXA_ESPURIOS:
                        z_us += (rng.uniform(ESPURIO_MIN, ESPURIO_MAX)
                                 * rng.choice([-1.0, 1.0]))
                    n_us_validos += 1

        z_por_canal = {CH_ENCODER: z_enc, CH_MAG: z_mag, CH_GYRO_Z: z_gz,
                       CH_GYRO_Y: z_gy, CH_TILT: z_tilt, CH_ULTRASSOM: z_us}

        # CADA CANAL E RELINEARIZADO NO ESTADO ATUAL, e nao no estado que havia
        # antes da primeira correcao do ciclo.
        ordem = [c[0] for c in obs(
            kf.x, sensores, d_cmd, em_movimento=em_mov, nivel=nivel,
            x_parede=x_parede)]
        for nome in ordem:
            z = z_por_canal.get(nome)
            if z is None:
                continue
            atual = [c for c in obs(
                kf.x, sensores, d_cmd, em_movimento=em_mov, nivel=nivel,
                x_parede=x_parede) if c[0] == nome]
            if not atual:
                continue
            _, H, R, usa_gate, z_pred = atual[0]
            if GATE_ENCODER and nome == CH_ENCODER:
                usa_gate = rej_enc_seguidas < N_ESCAPE_ENCODER
            aceito, nis = kf.update_scalar(
                z, H, R, is_angle=(nome == CH_MAG), gate=usa_gate,
                gate_threshold=GATING_THRESHOLD_99, channel=nome, z_pred=z_pred)
            nis_por_canal.setdefault(nome, []).append(nis)
            if nome == CH_ENCODER:
                rej_enc_seguidas = 0 if aceito else rej_enc_seguidas + 1
                if not aceito:
                    n_rej_enc += 1
            elif not aceito:
                n_rejeitados += 1

        # ZUPT + ZARU com o carro parado, como no `bancada.analise_fusao.rodar`
        if ZUPT and variante and fusao_giro.esta_parado(
                sensores, em_mov, 0.0 if abs(z_enc) < 3.0 * sigma_enc_verdade else z_enc):
            for nome, zz in ((fusao_giro.CH_ZUPT, 0.0), (fusao_giro.CH_ZARU, z_gz)):
                atual = [c for c in fusao_giro.canais_parado(kf.x, sensores) if c[0] == nome]
                if atual:
                    _, H, R, _, z_pred = atual[0]
                    kf.update_scalar(zz, H, R, gate=False, channel=nome, z_pred=z_pred)

        x_est[k] = kf.x.ravel()
        P_hist[k] = kf.P

    return dict(x_est=x_est, P_hist=P_hist, nis=nis_por_canal,
                innov=dict(kf.innov_history), n_rejeitados=n_rejeitados,
                n_us_validos=n_us_validos, n_rej_enc=n_rej_enc, kf=kf)


def _posto(sens, traj_ref, x_parede=None, nivel="arfagem"):
    """O posto da configuracao na formulacao em uso."""
    if FORMULACAO_PADRAO == "variante":
        tr = [(x, dt, d, em, float(x[IV, 0]) / ENTRE_EIXOS_L * np.tan(d))
              for x, dt, d, em in traj_ref]
        return fusao_giro.observabilidade(sens, tr, nivel=nivel,
                                          x_parede=x_parede)
    return fusao.observabilidade(sens, traj_ref, x_parede=x_parede,
                                 nivel=nivel)


def _nees(x_true, x_est, P_hist, burn_in):
    """NEES medio depois do transitorio, com wrap nos estados de angulo."""
    erros = x_true - x_est
    for i in ANGLE_INDICES:
        erros[:, i] = np.arctan2(np.sin(erros[:, i]), np.cos(erros[:, i]))
    vals = []
    for k in range(burn_in, len(erros)):
        e = erros[k].reshape(-1, 1)
        P = P_hist[k] + np.eye(DIM_X) * 1e-12
        try:
            vals.append(float((e.T @ np.linalg.solve(P, e)).item()))
        except np.linalg.LinAlgError:
            continue
    return float(np.mean(vals)) if vals else float("nan")


def _monte_carlo(sensores, perfil, n_passos, dt, n_runs, semente,
                 x_parede=None, espurios=False, nivel="arfagem",
                 formulacao=False, chavear=True, escapatoria=None):
    # `formulacao` aqui e o CENARIO --formulacao (verdade sorteada do
    # processo), e nao a escolha vigente/variante, que vem de
    # FORMULACAO_PADRAO.
    """Roda `n_runs` realizacoes e agrega. Uma semente por realizacao."""
    burn_in = int(1.0 / dt)
    nees_list, innov_list = [], []
    nis_acc, err_final, cobertura = {}, [], []
    # erro longitudinal: na frenagem a metrica e a distancia a parede (eixo x)
    err_long = []
    rej, us_val, total_us = 0, 0, 0

    for r in range(n_runs):
        rng = np.random.default_rng(semente + r)
        if formulacao:
            x_true, deltas, d_reais = gera_verdade(
                perfil, n_passos, dt, rng=rng, sigma_steer=SIGMA_STEER,
                sigma_acc=SIGMA_ACC, sigma_grade=SIGMA_GRADE, nivel=nivel)
        else:
            x_true, deltas, d_reais = gera_verdade(perfil, n_passos, dt,
                                                   nivel=nivel)

        # no cenario --formulacao, o encoder recebe o ruido que o R declara
        s_enc = (SIGMA_ENC_OPERACAO if (formulacao and FORMULACAO_PADRAO == "variante")
                 else SIGMA_ENC)
        out = _roda(x_true, deltas, sensores, dt, rng, x_parede=x_parede,
                    espurios=espurios, nivel=nivel, chavear=chavear,
                    deltas_reais=d_reais, escapatoria=escapatoria,
                    sigma_enc_verdade=s_enc)

        nees_list.append(_nees(x_true, out["x_est"], out["P_hist"], burn_in))
        innov_list.append(out["innov"])
        for c, v in out["nis"].items():
            nis_acc.setdefault(c, []).extend(v[burn_in:])
        rej += out["n_rejeitados"]
        us_val += out["n_us_validos"]
        total_us += n_passos

        e = out["x_est"][-1] - x_true[-1]
        err_final.append(float(np.hypot(e[IX], e[IY])))
        err_long.append(float(abs(e[IX])))
        var_xy = out["P_hist"][-1][IX, IX] + out["P_hist"][-1][IY, IY]
        cobertura.append(err_final[-1] <= 3.0 * np.sqrt(max(var_xy, 1e-18)))

    return dict(nees=np.array(nees_list), nis=nis_acc, innov=innov_list,
                err_final=np.array(err_final), err_long=np.array(err_long),
                cobertura=float(np.mean(cobertura)),
                taxa_rej=rej / max(us_val, 1),
                taxa_us=us_val / max(total_us, 1))


# Cenario 1: formulacao
def dim_efetiva(nivel):
    """Quantos estados o NEES deve esperar."""
    return DIM_X if nivel != "plano" else DIM_X - 1


def run_formulacao(n_runs=N_RUNS_PADRAO, dt=DT_LOOP, duracao=8.0,
                   nivel=None):
    nivel = nivel or nivel_padrao()
    n = int(duracao / dt)
    # A PAREDE FICA PERTO DE PROPOSITO.
    x_parede = 1.5
    dim_esp = dim_efetiva(nivel)
    # O perfil mantem v = 0,8 m/s DESDE k=0, sem arrancada.
    perfil = lambda k: (0.8, np.deg2rad(10.0), 0.0)

    print("=" * 78)
    print("FORMULACAO — verdade sorteada do processo que o filtro assume")
    print("   nivel: %s" % nivel)
    print("=" * 78)
    print("")
    print("  Sem discrepancia de modelo, o NEES tem de bater em %d." % dim_esp)
    print("  %d realizacoes, %.0f s cada, parede a %.1f m, v = 0,8 m/s."
          % (n_runs, duracao, x_parede))
    print("")
    print("  DUAS ESCOLHAS DE METODO:")
    print("")
    print("  1. A DIMENSAO ESPERADA E %d, E NAO %d. No nivel `plano` o `theta`"
          % (dim_esp, DIM_X))
    print("     fica travado, com variancia 1e-12 e erro zero: nao e grau de")
    print("     liberdade. Comparar com %d reprovaria um filtro correto." % DIM_X)
    print("")
    print("  2. O VEREDICTO USA A MEDIANA entre realizacoes, e nao a media. A")
    print("     distribuicao do NEES por realizacao e assimetrica a direita")
    print("     (medido: mediana 7,80 contra media 9,13, com maximo 40 e")
    print("     nenhuma acima de 100). A media persegue a cauda; a mediana diz")
    print("     como o filtro se comporta na corrida tipica. As duas saem")
    print("     abaixo, porque a diferenca entre elas tambem e informacao.")
    print("")
    print("  CHAVEAMENTO DO R DESLIGADO AQUI, de proposito. O detector de")
    print("  movimento erra nas transicoes, e um erro dele faz o filtro usar um")
    print("  R diferente do ruido injetado. Isso e discrepancia de MODELO, e")
    print("  este teste existe para isolar a FORMULACAO dela. O detector e")
    print("  exercitado nos cenarios realistas, onde o erro dele conta.")
    print("")
    print("  UMA CONFIGURACAO PODE DIVERGIR, e divergencia nao e reprovacao do")
    print("  filtro: e resultado sobre a configuracao. Uma realizacao conta")
    print("  como divergente quando o NEES passa de %d, cerca de %dx a" % (LIMIAR_DIVERGENCIA, LIMIAR_DIVERGENCIA // dim_esp))
    print("  dimensao. A mediana e calculada sobre as que NAO divergiram,")
    print("  porque NEES de filtro divergido nao mede consistencia de nada.")
    print("")
    print("  %-19s %7s %7s %8s %12s %8s"
          % ("configuracao", "posto", "diverg", "mediana", "IC 95% med",
             "veredicto"))
    print("  " + "-" * 70)

    traj_ref = [(np.array([[0.], [0.], [0.], [0.], [0.8], [0.], [0.]]), dt,
                 np.deg2rad(10.0), True)] * 60
    ok = True
    guardados = {}
    divergentes = []
    for rotulo, sens in fusao.configuracoes():
        r = _monte_carlo(sens, perfil, n, dt, n_runs, 1000, x_parede=x_parede,
                         formulacao=True, chavear=False, nivel=nivel)
        guardados[rotulo] = r
        posto, dim = _posto(sens, traj_ref, x_parede=x_parede, nivel=nivel)
        nees = np.asarray(r["nees"], dtype=float)
        nees = nees[np.isfinite(nees)]
        frac_div = float(np.mean(nees > LIMIAR_DIVERGENCIA))
        sadias = nees[nees <= LIMIAR_DIVERGENCIA]
        if len(sadias) < 5:
            print("  %-19s %5d/%d %6.0f%% %8s %12s %8s"
                  % (rotulo, posto, dim, 100 * frac_div, "-", "-", "DIVERGE"))
            divergentes.append(rotulo)
            continue
        med = float(np.median(sadias))
        # IC da mediana por bootstrap: a distribuicao e assimetrica, entao a
        # formula normal do erro padrao da media nao serve aqui.
        bs = np.random.default_rng(7).choice(sadias, size=(2000, len(sadias)))
        lo, hi = np.percentile(np.median(bs, axis=1), [2.5, 97.5])
        # O CRITERIO E ASSIMETRICO, e de proposito.
        if frac_div > FRACAO_DIVERGENCIA_TOLERADA:
            divergentes.append(rotulo)
            marca = "DIVERGE"
        elif lo <= dim_esp <= hi:
            marca = "OK"
        elif hi < dim_esp:
            marca = "CONSERV"
        else:
            marca = "FALHOU"
            ok = False
        print("  %-19s %5d/%d %6.0f%% %8.2f [%5.2f,%5.2f] %8s"
              % (rotulo, posto, dim, 100 * frac_div, med, lo, hi, marca))

    print("")
    print("  NIS POR CANAL (esperado 1,0 — nao depende de verdade nenhuma)")
    print("  " + "-" * 64)
    r = guardados["os tres"]
    for c in sorted(r["nis"]):
        m = float(np.mean(r["nis"][c]))
        nis_ok, _ = nis_verdict(m, label=c)
        print("  %-19s %9.3f   %s" % (c, m, "OK" if nis_ok else "FALHOU"))
        if not nis_ok:
            ok = False

    print("")
    print("  BRANCURA DA INOVACAO (r_1 medio, banda +-1,96/raiz(M N))")
    print("  " + "-" * 64)
    for c in sorted(r["nis"]):
        serie = [d[c] for d in r["innov"] if c in d and len(d[c]) > 20]
        if not serie:
            continue
        br_ok, r1, msg = innovation_whiteness_verdict_mc(serie, label=c)
        print("  %s" % msg)
        if not br_ok:
            ok = False

    if divergentes:
        print("")
        print("  CONFIGURACOES DIVERGENTES: %s" % ", ".join(divergentes))
        print("")
        print("  Isto e RESULTADO, e nao falha do filtro. O ultrassom sozinho")
        print("  mede (x_parede - x)/cos(psi): uma unica distancia escalar que")
        print("  depende da orientacao, sem nenhum sensor que informe a")
        print("  orientacao. Medido em 200 realizacoes: erro final de psi com")
        print("  mediana de 34,5 graus e maximo de 177,8, e a correlacao entre")
        print("  o NEES e o erro de psi e de 0,64.")
        print("")
        print("  A CONCLUSAO E MAIS FORTE QUE A DA TABELA DE POSTO. O posto diz")
        print("  que parte do estado nao e observavel; isto diz que a parte nao")
        print("  observavel DESTROI a observavel, porque o modelo de medida e")
        print("  nao linear justamente na direcao cega.")

    print("")
    print("  RESULTADO DO PORTAO: %s" % ("APROVADO" if ok else "REPROVADO"))
    return ok


# Cenario 2: reta
def run_reta(n_runs=N_RUNS_PADRAO, dt=DT_LOOP):
    duracao = 7.0 + T_PARADO_LARGADA
    n = int(duracao / dt)
    print("")
    print("=" * 78)
    print("RETA — frenagem contra parede, delta = 0")
    print("=" * 78)
    print("")
    print("  E O TESTE DE REGRESSAO contra a formulacao antiga: com delta = 0")
    print("  a cinematica de Ackerman se reduz ao caso 1D, e o erro de parada")
    print("  tem de continuar na mesma ordem de grandeza.")
    print("")
    print("  largada a %.2f m da parede, alcance de confianca do US: %.2f m"
          % (D_LARGADA, US_MAX_RANGE))
    print("  ecos espurios injetados: %.1f%% (medido: 0,4 a 1,2%%)"
          % (100 * TAXA_ESPURIOS))
    print("")
    print("  %-19s %10s %10s %10s %8s %7s"
          % ("configuracao", "mediana", "media", "p90", "cobert.", "rejeic."))
    print("  " + "-" * 70)

    perfil = perfil_frenagem(dt)
    nivel = nivel_padrao()
    # largada: o veiculo comeca em x tal que a parede fique a D_LARGADA
    x0 = X_PAREDE - D_LARGADA
    guarda = {}
    for rotulo, sens in fusao.configuracoes():
        def p(k, _p=perfil):
            return _p(k)
        r = _monte_carlo(sens, p, n, dt, n_runs, 2000, x_parede=X_PAREDE,
                         espurios=True, nivel=nivel)
        e = r["err_final"]
        guarda[rotulo] = (sens, e)
        el = r["err_long"]
        print("  %-19s %8.1f mm %7.1f mm %7.1f mm %7.0f%% %6.2f%%   |ex| med %6.1f mm p90 %6.1f mm"
              % (rotulo, 1000 * float(np.median(e)), 1000 * float(np.mean(e)),
                 1000 * float(np.percentile(e, 90)),
                 100 * r["cobertura"], 100 * r["taxa_rej"],
                 1000 * float(np.median(el)), 1000 * float(np.percentile(el, 90))))


    # O MODO DE FALHA, para comparacao: o MESMO filtro sem a escapatoria
    print("")
    if FORMULACAO_PADRAO == "variante":
        print("  Na variante a escapatoria so e ligada SEM encoder; nas outras a")
        print("  tabela abaixo e a de cima coincidem por construcao.")
    print("  SEM A ESCAPATORIA DO GATE: o mesmo filtro, mesmo criterio")
    print("  de aceitacao, para efeito de comparacao:")
    print("")
    print("  %-19s %10s %10s %10s %9s"
          % ("configuracao", "mediana", "p99", "maximo", "cobert."))
    print("  " + "-" * 64)
    for rotulo, (sens, e_com) in guarda.items():
        r = _monte_carlo(sens, perfil, n, dt, n_runs, 2000, x_parede=X_PAREDE,
                         espurios=True, escapatoria=False, nivel=nivel)
        e = r["err_final"]
        igual = (abs(float(np.median(e)) - float(np.median(e_com))) < 1e-12
                 and abs(float(e.max()) - float(e_com.max())) < 1e-12)
        print("  %-19s %8.1f mm %7.1f mm %7.1f mm %7.0f%%%s"
              % (rotulo, 1000 * float(np.median(e)),
                 1000 * float(np.percentile(e, 99)), 1000 * float(e.max()),
                 100 * r["cobertura"],
                 "   (identico)" if igual else ""))
    print("")
    print("  As configuracoes marcadas `identico` nao acionam a escapatoria:")
    print("  onde ha ENCODER a velocidade e observavel, o filtro nao colapsa")
    print("  `P` sobre uma velocidade errada, e o gate nunca trava.")

    print("")
    print("  Nota: a largada fica a %.2f m e o ultrassom so vale ate %.2f m,"
          % (D_LARGADA, US_MAX_RANGE))
    print("  entao o filtro passa parte do trecho predizendo sem medida")
    print("  nova. Isso NAO e defeito.")
    print("")
    print("  POR QUE A MEDIANA VEM PRIMEIRO: nas")
    print("  configuracoes com ultrassom a distribuicao do erro tem CAUDA")
    print("  PESADA -- corridas raras chegam a 2,5 e 3,2 m. A media segue a")
    print("  cauda e nao o caso tipico: em 600 corridas, `ultrassom+imu` deu")
    print("  mediana 15,5 mm contra media de 48 a 75 mm, e a media variava")
    print("  com a semente enquanto mediana e p90 ficavam estaveis. Citar a")
    print("  media sozinha, com erro padrao, subestima muito a incerteza.")
    return True


# Cenario 3: circulo
def run_circulo(n_runs=N_RUNS_PADRAO, dt=DT_LOOP, graus=15.0):
    duracao = 12.0 + T_PARADO_LARGADA
    n = int(duracao / dt)
    delta = np.deg2rad(graus)
    R_geo = ENTRE_EIXOS_L / np.tan(delta)
    print("")
    print("=" * 78)
    print("CIRCULO — delta constante de %.0f graus" % graus)
    print("=" * 78)
    print("")
    print("  raio geometrico L/tan(delta) = %.4f m, com L = %.3f m"
          % (R_geo, ENTRE_EIXOS_L))
    print("  Sem marco conhecido: o ultrassom NAO entra, e a posicao e")
    print("  navegacao estimada em todas as sete configuracoes.")
    print("")
    print("  %-19s %7s %9s %9s %10s"
          % ("configuracao", "posto", "mediana", "media", "cobertura"))
    print("  " + "-" * 56)

    perfil = perfil_circulo(dt, v=0.8, delta=delta)
    x_ref = np.array([[0.], [0.], [0.], [0.], [0.8], [0.], [0.]])
    traj_ref = [(x_ref, dt, delta, True)] * 60
    for rotulo, sens in fusao.configuracoes():
        r = _monte_carlo(sens, perfil, n, dt, n_runs, 3000, x_parede=None,
                         nivel=nivel_padrao())
        posto, dim = _posto(sens, traj_ref, x_parede=None, nivel=nivel_padrao())
        e = r["err_final"]
        print("  %-19s %5d/%d %8.3f m %8.3f m %8.0f%%"
              % (rotulo, posto, dim, float(np.median(e)), float(np.mean(e)),
                 100 * r["cobertura"]))
    return True


# Cenario 4: corredor
def run_corredor(n_runs=N_RUNS_PADRAO, dt=DT_LOOP):
    duracao = 16.0 + T_PARADO_LARGADA
    n = int(duracao / dt)
    print("")
    print("=" * 78)
    print("CORREDOR — retas e curvas, sem marco de posicao conhecida")
    print("=" * 78)
    print("")
    print("  A pergunta que este cenario responde nao e")
    print("  'qual configuracao acerta a posicao', porque NENHUMA ancora")
    print("  posicao aqui: e 'qual deriva menos'.")
    print("")
    print("  %-19s %7s %9s %9s %10s"
          % ("configuracao", "posto", "mediana", "media", "cobertura"))
    print("  " + "-" * 56)

    perfil = perfil_corredor(dt)
    x_ref = np.array([[0.], [0.], [0.], [0.], [0.8], [0.], [0.]])
    traj_ref = ([(x_ref, dt, 0.0, True)] * 40
                + [(x_ref, dt, np.deg2rad(18.0), True)] * 20)
    melhor, pior = None, None
    for rotulo, sens in fusao.configuracoes():
        r = _monte_carlo(sens, perfil, n, dt, n_runs, 4000, x_parede=None,
                         nivel=nivel_padrao())
        posto, dim = _posto(sens, traj_ref, x_parede=None, nivel=nivel_padrao())
        e = r["err_final"]
        d = float(np.median(e))      # mediana: a cauda e pesada, ver --reta
        print("  %-19s %5d/%d %8.3f m %8.3f m %8.0f%%"
              % (rotulo, posto, dim, d, float(np.mean(e)),
                 100 * r["cobertura"]))
        if melhor is None or d < melhor[1]:
            melhor = (rotulo, d)
        if pior is None or d > pior[1]:
            pior = (rotulo, d)
    print("")
    print("  menor deriva: %s (%.3f m)   maior: %s (%.3f m)   razao %.1fx"
          % (melhor[0], melhor[1], pior[0], pior[1], pior[1] / max(melhor[1], 1e-9)))
    return True


# Cenario 5: rampa
def run_rampa(n_runs=N_RUNS_PADRAO, dt=DT_LOOP):
    duracao = 12.0 + T_PARADO_LARGADA
    n = int(duracao / dt)
    print("")
    print("=" * 78)
    print("RAMPA — validacao da arfagem, SO EM SIMULACAO")
    print("=" * 78)
    print("")
    print("  RESSALVA: nao ha ensaio em rampa. Este")
    print("  cenario mostra que a formulacao e consistente com theta != 0, e")
    print("  nada alem disso. A validacao com dados reais se restringe")
    print("  a condicao theta ~ 0.")
    print("")
    print("  perfil: plano, subida a 8 graus, plano")
    print("")
    print("  %-19s %13s %14s" % ("configuracao", "erro de theta", "deriva final"))
    print("  " + "-" * 50)

    perfil = perfil_rampa(dt)
    for rotulo, sens in fusao.configuracoes():
        errs_th, errs_d = [], []
        for r in range(min(n_runs, 60)):
            rng = np.random.default_rng(5000 + r)
            x_true, deltas, d_reais = gera_verdade(perfil, n, dt)
            out = _roda(x_true, deltas, sens, dt, rng, x_parede=None,
                        deltas_reais=d_reais)
            e_th = np.degrees(np.abs(x_true[:, ITHETA] - out["x_est"][:, ITHETA]))
            errs_th.append(float(np.mean(e_th[int(1.0 / dt):])))
            e = out["x_est"][-1] - x_true[-1]
            errs_d.append(float(np.hypot(e[IX], e[IY])))
        print("  %-19s %9.2f graus %10.3f m"
              % (rotulo, float(np.mean(errs_th)), float(np.mean(errs_d))))
    print("")
    print("  So as configuracoes COM IMU tem canal que observe a inclinacao.")
    print("  Nas demais o erro de theta e o proprio desvio da via, e a")
    print("  covariancia tem de crescer para admitir isso.")
    return True


# Varredura de sensibilidade ao SIGMA_GRADE
def run_varredura(n_runs=120, dt=DT_LOOP, duracao=10.0):
    """Como o NEES responde ao unico parametro de sintonia sem medicao."""
    n = int(duracao / dt)
    sens = (fusao.ENCODER,)

    def perfil_const(k):
        return (0.8, np.deg2rad(10.0), 0.0)

    print("")
    print("=" * 78)
    print("VARREDURA DE SENSIBILIDADE AO SIGMA_GRADE")
    print("=" * 78)
    print("")
    print("  O SIGMA_GRADE e o unico parametro de sintonia sem valor medido,")
    print("  e entrou no lugar do SIGMA_JERK aposentado. Esta varredura diz se")
    print("  ele importa — e, ao contrario do jerk, ele importa.")
    print("")
    print("  MECANISMO: a posicao propaga por v*cos(theta), um PRODUTO de duas")
    print("  grandezas incertas. O EKF lineariza em primeira ordem, e o termo")
    print("  -v*dtheta^2/2 e invisivel para F e nao esta em Q. Quando `theta`")
    print("  vaga o bastante, esse termo cresce com t^2 e o filtro fica")
    print("  superconfiante.")
    print("")
    print("  %-24s %8s %8s %8s" % ("sigma_grade", "t=1s", "t=5s", "t=10s"))
    print("  " + "-" * 52)

    casos = [("nivel plano (theta=0)", None)]
    casos += [("%.1f graus/raiz(s)" % g, np.deg2rad(g))
              for g in (0.1, 0.5, 1.0, 2.0, 5.0)]

    for rotulo, sg in casos:
        nivel = "plano" if sg is None else "arfagem"
        sgv = 0.0 if sg is None else sg
        tab = np.zeros(n)
        for r in range(n_runs):
            rng = np.random.default_rng(9000 + r)
            xt, dc, dr = gera_verdade(perfil_const, n, dt, rng=rng,
                                      sigma_steer=SIGMA_STEER,
                                      sigma_acc=SIGMA_ACC, sigma_grade=sgv,
                                      nivel=nivel)
            out = _roda(xt, dc, sens, dt, rng, x_parede=1.5, chavear=False,
                        deltas_reais=dr, nivel=nivel)
            e = xt - out["x_est"]
            for i in ANGLE_INDICES:
                e[:, i] = np.arctan2(np.sin(e[:, i]), np.cos(e[:, i]))
            for k in range(n):
                ee = e[k].reshape(-1, 1)
                P = out["P_hist"][k] + np.eye(DIM_X) * 1e-12
                tab[k] += float((ee.T @ np.linalg.solve(P, ee)).item())
        m = tab / n_runs
        marca = "  <- em uso" if (sg is not None
                                  and abs(sg - SIGMA_GRADE) < 1e-9) else ""
        print("  %-24s %8.2f %8.2f %8.2f%s"
              % (rotulo, m[20], m[int(5.0 / dt)], m[-1], marca))

    print("")
    print("  LEITURA: no nivel plano o NEES fica estavel, o que mostra que a")
    print("  formulacao esta certa e o crescimento e da arfagem. Ate 0,5 o")
    print("  filtro e consistente no horizonte dos ensaios (7 s na frenagem,")
    print("  16 s no corredor). Acima disso ele fica superconfiante.")
    print("")
    print("  ADOTADO 0,5 graus/raiz(s), e a justificativa e FISICA e nao de")
    print("  ajuste: um corredor de predio nao se inclina 6 graus de forma")
    print("  aleatoria em 10 s, que e o que 2,0 implicava. Caimento de")
    print("  drenagem e de 1 a 2%% (0,6 a 1,2 graus) e rampa de acessibilidade")
    print("  e deliberada e sustentada, nao passeio aleatorio.")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formulacao", action="store_true")
    ap.add_argument("--reta", action="store_true")
    ap.add_argument("--circulo", action="store_true")
    ap.add_argument("--corredor", action="store_true")
    ap.add_argument("--rampa", action="store_true")
    ap.add_argument("--varredura", action="store_true")
    ap.add_argument("--runs", type=int, default=N_RUNS_PADRAO)
    ap.add_argument("--suspende-parado", action="store_true", dest="suspende_parado",
                    help="EM TESTE: sem a entrada do acelerometro com o carro parado")
    ap.add_argument("--suspende-giro", action="store_true", dest="suspende_giro",
                    help="EM TESTE: com --suspende-parado, sem a do giroscopio tambem")
    ap.add_argument("--sem-zupt", action="store_true", dest="sem_zupt",
                    help="desliga a ZUPT + ZARU da variante (para comparar)")
    ap.add_argument("--sem-mag", action="store_true", dest="sem_mag",
                    help="sem o canal do magnetometro, como o filtro oficial "
                         "roda nos dados reais")
    ap.add_argument("--vigente", action="store_true",
                    help="roda o filtro de fusao.py (esterco como entrada), para "
                         "comparar; o padrao e a VARIANTE (giroscopio como entrada)")
    args = ap.parse_args(argv)
    global FORMULACAO_PADRAO
    FORMULACAO_PADRAO = "vigente" if args.vigente else "variante"
    global SUSPENDE_PARADO, SUSPENDE_GIRO
    SUSPENDE_PARADO, SUSPENDE_GIRO = args.suspende_parado, args.suspende_giro
    global ZUPT, SEM_MAG
    ZUPT = not args.sem_zupt
    SEM_MAG = args.sem_mag
    if SEM_MAG:
        print("SEM O MAGNETOMETRO: canal suprimido, como nos dados reais")

    if FORMULACAO_PADRAO == "variante" and not args.vigente:
        print("ZUPT + ZARU com o carro parado: %s" % ("LIGADA" if ZUPT else "desligada"))
    if SUSPENDE_PARADO:
        print("EM TESTE: entrada do acelerometro suspensa parado%s"
              % (" (e a do giroscopio)" if SUSPENDE_GIRO else ""))
    print("FORMULACAO DO FILTRO FUNDIDO: %s" % FORMULACAO_PADRAO.upper())
    print("")

    algum = any([args.formulacao, args.reta, args.circulo, args.corredor,
                 args.rampa, args.varredura])
    ok = True
    if args.formulacao or not algum:
        ok &= run_formulacao(n_runs=args.runs)
    if args.reta or not algum:
        run_reta(n_runs=args.runs)
    if args.circulo or not algum:
        run_circulo(n_runs=args.runs)
    if args.corredor or not algum:
        run_corredor(n_runs=args.runs)
    # a rampa e a varredura do SIGMA_GRADE sao da arfagem; a variante e plana
    arfagem = FORMULACAO_PADRAO == "vigente"
    if (args.rampa or args.varredura) and not arfagem:
        print("  (--rampa e --varredura sao da arfagem: rode com --vigente)")
    if arfagem and (args.rampa or not algum):
        run_rampa(n_runs=args.runs)
    if arfagem and (args.varredura or not algum):
        run_varredura()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
