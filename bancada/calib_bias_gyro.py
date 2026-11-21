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
"""Calibração do viés do giroscópio, e de quebra o SIGMA_GYRO."""
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
import argparse
import math
import statistics
import time

import numpy as np

from kalman.config import SIGMA_GYRO

EIXOS = ("gx", "gy", "gz")

# Acima disto, para um MPU-9250 em repouso, quase certamente o veículo se
# mexeu ou há vibração. Não é limite de especificação: é critério de sanidade.
DESVIO_SUSPEITO_DPS = 2.0


def coletar_bruto(imu, n_amostras, delay=0.005):
    """N amostras do giroscópio sem correção de viés."""
    dados = {e: [] for e in EIXOS}
    print("Coletando {} amostras. NAO ENCOSTE no veiculo.".format(n_amostras))
    for i in range(n_amostras):
        gx, gy, gz = imu._ler_giro_dps_bruto()
        dados["gx"].append(gx)
        dados["gy"].append(gy)
        dados["gz"].append(gz)
        if (i + 1) % max(1, n_amostras // 10) == 0:
            print("  {:>6}/{} ...".format(i + 1, n_amostras))
        if delay > 0:
            time.sleep(delay)
    return dados


def convergencia(serie, pontos=8):
    """Como a estimativa de viés evolui com o número de amostras."""
    n = len(serie)
    if n < 20:
        return []
    marcos = [int(n * (i + 1) / pontos) for i in range(pontos)]
    fora = []
    final = statistics.mean(serie)
    for m in marcos:
        if m < 2:
            continue
        sub = serie[:m]
        med = statistics.mean(sub)
        dp = statistics.stdev(sub)
        erro_padrao = dp / math.sqrt(m)
        fora.append({"n": m, "media": med, "desvio": dp,
                     "erro_padrao": erro_padrao, "delta_final": med - final})
    return fora


def relatorio(dados, delay):
    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    n = len(dados["gx"])
    dur = n * delay
    print("  amostras : {}   (~{:.1f}s de coleta)".format(n, dur))
    print()
    print("  {:<5} {:>12} {:>12} {:>14}".format(
        "eixo", "bias[deg/s]", "sigma[deg/s]", "deriva[deg/min]"))
    print("  " + "-" * 46)

    resumo = {}
    suspeito = False
    for e in EIXOS:
        s = dados[e]
        media = statistics.mean(s)
        desvio = statistics.stdev(s) if len(s) > 1 else 0.0
        resumo[e] = {"bias": media, "sigma": desvio}
        if desvio > DESVIO_SUSPEITO_DPS:
            suspeito = True
        print("  {:<5} {:>+12.4f} {:>12.4f} {:>+14.1f}".format(
            e, media, desvio, media * 60.0))

    sigma_medio = statistics.mean([resumo[e]["sigma"] for e in EIXOS])
    print()
    print("  SIGMA_GYRO (media dos 3 eixos): {:.4f} deg/s = {:.6f} rad/s".format(
        sigma_medio, math.radians(sigma_medio)))
    print("  valor em config.py            : {:.4f} deg/s".format(
        math.degrees(SIGMA_GYRO)))
    razao = sigma_medio / math.degrees(SIGMA_GYRO) if SIGMA_GYRO > 0 else 0.0
    print("  razao medido/config           : {:.2f}x".format(razao))

    if suspeito:
        print()
        print("  *** ATENCAO: desvio acima de {:.1f} deg/s ***".format(
            DESVIO_SUSPEITO_DPS))
        print("  Para um MPU-9250 em repouso isso e alto demais. Provavelmente")
        print("  o veiculo se mexeu, esta sobre superficie que vibra, ou o")
        print("  motor estava ligado. O bias medido aqui NAO e confiavel.")

    # --- convergência
    print()
    print("  Convergencia da estimativa de bias (eixo com maior bias):")
    pior = max(EIXOS, key=lambda e: abs(resumo[e]["bias"]))
    conv = convergencia(dados[pior])
    if conv:
        print("  eixo {}:".format(pior))
        print("    {:>7} {:>12} {:>13} {:>13}".format(
            "N", "bias", "erro_padrao", "dif p/ final"))
        print("    " + "-" * 48)
        for c in conv:
            print("    {:>7} {:>+12.4f} {:>13.4f} {:>+13.4f}".format(
                c["n"], c["media"], c["erro_padrao"], c["delta_final"]))

        # A partir de que N a estimativa fica dentro do proprio erro padrao?
        estavel = None
        for c in conv:
            if abs(c["delta_final"]) <= c["erro_padrao"]:
                estavel = c["n"]
                break
        print()
        if estavel:
            print("    Estabiliza a partir de ~{} amostras.".format(estavel))
            if estavel > 100:
                print("    Os 100 do construtor do driver sao POUCOS —")
                print("    considerar aumentar `amostras_calib`.")
            else:
                print("    Os 100 do construtor do driver sao suficientes.")
        else:
            print("    Nao estabilizou dentro do erro padrao. Coletar mais.")

    print()
    print("  Para aplicar, editar kalman/config.py:")
    print("      SIGMA_GYRO = np.deg2rad({:.4f})  # MEDIDO, {} amostras".format(
        sigma_medio, n))
    print()
    print("  O bias NAO vai para config.py: ele e condicao inicial do estado")
    print("  de vies, medido a cada partida pelo construtor do driver,")
    print("  porque muda com temperatura e entre ligamentos.")

    return resumo


def _autoteste():
    """Verdade conhecida: viés e sigma injetados, o script tem de recuperá-los."""
    from bancada.drivers.base import FonteI2CFake
    from bancada.drivers.imu import IMU

    print("=== bancada.calib_bias_gyro — autoteste ===\n")

    BIAS = (0.35, -1.33, 0.48)  # deg/s, da ordem dos valores medidos
    RUIDO = 0.30                 # deg/s

    fonte = FonteI2CFake(gyro_dps=BIAS, ruido=RUIDO, seed=7)
    # calibrar=False: sem isso o construtor subtrairia o viés que se quer medir
    imu = IMU(fonte=fonte, calibrar=False)
    try:
        dados = coletar_bruto(imu, 1500, delay=0.0)
    finally:
        imu.close()

    resumo = relatorio(dados, delay=0.005)

    print()
    ok = True
    for e, verdade in zip(EIXOS, BIAS):
        medido = resumo[e]["bias"]
        err = abs(medido - verdade)
        bom = err < 0.05
        ok &= bom
        print("  {} : medido {:+.4f}  verdadeiro {:+.4f}  erro {:.4f} -> {}".format(
            e, medido, verdade, err, "OK" if bom else "FALHOU"))

    s_med = statistics.mean([resumo[e]["sigma"] for e in EIXOS])
    err_s = abs(s_med - RUIDO) / RUIDO
    ok_s = err_s < 0.10
    ok &= ok_s
    print("  sigma : medido {:.4f}  verdadeiro {:.4f}  erro {:.1f}% -> {}".format(
        s_med, RUIDO, 100 * err_s, "OK" if ok_s else "FALHOU"))

    print("\n=== RESULTADO: {} ===".format("OK" if ok else "FALHOU"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Calibracao do vies do giroscopio e do SIGMA_GYRO")
    ap.add_argument("--amostras", type=int, default=2000,
                    help="numero de amostras (padrao 2000). Mais amostras, "
                         "melhor a estimativa da media — o erro cai com 1/sqrt(N).")
    ap.add_argument("--delay", type=float, default=0.005,
                    help="intervalo entre amostras em s (padrao 0.005)")
    ap.add_argument("--bus", type=int, default=4,
                    help="bus I2C (padrao 4; tentar 1 se falhar)")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste com verdade conhecida, sem hardware")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    from bancada.drivers.imu import IMU

    print("=" * 70)
    print("CALIBRACAO DO BIAS DO GIROSCOPIO")
    print("=" * 70)
    print("O veiculo precisa estar PARADO, sobre superficie firme,")
    print("com o motor DESLIGADO. Nao segure na mao.")
    print()
    input("Tecle ENTER quando estiver assim...")
    print()

    # calibrar=False é essencial: calibrar antes de medir o viés seria circular
    imu = IMU(bus_id=args.bus, calibrar=False)
    try:
        dados = coletar_bruto(imu, args.amostras, delay=args.delay)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        dados = None
    finally:
        imu.close()

    if dados and len(dados["gx"]) > 20:
        relatorio(dados, delay=args.delay)
    else:
        print("Amostras insuficientes.")
