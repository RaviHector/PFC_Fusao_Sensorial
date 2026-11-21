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
"""Ensaio de empurrão à mão: o raio efetivo da roda, sem patinação."""
import argparse
import csv
import io
import os
import sys

if __package__ in (None, ""):
    _AQUI = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_AQUI))

import numpy as np

from kalman.config import SIGMA_US

# Abaixo disto o veículo é considerado parado. O encoder em repouso devolve
# exatamente 0,000, e o ruído no empurrão lento fica em 0,006 m/s.
V_PARADO = 0.02          # [m/s]
MIN_PATAMAR = 20         # amostras (1,0 s a 20 Hz) para valer como patamar


def ler(caminho):
    linhas = [L for L in io.open(caminho, encoding="utf-8")
              if not L.startswith("#")]
    r = list(csv.DictReader(linhas))
    t = np.array([float(x["t_s"]) for x in r])
    v = np.array([float(x["vel_m_s"]) for x in r])
    d = np.array([float(x["dist_m"]) for x in r])
    ok = np.array([x["us_valido"] == "1" for x in r])
    return t, v, d, ok


def _cabecalho(caminho, chave):
    for L in io.open(caminho, encoding="utf-8"):
        if not L.startswith("#"):
            break
        if L[1:].strip().startswith(chave + ":"):
            return L.split(":", 1)[1].strip()
    return ""


def patamares(t, v):
    """Acha o trecho parado do início e o do fim."""
    parado = np.abs(v) < V_PARADO

    i = 0
    while i < len(parado) and parado[i]:
        i += 1
    ini = slice(0, i)

    j = len(parado)
    while j > 0 and parado[j - 1]:
        j -= 1
    fim = slice(j, len(parado))

    moveu_no_fim = (len(parado) - j) < MIN_PATAMAR
    return ini, fim, moveu_no_fim


def analisar(caminho, d_ini_trena, d_fim_trena, verbose=True):
    t, v, d, ok = ler(caminho)
    ini, fim, moveu = patamares(t, v)
    n_ini, n_fim = ini.stop - ini.start, fim.stop - fim.start

    nome = os.path.basename(caminho)
    cond = _cabecalho(caminho, "condicao")
    if verbose:
        print("=" * 74)
        print("%s   [%s]" % (nome, cond))
        print("=" * 74)

    d_trena = d_ini_trena - d_fim_trena

    # --- ultrassom: média de cada patamar
    m_ini = ok.copy(); m_ini[fim.start:] = False; m_ini[ini.stop:] = False
    m_fim = np.zeros_like(ok); m_fim[fim] = ok[fim]
    us_ini = float(np.mean(d[m_ini])) if m_ini.any() else float("nan")
    us_fim = float(np.mean(d[m_fim])) if m_fim.any() else float("nan")
    d_us = us_ini - us_fim
    # incerteza da diferença de duas médias
    e_us = SIGMA_US * np.sqrt(1.0 / max(n_ini, 1) + 1.0 / max(n_fim, 1))

    # --- encoder: integral da velocidade
    d_enc = float(np.trapezoid(v, t)) if hasattr(np, "trapezoid") \
        else float(np.trapz(v, t))

    if verbose:
        print("  patamar inicial : %3d amostras (%.1f s)" % (n_ini, n_ini * 0.05))
        print("  patamar final   : %3d amostras (%.1f s)%s"
              % (n_fim, n_fim * 0.05,
                 "   <<< CURTO, ver aviso" if moveu else ""))
        print("")
        print("  %-26s %10s %10s" % ("deslocamento por", "valor", "vs trena"))
        print("  " + "-" * 50)
        print("  %-26s %8.4f m %10s" % ("trena (marcas)", d_trena, "--"))
        print("  %-26s %8.4f m %+9.1f%%"
              % ("ultrassom", d_us, 100 * (d_us - d_trena) / d_trena))
        print("  %-26s %8.4f m %+9.1f%%"
              % ("encoder", d_enc, 100 * (d_enc - d_trena) / d_trena))
        print("")
        print("  ENCODER contra ULTRASSOM: %+.4f m  (%+.2f%%)"
              % (d_enc - d_us, 100 * (d_enc - d_us) / d_us))
        print("  incerteza do ultrassom  : +-%.4f m (%.2f%%)"
              % (e_us, 100 * e_us / d_trena))
        print("")
        print("  ultrassom parado: inicio %.4f m, fim %.4f m" % (us_ini, us_fim))
        print("  offset da marca : inicio %+.3f m, fim %+.3f m"
              % (us_ini - d_ini_trena, us_fim - d_fim_trena))
        if moveu:
            print("")
            print("  AVISO: o patamar final tem so %.1f s (%d amostras)."
                  % (n_fim * 0.05, n_fim))
            print("  O veiculo parou, mas sem folga para a media render")
            print("  precisao: o erro da leitura de chegada fica em +-%.1f mm"
                  % (1000 * SIGMA_US / np.sqrt(max(n_fim, 1))))
            print("  contra os +-1,8 mm de um patamar de 5 s. O numero acima")
            print("  vale, com essa incerteza maior.")

    return dict(nome=nome, cond=cond, d_trena=d_trena, d_us=d_us, d_enc=d_enc,
                e_us=e_us, n_ini=n_ini, n_fim=n_fim, moveu=moveu)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="CSV do bancada.corrida --sem-motor")
    ap.add_argument("--de", type=float, help="marca de partida, em m da parede")
    ap.add_argument("--para", type=float, help="marca de chegada, em m da parede")
    args = ap.parse_args(argv)

    if args.csv and args.de is not None and args.para is not None:
        alvos = [(c, args.de, args.para) for c in args.csv]
    else:
        ap.error("informe o CSV com --de e --para")

    saidas = []
    for caminho, a, b in alvos:
        if not os.path.exists(caminho):
            print("  [!] nao encontrado: %s" % caminho)
            continue
        saidas.append(analisar(caminho, a, b))
        print("")

    # Entram todas, inclusive as de patamar curto: valem, só com incerteza
    # maior.
    validas = saidas
    if len(validas) >= 2:
        print("=" * 74)
        print("VEREDICTO — encoder contra ultrassom")
        print("=" * 74)
        print("  %-22s %10s %12s %10s" % ("corrida", "desloc.", "enc - us", "%"))
        print("  " + "-" * 58)
        for s in validas:
            print("  %-22s %8.3f m %10.4f m %+9.2f%%"
                  % (s["cond"][:22], s["d_trena"], s["d_enc"] - s["d_us"],
                     100 * (s["d_enc"] - s["d_us"]) / s["d_us"]))
        difs = np.array([s["d_enc"] - s["d_us"] for s in validas])
        pcts = np.array([100 * (s["d_enc"] - s["d_us"]) / s["d_us"]
                         for s in validas])
        # Espalhamento relativo, e não absoluto: metros e porcentagem não se
        # comparam.
        cv_m = abs(difs.std() / difs.mean()) * 100
        cv_p = abs(pcts.std() / pcts.mean()) * 100
        print("")
        print("  se fosse DESVIO FIXO   : %+.4f m, variando %.0f%% entre corridas"
              % (difs.mean(), cv_m))
        print("  se fosse ERRO DE ESCALA: %+.2f %%, variando %.0f%% entre corridas"
              % (pcts.mean(), cv_p))
        print("")
        print("  COMO LER. A hipotese certa e a que VARIA MENOS entre corridas.")
        print("  Mas so vale concluir se uma variar bem menos que a outra: com")
        print("  as duas na mesma faixa, o dado nao separa as hipoteses, e o")
        print("  que falta e repetir o MESMO deslocamento algumas vezes para")
        print("  medir quanto e simples espalhamento de corrida para corrida.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
