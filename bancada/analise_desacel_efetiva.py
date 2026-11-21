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
Desaceleração efetiva: a que reproduz a distância de parada real, contada desde
o comando de corte do motor.
"""
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
import csv
import glob

import numpy as np

# Abaixo disto o veículo está parado. Mesmo limiar de analise_costeio.py.
V_PARADO = 0.05


def ler_csv(caminho):
    """Devolve (cabecalho, t, v)."""
    with open(caminho, "r", encoding="utf-8") as f:
        linhas_todas = f.readlines()
    cab = {}
    for ln in linhas_todas:
        if ln.startswith("#") and ":" in ln:
            chave, valor = ln[1:].split(":", 1)
            cab[chave.strip()] = valor.strip()
    leitor = csv.DictReader([ln for ln in linhas_todas if not ln.startswith("#")])
    t, v = [], []
    for linha in leitor:
        try:
            bruto = linha.get("enc_valido", linha.get("valido", 1))
            if int(float(bruto)) != 1:
                continue
            t.append(float(linha["t_s"]))
            v.append(float(linha["vel_m_s"]))
        except (TypeError, ValueError, KeyError):
            continue
    return cab, np.asarray(t, dtype=float), np.asarray(v, dtype=float)


def medir(caminho):
    """Devolve dict com a desaceleração efetiva de uma corrida, ou None."""
    cab, t, v = ler_csv(caminho)
    if "corte_s" not in cab or len(t) < 10:
        return None
    i0 = int(np.searchsorted(t, float(cab["corte_s"])))
    if i0 >= len(v) - 1:
        return None
    i1 = i0
    while i1 < len(v) - 1 and v[i1] > V_PARADO:
        i1 += 1
    if i1 <= i0 or v[i0] <= V_PARADO:
        return None

    d_real = float(np.trapezoid(v[i0:i1 + 1], t[i0:i1 + 1]))
    if d_real <= 0:
        return None
    return {
        "nome": cab.get("condicao", caminho),
        "v0": float(v[i0]),
        "d": d_real,
        "t": float(t[i1] - t[i0]),
        "a_ef": float(v[i0] ** 2 / (2.0 * d_real)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSVs de corrida (precisam de corte_s)")
    ap.add_argument("--alcance", type=float, default=1.50,
                    help="alcance util do ultrassom, para o teto [m]")
    args = ap.parse_args(argv)

    caminhos = []
    for padrao in args.csv:
        caminhos.extend(sorted(glob.glob(padrao)) or [padrao])

    print("=" * 72)
    print("DESACELERACAO EFETIVA (desde o comando de corte ate a parada)")
    print("=" * 72)
    print(f"  {'corrida':<30} {'v_corte':>8} {'d_real':>8} {'t':>6} {'a_ef':>8}")
    res = []
    for caminho in caminhos:
        r = medir(caminho)
        if r is None:
            continue
        res.append(r)
        print(f"  {r['nome']:<30} {r['v0']:8.3f} {r['d']:8.3f} "
              f"{r['t']:6.2f} {r['a_ef']:8.4f}")

    if len(res) < 2:
        print("\n  Menos de duas corridas utilizaveis.")
        return 1

    a = np.array([r["a_ef"] for r in res])
    v0 = np.array([r["v0"] for r in res])
    media, desvio = float(a.mean()), float(a.std(ddof=1))

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print(f"  n = {len(a)}")
    print(f"  desaceleracao efetiva : {media:.4f} m/s^2")
    print(f"  desvio entre corridas : {desvio:.4f} m/s^2 ({100 * desvio / media:.1f}%)")
    print(f"  teto em {args.alcance:.2f} m       : "
          f"{np.sqrt(2.0 * media * args.alcance):.2f} m/s")

    if len(a) >= 3:
        corr = float(np.corrcoef(v0, a)[0, 1])
        k, b = np.polyfit(v0, a, 1)
        print(f"  correlacao com v_corte: {corr:+.3f}   (ajuste: a = {k:+.3f} v {b:+.3f})")
        if corr > 0.5:
            print("  -> efetiva cresce com v, como a diluicao pela transicao preve.")
            print("     NAO confundir com arrasto: a de REGIME nao mostra o efeito.")

    print("\n  distancias de parada previstas:")
    for rot, vv in (("baixo", 0.54), ("medio", 0.94), ("alto ", 1.31)):
        d = vv ** 2 / (2.0 * media)
        print(f"    {rot}  v = {vv:.2f} m/s -> {d:.2f} m "
              f"(folga em {args.alcance:.2f} m: {args.alcance - d:.2f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
