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
"""Desaceleração por costeio (motor cortado), a partir dos CSVs do encoder."""
# --- caminho de importação
# Permite rodar direto, sem -m.
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
import sys

import numpy as np

# Um trecho só vale como costeio se for longo o bastante para a reta
# significar alguma coisa. 6 amostras a 20 Hz são 0,3 s.
MIN_AMOSTRAS = 6

FRAC_TOPO = 0.85
FRAC_BASE = 0.20

# Abaixo disto a reta não descreve o trecho, e o número não deve ser usado.
R2_MINIMO = 0.90


def ler_csv(caminho):
    """Devolve (t, v) das amostras válidas. Ignora o cabeçalho comentado."""
    with open(caminho, "r", encoding="utf-8") as f:
        linhas = [ln for ln in f if not ln.startswith("#")]
    leitor = csv.DictReader(linhas)
    t, v = [], []
    for linha in leitor:
        try:
            if int(float(linha.get("valido", 1))) != 1:
                continue
            t.append(float(linha["t_s"]))
            v.append(float(linha["vel_m_s"]))
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(t, dtype=float), np.asarray(v, dtype=float)


def achar_costeios(t, v, v_min, v_parado):
    """Localiza os trechos de costeio, por frações do pico de cada corrida."""
    trechos = []
    n = len(v)
    i = 0
    while i < n:
        if v[i] <= v_parado:
            i += 1
            continue
        # região contígua acima do limiar de parado: uma corrida
        j = i
        while j + 1 < n and v[j + 1] > v_parado:
            j += 1

        pico = float(np.max(v[i:j + 1]))
        if pico >= v_min:
            topo = pico * FRAC_TOPO
            base = max(pico * FRAC_BASE, v_parado)

            # último instante ainda no topo: é o corte do motor
            ini = None
            for k in range(j, i - 1, -1):
                if v[k] >= topo:
                    ini = k
                    break
            if ini is not None:
                fim = j
                for k in range(ini + 1, j + 1):
                    if v[k] <= base:
                        fim = k
                        break
                if fim - ini >= MIN_AMOSTRAS:
                    trechos.append((ini, fim))
        i = j + 1
    return trechos


def _r2(v, pred):
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def ajustar(t, v):
    """Reta por mínimos quadrados. Devolve (inclinacao, R2)."""
    a, b = np.polyfit(t, v, 1)
    return float(a), _r2(v, a * t + b)


def ajustar_exp(t, v):
    """Ajusta v = v0*exp(-(t-t0)/tau) e devolve (tau, R2)."""
    t0 = t[0]
    mask = v > 1e-6
    if int(np.sum(mask)) < MIN_AMOSTRAS:
        return None, 0.0
    k, b = np.polyfit(t[mask] - t0, np.log(v[mask]), 1)
    if k >= 0:
        return None, 0.0
    tau = -1.0 / float(k)
    return tau, _r2(v, np.exp(b) * np.exp(-(t - t0) / tau))


def medir_transicao(t, v, ini, a_costeio, n_suav=3):
    """Mede quanto tempo a aceleração leva para ir do patamar ao costeio."""
    ini_j = max(0, ini - 12)
    fim_j = min(len(v) - 1, ini + 12)
    if fim_j - ini_j < 2 * n_suav + 2:
        return None, None

    dt = float(np.median(np.diff(t[ini_j:fim_j + 1])))
    if dt <= 0:
        return None, None

    a = np.gradient(v[ini_j:fim_j + 1], dt)
    nucleo = np.ones(n_suav) / n_suav
    a = np.convolve(a, nucleo, mode="same")

    # degrau de 0 (patamar) até -a_costeio; cruzamentos em 20% e 80%
    alvo_lo, alvo_hi = -0.20 * a_costeio, -0.80 * a_costeio
    k_lo = k_hi = None
    for k in range(len(a)):
        if k_lo is None and a[k] <= alvo_lo:
            k_lo = k
        if k_lo is not None and a[k] <= alvo_hi:
            k_hi = k
            break
    if k_lo is None or k_hi is None or k_hi <= k_lo:
        return None, None

    t_tr = (k_hi - k_lo) * dt
    if t_tr <= 0:
        return None, None
    # 20% a 80% cobre 60% do degrau
    return t_tr, 0.60 * a_costeio / t_tr


def analisar(caminho, v_min, v_parado):
    t, v = ler_csv(caminho)
    if len(t) < MIN_AMOSTRAS:
        print(f"  {caminho}: poucas amostras validas ({len(t)})")
        return []

    resultados = []
    for ini, fim in achar_costeios(t, v, v_min, v_parado):
        a, r2 = ajustar(t[ini:fim + 1], v[ini:fim + 1])
        if a >= 0:
            continue
        tau, r2exp = ajustar_exp(t[ini:fim + 1], v[ini:fim + 1])
        t_tr, s_jerk = medir_transicao(t, v, ini, -a)
        resultados.append({
            "t_tr": t_tr, "s_jerk": s_jerk,
            "t0": t[ini], "t1": t[fim],
            "v0": v[ini], "v1": v[fim],
            "a": -a, "r2": r2, "bom": r2 >= R2_MINIMO,
            "tau": tau, "r2exp": r2exp,
            "dur": t[fim] - t[ini], "n": fim - ini + 1,
        })
    return resultados


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="CSVs de encoder")
    ap.add_argument("--v-min", type=float, default=0.30, dest="v_min",
                    help="velocidade minima do pico para o trecho contar [m/s]")
    ap.add_argument("--v-parado", type=float, default=0.05, dest="v_parado",
                    help="abaixo disto considera-se parado [m/s]")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste com verdade conhecida")
    args = ap.parse_args(argv)

    if args.fake:
        return autoteste()

    caminhos = []
    for padrao in args.csv:
        caminhos.extend(sorted(glob.glob(padrao)) or [padrao])
    if not caminhos:
        ap.error("informe ao menos um CSV, ou use --fake")

    todos = []
    print("=" * 70)
    print("DESACELERACAO POR COSTEIO")
    print("=" * 70)
    for caminho in caminhos:
        res = analisar(caminho, args.v_min, args.v_parado)
        nome = caminho.split("/")[-1].split("\\")[-1]
        if not res:
            print(f"\n  {nome}: nenhum trecho de costeio identificado")
            continue
        print(f"\n  {nome}")
        for r in res:
            marca = "" if r["bom"] else "   <- R2 BAIXO, descartado"
            print(f"    v {r['v0']:.3f} -> {r['v1']:.3f} m/s  em {r['dur']:.2f} s "
                  f"({r['n']} amostras)   a = {r['a']:.4f} m/s^2   "
                  f"R2 = {r['r2']:.4f}{marca}")
            if r["tau"] is not None:
                vence = "EXPONENCIAL" if r["r2exp"] > r["r2"] else "reta"
                print(f"        modelo: reta R2 = {r['r2']:.4f} | "
                      f"exp R2 = {r['r2exp']:.4f} (tau = {r['tau']:.3f} s)"
                      f"  -> vence {vence}")
        todos.extend(res)

    if not todos:
        print("\n  Nenhum costeio encontrado. Baixe --v-min se as corridas")
        print("  foram mais lentas que 0,30 m/s.")
        return 1

    # Só entram na média os trechos em que a reta de fato descreve o dado:
    # um R2 baixo significa desaceleração não constante.
    bons = [r for r in todos if r["bom"]]
    descartados = len(todos) - len(bons)
    if not bons:
        print(f"\n  Os {len(todos)} trechos tem R2 abaixo de {R2_MINIMO:.2f}.")
        print("  Nenhum descreve desaceleracao constante. Refazer o ensaio.")
        return 1

    a = np.array([r["a"] for r in bons])
    v0 = np.array([r["v0"] for r in bons])
    media, desvio = float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"  trechos aproveitados: {len(a)} de {len(todos)}"
          + (f" ({descartados} com R2 baixo)" if descartados else ""))
    print(f"  desaceleracao media : {media:.4f} m/s^2")
    if len(a) > 1:
        print(f"  desvio entre trechos: {desvio:.4f} m/s^2 "
              f"({100.0 * desvio / media:.1f}%)")
    print(f"  mu implicito        : {media / 9.81:.4f}   "
          f"(class_car.py declara MI = 0.04)")

    # O teto de velocidade sai daqui: v_max = sqrt(2*a*d), com d = alcance util.
    for margem, util in ((0.0, 2.00), (0.3, 1.70), (0.5, 1.50)):
        print(f"  teto com margem de {margem:.1f} m: "
              f"{np.sqrt(2.0 * media * util):.2f} m/s")

    # A dependência com a velocidade separa rolamento de arrasto: mu*g não
    # depende de v, o arrasto depende de v^2.
    if len(a) >= 3:
        corr = float(np.corrcoef(v0, a)[0, 1])
        print(f"\n  correlacao entre v inicial e desaceleracao: {corr:+.3f}")
        if abs(corr) < 0.5:
            print("  -> sem dependencia clara da velocidade: o rolamento domina,")
            print("     e desprezar o arrasto esta correto.")
        else:
            print("  -> HA dependencia da velocidade. Um termo alem do rolamento")
            print("     atua, provavelmente arrasto ou freio do ESC. Investigar.")

    # O jerk sai da transição, e não do costeio. Ver medir_transicao.
    jerks = [r["s_jerk"] for r in todos if r.get("s_jerk")]
    if jerks:
        jm = float(np.mean(jerks))
        ttr = float(np.mean([r["t_tr"] for r in todos if r.get("s_jerk")]))
        print(f"\n  transicao plato -> costeio: {ttr:.3f} s em media "
              f"({len(jerks)} trechos)")
        print(f"  SIGMA_JERK implicado      : {jm:.1f} m/s^3   "
              f"(limite superior, ver ressalva)")
        print(f"  com sigma_jerk = 1,0 seriam {1.043 / (1.0 * 0.05):.0f} amostras "
              f"({1.043 / 1.0:.1f} s) para acompanhar o corte,")
        print("  mais que a duracao do proprio costeio: o filtro terminaria a")
        print("  frenagem sem ter notado que o motor foi cortado.")

    # Qual modelo descreve melhor a queda indica o mecanismo. Ver a
    # docstring de ajustar_exp.
    comparaveis = [r for r in todos if r["tau"] is not None]
    if comparaveis:
        n_exp = sum(1 for r in comparaveis if r["r2exp"] > r["r2"])
        print(f"\n  mecanismo: exponencial vence em {n_exp} de "
              f"{len(comparaveis)} trechos")
        if n_exp >= 0.7 * len(comparaveis):
            print("  -> decaimento EXPONENCIAL: torque proporcional a rotacao.")
            print("     E freio eletrico do ESC, e nao atrito seco: ha TORQUE")
            print("     nas rodas durante a parada.")
        elif n_exp <= 0.3 * len(comparaveis):
            print("  -> decaimento LINEAR: torque constante, atrito seco do")
            print("     conjunto motor-reducao, sem torque util nas rodas.")
        else:
            print("  -> indefinido: os dois modelos empatam. Mais corridas, ou")
            print("     ensaio no cavalete, onde o rolamento sai da conta.")

    if media > 3.0 * 0.3924:
        print("\n  ATENCAO: desaceleracao muito acima de mu*g com MI=0.04.")
        print("  Suspeita principal: o ESC aplica FREIO no neutro, e nao ha")
        print("  costeio livre.")
    return 0


def autoteste():
    """Gera corridas com desaceleração conhecida e confere a recuperação."""
    print("=== autoteste de analise_costeio ===")
    rng = np.random.default_rng(42)
    dt, a_verdade, sigma = 0.05, 0.3924, 0.0189
    t_all, v_all, tempo = [], [], 0.0
    for i, v0 in enumerate((0.6, 0.8, 1.0)):
        for k in range(15):                      # arrancada
            t_all.append(tempo); v_all.append(v0 * k / 15.0); tempo += dt
        for k in range(50):  # patamar ondulado, como no real
            t_all.append(tempo)
            v_all.append(v0 + 0.02 * np.sin(0.7 * k) + 0.01 * np.sin(2.3 * k))
            tempo += dt
        v = v0                                   # costeio, a verdade
        while v > 0.0:
            t_all.append(tempo); v_all.append(v); tempo += dt
            v -= a_verdade * dt
        for _ in range(10):                      # parado
            t_all.append(tempo); v_all.append(0.0); tempo += dt
    t = np.asarray(t_all)
    v = np.asarray(v_all) + rng.normal(0.0, sigma, len(v_all))

    trechos = achar_costeios(t, v, 0.30, 0.05)
    print(f"  trechos encontrados: {len(trechos)} (esperado 3)")
    ok = len(trechos) == 3
    for ini, fim in trechos:
        est, r2 = ajustar(t[ini:fim + 1], v[ini:fim + 1])
        erro = 100.0 * abs(-est - a_verdade) / a_verdade
        print(f"    a = {-est:.4f} m/s^2  (verdade {a_verdade:.4f}, "
              f"erro {erro:.2f}%)  R2 = {r2:.4f}")
        ok = ok and erro < 5.0 and r2 >= R2_MINIMO
    print(f"\n=== RESULTADO: {'OK' if ok else 'FALHOU'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
