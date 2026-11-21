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
Análise dos CSVs de `coleta.py`, `coleta_multi.py` e `corrida.py`: o ruído de
cada canal, de onde saem os R dos filtros.
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
import math
import os

import numpy as np

# Mapa coluna -> constante de kalman/config.py que ela alimenta, para o
# relatório dizer diretamente o que escrever no config.
ALVO_CONFIG = {
    "vel_m_s": ("SIGMA_ENC", "m/s"),
    "dist_m": ("SIGMA_US", "m"),
    "roll_rad": ("SIGMA_ACC_TILT", "rad"),
    "pitch_rad": ("SIGMA_ACC_TILT", "rad"),
    "yaw_rad": ("SIGMA_MAG", "rad"),
    "gx_rads": ("SIGMA_GYRO", "rad/s"),
    "gy_rads": ("SIGMA_GYRO", "rad/s"),
    "gz_rads": ("SIGMA_GYRO", "rad/s"),
}

# Colunas que são ângulos e precisam de unwrap antes de qualquer estatística.
ANGULARES = {"yaw_rad"}

# Canais cuja derivada alimenta o ruído de processo Q.
DERIVAR_PARA = {
    "vel_m_s": ("SIGMA_ACC (1a) / SIGMA_JERK (2a)", "m/s"),
    "dist_m": ("SIGMA_US_ACCEL (2a)", "m"),
}

# `enc_valido` e `imu_valido` vêm de bancada/coleta_multi.py.
NAO_ANALISAR = {"t_s", "dt_medido_s", "valido", "motivo", "accel_confiavel",
                "enc_valido", "imu_valido",
                # Colunas do `bancada.corrida`. `us_motivo` é texto ("acima_max",
                # "salto_impossivel").
                "us_valido", "us_novo", "us_motivo"}


def ler_csv(caminho):
    """Devolve (meta: dict, colunas: dict[str, list], n_total, n_validas)."""
    meta = {}
    linhas_dados = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("#"):
                if ":" in linha:
                    k, v = linha[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
            else:
                linhas_dados.append(linha)

    leitor = csv.DictReader(linhas_dados)
    colunas = {c: [] for c in (leitor.fieldnames or [])}
    n_total = 0
    n_validas = 0
    for reg in leitor:
        n_total += 1
        valido = reg.get("valido", "1").strip() in ("1", "True", "true")
        n_validas += int(valido)
        for c, v in reg.items():
            colunas[c].append((v, valido))
    return meta, colunas, n_total, n_validas


def serie_valida(colunas, nome):
    """Extrai a série numérica de uma coluna, só das amostras válidas."""
    fora = []
    for v, valido in colunas[nome]:
        if not valido or v == "":
            continue
        try:
            fora.append(float(v))
        except ValueError:
            continue
    return np.asarray(fora, dtype=float)


def detrend(y, modo="linear"):
    """Remove a tendência determinística, deixando só o resíduo."""
    if modo == "none" or len(y) < 3:
        return y - np.mean(y)
    grau = {"linear": 1, "poly2": 2, "poly3": 3}.get(modo, 1)
    if len(y) <= grau + 1:
        return y - np.mean(y)
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, grau)
    return y - np.polyval(coef, x)


def autocorrelacao(y, max_lag=40):
    """r_k para k = 0..max_lag, normalizada (r_0 = 1)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return np.array([1.0])
    d = y - np.mean(y)
    denom = float(np.dot(d, d))
    if denom <= 0:
        return np.array([1.0])
    max_lag = int(min(max_lag, n - 2))
    return np.array([float(np.dot(d[: n - k], d[k:]) / denom)
                     for k in range(max_lag + 1)])


def veredicto_brancura(r, n, alpha=0.05):
    """O ruído é branco?"""
    banda = 1.96 / math.sqrt(n) if n > 0 else float("inf")
    lags = r[1:]
    if len(lags) == 0:
        return True, {"banda": banda, "frac_fora": 0.0, "r1": 0.0, "n_lags": 0}
    fora = int(np.sum(np.abs(lags) > banda))
    frac = fora / len(lags)
    r1 = float(lags[0])

    # Dois critérios independentes; basta um falhar para não ser branco.
    ok_frac = frac <= max(3 * alpha, 0.15)
    ok_r1 = abs(r1) <= banda * 2.0
    return (ok_frac and ok_r1), {
        "banda": banda, "frac_fora": frac, "r1": r1,
        "n_lags": len(lags), "n_fora": fora,
        "ok_frac": ok_frac, "ok_r1": ok_r1,
    }


def analisar_coluna(nome, y, dt_medio, modo_detrend="linear", max_lag=40):
    """Estatísticas completas de um canal. Retorna dict."""
    if nome in ANGULARES:
        y = np.unwrap(y)  # antes de qualquer estatistica

    res = detrend(y, modo_detrend)
    n = len(res)
    sigma = float(np.std(res, ddof=1)) if n > 1 else 0.0
    r = autocorrelacao(res, max_lag)
    branco, diag = veredicto_brancura(r, n)

    # Assimetria e curtose: quão longe de gaussiano.
    if n > 3 and sigma > 0:
        z = res / sigma
        assimetria = float(np.mean(z ** 3))
        curtose = float(np.mean(z ** 4) - 3.0)
    else:
        assimetria = curtose = 0.0

    return {
        "coluna": nome, "n": n, "media_bruta": float(np.mean(y)) if n else 0.0,
        "sigma": sigma, "R": sigma ** 2,
        "min": float(np.min(y)) if n else 0.0,
        "max": float(np.max(y)) if n else 0.0,
        "r": r, "branco": branco, "diag": diag,
        "assimetria": assimetria, "curtose": curtose,
        "dt_medio": dt_medio,
    }


def sigma_derivada(y, dt, sigma_y, ordem=1):
    """
    Estima o sigma da derivada de `y`, descontando a amplificação do ruído.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < ordem + 2 or dt <= 0:
        return None

    d = y.copy()
    for _ in range(ordem):
        d = np.diff(d) / dt

    sigma_bruto = float(np.std(d, ddof=1))

    # Propagação do ruído por `ordem` diferenciações sucessivas.
    # 1a diferença: var *= 2/dt^2 ; 2a diferença: var *= 6/dt^4 (coef 1,-2,1)
    fator = {1: 2.0 / dt ** 2, 2: 6.0 / dt ** 4, 3: 20.0 / dt ** 6}.get(ordem)
    if fator is None:
        return None
    var_ruido = (sigma_y ** 2) * fator
    var_corr = sigma_bruto ** 2 - var_ruido

    return {
        "ordem": ordem,
        "sigma_bruto": sigma_bruto,
        "sigma_ruido": math.sqrt(var_ruido),
        "sigma_corrigido": math.sqrt(var_corr) if var_corr > 0 else 0.0,
        "dominado_por_ruido": var_corr <= 0,
        "fracao_ruido": var_ruido / (sigma_bruto ** 2) if sigma_bruto > 0 else 1.0,
    }


def relatorio_derivada(nome, d, unidade_base):
    """Imprime o resultado de `sigma_derivada`, com a crítica do resultado."""
    if d is None:
        return
    rotulo = {1: "1a derivada", 2: "2a derivada", 3: "3a derivada"}[d["ordem"]]
    unid = unidade_base + "/s" * d["ordem"]
    print("  --- {} de {} [{}] ---".format(rotulo, nome, unid))
    print("    sigma bruto (derivada direta) : {:.4f}".format(d["sigma_bruto"]))
    print("    parcela que e AMPLIFICACAO    : {:.4f}  ({:.0f}% da variancia)".format(
        d["sigma_ruido"], 100 * d["fracao_ruido"]))
    if d["dominado_por_ruido"]:
        print("    sigma corrigido               : INDETERMINADO")
        print("    O ruido de medicao domina inteiramente a derivada. Este")
        print("    ensaio NAO tem informacao sobre essa grandeza. Repetir com")
        print("    manobras mais vigorosas, ou aceitar que o valor nao e")
        print("    mensuravel por este caminho.")
    else:
        print("    sigma CORRIGIDO               : {:.4f}   <- use este".format(
            d["sigma_corrigido"]))
        if d["fracao_ruido"] > 0.8:
            print("    ATENCAO: {:.0f}% da variancia era ruido. A correcao e".format(
                100 * d["fracao_ruido"]))
            print("    grande demais para o resultado ser confiavel. Prefira um")
            print("    ensaio com aceleracoes mais fortes.")
    print()

def relatorio(res, meta):
    """Imprime o relatório de um canal."""
    nome = res["coluna"]
    alvo = ALVO_CONFIG.get(nome)
    print("--- " + nome + " " + "-" * max(0, 62 - len(nome)))
    print("  amostras validas : {}".format(res["n"]))
    print("  media (bruta)    : {:+.6f}".format(res["media_bruta"]))
    print("  faixa            : {:+.6f} .. {:+.6f}".format(res["min"], res["max"]))
    print("  SIGMA (residuo)  : {:.6f}".format(res["sigma"]))
    print("  R = sigma^2      : {:.3e}".format(res["R"]))
    if alvo:
        const, unid = alvo
        print("  -> alimenta {} [{}]".format(const, unid))
        if unid == "rad":
            print("     equivalente  : {:.3f} graus".format(math.degrees(res["sigma"])))

    d = res["diag"]
    print("  autocorrelacao   : r_1 = {:+.4f}   banda 95% = +-{:.4f}".format(
        d["r1"], d["banda"]))
    print("                     {}/{} lags fora ({:.1f}%, esperado ~5%)".format(
        d.get("n_fora", 0), d["n_lags"], 100 * d["frac_fora"]))
    if res["branco"]:
        print("  VEREDICTO        : ruido BRANCO — R e utilizavel")
    else:
        print("  VEREDICTO        : *** NAO E BRANCO ***")
        if not d["ok_r1"]:
            print("     r_1 alto: amostras vizinhas correlacionadas.")
            print("     Suspeitas, nesta ordem:")
            print("       1. media movel no caminho do dado")
            print("       2. leitura repetida sendo contada como nova")
            print("       3. o sensor e mesmo mais lento que a taxa de amostragem")
        if not d["ok_frac"]:
            print("     muitos lags fora da banda: ha estrutura no ruido.")
        print("     R calculado daqui SUBESTIMA a incerteza. Nao usar ainda.")

    print("  assimetria       : {:+.3f}  (0 = simetrico)".format(res["assimetria"]))
    print("  curtose          : {:+.3f}  (0 = gaussiano)".format(res["curtose"]))
    print()


def figura(resultados, caminho_png, titulo=""):
    """Três painéis por canal: série temporal, histograma e autocorrelação."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(resultados)
    if n == 0:
        return None
    fig, axs = plt.subplots(n, 3, figsize=(15, 2.8 * n), squeeze=False)
    for i, res in enumerate(resultados):
        # --- painel 1: a série no tempo, com a tendência removida sobreposta
        ax = axs[i][0]
        y = res.get("serie")
        ts = res.get("ts")
        if y is not None and len(y) > 1:
            x = ts if (ts is not None and len(ts) == len(y)) else np.arange(len(y))
            rot = "tempo [s]" if ts is not None else "amostra"
            ax.plot(x, y, lw=0.7, color="tab:red", label="bruto")
            tend = y - res["residuo"]
            ax.plot(x, tend, lw=1.6, color="k", ls="--", label="tendencia removida")
            ax.set_xlabel(rot, fontsize=8)
            ax.legend(fontsize=7)
            faixa = float(np.max(y) - np.min(y))
            variacao = 100 * faixa / abs(float(np.mean(y))) if np.mean(y) else 0.0
            titulo_p = "{} — serie no tempo".format(res["coluna"])
            sub = "faixa {:.4f} ({:.0f}% da media)".format(faixa, variacao)
            ax.set_title(titulo_p + chr(10) + sub, fontsize=9)
        ax.grid(alpha=0.3)

        ax = axs[i][1]
        y = res.get("residuo")
        if y is not None and len(y) > 1:
            ax.hist(y, bins=40, color="tab:blue", alpha=0.75, density=True)
            s = res["sigma"]
            if s > 0:
                xs = np.linspace(float(np.min(y)), float(np.max(y)), 200)
                pdf = np.exp(-0.5 * (xs / s) ** 2) / (s * math.sqrt(2 * math.pi))
                ax.plot(xs, pdf, "r--", lw=1.5,
                        label="N(0, {:.4f}^2)".format(s))
                ax.legend(fontsize=7)
        ax.set_title(res["coluna"] + " — histograma do residuo", fontsize=9)
        ax.grid(alpha=0.3)

        ax = axs[i][2]
        r = res["r"]
        lags = np.arange(len(r))
        ax.stem(lags, r, basefmt=" ")
        banda = res["diag"]["banda"]
        ax.axhspan(-banda, banda, color="green", alpha=0.15,
                   label="banda 95% (+-{:.3f})".format(banda))
        ax.axhline(0, color="k", lw=0.8)
        cor = "green" if res["branco"] else "red"
        rotulo = "BRANCO" if res["branco"] else "NAO BRANCO"
        ax.set_title(res["coluna"] + " — autocorrelacao (" + rotulo + ")",
                     fontsize=9, color=cor)
        ax.set_xlabel("lag")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    if titulo:
        fig.suptitle(titulo, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(caminho_png, dpi=110)
    plt.close(fig)
    return caminho_png


def analisar_arquivo(caminho, janela=None, modo_detrend="linear",
                     max_lag=40, png=True, derivar=False, sigma_sensor=None):
    meta, colunas, n_total, n_validas = ler_csv(caminho)

    print("=" * 70)
    print("ANALISE — " + os.path.basename(caminho))
    print("=" * 70)
    for k, v in meta.items():
        print("  {}: {}".format(k, v))
    taxa = 100.0 * n_validas / n_total if n_total else 0.0
    print("  amostras: {} total, {} validas ({:.1f}%)".format(
        n_total, n_validas, taxa))

    if "dt_medido_s" in colunas:
        dts = serie_valida(colunas, "dt_medido_s")
    else:
        dts = np.array([])
    dt_medio = float(np.median(dts[1:])) if len(dts) > 2 else 0.0
    if dt_medio > 0:
        print("  dt mediano: {:.1f} ms ({:.1f} Hz)".format(
            dt_medio * 1000, 1 / dt_medio))
    print("  detrend: " + modo_detrend)
    if janela:
        print("  janela: {:.1f}s .. {:.1f}s".format(janela[0], janela[1]))
    print()

    # recorte por janela de tempo, se pedido
    mascara = None
    if janela and "t_s" in colunas:
        ts_all = np.array([float(v) if v else np.nan
                           for v, _ in colunas["t_s"]], dtype=float)
        mascara = (ts_all >= janela[0]) & (ts_all <= janela[1])

    resultados = []
    for nome in colunas:
        if nome in NAO_ANALISAR:
            continue
        # Uma coluna de texto que não esteja na lista de exclusão não pode
        # derrubar a análise das outras.
        try:
            if mascara is not None:
                pares = [p for p, m in zip(colunas[nome], mascara) if m]
                y = np.asarray([float(v) for v, ok in pares if ok and v != ""],
                               dtype=float)
            else:
                y = serie_valida(colunas, nome)
        except ValueError:
            print("--- {}: coluna nao numerica, pulando\n".format(nome))
            continue
        if len(y) < 10:
            print("--- {}: apenas {} amostras validas, pulando\n".format(nome, len(y)))
            continue
        res = analisar_coluna(nome, y, dt_medio, modo_detrend, max_lag)
        yy = np.unwrap(y) if nome in ANGULARES else y
        res["residuo"] = detrend(yy, modo_detrend)
        res["serie"] = yy
        res["ts"] = None
        relatorio(res, meta)

        # Ruído de processo (Q): vem da derivada, e não da própria série.
        # Só faz sentido para os canais listados em DERIVAR_PARA.
        if derivar and nome in DERIVAR_PARA and dt_medio > 0:
            alvo_q, unid_base = DERIVAR_PARA[nome]
            # Qual sigma usar para descontar a amplificação do ruído?
            sig_ruido = sigma_sensor if sigma_sensor is not None else res["sigma"]
            print("  >>> ruido de PROCESSO a partir deste canal ({}):".format(alvo_q))
            if sigma_sensor is not None:
                print("      usando sigma_sensor={:.5f} (informado)".format(sigma_sensor))
            else:
                print("      usando sigma={:.5f} desta propria serie.".format(res["sigma"]))
                print("      ATENCAO: se este ensaio tem variacao REAL da grandeza,")
                print("      este valor inclui o sinal e a correcao vai subtrair")
                print("      demais. Informe --sigma-sensor com o valor medido")
                print("      num ensaio estavel.")
            for ordem in (1, 2):
                d = sigma_derivada(yy, dt_medio, sig_ruido, ordem=ordem)
                relatorio_derivada(nome, d, unid_base)

        resultados.append(res)

    if png and resultados:
        saida = os.path.splitext(caminho)[0] + "_analise.png"
        try:
            figura(resultados, saida, titulo=os.path.basename(caminho))
            print("Figura: " + saida)
        except Exception as e:
            print("(figura nao gerada: {}: {})".format(type(e).__name__, e))

    nao_brancos = [r["coluna"] for r in resultados if not r["branco"]]
    print()
    if nao_brancos:
        print("ATENCAO: canais NAO brancos: " + ", ".join(nao_brancos))
        print("Os R desses canais nao devem ir para config.py ainda.")
    elif resultados:
        print("Todos os canais passaram no teste de brancura.")
        print("Os SIGMA acima podem ir para kalman/config.py.")
    return resultados


def _autoteste():
    """Autoteste com verdade conhecida."""
    print("=== bancada.analise — autoteste com verdade conhecida ===\n")
    rng = np.random.default_rng(42)
    n = 2000
    sigma_verdade = 0.037
    ok_tudo = True

    # --- 1. ruído branco puro
    y = rng.normal(0.0, sigma_verdade, n)
    res = analisar_coluna("branco_puro", y, 0.05, "linear")
    err = abs(res["sigma"] - sigma_verdade) / sigma_verdade
    ok1 = err < 0.05 and res["branco"]
    ok_tudo &= ok1
    print("1) ruido branco puro (sigma verdadeiro = {:.4f})".format(sigma_verdade))
    print("   sigma medido = {:.4f}  (erro {:.1f}%)".format(res["sigma"], 100 * err))
    print("   r_1 = {:+.4f}   veredicto: {}".format(
        res["diag"]["r1"], "BRANCO" if res["branco"] else "NAO BRANCO"))
    print("   -> {}\n".format("OK" if ok1 else "FALHOU"))

    # --- 2. tendência + ruído
    rampa = np.linspace(0.0, 2.0, n)
    y2 = rampa + rng.normal(0.0, sigma_verdade, n)
    res_sem = analisar_coluna("com_rampa", y2, 0.05, "none")
    res_com = analisar_coluna("com_rampa", y2, 0.05, "linear")
    err2 = abs(res_com["sigma"] - sigma_verdade) / sigma_verdade
    ok2 = err2 < 0.05 and res_sem["sigma"] > 5 * sigma_verdade
    ok_tudo &= ok2
    print("2) rampa 0->2 somada ao mesmo ruido")
    print("   SEM detrend: sigma = {:.4f}  <- inflado pela rampa".format(
        res_sem["sigma"]))
    print("   COM detrend: sigma = {:.4f}  (erro {:.1f}%)".format(
        res_com["sigma"], 100 * err2))
    print("   -> {}\n".format("OK" if ok2 else "FALHOU"))

    # --- 3. média móvel
    k = 5
    branco = rng.normal(0.0, sigma_verdade, n + k)
    suave = np.convolve(branco, np.ones(k) / k, mode="valid")
    res3 = analisar_coluna("media_movel", suave, 0.05, "linear")
    ok3 = not res3["branco"]
    ok_tudo &= ok3
    print("3) mesmo ruido passado por MEDIA MOVEL de {} amostras".format(k))
    print("   r_1 = {:+.4f}  (banda +-{:.4f})".format(
        res3["diag"]["r1"], res3["diag"]["banda"]))
    print("   {}/{} lags fora da banda".format(
        res3["diag"]["n_fora"], res3["diag"]["n_lags"]))
    print("   veredicto: {}".format("BRANCO" if res3["branco"] else "NAO BRANCO"))
    print("   -> {} (tinha que ACUSAR)\n".format("OK" if ok3 else "FALHOU"))

    # --- 4. leitura repetida
    base = rng.normal(0.0, sigma_verdade, n // 2)
    repetido = np.repeat(base, 2)  # cada leitura contada duas vezes
    res4 = analisar_coluna("repetido", repetido, 0.05, "linear")
    ok4 = not res4["branco"]
    ok_tudo &= ok4
    print("4) cada leitura repetida 2x")
    print("   sigma medido = {:.4f}  (verdadeiro {:.4f})".format(
        res4["sigma"], sigma_verdade))
    print("   r_1 = {:+.4f}  (banda +-{:.4f})".format(
        res4["diag"]["r1"], res4["diag"]["banda"]))
    print("   veredicto: {}".format("BRANCO" if res4["branco"] else "NAO BRANCO"))
    print("   -> {} (tinha que ACUSAR)\n".format("OK" if ok4 else "FALHOU"))

    # --- 5. ângulo com wraparound
    yaw = np.linspace(-3.0, 3.0, n) + rng.normal(0.0, 0.01, n)
    yaw_wrap = (yaw + np.pi) % (2 * np.pi) - np.pi
    res5 = analisar_coluna("yaw_rad", yaw_wrap, 0.05, "linear")
    ok5 = res5["sigma"] < 0.05
    ok_tudo &= ok5
    print("5) yaw atravessando +-pi (unwrap obrigatorio)")
    print("   sigma medido = {:.5f}  (sem unwrap daria ~2)".format(res5["sigma"]))
    print("   -> {}\n".format("OK" if ok5 else "FALHOU"))

    print("=== RESULTADO: {} ===".format("OK" if ok_tudo else "FALHOU"))
    return ok_tudo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analise dos CSVs de coleta.py — sigma, R e teste de brancura")
    ap.add_argument("csv", nargs="*", help="arquivo(s) CSV de bancada.coleta")
    ap.add_argument("--janela", nargs=2, type=float, metavar=("T0", "T1"),
                    help="analisa so o trecho entre T0 e T1 segundos. Use para "
                         "isolar a parte ESTACIONARIA do ensaio — medir sigma "
                         "num trecho onde a grandeza varia infla o resultado.")
    ap.add_argument("--detrend", default="linear",
                    choices=["none", "linear", "poly2", "poly3"],
                    help="remocao de tendencia (padrao linear). NUNCA e media "
                         "movel: ela introduziria a correlacao que o teste "
                         "de brancura procura.")
    ap.add_argument("--max-lag", type=int, default=40, dest="max_lag",
                    help="lags da autocorrelacao (padrao 40)")
    ap.add_argument("--sem-png", action="store_true", dest="sem_png",
                    help="nao gera figura")
    ap.add_argument("--derivar", action="store_true",
                    help="alem do R, estima o ruido de PROCESSO (Q) derivando "
                         "a serie, com desconto da amplificacao do ruido. "
                         "Use num ensaio de conducao REPRESENTATIVA, nao num "
                         "ensaio parado.")
    ap.add_argument("--sigma-sensor", type=float, default=None, dest="sigma_sensor",
                    help="ruido do SENSOR [mesma unidade do canal], medido num "
                         "ensaio ESTAVEL. Usado com --derivar para descontar a "
                         "amplificacao. Sem isto, usa o sigma da propria serie — "
                         "o que subtrai demais se a grandeza variou de verdade.")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste com verdade conhecida, sem CSV")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    if not args.csv:
        ap.error("informe ao menos um CSV (ou use --fake)")

    # `--sigma-sensor` só tem efeito com `--derivar`.
    if args.sigma_sensor is not None and not args.derivar:
        print("[analise] --sigma-sensor foi informado sem --derivar.")
        print("[analise] Ele so serve para descontar a amplificacao na derivada,")
        print("[analise] entao --derivar foi ligado automaticamente.")
        print()
        args.derivar = True

    for caminho in args.csv:
        analisar_arquivo(caminho, janela=args.janela,
                         modo_detrend=args.detrend, max_lag=args.max_lag,
                         png=not args.sem_png, derivar=args.derivar,
                         sigma_sensor=args.sigma_sensor)
        print()
