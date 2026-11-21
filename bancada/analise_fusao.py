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
"""Roda as sete configurações do filtro fundido sobre um arquivo real."""
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
import sys

import numpy as np

from kalman.config import (GATE_ESCAPE_FATOR, GATE_ESCAPE_N,
                           GATING_THRESHOLD_99)
from kalman.core import KalmanFilter
from kalman.filtros import fusao, fusao_giro
from kalman.filtros.fusao import (ANGLE_INDICES, CH_ENCODER, CH_GYRO_Y,
                                  CH_GYRO_Z, CH_MAG, CH_TILT, CH_ULTRASSOM,
                                  DIM_X, IPSI, ITHETA, IV, IX, IY)
from kalman.movimento import EstadoMovimento
from kalman.utils import wrap

V_PARADO = 0.05  # [m/s] abaixo disto o encoder considera o veículo parado
N_PARADO_MIN = 20   # amostras seguidas abaixo de V_PARADO para contar parada


# Leitura
def _num(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ler(caminho):
    """Cabeçalho e amostras do CSV, aceitando os dois formatos de coleta."""
    cab = {}
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    for l in linhas:
        if l.startswith("#") and ":" in l:
            k, v = l[1:].split(":", 1)
            cab[k.strip()] = v.strip()

    leitor = csv.DictReader(l for l in linhas if not l.startswith("#"))
    campos = leitor.fieldnames or []
    tipo = "pilotado" if "delta_cmd_rad" in campos else "corrida"

    falta = [c for c in ("t_s", "dt_medido_s", "vel_m_s", "yaw_rad",
                         "pitch_rad", "gy_rads", "gz_rads", "dist_m")
             if c not in campos]
    if falta:
        raise SystemExit("[analise] o CSV nao tem as colunas %s.\n"
                         "          Isto e coleta de sensor unico? Use o CSV "
                         "do corrida.py ou do pilotar.py." % falta)

    delta_fixo = 0.0
    if tipo == "corrida":
        g = _num(cab.get("steer_cmd_deg", "0"))
        delta_fixo = np.deg2rad(g if g is not None else 0.0)

    amostras = []
    for r in leitor:
        if not r.get("t_s"):
            continue
        d_cmd = (_num(r.get("delta_cmd_rad")) if tipo == "pilotado"
                 else delta_fixo)
        amostras.append({
            "t": float(r["t_s"]),
            "dt": _num(r["dt_medido_s"]) or 0.05,
            "enc_ok": r.get("enc_valido") == "1",
            "v": _num(r["vel_m_s"]),
            "imu_ok": r.get("imu_valido") == "1",
            "ax": _num(r.get("ax_ms2")),
            "ay": _num(r.get("ay_ms2")),
            "az": _num(r.get("az_ms2")),
            "pitch": _num(r["pitch_rad"]),
            "yaw": _num(r["yaw_rad"]),
            "gy": _num(r["gy_rads"]),
            "gz": _num(r["gz_rads"]),
            "us_ok": r.get("us_valido") == "1",
            "us_novo": r.get("us_novo") == "1",
            "d": _num(r["dist_m"]),
            "delta": 0.0 if d_cmd is None else d_cmd,
            "marca": int(_num(r.get("marca")) or 0),
        })
    if not amostras:
        raise SystemExit("[analise] nenhuma amostra no arquivo")
    return cab, amostras, tipo


# O núcleo: uma passada do filtro sobre as amostras
def rodar(amostras, sensores, pose0, v0, x_parede=None, nivel=None,
          escapatoria=None, trim_steer=0.0, formulacao="variante",
          suspende_parado=False, suspende_giro_parado=False, zupt=None,
          sigma_pos0=None, gate_encoder=False, n_escape_encoder=12):
    """Uma passada de uma configuração sobre as amostras reais."""
    if formulacao not in fusao_giro.FORMULACOES:
        raise ValueError("formulacao deve ser 'variante' ou 'vigente'")
    variante = formulacao == "variante"
    obs = fusao_giro.observation if variante else fusao.observation
    if nivel is None:
        nivel = fusao_giro.NIVEL if variante else "arfagem"
    if escapatoria is None:
        escapatoria = fusao_giro.usa_escapatoria(sensores) if variante else True
    kf = KalmanFilter(dim_x=DIM_X, angle_indices=ANGLE_INDICES,
                      gate_escape_after=(GATE_ESCAPE_N if escapatoria
                                         else None),
                      gate_escape_factor=GATE_ESCAPE_FATOR)
    kw0 = {} if sigma_pos0 is None else dict(sigma_pos0=sigma_pos0)
    x0, P0 = fusao.initial(sensores, pose0=pose0, v0=v0, nivel=nivel,
                           x_parede=x_parede, **kw0)
    kf.x, kf.P = x0.copy(), P0.copy()
    if zupt is None:
        zupt = variante

    det = EstadoMovimento()
    n = len(amostras)
    x_est = np.zeros((n, DIM_X))
    P_hist = np.zeros((n, DIM_X, DIM_X))
    n_rej, n_us = 0, 0
    rej_enc_seguidas, n_rej_enc = 0, 0

    for k, a in enumerate(amostras):
        dt = a["dt"] if a["dt"] > 1e-6 else 0.05
        d_cmd = float(a["delta"]) + float(trim_steer)

        # O detector recebe a leitura, e não a verdade, como no veículo.
        if a["imu_ok"] and None not in (a["ax"], a["ay"], a["az"]):
            em_mov = det.atualiza(EstadoMovimento.norma(a["ax"], a["ay"],
                                                        a["az"]))
        else:
            em_mov = det.atualiza(None)

        if k > 0:
            if variante:
                gz = (float(a["gz"]) if (a["imu_ok"] and a["gz"] is not None)
                      else None)
                ax = (float(a["ax"]) if (a["imu_ok"] and a["ax"] is not None)
                      else None)
                parado = suspende_parado and fusao_giro.esta_parado(
                    sensores, em_mov, a["v"] if a["enc_ok"] else None)
                kf.F, kf.Q, kf.f = fusao_giro.passo(
                    kf.x, dt, d_cmd, sensores, gz=gz, em_movimento=em_mov,
                    nivel=nivel, ax=ax, parado=parado,
                    suspende_giro=suspende_giro_parado)
            else:
                kf.F, kf.Q = fusao.matrices(kf.x, dt, d_cmd, nivel=nivel)
                kf.f = (lambda dd: (lambda xx, uu: fusao.propaga(
                    xx, dt, dd, nivel=nivel)))(d_cmd)
            kf.predict()

        z_por_canal = {
            CH_ENCODER: a["v"] if a["enc_ok"] else None,
            CH_MAG: a["yaw"] if a["imu_ok"] else None,
            CH_GYRO_Z: a["gz"] if a["imu_ok"] else None,
            CH_GYRO_Y: a["gy"] if a["imu_ok"] else None,
            CH_TILT: a["pitch"] if a["imu_ok"] else None,
            CH_ULTRASSOM: (a["d"] if (a["us_ok"] and a["us_novo"]) else None),
        }
        if z_por_canal[CH_ULTRASSOM] is not None:
            n_us += 1

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
            if gate_encoder and nome == CH_ENCODER:
                usa_gate = rej_enc_seguidas < n_escape_encoder
            aceito, _ = kf.update_scalar(
                z, H, R, is_angle=(nome == CH_MAG), gate=usa_gate,
                gate_threshold=GATING_THRESHOLD_99, channel=nome,
                z_pred=z_pred)
            if nome == CH_ENCODER:
                rej_enc_seguidas = 0 if aceito else rej_enc_seguidas + 1
                if not aceito:
                    n_rej_enc += 1
            elif not aceito:
                n_rej += 1

        # ZUPT + ZARU com o carro parado. Pseudo-medidas lineares, recalculadas
        # a cada canal, como os demais, para o z_pred ser o atual.
        if zupt and variante and fusao_giro.esta_parado(
                sensores, em_mov, a["v"] if a["enc_ok"] else None):
            z_pseudo = {fusao_giro.CH_ZUPT: 0.0,
                        fusao_giro.CH_ZARU: (a["gz"] if (a["imu_ok"] and a["gz"] is not None)
                                             else None)}
            for nome in (fusao_giro.CH_ZUPT, fusao_giro.CH_ZARU):
                z = z_pseudo[nome]
                if z is None:
                    continue
                atual = [c for c in fusao_giro.canais_parado(kf.x, sensores) if c[0] == nome]
                if atual:
                    _, H, R, _, z_pred = atual[0]
                    kf.update_scalar(z, H, R, gate=False, channel=nome, z_pred=z_pred)

        x_est[k] = kf.x.ravel()
        P_hist[k] = kf.P

    return dict(x_est=x_est, P_hist=P_hist, n_rejeitados=n_rej,
                n_us_validos=n_us, n_rej_enc=n_rej_enc, nis=dict(kf.nis_history),
                escapes=dict(kf.escape_count), kf=kf)


# Instantes de referência, do encoder
def psi_ref_automatico(amostras, n_min=N_PARADO_MIN):
    """A leitura do magnetômetro que corresponde ao eixo `x` do mapa."""
    quietos = []
    for a in amostras:
        # Amostra de encoder inválida não é evidência de movimento: pula, em
        # vez de encerrar o trecho parado.
        if not a["enc_ok"] or a["v"] is None:
            continue
        if abs(a["v"]) > V_PARADO:
            break
        if a["imu_ok"] and a["yaw"] is not None:
            quietos.append(float(a["yaw"]))
    if len(quietos) < n_min:
        return None
    # mediana circular: a aritmética erraria com leituras em torno de +-pi
    s = np.median(np.sin(quietos))
    c = np.median(np.cos(quietos))
    return float(np.arctan2(s, c))


def aplica_psi_ref(amostras, psi_ref):
    """
    Desconta o `psi_ref` do yaw de todas as amostras, devolvendo uma cópia.
    """
    if not psi_ref:
        return amostras
    saida = []
    for a in amostras:
        b = dict(a)
        if b["yaw"] is not None:
            b["yaw"] = wrap(float(b["yaw"]) - psi_ref)
        saida.append(b)
    return saida


def instante_da_parada(amostras):
    """Última amostra com movimento, pelo encoder."""
    movs = [i for i, a in enumerate(amostras)
            if a["enc_ok"] and a["v"] is not None and abs(a["v"]) > V_PARADO]
    if not movs:
        raise SystemExit("[analise] o veiculo nao se moveu neste arquivo")
    return movs[-1]


def paradas(amostras, n_min=N_PARADO_MIN):
    """Índices do meio de cada trecho parado, pelo encoder."""
    paras, i, n = [], 0, len(amostras)
    while i < n:
        a = amostras[i]
        quieto = (a["enc_ok"] and a["v"] is not None
                  and abs(a["v"]) <= V_PARADO)
        if quieto:
            j = i
            while j < n:
                b = amostras[j]
                if not (b["enc_ok"] and b["v"] is not None
                        and abs(b["v"]) <= V_PARADO):
                    break
                j += 1
            if (j - i) >= n_min:
                paras.append((i + j) // 2)
            i = j
        else:
            i += 1
    return paras


def marcas(amostras):
    """Índices do meio de cada valor da coluna `marca`, se ela existir."""
    vistos, fora = {}, []
    for i, a in enumerate(amostras):
        m = a["marca"]
        if m > 0:
            vistos.setdefault(m, []).append(i)
    for m in sorted(vistos):
        idx = vistos[m]
        fora.append((m, idx[len(idx) // 2]))
    return fora


# Métrica 1: frenagem em reta
def metrica_frenagem(amostras, d0, dfinal, x_parede=6.0, escapatoria=None,
                     trim_steer=0.0, formulacao="variante"):
    """Erro de posição no instante da parada, nas sete configurações."""
    i_par = instante_da_parada(amostras)
    if dfinal is not None:
        d_ref, fonte = float(dfinal), "trena"
    else:
        depois = [a["d"] for a in amostras[i_par:]
                  if a["us_ok"] and a["us_novo"] and a["d"] is not None]
        if len(depois) < 10:
            raise SystemExit(
                "[analise] sem --dfinal e sem ultrassom apos a parada.\n"
                "          Meca a distancia final com trena.")
        d_ref, fonte = float(np.median(depois)), "ULTRASSOM (SUBSTITUTO)"

    x_verdade = x_parede - d_ref
    pose0 = (x_parede - float(d0), 0.0, 0.0)

    print("")
    print("=" * 76)
    print("FRENAGEM EM RETA — erro no instante da parada")
    print("=" * 76)
    print("  parada na amostra %d (t = %.2f s), pelo encoder"
          % (i_par, amostras[i_par]["t"]))
    print("  largada a %.3f m da parede (trena)" % d0)
    print("  parou a  %.3f m da parede  [%s]" % (d_ref, fonte))
    if fonte != "trena":
        print("  ATENCAO: sem trena final, as configuracoes com ultrassom sao")
        print("           julgadas contra o proprio sensor delas. Nao usar")
        print("           este numero no resultado do trabalho.")
    print("")
    print("  %-19s %11s %11s %9s %8s"
          % ("configuracao", "x estimado", "erro", "3 sigma", "rejeic."))
    print("  " + "-" * 64)

    linhas = []
    for rotulo, sens in fusao.configuracoes():
        out = rodar(amostras, sens, pose0, 0.0, x_parede=x_parede,
                    escapatoria=escapatoria, trim_steer=trim_steer,
                    formulacao=formulacao)
        xe = out["x_est"][i_par, IX]
        err = xe - x_verdade
        sig = float(np.sqrt(max(out["P_hist"][i_par][IX, IX], 1e-18)))
        linhas.append((rotulo, err, sig, out))
        print("  %-19s %9.4f m %8.1f mm %7.1f mm %7d"
              % (rotulo, xe, 1000 * err, 3000 * sig, out["n_rejeitados"]))
    return linhas


# Métrica 2: trajetória com marcos medidos
def le_marcos(caminho):
    """
    Marcos medidos com trena: CSV com colunas `n,x,y` e `psi_deg` opcional.
    """
    fora = []
    with open(caminho, encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if not r.get("x"):
                continue
            fora.append(dict(n=int(_num(r.get("n")) or (len(fora) + 1)),
                             x=float(r["x"]), y=float(r["y"]),
                             psi=(np.deg2rad(_num(r["psi_deg"]))
                                  if r.get("psi_deg") else None)))
    if not fora:
        raise SystemExit("[analise] nenhum marco no arquivo de marcos")
    return fora


def metrica_trajetoria(amostras, marcos_lista, pose0, x_parede=None,
                       escapatoria=None, trim_steer=0.0, formulacao="variante"):
    """Erro de posição em cada marco medido, nas sete configurações."""
    idx_marca = marcas(amostras)
    if idx_marca:
        casados = []
        por_n = {n: i for n, i in idx_marca}
        for m in marcos_lista:
            if m["n"] in por_n:
                casados.append((m, por_n[m["n"]]))
        origem = "coluna `marca` do CSV"
    else:
        paras = paradas(amostras)
        casados = list(zip(marcos_lista, paras))
        origem = "paradas detectadas pelo encoder"
        if len(paras) != len(marcos_lista):
            print("  AVISO: %d marcos medidos contra %d paradas detectadas."
                  % (len(marcos_lista), len(paras)))
            print("         Casados na ordem, so os %d primeiros."
                  % len(casados))

    if not casados:
        raise SystemExit(
            "[analise] nenhum marco casou com o arquivo.\n"
            "          Com a coluna `marca`, confira os numeros; sem ela, o\n"
            "          carro precisa PARAR (>= %d amostras) em cada marco."
            % N_PARADO_MIN)

    print("")
    print("=" * 76)
    print("TRAJETORIA — erro de posicao nos marcos medidos")
    print("=" * 76)
    print("  %d marcos, casados pela %s" % (len(casados), origem))
    print("  largada em (%.3f, %.3f) m, psi = %.2f graus"
          % (pose0[0], pose0[1], np.rad2deg(pose0[2])))
    print("")
    print("  %-19s %9s %9s %9s %9s"
          % ("configuracao", "marco 1", "...", "ultimo", "rms"))
    print("  " + "-" * 62)

    linhas = []
    for rotulo, sens in fusao.configuracoes():
        out = rodar(amostras, sens, pose0, 0.0, x_parede=x_parede,
                    escapatoria=escapatoria, trim_steer=trim_steer,
                    formulacao=formulacao)
        errs = []
        for m, i in casados:
            ex = out["x_est"][i, IX] - m["x"]
            ey = out["x_est"][i, IY] - m["y"]
            errs.append(float(np.hypot(ex, ey)))
        errs = np.array(errs)
        linhas.append((rotulo, errs, out))
        print("  %-19s %7.1f mm %7s %7.1f mm %7.1f mm"
              % (rotulo, 1000 * errs[0], "", 1000 * errs[-1],
                 1000 * np.sqrt((errs ** 2).mean())))

    print("")
    print("  erro em CADA marco, em mm:")
    print("  %-19s %s" % ("configuracao",
                          " ".join("%7d" % m["n"] for m, _ in casados)))
    print("  " + "-" * (20 + 8 * len(casados)))
    for rotulo, errs, _ in linhas:
        print("  %-19s %s" % (rotulo,
                              " ".join("%7.0f" % (1000 * e) for e in errs)))
    return linhas


# Métrica 3: distância à pista medida (sem parar, sem sincronizar tempo)
def le_pista(caminho):
    """
    A pista medida com trena: CSV com colunas `x,y`, um vértice por linha.
    """
    pts = []
    with open(caminho, encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if not r.get("x"):
                continue
            pts.append((float(r["x"]), float(r["y"])))
    if len(pts) < 2:
        raise SystemExit("[analise] a pista precisa de pelo menos 2 vertices")
    return np.asarray(pts, dtype=float)


def dist_a_pista(px, py, pista):
    """
    Distância do ponto (px, py) à poligonal `pista`, e a abscissa curvilínea.
    """
    a = pista[:-1]
    b = pista[1:]
    ab = b - a
    ap = np.stack([px - a[:, 0], py - a[:, 1]], axis=-1)
    ll = np.sum(ab * ab, axis=1)
    ll = np.where(ll < 1e-12, 1e-12, ll)
    t = np.clip(np.sum(ap * ab, axis=1) / ll, 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = np.hypot(px - proj[:, 0], py - proj[:, 1])
    i = int(np.argmin(d))
    # abscissa: comprimento acumulado até o vértice i, mais a fração no
    # segmento. Diz onde na pista o erro foi maior.
    comp = np.concatenate([[0.0], np.cumsum(np.hypot(ab[:, 0], ab[:, 1]))])
    return float(d[i]), float(comp[i] + t[i] * np.hypot(*ab[i]))


def metrica_pista(amostras, pista, pose0, fim=None, x_parede=None,
                  escapatoria=None, trim_steer=0.0, formulacao="variante"):
    """Erro perpendicular à pista medida, ao longo da corrida inteira."""
    comp_total = float(np.sum(np.hypot(np.diff(pista[:, 0]),
                                       np.diff(pista[:, 1]))))
    i_par = instante_da_parada(amostras)

    print("")
    print("=" * 76)
    print("TRAJETORIA — distancia a PISTA medida")
    print("=" * 76)
    print("  pista: %d vertices, %.3f m de comprimento"
          % (len(pista), comp_total))
    print("  largada em (%.3f, %.3f) m, psi = %.2f graus"
          % (pose0[0], pose0[1], np.rad2deg(pose0[2])))
    print("  parada na amostra %d (t = %.2f s), pelo encoder"
          % (i_par, amostras[i_par]["t"]))
    print("")
    print("  A distancia a pista mede o erro PERPENDICULAR. Ela NAO ve erro AO")
    print("  LONGO da pista, e o piso dela e o quanto o carro saiu da fita --")
    print("  por isso as colunas do fim e do comprimento estao ao lado.")
    print("")
    print("  %-19s %9s %9s %9s %10s %9s"
          % ("configuracao", "mediana", "p90", "maximo", "erro no fim",
             "comprim."))
    print("  " + "-" * 70)

    # O trecho em que o carro andou não depende da configuração: sai do
    # encoder e é o mesmo para as sete.
    movs = [k for k, a in enumerate(amostras)
            if a["enc_ok"] and a["v"] is not None and abs(a["v"]) > V_PARADO]
    k0, k1 = (movs[0], movs[-1]) if movs else (0, len(amostras) - 1)

    linhas = []
    for rotulo, sens in fusao.configuracoes():
        out = rodar(amostras, sens, pose0, 0.0, x_parede=x_parede,
                    escapatoria=escapatoria, trim_steer=trim_steer,
                    formulacao=formulacao)
        xs, ys = out["x_est"][:, IX], out["x_est"][:, IY]
        ds = np.array([dist_a_pista(xs[k], ys[k], pista)[0]
                       for k in range(len(amostras))])
        dsm = ds[k0:k1 + 1]
        comp = float(np.sum(np.hypot(np.diff(xs[k0:k1 + 1]),
                                     np.diff(ys[k0:k1 + 1]))))
        e_fim = (float(np.hypot(xs[i_par] - fim[0], ys[i_par] - fim[1]))
                 if fim is not None else float("nan"))
        linhas.append((rotulo, dsm, e_fim, comp, out))
        print("  %-19s %6.1f mm %6.1f mm %6.1f mm %7s %8.3f m"
              % (rotulo, 1000 * np.median(dsm), 1000 * np.percentile(dsm, 90),
                 1000 * dsm.max(),
                 ("%.1f mm" % (1000 * e_fim)) if fim is not None else "  --  ",
                 comp))

    if fim is None:
        print("")
        print("  SEM --fim: nenhuma coluna aqui pega erro AO LONGO da pista.")
        print("  Meca o (x, y) do eixo traseiro com o carro parado no final.")
    else:
        print("")
        print("  comprimento verdadeiro da pista: %.3f m. A diferenca contra a"
              % comp_total)
        print("  coluna 'comprim.' e erro de ESCALA (raio da roda, patinacao).")

    # por trecho, pela coluna `marca` (o Create marca a troca de trecho)
    trechos = []
    ini = k0
    marca_atual = amostras[k0]["marca"]
    for k in range(k0 + 1, k1 + 1):
        if amostras[k]["marca"] != marca_atual:
            trechos.append((marca_atual, ini, k - 1))
            marca_atual, ini = amostras[k]["marca"], k
    trechos.append((marca_atual, ini, k1))

    if len(trechos) > 1:
        print("")
        print("  POR TRECHO (fronteiras pelo Create; mediana do erro em mm):")
        print("")
        cab = "  %-19s" % "configuracao"
        for m, a, b in trechos:
            cab += " %9s" % ("trecho %d" % m)
        print(cab)
        print("  " + "-" * (20 + 10 * len(trechos)))
        for rotulo, dsm, _, _, out in linhas:
            lin = "  %-19s" % rotulo
            for m, a, b in trechos:
                seg = dsm[a - k0:b - k0 + 1]
                lin += " %9.1f" % (1000 * np.median(seg) if len(seg)
                                   else float("nan"))
            print(lin)
        print("")
        print("  duracao de cada trecho, em s: %s"
              % ", ".join("%d=%.1f" % (m, amostras[b]["t"] - amostras[a]["t"])
                          for m, a, b in trechos))
    else:
        print("")
        print("  (sem coluna `marca` variando: aperte o Create nas transicoes")
        print("   de trecho para que o erro saia separado por reta e curva)")

    return linhas


# Autoteste
def _autoteste():
    print("=== autoteste de bancada.analise_fusao (sem hardware) ===\n")
    ok = True

    def confere(rot, cond):
        nonlocal ok
        print("  %-56s %s" % (rot, "OK" if cond else "FALHOU"))
        ok = ok and bool(cond)

    # --- sintetiza um CSV de frenagem, verdade conhecida
    import os
    import shutil
    import tempfile

    # pasta local _teste_tmp, apagada no fim (ou o temporário do sistema)
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmpdir = os.path.join(raiz, "_teste_tmp")
    try:
        os.makedirs(tmpdir, exist_ok=True)
        with open(os.path.join(tmpdir, ".escrita"), "w") as _f:
            _f.write("ok")
    except OSError:
        tmpdir = tempfile.gettempdir()

    dt, n = 0.05, 200
    x_parede, d0 = 6.0, 4.75
    x = x_parede - d0
    v = 0.0
    linhas = ["# coleta: corrida (teste)", "# steer_cmd_deg: 0.0"]
    cols = ("t_s,dt_medido_s,enc_valido,vel_m_s,imu_valido,ax_ms2,ay_ms2,"
            "az_ms2,roll_rad,pitch_rad,yaw_rad,gx_rads,gy_rads,gz_rads,"
            "us_valido,us_novo,dist_m,us_motivo")
    linhas.append(cols)
    xs = []
    rng = np.random.default_rng(7)
    for k in range(n):
        t = k * dt
        v_ant = v
        if t < 2.0:
            v = 0.0
        elif t < 6.0:
            v = min(0.85, v + 0.4 * dt)
        else:
            v = max(0.0, v - 0.39 * dt)
        # o acelerômetro vê a aceleração verdadeira do passo, mais ruído (na
        # variante ele é a entrada da velocidade)
        a_ver = (v - v_ant) / dt
        x += v * dt
        xs.append(x)
        d_us = (x_parede - x) - 0.3973  # o braço do sensor
        novo = 1 if k % 4 != 3 else 0
        val = 1 if (0.0 < d_us <= 2.0) else 0
        linhas.append(
            "%.6f,%.6f,1,%.6f,1,%.4f,%.4f,%.4f,0,0,0,0,%.5f,%.5f,%d,%d,%s,"
            % (t, dt, v + rng.normal(0, 0.01),
               a_ver + rng.normal(0, 0.2), rng.normal(0, 0.2), 9.81 + rng.normal(0, 0.4),
               rng.normal(0, 0.003), rng.normal(0, 0.003),
               val, novo, ("%.4f" % (d_us + rng.normal(0, 0.018))
                           if val else "")))

    tmp = os.path.join(tmpdir, "teste_analise_fusao.csv")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(linhas) + "\n")

    cab, am, tipo = ler(tmp)
    confere("le o CSV do corrida.py", tipo == "corrida" and len(am) == n)
    confere("delta constante vem do cabecalho",
            all(abs(a["delta"]) < 1e-12 for a in am))

    i_par = instante_da_parada(am)
    confere("acha o instante da parada pelo encoder", 120 < i_par < n)

    x_fim = xs[i_par]
    d_fim = x_parede - x_fim
    linhas_r = metrica_frenagem(am, d0, d_fim, x_parede=x_parede)
    confere("roda as SETE configuracoes", len(linhas_r) == 7)

    por_rot = {r: (e, s) for r, e, s, _ in linhas_r}
    confere("'os tres' erra menos de 10 cm sobre dado sintetico",
            abs(por_rot["os tres"][0]) < 0.10)
    confere("'os tres' erra menos que 'imu' sozinho",
            abs(por_rot["os tres"][0]) < abs(por_rot["imu"][0]))
    confere("a incerteza declarada e finita e positiva",
            all(0 < s < 100 for _, s in por_rot.values()))

    # --- trajetória, com marcas explícitas
    lin2 = list(linhas)
    lin2[2] = cols + ",delta_cmd_rad,marca"
    for i in range(3, len(lin2)):
        k = i - 3
        m = 1 if 60 <= k <= 80 else (2 if 150 <= k <= 170 else 0)
        lin2[i] = lin2[i] + ",0.0,%d" % m
    tmp2 = os.path.join(tmpdir, "teste_analise_traj.csv")
    with open(tmp2, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lin2) + "\n")
    _, am2, tipo2 = ler(tmp2)
    confere("le o CSV do pilotar.py", tipo2 == "pilotado")
    mk = marcas(am2)
    confere("acha as duas marcas", [m for m, _ in mk] == [1, 2])

    marcos_lista = [dict(n=1, x=xs[mk[0][1]], y=0.0, psi=None),
                    dict(n=2, x=xs[mk[1][1]], y=0.0, psi=None)]
    lin_t = metrica_trajetoria(am2, marcos_lista,
                               (x_parede - d0, 0.0, 0.0), x_parede=x_parede)
    confere("roda a trajetoria nas sete", len(lin_t) == 7)
    e3 = {r: e for r, e, _ in lin_t}["os tres"]
    confere("'os tres' erra menos de 15 cm nos marcos", e3.max() < 0.15)

    # --- casamento por paradas, sem coluna marca
    ps = paradas(am)
    confere("detecta a parada final sem a coluna marca", len(ps) >= 1)

    # --- recusa arquivo de sensor unico
    tmp3 = os.path.join(tmpdir, "teste_analise_ruim.csv")
    with open(tmp3, "w", encoding="utf-8", newline="\n") as f:
        f.write("# sensor: ultrassom\nt_s,dt_medido_s,valido,dist_m\n0,0.05,1,1.5\n")
    try:
        ler(tmp3)
        confere("recusa CSV de sensor unico", False)
    except SystemExit as e:
        confere("recusa CSV de sensor unico, dizendo o que falta",
                "colunas" in str(e))

    # --- a métrica da pista, contra geometria conhecida
    quad = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.], [0., 0.]])
    d, _ = dist_a_pista(2.0, 0.0, quad)
    confere("ponto SOBRE a pista da distancia zero", abs(d) < 1e-12)
    d, _ = dist_a_pista(2.0, 0.5, quad)
    confere("0,5 m ao lado da reta da 0,5 m", abs(d - 0.5) < 1e-12)
    d, _ = dist_a_pista(2.0, -0.3, quad)
    confere("o lado nao importa: -0,3 m tambem da 0,3 m", abs(d - 0.3) < 1e-12)
    d, _ = dist_a_pista(-1.0, -1.0, quad)
    confere("fora da ponta do segmento, mede ate o VERTICE",
            abs(d - np.hypot(1, 1)) < 1e-12)
    _, sab = dist_a_pista(4.0, 2.0, quad)
    confere("a abscissa curvilinea acumula os segmentos anteriores",
            abs(sab - 6.0) < 1e-9)
    d, _ = dist_a_pista(2.0, 2.0, quad)
    confere("no centro do quadrado, mede ate o lado MAIS PROXIMO",
            abs(d - 2.0) < 1e-12)

    # Agora a métrica inteira, sobre a corrida sintética em linha reta.
    pista_reta = np.array([[xs[0], 0.0], [xs[-1] + 1.0, 0.0]])
    lin_p = metrica_pista(am, pista_reta, (x_parede - d0, 0.0, 0.0),
                          fim=(xs[i_par], 0.0), x_parede=x_parede)
    confere("roda a metrica da pista nas sete", len(lin_p) == 7)
    por_p = {r: (dsm, ef) for r, dsm, ef, _, _ in lin_p}
    confere("em reta, o erro PERPENDICULAR e pequeno em todas",
            all(dd.max() < 0.05 for dd, _ in por_p.values()))
    confere("mas o erro NO FIM separa as configuracoes, e a perpendicular NAO",
            por_p["os tres"][1] < 0.05 < por_p["imu"][1]
            and por_p["imu"][0].max() < 0.05)

    # --- psi_ref na entrada
    # O CSV sintético grava yaw = 0 com o carro apontado em +x.
    OFFSET = np.deg2rad(-124.0)  # zero do magnetômetro deslocado
    am_desl = [dict(a) for a in am]
    for a in am_desl:
        if a["yaw"] is not None:
            a["yaw"] = wrap(a["yaw"] + OFFSET)

    confere("psi_ref automatico recupera o offset do magnetometro",
            abs(wrap(psi_ref_automatico(am_desl) - OFFSET)) < np.deg2rad(0.5))

    am_corr = aplica_psi_ref(am_desl, psi_ref_automatico(am_desl))
    yaws = [a["yaw"] for a in am_corr[:N_PARADO_MIN] if a["yaw"] is not None]
    confere("apos o desconto o yaw volta a zero na largada",
            max(abs(y) for y in yaws) < np.deg2rad(1.0))

    confere("aplica_psi_ref NAO muta a lista original",
            all(abs(wrap(a["yaw"] - b["yaw"])) < 1e-12
                for a, b in zip(am_desl, [dict(x) for x in am_desl])
                if a["yaw"] is not None))

    confere("psi_ref zero devolve a MESMA lista, sem copiar",
            aplica_psi_ref(am, 0.0) is am)

    confere("sem trecho parado, o automatico devolve None e nao zero",
            psi_ref_automatico(am[i_par:]) is None)

    # Sem o desconto, o filtro propagaria x = v cos(psi) para o lado errado.
    lin_ruim = metrica_frenagem(am_desl, d0, d_fim, x_parede=x_parede)
    lin_bom = metrica_frenagem(am_corr, d0, d_fim, x_parede=x_parede)
    e_ruim = {r: e for r, e, _, _ in lin_ruim}["os tres"]
    e_bom = {r: e for r, e, _, _ in lin_bom}["os tres"]
    confere("sem psi_ref o zero deslocado estraga 'os tres'", abs(e_ruim) > 0.5)
    confere("com psi_ref 'os tres' volta ao normal", abs(e_bom) < 0.10)
    confere("e a correcao e grande: ao menos 10x", abs(e_ruim) > 10 * abs(e_bom))

    if tmpdir.endswith("_teste_tmp"):
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n=== RESULTADO: %s ===" % ("OK" if ok else "FALHOU"))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="roda as sete configuracoes do filtro fundido sobre um "
                    "CSV real")
    ap.add_argument("arquivo", nargs="?", help="CSV do corrida.py ou pilotar.py")
    ap.add_argument("--frenagem", action="store_true",
                    help="metrica de frenagem em reta (exige --d0)")
    ap.add_argument("--trajetoria", action="store_true",
                    help="metrica de trajetoria (exige --marcos)")
    ap.add_argument("--d0", type=float, default=None,
                    help="[frenagem] distancia de trena a parede na largada [m]")
    ap.add_argument("--dfinal", type=float, default=None,
                    help="[frenagem] distancia de trena com o veiculo PARADO "
                         "[m]. Sem ela usa-se o ultrassom, com ressalva grave")
    ap.add_argument("--x-parede", type=float, default=6.0, dest="x_parede",
                    help="posicao da parede no mapa [m] (default 6,0)")
    ap.add_argument("--marcos", default=None,
                    help="[trajetoria] CSV com n,x,y e psi_deg opcional: "
                         "marcos onde o carro PAROU, verdade pontual")
    ap.add_argument("--pista", default=None,
                    help="[trajetoria] CSV com x,y: os VERTICES da pista "
                         "medida (a fita no chao). Mede o erro PERPENDICULAR "
                         "a ela, sem precisar parar nem sincronizar tempo")
    ap.add_argument("--fim", default=None,
                    help="[trajetoria] x,y medidos com o carro PARADO no "
                         "final. E a unica medida que pega erro AO LONGO da "
                         "pista, que a distancia perpendicular nao ve")
    ap.add_argument("--pose0", default=None,
                    help="[trajetoria] largada como x,y,psi_deg")
    ap.add_argument("--trim-steer", type=float, default=0.0, dest="trim_steer",
                    help="offset constante somado ao delta comandado, em graus "
                         "(compensar no filtro um zero de direcao deslocado)")
    ap.add_argument("--psi-ref", default=None, dest="psi_ref",
                    help="leitura do magnetometro que corresponde ao eixo "
                         "x do mapa, em graus, descontada do yaw na entrada. "
                         "Use 'auto' para a mediana do trecho parado inicial "
                         "(supoe largada apontada no sentido de x crescente). "
                         "SEM ISTO o filtro propaga x = v cos(psi) na direcao "
                         "errada em toda configuracao que use o magnetometro")
    ap.add_argument("--formulacao", choices=("variante", "vigente"),
                    default="variante",
                    help="variante (PADRAO, fusao_giro.py): o giroscopio como "
                         "entrada do rumo; vigente (fusao.py): o esterco "
                         "comandado como entrada, para comparar")
    ap.add_argument("--sem-escapatoria", action="store_true",
                    dest="sem_escapatoria",
                    help="desliga a escapatoria do gate, para comparar")
    ap.add_argument("--fake", action="store_true", help="autoteste, sem arquivo")
    args = ap.parse_args(argv)

    if args.fake:
        return 0 if _autoteste() else 1
    if not args.arquivo:
        ap.error("informe o CSV, ou use --fake")
    if not (args.frenagem or args.trajetoria):
        ap.error("escolha --frenagem ou --trajetoria")

    cab, am, tipo = ler(args.arquivo)
    print("arquivo: %s  (%s, %d amostras, %.1f s)"
          % (args.arquivo, tipo, len(am), am[-1]["t"]))
    if cab.get("condicao"):
        print("condicao: %s" % cab["condicao"])
    if cab.get("temp_ar_c"):
        print("temperatura do ar no cabecalho: %s C" % cab["temp_ar_c"])
    esc = False if args.sem_escapatoria else None  # None: o padrão da formulação
    print("formulacao: %s" % args.formulacao)
    trim = np.deg2rad(args.trim_steer)

    # --- psi_ref, na entrada
    if args.psi_ref is None:
        auto = psi_ref_automatico(am)
        print("")
        print("AVISO: sem --psi-ref, o yaw do magnetometro entra CRU.")
        print("       O canal supoe que a leitura E o rumo no mapa, e entre as")
        print("       duas ha uma constante. Sem desconta-la, toda")
        print("       configuracao que usa o magnetometro sai errada.")
        if auto is not None:
            print("       O trecho parado inicial sugere --psi-ref %+.1f"
                  % np.rad2deg(auto))
        print("")
    elif str(args.psi_ref).strip().lower() == "auto":
        auto = psi_ref_automatico(am)
        if auto is None:
            ap.error("--psi-ref auto exige trecho parado no inicio "
                     "(>= %d amostras); passe o valor em graus" % N_PARADO_MIN)
        am = aplica_psi_ref(am, auto)
        print("psi_ref automatico: %+.1f graus (mediana do trecho parado)"
              % np.rad2deg(auto))
    else:
        pr = np.deg2rad(float(args.psi_ref))
        am = aplica_psi_ref(am, pr)
        print("psi_ref: %+.1f graus (informado)" % float(args.psi_ref))

    if args.trajetoria and args.pista:
        pista = le_pista(args.pista)
        fim = ([float(v) for v in args.fim.split(",")] if args.fim else None)
        if args.pose0:
            p = [float(v) for v in args.pose0.split(",")]
            pose0 = (p[0], p[1], np.deg2rad(p[2] if len(p) > 2 else 0.0))
        else:
            pose0 = (float(pista[0][0]), float(pista[0][1]), 0.0)
            print("AVISO: sem --pose0, largada no 1o vertice com psi = 0.")
            print("       A ORIENTACAO INICIAL gira a trajetoria inteira; se")
            print("       ela foi medida, passe em --pose0 x,y,psi_deg.")
        metrica_pista(am, pista, pose0, fim=fim, escapatoria=esc,
                      trim_steer=trim, formulacao=args.formulacao)
        return 0

    if args.frenagem:
        if args.d0 is None:
            ap.error("--frenagem exige --d0 (trena na largada)")
        metrica_frenagem(am, args.d0, args.dfinal, x_parede=args.x_parede,
                         escapatoria=esc, trim_steer=trim,
                         formulacao=args.formulacao)
    else:
        if not args.marcos:
            ap.error("--trajetoria exige --pista (fita medida) ou --marcos "
                     "(pontos onde o carro parou)")
        marcos_lista = le_marcos(args.marcos)
        if args.pose0:
            p = [float(s) for s in args.pose0.split(",")]
            pose0 = (p[0], p[1], np.deg2rad(p[2] if len(p) > 2 else 0.0))
        else:
            m0 = marcos_lista[0]
            pose0 = (m0["x"], m0["y"], m0["psi"] or 0.0)
            print("AVISO: sem --pose0, usando o primeiro marco como largada.")
            print("       A ORIENTACAO INICIAL gira a trajetoria inteira; se")
            print("       ela foi medida, passe em --pose0 x,y,psi_deg.")
        metrica_trajetoria(am, marcos_lista, pose0,
                           escapatoria=esc, trim_steer=trim,
                           formulacao=args.formulacao)
    return 0


if __name__ == "__main__":
    sys.exit(main())
