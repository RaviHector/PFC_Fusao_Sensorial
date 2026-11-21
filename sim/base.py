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
"""Infraestrutura comum dos simuladores: NEES, NIS, brancura, cobertura e figuras."""
# --- bootstrap de path
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

import os

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as _t_dist

from kalman.utils import wrap


# NEES: e = x_verdadeiro - x_estimado (com wrap nos ângulos).
# NEES_k = e^T P_k^-1 e. Num filtro consistente, a média tende a dim_x.
def compute_nees(x_true, x_est, P_hist, angle_indices=None):
    """
    x_true, x_est: (n, dim_x) P_hist: (n, dim_x, dim_x) Retorna (serie_nees:
    (n,), media: float).
    """
    angle_indices = list(angle_indices) if angle_indices else []
    x_true = np.asarray(x_true, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    n, dim_x = x_true.shape

    nees = np.zeros(n)
    for k in range(n):
        e = x_true[k] - x_est[k]
        for idx in angle_indices:
            e[idx] = wrap(e[idx])
        e = e.reshape(-1, 1)
        nees[k] = float((e.T @ np.linalg.inv(P_hist[k]) @ e).item())
    return nees, float(np.mean(nees))


def nees_verdict(nees_mean, dim_x, n_samples, alpha=0.05):
    """Teste de consistência do NEES via intervalo de confiança qui-quadrado"""
    df = n_samples * dim_x
    lo = chi2.ppf(alpha / 2, df) / n_samples
    hi = chi2.ppf(1 - alpha / 2, df) / n_samples
    ok = lo <= nees_mean <= hi
    msg = (f"NEES medio = {nees_mean:.2f} (esperado ~{dim_x}, N={n_samples}, "
           f"IC {int((1-alpha)*100)}% = [{lo:.2f}, {hi:.2f}]): {'OK' if ok else 'FALHOU'}")
    return ok, msg


def nees_verdict_mc(nees_means, dim_x, alpha=0.05):
    """
    Teste de consistência do NEES por Monte Carlo — várias REALIZAÇÕES
    INDEPENDENTES (sementes diferentes), não amostras de tempo de uma única
    corrida.
    """
    nees_means = np.asarray(nees_means, dtype=float)
    m = len(nees_means)
    mean = float(np.mean(nees_means))
    se = float(np.std(nees_means, ddof=1)) / np.sqrt(m)
    tcrit = float(_t_dist.ppf(1 - alpha / 2, df=m - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se
    ok = lo <= dim_x <= hi
    msg = (f"NEES medio (Monte Carlo, {m} realizacoes) = {mean:.2f} "
           f"(esperado {dim_x}; IC {int((1-alpha)*100)}% da media = "
           f"[{lo:.2f}, {hi:.2f}]): {'OK' if ok else 'FALHOU'}")
    return ok, msg


def nis_verdict(nis_mean, lo=0.5, hi=2.0, label=""):
    ok = lo <= nis_mean <= hi
    msg = (f"NIS medio [{label}] = {nis_mean:.2f} "
           f"(esperado ~1, faixa [{lo}, {hi}]): {'OK' if ok else 'FALHOU'}")
    return ok, msg


def innovation_autocorr(nu, nlags=15):
    """Autocorrelacao amostral da inovacao, de 0 a nlags, normalizada."""
    nu = np.asarray(nu, dtype=float)
    den = float(np.dot(nu, nu))
    if den <= 0.0:
        return np.zeros(nlags + 1)
    n = len(nu)
    return np.array([float(np.dot(nu[:n - k], nu[k:])) / den
                     for k in range(nlags + 1)])


def innovation_whiteness_verdict(nu, nlags=15, alpha=0.05, label=""):
    """Testa se a inovacao e BRANCA, e nao apenas do tamanho certo."""
    nu = np.asarray(nu, dtype=float)
    n = len(nu)
    if n < 3 * nlags:
        return True, 0.0, (f"brancura da inovacao [{label}]: amostras de menos "
                           f"({n} para {nlags} defasagens), teste pulado")

    ac = innovation_autocorr(nu, nlags)
    banda = 1.96 / np.sqrt(n)
    fora = int(np.sum(np.abs(ac[1:]) > banda))
    frac_fora = fora / nlags
    # 5% e o esperado por acaso; 20% e o limiar de reprovacao.
    ok = frac_fora <= 0.20
    r1 = float(ac[1])
    msg = (f"brancura da inovacao [{label}]: r_1 = {r1:+.3f}, "
           f"{fora}/{nlags} defasagens fora da banda +-{banda:.3f} "
           f"({frac_fora*100:.0f}%, esperado ~5%): {'OK' if ok else 'FALHOU'}")
    return ok, r1, msg


def innovation_decomposicao_mc(nu_list, nlags=15):
    """Separa o r_1 em VIES e ESTRUTURA TEMPORAL."""
    todas = [np.asarray(nu, dtype=float) for nu in nu_list]
    boas = [nu for nu in todas if len(nu) >= 3 * nlags]
    if not boas:
        return 0.0, 0.0, 0.0, 0, len(todas)

    media = float(np.mean([nu.mean() for nu in boas]))
    com = float(np.mean([innovation_autocorr(nu, 1)[1] for nu in boas]))
    sem = float(np.mean([innovation_autocorr(nu - nu.mean(), 1)[1]
                         for nu in boas]))
    return media, com, sem, len(boas), len(todas)


def innovation_whiteness_verdict_mc(nu_list, nlags=15, label="",
                                    criterio=True, frac_max=0.20):
    """Brancura da inovacao sobre VARIAS realizacoes independentes."""
    series = [np.asarray(nu, dtype=float) for nu in nu_list]
    series = [nu for nu in series if len(nu) >= 3 * nlags]
    if not series:
        return True, 0.0, (f"brancura da inovacao [{label}]: amostras de menos "
                           f"em todas as realizacoes, teste pulado")

    acs = np.array([innovation_autocorr(nu, nlags) for nu in series])
    ac = acs.mean(axis=0)
    n_total = sum(len(nu) for nu in series)
    banda = 1.96 / np.sqrt(n_total)

    fora = int(np.sum(np.abs(ac[1:]) > banda))
    frac_fora = fora / nlags
    r1 = float(ac[1])
    passou = frac_fora <= frac_max
    ok = True if not criterio else passou

    veredicto = "OK" if passou else "FALHOU"
    if not criterio:
        veredicto += " (informativo, nao e criterio aqui)"
    msg = (f"brancura da inovacao [{label}]: r_1 = {r1:+.3f}, "
           f"{fora}/{nlags} defasagens fora de +-{banda:.4f} "
           f"({len(series)} realizacoes, {n_total} amostras): {veredicto}")
    return ok, r1, msg


def coverage_verdict(err, var_vals, k=3.0, min_frac=0.95, label=""):
    """Fração dos erros dentro da banda +-k*sigma de P."""
    band = k * np.sqrt(np.maximum(np.asarray(var_vals), 0.0))
    inside = np.abs(err) <= band
    frac = float(np.mean(inside))
    ok = frac >= min_frac
    msg = (f"cobertura +-{int(k)}sigma [{label}] = {frac*100:.1f}% "
           f"(esperado >= {min_frac*100:.0f}%): {'OK' if ok else 'FALHOU'}")
    return ok, frac, msg


# Gráficos — cada função desenha num eixo (ax) dado, para o sim_*.py compor
# a figura combinada com o layout que fizer sentido para aquele sensor.
def draw_state_vs_time(ax, t, true_vals, est_vals, raw_t=None, raw_vals=None,
                        ylabel="", scale=1.0, raw_label="bruto"):
    ax.plot(t, np.asarray(true_vals) * scale, "k--", label="verdade")
    if raw_t is not None and raw_vals is not None:
        ax.plot(raw_t, np.asarray(raw_vals) * scale, ".", alpha=0.3, label=raw_label)
    ax.plot(t, np.asarray(est_vals) * scale, label="Kalman")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def draw_error_band(ax, t, true_vals, est_vals, var_vals, ylabel="",
                     scale=1.0, is_angle=False, k=3.0):
    """Desenha erro + banda +-k*sigma. Retorna o erro (unidades nativas, rad/m)."""
    true_vals = np.asarray(true_vals, dtype=float)
    est_vals = np.asarray(est_vals, dtype=float)
    var_vals = np.asarray(var_vals, dtype=float)

    err = wrap(true_vals - est_vals) if is_angle else (true_vals - est_vals)
    band = k * np.sqrt(np.maximum(var_vals, 0.0))

    ax.plot(t, err * scale, label="erro")
    ax.plot(t, band * scale, "r--", label=f"+-{int(k)}sigma", linewidth=1)
    ax.plot(t, -band * scale, "r--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return err


def draw_nis(ax, t_nis, nis_vals, threshold=None, label=""):
    ax.plot(t_nis, nis_vals, ".", ms=3, alpha=0.6, label=label)
    ax.axhline(1.0, color="g", ls=":", label="referencia (=1)")
    if threshold is not None:
        ax.axhline(threshold, color="r", ls="--", label="limiar de gating")
    ax.set_ylabel("NIS")
    ax.legend(fontsize=8)


def output_path(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_output", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Figura salva em {path}")


# Autoteste — `python -m sim.base`
def _autoteste():
    """
    Verifica que o teste de brancura PEGA um modelo errado que o NIS deixa
    passar.
    """
    import numpy as _np
    from kalman.config import DT_LOOP, SIGMA_ENC, SIGMA_ACC
    from kalman.core import KalmanFilter
    from kalman.filtros import encoder2
    from sim.sim_encoder_2 import true_trajectory_stochastic

    def _corrida(fator_F, seed):
        dt = DT_LOOP
        t, _, v, _ = true_trajectory_stochastic(dt, 20.0, SIGMA_ACC, seed)
        rng = _np.random.default_rng(seed + 7777)
        raw = v + rng.normal(0, SIGMA_ENC, len(t))
        F, H, Q, R = encoder2.matrices(dt, sigma_acc=SIGMA_ACC)
        F = F.copy()
        F[1, 1] = fator_F
        kf = KalmanFilter(dim_x=2)
        kf.x, kf.P = encoder2.initial()
        kf.F, kf.Q = F, Q
        nis = []
        for k in range(len(t)):
            kf.predict()
            _, z = kf.update_scalar(raw[k], H, R, channel="v")
            nis.append(z)
        b = int(1.0 / dt)
        return _np.array(kf.innov_history["v"])[b:], float(_np.mean(nis[b:]))

    print("=== sim.base — o teste de brancura pega o que o NIS nao pega? ===")
    veredictos = {}
    for nome, fF in [("F correto", 1.0), ("F errado (v decai 2%/passo)", 0.98)]:
        pares = [_corrida(fF, 1000 + i) for i in range(20)]
        ok_nis, msg_nis = nis_verdict(
            float(_np.mean([p[1] for p in pares])), label="v")
        ok_bra, r1, _ = innovation_whiteness_verdict_mc(
            [p[0] for p in pares], label="v")
        print("  %-30s NIS %-7s (%.2f)   brancura %-7s (r_1 = %+.3f)"
              % (nome, "OK" if ok_nis else "FALHOU",
                 float(_np.mean([p[1] for p in pares])),
                 "OK" if ok_bra else "FALHOU", r1))
        veredictos[nome] = (ok_nis, ok_bra)

    ok_nis_c, ok_bra_c = veredictos["F correto"]
    ok_nis_e, ok_bra_e = veredictos["F errado (v decai 2%/passo)"]
    assert ok_nis_c and ok_bra_c, "o filtro CORRETO tem de passar nos dois"
    assert ok_nis_e, ("o NIS deveria APROVAR o F errado — se ele reprovou, o "
                      "caso perdeu a graca e precisa ser reescolhido")
    assert not ok_bra_e, "a brancura tem de REPROVAR o F errado"
    print("")
    print("  O NIS aprova o modelo errado e a brancura o reprova: e a razao")
    print("  de o segundo teste existir ao lado do primeiro.")
    print("\n=== RESULTADO: OK ===")


if __name__ == "__main__":
    _autoteste()
