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
"""Driver do ultrassom HC-SR04 (GPIO da Raspberry)."""
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
import threading
import time

from kalman.config import (
    DT_LOOP,
    MAX_JUMP_MARGIN,
    SOUND_TEMP_C,
    US_ECHO_TIMEOUT,
    US_MAX_RANGE,
    US_MAX_RANGE_FISICO,
    US_MIN_RANGE,
    V_MAX_VEICULO,
    sound_speed,
)
from kalman.utils import Timer


class Ultrassom:
    """Driver de dados brutos."""

    def __init__(self, fonte=None, triggerPin=None, echoPin=None,
                 temp_c=SOUND_TEMP_C, timeout_s=US_ECHO_TIMEOUT,
                 min_range=US_MIN_RANGE, max_range=US_MAX_RANGE,
                 usar_thread=False, periodo_thread=0.06):
        if fonte is None:
            from bancada.drivers.base import FonteGPIO

            fonte = FonteGPIO(triggerPin=triggerPin, echoPin=echoPin)

        self.fonte = fonte
        self.temp_c = temp_c
        self.c_som = sound_speed(temp_c)
        self.timeout_s = timeout_s
        self.min_range = min_range
        self.max_range = max_range

        # Coerência entre o timeout e o alcance físico (não o de confiança): um
        # eco vindo de `US_MAX_RANGE_FISICO` leva 2*d/c para voltar.
        t_necessario = 2.0 * US_MAX_RANGE_FISICO / self.c_som
        if self.timeout_s < t_necessario:
            print(f"[ultrassom] AVISO: timeout={self.timeout_s*1000:.0f} ms e menor que "
                  f"os {t_necessario*1000:.0f} ms necessarios para o alcance fisico "
                  f"de {US_MAX_RANGE_FISICO} m a {self.temp_c} C. Alvos alem do limite "
                  f"de confianca serao contados como 'sem eco' em vez de 'fora de faixa'.")

        self.ultima_dist = None
        self.ultimo_t = None

        # Contadores de auditoria da coleta
        self.n_ok = 0
        self.n_sem_eco = 0
        self.n_fora_faixa = 0
        self.n_salto = 0
        # instante da última medida válida (watchdog, ver sensor_vivo)
        self._t_ultima_valida = None
        # Motivo da última rejeição.
        self._motivo = ""

        # Thread opcional de aquisição
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._ultima_amostra = (None, False, None)
        if usar_thread:
            self.periodo_thread = periodo_thread
            self._thread = threading.Thread(target=self._laco, daemon=True)
            self._thread.start()

    def _medir_pulso(self, nivel):
        """Espera o pino de eco atingir `nivel`, com prazo."""
        deadline = time.perf_counter() + self.timeout_s
        while self.fonte.nivel_echo() != nivel:
            if time.perf_counter() > deadline:
                return False, None
        return True, time.perf_counter()

    # Watchdog de sensor vivo.
    SENSOR_TIMEOUT_S = 0.30

    def sensor_vivo(self):
        """True se houve medida válida nos últimos SENSOR_TIMEOUT_S."""
        if self._t_ultima_valida is None:
            return False
        return (Timer.now() - self._t_ultima_valida) <= self.SENSOR_TIMEOUT_S

    def tempo_sem_medida(self):
        """Segundos desde a última medida válida (inf se nunca houve)."""
        if self._t_ultima_valida is None:
            return float("inf")
        return Timer.now() - self._t_ultima_valida

    def ler(self):
        """Uma medida."""
        t_agora = Timer.now()

        try:
            self.fonte.disparar()
        except Exception as e:
            print(f"[ultrassom] falha ao disparar: {type(e).__name__}: {e}")
            self.n_sem_eco += 1
            self._motivo = "falha_trigger"
            return None, False, t_agora

        ok_subida, t_subida = self._medir_pulso(1)
        if not ok_subida:
            self.n_sem_eco += 1
            self._motivo = "sem_eco_subida"
            return None, False, t_agora

        ok_descida, t_descida = self._medir_pulso(0)
        if not ok_descida:
            self.n_sem_eco += 1
            self._motivo = "sem_eco_descida"
            return None, False, t_agora

        largura = t_descida - t_subida
        dist = largura * self.c_som / 2.0

        # --- validação física (não é o gate estatístico)
        if not (self.min_range <= dist <= self.max_range):
            self.n_fora_faixa += 1
            self._motivo = ("abaixo_min" if dist < self.min_range else "acima_max")
            return None, False, t_agora

        if self.ultima_dist is not None and self.ultimo_t is not None:
            dt = t_agora - self.ultimo_t
            salto_max = V_MAX_VEICULO * dt + MAX_JUMP_MARGIN
            if abs(dist - self.ultima_dist) > salto_max:
                self.n_salto += 1
                self._motivo = "salto_impossivel"
                return None, False, t_agora

        self.ultima_dist = dist
        self.ultimo_t = t_agora
        self.n_ok += 1
        self._t_ultima_valida = Timer.now()
        self._motivo = ""
        return dist, True, t_agora

    def _laco(self):
        """Laço da thread de aquisição."""
        while not self._stop.is_set():
            amostra = self.ler()
            with self._lock:
                self._ultima_amostra = amostra
            time.sleep(self.periodo_thread)

    def ler_da_thread(self):
        """Última amostra colhida pela thread. Só use se `usar_thread=True`."""
        with self._lock:
            return self._ultima_amostra

    def ultimo_motivo(self):
        """Motivo da última rejeição (string vazia se a última leitura foi válida)."""
        return self._motivo

    def estatisticas(self):
        total = self.n_ok + self.n_sem_eco + self.n_fora_faixa + self.n_salto
        return {
            "ok": self.n_ok,
            "sem_eco": self.n_sem_eco,
            "fora_faixa": self.n_fora_faixa,
            "salto_rejeitado": self.n_salto,
            "total": total,
            "taxa_ok": (self.n_ok / total) if total else 0.0,
            "c_som": self.c_som,
            "temp_c": self.temp_c,
        }

    def plot(self, duration=60, filtro="ultrassom2"):
        """Depuração ao vivo."""
        import matplotlib.pyplot as plt

        from kalman.core import KalmanFilter
        from kalman.filtros import ultrassom1, ultrassom2

        usa2 = filtro == "ultrassom2"
        mod = ultrassom2 if usa2 else ultrassom1

        plt.ion()
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        (l_bruto,) = ax.plot([], [], "r.", alpha=0.5, label="bruto")
        (l_kf,) = ax.plot([], [], "g-", linewidth=2, label=f"Kalman ({filtro})")
        # Banda de +-3 sigma: quando a leitura é rejeitada, a estimativa
        # continua pela predição e a banda abre.
        (l_sup,) = ax.plot([], [], "g:", linewidth=1, alpha=0.7, label="+-3sigma")
        (l_inf,) = ax.plot([], [], "g:", linewidth=1, alpha=0.7)
        # Faixa de confiança: fora dela a leitura é descartada.
        ax.axhline(self.min_range, color="k", ls="--", alpha=0.4)
        ax.axhline(self.max_range, color="k", ls="--", alpha=0.4)
        ax.axhspan(self.min_range, self.max_range, color="green", alpha=0.05,
                    label=f"faixa valida ({self.min_range}-{self.max_range} m)")
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Distancia [m]")
        ax.set_title(f"Ultrassom — bruto vs {filtro}")
        ax.grid(True)
        ax.legend(fontsize=8)
        (l_p,) = axp.plot([], [], "b-")
        axp.set_ylabel("P[0,0]")
        axp.set_xlabel("Tempo [s]")
        axp.grid(True)

        kf = None
        ts, ds, ks, ps = [], [], [], []
        t0 = Timer.now()
        timer = Timer()
        n_loop = 0
        janela = 300

        try:
            while Timer.now() - t0 < duration:
                t_iter = Timer.now()
                dt = timer.tick()
                dist, ok, _ = self.ler()

                if kf is None:
                    if not ok:
                        sobra = DT_LOOP - (Timer.now() - t_iter)
                        if sobra > 0:
                            time.sleep(sobra)
                        continue  # espera a primeira medida valida para inicializar
                    kf = KalmanFilter(dim_x=2 if usa2 else 1)
                    kf.x, kf.P = mod.initial(dist)

                F, H, Q, R = mod.matrices(dt)
                kf.F, kf.Q = F, Q
                kf.predict()
                if ok:
                    kf.update_scalar(dist, H, R, channel="d")
                # sem eco -> só predição; P cresce

                ts.append(Timer.now() - t0)
                ds.append(dist if ok else float("nan"))
                ks.append(kf.x[0, 0])
                ps.append(kf.P[0, 0])
                n_loop += 1

                if n_loop % 5 == 0:
                    import numpy as _np

                    jt = ts[-janela:]
                    jk = _np.asarray(ks[-janela:])
                    j3 = 3.0 * _np.sqrt(_np.maximum(_np.asarray(ps[-janela:]), 0.0))
                    l_bruto.set_data(jt, ds[-janela:])
                    l_kf.set_data(jt, jk)
                    l_sup.set_data(jt, jk + j3)
                    l_inf.set_data(jt, jk - j3)
                    l_p.set_data(jt, ps[-janela:])
                    for a in (ax, axp):
                        a.relim()
                        a.autoscale_view()
                    plt.pause(0.001)

                # Limita o laço a DT_LOOP (20 Hz = 50 ms).
                sobra = DT_LOOP - (Timer.now() - t_iter)
                if sobra > 0:
                    time.sleep(sobra)

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuario")
        finally:
            plt.ioff()
            print(f"[ultrassom] estatisticas: {self.estatisticas()}")
            plt.show()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.fonte.fechar()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Autoteste --fake
def _autoteste():
    from bancada.drivers.base import FonteGPIOFake

    print("=== bancada.drivers.ultrassom — autoteste (--fake) ===\n")

    # 1) medida normal --------------------------------------------------
    us = Ultrassom(fonte=FonteGPIOFake(distancia_m=1.5), temp_c=20.0)
    d, ok, _ = us.ler()
    ok_normal = ok and abs(d - 1.5) < 0.05
    print(f"1) medida normal: d={d:.3f} m, ok={ok} (alvo 1.500) -> "
          f"{'OK' if ok_normal else 'FALHOU'}")

    # 2) sem eco -------------------------------------------------------------
    us2 = Ultrassom(fonte=FonteGPIOFake(sem_eco=True), temp_c=20.0, timeout_s=0.03)
    t_ini = time.perf_counter()
    d2, ok2, _ = us2.ler()
    decorrido = time.perf_counter() - t_ini

    nao_travou = decorrido < 0.5  # sem prazo, o laço travaria aqui
    ok_sem_eco = (not ok2) and (d2 is None)
    print(f"\n2) SEM ECO:")
    print(f"   retornou em {decorrido*1000:.0f} ms (timeout={us2.timeout_s*1000:.0f} ms) -> "
          f"{'OK' if nao_travou else 'FALHOU (travou)'}")
    print(f"   ok={ok2}, dist={d2} — nao inventou maxRange -> "
          f"{'OK' if ok_sem_eco else 'FALHOU'}")

    us3 = Ultrassom(fonte=FonteGPIOFake(distancia_m=0.02), temp_c=20.0)
    d3, ok3, _ = us3.ler()
    ok_perto = (not ok3) and us3.n_fora_faixa == 1
    print(f"\n3) abaixo de US_MIN_RANGE ({US_MIN_RANGE} m): ok={ok3}, "
          f"fora_faixa={us3.n_fora_faixa} -> {'OK' if ok_perto else 'FALHOU'}")

    us4 = Ultrassom(fonte=FonteGPIOFake(distancia_m=6.0), temp_c=20.0, timeout_s=0.05)
    d4, ok4, _ = us4.ler()
    ok_longe = not ok4
    print(f"   acima de US_MAX_RANGE ({US_MAX_RANGE} m): ok={ok4} -> "
          f"{'OK' if ok_longe else 'FALHOU'}")

    fonte_salto = FonteGPIOFake(distancia_m=0.5)
    us5 = Ultrassom(fonte=fonte_salto, temp_c=20.0)
    us5.ler()                       # ancora em ~0.5 m
    fonte_salto.distancia_m = 1.9   # salto de 1.4 m entre amostras consecutivas
    d5, ok5, _ = us5.ler()
    ok_salto = (not ok5) and us5.n_salto == 1 and us5.n_fora_faixa == 0
    print(f"\n4) salto impossivel (0.5 -> 1.9 m, ambos na faixa): ok={ok5}, "
          f"salto_rejeitado={us5.n_salto}, fora_faixa={us5.n_fora_faixa} -> "
          f"{'OK' if ok_salto else 'FALHOU'}")

    us20 = Ultrassom(fonte=FonteGPIOFake(distancia_m=1.5, temp_c=20.0), temp_c=20.0)
    d20, _, _ = us20.ler()
    # A fonte simula a 20 C, mas o driver calcula a 30 C -> deve superestimar.
    us30 = Ultrassom(fonte=FonteGPIOFake(distancia_m=1.5, temp_c=20.0), temp_c=30.0)
    d30, _, _ = us30.ler()
    razao = d30 / d20
    razao_esperada = sound_speed(30.0) / sound_speed(20.0)
    ok_temp = abs(razao - razao_esperada) < 0.005
    print(f"\n5) velocidade do som: c(20C)={sound_speed(20.0):.1f}, c(30C)={sound_speed(30.0):.1f} m/s")
    print(f"   d(30C)/d(20C) = {razao:.4f}, esperado {razao_esperada:.4f} -> "
          f"{'OK' if ok_temp else 'FALHOU'}")

    # 6) acoplado ao filtro: sem eco => P cresce -------------------------
    from kalman.core import KalmanFilter
    from kalman.filtros import ultrassom2

    fonte6 = FonteGPIOFake(distancia_m=1.5)
    us6 = Ultrassom(fonte=fonte6, temp_c=20.0, timeout_s=0.01)
    dt = 0.05
    F, H, Q, R = ultrassom2.matrices(dt)
    d0, ok0, _ = us6.ler()
    kf = KalmanFilter(dim_x=2)
    kf.x, kf.P = ultrassom2.initial(d0)
    kf.F, kf.Q = F, Q

    for _ in range(10):  # com eco: P converge
        kf.predict()
        d, ok, _ = us6.ler()
        if ok:
            kf.update_scalar(d, H, R, channel="d")
    p_com_eco = kf.P[0, 0]

    fonte6.sem_eco = True
    for _ in range(10):  # sem eco: so predicao
        kf.predict()
        d, ok, _ = us6.ler()
        if ok:
            kf.update_scalar(d, H, R, channel="d")
    p_sem_eco = kf.P[0, 0]

    ok_p = p_sem_eco > p_com_eco
    print(f"\n6) acoplado ao filtro: P[0,0] com eco={p_com_eco:.6f}, "
          f"apos 10 sem eco={p_sem_eco:.6f}")
    print(f"   P cresceu sem medida -> {'OK' if ok_p else 'FALHOU'}")

    tudo = all([ok_normal, nao_travou, ok_sem_eco, ok_perto, ok_longe,
                ok_salto, ok_temp, ok_p])
    print(f"\n=== RESULTADO: {'OK' if tudo else 'FALHOU'} ===")
    return tudo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Driver do ultrassom")
    ap.add_argument("--fake", action="store_true", help="autoteste sem hardware")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--temp", type=float, default=SOUND_TEMP_C,
                    help="temperatura do ensaio em C")
    ap.add_argument("--filtro", choices=["ultrassom1", "ultrassom2"],
                    default="ultrassom2")
    ap.add_argument("--min-range", type=float, default=US_MIN_RANGE,
                    dest="min_range",
                    help=f"limite inferior de confianca [m] (padrao {US_MIN_RANGE}). "
                         "Leituras abaixo disto sao REJEITADAS e o filtro so prediz.")
    ap.add_argument("--max-range", type=float, default=US_MAX_RANGE,
                    dest="max_range",
                    help=f"limite superior de confianca [m] (padrao {US_MAX_RANGE})")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    us = Ultrassom(temp_c=args.temp,
                   min_range=args.min_range, max_range=args.max_range)
    print(f"[ultrassom] faixa de confianca: {args.min_range} a {args.max_range} m "
          f"| fora dela a leitura e rejeitada e o filtro EXTRAPOLA")
    try:
        us.plot(duration=args.duration, filtro=args.filtro)
    finally:
        us.close()
