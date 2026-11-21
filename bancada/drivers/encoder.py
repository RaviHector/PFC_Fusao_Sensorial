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
"""Driver do encoder (Arduino Nano pela serial)."""
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
import time

from kalman.config import (DT_LOOP, ENCODER_BAUDRATE, RAIO_RODA,
                           REDUCAO_EIXO, SIGMA_ENC)
from kalman.utils import Timer


def find_arduino():
    """Procura o Nano nas portas seriais. Retorna o device ou None."""
    from serial.tools import list_ports  # importação tardia

    for p in list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if ("arduino" in desc or "wch" in desc or "1a86" in hwid
                or "ftdi" in desc or "0403" in hwid):
            return p.device
    return None


class Encoder:
    """Driver de dados brutos."""

    def __init__(self, fonte=None, port=None, baudrate=ENCODER_BAUDRATE,
                 raio_roda=RAIO_RODA, reducao=REDUCAO_EIXO):
        """
        fonte: objeto com `ler_todas()` e `fechar()` (ver drivers/base.py).
        """
        if fonte is None:
            from bancada.drivers.base import FonteSerial

            port = port or find_arduino()
            if not port:
                raise RuntimeError("Arduino Nano nao encontrado!")
            print(f"\033[32mEncoder conectando em: {port} @ {baudrate} baud\033[0m")
            fonte = FonteSerial(port, baudrate)

        self.fonte = fonte
        self.raio_roda = raio_roda
        self.reducao = reducao

        self.vel = 0.0  # última velocidade VÁLIDA (não é devolvida como nova)
        self.timer = Timer()
        # instante da ultima medida valida (watchdog — ver sensor_vivo)
        self._t_ultima_valida = None

        # Contadores, que vão para o CSV e permitem auditar a coleta.
        self.n_lidas = 0        # linhas que viraram velocidade válida
        self.n_vazias = 0       # chamadas sem nenhuma linha no buffer
        self.n_descartadas = 0  # linhas mais antigas jogadas fora (só a última vale)
        self.n_erro_parse = 0   # linhas que não converteram para float finito

    # Watchdog de sensor vivo.
    SENSOR_TIMEOUT_S = 0.30

    def sensor_vivo(self):
        """True se houve medida valida nos ultimos SENSOR_TIMEOUT_S."""
        if self._t_ultima_valida is None:
            return False
        return (Timer.now() - self._t_ultima_valida) <= self.SENSOR_TIMEOUT_S

    def tempo_sem_medida(self):
        """Segundos desde a ultima medida valida (inf se nunca houve)."""
        if self._t_ultima_valida is None:
            return float("inf")
        return Timer.now() - self._t_ultima_valida

    def getVel(self):
        """Retorna `(vel_m_s, is_new)`."""
        try:
            linhas = self.fonte.ler_todas()
        except Exception as e:  # falha de I/O é diferente de "sem dado"
            self.n_erro_parse += 1
            print(f"[encoder] falha de leitura: {type(e).__name__}: {e}")
            return self.vel, False

        if not linhas:
            self.n_vazias += 1
            return self.vel, False

        # Só a linha mais recente interessa; as anteriores já estão obsoletas.
        self.n_descartadas += len(linhas) - 1
        bruta = linhas[-1]

        try:
            rpm_motor = float(bruta.decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError, AttributeError):
            # Linha corrompida (baud errado, ruído na serial, linha parcial).
            # Não é `except:` nu: KeyboardInterrupt continua passando.
            self.n_erro_parse += 1
            return self.vel, False

        if math.isnan(rpm_motor) or math.isinf(rpm_motor):
            self.n_erro_parse += 1
            return self.vel, False

        rpm_roda = rpm_motor / self.reducao
        self.vel = self.raio_roda * (math.pi / 30.0) * rpm_roda
        self.n_lidas += 1
        self._t_ultima_valida = Timer.now()
        return self.vel, True

    def getVelLegacy(self):
        """Assinatura antiga (`-> float`), para compatibilidade. Preferir `getVel()`."""
        vel, _ = self.getVel()
        return vel

    def estatisticas(self):
        total = self.n_lidas + self.n_vazias + self.n_erro_parse
        return {
            "lidas": self.n_lidas,
            "vazias": self.n_vazias,
            "descartadas": self.n_descartadas,
            "erro_parse": self.n_erro_parse,
            "total_chamadas": total,
            "taxa_novas": (self.n_lidas / total) if total else 0.0,
        }

    def plot(self, duration=30):
        """Depuração ao vivo: velocidade bruta contra o Kalman de 2 estados."""
        import matplotlib.pyplot as plt

        from kalman.core import KalmanFilter
        from kalman.filtros import encoder2

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        (l_bruto,) = ax.plot([], [], "r-", alpha=0.5, label="Encoder (bruto)")
        (l_kf,) = ax.plot([], [], "g-", linewidth=2, label="Kalman (velocidade)")
        ax.set_xlabel("Tempo [s]")
        ax.set_ylabel("Velocidade [m/s]")
        ax.set_title("Encoder — bruto vs Kalman 2 estados")
        ax.grid(True)
        ax.legend()

        kf = KalmanFilter(dim_x=2)
        kf.x, kf.P = encoder2.initial()

        ts, vs_bruto, vs_kf = [], [], []
        t0 = Timer.now()
        self.timer.tick()
        n_loop = 0
        janela = 200

        try:
            while Timer.now() - t0 < duration:
                t_iter = Timer.now()
                dt = self.timer.tick()
                vel, is_new = self.getVel()

                # Matrizes remontadas com o dt medido a cada iteração.
                F, H, Q, R = encoder2.matrices(dt)
                kf.F, kf.Q = F, Q
                kf.predict()
                if is_new:
                    kf.update_scalar(vel, H, R, channel="v")
                # sem medida nova -> só predição, P cresce

                ts.append(Timer.now() - t0)
                vs_bruto.append(vel)
                vs_kf.append(kf.x[1, 0])
                n_loop += 1

                if n_loop % 5 == 0:
                    l_bruto.set_data(ts[-janela:], vs_bruto[-janela:])
                    l_kf.set_data(ts[-janela:], vs_kf[-janela:])
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.001)

                # Limita o laço a DT_LOOP (20 Hz), a mesma taxa de
                # class_car.py.
                sobra = DT_LOOP - (Timer.now() - t_iter)
                if sobra > 0:
                    time.sleep(sobra)

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuario")
        finally:
            plt.ioff()
            print(f"[encoder] estatisticas: {self.estatisticas()}")
            plt.show()

    def amostrar(self, duration=15.0, csv=None):
        """Coleta sem janela gráfica e imprime um resumo, útil por SSH."""
        print(f"Amostrando {duration:.0f}s @ {getattr(self.fonte, 'baudrate', '?')} baud...")
        print("Mantenha o veiculo em movimento durante a coleta.")
        print("MELHOR: acelerador CONSTANTE pelo controle remoto, veiculo no")
        print("cavalete. Com o motor acionando, o diferencial nao atrapalha —")
        print("ele so absorve movimento quando se forca UMA roda de fora.")
        print("")

        t0 = Timer.now()
        vistos = []
        registro = []  # (t, vel, is_new): dado bruto, sem filtro
        while Timer.now() - t0 < duration:
            t_iter = Timer.now()
            vel, is_new = self.getVel()
            if is_new:
                vistos.append(vel)
            registro.append((t_iter - t0, vel, int(is_new)))
            time.sleep(DT_LOOP)

        if csv:
            import csv as _csv

            with open(csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["t_s", "vel_m_s", "is_new"])
                w.writerows(registro)
            print("")
            print(f"Dados brutos salvos em {csv} ({len(registro)} amostras)")

        est = self.estatisticas()
        print(f"estatisticas: {est}")

        total = est["lidas"] + est["erro_parse"]
        frac_erro = (est["erro_parse"] / total) if total else 1.0
        print()
        if est["lidas"] == 0:
            print("==> NENHUMA LEITURA VALIDA. Baud errado, ou o Nano nao transmite.")
        elif frac_erro > 0.20:
            print(f"==> BAUD SUSPEITO: {frac_erro*100:.0f}% das linhas nao viraram numero.")
            print("    Tente outro valor em --baud.")
        else:
            vmin, vmax = (min(vistos), max(vistos)) if vistos else (0.0, 0.0)
            print(f"==> BAUD PLAUSIVEL: so {frac_erro*100:.0f}% de erro de parse.")
            print(f"    Velocidade observada: {vmin:+.3f} a {vmax:+.3f} m/s")
            if abs(vmax - vmin) < 1e-6:
                print("    MAS o valor nunca mudou — a roda girou? Se girou e nao mudou,")
                print("    o dado pode nao ser RPM (ver encoder_test.ino vs odometer.ino).")
            elif abs(vmax) > 50.0:
                print("    ATENCAO: valores absurdos (>50 m/s). O sketch pode estar")
                print("    enviando CONTAGEM ACUMULADA em vez de RPM — ver encoder_test.ino.")

            # com velocidade constante, o desvio estima o SIGMA_ENC
            if len(vistos) >= 30:
                import statistics

                meio = vistos[len(vistos) // 4: 3 * len(vistos) // 4]
                if len(meio) >= 10:
                    media = statistics.mean(meio)
                    desvio = statistics.pstdev(meio)
                    print()
                    print("    --- se a velocidade foi mantida constante ---")
                    print(f"    media do trecho central: {media:+.4f} m/s")
                    print(f"    desvio-padrao          : {desvio:.4f} m/s"
                          f"  <- estimativa preliminar de SIGMA_ENC")
                    print(f"    (config.py usa {SIGMA_ENC} m/s)")
                    piso = 0.0017
                    if desvio < piso:
                        print(f"    ATENCAO: abaixo do piso de quantizacao ({piso} m/s).")
                        print("    Fisicamente impossivel — a velocidade variou pouco demais,")
                        print("    ou o dado esta sendo repetido. Colete de novo.")

    def close(self):
        self.fonte.fechar()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Autoteste --fake
def _autoteste():
    from bancada.drivers.base import FonteSerialFake

    print("=== bancada.drivers.encoder — autoteste (--fake) ===\n")

    # Roteiro que exercita os caminhos de leitura.
    # 780 rpm no motor / 7.80 = 100 rpm na roda -> 0.08 * pi/30 * 100 = 0.8378 m/s
    roteiro = [
        ["780.0\n"],                 # 0: leitura boa
        [],                          # 1: BUFFER VAZIO -> is_new deve ser False
        ["abc\n"],                   # 2: linha corrompida
        ["nan\n"],                   # 3: NaN explicito
        ["390.0\n", "780.0\n"],      # 4: duas linhas -> usa a ultima, descarta 1
        [],                          # 5: vazio de novo
    ]
    enc = Encoder(fonte=FonteSerialFake(roteiro))

    esperado_is_new = [True, False, False, False, True, False]
    obtido_is_new, obtido_vel = [], []
    for _ in range(len(roteiro)):
        v, novo = enc.getVel()
        obtido_is_new.append(novo)
        obtido_vel.append(v)

    ok_flags = obtido_is_new == esperado_is_new
    print(f"1) is_new por chamada: {obtido_is_new}")
    print(f"   esperado:           {esperado_is_new}")
    print(f"   -> {'OK' if ok_flags else 'FALHOU'}")

    v_ref = RAIO_RODA * (math.pi / 30.0) * (780.0 / REDUCAO_EIXO)
    ok_valor = abs(obtido_vel[0] - v_ref) < 1e-9
    print(f"\n2) conversao rpm->m/s: {obtido_vel[0]:.4f} m/s (esperado {v_ref:.4f}) -> "
          f"{'OK' if ok_valor else 'FALHOU'}")

    # Nas chamadas sem dado novo, o valor devolvido é o anterior, e a
    # flag avisa.
    ok_b2 = (obtido_vel[1] == obtido_vel[0]) and (obtido_is_new[1] is False)
    print(f"\n3) sem dado novo devolve valor antigo MAS com is_new=False: "
          f"{'OK' if ok_b2 else 'FALHOU'}")

    est = enc.estatisticas()
    ok_cont = (est["lidas"] == 2 and est["vazias"] == 2
               and est["erro_parse"] == 2 and est["descartadas"] == 1)
    print(f"\n4) contadores: {est}")
    print(f"   esperado lidas=2, vazias=2, erro_parse=2, descartadas=1 -> "
          f"{'OK' if ok_cont else 'FALHOU'}")

    # KeyboardInterrupt não pode ser engolido.
    class FonteQueInterrompe:
        def ler_todas(self):
            raise KeyboardInterrupt()

        def fechar(self):
            pass

    enc2 = Encoder(fonte=FonteQueInterrompe())
    try:
        enc2.getVel()
        ok_ki = False
    except KeyboardInterrupt:
        ok_ki = True
    print(f"\n5) KeyboardInterrupt propaga (nao e engolido): "
          f"{'OK' if ok_ki else 'FALHOU'}")

    tudo = all([ok_flags, ok_valor, ok_b2, ok_cont, ok_ki])
    print(f"\n=== RESULTADO: {'OK' if tudo else 'FALHOU'} ===")
    return tudo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Driver do encoder")
    ap.add_argument("--fake", action="store_true", help="autoteste sem hardware")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--baud", type=int, default=ENCODER_BAUDRATE,
                    help=f"baud da serial (padrao {ENCODER_BAUDRATE}). Use para "
                         "descobrir o baud correto: rode com um valor e olhe "
                         "erro_parse nas estatisticas — o baud certo da ~0.")
    ap.add_argument("--sem-plot", action="store_true", dest="sem_plot",
                    help="so estatisticas, sem janela grafica (util por SSH)")
    ap.add_argument("--csv", default=None,
                    help="salva as amostras BRUTAS num CSV (t_s, vel_m_s, is_new). "
                         "Com velocidade constante, da uma estimativa do ruido.")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    enc = Encoder(baudrate=args.baud)
    try:
        if args.sem_plot:
            enc.amostrar(duration=args.duration, csv=args.csv)
        else:
            enc.plot(duration=args.duration)
    finally:
        enc.close()
