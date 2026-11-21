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
Corrida com acelerador constante, gravando os três sensores no mesmo laço.
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
import datetime
import os
import threading
import time

CAUDA_PADRAO = 5.0  # [s] tempo mínimo de gravação após o corte do motor
ARRANQUE_PADRAO = 1.0  # [s] trecho parado no inicio, para o log ter referencia

# v_real = 0,942 * velmax - 0,095, medido em cinco corridas
CALIB_GANHO = 0.942
CALIB_OFFSET = -0.095

# Abaixo deste velmax o veículo não sustenta movimento: perto da zona morta o
# motor entrega tração aos trancos (a 0,30 mal sai do lugar).
VELMAX_MINIMO_UTIL = 0.7

# Colunas do CSV de corrida: os três sensores no mesmo laço e no mesmo t_s,
# para que as sete configurações do filtro rodem sobre os mesmos dados.
COLUNAS = [
    "t_s", "dt_medido_s",
    "enc_valido", "vel_m_s",
    "imu_valido", "ax_ms2", "ay_ms2", "az_ms2",
    "roll_rad", "pitch_rad", "yaw_rad", "gx_rads", "gy_rads", "gz_rads",
    "us_valido", "us_novo", "dist_m", "us_motivo",
]


def _perfil(tempo, arranque, corte):
    """Resolve os instantes da corrida, com as checagens de coerência."""
    if corte is None:
        corte = tempo - CAUDA_PADRAO
    if corte <= arranque:
        raise SystemExit(
            f"[corrida] --corte ({corte:.1f} s) precisa ser maior que "
            f"--arranque ({arranque:.1f} s)")
    if corte > tempo:
        raise SystemExit(
            f"[corrida] --corte ({corte:.1f} s) passa do --tempo ({tempo:.1f} s); "
            f"o motor nunca seria cortado dentro da gravacao")
    return arranque, corte


def _valida_steer(steer_deg):
    """
    Converte o `--steer` para radianos e recusa o que o servo não entrega.
    """
    import numpy as np
    from kalman.config import STEER_MAX

    steer_rad = np.deg2rad(float(steer_deg))
    if abs(steer_rad) > STEER_MAX + 1e-9:
        raise SystemExit(
            "[corrida] --steer %.1f graus passa do limite de %.1f graus do "
            "class_servos, e seria cortado em silencio."
            % (float(steer_deg), np.rad2deg(STEER_MAX)))
    return steer_rad


def _avisa_velmax(velmax):
    """
    Avisa se o velmax pedido cai na faixa em que o veículo não anda direito.
    """
    if velmax < VELMAX_MINIMO_UTIL:
        print(f"[corrida] AVISO: velmax={velmax:.2f} esta abaixo de "
              f"{VELMAX_MINIMO_UTIL:.2f}, o minimo utilizavel medido.")
        print("[corrida]   Nessa faixa o motor fica na zona morta e a tracao "
              "sai aos trancos.")
        print("[corrida]   Medido: velmax 0,30 -> 0,12 m/s (nao andou); "
              "0,50 -> 0,15 m/s com sigma 0,151.")
    v = CALIB_GANHO * velmax + CALIB_OFFSET
    print(f"[corrida] velmax={velmax:.2f} deve dar ~{v:.2f} m/s "
          f"(pela calibracao; o encoder mede o valor real)")


def _descreve(tempo, arranque, corte, velmax, steer_deg=0.0):
    import numpy as np
    from kalman.config import ENTRE_EIXOS_L

    _avisa_velmax(velmax)
    print("[corrida] perfil:")
    print(f"    0,0 a {arranque:4.1f} s   parado, motor em repouso")
    print(f"  {arranque:5.1f} a {corte:4.1f} s   set_torque(1.0), acelerador "
          f"sobe e SATURA em velmax={velmax:.2f}")
    if corte >= tempo:
        print(f"  {corte:5.1f} s          fim da gravacao com o motor AINDA "
              f"LIGADO (sem costeio)")
    else:
        print(f"  {corte:5.1f} a {tempo:4.1f} s   motor cortado, veiculo costeia "
              f"({tempo - corte:.1f} s de cauda)")

    if abs(steer_deg) < 1e-9:
        print("[corrida] estercamento: ZERO, linha reta")
        return
    # O raio previsto é impresso porque decide se o ensaio cabe no espaço
    # disponível.
    raio = ENTRE_EIXOS_L / np.tan(np.deg2rad(abs(steer_deg)))
    print(f"[corrida] estercamento CONSTANTE de {steer_deg:+.1f} graus, a "
          f"corrida inteira")
    print(f"[corrida]   raio previsto R = L/tan(delta) = {raio:.2f} m, "
          f"diametro {2*raio:.2f} m")
    print(f"[corrida]   com L = {ENTRE_EIXOS_L:.3f} m, que e DECLARADO e nao "
          f"medido; conferir com trena")


def _cortar_motor(servos):
    """Zera a tração de verdade."""
    servos.set_torque(0.0)          # para de integrar
    with servos.lock:
        servos.dth_pwm = 0.0
        servos.th_pwm = 0.0  # e zera o que já foi acumulado


def _piloto(servos, parar, arranque, corte, pronto):
    """Thread que aciona."""
    def espera_ate(t_alvo, t0):
        while not parar.is_set() and (time.monotonic() - t0) < t_alvo:
            time.sleep(0.05)

    # o t0 do piloto passa a ser o mesmo instante do t0 da gravação
    while not pronto.wait(timeout=0.05):
        if parar.is_set():
            return
    t0 = time.monotonic()
    espera_ate(arranque, t0)
    if parar.is_set():
        return
    servos.set_forward()
    servos.set_torque(1.0)
    print(f"[corrida] t={arranque:.1f}s  acelerador liberado")

    espera_ate(corte, t0)
    _cortar_motor(servos)
    print(f"[corrida] t={corte:.1f}s  motor cortado, costeando")


def _coletar_tres(args, arranque, corte, pronto=None):
    """Laço único com encoder, IMU e ultrassom, um t_s só para os três."""
    from bancada.drivers.encoder import Encoder
    from bancada.drivers.imu import IMU
    from bancada.drivers.ultrassom import Ultrassom
    from kalman.utils import Timer

    # Se um dos três drivers falhar ao subir, os anteriores precisam ser
    # fechados, senão deixam o barramento preso e quebram a execução seguinte.
    enc = imu = us = None
    try:
        enc = Encoder()
        imu = IMU()
        us = Ultrassom(temp_c=args.temp, usar_thread=True,
                       **({} if args.max_range is None
                          else {"max_range": args.max_range}))
    except Exception:
        for d in (enc, imu, us):
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
        raise

    carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join("dados", f"corrida_{carimbo}.csv")
    os.makedirs("dados", exist_ok=True)
    f = open(caminho, "w", newline="", encoding="utf-8")
    f.write(f"# coleta: corrida (encoder + imu + ultrassom, laco unico)\n")
    f.write(f"# gravado_em: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# duracao_pedida_s: {args.tempo}\n")
    f.write(f"# dt_alvo_s: {args.dt}\n")
    f.write(f"# velmax_mps: {args.velmax}\n")
    f.write(f"# steer_cmd_deg: {getattr(args, 'steer', 0.0)}\n")
    f.write(f"# arranque_s: {arranque}\n")
    f.write(f"# corte_s: {corte}\n")
    f.write(f"# temp_ar_c: {args.temp}\n")
    f.write(f"# condicao: {args.condicao or ''}\n")
    f.write("# sincronizado: sim -- os tres sensores na mesma linha, mesmo t_s\n")
    w = csv.writer(f)
    w.writerow(COLUNAS)

    print(f"[corrida] gravando em {caminho}")
    print("[corrida] Ctrl+C interrompe, CORTA O MOTOR e fecha o arquivo.")

    # Libera o piloto agora, para que os dois relógios sejam o mesmo (ver a
    # docstring de `_piloto`).
    t0 = t_ant = Timer.now()
    if pronto is not None:
        pronto.set()
    t_us_ant = None
    n = n_e = n_i = n_u = n_novo = 0
    try:
        while True:
            t_it = Timer.now()
            if t_it - t0 >= args.tempo:
                break
            dt = t_it - t_ant
            t_ant = t_it

            try:
                vel, ok_e = enc.getVel()
            except Exception:
                vel, ok_e = 0.0, False

            try:
                ax, ay, az = imu.getAccel()
                roll, pitch, yaw, _ = imu.getEuler_rad()
                gx, gy, gz = imu.getGyro_rads()
                ok_i = yaw is not None
            except Exception:
                ax = ay = az = roll = pitch = yaw = gx = gy = gz = None
                ok_i = False

            try:
                dist, ok_u, t_us = us.ler_da_thread()
                novo = (t_us is not None) and (t_us != t_us_ant)
                t_us_ant = t_us
                motivo = "" if ok_u else us.ultimo_motivo()
            except Exception:
                dist, ok_u, novo, motivo = None, False, False, "erro"

            def s(v):
                return "" if v is None else f"{v:.6f}"

            w.writerow([f"{t_it - t0:.6f}", f"{dt:.6f}",
                        int(bool(ok_e)), s(vel),
                        int(bool(ok_i)), s(ax), s(ay), s(az),
                        s(roll), s(pitch), s(yaw), s(gx), s(gy), s(gz),
                        int(bool(ok_u)), int(bool(novo)), s(dist), motivo])
            n += 1
            n_e += int(bool(ok_e))
            n_i += int(bool(ok_i))
            n_u += int(bool(ok_u))
            n_novo += int(bool(novo))

            resto = args.dt - (Timer.now() - t_it)
            if resto > 0:
                time.sleep(resto)
    except KeyboardInterrupt:
        print("\n[corrida] interrompido pelo operador")
    finally:
        f.close()
        # Fechar os três.
        for d in (enc, imu, us):
            try:
                d.close()
            except Exception:
                pass

    if n:
        print(f"[corrida] {n} amostras em {Timer.now() - t0:.1f} s")
        print(f"[corrida]   encoder   validas: {100*n_e/n:5.1f}%")
        print(f"[corrida]   imu       validas: {100*n_i/n:5.1f}%")
        print(f"[corrida]   ultrassom validas: {100*n_u/n:5.1f}%  "
              f"(novas: {100*n_novo/n:5.1f}%)")
        try:
            print(f"[corrida]   ultrassom stats: {us.estatisticas()}")
        except Exception:
            pass
    return caminho


def _abrir_servos(velmax, steer_rad=0.0, tentativas=3, espera=0.8):
    """Cria os Servos, com nova tentativa se o PCA9685 não responder no I2C."""
    from class_servos import Servos

    for k in range(1, tentativas + 1):
        try:
            return Servos(steering=steer_rad, velmax=velmax, ultrasonic=False)
        except (ValueError, OSError) as e:
            if k == tentativas:
                print("")
                print("[corrida] O controlador de servos (PCA9685, 0x40) nao "
                      "respondeu em %d tentativas." % tentativas)
                print("[corrida] Erro: %s" % e)
                print("[corrida] O que conferir, em ordem:")
                print("           1. o carrinho esta LIGADO e a bateria tem carga")
                print("           2. `i2cdetect -y 1` mostra 40 e 68")
                print("           3. desligar e ligar o carrinho costuma resolver")
                raise SystemExit(1)
            print("[corrida] I2C nao respondeu (tentativa %d de %d), "
                  "repetindo em %.1f s..." % (k, tentativas, espera))
            time.sleep(espera)


def correr(args):
    arranque, corte = _perfil(args.tempo, args.arranque, args.corte)
    steer_rad = _valida_steer(getattr(args, "steer", 0.0))

    if getattr(args, "sem_motor", False):
        if abs(steer_rad) > 1e-9:
            print("[corrida] AVISO: --steer e IGNORADO com --sem-motor. Os "
                  "servos nao sao abertos, entao a direcao fica onde estiver.")
        print("[corrida] MODO SEM MOTOR: o acionamento nao sera tocado.")
        print("[corrida] Empurre o veiculo a mao, devagar e em linha reta.")
        print("[corrida] Anote a distancia de trena no --condicao.")
        print("[corrida] gravando por %.1f s" % args.tempo)
        _coletar_tres(args, arranque, corte, pronto=None)
        return

    _descreve(args.tempo, arranque, corte, args.velmax,
              getattr(args, "steer", 0.0))

    servos = _abrir_servos(args.velmax, steer_rad)
    parar = threading.Event()
    pronto = threading.Event()
    piloto = threading.Thread(target=_piloto,
                              args=(servos, parar, arranque, corte, pronto),
                              daemon=True)
    try:
        piloto.start()
        _coletar_tres(args, arranque, corte, pronto)
    finally:
        # A ordem importa: cortar o motor antes de fechar. Um `close()` que
        # falhe no meio não pode deixar o veículo acelerando.
        parar.set()
        try:
            _cortar_motor(servos)
        finally:
            servos.close()
        print("[corrida] motor cortado e servos encerrados")


def autoteste():
    """Verdade conhecida: o perfil resolvido, sem tocar em hardware."""
    print("=== autoteste de bancada.corrida (sem hardware) ===")
    ok = True

    a, c = _perfil(tempo=12.0, arranque=1.0, corte=None)
    print(f"  corte default com tempo=12: {c:.1f} s (esperado 7.0)")
    ok &= abs(c - 7.0) < 1e-9

    a, c = _perfil(tempo=20.0, arranque=1.0, corte=8.0)
    print(f"  corte explicito: {c:.1f} s (esperado 8.0)")
    ok &= abs(c - 8.0) < 1e-9

    for tempo, arranque, corte in ((12.0, 1.0, 0.5), (12.0, 1.0, 30.0)):
        try:
            _perfil(tempo, arranque, corte)
            print(f"  ACEITOU perfil invalido ({tempo}, {arranque}, {corte})")
            ok = False
        except SystemExit:
            print(f"  rejeitou perfil invalido corte={corte}: OK")

    # Importar de verdade: um erro de nome só apareceria na Pi, com o
    # carrinho montado.
    try:
        from bancada.drivers.encoder import Encoder          # noqa: F401
        from bancada.drivers.imu import IMU                  # noqa: F401
        from bancada.drivers.ultrassom import Ultrassom      # noqa: F401
        from kalman.utils import Timer                       # noqa: F401
        print("  imports dos tres drivers: OK")
    except Exception as e:
        print(f"  imports FALHARAM: {e}")
        ok = False

    print(f"  colunas do CSV: {len(COLUNAS)}")
    ok &= len(COLUNAS) == len(set(COLUNAS))
    ok &= COLUNAS[:2] == ["t_s", "dt_medido_s"]
    ok &= "us_novo" in COLUNAS

    # --steer: converte, e recusa o que o servo cortaria em silêncio
    import numpy as _np

    st = _valida_steer(15.0)
    print(f"  --steer 15 -> {_np.rad2deg(st):.1f} graus")
    ok &= abs(st - _np.deg2rad(15.0)) < 1e-12
    ok &= abs(_valida_steer(0.0)) < 1e-12
    for ruim in (25.0, -30.0):
        try:
            _valida_steer(ruim)
            print(f"  ACEITOU --steer {ruim} graus, que passa do limite")
            ok = False
        except SystemExit:
            print(f"  rejeitou --steer {ruim} graus: OK")

    print("")
    _descreve(12.0, 1.0, 7.0, 0.5)
    print("")
    _descreve(30.0, 1.0, 30.0, 1.0, steer_deg=15.0)
    print("")
    print("=== RESULTADO:", "OK" if ok else "FALHOU", "===")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--velmax", type=float, default=0.5,
                    help="teto de velocidade [m/s], 0.3 a 1.5 (padrao 0.5)")
    ap.add_argument("--tempo", type=float, default=12.0,
                    help="duracao da gravacao [s] (padrao 12)")
    ap.add_argument("--corte", type=float, default=None,
                    help=f"instante de cortar o motor [s] "
                         f"(padrao: tempo - {CAUDA_PADRAO:.0f})")
    ap.add_argument("--arranque", type=float, default=ARRANQUE_PADRAO,
                    help="trecho parado no inicio [s] (padrao 1.0)")
    ap.add_argument("--dt", type=float, default=0.05,
                    help="periodo do laco [s] (padrao 0.05, os 20 Hz do veiculo)")
    ap.add_argument("--temp", type=float, default=24.0,
                    help="temperatura do ar [C]; a velocidade do som depende dela")
    ap.add_argument("--condicao", default=None,
                    help="texto que vai no cabecalho do CSV")
    ap.add_argument("--max-range", type=float, default=None,
                    dest="max_range",
                    help="corte de confianca do ultrassom [m]. Padrao: o "
                         "US_MAX_RANGE do config (2,00). So mexer em ensaio "
                         "que exige largada parada fora da faixa.")
    ap.add_argument("--steer", type=float, default=0.0,
                    help="estercamento CONSTANTE [graus], -20 a 20 (padrao 0, "
                         "reta). Vai pelo construtor dos Servos, para valer "
                         "desde o primeiro instante. Para o ensaio de circulo.")
    ap.add_argument("--sem-motor", action="store_true", dest="sem_motor",
                    help="grava os tres sensores SEM acionar o motor; para o "
                         "ensaio de empurrao a mao (raio efetivo da roda)")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste, sem hardware")
    args = ap.parse_args(argv)

    if args.fake:
        return 0 if autoteste() else 1
    correr(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
