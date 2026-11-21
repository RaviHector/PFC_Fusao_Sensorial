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
"""Coleta de dados brutos de um sensor, para os ensaios de caracterização."""
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
import math
import os
import time

from kalman.config import DT_LOOP, SOUND_TEMP_C
from kalman.utils import Timer

# Colunas mínimas, comuns a todos os sensores.
COLUNAS_BASE = ["t_s", "dt_medido_s", "valido"]

# Colunas extras por sensor: o valor bruto de cada canal.
COLUNAS_EXTRA = {
    "encoder": ["vel_m_s"],
    "ultrassom": ["dist_m", "motivo"],
    "imu": ["ax_ms2", "ay_ms2", "az_ms2",
            "roll_rad", "pitch_rad", "yaw_rad",
            "gx_rads", "gy_rads", "gz_rads", "accel_confiavel"],
}

SENSORES = tuple(COLUNAS_EXTRA)


def _cabecalho_meta(sensor, args, extras=None):
    """
    Linhas de metadados, comentadas com `#`, no topo do CSV: temperatura, baud,
    duração e condição do ensaio.
    """
    meta = {
        "sensor": sensor,
        "gravado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "duracao_pedida_s": args.tempo,
        "dt_alvo_s": DT_LOOP,
        "condicao": args.condicao or "(nao informada)",
    }
    if sensor == "ultrassom":
        meta["temp_c"] = args.temp
    if extras:
        meta.update(extras)
    return [f"# {k}: {v}" for k, v in meta.items()]


def _abrir_saida(caminho, sensor, args, extras=None):
    colunas = COLUNAS_BASE + COLUNAS_EXTRA[sensor]
    os.makedirs(os.path.dirname(os.path.abspath(caminho)) or ".", exist_ok=True)
    f = open(caminho, "w", newline="", encoding="utf-8")
    for linha in _cabecalho_meta(sensor, args, extras):
        f.write(linha + "\n")
    w = csv.writer(f)
    w.writerow(colunas)
    return f, w


def coletar(sensor, driver, tempo, caminho, args, extras=None):
    """Laço de coleta."""
    f, w = _abrir_saida(caminho, sensor, args, extras)
    n = 0
    n_validas = 0
    t0 = Timer.now()
    t_anterior = t0

    print(f"[coleta] sensor={sensor}  duracao={tempo:.0f}s  saida={caminho}")
    print("[coleta] gravando dado BRUTO — nenhum filtro no caminho")
    if args.condicao:
        print(f"[coleta] condicao: {args.condicao}")
    print("[coleta] Ctrl+C interrompe e fecha o arquivo corretamente.")

    try:
        while True:
            t_iter = Timer.now()
            if t_iter - t0 >= tempo:
                break

            dt = t_iter - t_anterior
            t_anterior = t_iter

            if sensor == "encoder":
                vel, ok = driver.getVel()
                linha = [f"{vel:.6f}"]
            elif sensor == "ultrassom":
                dist, ok, _ = driver.ler()
                motivo = "" if ok else driver.ultimo_motivo()
                linha = ["" if dist is None else f"{dist:.6f}", motivo]
            else:  # imu
                ax, ay, az = driver.getAccel()
                roll, pitch, yaw, info = driver.getEuler_rad()
                gx, gy, gz = driver.getGyro_rads()
                ok = yaw is not None
                linha = [
                    f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                    f"{roll:.6f}", f"{pitch:.6f}",
                    "" if yaw is None else f"{yaw:.6f}",
                    f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}",
                    int(bool(info.get("accel_confiavel", True))),
                ]

            w.writerow([f"{t_iter - t0:.6f}", f"{dt:.6f}", int(bool(ok))] + linha)
            n += 1
            n_validas += int(bool(ok))

            if n % 100 == 0:
                pct = 100.0 * n_validas / n
                print(f"  {t_iter - t0:5.1f}s  {n:5d} amostras  {pct:5.1f}% validas")

            sobra = DT_LOOP - (Timer.now() - t_iter)
            if sobra > 0:
                time.sleep(sobra)

    except KeyboardInterrupt:
        print("\n[coleta] interrompido — arquivo fechado com o que ja foi gravado")
    finally:
        f.close()

    dur = Timer.now() - t0
    print()
    print(f"[coleta] {n} amostras em {dur:.1f}s = {n / dur if dur else 0:.1f} Hz")
    print(f"[coleta] validas: {n_validas}/{n} = {100.0 * n_validas / n if n else 0:.1f}%")
    est = driver.estatisticas()
    print(f"[coleta] estatisticas do driver: {est}")
    print()
    print(f"Proximo passo:  python -m bancada.analise {caminho}")
    return est


def _driver_fake(sensor, seed=0):
    """Driver com fonte sintética, para o autoteste."""
    import random

    from bancada.drivers.base import FonteGPIOFake, FonteI2CFake, FonteSerialFake

    rng = random.Random(seed)

    if sensor == "encoder":
        # RPM constante + ruído gaussiano. Em m/s isso vira SIGMA_ENC_FAKE.
        from bancada.drivers.encoder import Encoder

        roteiro = [["%.4f\n" % (780.0 + rng.gauss(0.0, 10.0))] for _ in range(4000)]
        return Encoder(fonte=FonteSerialFake(roteiro))

    if sensor == "ultrassom":
        from bancada.drivers.ultrassom import Ultrassom

        return Ultrassom(fonte=FonteGPIOFake(distancia_m=1.0), temp_c=24.0)

    from bancada.drivers.imu import IMU

    return IMU(fonte=FonteI2CFake(accel_g=(0.0, 0.0, 1.0),
                                  gyro_dps=(0.0, 0.0, 0.0),
                                  ruido=0.01, seed=seed))


def _autoteste():
    import tempfile

    print("=== bancada.coleta — autoteste ===\n")
    ok_tudo = True
    tmp = tempfile.mkdtemp(prefix="coleta_")

    class ArgsFake:
        tempo = 2.0
        temp = 24.0
        condicao = "autoteste sintetico"

    for sensor in SENSORES:
        caminho = os.path.join(tmp, f"{sensor}.csv")
        drv = _driver_fake(sensor)
        try:
            coletar(sensor, drv, 2.0, caminho, ArgsFake())
        finally:
            drv.close()

        with open(caminho, encoding="utf-8") as f:
            linhas = f.read().splitlines()
        meta = [l for l in linhas if l.startswith("#")]
        dados = [l for l in linhas if not l.startswith("#")]
        cab = dados[0].split(",")
        esperado = COLUNAS_BASE + COLUNAS_EXTRA[sensor]

        ok_cab = cab == esperado
        ok_meta = len(meta) >= 4
        ok_n = len(dados) - 1 > 10
        # dt_medido tem de ser plausível: nem zero, nem absurdo
        dts = [float(l.split(",")[1]) for l in dados[1:]]
        ok_dt = all(0.0 < d < 1.0 for d in dts[1:])

        bom = ok_cab and ok_meta and ok_n and ok_dt
        ok_tudo &= bom
        print(f"  {sensor:<10} colunas={'OK' if ok_cab else 'X'} "
              f"meta={'OK' if ok_meta else 'X'} "
              f"n={len(dados) - 1} "
              f"dt={'OK' if ok_dt else 'X'} -> {'OK' if bom else 'FALHOU'}\n")

    print(f"CSVs em {tmp}")
    print(f"\n=== RESULTADO: {'OK' if ok_tudo else 'FALHOU'} ===")
    return ok_tudo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Coleta de dados brutos de um sensor")
    ap.add_argument("--sensor", choices=SENSORES,
                    help="qual sensor coletar")
    ap.add_argument("--tempo", type=float, default=60.0,
                    help="duracao em segundos (padrao 60)")
    ap.add_argument("--saida", default=None,
                    help="caminho do CSV (padrao: dados/<sensor>_<datahora>.csv)")
    ap.add_argument("--temp", type=float, default=SOUND_TEMP_C,
                    help=f"temperatura do ar em C, so para o ultrassom "
                         f"(padrao {SOUND_TEMP_C}). MEDIR de verdade — 20 vs 25 C "
                         f"da 0,9%% na velocidade do som.")
    ap.add_argument("--min-range", type=float, default=None, dest="min_range",
                    help="[ultrassom] limite inferior de confianca [m]. O padrao "
                         "vem de US_MIN_RANGE (0,03 m) e REJEITA tudo abaixo. "
                         "Para medir a partir de onde o sensor deixa de "
                         "acompanhar, e preciso baixa-lo (ex. 0.01), "
                         "senao as leituras que interessam sao descartadas "
                         "antes de aparecerem no CSV.")
    ap.add_argument("--condicao", default=None,
                    help="descricao da condicao do ensaio, gravada no cabecalho. "
                         "Ex.: 'parado, piso liso, 1m da parede'. Sem isto "
                         "os CSVs de condicoes diferentes ficam "
                         "indistinguiveis.")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste com dados sinteticos, sem hardware")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    if not args.sensor:
        ap.error("--sensor e obrigatorio (ou use --fake)")

    if args.saida is None:
        carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join("dados", f"{args.sensor}_{carimbo}.csv")

    if args.sensor == "encoder":
        from bancada.drivers.encoder import Encoder

        drv = Encoder()
    elif args.sensor == "ultrassom":
        from bancada.drivers.ultrassom import Ultrassom

        # `min_range` só é repassado quando o usuário pede, para o padrão de
        # US_MIN_RANGE continuar valendo em todos os demais ensaios.
        kw = {} if args.min_range is None else {"min_range": args.min_range}
        drv = Ultrassom(temp_c=args.temp, **kw)
    else:
        from bancada.drivers.imu import IMU

        drv = IMU()

    extras = None
    if args.sensor == "ultrassom":
        extras = {"min_range_m": drv.min_range, "max_range_m": drv.max_range}

    try:
        coletar(args.sensor, drv, args.tempo, args.saida, args, extras)
    finally:
        drv.close()
