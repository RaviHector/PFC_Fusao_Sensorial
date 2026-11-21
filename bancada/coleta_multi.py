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
"""Coleta sincronizada de encoder e IMU, num laço só."""
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
import io
import math
import os
import sys
import time

from kalman.config import DT_LOOP
from kalman.utils import Timer

# Uma linha por instante, com os dois sensores.
COLUNAS = [
    "t_s", "dt_medido_s",
    "enc_valido", "vel_m_s",
    "imu_valido", "ax_ms2", "ay_ms2", "az_ms2",
    "roll_rad", "pitch_rad", "yaw_rad",
    "gx_rads", "gy_rads", "gz_rads",
]


def _cabecalho(args, extras):
    linhas = {
        "coleta": "multi (encoder + imu, laco unico)",
        "gravado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "duracao_pedida_s": args.tempo,
        "dt_alvo_s": DT_LOOP,
        "condicao": args.condicao or "(nao informada)",
        "sincronizado": "sim — os dois sensores na mesma linha, mesmo t_s",
    }
    linhas.update(extras or {})
    return [f"# {k}: {v}" for k, v in linhas.items()]


def _abrir(caminho, args, extras=None):
    os.makedirs(os.path.dirname(os.path.abspath(caminho)) or ".", exist_ok=True)
    f = io.open(caminho, "w", newline="", encoding="utf-8")
    for linha in _cabecalho(args, extras):
        f.write(linha + "\n")
    w = csv.writer(f)
    w.writerow(COLUNAS)
    return f, w


def _linha_imu(imu):
    """Lê a IMU e devolve (ok, campos). Uma falha de leitura não derruba a coleta."""
    try:
        ax, ay, az = imu.getAccel()
        gx, gy, gz = imu.getGyro_rads()
        roll, pitch, yaw, _info = imu.getEuler_rad()
        ok = yaw is not None
        return ok, [
            f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
            f"{roll:.6f}", f"{pitch:.6f}",
            "" if yaw is None else f"{yaw:.6f}",
            f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}",
        ]
    except Exception:
        return False, [""] * 9


def coletar(args):
    from bancada.drivers.encoder import Encoder
    from bancada.drivers.imu import IMU

    enc = Encoder()
    imu = IMU()

    carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join("dados", f"multi_{carimbo}.csv")
    f, w = _abrir(caminho, args)

    print(f"[coleta_multi] gravando em {caminho}")
    if args.condicao:
        print(f"[coleta_multi] condicao: {args.condicao}")
    print("[coleta_multi] Ctrl+C interrompe e fecha o arquivo corretamente.")

    t0 = t_ant = Timer.now()
    n = n_enc = n_imu = 0
    try:
        while True:
            t_iter = Timer.now()
            if t_iter - t0 >= args.tempo:
                break
            dt = t_iter - t_ant
            t_ant = t_iter

            try:
                vel, ok_e = enc.getVel()
            except Exception:
                vel, ok_e = 0.0, False
            ok_i, campos = _linha_imu(imu)

            w.writerow([f"{t_iter - t0:.6f}", f"{dt:.6f}",
                        int(bool(ok_e)), f"{vel:.6f}",
                        int(bool(ok_i))] + campos)
            n += 1
            n_enc += int(bool(ok_e))
            n_imu += int(bool(ok_i))

            if n % 100 == 0:
                print(f"  {t_iter - t0:5.1f}s  {n:5d} amostras  "
                      f"encoder {100.0 * n_enc / n:5.1f}%  "
                      f"imu {100.0 * n_imu / n:5.1f}%")

            sobra = DT_LOOP - (Timer.now() - t_iter)
            if sobra > 0:
                time.sleep(sobra)
    except KeyboardInterrupt:
        print("\n[coleta_multi] interrompido; o arquivo esta integro.")
    finally:
        f.close()
        for d in (enc, imu):
            try:
                d.close()
            except Exception:
                pass

    print(f"\n  amostras : {n}")
    if n:
        print(f"  encoder  : {n_enc} validas ({100.0 * n_enc / n:.1f}%)")
        print(f"  imu      : {n_imu} validas ({100.0 * n_imu / n:.1f}%)")
    print(f"  arquivo  : {caminho}")
    return 0


def autoteste():
    """Exercita o formato do arquivo sem hardware, com sensores simulados."""
    print("=== autoteste de coleta_multi ===")

    class ArgsFake:
        tempo = 1.0
        condicao = "autoteste sintetico"

    args = ArgsFake()
    # Na pasta corrente, e não no temp do sistema, que pode estar sem espaço.
    caminho = "_autoteste_coleta_multi.csv"
    f, w = _abrir(caminho, args, {"fake": "sim"})

    t0 = Timer.now()
    n = 0
    while Timer.now() - t0 < args.tempo:
        t = Timer.now() - t0
        vel = 0.5 + 0.01 * math.sin(6.0 * t)
        ax = -1.04 if t > 0.5 else 0.0
        w.writerow([f"{t:.6f}", f"{DT_LOOP:.6f}", 1, f"{vel:.6f}", 1,
                    f"{ax:.6f}", "0.000000", "9.810000",
                    "0.000000", "0.000000", "0.000000",
                    "0.000000", "0.000000", "0.000000"])
        n += 1
        time.sleep(DT_LOOP)
    f.close()

    linhas = [l for l in io.open(caminho, encoding="utf-8") if not l.startswith("#")]
    leitor = csv.DictReader(linhas)
    dados = list(leitor)

    ok = True
    if leitor.fieldnames != COLUNAS:
        print(f"  FALHOU: cabecalho divergente\n    {leitor.fieldnames}")
        ok = False
    else:
        print(f"  cabecalho: {len(COLUNAS)} colunas, confere")

    if len(dados) < 10:
        print(f"  FALHOU: poucas linhas ({len(dados)})")
        ok = False
    else:
        print(f"  linhas gravadas: {len(dados)}")

    # os dois sensores na mesma linha e no mesmo t_s
    for d in dados:
        if d["vel_m_s"] == "" or d["ax_ms2"] == "" or d["t_s"] == "":
            print("  FALHOU: linha com sensor faltando")
            ok = False
            break
    else:
        print("  os dois sensores presentes em todas as linhas")

    ts = [float(d["t_s"]) for d in dados]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        print("  FALHOU: t_s nao e monotonico")
        ok = False
    else:
        print(f"  t_s monotonico, de {ts[0]:.3f} a {ts[-1]:.3f} s")

    try:
        os.remove(caminho)
    except OSError:
        pass
    print(f"\n=== RESULTADO: {'OK' if ok else 'FALHOU'} ===")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tempo", type=float, default=20.0,
                    help="duracao maxima em segundos (Ctrl+C encerra antes)")
    ap.add_argument("--condicao", default=None,
                    help="descricao do ensaio, gravada no cabecalho. "
                         "SEMPRE preencher: e o que distingue dois CSVs depois.")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste sem hardware")
    args = ap.parse_args(argv)

    if args.fake:
        return autoteste()
    return coletar(args)


if __name__ == "__main__":
    sys.exit(main())
