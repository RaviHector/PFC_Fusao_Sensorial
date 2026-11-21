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
"""Verifica o ambiente da Raspberry: o que existe, e não o que instalar."""
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
import importlib.util
import os
import platform
import subprocess
import sys

VERDE = "\033[32m"
VERM = "\033[31m"
AMAR = "\033[33m"
FIM = "\033[0m"


def _cor(txt, c):
    return f"{c}{txt}{FIM}" if sys.stdout.isatty() else txt


def _tem(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# (módulo, pacote pip, quem usa, obrigatório?)
PACOTES = [
    ("numpy", "python3-numpy (apt)", "tudo", True),
    ("serial", "pyserial", "encoder — serial com o Nano", True),
    ("smbus2", "smbus2", "IMU — I2C", True),
    ("matplotlib", "python3-matplotlib (apt)", "os plot() de bancada e o main.py", True),
    ("lgpio", "lgpio", "ultrassom na Pi 5", False),
    ("RPi", "RPi.GPIO (ja vem no Raspberry Pi OS)", "ultrassom na Pi 3/4", False),
    ("adafruit_servokit", "adafruit-circuitpython-servokit", "class_servos.py", True),
    ("smbus", "smbus (alternativa ao smbus2)", "IMU — fallback", False),
    ("cv2", "opencv-python", "class_camera.py — SO se usar camera", False),
    ("scipy", "scipy", "NAO precisa na Pi", False),
]


def checar_pacotes():
    print("=" * 66)
    print("1. PACOTES PYTHON")
    print("=" * 66)
    faltando_obrig = []
    for mod, pip, quem, obrig in PACOTES:
        ok = _tem(mod)
        if ok:
            marca = _cor("  OK  ", VERDE)
        elif obrig:
            marca = _cor(" FALTA", VERM)
            faltando_obrig.append(pip)
        else:
            marca = _cor("  --  ", AMAR)
        print(f"  [{marca}] {mod:<20} {quem}")

    # gpio: precisa de um dos dois, conforme o modelo
    if not _tem("lgpio") and not _tem("RPi"):
        print()
        print(_cor("  ATENCAO: nem lgpio nem RPi.GPIO encontrados.", VERM))
        print("  O ultrassom nao vai funcionar. Pi 5 usa lgpio; Pi 3/4 usa RPi.GPIO.")
        faltando_obrig.append("lgpio  # (ou RPi.GPIO, conforme o modelo)")

    print()
    if faltando_obrig:
        print(_cor("  Instalar so o que falta:", AMAR))
        print("    pip3 install --break-system-packages " +
              " ".join(p.split()[0] for p in faltando_obrig
                       if "apt" not in p and "#" not in p))
        for p in faltando_obrig:
            if "apt" in p:
                print(f"    sudo apt install -y {p.split()[0]}")
    else:
        print(_cor("  Nada falta. Nao instale nada.", VERDE))
    return not faltando_obrig


def checar_i2c():
    print()
    print("=" * 66)
    print("2. BARRAMENTO I2C (IMU)")
    print("=" * 66)
    achou = False
    for bus in (1, 4, 0):
        dev = f"/dev/i2c-{bus}"
        if not os.path.exists(dev):
            continue
        print(f"  {dev} existe", end="")
        try:
            saida = subprocess.run(["i2cdetect", "-y", str(bus)],
                                   capture_output=True, text=True, timeout=5).stdout
            if "68" in saida.replace("60:", ""):
                print(_cor("  -> MPU-9250 encontrado no endereco 0x68", VERDE))
                print(f"       usar bus_id={bus} ao instanciar a IMU")
                achou = True
            else:
                print(_cor("  -> nenhum 0x68 aqui", AMAR))
        except FileNotFoundError:
            print("  (i2cdetect nao instalado: sudo apt install i2c-tools)")
        except (subprocess.SubprocessError, OSError) as e:
            print(f"  (nao consegui rodar i2cdetect: {type(e).__name__})")
    if not achou:
        print()
        print(_cor("  IMU nao encontrada.", VERM))
        print("  Verificar: I2C habilitado no raspi-config? cabo conectado?")
    return achou


def checar_serial():
    print()
    print("=" * 66)
    print("3. PORTAS SERIAIS (encoder)")
    print("=" * 66)
    if not _tem("serial"):
        print("  pyserial nao instalado — pulando")
        return False
    from serial.tools import list_ports

    portas = list(list_ports.comports())
    if not portas:
        print(_cor("  Nenhuma porta serial encontrada.", VERM))
        print("  O Nano esta conectado? Cabo de DADOS (alguns cabos so dao energia)?")
        return False
    achou_nano = False
    for p in portas:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        e_nano = any(k in desc or k in hwid
                     for k in ("arduino", "wch", "1a86", "ftdi", "0403"))
        marca = _cor(" NANO ", VERDE) if e_nano else "      "
        print(f"  [{marca}] {p.device:<16} {p.description}")
        achou_nano |= e_nano
    if not achou_nano:
        print()
        print(_cor("  Nenhuma porta parece ser o Nano.", AMAR))
        print("  `find_arduino()` procura por: arduino, wch, 1a86, ftdi, 0403.")
    return achou_nano


def checar_permissoes():
    print()
    print("=" * 66)
    print("4. GRUPOS DO USUARIO")
    print("=" * 66)
    print("  Falta de grupo aparece como PermissionError e parece falha de")
    print("  hardware. Vale conferir antes de culpar o sensor.")
    print()
    try:
        grupos = subprocess.run(["groups"], capture_output=True, text=True,
                                timeout=5).stdout.split()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        print("  (nao consegui listar os grupos)")
        return True
    ok = True
    for g, para in (("dialout", "serial / encoder"),
                    ("i2c", "IMU"),
                    ("gpio", "ultrassom")):
        tem = g in grupos
        ok &= tem
        marca = _cor("  OK  ", VERDE) if tem else _cor(" FALTA", VERM)
        print(f"  [{marca}] {g:<10} {para}")
    if not ok:
        print()
        print(_cor("  Adicionar e RELOGAR (nao basta sudo):", AMAR))
        print("    sudo usermod -aG dialout,i2c,gpio $USER")
    return ok


def main():
    print()
    print("#" * 66)
    print("#  CHECAGEM DE AMBIENTE — Raspberry Pi")
    print("#" * 66)
    print(f"  python  : {platform.python_version()}")
    print(f"  sistema : {platform.system()} {platform.release()}")
    print(f"  maquina : {platform.machine()}")
    try:
        with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as f:
            print(f"  modelo  : {f.read().strip(chr(0))}")
    except OSError:
        print("  modelo  : (nao e uma Raspberry — checagem parcial)")
    print()

    r_pac = checar_pacotes()
    r_i2c = checar_i2c()
    r_ser = checar_serial()
    r_per = checar_permissoes()

    print()
    print("#" * 66)
    itens = [("pacotes", r_pac), ("I2C/IMU", r_i2c),
             ("serial/Nano", r_ser), ("grupos", r_per)]
    for nome, ok in itens:
        print(f"  {nome:<14} {_cor('OK', VERDE) if ok else _cor('PENDENTE', VERM)}")
    tudo = all(ok for _, ok in itens)
    print("#" * 66)
    if tudo:
        print(_cor("  Ambiente pronto. Seguir para os autotestes --fake.", VERDE))
    else:
        print(_cor("  Resolver os pendentes acima antes dos ensaios.", AMAR))
        print("  Se so o I2C ou so a serial estiver pendente, os outros")
        print("  sensores ainda podem ser testados normalmente.")
    return 0 if tudo else 1


if __name__ == "__main__":
    raise SystemExit(main())
