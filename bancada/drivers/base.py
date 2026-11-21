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
"""Camada de I/O dos drivers: fontes reais e falsas, intercambiáveis."""
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

import random
import time

from kalman.config import sound_speed

# Detecção da placa — usada pela fonte GPIO
def detect_rpi_version():
    """5 para Raspberry Pi 5 (lgpio/BCM), 4 para as anteriores (RPi.GPIO/BOARD)."""
    try:
        with open("/proc/device-tree/model") as f:
            return 5 if "Raspberry Pi 5" in f.read() else 4
    except Exception:
        return 4


# SERIAL — encoder (Arduino Nano)
class FonteSerial:
    """Serial real. `ler_todas()` devolve todas as linhas completas no buffer."""

    # Espera do boot do Nano depois de abrir a porta: abrir a serial reinicia o
    # Arduino, que leva ~1 s para voltar a transmitir.
    ESPERA_BOOT_S = 1.5

    def __init__(self, port, baudrate, timeout=0.2, esperar_boot=True):
        import serial  # importação tardia: só existe na Pi

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.port = port
        self.baudrate = baudrate

        if esperar_boot:
            time.sleep(self.ESPERA_BOOT_S)
            self.ser.reset_input_buffer()   # descarta o lixo do bootloader

    # Teto de linhas por chamada. A 33 Hz de firmware contra 20 Hz de laço,
    # o normal é 1 ou 2 linhas; 50 é folga enorme para qualquer atraso.
    MAX_LINHAS_POR_CHAMADA = 50

    def ler_todas(self):
        """Drena as linhas já recebidas, com teto."""
        linhas = []
        # `in_waiting` conta bytes, não linhas; o laço drena o que já chegou.
        while self.ser.in_waiting and len(linhas) < self.MAX_LINHAS_POR_CHAMADA:
            linhas.append(self.ser.readline())
        return linhas

    def fechar(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass


class FonteSerialFake:
    """Serial sintética roteirizada."""

    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.i = 0
        self.fechada = False

    def ler_todas(self):
        if self.i >= len(self.roteiro):
            return []
        linhas = self.roteiro[self.i]
        self.i += 1
        return [l.encode() if isinstance(l, str) else l for l in linhas]

    def fechar(self):
        self.fechada = True


# GPIO — ultrassom HC-SR04
class FonteGPIO:
    """GPIO real."""

    # Rótulos que identificam o controlador do conector de 40 pinos.
    # Na Pi 5 quem o expõe é o RP1; nas anteriores, o `pinctrl-bcm...`.
    ROTULOS_HEADER = ("rp1", "pinctrl", "bcm2835", "bcm2711")

    @staticmethod
    def _chips_disponiveis():
        """Indices dos /dev/gpiochip* que existem, em ordem."""
        import glob as _glob
        import re as _re

        ns = []
        for cam in _glob.glob("/dev/gpiochip*"):
            m = _re.search(r"(\d+)$", cam)
            if m:
                ns.append(int(m.group(1)))
        return sorted(ns)

    @staticmethod
    def _abrir_chip(GPIO, pinos_necessarios):
        """Abre o gpiochip que controla o conector de 40 pinos."""
        disponiveis = FonteGPIO._chips_disponiveis()
        if not disponiveis:
            raise RuntimeError(
                "nenhum /dev/gpiochip* encontrado. O kernel nao expos o "
                "controlador de GPIO.")

        erros = []
        candidatos = []  # (prioridade, indice, handle)
        for n in disponiveis:
            try:
                h = GPIO.gpiochip_open(n)
            except Exception as e:
                erros.append("gpiochip%d: %s" % (n, e))
                continue
            linhas, rotulo = None, ""
            try:
                info = GPIO.gpio_get_chip_info(h)
                linhas = info[1]
                rotulo = " ".join(str(x) for x in info[2:]).lower()
            except Exception:
                pass
            if linhas is not None and linhas <= max(pinos_necessarios):
                GPIO.gpiochip_close(h)
                erros.append("gpiochip%d: so %d linhas" % (n, linhas))
                continue
            prio = 0 if any(r in rotulo for r in FonteGPIO.ROTULOS_HEADER) else 1
            candidatos.append((prio, n, h, linhas, rotulo))

        if not candidatos:
            raise RuntimeError(
                "nenhum gpiochip utilizavel entre "
                + ", ".join(str(n) for n in disponiveis)
                + ". Detalhes: " + "; ".join(erros)
                + ". Conferir `ls -l /dev/gpiochip*` e o grupo `gpio` em `id`.")

        candidatos.sort(key=lambda c: (c[0], c[1]))
        escolhido = candidatos[0]
        for c in candidatos[1:]:
            try:
                GPIO.gpiochip_close(c[2])
            except Exception:
                pass
        return escolhido[2], escolhido[1]

    def __init__(self, triggerPin=None, echoPin=None):
        self.rpi_version = detect_rpi_version()

        if self.rpi_version == 5:
            import lgpio as GPIO

            self.GPIO = GPIO
            self.triggerPin = triggerPin if triggerPin is not None else 24
            self.echoPin = echoPin if echoPin is not None else 8
            self.handleChip, self.chip_id = self._abrir_chip(
                GPIO, (self.triggerPin, self.echoPin))
            # O índice do chip varia entre Pi e entre versões de kernel; imprimi-lo
            # facilita o diagnóstico.
            print("[gpio] usando /dev/gpiochip%d  (trigger=%d, echo=%d)"
                  % (self.chip_id, self.triggerPin, self.echoPin))
            GPIO.gpio_claim_output(self.handleChip, self.triggerPin)
            GPIO.gpio_claim_input(self.handleChip, self.echoPin)
        else:
            import RPi.GPIO as GPIO

            self.GPIO = GPIO
            self.triggerPin = triggerPin if triggerPin is not None else 18
            self.echoPin = echoPin if echoPin is not None else 24
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.triggerPin, GPIO.OUT)
            GPIO.setup(self.echoPin, GPIO.IN)

    def disparar(self):
        if self.rpi_version == 5:
            self.GPIO.gpio_write(self.handleChip, self.triggerPin, 1)
            time.sleep(0.00002)  # 20 us
            self.GPIO.gpio_write(self.handleChip, self.triggerPin, 0)
        else:
            self.GPIO.output(self.triggerPin, True)
            time.sleep(0.00002)
            self.GPIO.output(self.triggerPin, False)

    def nivel_echo(self):
        if self.rpi_version == 5:
            return self.GPIO.gpio_read(self.handleChip, self.echoPin)
        return self.GPIO.input(self.echoPin)

    def fechar(self):
        try:
            if self.rpi_version == 5:
                self.GPIO.gpiochip_close(self.handleChip)
            else:
                self.GPIO.cleanup()
        except Exception:
            pass


class FonteGPIOFake:
    """Simula o pino de eco em tempo real, a partir de uma distância alvo."""

    def __init__(self, distancia_m=1.0, sem_eco=False, temp_c=20.0, atraso_subida=0.0005):
        self.distancia_m = distancia_m
        self.sem_eco = sem_eco
        self.c = sound_speed(temp_c)
        self.atraso_subida = atraso_subida
        self.t_sobe = None
        self.t_desce = None
        self.fechada = False

    def disparar(self):
        if self.sem_eco:
            self.t_sobe = self.t_desce = None
            return
        agora = time.perf_counter()
        self.t_sobe = agora + self.atraso_subida
        # ida e volta: 2*d/c
        self.t_desce = self.t_sobe + 2.0 * self.distancia_m / self.c

    def nivel_echo(self):
        if self.t_sobe is None:
            return 0
        agora = time.perf_counter()
        return 1 if self.t_sobe <= agora < self.t_desce else 0

    def fechar(self):
        self.fechada = True


class FonteSerialESP32:
    """Esqueleto, não implementado: ultrassom medido por um ESP32 e lido pela serial."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Firmware do ESP32 ainda não existe"
        )


# I2C — IMU MPU-9250
class FonteI2C:
    """I2C real. `ler_bloco` lê os registradores de uma vez (leitura atômica)."""

    def __init__(self, bus_id=4):
        try:
            from smbus2 import SMBus
        except Exception:
            from smbus import SMBus

        self.bus = SMBus(bus_id)
        self.bus_id = bus_id

    def ler_bloco(self, addr, reg, n):
        return self.bus.read_i2c_block_data(addr, reg, n)

    def ler_byte(self, addr, reg):
        return self.bus.read_byte_data(addr, reg)

    def escrever_byte(self, addr, reg, valor):
        self.bus.write_byte_data(addr, reg, valor)

    def fechar(self):
        try:
            self.bus.close()
        except Exception:
            pass


class FonteI2CFake:
    """I2C sintético."""

    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    MAG_ADDR = 0x0C
    MAG_ST1 = 0x02
    MAG_HXL = 0x03
    MAG_WIA = 0x00
    MAG_ASAX = 0x10

    LSB_PER_G = 16384.0
    LSB_PER_DPS = 131.0
    MAG_UT_LSB = 0.15

    def __init__(self, accel_g=(0.0, 0.0, 1.0), gyro_dps=(0.0, 0.0, 0.0),
                 mag_uT=(30.0, 0.0, -20.0), drdy=True, ruido=0.0, seed=0):
        self.accel_g = list(accel_g)
        self.gyro_dps = list(gyro_dps)
        self.mag_uT = list(mag_uT)
        self.drdy = drdy
        self.ruido = ruido
        self.rng = random.Random(seed)
        self.fechada = False
        self.escritas = []

    # --- codificação
    @staticmethod
    def _para_be16(valor):
        """int com sinal -> (high, low), big-endian, complemento de dois."""
        v = int(round(valor))
        v = max(-32768, min(32767, v))
        if v < 0:
            v += 65536
        return (v >> 8) & 0xFF, v & 0xFF

    @staticmethod
    def _para_le16(valor):
        """int com sinal -> (low, high), little-endian (o AK8963 é assim)."""
        v = int(round(valor))
        v = max(-32768, min(32767, v))
        if v < 0:
            v += 65536
        return v & 0xFF, (v >> 8) & 0xFF

    def _com_ruido(self, v):
        return v + self.rng.gauss(0.0, self.ruido) if self.ruido else v

    # --- interface
    def ler_bloco(self, addr, reg, n):
        if addr == self.MAG_ADDR:
            return self._ler_bloco_mag(reg, n)

        if reg == self.ACCEL_XOUT_H:
            dados = []
            for g in self.accel_g:
                dados.extend(self._para_be16(self._com_ruido(g) * self.LSB_PER_G))
            return (dados + [0] * n)[:n]

        if reg == self.GYRO_XOUT_H:
            dados = []
            for w in self.gyro_dps:
                dados.extend(self._para_be16(self._com_ruido(w) * self.LSB_PER_DPS))
            return (dados + [0] * n)[:n]

        return [0] * n

    def _ler_bloco_mag(self, reg, n):
        if reg == self.MAG_ST1:
            # bit 0 = DRDY; o resto do bloco é HXL..ST2
            bloco = [0x01 if self.drdy else 0x00]
            for m in self.mag_uT:
                bloco.extend(self._para_le16(self._com_ruido(m) / self.MAG_UT_LSB))
            bloco.append(0x00)  # ST2 sem overflow
            return (bloco + [0] * n)[:n]

        if reg == self.MAG_HXL:
            bloco = []
            for m in self.mag_uT:
                bloco.extend(self._para_le16(self._com_ruido(m) / self.MAG_UT_LSB))
            bloco.append(0x00)  # ST2
            return (bloco + [0] * n)[:n]

        return [0] * n

    def ler_byte(self, addr, reg):
        if addr == self.MAG_ADDR:
            if reg == self.MAG_WIA:
                return 0x48  # identificação do AK8963
            if reg == self.MAG_ST1:
                return 0x01 if self.drdy else 0x00
            if self.MAG_ASAX <= reg < self.MAG_ASAX + 3:
                return 128  # ajuste de fábrica neutro
        return 0x00

    def escrever_byte(self, addr, reg, valor):
        self.escritas.append((addr, reg, valor))

    def fechar(self):
        self.fechada = True


if __name__ == "__main__":
    print("=== bancada.drivers.base — autoteste das fontes falsas ===\n")

    # --- serial
    fs = FonteSerialFake([["100.0\n"], [], ["abc\n"], ["200.0\n", "201.0\n"]])
    esperado = [1, 0, 1, 2]
    obtido = [len(fs.ler_todas()) for _ in range(4)]
    ok_serial = obtido == esperado
    print(f"serial fake: linhas por chamada {obtido}, esperado {esperado} -> "
          f"{'OK' if ok_serial else 'FALHOU'}")

    # --- gpio
    fg = FonteGPIOFake(distancia_m=1.0)
    fg.disparar()
    t0 = time.perf_counter()
    while fg.nivel_echo() == 0 and time.perf_counter() - t0 < 0.05:
        pass
    t_sobe = time.perf_counter()
    while fg.nivel_echo() == 1 and time.perf_counter() - t_sobe < 0.05:
        pass
    largura = time.perf_counter() - t_sobe
    d_medido = largura * sound_speed(20.0) / 2.0
    ok_gpio = abs(d_medido - 1.0) < 0.05
    print(f"gpio fake: distancia reconstruida {d_medido:.3f} m (alvo 1.000) -> "
          f"{'OK' if ok_gpio else 'FALHOU'}")

    fg_sem = FonteGPIOFake(sem_eco=True)
    fg_sem.disparar()
    ok_sem_eco = fg_sem.nivel_echo() == 0
    print(f"gpio fake sem_eco: pino permanece baixo -> {'OK' if ok_sem_eco else 'FALHOU'}")

    # --- i2c
    fi = FonteI2CFake(accel_g=(0.0, 0.0, 1.0), gyro_dps=(10.0, -5.0, 2.5))
    bloco = fi.ler_bloco(0x68, FonteI2CFake.ACCEL_XOUT_H, 6)
    az = ((bloco[4] << 8) | bloco[5])
    if az >= 0x8000:
        az -= 65536
    az_g = az / FonteI2CFake.LSB_PER_G
    ok_accel = abs(az_g - 1.0) < 1e-3
    print(f"i2c fake: az reconstruido {az_g:.4f} g (alvo 1.0) -> "
          f"{'OK' if ok_accel else 'FALHOU'}")

    bloco_g = fi.ler_bloco(0x68, FonteI2CFake.GYRO_XOUT_H, 6)
    gy = ((bloco_g[2] << 8) | bloco_g[3])
    if gy >= 0x8000:
        gy -= 65536
    gy_dps = gy / FonteI2CFake.LSB_PER_DPS
    ok_gyro = abs(gy_dps - (-5.0)) < 1e-2
    print(f"i2c fake: gy reconstruido {gy_dps:.4f} deg/s (alvo -5.0, sinal negativo) -> "
          f"{'OK' if ok_gyro else 'FALHOU'}")

    ok_drdy = (FonteI2CFake(drdy=True).ler_byte(0x0C, 0x02) & 0x01) == 1 and \
              (FonteI2CFake(drdy=False).ler_byte(0x0C, 0x02) & 0x01) == 0
    print(f"i2c fake: bit DRDY responde ao flag -> {'OK' if ok_drdy else 'FALHOU'}")

    tudo = all([ok_serial, ok_gpio, ok_sem_eco, ok_accel, ok_gyro, ok_drdy])
    print(f"\n=== RESULTADO: {'OK' if tudo else 'FALHOU'} ===")
    if not tudo:
        raise SystemExit(1)
