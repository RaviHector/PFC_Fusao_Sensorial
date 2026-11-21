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
Driver da IMU MPU-9250 (acelerômetro e giroscópio) com o magnetômetro AK8963.
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
import math
import time

import numpy as np

from kalman.config import ACCEL_GATE_MARGIN, DT_LOOP, GRAVITY
from kalman.config import AMOSTRAS_CALIB_GIRO

# Registradores
PWR_MGMT_1 = 0x6B
INT_PIN_CFG = 0x37
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

MAG_ADDR = 0x0C
MAG_WIA = 0x00
MAG_ST1 = 0x02
MAG_HXL = 0x03
MAG_CNTL1 = 0x0A
MAG_ASAX = 0x10

# Escalas
LSB_PER_G = 16384.0    # +-2 g
LSB_PER_DPS = 131.0    # +-250 deg/s
MAG_UT_LSB = 0.15      # uT por LSB, AK8963 16 bits


def _be16(high, low):
    """Dois bytes big-endian -> int com sinal (complemento de dois)."""
    v = (high << 8) | low
    return v - 65536 if v >= 0x8000 else v


def _le16(low, high):
    """Dois bytes little-endian -> int com sinal. O AK8963 é little-endian."""
    v = (high << 8) | low
    return v - 65536 if v >= 0x8000 else v


class IMU:
    """Driver de dados brutos."""

    def __init__(self, fonte=None, address=0x68, bus_id=4,
                 tilt_formula="atual", do_wake=True, calibrar=True,
                 amostras_calib=AMOSTRAS_CALIB_GIRO, delay_calib=0.01):
        if tilt_formula not in ("atual", "literatura"):
            raise ValueError("tilt_formula deve ser 'atual' ou 'literatura'")

        if fonte is None:
            from bancada.drivers.base import FonteI2C

            fonte = FonteI2C(bus_id=bus_id)

        self.fonte = fonte
        self.address = address
        self.tilt_formula = tilt_formula

        # Viés do giroscópio [deg/s]
        self.bias_gx = self.bias_gy = self.bias_gz = 0.0
        # Offsets do acelerômetro [g]
        self.offset_ax = self.offset_ay = self.offset_az = 0.0
        # Ajuste de fábrica do magnetômetro
        self.mag_adj = (1.0, 1.0, 1.0)
        self.mag_bias_counts = None
        self.mag_scale = None

        # Contadores
        self.n_leituras = 0
        self.n_mag_sem_drdy = 0
        self.n_accel_rejeitado = 0

        # os sleeps dão tempo ao MPU e ao bypass do AK8963
        import time

        if do_wake:
            try:
                self.fonte.escrever_byte(self.address, PWR_MGMT_1, 0x00)
                time.sleep(0.05)
            except Exception as e:
                raise RuntimeError(f"Falha ao acordar a IMU em 0x{address:02x}: {e}")

        # Bypass para falar direto com o AK8963
        try:
            cur = self.fonte.ler_byte(self.address, INT_PIN_CFG)
            self.fonte.escrever_byte(self.address, INT_PIN_CFG, cur | 0x02)
            time.sleep(0.01)
        except Exception:
            try:
                self.fonte.escrever_byte(self.address, INT_PIN_CFG, 0x02)
                time.sleep(0.01)
            except Exception:
                pass

        # Magnetômetro: `mag_ok` é repassado ao filtro
        try:
            self._init_mag()
            self.mag_ok = True
        except Exception:
            self.mag_ok = False

        if calibrar:
            self._calibrar_accel_giro(amostras_calib, delay_calib)

    def _init_mag(self):
        """Sequência de inicialização do AK8963."""
        import time

        wia = self.fonte.ler_byte(MAG_ADDR, MAG_WIA)
        if wia != 0x48:
            raise RuntimeError(f"AK8963 nao encontrado (WIA=0x{wia:02x})")

        self.fonte.escrever_byte(MAG_ADDR, MAG_CNTL1, 0x00)  # power down
        time.sleep(0.01)

        self.fonte.escrever_byte(MAG_ADDR, MAG_CNTL1, 0x0F)  # fuse rom
        time.sleep(0.01)

        asa = [self.fonte.ler_byte(MAG_ADDR, MAG_ASAX + i) for i in range(3)]
        self.mag_adj = tuple(((a - 128) / 256.0) + 1.0 for a in asa)

        self.fonte.escrever_byte(MAG_ADDR, MAG_CNTL1, 0x00)  # power down
        time.sleep(0.01)

        self.fonte.escrever_byte(MAG_ADDR, MAG_CNTL1, 0x16)  # 16 bits, 100 Hz
        time.sleep(0.01)

    def _calibrar_accel_giro(self, amostras, delay):
        """Média de N amostras com o veículo parado."""
        import time

        sax = say = saz = 0.0
        sgx = sgy = sgz = 0.0
        n = 0
        for _ in range(amostras):
            try:
                ax, ay, az = self._ler_accel_g_bruto()
                gx, gy, gz = self._ler_giro_dps_bruto()
            except Exception:
                continue
            sax += ax; say += ay; saz += az
            sgx += gx; sgy += gy; sgz += gz
            n += 1
            if delay:
                time.sleep(delay)

        if n == 0:
            return

        self.offset_ax = sax / n
        self.offset_ay = say / n
        self.offset_az = (saz / n) - 1.0  # em repouso, az deve ler +1 g
        self.bias_gx = sgx / n
        self.bias_gy = sgy / n
        self.bias_gz = sgz / n

    # Leitura de baixo nível: uma transação por sensor
    def _ler_accel_g_bruto(self):
        b = self.fonte.ler_bloco(self.address, ACCEL_XOUT_H, 6)
        return (_be16(b[0], b[1]) / LSB_PER_G,
                _be16(b[2], b[3]) / LSB_PER_G,
                _be16(b[4], b[5]) / LSB_PER_G)

    def _ler_giro_dps_bruto(self):
        b = self.fonte.ler_bloco(self.address, GYRO_XOUT_H, 6)
        return (_be16(b[0], b[1]) / LSB_PER_DPS,
                _be16(b[2], b[3]) / LSB_PER_DPS,
                _be16(b[4], b[5]) / LSB_PER_DPS)

    def _ler_mag_uT(self):
        """Retorna `(mx, my, mz, is_new)` em uT."""
        if not self.mag_ok:
            return None, None, None, False

        b = self.fonte.ler_bloco(MAG_ADDR, MAG_ST1, 8)
        st1 = b[0]
        if not (st1 & 0x01):  # DRDY baixo => nenhuma amostra nova
            self.n_mag_sem_drdy += 1
            return None, None, None, False

        st2 = b[7]
        if st2 & 0x08:  # overflow do sensor => dado invalido
            return None, None, None, False

        mx = _le16(b[1], b[2])
        my = _le16(b[3], b[4])
        mz = _le16(b[5], b[6])

        if self.mag_bias_counts is not None and self.mag_scale is not None:
            mx = (mx - self.mag_bias_counts[0]) * self.mag_scale[0]
            my = (my - self.mag_bias_counts[1]) * self.mag_scale[1]
            mz = (mz - self.mag_bias_counts[2]) * self.mag_scale[2]

        return (mx * MAG_UT_LSB * self.mag_adj[0],
                my * MAG_UT_LSB * self.mag_adj[1],
                mz * MAG_UT_LSB * self.mag_adj[2],
                True)

    # API pública: SI com radianos
    def getAccel(self):
        """Aceleração em m/s²."""
        ax, ay, az = self._ler_accel_g_bruto()
        return ((ax - self.offset_ax) * GRAVITY,
                (ay - self.offset_ay) * GRAVITY,
                (az - self.offset_az) * GRAVITY)

    def getAccel_g(self):
        ax, ay, az = self._ler_accel_g_bruto()
        return (ax - self.offset_ax, ay - self.offset_ay, az - self.offset_az)

    def getGyro_rads(self):
        """Velocidade angular em rad/s, já sem o viés. É a entrada dos filtros inerciais."""
        gx, gy, gz = self._ler_giro_dps_bruto()
        return (math.radians(gx - self.bias_gx),
                math.radians(gy - self.bias_gy),
                math.radians(gz - self.bias_gz))

    def getGyro_deg(self):
        """Só para depuração. Não usar como entrada de filtro."""
        gx, gy, gz = self._ler_giro_dps_bruto()
        return (gx - self.bias_gx, gy - self.bias_gy, gz - self.bias_gz)

    def accel_confiavel(self):
        """
        O acelerômetro só informa rolagem e arfagem quando a única aceleração
        sentida é a gravidade.
        """
        ax, ay, az = self.getAccel()
        norma = math.sqrt(ax * ax + ay * ay + az * az)
        ok = abs(norma - GRAVITY) <= ACCEL_GATE_MARGIN
        if not ok:
            self.n_accel_rejeitado += 1
        return ok, norma

    def _tilt_compensa(self, mx, my, mz, roll, pitch):
        """Projeta o vetor do magnetômetro no plano horizontal."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)

        if self.tilt_formula == "literatura":
            mx_c = mx * cp + my * sr * sp + mz * cr * sp
            my_c = my * cr - mz * sr
        else:  # 'atual'
            mx_c = mx * cp + mz * sp
            my_c = mx * sr * sp + my * cr - mz * sr * cp

        return mx_c, my_c

    def getEuler_rad(self):
        """Retorna `(roll, pitch, yaw, info)` em radianos."""
        self.n_leituras += 1

        ax, ay, az = self.getAccel_g()
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        accel_ok, norma = self.accel_confiavel()

        yaw = None
        mx, my, mz, mag_novo = self._ler_mag_uT()
        if mag_novo:
            mx_c, my_c = self._tilt_compensa(mx, my, mz, roll, pitch)
            yaw = math.atan2(my_c, mx_c)  # em [-pi, pi)

        info = {
            "accel_ok": accel_ok,
            "norma_accel": norma,
            "mag_novo": mag_novo,
            "mag_ok": self.mag_ok,
        }
        return roll, pitch, yaw, info

    def getEuler_deg(self):
        """Só para depuração."""
        r, p, y, info = self.getEuler_rad()
        return (math.degrees(r), math.degrees(p),
                None if y is None else math.degrees(y) % 360.0, info)

    def estatisticas(self):
        return {
            "leituras": self.n_leituras,
            "mag_sem_drdy": self.n_mag_sem_drdy,
            "accel_rejeitado_C9": self.n_accel_rejeitado,
            "mag_ok": self.mag_ok,
            "tilt_formula": self.tilt_formula,
            "bias_giro_dps": (self.bias_gx, self.bias_gy, self.bias_gz),
        }

    @staticmethod
    def _quebrar_wrap(ts, ys, limiar=180.0):
        """Insere NaN onde a série salta o wrap, só para o desenho."""
        if len(ts) < 2:
            return list(ts), list(ys)
        ts_out, ys_out = [ts[0]], [ys[0]]
        for i in range(1, len(ys)):
            a, b = ys[i - 1], ys[i]
            if a == a and b == b and abs(b - a) > limiar:  # a==a descarta NaN
                ts_out.append(ts[i])
                ys_out.append(float("nan"))
            ts_out.append(ts[i])
            ys_out.append(ys[i])
        return ts_out, ys_out

    def plot(self, duration=60):
        """Bancada ao vivo: roll/pitch/yaw brutos contra o filtro de 6 estados."""
        import matplotlib.pyplot as plt

        from kalman.core import KalmanFilter
        from kalman.filtros import imu6
        from kalman.utils import Timer, wrap

        plt.ion()
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        rot = ["roll", "pitch", "yaw"]
        linhas_b, linhas_k = [], []
        for i, nome in enumerate(rot):
            (lb,) = axs[i].plot([], [], "r.", alpha=0.4, label=f"{nome} bruto")
            (lk,) = axs[i].plot([], [], "g-", linewidth=2, label=f"{nome} Kalman")
            axs[i].set_ylabel(f"{nome} [graus]")
            axs[i].grid(True)
            axs[i].legend(fontsize=8)
            linhas_b.append(lb)
            linhas_k.append(lk)
        axs[2].set_xlabel("Tempo [s]")

        r0, p0, y0, _ = self.getEuler_rad()
        kf = KalmanFilter(dim_x=6, angle_indices=[0, 1, 2])
        kf.x, kf.P = imu6.initial(rpy0=(r0, p0, y0 if y0 is not None else 0.0))

        ts, brutos, kfs = [], [[], [], []], [[], [], []]
        t0 = Timer.now()
        timer = Timer()
        n_loop = 0
        janela = 300

        try:
            while Timer.now() - t0 < duration:
                t_iter = Timer.now()
                dt = timer.tick()
                roll, pitch, yaw, info = self.getEuler_rad()
                gx, gy, gz = self.getGyro_rads()

                F, Gamma, Q = imu6.matrices(dt)
                kf.F, kf.Gamma, kf.Q = F, Gamma, Q
                kf.predict(u=[gx, gy, gz])

                # H de 3 linhas com magnetômetro novo, 2 linhas sem
                mag_disponivel = yaw is not None
                H, R = imu6.observation(mag_ok=mag_disponivel)
                if info["accel_ok"]:
                    kf.update_scalar(roll, H[0], R[0, 0], is_angle=True, channel="roll")
                    kf.update_scalar(pitch, H[1], R[1, 1], is_angle=True, channel="pitch")
                if mag_disponivel:
                    kf.update_scalar(yaw, H[2], R[2, 2], is_angle=True, channel="yaw")

                ts.append(Timer.now() - t0)
                for i, v in enumerate([roll, pitch, yaw]):
                    brutos[i].append(float("nan") if v is None else math.degrees(v))
                    kfs[i].append(math.degrees(wrap(kf.x[i, 0])))
                n_loop += 1

                if n_loop % 5 == 0:
                    for i in range(3):
                        tb, yb = ts[-janela:], brutos[i][-janela:]
                        tk, yk = ts[-janela:], kfs[i][-janela:]
                        if i == 2:  # yaw: unico que atravessa o wrap
                            tb, yb = self._quebrar_wrap(tb, yb)
                            tk, yk = self._quebrar_wrap(tk, yk)
                        linhas_b[i].set_data(tb, yb)
                        linhas_k[i].set_data(tk, yk)
                        axs[i].relim()
                        axs[i].autoscale_view()
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
            print(f"[imu] estatisticas: {self.estatisticas()}")
            plt.show()

    def close(self):
        """Fecha o barramento I2C."""
        self.fonte.fechar()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Autoteste --fake
def _autoteste():
    from bancada.drivers.base import FonteI2CFake

    print("=== bancada.drivers.imu — autoteste (--fake) ===\n")

    # 1) leitura em bloco e unidades ----------------------------------------
    fonte = FonteI2CFake(accel_g=(0.0, 0.0, 1.0), gyro_dps=(10.0, -5.0, 2.5),
                         mag_uT=(30.0, 0.0, -20.0))
    imu = IMU(fonte=fonte, calibrar=False)

    gx, gy, gz = imu.getGyro_rads()
    esperado = math.radians(10.0)
    ok_rad = abs(gx - esperado) < 1e-6
    print(f"1) giro em rad/s: gx={gx:.6f} (esperado {esperado:.6f} = 10 deg/s) -> "
          f"{'OK' if ok_rad else 'FALHOU'}")
    ok_nao_deg = abs(gx - 10.0) > 1.0
    print(f"   e NAO em graus (gx != 10.0) -> {'OK' if ok_nao_deg else 'FALHOU'}")

    # 2) roll/pitch com o sensor nivelado --------------------------------
    roll, pitch, yaw, info = imu.getEuler_rad()
    ok_nivel = abs(roll) < 1e-6 and abs(pitch) < 1e-6
    print(f"\n2) nivelado (az=+1g): roll={math.degrees(roll):.3f}, "
          f"pitch={math.degrees(pitch):.3f} graus -> {'OK' if ok_nivel else 'FALHOU'}")

    # 3) DRDY ---------------------------------------------------------------
    ok_com_drdy = yaw is not None and info["mag_novo"]
    fonte.drdy = False
    _, _, yaw2, info2 = imu.getEuler_rad()
    ok_sem_drdy = (yaw2 is None) and (not info2["mag_novo"])
    print(f"\n3) DRDY alto -> yaw disponivel: {'OK' if ok_com_drdy else 'FALHOU'}")
    print(f"   DRDY baixo -> yaw=None, mag_novo=False: "
          f"{'OK' if ok_sem_drdy else 'FALHOU'} (contador={imu.n_mag_sem_drdy})")
    fonte.drdy = True

    # 4) gate do acelerômetro -----------------------------------------------
    ok_g, norma_g = imu.accel_confiavel()
    fonte.accel_g = [0.0, 0.0, 1.5]  # 1.5 g: o veiculo esta acelerando
    ok_ruim, norma_ruim = imu.accel_confiavel()
    ok_c9 = ok_g and (not ok_ruim)
    print(f"\n4) gate do acelerometro, parado: |a|={norma_g:.3f} m/s2 -> aceito={ok_g}")
    print(f"        acelerando: |a|={norma_ruim:.3f} m/s2 -> aceito={ok_ruim}")
    print(f"        gate atua -> {'OK' if ok_c9 else 'FALHOU'}")
    fonte.accel_g = [0.0, 0.0, 1.0]

    # 5) tilt: as duas formulas coincidem quando roll OU pitch e zero ----
    imu_a = IMU(fonte=FonteI2CFake(), calibrar=False, tilt_formula="atual")
    imu_l = IMU(fonte=FonteI2CFake(), calibrar=False, tilt_formula="literatura")
    m = (25.0, 8.0, -18.0)

    casos_iguais = [(0.0, 0.0), (0.0, math.radians(20)), (math.radians(20), 0.0)]
    max_dif_zero = 0.0
    for r, p in casos_iguais:
        xa, ya = imu_a._tilt_compensa(*m, r, p)
        xl, yl = imu_l._tilt_compensa(*m, r, p)
        d = abs(math.degrees(math.atan2(ya, xa) - math.atan2(yl, xl)))
        max_dif_zero = max(max_dif_zero, d)
    ok_iguais = max_dif_zero < 1e-9

    r, p = math.radians(30), math.radians(20)
    xa, ya = imu_a._tilt_compensa(*m, r, p)
    xl, yl = imu_l._tilt_compensa(*m, r, p)
    dif_30_20 = abs(math.degrees(math.atan2(ya, xa) - math.atan2(yl, xl)))
    ok_diferem = dif_30_20 > 1.0

    print(f"\n5) tilt — roll=0 ou pitch=0: diferenca maxima {max_dif_zero:.2e} graus -> "
          f"{'OK (coincidem)' if ok_iguais else 'FALHOU'}")
    print(f"   roll=30, pitch=20: diferenca {dif_30_20:.1f} graus -> "
          f"{'OK (divergem, como esperado)' if ok_diferem else 'FALHOU'}")

    # 6) close() fecha o barramento de verdade ---------------------------
    f6 = FonteI2CFake()
    imu6_ = IMU(fonte=f6, calibrar=False)
    imu6_.close()
    ok_close = f6.fechada
    print(f"\n6) close() fecha o barramento: "
          f"{'OK' if ok_close else 'FALHOU'}")

    # 7) mag_ok=False -> H de 2 linhas --------------------------------------
    from kalman.filtros import imu6 as filtro_imu6

    f7 = FonteI2CFake()
    f7.ler_byte = lambda addr, reg: 0x00 if reg == MAG_WIA else 0x00  # WIA errado
    imu7 = IMU(fonte=f7, calibrar=False)
    H_sem, R_sem = filtro_imu6.observation(mag_ok=imu7.mag_ok)
    ok_b7 = (not imu7.mag_ok) and H_sem.shape[0] == 2
    print(f"\n7) sem magnetometro: mag_ok={imu7.mag_ok}, "
          f"H tem {H_sem.shape[0]} linhas (esperado 2) -> {'OK' if ok_b7 else 'FALHOU'}")

    tudo = all([ok_rad, ok_nao_deg, ok_nivel, ok_com_drdy, ok_sem_drdy,
                ok_c9, ok_iguais, ok_diferem, ok_close, ok_b7])
    print(f"\n=== RESULTADO: {'OK' if tudo else 'FALHOU'} ===")
    return tudo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Driver da IMU MPU-9250")
    ap.add_argument("--fake", action="store_true", help="autoteste sem hardware")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--tilt", choices=["atual", "literatura"], default="atual",
                    help="formula de compensacao de tilt (ver docstring)")
    args = ap.parse_args()

    if args.fake:
        raise SystemExit(0 if _autoteste() else 1)

    imu = IMU(tilt_formula=args.tilt)
    try:
        imu.plot(duration=args.duration)
    finally:
        imu.close()
