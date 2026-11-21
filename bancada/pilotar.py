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
Pilotar o veículo com um controle de PS5 (DualSense) ligado por Bluetooth na
Raspberry, gravando os três sensores e o esterçamento comandado.
"""
import argparse
import csv
import datetime
import os
import time

import numpy as np

if __package__ in (None, ""):
    import sys as _sys

    _d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

from bancada.corrida import COLUNAS as COLUNAS_CORRIDA
from bancada.corrida import ARRANQUE_PADRAO, _avisa_velmax
from kalman.config import STEER_MAX

# Colunas do CSV: as do `corrida.py`, mais o que o controle comandou.
COLUNAS = COLUNAS_CORRIDA + [
    "delta_cmd_rad", "delta_pedido_rad", "acel_frac",
    "modo_steer", "modo_acel", "armado", "marca",
]
# `marca` é um contador, e não um pulso: sobe de 1 a cada toque no Create e
# permanece no valor novo.

N_MEDIA_STEER = 10  # o `class_servos.st_filt` é MovingAverage(n=10)
N_MEDIA_STEER_PILOTO = 3  # o padrão ao pilotar; ver `_ajusta_media_steer`
ZONA_MORTA = 0.08  # fração da alavanca ignorada em torno do centro
TIMEOUT_CONTROLE = 0.5    # [s] sem evento nenhum acelerando -> parada
PERIODO_PADRAO = 0.05     # [s] os 20 Hz do veiculo

AUTO, MANUAL = "auto", "manual"


# Mapeamento da alavanca e do gatilho
def aplica_zona_morta(frac, zona=ZONA_MORTA):
    """Zera a alavanca perto do centro e reescala o resto."""
    f = float(np.clip(frac, -1.0, 1.0))
    if abs(f) <= zona:
        return 0.0
    return float(np.sign(f) * (abs(f) - zona) / (1.0 - zona))


def delta_da_alavanca(frac, steer_max=STEER_MAX, inverter=False):
    """Fração da alavanca (-1 a +1) para esterçamento em radianos."""
    f = aplica_zona_morta(frac)
    d = -f * float(steer_max)
    return float(-d if inverter else d)


def acel_do_gatilho(frac, cheio=False):
    """Gatilho (0 a 1) para fração de acelerador; `cheio` satura em 1."""
    if cheio:
        return 1.0
    return float(np.clip(frac, 0.0, 1.0))


class MediaMovel:
    """
    A mesma média móvel do `class_servos.st_filt`, para saber o que chegou ao
    servo.
    """

    def __init__(self, n=N_MEDIA_STEER, inicial=0.0):
        from collections import deque

        self.buf = deque([float(inicial)] * int(n), maxlen=int(n))

    def filtra(self, x):
        self.buf.append(float(x))
        return float(np.mean(self.buf))


# O estado que o controle entrega
class Comando:
    """O que o operador está pedindo agora. Só dados, nenhuma decisão."""

    def __init__(self):
        self.eixo_steer = 0.0      # -1 a +1
        self.gatilho = 0.0         # 0 a 1
        self.acel_cheio = False    # R1
        self.homem_morto = False   # L1
        self.parada = False  # círculo (evento)
        self.rearmar = False       # X (evento)
        self.marcar = False        # Create (evento): marca um ponto medido
        self.troca_steer = False   # quadrado (evento)
        self.troca_acel = False  # triângulo (evento)
        self.sair = False          # Options (evento)
        self.t_evento = None  # instante do último evento recebido

    def limpa_eventos(self):
        """Consome os eventos de botão, preservando os eixos e os apertos."""
        self.parada = self.rearmar = self.marcar = False
        self.troca_steer = self.troca_acel = self.sair = False


# Leitura do controle
MAPA_BOTOES = {
    "BTN_SELECT": "marcar",
    "BTN_TR": "acel_cheio",
    "BTN_TL": "homem_morto",
    "BTN_EAST": "parada",
    "BTN_SOUTH": "rearmar",
    "BTN_WEST": "troca_steer",
    "BTN_NORTH": "troca_acel",
    "BTN_START": "sair",
}
SEGURAR = ("acel_cheio", "homem_morto")   # valem enquanto pressionados


class ControlePS5:
    """Lê o DualSense por evdev, sem bloquear o laço."""

    def __init__(self, caminho=None):
        from evdev import InputDevice, ecodes, list_devices

        self.ecodes = ecodes
        if caminho is None:
            caminho = self._acha(InputDevice, list_devices)
        self.dev = InputDevice(caminho)
        self.caminho = caminho
        self.dev.grab()  # ninguém mais recebe estes eventos
        self.info_eixos = self._faixas()
        print("[pilotar] controle: %s em %s" % (self.dev.name, caminho))

    @staticmethod
    def _acha(InputDevice, list_devices, ecodes=None):
        """Acha o gamepad, e não os irmãos dele."""
        if ecodes is None:
            from evdev import ecodes

        candidatos, vistos = [], []
        for c in list_devices():
            try:
                d = InputDevice(c)
            except Exception:
                continue
            nome = (d.name or "").lower()
            if "dualsense" in nome or "wireless controller" in nome:
                caps = d.capabilities()
                eixos = set(cod for cod, _ in caps.get(ecodes.EV_ABS, []))
                teclas = set(caps.get(ecodes.EV_KEY, []))
                precisa_eixos = {ecodes.ABS_X, ecodes.ABS_RZ}
                precisa_teclas = set(
                    getattr(ecodes, n) for n in MAPA_BOTOES
                    if getattr(ecodes, n, None) is not None)
                completo = (precisa_eixos <= eixos
                            and bool(precisa_teclas) and precisa_teclas <= teclas)
                vistos.append((c, d.name, completo))
                if completo:
                    candidatos.append((c, d.name))
            d.close()

        if not candidatos:
            if vistos:
                lista = "\n".join("    %s  %s" % (c, n) for c, n, _ in vistos)
                raise SystemExit(
                    "[pilotar] achei o controle, mas nenhum dos dispositivos\n"
                    "  tem a assinatura do gamepad (ABS_X, ABS_RZ e os botoes).\n"
                    "  Encontrados:\n%s\n"
                    "  Rode --listar e passe o certo em --dispositivo." % lista)
            raise SystemExit(
                "[pilotar] nenhum controle encontrado. Pareie primeiro "
                "(bluetoothctl) e confira com --listar.")
        return candidatos[0][0]

    def _faixas(self):
        """Mínimo e máximo de cada eixo, para normalizar sem chutar escala."""
        faixas = {}
        for cod, info in self.dev.capabilities().get(self.ecodes.EV_ABS, []):
            faixas[cod] = (info.min, info.max)
        return faixas

    def _normaliza(self, cod, valor):
        lo, hi = self.info_eixos.get(cod, (0, 255))
        if hi == lo:
            return 0.0
        return (float(valor) - lo) / (hi - lo)     # 0 a 1

    def atualiza(self, cmd):
        """Consome os eventos pendentes e atualiza `cmd`."""
        import select

        from evdev import ecodes

        try:
            prontos, _, _ = select.select([self.dev.fd], [], [], 0)
            if not prontos:
                return True
            for ev in self.dev.read():
                cmd.t_evento = time.monotonic()
                if ev.type == ecodes.EV_ABS:
                    if ev.code == ecodes.ABS_X:
                        cmd.eixo_steer = 2.0 * self._normaliza(ev.code, ev.value) - 1.0
                    elif ev.code == ecodes.ABS_RZ:
                        cmd.gatilho = self._normaliza(ev.code, ev.value)
                elif ev.type == ecodes.EV_KEY:
                    nome = ecodes.KEY.get(ev.code) or ecodes.BTN.get(ev.code)
                    nomes = [nome] if isinstance(nome, str) else list(nome or [])
                    for n in nomes:
                        campo = MAPA_BOTOES.get(n)
                        if campo is None:
                            continue
                        if campo in SEGURAR:
                            setattr(cmd, campo, ev.value != 0)
                        elif ev.value == 1:  # só na descida da tecla
                            setattr(cmd, campo, True)
        except BlockingIOError:
            pass                                   # nada novo, o caso comum
        except OSError:
            return False                           # controle sumiu
        return True

    def esta_vivo(self):
        """O controle ainda responde?"""
        if not os.path.exists(self.caminho):
            return False
        try:
            self.dev.active_keys()
            return True
        except (OSError, IOError):
            return False

    def fechar(self):
        try:
            self.dev.ungrab()
        except Exception:
            pass
        try:
            self.dev.close()
        except Exception:
            pass

    @staticmethod
    def listar():
        """Imprime os dispositivos de entrada e os eixos de cada um."""
        from evdev import InputDevice, ecodes, list_devices

        achou = False
        for c in sorted(list_devices()):
            try:
                d = InputDevice(c)
            except Exception as e:
                print("  %s  (sem acesso: %s)" % (c, e))
                continue
            achou = True
            print("  %s  %s" % (c, d.name))
            caps = d.capabilities(verbose=False)
            eixos = [ecodes.ABS.get(cod, cod)
                     for cod, _ in caps.get(ecodes.EV_ABS, [])]
            botoes = []
            for cod in caps.get(ecodes.EV_KEY, []):
                n = ecodes.BTN.get(cod) or ecodes.KEY.get(cod) or cod
                botoes.append(n if isinstance(n, str) else str(n))
            print("     eixos : %s" % ", ".join(str(e) for e in eixos))
            print("     botoes: %s" % ", ".join(botoes[:20]))
            d.close()
        if not achou:
            print("  nenhum dispositivo legivel. Falta estar no grupo `input`?")


class ControleFalso:
    """Controle de mentira, para o autoteste: uma lista de (instante, ação)."""

    def __init__(self, roteiro):
        self.roteiro = sorted(roteiro, key=lambda p: p[0])
        self.t0 = None

    def atualiza(self, cmd):
        agora = time.monotonic()
        if self.t0 is None:
            self.t0 = agora
        while self.roteiro and self.roteiro[0][0] <= agora - self.t0:
            _, acao = self.roteiro.pop(0)
            for k, v in acao.items():
                setattr(cmd, k, v)
            cmd.t_evento = agora
        return True

    def esta_vivo(self):
        return True

    def fechar(self):
        pass


# A máquina de estados: decide o que vai para os servos
class Piloto:
    """Junta o comando do controle com os dois modos e devolve o que atuar."""

    def __init__(self, steer_auto=0.0, arranque=ARRANQUE_PADRAO, corte=None,
                 modo_steer=MANUAL, modo_acel=MANUAL, homem_morto=False,
                 inverter_steer=False, timeout=TIMEOUT_CONTROLE,
                 n_media=N_MEDIA_STEER):
        self.steer_auto = float(steer_auto)
        self.arranque = float(arranque)
        self.corte = corte
        self.modo_steer = modo_steer
        self.modo_acel = modo_acel
        self.exige_homem_morto = bool(homem_morto)
        self.inverter_steer = bool(inverter_steer)
        self.timeout = float(timeout)

        self.armado = True
        self.motivo_parada = ""
        self.t_auto_acel = 0.0     # quando o acelerador entrou em automatico
        self.n_media = int(n_media)
        self.media = MediaMovel(self.n_media)

    # --- modos
    def _alterna(self, cmd, t):
        if cmd.troca_steer:
            self.modo_steer = AUTO if self.modo_steer == MANUAL else MANUAL
            print("[pilotar] direcao -> %s" % self.modo_steer.upper())
        if cmd.troca_acel:
            self.modo_acel = AUTO if self.modo_acel == MANUAL else MANUAL
            self.t_auto_acel = t
            print("[pilotar] acelerador -> %s" % self.modo_acel.upper())

    # --- perfil automatico do acelerador
    def _acel_auto(self, t):
        """
        Mesmo perfil do `corrida.py`, contado do instante em que este eixo
        entrou em automatico: parado, acelerador cheio, corte.
        """
        dt_auto = t - self.t_auto_acel
        if dt_auto < self.arranque:
            return 0.0
        if self.corte is not None and dt_auto >= self.corte:
            return 0.0
        return 1.0

    # --- decisão
    def passo(self, cmd, t, controle_vivo=True):
        """
        Devolve (delta_cmd, delta_pedido, acel, modo_steer, modo_acel, armado).
        """
        self._alterna(cmd, t)

        # --- as razões para parar
        if cmd.parada:
            self.armado, self.motivo_parada = False, "botao de parada"
        if not controle_vivo:
            self.armado, self.motivo_parada = False, "controle desconectado"

        # Não há parada por "faz N segundos que não chega evento": segurar o R1
        # não gera eventos.

        if cmd.rearmar and not self.armado:
            self.armado, self.motivo_parada = True, ""
            print("[pilotar] rearmado")

        # --- direção
        if self.modo_steer == AUTO:
            pedido = self.steer_auto
        else:
            pedido = delta_da_alavanca(cmd.eixo_steer,
                                       inverter=self.inverter_steer)

        # --- acelerador
        if not self.armado:
            acel = 0.0
        elif self.modo_acel == AUTO:
            acel = self._acel_auto(t)
        else:
            acel = acel_do_gatilho(cmd.gatilho, cmd.acel_cheio)
            if self.exige_homem_morto and not cmd.homem_morto:
                acel = 0.0

        if not self.armado:
            pedido = 0.0 if self.modo_steer == MANUAL else pedido

        pedido = float(np.clip(pedido, -STEER_MAX, STEER_MAX))
        efetivo = self.media.filtra(pedido)
        return (efetivo, pedido, float(acel), self.modo_steer, self.modo_acel,
                self.armado)


# Atuação
def aplica(servos, delta, acel, por_taxa=False):
    """Manda direção e acelerador para os servos."""
    servos.set_steer(float(delta))
    if por_taxa:
        servos.set_torque(float(acel))
        return
    with servos.lock:
        servos.dth_pwm = 0.0
        servos.th_pwm = float(np.clip(acel, 0.0, 1.0)) * servos.max_throttle


def parar_tudo(servos):
    """Corta a tração e centra a direção. Seguro de chamar várias vezes."""
    try:
        servos.set_torque(0.0)
        with servos.lock:
            servos.dth_pwm = 0.0
            servos.th_pwm = 0.0
    except Exception:
        pass
    try:
        servos.set_steer(0.0)
    except Exception:
        pass


class AvisoSonoro:
    """Decide quando avisar o operador pelo buzzer."""

    T_LARGADA = 2.0  # [s] parado na largada
    T_CAUDA = 5.0  # [s] parado depois da parada
    V_ANDANDO = 0.10    # [m/s] acima disto o carro saiu do lugar
    V_PARADO = 0.05  # [m/s] abaixo disto está parado (mesmo limiar da análise)

    def __init__(self, t_largada=None, t_cauda=None):
        self.t_largada = self.T_LARGADA if t_largada is None else float(t_largada)
        self.t_cauda = self.T_CAUDA if t_cauda is None else float(t_cauda)
        self.fase = "largada"
        self.t_parou = None

    def passo(self, t, v):
        """`t` em segundos desde o início; `v` do encoder, ou None."""
        vv = 0.0 if v is None else abs(float(v))

        if self.fase == "largada":
            if t >= self.t_largada:
                self.fase = "espera_movimento"
                return "pode_andar"
            return None

        if self.fase == "espera_movimento":
            if vv > self.V_ANDANDO:
                self.fase = "andando"
            return None

        if self.fase == "andando":
            if vv <= self.V_PARADO:
                if self.t_parou is None:
                    self.t_parou = t
                elif (t - self.t_parou) >= self.t_cauda:
                    self.fase = "fim"
                    return "pode_encerrar"
            else:
                # voltou a andar: a contagem da cauda recomeça do zero
                self.t_parou = None
            return None

        return None


def _abre_buzzer(ligado=True):
    """O `Buzzer` da raiz, ou None se não der."""
    if not ligado:
        return None
    try:
        from class_buzzer import Buzzer
        return Buzzer()
    except Exception as e:
        print("[pilotar] sem buzzer (%s); segue sem aviso sonoro" % e)
        return None


def _beep(bz, padrao):
    """Toca sem deixar erro escapar para o laco."""
    if bz is None:
        return
    try:
        bz.beep(padrao)
    except Exception:
        pass


def _testar_buzzer():
    """Toca os dois avisos e sai."""
    bz = _abre_buzzer(True)
    if bz is None:
        print("[pilotar] o buzzer NAO abriu. Os ensaios seguem normalmente, "
              "mas conte os 2 s e os 5 s de cabeca.")
        return 1
    try:
        print("[pilotar] UM beep = os 2 s de largada acabaram, pode andar")
        _beep(bz, 0.15)
        time.sleep(1.5)
        print("[pilotar] DOIS beeps = 5 s parado, pode apertar Options")
        _beep(bz, [0.15, 0.15])
        time.sleep(1.5)
        print("[pilotar] se voce ouviu os dois, esta pronto.")
        return 0
    finally:
        try:
            bz.close()
        except Exception:
            pass


def _aplica_trim_steer(servos, graus):
    """
    Ajuste fino do zero da direção, em graus, nesta instância de `Servos`.
    """
    if abs(float(graus)) < 1e-12:
        return 0.0
    g = float(np.clip(graus, -10.0, 10.0))
    if abs(g - float(graus)) > 1e-9:
        print("[pilotar] AVISO: trim pedido de %.2f graus, mas o "
              "`class_servos.set_trim` satura em +-10. Usando %.2f."
              % (float(graus), g))
    servos.set_trim(steer=np.deg2rad(g))
    print("[pilotar] trim da direcao: %+.2f graus" % g)
    return g


def _ajusta_media_steer(servos, n):
    """Troca a janela da média móvel da direção, só nesta instância."""
    if int(n) == N_MEDIA_STEER:
        return  # nada a fazer: é o padrão do class_servos
    import class_filter

    anterior = getattr(servos, "st_filt", None)
    if anterior is None:
        raise SystemExit(
            "[pilotar] --media-steer pedido, mas este `Servos` nao tem "
            "`st_filt`. O class_servos.py mudou; conferir antes de rodar.")
    servos.st_filt = class_filter.MovingAverage(n=int(n), initial=0.0)
    print("[pilotar] media movel da direcao: %d amostras (%.2f s para o "
          "angulo cheio), no lugar das %d do class_servos"
          % (int(n), int(n) * PERIODO_PADRAO, N_MEDIA_STEER))


# O laço
def pilotar(args):
    from bancada.corrida import _abrir_servos
    from bancada.drivers.encoder import Encoder
    from bancada.drivers.imu import IMU
    from bancada.drivers.ultrassom import Ultrassom
    from kalman.utils import Timer

    controle = ControlePS5(args.dispositivo)
    piloto = Piloto(steer_auto=np.deg2rad(args.steer), arranque=args.arranque,
                    corte=args.corte, modo_steer=args.modo_steer,
                    modo_acel=args.modo_acel, homem_morto=args.homem_morto,
                    inverter_steer=args.inverter_steer,
                    timeout=args.timeout_controle,
                    n_media=args.media_steer)

    bz = _abre_buzzer(not args.sem_buzzer)
    avisos = AvisoSonoro()

    enc = imu = us = servos = None
    try:
        enc = Encoder()
        imu = IMU()
        us = Ultrassom(temp_c=args.temp, usar_thread=True)
        servos = _abrir_servos(args.velmax, 0.0)
        _ajusta_media_steer(servos, int(args.media_steer))
        trim_usado = _aplica_trim_steer(servos, args.trim_steer)
    except Exception:
        for d in (enc, imu, us):
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
        if servos is not None:
            try:
                servos.close()
            except Exception:
                pass
        controle.fechar()
        raise

    carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join("dados", "pilotado_%s.csv" % carimbo)
    os.makedirs("dados", exist_ok=True)
    f = open(caminho, "w", newline="", encoding="utf-8")
    f.write("# coleta: pilotado (encoder + imu + ultrassom, laco unico)\n")
    f.write("# gravado_em: %s\n"
            % datetime.datetime.now().isoformat(timespec="seconds"))
    f.write("# controle: %s\n" % controle.dev.name)
    f.write("# duracao_pedida_s: %s\n" % args.tempo)
    f.write("# dt_alvo_s: %s\n" % args.dt)
    f.write("# velmax_mps: %s\n" % args.velmax)
    f.write("# steer_auto_deg: %s\n" % args.steer)
    f.write("# modo_steer_inicial: %s\n" % args.modo_steer)
    f.write("# modo_acel_inicial: %s\n" % args.modo_acel)
    f.write("# arranque_s: %s\n" % args.arranque)
    f.write("# corte_s: %s\n" % args.corte)
    f.write("# temp_ar_c: %s\n" % args.temp)
    f.write("# condicao: %s\n" % (args.condicao or ""))
    f.write("# media_steer_n: %d   (janela da media movel da direcao; "
            "%d = padrao do class_servos)\n"
            % (int(args.media_steer), N_MEDIA_STEER))
    f.write("# trim_steer_deg: %+.2f   (ajuste do zero da direcao; o "
            "delta_cmd_rad NAO o inclui)\n" % trim_usado)
    f.write("# acelerador: %s\n" % ("taxa (set_torque)" if args.acel_taxa
                                    else "posicao (th_pwm)"))
    f.write("# delta_cmd_rad: JA filtrado pela media movel do class_servos;"
            " e a entrada do filtro\n")
    f.write("# sincronizado: sim -- os tres sensores na mesma linha, mesmo t_s\n")
    w = csv.writer(f)
    w.writerow(COLUNAS)

    print("[pilotar] gravando em %s" % caminho)
    print("[pilotar] circulo = PARADA, X = rearmar, quadrado = modo da "
          "direcao, triangulo = modo do acelerador, Options = encerrar")
    print("[pilotar] Create = MARCAR PONTO (para casar com marco de trena)")
    marca = 0
    print("[pilotar] direcao: %s   acelerador: %s"
          % (args.modo_steer.upper(), args.modo_acel.upper()))

    cmd = Comando()
    t0 = t_ant = Timer.now()
    t_us_ant = None
    n = n_e = n_i = n_u = n_novo = 0
    paradas = 0
    try:
        while True:
            t_it = Timer.now()
            if t_it - t0 >= args.tempo:
                break
            dt = t_it - t_ant
            t_ant = t_it

            vivo = controle.atualiza(cmd) and controle.esta_vivo()
            armado_antes = piloto.armado
            delta, pedido, acel, m_s, m_a, armado = piloto.passo(
                cmd, t_it, controle_vivo=vivo)
            if armado_antes and not armado:
                paradas += 1
                print("\n[pilotar] PARADA: %s" % piloto.motivo_parada)
                parar_tudo(servos)
            if cmd.marcar:
                marca += 1
                print("[pilotar] MARCA %d em t = %.2f s" % (marca, t_it - t0))
            if cmd.sair:
                print("[pilotar] encerrando a pedido do operador")
                break
            cmd.limpa_eventos()

            if armado:
                aplica(servos, delta, acel, por_taxa=args.acel_taxa)
            else:
                parar_tudo(servos)

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
                return "" if v is None else "%.6f" % v

            w.writerow(["%.6f" % (t_it - t0), "%.6f" % dt,
                        int(bool(ok_e)), s(vel),
                        int(bool(ok_i)), s(ax), s(ay), s(az),
                        s(roll), s(pitch), s(yaw), s(gx), s(gy), s(gz),
                        int(bool(ok_u)), int(bool(novo)), s(dist), motivo,
                        "%.6f" % delta, "%.6f" % pedido, "%.4f" % acel,
                        m_s, m_a, int(bool(armado)), marca])
            aviso = avisos.passo(t_it - t0, vel if ok_e else None)
            if aviso == "pode_andar":
                _beep(bz, 0.15)
                print("[pilotar] 2 s de largada cumpridos — pode andar")
            elif aviso == "pode_encerrar":
                _beep(bz, [0.15, 0.15])
                print("[pilotar] 5 s parado cumpridos — pode apertar Options")

            n += 1
            n_e += int(bool(ok_e))
            n_i += int(bool(ok_i))
            n_u += int(bool(ok_u))
            n_novo += int(bool(novo))

            resto = args.dt - (Timer.now() - t_it)
            if resto > 0:
                time.sleep(resto)
    except KeyboardInterrupt:
        print("\n[pilotar] interrompido pelo operador")
    finally:
        parar_tudo(servos)
        f.close()
        controle.fechar()
        for d in (enc, imu, us, servos):
            try:
                d.close()
            except Exception:
                pass
        if bz is not None:
            try:
                bz.close()
            except Exception:
                pass

    if n:
        print("[pilotar] %d amostras em %.1f s" % (n, Timer.now() - t0))
        print("[pilotar]   encoder   validas: %5.1f%%" % (100 * n_e / n))
        print("[pilotar]   imu       validas: %5.1f%%" % (100 * n_i / n))
        print("[pilotar]   ultrassom validas: %5.1f%%  (novas: %5.1f%%)"
              % (100 * n_u / n, 100 * n_novo / n))
        print("[pilotar]   paradas de emergencia: %d" % paradas)
    return caminho


# Autoteste, sem hardware
def autoteste():
    print("=" * 70)
    print("AUTOTESTE  bancada/pilotar.py  (sem hardware)")
    print("=" * 70)
    ok = True

    def confere(rotulo, cond):
        nonlocal ok
        print("  %-52s %s" % (rotulo, "OK" if cond else "FALHOU"))
        ok = ok and bool(cond)

    # --- zona morta e reescala
    confere("alavanca no centro da zona morta -> 0",
            aplica_zona_morta(0.05) == 0.0)
    confere("alavanca no fim -> 1", abs(aplica_zona_morta(1.0) - 1.0) < 1e-12)
    confere("reescala e continua na borda",
            abs(aplica_zona_morta(ZONA_MORTA + 1e-9)) < 1e-6)
    confere("simetrica", aplica_zona_morta(-0.5) == -aplica_zona_morta(0.5))

    # --- mapeamento do esterçamento
    d_esq = delta_da_alavanca(-1.0)
    confere("alavanca para a esquerda da delta POSITIVO (ISO 8855)",
            d_esq > 0)
    confere("no limite, delta = STEER_MAX", abs(d_esq - STEER_MAX) < 1e-9)
    confere("--inverter-steer troca o sinal",
            delta_da_alavanca(-1.0, inverter=True) < 0)
    confere("centro da alavanca da delta = 0", delta_da_alavanca(0.0) == 0.0)

    # --- gatilho
    confere("gatilho a meio curso", abs(acel_do_gatilho(0.5) - 0.5) < 1e-12)
    confere("R1 satura em 100%", acel_do_gatilho(0.2, cheio=True) == 1.0)

    # --- média móvel: o servo não recebe o pedido da alavanca
    p = Piloto(modo_steer=MANUAL)
    cmd = Comando()
    cmd.eixo_steer = -1.0
    cmd.t_evento = time.monotonic()
    efetivo, pedido, _, _, _, _ = p.passo(cmd, 0.0)
    confere("1a amostra: o servo recebe 1/10 do pedido",
            abs(efetivo - pedido / N_MEDIA_STEER) < 1e-9)
    for k in range(N_MEDIA_STEER):
        efetivo, pedido, _, _, _, _ = p.passo(cmd, 0.05 * (k + 1))
    confere("depois de 10 amostras o servo recebe o pedido inteiro",
            abs(efetivo - pedido) < 1e-9)

    # --- janela da média móvel ajustável (--media-steer)
    p = Piloto(modo_steer=MANUAL, n_media=4)
    cmd = Comando()
    cmd.eixo_steer = -1.0
    cmd.t_evento = time.monotonic()
    efetivo, pedido, _, _, _, _ = p.passo(cmd, 0.0)
    confere("--media-steer 4: 1a amostra da 1/4 do pedido",
            abs(efetivo - pedido / 4.0) < 1e-9)
    for k in range(4):
        efetivo, pedido, _, _, _, _ = p.passo(cmd, 0.05 * (k + 1))
    confere("--media-steer 4: pedido inteiro em 4 amostras",
            abs(efetivo - pedido) < 1e-9)
    p10 = Piloto(modo_steer=MANUAL)
    c10 = Comando()
    c10.eixo_steer, c10.t_evento = -1.0, time.monotonic()
    for k in range(4):
        e10 = p10.passo(c10, 0.05 * k)[0]
    confere("--media-steer 4 chega antes da janela de 10", efetivo > e10)
    confere("a replica do class_servos continua em 10", N_MEDIA_STEER == 10)
    confere("o padrao ao pilotar e 3",
            N_MEDIA_STEER_PILOTO == 3)
    confere("a rampa de 20 graus com janela 3 nao satura o servo",
            np.rad2deg(np.deg2rad(20.0)
                       / (N_MEDIA_STEER_PILOTO * PERIODO_PADRAO)) < 600.0)

    # --- os dois modos são independentes
    p = Piloto(steer_auto=np.deg2rad(15.0), modo_steer=MANUAL, modo_acel=AUTO,
               arranque=1.0, corte=5.0)
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    cmd.troca_steer = True
    _, _, _, m_s, m_a, _ = p.passo(cmd, 0.0)
    cmd.limpa_eventos()
    confere("quadrado alterna SO a direcao", m_s == AUTO and m_a == AUTO)
    cmd.troca_acel = True
    _, _, _, m_s, m_a, _ = p.passo(cmd, 0.1)
    cmd.limpa_eventos()
    confere("triangulo alterna SO o acelerador",
            m_s == AUTO and m_a == MANUAL)

    # --- perfil automatico do acelerador
    p = Piloto(modo_acel=AUTO, arranque=1.0, corte=5.0)
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    a_parado = p.passo(cmd, 0.5)[2]
    a_andando = p.passo(cmd, 2.0)[2]
    a_cortado = p.passo(cmd, 6.0)[2]
    confere("automatico: parado no arranque, cheio depois, zero no corte",
            a_parado == 0.0 and a_andando == 1.0 and a_cortado == 0.0)

    # --- parada de emergência e rearme
    p = Piloto(modo_steer=MANUAL, modo_acel=MANUAL)
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    cmd.gatilho = 1.0
    confere("acelera antes da parada", p.passo(cmd, 0.0)[2] == 1.0)
    cmd.parada = True
    _, _, acel, _, _, armado = p.passo(cmd, 0.1)
    cmd.limpa_eventos()
    confere("circulo corta a tracao e desarma", acel == 0.0 and not armado)
    confere("segue desarmado enquanto nao rearmar",
            p.passo(cmd, 0.2)[2] == 0.0 and not p.armado)
    cmd.rearmar = True
    _, _, acel, _, _, armado = p.passo(cmd, 0.3)
    cmd.limpa_eventos()
    confere("X rearma e volta a acelerar", armado and acel == 1.0)

    # --- controle desconectado
    p = Piloto()
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    cmd.gatilho = 1.0
    p.passo(cmd, 0.0)
    _, _, acel, _, _, armado = p.passo(cmd, 0.1, controle_vivo=False)
    confere("controle desconectado corta a tracao",
            acel == 0.0 and not armado and "desconect" in p.motivo_parada)

    # --- o silêncio do controle não é defeito
    # segurar o R1 não gera evento; acelerar em reta não pode cortar a tração
    p = Piloto(timeout=0.5)
    cmd = Comando()
    cmd.t_evento = 10.0
    cmd.gatilho = 1.0
    confere("logo apos o ultimo evento, acelera", p.passo(cmd, 10.2)[2] == 1.0)
    confere("MUITO tempo sem evento, segurando o gatilho, CONTINUA acelerando",
            p.passo(cmd, 60.0)[2] == 1.0 and p.armado)

    cmd2 = Comando()
    cmd2.t_evento = 10.0
    cmd2.acel_cheio = True
    p2 = Piloto(timeout=0.5)
    confere("idem com o R1 segurado, que e o caso que quebrava",
            p2.passo(cmd2, 10.2)[2] == 1.0
            and p2.passo(cmd2, 300.0)[2] == 1.0 and p2.armado)

    # o que deve cortar continua cortando: o controle sumir de verdade
    p3 = Piloto(timeout=0.5)
    cmd3 = Comando()
    cmd3.acel_cheio = True
    cmd3.t_evento = 10.0
    confere("mas o controle SUMIR corta, mesmo com evento recente",
            p3.passo(cmd3, 10.1, controle_vivo=False)[2] == 0.0
            and not p3.armado and "desconect" in p3.motivo_parada)

    # --- homem-morto
    p = Piloto(homem_morto=True)
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    cmd.gatilho = 1.0
    confere("sem L1 nao anda", p.passo(cmd, 0.0)[2] == 0.0)
    cmd.homem_morto = True
    confere("com L1 anda", p.passo(cmd, 0.1)[2] == 1.0)

    # --- o automático não depende do gatilho
    p = Piloto(modo_acel=AUTO, arranque=0.0, corte=None)
    cmd = Comando()
    cmd.t_evento = time.monotonic()
    confere("em automatico, acelera sem gatilho", p.passo(cmd, 1.0)[2] == 1.0)

    # --- controle de mentira dirigindo a máquina de estados
    p = Piloto(modo_steer=MANUAL, modo_acel=MANUAL)
    falso = ControleFalso([(0.0, {"gatilho": 0.4}),
                           (0.0, {"eixo_steer": -0.5})])
    cmd = Comando()
    falso.atualiza(cmd)
    _, pedido, acel, _, _, _ = p.passo(cmd, 0.0)
    confere("controle falso: gatilho e alavanca chegam",
            abs(acel - 0.4) < 1e-9 and pedido > 0)

    # --- o botão de marcar ponto (Create)
    confere("Create esta mapeado para marcar",
            MAPA_BOTOES.get("BTN_SELECT") == "marcar")
    confere("marcar NAO e um aperto continuo, e evento",
            "marcar" not in SEGURAR)
    cm = Comando()
    cm.marcar = True
    cm.limpa_eventos()
    confere("limpa_eventos consome a marca (senao contaria todo ciclo)",
            cm.marcar is False)
    confere("a coluna `marca` esta no CSV", "marca" in COLUNAS)

    # --- trim da direção
    class _ServoTrim:
        def __init__(self):
            self.chamadas = []

        def set_trim(self, steer=0.0, throttle=0.0, pan=0.0):
            self.chamadas.append((steer, throttle, pan))

    st = _ServoTrim()
    confere("trim zero NAO chama set_trim (nao mexe no que nao precisa)",
            _aplica_trim_steer(st, 0.0) == 0.0 and not st.chamadas)
    g = _aplica_trim_steer(st, 1.5)
    confere("trim de 1,5 graus chega em RADIANOS ao servo",
            abs(g - 1.5) < 1e-9 and len(st.chamadas) == 1
            and abs(st.chamadas[0][0] - np.deg2rad(1.5)) < 1e-12)
    confere("trim satura em +-10, como o class_servos",
            abs(_aplica_trim_steer(st, 25.0) - 10.0) < 1e-9
            and abs(_aplica_trim_steer(st, -25.0) + 10.0) < 1e-9)

    # --- aviso sonoro: a máquina de estados, sem GPIO
    av = AvisoSonoro()
    confere("antes dos 2 s nao avisa nada",
            av.passo(0.0, 0.0) is None and av.passo(1.9, 0.0) is None)
    confere("aos 2 s avisa 'pode_andar'", av.passo(2.0, 0.0) == "pode_andar")
    confere("e NAO repete o aviso de largada", av.passo(2.5, 0.0) is None)
    confere("parado logo apos a largada NAO dispara a cauda",
            all(av.passo(2.5 + 0.5 * k, 0.0) is None for k in range(1, 20)))

    confere("depois de andar e parar, avisa aos 5 s",
            av.passo(13.0, 0.8) is None            # andou
            and av.passo(14.0, 0.0) is None        # parou, conta do 14
            and av.passo(18.9, 0.0) is None
            and av.passo(19.0, 0.0) == "pode_encerrar")
    confere("e nao repete o aviso da cauda", av.passo(25.0, 0.0) is None)

    # uma parada no meio da corrida não vale; a contagem recomeça quando o
    # carro volta a andar.
    av2 = AvisoSonoro()
    av2.passo(2.0, 0.0)
    av2.passo(3.0, 0.8)
    av2.passo(4.0, 0.0)                            # para
    av2.passo(6.0, 0.8)                            # volta a andar aos 6
    confere("parada no meio nao conta: a cauda recomeca do zero",
            av2.passo(9.5, 0.0) is None            # 3,5 s parado
            and av2.passo(14.6, 0.0) == "pode_encerrar")

    confere("encoder invalido (None) conta como parado",
            AvisoSonoro().passo(2.0, None) == "pode_andar")

    av3 = AvisoSonoro(t_largada=0.5, t_cauda=1.0)
    confere("os dois tempos sao parametrizaveis",
            av3.passo(0.5, 0.0) == "pode_andar"
            and av3.passo(1.0, 0.8) is None
            and av3.passo(1.5, 0.0) is None
            and av3.passo(2.5, 0.0) == "pode_encerrar")

    confere("o buzzer desligado devolve None e nao levanta",
            _abre_buzzer(False) is None)
    _beep(None, 0.1)  # não pode levantar
    confere("_beep com buzzer None e inofensivo", True)

    class _BzRuim:
        def beep(self, *a, **k):
            raise RuntimeError("sem GPIO")
    _beep(_BzRuim(), 0.1)
    confere("_beep engole erro do buzzer, para nao derrubar a corrida", True)

    # --- colunas do CSV
    confere("CSV comeca com as colunas do corrida.py",
            COLUNAS[:len(COLUNAS_CORRIDA)] == COLUNAS_CORRIDA)
    confere("CSV tem delta_cmd_rad e delta_pedido_rad",
            "delta_cmd_rad" in COLUNAS and "delta_pedido_rad" in COLUNAS)

    print("")
    print("=== RESULTADO: %s ===" % ("OK" if ok else "FALHOU"))
    if ok:
        print("")
        print("Na Raspberry, antes do primeiro ensaio:")
        print("  1. parear o controle:  bluetoothctl  (scan on, pair, trust,")
        print("     connect) com o DualSense em modo de pareamento (PS+Create)")
        print("  2. python3 -m bancada.pilotar --listar   e conferir ABS_X,")
        print("     ABS_RZ e os botoes BTN_*")
        print("  3. conferir a convencao de sinais: a")
        print("     alavanca para a esquerda tem de virar a roda para a")
        print("     esquerda. Se nao, use --inverter-steer")
        print("  4. primeiro ensaio com o carro no cavalete, roda no ar")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tempo", type=float, default=40.0,
                    help="duracao da gravacao [s]")
    ap.add_argument("--dt", type=float, default=PERIODO_PADRAO,
                    help="periodo do laco [s] (padrao 0.05, os 20 Hz)")
    ap.add_argument("--velmax", type=float, default=1.0,
                    help="teto de velocidade do class_servos [m/s]")
    ap.add_argument("--steer", type=float, default=0.0,
                    help="estercamento fixo [graus] no modo AUTOMATICO")
    ap.add_argument("--modo-steer", choices=(MANUAL, AUTO), default=MANUAL,
                    dest="modo_steer", help="modo inicial da direcao")
    ap.add_argument("--modo-acel", choices=(MANUAL, AUTO), default=MANUAL,
                    dest="modo_acel", help="modo inicial do acelerador")
    ap.add_argument("--arranque", type=float, default=ARRANQUE_PADRAO,
                    help="trecho parado do perfil automatico [s]")
    ap.add_argument("--corte", type=float, default=None,
                    help="instante do corte no perfil automatico [s]")
    ap.add_argument("--temp", type=float, default=24.0,
                    help="temperatura do ar [C], para o ultrassom")
    ap.add_argument("--condicao", default=None, help="rotulo do ensaio")
    ap.add_argument("--dispositivo", default=None,
                    help="/dev/input/eventN do controle (padrao: procura)")
    ap.add_argument("--media-steer", type=int, default=N_MEDIA_STEER_PILOTO,
                    dest="media_steer",
                    help="janela da media movel da DIRECAO, em amostras. "
                         "Padrao %d (%.2f s para o angulo cheio, a 20 Hz), "
                         "escolhido no cavalete; o class_servos usa "
                         "%d (%.2f s), que ficou lento para pilotar. Ver a "
                         "conta em _ajusta_media_steer antes de mudar"
                         % (N_MEDIA_STEER_PILOTO,
                            N_MEDIA_STEER_PILOTO * PERIODO_PADRAO,
                            N_MEDIA_STEER, N_MEDIA_STEER * PERIODO_PADRAO))
    ap.add_argument("--trim-steer", type=float, default=0.0,
                    dest="trim_steer",
                    help="ajuste fino do ZERO da direcao, em GRAUS, para o "
                         "carro andar reto com --steer 0. Satura em +-10 "
                         "(limite do class_servos.set_trim). Positivo vira "
                         "para a ESQUERDA, como o delta. Ajuste ate o desvio "
                         "lateral sumir; o valor usado vai no cabecalho do CSV")
    ap.add_argument("--inverter-steer", action="store_true",
                    dest="inverter_steer",
                    help="troca o sinal do estercamento")
    ap.add_argument("--homem-morto", action="store_true", dest="homem_morto",
                    help="so acelera com L1 pressionado")
    ap.add_argument("--acel-taxa", action="store_true", dest="acel_taxa",
                    help="gatilho comanda TAXA (set_torque), nao posicao")
    ap.add_argument("--timeout-controle", type=float,
                    default=TIMEOUT_CONTROLE, dest="timeout_controle",
                    help="OBSOLETO e SEM EFEITO. Aceito para "
                         "nao quebrar linha de comando antiga. O controle "
                         "sumir passou a ser detectado por sondagem do "
                         "dispositivo, e nao por silencio de eventos")
    ap.add_argument("--listar", action="store_true",
                    help="lista os dispositivos de entrada e sai")
    ap.add_argument("--sem-buzzer", action="store_true", dest="sem_buzzer",
                    help="desliga o aviso sonoro (um beep quando os 2 s de "
                         "largada acabam, dois quando os 5 s de cauda acabam)")
    ap.add_argument("--testar-buzzer", action="store_true",
                    dest="testar_buzzer",
                    help="toca os dois avisos e sai; use ANTES dos ensaios "
                         "para conferir que o buzzer responde")
    ap.add_argument("--fake", action="store_true",
                    help="autoteste, sem hardware")
    args = ap.parse_args(argv)

    if args.fake:
        return 0 if autoteste() else 1
    if args.listar:
        ControlePS5.listar()
        return 0
    if args.testar_buzzer:
        return _testar_buzzer()
    if args.corte is not None and args.corte <= args.arranque:
        raise SystemExit("[pilotar] --corte precisa ser maior que --arranque")
    _avisa_velmax(args.velmax)
    pilotar(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
