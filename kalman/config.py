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
"""Constantes dos filtros. A procedência de cada valor está no kalman/README.md."""
import numpy as np

# --- geometria do veículo
RAIO_RODA = 0.07779      # [m] raio efetivo (o geométrico é 0,08)
REDUCAO_EIXO = 7.80      # motor -> roda
ENTRE_EIXOS_L = 0.36     # [m]
ELL_S = 0.40             # [m] face do ultrassom ao eixo traseiro
STEER_MAX = np.deg2rad(20.0)   # [rad] limite de software do class_servos

# delta_real = GANHO_STEER * delta_comandado + TRIM_STEER
GANHO_STEER = 1.0
TRIM_STEER = 0.0         # [rad]
SIGMA_STEER = np.deg2rad(1.0)  # [rad] erro do esterçamento comandado

# --- taxas
F_LOOP_CONTROLE = 20.0   # [Hz]
DT_LOOP = 1.0 / F_LOOP_CONTROLE
F_ENCODER_FIRMWARE = 31.8  # [Hz] medido
ENCODER_RESOLUCAO = 360  # pulsos por volta do motor

# --- encoder
SIGMA_ACC = 1.26         # [m/s^2] aceleração desconhecida (ruído de processo de v)
SIGMA_JERK = 33.0        # [m/s^3] jerk, filtro fundido de fusao.py
SIGMA_JERK_ENC3 = 1.0    # [m/s^3] jerk, encoder de 3 estados
SIGMA_ENC = 0.0239       # [m/s] velocidade constante
SIGMA_ENC_OPERACAO = 0.12  # [m/s] em operação, inclui o erro de escala (fusao_giro)
SIGMA_ZUPT = 0.01        # [m/s] pseudo-medida v = 0 com o carro parado

# --- ultrassom
SIGMA_US = 0.018         # [m] em operação
SIGMA_US_WALK = 0.05     # [m] passeio aleatório, ultrassom de 1 estado
SIGMA_US_ACCEL = 0.30    # [m/s^2] aceleração relativa, ultrassom de 2 estados
US_MIN_RANGE = 0.03      # [m]
US_MAX_RANGE = 2.00      # [m] limite de confiança
ALFA_US = np.deg2rad(15.4)  # [rad] meia-abertura do feixe
US_MAX_RANGE_FISICO = 4.00  # [m] só para o timeout
MAX_JUMP_MARGIN = 0.50   # [m] margem do teste de salto
V_MAX_VEICULO = 5.0      # [m/s] máximo físico, para o teste de salto
US_ECHO_TIMEOUT = 0.030  # [s] cobre US_MAX_RANGE_FISICO a 0 °C

# --- unidade inercial
SIGMA_GYRO = np.deg2rad(0.147)     # [rad/s] parado
AMOSTRAS_CALIB_GIRO = 250          # amostras para o viés inicial
SIGMA_BIAS = np.deg2rad(0.05)      # [rad/s/sqrt(s)] passeio do viés, filtros inerciais
SIGMA_ACC_MEAS_PARADO = 0.18       # [m/s^2] acelerômetro x, parado
SIGMA_ACC_MEAS_MOVEL = 1.07        # [m/s^2] acelerômetro x, cruzeiro
SIGMA_BIAS_FUNDIDO = np.deg2rad(5e-4)  # [rad/s/sqrt(s)] passeio do viés, filtro fundido
SIGMA_MAG = np.deg2rad(1.35)       # [rad] parado, em bancada
SIGMA_ACC_TILT = np.deg2rad(9.12)          # [rad] inclinação, filtros inerciais
SIGMA_ACC_TILT_PARADO = np.deg2rad(1.18)   # [rad]
SIGMA_ACC_TILT_MOVEL = np.deg2rad(19.09)   # [rad]
SIGMA_GRADE = np.deg2rad(0.5)      # [rad/sqrt(s)] passeio da arfagem (não medido)
SIGMA_GYRO_PARADO = np.deg2rad(0.147)  # [rad/s]
SIGMA_GYRO_MOVEL = np.deg2rad(5.41)    # [rad/s]

# --- detector de movimento (kalman/movimento.py)
JANELA_MOVIMENTO = 5        # amostras (0,25 s a 20 Hz)
LIMIAR_MOV_LIGA = 0.90      # [m/s^2]
LIMIAR_MOV_DESLIGA = 0.60   # [m/s^2]
N_DEBOUNCE_MOV_LIGA = 1     # amostras
N_DEBOUNCE_MOV_DESLIGA = 6  # amostras (0,30 s)

GRAVITY = 9.80665           # [m/s^2]
ACCEL_GATE_MARGIN = 0.5     # [m/s^2] tolerância de | |a| - g | para corrigir a atitude

# --- incertezas iniciais, filtros individuais
SIGMA_A0_ENC3 = 0.1         # [m/s^2]
SIGMA_D0_US2 = 0.1          # [m/s]

# --- gate qui-quadrado, 1 grau de liberdade
GATING_THRESHOLD_95 = 3.84
GATING_THRESHOLD_99 = 6.63
GATING_THRESHOLD_999 = 10.83
GATING_THRESHOLD = GATING_THRESHOLD_99

# --- P0 do filtro fundido (procedimento de largada)
SIGMA_POS0 = 0.01               # [m] trena
SIGMA_PSI0 = np.deg2rad(2.0)    # [rad] esquadro
SIGMA_V0 = 0.05                 # [m/s]
SIGMA_THETA0 = np.deg2rad(5.0)  # [rad]
SIGMA_BIAS0 = np.deg2rad(0.5)   # [rad/s]

# --- escapatória do gate (filtro fundido)
GATE_ESCAPE_N = 3           # rejeições seguidas antes de inflar P
GATE_ESCAPE_FATOR = 2.0     # fator de P a cada rejeição além do limite

# --- velocidade do som
SOUND_TEMP_C = 20.0         # [°C] ajustar por sessão


def sound_speed(temp_c=SOUND_TEMP_C):
    """Velocidade do som no ar [m/s], com T em °C."""
    return 331.3 + 0.606 * temp_c


ENCODER_BAUDRATE = 115200
