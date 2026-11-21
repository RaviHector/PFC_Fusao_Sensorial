# Fusão sensorial para o veículo de escala reduzida

Código do trabalho de fusão de encoder, unidade inercial (MPU-9250) e ultrassom
(HC-SR04) por filtro de Kalman estendido, no veículo de escala reduzida do LSI
(Raspberry Pi 5 + Arduino Nano). Este repositório traz o código que roda no
carrinho e os scripts para refazer os ensaios de caracterização dos sensores.

## Estrutura

| pasta / arquivo | o que é |
|---|---|
| `class_*.py`, `main.py` | o código do carrinho (classes de sensores, servos, buzzer, câmera e o laço de controle) |
| `arduino_encoder/` | firmware do Arduino Nano que lê o encoder e envia o RPM pela serial |
| `kalman/` | os filtros de Kalman: o ciclo genérico, os seis filtros individuais, o filtro fundido e as constantes |
| `bancada/` | drivers dos sensores para os ensaios, coleta de dados, corridas programadas, pilotagem por controle de PS5 e as análises |
| `sim/` | simuladores com verdade conhecida, que validam a consistência dos filtros (NEES, NIS) |
| `testar_tudo.py` | roda todos os autotestes que não precisam de hardware (inclui os simuladores dos filtros individuais) |

Cada pasta tem um README com o que cada arquivo faz.

## Instalação

No Raspberry Pi OS:

```
sudo apt install python3-numpy python3-matplotlib python3-evdev
pip install pyserial smbus2 lgpio adafruit-circuitpython-servokit
```

`python3-evdev` só é preciso para pilotar com o controle de PS5
(`bancada/pilotar.py`). A câmera (`class_camera.py`) precisa de `opencv-python` e
`ultralytics` e é opcional. Para conferir o que falta na Raspberry, sem instalar
nada:

```
python3 -m bancada.checar_ambiente
```

O usuário precisa estar nos grupos `dialout`, `i2c` e `gpio`.

Fora da Raspberry (para rodar os filtros, os simuladores e as análises sobre
arquivos gravados) bastam `numpy`, `matplotlib` e `scipy`; o `scipy` só é usado
pelos simuladores.

## Uso rápido

Todos os comandos são executados a partir da raiz do repositório.

```
python testar_tudo.py                                   # autotestes, sem hardware
python -m sim.sim_fusao --reta --runs 100               # filtro fundido em simulação
python3 main.py                                         # laço de controle do carrinho
python3 -m bancada.corrida --velmax 1.0 --tempo 20 --corte 8 --condicao frenagem_c1
python -m bancada.analise_fusao dados/corrida_X.csv --frenagem --d0 4.80 --dfinal 1.39
```

Os scripts de `bancada/` gravam os CSVs em `dados/`, com um cabeçalho de
metadados (data, condição, temperatura, parâmetros).

## Ensaios de caracterização

Cada parâmetro de `kalman/config.py` que vem de medida tem um ensaio para
refazê-lo. Os passos estão no `bancada/README.md`; em resumo:

| parâmetro | ensaio | scripts |
|---|---|---|
| ruído do encoder (`SIGMA_ENC`) | corridas a velocidade constante no chão, e no cavalete com a roda no ar | `corrida` ou `coleta`, depois `analise` |
| aceleração desconhecida (`SIGMA_ACC`) | corridas de costeio | `corrida --corte`, depois `analise --derivar` |
| desaceleração por costeio | corridas com corte do motor | `corrida --corte`, depois `analise_costeio` e `analise_desacel_efetiva` |
| ruído do acelerômetro, parado e em cruzeiro | corridas e coletas paradas | `corrida` ou `coleta_multi`, depois `analise` |
| ruído e viés do giroscópio (`SIGMA_GYRO`, `AMOSTRAS_CALIB_GIRO`) | veículo parado | `calib_bias_gyro` |
| ruído do magnetômetro e da inclinação | veículo parado, duas orientações | `coleta --sensor imu`, depois `analise` |
| ruído do ultrassom e alcance mínimo | parado diante de parede a várias distâncias | `coleta --sensor ultrassom`, depois `analise` |
| braço e abertura do feixe do ultrassom (`ELL_S`, `ALFA_US`) | parede plana em várias orientações | `coleta --sensor ultrassom` |
| raio efetivo da roda (`RAIO_RODA`) | empurrão à mão entre duas marcas | `corrida --sem-motor`, depois `analise_empurrao` |

## Convenções

- Dentro dos filtros, SI com ângulos em radianos. Os drivers de `bancada/`
  entregam radianos; a conversão de graus acontece só na entrada.
- Sem medida nova, só predição: nenhum driver devolve a última leitura como se
  fosse nova, nem um valor inventado quando não há eco.
- Rumo positivo no sentido anti-horário (ISO 8855); `delta > 0` vira à esquerda.
