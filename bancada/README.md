# bancada/

Código que roda na Raspberry para os ensaios: drivers dos sensores, coleta,
corridas e análises. Todo script aceita `--help`, e os que mexem com hardware
têm um `--fake`, que roda um autoteste sem hardware.

## Arquivos

| arquivo | o que faz |
|---|---|
| `drivers/base.py` | fontes de I/O reais e falsas (serial, GPIO, I2C), injetadas nos drivers |
| `drivers/encoder.py` | encoder pela serial do Nano; `getVel()` devolve `(vel, is_new)` |
| `drivers/imu.py` | MPU-9250 e AK8963 por I2C, leitura em bloco, radianos |
| `drivers/ultrassom.py` | HC-SR04 por GPIO, com prazo no eco e validação de faixa e de salto |
| `checar_ambiente.py` | confere pacotes, I2C, serial, GPIO e grupos do usuário |
| `coleta.py` | grava um sensor bruto num CSV |
| `coleta_multi.py` | grava encoder e IMU no mesmo laço |
| `corrida.py` | corrida com acelerador constante, gravando os três sensores; corte do motor, esterçamento fixo, ou sem motor |
| `pilotar.py` | pilotagem com controle de PS5 por Bluetooth, gravando os três sensores e o esterçamento comandado |
| `teste_controle.py` | mostra o que o controle está mandando, sem mexer no carro |
| `analise.py` | desvio-padrão, autocorrelação e distribuição de cada canal de um CSV; ruído de processo pela derivada |
| `analise_costeio.py` | desaceleração de regime no costeio, mecanismo (linear ou exponencial) e duração da transição |
| `analise_desacel_efetiva.py` | desaceleração que reproduz a distância de parada, contada desde o comando de corte |
| `analise_empurrao.py` | raio efetivo da roda, pelo empurrão à mão |
| `calib_bias_gyro.py` | viés e ruído do giroscópio, e quantas amostras bastam |
| `analise_fusao.py` | roda as sete configurações do filtro fundido sobre um CSV gravado (frenagem ou trajetória) |

## Drivers

- Sem medida nova, só predição. `encoder.getVel()` devolve `(vel, is_new)`;
  `ultrassom.ler()` devolve `(dist, ok, t)`, com `ok=False` quando não há eco,
  a leitura está fora da faixa ou o salto é fisicamente impossível. Nenhum
  driver repete a última leitura nem devolve o alcance máximo no lugar de uma
  medida.
- `sensor_vivo()` diz se o sensor ainda responde (sem medida há 0,3 s é
  falha); é diferente de `is_new`, que só diz se há medida neste ciclo.
- A leitura da serial tem teto de linhas por chamada (com o baud errado, ela
  nunca esvaziaria); só a linha mais recente vale.
- A IMU é lida em bloco, numa transação I2C por sensor, e o DRDY do
  magnetômetro é testado. Ângulos e taxas saem em radianos.
- No ultrassom, a espera pelo eco tem prazo, o tempo é medido com
  `perf_counter`, e a velocidade do som sai da temperatura (`--temp`).
- A fonte de I/O é injetada no construtor; `--fake` usa fontes sintéticas com
  verdade conhecida.

## Corrida e pilotagem

`corrida.py`: `Servos(velmax=V)` define um teto de PWM, e `set_torque(1.0)` faz
o acelerador subir até ele e ficar lá (PWM constante). `set_torque` é taxa, e
não posição: para cortar o motor o script zera o `th_pwm` diretamente. A
velocidade real segue `v = 0,942 velmax - 0,095` (abaixo de velmax 0,7 o
veículo anda aos trancos). O ultrassom roda numa thread, e a coluna `us_novo`
marca as amostras com leitura nova.

`pilotar.py` (controle de PS5 por Bluetooth; precisa de
`sudo apt install python3-evdev`). O esterçamento comandado vai gravado por
amostra: `delta_pedido_rad` (a alavanca) e `delta_cmd_rad` (depois da média
móvel do `class_servos`, o que chegou ao servo, e a entrada do filtro).

| comando | função |
|---|---|
| alavanca esquerda, X | esterçamento contínuo, até `STEER_MAX` |
| gatilho R2 | acelerador contínuo (por posição) |
| R1 | acelerador 100% enquanto pressionado |
| quadrado / triângulo | alterna manual/automático da direção / do acelerador |
| círculo | parada de emergência: corta a tração, centra a direção e trava |
| X | rearma |
| L1 | homem-morto, com `--homem-morto` |
| Create | marca um ponto (incrementa a coluna `marca`) |
| Options | encerra a gravação |

Em automático, a direção usa o `--steer` fixo e o acelerador segue o perfil do
`corrida.py`. Se o controle sumir (erro de leitura do descritor), o carro para.
O buzzer avisa quando passaram os 2 s parados da largada e quando o carro está
parado há 5 s depois da corrida. `python3 -m bancada.teste_controle` mostra o
que o controle manda, sem mexer no carro.

## Como refazer os ensaios de caracterização

Regras que valem para todos:

- Comando constante. Use `corrida.py` (acelerador saturado no teto do
  `--velmax`); conduzida à mão, a corrida mede a correção humana, e não o
  sensor.
- Aceitação pelo sinal da autocorrelação de lag 1 (`r_1` que o `analise.py`
  imprime): no encoder ela deve ser negativa, típica da contagem de pulsos. `r_1`
  positivo indica que se mediu a variação da grandeza, e não o ruído.
- Em movimento, analise só o patamar (`--janela T0 T1`, com a velocidade acima
  de ~90% do pico) e remova a tendência com `--detrend poly2`.
- Registre a condição em `--condicao`; ela vai para o cabeçalho do CSV.

### Encoder

Cinco corridas no chão, em linha reta, a ~1 m/s:

```
python3 -m bancada.corrida --velmax 1.2 --tempo 12 --condicao patamar_c1
python -m bancada.analise dados/corrida_X.csv --janela 4 10 --detrend poly2
```

O desvio do canal `vel_m_s` é o `SIGMA_ENC`. No cavalete, com a roda no ar:
`python3 -m bancada.coleta --sensor encoder --tempo 60`.

### Desaceleração por costeio e aceleração desconhecida

Corridas com o motor cortado no meio, gravando até parar:

```
python3 -m bancada.corrida --velmax 1.2 --tempo 20 --corte 8 --condicao frenagem_c1
python3 -m bancada.analise_costeio dados/corrida_*.csv
python3 -m bancada.analise_desacel_efetiva dados/corrida_*.csv
python -m bancada.analise dados/corrida_X.csv --derivar --sigma-sensor 0.0189
```

O `--derivar` estima o `SIGMA_ACC` pela derivada da velocidade, descontando o
ruído que a derivação amplifica (`--sigma-sensor` é o ruído do encoder no
cavalete).

### Unidade inercial

Giroscópio parado, numa superfície firme e sem vibração:

```
python -m bancada.calib_bias_gyro --amostras 2000
```

Dá o viés de cada eixo, o `SIGMA_GYRO` e a convergência da média com N
(`AMOSTRAS_CALIB_GIRO`).

Acelerômetro, giroscópio em movimento, magnetômetro e inclinação: o veículo
parado e ligado (`coleta --sensor imu --tempo 60`, em duas orientações para o
magnetômetro) e as corridas de patamar (`corrida.py`), com o `analise.py` sobre
os canais `ax`, `gz`, `yaw` e `pitch` no trecho correspondente. Os valores
parado e em movimento são diferentes, e o filtro usa os dois.

### Ultrassom

Ruído: o veículo parado diante de uma parede lisa e perpendicular, a 0,30,
0,60, 0,90, 1,20 e 1,50 m, 60 s em cada:

```
python3 -m bancada.coleta --sensor ultrassom --tempo 60 --temp 24 --condicao "parede 0,60 m"
python -m bancada.analise dados/ultrassom_X.csv
```

Meça a temperatura do ar (`--temp`): ela entra na velocidade do som. Parado, o
desvio-padrão bruto é dominado por poucos ecos espúrios; para o ruído de fundo,
use uma estatística robusta (1,4826 vezes o desvio absoluto mediano). Em
operação o ruído é bem maior que parado.

Alcance mínimo: alvo plano e rígido de 0,10 a 0,01 m, com
`--min-range 0.01` para que as leituras próximas não sejam descartadas.

Braço `ELL_S` e meia-abertura do feixe `ALFA_US`: o veículo a ~2 m de uma parede
plana, em cinco orientações (perpendicular e inclinado dos dois lados), ~400
disparos em cada, com a distância da face do sensor à parede medida com trena.
Com o carro perpendicular, `ELL_S = d_perp - média(z)`, sendo `d_perp` a
distância do eixo traseiro à parede. Dentro do cone a leitura quase não muda com
o ângulo; fora dele, `cos(psi - alfa) = z_perp / z_incl` dá a meia-abertura.

### Raio efetivo da roda

Empurrando o veículo à mão não há torque na roda, e portanto não há patinação.
Deixe-o parado ~5 s diante da parede, empurre até a marca seguinte e deixe-o
parado mais ~5 s:

```
python3 -m bancada.corrida --sem-motor --tempo 20 --condicao empurrao_c1
python -m bancada.analise_empurrao dados/corrida_X.csv --de 1.75 --para 0.25
```

`--de` e `--para` são as marcas, em metros da parede. A razão entre a distância
do encoder e a do ultrassom, ponderada sobre várias corridas e deslocamentos,
dá o erro de escala do encoder e dele o `RAIO_RODA`.

## Rodar o filtro sobre um ensaio

Frenagem em reta contra parede, com a largada (`--d0`) e a parada (`--dfinal`)
medidas com trena na face do ultrassom:

```
python -m bancada.analise_fusao dados/corrida_X.csv --frenagem --d0 4.80 --dfinal 1.39
```

Trajetória com marcos medidos (`pilotar.py`, botão Create em cada marco):

```
python -m bancada.analise_fusao dados/pilotado_X.csv --trajetoria --marcos marcos.csv
```

`marcos.csv` tem as colunas `n,x,y`. Sem medida do magnetômetro que corresponda
ao eixo x do mapa, use `--psi-ref auto` (largada apontada no sentido de x
crescente).
