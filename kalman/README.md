# kalman/

Os filtros de Kalman. Cada módulo de `filtros/` só monta as matrizes (F, Q, H,
R, P0) de um filtro; quem roda o ciclo de predição e atualização é o `core.py`.

| arquivo | o que faz |
|---|---|
| `config.py` | todas as constantes: geometria, taxas, ruídos, limites dos sensores, incertezas iniciais. Nenhum outro arquivo declara um sigma próprio |
| `core.py` | `KalmanFilter`: predição linear ou estendida, atualização escalar sequencial com forma de Joseph, gate qui-quadrado, histórico de NIS e inovações, escapatória do gate |
| `utils.py` | `wrap()` de ângulos, conversões e cronômetro monotônico |
| `movimento.py` | detector de movimento pelo espalhamento da aceleração |
| `filtros/encoder2.py` | encoder, 2 estados `[p, v]` |
| `filtros/encoder3.py` | encoder, 3 estados `[p, v, a]` |
| `filtros/ultrassom1.py` | ultrassom, 1 estado `[d]` |
| `filtros/ultrassom2.py` | ultrassom, 2 estados `[d, ddot]` |
| `filtros/imu2.py` | inercial, 2 estados `[psi, b]` no eixo z |
| `filtros/imu6.py` | inercial, 6 estados (atitude e vieses), com ou sem magnetômetro |
| `filtros/fusao.py` | EKF de 7 estados sobre a cinemática de Ackermann, com o esterçamento comandado como entrada |
| `filtros/fusao_giro.py` | o filtro fundido usado: giroscópio e acelerômetro como entradas |

Cada módulo roda um autoteste ou demonstração: `python -m kalman.core`,
`python -m kalman.filtros.fusao_giro` etc.

## Ciclo (`core.py`)

- Atualização escalar, um canal por vez: cada sensor corrige quando tem dado
  novo, e um outlier de um canal não descarta os demais.
- Modo EKF por dois ganchos opcionais: `self.f` (propagação não linear; `F`
  passa a ser o jacobiano) e `z_pred` em `update_scalar` (medida prevista; `H`
  passa a ser o jacobiano de h). Sem eles, o filtro é o linear.
- Ângulos (`angle_indices`) passam por `wrap()` na inovação e no estado
  corrigido, nunca na predição.
- Gate: a medida é rejeitada se NIS > 6,63 (99%, 1 grau de liberdade), e x e P
  ficam como estavam.
- Escapatória do gate: sem encoder, a velocidade é pouco observável e o filtro
  pode fixar-se num valor errado com P pequeno; daí em diante toda medida cai
  fora do gate. Depois de `GATE_ESCAPE_N` rejeições seguidas num canal, P é
  inflado a cada nova rejeição até o gate voltar a aceitar. Desligada por
  padrão; só o filtro fundido a liga, e só sem encoder.

## Filtros individuais

- `encoder2`: a posição não é observável (posto 1 de 2); P[0,0] cresce sem
  parar. `encoder3`: a aceleração é observável, inferida das velocidades.
- `ultrassom1`: passeio aleatório; com o veículo se aproximando, a estimativa
  fica atrás da distância real (viés, e não ruído). `ultrassom2` elimina o
  atraso, e com o alvo imóvel `ddot = -v`.
- `imu2` e `imu6`: o viés do giroscópio é estado, observável pela correção do
  ângulo; Q do viés nunca pode ser zero. O rumo nunca vem do acelerômetro, e
  sem magnetômetro o rumo e o viés em z não são observáveis.

## Filtro fundido

Estado `[x, y, psi, theta, v, b_gz, b_gy]`, o mesmo nas sete configurações (os
três sensores isolados, os três pares e os três juntos); só o conjunto de canais
muda. A posição é a do eixo traseiro.

Propagação (`fusao.py`), com o esterçamento `delta` como entrada:

```
dx/dt   = v cos(theta) cos(psi)        dpsi/dt = (v/L) tan(delta) / cos(theta)
dy/dt   = v cos(theta) sin(psi)        theta, v, vieses: constantes + ruído
```

integrada pelo arco exato (com v e delta constantes no passo, a trajetória é um
arco de círculo), e não por Euler, cujo erro em curva é sistemático. Q tem três
fontes: aceleração desconhecida (entra em x, y, psi e v, correlacionadas),
erro de esterçamento (entra em psi) e passeios aleatórios de theta e dos vieses.

Canais de medida:

| sensor | medida | observação |
|---|---|---|
| encoder | `z = v` | |
| ultrassom | `z = d_s / cos(phi)`, `d_s = (x_parede - x) - ELL_S cos(psi)`, `phi = max(|psi| - ALFA_US, 0)` | feixe em cone: dentro da abertura a leitura quase não depende de psi; com gate |
| giroscópio z | `z = (v/L) tan(delta) + b_gz` | o R soma o erro de esterçamento, com peso `(v/L) sec²(delta)` |
| giroscópio y | `z = b_gy` | |
| acelerômetro | `z = theta` | R parado ou em movimento |
| magnetômetro | `z = psi` | fora do filtro usado (ver abaixo) |

`fusao_giro.py` (o usado) muda, quando a configuração tem IMU:

1. o giroscópio move o rumo: `dpsi/dt = (gz - b_gz) / cos(theta)`, e sai das
   medidas; o ruído dele passa para Q;
2. o acelerômetro move a velocidade: `v' = v + a_x dt`, com o ruído medido em Q,
   e o nível é o plano (theta fixo em zero);
3. a pose anda com a velocidade média do passo (`V_MEDIA_NO_ARCO`);
4. o encoder entra com `SIGMA_ENC_OPERACAO`;
5. com o carro parado, pseudo-medidas de velocidade zero (ZUPT) e de giro zero
   (ZARU, que mede o viés do giroscópio);
6. a escapatória do gate só nas configurações sem encoder.

Sem IMU, usa a propagação de `fusao.py`. Nos nomes, "variante" é o
`fusao_giro.py` e "vigente" é o `fusao.py`.

O magnetômetro fica fora: dentro de prédios o zero dele muda de lugar para
lugar (erro de dezenas de graus), e no aberto o erro em movimento é de ~10°,
maior que a deriva do giroscópio nos percursos ensaiados. Para tirá-lo, basta
não fornecer o rumo dele nas amostras.

P0 vem do procedimento de largada (trena e esquadro), e não dos sensores. Folga
grande demais não é conservadora: com muita incerteza em psi o jacobiano do
ultrassom é avaliado longe do ponto verdadeiro e o EKF diverge.

## Detector de movimento (`movimento.py`)

Olha o desvio padrão móvel do módulo da aceleração, e não a média (que é zero
tanto parado quanto em velocidade constante). Parado e ligado, o espalhamento é
~0,15 m/s²; rolando, ~2,7 m/s², inclusive no costeio com o motor cortado. A
histerese é assimétrica: liga com uma amostra acima de 0,90 m/s² e desliga com
seis seguidas abaixo de 0,60 m/s², para não declarar "parado" com o carro ainda
andando.

## Procedência dos parâmetros (`config.py`)

| constante | valor | como foi obtido |
|---|---|---|
| `RAIO_RODA` | 0,07779 m | empurrão à mão, 12 corridas, encoder contra ultrassom com as pontas paradas (`bancada.analise_empurrao`) |
| `ENTRE_EIXOS_L` | 0,36 m | trena |
| `ELL_S` | 0,40 m | régua (±2 cm); com o carro perpendicular à parede deu 0,3973 m |
| `ALFA_US` | 15,4° | parede plana, 5 orientações, 400 disparos cada (catálogo: 15°) |
| `F_ENCODER_FIRMWARE` | 31,8 Hz | contagem de linhas na serial |
| `SIGMA_ENC` | 0,0239 m/s | 5 corridas a ~1 m/s, autocorrelação de lag 1 negativa (−0,20); no cavalete, 0,0189 |
| `SIGMA_ENC_OPERACAO` | 0,12 m/s | frenagens com trena: o encoder contou +1 a +2% (erro de escala), convertido por sigma = \|e\| / (dt √N) |
| `SIGMA_ACC` | 1,26 m/s² | 5 corridas de costeio, derivada da velocidade descontado o ruído da derivação |
| `SIGMA_US` | 0,018 m | em aproximação, confirmado pelo NIS ≈ 1 sobre dado real; parado o ruído é ~2 mm |
| `US_MIN_RANGE` | 0,03 m | 8 distâncias de 0,10 a 0,01 m; a leitura deixa de acompanhar abaixo de 0,02 m |
| `SIGMA_GYRO` | 0,147 °/s | duas calibrações de 2000 amostras, parado (`bancada.calib_bias_gyro`) |
| `AMOSTRAS_CALIB_GIRO` | 250 | convergência da média do viés (erro padrão 0,009 °/s) |
| `SIGMA_GYRO_MOVEL` | 5,41 °/s | patamares de cruzeiro |
| `SIGMA_ACC_MEAS_PARADO` / `_MOVEL` | 0,18 / 1,07 m/s² | parado e ligado / patamar de cruzeiro |
| `SIGMA_BIAS_FUNDIDO` | 5e-4 °/s/√s | limite superior, 40 min parado |
| `SIGMA_MAG` | 1,35° | parado, em bancada, duas orientações |
| `SIGMA_ACC_TILT_PARADO` / `_MOVEL` | 1,18° / 19,09° | parado e ligado / cruzeiro |
| `SIGMA_US_WALK`, `SIGMA_US_ACCEL`, `SIGMA_JERK_ENC3` | | sintonia por consistência (NEES) em simulação |
| `SIGMA_GRADE`, `SIGMA_STEER`, `SIGMA_ZUPT`, `GATE_ESCAPE_FATOR` | | escolhidos; não medidos |
| `US_MAX_RANGE` | 2,00 m | limite de confiança: acima disso a leitura degrada |
| `US_ECHO_TIMEOUT` | 30 ms | 2 × 4,0 m / 331,3 m/s = 24 ms, com folga; aumentar junto com `US_MAX_RANGE_FISICO` |
| `GATE_ESCAPE_N` | 3 | com 0,4 a 1,2% de ecos espúrios, três seguidos são muito improváveis |
