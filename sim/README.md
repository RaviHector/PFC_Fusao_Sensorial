# sim/

Simuladores com verdade conhecida. Cada um gera a trajetória verdadeira, sorteia
as medidas com o ruído declarado em `kalman/config.py`, roda o filtro e confere
se a incerteza que ele declara bate com o erro que ele comete. Rodam no PC, sem
hardware, e salvam as figuras em `sim/_output/`.

| arquivo | o que faz |
|---|---|
| `base.py` | NEES, NIS, teste de brancura das inovações, cobertura das bandas de ±k sigma, intervalos de confiança e figuras |
| `sim_encoder_2.py` | encoder de 2 estados: a posição não é observável e P[0,0] cresce sem parar |
| `sim_encoder_3.py` | encoder de 3 estados |
| `sim_ultrassom_1.py` | ultrassom de 1 estado, e o atraso sistemático na aproximação |
| `sim_ultrassom_2.py` | ultrassom de 2 estados, e a perda de alvo (P cresce sem medida) |
| `sim_imu_2.py` | inercial de 2 estados: convergência do viés e o rumo passando por ±180° |
| `sim_imu_6.py` | inercial de 6 estados, com e sem magnetômetro |
| `sim_fusao.py` | filtro fundido nas sete configurações, em vários cenários |

```
python -m sim.sim_encoder_2
python -m sim.sim_fusao --reta --runs 100
```

Critérios: o NEES médio tem de cair no intervalo de confiança qui-quadrado em
torno da dimensão do estado; o NIS médio, em torno de 1 por canal; as inovações
normalizadas não podem ter autocorrelação significativa; e pelo menos 95% dos
erros têm de ficar dentro da banda de ±3 sigma.

`sim_fusao.py` sem opções roda os cenários de formulação (o modelo do filtro
igual ao da verdade), reta (frenagem contra parede, com ecos espúrios do
ultrassom), círculo e corredor (curvas sem marco de posição), com 200
realizações cada; leva vários minutos. Cada cenário pode ser rodado sozinho
(`--formulacao`, `--reta`, `--circulo`, `--corredor`), e `--runs` muda o
número de realizações. `--vigente` roda a formulação de `fusao.py` (esterçamento
como entrada); `--rampa` e `--varredura` (sensibilidade ao `SIGMA_GRADE`) só
valem nela, porque a formulação usada é plana. `--sem-mag` tira o canal do
magnetômetro e `--sem-zupt` desliga a ZUPT e a ZARU.

Os resultados são apresentados pela mediana do erro, além da média: nas
configurações com ultrassom a distribuição do erro tem cauda pesada.
