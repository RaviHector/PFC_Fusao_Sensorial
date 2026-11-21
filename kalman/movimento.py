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
"""Detector de movimento a partir do próprio acelerômetro."""
import numpy as np

from kalman.config import (JANELA_MOVIMENTO, LIMIAR_MOV_DESLIGA,
                           LIMIAR_MOV_LIGA, N_DEBOUNCE_MOV_DESLIGA,
                           N_DEBOUNCE_MOV_LIGA)


class EstadoMovimento:
    """Schmitt trigger com debounce sobre o desvio padrão móvel de ||a||."""

    def __init__(self, em_movimento=False):
        self.em_movimento = bool(em_movimento)
        self.n_transicoes = 0
        self._buf = []
        self._contador = 0

    def atualiza(self, a_norma):
        """
        Recebe ||a|| desta amostra [m/s^2] e devolve True se em movimento.
        """
        if a_norma is None:
            self._contador = 0
            return self.em_movimento

        self._buf.append(float(a_norma))
        if len(self._buf) > JANELA_MOVIMENTO:
            self._buf.pop(0)
        if len(self._buf) < JANELA_MOVIMENTO:
            return self.em_movimento  # janela ainda nao cheia

        s = float(np.std(self._buf, ddof=1))
        if self.em_movimento:
            pede_troca = s < LIMIAR_MOV_DESLIGA
        else:
            pede_troca = s > LIMIAR_MOV_LIGA

        if not pede_troca:
            self._contador = 0
            return self.em_movimento

        self._contador += 1
        exigido = (N_DEBOUNCE_MOV_DESLIGA if self.em_movimento
                   else N_DEBOUNCE_MOV_LIGA)
        if self._contador >= exigido:
            self.em_movimento = not self.em_movimento
            self.n_transicoes += 1
            self._contador = 0
        return self.em_movimento

    @staticmethod
    def norma(ax, ay, az):
        """Conveniencia: ||a|| a partir dos tres eixos."""
        return float(np.sqrt(ax * ax + ay * ay + az * az))

    def __repr__(self):
        estado = "movimento" if self.em_movimento else "parado"
        return "EstadoMovimento(%s, %d transicoes)" % (estado, self.n_transicoes)


def _autoteste():
    import random

    print("=" * 70)
    print("AUTOTESTE  kalman/movimento.py")
    print("=" * 70)
    ok = True
    G = 9.80665
    random.seed(42)

    def alimenta(e, n, sigma, base=G):
        r = e.em_movimento
        for _ in range(n):
            r = e.atualiza(base + random.gauss(0.0, sigma))
        return r

    e = EstadoMovimento()
    assert e.em_movimento is False
    print("  comeca parado                                       OK")

    # PARADO: espalhamento medido de 0,145 m/s2 -> nunca liga
    e1 = EstadoMovimento()
    alimenta(e1, 400, 0.145)
    if e1.n_transicoes == 0:
        print("  parado (sigma 0,145) nao liga em 400 amostras       OK")
    else:
        print("  FALHOU: %d transicoes com o veiculo parado" % e1.n_transicoes)
        ok = False

    # Velocidade constante tem aceleração média zero, igual a parado.
    # O detector precisa acusar movimento mesmo assim.
    e2 = EstadoMovimento()
    r = alimenta(e2, 40, 2.712)  # cruzeiro: media ~g, espalhamento alto
    if r:
        print("  cruzeiro (media de a = 0) detectado como movimento  OK")
    else:
        print("  FALHOU: cruzeiro nao detectado -- media zero enganou")
        ok = False

    # COSTEIO: o par dificil, motor desligado
    e3 = EstadoMovimento()
    r = alimenta(e3, 40, 2.749)
    if r:
        print("  costeio (motor cortado) detectado como movimento    OK")
    else:
        print("  FALHOU: costeio nao detectado")
        ok = False

    # Chegada ao repouso: a vibração decai de 2,7 para 0,145 conforme o veículo
    # costeia até parar, e no caminho atravessa a zona morta.
    e4 = EstadoMovimento()
    alimenta(e4, 30, 2.7)  # rolando
    assert e4.em_movimento, "deveria estar em movimento antes de desacelerar"
    n_antes = e4.n_transicoes
    for k in range(40):  # decaimento exponencial de 2,7 ate 0,145
        sigma = 0.145 + (2.7 - 0.145) * np.exp(-k / 6.0)
        e4.atualiza(G + random.gauss(0.0, sigma))
    alimenta(e4, 60, 0.145)  # parado
    novas = e4.n_transicoes - n_antes
    if not e4.em_movimento and novas == 1:
        print("  chegada ao repouso: 1 transicao, sem piscar          OK")
    else:
        print("  FALHOU: %d transicoes na chegada, estado final %s"
              % (novas, "movimento" if e4.em_movimento else "parado"))
        ok = False

    # Assimetria, primeiro caso: uma queda curta de vibração com o veículo
    # andando (costeio lento, trecho liso) não pode virar "parado".
    def rola(e, n):
        for k in range(n):
            e.atualiza(G + (2.7 if k % 2 == 0 else -2.7))

    e7 = EstadoMovimento()
    rola(e7, 30)
    assert e7.em_movimento
    n_antes = e7.n_transicoes
    for _ in range(8):
        e7.atualiza(G)                                 # queda curta, 0,40 s
    rola(e7, 30)
    if e7.em_movimento and e7.n_transicoes == n_antes:
        print("  queda curta de vibracao andando nao vira 'parado'    OK")
    else:
        print("  FALHOU: queda curta virou 'parado'")
        ok = False

    # Segundo caso: partida com vibração moderada, entre o limiar de ligar
    # (0,90) e 1,20, tem de ser vista.
    e8 = EstadoMovimento()
    alimenta(e8, 40, 0.145)
    assert not e8.em_movimento
    r = False
    for k in range(40):
        r = e8.atualiza(G + (1.0 if k % 2 == 0 else -1.0))
    if r:
        print("  partida com vibracao moderada (~1,10) detectada      OK")
    else:
        print("  FALHOU: partida moderada nao detectada")
        ok = False

    # sem leitura valida preserva o estado
    e5 = EstadoMovimento()
    alimenta(e5, 40, 2.7)
    assert e5.atualiza(None) is True
    print("  sem leitura valida preserva o estado                OK")

    # a janela precisa encher antes de qualquer decisao
    e6 = EstadoMovimento()
    for _ in range(JANELA_MOVIMENTO - 1):
        assert e6.atualiza(G + 5.0) is False
    print("  nao decide antes de a janela encher                 OK")

    print("")
    print("=== RESULTADO: %s ===" % ("OK" if ok else "FALHOU"))
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _autoteste() else 1)
