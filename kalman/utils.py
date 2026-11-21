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
"""wrap(), conversões de unidade e cronômetro monotônico."""
import time

import numpy as np


def wrap(angle):
    """Normaliza um ângulo [rad] para o intervalo [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def deg2rad(deg):
    return np.deg2rad(deg)


def rad2deg(rad):
    return np.rad2deg(rad)


def safe_dt(dt, fallback=1e-3):
    """Guarda contra dt <= 0 (relógio parado, ou primeira iteração)."""
    return dt if dt > 0.0 else fallback


class Timer:
    """Cronômetro monotônico para medir o dt entre iterações do filtro."""

    def __init__(self):
        self._last = time.perf_counter()

    def tick(self):
        """Retorna o dt [s] desde a última chamada e reinicia a contagem."""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        return safe_dt(dt)

    def peek(self):
        """Retorna o tempo decorrido [s] desde a última chamada, sem reiniciar."""
        return time.perf_counter() - self._last

    @staticmethod
    def now():
        return time.perf_counter()
