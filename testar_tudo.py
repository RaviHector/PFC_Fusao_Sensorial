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
Roda todos os autotestes que não precisam de hardware e resume o resultado.

    python testar_tudo.py
"""
import os
import subprocess
import sys

TESTES = [
    ["-m", "kalman.core"],
    ["-m", "kalman.movimento"],
    ["-m", "kalman.filtros.encoder2"],
    ["-m", "kalman.filtros.encoder3"],
    ["-m", "kalman.filtros.ultrassom1"],
    ["-m", "kalman.filtros.ultrassom2"],
    ["-m", "kalman.filtros.imu2"],
    ["-m", "kalman.filtros.imu6"],
    ["-m", "kalman.filtros.fusao"],
    ["-m", "kalman.filtros.fusao_giro"],
    ["-m", "sim.sim_encoder_2"],
    ["-m", "sim.sim_encoder_3"],
    ["-m", "sim.sim_ultrassom_1"],
    ["-m", "sim.sim_ultrassom_2"],
    ["-m", "sim.sim_imu_2"],
    ["-m", "sim.sim_imu_6"],
    ["-m", "bancada.drivers.base", "--fake"],
    ["-m", "bancada.drivers.encoder", "--fake"],
    ["-m", "bancada.drivers.imu", "--fake"],
    ["-m", "bancada.drivers.ultrassom", "--fake"],
    ["-m", "bancada.coleta", "--fake"],
    ["-m", "bancada.coleta_multi", "--fake"],
    ["-m", "bancada.corrida", "--fake"],
    ["-m", "bancada.pilotar", "--fake"],
    ["-m", "bancada.analise", "--fake"],
    ["-m", "bancada.analise_costeio", "--fake"],
    ["-m", "bancada.calib_bias_gyro", "--fake"],
    ["-m", "bancada.analise_fusao", "--fake"],
]


def main():
    raiz = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, MPLBACKEND="Agg")
    falhas = 0
    for args in TESTES:
        r = subprocess.run([sys.executable] + args, cwd=raiz, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        saida = r.stdout + r.stderr
        ok = r.returncode == 0 and ("RESULTADO" not in saida or "RESULTADO: OK" in saida)
        falhas += not ok
        print("%-6s %s" % ("OK" if ok else "FALHOU", " ".join(args[1:])))
        if not ok:
            print("\n".join(saida.strip().splitlines()[-15:]))
    print("\n%d de %d autotestes passaram." % (len(TESTES) - falhas, len(TESTES)))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
