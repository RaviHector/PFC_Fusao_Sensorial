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
"""Mostra, ao vivo, o que o controle de PS5 está mandando."""
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
import time

from bancada.pilotar import Comando, ControlePS5

# nome curto -> (rótulo que o operador reconhece, e se é evento ou contínuo)
BOTOES = [
    ("acel_cheio", "R1"),
    ("homem_morto", "L1"),
    ("parada", "circulo"),
    ("rearmar", "X"),
    ("troca_steer", "quadrado"),
    ("troca_acel", "triangulo"),
    ("marcar", "Create"),
    ("sair", "Options"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dispositivo", default=None,
                    help="/dev/input/eventN; sem isto, procura sozinho")
    ap.add_argument("--tempo", type=float, default=120.0,
                    help="duracao do teste [s]")
    args = ap.parse_args(argv)

    controle = ControlePS5(args.dispositivo)
    print("controle: %s" % controle.dev.name)
    print("")
    print("Aperte os botoes um a um. Cada um deve acender do lado esquerdo.")
    print("O gatilho R2 e a alavanca esquerda mexem os numeros.")
    print("Ctrl+C para sair.")
    print("")
    print("  ATENCAO: este teste NAO mexe no carro. Se tudo aparecer aqui e")
    print("  mesmo assim ele nao andar, o problema nao e o controle.")
    print("")

    cmd = Comando()
    t0 = time.monotonic()
    vistos = set()
    ult = ""

    try:
        while (time.monotonic() - t0) < args.tempo:
            t = time.monotonic()
            vivo_leitura = controle.atualiza(cmd)
            vivo = vivo_leitura and controle.esta_vivo()

            apertados = [rot for attr, rot in BOTOES if getattr(cmd, attr)]
            for r in apertados:
                if r not in vistos:
                    vistos.add(r)
                    print("\n  [OK] %s chegou" % r)

            idade = ("--" if cmd.t_evento is None
                     else "%4.1f s" % (t - cmd.t_evento))
            linha = ("  gatilho %4.2f | alavanca %+5.2f | vivo %-3s | "
                     "ultimo evento ha %s | %s"
                     % (cmd.gatilho, cmd.eixo_steer, "sim" if vivo else "NAO",
                        idade, ", ".join(apertados) if apertados else "-"))
            if linha != ult:
                print("\r" + linha.ljust(100), end="", flush=True)
                ult = linha

            cmd.limpa_eventos()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        controle.fechar()

    print("\n")
    print("=" * 62)
    faltaram = [rot for _, rot in BOTOES if rot not in vistos]
    print("  CHEGARAM : %s" % (", ".join(sorted(vistos)) or "nenhum"))
    print("  FALTARAM : %s" % (", ".join(faltaram) or "nenhum"))
    print("=" * 62)
    if faltaram:
        print("  Se voce apertou um dos que faltaram, o mapeamento esta")
        print("  errado para ele. Rode `python3 -m bancada.pilotar --listar`")
        print("  e me mande a linha do gamepad.")
    else:
        print("  O controle esta 100%. Se o carro nao anda, o problema esta")
        print("  no carro: chave do ESC, bateria de tracao ou conector.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
