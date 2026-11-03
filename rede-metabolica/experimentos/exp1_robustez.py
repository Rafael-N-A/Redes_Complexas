"""Experimento 1 — Robustez estrutural.

Pergunta: a rede da E. coli tolera remoções aleatórias e é frágil a remoções
direcionadas aos nós mais conectados, como prevê a teoria de redes livres de escala?

Lê:    dados/<org>/reacoes.csv
Grava: resultados/tabelas/exp1_curvas_<org>_<versao>.csv   (formato longo, uma linha por passo)
       resultados/tabelas/exp1_indices_<org>_<versao>.csv  (índice R por estratégia)
       resultados/figuras/exp1_<metrica>_<org>_<versao>.png
           para metrica em maior_fraco, maior_forte, eficiencia, alcance

É a etapa mais longa do pipeline (~35 min nas duas versões): deixe rodando
em segundo plano e use --so-figuras para ajustar os gráficos depois.

Uso, na raiz:
    python -m experimentos.exp1_robustez --org eco --versao ambas
    python -m experimentos.exp1_robustez --so-figuras
"""
from __future__ import annotations

import argparse

import config
from experimentos.biblioteca import figuras, grafo, robustez  # noqa: F401

METRICAS_FIGURA = ("maior_fraco", "maior_forte", "eficiencia", "alcance")


def rodar(org: str, versao: str) -> None:
    """Roda todas as estratégias de config.ESTRATEGIAS e grava curvas e índices.

    Passos:
      1. reacoes = grafo.carregar_tabelas(org)["reacoes"]
      2. G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))
      3. curvas, indices = robustez.comparar_estrategias(G, nutrientes=config.NUTRIENTES)
      4. Grave em config.arquivo_tabela("exp1", "curvas", org, versao) e ("exp1", "indices", ...).
    """
    raise NotImplementedError


def plotar(org: str, versao: str) -> None:
    """Lê exp1_curvas_<org>_<versao>.csv e grava uma figura por métrica de METRICAS_FIGURA.

    A figura principal do artigo é a de maior_fraco (curva clássica de
    Albert, Jeong e Barabási, 2000); a eficiência só tem pontos a cada 5%.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: robustez estrutural (remoção de nós).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--versao", choices=[*config.VERSOES_MOEDA, "ambas"], default="ambas",
                        help="com ou sem metabólitos-moeda")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    versoes = list(config.VERSOES_MOEDA) if args.versao == "ambas" else [args.versao]
    for versao in versoes:
        if not args.so_figuras:
            rodar(args.org, versao)
        plotar(args.org, versao)


if __name__ == "__main__":
    main()
