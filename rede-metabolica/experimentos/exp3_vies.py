"""Experimento 3 — Viés por substância (cobre o item "spreading" do roteiro).

Pergunta: remover uma substância específica (piruvato, acetil-CoA, um
intermediário do ciclo de Krebs) dispara uma cascata de falhas? Até onde ela
se espalha?

Roda numa versão só (config.VERSAO_CASCATA): com a moeda sempre disponível,
a cascata dá o mesmo resultado com ou sem moeda.

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp3_substancias_<org>.csv  (6 escolhidas + 12 controles)
       resultados/tabelas/exp3_avalanches_<org>.csv   (avalanche de todos os metabólitos)
       resultados/figuras/exp3_avalanche_vs_grau_<org>.png
       resultados/figuras/exp3_distribuicao_<org>.png

Uso, na raiz:
    python -m experimentos.exp3_vies --org eco
    python -m experimentos.exp3_vies --so-figuras
"""
from __future__ import annotations

import argparse

import networkx as nx
import numpy as np

import config
from experimentos.biblioteca import cascata, figuras, grafo, metricas  # noqa: F401


def escolher_controles(G: nx.DiGraph, excluir: set[str], n: int,
                       rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Os dois grupos de controle.

    Retorna (hubs, baixo_grau):
      hubs:       os n nós de maior grau total, fora de excluir
      baixo_grau: n nós sorteados com rng entre a fração config.QUANTIL_BAIXO_GRAU
                  de menor grau, fora de excluir
    """
    raise NotImplementedError


def rodar(org: str) -> None:
    """Calcula as avalanches e grava as duas tabelas.

    Passos:
      1. tabelas = grafo.carregar_tabelas(org); remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
      2. B = grafo.grafo_bipartido(tabelas["reacoes"], remover_moeda=remover)
         G = grafo.grafo_simples(grafo.grafo_metabolitos(tabelas["reacoes"], remover_moeda=remover))
      3. Confira se os códigos de config.SUBSTANCIAS_EXP3 estão em B; avise os que faltarem.
      4. hubs, baixo = escolher_controles(G, set(config.SUBSTANCIAS_EXP3), config.N_CONTROLES_EXP3,
                                          np.random.default_rng(config.SEMENTE))
      5. Para cada metabólito dos três grupos: grupo, metabolito, nome, grau,
         avalanche = len(cascata.cascata(B, metabolitos_removidos=[m])).
      6. todos = cascata.avalanches_todos(B), junto com o grau de cada metabólito.
      7. Grave em config.arquivo_tabela("exp3", "substancias", org) e ("exp3", "avalanches", org).
    """
    raise NotImplementedError


def plotar(org: str) -> None:
    """Lê as duas tabelas e grava avalanche x grau e a distribuição dos tamanhos."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 3: viés por substância (cascata de falhas).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        rodar(args.org)
    plotar(args.org)


if __name__ == "__main__":
    main()
