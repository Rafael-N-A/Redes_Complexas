"""Modelos nulos (Exp 4): redes aleatórias para comparar com a E. coli.

Se a E. coli e o modelo nulo se comportam igual, a propriedade vem só da
distribuição de graus; se diferem, há organização biológica além dela.

Os modelos nulos não têm reações, só arestas. Por isso neles só rodam as
métricas descritivas e a robustez (maior componente, índice R, alcance por
busca em largura); a cascata não se aplica.
"""
from __future__ import annotations

from collections.abc import Iterator

import networkx as nx

import config


def rewiring(G: nx.DiGraph, trocas_por_aresta: int = config.TROCAS_POR_ARESTA,
             semente: int | None = None) -> nx.DiGraph:
    """Rede com os mesmos graus de entrada e saída de G, arestas embaralhadas.

    Usa nx.directed_edge_swap numa cópia de G sem pesos, com
    nswap = trocas_por_aresta * número de arestas e max_tries = 10 * nswap,
    porque muitas trocas são recusadas.
    """
    raise NotImplementedError


def erdos_renyi(n: int, m: int, semente: int | None = None) -> nx.DiGraph:
    """Rede dirigida com n nós e m arestas sorteadas (nx.gnm_random_graph, directed=True).

    Os nós são renomeados para os códigos dos metabólitos de G por quem chama,
    se for preciso medir alcance a partir dos nutrientes; sem isso, alcance não se aplica.
    """
    raise NotImplementedError


def gerar_nulos(G: nx.DiGraph, tipo: str, n_redes: int = config.N_MODELOS_NULOS,
                semente: int = config.SEMENTE) -> Iterator[nx.DiGraph]:
    """n_redes modelos nulos do tipo "rewiring" ou "erdos_renyi" a partir de G.

    A rede i usa a semente semente * 1000 + i, para cada uma ser reprodutível sozinha.
    """
    raise NotImplementedError
