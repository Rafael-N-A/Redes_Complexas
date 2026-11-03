"""Estratégias de remoção de nós e curvas de robustez (Albert, Jeong e Barabási, 2000).

Estratégias (nomes usados em config.ESTRATEGIAS):
    aleatoria               ordem sorteada; média de várias repetições
    grau                    maior grau primeiro, calculado uma vez
    grau_adaptativo         recalcula o grau depois de cada remoção
    betweenness             maior betweenness primeiro, calculada uma vez
    betweenness_adaptativo  recalcula a cada config.PASSO_BETWEENNESS remoções

As métricas de dano vêm de metricas.py; este módulo só decide a ordem e
mede ao longo do caminho.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

import config


def ordem_inicial(G: nx.DiGraph, estrategia: str,
                  rng: np.random.Generator | None = None) -> list[str]:
    """Ordem de remoção das estratégias estáticas (aleatoria, grau, betweenness).

    Empates no grau ou na betweenness são desfeitos pelo código do nó, para
    que a ordem seja sempre a mesma.
    """
    raise NotImplementedError


def proximo_adaptativo(G: nx.DiGraph, estrategia: str) -> str:
    """Próximo nó a remover nas estratégias adaptativas, no estado atual de G.

    grau_adaptativo: maior grau total agora. betweenness_adaptativo: maior
    betweenness agora (quem chama decide recalcular só a cada passo_betweenness).
    """
    raise NotImplementedError


def simular_remocao(G: nx.DiGraph, estrategia: str,
                    rng: np.random.Generator | None = None,
                    nutrientes: Iterable[str] = (),
                    passo_eficiencia: float = config.PASSO_EFICIENCIA,
                    passo_betweenness: int = config.PASSO_BETWEENNESS,
                    medir_eficiencia: bool = True) -> pd.DataFrame:
    """Remove os nós de G um a um, na ordem da estratégia, até não sobrar nenhum.

    Uma linha por passo, colunas:
        n_removidos, fracao_removida,
        maior_fraco, maior_forte  (frações do tamanho original, a cada passo)
        alcance                   (a partir dos nutrientes, a cada passo)
        eficiencia                (só nos passos em que a fração removida cruza
                                   um múltiplo de passo_eficiencia; NaN nos demais
                                   e em todos os passos se medir_eficiencia=False)
    Trabalha numa cópia de G. rng é obrigatório na estratégia aleatória.
    """
    raise NotImplementedError


def curva_aleatoria(G: nx.DiGraph, n_rep: int, semente: int,
                    nutrientes: Iterable[str] = (),
                    medir_eficiencia: bool = True) -> pd.DataFrame:
    """Média e desvio padrão de n_rep simulações aleatórias, por fração removida.

    Colunas: as de simular_remocao, mais <metrica>_dp para cada métrica.
    Cada repetição usa np.random.default_rng([semente, i]), para ser reprodutível.
    """
    raise NotImplementedError


def comparar_estrategias(G: nx.DiGraph, estrategias: Iterable[str] = config.ESTRATEGIAS,
                         n_rep: int = config.N_REPETICOES_ALEATORIAS,
                         semente: int = config.SEMENTE,
                         nutrientes: Iterable[str] = (),
                         medir_eficiencia: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda todas as estratégias em G. É o que o exp1 e o exp4 chamam.

    Retorna (curvas, indices):
        curvas:  formato longo, coluna "estrategia" + colunas de simular_remocao
        indices: uma linha por estratégia, com R_fraco e R_forte (metricas.indice_robustez)
    """
    raise NotImplementedError
