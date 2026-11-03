"""Métricas usadas por mais de um experimento.

Descritivas (exp0, exp4): tamanho, graus, clustering, caminhos, bow-tie.
De dano (exp1, exp4): maior componente, índice R, eficiência, alcance.
Topológicas para o exp2: centralidades e choke points.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Descritivas
# ---------------------------------------------------------------------------
def graus(G: nx.DiGraph) -> pd.DataFrame:
    """Uma linha por nó: grau de entrada, de saída e total (índice = nó)."""
    raise NotImplementedError


def distribuicao_graus(G: nx.DiGraph) -> pd.DataFrame:
    """Colunas: tipo ("entrada"/"saida"/"total"), grau, n_nos. Base da figura log-log."""
    raise NotImplementedError


def top_por_grau(G: nx.DiGraph, k: int = 10, nomes: dict[str, str] | None = None) -> pd.DataFrame:
    """Os k nós de maior grau total, com nome, para conferir os metabólitos-moeda."""
    raise NotImplementedError


def caminho_medio_e_diametro(G: nx.DiGraph) -> tuple[float, int]:
    """Calculados no maior componente fortemente conexo.

    Num grafo dirigido, a distância só existe entre pares mutuamente
    alcançáveis; fora do maior componente forte, muitos pares não têm caminho.
    """
    raise NotImplementedError


def bow_tie(G: nx.DiGraph) -> dict[str, int]:
    """Tamanhos de núcleo (maior componente forte), entrada, saída e restante (Ma e Zeng, 2003b).

    entrada: alcança o núcleo mas não é alcançada por ele
    saida:   é alcançada pelo núcleo mas não o alcança
    """
    raise NotImplementedError


def ajuste_lei_potencia(valores_grau: Iterable[int]) -> dict:
    """Opcional: teste de Clauset et al. com o pacote powerlaw.

    Devolver alpha, xmin e a comparação com a log-normal (R e p).
    """
    raise NotImplementedError


def tabela_descritiva(G: nx.DiGraph) -> dict:
    """Uma linha da Tabela 1 do artigo para o grafo simples dirigido G.

    Chaves: nos, arestas, densidade, grau_medio, clustering_medio (na versão
    não dirigida), caminho_medio, diametro, bow-tie (núcleo, entrada, saída,
    restante) e assortatividade. É reaproveitada pelo exp4 para os outros
    organismos e para os modelos nulos.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Dano (curvas de robustez)
# ---------------------------------------------------------------------------
def fracao_maior_componente(G: nx.DiGraph, n_original: int, forte: bool = False) -> float:
    """Tamanho do maior componente fraco (ou forte) dividido por n_original."""
    raise NotImplementedError


def indice_robustez(fracoes_maior_componente: Iterable[float]) -> float:
    """Índice R de Schneider et al. (2011): média da fração no maior
    componente após cada remoção, de 1 até N nós removidos."""
    raise NotImplementedError


def eficiencia_global_dirigida(G: nx.DiGraph, n_original: int) -> float:
    """Média de 1/d(i, j) sobre os pares ordenados i != j, dividida por n_original*(n_original-1).

    nx.global_efficiency não aceita grafo dirigido, por isso a versão própria.
    Normalizar pelo tamanho original faz a curva cair conforme a rede se parte.
    É cara (distâncias entre todos os pares): o exp1 só a mede a cada
    config.PASSO_EFICIENCIA da rede removida.
    """
    raise NotImplementedError


def alcance(G: nx.DiGraph, fontes: Iterable[str], n_original: int) -> float:
    """Fração de metabólitos alcançáveis por busca em largura a partir das fontes.

    Versão topológica (cada aresta vale sozinha), que funciona também nos
    modelos nulos. A versão que exige todos os substratos é cascata.disponiveis.
    Fontes ausentes do grafo são ignoradas.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Para o exp2
# ---------------------------------------------------------------------------
def betweenness(G: nx.DiGraph, k: int | None = None, semente: int | None = None) -> pd.Series:
    """Betweenness de cada nó; com k, a versão aproximada por amostragem (mais rápida)."""
    raise NotImplementedError


def choke_points(reacoes: pd.DataFrame) -> set[str]:
    """Reações que são a única produtora ou a única consumidora de algum metabólito
    (Rahman e Schomburg, 2006).

    Uma reação reversível conta como produtora e consumidora dos dois lados.
    """
    raise NotImplementedError
