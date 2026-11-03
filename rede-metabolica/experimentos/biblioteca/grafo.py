"""Lê as tabelas do contrato e monta as visões do grafo.

Todas as visões saem da tabela "reacoes", que guarda cada reação inteira
(todos os substratos, produtos e genes juntos). Assim nenhuma visão perde
a informação de que uma reação precisa de todos os substratos ao mesmo tempo.

Visões:
    grafo_metabolitos   MultiDiGraph, metabólito -> metabólito, uma aresta por reação
    grafo_simples       DiGraph com peso = número de reações entre o par (métricas clássicas)
    grafo_nao_dirigido  Graph, para o coeficiente de clustering
    grafo_bipartido     DiGraph metabólito -> reação -> metabólito (cascata e nocaute)
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

import config
import contrato


def carregar_tabelas(org: str) -> dict[str, pd.DataFrame]:
    """Lê reacoes, metabolitos e genes de dados/<org>/ (listas já como list[str])."""
    return {t: contrato.carregar(org, t) for t in ("reacoes", "metabolitos", "genes")}


def nomes_metabolitos(metabolitos: pd.DataFrame) -> dict[str, str]:
    """C##### -> nome legível, para tabelas e figuras."""
    return dict(zip(metabolitos["metabolito"], metabolitos["nome"]))


def grafo_metabolitos(reacoes: pd.DataFrame, remover_moeda: bool = True,
                      so_irreversiveis: bool = False) -> nx.MultiDiGraph:
    """Rede metabólito-cêntrica dirigida (a modelagem principal do plano).

    Para cada reação, uma aresta de cada substrato para cada produto, com
    key = código R##### e atributos reacao, reversivel, genes, vias.
    Reação reversível também gera as arestas de volta (produto -> substrato),
    com o mesmo código.

    remover_moeda:     tira os metabólitos de config.METABOLITOS_MOEDA antes de criar as arestas
    so_irreversiveis:  teste de sensibilidade do plano (ignora as reações reversíveis)
    """
    raise NotImplementedError


def grafo_simples(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Colapsa arestas paralelas: uma aresta por par, peso = número de reações."""
    raise NotImplementedError


def grafo_nao_dirigido(G: nx.DiGraph | nx.MultiDiGraph) -> nx.Graph:
    """Versão não dirigida e simples, para o coeficiente de clustering."""
    raise NotImplementedError


def grafo_bipartido(reacoes: pd.DataFrame, remover_moeda: bool = True) -> nx.DiGraph:
    """Rede bipartida metabólito-reação, usada pela cascata e pelo nocaute.

    Nós de metabólito: id = C#####, atributo tipo="metabolito".
    Nós de reação: um por sentido possível, id = "R#####:ida" (e "R#####:volta"
    se a reação for reversível), atributos tipo="reacao", reacao="R#####", sentido.
    Arestas: substrato -> nó de reação -> produto (na volta, os papéis se invertem).

    Com remover_moeda=True, os metabólitos-moeda saem das listas de substratos
    e produtos. Para a cascata, isso equivale a tratá-los como sempre disponíveis.
    """
    raise NotImplementedError
