"""Nocaute de genes (Exp 2): deleção de gene -> reações que somem -> dano.

Regra das isoenzimas (OU): uma reação só some quando TODOS os genes ligados
a ela foram removidos. Deletar um gene de uma reação que tem outro gene
deixa a reação funcionando.

Limitação declarada no artigo: o KGML não diz se os genes de uma reação são
alternativas (isoenzimas, OU) ou peças do mesmo complexo (E). Aqui todos
valem como OU, o que subestima a fragilidade nos complexos.

Também ficam aqui as pontuações por gene usadas nos rankings do Exp 2
(dano simulado, grau, betweenness, choke points).
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd

import config


def reacoes_por_gene(reacoes: pd.DataFrame) -> dict[str, set[str]]:
    """gene -> códigos R##### das reações em que ele aparece (coluna reacoes.genes)."""
    raise NotImplementedError


def reacoes_perdidas(genes_removidos: Iterable[str], reacoes: pd.DataFrame) -> set[str]:
    """Códigos R##### cujos genes estão TODOS em genes_removidos.

    Reação sem nenhum gene na tabela nunca some (espontânea ou sem enzima
    conhecida no KEGG).
    """
    raise NotImplementedError


def dano_nocaute(B: nx.DiGraph, reacoes: pd.DataFrame, genes_removidos: Iterable[str],
                 medida: str = config.DANO_NOCAUTE) -> int:
    """Número de metabólitos perdidos ao deletar genes_removidos.

    medida="avalanche": cascata.cascata(B, reacoes_removidas=...) com as fontes externas.
    medida="alcance":   a mesma cascata, mas partindo só de config.NUTRIENTES
                        (mais a moeda, como em cascata.fontes_externas).
    """
    raise NotImplementedError


def dano_todos_genes(B: nx.DiGraph, reacoes: pd.DataFrame,
                     medida: str = config.DANO_NOCAUTE) -> pd.DataFrame:
    """Nocaute de cada gene da rede, um de cada vez.

    Uma linha por gene, colunas:
        gene, n_reacoes (em que aparece), n_reacoes_perdidas, dano
    Muitos genes terão dano 0 (todas as reações deles têm isoenzima); isso é
    esperado e o AUC lida bem com empates.
    """
    raise NotImplementedError


def centralidade_por_gene(reacoes: pd.DataFrame, valores_metabolito: pd.Series,
                          agregacao: str = config.AGREGACAO_CENTRALIDADE_GENE) -> pd.Series:
    """Leva uma medida por metabólito (grau ou betweenness) para uma por gene.

    Para cada gene, junta os metabólitos (substratos e produtos) das reações
    em que ele aparece e aplica a agregação: "max" ou "soma".
    Metabólitos fora de valores_metabolito (por exemplo, moeda removida) contam 0.
    Índice = gene.
    """
    raise NotImplementedError


def genes_em_choke_points(reacoes: pd.DataFrame, choke: set[str]) -> pd.Series:
    """True para o gene que aparece em pelo menos uma reação choke point.

    choke vem de metricas.choke_points(reacoes). Índice = gene.
    """
    raise NotImplementedError
