"""Gabarito biológico e métricas de acerto (Exp 2).

O gabarito é a lista de candidatos a essenciais da coleção Keio
(Baba et al., 2006), em dados/keio/essenciais.csv (ver o LEIAME.md da pasta).
Os genes da E. coli no KEGG usam o mesmo identificador (b-number), então o
cruzamento é direto.

A comparação fica restrita aos genes presentes na rede: genes essenciais
fora do metabolismo (ribossomo, replicação) não têm como ser previstos pela
topologia e tornariam a validação injusta (risco do plano).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

import config


def carregar_keio(caminho: Path = config.ARQ_KEIO) -> set[str]:
    """Lê a coluna "gene" de essenciais.csv. O plano espera 303 genes.

    Levanta FileNotFoundError com a instrução do LEIAME.md se o arquivo não existir.
    """
    raise NotImplementedError


def rotulos(genes_rede: Iterable[str], essenciais: set[str]) -> pd.Series:
    """y: True para os genes da rede que estão na Keio. Índice = gene da rede.

    Genes essenciais que não aparecem na rede ficam de fora (ver o docstring do módulo);
    quem chama deve reportar quantos foram.
    """
    raise NotImplementedError


def auc(y: pd.Series, pontuacao: pd.Series) -> float:
    """Área sob a curva ROC (sklearn.metrics.roc_auc_score).

    Maior pontuação = mais provável de ser essencial. Alinha y e pontuação
    pelo índice antes de calcular.
    """
    raise NotImplementedError


def precisao_k(y: pd.Series, pontuacao: pd.Series, k: int) -> float:
    """Fração de essenciais entre os k genes de maior pontuação.

    Empates na fronteira do top-k são desfeitos pelo código do gene, para o
    resultado não depender da ordem das linhas.
    """
    raise NotImplementedError


def avaliar_rankings(y: pd.Series, rankings: dict[str, pd.Series],
                     ks: Iterable[int] = config.K_PRECISAO) -> pd.DataFrame:
    """Uma linha por ranking (ex.: "dano", "grau", "betweenness"), colunas:
    ranking, auc, precisao_10, precisao_25, precisao_50 (conforme ks), n_genes, n_essenciais.
    """
    raise NotImplementedError


def curva_roc(y: pd.Series, pontuacao: pd.Series) -> pd.DataFrame:
    """Pontos da curva ROC para a figura: colunas fpr, tpr (sklearn.metrics.roc_curve)."""
    raise NotImplementedError
