"""Gabarito biológico e métricas de acerto (Exp 2).

O gabarito é a lista de candidatos a essenciais da coleção Keio
(Baba et al., 2006), em dados/keio/essenciais.csv (ver o LEIAME.md da pasta).
Os genes da E. coli no KEGG usam o mesmo identificador (b-number), então o
cruzamento é direto.

A comparação fica restrita aos genes presentes na rede: genes essenciais
fora do metabolismo (ribossomo, replicação) não têm como ser previstos pela
topologia e tornariam a validação injusta.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import config


def carregar_keio(caminho: Path | None = None) -> set[str]:
    """Lê a coluna "gene" de essenciais.csv (a Tabela Suplementar 6 tem 300 candidatos;
    o texto de Baba et al. fala em 303; ver o LEIAME.md da pasta).

    caminho: config.ARQ_KEIO se não for dado (lido na hora da chamada).
    Levanta FileNotFoundError com a instrução do LEIAME.md se o arquivo não existir.
    """
    caminho = Path(config.ARQ_KEIO if caminho is None else caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"{config.caminho_legivel(caminho)} não existe. Baixe a Tabela Suplementar 6 de Baba et al. "
            f"(2006) e converta com python -m coleta.converter_keio (ver dados/keio/LEIAME.md).")
    tabela = pd.read_csv(caminho, dtype=str)
    if "gene" not in tabela.columns:
        raise ValueError(f"{config.caminho_legivel(caminho)} precisa de uma coluna 'gene' com os b-numbers")
    return {g.strip() for g in tabela["gene"].dropna() if g.strip()}


def rotulos(genes_rede: Iterable[str], essenciais: set[str]) -> pd.Series:
    """y: True para os genes da rede que estão na Keio. Índice = gene da rede.

    Genes essenciais que não aparecem na rede ficam de fora (ver o docstring do módulo);
    quem chama deve reportar quantos foram.
    """
    genes = list(dict.fromkeys(genes_rede))
    y = pd.Series([g in essenciais for g in genes], index=genes, name="essencial", dtype=bool)
    y.index.name = "gene"
    return y


def _alinhar(y: pd.Series, pontuacao: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """y e pontuação na ordem de y; gene sem pontuação conta 0."""
    return y.to_numpy(dtype=bool), pontuacao.reindex(y.index).fillna(0.0).to_numpy(dtype=float)


def auc(y: pd.Series, pontuacao: pd.Series) -> float:
    """Área sob a curva ROC (sklearn.metrics.roc_auc_score).

    Maior pontuação = mais provável de ser essencial. Alinha y e pontuação
    pelo índice antes de calcular. NaN se y tiver uma classe só.
    """
    rotulo, valores = _alinhar(y, pontuacao)
    if rotulo.all() or not rotulo.any():
        return math.nan
    return float(roc_auc_score(rotulo, valores))


def teste_auc(y: pd.Series, pontuacao: pd.Series) -> dict:
    """AUC com intervalo de 95% e teste contra o acaso (AUC = 0,5).

    p: teste de Mann-Whitney bilateral (o AUC é a estatística U normalizada),
       então também acusa um ranking significativamente PIOR que o acaso.
    ic_inf, ic_sup: intervalo de 95% pelo erro padrão de Hanley e McNeil (1982).
    """
    rotulo, valores = _alinhar(y, pontuacao)
    n1, n0 = int(rotulo.sum()), int((~rotulo).sum())
    a = auc(y, pontuacao)
    if math.isnan(a):
        return {"auc": a, "ic_inf": math.nan, "ic_sup": math.nan, "p": math.nan}
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    ep = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    p = float(mannwhitneyu(valores[rotulo], valores[~rotulo], alternative="two-sided").pvalue)
    return {"auc": a, "ic_inf": max(0.0, a - 1.96 * ep), "ic_sup": min(1.0, a + 1.96 * ep), "p": p}


def precisao_k(y: pd.Series, pontuacao: pd.Series, k: int) -> float:
    """Fração de essenciais entre os k genes de maior pontuação.

    Empates na fronteira do top-k são desfeitos pelo código do gene, para o
    resultado não depender da ordem das linhas.
    """
    rotulo, valores = _alinhar(y, pontuacao)
    tabela = pd.DataFrame({"gene": y.index, "pontuacao": valores, "essencial": rotulo})
    topo = tabela.sort_values(["pontuacao", "gene"], ascending=[False, True]).head(k)
    return float(topo["essencial"].mean()) if len(topo) else math.nan


def avaliar_rankings(y: pd.Series, rankings: dict[str, pd.Series],
                     ks: Iterable[int] = config.K_PRECISAO) -> pd.DataFrame:
    """Uma linha por ranking (ex.: "dano", "grau", "betweenness"), colunas:
    ranking, auc, auc_ic_inf, auc_ic_sup, p_auc, precisao_10, precisao_25, precisao_50
    (conforme ks), n_genes, n_essenciais, taxa_base.

    taxa_base é a fração de essenciais entre os genes: a precisão de um ranking ao acaso.
    """
    ks = list(ks)
    linhas = []
    for nome, pontuacao in rankings.items():
        teste = teste_auc(y, pontuacao)
        linha = {"ranking": nome, "auc": teste["auc"], "auc_ic_inf": teste["ic_inf"],
                 "auc_ic_sup": teste["ic_sup"], "p_auc": teste["p"]}
        linha.update({f"precisao_{k}": precisao_k(y, pontuacao, k) for k in ks})
        linha.update({"n_genes": len(y), "n_essenciais": int(y.sum()), "taxa_base": float(y.mean())})
        linhas.append(linha)
    colunas = ["ranking", "auc", "auc_ic_inf", "auc_ic_sup", "p_auc",
               *[f"precisao_{k}" for k in ks], "n_genes", "n_essenciais", "taxa_base"]
    return pd.DataFrame(linhas, columns=colunas)
