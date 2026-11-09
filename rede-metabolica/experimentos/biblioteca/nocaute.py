"""Nocaute de genes (Exp 2): deleção de gene -> reações que somem -> dano.

Regra das isoenzimas (OU): uma reação só some quando TODOS os genes ligados
a ela foram removidos. Deletar um gene de uma reação que tem outro gene
deixa a reação funcionando.

Limitação: o KGML não diz se os genes de uma reação são
alternativas (isoenzimas, OU) ou peças do mesmo complexo (E). Aqui todos
valem como OU, o que subestima a fragilidade nos complexos.

Também ficam aqui as pontuações por gene usadas nos rankings do Exp 2
(dano simulado, grau, betweenness, choke points).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import networkx as nx
import pandas as pd

import config
from experimentos.biblioteca import cascata, grafo

MEDIDAS = ("avalanche", "alcance")


def reacoes_por_gene(reacoes: pd.DataFrame) -> dict[str, set[str]]:
    """gene -> códigos R##### das reações em que ele aparece (coluna reacoes.genes)."""
    resultado: dict[str, set[str]] = defaultdict(set)
    for codigo, genes in zip(reacoes["reacao"], reacoes["genes"]):
        for gene in genes:
            resultado[gene].add(codigo)
    return dict(resultado)


def reacoes_perdidas(genes_removidos: Iterable[str], reacoes: pd.DataFrame) -> set[str]:
    """Códigos R##### cujos genes estão TODOS em genes_removidos.

    Reação sem nenhum gene na tabela nunca some (espontânea ou sem enzima
    conhecida no KEGG).
    """
    removidos = set(genes_removidos)
    return {codigo for codigo, genes in zip(reacoes["reacao"], reacoes["genes"])
            if genes and set(genes) <= removidos}


def fontes_da_medida(B: nx.DiGraph, medida: str = config.DANO_NOCAUTE) -> set[str]:
    """Fontes da cascata para cada medida de dano, sempre na rede intacta B.

    "avalanche": cascata.fontes_externas(B), o "meio rico" do artigo (a Keio foi
                 medida em LB);
    "alcance":   config.NUTRIENTES mais a moeda presentes em B, o meio mínimo
                 (na eco sem moeda, só glicose e sulfato; apaga 47% da rede intacta).
    """
    if medida == "avalanche":
        return cascata.fontes_externas(B)
    if medida == "alcance":
        metabolitos = set(grafo.nos_metabolito(B))
        return metabolitos & (set(config.NUTRIENTES) | set(config.METABOLITOS_MOEDA))
    raise ValueError(f"medida de dano desconhecida: {medida!r} (use {' ou '.join(MEDIDAS)})")


def dano_nocaute(B: nx.DiGraph, reacoes: pd.DataFrame, genes_removidos: Iterable[str],
                 medida: str = config.DANO_NOCAUTE, fontes: Iterable[str] | None = None) -> int:
    """Número de metabólitos perdidos ao deletar genes_removidos.

    medida="avalanche": cascata.cascata(B, reacoes_removidas=...) com as fontes externas.
    medida="alcance":   a mesma cascata, mas partindo só de config.NUTRIENTES
                        (mais a moeda, como em cascata.fontes_externas).
    fontes: as de fontes_da_medida(B, medida), para não recalcular a cada gene.
    """
    perdidas = reacoes_perdidas(genes_removidos, reacoes)
    if not perdidas:
        return 0
    fontes = fontes_da_medida(B, medida) if fontes is None else set(fontes)
    return len(cascata.cascata(B, reacoes_removidas=perdidas, fontes=fontes))


def dano_todos_genes(B: nx.DiGraph, reacoes: pd.DataFrame,
                     medida: str = config.DANO_NOCAUTE) -> pd.DataFrame:
    """Nocaute de cada gene da rede, um de cada vez.

    Uma linha por gene, colunas:
        gene, n_reacoes (em que aparece), n_reacoes_perdidas, dano
    Muitos genes terão dano 0 (todas as reações deles têm isoenzima); isso é
    esperado e o AUC lida bem com empates. Ordenada pelo gene.
    """
    por_gene = reacoes_por_gene(reacoes)
    genes_da_reacao = {codigo: set(genes) for codigo, genes in zip(reacoes["reacao"], reacoes["genes"])}
    fontes = fontes_da_medida(B, medida)
    linhas = []
    for gene in sorted(por_gene):
        perdidas = {c for c in por_gene[gene] if genes_da_reacao[c] <= {gene}}
        dano = len(cascata.cascata(B, reacoes_removidas=perdidas, fontes=fontes)) if perdidas else 0
        linhas.append({"gene": gene, "n_reacoes": len(por_gene[gene]),
                       "n_reacoes_perdidas": len(perdidas), "dano": dano})
    return pd.DataFrame(linhas, columns=["gene", "n_reacoes", "n_reacoes_perdidas", "dano"])


def centralidade_por_gene(reacoes: pd.DataFrame, valores_metabolito: pd.Series,
                          agregacao: str = config.AGREGACAO_CENTRALIDADE_GENE) -> pd.Series:
    """Leva uma medida por metabólito (grau ou betweenness) para uma por gene.

    Para cada gene, junta os metabólitos (substratos e produtos) das reações
    em que ele aparece e aplica a agregação: "max" ou "soma".
    Metabólitos fora de valores_metabolito (por exemplo, moeda removida) contam 0.
    Índice = gene.
    """
    if agregacao not in ("max", "soma"):
        raise ValueError(f"agregação desconhecida: {agregacao!r} (use max ou soma)")
    metabolitos_do_gene: dict[str, set[str]] = defaultdict(set)
    for substratos, produtos, genes in zip(reacoes["substratos"], reacoes["produtos"], reacoes["genes"]):
        for gene in genes:
            metabolitos_do_gene[gene].update(substratos, produtos)
    valores = valores_metabolito.to_dict()
    pontuacao = {}
    for gene in sorted(metabolitos_do_gene):
        lista = [float(valores.get(m, 0.0)) for m in metabolitos_do_gene[gene]] or [0.0]
        pontuacao[gene] = max(lista) if agregacao == "max" else sum(lista)
    serie = pd.Series(pontuacao, name=valores_metabolito.name, dtype=float)
    serie.index.name = "gene"
    return serie


def genes_em_choke_points(reacoes: pd.DataFrame, choke: set[str]) -> pd.Series:
    """True para o gene que aparece em pelo menos uma reação choke point.

    choke vem de metricas.choke_points(reacoes). Índice = gene.
    """
    por_gene = reacoes_por_gene(reacoes)
    serie = pd.Series({gene: bool(por_gene[gene] & set(choke)) for gene in sorted(por_gene)},
                      name="choke", dtype=bool)
    serie.index.name = "gene"
    return serie
