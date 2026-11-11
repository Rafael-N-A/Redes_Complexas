"""Nocaute de genes (Exp 2): deleção de gene -> reações que somem -> dano.

Regra das isoenzimas (OU): uma reação só some quando TODOS os genes ligados
a ela foram removidos. Deletar um gene de uma reação que tem outro gene
deixa a reação funcionando.

Limitação: o KGML não diz se os genes de uma reação são
alternativas (isoenzimas, OU) ou peças do mesmo complexo (E). Aqui todos
valem como OU, o que subestima a fragilidade nos complexos.

Duas medidas do dano de um nocaute:
    funcional  (dano_todos_genes)            metabólitos que a rede intacta produz e
               deixam de ser produzidos (a avalanche da cascata, na rede bipartida);
    estrutural (dano_estrutural_todos_genes) metabólitos que deixam o maior componente
               forte e o maior fraco, e a queda da eficiência, no grafo metabólito-cêntrico
               sem as arestas das reações perdidas.
O nocaute remove arestas (reações), nunca metabólitos.

Também fica aqui a pontuação por gene das centralidades usadas nos rankings
do Exp 2 (centralidade_por_gene).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import networkx as nx
import pandas as pd

import config
from experimentos.biblioteca import cascata, grafo, metricas

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
                 (na eco sem moeda, só glicose e sulfato; apaga 47% da rede intacta);
                 usado na checagem da via do chiquimato (checar_inferencias.py).
    """
    if medida == "avalanche":
        return cascata.fontes_externas(B)
    if medida == "alcance":
        metabolitos = set(grafo.nos_metabolito(B))
        return metabolitos & (set(config.NUTRIENTES) | set(config.METABOLITOS_MOEDA))
    raise ValueError(f"medida de dano desconhecida: {medida!r} (use {' ou '.join(MEDIDAS)})")


def dano_todos_genes(B: nx.DiGraph, reacoes: pd.DataFrame,
                     medida: str = config.DANO_NOCAUTE) -> pd.DataFrame:
    """Nocaute de cada gene da rede, um de cada vez.

    O dano é o tamanho da cascata (cascata.cascata) com as fontes de
    fontes_da_medida(B, medida). Uma linha por gene, colunas:
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


def grafo_sem_reacoes(M: nx.MultiDiGraph, perdidas: Iterable[str], nos: Iterable[str] | None = None) -> nx.DiGraph:
    """O grafo simples de M sem as arestas das reações perdidas, com todos os nós.

    Copia M, tira as arestas cuja key (o código R#####) está em perdidas, nos dois
    sentidos das reversíveis, e colapsa as paralelas (grafo.grafo_simples): um par
    só perde a aresta se todas as reações entre os dois se perderam. nos: os nós
    que o resultado precisa ter (padrão: os de M); quem ficou sem nenhuma aresta
    continua lá, isolado, para continuar contando em N.
    """
    perdidas = set(perdidas)
    copia = M.copy()
    copia.remove_edges_from([(u, v, k) for u, v, k in M.edges(keys=True) if k in perdidas])
    H = grafo.grafo_simples(copia)
    H.add_nodes_from(M.nodes if nos is None else nos)
    return H


COLUNAS_DANO_ESTRUTURAL = ("gene", "n_reacoes", "n_reacoes_perdidas", "arestas_perdidas",
                           *metricas.COLUNAS_DANO_ESTRUTURAL)


def dano_estrutural_todos_genes(M: nx.MultiDiGraph, reacoes: pd.DataFrame) -> pd.DataFrame:
    """Nocaute de cada gene da rede, um de cada vez, medido na estrutura do grafo.

    M é o grafo metabólito-cêntrico (grafo.grafo_metabolitos, sem moeda no Exp 2).
    Para cada gene:
      1. as reações perdidas, pela regra OU e com a mesma lógica de dano_todos_genes;
      2. grafo_sem_reacoes(M, perdidas): o grafo simples sem as arestas delas, com
         todos os nós de M (N fica igual ao da rede intacta);
      3. metricas.queda_estrutural em relação ao grafo simples intacto.
    Gene que não perde nenhuma reação tem dano 0, sem recalcular nada. Reação
    perdida que só liga metabólitos-moeda não tem aresta no grafo e não muda nada.

    Uma linha por gene, ordenada, colunas:
        gene, n_reacoes, n_reacoes_perdidas (como em dano_todos_genes),
        arestas_perdidas (do grafo simples),
        dano_forte, dano_fraco, dano_forte_fracao, dano_fraco_fracao, dano_eficiencia
    (ver metricas.queda_estrutural). Leva ~30 s na eco (574 dos 952 genes perdem reação).
    """
    G = grafo.grafo_simples(M)
    nos = list(G.nodes)
    n = len(nos)
    antes = metricas.medidas_estruturais(G, n)
    por_gene = reacoes_por_gene(reacoes)
    genes_da_reacao = {codigo: set(genes) for codigo, genes in zip(reacoes["reacao"], reacoes["genes"])}
    zero = {coluna: 0 for coluna in ("dano_forte", "dano_fraco")}
    zero.update({coluna: 0.0 for coluna in ("dano_forte_fracao", "dano_fraco_fracao", "dano_eficiencia")})
    linhas = []
    for gene in sorted(por_gene):
        perdidas = {c for c in por_gene[gene] if genes_da_reacao[c] <= {gene}}
        linha = {"gene": gene, "n_reacoes": len(por_gene[gene]), "n_reacoes_perdidas": len(perdidas)}
        if perdidas:
            H = grafo_sem_reacoes(M, perdidas, nos)
            linha["arestas_perdidas"] = G.number_of_edges() - H.number_of_edges()
            linha.update(metricas.queda_estrutural(antes, metricas.medidas_estruturais(H, n), n))
        else:
            linha.update(arestas_perdidas=0, **zero)
        linhas.append(linha)
    return pd.DataFrame(linhas, columns=list(COLUNAS_DANO_ESTRUTURAL))


def centralidade_por_gene(reacoes: pd.DataFrame, valores_metabolito: pd.Series,
                          agregacao: str = config.AGREGACAO_CENTRALIDADE_GENE) -> pd.Series:
    """Leva uma medida por metabólito (grau, betweenness ou PageRank) para uma por gene.

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
