"""Lê as tabelas do contrato e monta as visões do grafo.

Todas as visões saem da tabela "reacoes", que guarda cada reação inteira
(todos os substratos, produtos e genes juntos). Assim nenhuma visão perde
a informação de que uma reação precisa de todos os substratos ao mesmo tempo.

Visões:
    grafo_metabolitos   MultiDiGraph, metabólito -> metabólito, uma aresta por reação
    grafo_simples       DiGraph com peso = número de reações entre o par (métricas clássicas)
    grafo_nao_dirigido  Graph, para o coeficiente de clustering e os modelos clássicos
    grafo_bipartido     DiGraph metabólito -> reação -> metabólito (cascata e nocaute)

Uso típico nos experimentos:
    reacoes = carregar_tabelas("eco")["reacoes"]
    G = grafo_simples(grafo_metabolitos(reacoes, remover_moeda=True))
    B = grafo_bipartido(reacoes, remover_moeda=True)
"""
from __future__ import annotations

from collections.abc import Iterable

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


def sem_moeda(metabolitos: Iterable[str], remover_moeda: bool = True) -> list[str]:
    """A lista sem os metabólitos de config.METABOLITOS_MOEDA (ou intacta, se remover_moeda=False)."""
    if not remover_moeda:
        return list(metabolitos)
    return [m for m in metabolitos if m not in config.METABOLITOS_MOEDA]


def reversivel(r: dict, conflito_como_irreversivel: bool = False) -> bool:
    """Se a reação vale nos dois sentidos no grafo.

    Por padrão, o que está na tabela (regra da coleta: vale o reversível).
    Com conflito_como_irreversivel=True, as reações em que as vias discordam
    (conflito_reversibilidade, 77 na E. coli) ficam irreversíveis, no sentido
    guardado na tabela, que é o da via irreversível: é a regra contrária
    ("vale o irreversível"), sem refazer a coleta. Serve só para conferir a
    escolha da regra (ver o README).
    """
    if conflito_como_irreversivel and r["conflito_reversibilidade"]:
        return False
    return bool(r["reversivel"])


def grafo_metabolitos(reacoes: pd.DataFrame, remover_moeda: bool = True,
                      conflito_como_irreversivel: bool = False) -> nx.MultiDiGraph:
    """Rede metabólito-cêntrica dirigida (a modelagem principal).

    Para cada reação, uma aresta de cada substrato para cada produto, com
    key = código R##### e atributos reacao, reversivel, genes, vias.
    Reação reversível também gera as arestas de volta (produto -> substrato),
    com o mesmo código.

    remover_moeda:     tira os metabólitos de config.METABOLITOS_MOEDA antes de criar as arestas
    conflito_como_irreversivel: a regra contrária de reversibilidade (ver reversivel)

    Duas escolhas que valem para todas as métricas:
      - um metabólito dos dois lados da mesma reação (ex.: R04241) não gera
        aresta dele para ele mesmo;
      - só entram os metabólitos com pelo menos uma aresta. Um produto que só
        se liga a metabólitos-moeda fica isolado depois da remoção e sai do grafo.
    """
    G = nx.MultiDiGraph()
    for r in reacoes.to_dict("records"):
        nos_dois_sentidos = reversivel(r, conflito_como_irreversivel)
        atributos = {"reacao": r["reacao"], "reversivel": nos_dois_sentidos,
                     "genes": r["genes"], "vias": r["vias"]}
        substratos = sem_moeda(r["substratos"], remover_moeda)
        produtos = sem_moeda(r["produtos"], remover_moeda)
        for a in substratos:
            for b in produtos:
                if a == b:
                    continue
                G.add_edge(a, b, key=r["reacao"], **atributos)
                if nos_dois_sentidos:
                    G.add_edge(b, a, key=r["reacao"], **atributos)
    return G


def grafo_simples(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Colapsa arestas paralelas: uma aresta por par, peso = número de reações.

    Atributos da aresta: weight (número de reações entre o par, na mesma
    direção) e reacoes (os códigos, em ordem).
    """
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes)
    for u, v, codigo in G.edges(keys=True):
        if S.has_edge(u, v):
            S[u][v]["reacoes"].append(codigo)
        else:
            S.add_edge(u, v, reacoes=[codigo])
    for _, _, dados in S.edges(data=True):
        dados["reacoes"].sort()
        dados["weight"] = len(dados["reacoes"])
    return S


def grafo_nao_dirigido(G: nx.DiGraph | nx.MultiDiGraph) -> nx.Graph:
    """Versão não dirigida e simples, para o coeficiente de clustering.

    u -> v e v -> u viram uma aresta só; arestas paralelas também.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    U.add_edges_from((u, v) for u, v in G.edges() if u != v)
    return U


def no_de_reacao(codigo: str, sentido: str) -> str:
    """Identificador do nó de reação na rede bipartida: "R00200:ida" ou "R00200:volta"."""
    return f"{codigo}:{sentido}"


def grafo_bipartido(reacoes: pd.DataFrame, remover_moeda: bool = True) -> nx.DiGraph:
    """Rede bipartida metabólito-reação, usada pela cascata e pelo nocaute.

    Nós de metabólito: id = C#####, atributo tipo="metabolito".
    Nós de reação: um por sentido possível, id = "R#####:ida" (e "R#####:volta"
    se a reação for reversível), atributos tipo="reacao", reacao="R#####",
    sentido e genes. Arestas: substrato -> nó de reação -> produto (na volta,
    os papéis se invertem).

    Com remover_moeda=True, os metabólitos-moeda saem das listas de substratos
    e produtos. Para a cascata, isso equivale a tratá-los como sempre disponíveis.
    Todo nó de reação é criado, mesmo sem substratos (só moeda) ou sem produtos:
    o número de nós de reação é sempre reações + reações reversíveis.
    """
    B = nx.DiGraph()
    for r in reacoes.to_dict("records"):
        substratos = sem_moeda(r["substratos"], remover_moeda)
        produtos = sem_moeda(r["produtos"], remover_moeda)
        sentidos = [("ida", substratos, produtos)]
        if reversivel(r):
            sentidos.append(("volta", produtos, substratos))
        for sentido, entram, saem in sentidos:
            no = no_de_reacao(r["reacao"], sentido)
            B.add_node(no, tipo="reacao", reacao=r["reacao"], sentido=sentido, genes=r["genes"])
            for m in entram:
                B.add_node(m, tipo="metabolito")
                B.add_edge(m, no)
            for m in saem:
                B.add_node(m, tipo="metabolito")
                B.add_edge(no, m)
    return B


def nos_metabolito(B: nx.DiGraph) -> set[str]:
    """Os nós de metabólito da rede bipartida."""
    return {n for n, tipo in B.nodes(data="tipo") if tipo == "metabolito"}
