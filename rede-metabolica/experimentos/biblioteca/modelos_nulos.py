"""Modelos nulos (Exp 4): redes aleatórias para comparar com a E. coli.

Se a E. coli e o modelo nulo se comportam igual, a propriedade vem só do que o
modelo preserva; se diferem, há organização além disso. Três modelos, do que
mais preserva ao que menos preserva:

    rewiring_reciproco  mesmos graus de entrada e saída e mesmos pares recíprocos
                        por nó (Milo et al., 2002)
    rewiring            mesmos graus de entrada e saída (Maslov e Sneppen, 2002)
    erdos_renyi         mesmos números de nós e de arestas (Erdős e Rényi, 1959)

Por que dois rewirings: cada reação reversível vira um par de arestas A→B e
B→A, e metade das arestas da E. coli está num par desses (reciprocidade 0,51;
Garlaschelli e Loffredo, 2004). O rewiring simples desfaz quase todos os pares
(reciprocidade ~0,01), e com isso muda o tamanho dos componentes por um motivo
químico conhecido, a reversibilidade. O rewiring recíproco separa os dois
efeitos: o que a E. coli tem de diferente dele não vem nem dos graus nem da
reversibilidade.

A troca é a de Maslov e Sneppen (2002): sorteia duas arestas a→b e c→d em
qualquer lugar da rede e troca as pontas (a→d e c→b), recusando laços e
arestas repetidas. A nx.directed_edge_swap não serve aqui: ela troca três
arestas de um caminho a→b→c→d, então nunca liga dois componentes fracos
diferentes, e a rede sorteada sai com exatamente os mesmos componentes fracos
da original (testado: 94 componentes e maior componente de 79,7% nas duas).

Os modelos nulos não têm reações, só arestas. Por isso neles só rodam as
métricas descritivas e a robustez (maior componente, índice R, alcance por
busca em largura); a cascata não se aplica.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterator

import networkx as nx
import numpy as np

import config

TIPOS = ("rewiring", "rewiring_reciproco", "erdos_renyi")

# Tentativas por troca pedida. Nas três redes sem moeda, 94% a 99% das
# tentativas são aceitas nos dois rewirings; 10 dá folga de sobra.
TENTATIVAS_POR_TROCA = 10


def _copia_simples(G: nx.DiGraph) -> nx.DiGraph:
    """Cópia sem atributos, com nós e arestas em ordem de código (o sorteio não depende da ordem de inserção)."""
    H = nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes))
    H.add_edges_from(sorted(G.edges))
    return H


def pares_reciprocos(G: nx.DiGraph) -> list[tuple]:
    """Pares {u, v} com u→v e v→u, como (u, v) com u < v, em ordem."""
    return sorted((u, v) for u, v in G.edges if u < v and G.has_edge(v, u))


def _trocar_simples(lista: list, i: int, j: int, arestas: set, evitar_reciproca: bool) -> bool:
    """Troca a→b, c→d por a→d, c→b, se não criar laço nem aresta repetida.

    Com evitar_reciproca, recusa também se d→a ou b→c já existirem (a troca
    criaria um par recíproco novo). Atualiza lista e arestas no lugar.
    """
    (a, b), (c, d) = lista[i], lista[j]
    if a == d or c == b or (a, d) in arestas or (c, b) in arestas:
        return False
    if evitar_reciproca and ((d, a) in arestas or (b, c) in arestas):
        return False
    arestas.difference_update({(a, b), (c, d)})
    arestas.update({(a, d), (c, b)})
    lista[i], lista[j] = (a, d), (c, b)
    return True


def _trocar_par(lista: list, i: int, j: int, arestas: set, inverter: bool) -> bool:
    """Troca os pares a↔b, c↔d por a↔d, c↔b (ou a↔c, d↔b, com inverter).

    Recusa se criar laço ou se já houver aresta, em qualquer sentido, entre
    as novas duplas. Preserva quantos pares recíprocos cada nó tem.
    """
    (a, b), (c, d) = lista[i], lista[j]
    if inverter:
        c, d = d, c
    if a == d or c == b:
        return False
    if {(a, d), (d, a), (c, b), (b, c)} & arestas:
        return False
    arestas.difference_update({(a, b), (b, a), (c, d), (d, c)})
    arestas.update({(a, d), (d, a), (c, b), (b, c)})
    lista[i], lista[j] = (min(a, d), max(a, d)), (min(c, b), max(c, b))
    return True


def rewiring(G: nx.DiGraph, trocas_por_aresta: int = config.TROCAS_POR_ARESTA,
             semente: int | None = None, preservar_reciprocidade: bool = False) -> nx.DiGraph:
    """Rede com os mesmos graus de entrada e saída de G, arestas embaralhadas.

    Faz trocas_por_aresta * (número de arestas) trocas aceitas de Maslov e
    Sneppen (2002), com até TENTATIVAS_POR_TROCA tentativas por troca.

    preservar_reciprocidade=True (rewiring recíproco, Milo et al., 2002): os
    pares recíprocos (a↔b) só trocam com outros pares e as arestas sem par só
    com outras sem par, sem criar par novo. Assim cada nó mantém o grau de
    entrada, o de saída e quantos pares recíprocos tem, e a reciprocidade da
    rede fica igual à de G. A cada tentativa, o tipo de troca é sorteado na
    proporção das arestas de cada tipo.

    A rede devolvida guarda em .graph["trocas"] quantas trocas foram aceitas e
    em .graph["tentativas"] quantas foram tentadas. Se as tentativas acabarem
    antes das trocas pedidas (rede pequena ou muito restrita), avisa com
    warnings.warn e devolve a rede como ficou.
    """
    H = _copia_simples(G)
    rng = np.random.default_rng(semente)
    arestas = set(H.edges)
    if preservar_reciprocidade:
        pares = pares_reciprocos(H)
        simples = sorted((u, v) for u, v in arestas if (v, u) not in arestas)
    else:
        pares, simples = [], sorted(arestas)
    nswap = trocas_por_aresta * len(arestas)
    max_tentativas = TENTATIVAS_POR_TROCA * nswap
    # Probabilidade de sortear uma troca de pares: a fração das arestas que está em pares.
    p_par = 2 * len(pares) / len(arestas) if arestas else 0.0

    trocas = tentativas = 0
    while trocas < nswap and tentativas < max_tentativas:
        tentativas += 1
        usar_pares = rng.random() < p_par
        lista = pares if usar_pares else simples
        if len(lista) < 2:
            continue
        i, j = rng.integers(len(lista), size=2)
        if i == j:
            continue
        if usar_pares:
            aceita = _trocar_par(lista, i, j, arestas, inverter=bool(rng.integers(2)))
        else:
            aceita = _trocar_simples(lista, i, j, arestas, evitar_reciproca=preservar_reciprocidade)
        trocas += aceita

    if trocas < nswap:
        warnings.warn(f"rewiring: só {trocas} de {nswap} trocas aceitas em {tentativas} tentativas",
                      stacklevel=2)
    R = nx.DiGraph()
    R.add_nodes_from(H.nodes)
    R.add_edges_from(sorted(arestas))
    R.graph.update(trocas=trocas, tentativas=tentativas)
    return R


def erdos_renyi(n: int, m: int, semente: int | None = None) -> nx.DiGraph:
    """Rede dirigida com n nós e m arestas sorteadas (nx.gnm_random_graph, directed=True).

    Os nós são os inteiros 0..n-1: não correspondem a metabólitos, então o
    alcance a partir dos nutrientes não se aplica nesse modelo.
    """
    return nx.gnm_random_graph(n, m, seed=semente, directed=True)


def gerar_nulos(G: nx.DiGraph, tipo: str, n_redes: int = config.N_MODELOS_NULOS,
                semente: int = config.SEMENTE) -> Iterator[nx.DiGraph]:
    """n_redes modelos nulos do tipo pedido (um de TIPOS) a partir de G.

    A rede i usa a semente semente * 1000 + i, para cada uma ser reprodutível sozinha.
    """
    if tipo not in TIPOS:
        raise ValueError(f"modelo nulo desconhecido: {tipo!r} (use um de {TIPOS})")
    for i in range(n_redes):
        semente_i = semente * 1000 + i
        if tipo == "erdos_renyi":
            yield erdos_renyi(G.number_of_nodes(), G.number_of_edges(), semente_i)
        else:
            yield rewiring(G, semente=semente_i, preservar_reciprocidade=(tipo == "rewiring_reciproco"))
