"""grafo.py e metricas.py em redes pequenas, de duas formas.

1. Rede de brinquedo com as respostas calculadas à mão (ver o desenho abaixo).
2. Comparação com uma implementação independente (força bruta ou a NetworkX)
   em 20 redes aleatórias pequenas, com semente fixa.

Rode na raiz do projeto: pytest
"""
from __future__ import annotations

import itertools
import math

import networkx as nx
import pandas as pd
import pytest

import contrato
from experimentos.biblioteca import grafo, metricas

ATP = "C00002"  # metabólito-moeda

# Rede de brinquedo, sem moeda:
#
#   X -> A -> B <-> C -> D -> A      (A, B, C, D formam o núcleo)
#        A ==> B por duas reações    (R1 e R8: peso 2)
#   F -> B, F -> G, B -> G           (R7: F + B -> B + G; o B dos dois lados não vira laço)
#   C -> E
#
# Bow-tie: núcleo {A, B, C, D}, entrada {X, F}, saída {E, G}, restante {}.
REACOES = [
    ("R1", ["A"], ["B"], False),
    ("R2", ["B"], ["C"], True),
    ("R3", ["C"], ["D"], False),
    ("R4", ["D", ATP], ["A"], False),
    ("R5", ["C"], ["E"], False),
    ("R6", ["X"], ["A"], False),
    ("R7", ["F", "B"], ["B", "G"], False),
    ("R8", ["A"], ["B"], False),
]


def tabela(linhas) -> pd.DataFrame:
    registros = [
        {"reacao": c, "substratos": s, "produtos": p, "reversivel": rev, "genes": [f"g_{c}"],
         "vias": ["eco00000"], "n_ocorrencias": 1, "conflito_reversibilidade": False,
         "orientacao_invertida": False, "lados_variam": False}
        for c, s, p, rev in linhas
    ]
    return pd.DataFrame(registros, columns=contrato.COLUNAS["reacoes"])


@pytest.fixture
def reacoes():
    return tabela(REACOES)


@pytest.fixture
def G(reacoes):
    return grafo.grafo_simples(grafo.grafo_metabolitos(reacoes))


# ---------------------------------------------------------------------------
# 1. Respostas calculadas à mão
# ---------------------------------------------------------------------------
def test_grafo_metabolitos(reacoes):
    M = grafo.grafo_metabolitos(reacoes)
    assert set(M.nodes) == {"A", "B", "C", "D", "E", "X", "F", "G"}  # sem ATP
    assert M.number_of_edges() == 11
    assert M.has_edge("B", "C", key="R2") and M.has_edge("C", "B", key="R2")  # reversível: ida e volta
    assert not M.has_edge("B", "B")                                            # sem laço
    assert set(M["A"]["B"]) == {"R1", "R8"}                                    # paralelas

    com_moeda = grafo.grafo_metabolitos(reacoes, remover_moeda=False)
    assert com_moeda.has_edge(ATP, "A", key="R4")

    irreversiveis = grafo.grafo_metabolitos(reacoes, so_irreversiveis=True)
    assert not irreversiveis.has_edge("B", "C") and not irreversiveis.has_edge("C", "B")


def test_metabolito_so_ligado_a_moeda_sai_do_grafo():
    M = grafo.grafo_metabolitos(tabela([("R1", [ATP], ["H"], False), ("R2", ["A"], ["B"], False)]))
    assert "H" not in M


def test_grafo_simples_e_nao_dirigido(reacoes, G):
    assert G.number_of_edges() == 10
    assert G["A"]["B"]["weight"] == 2 and G["A"]["B"]["reacoes"] == ["R1", "R8"]
    assert sum(w for _, _, w in G.edges(data="weight")) == 11

    U = grafo.grafo_nao_dirigido(G)
    assert U.number_of_edges() == 9  # B -> C e C -> B viram uma aresta
    assert nx.triangles(U)["G"] == 1  # o triângulo F, B, G


def test_grafo_bipartido(reacoes):
    B = grafo.grafo_bipartido(reacoes)
    assert grafo.nos_reacao(B) == {f"R{i}:ida" for i in range(1, 9)} | {"R2:volta"}
    assert set(B.predecessors("R2:volta")) == {"C"} and set(B.successors("R2:volta")) == {"B"}
    assert set(B.predecessors("R4:ida")) == {"D"}                        # ATP removido
    assert set(B.predecessors("R7:ida")) == {"F", "B"} and set(B.successors("R7:ida")) == {"B", "G"}
    assert ATP not in B

    com_moeda = grafo.grafo_bipartido(reacoes, remover_moeda=False)
    assert set(com_moeda.predecessors("R4:ida")) == {"D", ATP}
    assert com_moeda.nodes[ATP]["tipo"] == "metabolito"


def test_graus_e_top(G):
    tabela_graus = metricas.graus(G)
    assert tabela_graus.loc["B"].to_dict() == {"entrada": 3, "saida": 2, "total": 5}
    assert tabela_graus["entrada"].sum() == tabela_graus["saida"].sum() == 10
    assert list(metricas.top_por_grau(G, 2)["metabolito"]) == ["B", "C"]

    dist = metricas.distribuicao_graus(G)
    total = dist[dist["tipo"] == "total"].set_index("grau")["n_nos"].to_dict()
    assert total == {1: 2, 2: 3, 3: 1, 4: 1, 5: 1}


def test_bow_tie_e_caminhos(G):
    partes = metricas.componentes_bow_tie(G)
    assert partes == {"nucleo": {"A", "B", "C", "D"}, "entrada": {"X", "F"},
                      "saida": {"E", "G"}, "restante": set()}
    caminho, diametro = metricas.caminho_medio_e_diametro(G)
    assert caminho == pytest.approx(22 / 12) and diametro == 3


def test_tabela_descritiva(G):
    t = metricas.tabela_descritiva(G)
    assert (t["nos"], t["arestas"], t["nucleo"], t["entrada"], t["saida"], t["restante"]) == (8, 10, 4, 2, 2, 0)
    assert t["densidade"] == pytest.approx(10 / 56)
    assert t["grau_medio"] == pytest.approx(10 / 8)
    assert t["maior_fraco"] == 1.0 and t["componentes_fracos"] == 1
    assert 0 < t["clustering_medio"] < 1 and not math.isnan(t["assortatividade"])


def test_metricas_de_dano(G):
    assert metricas.fracao_maior_componente(G, 8) == 1.0
    assert metricas.fracao_maior_componente(G, 8, forte=True) == 0.5
    assert metricas.alcance(G, ["X"], 8) == pytest.approx(7 / 8)  # tudo menos F
    assert metricas.alcance(G, ["Z"], 8) == 0.0                     # fonte ausente
    assert metricas.indice_robustez([0.5, 0.25, 0.0]) == pytest.approx(0.25)

    caminho = nx.DiGraph([("a", "b"), ("b", "c")])  # 1/1 + 1/2 + 1/1 = 2,5
    assert metricas.eficiencia_global_dirigida(caminho, 3) == pytest.approx(2.5 / 6)
    assert metricas.eficiencia_global_dirigida(caminho, 4) == pytest.approx(2.5 / 12)


def test_choke_points(reacoes):
    # C só é produzido por R2; D só por R3 e só consumido por R4; E só por R5;
    # X só consumido por R6; F e G só por R7. A -> B tem duas reações (R1, R8).
    assert metricas.choke_points(reacoes) == {"R2", "R3", "R4", "R5", "R6", "R7"}


# ---------------------------------------------------------------------------
# 2. Contra uma implementação independente, em redes aleatórias
# ---------------------------------------------------------------------------
REDES_ALEATORIAS = [nx.gnp_random_graph(15, 0.12, seed=s, directed=True) for s in range(20)]


@pytest.mark.parametrize("D", REDES_ALEATORIAS)
def test_eficiencia_contra_floyd_warshall(D):
    distancias = nx.floyd_warshall(D)
    esperado = sum(1 / distancias[i][j] for i, j in itertools.permutations(D, 2)
                   if math.isfinite(distancias[i][j]))
    n = D.number_of_nodes()
    assert metricas.eficiencia_global_dirigida(D, n) == pytest.approx(esperado / (n * (n - 1)))


@pytest.mark.parametrize("D", REDES_ALEATORIAS[:5])
def test_eficiencia_simetrica_igual_a_da_networkx(D):
    U = D.to_undirected()
    simetrica = U.to_directed()
    n = U.number_of_nodes()
    assert metricas.eficiencia_global_dirigida(simetrica, n) == pytest.approx(nx.global_efficiency(U))


@pytest.mark.parametrize("D", REDES_ALEATORIAS)
def test_bow_tie_pela_definicao(D):
    partes = metricas.componentes_bow_tie(D)
    nucleo = partes["nucleo"]
    c = next(iter(nucleo))
    for v in D:
        if v in nucleo:
            continue
        chega, alcancado = nx.has_path(D, v, c), nx.has_path(D, c, v)
        esperado = "entrada" if chega else "saida" if alcancado else "restante"
        assert v in partes[esperado]
    assert sum(map(len, partes.values())) == D.number_of_nodes()


@pytest.mark.parametrize("D", REDES_ALEATORIAS)
def test_alcance_e_caminhos_contra_networkx(D):
    fontes = [0, 3]
    esperado = set(fontes) | nx.descendants(D, 0) | nx.descendants(D, 3)
    assert metricas.alcancaveis(D, fontes) == esperado

    nucleo = metricas.maior_componente_forte(D)
    if len(nucleo) >= 2:
        H = D.subgraph(nucleo)
        caminho, diametro = metricas.caminho_medio_e_diametro(D)
        assert caminho == pytest.approx(nx.average_shortest_path_length(H))
        assert diametro == nx.diameter(H)
