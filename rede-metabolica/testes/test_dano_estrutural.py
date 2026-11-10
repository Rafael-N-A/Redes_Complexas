"""Dano estrutural de uma remoção (metricas.py) e do nocaute de genes (nocaute.py).

Redes de brinquedo com o dano feito à mão (uma cadeia e um ciclo), a eficiência
rápida (scipy) contra a de Python puro e a harmônica contra a da NetworkX, em
redes sorteadas e na rede real.
"""
from __future__ import annotations

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import contrato
from experimentos.biblioteca import grafo, metricas, nocaute


def tabela(*reacoes) -> pd.DataFrame:
    """Cada item: (codigo, substratos, produtos, genes) ou (..., reversivel)."""
    linhas = []
    for item in reacoes:
        codigo, substratos, produtos, genes, *resto = item
        linhas.append({"reacao": codigo, "substratos": list(substratos), "produtos": list(produtos),
                       "reversivel": bool(resto and resto[0]), "genes": list(genes), "vias": ["eco00000"],
                       "n_ocorrencias": 1, "conflito_reversibilidade": False, "orientacao_invertida": False,
                       "lados_variam": False})
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


# Ciclo A -> B -> C -> A com uma cauda C -> D -> E (5 nós):
#   núcleo (maior componente forte) = {A, B, C}; maior componente fraco = os 5.
#   R3 (C -> A) tem duas isoenzimas, g3 e g4: nenhuma sozinha derruba a reação.
CICLO = tabela(("R1", ["A"], ["B"], ["g1"]), ("R2", ["B"], ["C"], ["g2"]),
               ("R3", ["C"], ["A"], ["g3", "g4"]), ("R4", ["C"], ["D"], ["g5"]),
               ("R5", ["D"], ["E"], ["g6"]))


def grafo_sorteado(semente: int, n: int = 30, p: float = 0.08, dirigido: bool = True) -> nx.Graph:
    G = nx.gnp_random_graph(n, p, directed=dirigido, seed=semente)
    return nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})


# ---------------------------------------------------------------------------
# Eficiência rápida, distâncias e harmônica
# ---------------------------------------------------------------------------
def test_eficiencia_rapida_igual_a_de_python_puro():
    for semente in range(15):
        G = grafo_sorteado(semente, n=35, p=0.06)
        for n_original in (35, 50):
            lenta = metricas.eficiencia_global_dirigida(G, n_original)
            assert math.isclose(metricas.eficiencia_global_dirigida_rapida(G, n_original), lenta, rel_tol=1e-9)
        H = G.copy()
        H.remove_nodes_from(list(H)[:10])          # rede partida, normalizada pelo tamanho original
        assert math.isclose(metricas.eficiencia_global_dirigida_rapida(H, 35),
                            metricas.eficiencia_global_dirigida(H, 35), rel_tol=1e-9)


def test_eficiencia_rapida_casos_de_borda():
    vazio = nx.DiGraph()
    vazio.add_nodes_from(["a", "b"])
    assert metricas.eficiencia_global_dirigida_rapida(vazio, 2) == 0.0       # sem arestas
    assert metricas.eficiencia_global_dirigida_rapida(nx.DiGraph([("a", "b")]), 1) == 0.0
    # a -> b -> c: pares (a,b)=1, (b,c)=1, (a,c)=1/2 -> 2,5 / (3*2)
    assert math.isclose(metricas.eficiencia_global_dirigida_rapida(nx.DiGraph([("a", "b"), ("b", "c")]), 3),
                        2.5 / 6)


def test_eficiencia_rapida_na_rede_real():
    reacoes = grafo.carregar_tabelas("eco")["reacoes"]
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=True))
    n = G.number_of_nodes()
    rapida = metricas.eficiencia_global_dirigida_rapida(G, n)
    assert math.isclose(rapida, metricas.eficiencia_global_dirigida(G, n), rel_tol=1e-9)
    assert round(rapida, 4) == 0.0393                                         # valor de referência


def test_matriz_distancias():
    G = nx.DiGraph([("a", "b"), ("b", "c")])
    D = metricas.matriz_distancias(G, ["a", "b", "c"])
    assert D[0, 2] == 2 and D[2, 0] == np.inf and D[1, 1] == 0
    U = nx.Graph([("a", "b"), ("b", "c")])
    assert metricas.matriz_distancias(U, ["a", "b", "c"])[2, 0] == 2          # não dirigido: volta também


def _harmonica_exata(G: nx.Graph) -> dict:
    valores = {no: Fraction(0) for no in G}
    for _, distancias in nx.all_pairs_shortest_path_length(G):
        for destino, d in distancias.items():
            if d > 0:
                valores[destino] += Fraction(1, d)
    return valores


@pytest.mark.parametrize("dirigido", [True, False])
def test_harmonica_igual_a_da_networkx(dirigido):
    for semente in range(10):
        G = grafo_sorteado(semente, n=30, p=0.08, dirigido=dirigido)
        nossa = metricas.harmonica(G)
        da_networkx = nx.harmonic_centrality(G)
        exata = _harmonica_exata(G)
        assert list(nossa.index) == sorted(G)
        for no in G:
            assert math.isclose(nossa[no], da_networkx[no], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(nossa[no], float(exata[no]), rel_tol=1e-12, abs_tol=1e-12)


def test_harmonica_feita_a_mao_e_empates_exatos():
    """a -> b -> c: b recebe de a (1), c de b (1) e de a (1/2). Na estrela de mão dupla, as folhas empatam bit a bit."""
    assert metricas.harmonica(nx.DiGraph([("a", "b"), ("b", "c")])).to_dict() == {"a": 0.0, "b": 1.0, "c": 1.5}
    estrela = nx.DiGraph()
    for folha in ("f1", "f2", "f3", "f4", "f5"):
        estrela.add_edges_from([("c", folha), (folha, "c")])
    valores = metricas.harmonica(estrela)
    assert valores["c"] == 5.0 and len({valores[f] for f in ("f1", "f2", "f3", "f4", "f5")}) == 1
    assert metricas.harmonica(nx.DiGraph()).empty


# ---------------------------------------------------------------------------
# Dano de uma remoção (metabólito)
# ---------------------------------------------------------------------------
def test_queda_estrutural():
    antes = {"maior_forte": 10, "maior_fraco": 20, "eficiencia": 0.5}
    depois = {"maior_forte": 4, "maior_fraco": 20, "eficiencia": 0.4}
    queda = metricas.queda_estrutural(antes, depois, 40)
    assert queda == pytest.approx({"dano_forte": 6, "dano_fraco": 0, "dano_forte_fracao": 0.15,
                                   "dano_fraco_fracao": 0.0, "dano_eficiencia": 0.2})
    sem_caminho = metricas.queda_estrutural({**antes, "eficiencia": 0.0}, {**depois, "eficiencia": 0.0}, 40)
    assert math.isnan(sem_caminho["dano_eficiencia"])


def test_dano_de_remover_cada_metabolito_do_ciclo_com_cauda():
    """Ciclo A -> B -> C -> A e cauda C -> D -> E.

    Tirar A ou B: o núcleo (3) vira nós soltos (1) -> queda 2; o fraco fica com 4 -> queda 1.
    Tirar C: núcleo 3 -> 1 e o fraco se parte em {A, B} e {D, E} -> 5 - 2 = 3.
    Tirar D: núcleo intacto; fraco {A, B, C} -> queda 2. Tirar E: só o fraco cai 1.
    """
    G = grafo.grafo_simples(grafo.grafo_metabolitos(CICLO))
    dano = metricas.dano_remocao_individual(G).set_index("metabolito")
    assert list(dano.index) == ["A", "B", "C", "D", "E"]
    assert dano["dano_forte"].to_dict() == {"A": 2, "B": 2, "C": 2, "D": 0, "E": 0}
    assert dano["dano_fraco"].to_dict() == {"A": 1, "B": 1, "C": 3, "D": 2, "E": 1}
    assert dano["dano_fraco_fracao"]["C"] == pytest.approx(3 / 5)
    e0 = metricas.eficiencia_global_dirigida(G, 5)
    for no in G:
        H = G.copy()
        H.remove_node(no)
        assert dano.loc[no, "dano_eficiencia"] == pytest.approx((e0 - metricas.eficiencia_global_dirigida(H, 5)) / e0)


# ---------------------------------------------------------------------------
# Nocaute estrutural de genes
# ---------------------------------------------------------------------------
def test_dano_estrutural_de_cada_gene_feito_a_mao():
    """g1 corta o ciclo (núcleo 3 -> 1, o fraco fica inteiro); g3 e g4 são isoenzimas (dano 0);
    g5 separa a cauda (fraco 5 -> 3); g6 solta E, que continua contando em N = 5."""
    M = grafo.grafo_metabolitos(CICLO)
    dano = nocaute.dano_estrutural_todos_genes(M, CICLO).set_index("gene")
    assert list(dano.index) == ["g1", "g2", "g3", "g4", "g5", "g6"]
    assert dano["n_reacoes_perdidas"].to_dict() == {"g1": 1, "g2": 1, "g3": 0, "g4": 0, "g5": 1, "g6": 1}
    assert dano["dano_forte"].to_dict() == {"g1": 2, "g2": 2, "g3": 0, "g4": 0, "g5": 0, "g6": 0}
    assert dano["dano_fraco"].to_dict() == {"g1": 0, "g2": 0, "g3": 0, "g4": 0, "g5": 2, "g6": 1}
    assert dano.loc["g6", "dano_fraco_fracao"] == pytest.approx(1 / 5)      # E isolado ainda conta em N
    assert (dano.loc[["g3", "g4"], ["arestas_perdidas", "dano_eficiencia"]] == 0).all().all()
    # g6: sem D -> E, somem os pares (A,E)=1/4, (B,E)=1/3, (C,E)=1/2 e (D,E)=1 de uma soma de 8,41667.
    soma = 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 + 1 / 2 + 1 / 2 + 1 / 3 + 1 + 1 + 1 / 2 + 1 / 2 + 1
    assert dano.loc["g6", "dano_eficiencia"] == pytest.approx((1 / 4 + 1 / 3 + 1 / 2 + 1) / soma)


def test_nocaute_estrutural_remove_arestas_e_nao_nos():
    M = grafo.grafo_metabolitos(CICLO)
    H = nocaute.grafo_sem_reacoes(M, {"R4", "R5"})
    assert set(H.nodes) == set(M.nodes)                                       # D e E ficam, isolados
    assert set(H.edges) == {("A", "B"), ("B", "C"), ("C", "A")}


def test_reversivel_perde_os_dois_sentidos_e_paralela_segura_a_aresta():
    """R1: A <-> B (g1, reversível); R2 e R3: B -> C por genes diferentes (g2, g3)."""
    rede = tabela(("R1", ["A"], ["B"], ["g1"], True), ("R2", ["B"], ["C"], ["g2"]), ("R3", ["B"], ["C"], ["g3"]))
    dano = nocaute.dano_estrutural_todos_genes(grafo.grafo_metabolitos(rede), rede).set_index("gene")
    assert dano.loc["g1", "arestas_perdidas"] == 2 and dano.loc["g1", "dano_forte"] == 1   # o par A <-> B some
    assert dano.loc["g2", "n_reacoes_perdidas"] == 1 and dano.loc["g2", "arestas_perdidas"] == 0
    assert dano.loc["g2", ["dano_forte", "dano_fraco", "dano_eficiencia"]].abs().sum() == 0  # R3 segura B -> C


def test_nocaute_estrutural_igual_a_recalcular_do_zero():
    """Em redes sorteadas, cada linha é a queda medida à parte com a eficiência de Python puro."""
    rng = np.random.default_rng(7)
    metabolitos = [f"M{i:02d}" for i in range(25)]
    reacoes = []
    for i in range(40):
        a, b = rng.choice(metabolitos, size=2, replace=False)
        genes = [f"g{j}" for j in rng.choice(12, size=rng.integers(0, 3), replace=False)]
        reacoes.append((f"R{i:03d}", [a], [b], genes, bool(rng.integers(2))))
    rede = tabela(*reacoes)
    M = grafo.grafo_metabolitos(rede)
    G = grafo.grafo_simples(M)
    n = G.number_of_nodes()
    dano = nocaute.dano_estrutural_todos_genes(M, rede).set_index("gene")
    e0 = metricas.eficiencia_global_dirigida(G, n)
    for gene in dano.index:
        perdidas = nocaute.reacoes_perdidas([gene], rede)
        H = nx.DiGraph()                     # o par fica se sobrar pelo menos uma reação entre os dois
        H.add_nodes_from(G.nodes)
        H.add_edges_from((u, v) for u, v, k in M.edges(keys=True) if k not in perdidas)
        forte = max(len(c) for c in nx.strongly_connected_components(H))
        fraco = max(len(c) for c in nx.weakly_connected_components(H))
        assert dano.loc[gene, "dano_forte"] == len(metricas.maior_componente_forte(G)) - forte
        assert dano.loc[gene, "dano_fraco"] == max(map(len, nx.weakly_connected_components(G))) - fraco
        assert dano.loc[gene, "dano_eficiencia"] == pytest.approx((e0 - metricas.eficiencia_global_dirigida(H, n)) / e0)
