"""Testes das estratégias de remoção e das curvas de robustez (robustez.py).

Quatro tipos de teste:
  1. ordens de remoção em grafos pequenos com a resposta feita à mão;
  2. uma curva inteira feita à mão (estrela com um par solto);
  3. resultados analíticos do índice R (grafo completo e estrela);
  4. comparação com uma implementação ingênua, que refaz cada passo do zero
     com funções da NetworkX em subgrafos, em grafos dirigidos sorteados.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from experimentos.biblioteca import metricas, robustez


def estrela_com_par_solto() -> nx.DiGraph:
    """h <-> x1, h <-> x2, h -> x3 e o par solto x4 -> x5 (6 nós).

    Graus totais: h=5, x1=2, x2=2, x3=1, x4=1, x5=1.
    """
    G = nx.DiGraph()
    G.add_edges_from([("h", "x1"), ("x1", "h"), ("h", "x2"), ("x2", "h"), ("h", "x3"), ("x4", "x5")])
    return G


def caminho_dirigido() -> nx.DiGraph:
    """a -> b -> c -> d -> e. Betweenness (pares ordenados): b=3, c=4, d=3, a=e=0."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])


# ---------------------------------------------------------------------------
# 1. Ordens de remoção
# ---------------------------------------------------------------------------
def test_ordem_por_grau_com_desempate_pelo_codigo():
    assert robustez.ordem_inicial(estrela_com_par_solto(), "grau") == ["h", "x1", "x2", "x3", "x4", "x5"]


def test_ordem_por_betweenness():
    assert robustez.ordem_inicial(caminho_dirigido(), "betweenness") == ["c", "b", "d", "a", "e"]


def test_ordem_aleatoria_e_permutacao_reprodutivel():
    G = nx.gnp_random_graph(30, 0.1, directed=True, seed=1)
    G = nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})
    ordem = robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 0]))
    assert sorted(ordem) == sorted(G.nodes)
    assert ordem == robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 0]))
    assert ordem != robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 1]))


def test_proximo_adaptativo():
    G = estrela_com_par_solto()
    assert robustez.proximo_adaptativo(G, "grau_adaptativo") == "h"
    G.remove_node("h")
    # Sem o hub, só x4 e x5 têm grau 1; empate desfeito pelo código.
    assert robustez.proximo_adaptativo(G, "grau_adaptativo") == "x4"
    assert robustez.proximo_adaptativo(caminho_dirigido(), "betweenness_adaptativo") == "c"


def test_erros_de_uso():
    G = caminho_dirigido()
    with pytest.raises(ValueError):
        robustez.ordem_inicial(G, "aleatoria")  # sem rng
    with pytest.raises(ValueError):
        robustez.ordem_inicial(G, "grau_adaptativo")  # não é estática
    with pytest.raises(ValueError):
        robustez.simular_remocao(G, "pagerank")
    with pytest.raises(ValueError):
        robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=0)


# ---------------------------------------------------------------------------
# 2. Curva feita à mão
# ---------------------------------------------------------------------------
def test_curva_grau_feita_a_mao():
    """Ordem h, x1, x2, x3, x4, x5.

    Passo 0: fraco {h,x1,x2,x3}=4, forte {h,x1,x2}=3, alcance a partir de h = 4.
    Sem h: sobra só o par x4 -> x5 (fraco 2, forte 1) até x4 sair.
    """
    curva = robustez.simular_remocao(estrela_com_par_solto(), "grau", nutrientes=["h"],
                                     medir_eficiencia=False)
    assert list(curva["n_removidos"]) == list(range(7))
    assert np.allclose(curva["fracao_removida"], np.arange(7) / 6)
    assert np.allclose(curva["maior_fraco"] * 6, [4, 2, 2, 2, 2, 1, 0])
    assert np.allclose(curva["maior_forte"] * 6, [3, 1, 1, 1, 1, 1, 0])
    assert np.allclose(curva["alcance"] * 6, [4, 0, 0, 0, 0, 0, 0])
    assert curva["eficiencia"].isna().all()


def test_curva_grau_adaptativo_feita_a_mao():
    """Ordem h, x4 (o par solto vira o de maior grau), x1, x2, x3, x5."""
    curva = robustez.simular_remocao(estrela_com_par_solto(), "grau_adaptativo", medir_eficiencia=False)
    assert np.allclose(curva["maior_fraco"] * 6, [4, 2, 1, 1, 1, 1, 0])
    assert curva["alcance"].isna().all()  # sem nutrientes, o alcance não é medido


def test_indices_da_curva_feita_a_mao():
    """R_fraco = (2+2+2+2+1+0)/6/6 = 1/4; R_forte = (1+1+1+1+1+0)/6/6 = 5/36."""
    _, indices = robustez.comparar_estrategias(estrela_com_par_solto(), ["grau"], medir_eficiencia=False)
    linha = indices.iloc[0]
    assert linha["estrategia"] == "grau" and linha["n_rep"] == 1
    assert math.isclose(linha["R_fraco"], 1 / 4) and math.isclose(linha["R_forte"], 5 / 36)
    assert math.isnan(linha["R_fraco_dp"])


# ---------------------------------------------------------------------------
# 3. Resultados analíticos do índice R
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("estrategia", ["aleatoria", "grau", "grau_adaptativo", "betweenness"])
def test_r_do_grafo_completo(estrategia):
    """No grafo completo, qualquer ordem deixa os N-Q restantes conectados:
    s(Q) = (N-Q)/N e R = (N-1)/(2N), o máximo possível."""
    n = 12
    G = nx.complete_graph([f"n{i:02d}" for i in range(n)], create_using=nx.DiGraph)
    _, indices = robustez.comparar_estrategias(G, [estrategia], n_rep=3, medir_eficiencia=False)
    assert math.isclose(indices["R_fraco"][0], (n - 1) / (2 * n))
    assert math.isclose(indices["R_forte"][0], (n - 1) / (2 * n))


def test_r_da_estrela_sob_ataque():
    """Estrela de mão dupla: o ataque tira o centro primeiro e sobram N-1 nós soltos.
    s(Q) = 1/N para Q = 1..N-1 e 0 no fim, então R = (N-1)/N²."""
    n = 10
    G = nx.DiGraph()
    for i in range(1, n):
        G.add_edges_from([("c", f"f{i}"), (f"f{i}", "c")])
    _, indices = robustez.comparar_estrategias(G, ["grau", "betweenness"], medir_eficiencia=False)
    assert np.allclose(indices["R_fraco"], (n - 1) / n**2)


# ---------------------------------------------------------------------------
# Eficiência a cada 5%
# ---------------------------------------------------------------------------
def test_marcos_da_eficiencia():
    assert robustez.marcos_eficiencia(6, 0.25) == {0, 2, 3, 5, 6}
    assert robustez.marcos_eficiencia(20, 0.05) == set(range(21))
    marcos = robustez.marcos_eficiencia(1111, 0.05)
    assert len(marcos) == 21 and {0, 56, 556, 1111} <= marcos  # teto(55,55) = 56, teto(555,5) = 556


def test_eficiencia_so_nos_marcos():
    G = nx.gnp_random_graph(40, 0.08, directed=True, seed=3)
    G = nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})
    curva = robustez.simular_remocao(G, "grau", passo_eficiencia=0.1)
    medidos = set(curva.loc[curva["eficiencia"].notna(), "n_removidos"])
    assert medidos == robustez.marcos_eficiencia(40, 0.1)
    assert math.isclose(curva["eficiencia"][0], metricas.eficiencia_global_dirigida(G, 40))
    assert curva["eficiencia"].iloc[-1] == 0.0


def test_nao_altera_o_grafo():
    G = estrela_com_par_solto()
    antes = (set(G.nodes), set(G.edges))
    robustez.comparar_estrategias(G, n_rep=2, nutrientes=["h"])
    assert (set(G.nodes), set(G.edges)) == antes


# ---------------------------------------------------------------------------
# Repetições aleatórias
# ---------------------------------------------------------------------------
def grafo_sorteado(semente: int, n: int = 25, p: float = 0.08) -> nx.DiGraph:
    G = nx.gnp_random_graph(n, p, directed=True, seed=semente)
    return nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})


def test_curva_aleatoria_com_uma_repeticao_e_a_propria_simulacao():
    G = grafo_sorteado(5)
    curva = robustez.curva_aleatoria(G, n_rep=1, semente=7, nutrientes=["n00"])
    simulada = robustez.simular_remocao(G, "aleatoria", rng=np.random.default_rng([7, 0]), nutrientes=["n00"])
    pd.testing.assert_frame_equal(curva[list(robustez.COLUNAS)], simulada, check_dtype=False)
    assert curva[list(robustez.COLUNAS_DP)].isna().all().all()


def test_curva_aleatoria_e_a_media_das_repeticoes():
    G = grafo_sorteado(6)
    curva = robustez.curva_aleatoria(G, n_rep=4, semente=3, nutrientes=["n01"], medir_eficiencia=False)
    reps = [robustez.simular_remocao(G, "aleatoria", rng=np.random.default_rng([3, i]),
                                     nutrientes=["n01"], medir_eficiencia=False) for i in range(4)]
    for coluna in ("maior_fraco", "maior_forte", "alcance"):
        valores = np.vstack([r[coluna] for r in reps])
        assert np.allclose(curva[coluna], valores.mean(axis=0))
        assert np.allclose(curva[f"{coluna}_dp"], valores.std(axis=0, ddof=1))


def test_comparar_estrategias_formato_e_reprodutibilidade():
    G = grafo_sorteado(8)
    curvas, indices = robustez.comparar_estrategias(G, n_rep=5, nutrientes=["n02"])
    assert list(curvas.columns) == ["estrategia", *robustez.COLUNAS, *robustez.COLUNAS_DP]
    assert list(indices.columns) == list(robustez.COLUNAS_INDICES)
    assert list(indices["estrategia"]) == list(robustez.ESTATICAS[:1]) + [
        "grau", "grau_adaptativo", "betweenness", "betweenness_adaptativo"]
    assert (curvas.groupby("estrategia").size() == G.number_of_nodes() + 1).all()
    for linha in indices.to_dict("records"):
        curva = curvas[curvas["estrategia"] == linha["estrategia"]]
        depois = curva[curva["n_removidos"] > 0]
        # Na aleatória, a média dos R é o R da curva média (o índice é linear).
        assert math.isclose(linha["R_fraco"], depois["maior_fraco"].mean())
        assert math.isclose(linha["R_forte"], depois["maior_forte"].mean())
    de_novo = robustez.comparar_estrategias(G, n_rep=5, nutrientes=["n02"])
    pd.testing.assert_frame_equal(curvas, de_novo[0])
    pd.testing.assert_frame_equal(indices, de_novo[1])


# ---------------------------------------------------------------------------
# 4. Implementação ingênua
# ---------------------------------------------------------------------------
def _ranking_ingenuo(H: nx.DiGraph, criterio: str) -> list:
    if criterio == "grau":
        valores = {no: H.in_degree(no) + H.out_degree(no) for no in H}
    else:
        valores = {no: round(v, 12) for no, v in nx.betweenness_centrality(H).items()}
    return sorted(H, key=lambda no: (-valores[no], no))


def _medidas_ingenuas(G: nx.DiGraph, restantes: set, n: int, nutrientes: list, eficiencia: bool) -> dict:
    H = G.subgraph(restantes)
    fracos = [len(c) for c in nx.weakly_connected_components(H)]
    fortes = [len(c) for c in nx.strongly_connected_components(H)]
    alcancados = set()
    for f in nutrientes:
        if f in H:
            alcancados |= nx.descendants(H, f) | {f}
    medidas = {"maior_fraco": max(fracos, default=0) / n, "maior_forte": max(fortes, default=0) / n,
               "alcance": len(alcancados) / n, "eficiencia": math.nan}
    if eficiencia:
        soma = sum(1 / d for _, dist in nx.all_pairs_shortest_path_length(H) for d in dist.values() if d > 0)
        medidas["eficiencia"] = soma / (n * (n - 1))
    return medidas


def simulacao_ingenua(G: nx.DiGraph, estrategia: str, nutrientes: list, passo_eficiencia: float,
                      passo_betweenness: int, rng=None, ate: int | None = None) -> pd.DataFrame:
    """Refaz cada passo do zero a partir de G, sem copiar nem remover nós de um grafo.

    ate: para depois de tantas remoções (o verificar_cascata_robustez usa na
    rede real, onde a versão adaptativa da betweenness feita assim é lenta).
    """
    n = G.number_of_nodes()
    restantes = set(G)
    if estrategia == "aleatoria":
        nos = sorted(G)
        ordem = [nos[i] for i in rng.permutation(n)]
    elif estrategia in ("grau", "betweenness"):
        ordem = _ranking_ingenuo(G, estrategia)
    multiplos = range(math.floor(1 / passo_eficiencia + 1e-9) + 1)
    marcos = {min(n, math.ceil(round(m * passo_eficiencia * n, 9))) for m in multiplos}
    linhas = [{"n_removidos": 0, "fracao_removida": 0.0,
               **_medidas_ingenuas(G, restantes, n, nutrientes, 0 in marcos)}]
    fila = []
    for k in range(1, (n if ate is None else min(n, ate)) + 1):
        H = G.subgraph(restantes)
        if estrategia == "grau_adaptativo":
            no = _ranking_ingenuo(H, "grau")[0]
        elif estrategia == "betweenness_adaptativo":
            if not fila:
                fila = _ranking_ingenuo(H, "betweenness")[:passo_betweenness]
            no = fila.pop(0)
        else:
            no = ordem[k - 1]
        restantes.discard(no)
        linhas.append({"n_removidos": k, "fracao_removida": k / n,
                       **_medidas_ingenuas(G, restantes, n, nutrientes, k in marcos)})
    return pd.DataFrame(linhas, columns=list(robustez.COLUNAS))


@pytest.mark.parametrize("estrategia", ["aleatoria", "grau", "grau_adaptativo",
                                        "betweenness", "betweenness_adaptativo"])
def test_igual_a_implementacao_ingenua(estrategia):
    for semente in range(15):
        G = grafo_sorteado(100 + semente, n=30, p=0.07)
        nutrientes = ["n00", "n05"]
        rng = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        rng_ingenuo = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        curva = robustez.simular_remocao(G, estrategia, rng=rng, nutrientes=nutrientes,
                                         passo_eficiencia=0.1, passo_betweenness=3)
        esperada = simulacao_ingenua(G, estrategia, nutrientes, 0.1, 3, rng_ingenuo)
        pd.testing.assert_frame_equal(curva, esperada, check_dtype=False, rtol=1e-12,
                                      obj=f"{estrategia}, semente={semente}")


def test_betweenness_adaptativo_nos_extremos():
    """Com passo = N, nunca recalcula (= estática); com passo = 1, recalcula a cada remoção."""
    G = grafo_sorteado(11, n=30, p=0.07)
    n = G.number_of_nodes()
    estatica = robustez.simular_remocao(G, "betweenness", medir_eficiencia=False)
    sem_recalcular = robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=n,
                                              medir_eficiencia=False)
    pd.testing.assert_frame_equal(estatica, sem_recalcular)
    a_cada_uma = robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=1,
                                          medir_eficiencia=False)
    esperada = simulacao_ingenua(G, "betweenness_adaptativo", [], 1.0, 1)
    pd.testing.assert_frame_equal(a_cada_uma[["n_removidos", "maior_fraco", "maior_forte"]],
                                  esperada[["n_removidos", "maior_fraco", "maior_forte"]], check_dtype=False)
