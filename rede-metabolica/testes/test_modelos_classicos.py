"""Modelos clássicos (modelos_classicos.py) e a seção deles no Exp 4.

A percolação reversa com union-find é conferida contra a simulação ingênua (remove
nó a nó e recalcula os componentes com a NetworkX); os geradores, pelo que
prometem (N exato, M exato ou perto do alvo, mesma semente = mesma rede); as
descritivas e os índices, em redes pequenas com a resposta feita à mão.
"""
from __future__ import annotations

import math
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
from experimentos import exp4_comparacao as exp4
from experimentos.biblioteca import grafo, metricas, modelos_classicos as mc, robustez


def maior_componente_ingenuo(G: nx.Graph, ordem: list) -> np.ndarray:
    """Remove ordem[:k] de uma cópia e recalcula o maior componente a cada passo."""
    n = G.number_of_nodes()
    H = G.copy()
    s = [max((len(c) for c in nx.connected_components(H)), default=0)]
    for no in ordem:
        H.remove_node(no)
        s.append(max((len(c) for c in nx.connected_components(H)), default=0))
    return np.array(s) / n


def rede_sorteada(semente: int, n: int = 40, p: float = 0.06) -> nx.Graph:
    G = nx.gnp_random_graph(n, p, seed=semente)
    return nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})


@pytest.fixture(scope="module")
def ecoli():
    reacoes = grafo.carregar_tabelas("eco")["reacoes"]
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=True))
    return mc.rede_nao_dirigida(G)


# ---------------------------------------------------------------------------
# Percolação reversa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("estrategia", ["aleatoria", "grau", "betweenness", "pagerank",
                                        "grau_adaptativo"])
def test_union_find_igual_a_simulacao_ingenua(estrategia):
    for semente in range(10):
        G = rede_sorteada(semente)               # redes partidas, com nós isolados
        rng = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        ordem = mc.ordem(G, estrategia, rng)
        assert np.allclose(mc.maior_componente_por_ordem(G, ordem), maior_componente_ingenuo(G, ordem))


def test_grau_adaptativo_igual_a_recalcular_a_cada_passo():
    """A sequência do grau adaptativo é a do maior grau na rede que sobrou (empates pelo código)."""
    for semente in range(5):
        G = rede_sorteada(100 + semente, n=30, p=0.1)
        H, esperada = G.copy(), []
        while len(H):
            no = min(H, key=lambda v: (-H.degree(v), v))
            esperada.append(no)
            H.remove_node(no)
        assert mc.ordem(G, "grau_adaptativo") == esperada


def test_union_find_casos_feitos_a_mao():
    """Caminho a - b - c - d - e: tirar c primeiro parte em dois pares."""
    G = nx.path_graph(["a", "b", "c", "d", "e"])
    s = mc.maior_componente_por_ordem(G, ["c", "a", "b", "d", "e"])
    assert np.allclose(s * 5, [5, 2, 2, 2, 1, 0])
    assert mc.indice_R(s) == pytest.approx((2 + 2 + 2 + 1 + 0) / 5 / 5)
    assert mc.fracao_ate_metade(s) == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        mc.maior_componente_por_ordem(G, ["a", "b"])        # ordem incompleta


def test_fracao_ate_metade():
    assert mc.fracao_ate_metade(np.array([0.8, 0.6, 0.41, 0.4, 0.0])) == pytest.approx(3 / 4)
    assert math.isnan(mc.fracao_ate_metade(np.array([0.0, 0.0])))


def test_robustez_rede_indices_e_reprodutibilidade():
    G = rede_sorteada(3, n=50, p=0.05)
    curvas, indices = mc.robustez_rede(G, ("aleatoria", "grau"), n_rep=4, semente=9)
    assert list(indices.columns) == list(mc.COLUNAS_INDICES)
    assert list(curvas.columns) == list(mc.COLUNAS_CURVA)
    por = indices.set_index("estrategia")
    s0 = len(mc.maior_componente(G)) / 50
    assert por.loc["grau", "S0"] == pytest.approx(s0)
    assert por.loc["aleatoria", "R_norm"] == pytest.approx(por.loc["aleatoria", "R"] / s0)
    reps = [mc.maior_componente_por_ordem(G, mc.ordem(G, "aleatoria", np.random.default_rng([9, i])))
            for i in range(4)]
    assert por.loc["aleatoria", "R"] == pytest.approx(np.mean([mc.indice_R(s) for s in reps]))
    assert por.loc["aleatoria", "R_dp"] == pytest.approx(np.std([mc.indice_R(s) for s in reps], ddof=1))
    aleatoria = curvas[curvas["estrategia"] == "aleatoria"]
    assert np.allclose(aleatoria["maior_componente"], np.vstack(reps).mean(axis=0))
    assert aleatoria["maior_componente_rel"].iloc[0] == pytest.approx(1.0)
    de_novo = mc.robustez_rede(G, ("aleatoria", "grau"), n_rep=4, semente=9)
    pd.testing.assert_frame_equal(indices, de_novo[1])


def test_aleatoria_da_ecoli_e_o_r_fraco_do_exp1(ecoli):
    """O maior componente não dirigido é o maior componente fraco, e as ordens sorteadas são as mesmas."""
    reacoes = grafo.carregar_tabelas("eco")["reacoes"]
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=True))
    _, exp1 = robustez.comparar_estrategias(G, ["aleatoria"], n_rep=3, medir_eficiencia=False)
    _, classico = mc.robustez_rede(ecoli, ("aleatoria",), n_rep=3, semente=config.SEMENTE)
    assert classico["R"].iloc[0] == pytest.approx(exp1["R_fraco"].iloc[0])


# ---------------------------------------------------------------------------
# Geradores
# ---------------------------------------------------------------------------
def test_rede_nao_dirigida_da_ecoli(ecoli):
    """Os números da Seção 1 do plano: N = 1.111, M = 1.277, grau máximo 27, 79,7% no maior componente."""
    assert ecoli.number_of_nodes() == 1111 and ecoli.number_of_edges() == 1277
    assert max(d for _, d in ecoli.degree()) == 27
    assert len(mc.maior_componente(ecoli)) / 1111 == pytest.approx(0.797, abs=5e-4)
    assert list(ecoli.nodes) == sorted(ecoli.nodes)


def test_parametros():
    assert mc.parametros(1111, 1277) == pytest.approx({"p_ba_misto": 2 - 1277 / 1111,
                                                       "p_newman_watts": 1277 / 1111 - 1})
    assert mc.parametros(168, 146) == {"p_ba_misto": 1.0, "p_newman_watts": 0.0}   # M < N: presos em [0, 1]


@pytest.mark.parametrize("modelo", config.MODELOS_CLASSICOS)
def test_geradores_n_exato_m_perto_do_alvo_e_reprodutiveis(ecoli, modelo):
    n, m = ecoli.number_of_nodes(), ecoli.number_of_edges()
    redes = list(mc.gerar_modelos(ecoli, modelo, n_redes=4, semente=7))
    assert all(R.number_of_nodes() == n and nx.number_of_selfloops(R) == 0 for R in redes)
    arestas = np.array([R.number_of_edges() for R in redes])
    if modelo in ("erdos_renyi", "configuracao"):
        assert (arestas == m).all()                                      # exato
    elif modelo == "ba_m1":
        assert (arestas == n - 1).all() and all(nx.is_tree(R) for R in redes)   # uma árvore, de propósito
    elif modelo == "watts_strogatz":
        assert (arestas == n * config.K_WATTS_STROGATZ // 2).all()        # o dobro, de propósito
        assert all(nx.is_connected(R) for R in redes)
    else:
        assert abs(arestas.mean() - m) / m < 0.02                        # BA misto e Newman–Watts: em média
    mesma = mc.gerar(modelo, ecoli, 7 * 1000)
    assert set(mesma.edges) == set(redes[0].edges)                       # mesma semente, mesma rede
    assert set(redes[0].edges) != set(redes[1].edges)                    # outra semente, outra rede
    assert mesma.graph["parametros"]


def test_configuracao_preserva_os_graus(ecoli):
    R = mc.configuracao(ecoli, semente=1)
    assert dict(R.degree()) == dict(ecoli.degree())
    assert len(set(R.edges) & set(ecoli.edges)) < 0.2 * ecoli.number_of_edges()  # embaralhou de fato


def test_modelo_desconhecido(ecoli):
    with pytest.raises(ValueError, match="modelo clássico desconhecido"):
        mc.gerar("estrela", ecoli, 0)


# ---------------------------------------------------------------------------
# Descritivas
# ---------------------------------------------------------------------------
def test_descritiva_feita_a_mao():
    """Triângulo a-b-c com cauda c-d e o par solto x-y (6 nós, 5 arestas)."""
    G = nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("x", "y")])
    d = mc.descritiva(G)
    assert d["nos"] == 6 and d["arestas"] == 5 and d["grau_max"] == 3
    assert d["grau_medio"] == pytest.approx(10 / 6)
    assert d["clustering_medio"] == pytest.approx((1 + 1 + 1 / 3) / 6)   # d, x e y contam 0
    assert d["maior_componente"] == pytest.approx(4 / 6)
    # no maior componente {a, b, c, d}: d fica a 2 de a e de b; o resto a 1 -> (8*1 + 4*2) / 12
    assert d["caminho_medio"] == pytest.approx(16 / 12) and d["diametro"] == 2
    assert d["caminho_medio"] == pytest.approx(nx.average_shortest_path_length(G.subgraph({"a", "b", "c", "d"})))


def test_descritiva_da_ecoli(ecoli):
    """Valores de referência do protótipo (Seção 8)."""
    d = mc.descritiva(ecoli)
    assert d["clustering_medio"] == pytest.approx(0.049, abs=5e-4)
    assert d["caminho_medio"] == pytest.approx(8.83, abs=5e-3)
    assert d["assortatividade"] == pytest.approx(0.091, abs=5e-4)


def test_sigma_e_ccdf():
    assert mc.sigma_mundo_pequeno(0.05, 9.0, 0.002, 8.0) == pytest.approx((0.05 / 0.002) / (9 / 8))
    assert math.isnan(mc.sigma_mundo_pequeno(0.05, 9.0, 0.0, 8.0))
    ccdf = mc.ccdf_graus([nx.path_graph(3), nx.path_graph(3)])          # graus 1, 2, 1 duas vezes
    assert list(ccdf["grau"]) == [1, 2] and list(ccdf["n_nos"]) == [4, 2]
    assert list(ccdf["ccdf_empirica"]) == pytest.approx([1.0, 2 / 6])


# ---------------------------------------------------------------------------
# A seção do Exp 4
# ---------------------------------------------------------------------------
def test_confronto_classicos():
    descritiva = pd.DataFrame([{"modelo": "eco", **{m: 1.0 for m in exp4.DESCRITIVAS_CLASSICOS}},
                               *[{"modelo": "erdos_renyi", **{m: v for m in exp4.DESCRITIVAS_CLASSICOS}}
                                 for v in (2.0, 4.0)]])
    indices = pd.DataFrame([{"modelo": r, "estrategia": "grau", **{m: v for m in exp4.INDICES_CLASSICOS}}
                            for r, v in (("eco", 0.1), ("erdos_renyi", 0.2), ("erdos_renyi", 0.3))])
    tabela = exp4.confronto_classicos(descritiva, indices, "eco")
    assert len(tabela) == len(exp4.DESCRITIVAS_CLASSICOS) + len(exp4.INDICES_CLASSICOS)
    assert "ba_misto_z" not in tabela                                   # só os modelos que existem
    linha = tabela[(tabela["metrica"] == "R_norm") & (tabela["estrategia"] == "grau")].iloc[0]
    assert linha["erdos_renyi_z"] == pytest.approx((0.1 - 0.25) / np.std([0.2, 0.3], ddof=1))


def test_sigma_pela_media_do_erdos_renyi():
    descritiva = pd.DataFrame({"modelo": ["eco", "erdos_renyi", "erdos_renyi"],
                               "clustering_medio": [0.05, 0.001, 0.003], "caminho_medio": [9.0, 7.0, 9.0]})
    sigma = exp4.acrescentar_sigma(descritiva)["sigma"]
    assert sigma.iloc[0] == pytest.approx((0.05 / 0.002) / (9.0 / 8.0))
    assert sigma.iloc[1:].mean() == pytest.approx(((0.001 / 0.002) / (7 / 8) + (0.003 / 0.002) / (9 / 8)) / 2)


@pytest.fixture
def exp4_pequeno(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_TABELAS", tmp_path / "tabelas")
    monkeypatch.setattr(config, "PASTA_FIGURAS", tmp_path / "figuras")
    monkeypatch.setattr(config, "N_MODELOS_NULOS", 2)
    monkeypatch.setattr(config, "N_REPETICOES_NULOS", 2)
    monkeypatch.setattr(config, "N_REPETICOES_ALEATORIAS", 3)
    return tmp_path


def _tabela(conteudo: str) -> pd.DataFrame:
    return pd.read_csv(config.arquivo_tabela("exp4", conteudo, "redes", config.VERSAO_COMPARACAO))


def test_classicos_de_ponta_a_ponta(exp4_pequeno):
    org = "mge"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        exp4.rodar([org], com_nulos=False)
    U = mc.rede_nao_dirigida(exp4.grafo_do_organismo(org))
    descritiva = _tabela("classicos_descritiva")
    assert list(descritiva["modelo"]) == [org] + [m for m in config.MODELOS_CLASSICOS for _ in range(2)]
    assert (descritiva["nos"] == U.number_of_nodes()).all()
    real = descritiva.iloc[0]
    assert real["arestas"] == U.number_of_edges() and real["parametros"] == "rede real"
    assert real["clustering_medio"] == pytest.approx(nx.average_clustering(U))
    assert "sigma" in descritiva.columns                                # o cálculo é testado à parte

    indices = _tabela("classicos_indices")
    assert len(indices) == len(descritiva) * len(config.ESTRATEGIAS_CLASSICOS)
    assert (indices.loc[indices["modelo"] == org, "n_rep"].max()) == config.N_REPETICOES_ALEATORIAS
    _, direto = mc.robustez_rede(U, config.ESTRATEGIAS_CLASSICOS, config.N_REPETICOES_ALEATORIAS, config.SEMENTE)
    do_org = indices[indices["modelo"] == org].set_index("estrategia")
    for linha in direto.to_dict("records"):
        assert do_org.loc[linha["estrategia"], "R_norm"] == pytest.approx(linha["R_norm"])

    curvas = _tabela("classicos_curvas")
    n = U.number_of_nodes()
    for modelo, n_redes in ((org, 1), *((m, 2) for m in config.MODELOS_CLASSICOS)):
        parte = curvas[curvas["modelo"] == modelo]
        assert len(parte) == (n + 1) * len(config.ESTRATEGIAS_CLASSICOS) and (parte["n_redes"] == n_redes).all()

    z = _tabela("classicos_z")
    assert set(z["rede_real"]) == {org} and {f"{m}_z" for m in config.MODELOS_CLASSICOS} <= set(z.columns)
    assert set(_tabela("classicos_ccdf")["modelo"]) == {org, *config.MODELOS_CLASSICOS}

    exp4.plotar()
    for conteudo in ("classicos_indices", "classicos_curvas", "classicos_graus"):
        assert config.arquivo_figura("exp4", conteudo, "redes", config.VERSAO_COMPARACAO).exists(), conteudo
    assert "modelos clássicos" in exp4._resumo()


def test_exp4_sem_classicos(exp4_pequeno):
    exp4.rodar(["mge"], com_nulos=False, com_classicos=False)
    assert not config.arquivo_tabela("exp4", "classicos_descritiva", "redes", config.VERSAO_COMPARACAO).exists()
    exp4.plotar()
    assert not config.arquivo_figura("exp4", "classicos_curvas", "redes", config.VERSAO_COMPARACAO).exists()
