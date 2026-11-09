"""Modelos nulos (modelos_nulos.py) e Exp 4 (exp4_comparacao.py).

Os modelos nulos são conferidos pelo que prometem preservar (graus, pares
recíprocos, n e m) e pelo que precisam mudar (as arestas, e os componentes
fracos, que a troca de três arestas da NetworkX nunca muda). O de ponta a
ponta roda no mge com poucas redes e repetições, numa pasta temporária.
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
from experimentos.biblioteca import metricas, modelos_nulos, robustez

ORG = "mge"


def rede_com_pares(semente: int = 0) -> nx.DiGraph:
    """Rede dirigida de 40 nós com muitos pares recíprocos, como as reações reversíveis."""
    base = nx.gnm_random_graph(40, 70, seed=semente, directed=True)
    G = nx.DiGraph()
    G.add_nodes_from(f"N{i:02d}" for i in base)
    for u, v in base.edges:
        G.add_edge(f"N{u:02d}", f"N{v:02d}")
        if (u + v) % 2 == 0:  # metade das arestas ganha a volta
            G.add_edge(f"N{v:02d}", f"N{u:02d}")
    return G


def pares_por_no(G: nx.DiGraph) -> dict:
    return {n: sum(1 for v in G.succ[n] if G.has_edge(v, n)) for n in G}


def mesmos_graus(A: nx.DiGraph, B: nx.DiGraph) -> bool:
    return set(A) == set(B) and all(A.in_degree(n) == B.in_degree(n) and A.out_degree(n) == B.out_degree(n)
                                    for n in A)


# ---------------------------------------------------------------------------
# Modelos nulos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reciproco", [False, True])
def test_rewiring_preserva_graus_e_muda_arestas(reciproco):
    G = rede_com_pares()
    R = modelos_nulos.rewiring(G, semente=1, preservar_reciprocidade=reciproco)
    assert mesmos_graus(G, R)
    assert R.number_of_edges() == G.number_of_edges() and nx.number_of_selfloops(R) == 0
    assert len(set(R.edges) & set(G.edges)) < 0.5 * G.number_of_edges()   # embaralhou de fato
    assert R.graph["trocas"] == config.TROCAS_POR_ARESTA * G.number_of_edges()
    assert R.graph["tentativas"] >= R.graph["trocas"]


def test_rewiring_reciproco_preserva_os_pares_de_cada_no():
    G = rede_com_pares()
    R = modelos_nulos.rewiring(G, semente=3, preservar_reciprocidade=True)
    assert pares_por_no(R) == pares_por_no(G)
    assert metricas.reciprocidade(R) == pytest.approx(metricas.reciprocidade(G))
    assert set(modelos_nulos.pares_reciprocos(R)) != set(modelos_nulos.pares_reciprocos(G))  # os pares trocaram


def test_rewiring_reciproco_nao_cria_pares_novos():
    """Rede densa sem nenhum par: as trocas de arestas sem par teriam muitas chances de formar um."""
    base = nx.gnm_random_graph(12, 45, seed=2, directed=True)
    G = nx.DiGraph((u, v) for u, v in base.edges if not (base.has_edge(v, u) and v < u))
    assert metricas.reciprocidade(G) == 0
    for semente in range(5):
        assert metricas.reciprocidade(modelos_nulos.rewiring(G, semente=semente, preservar_reciprocidade=True)) == 0
    assert max(metricas.reciprocidade(modelos_nulos.rewiring(G, semente=s)) for s in range(5)) > 0


def test_rewiring_simples_desfaz_os_pares():
    G = rede_com_pares()
    R = modelos_nulos.rewiring(G, semente=3)
    assert metricas.reciprocidade(R) < 0.5 * metricas.reciprocidade(G)


def test_rewiring_reprodutivel_pela_semente():
    G = rede_com_pares()
    assert set(modelos_nulos.rewiring(G, semente=7).edges) == set(modelos_nulos.rewiring(G, semente=7).edges)
    assert set(modelos_nulos.rewiring(G, semente=7).edges) != set(modelos_nulos.rewiring(G, semente=8).edges)


def test_rewiring_liga_componentes_fracos_diferentes():
    """Dois ciclos separados: a troca de Maslov e Sneppen sorteia arestas em qualquer lugar e mistura os nós.

    É o que a nx.directed_edge_swap não faz (troca só dentro de um caminho a→b→c→d).
    (O número de componentes não serve de teste aqui: com grau 1 em tudo, cada
    troca muda o número de ciclos em ±1, e um número par de trocas volta à mesma paridade.)
    """
    G = nx.DiGraph()
    nx.add_cycle(G, ["a1", "a2", "a3", "a4"])
    nx.add_cycle(G, ["b1", "b2", "b3", "b4"])

    def misturados(R):
        return any({n[0] for n in c} == {"a", "b"} for c in nx.weakly_connected_components(R))

    assert all(misturados(modelos_nulos.rewiring(G, semente=s)) for s in range(5))
    sem_mistura = nx.directed_edge_swap(G.copy(), nswap=20, max_tries=10_000, seed=0)
    assert not misturados(sem_mistura)  # o motivo de não usá-la


def test_rewiring_avisa_quando_nao_consegue_trocar():
    G = nx.DiGraph([("a", "b"), ("a", "c")])   # toda troca repetiria uma aresta
    with pytest.warns(UserWarning, match="trocas aceitas"):
        R = modelos_nulos.rewiring(G, trocas_por_aresta=1, semente=0)
    assert set(R.edges) == set(G.edges) and R.graph["trocas"] == 0


def test_rewiring_nao_mexe_no_original():
    G = rede_com_pares()
    antes = set(G.edges)
    modelos_nulos.rewiring(G, semente=0, preservar_reciprocidade=True)
    assert set(G.edges) == antes


def test_erdos_renyi():
    N = modelos_nulos.erdos_renyi(50, 80, semente=4)
    assert N.is_directed() and N.number_of_nodes() == 50 and N.number_of_edges() == 80
    assert nx.number_of_selfloops(N) == 0
    assert set(N.edges) == set(modelos_nulos.erdos_renyi(50, 80, semente=4).edges)


def test_gerar_nulos_usa_uma_semente_por_rede():
    G = rede_com_pares()
    redes = list(modelos_nulos.gerar_nulos(G, "rewiring_reciproco", n_redes=3, semente=5))
    assert len(redes) == 3
    for i, R in enumerate(redes):
        sozinha = modelos_nulos.rewiring(G, semente=5 * 1000 + i, preservar_reciprocidade=True)
        assert set(R.edges) == set(sozinha.edges)
    er = list(modelos_nulos.gerar_nulos(G, "erdos_renyi", n_redes=2, semente=5))
    assert all(N.number_of_nodes() == 40 and N.number_of_edges() == G.number_of_edges() for N in er)
    with pytest.raises(ValueError, match="modelo nulo desconhecido"):
        next(modelos_nulos.gerar_nulos(G, "configuracao"))


def test_reciprocidade():
    G = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c")])
    assert metricas.reciprocidade(G) == pytest.approx(2 / 3)
    assert math.isnan(metricas.reciprocidade(nx.DiGraph()))
    assert metricas.tabela_descritiva(G)["reciprocidade"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Peças do Exp 4
# ---------------------------------------------------------------------------
def test_indices_normalizados_batem_com_a_curva_relativa():
    G = nx.DiGraph()
    nx.add_path(G, ["A", "B", "C", "D", "E"])
    G.add_edge("C00031", "A")                     # o "nutriente" alcança A..E: 6 de 9 nós
    G.add_edges_from([("X", "Y"), ("Y", "Z")])   # componente separado: S0 = 6/9
    _, curvas, indices = exp4.medir_rede(G, ["C00031"], n_rep=3, semente=0)
    for linha in indices.to_dict("records"):
        curva = curvas[curvas["estrategia"] == linha["estrategia"]]
        depois = curva[curva["n_removidos"] > 0]
        assert linha["S0_fraco"] == pytest.approx(6 / 9)
        assert linha["alcance0"] == pytest.approx(6 / 9)
        assert linha["R_fraco_norm"] == pytest.approx(metricas.indice_robustez(depois["maior_fraco_rel"]))
        assert linha["R_fraco_norm"] == pytest.approx(linha["R_fraco"] / linha["S0_fraco"])
        assert linha["R_alcance_norm"] == pytest.approx(linha["R_alcance"] / linha["alcance0"])
    rel = curvas.loc[curvas["n_removidos"] == 0, ["maior_fraco_rel", "maior_forte_rel"]]
    assert (rel == 1).all().all()


def test_relativas_sem_alcance_ficam_nan():
    G = nx.gnm_random_graph(20, 30, seed=1, directed=True)
    _, curvas, indices = exp4.medir_rede(G, (), n_rep=2, semente=0)
    assert curvas["alcance_rel"].isna().all() and indices["R_alcance_norm"].isna().all()


def test_media_das_redes():
    a = pd.DataFrame({"estrategia": "grau", "n_removidos": [0, 1], "fracao_removida": [0, 0.5],
                      **{m: [1.0, 0.4] for m in robustez.METRICAS},
                      **{f"{m}_rel": [1.0, 0.5] for m in exp4.RELATIVAS}})
    b = a.copy()
    b["maior_fraco"] = [1.0, 0.2]
    media = exp4.media_das_redes([a, b])
    passo1 = media[media["n_removidos"] == 1].iloc[0]
    assert passo1["maior_fraco"] == pytest.approx(0.3)
    assert passo1["maior_fraco_dp"] == pytest.approx(np.std([0.4, 0.2], ddof=1))
    assert passo1["maior_forte_dp"] == pytest.approx(0) and passo1["n_redes"] == 2


def test_escore_z():
    assert exp4.escore_z(3.0, [1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0, 1.0))
    media, dp, z = exp4.escore_z(5.0, [2.0, 2.0])
    assert media == 2.0 and dp == 0.0 and math.isnan(z)        # nulos todos iguais: z indefinido
    assert all(math.isnan(v) for v in exp4.escore_z(1.0, []))
    media, dp, z = exp4.escore_z(0.6, [0.5116550116550117] * 10)  # dp de arredondamento, não variação
    assert dp == 0.0 and math.isnan(z)
    assert math.isnan(exp4.escore_z(1.0, [2.0])[2])                # uma rede só: sem dp


def test_confronto_nulos():
    descritiva = pd.DataFrame([{"rede": "eco", **{m: 1.0 for m in exp4.DESCRITIVAS_NULOS}},
                               *[{"rede": "rewiring", **{m: v for m in exp4.DESCRITIVAS_NULOS}}
                                 for v in (2.0, 4.0)]])
    indices = pd.DataFrame([{"rede": r, "estrategia": "grau", **{m: v for m in exp4.INDICES_NULOS}}
                            for r, v in (("eco", 0.1), ("rewiring", 0.2), ("rewiring", 0.3))])
    tabela = exp4.confronto_nulos(descritiva, indices, "eco")
    assert len(tabela) == len(exp4.DESCRITIVAS_NULOS) + len(exp4.INDICES_NULOS)
    assert "erdos_renyi_z" not in tabela                        # só os nulos que existem
    linha = tabela[tabela["metrica"] == "reciprocidade"].iloc[0]
    assert linha["rewiring_media"] == 3.0 and linha["rewiring_z"] == pytest.approx((1 - 3) / math.sqrt(2))
    linha = tabela[(tabela["metrica"] == "metade_fraco") & (tabela["estrategia"] == "grau")].iloc[0]
    assert linha["real"] == 0.1 and linha["rewiring_z"] == pytest.approx((0.1 - 0.25) / np.std([0.2, 0.3], ddof=1))


# ---------------------------------------------------------------------------
# De ponta a ponta
# ---------------------------------------------------------------------------
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


def test_exp4_de_ponta_a_ponta(exp4_pequeno):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)             # todas as trocas pedidas são aceitas
        exp4.rodar([ORG])
    G = exp4.grafo_do_organismo(ORG)

    descritiva = _tabela("descritiva")
    assert list(descritiva["rede"]) == [ORG] + [t for t in modelos_nulos.TIPOS for _ in range(2)]
    real = descritiva.iloc[0]
    esperado = metricas.tabela_descritiva(G)
    assert real["nucleo"] == esperado["nucleo"] and real["reciprocidade"] == pytest.approx(esperado["reciprocidade"])
    assert not math.isnan(real["lp_alpha"])
    nulos = descritiva.iloc[1:]
    assert (nulos["nos"] == G.number_of_nodes()).all() and (nulos["arestas"] == G.number_of_edges()).all()
    reciproco = nulos[nulos["rede"] == "rewiring_reciproco"]
    assert np.allclose(reciproco["reciprocidade"], esperado["reciprocidade"])
    assert (nulos.loc[nulos["rede"] == "rewiring", "reciprocidade"] < esperado["reciprocidade"]).all()
    assert (nulos.loc[nulos["rede"] != "erdos_renyi", "trocas"] == config.TROCAS_POR_ARESTA * G.number_of_edges()).all()

    # Os índices do organismo são os do Exp 1 com as mesmas repetições.
    indices = _tabela("indices")
    _, ind_direto = robustez.comparar_estrategias(G, config.ESTRATEGIAS_COMPARACAO, n_rep=3,
                                                  semente=config.SEMENTE, nutrientes=config.NUTRIENTES,
                                                  medir_eficiencia=False)
    do_org = indices[indices["rede"] == ORG].set_index("estrategia")
    for linha in ind_direto.to_dict("records"):
        assert do_org.loc[linha["estrategia"], "R_fraco"] == pytest.approx(linha["R_fraco"])
        assert do_org.loc[linha["estrategia"], "metade_forte"] == pytest.approx(linha["metade_forte"])
    assert len(indices) == (1 + 2 * len(modelos_nulos.TIPOS)) * len(config.ESTRATEGIAS_COMPARACAO)
    assert indices.loc[indices["rede"] == "erdos_renyi", "R_alcance"].isna().all()
    assert indices.loc[indices["rede"] == "rewiring", "R_alcance"].notna().all()

    curvas = _tabela("curvas")
    n = G.number_of_nodes()
    for rede, n_redes in ((ORG, 1), *((t, 2) for t in modelos_nulos.TIPOS)):
        parte = curvas[curvas["rede"] == rede]
        assert len(parte) == (n + 1) * len(config.ESTRATEGIAS_COMPARACAO) and (parte["n_redes"] == n_redes).all()

    confronto = _tabela("nulos")
    assert set(confronto["rede_real"]) == {ORG}
    assert {f"{t}_z" for t in modelos_nulos.TIPOS} <= set(confronto.columns)
    linha = confronto[confronto["metrica"] == "nucleo"].iloc[0]
    rew = descritiva.loc[descritiva["rede"] == "rewiring", "nucleo"]
    assert linha["real"] == real["nucleo"] and linha["rewiring_media"] == pytest.approx(rew.mean())

    ccdf = _tabela("ccdf")
    assert set(ccdf["rede"]) == {ORG} and ccdf["ccdf_empirica"].iloc[0] == pytest.approx(1.0)

    exp4.plotar()
    for conteudo, org in (("indices", "redes"), ("curvas", "organismos"), ("curvas", "nulos"), ("graus", "redes")):
        assert config.arquivo_figura("exp4", conteudo, org, config.VERSAO_COMPARACAO).exists()
    assert "escores z" in exp4._resumo()


def test_exp4_sem_nulos(exp4_pequeno):
    exp4.rodar([ORG], com_nulos=False)
    assert set(_tabela("descritiva")["rede"]) == {ORG}
    assert not config.arquivo_tabela("exp4", "nulos", "redes", config.VERSAO_COMPARACAO).exists()
    exp4.plotar()
    assert not config.arquivo_figura("exp4", "curvas", "nulos", config.VERSAO_COMPARACAO).exists()
    assert config.arquivo_figura("exp4", "indices", "redes", config.VERSAO_COMPARACAO).exists()


def test_exp4_figuras_sem_tabelas(exp4_pequeno):
    with pytest.raises(FileNotFoundError, match="faltam as tabelas do Exp 4"):
        exp4.plotar()
