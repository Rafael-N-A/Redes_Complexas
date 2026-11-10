"""Exp 3: escolha dos controles, profundidade da avalanche e o experimento de ponta a ponta.

O de ponta a ponta roda no mge (pequeno; três das seis substâncias nem
existem nele, o que também testa o aviso). As tabelas vão para uma pasta
temporária e são conferidas contra a cascata chamada à parte.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
from experimentos import exp3_vies
from experimentos.biblioteca import cascata, grafo, metricas

ORG = "mge"


def grafo_estrela() -> nx.DiGraph:
    """Hubs H1 (grau 4) e H2 (grau 3, empatado com H3), e folhas de grau 1."""
    G = nx.DiGraph()
    G.add_edges_from([("H1", "a"), ("H1", "b"), ("c", "H1"), ("H1", "H2"),
                      ("H2", "d"), ("e", "H2"),
                      ("H3", "f"), ("H3", "g"), ("h", "H3")])
    return G


# ---------------------------------------------------------------------------
# Controles e profundidade
# ---------------------------------------------------------------------------
def test_hubs_por_grau_com_empate_pelo_codigo():
    hubs, _ = exp3_vies.escolher_controles(grafo_estrela(), set(), 3, np.random.default_rng(0))
    assert hubs == ["H1", "H2", "H3"]                     # H2 e H3 empatam em 3: vale o código
    hubs, _ = exp3_vies.escolher_controles(grafo_estrela(), {"H1"}, 2, np.random.default_rng(0))
    assert hubs == ["H2", "H3"]                           # os excluídos não entram


def test_baixo_grau_sorteado_no_quantil():
    G = grafo_estrela()
    graus = metricas.graus(G)["total"]
    limite = np.quantile(graus, config.QUANTIL_BAIXO_GRAU)
    _, baixo = exp3_vies.escolher_controles(G, {"a"}, 3, np.random.default_rng(7))
    assert len(baixo) == 3 and len(set(baixo)) == 3 and "a" not in baixo
    assert all(graus[m] <= limite for m in baixo)
    _, de_novo = exp3_vies.escolher_controles(G, {"a"}, 3, np.random.default_rng(7))
    assert de_novo == baixo                               # mesma semente, mesmo sorteio


def test_controles_so_entre_os_candidatos():
    hubs, baixo = exp3_vies.escolher_controles(grafo_estrela(), set(), 5, np.random.default_rng(0),
                                               candidatos={"H2", "b", "d"})
    assert hubs == ["H2", "b", "d"]
    assert baixo == ["b", "d"]                            # menos candidatos que n: vêm todos


def test_profundidade():
    G = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "D")])
    assert exp3_vies.profundidade(G, "A", {"B", "C", "D"}) == 2
    assert exp3_vies.profundidade(G, "A", set()) == 0
    assert exp3_vies.profundidade(G, "X", set()) == 0     # fora do grafo e sem perda
    with pytest.raises(ValueError):
        exp3_vies.profundidade(G, "B", {"D"})             # D não descende de B


def test_posicao_do_maior_para_o_menor():
    assert list(exp3_vies._posicao(pd.Series([5, 9, 5, 0]))) == [2, 1, 2, 4]


# ---------------------------------------------------------------------------
# De ponta a ponta
# ---------------------------------------------------------------------------
@pytest.fixture
def resultados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_TABELAS", tmp_path / "tabelas")
    monkeypatch.setattr(config, "PASTA_FIGURAS", tmp_path / "figuras")
    return tmp_path


@pytest.fixture
def redes():
    reacoes = grafo.carregar_tabelas(ORG)["reacoes"]
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))
    return B, G


def test_exp3_de_ponta_a_ponta(resultados_temporarios, redes, capsys):
    B, G = redes
    escolhidos = exp3_vies.rodar(ORG)
    aviso = capsys.readouterr().out
    faltando = [m for m in config.SUBSTANCIAS_EXP3 if m not in B]
    assert faltando and all(m in aviso for m in faltando)

    assert list(escolhidos.columns) == exp3_vies.COLUNAS_SUBSTANCIAS
    grupos = escolhidos.groupby("grupo")["metabolito"].apply(list).to_dict()
    assert grupos["substancia"] == [m for m in config.SUBSTANCIAS_EXP3 if m in B]
    assert len(grupos["hub"]) == len(grupos["baixo_grau"]) == config.N_CONTROLES_EXP3

    produzidos, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
    assert set(grupos["hub"]) | set(grupos["baixo_grau"]) <= produzidos
    assert not set(grupos["hub"]) & set(config.SUBSTANCIAS_EXP3)
    graus = metricas.graus(G)["total"]
    assert all(graus[m] <= np.quantile(graus, config.QUANTIL_BAIXO_GRAU) for m in grupos["baixo_grau"])

    for linha in escolhidos.to_dict("records"):
        perdidos = cascata.cascata(B, [linha["metabolito"]])
        assert linha["avalanche"] == len(perdidos)
        assert math.isclose(linha["fracao"], len(perdidos) / len(produzidos))
        assert linha["profundidade"] == exp3_vies.profundidade(G, linha["metabolito"], perdidos)

    todos = pd.read_csv(config.arquivo_tabela("exp3", "avalanches", ORG))
    assert set(todos["metabolito"]) == produzidos
    assert todos["avalanche"].is_monotonic_decreasing
    pos = todos.set_index("metabolito")["avalanche"].rank(ascending=False, method="min")
    for linha in escolhidos.dropna(subset=["posicao_avalanche"]).to_dict("records"):
        assert linha["posicao_avalanche"] == pos[linha["metabolito"]]

    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", ORG)).iloc[0]
    assert resumo["produzidos"] == len(produzidos)
    assert resumo["sem_avalanche"] == (todos["avalanche"] == 0).sum()
    assert resumo["avalanche_max"] == todos["avalanche"].max()

    exp3_vies.plotar(ORG)
    for conteudo in ("avalanche_vs_grau", "distribuicao"):
        assert config.arquivo_figura("exp3", conteudo, ORG).exists()


def test_toda_perda_descende_do_removido(redes):
    """A profundidade sempre existe: o que se perde está a jusante do removido no grafo."""
    B, G = redes
    fontes = cascata.fontes_externas(B)
    produzidos, _ = cascata.disponiveis(B, fontes)
    for m in sorted(produzidos):
        perdidos = cascata.cascata(B, [m], fontes=fontes)
        if perdidos:
            assert perdidos <= nx.descendants(G, m), m


# ---------------------------------------------------------------------------
# Dano estrutural de cada metabólito (P2 sem gabarito)
# ---------------------------------------------------------------------------
def test_exp3_dano_estrutural_de_ponta_a_ponta(resultados_temporarios, redes):
    B, G = redes
    escolhidos = exp3_vies.rodar(ORG)
    dano = pd.read_csv(config.arquivo_tabela("exp3", "dano", ORG))
    assert list(dano.columns) == exp3_vies.COLUNAS_DANO
    assert set(dano["metabolito"]) == set(G.nodes)                       # todos os nós do grafo, uma vez cada
    direto = metricas.dano_remocao_individual(G).set_index("metabolito")
    por = dano.set_index("metabolito")
    for coluna in ("dano_forte", "dano_fraco", "dano_eficiencia"):
        assert np.allclose(por[coluna], direto[coluna].reindex(por.index)), coluna
    avalanches = pd.read_csv(config.arquivo_tabela("exp3", "avalanches", ORG)).set_index("metabolito")["avalanche"]
    produzidos = por[por["produzido"]]
    assert (produzidos["avalanche"] == avalanches.reindex(produzidos.index)).all()
    assert por.loc[~por["produzido"], "avalanche"].isna().all()
    assert dano["dano_forte"].is_monotonic_decreasing

    correlacoes = pd.read_csv(config.arquivo_tabela("exp3", "correlacoes", ORG)).set_index(["medida", "outra"])
    assert len(correlacoes) == len(exp3_vies.MEDIDAS_DANO) * len(exp3_vies.CENTRALIDADES) + 3
    parte = dano.dropna(subset=["avalanche"])
    assert math.isclose(correlacoes.loc[("dano_eficiencia", "avalanche"), "spearman"],
                        exp3_vies._spearman(parte["dano_eficiencia"], parte["avalanche"])[0])
    assert correlacoes.loc[("dano_forte", "grau"), "n"] == G.number_of_nodes()

    no_grafo = escolhidos.dropna(subset=["dano_forte"])
    assert (no_grafo["dano_forte"].astype(int) == direto["dano_forte"].reindex(no_grafo["metabolito"]).to_numpy()).all()
    exemplos = pd.read_csv(config.arquivo_tabela("exp3", "exemplos", ORG))
    assert list(exemplos.columns) == exp3_vies.COLUNAS_EXEMPLOS
    assert {"estrutural_sem_funcional", "funcional_sem_estrutural"} <= set(exemplos["grupo"])
    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", ORG)).iloc[0]
    assert resumo["nucleo_intacto"] == len(metricas.maior_componente_forte(G))
    assert resumo["nucleo_quebra"] == (dano["dano_forte"] >= 2).sum()

    exp3_vies.plotar(ORG)
    assert config.arquivo_figura("exp3", "dano_vs_avalanche", ORG).exists()
    assert "dano estrutural" in exp3_vies._resumo(escolhidos, ORG)


def dano_de_brinquedo() -> pd.DataFrame:
    return pd.DataFrame({
        "metabolito": ["C1", "C2", "C3", "C4", "C5", "C6"], "nome": ["um", "dois", "tres", "quatro", "cinco", "seis"],
        "produzido": [True, True, True, True, True, False],
        "grau": [10, 2, 5, 3, 1, 1], "betweenness": [0.5, 0.0, 0.2, 0.1, 0.0, 0.0],
        "harmonica": [9.0, 1.0, 4.0, 2.0, 0.5, 0.0], "pagerank": [0.3, 0.1, 0.2, 0.2, 0.1, 0.1],
        "dano_forte": [30, 25, 10, 0, 0, 0], "dano_fraco": [5, 1, 2, 0, 1, 0],
        "dano_forte_fracao": 0.0, "dano_fraco_fracao": 0.0,
        "dano_eficiencia": [0.2, 0.05, 0.06, 0.01, 0.0, 0.0],
        "avalanche": [40.0, 0.0, 12.0, 30.0, 1.0, np.nan],
    })


def test_discordantes():
    """C2 quebra o núcleo (25) e não para nada; C4 para 30 sem tocar o núcleo; C6 não é produzido e fica de fora."""
    tabela = exp3_vies.discordantes(dano_de_brinquedo(), k=1)
    assert tabela.set_index("grupo")["metabolito"].to_dict() == {"estrutural_sem_funcional": "C2",
                                                                 "funcional_sem_estrutural": "C4"}
    assert (tabela["discordancia"] > 0).tolist() == [True, False]
    assert "C6" not in set(exp3_vies.discordantes(dano_de_brinquedo(), k=6)["metabolito"])


def test_correlacoes_dano_usam_os_pares_definidos():
    tabela = exp3_vies.correlacoes_dano(dano_de_brinquedo()).set_index(["medida", "outra"])
    assert tabela.loc[("dano_forte", "grau"), "n"] == 6                    # estrutural: todos
    assert tabela.loc[("dano_forte", "avalanche"), "n"] == 5               # com avalanche: só os produzidos
    assert ("avalanche", "avalanche") not in tabela.index


def test_exemplos_dano_metabolitos(capsys):
    exemplos = exp3_vies.exemplos_dano(dano_de_brinquedo(), {"C1": "alto nos dois", "C9": "não existe"}, k=1)
    assert "C9" in capsys.readouterr().out                                 # avisa o que não está no grafo
    assert list(exemplos["grupo"]) == ["exemplo", "estrutural_sem_funcional", "funcional_sem_estrutural"]
    primeiro = exemplos.iloc[0]
    assert primeiro["metabolito"] == "C1" and primeiro["motivo"] == "alto nos dois"
    assert primeiro["posicao_dano_forte"] == 1 and primeiro["posicao_avalanche"] == 1
    assert exemplos.set_index("metabolito").loc["C2", "nome_curto"] == "dois"   # sem nome curto: o do KEGG
