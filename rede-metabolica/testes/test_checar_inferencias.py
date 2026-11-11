"""Checagem das inferências do artigo (experimentos/checar_inferencias.py).

Redes de brinquedo com a resposta feita à mão para cada uma das quatro partes,
e a checagem inteira na E. coli (~10 s), com a contagem dos genes conferida por
um caminho independente (refazer a rede sem as linhas das reações perdidas).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
import contrato
from experimentos import checar_inferencias as ci
from experimentos.biblioteca import grafo, metricas, nocaute, validacao
from experimentos.biblioteca import modelos_classicos as mc

ORG = "eco"


def tabela(*reacoes) -> pd.DataFrame:
    """Cada item: (codigo, substratos, produtos, genes)."""
    linhas = [{"reacao": codigo, "substratos": list(s), "produtos": list(p), "reversivel": False,
               "genes": list(genes), "vias": ["eco00000"], "n_ocorrencias": 1,
               "conflito_reversibilidade": False, "orientacao_invertida": False, "lados_variam": False}
              for codigo, s, p, genes in reacoes]
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


# I -> A, ciclo A -> B -> C -> A, cauda C -> D -> E (6 nós).
#   gravata: núcleo {A, B, C}, entrada {I}, saída {D, E}.
#   g3 e g4 são isoenzimas de R3: nenhuma sozinha derruba a reação.
GRAVATA = tabela(("R0", ["I"], ["A"], ["g0"]), ("R1", ["A"], ["B"], ["g1"]), ("R2", ["B"], ["C"], ["g2"]),
                 ("R3", ["C"], ["A"], ["g3", "g4"]), ("R4", ["C"], ["D"], ["g5"]), ("R5", ["D"], ["E"], ["g6"]))
GENES_EXP2 = pd.DataFrame({"gene": ["g0", "g1", "g2", "g3", "g4", "g5", "g6"],
                           "simbolo": ["s0", "s1", "s2", "s3", "s4", "s5", "s6"],
                           "essencial": [False, False, False, False, False, True, True],
                           "dano_fraco": [1, 0, 0, 0, 0, 2, 1]})


@pytest.fixture
def resultados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_TABELAS", tmp_path / "tabelas")
    monkeypatch.setattr(config, "PASTA_FIGURAS", tmp_path / "figuras")
    return tmp_path


# ---------------------------------------------------------------------------
# (1) Rankings na gravata-borboleta
# ---------------------------------------------------------------------------
def gravata_com_par_solto() -> nx.DiGraph:
    """a -> c1 <-> c2 -> s e o par solto x -> y. Graus totais: c1 = c2 = 3, os outros 1."""
    G = nx.DiGraph()
    G.add_edges_from([("a", "c1"), ("c1", "c2"), ("c2", "c1"), ("c2", "s"), ("x", "y")])
    return G


def test_lado_bow_tie():
    assert ci.lado_bow_tie(gravata_com_par_solto()) == {"c1": "nucleo", "c2": "nucleo", "a": "entrada",
                                                        "s": "saida", "x": "restante", "y": "restante"}


def test_tamanhos_top():
    assert ci.tamanhos_top(1111) == [39, 111]          # os k da E. coli: 3,5% e 10%
    assert ci.tamanhos_top(6, (0.35, 0.5, 0.01, 2.0)) == [2, 3, 1, 6]


def test_rankings_bow_tie_feito_a_mao():
    G = gravata_com_par_solto()
    rankings, primeiros = ci.rankings_bow_tie(G, ["grau"], fracoes=(0.35, 0.5), protegidos=(), n_primeiros=3)
    assert list(rankings["k"]) == [2, 3]
    top2, top3 = rankings.to_dict("records")
    assert (top2["nucleo"], top2["entrada"], top2["sumidouros"]) == (1.0, 0.0, 0.0)
    # O terceiro é o primeiro dos empatados com grau 1, pelo código: a (entrada).
    assert top3["nucleo"] == pytest.approx(2 / 3) and top3["entrada"] == pytest.approx(1 / 3)
    # Entrada: c1 = 2, c2 = 1, a = 0; saída: c1 = 1, c2 = 2, a = 1.
    assert top3["grau_entrada_medio"] == pytest.approx(1.0) and top3["grau_saida_medio"] == pytest.approx(4 / 3)
    assert list(primeiros["metabolito"]) == ["c1", "c2", "a"]
    assert list(primeiros["lado"]) == ["nucleo", "nucleo", "entrada"]


def test_rankings_bow_tie_respeita_os_protegidos():
    """Protegido, a sai do topo: o terceiro passa a ser s (saída e sumidouro)."""
    rankings, primeiros = ci.rankings_bow_tie(gravata_com_par_solto(), ["grau"], fracoes=(0.5,),
                                              protegidos={"a"}, n_primeiros=3, nomes={"s": "Saída"})
    linha = rankings.iloc[0]
    assert linha["saida"] == pytest.approx(1 / 3) and linha["entrada"] == 0
    assert linha["sumidouros"] == pytest.approx(1 / 3)
    assert list(primeiros["nome"]) == ["c1", "c2", "Saída"]


# ---------------------------------------------------------------------------
# (2) Direção dos ramos
# ---------------------------------------------------------------------------
def test_classificar_lado():
    assert [ci.classificar_lado(e, s) for e, s in ((0, 3), (2, 1), (1, 1), (0, 0))] == \
        ["saida", "entrada", "empate", "empate"]


def test_genes_por_lado_feito_a_mao():
    M = grafo.grafo_metabolitos(GRAVATA, remover_moeda=False)
    genes = ci.genes_por_lado(M, GRAVATA, GENES_EXP2).set_index("gene")
    # g1 e g2 cortam o ciclo, mas a rede continua fracamente conexa; g3 e g4 não derrubam R3.
    assert list(genes.index) == ["g0", "g5", "g6"]
    assert genes.loc["g0", ["n", "entrada", "saida", "lado"]].tolist() == [1, 1, 0, "entrada"]
    assert genes.loc["g5", ["n", "entrada", "saida", "lado"]].tolist() == [2, 0, 2, "saida"]   # D e E
    assert genes.loc["g6", ["n", "saida", "lado"]].tolist() == [1, 1, "saida"]               # só E
    assert (genes["n"] == genes["dano_fraco_exp2"]).all()
    assert genes.loc["g5", "simbolo"] == "s5" and bool(genes.loc["g5", "essencial"])


def test_taxas_por_lado():
    genes = pd.DataFrame({"gene": list("abcde"), "n": [1, 6, 7, 2, 9], "essencial": [False, True, False, True, True],
                          "lado": ["entrada", "saida", "saida", "empate", "entrada"]})
    taxas = ci.taxas_por_lado(genes, minimos=(1, 5)).set_index(["minimo", "lado"])
    assert taxas.loc[(1, "saida"), ["genes", "essenciais"]].tolist() == [2, 1]
    assert taxas.loc[(1, "entrada"), "taxa"] == 0.5 and taxas.loc[(1, "empate"), "taxa"] == 1.0
    assert list(taxas.loc[5].index) == ["entrada", "saida"]             # o empate (n = 2) sai
    assert taxas.loc[(5, "entrada"), "taxa"] == 1.0


# ---------------------------------------------------------------------------
# (3) Via do chiquimato
# ---------------------------------------------------------------------------
def test_via_chiquimato_feito_a_mao():
    M = grafo.grafo_metabolitos(GRAVATA, remover_moeda=False)
    B = grafo.grafo_bipartido(GRAVATA, remover_moeda=False)
    tabela_aro = ci.via_chiquimato(M, B, GRAVATA, GENES_EXP2, simbolos=("s5", "s0", "nao_existe"),
                                   aromaticos={"E": "Ee", "B": "Bb"}, nomes={"D": "Dd"})
    assert list(tabela_aro["simbolo"]) == ["s5", "s0"]                 # o que não existe fica de fora
    s5, s0 = tabela_aro.to_dict("records")
    # Sem C -> D, D e E deixam de ser produzidos (a partir de I, a fonte externa); o núcleo fica.
    assert (s5["saem_nucleo"], s5["nao_produzidos_rico"], s5["aromaticos_rico"]) == (0, 2, "Ee")
    assert s5["nomes_rico"] == "Dd; E"
    # Sem I -> A, nada mais é produzido; o ciclo continua lá (o núcleo não muda).
    assert (s0["saem_nucleo"], s0["nao_produzidos_rico"], s0["aromaticos_rico"]) == (0, 5, "Ee, Bb")
    # No meio mínimo, nenhum nutriente está na rede: nada é produzido nem na rede intacta.
    assert s0["nao_produzidos_minimo"] == 0 and s0["aromaticos_minimo"] == ""


def test_aromaticos_nas_fontes():
    B = grafo.grafo_bipartido(GRAVATA, remover_moeda=False)
    assert ci.aromaticos_nas_fontes(B, {"I": "Ii", "A": "Aa"}) == {"Ii": True, "Aa": False}


# ---------------------------------------------------------------------------
# (4) Pontes
# ---------------------------------------------------------------------------
def test_pontes_feito_a_mao():
    assert ci.pontes(nx.path_graph(["a", "b", "c", "d"])) == {
        "nos": 4, "arestas": 3, "pontes": 3, "pontes_internas": 1, "articulacao": 2}
    triangulo_com_cauda = nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    assert ci.pontes(triangulo_com_cauda) == {
        "nos": 4, "arestas": 4, "pontes": 1, "pontes_internas": 0, "articulacao": 1}


def test_pontes_contra_configuracao_usa_as_redes_do_exp4():
    U = mc.rede_nao_dirigida(nx.gnm_random_graph(40, 60, seed=3, directed=True))
    tabela_pontes = ci.pontes_contra_configuracao(U, n_redes=3, semente=7)
    assert list(tabela_pontes["rede"]) == ["eco", "configuracao", "configuracao", "configuracao"]
    assert tabela_pontes["indice"].isna().tolist() == [True, False, False, False]
    assert tabela_pontes.iloc[0][["pontes", "articulacao"]].tolist() == [ci.pontes(U)["pontes"],
                                                                          ci.pontes(U)["articulacao"]]
    for i, C in enumerate(mc.gerar_modelos(U, "configuracao", 3, 7)):
        assert sorted(d for _, d in C.degree()) == sorted(d for _, d in U.degree())   # mesmos graus
        assert tabela_pontes.iloc[i + 1]["pontes"] == ci.pontes(C)["pontes"]


# ---------------------------------------------------------------------------
# A checagem inteira na E. coli
# ---------------------------------------------------------------------------
def test_sem_a_tabela_do_exp2_pede_para_rodar_antes(resultados_temporarios):
    with pytest.raises(FileNotFoundError, match="Exp 2"):
        ci.rodar(ORG)


def test_checar_inferencias_na_ecoli(resultados_temporarios):
    tabelas = grafo.carregar_tabelas(ORG)
    reacoes = tabelas["reacoes"]
    essenciais = validacao.carregar_keio()
    # Uma tabela exp2_genes mínima: só as colunas que a checagem lê (dano_fraco = -1 de propósito).
    genes_exp2 = tabelas["genes"][["gene", "simbolo"]].assign(
        essencial=lambda t: t["gene"].isin(essenciais), dano_fraco=-1)
    config.PASTA_TABELAS.mkdir(parents=True)
    genes_exp2.to_csv(config.arquivo_tabela("exp2", "genes", ORG), index=False)

    r = ci.rodar(ORG)
    colunas = {"rankings": ci.COLUNAS_RANKINGS, "primeiros": ci.COLUNAS_PRIMEIROS, "genes": ci.COLUNAS_GENES,
               "lados": ci.COLUNAS_LADOS, "chiquimato": ci.COLUNAS_CHIQUIMATO, "pontes": ci.COLUNAS_PONTES}
    for conteudo, esperadas in colunas.items():
        assert list(pd.read_csv(config.arquivo_tabela("inferencias", conteudo, ORG)).columns) == esperadas

    # (1) os k da E. coli; os nutrientes protegidos nunca aparecem no topo
    rankings = r["rankings"]
    assert list(rankings["k"]) == [39, 111] * 3
    assert np.allclose(rankings[list(ci.LADOS)].sum(axis=1), 1.0)
    assert not set(r["primeiros"]["metabolito"]) & set(config.NUTRIENTES)
    assert (r["primeiros"].groupby("centralidade").size() == config.N_PRIMEIROS_INFERENCIAS).all()

    # (2) cada gene conferido por outro caminho: a rede refeita sem as linhas das reações perdidas
    genes = r["genes"]
    assert (genes["n"] == genes[list(ci.LADOS)].sum(axis=1)).all() and (genes["n"] >= 1).all()
    assert (genes["lado"] == [ci.classificar_lado(e, s) for e, s in zip(genes["entrada"], genes["saida"])]).all()
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))
    fraco0 = max(nx.weakly_connected_components(G), key=len)
    for linha in genes.sort_values(["n", "gene"], ascending=[False, True]).head(5).to_dict("records"):
        perdidas = nocaute.reacoes_perdidas([linha["gene"]], reacoes)
        H = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes[~reacoes["reacao"].isin(perdidas)],
                                                        remover_moeda=remover))
        H.add_nodes_from(G.nodes)
        assert linha["n"] == len(fraco0 - max(nx.weakly_connected_components(H), key=len)), linha["gene"]
    lados = r["lados"]
    assert lados.groupby("minimo")["genes"].sum().loc[1] == len(genes)

    # (3) os quatro genes aro: Phe e Tyr saem do núcleo; no meio mínimo, os três aromáticos se perdem
    aro = r["chiquimato"].set_index("simbolo")
    assert list(aro.index) == list(config.GENES_CHIQUIMATO)
    assert (aro["aromaticos_nucleo"] == "Phe, Tyr").all()
    assert (aro["aromaticos_minimo"] == "Phe, Tyr, Trp").all()
    assert not any(r["aromaticos_nas_fontes"].values())

    # (4) a E. coli e as redes de configuração do Exp 4
    pontes = r["pontes"]
    assert list(pontes["rede"]) == ["eco"] + ["configuracao"] * config.N_MODELOS_NULOS
    assert pontes["nos"].nunique() == 1 and pontes["arestas"].nunique() == 1
    assert (pontes["pontes_internas"] <= pontes["pontes"]).all()
    assert "contagem igual à do Exp 2: NÃO" in ci._resumo(r)       # dano_fraco = -1 na tabela falsa


def test_resumo_confere_com_o_exp2_quando_a_tabela_e_a_verdadeira(resultados_temporarios, monkeypatch):
    """Com o dano_fraco da própria checagem na tabela, o resumo diz que confere."""
    M = grafo.grafo_metabolitos(GRAVATA, remover_moeda=False)
    genes = ci.genes_por_lado(M, GRAVATA, GENES_EXP2)
    r = {"rede": {"nos": 6, "arestas": 6, "gravata": metricas.bow_tie(grafo.grafo_simples(M))},
         "rankings": ci.rankings_bow_tie(grafo.grafo_simples(M), protegidos=())[0],
         "primeiros": ci.rankings_bow_tie(grafo.grafo_simples(M), protegidos=())[1],
         "genes": genes, "lados": ci.taxas_por_lado(genes),
         "chiquimato": ci.via_chiquimato(M, grafo.grafo_bipartido(GRAVATA, remover_moeda=False), GRAVATA,
                                         GENES_EXP2, simbolos=("s5",)),
         "aromaticos_nas_fontes": {"Phe": False},
         "pontes": ci.pontes_contra_configuracao(nx.path_graph(5), n_redes=2)}
    texto = ci._resumo(r)
    assert "contagem igual à do Exp 2: sim" in texto and "s5* 2 (saida)" in texto
