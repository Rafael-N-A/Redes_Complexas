"""Exp 2 de ponta a ponta na E. coli (o único organismo com gabarito; ~40 s, quase tudo nos nocautes estruturais).

As tabelas vão para uma pasta temporária; os valores são conferidos contra
as funções da biblioteca chamadas à parte.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import spearmanr

import config
from experimentos import exp2_biologia
from experimentos.biblioteca import grafo, metricas, nocaute, validacao

ORG = "eco"


@pytest.fixture
def resultados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_TABELAS", tmp_path / "tabelas")
    monkeypatch.setattr(config, "PASTA_FIGURAS", tmp_path / "figuras")
    return tmp_path


def ler(conteudo: str) -> pd.DataFrame:
    return pd.read_csv(config.arquivo_tabela("exp2", conteudo, ORG))


def test_exp2_de_ponta_a_ponta(resultados_temporarios):
    avaliacao = exp2_biologia.rodar(ORG)
    reacoes = grafo.carregar_tabelas(ORG)["reacoes"]
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    essenciais = validacao.carregar_keio()

    genes = ler("genes")
    assert list(genes["gene"]) == sorted(nocaute.reacoes_por_gene(reacoes))
    assert genes["essencial"].sum() == len(essenciais & set(genes["gene"]))
    dano = nocaute.dano_todos_genes(B, reacoes).set_index("gene")["dano"]
    assert (genes.set_index("gene")["dano"] == dano).all()
    # Dano estrutural: alguns genes refeitos à parte (grafo sem as reações perdidas, medido do zero).
    M = grafo.grafo_metabolitos(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(M)
    n = G.number_of_nodes()
    antes = metricas.medidas_estruturais(G, n)
    por_gene = genes.set_index("gene")
    amostra = por_gene[por_gene["simbolo"].isin(["fabG", "purH", "thyA", "aroA", "lpxC", "eno", "accA"])].index
    for gene in [*amostra, *por_gene.index[::97]]:
        H = nocaute.grafo_sem_reacoes(M, nocaute.reacoes_perdidas([gene], reacoes), G.nodes)
        queda = metricas.queda_estrutural(antes, metricas.medidas_estruturais(H, n), n)
        for coluna in ("dano_forte", "dano_fraco", "dano_eficiencia"):
            assert por_gene.loc[gene, coluna] == pytest.approx(queda[coluna]), (gene, coluna)
    # Quem não perde nenhuma reação (toda reação tem isoenzima) tem dano estrutural 0.
    sem_perda = genes[genes["n_reacoes_perdidas"] == 0]
    assert len(sem_perda) == len(genes) - 574
    assert (sem_perda[["arestas_perdidas", "dano_forte", "dano_fraco", "dano_eficiencia"]] == 0).all().all()

    # Os rankings saem para os dois subconjuntos: todos os genes e só os que perdem reação.
    assert list(dict.fromkeys(avaliacao["subconjunto"])) == ["todos", "perde_reacao"]
    todos = avaliacao[avaliacao["subconjunto"] == "todos"]
    assert list(todos["ranking"]) == ["dano", "dano_forte", "dano_fraco", "dano_eficiencia", "grau", "betweenness",
                                      "harmonica", "pagerank", "dano_meio_minimo", "reacoes_perdidas"]
    assert avaliacao["papel"].notna().all()
    y = genes.set_index("gene")["essencial"]
    for nome, coluna in (("dano", "dano"), ("dano_forte", "dano_forte"), ("dano_eficiencia", "dano_eficiencia"),
                         ("grau", "grau"), ("reacoes_perdidas", "n_reacoes_perdidas")):
        esperado = validacao.auc(y, genes.set_index("gene")[coluna])
        assert math.isclose(todos.set_index("ranking").loc[nome, "auc"], esperado)
    perde = genes[genes["n_reacoes_perdidas"] > 0].set_index("gene")
    subconjunto = avaliacao[avaliacao["subconjunto"] == "perde_reacao"].set_index("ranking")
    assert (subconjunto["n_genes"] == len(perde)).all()
    assert math.isclose(subconjunto.loc["dano", "auc"], validacao.auc(perde["essencial"], perde["dano"]))
    # Valores de referência dos protótipos (Seção 8 do plano)
    assert round(todos.set_index("ranking").loc["dano", "auc"], 3) == 0.756
    assert round(subconjunto.loc["dano_forte", "auc"], 3) == 0.513

    correlacoes = ler("correlacoes").set_index(["subconjunto", "medida_x", "medida_y"])
    rho = correlacoes.loc[("perde_reacao", "dano", "dano_eficiencia"), "spearman"]
    assert math.isclose(rho, spearmanr(perde["dano"], perde["dano_eficiencia"]).statistic)

    casos = ler("exemplos_dano")
    assert list(dict.fromkeys(casos["grupo"])) == list(config.EXEMPLOS_DANO_EXP2)
    grupo = casos.set_index("simbolo")["essencial"]
    assert not grupo[["aroA", "purH", "thyA"]].any() and grupo[["fabG", "lpxC", "accA", "eno"]].all()
    assert (casos.loc[casos["grupo"] == "essencial_dano_zero", ["dano", "dano_forte"]] == 0).all().all()

    resumo_rede = ler("resumo").iloc[0]
    assert (resumo_rede["nucleo_intacto"], resumo_rede["maior_fraco_intacto"]) == (415, 885)
    assert resumo_rede["genes_com_reacao_perdida"] == 574

    choke = ler("choke").iloc[0]
    assert choke["genes_choke"] == genes["choke"].sum()
    assert choke["essenciais_choke"] == (genes["choke"] & genes["essencial"]).sum()
    assert choke["reacoes_choke"] == len(metricas.choke_points(reacoes))

    resumo = ler("resumo").iloc[0]
    assert resumo["essenciais_fora_da_rede"] == len(essenciais - set(genes["gene"]))
    assert resumo["genes_com_dano"] == (genes["dano"] > 0).sum()

    exemplos = ler("exemplos")
    assert 0 < len(exemplos) <= exp2_biologia.N_EXEMPLOS
    assert set(exemplos["gene"]) <= set(genes.loc[genes["essencial"], "gene"])
    assert (exemplos["percentil_grau"] <= 0.5).all() and (exemplos["dano"] > 0).all()
    assert exemplos["dano"].is_monotonic_decreasing

    roc = ler("roc")
    assert set(roc["ranking"]) == set(avaliacao["ranking"])
    assert set(roc["subconjunto"]) == {"todos", "perde_reacao"}

    # (d) essenciais por faixa de dano: só os 574 que perdem reação, cada um numa faixa por medida
    faixas = ler("essenciais_por_faixa")
    assert list(faixas.columns) == exp2_biologia.COLUNAS_FAIXAS
    assert list(dict.fromkeys(faixas["medida"])) == list(exp2_biologia.MEDIDAS_FAIXAS)
    for medida, parte in faixas.groupby("medida"):
        assert parte["genes"].sum() == 574 and parte["essenciais"].sum() == perde["essencial"].sum()
        dez_ou_mais = parte.set_index("faixa").loc["10 ou mais"]
        assert dez_ou_mais["genes"] == (perde[medida] >= 10).sum()
        assert dez_ou_mais["essenciais"] == perde.loc[perde[medida] >= 10, "essencial"].sum()
    # Os números da Figura 4 do artigo
    cascata = faixas[faixas["medida"] == "dano"].set_index("faixa")
    assert (cascata.loc["10 ou mais", "essenciais"], cascata.loc["10 ou mais", "genes"]) == (36, 52)

    exp2_biologia.plotar(ORG)
    for conteudo in ("roc", "dispersao", "auc", "essenciais_por_faixa"):
        assert config.arquivo_figura("exp2", conteudo, ORG).exists(), conteudo
    texto = exp2_biologia._resumo(avaliacao, ORG)
    assert "Spearman" in texto and "36/52 (69%)" in texto


def test_essenciais_por_faixa_feito_a_mao():
    """Faixas 0, 1–2, 3–9 e 10+; quem não perde reação fica de fora; faixa vazia com taxa NaN."""
    genes = pd.DataFrame({"gene": [f"g{i}" for i in range(7)],
                          "n_reacoes_perdidas": [1, 1, 1, 1, 1, 1, 0],
                          "essencial": [True, False, False, True, True, False, True],
                          "dano":       [0, 0, 2, 3, 12, 40, 99],
                          "dano_forte": [0, 1, 1, 0, 0, 10, 99]})
    faixas = exp2_biologia.essenciais_por_faixa(genes).set_index(["medida", "faixa"])
    assert faixas.loc[("dano", "0"), ["genes", "essenciais"]].tolist() == [2, 1]
    assert faixas.loc[("dano", "1–2"), ["genes", "essenciais"]].tolist() == [1, 0]
    assert faixas.loc[("dano", "3–9"), ["genes", "essenciais", "taxa"]].tolist() == [1, 1, 1.0]
    assert faixas.loc[("dano", "10 ou mais"), ["genes", "essenciais"]].tolist() == [2, 1]   # g6 (99) fica de fora
    assert faixas.loc[("dano_forte", "1–2"), "genes"] == 2
    assert faixas.loc[("dano_forte", "3–9"), "genes"] == 0 and math.isnan(faixas.loc[("dano_forte", "3–9"), "taxa"])
    assert faixas.loc[("dano", "1–2"), ["minimo", "maximo"]].tolist() == [1, 2]
    assert pd.isna(faixas.loc[("dano", "10 ou mais"), "maximo"])
    assert list(faixas.index.get_level_values("medida").unique()) == ["dano_forte", "dano"]


def test_exemplos_de_baixa_centralidade():
    """Só essenciais, abaixo da mediana em grau E em betweenness, com dano > 0; maior dano primeiro."""
    genes = pd.DataFrame({
        "gene": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        "simbolo": "", "descricao": "",
        "essencial": [True, True, True, True, False, True, False, False],
        "grau":        [1, 1, 9, 1, 1, 1, 5, 6],
        "betweenness": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.5, 0.6],
        "dano":        [3, 8, 50, 40, 60, 0, 1, 1],
        "n_reacoes_perdidas": 1, "choke": False,
    })
    exemplos = exp2_biologia.exemplos_baixa_centralidade(genes, n=5)
    # g3: grau alto; g4: betweenness alta; g5: não essencial; g6: sem dano
    assert list(exemplos["gene"]) == ["g2", "g1"]


def genes_de_brinquedo() -> pd.DataFrame:
    return pd.DataFrame({
        "gene": [f"g{i}" for i in range(1, 9)],
        "simbolo": ["aroA", "aroB", "fabG", "accA", "xyzA", "xyzB", "xyzC", "xyzD"],
        "descricao": "",
        "essencial": [False, False, True, True, False, True, False, False],
        "n_reacoes": 1, "n_reacoes_perdidas": [1, 1, 1, 0, 1, 1, 0, 1],
        "dano": [1, 7, 50, 0, 3, 9, 0, 2], "dano_forte": [20, 20, 20, 0, 1, 0, 0, 5],
        "dano_fraco": [0, 0, 18, 0, 1, 2, 0, 0], "dano_eficiencia": [0.03, 0.04, 0.07, 0.0, 0.01, 0.02, 0.0, 0.005],
        "grau": [3, 5, 5, 40, 2, 2, 1, 1],
    })


def test_exemplos_dano():
    grupos = {"estrutural": (("aroA", "aroB", "naoExiste"), "quebram o núcleo"),
              "zero": (("accA",), "complexo")}
    casos = exp2_biologia.exemplos_dano(genes_de_brinquedo(), grupos)
    assert list(casos.columns) == exp2_biologia.COLUNAS_EXEMPLOS_DANO
    assert list(casos["simbolo"]) == ["aroA", "aroB", "accA"]          # o que não está na rede fica de fora
    assert list(casos["motivo"]) == ["quebram o núcleo", "quebram o núcleo", "complexo"]
    por = casos.set_index("simbolo")
    assert por.loc["aroA", "posicao_dano_forte"] == 1 and por.loc["aroB", "posicao_dano"] == 3   # fabG 50, xyzB 9, aroB 7


def test_avaliar_subconjuntos():
    genes = genes_de_brinquedo().set_index("gene")
    y = genes["essencial"]
    perde = genes["n_reacoes_perdidas"] > 0
    avaliacao = exp2_biologia.avaliar_subconjuntos(y, genes, {"dano": "dano", "grau": "grau"}, perde)
    assert list(avaliacao.columns[:3]) == ["subconjunto", "ranking", "papel"]
    sub = avaliacao.set_index(["subconjunto", "ranking"])
    assert sub.loc[("todos", "dano"), "n_genes"] == 8 and sub.loc[("perde_reacao", "dano"), "n_genes"] == 6
    assert math.isclose(sub.loc[("perde_reacao", "grau"), "auc"], validacao.auc(y[perde], genes.loc[perde, "grau"]))


def test_correlacoes_entre_as_medidas():
    genes = genes_de_brinquedo()
    tabela = exp2_biologia.correlacoes(genes).set_index(["subconjunto", "medida_x", "medida_y"])
    perde = genes[genes["n_reacoes_perdidas"] > 0]
    assert tabela.loc[("perde_reacao", "dano", "dano_forte"), "n_genes"] == len(perde)
    assert math.isclose(tabela.loc[("todos", "dano", "dano_eficiencia"), "spearman"],
                        spearmanr(genes["dano"], genes["dano_eficiencia"]).statistic)


def test_rotulos_da_dispersao_juntam_os_genes_no_mesmo_ponto():
    casos = exp2_biologia.exemplos_dano(genes_de_brinquedo(), {
        "estrutural_alto_nao_essencial": (("aroA", "aroB"), ""), "funcional_alto_essencial": (("fabG",), ""),
        "essencial_dano_zero": (("accA",), "")})
    assert exp2_biologia.rotulos_dispersao(casos) == {"aroA–aroB": ["aroA", "aroB"], "fabG": ["fabG"]}


def test_exp2_so_para_a_ecoli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["exp2_biologia", "--org", "bsu"])
    with pytest.raises(SystemExit):
        exp2_biologia.main()


def test_exp2_sem_a_lista_da_keio(resultados_temporarios, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQ_KEIO", tmp_path / "nao_existe.csv")
    with pytest.raises(FileNotFoundError, match="LEIAME"):
        exp2_biologia.rodar(ORG)
