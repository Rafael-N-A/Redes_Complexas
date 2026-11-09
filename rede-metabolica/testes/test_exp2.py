"""Exp 2 de ponta a ponta na E. coli (o único organismo com gabarito; roda em segundos).

As tabelas vão para uma pasta temporária; os valores são conferidos contra
as funções da biblioteca chamadas à parte.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

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
    B = grafo.grafo_bipartido(reacoes, remover_moeda=config.VERSOES_MOEDA[config.VERSAO_CASCATA])
    essenciais = validacao.carregar_keio()

    genes = ler("genes")
    assert list(genes["gene"]) == sorted(nocaute.reacoes_por_gene(reacoes))
    assert genes["essencial"].sum() == len(essenciais & set(genes["gene"]))
    dano = nocaute.dano_todos_genes(B, reacoes).set_index("gene")["dano"]
    assert (genes.set_index("gene")["dano"] == dano).all()

    assert list(avaliacao["ranking"]) == ["dano", "grau", "betweenness", "dano_meio_minimo", "reacoes_perdidas"]
    assert avaliacao["papel"].notna().all()
    y = genes.set_index("gene")["essencial"]
    for nome, coluna in (("dano", "dano"), ("grau", "grau"), ("reacoes_perdidas", "n_reacoes_perdidas")):
        esperado = validacao.auc(y, genes.set_index("gene")[coluna])
        assert math.isclose(avaliacao.set_index("ranking").loc[nome, "auc"], esperado)

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
    exp2_biologia.plotar(ORG)
    assert config.arquivo_figura("exp2", "roc", ORG).exists()


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


def test_exp2_so_para_a_ecoli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["exp2_biologia", "--org", "bsu"])
    with pytest.raises(SystemExit):
        exp2_biologia.main()


def test_exp2_sem_a_lista_da_keio(resultados_temporarios, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQ_KEIO", tmp_path / "nao_existe.csv")
    with pytest.raises(FileNotFoundError, match="LEIAME"):
        exp2_biologia.rodar(ORG)
