"""Exp 0 e Exp 1 de ponta a ponta, na rede do mge (pequena, roda em segundos).

As tabelas e figuras vão para uma pasta temporária: resultados/ não muda.
Os valores são conferidos contra as funções da biblioteca chamadas à parte.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from experimentos import exp0_descritiva, exp1_robustez
from experimentos.biblioteca import grafo, metricas, robustez

ORG = "mge"


@pytest.fixture
def resultados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_TABELAS", tmp_path / "tabelas")
    monkeypatch.setattr(config, "PASTA_FIGURAS", tmp_path / "figuras")
    return tmp_path


def grafo_simples(versao: str):
    reacoes = grafo.carregar_tabelas(ORG)["reacoes"]
    return grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))


def test_exp0_tabelas_e_figura(resultados_temporarios):
    linha = exp0_descritiva.rodar(ORG, "sem_moeda")
    G = grafo_simples("sem_moeda")
    esperado = metricas.tabela_descritiva(G)
    for chave, valor in esperado.items():
        assert linha[chave] == pytest.approx(valor, nan_ok=True), chave
    assert linha["reacoes"] == len(grafo.carregar_tabelas(ORG)["reacoes"])

    tabela1 = pd.read_csv(config.arquivo_tabela("exp0", "tabela1", ORG, "sem_moeda"))
    assert tabela1["nucleo"].item() == esperado["nucleo"]
    ajuste = metricas.ajuste_lei_potencia(metricas.graus(G)["total"])
    assert tabela1["lp_alpha"].item() == pytest.approx(ajuste["alpha"])

    bowtie = pd.read_csv(config.arquivo_tabela("exp0", "bowtie", ORG, "sem_moeda"))
    assert len(bowtie) == G.number_of_nodes()
    assert bowtie["componente"].value_counts().to_dict() == {k: v for k, v in metricas.bow_tie(G).items() if v}
    assert list(bowtie["componente"].unique()) == [c for c in exp0_descritiva.ORDEM_BOWTIE
                                                   if c in set(bowtie["componente"])]

    top10 = pd.read_csv(config.arquivo_tabela("exp0", "top10", ORG, "sem_moeda"))
    assert len(top10) == 10 and top10["total"].is_monotonic_decreasing
    lei = pd.read_csv(config.arquivo_tabela("exp0", "lei_potencia", ORG, "sem_moeda"))
    assert list(lei["tipo"]) == list(exp0_descritiva.TIPOS_GRAU)

    exp0_descritiva.plotar(ORG, "sem_moeda")
    assert config.arquivo_figura("exp0", "graus", ORG, "sem_moeda").exists()


def test_exp0_combina_as_duas_versoes(resultados_temporarios):
    assert exp0_descritiva.combinar_versoes(ORG) is None    # ainda não há nenhuma versão
    linhas = {v: exp0_descritiva.rodar(ORG, v) for v in config.VERSOES_MOEDA}
    combinada = exp0_descritiva.combinar_versoes(ORG)
    assert list(combinada.columns) == ["metrica", *config.VERSOES_MOEDA]
    combinada = combinada.set_index("metrica")
    for versao, linha in linhas.items():
        assert combinada.loc["nos", versao] == linha["nos"]
    assert config.arquivo_tabela("exp0", "tabela1", ORG).exists()


def test_exp1_curvas_indices_e_figuras(resultados_temporarios, monkeypatch):
    monkeypatch.setattr(config, "N_REPETICOES_ALEATORIAS", 3)
    indices = exp1_robustez.rodar(ORG, "sem_moeda")
    curvas = pd.read_csv(config.arquivo_tabela("exp1", "curvas", ORG, "sem_moeda"))
    G = grafo_simples("sem_moeda")
    assert list(indices["estrategia"]) == list(config.ESTRATEGIAS)
    assert (curvas.groupby("estrategia").size() == G.number_of_nodes() + 1).all()
    assert indices.set_index("estrategia").loc["aleatoria", "n_rep"] == 3

    # O mesmo que chamar a biblioteca direto, com a mesma semente.
    _, direto = robustez.comparar_estrategias(G, ["aleatoria", "grau"], n_rep=3, semente=config.SEMENTE,
                                              nutrientes=config.NUTRIENTES)
    gravado = pd.read_csv(config.arquivo_tabela("exp1", "indices", ORG, "sem_moeda")).set_index("estrategia")
    for linha in direto.to_dict("records"):
        assert math.isclose(gravado.loc[linha["estrategia"], "R_fraco"], linha["R_fraco"])
        assert math.isclose(gravado.loc[linha["estrategia"], "metade_fraco"], linha["metade_fraco"])

    exp1_robustez.plotar(ORG, "sem_moeda")
    for conteudo in (*exp1_robustez.METRICAS_FIGURA, "painel"):
        assert config.arquivo_figura("exp1", conteudo, ORG, "sem_moeda").exists(), conteudo


def test_exp1_roda_so_sem_moeda_por_padrao(monkeypatch):
    chamadas = []
    monkeypatch.setattr(exp1_robustez, "rodar", lambda org, versao: chamadas.append(versao) or pd.DataFrame(
        columns=robustez.COLUNAS_INDICES))
    monkeypatch.setattr(exp1_robustez, "plotar", lambda org, versao: None)
    monkeypatch.setattr("sys.argv", ["exp1_robustez", "--org", ORG])
    exp1_robustez.main()
    assert chamadas == [config.VERSAO_ROBUSTEZ]


def test_exp1_centralidades_novas_e_r_normalizado(resultados_temporarios, monkeypatch):
    monkeypatch.setattr(config, "N_REPETICOES_ALEATORIAS", 2)
    indices = exp1_robustez.rodar(ORG, "sem_moeda")
    assert set(indices["estrategia"]) == set(config.ESTRATEGIAS)
    assert not any("harmonica" in e for e in indices["estrategia"])   # a harmônica saiu da remoção
    assert (indices["R_fraco_norm"] == indices["R_fraco"] / indices["S0_fraco"]).all()
    assert (indices["R_forte_norm"] == indices["R_forte"] / indices["S0_forte"]).all()
    exp1_robustez.plotar(ORG, "sem_moeda")
    assert config.arquivo_figura("exp1", "estatica_adaptativa", ORG, "sem_moeda").exists()
    assert config.arquivo_figura("exp1", "maior_fraco_inteira", ORG, "sem_moeda").exists()
    assert "R_fraco_norm" in exp1_robustez._resumo(indices)
