"""Nocaute de genes (nocaute.py) e métricas de acerto contra a Keio (validacao.py).

Redes e rankings de brinquedo com a resposta óbvia, e as métricas conferidas
contra o sklearn e o scipy chamados à parte.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import config
import contrato
from experimentos.biblioteca import cascata, grafo, nocaute, validacao


def tabela(*reacoes) -> pd.DataFrame:
    """Cada item: (codigo, substratos, produtos, genes)."""
    linhas = [{"reacao": c, "substratos": list(s), "produtos": list(p), "reversivel": False,
               "genes": list(g), "vias": ["eco00000"], "n_ocorrencias": 1,
               "conflito_reversibilidade": False, "orientacao_invertida": False, "lados_variam": False}
              for c, s, p, g in reacoes]
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


# X -> A (g1) -> B (g2 ou g3, isoenzimas) -> C (g4); A -> D (g4); D -> E sem gene
REDE = tabela(("R1", ["X"], ["A"], ["g1"]), ("R2", ["A"], ["B"], ["g2", "g3"]),
              ("R3", ["B"], ["C"], ["g4"]), ("R4", ["A"], ["D"], ["g4"]), ("R5", ["D"], ["E"], []))


# ---------------------------------------------------------------------------
# nocaute.py
# ---------------------------------------------------------------------------
def test_reacoes_por_gene():
    assert nocaute.reacoes_por_gene(REDE) == {"g1": {"R1"}, "g2": {"R2"}, "g3": {"R2"}, "g4": {"R3", "R4"}}


def test_reacoes_perdidas_regra_ou_e_reacao_sem_gene():
    assert nocaute.reacoes_perdidas(["g2"], REDE) == set()          # g3 ainda catalisa R2
    assert nocaute.reacoes_perdidas(["g2", "g3"], REDE) == {"R2"}
    assert nocaute.reacoes_perdidas(["g4"], REDE) == {"R3", "R4"}
    todos = ["g1", "g2", "g3", "g4"]
    assert "R5" not in nocaute.reacoes_perdidas(todos, REDE)          # sem gene: nunca some


def test_dano_de_cada_gene():
    B = grafo.grafo_bipartido(REDE)
    dano = nocaute.dano_todos_genes(B, REDE, medida="avalanche").set_index("gene")
    assert list(dano.index) == ["g1", "g2", "g3", "g4"]
    assert dano["dano"].to_dict() == {"g1": 5, "g2": 0, "g3": 0, "g4": 3}   # g1 derruba A, B, C, D, E
    assert dano["n_reacoes"].to_dict() == {"g1": 1, "g2": 1, "g3": 1, "g4": 2}
    assert dano["n_reacoes_perdidas"].to_dict() == {"g1": 1, "g2": 0, "g3": 0, "g4": 2}
    for gene in dano.index:
        assert dano.loc[gene, "dano"] == nocaute.dano_nocaute(B, REDE, [gene], "avalanche")
    assert nocaute.dano_nocaute(B, REDE, ["g2", "g3"], "avalanche") == 2  # B e C


def test_fontes_de_cada_medida(monkeypatch):
    B = grafo.grafo_bipartido(REDE)
    assert nocaute.fontes_da_medida(B, "avalanche") == cascata.fontes_externas(B)
    monkeypatch.setattr(config, "NUTRIENTES", {"A": "nutriente de teste"})
    assert nocaute.fontes_da_medida(B, "alcance") == {"A"}
    # Partindo só de A, X nunca está disponível e o nocaute de g1 não muda nada.
    assert nocaute.dano_nocaute(B, REDE, ["g1"], "alcance") == 0
    with pytest.raises(ValueError):
        nocaute.fontes_da_medida(B, "outra")


def test_meio_minimo_inclui_a_moeda_na_versao_com_moeda(monkeypatch):
    """Na versão com moeda, o ATP fica em B e precisa ser fonte também no meio mínimo."""
    atp = "C00002"
    rede = tabela(("R1", ["N", atp], ["A"], ["g1"]))
    monkeypatch.setattr(config, "NUTRIENTES", {"N": "nutriente de teste"})
    B = grafo.grafo_bipartido(rede, remover_moeda=False)
    assert nocaute.fontes_da_medida(B, "alcance") == {"N", atp}
    assert nocaute.dano_nocaute(B, rede, ["g1"], "alcance") == 1


def test_centralidade_por_gene():
    valores = pd.Series({"X": 1.0, "A": 5.0, "B": 2.0, "C": 7.0, "D": 3.0})  # E fora: conta 0
    maximo = nocaute.centralidade_por_gene(REDE, valores, "max")
    assert maximo.to_dict() == {"g1": 5.0, "g2": 5.0, "g3": 5.0, "g4": 7.0}
    soma = nocaute.centralidade_por_gene(REDE, valores, "soma")
    assert soma.to_dict() == {"g1": 6.0, "g2": 7.0, "g3": 7.0, "g4": 2 + 7 + 5 + 3}
    with pytest.raises(ValueError):
        nocaute.centralidade_por_gene(REDE, valores, "media")


def test_genes_em_choke_points():
    assert nocaute.genes_em_choke_points(REDE, {"R4"}).to_dict() == {"g1": False, "g2": False,
                                                                     "g3": False, "g4": True}


# ---------------------------------------------------------------------------
# validacao.py
# ---------------------------------------------------------------------------
def test_carregar_keio(tmp_path):
    arquivo = tmp_path / "essenciais.csv"
    arquivo.write_text("gene\nb0001\n b0002 \n\nb0001\n", encoding="utf-8")
    assert validacao.carregar_keio(arquivo) == {"b0001", "b0002"}
    with pytest.raises(FileNotFoundError, match="converter_keio"):
        validacao.carregar_keio(tmp_path / "falta.csv")
    errado = tmp_path / "errado.csv"
    errado.write_text("b_number\nb0001\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validacao.carregar_keio(errado)


def test_rotulos_so_dos_genes_da_rede():
    y = validacao.rotulos(["g1", "g2", "g3"], {"g2", "fora"})
    assert y.to_dict() == {"g1": False, "g2": True, "g3": False} and y.dtype == bool


def test_auc_casos_obvios_e_alinhamento():
    y = validacao.rotulos(["a", "b", "c", "d"], {"a", "b"})
    assert validacao.auc(y, pd.Series({"d": 0, "c": 1, "b": 2, "a": 3})) == 1.0   # ordem do índice não importa
    assert validacao.auc(y, pd.Series({"a": 0, "b": 0, "c": 5, "d": 5})) == 0.0
    assert validacao.auc(y, pd.Series(0.0, index=y.index)) == 0.5                 # empate total
    assert validacao.auc(y, pd.Series({"a": 2, "b": 2})) == 1.0                   # gene sem pontuação vale 0
    assert math.isnan(validacao.auc(validacao.rotulos(["a"], {"a"}), pd.Series({"a": 1})))


def test_teste_auc_contra_scipy_e_sklearn():
    rng = np.random.default_rng(4)
    genes = [f"g{i:03d}" for i in range(200)]
    y = validacao.rotulos(genes, set(genes[:30]))
    pontuacao = pd.Series(rng.normal(size=200) + y.to_numpy() * 0.8, index=genes)
    teste = validacao.teste_auc(y, pontuacao)
    assert math.isclose(teste["auc"], roc_auc_score(y, pontuacao))
    esperado = mannwhitneyu(pontuacao[y], pontuacao[~y], alternative="two-sided").pvalue
    assert math.isclose(teste["p"], esperado)
    assert teste["ic_inf"] < teste["auc"] < teste["ic_sup"]


def test_precisao_k_desempata_pelo_gene():
    y = validacao.rotulos(["a", "b", "c", "d"], {"b"})
    pontuacao = pd.Series({"d": 1, "c": 1, "b": 1, "a": 2})
    assert validacao.precisao_k(y, pontuacao, 1) == 0.0            # a
    assert validacao.precisao_k(y, pontuacao, 2) == 0.5            # a, b (b vem antes de c e d)
    assert validacao.precisao_k(y, pontuacao, 10) == 0.25          # k maior que o número de genes


def test_avaliar_rankings_e_curva_roc():
    y = validacao.rotulos(["a", "b", "c", "d"], {"a"})
    rankings = {"bom": pd.Series({"a": 3, "b": 2, "c": 1, "d": 0}),
                "ruim": pd.Series({"a": 0, "b": 2, "c": 1, "d": 3})}
    tabela_ = validacao.avaliar_rankings(y, rankings, ks=(1, 2))
    assert list(tabela_.columns) == ["ranking", "auc", "auc_ic_inf", "auc_ic_sup", "p_auc", "precisao_1",
                                     "precisao_2", "n_genes", "n_essenciais", "taxa_base"]
    assert tabela_.set_index("ranking")["auc"].to_dict() == {"bom": 1.0, "ruim": 0.0}
    assert tabela_["taxa_base"].iloc[0] == 0.25
    roc = validacao.curva_roc(y, rankings["bom"])
    assert (roc.iloc[0]["fpr"], roc.iloc[0]["tpr"]) == (0.0, 0.0)
    assert (roc.iloc[-1]["fpr"], roc.iloc[-1]["tpr"]) == (1.0, 1.0)
