"""Regras do modelo em redes de brinquedo.

Cada teste monta uma rede de 2 a 4 metabólitos em que a resposta certa é
óbvia, e confere se a biblioteca chega nela. Rode na raiz do projeto:

    python -m pytest testes/

Enquanto as funções da biblioteca não estiverem implementadas, os testes
que dependem delas falham com NotImplementedError: essa é a lista do que
falta. O teste do contrato já passa.
"""
from __future__ import annotations

import pandas as pd

import config
import contrato
from coleta import montar_tabelas
from experimentos.biblioteca import cascata, grafo, nocaute

ATP, ADP = "C00002", "C00008"  # dois metabólitos-moeda de config.METABOLITOS_MOEDA


def tabela_reacoes(*reacoes) -> pd.DataFrame:
    """Monta uma tabela "reacoes" do contrato.

    Cada item: (codigo, substratos, produtos, reversivel, genes).
    """
    linhas = [
        {
            "reacao": codigo,
            "substratos": list(substratos),
            "produtos": list(produtos),
            "reversivel": reversivel,
            "genes": list(genes),
            "vias": ["eco00000"],
            "n_ocorrencias": 1,
            "conflito_reversibilidade": False,
            "orientacao_invertida": False,
            "lados_variam": False,
        }
        for codigo, substratos, produtos, reversivel, genes in reacoes
    ]
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


def test_regra_e_dos_substratos():
    """A + B -> C: sem B, a reação não ocorre e C se perde (como na citrato sintase)."""
    reacoes = tabela_reacoes(("R1", ["A", "B"], ["C"], False, ["g1"]))
    B = grafo.grafo_bipartido(reacoes)
    assert cascata.cascata(B, metabolitos_removidos=["B"]) == {"C"}


def test_regra_ou_das_produtoras():
    """C é produzido por R1 (de A) e por R2 (de D): sem A, R2 ainda produz C."""
    reacoes = tabela_reacoes(
        ("R1", ["A"], ["C"], False, ["g1"]),
        ("R2", ["D"], ["C"], False, ["g2"]),
    )
    B = grafo.grafo_bipartido(reacoes)
    assert cascata.cascata(B, metabolitos_removidos=["A"]) == set()


def test_regra_ou_das_isoenzimas():
    """R1 tem os genes g1 e g2: só some quando os dois são deletados."""
    reacoes = tabela_reacoes(("R1", ["A"], ["C"], False, ["g1", "g2"]))
    assert nocaute.reacoes_perdidas(["g1"], reacoes) == set()
    assert nocaute.reacoes_perdidas(["g1", "g2"], reacoes) == {"R1"}

    B = grafo.grafo_bipartido(reacoes)
    assert cascata.cascata(B, reacoes_removidas=["R1"]) == {"C"}


def test_ciclo_reversivel_nao_se_sustenta_sozinho():
    """X -> A <-> B, com X sem produtor: sem X, A e B se perdem.

    Partindo de "tudo vivo", A e B se sustentariam um ao outro pela reação
    reversível. É o caso que justifica a expansão a partir das fontes.
    """
    reacoes = tabela_reacoes(
        ("R1", ["X"], ["A"], False, ["g1"]),
        ("R2", ["A"], ["B"], True, ["g2"]),
    )
    B = grafo.grafo_bipartido(reacoes)
    assert cascata.cascata(B, metabolitos_removidos=["X"]) == {"A", "B"}


def test_moeda_sempre_disponivel_nas_duas_versoes():
    """A + ATP -> B + ADP e ADP -> ATP: ATP e ADP só se produzem um do outro.

    Sem a moeda entre as fontes, a versão com moeda nunca produziria B.
    Nas duas versões, B tem que estar disponível.
    """
    reacoes = tabela_reacoes(
        ("R1", ["A", ATP], ["B", ADP], False, ["g1"]),
        ("R2", [ADP], [ATP], False, ["g2"]),
    )
    for remover_moeda in (True, False):
        B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
        disponiveis, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
        assert "B" in disponiveis, f"remover_moeda={remover_moeda}"


def test_fusao_com_sentido_invertido():
    """A mesma reação irreversível desenhada A -> B numa via e B -> A em outra.

    Vira uma linha só, no sentido da primeira via, marcada como invertida e
    reversível, porque as duas direções aparecem no KEGG.
    """
    reacoes_por_via = pd.DataFrame(
        [
            {"via": "eco00010", "entrada_kgml": 1, "reacoes": ["R1"], "tipo": "irreversible",
             "substratos": ["A"], "produtos": ["B"], "genes": ["g1"]},
            {"via": "eco00020", "entrada_kgml": 7, "reacoes": ["R1"], "tipo": "irreversible",
             "substratos": ["B"], "produtos": ["A"], "genes": ["g2"]},
        ],
        columns=contrato.COLUNAS["reacoes_por_via"],
    )
    genes_por_via = pd.DataFrame(columns=contrato.COLUNAS["genes_por_via"])

    fundida = montar_tabelas.fundir_reacoes(reacoes_por_via, genes_por_via)

    assert list(fundida.columns) == contrato.COLUNAS["reacoes"]
    assert len(fundida) == 1
    linha = fundida.iloc[0]
    assert linha["reacao"] == "R1"
    assert linha["substratos"] == ["A"] and linha["produtos"] == ["B"]
    assert linha["orientacao_invertida"]
    assert linha["reversivel"]
    assert sorted(linha["genes"]) == ["g1", "g2"]
    assert linha["n_ocorrencias"] == 2


def test_fusao_vale_o_reversivel():
    """A primeira via desenha a reação reversível (A <-> B); a segunda, irreversível B -> A.

    Vale o reversível: a reação fica reversível e marcada com conflito. O
    sentido guardado é o da via irreversível (B -> A), para o grafo poder
    reconstruir a regra contrária no teste de sensibilidade.
    """
    reacoes_por_via = pd.DataFrame(
        [
            {"via": "eco00010", "entrada_kgml": 1, "reacoes": ["R1"], "tipo": "reversible",
             "substratos": ["A"], "produtos": ["B"], "genes": ["g1"]},
            {"via": "eco00020", "entrada_kgml": 7, "reacoes": ["R1"], "tipo": "irreversible",
             "substratos": ["B"], "produtos": ["A"], "genes": ["g1"]},
        ],
        columns=contrato.COLUNAS["reacoes_por_via"],
    )
    genes_por_via = pd.DataFrame(columns=contrato.COLUNAS["genes_por_via"])

    linha = montar_tabelas.fundir_reacoes(reacoes_por_via, genes_por_via).iloc[0]

    assert linha["substratos"] == ["B"] and linha["produtos"] == ["A"]
    assert linha["reversivel"]
    assert linha["conflito_reversibilidade"] and linha["orientacao_invertida"]


def test_contrato_ida_e_volta(tmp_path, monkeypatch):
    """salvar() e carregar() devolvem a mesma tabela, listas vazias incluídas."""
    monkeypatch.setattr(config, "DADOS", tmp_path)
    original = tabela_reacoes(
        ("R1", ["A", "B"], ["C"], True, []),
        ("R2", ["C"], ["D"], False, ["g1", "g2"]),
    )
    contrato.salvar(original, "teste", "reacoes")
    lida = contrato.carregar("teste", "reacoes")
    pd.testing.assert_frame_equal(lida, original)
