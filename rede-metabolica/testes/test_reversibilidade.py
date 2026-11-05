"""A regra "vale o reversível", da camada bruta até as métricas.

Reproduz o caso real que motivou a regra. No KEGG da E. coli, a succinato
desidrogenase (R02164) e a succinil-CoA sintetase (R00405) aparecem
reversíveis no ciclo de Krebs (eco00020), mas irreversíveis no sentido
redutivo em outras vias: fumarato -> succinato no eco00620 e
succinato -> succinil-CoA no eco00720.

  - Vale o reversível (a regra da coleta): o ciclo fecha, os 7 no núcleo.
  - Vale o irreversível (conflito_como_irreversivel=True): succinato e
    succinil-CoA saem do núcleo, como aconteceu na primeira versão da rede.

Rode na raiz do projeto: pytest
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
import contrato
from coleta import montar_tabelas
from experimentos.biblioteca import grafo, metricas

OAA, CIT, OG, SCOA, SUC, FUM, MAL = "C00036", "C00158", "C00026", "C00091", "C00042", "C00122", "C00149"
KREBS = {OAA, CIT, OG, SCOA, SUC, FUM, MAL}

# (via, entrada, código, tipo, substratos, produtos), como o KGML desenha cada via
BRUTA = [
    ("eco00020", 1, "R90001", "irreversible", [OAA], [CIT]),   # citrato sintase
    ("eco00020", 2, "R90002", "irreversible", [CIT], [OG]),    # até o 2-oxoglutarato (resumido)
    ("eco00020", 3, "R90003", "irreversible", [OG], [SCOA]),   # 2-oxoglutarato desidrogenase
    ("eco00020", 4, "R00405", "reversible", [SUC], [SCOA]),
    ("eco00020", 5, "R02164", "reversible", [SUC], [FUM]),
    ("eco00020", 6, "R01082", "reversible", [MAL], [FUM]),
    ("eco00020", 7, "R90004", "reversible", [MAL], [OAA]),     # malato desidrogenase
    ("eco00620", 1, "R02164", "irreversible", [FUM], [SUC]),   # fumarato redutase (anaeróbio)
    ("eco00720", 1, "R00405", "irreversible", [SUC], [SCOA]),  # ciclo de Krebs redutivo
    ("eco00720", 2, "R02164", "reversible", [SUC], [FUM]),
]


def camada_bruta(linhas) -> tuple[pd.DataFrame, pd.DataFrame]:
    reacoes_por_via = pd.DataFrame(
        [{"via": via, "entrada_kgml": e, "reacoes": [c], "tipo": tipo, "substratos": s,
          "produtos": p, "genes": [f"g_{c}"]} for via, e, c, tipo, s, p in linhas],
        columns=contrato.COLUNAS["reacoes_por_via"],
    )
    return reacoes_por_via, pd.DataFrame(columns=contrato.COLUNAS["genes_por_via"])


@pytest.fixture
def reacoes():
    return montar_tabelas.fundir_reacoes(*camada_bruta(BRUTA)).set_index("reacao", drop=False)


# ---------------------------------------------------------------------------
# A regra, caso a caso (a tabela das quatro combinações)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tipos_e_sentidos, reversivel_esperado, conflito_esperado", [
    ([("reversible", "ida"), ("reversible", "ida")], True, False),
    ([("irreversible", "ida"), ("irreversible", "ida")], False, False),
    ([("irreversible", "ida"), ("irreversible", "volta")], True, False),
    ([("reversible", "ida"), ("irreversible", "volta")], True, True),
])
def test_as_quatro_combinacoes(tipos_e_sentidos, reversivel_esperado, conflito_esperado):
    linhas = [(f"eco000{i}0", 1, "R1", tipo, ["A"], ["B"]) if sentido == "ida"
              else (f"eco000{i}0", 1, "R1", tipo, ["B"], ["A"])
              for i, (tipo, sentido) in enumerate(tipos_e_sentidos, start=1)]
    linha = montar_tabelas.fundir_reacoes(*camada_bruta(linhas)).iloc[0]
    assert linha["reversivel"] == reversivel_esperado
    assert linha["conflito_reversibilidade"] == conflito_esperado


def test_fusao_do_caso_real(reacoes):
    for codigo in ("R00405", "R02164"):
        assert reacoes.loc[codigo, "reversivel"] and reacoes.loc[codigo, "conflito_reversibilidade"]
    # o sentido guardado é o da primeira via irreversível
    assert reacoes.loc["R02164", "substratos"] == [FUM] and reacoes.loc["R02164", "produtos"] == [SUC]
    assert reacoes.loc["R00405", "substratos"] == [SUC] and reacoes.loc["R00405", "produtos"] == [SCOA]
    assert not reacoes.loc["R90001", "reversivel"]  # irreversível nas vias em que aparece


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------
def test_grafo_com_as_duas_regras(reacoes):
    padrao = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes))
    assert padrao.has_edge(SUC, FUM) and padrao.has_edge(FUM, SUC)
    assert padrao.has_edge(SCOA, SUC) and padrao.has_edge(SUC, SCOA)

    antiga = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, conflito_como_irreversivel=True))
    assert antiga.has_edge(FUM, SUC) and not antiga.has_edge(SUC, FUM)
    assert antiga.has_edge(SUC, SCOA) and not antiga.has_edge(SCOA, SUC)
    assert antiga.has_edge(MAL, FUM) and antiga.has_edge(FUM, MAL)  # sem conflito: não muda

    so_irrev = grafo.grafo_metabolitos(reacoes, so_irreversiveis=True, conflito_como_irreversivel=True)
    assert so_irrev.has_edge(FUM, SUC) and not so_irrev.has_edge(MAL, FUM)


def test_bipartido_com_as_duas_regras(reacoes):
    padrao = grafo.grafo_bipartido(reacoes)
    antiga = grafo.grafo_bipartido(reacoes, conflito_como_irreversivel=True)
    assert {"R02164:volta", "R00405:volta"} <= grafo.nos_reacao(padrao)
    assert not {"R02164:volta", "R00405:volta"} & grafo.nos_reacao(antiga)
    assert set(antiga.predecessors("R02164:ida")) == {FUM} and set(antiga.successors("R02164:ida")) == {SUC}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_ciclo_de_krebs_fecha_so_com_vale_o_reversivel(reacoes):
    padrao = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes))
    assert metricas.maior_componente_forte(padrao) == KREBS
    assert metricas.bow_tie(padrao) == {"nucleo": 7, "entrada": 0, "saida": 0, "restante": 0}

    antiga = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, conflito_como_irreversivel=True))
    partes = metricas.componentes_bow_tie(antiga)
    assert partes["nucleo"] == {FUM, MAL, OAA}
    assert partes["saida"] == {CIT, OG, SCOA, SUC}  # o núcleo chega neles, eles não voltam

    caminho_padrao, _ = metricas.caminho_medio_e_diametro(padrao)
    caminho_antiga, _ = metricas.caminho_medio_e_diametro(antiga)
    assert caminho_antiga < caminho_padrao  # núcleo menor, caminhos mais curtos dentro dele


def test_choke_points_com_as_duas_regras(reacoes):
    padrao = metricas.choke_points(reacoes)
    antiga = metricas.choke_points(reacoes, conflito_como_irreversivel=True)
    # com a regra antiga, R02164 vira a única produtora do succinato
    assert "R02164" in antiga
    assert padrao != antiga


def test_relatorio_confere_a_regra(reacoes, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARQ_COMPOSTOS", tmp_path / "nao_existe.tsv")
    monkeypatch.setattr(config, "ARQ_GLICANOS", tmp_path / "nao_existe.tsv")
    reacoes_por_via, genes_por_via = camada_bruta(BRUTA)
    tabela = reacoes.reset_index(drop=True)
    metabolitos = montar_tabelas.montar_metabolitos(pd.DataFrame(columns=["via", "metabolito"]), tabela)
    genes = pd.DataFrame(columns=contrato.COLUNAS["genes"])

    rel = montar_tabelas.relatorio(reacoes_por_via, genes_por_via, tabela, metabolitos, genes, [],
                                   {"vias": [{"codigo": v} for v in ("eco00020", "eco00620", "eco00720")]})

    assert rel["fusao"]["conflito_reversibilidade"] == 2
    assert rel["fusao"]["reversiveis_por_sentidos_opostos"] == 0
    assert rel["fusao"]["irreversiveis_com_sentido_de_outra_via"] == 0
    assert all(c["ok"] for c in rel["conferencias"].values())
