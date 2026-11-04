"""Coleta sem internet: KGML e respostas da API de mentira, gravados numa pasta temporária.

Conferem o que dá para conferir antes do primeiro download de verdade:
leitura do KGML, filtro das vias, cache, formato das listas e o caminho
completo da camada bruta até o relatório. Rode na raiz do projeto: pytest
"""
from __future__ import annotations

import json

import pytest

import config
from coleta import baixar_kgml, montar_tabelas

# Uma via de mentira com os casos que importam:
#   - prefixo de organismo que não é eco (bsu:), para o Exp 4;
#   - uma <reaction> com dois códigos R e dois genes na mesma entrada;
#   - um substrato sem o atributo name (vem da entrada) e um <alt> que é ignorado;
#   - uma entrada de gene que cita R00003, mas sem <reaction> na via (a lacuna);
#   - um glicano e uma entrada de mapa, que não é composto.
KGML = """<?xml version="1.0"?>
<!DOCTYPE pathway SYSTEM "https://www.kegg.jp/kegg/xml/KGML_v0.7.2_.dtd">
<pathway name="path:bsu00010" org="bsu" number="00010" title="Teste">
    <entry id="10" name="bsu:BSU00010 bsu:BSU00020" type="gene" reaction="rn:R00001 rn:R00002"/>
    <entry id="11" name="bsu:BSU00030" type="gene" reaction="rn:R00003"/>
    <entry id="20" name="cpd:C00031" type="compound"/>
    <entry id="21" name="cpd:C00092" type="compound"/>
    <entry id="23" name="cpd:C00111" type="compound"/>
    <entry id="24" name="gl:G00001" type="compound"/>
    <entry id="30" name="path:bsu00020" type="map"/>
    <reaction id="10" name="rn:R00001 rn:R00002" type="irreversible">
        <substrate id="20" name="cpd:C00031"/>
        <substrate id="23"/>
        <product id="21" name="cpd:C00092"><alt name="cpd:C99999"/></product>
    </reaction>
</pathway>
"""


@pytest.fixture
def dados_temporarios(tmp_path, monkeypatch):
    """Aponta config para uma pasta vazia; nada toca dados/ de verdade."""
    monkeypatch.setattr(config, "DADOS", tmp_path)
    monkeypatch.setattr(config, "ARQ_COMPOSTOS", tmp_path / "kgml" / "_compostos.tsv")
    monkeypatch.setattr(config, "ARQ_GLICANOS", tmp_path / "kgml" / "_glicanos.tsv")
    monkeypatch.setattr(config, "PAUSA_ENTRE_REQUISICOES", 0)
    return tmp_path


def test_ler_kgml(tmp_path):
    arquivo = tmp_path / "bsu00010.xml"
    arquivo.write_text(KGML, encoding="utf-8")

    reacoes, genes, compostos = montar_tabelas.ler_kgml(arquivo)

    assert reacoes == [{
        "via": "bsu00010", "entrada_kgml": 10, "reacoes": ["R00001", "R00002"],
        "tipo": "irreversible", "substratos": ["C00031", "C00111"], "produtos": ["C00092"],
        "genes": ["BSU00010", "BSU00020"],
    }]
    assert genes == [
        {"via": "bsu00010", "entrada_kgml": 10, "genes": ["BSU00010", "BSU00020"],
         "reacoes": ["R00001", "R00002"]},
        {"via": "bsu00010", "entrada_kgml": 11, "genes": ["BSU00030"], "reacoes": ["R00003"]},
    ]
    assert [c["metabolito"] for c in compostos] == ["C00031", "C00092", "C00111", "G00001"]


def test_filtro_das_vias():
    vias = [("eco00010", "Glicólise"), ("eco00970", "Aminoacyl-tRNA"), ("eco01040", "Ácidos graxos"),
            ("eco01100", "Mapa global"), ("eco01200", "Carbono"), ("eco02010", "Transportadores ABC")]
    assert [c for c, _ in baixar_kgml.filtrar_vias_regulares(vias)] == ["eco00010", "eco01040"]


def test_cache_e_validacao(dados_temporarios, monkeypatch):
    chamadas = []

    def falsa(caminho_api):
        chamadas.append(caminho_api)
        return KGML

    monkeypatch.setattr(baixar_kgml, "requisitar", falsa)
    destino = dados_temporarios / "kgml" / "bsu" / "bsu00010.xml"

    baixar_kgml.baixar_com_cache("get/bsu00010/kgml", destino)
    baixar_kgml.baixar_com_cache("get/bsu00010/kgml", destino)
    assert chamadas == ["get/bsu00010/kgml"]  # a segunda veio do cache

    recusado = dados_temporarios / "recusado.xml"
    with pytest.raises(baixar_kgml.ErroKEGG):
        baixar_kgml.baixar_com_cache("x", recusado, validar=lambda texto: False)
    assert not recusado.exists() and not list(dados_temporarios.glob("*.parcial"))


def test_via_sem_kgml_vira_none(dados_temporarios, monkeypatch):
    def nao_encontrada(caminho_api):
        raise baixar_kgml.ErroKEGG("HTTP 404", status=404)

    monkeypatch.setattr(baixar_kgml, "requisitar", nao_encontrada)
    assert baixar_kgml.baixar_kgml("bsu", "bsu99999") is None


def test_formatos_da_lista_de_genes(dados_temporarios):
    pasta = config.pasta_brutos("eco")
    pasta.mkdir(parents=True)
    (pasta / "_genes.tsv").write_text(
        "eco:b0001\tCDS\t190..255\tthrL; thr operon leader peptide\n"
        "eco:b4810\tCDS\t1..90\tuncharacterized protein\n"
        "eco:b0114\taceE; pyruvate dehydrogenase E1 component\n",
        encoding="utf-8",
    )
    genes = montar_tabelas.montar_genes("eco").to_dict("records")
    assert genes == [
        {"gene": "b0001", "simbolo": "thrL", "descricao": "thr operon leader peptide"},
        {"gene": "b4810", "simbolo": "", "descricao": "uncharacterized protein"},
        {"gene": "b0114", "simbolo": "aceE", "descricao": "pyruvate dehydrogenase E1 component"},
    ]


def test_da_camada_bruta_ao_relatorio(dados_temporarios):
    pasta = config.pasta_brutos("bsu")
    pasta.mkdir(parents=True)
    (pasta / "bsu00010.xml").write_text(KGML, encoding="utf-8")
    (pasta / "bsu00020.xml").write_text("isto não é XML", encoding="utf-8")
    metadados = {"org": "bsu", "vias": [{"codigo": "bsu00010", "nome": "Teste"},
                                        {"codigo": "bsu00020", "nome": "Quebrada"}]}
    (pasta / "_metadados.json").write_text(json.dumps(metadados), encoding="utf-8")
    config.ARQ_COMPOSTOS.write_text("cpd:C00031\tD-Glucose; Grape sugar\nC00092\tD-Glucose 6-phosphate\n",
                                    encoding="utf-8")

    rpv, gpv, cpv, falhas = montar_tabelas.montar_camada_bruta("bsu")
    reacoes = montar_tabelas.fundir_reacoes(rpv, gpv)
    metabolitos = montar_tabelas.montar_metabolitos(cpv, reacoes)
    rel = montar_tabelas.relatorio(rpv, gpv, reacoes, metabolitos, montar_tabelas.montar_genes("bsu"),
                                   falhas, metadados)

    assert [f["via"] for f in falhas] == ["bsu00020"]  # o arquivo quebrado não derruba os outros
    assert list(reacoes["reacao"]) == ["R00001", "R00002"]
    assert all(g == ["BSU00010", "BSU00020"] for g in reacoes["genes"])
    nomes = dict(zip(metabolitos["metabolito"], metabolitos["nome"]))
    assert nomes["C00031"] == "D-Glucose" and nomes["C00092"] == "D-Glucose 6-phosphate"

    assert rel["lacuna_reaction"]["sem_reaction_na_mesma_via"] == 1
    assert rel["lacuna_reaction"]["exemplos_codigos"] == ["R00003"]
    assert all(c["ok"] for c in rel["conferencias"].values())
    json.dumps(rel)  # o relatório precisa poder ser gravado em JSON
