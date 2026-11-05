"""Coleta, passo 2: transforma os KGML baixados nas tabelas do contrato.

Lê dados/kgml/<org>/ e grava em dados/<org>/:
    reacoes_por_via.csv, genes_por_via.csv   camada bruta (nada fundido)
    reacoes.csv, metabolitos.csv, genes.csv  camada fundida
    relatorio_extracao.json                  contagens, conflitos e conferências

O formato de cada tabela está em contrato.py. Só são lidas as vias listadas
em dados/kgml/<org>/_metadados.json, que o baixar_kgml grava.

Uso, na raiz do projeto:
    python -m coleta.montar_tabelas --org eco
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

import config
import contrato


# ---------------------------------------------------------------------------
# Leitura do KGML
# ---------------------------------------------------------------------------
def _unicos(itens: Iterable[str]) -> list[str]:
    """Remove repetidos mantendo a ordem de aparição."""
    vistos: dict[str, None] = {}
    for item in itens:
        if item:
            vistos.setdefault(item, None)
    return list(vistos)


def _codigo(nome: str) -> str:
    """Tira o prefixo do KEGG: "cpd:C00024" -> "C00024", "bsu:BSU00010" -> "BSU00010"."""
    return nome.strip().split(":", 1)[-1]


def _codigos(texto: str | None) -> list[str]:
    """Separa um atributo com vários nomes ("rn:R00710 rn:R00711") e tira os prefixos."""
    return _unicos(_codigo(parte) for parte in (texto or "").split())


def _participantes(reacao: ET.Element, papel: str, entradas: dict[str, ET.Element]) -> list[str]:
    """Códigos dos <substrate> ou <product> de uma reação, na ordem do KGML.

    Usa o atributo name do próprio elemento; se faltar, o name da entrada
    apontada pelo id. Os elementos <alt> (nomes alternativos) são ignorados.
    """
    codigos: list[str] = []
    for elemento in reacao.findall(papel):
        nome = elemento.get("name")
        if not nome:
            entrada = entradas.get(elemento.get("id", ""))
            nome = entrada.get("name") if entrada is not None else ""
        codigos.extend(_codigos(nome))
    return _unicos(codigos)


def ler_kgml(caminho: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Lê um KGML e devolve três listas de linhas, sem fundir nada.

    Retorna (reacoes, genes, compostos):
      - reacoes:   uma linha por <reaction>, com as colunas de "reacoes_por_via"
      - genes:     uma linha por <entry type="gene">, colunas de "genes_por_via"
      - compostos: uma linha por <entry type="compound">: {"via", "metabolito"}

    Lê o XML com xml.etree, e não com o Bio.KEGG.KGML do Biopython, por dois
    motivos: o Biopython guarda substratos e produtos num conjunto (perde a
    ordem do KGML) e recusa o arquivo inteiro se uma reação apontar para uma
    entrada que não existe.

    Os prefixos rn:, cpd:, gl: e o do organismo (eco:, bsu:, mge:...) saem
    pelo texto depois do primeiro ":", e não por um prefixo fixo, para que
    bsu e mge também tenham genes. Levanta ET.ParseError ou ValueError se o
    arquivo não for um KGML válido.
    """
    raiz = ET.parse(caminho).getroot()
    if raiz.tag != "pathway":
        raise ValueError(f"a raiz do XML é <{raiz.tag}>, esperado <pathway>")
    via = _codigo(raiz.get("name", "")) or caminho.stem
    entradas = {e.get("id", ""): e for e in raiz.findall("entry")}

    reacoes = []
    for elemento in raiz.findall("reaction"):
        id_reacao = elemento.get("id", "")
        entrada = entradas.get(id_reacao)
        ligada_a_gene = entrada is not None and entrada.get("type") == "gene"
        reacoes.append({
            "via": via,
            "entrada_kgml": int(id_reacao),
            "reacoes": _codigos(elemento.get("name")),
            "tipo": elemento.get("type", ""),
            "substratos": _participantes(elemento, "substrate", entradas),
            "produtos": _participantes(elemento, "product", entradas),
            "genes": _codigos(entrada.get("name")) if ligada_a_gene else [],
        })

    genes = []
    compostos = []
    for id_entrada, entrada in entradas.items():
        tipo = entrada.get("type")
        if tipo == "gene":
            genes.append({
                "via": via,
                "entrada_kgml": int(id_entrada),
                "genes": _codigos(entrada.get("name")),
                "reacoes": _codigos(entrada.get("reaction")),
            })
        elif tipo == "compound":
            compostos.extend({"via": via, "metabolito": c} for c in _codigos(entrada.get("name")))

    return reacoes, genes, compostos


def ler_metadados(org: str) -> dict:
    """Lê dados/kgml/<org>/_metadados.json, gravado pelo baixar_kgml."""
    arquivo = config.pasta_brutos(org) / "_metadados.json"
    if not arquivo.exists():
        raise FileNotFoundError(f"{config.caminho_legivel(arquivo)} não existe. "
                                f"Rode antes: python -m coleta.baixar_kgml --org {org}")
    return json.loads(arquivo.read_text(encoding="utf-8"))


def montar_camada_bruta(org: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Aplica ler_kgml a todas as vias listadas nos metadados do organismo.

    Retorna (reacoes_por_via, genes_por_via, compostos_por_via, falhas_leitura),
    as duas primeiras já no formato do contrato. Um arquivo que não dá para
    ler entra em falhas_leitura e não interrompe os outros.
    """
    pasta = config.pasta_brutos(org)
    linhas_reacoes: list[dict] = []
    linhas_genes: list[dict] = []
    linhas_compostos: list[dict] = []
    falhas: list[dict] = []

    for via in (v["codigo"] for v in ler_metadados(org)["vias"]):
        arquivo = pasta / f"{via}.xml"
        if not arquivo.exists():
            falhas.append({"via": via, "motivo": "o arquivo não está no cache"})
            continue
        try:
            reacoes, genes, compostos = ler_kgml(arquivo)
        except (ET.ParseError, ValueError) as erro:
            falhas.append({"via": via, "motivo": f"{erro.__class__.__name__}: {erro}"})
            continue
        linhas_reacoes.extend(reacoes)
        linhas_genes.extend(genes)
        linhas_compostos.extend(compostos)

    return (
        pd.DataFrame(linhas_reacoes, columns=contrato.COLUNAS["reacoes_por_via"]),
        pd.DataFrame(linhas_genes, columns=contrato.COLUNAS["genes_por_via"]),
        pd.DataFrame(linhas_compostos, columns=["via", "metabolito"]),
        falhas,
    )


# ---------------------------------------------------------------------------
# Camada fundida
# ---------------------------------------------------------------------------
def _ocorrencias_por_codigo(reacoes_por_via: pd.DataFrame) -> dict[str, list[dict]]:
    """Regra 1: código R -> linhas brutas em que ele aparece, ordenadas por (via, entrada)."""
    linhas = sorted(reacoes_por_via.to_dict("records"),
                    key=lambda linha: (linha["via"], int(linha["entrada_kgml"])))
    ocorrencias: dict[str, list[dict]] = defaultdict(list)
    for linha in linhas:
        for codigo in linha["reacoes"]:
            ocorrencias[codigo].append(linha)
    return ocorrencias


def _esta_invertida(substratos: Iterable[str], produtos: Iterable[str],
                    ref_substratos: set[str], ref_produtos: set[str]) -> bool:
    """True se os lados coincidem mais com os da referência trocados do que como estão."""
    s, p = set(substratos), set(produtos)
    direto = len(s & ref_substratos) + len(p & ref_produtos)
    inverso = len(s & ref_produtos) + len(p & ref_substratos)
    return inverso > direto


def fundir_reacoes(reacoes_por_via: pd.DataFrame, genes_por_via: pd.DataFrame) -> pd.DataFrame:
    """Funde a camada bruta numa linha por código R##### (tabela "reacoes").

    Regras de fusão:
      1. Uma linha bruta com vários códigos R vira uma ocorrência para cada código.
      2. Orientação: a referência é a primeira ocorrência irreversível
         (ordenando por via), porque só uma irreversível afirma um sentido;
         se todas forem reversíveis, a primeira ocorrência. Uma ocorrência
         cujos substratos coincidem mais com os produtos da referência do que
         com os substratos é invertida antes de juntar (orientacao_invertida = True).
      3. substratos e produtos = união das ocorrências já alinhadas;
         lados_variam = True se as ocorrências listam conjuntos diferentes.
      4. Vale o reversível: reversivel = True se alguma ocorrência for
         "reversible", ou se há ocorrências irreversíveis desenhadas em
         sentidos opostos. Só fica False quando todas as ocorrências são
         irreversíveis e no mesmo sentido. O mapa irreversível marca o sentido
         em que a reação funciona naquela via (ex.: a fumarato redutase no
         eco00620, o ciclo de Krebs redutivo no eco00720), e não que ela só
         aconteça nesse sentido.
         conflito_reversibilidade = True se as vias discordam do tipo. Nesse
         caso, pela regra 2, substratos e produtos ficam no sentido da via
         irreversível, o que permite reconstruir no grafo a regra contrária
         (grafo.reversivel com conflito_como_irreversivel=True).
      5. genes = união dos genes das linhas de reacoes_por_via e das entradas
         de genes_por_via cujo atributo reaction cita o código. Isso cobre as
         entradas de gene sem tag <reaction>.
      6. vias = união; n_ocorrencias = número de ocorrências fundidas.
    """
    genes_citados: dict[str, set[str]] = defaultdict(set)
    for linha in genes_por_via.to_dict("records"):
        for codigo in linha["reacoes"]:
            genes_citados[codigo].update(linha["genes"])

    ocorrencias = _ocorrencias_por_codigo(reacoes_por_via)
    linhas = []
    for codigo in sorted(ocorrencias):
        grupo = ocorrencias[codigo]
        referencia = next((o for o in grupo if o["tipo"] != "reversible"), grupo[0])
        ref_substratos, ref_produtos = set(referencia["substratos"]), set(referencia["produtos"])
        substratos: list[str] = []
        produtos: list[str] = []
        genes: set[str] = set(genes_citados.get(codigo, ()))
        vias: set[str] = set()
        tipos: list[str] = []
        sentidos_irreversiveis: set[bool] = set()
        alguma_invertida = lados_variam = False

        # a referência primeiro, para as listas começarem na ordem dela
        for ocorrencia in [referencia, *(o for o in grupo if o is not referencia)]:
            s, p = ocorrencia["substratos"], ocorrencia["produtos"]
            invertida = _esta_invertida(s, p, ref_substratos, ref_produtos)
            if invertida:
                s, p = p, s
                alguma_invertida = True
            if (set(s), set(p)) != (ref_substratos, ref_produtos):
                lados_variam = True

            substratos.extend(m for m in s if m not in substratos)
            produtos.extend(m for m in p if m not in produtos)
            genes.update(ocorrencia["genes"])
            vias.add(ocorrencia["via"])
            tipos.append(ocorrencia["tipo"])
            if ocorrencia["tipo"] != "reversible":
                sentidos_irreversiveis.add(invertida)

        alguma_reversivel = any(tipo == "reversible" for tipo in tipos)
        irreversiveis_opostas = sentidos_irreversiveis == {True, False}
        linhas.append({
            "reacao": codigo,
            "substratos": substratos,
            "produtos": produtos,
            "reversivel": alguma_reversivel or irreversiveis_opostas,
            "genes": sorted(genes),
            "vias": sorted(vias),
            "n_ocorrencias": len(grupo),
            "conflito_reversibilidade": len(set(tipos)) > 1,
            "orientacao_invertida": alguma_invertida,
            "lados_variam": lados_variam,
        })

    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


def _ler_nomes(arquivo: Path) -> dict[str, str]:
    """Resposta de list/compound ou list/glycan -> código -> primeiro nome.

    Aceita as linhas com e sem prefixo ("cpd:C00001<tab>H2O; Water" ou
    "C00001<tab>H2O; Water"). Arquivo ausente devolve um dicionário vazio.
    """
    if not arquivo.exists():
        return {}
    nomes = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        codigo, sep, resto = linha.partition("\t")
        if sep:
            nomes[_codigo(codigo)] = resto.split(";")[0].strip()
    return nomes


def _tipo_id(codigo: str) -> str:
    return {"C": "composto", "G": "glicano"}.get(codigo[:1], "outro")


def montar_metabolitos(compostos_por_via: pd.DataFrame, reacoes: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela "metabolitos": todos os compostos vistos em alguma via.

    Nome: primeiro nome de config.ARQ_COMPOSTOS ou config.ARQ_GLICANOS; vazio
    se o código não estiver em nenhum dos dois. tipo_id pelo prefixo do
    código (C -> composto, G -> glicano, outro). vias: as vias em que aparece
    como entrada de composto ou em alguma reação. em_reacao = aparece em
    reacoes. Compostos fora de reações também entram, para não perder informação.
    """
    nomes = {**_ler_nomes(config.ARQ_GLICANOS), **_ler_nomes(config.ARQ_COMPOSTOS)}

    vias_de: dict[str, set[str]] = defaultdict(set)
    for linha in compostos_por_via.to_dict("records"):
        vias_de[linha["metabolito"]].add(linha["via"])

    em_reacao: set[str] = set()
    for linha in reacoes.to_dict("records"):
        for metabolito in [*linha["substratos"], *linha["produtos"]]:
            em_reacao.add(metabolito)
            vias_de[metabolito].update(linha["vias"])

    linhas = [
        {
            "metabolito": m,
            "nome": nomes.get(m, ""),
            "tipo_id": _tipo_id(m),
            "vias": sorted(vias_de[m]),
            "em_reacao": m in em_reacao,
        }
        for m in sorted(vias_de)
    ]
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["metabolitos"])


def montar_genes(org: str) -> pd.DataFrame:
    """Monta a tabela "genes" a partir de dados/kgml/<org>/_genes.tsv.

    Aceita os dois formatos da resposta de list/<org>:
        "eco:b0001<tab>CDS<tab>190..255<tab>thrL; thr operon leader peptide"  (atual)
        "eco:b0001<tab>thrL; thr operon leader peptide"                       (antigo)
    O símbolo e a descrição vêm da última coluna, separados pelo primeiro ";".
    Sem ";", o gene não tem símbolo e o texto todo vira a descrição.
    """
    arquivo = config.pasta_brutos(org) / "_genes.tsv"
    linhas = []
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            partes = linha.split("\t")
            if len(partes) < 2:
                continue
            simbolo, sep, descricao = partes[-1].partition(";")
            if not sep:
                simbolo, descricao = "", partes[-1]
            linhas.append({"gene": _codigo(partes[0]), "simbolo": simbolo.strip(),
                           "descricao": descricao.strip()})
    tabela = pd.DataFrame(linhas, columns=contrato.COLUNAS["genes"])
    return tabela.drop_duplicates("gene").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def _amostra(itens: Iterable[str], n: int = 10) -> list[str]:
    return sorted(itens)[:n]


def _conferir_codigos(codigos: dict[str, str], nomes_kegg: dict[str, str],
                      na_rede: set[str]) -> dict[str, dict]:
    return {
        codigo: {"nome_config": nome, "nome_kegg": nomes_kegg.get(codigo, ""),
                 "na_rede": codigo in na_rede}
        for codigo, nome in codigos.items()
    }


def relatorio(reacoes_por_via: pd.DataFrame, genes_por_via: pd.DataFrame,
              reacoes: pd.DataFrame, metabolitos: pd.DataFrame, genes: pd.DataFrame,
              falhas_leitura: list[dict], metadados: dict) -> dict:
    """Contagens para conferir a extração (e citar na metodologia).

    Inclui:
      - número de vias, de linhas brutas, de reações, de metabólitos e de genes
      - entradas de gene sem <reaction> correspondente na mesma via (número e fração):
        acima de config.LIMITE_LACUNA_GENES, reparo_necessario = True
        (a rede precisa ser reparada com o mapa global eco01100)
      - as três conferências de que nada se perdeu:
          soma de n_ocorrencias == número de pares (linha bruta, código R);
          todo metabólito de reacoes_por_via está em metabolitos;
          todo gene ligado a uma reação da camada bruta está em reacoes.genes
      - a conferência da regra 4: toda reação com conflito de reversibilidade
        ficou reversível, e nenhuma irreversível ficou no sentido de uma via
        que não é a primeira (com "vale o reversível", as duas listas ficam vazias)
      - reações com conflito de reversibilidade (todas reversíveis), com
        orientação invertida e reversíveis só por estarem desenhadas como
        irreversíveis em sentidos opostos
      - reações com um metabólito dos dois lados, linhas com vários códigos R
      - metabólitos que não aparecem em nenhuma reação
      - os metabólitos mais frequentes (para conferir a lista de moeda) e os
        códigos do config.py (nutrientes, substâncias do Exp 3, moeda) na rede
    """
    nomes = dict(zip(metabolitos["metabolito"], metabolitos["nome"]))
    genes_de = {linha["reacao"]: set(linha["genes"]) for linha in reacoes.to_dict("records")}

    # --- lacuna de <reaction> ---------------------------------------------------
    ids_com_reaction = set(zip(reacoes_por_via["via"], reacoes_por_via["entrada_kgml"].astype(int)))
    citam_reacao = genes_por_via[genes_por_via["reacoes"].map(len) > 0]
    sem_reaction = [
        (linha["via"], int(linha["entrada_kgml"]))
        for linha in citam_reacao.to_dict("records")
        if (linha["via"], int(linha["entrada_kgml"])) not in ids_com_reaction
    ]
    fracao_lacuna = len(sem_reaction) / len(citam_reacao) if len(citam_reacao) else 0.0
    codigos_citados = {c for lista in genes_por_via["reacoes"] for c in lista}
    codigos_sem_reaction = codigos_citados - set(reacoes["reacao"])
    vias_com_genes = set(citam_reacao["via"])
    vias_sem_reaction = vias_com_genes - set(reacoes_por_via["via"])

    # --- fusão -----------------------------------------------------------------
    tipos_de: dict[str, set[str]] = defaultdict(set)
    for linha in reacoes_por_via.to_dict("records"):
        for codigo in linha["reacoes"]:
            tipos_de[codigo].add(linha["tipo"])
    registros = reacoes.to_dict("records")
    por_sentidos_opostos = [r["reacao"] for r in registros
                            if r["reversivel"] and "reversible" not in tipos_de[r["reacao"]]]
    conflito_irreversivel = [r["reacao"] for r in registros
                             if r["conflito_reversibilidade"] and not r["reversivel"]]
    primeira = {codigo: grupo[0] for codigo, grupo in _ocorrencias_por_codigo(reacoes_por_via).items()}
    sentido_de_outra_via = [
        r["reacao"] for r in registros
        if not r["reversivel"] and _esta_invertida(primeira[r["reacao"]]["substratos"],
                                                   primeira[r["reacao"]]["produtos"],
                                                   set(r["substratos"]), set(r["produtos"]))
    ]
    dois_lados = [r["reacao"] for r in registros if set(r["substratos"]) & set(r["produtos"])]
    lado_vazio = [r["reacao"] for r in registros if not r["substratos"] or not r["produtos"]]

    # --- conferências ----------------------------------------------------------
    pares = int(reacoes_por_via["reacoes"].map(len).sum())
    soma = int(reacoes["n_ocorrencias"].sum())
    citados_bruta = {m for col in ("substratos", "produtos") for lista in reacoes_por_via[col] for m in lista}
    faltando_metabolitos = citados_bruta - set(metabolitos["metabolito"])
    genes_faltando = set()
    for tabela in (reacoes_por_via, genes_por_via):
        for linha in tabela.to_dict("records"):
            for codigo in linha["reacoes"]:
                if codigo in genes_de:
                    genes_faltando.update(f"{codigo}:{g}" for g in set(linha["genes"]) - genes_de[codigo])

    # --- metabólitos e genes ---------------------------------------------------
    na_rede = set(metabolitos.loc[metabolitos["em_reacao"], "metabolito"])
    sem_nome = metabolitos.loc[metabolitos["nome"] == "", "metabolito"]
    frequencia = Counter(m for r in registros for m in set(r["substratos"]) | set(r["produtos"]))
    genes_nas_reacoes = {g for lista in reacoes["genes"] for g in lista}

    return {
        "org": metadados.get("org"),
        "data_download": metadados.get("data_download"),
        "info_kegg": (metadados.get("info_kegg") or "").splitlines()[:2],
        "contagens": {
            "vias_nos_metadados": len(metadados.get("vias", [])),
            "vias_lidas": len(metadados.get("vias", [])) - len(falhas_leitura),
            "linhas_brutas_reacao": len(reacoes_por_via),
            "pares_linha_codigo": pares,
            "linhas_com_varios_codigos": int((reacoes_por_via["reacoes"].map(len) > 1).sum()),
            "reacoes": len(reacoes),
            "reacoes_reversiveis": int(reacoes["reversivel"].sum()),
            "metabolitos": len(metabolitos),
            "metabolitos_em_reacao": len(na_rede),
            "metabolitos_fora_de_reacao": len(metabolitos) - len(na_rede),
            "metabolitos_sem_nome": len(sem_nome),
            "entradas_de_gene": len(genes_por_via),
            "genes_na_tabela": len(genes),
            "genes_nas_reacoes": len(genes_nas_reacoes),
            "genes_nas_reacoes_sem_cadastro": len(genes_nas_reacoes - set(genes["gene"])),
        },
        "lacuna_reaction": {
            "entradas_de_gene_que_citam_reacao": len(citam_reacao),
            "sem_reaction_na_mesma_via": len(sem_reaction),
            "fracao": round(fracao_lacuna, 4),
            "limite": config.LIMITE_LACUNA_GENES,
            "reparo_necessario": fracao_lacuna > config.LIMITE_LACUNA_GENES,
            "codigos_citados_sem_reaction_em_nenhuma_via": len(codigos_sem_reaction),
            "exemplos_codigos": _amostra(codigos_sem_reaction),
            "vias_com_genes_e_nenhuma_reaction": _amostra(vias_sem_reaction, 50),
        },
        "fusao": {
            "conflito_reversibilidade": int(reacoes["conflito_reversibilidade"].sum()),
            "orientacao_invertida": int(reacoes["orientacao_invertida"].sum()),
            "reversiveis_por_sentidos_opostos": len(por_sentidos_opostos),
            "exemplos_reversiveis_por_sentidos_opostos": _amostra(por_sentidos_opostos),
            "irreversiveis_com_sentido_de_outra_via": len(sentido_de_outra_via),
            "exemplos_irreversiveis_com_sentido_de_outra_via": _amostra(sentido_de_outra_via),
            "lados_variam": int(reacoes["lados_variam"].sum()),
            "metabolito_dos_dois_lados": len(dois_lados),
            "exemplos_metabolito_dos_dois_lados": _amostra(dois_lados),
            "sem_substrato_ou_sem_produto": len(lado_vazio),
            "exemplos_sem_substrato_ou_sem_produto": _amostra(lado_vazio),
        },
        "conferencias": {
            "soma_ocorrencias": {"ok": soma == pares, "soma_n_ocorrencias": soma, "pares": pares},
            "metabolitos_completos": {"ok": not faltando_metabolitos,
                                      "faltando": _amostra(faltando_metabolitos)},
            "genes_completos": {"ok": not genes_faltando, "faltando": _amostra(genes_faltando)},
            "conflitos_reversiveis": {"ok": not conflito_irreversivel and not sentido_de_outra_via,
                                      "conflitos_que_ficaram_irreversiveis": _amostra(conflito_irreversivel),
                                      "irreversiveis_com_sentido_de_outra_via": _amostra(sentido_de_outra_via)},
        },
        "mais_frequentes": [
            {"metabolito": m, "nome": nomes.get(m, ""), "n_reacoes": n,
             "na_lista_de_moeda": m in config.METABOLITOS_MOEDA}
            for m, n in frequencia.most_common(20)
        ],
        "codigos_do_config": {
            "nutrientes": _conferir_codigos(config.NUTRIENTES, nomes, na_rede),
            "substancias_exp3": _conferir_codigos(config.SUBSTANCIAS_EXP3, nomes, na_rede),
            "moeda_fora_da_rede": sorted(set(config.METABOLITOS_MOEDA) - na_rede),
        },
        "falhas_leitura": falhas_leitura,
        "falhas_download": metadados.get("falhas", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta as tabelas do contrato a partir dos KGML.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    args = parser.parse_args()
    org = args.org

    try:
        metadados = ler_metadados(org)
        reacoes_por_via, genes_por_via, compostos_por_via, falhas = montar_camada_bruta(org)
    except FileNotFoundError as erro:
        sys.exit(str(erro))
    if reacoes_por_via.empty:
        sys.exit(f"Nenhuma <reaction> encontrada nos KGML de {org}. Confira dados/kgml/{org}/.")

    reacoes = fundir_reacoes(reacoes_por_via, genes_por_via)
    metabolitos = montar_metabolitos(compostos_por_via, reacoes)
    genes = montar_genes(org)

    contrato.salvar(reacoes_por_via, org, "reacoes_por_via")
    contrato.salvar(genes_por_via, org, "genes_por_via")
    contrato.salvar(reacoes, org, "reacoes")
    contrato.salvar(metabolitos, org, "metabolitos")
    contrato.salvar(genes, org, "genes")

    rel = relatorio(reacoes_por_via, genes_por_via, reacoes, metabolitos, genes, falhas, metadados)
    destino = config.pasta_processados(org) / "relatorio_extracao.json"
    destino.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rel, indent=2, ensure_ascii=False))
    print(f"\nTabelas em {config.caminho_legivel(config.pasta_processados(org))}/, "
          f"relatório em {config.caminho_legivel(destino)}")


if __name__ == "__main__":
    main()
