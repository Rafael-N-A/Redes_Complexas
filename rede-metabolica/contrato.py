"""Contrato dos dados: o formato das tabelas que a coleta grava e os
experimentos leem.

É a única interface entre `coleta/` e `experimentos/`. Se uma coluna mudar,
muda aqui, e `salvar()` recusa qualquer tabela fora do formato.

Para não perder informação do KGML, os dados ficam em duas camadas:

1. Camada bruta (uma linha por elemento do KGML, sem fundir nada)
   - reacoes_por_via: cada <reaction> de cada via, como está no arquivo.
   - genes_por_via:   cada <entry type="gene"> de cada via, como está no arquivo.
     Inclui as entradas de gene cuja reação não tem tag <reaction> (risco do plano).

2. Camada fundida (deduplicada, é a que os experimentos usam)
   - reacoes:     uma linha por código R#####, com todos os substratos e
                  produtos, genes e vias; marca conflitos entre vias.
   - metabolitos: uma linha por composto (C#####, ou G##### para glicanos).
   - genes:       b-number -> símbolo e descrição (para a discussão do Exp 2).

Em disco, colunas de lista são texto separado por ";". Em memória, depois
de `carregar()`, são `list[str]`.
"""
from __future__ import annotations

import pandas as pd

import config

SEP = ";"

COLUNAS = {
    # --- camada bruta -------------------------------------------------------
    "reacoes_por_via": [
        "via",            # eco00020
        "entrada_kgml",   # id do <reaction> (= id da entrada de gene ligada a ele)
        "reacoes",        # lista: atributo name, pode ter mais de um R#####
        "tipo",           # "reversible" ou "irreversible", como está no KGML
        "substratos",     # lista: C#####, na ordem do KGML
        "produtos",       # lista: C#####, na ordem do KGML
        "genes",          # lista: genes da entrada ligada, sem o prefixo do organismo
    ],
    "genes_por_via": [
        "via",
        "entrada_kgml",   # id da <entry type="gene">
        "genes",          # lista: genes da entrada (atributo name), sem o prefixo
        "reacoes",        # lista: R##### do atributo reaction da entrada
    ],
    # --- camada fundida -----------------------------------------------------
    "reacoes": [
        "reacao",                    # R#####
        "substratos",                # lista: união entre as vias, já alinhada no mesmo sentido
        "produtos",                  # lista: idem
        "reversivel",                # bool, pela regra de fusão (ver coleta/montar_tabelas.py)
        "genes",                     # lista: união dos genes ligados à reação em todas as vias
        "vias",                      # lista: vias em que a reação aparece
        "n_ocorrencias",             # quantas linhas da camada bruta viraram esta
        "conflito_reversibilidade",  # bool: vias discordam do tipo
        "orientacao_invertida",      # bool: alguma via desenhou a reação no sentido oposto
        "lados_variam",              # bool: vias listam conjuntos diferentes de substratos/produtos
    ],
    "metabolitos": [
        "metabolito",  # C##### (ou G##### para glicanos)
        "nome",        # nome legível do KEGG
        "tipo_id",     # "composto" ou "glicano"
        "vias",        # lista: vias em que aparece
        "em_reacao",   # bool: aparece como substrato ou produto de alguma reação
    ],
    "genes": [
        "gene",       # id do KEGG sem o prefixo; na E. coli é o b-number, o mesmo da Keio
        "simbolo",    # gltA, aceE...
        "descricao",
    ],
}

COLUNAS_LISTA = {"reacoes", "substratos", "produtos", "genes", "vias"}
COLUNAS_BOOL = {"reversivel", "conflito_reversibilidade", "orientacao_invertida",
                "lados_variam", "em_reacao"}
COLUNAS_INT = {"entrada_kgml", "n_ocorrencias"}


def juntar(itens) -> str:
    """Lista -> texto separado por ';'. Recusa itens que contenham ';'."""
    itens = list(itens)
    for item in itens:
        if SEP in str(item):
            raise ValueError(f"item {item!r} contém o separador {SEP!r}")
    return SEP.join(str(i) for i in itens)


def separar(texto: str) -> list[str]:
    """Texto separado por ';' -> lista. Texto vazio vira lista vazia."""
    return [] if texto == "" else texto.split(SEP)


def caminho(org: str, tabela: str):
    return config.pasta_processados(org) / f"{tabela}.csv"


def validar(df: pd.DataFrame, tabela: str) -> None:
    """Garante que o DataFrame tem exatamente as colunas do contrato, na ordem."""
    esperado = COLUNAS[tabela]
    if list(df.columns) != esperado:
        faltando = [c for c in esperado if c not in df.columns]
        sobrando = [c for c in df.columns if c not in esperado]
        raise ValueError(
            f"tabela '{tabela}' fora do contrato. Faltando: {faltando}. "
            f"Sobrando: {sobrando}. Ordem esperada: {esperado}"
        )


def salvar(df: pd.DataFrame, org: str, tabela: str) -> None:
    """Valida e grava uma tabela em dados/<org>/<tabela>.csv."""
    validar(df, tabela)
    df = df.copy()
    for col in COLUNAS_LISTA & set(df.columns):
        df[col] = df[col].map(juntar)
    destino = caminho(org, tabela)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)


def carregar(org: str, tabela: str) -> pd.DataFrame:
    """Lê dados/<org>/<tabela>.csv e converte listas, booleanos e inteiros."""
    origem = caminho(org, tabela)
    if not origem.exists():
        raise FileNotFoundError(
            f"{origem} não existe. Rode antes: "
            f"python -m coleta.baixar_kgml --org {org} && "
            f"python -m coleta.montar_tabelas --org {org}"
        )
    df = pd.read_csv(origem, dtype=str, keep_default_na=False)
    validar(df, tabela)
    for col in df.columns:
        if col in COLUNAS_LISTA:
            df[col] = df[col].map(separar)
        elif col in COLUNAS_BOOL:
            df[col] = df[col].map({"True": True, "False": False})
        elif col in COLUNAS_INT:
            df[col] = df[col].astype(int)
    return df
