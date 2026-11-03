"""Coleta, passo 2: transforma os KGML baixados nas tabelas do contrato.

Lê dados/kgml/<org>/ e grava em dados/<org>/:
    reacoes_por_via.csv, genes_por_via.csv   camada bruta (nada fundido)
    reacoes.csv, metabolitos.csv, genes.csv  camada fundida
    relatorio_extracao.json                  contagens e conflitos

O formato de cada tabela está em contrato.py.

Uso:
    python -m coleta.montar_tabelas --org eco
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
import contrato


def ler_kgml(caminho: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Lê um KGML e devolve três listas de linhas, sem fundir nada.

    Retorna (reacoes, genes, compostos):
      - reacoes:   uma linha por <reaction>, com as colunas de "reacoes_por_via"
      - genes:     uma linha por <entry type="gene">, colunas de "genes_por_via"
      - compostos: uma linha por <entry type="compound">: {"via", "metabolito"}

    Dicas (Biopython):
        from Bio.KEGG.KGML import KGML_parser
        p = KGML_parser.read(open(caminho))
        p.reactions  -> r.id, r.name ("rn:R00351", pode ter mais de um, separados
                        por espaço), r.type, r.substrates / r.products (entradas
                        com .name "cpd:C00024"), r.entry (entrada de gene ligada)
        p.genes      -> g.id, g.name ("eco:b0114 eco:b0115"), g.reaction ("rn:R... rn:R...")
        p.compounds  -> c.name ("cpd:C00024" ou "gl:G00001" para glicanos)
    Remova os prefixos rn:, cpd:, gl: e o do organismo (eco:, bsu:, mge:...),
    e separe os nomes múltiplos. Use o texto depois do primeiro ":" em vez de
    fixar "eco:", senão bsu e mge saem sem genes.
    """
    raise NotImplementedError


def montar_camada_bruta(org: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica ler_kgml a todos os .xml do organismo.

    Retorna (reacoes_por_via, genes_por_via, compostos_por_via), as duas
    primeiras já no formato do contrato.
    """
    raise NotImplementedError


def fundir_reacoes(reacoes_por_via: pd.DataFrame, genes_por_via: pd.DataFrame) -> pd.DataFrame:
    """Funde a camada bruta numa linha por código R##### (tabela "reacoes").

    Regras de fusão:
      1. Uma linha bruta com vários códigos R vira uma ocorrência para cada código.
      2. Orientação: a primeira ocorrência (ordenando por via) é a referência.
         Uma ocorrência cujos substratos coincidem mais com os produtos da
         referência do que com os substratos é invertida antes de juntar
         (orientacao_invertida = True).
      3. substratos e produtos = união das ocorrências já alinhadas;
         lados_variam = True se as ocorrências listam conjuntos diferentes.
      4. reversivel = True se todas as ocorrências forem "reversible", OU se
         há ocorrências irreversíveis desenhadas em sentidos opostos (as duas
         direções aparecem no KEGG). Nos demais casos, False.
         conflito_reversibilidade = True se as vias discordam do tipo.
      5. genes = união dos genes das linhas de reacoes_por_via e das entradas
         de genes_por_via cujo atributo reaction cita o código. Isso cobre as
         entradas de gene sem tag <reaction>.
      6. vias = união; n_ocorrencias = número de ocorrências fundidas.
    """
    raise NotImplementedError


def montar_metabolitos(compostos_por_via: pd.DataFrame, reacoes: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela "metabolitos": todos os compostos vistos em alguma via.

    Nome: primeiro nome de config.ARQ_COMPOSTOS ou config.ARQ_GLICANOS
    (cada linha é "cpd:C00001<tab>H2O; Water"). tipo_id pelo prefixo do
    código (C -> composto, G -> glicano). em_reacao = aparece em reacoes.
    Compostos fora de reações também entram, para não perder informação.
    """
    raise NotImplementedError


def montar_genes(org: str) -> pd.DataFrame:
    """Monta a tabela "genes" a partir de dados/kgml/<org>/_genes.tsv.

    Confira o formato da resposta de list/<org> na primeira execução:
    o símbolo e a descrição costumam vir na última coluna, separados por ";".
    """
    raise NotImplementedError


def relatorio(reacoes_por_via: pd.DataFrame, genes_por_via: pd.DataFrame,
              reacoes: pd.DataFrame, metabolitos: pd.DataFrame) -> dict:
    """Contagens para conferir a extração (e citar na metodologia).

    Inclua pelo menos:
      - número de vias, de linhas brutas, de reações, de metabólitos e de genes
      - entradas de gene sem <reaction> correspondente na mesma via (número e fração):
        acima de config.LIMITE_LACUNA_GENES, marque reparo_necessario = True
        (a rede precisa ser reparada com o mapa global eco01100)
      - as três conferências de que nada se perdeu:
          soma de n_ocorrencias == número de pares (linha bruta, código R);
          todo metabólito de reacoes_por_via está em metabolitos;
          todo gene de genes_por_via está em reacoes.genes ou entre os sem reação
      - reações com conflito de reversibilidade, com orientação invertida e,
        entre estas, quantas são irreversíveis nas duas vias (conflito real)
      - reações com um metabólito dos dois lados, linhas com vários códigos R
      - metabólitos que não aparecem em nenhuma reação
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta as tabelas do contrato a partir dos KGML.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    args = parser.parse_args()
    org = args.org

    reacoes_por_via, genes_por_via, compostos_por_via = montar_camada_bruta(org)
    reacoes = fundir_reacoes(reacoes_por_via, genes_por_via)
    metabolitos = montar_metabolitos(compostos_por_via, reacoes)
    genes = montar_genes(org)

    contrato.salvar(reacoes_por_via, org, "reacoes_por_via")
    contrato.salvar(genes_por_via, org, "genes_por_via")
    contrato.salvar(reacoes, org, "reacoes")
    contrato.salvar(metabolitos, org, "metabolitos")
    contrato.salvar(genes, org, "genes")

    rel = relatorio(reacoes_por_via, genes_por_via, reacoes, metabolitos)
    destino = config.pasta_processados(org) / "relatorio_extracao.json"
    destino.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
