"""Converte a tabela suplementar da Keio em dados/keio/essenciais.csv.

Fonte: Baba et al. (2006), Mol. Syst. Biol., Tabela Suplementar 6
("essential gene candidates"), arquivo msb4100050-s8.xls do material
suplementar (https://pmc.ncbi.nlm.nih.gov/articles/PMC1681482/).

O script procura, em todas as abas, a coluna com mais b-numbers (b0001,
b2697...), grava essenciais.csv com uma coluna "gene" e confere a contagem
contra os 303 candidatos do artigo. As linhas de dados são as que têm um
número ECK na primeira coluna; um gene da tabela pode não ter b-number
(anotado só na W3110, não na MG1655): ele fica de fora do CSV, porque o
KEGG da E. coli (eco = MG1655) usa b-numbers, e o script lista quais são.
Se o número de linhas de dados não bater com 303, ele não grava e mostra
as colunas para você filtrar; --forcar grava assim mesmo.

Uso, na raiz do projeto (rede-metabolica):
    pip install xlrd               # só para ler o .xls antigo; .xlsx e .csv não precisam
    python -m coleta.converter_keio ~/Downloads/msb4100050-s8.xls
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

import config

ESPERADO = 303  # candidatos a essenciais em Baba et al. (2006)
B_NUMBER = re.compile(r"^b\d{4}$")
ECK = re.compile(r"^ECK\d{4}$")


def ler_abas(caminho: Path) -> dict[str, pd.DataFrame]:
    """Todas as abas como texto cru, sem adivinhar cabeçalho."""
    if caminho.suffix.lower() == ".csv":
        return {caminho.stem: pd.read_csv(caminho, header=None, dtype=str)}
    with warnings.catch_warnings():  # o pandas avisa que o leitor de .xls (xlrd) vai sair; aqui ainda funciona
        warnings.simplefilter("ignore")
        return pd.read_excel(caminho, sheet_name=None, header=None, dtype=str)


def coluna_de_b_numbers(abas: dict[str, pd.DataFrame]) -> tuple[str, int, list[str]]:
    """(aba, coluna, b-numbers na ordem da tabela) da coluna com mais b-numbers."""
    melhor = ("", -1, [])
    for nome, tabela in abas.items():
        for coluna in tabela.columns:
            valores = tabela[coluna].dropna().astype(str).str.strip()
            achados = [v for v in valores if B_NUMBER.match(v)]
            if len(achados) > len(melhor[2]):
                melhor = (nome, coluna, achados)
    return melhor


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte a tabela suplementar da Keio em essenciais.csv.")
    parser.add_argument("arquivo", type=Path, help="msb4100050-s8.xls (ou .xlsx/.csv salvo a partir dele)")
    parser.add_argument("--destino", type=Path, default=config.ARQ_KEIO)
    parser.add_argument("--forcar", action="store_true", help=f"grava mesmo se não der {ESPERADO} genes")
    args = parser.parse_args()

    abas = ler_abas(args.arquivo)
    aba, coluna, achados = coluna_de_b_numbers(abas)
    if not achados:
        sys.exit("Nenhum b-number (b0001, b0002...) encontrado em nenhuma aba.")
    genes = sorted(set(achados))
    repetidos = len(achados) - len(genes)
    print(f"aba '{aba}', coluna {coluna}: {len(genes)} b-numbers distintos"
          + (f" ({repetidos} repetidos ignorados)" if repetidos else ""))

    tabela = abas[aba]
    primeira = tabela.iloc[:, 0].astype(str).str.strip()
    dados = tabela[primeira.str.match(ECK)]
    n_linhas = len(dados) if len(dados) else len(genes)
    if len(dados):
        sem_b = dados[~dados[coluna].astype(str).str.strip().str.match(B_NUMBER)]
        print(f"{len(dados)} linhas de dados (com número ECK)")
        if len(sem_b):
            print(f"{len(sem_b)} sem b-number da MG1655, ficam de fora do CSV:")
            for linha in sem_b.itertuples(index=False):
                print("   " + "  ".join(str(v) for v in linha[:4] if str(v) != "nan"))
        for j in tabela.columns:  # a coluna "genetic" deve ser E em todas as linhas
            if tabela[j].head(6).astype(str).str.contains("genetic", case=False).any():
                print(f"coluna '5. genetic': {dados[j].value_counts(dropna=False).to_dict()}")

    if n_linhas != ESPERADO and not args.forcar:
        print(f"\nO artigo relata {ESPERADO} candidatos; a tabela tem {n_linhas}. Nada foi gravado.")
        print("Primeiras linhas da aba, para ver se há uma coluna separando os essenciais da Keio:")
        print(tabela.head(8).to_string())
        sys.exit("Filtre as linhas da Keio (ou me mande o arquivo) e rode de novo; --forcar grava assim mesmo.")

    args.destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes}).to_csv(args.destino, index=False)
    print(f"gravado: {config.caminho_legivel(args.destino)}")
    nota = f"Tabela Suplementar 6 (msb4100050-s8.xls), aba '{aba}': {n_linhas} candidatos"
    if n_linhas != len(genes):
        nota += f", {len(genes)} com b-number da MG1655 (os demais só têm anotação W3110)"
    print(f"Anote no LEIAME.md: {nota}.")


if __name__ == "__main__":
    main()
