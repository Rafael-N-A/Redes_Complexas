"""Experimento 0 — Análise descritiva (a base que o roteiro pede).

Caracteriza a rede e confere a lista de metabólitos-moeda pelos nós de maior
grau. Produz a Tabela 1 e a figura de distribuição de graus do artigo.

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp0_tabela1_<org>_<versao>.csv
       resultados/tabelas/exp0_top10_<org>_<versao>.csv
       resultados/tabelas/exp0_graus_<org>_<versao>.csv
       resultados/figuras/exp0_graus_<org>_<versao>.png

Uso, na raiz:
    python -m experimentos.exp0_descritiva --org eco --versao ambas
    python -m experimentos.exp0_descritiva --so-figuras
"""
from __future__ import annotations

import argparse

import config
from experimentos.biblioteca import figuras, grafo, metricas  # noqa: F401


def rodar(org: str, versao: str) -> None:
    """Calcula as métricas descritivas e grava as três tabelas.

    Passos:
      1. tabelas = grafo.carregar_tabelas(org)
      2. M = grafo.grafo_metabolitos(tabelas["reacoes"], remover_moeda=config.VERSOES_MOEDA[versao])
         G = grafo.grafo_simples(M)
      3. tabela1 = metricas.tabela_descritiva(G)  (uma linha; acrescente org e versao)
         top10   = metricas.top_por_grau(G, 10, grafo.nomes_metabolitos(tabelas["metabolitos"]))
         graus   = metricas.distribuicao_graus(G)
      4. Grave cada uma em config.arquivo_tabela("exp0", <conteudo>, org, versao).
      5. Opcional (só se sobrar tempo): metricas.ajuste_lei_potencia, numa quarta tabela.

    Na versão com_moeda, confira se o top-10 é mesmo de metabólitos-moeda e
    ajuste config.METABOLITOS_MOEDA se aparecer algum que falta.
    """
    raise NotImplementedError


def plotar(org: str, versao: str) -> None:
    """Lê exp0_graus_<org>_<versao>.csv e grava a figura de distribuição de graus.

    figuras.estilo(); fig = figuras.distribuicao_graus(tabela);
    figuras.salvar(fig, config.arquivo_figura("exp0", "graus", org, versao)).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 0: análise descritiva da rede.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--versao", choices=[*config.VERSOES_MOEDA, "ambas"], default="ambas",
                        help="com ou sem metabólitos-moeda")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    versoes = list(config.VERSOES_MOEDA) if args.versao == "ambas" else [args.versao]
    for versao in versoes:
        if not args.so_figuras:
            rodar(args.org, versao)
        plotar(args.org, versao)


if __name__ == "__main__":
    main()
