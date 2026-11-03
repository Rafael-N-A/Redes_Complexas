"""Experimento 2 — Topologia vs. biologia.

Pergunta: os genes cuja deleção mais danifica a rede coincidem com os genes
essenciais medidos em laboratório (coleção Keio)? A literatura diz que a
correlação com centralidade é fraca (Samal et al., 2006).

Só para a E. coli, o único organismo com gabarito. Roda na versão
config.VERSAO_CASCATA, como o Exp 3, porque o dano vem da mesma cascata.

Lê:    dados/eco/reacoes.csv, genes.csv; dados/keio/essenciais.csv
Grava: resultados/tabelas/exp2_genes_eco.csv     (uma linha por gene: essencial, dano, grau, betweenness, choke)
       resultados/tabelas/exp2_rankings_eco.csv  (AUC e precisão@k de cada ranking)
       resultados/tabelas/exp2_choke_eco.csv     (essenciais dentro e fora dos choke points + teste)
       resultados/tabelas/exp2_roc_eco.csv       (pontos das curvas ROC)
       resultados/figuras/exp2_roc_eco.png

Uso, na raiz:
    python -m experimentos.exp2_biologia
    python -m experimentos.exp2_biologia --so-figuras
"""
from __future__ import annotations

import argparse

import config
from experimentos.biblioteca import cascata, figuras, grafo, metricas, nocaute, validacao  # noqa: F401


def rodar(org: str) -> None:
    """Simula o nocaute de cada gene, monta os três rankings e compara com a Keio.

    Passos:
      1. tabelas = grafo.carregar_tabelas(org); remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
         B = grafo.grafo_bipartido(tabelas["reacoes"], remover_moeda=remover)
         G = grafo.grafo_simples(grafo.grafo_metabolitos(tabelas["reacoes"], remover_moeda=remover))
      2. dano = nocaute.dano_todos_genes(B, tabelas["reacoes"])
      3. grau = nocaute.centralidade_por_gene(tabelas["reacoes"], metricas.graus(G)["total"])
         betw = nocaute.centralidade_por_gene(tabelas["reacoes"], metricas.betweenness(G))
      4. y = validacao.rotulos(dano["gene"], validacao.carregar_keio())
         Reporte quantos essenciais da Keio ficaram fora da rede.
      5. rankings = {"dano": ..., "grau": grau, "betweenness": betw}
         validacao.avaliar_rankings(y, rankings) e validacao.curva_roc para cada um.
      6. Choke points: metricas.choke_points -> nocaute.genes_em_choke_points;
         tabela 2x2 (choke x essencial) e scipy.stats.fisher_exact.
      7. Grave as quatro tabelas com config.arquivo_tabela("exp2", <conteudo>, org).

    Para a discussão: separe dois ou três genes essenciais com baixa
    centralidade (o plano diz que rendem uma boa discussão).
    """
    raise NotImplementedError


def plotar(org: str) -> None:
    """Lê exp2_roc e exp2_rankings e grava a figura com as três curvas ROC."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2: topologia vs. biologia (nocaute x Keio).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL,
                        help="código KEGG; só a E. coli (eco) tem gabarito")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if args.org != "eco":
        parser.error("o gabarito da Keio só existe para a E. coli (--org eco)")
    if not args.so_figuras:
        rodar(args.org)
    plotar(args.org)


if __name__ == "__main__":
    main()
