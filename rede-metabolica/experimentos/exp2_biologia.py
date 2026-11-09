"""Experimento 2 — Topologia vs. biologia.

Pergunta: os genes cuja deleção mais danifica a rede coincidem com os genes
essenciais medidos em laboratório (coleção Keio)? A literatura diz que a
correlação com centralidade é fraca (Samal et al., 2006).

Só para a E. coli, o único organismo com gabarito. Roda na versão
config.VERSAO_CASCATA, como o Exp 3, porque o dano vem da mesma cascata.

Rankings (maior pontuação = mais provável de ser essencial):
  dano               metabólitos perdidos na cascata ao deletar o gene, meio rico
                     (config.DANO_NOCAUTE; a Keio foi medida em LB)      [principal]
  grau, betweenness  centralidade dos metabólitos das reações do gene     [principal]
  dano_meio_minimo   o mesmo dano partindo só do meio mínimo             [sensibilidade]
  reacoes_perdidas   só quantas reações o gene derruba, sem a cascata    [linha de base]
A linha de base mostra quanto a propagação pela rede acrescenta à perda direta.
Os choke points são sim/não: entram num teste exato de Fisher, não num ranking.

Lê:    dados/eco/reacoes.csv, genes.csv; dados/keio/essenciais.csv
Grava: resultados/tabelas/exp2_genes_eco.csv     (uma linha por gene: essencial, danos, grau, betweenness, choke)
       resultados/tabelas/exp2_rankings_eco.csv  (AUC com IC 95% e p, precisão@k de cada ranking)
       resultados/tabelas/exp2_choke_eco.csv     (essenciais dentro e fora dos choke points + teste)
       resultados/tabelas/exp2_roc_eco.csv       (pontos das curvas ROC)
       resultados/tabelas/exp2_exemplos_eco.csv  (essenciais de baixa centralidade que o dano acerta)
       resultados/tabelas/exp2_resumo_eco.csv    (cobertura da Keio na rede)
       resultados/figuras/exp2_roc_eco.png

Uso, na raiz:
    python -m experimentos.exp2_biologia
    python -m experimentos.exp2_biologia --so-figuras
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import fisher_exact

import config
from experimentos.biblioteca import figuras, grafo, metricas, nocaute, validacao

MEIO = {"avalanche": "meio_rico", "alcance": "meio_minimo"}  # medida de dano -> nome da coluna
PAPEIS = {"dano": "principal", "grau": "principal", "betweenness": "principal",
          "dano_meio_minimo": "sensibilidade", "dano_meio_rico": "sensibilidade",
          "reacoes_perdidas": "linha de base"}
RANKINGS_FIGURA = ("dano", "grau", "betweenness")
N_EXEMPLOS = 5


def _gravar(tabela: pd.DataFrame, conteudo: str, org: str) -> None:
    destino = config.arquivo_tabela("exp2", conteudo, org)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)


def tabela_choke(genes: pd.DataFrame, n_choke: int, n_reacoes: int) -> pd.DataFrame:
    """Uma linha: essenciais dentro e fora dos choke points, razão de chances e p de Fisher.

    genes precisa das colunas essencial e choke (booleanas).
    """
    dentro, fora = genes[genes["choke"]], genes[~genes["choke"]]
    tabela_2x2 = [[int(dentro["essencial"].sum()), int((~dentro["essencial"]).sum())],
                  [int(fora["essencial"].sum()), int((~fora["essencial"]).sum())]]
    razao, p = fisher_exact(tabela_2x2, alternative="two-sided")
    return pd.DataFrame([{
        "reacoes_choke": n_choke, "reacoes_total": n_reacoes,
        "genes_choke": len(dentro), "essenciais_choke": tabela_2x2[0][0],
        "genes_outros": len(fora), "essenciais_outros": tabela_2x2[1][0],
        "fracao_essenciais_choke": tabela_2x2[0][0] / len(dentro) if len(dentro) else float("nan"),
        "fracao_essenciais_outros": tabela_2x2[1][0] / len(fora) if len(fora) else float("nan"),
        "razao_de_chances": float(razao), "p_fisher": float(p),
    }])


def exemplos_baixa_centralidade(genes: pd.DataFrame, n: int = N_EXEMPLOS) -> pd.DataFrame:
    """Essenciais com centralidade abaixo da mediana (grau e betweenness) que o dano acerta.

    São os casos que a topologia sozinha não vê e a cascata vê: ordenados pelo
    dano (maior primeiro) e, no empate, pelo grau (menor primeiro). Para a
    discussão do artigo (o plano pede dois ou três).
    """
    tabela = genes.copy()
    tabela["percentil_grau"] = tabela["grau"].rank(pct=True)
    tabela["percentil_betweenness"] = tabela["betweenness"].rank(pct=True)
    baixos = tabela[tabela["essencial"] & (tabela["percentil_grau"] <= 0.5)
                    & (tabela["percentil_betweenness"] <= 0.5) & (tabela["dano"] > 0)]
    baixos = baixos.sort_values(["dano", "grau", "gene"], ascending=[False, True, True]).head(n)
    return baixos[["gene", "simbolo", "descricao", "grau", "percentil_grau", "betweenness",
                   "percentil_betweenness", "dano", "n_reacoes_perdidas", "choke"]].reset_index(drop=True)


def rodar(org: str) -> pd.DataFrame:
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

    Para a discussão: separe dois ou três genes essenciais com baixa centralidade.
    Devolve a tabela de rankings.
    """
    tabelas = grafo.carregar_tabelas(org)
    reacoes = tabelas["reacoes"]
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))

    # Medida principal (config.DANO_NOCAUTE) e a outra como sensibilidade.
    sensibilidade = next(m for m in nocaute.MEDIDAS if m != config.DANO_NOCAUTE)
    coluna_sens = f"dano_{MEIO[sensibilidade]}"
    dano = nocaute.dano_todos_genes(B, reacoes, medida=config.DANO_NOCAUTE).set_index("gene")
    dano_sens = nocaute.dano_todos_genes(B, reacoes, medida=sensibilidade).set_index("gene")["dano"]
    grau = nocaute.centralidade_por_gene(reacoes, metricas.graus(G)["total"])
    betw = nocaute.centralidade_por_gene(reacoes, metricas.betweenness(G))

    essenciais = validacao.carregar_keio()
    y = validacao.rotulos(dano.index, essenciais)
    fora_da_rede = sorted(essenciais - set(y.index))

    choke = metricas.choke_points(reacoes)
    no_choke = nocaute.genes_em_choke_points(reacoes, choke)
    descricao = tabelas["genes"].set_index("gene")
    genes = pd.DataFrame({
        "gene": y.index,
        "simbolo": descricao["simbolo"].reindex(y.index).fillna("").to_numpy(),
        "descricao": descricao["descricao"].reindex(y.index).fillna("").to_numpy(),
        "essencial": y.to_numpy(),
        "n_reacoes": dano["n_reacoes"].reindex(y.index).to_numpy(),
        "n_reacoes_perdidas": dano["n_reacoes_perdidas"].reindex(y.index).to_numpy(),
        "dano": dano["dano"].reindex(y.index).to_numpy(),
        coluna_sens: dano_sens.reindex(y.index).to_numpy(),
        "grau": grau.reindex(y.index).fillna(0).to_numpy(),
        "betweenness": betw.reindex(y.index).fillna(0).to_numpy(),
        "choke": no_choke.reindex(y.index).fillna(False).to_numpy(dtype=bool),
    })
    pontuacoes = genes.set_index("gene")
    rankings = {nome: pontuacoes[coluna] for nome, coluna in (
        ("dano", "dano"), ("grau", "grau"), ("betweenness", "betweenness"),
        (coluna_sens, coluna_sens), ("reacoes_perdidas", "n_reacoes_perdidas"))}
    avaliacao = validacao.avaliar_rankings(y, rankings)
    avaliacao.insert(1, "papel", avaliacao["ranking"].map(PAPEIS))

    roc = pd.concat([validacao.curva_roc(y, rankings[nome]).assign(ranking=nome) for nome in rankings],
                    ignore_index=True)[["ranking", "fpr", "tpr"]]
    resumo = pd.DataFrame([{
        "org": org, "versao": config.VERSAO_CASCATA, "medida_dano": config.DANO_NOCAUTE,
        "agregacao_centralidade": config.AGREGACAO_CENTRALIDADE_GENE,
        "genes_rede": len(y), "essenciais_keio": len(essenciais), "essenciais_na_rede": int(y.sum()),
        "essenciais_fora_da_rede": len(fora_da_rede),
        "genes_com_reacao_perdida": int((genes["n_reacoes_perdidas"] > 0).sum()),
        "genes_com_dano": int((genes["dano"] > 0).sum()),
        "reacoes_choke": len(choke), "reacoes_total": len(reacoes),
    }])

    _gravar(genes, "genes", org)
    _gravar(avaliacao, "rankings", org)
    _gravar(tabela_choke(genes, len(choke), len(reacoes)), "choke", org)
    _gravar(roc, "roc", org)
    _gravar(exemplos_baixa_centralidade(genes), "exemplos", org)
    _gravar(resumo, "resumo", org)
    return avaliacao


def plotar(org: str) -> None:
    """Lê exp2_roc e exp2_rankings e grava a figura com as três curvas ROC
    (dano, grau e betweenness; a sensibilidade e a linha de base ficam só na tabela)."""
    figuras.estilo()
    roc = pd.read_csv(config.arquivo_tabela("exp2", "roc", org))
    avaliacao = pd.read_csv(config.arquivo_tabela("exp2", "rankings", org)).set_index("ranking")
    rocs = {nome: roc[roc["ranking"] == nome] for nome in RANKINGS_FIGURA if nome in set(roc["ranking"])}
    aucs = {nome: float(avaliacao.loc[nome, "auc"]) for nome in rocs}
    figuras.salvar(figuras.curvas_roc(rocs, aucs), config.arquivo_figura("exp2", "roc", org))


def _resumo(avaliacao: pd.DataFrame, org: str) -> str:
    resumo = pd.read_csv(config.arquivo_tabela("exp2", "resumo", org)).iloc[0]
    choke = pd.read_csv(config.arquivo_tabela("exp2", "choke", org)).iloc[0]
    linhas = [f"{org}: {resumo['genes_rede']} genes na rede, {resumo['essenciais_na_rede']} essenciais da Keio "
              f"({resumo['essenciais_fora_da_rede']} dos {resumo['essenciais_keio']} ficam fora da rede); "
              f"{resumo['genes_com_dano']} genes com dano > 0",
              f"  {'ranking':18s} {'AUC':>5s} {'IC 95%':>13s} {'p':>8s} {'prec@10':>8s} {'prec@50':>8s}"]
    for linha in avaliacao.to_dict("records"):
        linhas.append(f"  {linha['ranking']:18s} {linha['auc']:5.3f} [{linha['auc_ic_inf']:.2f}, "
                      f"{linha['auc_ic_sup']:.2f}] {linha['p_auc']:8.1e} {linha['precisao_10']:8.2f} "
                      f"{linha['precisao_50']:8.2f}")
    linhas.append(f"  choke points: {choke['fracao_essenciais_choke']:.0%} essenciais "
                  f"(contra {choke['fracao_essenciais_outros']:.0%} fora), razão de chances "
                  f"{choke['razao_de_chances']:.2f}, p = {choke['p_fisher']:.1e}; taxa base "
                  f"{avaliacao['taxa_base'].iloc[0]:.0%}")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2: topologia vs. biologia (nocaute x Keio).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL,
                        help="código KEGG; só a E. coli (eco) tem gabarito")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if args.org != "eco":
        parser.error("o gabarito da Keio só existe para a E. coli (--org eco)")
    if not args.so_figuras:
        print(_resumo(rodar(args.org), args.org))
    plotar(args.org)


if __name__ == "__main__":
    main()
