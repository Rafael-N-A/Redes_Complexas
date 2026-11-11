"""Experimento 2 — Dano x essencialidade dos genes (P2: o gene cuja deleção causa mais dano
na rede é essencial?). Tabela 3 e Figura 4 do artigo.

A resposta depende de como o dano é medido. Duas medidas principais:
  estrutural  "quebrar o maior componente": quantos metabólitos deixam o maior
              componente forte (o núcleo) e o maior componente fraco quando o
              gene é nocauteado (nocaute.dano_estrutural_todos_genes). É a mesma
              régua das curvas do Exp 1, e é o que liga P1 a P2;
  funcional   "deixar de produzir compostos": quantos metabólitos que a rede
              intacta produz deixam de ser produzidos (a avalanche da cascata,
              nocaute.dano_todos_genes, no meio rico; a Keio foi medida em LB);
e uma complementar, a queda relativa da eficiência global dirigida, (E0 - E)/E0.
O nocaute remove as reações que só aquele gene catalisa (regra OU): remove
arestas da rede, nunca metabólitos.

Só para a E. coli, o único organismo com gabarito (coleção Keio, Baba et al.,
2006): 103 dos 300 genes essenciais estão entre os 952 genes da rede (os
positivos); os outros 849 são os negativos. Roda na versão config.VERSAO_PRINCIPAL.
A literatura diz que a correlação com centralidade é fraca (Samal et al., 2006).

Rankings (maior pontuação = mais provável de ser essencial), os blocos da Tabela 3:
  dano               metabólitos perdidos na cascata ao deletar o gene, meio rico
                     (config.DANO_NOCAUTE)                                 [funcional]
  dano_forte         metabólitos que deixam o maior componente forte      [estrutural]
  dano_fraco         metabólitos que deixam o maior componente fraco      [estrutural]
  dano_eficiencia    queda relativa da eficiência global dirigida         [estrutural, complementar]
  grau, betweenness, pagerank  centralidade do metabólito mais central das
                     reações do gene (config.AGREGACAO_CENTRALIDADE_GENE)  [centralidade]
  reacoes_perdidas   só quantas reações o gene derruba, sem a cascata    [linha de base]
A linha de base mostra quanto a propagação pela rede acrescenta à perda direta.

Checagem das isoenzimas: 378 dos 952 genes não perdem nenhuma reação (toda
reação deles tem outro gene) e têm dano 0 em qualquer medida. Para mostrar que
a vantagem da cascata não vem só de saber quem tem isoenzima, a avaliação se
repete só com os 574 genes que perdem pelo menos uma reação (coluna
subconjunto: "todos" ou "perde_reacao" nas tabelas de rankings e correlações).

Lê:    dados/eco/reacoes.csv, genes.csv; dados/keio/essenciais.csv
Grava: resultados/tabelas/exp2_genes_eco.csv        (uma linha por gene: essencial, danos estrutural e
                                                     funcional, centralidades)
       resultados/tabelas/exp2_rankings_eco.csv     (Tabela 3: por subconjunto e ranking, AUC com IC 95% e p,
                                                     precisão@k)
       resultados/tabelas/exp2_correlacoes_eco.csv  (Spearman entre o dano estrutural e o funcional)
       resultados/tabelas/exp2_exemplos_dano_eco.csv (os três grupos de casos de config.EXEMPLOS_DANO_EXP2)
       resultados/tabelas/exp2_essenciais_por_faixa_eco.csv (genes e essenciais por faixa de dano, nos que
                                                     perdem reação; estrutural e funcional)
       resultados/tabelas/exp2_resumo_eco.csv       (cobertura da Keio na rede e a rede intacta)
       resultados/figuras/exp2_essenciais_por_faixa_eco.png  (Figura 4: % de essenciais por faixa de dano,
                                                     estrutural x funcional)

Uso, na raiz:
    python -m experimentos.exp2_biologia
    python -m experimentos.exp2_biologia --so-figuras
"""
from __future__ import annotations

import argparse
import math
import time

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr

import config
from experimentos.biblioteca import figuras, grafo, metricas, nocaute, validacao

PAPEIS = {"dano": "principal (funcional)", "dano_forte": "principal (estrutural)",
          "dano_fraco": "estrutural", "dano_eficiencia": "complementar (estrutural)",
          "grau": "centralidade", "betweenness": "centralidade", "pagerank": "centralidade",
          "reacoes_perdidas": "linha de base"}
# ranking -> coluna da tabela de genes; a ordem é a das tabelas.
RANKINGS = {"dano": "dano", "dano_forte": "dano_forte", "dano_fraco": "dano_fraco",
            "dano_eficiencia": "dano_eficiencia", "grau": "grau", "betweenness": "betweenness",
            "pagerank": "pagerank", "reacoes_perdidas": "n_reacoes_perdidas"}
SUBCONJUNTOS = ("todos", "perde_reacao")
# Pares do Spearman (dano funcional x estrutural e entre as estruturais).
PARES_CORRELACAO = (("dano", "dano_forte"), ("dano", "dano_fraco"), ("dano", "dano_eficiencia"),
                    ("dano_forte", "dano_eficiencia"), ("dano_forte", "dano_fraco"))
# Figura 4, % de essenciais por faixa de dano: a principal estrutural e a funcional, nesta ordem.
MEDIDAS_FAIXAS = ("dano_forte", "dano")
COLUNAS_FAIXAS = ["medida", "faixa", "minimo", "maximo", "genes", "essenciais", "taxa"]


def _gravar(tabela: pd.DataFrame, conteudo: str, org: str) -> None:
    destino = config.arquivo_tabela("exp2", conteudo, org)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)


COLUNAS_EXEMPLOS_DANO = ["grupo", "gene", "simbolo", "descricao", "essencial", "n_reacoes", "n_reacoes_perdidas",
                         "dano", "posicao_dano", "dano_forte", "posicao_dano_forte", "dano_fraco",
                         "dano_eficiencia", "posicao_dano_eficiencia", "grau", "motivo"]


def exemplos_dano(genes: pd.DataFrame, grupos: dict = config.EXEMPLOS_DANO_EXP2) -> pd.DataFrame:
    """Os casos da discussão: uma linha por gene de cada grupo de config.EXEMPLOS_DANO_EXP2.

    grupos: grupo -> (símbolos, motivo). Junta as medidas de dano e a posição do
    gene em cada ranking (1 = maior dano entre todos os genes da rede; empates
    ficam com a melhor posição). Símbolos que não estão na rede são avisados e
    ficam de fora.
    """
    tabela = genes.copy()
    for coluna in ("dano", "dano_forte", "dano_eficiencia"):
        tabela[f"posicao_{coluna}"] = tabela[coluna].rank(ascending=False, method="min").astype(int)
    por_simbolo = tabela.set_index("simbolo")
    linhas, faltando = [], []
    for grupo, (simbolos, motivo) in grupos.items():
        for simbolo in simbolos:
            if simbolo not in por_simbolo.index:
                faltando.append(simbolo)
                continue
            linha = por_simbolo.loc[[simbolo]].iloc[0]
            linhas.append({"grupo": grupo, "simbolo": simbolo, "motivo": motivo,
                           **{c: linha[c] for c in COLUNAS_EXEMPLOS_DANO if c not in ("grupo", "simbolo", "motivo")}})
    if faltando:
        print(f"aviso: genes dos exemplos fora da rede: {', '.join(faltando)}")
    return pd.DataFrame(linhas, columns=COLUNAS_EXEMPLOS_DANO)


def avaliar_subconjuntos(y: pd.Series, pontuacoes: pd.DataFrame, rankings: dict[str, str],
                         perde_reacao: pd.Series) -> pd.DataFrame:
    """validacao.avaliar_rankings em todos os genes e só nos que perdem pelo menos uma reação.

    rankings: nome -> coluna de pontuacoes. Coluna nova "subconjunto" ("todos" ou
    "perde_reacao") e "papel" (PAPEIS), antes das de avaliar_rankings.
    """
    partes = []
    for subconjunto in SUBCONJUNTOS:
        manter = y.index if subconjunto == "todos" else y.index[perde_reacao.reindex(y.index).to_numpy(dtype=bool)]
        series = {nome: pontuacoes.loc[manter, coluna] for nome, coluna in rankings.items()}
        avaliacao = validacao.avaliar_rankings(y.loc[manter], series)
        avaliacao.insert(0, "subconjunto", subconjunto)
        avaliacao.insert(2, "papel", avaliacao["ranking"].map(PAPEIS))
        partes.append(avaliacao)
    return pd.concat(partes, ignore_index=True)


def correlacoes(genes: pd.DataFrame, pares=PARES_CORRELACAO) -> pd.DataFrame:
    """Spearman entre as medidas de dano de cada gene, em todos e só nos que perdem reação."""
    linhas = []
    for subconjunto in SUBCONJUNTOS:
        parte = genes if subconjunto == "todos" else genes[genes["n_reacoes_perdidas"] > 0]
        for x, y in pares:
            if parte[x].nunique() < 2 or parte[y].nunique() < 2:
                rho, p = float("nan"), float("nan")
            else:
                resultado = spearmanr(parte[x], parte[y])
                rho, p = float(resultado.statistic), float(resultado.pvalue)
            linhas.append({"subconjunto": subconjunto, "medida_x": x, "medida_y": y,
                           "spearman": rho, "p": p, "n_genes": len(parte)})
    return pd.DataFrame(linhas)


def essenciais_por_faixa(genes: pd.DataFrame, medidas=MEDIDAS_FAIXAS,
                         faixas=config.FAIXAS_DANO_EXP2) -> pd.DataFrame:
    """Quantos genes e quantos essenciais em cada faixa de dano, por medida.

    Só os genes que perdem pelo menos uma reação (os outros têm dano 0 em
    qualquer medida). faixas: (rótulo, maior dano da faixa) em ordem crescente;
    a faixa vai do teto da anterior + 1 até o seu (a primeira começa em 0).
    Uma linha por medida e faixa (COLUNAS_FAIXAS); taxa = essenciais / genes
    (NaN numa faixa vazia). minimo e maximo são os limites inteiros da faixa
    (maximo vazio na última, sem teto).
    """
    perdem = genes[genes["n_reacoes_perdidas"] > 0]
    limites = [-1, *(teto for _, teto in faixas)]
    rotulos = [rotulo for rotulo, _ in faixas]
    linhas = []
    for medida in medidas:
        faixa = pd.cut(perdem[medida], limites, labels=rotulos)
        contagem = perdem.groupby(faixa, observed=False)["essencial"].agg(["size", "sum"])
        for (rotulo, teto), anterior in zip(faixas, limites):
            n, essenciais = int(contagem.loc[rotulo, "size"]), int(contagem.loc[rotulo, "sum"])
            linhas.append({"medida": medida, "faixa": rotulo, "minimo": int(anterior) + 1,
                           "maximo": None if math.isinf(teto) else int(teto), "genes": n,
                           "essenciais": essenciais, "taxa": essenciais / n if n else float("nan")})
    tabela = pd.DataFrame(linhas, columns=COLUNAS_FAIXAS)
    tabela["maximo"] = tabela["maximo"].astype("Int64")
    return tabela


def rodar(org: str) -> pd.DataFrame:
    """Simula o nocaute de cada gene, mede o dano de três formas e compara os rankings com a Keio.

    Passos:
      1. B = grafo bipartido e M = grafo metabólito-cêntrico da versão principal; G = grafo_simples(M).
      2. funcional: nocaute.dano_todos_genes(B, reacoes);
         estrutural: nocaute.dano_estrutural_todos_genes(M, reacoes).
      3. centralidades por gene: nocaute.centralidade_por_gene com o grau, a betweenness
         e o PageRank (nx.pagerank) dos metabólitos de G.
      4. y = validacao.rotulos(genes, validacao.carregar_keio()); reporta quantos essenciais
         da Keio ficaram fora da rede.
      5. avaliar_subconjuntos: validacao.avaliar_rankings (AUC com IC 95%, p de Mann–Whitney,
         precisão@k) em todos os genes e nos que perdem reação.
      6. Spearman entre o dano estrutural e o funcional (correlacoes).
      7. Exemplos (exemplos_dano, config.EXEMPLOS_DANO_EXP2) e essenciais por faixa de dano
         (essenciais_por_faixa, config.FAIXAS_DANO_EXP2).
      8. Grava as tabelas com config.arquivo_tabela("exp2", <conteudo>, org).
    Devolve a tabela de rankings (os dois subconjuntos).
    """
    inicio = time.perf_counter()
    essenciais = validacao.carregar_keio()  # primeiro: sem o gabarito, não vale a pena simular os nocautes
    tabelas = grafo.carregar_tabelas(org)
    reacoes = tabelas["reacoes"]
    remover = config.VERSOES_MOEDA[config.VERSAO_PRINCIPAL]
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    M = grafo.grafo_metabolitos(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(M)

    dano = nocaute.dano_todos_genes(B, reacoes, medida=config.DANO_NOCAUTE).set_index("gene")
    estrutural = nocaute.dano_estrutural_todos_genes(M, reacoes).set_index("gene")
    centralidades = {
        "grau": metricas.graus(G)["total"],
        "betweenness": metricas.betweenness(G),
        "pagerank": pd.Series(nx.pagerank(G), name="pagerank", dtype=float),
    }
    por_gene = {nome: nocaute.centralidade_por_gene(reacoes, valores) for nome, valores in centralidades.items()}

    y = validacao.rotulos(dano.index, essenciais)
    fora_da_rede = sorted(essenciais - set(y.index))

    descricao = tabelas["genes"].set_index("gene")
    genes = pd.DataFrame({
        "gene": y.index,
        "simbolo": descricao["simbolo"].reindex(y.index).fillna("").to_numpy(),
        "descricao": descricao["descricao"].reindex(y.index).fillna("").to_numpy(),
        "essencial": y.to_numpy(),
        "n_reacoes": dano["n_reacoes"].reindex(y.index).to_numpy(),
        "n_reacoes_perdidas": dano["n_reacoes_perdidas"].reindex(y.index).to_numpy(),
        "dano": dano["dano"].reindex(y.index).to_numpy(),
        **{coluna: estrutural[coluna].reindex(y.index).to_numpy()
           for coluna in ("arestas_perdidas", *metricas.COLUNAS_DANO_ESTRUTURAL)},
        **{nome: serie.reindex(y.index).fillna(0).to_numpy() for nome, serie in por_gene.items()},
    })
    pontuacoes = genes.set_index("gene")
    perde_reacao = pontuacoes["n_reacoes_perdidas"] > 0
    avaliacao = avaliar_subconjuntos(y, pontuacoes, RANKINGS, perde_reacao)

    antes = metricas.medidas_estruturais(G, G.number_of_nodes())
    resumo = pd.DataFrame([{
        "org": org, "versao": config.VERSAO_PRINCIPAL, "medida_dano": config.DANO_NOCAUTE,
        "agregacao_centralidade": config.AGREGACAO_CENTRALIDADE_GENE,
        "genes_rede": len(y), "essenciais_keio": len(essenciais), "essenciais_na_rede": int(y.sum()),
        "essenciais_fora_da_rede": len(fora_da_rede),
        "genes_com_reacao_perdida": int(perde_reacao.sum()),
        "essenciais_com_reacao_perdida": int((y & perde_reacao.reindex(y.index)).sum()),
        "genes_com_dano": int((genes["dano"] > 0).sum()),
        "genes_com_dano_forte": int((genes["dano_forte"] > 0).sum()),
        "genes_com_dano_eficiencia": int((genes["dano_eficiencia"] > 0).sum()),
        "reacoes_total": len(reacoes),
        "metabolitos_grafo": G.number_of_nodes(), "arestas_grafo": G.number_of_edges(),
        "nucleo_intacto": antes["maior_forte"], "maior_fraco_intacto": antes["maior_fraco"],
        "eficiencia_intacta": antes["eficiencia"],
    }])

    _gravar(genes, "genes", org)
    _gravar(avaliacao, "rankings", org)
    _gravar(correlacoes(genes), "correlacoes", org)
    _gravar(exemplos_dano(genes), "exemplos_dano", org)
    _gravar(essenciais_por_faixa(genes), "essenciais_por_faixa", org)
    _gravar(resumo, "resumo", org)
    print(f"{org}: nocautes de {len(y)} genes ({time.perf_counter() - inicio:.0f} s)")
    return avaliacao


def plotar(org: str) -> None:
    """Lê exp2_essenciais_por_faixa e grava a Figura 4: % de essenciais por faixa de dano,
    estrutural (queda do núcleo) x funcional (cascata), nos genes que perdem reação."""
    figuras.estilo()
    faixas = pd.read_csv(config.arquivo_tabela("exp2", "essenciais_por_faixa", org))
    figuras.salvar(figuras.barras_essenciais_por_faixa(faixas),
                   config.arquivo_figura("exp2", "essenciais_por_faixa", org))


def _resumo(avaliacao: pd.DataFrame, org: str) -> str:
    resumo = pd.read_csv(config.arquivo_tabela("exp2", "resumo", org)).iloc[0]
    correlacao = pd.read_csv(config.arquivo_tabela("exp2", "correlacoes", org))
    exemplos = pd.read_csv(config.arquivo_tabela("exp2", "exemplos_dano", org))
    faixas = pd.read_csv(config.arquivo_tabela("exp2", "essenciais_por_faixa", org))
    linhas = [f"{org}: {resumo['genes_rede']} genes na rede, {resumo['essenciais_na_rede']} essenciais da Keio "
              f"({resumo['essenciais_fora_da_rede']} dos {resumo['essenciais_keio']} ficam fora da rede); "
              f"{resumo['genes_com_reacao_perdida']} genes perdem pelo menos uma reação, "
              f"{resumo['genes_com_dano']} com dano funcional > 0, {resumo['genes_com_dano_forte']} tiram "
              f"metabólitos do núcleo",
              f"  rede intacta: núcleo (maior componente forte) = {resumo['nucleo_intacto']}, maior componente "
              f"fraco = {resumo['maior_fraco_intacto']}, eficiência = {resumo['eficiencia_intacta']:.4f}",
              f"  {'ranking':18s} {'AUC (952)':>9s} {'IC 95%':>13s} {'p':>8s} {'prec@10':>8s}"
              f" {'AUC (perde reação)':>19s}"]
    por_sub = avaliacao.set_index(["subconjunto", "ranking"])
    for linha in avaliacao[avaliacao["subconjunto"] == "todos"].to_dict("records"):
        perde = por_sub.loc[("perde_reacao", linha["ranking"]), "auc"]
        linhas.append(f"  {linha['ranking']:18s} {linha['auc']:9.3f} [{linha['auc_ic_inf']:.2f}, "
                      f"{linha['auc_ic_sup']:.2f}] {linha['p_auc']:8.1e} {linha['precisao_10']:8.2f} "
                      f"{perde:19.3f}")
    linhas.append(f"  taxa base (essenciais / genes da rede): {avaliacao['taxa_base'].iloc[0]:.0%}")
    perde = correlacao[correlacao["subconjunto"] == "perde_reacao"].set_index(["medida_x", "medida_y"])
    linhas.append("  Spearman nos genes que perdem reação: cascata x eficiência = "
                  f"{perde.loc[('dano', 'dano_eficiencia'), 'spearman']:.2f}, cascata x núcleo = "
                  f"{perde.loc[('dano', 'dano_forte'), 'spearman']:.2f}, cascata x componente fraco = "
                  f"{perde.loc[('dano', 'dano_fraco'), 'spearman']:.2f}")
    linhas.append(f"  {'exemplo':30s} {'gene':6s} {'essencial':>9s} {'cascata':>7s} {'núcleo':>6s} {'eficiência':>10s}")
    for linha in exemplos.to_dict("records"):
        linhas.append(f"  {linha['grupo']:30s} {linha['simbolo']:6s} {str(bool(linha['essencial'])):>9s} "
                      f"{linha['dano']:7d} {linha['dano_forte']:6d} {linha['dano_eficiencia']:10.3f}")
    linhas.append(f"  essenciais por faixa de dano ({resumo['genes_com_reacao_perdida']} genes que perdem reação;"
                  f" faixas {', '.join(faixas['faixa'].drop_duplicates())}):")
    for medida, parte in faixas.groupby("medida", sort=False):
        linhas.append(f"    {medida:12s} " + "   ".join(
            f"{f['essenciais']}/{f['genes']} ({f['taxa']:.0%})" for f in parte.to_dict("records")))
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2: dano do nocaute de cada gene x essencialidade (Keio).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL,
                        help="código KEGG; só a E. coli (eco) tem gabarito")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só a figura, sem recalcular")
    args = parser.parse_args()

    if args.org != "eco":
        parser.error("o gabarito da Keio só existe para a E. coli (--org eco)")
    if not args.so_figuras:
        print(_resumo(rodar(args.org), args.org))
    plotar(args.org)


if __name__ == "__main__":
    main()
