"""Checagem, nos dados, das inferências do artigo (v2.0). Só para a E. coli.

Não é um experimento novo: confere quatro afirmações da discussão com os mesmos
grafos e as mesmas funções dos Exps 1, 2 e 4.
  (1) rankings    onde ficam, na gravata-borboleta, os primeiros metabólitos que a
                  remoção estática do Exp 1 tira por grau, betweenness e PageRank
                  (robustez.ordem_de_remocao, nutrientes protegidos): frações no
                  núcleo, na entrada, na saída e no resto, graus médios de entrada e
                  de saída e a fração de sumidouros (grau de saída 0), para os
                  primeiros k = round(f * N), f em config.TOPS_INFERENCIAS.
  (2) direção     genes cujo nocaute (regra OU, nocaute.reacoes_perdidas) tira
                  metabólitos do maior componente fraco, classificados pelo lado da
                  gravata-borboleta do que sai: "saida" se saem mais metabólitos da
                  saída que da entrada, "entrada" no contrário, "empate" se igual;
                  a taxa de essenciais (Keio) em cada lado, entre os genes que tiram
                  pelo menos n metabólitos (n em config.MINIMOS_LADO_INFERENCIAS).
                  Confere a contagem com a coluna dano_fraco do Exp 2.
  (3) chiquimato  nocaute de aroA a aroD (config.GENES_CHIQUIMATO): quantos
                  metabólitos saem do núcleo e quantos deixam de ser produzidos no
                  meio rico (fontes da avalanche) e no meio mínimo (fontes do
                  alcance), e quais aromáticos (config.AROMATICOS) estão entre eles.
  (4) pontes      na versão não dirigida (a do Exp 4): pontes, pontes internas (sem
                  folha nas pontas) e pontos de articulação da E. coli e das
                  config.N_MODELOS_NULOS redes do modelo de configuração do Exp 4
                  (modelos_classicos.gerar_modelos, as mesmas sementes).

Versões: (1) a do Exp 1 (config.VERSAO_ROBUSTEZ), (2) e (3) a do Exp 2
(config.VERSAO_CASCATA), (4) a do Exp 4 (config.VERSAO_COMPARACAO); hoje, as três
são sem moeda.

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
       resultados/tabelas/exp2_genes_<org>.csv  (rode antes o Exp 2)
Grava: resultados/tabelas/inferencias_rankings_<org>.csv   (1: um ranking e um k por linha)
       resultados/tabelas/inferencias_primeiros_<org>.csv  (1: os primeiros de cada ranking)
       resultados/tabelas/inferencias_genes_<org>.csv      (2: um gene por linha, com o lado)
       resultados/tabelas/inferencias_lados_<org>.csv      (2: genes e essenciais por lado)
       resultados/tabelas/inferencias_chiquimato_<org>.csv (3: um gene aro por linha)
       resultados/tabelas/inferencias_pontes_<org>.csv     (4: E. coli e cada rede de configuração)

Uso, na raiz:
    python -m experimentos.checar_inferencias            # ~15 s
    python -m experimentos.checar_inferencias --org eco
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import cascata, grafo, metricas, nocaute, robustez
from experimentos.biblioteca import modelos_classicos as mc

LADOS = ("nucleo", "entrada", "saida", "restante")          # as partes de metricas.componentes_bow_tie
CENTRALIDADES = ("grau", "betweenness", "pagerank")         # as estáticas do Exp 1, menos a aleatória
N_GENES_RESUMO = 12                                         # genes que mais tiram do maior fraco, no resumo

COLUNAS_RANKINGS = ["centralidade", "k", "fracao_k", *LADOS, "grau_entrada_medio", "grau_saida_medio",
                    "sumidouros"]
COLUNAS_PRIMEIROS = ["centralidade", "posicao", "metabolito", "nome", "lado", "grau_entrada", "grau_saida"]
COLUNAS_GENES = ["gene", "simbolo", "essencial", "n", "dano_fraco_exp2", *LADOS, "lado"]
COLUNAS_LADOS = ["minimo", "lado", "genes", "essenciais", "taxa"]
COLUNAS_CHIQUIMATO = ["simbolo", "gene", "n_reacoes_perdidas", "saem_nucleo", "aromaticos_nucleo",
                      "nao_produzidos_rico", "aromaticos_rico", "nao_produzidos_minimo", "aromaticos_minimo",
                      "nomes_nucleo", "nomes_rico"]
COLUNAS_PONTES = ["rede", "indice", "nos", "arestas", "pontes", "pontes_internas", "articulacao"]


# ---------------------------------------------------------------------------
# (1) Os primeiros de cada ranking na gravata-borboleta
# ---------------------------------------------------------------------------
def lado_bow_tie(G: nx.DiGraph) -> dict:
    """nó -> parte da gravata-borboleta (nucleo, entrada, saida ou restante)."""
    return {no: parte for parte, nos in metricas.componentes_bow_tie(G).items() for no in nos}


def tamanhos_top(n: int, fracoes: Iterable[float] = config.TOPS_INFERENCIAS) -> list[int]:
    """k = round(f * n) para cada fração, pelo menos 1 e no máximo n."""
    return [min(n, max(1, round(f * n))) for f in fracoes]


def rankings_bow_tie(G: nx.DiGraph, centralidades: Iterable[str] = CENTRALIDADES,
                     fracoes: Iterable[float] = config.TOPS_INFERENCIAS,
                     protegidos: Iterable[str] = config.NUTRIENTES,
                     n_primeiros: int = config.N_PRIMEIROS_INFERENCIAS,
                     nomes: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Onde ficam os primeiros removidos de cada estratégia estática.

    A ordem é a do Exp 1 (robustez.ordem_de_remocao com os nutrientes protegidos,
    que vão para o fim e nunca entram no topo). Retorna (rankings, primeiros):
        rankings:  uma linha por centralidade e k (COLUNAS_RANKINGS); as partes da
                   gravata e os sumidouros em fração dos k
        primeiros: os n_primeiros de cada ranking, com a parte e os graus
    """
    lado = lado_bow_tie(G)
    entrada, saida = dict(G.in_degree()), dict(G.out_degree())
    nomes = nomes or {}
    linhas, primeiros = [], []
    for centralidade in centralidades:
        ordem = robustez.ordem_de_remocao(G, centralidade, protegidos=protegidos)
        for k, f in zip(tamanhos_top(G.number_of_nodes(), fracoes), fracoes):
            topo = ordem[:k]
            partes = Counter(lado[no] for no in topo)
            linhas.append({"centralidade": centralidade, "k": k, "fracao_k": f,
                           **{p: partes[p] / k for p in LADOS},
                           "grau_entrada_medio": float(np.mean([entrada[no] for no in topo])),
                           "grau_saida_medio": float(np.mean([saida[no] for no in topo])),
                           "sumidouros": sum(saida[no] == 0 for no in topo) / k})
        primeiros += [{"centralidade": centralidade, "posicao": i + 1, "metabolito": no,
                       "nome": nomes.get(no, no), "lado": lado[no],
                       "grau_entrada": entrada[no], "grau_saida": saida[no]}
                      for i, no in enumerate(ordem[:n_primeiros])]
    return (pd.DataFrame(linhas, columns=COLUNAS_RANKINGS),
            pd.DataFrame(primeiros, columns=COLUNAS_PRIMEIROS))


# ---------------------------------------------------------------------------
# (2) A direção dos ramos: que lado da gravata cada nocaute desliga
# ---------------------------------------------------------------------------
def classificar_lado(entrada: int, saida: int) -> str:
    """"saida" se saem mais metabólitos da saída, "entrada" no contrário, "empate" se igual."""
    return "saida" if saida > entrada else "entrada" if entrada > saida else "empate"


def genes_por_lado(M: nx.MultiDiGraph, reacoes: pd.DataFrame, genes_exp2: pd.DataFrame) -> pd.DataFrame:
    """Os genes cujo nocaute tira algum metabólito do maior componente fraco.

    Para cada gene com reação perdida (regra OU), refaz o grafo sem as reações
    (nocaute.grafo_sem_reacoes) e conta quantos metabólitos do maior componente
    fraco da rede intacta ficam fora do novo, por parte da gravata (a da rede
    intacta). genes_exp2: a tabela exp2_genes (gene, simbolo, essencial,
    dano_fraco). Uma linha por gene (COLUNAS_GENES), em ordem de código.
    """
    G = grafo.grafo_simples(M)
    nos = list(G.nodes)
    lado = lado_bow_tie(G)
    fraco0 = max(nx.weakly_connected_components(G), key=len) if nos else set()
    tabela = genes_exp2.set_index("gene")
    linhas = []
    for gene in sorted(nocaute.reacoes_por_gene(reacoes)):
        perdidas = nocaute.reacoes_perdidas([gene], reacoes)
        if not perdidas:
            continue
        H = nocaute.grafo_sem_reacoes(M, perdidas, nos)
        saiu = fraco0 - max(nx.weakly_connected_components(H), key=len)
        if not saiu:
            continue
        partes = Counter(lado[no] for no in saiu)
        linha = tabela.loc[gene] if gene in tabela.index else None
        linhas.append({"gene": gene,
                       "simbolo": linha["simbolo"] if linha is not None else gene,
                       "essencial": bool(linha["essencial"]) if linha is not None else False,
                       "n": len(saiu),
                       "dano_fraco_exp2": int(linha["dano_fraco"]) if linha is not None else -1,
                       **{p: partes[p] for p in LADOS},
                       "lado": classificar_lado(partes["entrada"], partes["saida"])})
    return pd.DataFrame(linhas, columns=COLUNAS_GENES)


def taxas_por_lado(genes: pd.DataFrame, minimos: Iterable[int] = config.MINIMOS_LADO_INFERENCIAS) -> pd.DataFrame:
    """Genes e essenciais em cada lado, entre os que tiram pelo menos `minimo` metabólitos."""
    linhas = []
    for minimo in minimos:
        sub = genes[genes["n"] >= minimo]
        for lado, grupo in sub.groupby("lado", sort=True):
            linhas.append({"minimo": minimo, "lado": lado, "genes": len(grupo),
                           "essenciais": int(grupo["essencial"].sum()),
                           "taxa": float(grupo["essencial"].mean())})
    return pd.DataFrame(linhas, columns=COLUNAS_LADOS)


# ---------------------------------------------------------------------------
# (3) A via do chiquimato
# ---------------------------------------------------------------------------
def via_chiquimato(M: nx.MultiDiGraph, B: nx.DiGraph, reacoes: pd.DataFrame, genes_exp2: pd.DataFrame,
                   simbolos: Iterable[str] = config.GENES_CHIQUIMATO,
                   aromaticos: dict = config.AROMATICOS, nomes: dict | None = None) -> pd.DataFrame:
    """O nocaute de cada gene de `simbolos`: o que sai do núcleo e o que deixa de ser produzido.

    Núcleo: o maior componente forte do grafo simples de M (metricas.maior_componente_forte).
    Produção: cascata.cascata na rede bipartida B, com as fontes da avalanche (meio
    rico) e as do alcance (meio mínimo), de nocaute.fontes_da_medida. Os símbolos
    que não estão na tabela do Exp 2 ficam de fora.
    """
    nomes = nomes or {}
    G = grafo.grafo_simples(M)
    nos = list(G.nodes)
    nucleo0 = metricas.maior_componente_forte(G)
    fontes = {medida: nocaute.fontes_da_medida(B, medida) for medida in ("avalanche", "alcance")}
    gene_do_simbolo = dict(zip(genes_exp2["simbolo"], genes_exp2["gene"]))

    def quais(conjunto: set) -> str:
        return ", ".join(nome for codigo, nome in aromaticos.items() if codigo in conjunto)

    def lista(conjunto: set) -> str:
        return "; ".join(sorted(nomes.get(no, no) for no in conjunto))

    linhas = []
    for simbolo in simbolos:
        if simbolo not in gene_do_simbolo:
            continue
        gene = gene_do_simbolo[simbolo]
        perdidas = nocaute.reacoes_perdidas([gene], reacoes)
        saiu = nucleo0 - metricas.maior_componente_forte(nocaute.grafo_sem_reacoes(M, perdidas, nos))
        rico = cascata.cascata(B, reacoes_removidas=perdidas, fontes=fontes["avalanche"])
        minimo = cascata.cascata(B, reacoes_removidas=perdidas, fontes=fontes["alcance"])
        linhas.append({"simbolo": simbolo, "gene": gene, "n_reacoes_perdidas": len(perdidas),
                       "saem_nucleo": len(saiu), "aromaticos_nucleo": quais(saiu),
                       "nao_produzidos_rico": len(rico), "aromaticos_rico": quais(rico),
                       "nao_produzidos_minimo": len(minimo), "aromaticos_minimo": quais(minimo),
                       "nomes_nucleo": lista(saiu), "nomes_rico": lista(rico)})
    return pd.DataFrame(linhas, columns=COLUNAS_CHIQUIMATO)


def aromaticos_nas_fontes(B: nx.DiGraph, aromaticos: dict = config.AROMATICOS) -> dict[str, bool]:
    """Cada aromático está entre as fontes do meio rico (nocaute.fontes_da_medida, "avalanche")?"""
    fontes = nocaute.fontes_da_medida(B, "avalanche")
    return {nome: codigo in fontes for codigo, nome in aromaticos.items()}


# ---------------------------------------------------------------------------
# (4) Pontes na versão não dirigida
# ---------------------------------------------------------------------------
def pontes(U: nx.Graph) -> dict:
    """Pontes, pontes internas (as duas pontas com grau > 1) e pontos de articulação de U."""
    lista = list(nx.bridges(U))
    internas = sum(U.degree(u) > 1 and U.degree(v) > 1 for u, v in lista)
    return {"nos": U.number_of_nodes(), "arestas": U.number_of_edges(), "pontes": len(lista),
            "pontes_internas": internas, "articulacao": sum(1 for _ in nx.articulation_points(U))}


def pontes_contra_configuracao(U: nx.Graph, n_redes: int = config.N_MODELOS_NULOS,
                               semente: int = config.SEMENTE) -> pd.DataFrame:
    """A rede U (rede = "eco", indice vazio) e cada rede do modelo de configuração do Exp 4.

    As redes de configuração são as de modelos_classicos.gerar_modelos(U,
    "configuracao", n_redes, semente): a i-ésima usa a semente semente * 1000 + i.
    """
    linhas = [{"rede": "eco", "indice": None, **pontes(U)}]
    linhas += [{"rede": "configuracao", "indice": i, **pontes(C)}
               for i, C in enumerate(mc.gerar_modelos(U, "configuracao", n_redes, semente))]
    tabela = pd.DataFrame(linhas, columns=COLUNAS_PONTES)
    tabela["indice"] = tabela["indice"].astype("Int64")
    return tabela


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _grafos(reacoes: pd.DataFrame) -> dict:
    """Grafos por versão (só as que as checagens usam), cada um construído uma vez."""
    versoes = {config.VERSAO_ROBUSTEZ, config.VERSAO_CASCATA, config.VERSAO_COMPARACAO}
    resultado = {}
    for versao in versoes:
        remover = config.VERSOES_MOEDA[versao]
        M = grafo.grafo_metabolitos(reacoes, remover_moeda=remover)
        resultado[versao] = {"M": M, "G": grafo.grafo_simples(M)}
    resultado[config.VERSAO_CASCATA]["B"] = grafo.grafo_bipartido(
        reacoes, remover_moeda=config.VERSOES_MOEDA[config.VERSAO_CASCATA])
    return resultado


def rodar(org: str = config.ORGANISMO_PRINCIPAL) -> dict:
    """Faz as quatro checagens, grava as tabelas e devolve um dicionário com elas."""
    arquivo_genes = config.arquivo_tabela("exp2", "genes", org)
    if not arquivo_genes.exists():
        raise FileNotFoundError(f"{arquivo_genes} não existe: rode antes o Exp 2 "
                                f"(python -m experimentos.exp2_biologia --org {org})")
    genes_exp2 = pd.read_csv(arquivo_genes)
    tabelas = grafo.carregar_tabelas(org)
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    grafos = _grafos(reacoes)
    G1 = grafos[config.VERSAO_ROBUSTEZ]["G"]
    M2, B2 = grafos[config.VERSAO_CASCATA]["M"], grafos[config.VERSAO_CASCATA]["B"]
    G4 = grafos[config.VERSAO_COMPARACAO]["G"]

    rankings, primeiros = rankings_bow_tie(G1, nomes=nomes)
    genes = genes_por_lado(M2, reacoes, genes_exp2)
    resultado = {
        "rankings": rankings,
        "primeiros": primeiros,
        "genes": genes,
        "lados": taxas_por_lado(genes),
        "chiquimato": via_chiquimato(M2, B2, reacoes, genes_exp2, nomes=nomes),
        "pontes": pontes_contra_configuracao(mc.rede_nao_dirigida(G4)),
    }
    config.PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    for conteudo, tabela in resultado.items():
        tabela.to_csv(config.arquivo_tabela("inferencias", conteudo, org), index=False)
    resultado["rede"] = {"nos": G1.number_of_nodes(), "arestas": G1.number_of_edges(),
                         "gravata": metricas.bow_tie(G1)}
    resultado["aromaticos_nas_fontes"] = aromaticos_nas_fontes(B2)
    return resultado


def _resumo(r: dict) -> str:
    linhas = [f"  rede ({config.VERSAO_ROBUSTEZ}): {r['rede']['nos']} nós, {r['rede']['arestas']} arestas;"
              f" gravata {r['rede']['gravata']}", "", "  (1) primeiros removidos de cada ranking (estáticas do Exp 1)"]
    for c in r["rankings"].to_dict("records"):
        linhas.append(f"      {c['centralidade']:11s} top {c['k']:3d}: núcleo {c['nucleo']:.0%}  entrada "
                      f"{c['entrada']:.0%}  saída {c['saida']:.0%}  resto {c['restante']:.0%} | grau de entrada "
                      f"{c['grau_entrada_medio']:.1f}, de saída {c['grau_saida_medio']:.1f} | sumidouros "
                      f"{c['sumidouros']:.0%}")
    for centralidade, grupo in r["primeiros"].groupby("centralidade", sort=False):
        linhas.append(f"      {centralidade}: " + "; ".join(
            f"{p['nome']} ({p['lado']}, {p['grau_entrada']}/{p['grau_saida']})" for p in grupo.to_dict("records")))
    genes = r["genes"]
    confere = bool((genes["n"] == genes["dano_fraco_exp2"]).all())
    linhas += ["", f"  (2) {len(genes)} genes tiram metabólitos do maior componente fraco "
                   f"(contagem igual à do Exp 2: {'sim' if confere else 'NÃO'})"]
    for c in r["lados"].to_dict("records"):
        linhas.append(f"      n >= {c['minimo']}  {c['lado']:8s} {c['genes']:4d} genes, {c['essenciais']:3d} "
                      f"essenciais (taxa {c['taxa']:.2f})")
    maiores = genes.sort_values(["n", "gene"], ascending=[False, True]).head(N_GENES_RESUMO)
    linhas.append("      os que mais tiram: " + "; ".join(
        f"{g['simbolo']}{'*' if g['essencial'] else ''} {g['n']} ({g['lado']})" for g in maiores.to_dict("records"))
        + "   (* essencial na Keio)")
    linhas += ["", "  (3) via do chiquimato; aromáticos entre as fontes do meio rico: "
               + ", ".join(f"{k} {'sim' if v else 'não'}" for k, v in r["aromaticos_nas_fontes"].items())]
    for c in r["chiquimato"].to_dict("records"):
        linhas.append(f"      {c['simbolo']}: saem do núcleo {c['saem_nucleo']} [{c['aromaticos_nucleo']}]; não "
                      f"produzidos no meio rico {c['nao_produzidos_rico']} [{c['aromaticos_rico']}], no mínimo "
                      f"{c['nao_produzidos_minimo']} [{c['aromaticos_minimo']}]")
    p = r["pontes"]
    eco = p[p["rede"] == "eco"].iloc[0]
    nulas = p[p["rede"] == "configuracao"][["pontes", "pontes_internas", "articulacao"]]
    linhas += ["", f"  (4) não dirigida ({config.VERSAO_COMPARACAO}): {eco['nos']} nós, {eco['arestas']} arestas",
               f"      E. coli: {eco['pontes']} pontes, {eco['pontes_internas']} internas, "
               f"{eco['articulacao']} pontos de articulação"]
    if len(nulas):
        media, dp = nulas.mean(), nulas.std(ddof=1) if len(nulas) > 1 else nulas.mean() * np.nan
        linhas.append(f"      configuração ({len(nulas)} redes): " + ", ".join(
            f"{coluna} {media[coluna]:.1f} ± {dp[coluna]:.1f}" for coluna in nulas.columns))
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checa nos dados as inferências do artigo.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    args = parser.parse_args()
    inicio = time.perf_counter()
    print(_resumo(rodar(args.org)))
    print(f"\n  ({time.perf_counter() - inicio:.0f} s; tabelas em {config.PASTA_TABELAS})")


if __name__ == "__main__":
    main()
