"""Experimento 3 — Viés por substância (cobre o item "spreading" do roteiro).

Pergunta: remover uma substância específica (piruvato, acetil-CoA, um
intermediário do ciclo de Krebs) dispara uma cascata de falhas? Até onde ela
se espalha?

Roda numa versão só (config.VERSAO_CASCATA): com a moeda sempre disponível,
a cascata dá o mesmo resultado com ou sem moeda.

Três grupos (config.SUBSTANCIAS_EXP3 e config.N_CONTROLES_EXP3):
  substancia  as 6 escolhidas por relevância biológica
  hub         controle: os 6 de maior grau fora das escolhidas
  baixo_grau  controle: 6 sorteados entre os de menor grau
E a avalanche de cada um dos metabólitos que a rede intacta produz, para a
distribuição dos tamanhos e para comparar a cascata com a centralidade.

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp3_substancias_<org>.csv  (6 escolhidas + 12 controles)
       resultados/tabelas/exp3_avalanches_<org>.csv   (avalanche de todos os metabólitos)
       resultados/tabelas/exp3_resumo_<org>.csv       (distribuição, ajuste e correlações)
       resultados/tabelas/exp3_ccdf_<org>.csv         (CCDF das avalanches >= 1 e ajustes)
       resultados/figuras/exp3_avalanche_vs_grau_<org>.png
       resultados/figuras/exp3_distribuicao_<org>.png

Uso, na raiz:
    python -m experimentos.exp3_vies --org eco
    python -m experimentos.exp3_vies --so-figuras
"""
from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from experimentos.biblioteca import cascata, figuras, grafo, metricas

COLUNAS_SUBSTANCIAS = ["grupo", "metabolito", "nome", "grau", "betweenness", "avalanche", "fracao",
                       "profundidade", "posicao_avalanche", "posicao_grau"]
COLUNAS_AVALANCHES = ["metabolito", "nome", "grau", "betweenness", "avalanche", "fracao"]


def escolher_controles(G: nx.DiGraph, excluir: set[str], n: int, rng: np.random.Generator,
                       candidatos: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Os dois grupos de controle.

    Retorna (hubs, baixo_grau):
      hubs:       os n nós de maior grau total, fora de excluir; empates
                  desfeitos pelo código, como em metricas.top_por_grau
      baixo_grau: n nós sorteados com rng entre a fração config.QUANTIL_BAIXO_GRAU
                  de menor grau, fora de excluir (todos, se forem menos de n)
    candidatos: se dado, os dois grupos saem só dele (no exp3, os metabólitos
    que a rede intacta produz: remover um que nunca é produzido daria avalanche
    0 por definição, e o controle não diria nada). O quantil de grau é sempre
    calculado sobre todos os nós de G.
    """
    graus = metricas.graus(G)["total"]
    limite = float(np.quantile(graus, config.QUANTIL_BAIXO_GRAU))
    permitidos = set(graus.index) - set(excluir)
    if candidatos is not None:
        permitidos &= set(candidatos)
    tabela = graus[graus.index.isin(permitidos)].rename("total").reset_index()
    tabela = tabela.sort_values(["total", "metabolito"], ascending=[False, True])
    hubs = tabela["metabolito"].head(n).tolist()
    baixos = sorted(tabela.loc[tabela["total"] <= limite, "metabolito"])
    baixo_grau = sorted(rng.choice(baixos, size=min(n, len(baixos)), replace=False).tolist()) if baixos else []
    return hubs, baixo_grau


def _posicao(serie: pd.Series) -> pd.Series:
    """Posição de cada valor do maior para o menor (1 = maior); empates ficam com a melhor posição."""
    return serie.rank(ascending=False, method="min").astype(int)


def profundidade(G: nx.DiGraph, removido: str, perdidos: set[str]) -> int:
    """Até onde a avalanche chega: a maior distância (em reações) de removido a um perdido.

    Todo metabólito perdido descende do removido no grafo metabólito-cêntrico
    (só perde a produção quem dependia dele), então a distância sempre existe.
    0 se não perdeu nada.
    """
    if not perdidos or removido not in G:
        return 0
    distancias = nx.single_source_shortest_path_length(G, removido)
    faltando = perdidos - set(distancias)
    if faltando:
        raise ValueError(f"metabólitos perdidos sem caminho a partir de {removido}: {sorted(faltando)[:5]}")
    return max(distancias[m] for m in perdidos)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    resultado = spearmanr(x, y)
    return float(resultado.statistic), float(resultado.pvalue)


def rodar(org: str) -> pd.DataFrame:
    """Calcula as avalanches e grava as tabelas. Devolve a tabela das substâncias e controles.

    Passos:
      1. tabelas = grafo.carregar_tabelas(org); remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
      2. B = grafo.grafo_bipartido(tabelas["reacoes"], remover_moeda=remover)
         G = grafo.grafo_simples(grafo.grafo_metabolitos(tabelas["reacoes"], remover_moeda=remover))
      3. Confira se os códigos de config.SUBSTANCIAS_EXP3 estão em B; avise os que faltarem.
      4. hubs, baixo = escolher_controles(G, set(config.SUBSTANCIAS_EXP3), config.N_CONTROLES_EXP3,
                                          np.random.default_rng(config.SEMENTE), candidatos=produzidos)
      5. Para cada metabólito dos três grupos: grupo, metabolito, nome, grau, betweenness,
         avalanche = len(cascata.cascata(B, metabolitos_removidos=[m])), fracao (do que a
         rede intacta produz), profundidade e as posições entre todos os produzidos.
      6. todos = cascata.avalanches_todos(B), junto com o grau e a betweenness de cada metabólito.
      7. Resumo: quantos não derrubam nenhum outro, média e máximo, ajuste de lei de potência
         das avalanches >= 1 (Clauset et al.) e correlação de Spearman da avalanche com o grau
         e com a betweenness (a centralidade prevê o tamanho da cascata?).
      8. Grave em config.arquivo_tabela("exp3", "substancias" | "avalanches" | "resumo" | "ccdf", org).

    O grau e a betweenness são os do grafo metabólito-cêntrico simples (os mesmos do Exp 1);
    metabólitos que só aparecem na rede bipartida (sem aresta no grafo) ficam com 0.
    """
    tabelas = grafo.carregar_tabelas(org)
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))

    fontes = cascata.fontes_externas(B)
    produzidos, _ = cascata.disponiveis(B, fontes)
    substancias = [m for m in config.SUBSTANCIAS_EXP3 if m in B]
    faltando = [f"{m} ({config.SUBSTANCIAS_EXP3[m]})" for m in config.SUBSTANCIAS_EXP3 if m not in B]
    if faltando:
        print(f"aviso: fora da rede de {org}, ficam de fora: {', '.join(faltando)}")

    graus = metricas.graus(G)["total"]
    entre = metricas.betweenness(G)

    # Avalanche de todos os metabólitos produzidos, com a centralidade de cada um.
    avalanches = cascata.avalanches_todos(B, fontes)
    todos = pd.DataFrame({"metabolito": avalanches.index,
                          "nome": [nomes.get(m, "") for m in avalanches.index],
                          "grau": graus.reindex(avalanches.index).fillna(0).astype(int).to_numpy(),
                          "betweenness": entre.reindex(avalanches.index).fillna(0.0).to_numpy(),
                          "avalanche": avalanches.to_numpy()})
    todos["fracao"] = todos["avalanche"] / len(produzidos)
    todos["posicao_avalanche"] = _posicao(todos["avalanche"])
    todos["posicao_grau"] = _posicao(todos["grau"])
    todos = todos.sort_values(["avalanche", "metabolito"], ascending=[False, True]).reset_index(drop=True)
    por_codigo = todos.set_index("metabolito")

    hubs, baixo = escolher_controles(G, set(config.SUBSTANCIAS_EXP3), config.N_CONTROLES_EXP3,
                                     np.random.default_rng(config.SEMENTE), candidatos=produzidos)
    linhas = []
    for grupo, membros in (("substancia", substancias), ("hub", hubs), ("baixo_grau", baixo)):
        for m in membros:
            perdidos = cascata.cascata(B, metabolitos_removidos=[m], fontes=fontes)
            produzido = m in por_codigo.index
            linhas.append({
                "grupo": grupo, "metabolito": m,
                "nome": config.SUBSTANCIAS_EXP3.get(m) or nomes.get(m, ""),
                "grau": int(graus.get(m, 0)), "betweenness": float(entre.get(m, 0.0)),
                "avalanche": len(perdidos), "fracao": len(perdidos) / len(produzidos),
                "profundidade": profundidade(G, m, perdidos),
                "posicao_avalanche": int(por_codigo.loc[m, "posicao_avalanche"]) if produzido else pd.NA,
                "posicao_grau": int(por_codigo.loc[m, "posicao_grau"]) if produzido else pd.NA,
            })
    escolhidos = pd.DataFrame(linhas, columns=COLUNAS_SUBSTANCIAS)

    positivos = todos.loc[todos["avalanche"] > 0, "avalanche"]
    ajuste = metricas.ajuste_lei_potencia(positivos)
    rho_grau, p_grau = _spearman(todos["grau"], todos["avalanche"])
    rho_bet, p_bet = _spearman(todos["betweenness"], todos["avalanche"])
    resumo = pd.DataFrame([{
        "org": org, "versao": config.VERSAO_CASCATA,
        "metabolitos": len(grafo.nos_metabolito(B)), "produzidos": len(produzidos),
        "sem_avalanche": int((todos["avalanche"] == 0).sum()),
        "fracao_sem_avalanche": float((todos["avalanche"] == 0).mean()),
        "avalanche_media": float(todos["avalanche"].mean()),
        "avalanche_max": int(todos["avalanche"].max()), "metabolito_max": todos["metabolito"].iloc[0],
        "spearman_grau": rho_grau, "p_grau": p_grau,
        "spearman_betweenness": rho_bet, "p_betweenness": p_bet,
        "lp_alpha": ajuste["alpha"], "lp_xmin": ajuste["xmin"], "lp_n_cauda": ajuste["n_cauda"],
        "lp_R_vs_lognormal": ajuste["R_vs_lognormal"], "lp_p": ajuste["p"],
    }])

    for conteudo, tabela in (("substancias", escolhidos), ("avalanches", todos[COLUNAS_AVALANCHES]),
                             ("resumo", resumo), ("ccdf", metricas.ccdf_com_ajustes(positivos))):
        destino = config.arquivo_tabela("exp3", conteudo, org)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(destino, index=False)
    return escolhidos


def plotar(org: str) -> None:
    """Lê as tabelas e grava avalanche x grau e a distribuição dos tamanhos.

    Avalanche x grau: os três grupos, com todos os metabólitos em cinza ao
    fundo. Distribuição: CCDF das avalanches >= 1 com os ajustes de lei de
    potência e log-normal, como a distribuição de graus do Exp 0.
    """
    figuras.estilo()
    escolhidos = pd.read_csv(config.arquivo_tabela("exp3", "substancias", org))
    todos = pd.read_csv(config.arquivo_tabela("exp3", "avalanches", org))
    ccdf = pd.read_csv(config.arquivo_tabela("exp3", "ccdf", org))
    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", org)).iloc[0]
    ajuste = {chave: resumo[f"lp_{chave}"] for chave in ("alpha", "xmin", "R_vs_lognormal", "p")}
    figuras.salvar(figuras.avalanche_vs_grau(escolhidos, todos),
                   config.arquivo_figura("exp3", "avalanche_vs_grau", org))
    figuras.salvar(figuras.distribuicao_avalanches(todos["avalanche"], ccdf, ajuste),
                   config.arquivo_figura("exp3", "distribuicao", org))


def _resumo(escolhidos: pd.DataFrame, org: str) -> str:
    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", org)).iloc[0]
    linhas = [f"{org}: {resumo['produzidos']} de {resumo['metabolitos']} metabólitos produzidos; "
              f"{resumo['sem_avalanche']} ({resumo['fracao_sem_avalanche']:.0%}) sem avalanche; "
              f"maior {resumo['avalanche_max']} ({resumo['metabolito_max']})",
              f"  Spearman avalanche x grau = {resumo['spearman_grau']:.2f}, "
              f"x betweenness = {resumo['spearman_betweenness']:.2f}; "
              f"lei de potência x log-normal R = {resumo['lp_R_vs_lognormal']:.2f} (p = {resumo['lp_p']:.3f})",
              f"  {'grupo':11s} {'metabólito':28s} {'grau':>4s} {'aval.':>5s} {'prof.':>5s} {'posição':>7s}"]
    for linha in escolhidos.to_dict("records"):
        linhas.append(f"  {linha['grupo']:11s} {str(linha['nome'])[:28]:28s} {linha['grau']:4d} "
                      f"{linha['avalanche']:5d} {linha['profundidade']:5d} {str(linha['posicao_avalanche']):>7s}")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 3: viés por substância (cascata de falhas).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        print(_resumo(rodar(args.org), args.org))
    plotar(args.org)


if __name__ == "__main__":
    main()
