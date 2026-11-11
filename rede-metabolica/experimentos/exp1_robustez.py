"""Experimento 1 — Remoção de nós por centralidade (P1: removendo metabólitos por grau,
intermediação e outras centralidades, como a rede se desfaz?). Tabela 2 e Figura 3 do artigo.

Estratégias (config.ESTRATEGIAS; ver robustez.py): a remoção aleatória (média de
config.N_REPETICOES_ALEATORIAS = 100 repetições) e três centralidades, cada uma
nas versões estática (calculada uma vez, na rede intacta) e adaptativa
(recalculada na rede que sobrou):
  grau          o grau total (entrada + saída) no grafo simples: um vizinho
                ligado por uma reação reversível (u -> v e v -> u) conta duas vezes;
                a versão adaptativa recalcula depois de cada remoção
  betweenness   a intermediação exata (metricas.betweenness); a adaptativa recalcula
                a cada config.PASSO_BETWEENNESS = 10 remoções
  pagerank      o PageRank (amortecimento 0,85, peso = número de reações entre o par);
                a adaptativa recalcula a cada 10 remoções
A centralidade de autovetor fica de fora: numa rede dirigida que não é
fortemente conexa ela não é bem definida.

A cada passo, mede o maior componente fraco, o núcleo (maior componente forte)
e o alcance: a fração de metabólitos a que se chega, por busca em largura, a
partir dos nutrientes de config.NUTRIENTES (na versão sem moeda, glicose e
sulfato). Os nutrientes não são removidos antes dos outros nós: vão para o fim
da ordem em todas as estratégias (robustez.simular_remocao). Sem isso, o ataque
por grau removia a glicose (grau 12, entre os 22 mais conectados) e o alcance
caía a zero por isso, e não pelo dano à rede.

Cada curva vira três números (robustez.comparar_estrategias): o índice R de
Schneider et al. (2011), a área sob a curva do maior componente; o R
normalizado, R dividido pelo maior componente na rede intacta (S0); e a fração
removida até o maior componente cair à metade.

Lê:    dados/<org>/reacoes.csv
Grava: resultados/tabelas/exp1_curvas_<org>_<versao>.csv   (formato longo, uma linha por estratégia e passo)
       resultados/tabelas/exp1_indices_<org>_<versao>.csv  (Tabela 2: por estratégia, índice R, R normalizado
                                                            e fração removida até o maior componente cair à
                                                            metade, nos componentes fraco e forte)
       resultados/figuras/exp1_maior_fraco_inteira_<org>_<versao>.png  (Figura 3: o maior componente fraco
                                                            na remoção aleatória e nas versões adaptativas)

Roda na versão sem moeda (config.VERSAO_PRINCIPAL). Leva ~4 min na eco (quase
tudo nas 100 repetições aleatórias). Use --so-figuras para refazer só a figura.

Uso, na raiz:
    python -m experimentos.exp1_robustez
    python -m experimentos.exp1_robustez --so-figuras
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from experimentos.biblioteca import figuras, grafo, robustez

# A figura: a remoção aleatória e uma curva por centralidade, a versão adaptativa
# (a mais danosa). As estáticas ficam na tabela de índices.
ESTRATEGIAS_FIGURA = ("aleatoria", "grau_adaptativo", "betweenness_adaptativo", "pagerank_adaptativo")


def rodar(org: str, versao: str = config.VERSAO_PRINCIPAL) -> pd.DataFrame:
    """Roda todas as estratégias de config.ESTRATEGIAS e grava curvas e índices.

    G = grafo simples metabólito-cêntrico da versão; robustez.comparar_estrategias
    com os nutrientes no fim da ordem. Devolve a tabela de índices.
    """
    reacoes = grafo.carregar_tabelas(org)["reacoes"]
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))
    inicio = time.perf_counter()
    curvas, indices = robustez.comparar_estrategias(G, config.ESTRATEGIAS,
                                                    n_rep=config.N_REPETICOES_ALEATORIAS,
                                                    semente=config.SEMENTE,
                                                    nutrientes=config.NUTRIENTES, progresso=True)
    for conteudo, tabela in (("curvas", curvas), ("indices", indices)):
        destino = config.arquivo_tabela("exp1", conteudo, org, versao)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(destino, index=False)
    protegidos = [n for n in config.NUTRIENTES if n in G]
    print(f"{org} {versao}: {G.number_of_nodes()} nós, {len(config.ESTRATEGIAS)} estratégias, "
          f"{config.N_REPETICOES_ALEATORIAS} repetições aleatórias, {len(protegidos)} nutrientes no fim da ordem "
          f"({', '.join(config.NUTRIENTES[n] for n in protegidos)}) ({time.perf_counter() - inicio:.0f} s)")
    return indices


def plotar(org: str, versao: str = config.VERSAO_PRINCIPAL) -> None:
    """Lê exp1_curvas e grava a Figura 3: o maior componente fraco, a curva inteira,
    na remoção aleatória (com a faixa de ±1 dp) e na versão adaptativa de cada centralidade."""
    figuras.estilo()
    todas = pd.read_csv(config.arquivo_tabela("exp1", "curvas", org, versao))
    curvas = todas[todas["estrategia"].isin(ESTRATEGIAS_FIGURA)]
    figuras.salvar(figuras.curvas_robustez(curvas, "maior_fraco"),
                   config.arquivo_figura("exp1", "maior_fraco_inteira", org, versao))


def _resumo(indices: pd.DataFrame) -> str:
    linhas = [f"  {'estratégia':24s} {'R_fraco':>8s} {'R_forte':>8s} {'R_fraco_norm':>12s} {'R_forte_norm':>12s}"
              f" {'metade (forte)':>15s} {'metade (fraco)':>15s}"]
    for linha in indices.to_dict("records"):
        dp = f" ± {linha['metade_fraco_dp']:.1%}" if linha["n_rep"] > 1 else ""
        linhas.append(f"  {linha['estrategia']:24s} {linha['R_fraco']:8.4f} {linha['R_forte']:8.4f} "
                      f"{linha['R_fraco_norm']:12.4f} {linha['R_forte_norm']:12.4f} "
                      f"{linha['metade_forte']:15.1%} {linha['metade_fraco']:15.1%}{dp}")
    if len(indices):
        primeira = indices.iloc[0]
        linhas.append(f"  (S0: maior componente fraco = {primeira['S0_fraco']:.3f} e forte = "
                      f"{primeira['S0_forte']:.3f} da rede intacta; R normalizado = R / S0)")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: remoção de nós por centralidade.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só a figura, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        print(_resumo(rodar(args.org)))
    plotar(args.org)


if __name__ == "__main__":
    main()
