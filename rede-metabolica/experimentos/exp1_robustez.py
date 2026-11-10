"""Experimento 1 — Robustez estrutural (P1: removendo nós por grau, betweenness e outras centralidades, como a rede se desfaz?).

Pergunta: a rede da E. coli tolera remoções aleatórias e é frágil a remoções
direcionadas aos nós mais centrais, como prevê a teoria de redes livres de escala?

Estratégias (config.ESTRATEGIAS; ver robustez.py): a remoção aleatória (média de
config.N_REPETICOES_ALEATORIAS = 100 repetições) e quatro centralidades, cada uma
nas versões estática (calculada uma vez, na rede intacta) e adaptativa
(recalculada na rede que sobrou):
  grau          o grau total (entrada + saída) no grafo simples: um vizinho
                ligado por uma reação reversível (u -> v e v -> u) conta duas vezes;
                a versão adaptativa recalcula depois de cada remoção
  betweenness   a exata (metricas.betweenness); a adaptativa recalcula a cada
                config.PASSO_BETWEENNESS = 10 remoções
  harmonica     a centralidade harmônica, a versão da proximidade (closeness) que
                funciona em rede desconexa; a adaptativa recalcula a cada 10 remoções
  pagerank      o PageRank (amortecimento 0,85, peso = número de reações entre o par);
                a adaptativa recalcula a cada 10 remoções
A centralidade de autovetor fica de fora: numa rede dirigida que não é
fortemente conexa ela não é bem definida.

Os nutrientes de config.NUTRIENTES (as fontes do alcance) não são removidos
antes dos outros nós: vão para o fim da ordem em todas as estratégias
(robustez.simular_remocao, proteger_fontes=True). Sem isso, o ataque por grau
removia a glicose (grau 12, entre os 22 mais conectados) e o alcance caía a zero
por isso, e não pelo dano à rede. Na eco sem moeda são 2 dos 1.111 nós (glicose
e sulfato; NH3 e ortofosfato são moeda e já não estão no grafo), então os
índices mudam só na margem.

Cada curva vira três números (robustez.comparar_estrategias): o índice R de
Schneider et al. (2011), a área sob a curva do maior componente; o R
normalizado, R dividido pelo maior componente na rede intacta (S0), como no
Exp 4; e a fração removida até o maior componente cair à metade.

Lê:    dados/<org>/reacoes.csv
Grava: resultados/tabelas/exp1_curvas_<org>_<versao>.csv   (formato longo, uma linha por passo)
       resultados/tabelas/exp1_indices_<org>_<versao>.csv  (por estratégia: índice R, R normalizado e
                                                            fração removida até o maior componente
                                                            cair à metade, nos componentes fraco e forte)
       resultados/figuras/exp1_<metrica>_<org>_<versao>.png
           para metrica em maior_fraco, maior_forte, eficiencia, alcance: a remoção
           aleatória e a versão adaptativa de cada centralidade (curva inteira e um
           detalhe dos primeiros 10% removidos; a eficiência, medida só a cada 5%,
           fica sem o detalhe)
       resultados/figuras/exp1_painel_<org>_<versao>.png  (as quatro métricas num painel 2x2, com as
                                                           mesmas curvas)
       resultados/figuras/exp1_estatica_adaptativa_<org>_<versao>.png  (para cada centralidade, a
                                                           versão estática x a adaptativa)
Todas as estratégias aparecem na tabela de índices.

Roda por padrão só na versão sem moeda (config.VERSAO_ROBUSTEZ), como os
Exps 2, 3 e 4: a versão com moeda só repetiria a mesma conclusão com o dobro
de figuras. Ela continua disponível com --versao com_moeda ou --versao ambas.

Leva ~4 min por versão na eco (quase tudo nas 100 repetições aleatórias). Use
--so-figuras para ajustar os gráficos sem recalcular.

Uso, na raiz:
    python -m experimentos.exp1_robustez --org eco
    python -m experimentos.exp1_robustez --versao ambas
    python -m experimentos.exp1_robustez --so-figuras
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from experimentos.biblioteca import figuras, grafo, robustez

METRICAS_FIGURA = ("maior_fraco", "maior_forte", "eficiencia", "alcance")
# O painel e as figuras de cada métrica: a remoção aleatória e uma curva por
# centralidade, a versão adaptativa (a mais danosa). As estáticas ficam na
# tabela e na figura estática x adaptativa.
ESTRATEGIAS_PAINEL = ("aleatoria", "grau_adaptativo", "betweenness_adaptativo",
                      "harmonica_adaptativo", "pagerank_adaptativo")
CENTRALIDADES = ("grau", "betweenness", "harmonica", "pagerank")
ZOOM = 0.1  # detalhe das figuras individuais: os ataques derrubam a rede nos primeiros 10%


def rodar(org: str, versao: str) -> pd.DataFrame:
    """Roda todas as estratégias de config.ESTRATEGIAS e grava curvas e índices.

    Passos:
      1. reacoes = grafo.carregar_tabelas(org)["reacoes"]
      2. G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))
      3. curvas, indices = robustez.comparar_estrategias(G, nutrientes=config.NUTRIENTES)
         (os nutrientes vão para o fim da ordem; indices já traz R_fraco, R_forte, os
         normalizados, metade_fraco e metade_forte, com os _dp)
      4. Grave em config.arquivo_tabela("exp1", "curvas", org, versao) e ("exp1", "indices", ...).
    Devolve a tabela de índices.
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


def plotar(org: str, versao: str) -> None:
    """Lê exp1_curvas_<org>_<versao>.csv e grava as figuras.

    Uma por métrica de METRICAS_FIGURA e o painel 2x2 com as quatro, todas com
    as estratégias de ESTRATEGIAS_PAINEL (a remoção aleatória e a versão
    adaptativa de cada centralidade); e a figura estática x adaptativa de cada
    centralidade (maior componente fraco). A figura principal do artigo é a de
    maior_fraco (curva clássica de Albert, Jeong e Barabási, 2000); a eficiência
    só tem pontos a cada 5%.
    """
    figuras.estilo()
    todas = pd.read_csv(config.arquivo_tabela("exp1", "curvas", org, versao))
    curvas = todas[todas["estrategia"].isin(ESTRATEGIAS_PAINEL)]
    for metrica in METRICAS_FIGURA:
        # A eficiência só tem pontos a cada 5%: no detalhe sobrariam três.
        fig = figuras.curvas_robustez(curvas, metrica, zoom=None if metrica == "eficiencia" else ZOOM)
        figuras.salvar(fig, config.arquivo_figura("exp1", metrica, org, versao))
    figuras.salvar(figuras.painel_robustez(curvas, METRICAS_FIGURA),
                   config.arquivo_figura("exp1", "painel", org, versao))
    centralidades = [c for c in CENTRALIDADES if c in set(todas["estrategia"])]
    if centralidades:
        figuras.salvar(figuras.estatica_vs_adaptativa(todas, centralidades, "maior_fraco"),
                       config.arquivo_figura("exp1", "estatica_adaptativa", org, versao))


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
    parser = argparse.ArgumentParser(description="Exp 1: robustez estrutural (remoção de nós).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--versao", choices=[*config.VERSOES_MOEDA, "ambas"],
                        default=config.VERSAO_ROBUSTEZ, help="com ou sem metabólitos-moeda")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    versoes = list(config.VERSOES_MOEDA) if args.versao == "ambas" else [args.versao]
    for versao in versoes:
        if not args.so_figuras:
            print(_resumo(rodar(args.org, versao)))
        plotar(args.org, versao)


if __name__ == "__main__":
    main()
