"""Experimento 1 — Robustez estrutural.

Pergunta: a rede da E. coli tolera remoções aleatórias e é frágil a remoções
direcionadas aos nós mais conectados, como prevê a teoria de redes livres de escala?

Lê:    dados/<org>/reacoes.csv
Grava: resultados/tabelas/exp1_curvas_<org>_<versao>.csv   (formato longo, uma linha por passo)
       resultados/tabelas/exp1_indices_<org>_<versao>.csv  (por estratégia: índice R e
                                                            fração removida até o maior
                                                            componente cair à metade)
       resultados/figuras/exp1_<metrica>_<org>_<versao>.png
           para metrica em maior_fraco, maior_forte, eficiencia, alcance
           (curva inteira e um detalhe dos primeiros 10% removidos; a eficiência,
           medida só a cada 5%, fica sem o detalhe)
       resultados/figuras/exp1_painel_<org>_<versao>.png  (as quatro métricas num painel 2x2)

Roda por padrão só na versão sem moeda (config.VERSAO_ROBUSTEZ), como os
Exps 2, 3 e 4: a versão com moeda só repetiria a mesma conclusão com o dobro
de figuras. Ela continua disponível com --versao com_moeda ou --versao ambas.

Leva ~2 min por versão na eco (medido pelo testes/verificar_cascata_robustez;
quase tudo nas 100 repetições aleatórias). Use --so-figuras para ajustar os
gráficos sem recalcular.

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
ZOOM = 0.1  # detalhe das figuras individuais: os ataques derrubam a rede nos primeiros 10%


def rodar(org: str, versao: str) -> pd.DataFrame:
    """Roda todas as estratégias de config.ESTRATEGIAS e grava curvas e índices.

    Passos:
      1. reacoes = grafo.carregar_tabelas(org)["reacoes"]
      2. G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))
      3. curvas, indices = robustez.comparar_estrategias(G, nutrientes=config.NUTRIENTES)
         (indices já traz R_fraco, R_forte, metade_fraco e metade_forte, com os _dp)
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
    print(f"{org} {versao}: {G.number_of_nodes()} nós, {len(config.ESTRATEGIAS)} estratégias, "
          f"{config.N_REPETICOES_ALEATORIAS} repetições aleatórias ({time.perf_counter() - inicio:.0f} s)")
    return indices


def plotar(org: str, versao: str) -> None:
    """Lê exp1_curvas_<org>_<versao>.csv e grava uma figura por métrica de METRICAS_FIGURA.

    A figura principal do artigo é a de maior_fraco (curva clássica de
    Albert, Jeong e Barabási, 2000); a eficiência só tem pontos a cada 5%.
    Grava também o painel 2x2 com as quatro métricas.
    """
    figuras.estilo()
    curvas = pd.read_csv(config.arquivo_tabela("exp1", "curvas", org, versao))
    for metrica in METRICAS_FIGURA:
        # A eficiência só tem pontos a cada 5%: no detalhe sobrariam três.
        fig = figuras.curvas_robustez(curvas, metrica, zoom=None if metrica == "eficiencia" else ZOOM)
        figuras.salvar(fig, config.arquivo_figura("exp1", metrica, org, versao))
    figuras.salvar(figuras.painel_robustez(curvas, METRICAS_FIGURA),
                   config.arquivo_figura("exp1", "painel", org, versao))


def _resumo(indices: pd.DataFrame) -> str:
    linhas = [f"  {'estratégia':24s} {'R_fraco':>8s} {'R_forte':>8s} {'metade (fraco)':>15s}"]
    for linha in indices.to_dict("records"):
        dp = f" ± {linha['metade_fraco_dp']:.1%}" if linha["n_rep"] > 1 else ""
        linhas.append(f"  {linha['estrategia']:24s} {linha['R_fraco']:8.4f} {linha['R_forte']:8.4f} "
                      f"{linha['metade_fraco']:15.1%}{dp}")
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
