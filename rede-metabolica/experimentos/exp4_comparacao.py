"""Experimento 4 — Comparação com outras redes.

Pergunta: os padrões da E. coli se repetem em outros organismos e diferem de
uma rede aleatória com a mesma distribuição de graus?

Duas camadas, com as mesmas funções da base (exp0) e do Exp 1:
  - outros organismos (config.ORGANISMOS_COMPARACAO), pelo mesmo pipeline de coleta;
  - modelos nulos da E. coli: rewiring (mesmos graus) e Erdős–Rényi (mesmos n e m).
Nos modelos nulos só rodam métricas e robustez: eles não têm reações.

Roda numa versão só (config.VERSAO_COMPARACAO) e, por padrão, sem medir a
eficiência (config.MEDIR_EFICIENCIA_EXP4), para caber no tempo.

Lê:    dados/<org>/reacoes.csv de cada organismo (rode a coleta de todos antes)
Grava: resultados/tabelas/exp4_descritiva_redes_<versao>.csv  (uma linha por rede)
       resultados/tabelas/exp4_indices_redes_<versao>.csv     (índice R por rede e estratégia)
       resultados/tabelas/exp4_curvas_redes_<versao>.csv      (organismos inteiros; nulos já como média por tipo)
       resultados/figuras/exp4_indices_redes_<versao>.png
       resultados/figuras/exp4_curvas_redes_<versao>.png      (E. coli x modelos nulos)

Uso, na raiz:
    python -m experimentos.exp4_comparacao
    python -m experimentos.exp4_comparacao --orgs eco bsu --sem-nulos
    python -m experimentos.exp4_comparacao --so-figuras
"""
from __future__ import annotations

import argparse

import config
from experimentos.biblioteca import figuras, grafo, metricas, modelos_nulos, robustez  # noqa: F401

TIPOS_NULOS = ("rewiring", "erdos_renyi")


def rodar(orgs: list[str], com_nulos: bool = True) -> None:
    """Calcula métricas e robustez de cada rede e grava as três tabelas.

    Passos:
      1. remover = config.VERSOES_MOEDA[config.VERSAO_COMPARACAO]
      2. Para cada org: G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))
           metricas.tabela_descritiva(G)
           robustez.comparar_estrategias(G, config.ESTRATEGIAS_COMPARACAO,
                                         nutrientes=config.NUTRIENTES,
                                         medir_eficiencia=config.MEDIR_EFICIENCIA_EXP4)
      3. Se com_nulos, a partir do G da E. coli, para cada tipo em TIPOS_NULOS:
           para cada rede de modelos_nulos.gerar_nulos(G_eco, tipo):
             a mesma descritiva e robustez, com n_rep=config.N_REPETICOES_NULOS
             (nutrientes só no rewiring, que mantém os nomes dos nós).
         Guarde as curvas dos nulos como média por tipo, para a tabela não passar de centenas de MB.
      4. Coluna "rede" em todas as tabelas: "eco", "bsu", "mge", "rewiring", "erdos_renyi";
         coluna "indice" para as redes nulas (0 a N_MODELOS_NULOS - 1).
      5. Grave com config.arquivo_tabela("exp4", <conteudo>, "redes", config.VERSAO_COMPARACAO).
    """
    raise NotImplementedError


def plotar() -> None:
    """Lê as tabelas e grava o índice R por rede e as curvas da E. coli x modelos nulos."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 4: comparação com outros organismos e modelos nulos.")
    parser.add_argument("--orgs", nargs="+",
                        default=[config.ORGANISMO_PRINCIPAL, *config.ORGANISMOS_COMPARACAO],
                        help="códigos KEGG; o primeiro é a referência dos modelos nulos")
    parser.add_argument("--sem-nulos", action="store_true", help="pula os modelos nulos (teste rápido)")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        rodar(args.orgs, com_nulos=not args.sem_nulos)
    plotar()


if __name__ == "__main__":
    main()
