"""Experimento 0 — Análise descritiva (a base que o roteiro pede).

Caracteriza a rede e confere a lista de metabólitos-moeda pelos nós de maior
grau. Produz a Tabela 1 e a figura de distribuição de graus do artigo.

Roda por padrão nas duas versões (--versao ambas): a sem moeda é a principal;
a com moeda é o teste de sensibilidade da Tabela 1.

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp0_tabela1_<org>_<versao>.csv   (uma linha: métricas + lei de potência)
       resultados/tabelas/exp0_tabela1_<org>.csv            (as duas versões lado a lado)
       resultados/tabelas/exp0_top10_<org>_<versao>.csv
       resultados/tabelas/exp0_graus_<org>_<versao>.csv     (distribuição: tipo, grau, n_nos)
       resultados/tabelas/exp0_lei_potencia_<org>_<versao>.csv (ajuste por tipo de grau)
       resultados/tabelas/exp0_ccdf_<org>_<versao>.csv      (CCDF do grau total e dos ajustes)
       resultados/tabelas/exp0_bowtie_<org>_<versao>.csv    (componente de cada metabólito)
       resultados/figuras/exp0_graus_<org>_<versao>.png

Uso, na raiz:
    python -m experimentos.exp0_descritiva --org eco --versao ambas
    python -m experimentos.exp0_descritiva --so-figuras
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from experimentos.biblioteca import figuras, grafo, metricas

TIPOS_GRAU = ("total", "entrada", "saida")
ORDEM_BOWTIE = ("nucleo", "entrada", "saida", "restante")


def _gravar(tabela: pd.DataFrame, conteudo: str, org: str, versao: str | None = None) -> None:
    destino = config.arquivo_tabela("exp0", conteudo, org, versao)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)


def tabela_bowtie(G, nomes: dict[str, str]) -> pd.DataFrame:
    """Uma linha por metabólito: componente do bow-tie e graus, do núcleo para o restante."""
    graus = metricas.graus(G)
    componente = {no: nome for nome, nos in metricas.componentes_bow_tie(G).items() for no in nos}
    tabela = graus.reset_index()
    tabela.insert(1, "nome", [nomes.get(m, "") for m in tabela["metabolito"]])
    tabela.insert(2, "componente", [componente[m] for m in tabela["metabolito"]])
    tabela["componente"] = pd.Categorical(tabela["componente"], categories=ORDEM_BOWTIE, ordered=True)
    tabela = tabela.sort_values(["componente", "total", "metabolito"], ascending=[True, False, True])
    tabela["componente"] = tabela["componente"].astype(str)
    return tabela.reset_index(drop=True)


def rodar(org: str, versao: str) -> dict:
    """Calcula as métricas descritivas e grava as tabelas. Devolve a linha da Tabela 1.

    Além das métricas de metricas.tabela_descritiva (no grafo simples dirigido),
    a Tabela 1 traz o tamanho da rede de origem (reações e metabólitos que
    aparecem nelas, depois de tirar a moeda, para comparar com o iML1515) e o
    ajuste de lei de potência do grau total (Clauset et al., 2009): alpha,
    xmin, nós na cauda e a comparação com a log-normal (R < 0 favorece a
    log-normal; p < 0,1, diferença significativa).

    Na versão com_moeda, o top-10 serve para conferir se os hubs são mesmo
    metabólitos-moeda (coluna moeda) e ajustar config.METABOLITOS_MOEDA se
    aparecer algum que falta.
    """
    tabelas = grafo.carregar_tabelas(org)
    remover = config.VERSOES_MOEDA[versao]
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)

    graus = metricas.graus(G)
    ajustes = pd.DataFrame([{"tipo": tipo, **metricas.ajuste_lei_potencia(graus[tipo])} for tipo in TIPOS_GRAU])
    total = ajustes.set_index("tipo").loc["total"]

    linha = {"org": org, "versao": versao,
             "reacoes": len(reacoes), "metabolitos_nas_reacoes": len(grafo.nos_metabolito(B)),
             **metricas.tabela_descritiva(G),
             "lp_alpha": total["alpha"], "lp_xmin": total["xmin"], "lp_n_cauda": total["n_cauda"],
             "lp_R_vs_lognormal": total["R_vs_lognormal"], "lp_p": total["p"]}

    _gravar(pd.DataFrame([linha]), "tabela1", org, versao)
    _gravar(metricas.top_por_grau(G, 10, nomes), "top10", org, versao)
    _gravar(metricas.distribuicao_graus(G), "graus", org, versao)
    _gravar(ajustes, "lei_potencia", org, versao)
    _gravar(metricas.ccdf_com_ajustes(graus["total"]), "ccdf", org, versao)
    _gravar(tabela_bowtie(G, nomes), "bowtie", org, versao)
    return linha


def combinar_versoes(org: str) -> pd.DataFrame | None:
    """Junta as Tabelas 1 das versões já gravadas: uma linha por métrica, uma coluna por versão.

    Grava exp0_tabela1_<org>.csv e devolve a tabela; None se faltar alguma versão.
    """
    linhas = {}
    for versao in config.VERSOES_MOEDA:
        caminho = config.arquivo_tabela("exp0", "tabela1", org, versao)
        if not caminho.exists():
            return None
        linhas[versao] = pd.read_csv(caminho).drop(columns=["org", "versao"]).iloc[0]
    combinada = pd.DataFrame(linhas)
    combinada.index.name = "metrica"
    combinada = combinada.reset_index()
    _gravar(combinada, "tabela1", org)
    return combinada


def plotar(org: str, versao: str) -> None:
    """Lê exp0_graus, exp0_ccdf e exp0_lei_potencia e grava a figura de distribuição de graus.

    Dois painéis: (a) P(k) de entrada, saída e total em log-log; (b) CCDF do
    grau total com os ajustes de lei de potência e log-normal.
    """
    figuras.estilo()
    dist = pd.read_csv(config.arquivo_tabela("exp0", "graus", org, versao))
    ccdf = pd.read_csv(config.arquivo_tabela("exp0", "ccdf", org, versao))
    ajustes = pd.read_csv(config.arquivo_tabela("exp0", "lei_potencia", org, versao)).set_index("tipo")
    fig = figuras.distribuicao_graus(dist, ccdf, ajustes.loc["total"].to_dict())
    figuras.salvar(fig, config.arquivo_figura("exp0", "graus", org, versao))


def _resumo(linha: dict) -> str:
    return (f"{linha['org']} {linha['versao']}: {linha['nos']} nós, {linha['arestas']} arestas, "
            f"núcleo {linha['nucleo']}, caminho médio {linha['caminho_medio']:.2f}, "
            f"clustering {linha['clustering_medio']:.3f}, lei de potência x log-normal "
            f"R = {linha['lp_R_vs_lognormal']:.2f} (p = {linha['lp_p']:.3f})")


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
            print(_resumo(rodar(args.org, versao)))
        plotar(args.org, versao)
    if not args.so_figuras and combinar_versoes(args.org) is not None:
        print(f"Tabela 1 com as duas versões: "
              f"{config.caminho_legivel(config.arquivo_tabela('exp0', 'tabela1', args.org))}")


if __name__ == "__main__":
    main()
