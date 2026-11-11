"""Experimento 0 — Análise descritiva da rede (Tabela 1 e Figura 2 do artigo).

Caracteriza a rede: tamanho, densidade, graus, agrupamento, componentes,
gravata-borboleta (núcleo, entrada e saída), caminho médio e diâmetro no núcleo
e o teste de lei de potência x log-normal da distribuição de graus (Clauset et
al., 2009).

A Tabela 1 sai nas duas versões, lado a lado: a sem moeda é a rede do artigo; a
com moeda é só a comparação citada no texto (o núcleo cresce de 415 para 497).
As outras tabelas e a figura são só da versão principal (config.VERSAO_PRINCIPAL).

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp0_tabela1_<org>.csv               (Tabela 1: uma linha por métrica, uma
                                                                coluna por versão; métricas + lei de potência)
       resultados/tabelas/exp0_top10_<org>_<versao>.csv         (os 10 metabólitos de maior grau)
       resultados/tabelas/exp0_graus_<org>_<versao>.csv         (distribuição: tipo, grau, n_nos)
       resultados/tabelas/exp0_lei_potencia_<org>_<versao>.csv  (ajuste por tipo de grau)
       resultados/tabelas/exp0_ccdf_<org>_<versao>.csv          (CCDF do grau total e dos ajustes)
       resultados/tabelas/exp0_bowtie_<org>_<versao>.csv        (parte da gravata-borboleta de cada metabólito)
       resultados/figuras/exp0_graus_<org>_<versao>.png         (o painel b é a Figura 2)

Uso, na raiz:
    python -m experimentos.exp0_descritiva
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


def rodar(org: str) -> pd.DataFrame:
    """Calcula a Tabela 1 nas duas versões e as tabelas da versão principal. Devolve a Tabela 1.

    Além das métricas de metricas.tabela_descritiva (no grafo simples dirigido),
    a Tabela 1 traz o tamanho da rede de origem (reações e metabólitos que
    aparecem nelas, depois de tirar a moeda) e o ajuste de lei de potência do
    grau total (Clauset et al., 2009): alpha, xmin, nós na cauda e a comparação
    com a log-normal (R < 0 favorece a log-normal; p < 0,1, diferença significativa).
    """
    tabelas = grafo.carregar_tabelas(org)
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    colunas = {}
    for versao, remover in config.VERSOES_MOEDA.items():
        principal = versao == config.VERSAO_PRINCIPAL
        G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))
        B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
        graus = metricas.graus(G)
        tipos = TIPOS_GRAU if principal else ("total",)
        ajustes = pd.DataFrame([{"tipo": tipo, **metricas.ajuste_lei_potencia(graus[tipo])} for tipo in tipos])
        total = ajustes.set_index("tipo").loc["total"]
        colunas[versao] = {"reacoes": len(reacoes), "metabolitos_nas_reacoes": len(grafo.nos_metabolito(B)),
                           **metricas.tabela_descritiva(G),
                           "lp_alpha": total["alpha"], "lp_xmin": total["xmin"], "lp_n_cauda": total["n_cauda"],
                           "lp_R_vs_lognormal": total["R_vs_lognormal"], "lp_p": total["p"]}
        if principal:
            _gravar(metricas.top_por_grau(G, 10, nomes), "top10", org, versao)
            _gravar(metricas.distribuicao_graus(G), "graus", org, versao)
            _gravar(ajustes, "lei_potencia", org, versao)
            _gravar(metricas.ccdf_com_ajustes(graus["total"]), "ccdf", org, versao)
            _gravar(tabela_bowtie(G, nomes), "bowtie", org, versao)

    tabela1 = pd.DataFrame({versao: pd.Series(linha, dtype=float) for versao, linha in colunas.items()})
    tabela1.index.name = "metrica"
    tabela1 = tabela1.reset_index()
    _gravar(tabela1, "tabela1", org)
    return tabela1


def plotar(org: str, versao: str = config.VERSAO_PRINCIPAL) -> None:
    """Lê exp0_graus, exp0_ccdf e exp0_lei_potencia e grava a figura de distribuição de graus.

    Dois painéis: (a) P(k) de entrada, saída e total em log-log; (b) CCDF do
    grau total com os ajustes de lei de potência e log-normal (a Figura 2).
    """
    figuras.estilo()
    dist = pd.read_csv(config.arquivo_tabela("exp0", "graus", org, versao))
    ccdf = pd.read_csv(config.arquivo_tabela("exp0", "ccdf", org, versao))
    ajustes = pd.read_csv(config.arquivo_tabela("exp0", "lei_potencia", org, versao)).set_index("tipo")
    fig = figuras.distribuicao_graus(dist, ccdf, ajustes.loc["total"].to_dict())
    figuras.salvar(fig, config.arquivo_figura("exp0", "graus", org, versao))


def _resumo(tabela1: pd.DataFrame, org: str) -> str:
    linhas = []
    for versao, coluna in tabela1.set_index("metrica").items():
        linhas.append(f"{org} {versao}: {coluna['nos']:.0f} nós, {coluna['arestas']:.0f} arestas, "
                      f"núcleo {coluna['nucleo']:.0f}, caminho médio {coluna['caminho_medio']:.2f}, "
                      f"clustering {coluna['clustering_medio']:.3f}, lei de potência x log-normal "
                      f"R = {coluna['lp_R_vs_lognormal']:.2f} (p = {coluna['lp_p']:.3f})")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 0: análise descritiva da rede.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só a figura, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        print(_resumo(rodar(args.org), args.org))
        print(f"Tabela 1: {config.caminho_legivel(config.arquivo_tabela('exp0', 'tabela1', args.org))}")
    plotar(args.org)


if __name__ == "__main__":
    main()
