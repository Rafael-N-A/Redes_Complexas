"""Experimento 4 — Comparação com os modelos clássicos (P3: a rede se parece com Erdős–Rényi,
Watts–Strogatz ou Barabási–Albert?). Tabela 4 do artigo.

Os modelos clássicos são não dirigidos, então a comparação usa a versão NÃO
dirigida da rede sem moeda (N = 1.111, M = 1.277). Para cada modelo de
config.MODELOS_CLASSICOS, config.N_REDES_POR_MODELO redes com o N e o M da
E. coli, ou o mais perto que os parâmetros inteiros permitem (ver
modelos_classicos.py): configuração (mesmos graus), Erdős–Rényi G(N, M),
Barabási–Albert misto (m = 1 ou 2) e com m = 1 (uma árvore), Newman–Watts
(k = 2) e Watts–Strogatz (k = 4).

Em cada rede: agrupamento, caminho médio (no maior componente), grau máximo e
robustez pelo maior componente conexo, por percolação reversa (union-find), na
remoção aleatória e no ataque por grau (config.ESTRATEGIAS_CLASSICOS). Como as
redes começam com maiores componentes de tamanhos diferentes, a robustez é o R
normalizado: o índice R de Schneider et al. (2011) dividido pelo maior
componente na rede intacta. Cada métrica da E. coli vira um escore z contra cada
modelo, z = (E. coli − média do modelo) / desvio padrão do modelo: |z| > 2
indica uma diferença que o modelo não explica. A tabela descritiva traz também
o sigma de mundo pequeno (C/C_ER)/(L/L_ER), com C_ER e L_ER as médias das redes
Erdős–Rényi.

Lê:    dados/<org>/reacoes.csv
Grava em resultados/tabelas/ (<v> = config.VERSAO_PRINCIPAL):
       exp4_classicos_descritiva_redes_<v>.csv  uma linha por rede (a E. coli e cada rede sorteada):
                                      M, grau médio e máximo, agrupamento, maior componente,
                                      caminho médio e diâmetro (no maior componente),
                                      assortatividade, sigma
       exp4_classicos_indices_redes_<v>.csv     uma linha por rede e estratégia: S0, R, R normalizado, metade
       exp4_classicos_z_redes_<v>.csv           E. coli x cada modelo: média, dp e escore z de cada métrica
A Tabela 4 do artigo é a média das redes de cada modelo (as duas primeiras tabelas).

Uso, na raiz:
    python -m experimentos.exp4_comparacao
"""
from __future__ import annotations

import argparse
import math
import time

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import grafo, modelos_classicos

# O que entra no confronto (escores z).
DESCRITIVAS_CLASSICOS = ("arestas", "grau_medio", "grau_max", "clustering_medio", "maior_componente",
                         "caminho_medio", "diametro", "assortatividade", "sigma")
INDICES_CLASSICOS = ("R_norm", "R", "metade")


def grafo_do_organismo(org: str, versao: str = config.VERSAO_PRINCIPAL) -> nx.DiGraph:
    """O grafo simples metabólito-cêntrico de org, na versão pedida (o mesmo do Exp 1)."""
    reacoes = grafo.carregar_tabelas(org)["reacoes"]
    return grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))


def escore_z(real: float, valores_modelo) -> tuple[float, float, float]:
    """(média, desvio padrão amostral, z) dos valores das redes de um modelo.

    z é NaN se o dp for indefinido ou zero, o que inclui o resto de
    arredondamento (~1e-16) do dp de valores iguais: é o caso das métricas que o
    modelo preserva por construção (as arestas no Erdős–Rényi e na configuração,
    o grau máximo na configuração).
    """
    valores = np.asarray(pd.Series(valores_modelo, dtype=float).dropna(), dtype=float)
    if len(valores) == 0:
        return math.nan, math.nan, math.nan
    media = float(valores.mean())
    dp = float(valores.std(ddof=1)) if len(valores) > 1 else math.nan
    if dp <= 1e-12 * max(1.0, abs(media)):
        dp = 0.0 if not math.isnan(dp) else dp
        return media, dp, math.nan
    return media, dp, (real - media) / dp if not math.isnan(real) else math.nan


def _em_ordem(valores, referencia) -> list:
    vistos = list(dict.fromkeys(valores))
    return [v for v in referencia if v in vistos] + [v for v in vistos if v not in referencia]


def acrescentar_sigma(descritiva: pd.DataFrame) -> pd.DataFrame:
    """Coluna sigma: (C/C_ER)/(L/L_ER) de cada rede, com C_ER e L_ER as médias das redes erdos_renyi.

    NaN em tudo se não houver redes Erdős–Rényi (modelos_classicos.sigma_mundo_pequeno).
    """
    descritiva = descritiva.copy()
    er = descritiva[descritiva["modelo"] == "erdos_renyi"]
    c_er = float(er["clustering_medio"].mean()) if len(er) else math.nan
    l_er = float(er["caminho_medio"].mean()) if len(er) else math.nan
    descritiva["sigma"] = [modelos_classicos.sigma_mundo_pequeno(c, l, c_er, l_er)
                           for c, l in zip(descritiva["clustering_medio"], descritiva["caminho_medio"])]
    return descritiva


def confronto_classicos(descritiva: pd.DataFrame, indices: pd.DataFrame, real: str) -> pd.DataFrame:
    """Uma linha por métrica (e estratégia, nos índices), com o valor da rede real e, para cada
    modelo, <modelo>_media, <modelo>_dp e <modelo>_z (escore_z)."""
    modelos = [m for m in config.MODELOS_CLASSICOS if m in set(descritiva["modelo"])]
    linhas = []

    def linha_de(metrica: str, estrategia: str, tabela: pd.DataFrame) -> dict:
        valor = float(tabela.loc[tabela["modelo"] == real, metrica].iloc[0])
        linha = {"rede_real": real, "metrica": metrica, "estrategia": estrategia, "real": valor}
        for modelo in modelos:
            amostra = tabela.loc[tabela["modelo"] == modelo, metrica]
            linha[f"{modelo}_media"], linha[f"{modelo}_dp"], linha[f"{modelo}_z"] = escore_z(valor, amostra)
        return linha

    for metrica in DESCRITIVAS_CLASSICOS:
        linhas.append(linha_de(metrica, "", descritiva))
    for estrategia in _em_ordem(indices["estrategia"], config.ESTRATEGIAS_CLASSICOS):
        da_estrategia = indices[indices["estrategia"] == estrategia]
        for metrica in INDICES_CLASSICOS:
            linhas.append(linha_de(metrica, estrategia, da_estrategia))
    return pd.DataFrame(linhas)


def _gravar(tabela: pd.DataFrame, conteudo: str) -> None:
    destino = config.arquivo_tabela("exp4", conteudo, "redes", config.VERSAO_PRINCIPAL)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)


def _ler(conteudo: str) -> pd.DataFrame:
    return pd.read_csv(config.arquivo_tabela("exp4", conteudo, "redes", config.VERSAO_PRINCIPAL))


def rodar(org: str = config.ORGANISMO_PRINCIPAL) -> pd.DataFrame:
    """A E. coli (não dirigida) contra os modelos clássicos. Grava as tabelas exp4_classicos_*.

    Passos:
      1. U = modelos_classicos.rede_nao_dirigida(grafo_do_organismo(org)).
      2. Na rede real: modelos_classicos.descritiva e robustez_rede com
         config.N_REPETICOES_ALEATORIAS repetições da aleatória (as mesmas ordens do
         Exp 1: o maior componente não dirigido é o maior componente fraco, então o R da
         aleatória é o R_fraco do Exp 1 sem proteger os nutrientes).
      3. Para cada modelo de config.MODELOS_CLASSICOS, config.N_REDES_POR_MODELO redes
         (modelos_classicos.gerar_modelos, semente config.SEMENTE * 1000 + i), cada uma com
         config.N_REPETICOES_MODELOS repetições da aleatória e a semente da própria rede.
      4. sigma de mundo pequeno (acrescentar_sigma) e o confronto (escores z).
    Devolve a tabela descritiva.
    """
    from tqdm import tqdm  # import tardio: só a barra de progresso precisa dele

    inicio = time.perf_counter()
    U = modelos_classicos.rede_nao_dirigida(grafo_do_organismo(org))
    descritivas, indices = [], []

    def medir(rede: nx.Graph, modelo: str, indice, n_rep: int, semente: int) -> None:
        descritivas.append({"modelo": modelo, "indice": indice,
                            "parametros": rede.graph.get("parametros", "rede real"),
                            **modelos_classicos.descritiva(rede)})
        ind = modelos_classicos.robustez_rede(rede, config.ESTRATEGIAS_CLASSICOS, n_rep, semente)
        indices.append(ind.assign(modelo=modelo, indice=indice))

    medir(U, org, pd.NA, config.N_REPETICOES_ALEATORIAS, config.SEMENTE)
    for modelo in config.MODELOS_CLASSICOS:
        redes = list(modelos_classicos.gerar_modelos(U, modelo, config.N_REDES_POR_MODELO, config.SEMENTE))
        for i, rede in enumerate(tqdm(redes, desc=modelo, unit="rede")):
            medir(rede, modelo, i, config.N_REPETICOES_MODELOS, config.SEMENTE * 1000 + i)

    descritiva = acrescentar_sigma(pd.DataFrame(descritivas))
    tabela_indices = pd.concat(indices, ignore_index=True)
    tabela_indices = tabela_indices[["modelo", "indice", *modelos_classicos.COLUNAS_INDICES]]
    _gravar(descritiva, "classicos_descritiva")
    _gravar(tabela_indices, "classicos_indices")
    _gravar(confronto_classicos(descritiva, tabela_indices, org), "classicos_z")
    print(f"Exp 4, modelos clássicos: {len(descritiva)} redes em {time.perf_counter() - inicio:.0f} s")
    return descritiva


def _resumo() -> str:
    """A Tabela 4 (médias das redes de cada modelo) e os escores z da rede real."""
    descritiva, indices, confronto = (_ler(f"classicos_{c}") for c in ("descritiva", "indices", "z"))
    media = descritiva.groupby("modelo", sort=False)[
        ["arestas", "grau_max", "clustering_medio", "maior_componente", "caminho_medio", "assortatividade",
         "sigma"]].mean()
    por = indices.groupby(["modelo", "estrategia"], sort=False)[["R", "R_norm", "metade"]].mean()
    linhas = ["modelos clássicos (não dirigidos; média das redes de cada modelo; R sem normalizar e normalizado)",
              f"  {'rede':15s} {'M':>6s} {'kmax':>5s} {'C':>6s} {'%comp':>6s} {'L':>6s} {'assort':>7s} {'sigma':>6s}"
              f" {'R aleat':>8s} {'Rn aleat':>8s} {'metade':>7s} {'R grau':>7s} {'Rn grau':>7s} {'metade':>7s}"]
    for modelo, d in media.iterrows():
        def valor(estrategia, coluna):
            return por.loc[(modelo, estrategia), coluna] if (modelo, estrategia) in por.index else math.nan
        linhas.append(f"  {str(modelo):15s} {d['arestas']:6.0f} {d['grau_max']:5.1f} {d['clustering_medio']:6.3f}"
                      f" {100 * d['maior_componente']:6.1f} {d['caminho_medio']:6.2f} {d['assortatividade']:7.3f}"
                      f" {d['sigma']:6.1f} {valor('aleatoria', 'R'):8.3f} {valor('aleatoria', 'R_norm'):8.3f}"
                      f" {valor('aleatoria', 'metade'):7.3f} {valor('grau', 'R'):7.3f} {valor('grau', 'R_norm'):7.3f}"
                      f" {valor('grau', 'metade'):7.3f}")
    modelos = [m for m in config.MODELOS_CLASSICOS if f"{m}_z" in confronto]
    linhas.append(f"  escores z de {confronto['rede_real'].iloc[0]} (R normalizado e descritivas)")
    linhas.append(f"    {'métrica':28s} {'real':>8s} " + " ".join(f"{m[:14]:>14s}" for m in modelos))
    escolhidas = confronto[(confronto["metrica"] == "R_norm") | confronto["estrategia"].fillna("").eq("")]
    for linha in escolhidas.fillna({"estrategia": ""}).to_dict("records"):
        nome = linha["metrica"] + (f" ({linha['estrategia']})" if linha["estrategia"] else "")
        linhas.append(f"    {nome:28s} {linha['real']:8.3f} "
                      + " ".join(f"{linha[f'{m}_z']:14.1f}" for m in modelos))
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 4: comparação com os modelos clássicos de rede.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    args = parser.parse_args()
    rodar(args.org)
    print(_resumo())


if __name__ == "__main__":
    main()
