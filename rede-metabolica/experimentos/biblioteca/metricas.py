"""Métricas usadas por mais de um experimento.

Descritivas (exp0, exp4): tamanho, graus, clustering, caminhos, bow-tie.
De dano (exp1, exp4): maior componente, índice R, eficiência, alcance.
Topológicas para o exp2: centralidades e choke points.

Todas recebem o grafo simples dirigido (grafo.grafo_simples), exceto
choke_points, que trabalha direto na tabela de reações.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import grafo


# ---------------------------------------------------------------------------
# Descritivas
# ---------------------------------------------------------------------------
def graus(G: nx.DiGraph) -> pd.DataFrame:
    """Uma linha por nó: grau de entrada, de saída e total (índice = nó)."""
    tabela = pd.DataFrame(
        {"entrada": dict(G.in_degree()), "saida": dict(G.out_degree())},
        columns=["entrada", "saida"],
    ).astype(int)
    tabela["total"] = tabela["entrada"] + tabela["saida"]
    tabela.index.name = "metabolito"
    return tabela.sort_index()


def distribuicao_graus(G: nx.DiGraph) -> pd.DataFrame:
    """Colunas: tipo ("entrada"/"saida"/"total"), grau, n_nos. Base da figura log-log.

    Inclui o grau 0; quem faz a figura em log-log tira esses pontos.
    """
    tabela = graus(G)
    linhas = []
    for tipo in ("entrada", "saida", "total"):
        for grau, n in tabela[tipo].value_counts().sort_index().items():
            linhas.append({"tipo": tipo, "grau": int(grau), "n_nos": int(n)})
    return pd.DataFrame(linhas, columns=["tipo", "grau", "n_nos"])


def top_por_grau(G: nx.DiGraph, k: int = 10, nomes: dict[str, str] | None = None) -> pd.DataFrame:
    """Os k nós de maior grau total, com nome, para conferir os metabólitos-moeda.

    Colunas: metabolito, nome, total, entrada, saida, moeda (está em
    config.METABOLITOS_MOEDA). Empates são desfeitos pelo código.
    """
    tabela = graus(G).reset_index()
    nomes = nomes or {}
    tabela["nome"] = [nomes.get(m, "") for m in tabela["metabolito"]]
    tabela["moeda"] = tabela["metabolito"].isin(list(config.METABOLITOS_MOEDA))
    tabela = tabela.sort_values(["total", "metabolito"], ascending=[False, True]).head(k)
    return tabela[["metabolito", "nome", "total", "entrada", "saida", "moeda"]].reset_index(drop=True)


def maior_componente_forte(G: nx.DiGraph) -> set:
    """Os nós do maior componente fortemente conexo (vazio se G não tem nós).

    Empate de tamanho: fica o que tem o menor nó, para o resultado não
    depender da ordem de inserção.
    """
    componentes = list(nx.strongly_connected_components(G))
    if not componentes:
        return set()
    return min(componentes, key=lambda c: (-len(c), min(map(str, c))))


def caminho_medio_e_diametro(G: nx.DiGraph) -> tuple[float, int]:
    """Calculados no maior componente fortemente conexo.

    Num grafo dirigido, a distância só existe entre pares mutuamente
    alcançáveis; fora do maior componente forte, muitos pares não têm caminho.
    Dentro dele, os caminhos mínimos nunca saem do componente, então basta
    o subgrafo. Devolve (nan, 0) se o componente tem menos de 2 nós.
    """
    nucleo = maior_componente_forte(G)
    if len(nucleo) < 2:
        return math.nan, 0
    H = G.subgraph(nucleo)
    soma = pares = diametro = 0
    for origem, distancias in nx.all_pairs_shortest_path_length(H):
        for destino, d in distancias.items():
            if destino != origem:
                soma += d
                pares += 1
                diametro = max(diametro, d)
    return soma / pares, diametro


def componentes_bow_tie(G: nx.DiGraph) -> dict[str, set]:
    """Os conjuntos da estrutura bow-tie (Ma e Zeng, 2003b).

    nucleo:   maior componente fortemente conexo
    entrada:  alcança o núcleo mas não é alcançada por ele
    saida:    é alcançada pelo núcleo mas não o alcança
    restante: o resto (tubos, gavinhas e componentes desconectados)
    """
    nucleo = maior_componente_forte(G)
    if not nucleo:
        return {"nucleo": set(), "entrada": set(), "saida": set(), "restante": set(G.nodes)}
    representante = next(iter(nucleo))
    saida = nx.descendants(G, representante) - nucleo
    entrada = nx.ancestors(G, representante) - nucleo
    restante = set(G.nodes) - nucleo - saida - entrada
    return {"nucleo": nucleo, "entrada": entrada, "saida": saida, "restante": restante}


def bow_tie(G: nx.DiGraph) -> dict[str, int]:
    """Tamanhos de núcleo (maior componente forte), entrada, saída e restante (Ma e Zeng, 2003b).

    entrada: alcança o núcleo mas não é alcançada por ele
    saida:   é alcançada pelo núcleo mas não o alcança
    """
    return {nome: len(nos) for nome, nos in componentes_bow_tie(G).items()}


def ajuste_lei_potencia(valores_grau: Iterable[int]) -> dict:
    """Opcional: teste de Clauset et al. com o pacote powerlaw.

    Devolve alpha, xmin, quantos nós ficam na cauda (grau >= xmin) e a
    comparação com a log-normal: R > 0 favorece a lei de potência, R < 0 a
    log-normal, e p < 0,1 indica que a diferença é significativa.
    """
    import powerlaw  # só aqui: o pacote é opcional e demora para importar

    dados = np.array([v for v in valores_grau if v > 0])
    if len(dados) < 10:
        return {"alpha": math.nan, "xmin": math.nan, "n_cauda": 0, "R_vs_lognormal": math.nan, "p": math.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = powerlaw.Fit(dados, discrete=True, verbose=False)
        R, p = ajuste.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    xmin = float(ajuste.power_law.xmin)
    return {"alpha": float(ajuste.power_law.alpha), "xmin": xmin,
            "n_cauda": int((dados >= xmin).sum()), "R_vs_lognormal": float(R), "p": float(p)}


def assortatividade(G: nx.DiGraph) -> float:
    """Assortatividade de grau (saída da origem x entrada do destino); nan se indefinida."""
    if G.number_of_edges() == 0:
        return math.nan
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        valor = nx.degree_assortativity_coefficient(G)
    return float(valor) if valor is not None else math.nan


def tabela_descritiva(G: nx.DiGraph) -> dict:
    """Uma linha da Tabela 1 do artigo para o grafo simples dirigido G.

    Chaves: nos, arestas, densidade, grau_medio, clustering_medio (na versão
    não dirigida), caminho_medio, diametro, bow-tie (núcleo, entrada, saída,
    restante) e assortatividade. É reaproveitada pelo exp4 para os outros
    organismos e para os modelos nulos.

    grau_medio é a média do grau de saída (igual à de entrada, = arestas/nós).
    maior_fraco é a fração dos nós no maior componente fraco.
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    caminho, diametro = caminho_medio_e_diametro(G)
    tamanhos = bow_tie(G)
    fracos = list(nx.weakly_connected_components(G)) if n else []
    return {
        "nos": n,
        "arestas": m,
        "densidade": nx.density(G) if n > 1 else 0.0,
        "grau_medio": m / n if n else 0.0,
        "clustering_medio": nx.average_clustering(grafo.grafo_nao_dirigido(G)) if n else 0.0,
        "componentes_fracos": len(fracos),
        "maior_fraco": max(map(len, fracos)) / n if n else 0.0,
        "caminho_medio": caminho,
        "diametro": diametro,
        "nucleo": tamanhos["nucleo"],
        "entrada": tamanhos["entrada"],
        "saida": tamanhos["saida"],
        "restante": tamanhos["restante"],
        "assortatividade": assortatividade(G),
    }


# ---------------------------------------------------------------------------
# Dano (curvas de robustez)
# ---------------------------------------------------------------------------
def fracao_maior_componente(G: nx.DiGraph, n_original: int, forte: bool = False) -> float:
    """Tamanho do maior componente fraco (ou forte) dividido por n_original."""
    if n_original <= 0 or G.number_of_nodes() == 0:
        return 0.0
    componentes = nx.strongly_connected_components(G) if forte else nx.weakly_connected_components(G)
    return max(map(len, componentes)) / n_original


def indice_robustez(fracoes_maior_componente: Iterable[float]) -> float:
    """Índice R de Schneider et al. (2011): média da fração no maior
    componente após cada remoção, de 1 até N nós removidos.

    Passe a curva sem o ponto inicial (0 nós removidos). R vai de 0 a 0,5:
    0,5 é a rede mais robusta possível, perto de 0 a mais frágil.
    """
    valores = np.asarray(list(fracoes_maior_componente), dtype=float)
    return float(valores.mean()) if len(valores) else math.nan


def eficiencia_global_dirigida(G: nx.DiGraph, n_original: int) -> float:
    """Média de 1/d(i, j) sobre os pares ordenados i != j, dividida por n_original*(n_original-1).

    nx.global_efficiency não aceita grafo dirigido, por isso a versão própria.
    Normalizar pelo tamanho original faz a curva cair conforme a rede se parte.
    É cara (distâncias entre todos os pares): o exp1 só a mede a cada
    config.PASSO_EFICIENCIA da rede removida.
    """
    if n_original < 2:
        return 0.0
    soma = 0.0
    for origem in G:
        for destino, d in nx.single_source_shortest_path_length(G, origem).items():
            if d > 0:
                soma += 1.0 / d
    return soma / (n_original * (n_original - 1))


def alcancaveis(G: nx.DiGraph, fontes: Iterable[str]) -> set:
    """Nós alcançáveis a partir das fontes (incluindo as próprias fontes presentes em G)."""
    vistos = {f for f in fontes if f in G}
    fila = deque(vistos)
    while fila:
        u = fila.popleft()
        for v in G.successors(u):
            if v not in vistos:
                vistos.add(v)
                fila.append(v)
    return vistos


def alcance(G: nx.DiGraph, fontes: Iterable[str], n_original: int) -> float:
    """Fração de metabólitos alcançáveis por busca em largura a partir das fontes.

    Versão topológica (cada aresta vale sozinha), que funciona também nos
    modelos nulos. A versão que exige todos os substratos é cascata.disponiveis.
    Fontes ausentes do grafo são ignoradas; as presentes contam como alcançadas.
    """
    if n_original <= 0:
        return 0.0
    return len(alcancaveis(G, fontes)) / n_original


# ---------------------------------------------------------------------------
# Para o exp2
# ---------------------------------------------------------------------------
def betweenness(G: nx.DiGraph, k: int | None = None, semente: int | None = None) -> pd.Series:
    """Betweenness de cada nó; com k, a versão aproximada por amostragem (mais rápida).

    Normalizada, dirigida e sem pesos. Com k >= número de nós, calcula a exata.
    """
    if k is not None and k >= G.number_of_nodes():
        k = None
    valores = nx.betweenness_centrality(G, k=k, seed=semente, normalized=True)
    return pd.Series(valores, name="betweenness", dtype=float).sort_index()


def choke_points(reacoes: pd.DataFrame, remover_moeda: bool = True,
                 conflito_como_irreversivel: bool = False) -> set[str]:
    """Reações que são a única produtora ou a única consumidora de algum metabólito
    (Rahman e Schomburg, 2006).

    Uma reação reversível conta como produtora e consumidora dos dois lados.
    Com remover_moeda=True (o padrão, como na cascata), os metabólitos-moeda
    não contam: ser a única reação que consome um deles não diz nada.
    conflito_como_irreversivel: o teste de sensibilidade de grafo.reversivel.
    """
    produtoras: dict[str, set[str]] = defaultdict(set)
    consumidoras: dict[str, set[str]] = defaultdict(set)
    for r in reacoes.to_dict("records"):
        substratos = grafo.sem_moeda(r["substratos"], remover_moeda)
        produtos = grafo.sem_moeda(r["produtos"], remover_moeda)
        for m in substratos:
            consumidoras[m].add(r["reacao"])
        for m in produtos:
            produtoras[m].add(r["reacao"])
        if grafo.reversivel(r, conflito_como_irreversivel):
            for m in substratos:
                produtoras[m].add(r["reacao"])
            for m in produtos:
                consumidoras[m].add(r["reacao"])

    choke: set[str] = set()
    for conjunto in [*produtoras.values(), *consumidoras.values()]:
        if len(conjunto) == 1:
            choke |= conjunto
    return choke
