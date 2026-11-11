"""Modelos clássicos de rede (Exp 4, P3): a E. coli se parece com Erdős–Rényi, Watts–Strogatz ou Barabási–Albert?

Os modelos clássicos são não dirigidos, então a comparação usa a versão não
dirigida e simples da rede sem moeda (rede_nao_dirigida): na eco, N = 1.111,
M = 1.277, grau médio 2,30, grau máximo 27 e 79,7% dos nós no maior componente.
Cada modelo tem o mesmo N e o mesmo M da E. coli, ou o mais perto que os
parâmetros inteiros permitem (os nomes são os de config.MODELOS_CLASSICOS):

    configuracao    mesmos graus: nx.double_edge_swap com config.TROCAS_POR_ARESTA
                    trocas por aresta (Maslov e Sneppen, 2002)                    [M exato]
    erdos_renyi     G(N, M): nx.gnm_random_graph (Erdős e Rényi, 1959)            [M exato]
    ba_misto        Barabási–Albert com m misto: nx.dual_barabasi_albert_graph(N, 1, 2, p).
                    Cada nó novo entra com 1 aresta com probabilidade p e com 2 no
                    resto; p = 2 - M/N (~0,85 na eco) dá M em média. O BA puro não
                    casa com a rede: com m = 1 é uma árvore (M = N - 1) e com m = 2
                    o número de arestas dobra.
    ba_m1           Barabási–Albert com m = 1 (Barabási e Albert, 1999), só como
                    referência: uma árvore, com M = N - 1.
    newman_watts    Newman–Watts com k = 2 e p = M/N - 1 (~0,15): um anel em que cada
                    aresta ganha um atalho com probabilidade p (nx.newman_watts_strogatz_graph).
                    Mesma densidade, clustering ~0.
    watts_strogatz  Watts–Strogatz com k = 4 e p = 0,1 (nx.connected_watts_strogatz_graph):
                    clustering alto, mas o dobro de arestas.
Com grau médio 2,3 não dá para casar a densidade e o clustering do Watts–Strogatz
ao mesmo tempo: com k = 2 a rede é um anel sem triângulos, e o clustering alto
exige k >= 4. Por isso as duas versões. Se M/N sair de [1, 2], os p ficam
presos em [0, 1] e o M não casa; a tabela descritiva mostra o M de cada rede.

Robustez (o foco): a métrica é o tamanho do maior componente conexo depois de
remover os nós numa ordem. Como a curva só depende da ordem, ela sai por
percolação reversa com union-find (maior_componente_por_ordem): os nós são
readicionados do último removido para o primeiro, acompanhando o maior
componente, em O(M) por curva (Newman e Ziff, 2000). A ordem das estratégias
estáticas e da aleatória é a de robustez.ordem_inicial; a do grau adaptativo
é calculada passo a passo (robustez.ordem_de_remocao), e só depois percorrida.
Índices (os mesmos do Exp 1): R (Schneider et al., 2011), R normalizado = R / S0
(S0 = maior componente na rede intacta; necessário porque BA e WS nascem 100%
conexos e a E. coli tem só 80% dos nós no maior componente) e a fração removida
até o maior componente cair à metade.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import grafo, metricas, robustez

COLUNAS_DESCRITIVA = ("nos", "arestas", "grau_medio", "grau_max", "clustering_medio", "maior_componente",
                      "caminho_medio", "diametro", "assortatividade")
COLUNAS_INDICES = ("estrategia", "n_rep", "S0", "R", "R_dp", "R_norm", "metade", "metade_dp")

# Limite de tentativas por troca pedida no modelo de configuração (o max_tries de
# nx.double_edge_swap, que para com erro se as tentativas acabarem antes das trocas).
TENTATIVAS_POR_TROCA = 10


# ---------------------------------------------------------------------------
# A rede real e os geradores
# ---------------------------------------------------------------------------
def rede_nao_dirigida(G: nx.DiGraph) -> nx.Graph:
    """A versão não dirigida e simples de G (grafo.grafo_nao_dirigido), com nós e arestas em ordem de código.

    u -> v e v -> u viram uma aresta só. A ordem fixa faz o modelo de configuração
    (que parte desta rede) não depender da ordem de inserção.
    """
    nao_dirigida = grafo.grafo_nao_dirigido(G)
    U = nx.Graph()
    U.add_nodes_from(sorted(nao_dirigida.nodes))
    U.add_edges_from(sorted(tuple(sorted(aresta)) for aresta in nao_dirigida.edges()))
    return U


def parametros(n: int, m: int) -> dict:
    """Os parâmetros que casam N e M: p do BA misto (2 - M/N) e do Newman–Watts (M/N - 1), presos em [0, 1]."""
    razao = m / n if n else 0.0
    return {"p_ba_misto": min(1.0, max(0.0, 2.0 - razao)),
            "p_newman_watts": min(1.0, max(0.0, razao - 1.0))}


def configuracao(U: nx.Graph, semente: int | None = None,
                 trocas_por_aresta: int = config.TROCAS_POR_ARESTA) -> nx.Graph:
    """Mesmos graus de U, arestas embaralhadas por trocas duplas (nx.double_edge_swap).

    Faz trocas_por_aresta * M trocas, com até TENTATIVAS_POR_TROCA tentativas
    por troca (Maslov e Sneppen, 2002). Os nós continuam os de U.
    """
    H = nx.Graph()
    H.add_nodes_from(sorted(U.nodes))
    H.add_edges_from(sorted(U.edges()))
    trocas = trocas_por_aresta * H.number_of_edges()
    if trocas and H.number_of_nodes() >= 4:
        nx.double_edge_swap(H, nswap=trocas, max_tries=TENTATIVAS_POR_TROCA * trocas, seed=semente)
    H.graph["trocas"] = trocas
    return H


def gerar(modelo: str, U: nx.Graph, semente: int) -> nx.Graph:
    """Uma rede do modelo pedido (um de config.MODELOS_CLASSICOS) com o N e o M de U.

    O texto dos parâmetros usados fica em .graph["parametros"].
    """
    n, m = U.number_of_nodes(), U.number_of_edges()
    p = parametros(n, m)
    if modelo == "configuracao":
        R = configuracao(U, semente)
        texto = f"mesmos graus, {R.graph['trocas']} trocas"
    elif modelo == "erdos_renyi":
        R, texto = nx.gnm_random_graph(n, m, seed=semente), f"G(N = {n}, M = {m})"
    elif modelo == "ba_misto":
        R = nx.dual_barabasi_albert_graph(n, 1, 2, p["p_ba_misto"], seed=semente)
        texto = f"m = 1 (prob. {p['p_ba_misto']:.3f}) ou 2"
    elif modelo == "ba_m1":
        R, texto = nx.barabasi_albert_graph(n, 1, seed=semente), "m = 1"
    elif modelo == "newman_watts":
        R = nx.newman_watts_strogatz_graph(n, 2, p["p_newman_watts"], seed=semente)
        texto = f"k = 2, p = {p['p_newman_watts']:.3f}"
    elif modelo == "watts_strogatz":
        R = nx.connected_watts_strogatz_graph(n, config.K_WATTS_STROGATZ, config.P_WATTS_STROGATZ,
                                              tries=100, seed=semente)
        texto = f"k = {config.K_WATTS_STROGATZ}, p = {config.P_WATTS_STROGATZ}"
    else:
        raise ValueError(f"modelo clássico desconhecido: {modelo!r} (use um de {config.MODELOS_CLASSICOS})")
    R.graph["parametros"] = texto
    return R


def gerar_modelos(U: nx.Graph, modelo: str, n_redes: int = config.N_REDES_POR_MODELO,
                  semente: int = config.SEMENTE) -> Iterator[nx.Graph]:
    """n_redes redes do modelo; a rede i usa a semente semente * 1000 + i, para ser reprodutível sozinha."""
    for i in range(n_redes):
        yield gerar(modelo, U, semente * 1000 + i)


# ---------------------------------------------------------------------------
# Descritivas
# ---------------------------------------------------------------------------
def maior_componente(G: nx.Graph) -> set:
    """Os nós do maior componente conexo (empate de tamanho: o que tem o menor nó)."""
    componentes = list(nx.connected_components(G))
    if not componentes:
        return set()
    return min(componentes, key=lambda c: (-len(c), min(c)))


def caminho_medio_e_diametro(G: nx.Graph) -> tuple[float, int]:
    """Caminho médio e diâmetro no maior componente conexo (distâncias da scipy).

    Os modelos têm maiores componentes de tamanhos diferentes, então o caminho é
    sempre medido dentro do maior componente da própria rede. (nan, 0) se ele
    tiver menos de 2 nós.
    """
    nucleo = maior_componente(G)
    if len(nucleo) < 2:
        return math.nan, 0
    D = metricas.matriz_distancias(G.subgraph(nucleo), sorted(nucleo))
    fora_da_diagonal = D[~np.eye(len(nucleo), dtype=bool)]
    return float(fora_da_diagonal.mean()), int(fora_da_diagonal.max())


def descritiva(G: nx.Graph) -> dict:
    """Uma linha da tabela descritiva dos modelos clássicos (chaves de COLUNAS_DESCRITIVA).

    clustering_medio: nx.average_clustering (nós de grau 0 ou 1 contam 0);
    maior_componente: fração dos nós; caminho_medio e diametro: no maior componente;
    assortatividade: de grau (metricas.assortatividade; nan se indefinida).
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    caminho, diametro = caminho_medio_e_diametro(G)
    return {
        "nos": n,
        "arestas": m,
        "grau_medio": 2 * m / n if n else 0.0,
        "grau_max": max((d for _, d in G.degree()), default=0),
        "clustering_medio": nx.average_clustering(G) if n else 0.0,
        "maior_componente": len(maior_componente(G)) / n if n else 0.0,
        "caminho_medio": caminho,
        "diametro": diametro,
        "assortatividade": metricas.assortatividade(G),
    }


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------
def maior_componente_por_ordem(G: nx.Graph, ordem: list) -> np.ndarray:
    """Tamanho do maior componente depois de remover ordem[:k], para k = 0..N, em fração de N.

    Percolação reversa com union-find: os nós são readicionados do último
    removido (ordem[-1]) para o primeiro, unindo cada um aos vizinhos já
    presentes e acompanhando o maior componente. s[N] = 0 (nada sobrou).
    ordem tem que ter cada nó de G uma vez. Dá a mesma curva da simulação que
    remove nó a nó e recalcula os componentes, em O(M).
    """
    n = G.number_of_nodes()
    if len(ordem) != n or set(ordem) != set(G.nodes):
        raise ValueError("a ordem tem que ter cada nó do grafo exatamente uma vez")
    if n == 0:
        return np.zeros(1)
    posicao = {no: i for i, no in enumerate(ordem)}
    vizinhos = [[posicao[w] for w in G[v] if w != v] for v in ordem]
    pai = list(range(n))
    tamanho = [1] * n
    presente = [False] * n

    def raiz(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    s = np.zeros(n + 1)
    maior = 0
    for k in range(n - 1, -1, -1):
        presente[k] = True
        maior = max(maior, 1)
        for w in vizinhos[k]:
            if presente[w]:
                a, b = raiz(k), raiz(w)
                if a != b:
                    if tamanho[a] < tamanho[b]:
                        a, b = b, a
                    pai[b] = a
                    tamanho[a] += tamanho[b]
                    maior = max(maior, tamanho[a])
        s[k] = maior
    return s / n


def indice_R(s: np.ndarray) -> float:
    """Índice R de uma curva de maior_componente_por_ordem (a média sem o passo 0)."""
    return metricas.indice_robustez(s[1:])


def fracao_ate_metade(s: np.ndarray) -> float:
    """Fração removida no primeiro passo em que a curva cai à metade do valor inicial ou menos (nan se s[0] = 0)."""
    if not s[0] > 0:
        return math.nan
    caiu = np.nonzero(s[1:] <= s[0] / 2)[0]
    return float((caiu[0] + 1) / (len(s) - 1)) if len(caiu) else math.nan


def ordem(G: nx.Graph, estrategia: str, rng: np.random.Generator | None = None) -> list:
    """A sequência de remoção (robustez.ordem_de_remocao, sem nós protegidos: aqui não há alcance)."""
    return robustez.ordem_de_remocao(G, estrategia, rng)


def robustez_rede(G: nx.Graph, estrategias: Iterable[str] = config.ESTRATEGIAS_CLASSICOS,
                  n_rep: int = config.N_REPETICOES_MODELOS,
                  semente: int = config.SEMENTE) -> pd.DataFrame:
    """Índices de robustez de uma rede não dirigida, uma linha por estratégia (COLUNAS_INDICES).

    S0 (maior componente na rede intacta, em fração de N), R, R_norm = R/S0 e
    metade; na aleatória, médias das n_rep repetições e _dp o desvio padrão entre
    elas (NaN nas outras). A repetição i da aleatória usa np.random.default_rng([semente, i]).
    """
    indices = []
    for estrategia in estrategias:
        if estrategia == "aleatoria":
            if n_rep < 1:
                raise ValueError("n_rep tem que ser pelo menos 1")
            series = [maior_componente_por_ordem(G, ordem(G, "aleatoria", np.random.default_rng([semente, i])))
                      for i in range(n_rep)]
        else:
            series = [maior_componente_por_ordem(G, ordem(G, estrategia))]
        s0 = float(series[0][0])
        Rs = np.array([indice_R(s) for s in series])
        metades = np.array([fracao_ate_metade(s) for s in series])
        desvio = (lambda v: float(v.std(ddof=1))) if len(series) > 1 else (lambda v: math.nan)
        indices.append({"estrategia": estrategia, "n_rep": len(series), "S0": s0,
                        "R": float(Rs.mean()), "R_dp": desvio(Rs),
                        "R_norm": float(Rs.mean()) / s0 if s0 > 0 else math.nan,
                        "metade": float(metades.mean()), "metade_dp": desvio(metades)})
    return pd.DataFrame(indices, columns=list(COLUNAS_INDICES))


def sigma_mundo_pequeno(C: float, L: float, C_er: float, L_er: float) -> float:
    """Sigma de mundo pequeno (Humphries e Gurney, 2008): (C / C_ER) / (L / L_ER).

    sigma > 1 indica mundo pequeno, mas C_ER é minúsculo nestas redes (~0,002),
    então sigma cresce muito com qualquer clustering: relate C e L também. nan
    se algum termo for zero ou indefinido.
    """
    termos = (C, L, C_er, L_er)
    if any(not np.isfinite(t) for t in termos) or C_er == 0 or L == 0 or L_er == 0:
        return math.nan
    return (C / C_er) / (L / L_er)
