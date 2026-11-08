"""Estratégias de remoção de nós e curvas de robustez (Albert, Jeong e Barabási, 2000).

Estratégias (nomes usados em config.ESTRATEGIAS):
    aleatoria               ordem sorteada; média de várias repetições
    grau                    maior grau primeiro, calculado uma vez
    grau_adaptativo         recalcula o grau depois de cada remoção
    betweenness             maior betweenness primeiro, calculada uma vez
    betweenness_adaptativo  recalcula a cada config.PASSO_BETWEENNESS remoções

O grau é o total (entrada + saída) no grafo simples, o mesmo de
metricas.graus. A betweenness é a exata de metricas.betweenness.

As métricas de dano vêm de metricas.py; este módulo só decide a ordem e
mede ao longo do caminho.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import metricas

ESTATICAS = ("aleatoria", "grau", "betweenness")
ADAPTATIVAS = ("grau_adaptativo", "betweenness_adaptativo")
METRICAS = ("maior_fraco", "maior_forte", "alcance", "eficiencia")
COLUNAS = ("n_removidos", "fracao_removida", *METRICAS)
COLUNAS_DP = tuple(f"{m}_dp" for m in METRICAS)
COLUNAS_INDICES = ("estrategia", "R_fraco", "R_forte", "R_fraco_dp", "R_forte_dp", "n_rep")

# A betweenness é arredondada antes de ordenar: dois nós com o mesmo valor
# exato podem diferir na 17ª casa pela ordem da soma, e o desempate pelo
# código só funciona se o empate aparecer.
_CASAS_BETWEENNESS = 12


def _por_valor(valores: dict) -> list:
    """Nós do maior para o menor valor; empates pelo código do nó."""
    return sorted(valores, key=lambda no: (-valores[no], no))


def _betweenness(G: nx.DiGraph) -> dict:
    exata = metricas.betweenness(G)
    return {no: round(float(v), _CASAS_BETWEENNESS) for no, v in exata.items()}


def _ranking_atual(G: nx.DiGraph, estrategia: str) -> list:
    if estrategia in ("grau", "grau_adaptativo"):
        return _por_valor(dict(G.degree()))
    if estrategia in ("betweenness", "betweenness_adaptativo"):
        return _por_valor(_betweenness(G))
    raise ValueError(f"estratégia desconhecida: {estrategia!r}")


def ordem_inicial(G: nx.DiGraph, estrategia: str,
                  rng: np.random.Generator | None = None) -> list[str]:
    """Ordem de remoção das estratégias estáticas (aleatoria, grau, betweenness).

    Empates no grau ou na betweenness são desfeitos pelo código do nó, para
    que a ordem seja sempre a mesma. Na aleatória, o sorteio parte dos nós em
    ordem de código, então a mesma semente dá a mesma ordem em qualquer máquina.
    """
    if estrategia == "aleatoria":
        if rng is None:
            raise ValueError("a estratégia aleatória precisa de um rng")
        nos = sorted(G.nodes)
        return [nos[i] for i in rng.permutation(len(nos))]
    if estrategia in ("grau", "betweenness"):
        return _ranking_atual(G, estrategia)
    raise ValueError(f"estratégia estática desconhecida: {estrategia!r} (use uma de {ESTATICAS})")


def proximo_adaptativo(G: nx.DiGraph, estrategia: str) -> str:
    """Próximo nó a remover nas estratégias adaptativas, no estado atual de G.

    grau_adaptativo: maior grau total agora. betweenness_adaptativo: maior
    betweenness agora (quem chama decide recalcular só a cada passo_betweenness).
    Empates pelo código do nó.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("o grafo está vazio")
    if estrategia == "grau_adaptativo":
        return min(G.nodes, key=lambda no: (-G.degree(no), no))
    if estrategia == "betweenness_adaptativo":
        return _ranking_atual(G, estrategia)[0]
    raise ValueError(f"estratégia adaptativa desconhecida: {estrategia!r} (use uma de {ADAPTATIVAS})")


def marcos_eficiencia(n: int, passo: float) -> set[int]:
    """Números de nós removidos em que a fração removida cruza um múltiplo de passo.

    O múltiplo m*passo é cruzado no primeiro k com k/n >= m*passo, isto é,
    k = teto(m*passo*n). Inclui sempre o 0 (a rede intacta).
    """
    if not 0 < passo <= 1:
        raise ValueError("passo_eficiencia tem que estar em (0, 1]")
    n_multiplos = math.floor(1 / passo + 1e-9)
    return {min(n, math.ceil(round(m * passo * n, 9))) for m in range(n_multiplos + 1)}


def _medir(H: nx.DiGraph, k: int, n: int, nutrientes: list, medir_eficiencia: bool) -> dict:
    return {
        "n_removidos": k,
        "fracao_removida": k / n,
        "maior_fraco": metricas.fracao_maior_componente(H, n),
        "maior_forte": metricas.fracao_maior_componente(H, n, forte=True),
        "alcance": metricas.alcance(H, nutrientes, n) if nutrientes else math.nan,
        "eficiencia": metricas.eficiencia_global_dirigida(H, n) if medir_eficiencia else math.nan,
    }


def simular_remocao(G: nx.DiGraph, estrategia: str,
                    rng: np.random.Generator | None = None,
                    nutrientes: Iterable[str] = (),
                    passo_eficiencia: float = config.PASSO_EFICIENCIA,
                    passo_betweenness: int = config.PASSO_BETWEENNESS,
                    medir_eficiencia: bool = True) -> pd.DataFrame:
    """Remove os nós de G um a um, na ordem da estratégia, até não sobrar nenhum.

    Uma linha por passo (de 0 a N nós removidos), colunas:
        n_removidos, fracao_removida,
        maior_fraco, maior_forte  (frações do tamanho original, a cada passo)
        alcance                   (a partir dos nutrientes, a cada passo; NaN
                                   em todos se nenhum nutriente for passado)
        eficiencia                (só nos passos em que a fração removida cruza
                                   um múltiplo de passo_eficiencia; NaN nos demais
                                   e em todos os passos se medir_eficiencia=False)
    Trabalha numa cópia de G. rng é obrigatório na estratégia aleatória.
    Na betweenness_adaptativo, remove em sequência os passo_betweenness
    primeiros do ranking e só então recalcula.
    """
    if estrategia not in ESTATICAS + ADAPTATIVAS:
        raise ValueError(f"estratégia desconhecida: {estrategia!r}")
    if passo_betweenness < 1:
        raise ValueError("passo_betweenness tem que ser pelo menos 1")
    H = G.copy()
    n = H.number_of_nodes()
    if n == 0:
        raise ValueError("o grafo está vazio")
    nutrientes = list(nutrientes)
    marcos = marcos_eficiencia(n, passo_eficiencia) if medir_eficiencia else set()

    fila = deque(ordem_inicial(H, estrategia, rng)) if estrategia in ESTATICAS else deque()
    linhas = [_medir(H, 0, n, nutrientes, 0 in marcos)]
    for k in range(1, n + 1):
        if estrategia == "grau_adaptativo":
            no = proximo_adaptativo(H, estrategia)
        elif estrategia == "betweenness_adaptativo":
            if not fila:
                fila.extend(_ranking_atual(H, estrategia)[:passo_betweenness])
            no = fila.popleft()
        else:
            no = fila.popleft()
        H.remove_node(no)
        linhas.append(_medir(H, k, n, nutrientes, k in marcos))
    return pd.DataFrame(linhas, columns=list(COLUNAS))


def _repeticoes_aleatorias(G: nx.DiGraph, n_rep: int, semente: int, nutrientes: Iterable[str],
                           medir_eficiencia: bool, progresso: bool = False) -> list[pd.DataFrame]:
    if n_rep < 1:
        raise ValueError("n_rep tem que ser pelo menos 1")
    nutrientes = list(nutrientes)
    repeticoes = range(n_rep)
    if progresso:
        from tqdm import tqdm  # import tardio: só quem pede a barra precisa do tqdm
        repeticoes = tqdm(repeticoes, desc="aleatoria", unit="rep")
    return [simular_remocao(G, "aleatoria", rng=np.random.default_rng([semente, i]),
                            nutrientes=nutrientes, medir_eficiencia=medir_eficiencia)
            for i in repeticoes]


def _media_das_repeticoes(repeticoes: list[pd.DataFrame]) -> pd.DataFrame:
    juntas = pd.concat(repeticoes, ignore_index=True)
    grupos = juntas.groupby("n_removidos", sort=True)
    curva = grupos[["fracao_removida", *METRICAS]].mean()
    desvio = grupos[list(METRICAS)].std(ddof=1)  # NaN com uma repetição só
    desvio.columns = list(COLUNAS_DP)
    return pd.concat([curva, desvio], axis=1).reset_index()[[*COLUNAS, *COLUNAS_DP]]


def curva_aleatoria(G: nx.DiGraph, n_rep: int, semente: int,
                    nutrientes: Iterable[str] = (),
                    medir_eficiencia: bool = True,
                    progresso: bool = False) -> pd.DataFrame:
    """Média e desvio padrão de n_rep simulações aleatórias, por fração removida.

    Colunas: as de simular_remocao, mais <metrica>_dp para cada métrica
    (desvio padrão amostral; NaN com n_rep=1).
    Cada repetição usa np.random.default_rng([semente, i]), para ser reprodutível.
    progresso=True mostra uma barra (tqdm) com as repetições.
    """
    return _media_das_repeticoes(
        _repeticoes_aleatorias(G, n_rep, semente, nutrientes, medir_eficiencia, progresso))


def _indice(curva: pd.DataFrame, coluna: str) -> float:
    """Índice R de uma curva: a média depois de cada remoção (sem o passo 0)."""
    return metricas.indice_robustez(curva.loc[curva["n_removidos"] > 0, coluna])


def comparar_estrategias(G: nx.DiGraph, estrategias: Iterable[str] = config.ESTRATEGIAS,
                         n_rep: int = config.N_REPETICOES_ALEATORIAS,
                         semente: int = config.SEMENTE,
                         nutrientes: Iterable[str] = (),
                         medir_eficiencia: bool = True,
                         progresso: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda todas as estratégias em G. É o que o exp1 e o exp4 chamam.

    Retorna (curvas, indices):
        curvas:  formato longo, coluna "estrategia" + colunas de simular_remocao
                 (+ as colunas _dp, preenchidas só na aleatória)
        indices: uma linha por estratégia, com R_fraco e R_forte (metricas.indice_robustez).
                 Na aleatória, R é a média dos R de cada repetição (igual ao R da
                 curva média) e R_*_dp é o desvio padrão entre as repetições; nas
                 determinísticas, R_*_dp é NaN e n_rep é 1.
    """
    nutrientes = list(nutrientes)
    curvas, indices = [], []
    for estrategia in estrategias:
        if estrategia == "aleatoria":
            repeticoes = _repeticoes_aleatorias(G, n_rep, semente, nutrientes, medir_eficiencia, progresso)
            curva = _media_das_repeticoes(repeticoes)
            r_fraco = np.array([_indice(r, "maior_fraco") for r in repeticoes])
            r_forte = np.array([_indice(r, "maior_forte") for r in repeticoes])
            dp = (lambda v: float(v.std(ddof=1)) if len(v) > 1 else math.nan)
            indices.append({"estrategia": estrategia,
                            "R_fraco": float(r_fraco.mean()), "R_forte": float(r_forte.mean()),
                            "R_fraco_dp": dp(r_fraco), "R_forte_dp": dp(r_forte), "n_rep": n_rep})
        else:
            curva = simular_remocao(G, estrategia, nutrientes=nutrientes, medir_eficiencia=medir_eficiencia)
            indices.append({"estrategia": estrategia,
                            "R_fraco": _indice(curva, "maior_fraco"), "R_forte": _indice(curva, "maior_forte"),
                            "R_fraco_dp": math.nan, "R_forte_dp": math.nan, "n_rep": 1})
        curva = curva.reindex(columns=[*COLUNAS, *COLUNAS_DP])
        curva.insert(0, "estrategia", estrategia)
        curvas.append(curva)
    return (pd.concat(curvas, ignore_index=True),
            pd.DataFrame(indices, columns=list(COLUNAS_INDICES)))
