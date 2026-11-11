"""Estratégias de remoção de nós e curvas de robustez (Albert, Jeong e Barabási, 2000).

Estratégias (nomes usados em config.ESTRATEGIAS), cada centralidade nas versões
estática (calculada uma vez, na rede intacta) e adaptativa (recalculada na rede
que sobrou):
    aleatoria               ordem sorteada; média de várias repetições
    grau                    maior grau primeiro, calculado uma vez
    grau_adaptativo         recalcula o grau depois de cada remoção
    betweenness             maior betweenness primeiro, calculada uma vez
    betweenness_adaptativo  recalcula a cada config.PASSO_BETWEENNESS (10) remoções
    pagerank                maior PageRank primeiro, calculado uma vez
    pagerank_adaptativo     recalcula a cada config.PASSO_BETWEENNESS remoções

O grau é o total (entrada + saída) no grafo simples, o mesmo de metricas.graus:
um vizinho ligado por uma reação reversível (u -> v e v -> u) conta duas vezes.
A betweenness é a exata de metricas.betweenness. O PageRank é o nx.pagerank
com os parâmetros padrão (amortecimento 0,85), com peso = número de reações
entre o par: o passeio aleatório segue cada reação com a mesma chance, como no
multigrafo de reações (no grafo não dirigido dos modelos clássicos não há peso).
A centralidade de autovetor fica de fora: numa rede dirigida que não é
fortemente conexa ela não é bem definida (quase todos os nós ficam com zero).

Fontes protegidas: os nutrientes de onde parte o alcance vão para o fim da
ordem em todas as estratégias. Sem isso, o ataque por grau removia a glicose
(grau 12, entre os 22 mais conectados da eco sem moeda) e o alcance caía a zero
por isso, e não pelo dano à rede.

Índices por estratégia (comparar_estrategias), para a Tabela 2 (Exp 1):
    R_fraco, R_forte            índice R (Schneider et al., 2011) do maior componente
    R_fraco_norm, R_forte_norm  o mesmo R dividido pelo maior componente na rede
                                intacta (S0_fraco, S0_forte)
    metade_fraco, metade_forte  fração removida até o maior componente cair à metade

A ordem de remoção (ordem_de_remocao) também é usada nos modelos clássicos do
Exp 4 (modelos_classicos.py) e na checagem dos primeiros removidos
(checar_inferencias.py).

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

ESTATICAS = ("aleatoria", "grau", "betweenness", "pagerank")
ADAPTATIVAS = ("grau_adaptativo", "betweenness_adaptativo", "pagerank_adaptativo")
# As adaptativas que recalculam em lote: removem em sequência os
# config.PASSO_BETWEENNESS primeiros do ranking e só então recalculam.
EM_LOTE = ("betweenness_adaptativo", "pagerank_adaptativo")
METRICAS = ("maior_fraco", "maior_forte", "alcance")
COLUNAS = ("n_removidos", "fracao_removida", *METRICAS)
COLUNAS_DP = tuple(f"{m}_dp" for m in METRICAS)
COLUNAS_INDICES = ("estrategia", "R_fraco", "R_forte", "R_fraco_dp", "R_forte_dp",
                   "metade_fraco", "metade_forte", "metade_fraco_dp", "metade_forte_dp", "n_rep",
                   "S0_fraco", "S0_forte", "R_fraco_norm", "R_forte_norm")

# As centralidades contínuas são arredondadas antes de ordenar: dois nós com o
# mesmo valor exato podem diferir na 17ª casa pela ordem da soma, e o desempate
# pelo código só funciona se o empate aparecer.
_CASAS_BETWEENNESS = 12


def _por_valor(valores: dict) -> list:
    """Nós do maior para o menor valor; empates pelo código do nó."""
    return sorted(valores, key=lambda no: (-valores[no], no))


def _arredondar(valores) -> dict:
    return {no: round(float(v), _CASAS_BETWEENNESS) for no, v in valores.items()}


def _betweenness(G: nx.DiGraph) -> dict:
    return _arredondar(metricas.betweenness(G))


def _pagerank(G: nx.DiGraph) -> dict:
    return _arredondar(nx.pagerank(G)) if G.number_of_nodes() else {}


_CENTRALIDADES = {
    "grau": lambda G: dict(G.degree()),
    "betweenness": _betweenness,
    "pagerank": _pagerank,
}


def _ranking_atual(G: nx.DiGraph, estrategia: str) -> list:
    """Nós de G do mais para o menos central, na centralidade da estratégia (estática ou adaptativa)."""
    criterio = estrategia.removesuffix("_adaptativo")
    if criterio not in _CENTRALIDADES:
        raise ValueError(f"estratégia desconhecida: {estrategia!r}")
    return _por_valor(_CENTRALIDADES[criterio](G))


def ordem_inicial(G: nx.DiGraph, estrategia: str,
                  rng: np.random.Generator | None = None) -> list[str]:
    """Ordem de remoção das estratégias estáticas (aleatoria, grau, betweenness, pagerank).

    Empates na centralidade são desfeitos pelo código do nó, para que a ordem
    seja sempre a mesma. Na aleatória, o sorteio parte dos nós em ordem de
    código, então a mesma semente dá a mesma ordem em qualquer máquina.
    Não protege nenhum nó (ver ordem_de_remocao).
    """
    if estrategia == "aleatoria":
        if rng is None:
            raise ValueError("a estratégia aleatória precisa de um rng")
        nos = sorted(G.nodes)
        return [nos[i] for i in rng.permutation(len(nos))]
    if estrategia in ESTATICAS:
        return _ranking_atual(G, estrategia)
    raise ValueError(f"estratégia estática desconhecida: {estrategia!r} (use uma de {ESTATICAS})")


def proximo_adaptativo(G: nx.DiGraph, estrategia: str, protegidos: Iterable = ()) -> str:
    """Próximo nó a remover nas estratégias adaptativas, no estado atual de G.

    grau_adaptativo: maior grau total agora. As de EM_LOTE: o primeiro do ranking
    atual (quem chama decide recalcular só a cada passo_betweenness).
    protegidos: só são escolhidos quando não sobra nenhum outro nó.
    Empates pelo código do nó.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("o grafo está vazio")
    protegidos = set(protegidos)
    if estrategia == "grau_adaptativo":
        candidatos = [no for no in G.nodes if no not in protegidos] or list(G.nodes)
        return min(candidatos, key=lambda no: (-G.degree(no), no))
    if estrategia in EM_LOTE:
        ranking = _ranking_atual(G, estrategia)
        return next((no for no in ranking if no not in protegidos), ranking[0])
    raise ValueError(f"estratégia adaptativa desconhecida: {estrategia!r} (use uma de {ADAPTATIVAS})")


def ordem_de_remocao(G: nx.DiGraph, estrategia: str, rng: np.random.Generator | None = None,
                     protegidos: Iterable = (),
                     passo_betweenness: int = config.PASSO_BETWEENNESS) -> list:
    """A sequência completa de remoção dos nós de G (todos), em qualquer estratégia.

    Estáticas: ordem_inicial. Adaptativas: remove os nós de uma cópia de G, um a
    um, escolhendo cada próximo no grafo que sobrou; as de EM_LOTE removem em
    sequência os passo_betweenness primeiros do ranking e só então recalculam.
    A curva de uma remoção só depende dessa sequência: simular_remocao mede ao
    longo dela, e modelos_classicos.maior_componente_por_ordem a percorre ao contrário.

    protegidos: os presentes em G vão para o fim, em todas as estratégias. Nas
    estáticas, a ordem dos outros não muda e os protegidos vêm depois, na ordem
    da própria estratégia. Nas adaptativas, nunca são escolhidos enquanto
    sobrar outro nó; no fim, saem na ordem da estratégia no que restou.
    """
    if estrategia not in ESTATICAS + ADAPTATIVAS:
        raise ValueError(f"estratégia desconhecida: {estrategia!r}")
    if passo_betweenness < 1:
        raise ValueError("passo_betweenness tem que ser pelo menos 1")
    restantes = {no for no in protegidos if no in G}
    if estrategia in ESTATICAS:
        ordem = ordem_inicial(G, estrategia, rng)
        return [no for no in ordem if no not in restantes] + [no for no in ordem if no in restantes]
    H = G.copy()
    ordem: list = []
    fila: deque = deque()
    while H.number_of_nodes():
        if estrategia == "grau_adaptativo":
            no = proximo_adaptativo(H, estrategia, restantes)
        else:
            if not fila:
                ranking = _ranking_atual(H, estrategia)
                livres = [n for n in ranking if n not in restantes] or ranking
                fila.extend(livres[:passo_betweenness])
            no = fila.popleft()
        H.remove_node(no)
        restantes.discard(no)
        ordem.append(no)
    return ordem


def _medir(H: nx.DiGraph, k: int, n: int, nutrientes: list) -> dict:
    return {
        "n_removidos": k,
        "fracao_removida": k / n,
        "maior_fraco": metricas.fracao_maior_componente(H, n),
        "maior_forte": metricas.fracao_maior_componente(H, n, forte=True),
        "alcance": metricas.alcance(H, nutrientes, n) if nutrientes else math.nan,
    }


def simular_remocao(G: nx.DiGraph, estrategia: str,
                    rng: np.random.Generator | None = None,
                    nutrientes: Iterable[str] = (),
                    passo_betweenness: int = config.PASSO_BETWEENNESS) -> pd.DataFrame:
    """Remove os nós de G um a um, na ordem da estratégia, até não sobrar nenhum.

    Uma linha por passo (de 0 a N nós removidos), colunas:
        n_removidos, fracao_removida,
        maior_fraco, maior_forte  (frações do tamanho original, a cada passo)
        alcance                   (a partir dos nutrientes, a cada passo; NaN
                                   em todos se nenhum nutriente for passado)
    Trabalha numa cópia de G. rng é obrigatório na estratégia aleatória. A ordem
    vem de ordem_de_remocao: nas de EM_LOTE, remove em sequência os
    passo_betweenness primeiros do ranking e só então recalcula.
    Os nutrientes presentes em G, que são as fontes do alcance, vão para o fim
    da ordem em todas as estratégias, para o alcance não cair a zero só porque a
    própria fonte foi removida. Na eco sem moeda são 2 dos 1.111 nós (glicose e
    sulfato; NH3 e ortofosfato são moeda).
    """
    if estrategia not in ESTATICAS + ADAPTATIVAS:
        raise ValueError(f"estratégia desconhecida: {estrategia!r}")
    if passo_betweenness < 1:
        raise ValueError("passo_betweenness tem que ser pelo menos 1")
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("o grafo está vazio")
    nutrientes = list(nutrientes)
    ordem = ordem_de_remocao(G, estrategia, rng, nutrientes, passo_betweenness)

    H = G.copy()
    linhas = [_medir(H, 0, n, nutrientes)]
    for k, no in enumerate(ordem, start=1):
        H.remove_node(no)
        linhas.append(_medir(H, k, n, nutrientes))
    return pd.DataFrame(linhas, columns=list(COLUNAS))


def _repeticoes_aleatorias(G: nx.DiGraph, n_rep: int, semente: int, nutrientes: Iterable[str],
                           progresso: bool = False) -> list[pd.DataFrame]:
    """n_rep simulações aleatórias; a repetição i usa np.random.default_rng([semente, i])."""
    if n_rep < 1:
        raise ValueError("n_rep tem que ser pelo menos 1")
    nutrientes = list(nutrientes)
    repeticoes = range(n_rep)
    if progresso:
        from tqdm import tqdm  # import tardio: só quem pede a barra precisa do tqdm
        repeticoes = tqdm(repeticoes, desc="aleatoria", unit="rep")
    return [simular_remocao(G, "aleatoria", rng=np.random.default_rng([semente, i]), nutrientes=nutrientes)
            for i in repeticoes]


def _media_das_repeticoes(repeticoes: list[pd.DataFrame]) -> pd.DataFrame:
    juntas = pd.concat(repeticoes, ignore_index=True)
    grupos = juntas.groupby("n_removidos", sort=True)
    curva = grupos[["fracao_removida", *METRICAS]].mean()
    desvio = grupos[list(METRICAS)].std(ddof=1)  # NaN com uma repetição só
    desvio.columns = list(COLUNAS_DP)
    return pd.concat([curva, desvio], axis=1).reset_index()[[*COLUNAS, *COLUNAS_DP]]


def _indice(curva: pd.DataFrame, coluna: str) -> float:
    """Índice R de uma curva: a média depois de cada remoção (sem o passo 0)."""
    return metricas.indice_robustez(curva.loc[curva["n_removidos"] > 0, coluna])


def fracao_ate_metade(curva: pd.DataFrame, coluna: str = "maior_fraco") -> float:
    """Fração removida no primeiro passo em que a métrica cai à metade do valor inicial ou menos.

    O valor inicial é o do passo 0 (rede intacta). NaN se ele for 0. Como a
    simulação vai até não sobrar nó, a métrica sempre chega a 0 no fim, e a
    resposta existe sempre que o valor inicial é positivo.
    """
    curva = curva.sort_values("n_removidos")
    inicial = float(curva[coluna].iloc[0])
    if not inicial > 0:
        return math.nan
    caiu = curva[(curva["n_removidos"] > 0) & (curva[coluna] <= inicial / 2)]
    return float(caiu["fracao_removida"].iloc[0]) if len(caiu) else math.nan


def _desvio(valores: np.ndarray) -> float:
    return float(valores.std(ddof=1)) if len(valores) > 1 else math.nan


def comparar_estrategias(G: nx.DiGraph, estrategias: Iterable[str] = config.ESTRATEGIAS,
                         n_rep: int = config.N_REPETICOES_ALEATORIAS,
                         semente: int = config.SEMENTE,
                         nutrientes: Iterable[str] = (),
                         progresso: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda todas as estratégias em G. É o que o Exp 1 chama.

    Retorna (curvas, indices):
        curvas:  formato longo, coluna "estrategia" + colunas de simular_remocao
                 (+ as colunas _dp, preenchidas só na aleatória)
        indices: uma linha por estratégia, com R_fraco e R_forte (metricas.indice_robustez)
                 e metade_fraco e metade_forte (fracao_ate_metade).
                 Na aleatória, cada índice é a média dos valores de cada repetição e
                 *_dp é o desvio padrão entre elas. O R da média é igual ao R da curva
                 média (o índice é linear); a metade não é, então metade_* pode
                 diferir um pouco do ponto em que a curva média cruza a metade.
                 Nas determinísticas, *_dp é NaN e n_rep é 1.
                 S0_fraco e S0_forte: o maior componente na rede intacta (fração dos
                 nós); R_fraco_norm = R_fraco / S0_fraco e R_forte_norm = R_forte / S0_forte
                 (a área sob a curva relativa ao início; NaN se S0 = 0).
    Os nutrientes vão para o fim da ordem em todas as estratégias (simular_remocao).
    """
    nutrientes = list(nutrientes)
    curvas, indices = [], []
    for estrategia in estrategias:
        if estrategia == "aleatoria":
            repeticoes = _repeticoes_aleatorias(G, n_rep, semente, nutrientes, progresso)
            curva = _media_das_repeticoes(repeticoes)
            linha = {"estrategia": estrategia, "n_rep": n_rep}
            for nome, funcao in (("R", _indice), ("metade", fracao_ate_metade)):
                for componente in ("fraco", "forte"):
                    valores = np.array([funcao(r, f"maior_{componente}") for r in repeticoes])
                    linha[f"{nome}_{componente}"] = float(valores.mean())
                    linha[f"{nome}_{componente}_dp"] = _desvio(valores)
        else:
            curva = simular_remocao(G, estrategia, nutrientes=nutrientes)
            linha = {"estrategia": estrategia, "n_rep": 1}
            for nome, funcao in (("R", _indice), ("metade", fracao_ate_metade)):
                for componente in ("fraco", "forte"):
                    linha[f"{nome}_{componente}"] = funcao(curva, f"maior_{componente}")
                    linha[f"{nome}_{componente}_dp"] = math.nan
        inicio = curva.loc[curva["n_removidos"] == 0].iloc[0]
        for componente in ("fraco", "forte"):
            s0 = float(inicio[f"maior_{componente}"])
            linha[f"S0_{componente}"] = s0
            linha[f"R_{componente}_norm"] = linha[f"R_{componente}"] / s0 if s0 > 0 else math.nan
        indices.append(linha)
        curva = curva.reindex(columns=[*COLUNAS, *COLUNAS_DP])
        curva.insert(0, "estrategia", estrategia)
        curvas.append(curva)
    return (pd.concat(curvas, ignore_index=True),
            pd.DataFrame(indices, columns=list(COLUNAS_INDICES)))
