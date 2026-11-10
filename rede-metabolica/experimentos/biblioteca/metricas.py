"""Métricas usadas por mais de um experimento.

Descritivas (exp0, exp4): tamanho, graus, clustering, caminhos, bow-tie.
De dano (exp1, exp4): maior componente, índice R, eficiência, alcance.
Dano estrutural de uma remoção (exp2, exp3): queda do maior componente forte,
do maior componente fraco e da eficiência (medidas_estruturais, queda_estrutural,
dano_remocao_individual).
Topológicas para o exp2: centralidades (betweenness, harmônica) e choke points.

Todas recebem o grafo simples dirigido (grafo.grafo_simples), exceto
choke_points, que trabalha direto na tabela de reações. matriz_distancias,
harmonica e assortatividade aceitam também o grafo não dirigido dos modelos
clássicos (modelos_classicos.py).
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

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


_MINIMO_AJUSTE = 10  # abaixo disso, o ajuste não faz sentido


def _ajustar(dados: np.ndarray):
    """powerlaw.Fit discreto (Clauset et al., 2009), sem avisos; None se houver poucos dados."""
    import powerlaw  # só aqui: o pacote é opcional e demora para importar

    if len(dados) < _MINIMO_AJUSTE:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return powerlaw.Fit(dados, discrete=True, verbose=False)


def ajuste_lei_potencia(valores_grau: Iterable[int]) -> dict:
    """Opcional: teste de Clauset et al. com o pacote powerlaw.

    Devolve alpha, xmin, quantos nós ficam na cauda (grau >= xmin) e a
    comparação com a log-normal: R > 0 favorece a lei de potência, R < 0 a
    log-normal, e p < 0,1 indica que a diferença é significativa.
    """
    dados = np.array([v for v in valores_grau if v > 0])
    ajuste = _ajustar(dados)
    if ajuste is None:
        return {"alpha": math.nan, "xmin": math.nan, "n_cauda": 0, "R_vs_lognormal": math.nan, "p": math.nan}
    with warnings.catch_warnings():  # o powerlaw ajusta as distribuições quando são acessadas
        warnings.simplefilter("ignore")
        R, p = ajuste.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        xmin, alpha = float(ajuste.power_law.xmin), float(ajuste.power_law.alpha)
    return {"alpha": alpha, "xmin": xmin,
            "n_cauda": int((dados >= xmin).sum()), "R_vs_lognormal": float(R), "p": float(p)}


def ccdf_com_ajustes(valores_grau: Iterable[int]) -> pd.DataFrame:
    """CCDF empírica P(K >= k) e as dos ajustes de lei de potência e log-normal.

    Uma linha por grau k de 1 ao máximo. Colunas: grau, n_nos (quantos nós têm
    exatamente esse grau; a figura só desenha a empírica onde n_nos > 0),
    ccdf_empirica, ccdf_lei_potencia, ccdf_lognormal. Os ajustes são os mesmos de
    ajuste_lei_potencia (mesmo xmin) e só existem de xmin em diante (NaN antes);
    estão multiplicados pela fração dos dados na cauda, para coincidirem com a
    empírica em xmin, como nas figuras de Clauset et al. (2009). Graus 0 ficam
    de fora, como no ajuste.
    """
    dados = np.array([v for v in valores_grau if v > 0], dtype=int)
    colunas = ["grau", "n_nos", "ccdf_empirica", "ccdf_lei_potencia", "ccdf_lognormal"]
    if len(dados) == 0:
        return pd.DataFrame(columns=colunas)
    ks = np.arange(1, dados.max() + 1)
    ordenados = np.sort(dados)
    empirica = 1.0 - np.searchsorted(ordenados, ks, side="left") / len(dados)
    n_nos = np.bincount(dados, minlength=dados.max() + 1)[1:]
    tabela = pd.DataFrame({"grau": ks, "n_nos": n_nos, "ccdf_empirica": empirica,
                           "ccdf_lei_potencia": math.nan, "ccdf_lognormal": math.nan})
    ajuste = _ajustar(dados)
    if ajuste is not None:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            xmin = int(ajuste.power_law.xmin)
            cauda = ks >= xmin
            fracao_cauda = float((dados >= xmin).mean())
            tabela.loc[cauda, "ccdf_lei_potencia"] = fracao_cauda * ajuste.power_law.ccdf(ks[cauda])
            tabela.loc[cauda, "ccdf_lognormal"] = fracao_cauda * ajuste.lognormal.ccdf(ks[cauda])
    return tabela[colunas]


def assortatividade(G: nx.DiGraph) -> float:
    """Assortatividade de grau (saída da origem x entrada do destino); nan se indefinida."""
    if G.number_of_edges() == 0:
        return math.nan
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        valor = nx.degree_assortativity_coefficient(G)
    return float(valor) if valor is not None else math.nan


def reciprocidade(G: nx.DiGraph) -> float:
    """Fração das arestas u→v que têm a volta v→u (Garlaschelli e Loffredo, 2004); nan sem arestas.

    Na rede metabólito-cêntrica, cada reação reversível vira um par desses.
    """
    if G.number_of_edges() == 0:
        return math.nan
    return float(nx.reciprocity(G))


def tabela_descritiva(G: nx.DiGraph) -> dict:
    """Uma linha da Tabela 1 do artigo para o grafo simples dirigido G.

    Chaves: nos, arestas, densidade, grau_medio, clustering_medio (na versão
    não dirigida), caminho_medio, diametro, bow-tie (núcleo, entrada, saída,
    restante), assortatividade e reciprocidade. É reaproveitada pelo exp4
    para os outros organismos e para os modelos nulos.

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
        "reciprocidade": reciprocidade(G),
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


def matriz_distancias(G: nx.Graph, nos: list | None = None) -> np.ndarray:
    """Distâncias (número de arestas) entre todos os pares, D[i, j] = d(nos[i], nos[j]).

    scipy.sparse.csgraph.shortest_path(method="D", unweighted=True), dirigida se G
    for dirigido. inf onde não há caminho e 0 na diagonal. nos: a ordem das linhas
    e colunas (padrão: a de G.nodes). Ocupa N² números (~10 MB na eco).
    """
    nos = list(G.nodes) if nos is None else list(nos)
    posicao = {no: i for i, no in enumerate(nos)}
    origem = [posicao[u] for u, _ in G.edges()]
    destino = [posicao[v] for _, v in G.edges()]
    A = csr_matrix((np.ones(len(origem)), (origem, destino)), shape=(len(nos), len(nos)))
    return shortest_path(A, method="D", directed=G.is_directed(), unweighted=True)


def eficiencia_global_dirigida_rapida(G: nx.DiGraph, n_original: int) -> float:
    """A mesma eficiência de eficiencia_global_dirigida, com as distâncias calculadas pela scipy.

    Soma 1/d(i, j) sobre os pares ordenados com caminho (matriz_distancias) e divide
    pelo mesmo n_original*(n_original-1). Dá o mesmo valor da versão em Python
    puro (até ~1e-11 relativo, a diferença de arredondamento das duas somas; ver
    testes/test_dano_estrutural.py) em menos da metade do tempo: é a usada nos
    ~1.700 nocautes e remoções individuais dos Exps 2 e 3.
    """
    if n_original < 2 or G.number_of_edges() == 0:
        return 0.0
    D = matriz_distancias(G)
    com_caminho = D[np.isfinite(D) & (D > 0)]
    return float((1.0 / com_caminho).sum()) / (n_original * (n_original - 1))


def harmonica(G: nx.Graph) -> pd.Series:
    """Centralidade harmônica de cada nó: soma de 1/d(v, u) sobre os nós v que alcançam u.

    É a nx.harmonic_centrality (Marchiori e Latora, 2000; Boldi e Vigna, 2014), a
    versão da proximidade (closeness) que funciona em rede desconexa: par sem
    caminho conta 0. Num grafo dirigido, conta as distâncias que chegam ao nó.
    Calculada pelas contagens de distância (quantos v estão a 1, 2, 3... arestas
    de u), e não somando 1/d na ordem em que a NetworkX visita as origens: assim
    dois nós com a mesma contagem têm exatamente o mesmo valor (o empate aparece e
    é desfeito pelo código do nó) e o resultado não muda de uma execução para
    outra. Os valores batem com os da NetworkX (testado). Índice = nó, ordenado.
    """
    nos = sorted(G.nodes)
    if not nos:
        return pd.Series(dtype=float, name="harmonica")
    D = matriz_distancias(G, nos)
    finitas = D[np.isfinite(D)]
    maior = int(finitas.max()) if finitas.size else 0
    valores = np.zeros(len(nos))
    for d in range(1, maior + 1):
        valores += (D == d).sum(axis=0) / d   # coluna u: os v com d(v, u) = d
    return pd.Series(valores, index=nos, name="harmonica")


# ---------------------------------------------------------------------------
# Dano estrutural de uma remoção (Exps 2 e 3)
# ---------------------------------------------------------------------------
def medidas_estruturais(G: nx.DiGraph, n_original: int) -> dict:
    """Tamanho (em nós) do maior componente forte e do maior fraco, e a eficiência global dirigida.

    A eficiência é a de eficiencia_global_dirigida_rapida, normalizada por n_original.
    """
    return {
        "maior_forte": max((len(c) for c in nx.strongly_connected_components(G)), default=0),
        "maior_fraco": max((len(c) for c in nx.weakly_connected_components(G)), default=0),
        "eficiencia": eficiencia_global_dirigida_rapida(G, n_original),
    }


COLUNAS_DANO_ESTRUTURAL = ("dano_forte", "dano_fraco", "dano_forte_fracao", "dano_fraco_fracao",
                           "dano_eficiencia")


def queda_estrutural(antes: dict, depois: dict, n_original: int) -> dict:
    """Dano estrutural de uma remoção, a partir de duas medidas_estruturais.

    dano_forte, dano_fraco: nós que deixam o maior componente forte (o núcleo) e o
        maior componente fraco, calculados como tamanho antes menos tamanho depois
        (a mesma régua das curvas do Exp 1);
    dano_forte_fracao, dano_fraco_fracao: os mesmos, divididos por n_original;
    dano_eficiencia: queda relativa da eficiência global, (E0 - E)/E0 (NaN se E0 = 0).
    """
    e0 = antes["eficiencia"]
    forte = antes["maior_forte"] - depois["maior_forte"]
    fraco = antes["maior_fraco"] - depois["maior_fraco"]
    return {"dano_forte": forte, "dano_fraco": fraco,
            "dano_forte_fracao": forte / n_original if n_original else math.nan,
            "dano_fraco_fracao": fraco / n_original if n_original else math.nan,
            "dano_eficiencia": (e0 - depois["eficiencia"]) / e0 if e0 > 0 else math.nan}


def dano_remocao_individual(G: nx.DiGraph) -> pd.DataFrame:
    """Remove cada nó de G sozinho e mede o dano estrutural (queda_estrutural) em relação a G.

    N é o número de nós de G, o mesmo antes e depois: a eficiência depois da
    remoção também é dividida por N*(N-1), então os pares do próprio nó removido
    contam como perdidos, como nas curvas do Exp 1. Uma linha por nó, em ordem de
    código. Colunas: metabolito e COLUNAS_DANO_ESTRUTURAL. Leva ~30 s na eco.
    """
    n = G.number_of_nodes()
    simples = nx.DiGraph()
    simples.add_nodes_from(G.nodes)
    simples.add_edges_from(G.edges())   # sem atributos: a cópia a cada remoção fica mais leve
    antes = medidas_estruturais(simples, n)
    linhas = []
    for no in sorted(simples.nodes):
        H = simples.copy()
        H.remove_node(no)
        linhas.append({"metabolito": no, **queda_estrutural(antes, medidas_estruturais(H, n), n)})
    return pd.DataFrame(linhas, columns=["metabolito", *COLUNAS_DANO_ESTRUTURAL])


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
