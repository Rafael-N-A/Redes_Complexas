"""Testes das estratégias de remoção e das curvas de robustez (robustez.py).

Quatro tipos de teste:
  1. ordens de remoção em grafos pequenos com a resposta feita à mão;
  2. uma curva inteira feita à mão (estrela com um par solto);
  3. resultados analíticos do índice R (grafo completo e estrela);
  4. comparação com uma implementação ingênua, que refaz cada passo do zero
     com funções da NetworkX em subgrafos, em grafos dirigidos sorteados.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
from experimentos.biblioteca import metricas, robustez


def estrela_com_par_solto() -> nx.DiGraph:
    """h <-> x1, h <-> x2, h -> x3 e o par solto x4 -> x5 (6 nós).

    Graus totais: h=5, x1=2, x2=2, x3=1, x4=1, x5=1.
    """
    G = nx.DiGraph()
    G.add_edges_from([("h", "x1"), ("x1", "h"), ("h", "x2"), ("x2", "h"), ("h", "x3"), ("x4", "x5")])
    return G


def caminho_dirigido() -> nx.DiGraph:
    """a -> b -> c -> d -> e. Betweenness (pares ordenados): b=3, c=4, d=3, a=e=0."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])


# ---------------------------------------------------------------------------
# 1. Ordens de remoção
# ---------------------------------------------------------------------------
def test_ordem_por_grau_com_desempate_pelo_codigo():
    assert robustez.ordem_inicial(estrela_com_par_solto(), "grau") == ["h", "x1", "x2", "x3", "x4", "x5"]


def test_ordem_por_betweenness():
    assert robustez.ordem_inicial(caminho_dirigido(), "betweenness") == ["c", "b", "d", "a", "e"]


def test_ordem_aleatoria_e_permutacao_reprodutivel():
    G = nx.gnp_random_graph(30, 0.1, directed=True, seed=1)
    G = nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})
    ordem = robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 0]))
    assert sorted(ordem) == sorted(G.nodes)
    assert ordem == robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 0]))
    assert ordem != robustez.ordem_inicial(G, "aleatoria", np.random.default_rng([42, 1]))


def test_proximo_adaptativo():
    G = estrela_com_par_solto()
    assert robustez.proximo_adaptativo(G, "grau_adaptativo") == "h"
    G.remove_node("h")
    # Sem o hub, só x4 e x5 têm grau 1; empate desfeito pelo código.
    assert robustez.proximo_adaptativo(G, "grau_adaptativo") == "x4"
    assert robustez.proximo_adaptativo(caminho_dirigido(), "betweenness_adaptativo") == "c"


def test_erros_de_uso():
    G = caminho_dirigido()
    with pytest.raises(ValueError):
        robustez.ordem_inicial(G, "aleatoria")  # sem rng
    with pytest.raises(ValueError):
        robustez.ordem_inicial(G, "grau_adaptativo")  # não é estática
    with pytest.raises(ValueError):
        robustez.simular_remocao(G, "autovetor")  # não existe (ver o docstring de robustez.py)
    with pytest.raises(ValueError):
        robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=0)


# ---------------------------------------------------------------------------
# 2. Curva feita à mão
# ---------------------------------------------------------------------------
def test_curva_grau_feita_a_mao():
    """Ordem h, x1, x2, x3, x4, x5 (sem proteger a fonte h).

    Passo 0: fraco {h,x1,x2,x3}=4, forte {h,x1,x2}=3, alcance a partir de h = 4.
    Sem h: sobra só o par x4 -> x5 (fraco 2, forte 1) até x4 sair.
    """
    curva = robustez.simular_remocao(estrela_com_par_solto(), "grau", nutrientes=["h"],
                                     medir_eficiencia=False, proteger_fontes=False)
    assert list(curva["n_removidos"]) == list(range(7))
    assert np.allclose(curva["fracao_removida"], np.arange(7) / 6)
    assert np.allclose(curva["maior_fraco"] * 6, [4, 2, 2, 2, 2, 1, 0])
    assert np.allclose(curva["maior_forte"] * 6, [3, 1, 1, 1, 1, 1, 0])
    assert np.allclose(curva["alcance"] * 6, [4, 0, 0, 0, 0, 0, 0])
    assert curva["eficiencia"].isna().all()


def test_curva_grau_adaptativo_feita_a_mao():
    """Ordem h, x4 (o par solto vira o de maior grau), x1, x2, x3, x5."""
    curva = robustez.simular_remocao(estrela_com_par_solto(), "grau_adaptativo", medir_eficiencia=False)
    assert np.allclose(curva["maior_fraco"] * 6, [4, 2, 1, 1, 1, 1, 0])
    assert curva["alcance"].isna().all()  # sem nutrientes, o alcance não é medido


def test_indices_da_curva_feita_a_mao():
    """R_fraco = (2+2+2+2+1+0)/6/6 = 1/4; R_forte = (1+1+1+1+1+0)/6/6 = 5/36.

    Metade: tirar o hub leva o fraco de 4 para 2 e o forte de 3 para 1, então
    os dois caem à metade logo na primeira remoção (1/6).
    """
    _, indices = robustez.comparar_estrategias(estrela_com_par_solto(), ["grau"], medir_eficiencia=False)
    linha = indices.iloc[0]
    assert linha["estrategia"] == "grau" and linha["n_rep"] == 1
    assert math.isclose(linha["R_fraco"], 1 / 4) and math.isclose(linha["R_forte"], 5 / 36)
    assert math.isclose(linha["metade_fraco"], 1 / 6) and math.isclose(linha["metade_forte"], 1 / 6)
    assert math.isnan(linha["R_fraco_dp"]) and math.isnan(linha["metade_fraco_dp"])


def test_fracao_ate_metade_em_curvas_montadas():
    fracao = [0, 0.25, 0.5, 0.75, 1]
    curva = pd.DataFrame({"n_removidos": range(5), "fracao_removida": fracao,
                          "maior_fraco": [1.0, 0.8, 0.5, 0.2, 0.0],   # cai exatamente à metade
                          "maior_forte": [0.8, 0.6, 0.41, 0.39, 0.0]})  # passa um pouco da metade
    assert robustez.fracao_ate_metade(curva) == 0.5
    assert robustez.fracao_ate_metade(curva, "maior_forte") == 0.75
    assert robustez.fracao_ate_metade(curva.iloc[::-1]) == 0.5  # a ordem das linhas não importa
    curva["maior_forte"] = 0.0
    assert math.isnan(robustez.fracao_ate_metade(curva, "maior_forte"))  # sem componente inicial


# ---------------------------------------------------------------------------
# 3. Resultados analíticos do índice R
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("estrategia", ["aleatoria", "grau", "grau_adaptativo", "betweenness"])
def test_r_do_grafo_completo(estrategia):
    """No grafo completo, qualquer ordem deixa os N-Q restantes conectados:
    s(Q) = (N-Q)/N e R = (N-1)/(2N), o máximo possível."""
    n = 12
    G = nx.complete_graph([f"n{i:02d}" for i in range(n)], create_using=nx.DiGraph)
    _, indices = robustez.comparar_estrategias(G, [estrategia], n_rep=3, medir_eficiencia=False)
    assert math.isclose(indices["R_fraco"][0], (n - 1) / (2 * n))
    assert math.isclose(indices["R_forte"][0], (n - 1) / (2 * n))
    # s(Q) = (N-Q)/N chega à metade em Q = N/2, qualquer que seja a ordem.
    assert math.isclose(indices["metade_fraco"][0], 0.5) and math.isclose(indices["metade_forte"][0], 0.5)


def test_r_da_estrela_sob_ataque():
    """Estrela de mão dupla: o ataque tira o centro primeiro e sobram N-1 nós soltos.
    s(Q) = 1/N para Q = 1..N-1 e 0 no fim, então R = (N-1)/N²."""
    n = 10
    G = nx.DiGraph()
    for i in range(1, n):
        G.add_edges_from([("c", f"f{i}"), (f"f{i}", "c")])
    _, indices = robustez.comparar_estrategias(G, ["grau", "betweenness"], medir_eficiencia=False)
    assert np.allclose(indices["R_fraco"], (n - 1) / n**2)
    assert np.allclose(indices["metade_fraco"], 1 / n)  # o centro sai primeiro e a rede se desfaz


# ---------------------------------------------------------------------------
# Eficiência a cada 5%
# ---------------------------------------------------------------------------
def test_marcos_da_eficiencia():
    assert robustez.marcos_eficiencia(6, 0.25) == {0, 2, 3, 5, 6}
    assert robustez.marcos_eficiencia(20, 0.05) == set(range(21))
    marcos = robustez.marcos_eficiencia(1111, 0.05)
    assert len(marcos) == 21 and {0, 56, 556, 1111} <= marcos  # teto(55,55) = 56, teto(555,5) = 556


def test_eficiencia_so_nos_marcos():
    G = nx.gnp_random_graph(40, 0.08, directed=True, seed=3)
    G = nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})
    curva = robustez.simular_remocao(G, "grau", passo_eficiencia=0.1)
    medidos = set(curva.loc[curva["eficiencia"].notna(), "n_removidos"])
    assert medidos == robustez.marcos_eficiencia(40, 0.1)
    assert math.isclose(curva["eficiencia"][0], metricas.eficiencia_global_dirigida(G, 40))
    assert curva["eficiencia"].iloc[-1] == 0.0


def test_nao_altera_o_grafo():
    G = estrela_com_par_solto()
    antes = (set(G.nodes), set(G.edges))
    robustez.comparar_estrategias(G, n_rep=2, nutrientes=["h"])
    assert (set(G.nodes), set(G.edges)) == antes


# ---------------------------------------------------------------------------
# Repetições aleatórias
# ---------------------------------------------------------------------------
def grafo_sorteado(semente: int, n: int = 25, p: float = 0.08) -> nx.DiGraph:
    G = nx.gnp_random_graph(n, p, directed=True, seed=semente)
    return nx.relabel_nodes(G, {i: f"n{i:02d}" for i in G})


def test_curva_aleatoria_com_uma_repeticao_e_a_propria_simulacao():
    G = grafo_sorteado(5)
    curva = robustez.curva_aleatoria(G, n_rep=1, semente=7, nutrientes=["n00"])
    simulada = robustez.simular_remocao(G, "aleatoria", rng=np.random.default_rng([7, 0]), nutrientes=["n00"])
    pd.testing.assert_frame_equal(curva[list(robustez.COLUNAS)], simulada, check_dtype=False)
    assert curva[list(robustez.COLUNAS_DP)].isna().all().all()


def test_curva_aleatoria_e_a_media_das_repeticoes():
    G = grafo_sorteado(6)
    curva = robustez.curva_aleatoria(G, n_rep=4, semente=3, nutrientes=["n01"], medir_eficiencia=False)
    reps = [robustez.simular_remocao(G, "aleatoria", rng=np.random.default_rng([3, i]),
                                     nutrientes=["n01"], medir_eficiencia=False) for i in range(4)]
    for coluna in ("maior_fraco", "maior_forte", "alcance"):
        valores = np.vstack([r[coluna] for r in reps])
        assert np.allclose(curva[coluna], valores.mean(axis=0))
        assert np.allclose(curva[f"{coluna}_dp"], valores.std(axis=0, ddof=1))


def test_indices_da_aleatoria_sao_a_media_das_repeticoes():
    """R e metade da aleatória: média e desvio padrão dos valores de cada repetição."""
    G = grafo_sorteado(6)
    _, indices = robustez.comparar_estrategias(G, ["aleatoria"], n_rep=4, semente=3, medir_eficiencia=False)
    reps = [robustez.simular_remocao(G, "aleatoria", rng=np.random.default_rng([3, i]),
                                     medir_eficiencia=False) for i in range(4)]
    linha = indices.iloc[0]
    for componente in ("fraco", "forte"):
        metades = np.array([robustez.fracao_ate_metade(r, f"maior_{componente}") for r in reps])
        assert math.isclose(linha[f"metade_{componente}"], metades.mean())
        assert math.isclose(linha[f"metade_{componente}_dp"], metades.std(ddof=1))
        Rs = np.array([r.loc[r["n_removidos"] > 0, f"maior_{componente}"].mean() for r in reps])
        assert math.isclose(linha[f"R_{componente}_dp"], Rs.std(ddof=1))


def test_comparar_estrategias_formato_e_reprodutibilidade():
    G = grafo_sorteado(8)
    curvas, indices = robustez.comparar_estrategias(G, n_rep=5, nutrientes=["n02"])
    assert list(curvas.columns) == ["estrategia", *robustez.COLUNAS, *robustez.COLUNAS_DP]
    assert list(indices.columns) == list(robustez.COLUNAS_INDICES)
    assert list(indices["estrategia"]) == list(robustez.ESTATICAS[:1]) + [
        "grau", "grau_adaptativo", "betweenness", "betweenness_adaptativo",
        "pagerank", "pagerank_adaptativo"]
    assert (curvas.groupby("estrategia").size() == G.number_of_nodes() + 1).all()
    for linha in indices.to_dict("records"):
        curva = curvas[curvas["estrategia"] == linha["estrategia"]]
        depois = curva[curva["n_removidos"] > 0]
        # Na aleatória, a média dos R é o R da curva média (o índice é linear).
        assert math.isclose(linha["R_fraco"], depois["maior_fraco"].mean())
        assert math.isclose(linha["R_forte"], depois["maior_forte"].mean())
    de_novo = robustez.comparar_estrategias(G, n_rep=5, nutrientes=["n02"])
    pd.testing.assert_frame_equal(curvas, de_novo[0])
    pd.testing.assert_frame_equal(indices, de_novo[1])


# ---------------------------------------------------------------------------
# 4. Implementação ingênua
# ---------------------------------------------------------------------------
def _ranking_ingenuo(H: nx.DiGraph, criterio: str) -> list:
    if criterio == "grau":
        valores = {no: H.in_degree(no) + H.out_degree(no) for no in H}
    elif criterio == "pagerank":
        valores = {no: round(v, 12) for no, v in nx.pagerank(H).items()} if len(H) else {}
    else:
        valores = {no: round(v, 12) for no, v in nx.betweenness_centrality(H).items()}
    return sorted(H, key=lambda no: (-valores[no], no))


def _medidas_ingenuas(G: nx.DiGraph, restantes: set, n: int, nutrientes: list, eficiencia: bool) -> dict:
    H = G.subgraph(restantes)
    fracos = [len(c) for c in nx.weakly_connected_components(H)]
    fortes = [len(c) for c in nx.strongly_connected_components(H)]
    alcancados = set()
    for f in nutrientes:
        if f in H:
            alcancados |= nx.descendants(H, f) | {f}
    medidas = {"maior_fraco": max(fracos, default=0) / n, "maior_forte": max(fortes, default=0) / n,
               "alcance": len(alcancados) / n, "eficiencia": math.nan}
    if eficiencia:
        soma = sum(1 / d for _, dist in nx.all_pairs_shortest_path_length(H) for d in dist.values() if d > 0)
        medidas["eficiencia"] = soma / (n * (n - 1))
    return medidas


def simulacao_ingenua(G: nx.DiGraph, estrategia: str, nutrientes: list, passo_eficiencia: float,
                      passo_betweenness: int, rng=None, ate: int | None = None,
                      proteger_fontes: bool = True) -> pd.DataFrame:
    """Refaz cada passo do zero a partir de G, sem copiar nem remover nós de um grafo.

    ate: para depois de tantas remoções (o verificar_cascata_robustez usa na
    rede real, onde a versão adaptativa da betweenness feita assim é lenta).
    proteger_fontes: como em robustez.simular_remocao (o mesmo padrão), os
    nutrientes presentes em G só saem quando não sobra mais nenhum outro nó.
    """
    n = G.number_of_nodes()
    restantes = set(G)
    protegidos = {f for f in nutrientes if f in G} if proteger_fontes else set()
    criterio = estrategia.removesuffix("_adaptativo")
    if estrategia == "aleatoria":
        nos = sorted(G)
        ordem = [nos[i] for i in rng.permutation(n)]
    elif estrategia == criterio:
        ordem = _ranking_ingenuo(G, estrategia)
    if estrategia == criterio:  # estáticas: os protegidos vão para o fim, sem mudar a ordem dos outros
        ordem = [no for no in ordem if no not in protegidos] + [no for no in ordem if no in protegidos]
    multiplos = range(math.floor(1 / passo_eficiencia + 1e-9) + 1)
    marcos = {min(n, math.ceil(round(m * passo_eficiencia * n, 9))) for m in multiplos}
    linhas = [{"n_removidos": 0, "fracao_removida": 0.0,
               **_medidas_ingenuas(G, restantes, n, nutrientes, 0 in marcos)}]
    fila = []
    for k in range(1, (n if ate is None else min(n, ate)) + 1):
        H = G.subgraph(restantes)
        livres = (restantes - protegidos) or restantes
        if estrategia == "grau_adaptativo":
            no = [m for m in _ranking_ingenuo(H, "grau") if m in livres][0]
        elif estrategia != criterio:
            if not fila:
                fila = [m for m in _ranking_ingenuo(H, criterio) if m in livres][:passo_betweenness]
            no = fila.pop(0)
        else:
            no = ordem[k - 1]
        restantes.discard(no)
        linhas.append({"n_removidos": k, "fracao_removida": k / n,
                       **_medidas_ingenuas(G, restantes, n, nutrientes, k in marcos)})
    return pd.DataFrame(linhas, columns=list(robustez.COLUNAS))


@pytest.mark.parametrize("estrategia", ["aleatoria", "grau", "grau_adaptativo",
                                        "betweenness", "betweenness_adaptativo"])
def test_igual_a_implementacao_ingenua(estrategia):
    for semente in range(15):
        G = grafo_sorteado(100 + semente, n=30, p=0.07)
        nutrientes = ["n00", "n05"]
        rng = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        rng_ingenuo = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        curva = robustez.simular_remocao(G, estrategia, rng=rng, nutrientes=nutrientes,
                                         passo_eficiencia=0.1, passo_betweenness=3)
        esperada = simulacao_ingenua(G, estrategia, nutrientes, 0.1, 3, rng_ingenuo)
        pd.testing.assert_frame_equal(curva, esperada, check_dtype=False, rtol=1e-12,
                                      obj=f"{estrategia}, semente={semente}")


def test_betweenness_adaptativo_nos_extremos():
    """Com passo = N, nunca recalcula (= estática); com passo = 1, recalcula a cada remoção."""
    G = grafo_sorteado(11, n=30, p=0.07)
    n = G.number_of_nodes()
    estatica = robustez.simular_remocao(G, "betweenness", medir_eficiencia=False)
    sem_recalcular = robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=n,
                                              medir_eficiencia=False)
    pd.testing.assert_frame_equal(estatica, sem_recalcular)
    a_cada_uma = robustez.simular_remocao(G, "betweenness_adaptativo", passo_betweenness=1,
                                          medir_eficiencia=False)
    esperada = simulacao_ingenua(G, "betweenness_adaptativo", [], 1.0, 1)
    pd.testing.assert_frame_equal(a_cada_uma[["n_removidos", "maior_fraco", "maior_forte"]],
                                  esperada[["n_removidos", "maior_fraco", "maior_forte"]], check_dtype=False)


# ---------------------------------------------------------------------------
# 5. PageRank e proteção das fontes
# ---------------------------------------------------------------------------
NOVAS = ("pagerank", "pagerank_adaptativo")


def estrela_de_mao_dupla(folhas=("f1", "f2", "f3", "f4"), inverter: bool = False) -> nx.DiGraph:
    """c <-> cada folha: c é o mais central e as folhas empatam."""
    G = nx.DiGraph()
    nos = ["c", *folhas]
    G.add_nodes_from(reversed(nos) if inverter else nos)
    for f in folhas:
        G.add_edges_from([("c", f), (f, "c")])
    return G


@pytest.mark.parametrize("estrategia", ["pagerank", "betweenness"])
def test_centralidades_desempatam_pelo_codigo(estrategia):
    """As folhas empatam; a ordem é a do código e não depende da ordem em que os nós entraram."""
    esperada = ["c", "f1", "f2", "f3", "f4"]
    assert robustez.ordem_inicial(estrela_de_mao_dupla(), estrategia) == esperada
    assert robustez.ordem_inicial(estrela_de_mao_dupla(inverter=True), estrategia) == esperada
    assert robustez.ordem_de_remocao(estrela_de_mao_dupla(inverter=True), f"{estrategia}_adaptativo") == esperada


def test_harmonica_nao_e_estrategia_de_remocao():
    """A harmônica saiu da remoção de nós (continua só como centralidade nos Exps 2 e 3)."""
    assert "harmonica" not in robustez.ESTATICAS and "harmonica_adaptativo" not in robustez.ADAPTATIVAS
    assert not any("harmonica" in e for e in (*config.ESTRATEGIAS, *config.ESTRATEGIAS_CLASSICOS))
    for estrategia in ("harmonica", "harmonica_adaptativo"):
        with pytest.raises(ValueError):
            robustez.simular_remocao(caminho_dirigido(), estrategia)


@pytest.mark.parametrize("estrategia", NOVAS)
def test_novas_centralidades_iguais_a_implementacao_ingenua(estrategia):
    for semente in range(8):
        G = grafo_sorteado(300 + semente, n=30, p=0.07)
        nutrientes = ["n00", "n05"]
        curva = robustez.simular_remocao(G, estrategia, nutrientes=nutrientes,
                                         passo_eficiencia=0.1, passo_betweenness=3)
        esperada = simulacao_ingenua(G, estrategia, nutrientes, 0.1, 3)
        pd.testing.assert_frame_equal(curva, esperada, check_dtype=False, rtol=1e-12,
                                      obj=f"{estrategia}, semente={semente}")


@pytest.mark.parametrize("estrategia", [*robustez.ESTATICAS, *robustez.ADAPTATIVAS])
def test_fontes_protegidas_ficam_no_fim_em_todas_as_estrategias(estrategia):
    """Os nutrientes (aqui o hub e um nó do meio) só saem depois de todos os outros."""
    for semente in range(5):
        G = grafo_sorteado(500 + semente, n=25, p=0.12)
        hub = max(sorted(G), key=G.degree)
        protegidos = [hub, "n07", "nao_esta_no_grafo"]
        rng = np.random.default_rng([semente, 0]) if estrategia == "aleatoria" else None
        ordem = robustez.ordem_de_remocao(G, estrategia, rng, protegidos=protegidos, passo_betweenness=4)
        assert sorted(ordem) == sorted(G)                       # todos os nós, uma vez cada
        assert set(ordem[-2:]) == {hub, "n07"}
        if estrategia in ("grau", "grau_adaptativo"):
            assert robustez.ordem_de_remocao(G, estrategia)[0] == hub   # sem proteção, o hub sai primeiro


def test_protecao_nao_muda_a_ordem_dos_outros_nas_estaticas():
    G = grafo_sorteado(9, n=25, p=0.12)
    for estrategia in robustez.ESTATICAS:
        rng = (lambda: np.random.default_rng([3, 0])) if estrategia == "aleatoria" else (lambda: None)
        livre = robustez.ordem_de_remocao(G, estrategia, rng())
        protegida = robustez.ordem_de_remocao(G, estrategia, rng(), protegidos=["n03"])
        assert protegida == [no for no in livre if no != "n03"] + ["n03"]


def test_curva_com_a_fonte_protegida_feita_a_mao():
    """Com h protegido, a ordem por grau vira x1, x2, x3, x4, x5, h.

    Sem x1: fraco {h,x2,x3}=3, forte {h,x2}=2, alcance {h,x2,x3}=3. Sem x2: o fraco
    empata em 2 ({h,x3} e {x4,x5}), o forte cai a 1, o alcance a 2. Sem x3: alcance
    só {h}. O alcance só zera quando a própria fonte sai, no último passo.
    """
    curva = robustez.simular_remocao(estrela_com_par_solto(), "grau", nutrientes=["h"], medir_eficiencia=False)
    assert np.allclose(curva["maior_fraco"] * 6, [4, 3, 2, 2, 1, 1, 0])
    assert np.allclose(curva["maior_forte"] * 6, [3, 2, 1, 1, 1, 1, 0])
    assert np.allclose(curva["alcance"] * 6, [4, 3, 2, 1, 1, 1, 0])


def test_proximo_adaptativo_com_protegidos():
    G = estrela_com_par_solto()
    assert robustez.proximo_adaptativo(G, "grau_adaptativo", protegidos={"h"}) == "x1"
    assert robustez.proximo_adaptativo(G, "pagerank_adaptativo", protegidos={"h"}) != "h"
    so_h = nx.DiGraph()
    so_h.add_node("h")
    assert robustez.proximo_adaptativo(so_h, "grau_adaptativo", protegidos={"h"}) == "h"  # não sobrou outro


def test_r_normalizado_feito_a_mao():
    """Estrela com par solto: S0_fraco = 4/6, R_fraco = 1/4 -> 3/8; S0_forte = 3/6, R_forte = 5/36 -> 5/18."""
    _, indices = robustez.comparar_estrategias(estrela_com_par_solto(), ["grau"], medir_eficiencia=False)
    linha = indices.iloc[0]
    assert math.isclose(linha["S0_fraco"], 4 / 6) and math.isclose(linha["S0_forte"], 3 / 6)
    assert math.isclose(linha["R_fraco_norm"], 3 / 8) and math.isclose(linha["R_forte_norm"], 5 / 18)


def test_r_normalizado_da_aleatoria():
    G = grafo_sorteado(6)
    _, indices = robustez.comparar_estrategias(G, ["aleatoria"], n_rep=4, semente=3, medir_eficiencia=False)
    linha = indices.iloc[0]
    s0 = metricas.fracao_maior_componente(G, G.number_of_nodes())
    assert math.isclose(linha["S0_fraco"], s0)
    assert math.isclose(linha["R_fraco_norm"], linha["R_fraco"] / s0)
