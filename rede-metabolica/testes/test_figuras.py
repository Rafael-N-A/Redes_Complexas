"""Testes das figuras (figuras.py) e da CCDF com ajustes (metricas.ccdf_com_ajustes).

As figuras são conferidas pelo que desenham (eixos, séries, rótulos, cores),
com tabelas pequenas montadas à mão no formato que os scripts gravam.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experimentos.biblioteca import figuras, metricas


@pytest.fixture(autouse=True)
def fechar_figuras():
    yield
    plt.close("all")


def curvas_exemplo(estrategias=("aleatoria", "grau", "grau_adaptativo")) -> pd.DataFrame:
    """Três estratégias em 11 passos; a aleatória com desvio padrão; eficiência só a cada 5 passos."""
    linhas = []
    for e in estrategias:
        for k in range(11):
            f = k / 10
            queda = 1 - f if e == "aleatoria" else max(0.0, 1 - 5 * f)
            linhas.append({"estrategia": e, "n_removidos": k, "fracao_removida": f,
                           "maior_fraco": 0.8 * queda, "maior_forte": 0.4 * queda,
                           "alcance": 0.6 * queda, "eficiencia": 0.04 * queda if k % 5 == 0 else np.nan,
                           "maior_fraco_dp": 0.05 if e == "aleatoria" else np.nan})
    return pd.DataFrame(linhas)


def linhas_com_rotulo(ax) -> dict:
    return {linha.get_label(): linha for linha in ax.get_lines() if not linha.get_label().startswith("_")}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def test_numeros_em_portugues():
    assert figuras.num(-2.5196) == "−2,52"
    assert figuras.num(1109, 0) == "1.109"
    assert figuras.num(0.0117, 3) == "0,012"
    assert figuras._virgula(0.25) == "0,25" and figuras._virgula(1.0) == "1"


def test_estilo_e_salvar(tmp_path):
    figuras.estilo()
    assert plt.rcParams["axes.spines.top"] is False and plt.rcParams["axes.spines.right"] is False
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    destino = figuras.salvar(fig, tmp_path / "sub" / "teste.png")
    assert destino.exists() and destino.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)  # salvar fecha a figura


def test_cor_segue_a_entidade_e_nao_a_ordem():
    """A estratégia aleatória é azul e as adaptativas tracejadas, em qualquer ordem."""
    for curvas in (curvas_exemplo(), curvas_exemplo(("grau_adaptativo", "grau", "aleatoria"))):
        ax = figuras.curvas_robustez(curvas, zoom=None).axes[0]
        linhas = linhas_com_rotulo(ax)
        assert linhas["Aleatória"].get_color() == figuras.CORES[0]
        assert linhas["Grau"].get_linestyle() == "-"
        assert linhas["Grau (adaptativo)"].get_linestyle() == "--"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Aleatória", "Grau", "Grau (adaptativo)"]


# ---------------------------------------------------------------------------
# Exp 0
# ---------------------------------------------------------------------------
def test_ccdf_com_ajustes():
    # Excesso de graus 1 e 2 sobre uma cauda de Zipf: o xmin sai acima de 1 (6 com esta semente),
    # então a fração da cauda que escala os ajustes é menor que 1.
    rng = np.random.default_rng(2)
    graus = np.concatenate([np.full(150, 1), np.full(120, 2), rng.zipf(2.3, 300) + 2, np.zeros(20, dtype=int)])
    graus = graus[graus < 300]
    tabela = metricas.ccdf_com_ajustes(graus)
    positivos = graus[graus > 0]
    assert list(tabela["grau"]) == list(range(1, positivos.max() + 1))
    assert tabela["n_nos"].sum() == len(positivos)                     # o grau 0 fica de fora
    assert tabela["ccdf_empirica"].iloc[0] == 1.0
    assert (np.diff(tabela["ccdf_empirica"]) <= 1e-12).all()           # não cresce
    for k in (2, 5, 10):
        assert math.isclose(tabela.loc[tabela["grau"] == k, "ccdf_empirica"].item(), (positivos >= k).mean())
    ajuste = metricas.ajuste_lei_potencia(graus)
    assert ajuste["xmin"] > 1
    antes = tabela["grau"] < ajuste["xmin"]
    assert tabela.loc[antes, ["ccdf_lei_potencia", "ccdf_lognormal"]].isna().all().all()
    no_xmin = tabela[tabela["grau"] == ajuste["xmin"]].iloc[0]
    # Os ajustes saem da mesma fração da cauda que a empírica em xmin.
    assert math.isclose(no_xmin["ccdf_lei_potencia"], no_xmin["ccdf_empirica"])
    assert math.isclose(no_xmin["ccdf_lognormal"], no_xmin["ccdf_empirica"])


def test_ccdf_com_poucos_dados_nao_ajusta():
    tabela = metricas.ccdf_com_ajustes([1, 2, 2, 3])
    assert list(tabela["n_nos"]) == [1, 2, 1]
    assert tabela[["ccdf_lei_potencia", "ccdf_lognormal"]].isna().all().all()


def test_distribuicao_graus():
    dist = pd.DataFrame({"tipo": ["entrada"] * 3 + ["saida"] * 3 + ["total"] * 3,
                         "grau": [0, 1, 2] * 3, "n_nos": [5, 3, 2] * 3})
    fig = figuras.distribuicao_graus(dist)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    entrada = linhas_com_rotulo(ax)["Entrada"]
    assert list(entrada.get_xdata()) == [1, 2]                         # grau 0 não vai ao log-log
    assert np.allclose(entrada.get_ydata(), [0.3, 0.2])                 # P(k) com o total de nós

    ccdf = metricas.ccdf_com_ajustes(np.random.default_rng(2).zipf(2.5, 300))
    ajuste = {"alpha": 2.5, "xmin": 2.0, "R_vs_lognormal": -1.2, "p": 0.04}
    fig = figuras.distribuicao_graus(dist, ccdf, ajuste)
    assert len(fig.axes) == 2
    rotulos = list(linhas_com_rotulo(fig.axes[1]))
    assert rotulos == ["Dados (grau total)", "Lei de potência (α = 2,50)", "Log-normal"]
    assert "R = −1,20, p = 0,040" in fig.axes[1].texts[0].get_text()


# ---------------------------------------------------------------------------
# Exp 1 e 4
# ---------------------------------------------------------------------------
def test_curvas_robustez_com_detalhe_e_faixa():
    fig = figuras.curvas_robustez(curvas_exemplo(), "maior_fraco", zoom=0.1)
    principal, detalhe = fig.axes
    assert detalhe.get_xlim() == (0, 0.1)
    assert len(principal.collections) == 1      # só a aleatória tem a faixa de ±1 dp
    assert principal.get_xlabel() == figuras.ROTULO_FRACAO_REMOVIDA


def test_eficiencia_so_nos_pontos_medidos():
    ax = figuras.curvas_robustez(curvas_exemplo(), "eficiencia", zoom=None).axes[0]
    aleatoria = linhas_com_rotulo(ax)["Aleatória"]
    assert list(aleatoria.get_xdata()) == [0.0, 0.5, 1.0]
    assert aleatoria.get_marker() == "o"


def test_painel_robustez():
    fig = figuras.painel_robustez(curvas_exemplo())
    assert len(fig.axes) == 4
    titulos = [ax.get_title(loc="left") for ax in fig.axes]  # títulos alinhados à esquerda (estilo)
    assert titulos == ["(a) Maior componente fraco", "(b) Maior componente forte",
                       "(c) Eficiência global", "(d) Alcance topológico"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Aleatória", "Grau", "Grau (adaptativo)"]


def test_curvas_por_rede_no_exp4():
    curvas = curvas_exemplo(("aleatoria",)).assign(rede="eco")
    curvas = pd.concat([curvas, curvas.assign(rede="rewiring")])
    ax = figuras.curvas_robustez(curvas, grupo="rede", zoom=None).axes[0]
    linhas = linhas_com_rotulo(ax)
    assert linhas["E. coli"].get_color() == figuras.CORES[0]
    assert "Rewiring (mesmos graus)" in linhas


def test_indices_por_rede():
    indices = pd.DataFrame([
        {"rede": "eco", "estrategia": "aleatoria", "R_fraco": 0.21},
        {"rede": "eco", "estrategia": "grau", "R_fraco": 0.05},
        *[{"rede": "erdos_renyi", "estrategia": e, "R_fraco": v, "indice": i}
          for i, (va, vg) in enumerate([(0.30, 0.20), (0.32, 0.22)])
          for e, v in (("aleatoria", va), ("grau", vg))],
    ])
    fig = figuras.indices_por_rede(indices)
    ax = fig.axes[0]
    alturas = sorted(round(b.get_height(), 3) for b in ax.patches)
    assert alturas == [0.05, 0.21, 0.21, 0.31]                         # nulos: média das redes sorteadas
    assert [t.get_text() for t in ax.get_xticklabels()] == ["E. coli", "Erdős–Rényi"]
    assert len(ax.containers) >= 3                                      # barras + traços de erro dos nulos


def indices_exemplo() -> pd.DataFrame:
    linhas = [{"rede": "eco", "estrategia": e, "R_fraco_norm": v, "metade_fraco": v / 2}
              for e, v in (("aleatoria", 0.26), ("grau", 0.06))]
    for tipo in ("rewiring", "rewiring_reciproco"):
        for i, desvio in enumerate((-0.01, 0.01)):
            linhas += [{"rede": tipo, "indice": i, "estrategia": e, "R_fraco_norm": v + desvio,
                        "metade_fraco": (v + desvio) / 2} for e, v in (("aleatoria", 0.3), ("grau", 0.1))]
    return pd.DataFrame(linhas)


def test_painel_indices_redes():
    fig = figuras.painel_indices_redes(indices_exemplo())
    eixos = [ax for ax in fig.axes if ax.patches]
    assert len(eixos) == 2
    assert [t.get_text() for t in eixos[1].get_xticklabels()] == [
        "E. coli", "Rewiring\n(graus)", "Rewiring\n(graus e\nreciprocidade)"]
    assert "R normalizado" in eixos[0].get_ylabel() and "metade" in eixos[1].get_ylabel()
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Aleatória", "Grau"]   # legenda uma vez só
    alturas = sorted(round(b.get_height(), 3) for b in eixos[0].patches)
    assert alturas == [0.06, 0.1, 0.1, 0.26, 0.3, 0.3]                     # nulos: média das redes


def test_painel_curvas_redes():
    base = curvas_exemplo(("aleatoria", "grau_adaptativo"))
    base["maior_fraco_rel"] = base["maior_fraco"] / 0.8
    base["maior_forte_rel"] = base["maior_forte"] / 0.4
    curvas = pd.concat([base.assign(rede=r) for r in ("eco", "rewiring", "rewiring_reciproco", "erdos_renyi")])
    fig = figuras.painel_curvas_redes(curvas)
    assert len(fig.axes) == 4
    assert [ax.get_title(loc="left") for ax in fig.axes] == [
        "(a) Aleatória", "(b) Grau (adaptativo)", "(c) Aleatória", "(d) Grau (adaptativo)"]
    assert fig.axes[1].get_xlim() == (0, 0.4) and fig.axes[0].get_xlim() == (0, 1)
    assert "relativo à rede intacta" in fig.axes[0].get_ylabel()
    linhas = linhas_com_rotulo(fig.axes[0])
    assert linhas["E. coli"].get_linestyle() == "-"
    assert linhas["Rewiring (mesmos graus)"].get_linestyle() == "--"            # nulos tracejados
    assert linhas["Rewiring (graus e reciprocidade)"].get_color() == figuras.CORES[5]
    assert [t.get_text() for t in fig.legends[0].get_texts()][0] == "E. coli"


def test_faixa_nao_passa_de_um():
    curvas = curvas_exemplo(("aleatoria",))
    curvas["maior_fraco_dp"] = 0.5
    ax = figuras.curvas_robustez(curvas, zoom=None).axes[0]
    faixa = ax.collections[0].get_paths()[0].vertices
    assert faixa[:, 1].max() <= 1 and faixa[:, 1].min() >= 0


def test_ccdf_por_rede():
    ccdf = pd.DataFrame([{"rede": r, "grau": k, "n_nos": n, "ccdf_empirica": c}
                         for r in ("mge", "eco")
                         for k, n, c in ((1, 5, 1.0), (2, 3, 0.5), (8, 0, 0.2), (9, 2, 0.2))])
    ax = figuras.ccdf_por_rede(ccdf).axes[0]
    linhas = linhas_com_rotulo(ax)
    assert list(linhas) == ["E. coli", "M. genitalium"]                      # na ordem de REDES
    assert list(linhas["E. coli"].get_xdata()) == [1, 2, 9]                  # só os graus observados
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"


# ---------------------------------------------------------------------------
# Exp 2 e 3
# ---------------------------------------------------------------------------
def test_curvas_roc():
    rocs = {"grau": pd.DataFrame({"fpr": [0, 0.5, 1], "tpr": [0, 0.6, 1]}),
            "dano": pd.DataFrame({"fpr": [0, 0.2, 1], "tpr": [0, 0.8, 1]})}
    fig = figuras.curvas_roc(rocs, {"dano": 0.8, "grau": 0.55})
    rotulos = list(linhas_com_rotulo(fig.axes[0]))
    assert rotulos == ["Ao acaso (AUC = 0,5)", "Dano na cascata (AUC = 0,80)", "Grau (AUC = 0,55)"]


def test_avalanche_vs_grau():
    tabela = pd.DataFrame({"grupo": ["substancia", "substancia", "hub", "baixo_grau"],
                           "metabolito": ["C00022", "C00024", "C00085", "C99999"],
                           "nome": ["Piruvato", "Acetil-CoA", "F6P", "X"],
                           "grau": [31, 26, 13, 1], "avalanche": [63, 59, 10, 0]})
    ax = figuras.avalanche_vs_grau(tabela).axes[0]
    assert sorted(t.get_text() for t in ax.texts) == ["Acetil-CoA", "Piruvato"]  # só as escolhidas têm nome
    assert len(ax.collections) == 3
    caixas = [texto.get_window_extent() for texto in ax.texts]
    assert not caixas[0].overlaps(caixas[1])                 # os nomes não se sobrepõem

    todos = pd.DataFrame({"grau": [1, 2, 31, 26, 13], "avalanche": [0, 1, 63, 59, 10]})
    ax = figuras.avalanche_vs_grau(tabela, todos).axes[0]
    fundo = ax.collections[0]
    assert len(ax.collections) == 4 and fundo.get_label() == "Todos os metabólitos"
    assert fundo.get_zorder() < ax.collections[1].get_zorder()   # o contexto fica atrás dos grupos


def test_distribuicao_avalanches():
    ax = figuras.distribuicao_avalanches(pd.Series([0, 0, 0, 1, 1, 2, 5, 63])).axes[0]
    assert ax.get_xscale() == "log"
    pontos = ax.get_lines()[0]
    assert list(pontos.get_xdata()) == [1, 2, 5, 63]
    assert np.allclose(pontos.get_ydata(), [1.0, 0.6, 0.4, 0.2])
    assert ax.get_title(loc="left") == "3 de 8 metabólitos (38%) sem avalanche"


def test_distribuicao_avalanches_com_ajustes():
    tamanhos = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.random.default_rng(3).zipf(2.2, 200)]))
    ccdf = metricas.ccdf_com_ajustes(tamanhos)
    ajuste = metricas.ajuste_lei_potencia(tamanhos)
    ax = figuras.distribuicao_avalanches(tamanhos, ccdf, ajuste).axes[0]
    rotulos = list(linhas_com_rotulo(ax))
    assert rotulos[0] == "Dados (avalanches ≥ 1)" and rotulos[2] == "Log-normal"
    assert rotulos[1].startswith("Lei de potência (α = ")
    assert ax.texts[0].get_text().startswith(f"xmin = {ajuste['xmin']:g}")


def test_ccdf_sem_ajuste_so_desenha_os_dados():
    """Redes pequenas (mge) podem ter menos de 10 avalanches: sem ajuste, só a empírica."""
    tamanhos = pd.Series([0, 1, 1, 3])
    ax = figuras.distribuicao_avalanches(tamanhos, metricas.ccdf_com_ajustes(tamanhos),
                                         metricas.ajuste_lei_potencia(tamanhos)).axes[0]
    assert list(linhas_com_rotulo(ax)) == ["Dados (avalanches ≥ 1)"] and not ax.texts


# ---------------------------------------------------------------------------
# Figuras novas (P1, P2 e P3)
# ---------------------------------------------------------------------------
def test_estrategias_novas_tem_cor_e_traco():
    """No painel do Exp 1 (aleatória + as quatro adaptativas), cores distintas e traços distintos."""
    painel = ("aleatoria", "grau_adaptativo", "betweenness_adaptativo", "harmonica_adaptativo", "pagerank_adaptativo")
    cores = [figuras.ESTRATEGIAS[e][1] for e in painel]
    tracos = [str(figuras.ESTRATEGIAS[e][2]) for e in painel]
    assert len(set(cores)) == len(painel) and len(set(tracos)) == len(painel)
    for centralidade in ("harmonica", "pagerank"):   # mesma cor nas duas versões, traço diferente
        estatica, adaptativa = figuras.ESTRATEGIAS[centralidade], figuras.ESTRATEGIAS[f"{centralidade}_adaptativo"]
        assert estatica[1] == adaptativa[1] and estatica[2] == "-" and adaptativa[2] != "-"


def test_estatica_vs_adaptativa():
    curvas = curvas_exemplo(("aleatoria", "grau", "grau_adaptativo", "harmonica", "harmonica_adaptativo"))
    fig = figuras.estatica_vs_adaptativa(curvas, ["grau", "harmonica"])
    visiveis = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in visiveis] == ["(a) Grau", "(b) Harmônica"]
    assert list(linhas_com_rotulo(visiveis[1])) == ["Aleatória", "Harmônica", "Harmônica (adaptativo)"]


def genes_exemplo() -> pd.DataFrame:
    return pd.DataFrame({"gene": ["g1", "g2", "g3", "g4"], "simbolo": ["aroA", "aroB", "fabG", "xyz"],
                         "essencial": [False, False, True, True], "dano": [1, 7, 50, 0],
                         "dano_forte": [20, 20, 20, 0], "dano_eficiencia": [0.03, 0.04, 0.07, 0.0]})


def test_dispersao_dano_genes():
    correlacao = pd.DataFrame({"subconjunto": "perde_reacao", "medida_x": "dano",
                               "medida_y": ["dano_forte", "dano_eficiencia"], "spearman": [0.23, 0.47]})
    fig = figuras.dispersao_dano_genes(genes_exemplo(), {"aroA–aroB": ["aroA", "aroB"], "fabG": ["fabG"]},
                                       correlacao)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert sorted(t.get_text() for t in ax.texts) == ["aroA–aroB", "fabG"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Não essencial", "Essencial (Keio)"]
    assert "0,23" in fig.texts[0].get_text() and "0,47" in fig.texts[0].get_text()


def test_barras_auc():
    avaliacao = pd.DataFrame([{"subconjunto": s, "ranking": r, "auc": a, "auc_ic_inf": a - 0.05, "auc_ic_sup": a + 0.05}
                              for s, r, a in (("todos", "dano", 0.76), ("todos", "grau", 0.42),
                                              ("perde_reacao", "dano", 0.77), ("perde_reacao", "grau", 0.41))])
    ax = figuras.barras_auc(avaliacao, ["dano", "grau"]).axes[0]
    assert sorted(round(b.get_width(), 2) for b in ax.patches) == [0.41, 0.42, 0.76, 0.77]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Dano na cascata", "Grau"]   # de cima para baixo
    assert any(np.allclose(l.get_xdata(), 0.5) for l in ax.get_lines())                 # a linha do acaso


def test_dano_vs_avalanche():
    dano = pd.DataFrame({"metabolito": ["C1", "C2", "C3", "C4"], "grau": [40, 6, 5, 1],
                         "dano_forte": [31, 36, 7, 0], "avalanche": [59.0, 0.0, 54.0, np.nan]})
    exemplos = pd.DataFrame({"metabolito": ["C1", "C2", "C4"], "nome_curto": ["Acetil-CoA", "FAICAR", "fora"]})
    correlacoes = pd.DataFrame({"medida": ["dano_forte", "dano_eficiencia"], "outra": "avalanche",
                                "spearman": [0.2, 0.42]})
    fig = figuras.dano_vs_avalanche(dano, exemplos, correlacoes)
    ax = fig.axes[0]
    assert len(ax.collections[0].get_offsets()) == 3                        # C4 não é produzido: fica de fora
    assert sorted(t.get_text() for t in ax.texts) == ["Acetil-CoA", "FAICAR"]
    assert len(fig.axes) == 2                                               # a barra de cores do grau


def test_figuras_dos_modelos_classicos():
    indices = pd.DataFrame([{"modelo": m, "indice": i, "estrategia": e, "R_norm": v, "metade": v}
                            for m, reps in (("eco", [None]), ("erdos_renyi", [0, 1]), ("ba_misto", [0, 1]))
                            for i in reps for e, v in (("aleatoria", 0.3), ("grau", 0.1), ("harmonica", 0.2))])
    fig = figuras.painel_indices_redes(indices, ("R_norm", "metade"), grupo="modelo")
    eixos = [ax for ax in fig.axes if ax.patches]
    assert [t.get_text() for t in eixos[1].get_xticklabels()] == ["E. coli", "Erdős–\nRényi", "BA misto\n(m = 1 ou 2)"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Aleatória", "Grau", "Harmônica"]
    ccdf = pd.DataFrame([{"modelo": m, "grau": k, "n_nos": 1, "ccdf_empirica": c}
                         for m in ("eco", "configuracao", "erdos_renyi")
                         for k, c in ((0, 1.0), (1, 0.9), (2, 0.5))])
    ax = figuras.ccdf_por_rede(ccdf, grupo="modelo").axes[0]
    linhas = linhas_com_rotulo(ax)
    assert list(linhas) == ["E. coli (= configuração)", "Erdős–Rényi"]      # a configuração coincide com a E. coli
    assert list(linhas["Erdős–Rényi"].get_xdata()) == [1, 2]                # grau 0 não vai ao log-log
