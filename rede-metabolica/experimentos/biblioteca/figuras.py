"""Figuras do artigo, com o mesmo estilo em todas.

Cada função recebe as tabelas que os scripts gravaram em resultados/tabelas/
e devolve uma figura do matplotlib. Quem grava é o script, com
config.arquivo_figura(...) e salvar(); a biblioteca não decide nada sobre arquivos.

Convenções (iguais em todas as figuras):
  - largura de uma coluna do modelo SBC (~16 cm); texto em português, com
    vírgula decimal nos eixos lineares;
  - cores categóricas numa ordem fixa, testada para daltonismo (CORES); cada
    entidade tem sempre a mesma cor: a estratégia aleatória é sempre azul, a
    E. coli é sempre azul no Exp 4;
  - estratégias adaptativas em linha tracejada, além da cor, para a figura
    continuar legível impressa em preto e branco;
  - grade e eixos em cinza claro, sem bordas de cima e da direita; legenda
    sempre que há mais de uma série.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Paleta e rótulos
# ---------------------------------------------------------------------------
# Ordem categórica validada (contraste e separação para daltonismo entre
# vizinhos; as três primeiras também entre todos os pares).
CORES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TINTA = "#0b0b0b"          # texto principal
TINTA_SECUNDARIA = "#52514e"
CINZA = "#898781"          # referências (diagonal da ROC, linhas de apoio)
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

LARGURA = 6.3              # polegadas (~16 cm)

ESTRATEGIAS = {  # nome -> (rótulo, cor, tracejada?)
    "aleatoria": ("Aleatória", CORES[0], False),
    "grau": ("Grau", CORES[1], False),
    "grau_adaptativo": ("Grau (adaptativo)", CORES[2], True),
    "betweenness": ("Betweenness", CORES[3], False),
    "betweenness_adaptativo": ("Betweenness (adaptativo)", CORES[4], True),
}

REDES = {  # Exp 4: nome -> (rótulo, cor)
    "eco": ("E. coli", CORES[0]),
    "bsu": ("B. subtilis", CORES[1]),
    "mge": ("M. genitalium", CORES[2]),
    "rewiring": ("Rewiring (mesmos graus)", CORES[3]),
    "erdos_renyi": ("Erdős–Rényi", CORES[4]),
}

METRICAS_ROBUSTEZ = {  # coluna das curvas -> rótulo do eixo y
    "maior_fraco": "Maior componente fraco\n(fração dos nós)",
    "maior_forte": "Maior componente forte\n(fração dos nós)",
    "eficiencia": "Eficiência global",
    "alcance": "Alcance topológico a partir\ndo meio mínimo (fração)",
}

TITULOS_ROBUSTEZ = {  # títulos curtos dos painéis
    "maior_fraco": "Maior componente fraco",
    "maior_forte": "Maior componente forte",
    "eficiencia": "Eficiência global",
    "alcance": "Alcance topológico",
}

GRUPOS_EXP3 = {  # coluna "grupo" da tabela exp3_substancias -> (rótulo, cor, marcador)
    "substancia": ("Substâncias escolhidas", CORES[0], "o"),
    "hub": ("Controle: maior grau", CORES[1], "s"),
    "baixo_grau": ("Controle: baixo grau", CORES[2], "^"),
}

RANKINGS_EXP2 = {  # Exp 2: nome do ranking -> rótulo
    "dano": "Dano simulado (cascata)",
    "grau": "Grau",
    "betweenness": "Betweenness",
}

ROTULO_FRACAO_REMOVIDA = "Fração de nós removidos"


def _virgula(valor: float, _posicao=None) -> str:
    """Número curto com vírgula decimal: 0.25 -> '0,25', 1.0 -> '1'."""
    return f"{valor:g}".replace(".", ",")


def num(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal, ponto de milhar e sinal de menos (−), para legendas e anotações."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "\u2212")


def _eixos_lineares_com_virgula(ax: plt.Axes, x: bool = True, y: bool = True) -> None:
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_virgula))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_virgula))


def _cor_e_estilo(grupo: str, valor: str, posicao: int) -> tuple[str, str, str]:
    """(rótulo, cor, estilo de linha) de uma série; cai na ordem de CORES se o valor for novo."""
    if grupo == "estrategia" and valor in ESTRATEGIAS:
        rotulo, cor, tracejada = ESTRATEGIAS[valor]
        return rotulo, cor, "--" if tracejada else "-"
    if grupo == "rede" and valor in REDES:
        rotulo, cor = REDES[valor]
        return rotulo, cor, "-"
    return str(valor), CORES[posicao % len(CORES)], "-"


def _ordenar(valores, referencia: dict) -> list:
    """Os valores na ordem do dicionário de referência; os desconhecidos no fim, na ordem em que vieram."""
    vistos = list(dict.fromkeys(valores))
    return [v for v in referencia if v in vistos] + [v for v in vistos if v not in referencia]


# ---------------------------------------------------------------------------
# Estilo e gravação
# ---------------------------------------------------------------------------
def estilo() -> None:
    """Ajusta o rcParams uma vez: fonte, tamanho, sem bordas de cima e da direita."""
    plt.rcParams.update({
        "figure.figsize": (LARGURA, 3.8),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": EIXO,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=list(CORES)),
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.markersize": 4.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "text.color": TINTA,
    })


def salvar(fig: plt.Figure, destino: Path) -> Path:
    """Grava fig em destino (dpi = config.DPI_FIGURA, bbox justo), fecha a figura
    e devolve o caminho. Cria a pasta se faltar."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=config.DPI_FIGURA, bbox_inches="tight")
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# Exp 0
# ---------------------------------------------------------------------------
_MARCADORES_GRAU = {"entrada": ("Entrada", CORES[0], "o"),
                    "saida": ("Saída", CORES[1], "s"),
                    "total": ("Total", CORES[2], "^")}


def distribuicao_graus(dist: pd.DataFrame, ccdf: pd.DataFrame | None = None,
                       ajuste: dict | None = None) -> plt.Figure:
    """Exp 0. Entrada, saída e total em log-log (tabela de metricas.distribuicao_graus).

    Com ccdf (tabela de metricas.ccdf_com_ajustes), um segundo painel com a CCDF
    do grau total e os ajustes de lei de potência e log-normal sobrepostos a
    partir de xmin, como em Clauset et al. (2009). ajuste (uma linha de
    metricas.ajuste_lei_potencia) acrescenta alpha, R e p na anotação.
    Graus 0 não aparecem no log-log.
    """
    estilo()
    paineis = 2 if ccdf is not None else 1
    fig, eixos = plt.subplots(1, paineis, figsize=(LARGURA, 2.9), squeeze=False)
    ax = eixos[0, 0]
    for tipo in _ordenar(dist["tipo"], _MARCADORES_GRAU):
        parte = dist[(dist["tipo"] == tipo) & (dist["grau"] > 0)]
        if parte.empty:
            continue
        rotulo, cor, marcador = _MARCADORES_GRAU.get(tipo, (tipo, CINZA, "o"))
        p_k = parte["n_nos"] / dist.loc[dist["tipo"] == tipo, "n_nos"].sum()
        ax.plot(parte["grau"], p_k, linestyle="none", marker=marcador, color=cor,
                markeredgecolor="white", markeredgewidth=0.6, label=rotulo, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grau k")
    ax.set_ylabel("P(k)")
    ax.set_title("(a) Distribuição de graus" if ccdf is not None else "Distribuição de graus")
    ax.legend(loc="lower left")

    if ccdf is not None:
        ax = eixos[0, 1]
        _desenhar_ccdf(ax, ccdf, ajuste, "Dados (grau total)")
        ax.set_xlabel("Grau k")
        ax.set_ylabel("P(K ≥ k)")
        ax.set_title("(b) CCDF e ajustes")
    fig.tight_layout()
    return fig


def _desenhar_ccdf(ax: plt.Axes, ccdf: pd.DataFrame, ajuste: dict | None, rotulo_dados: str) -> None:
    """CCDF empírica (só nos valores observados) e os ajustes de metricas.ccdf_com_ajustes, em log-log.

    Com ajuste (uma linha de metricas.ajuste_lei_potencia), põe alpha na legenda
    e xmin, R e p no canto de cima.
    """
    empirica = ccdf[(ccdf["n_nos"] > 0) & (ccdf["ccdf_empirica"] > 0)]
    ax.plot(empirica["grau"], empirica["ccdf_empirica"], linestyle="none", marker="o",
            color=CORES[0], markeredgecolor="white", markeredgewidth=0.6, label=rotulo_dados)
    lp = ccdf.dropna(subset=["ccdf_lei_potencia"])
    ln = ccdf.dropna(subset=["ccdf_lognormal"])
    rotulo_lp = "Lei de potência"
    if ajuste is not None and np.isfinite(ajuste.get("alpha", np.nan)):
        rotulo_lp += f" (α = {num(ajuste['alpha'])})"
    if not lp.empty:
        ax.plot(lp["grau"], lp["ccdf_lei_potencia"], color=CORES[1], label=rotulo_lp)
        ax.plot(ln["grau"], ln["ccdf_lognormal"], color=CORES[2], linestyle="--", label="Log-normal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    if ajuste is not None and np.isfinite(ajuste.get("R_vs_lognormal", np.nan)):
        ax.text(0.97, 0.97, f"xmin = {ajuste['xmin']:g}\nR = {num(ajuste['R_vs_lognormal'])}, "
                            f"p = {num(ajuste['p'], 3)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TINTA_SECUNDARIA)


# ---------------------------------------------------------------------------
# Exp 1 e 4
# ---------------------------------------------------------------------------
def _desenhar_curvas(ax: plt.Axes, curvas: pd.DataFrame, metrica: str, grupo: str) -> None:
    ordem = _ordenar(curvas[grupo], ESTRATEGIAS if grupo == "estrategia" else REDES)
    for i, valor in enumerate(ordem):
        curva = curvas[curvas[grupo] == valor].sort_values("fracao_removida")
        curva = curva.dropna(subset=[metrica])  # a eficiência só existe nos marcos
        if curva.empty:
            continue
        rotulo, cor, estilo_linha = _cor_e_estilo(grupo, valor, i)
        esparsa = len(curva) <= 25  # eficiência: um ponto a cada 5%
        ax.plot(curva["fracao_removida"], curva[metrica], color=cor, linestyle=estilo_linha,
                marker="o" if esparsa else None, markersize=3.5, markeredgecolor="white",
                markeredgewidth=0.5, label=rotulo, zorder=3)
        coluna_dp = f"{metrica}_dp"
        if coluna_dp in curva and curva[coluna_dp].notna().any():
            dp = curva[coluna_dp].fillna(0)
            ax.fill_between(curva["fracao_removida"], (curva[metrica] - dp).clip(lower=0),
                            curva[metrica] + dp, color=cor, alpha=0.15, linewidth=0, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _eixos_lineares_com_virgula(ax)


def curvas_robustez(curvas: pd.DataFrame, metrica: str = "maior_fraco",
                    grupo: str = "estrategia", zoom: float | None = 0.1) -> plt.Figure:
    """Exp 1 e 4. Uma curva por estratégia: metrica x fracao_removida.

    Na estratégia aleatória, a faixa de ±1 desvio padrão (<metrica>_dp).
    grupo: coluna que separa as curvas ("estrategia" no Exp 1; "rede" no Exp 4,
    com uma estratégia já filtrada). zoom: se não for None, um detalhe dos
    primeiros 0..zoom da fração removida, onde os ataques derrubam a rede.
    """
    estilo()
    if zoom:
        fig, (ax, detalhe) = plt.subplots(1, 2, figsize=(LARGURA, 3.0), width_ratios=(1.6, 1))
    else:
        fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 3.0))
        detalhe = None
    _desenhar_curvas(ax, curvas, metrica, grupo)
    ax.set_xlabel(ROTULO_FRACAO_REMOVIDA)
    ax.set_ylabel(METRICAS_ROBUSTEZ.get(metrica, metrica))
    ax.legend(loc="upper right")
    if detalhe is not None:
        _desenhar_curvas(detalhe, curvas, metrica, grupo)
        detalhe.set_xlim(0, zoom)
        detalhe.set_xlabel(ROTULO_FRACAO_REMOVIDA)
        detalhe.set_title(f"Detalhe: até {_virgula(zoom * 100)}% removidos", fontweight="normal",
                          color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return fig


def painel_robustez(curvas: pd.DataFrame, metricas: tuple[str, ...] = tuple(METRICAS_ROBUSTEZ),
                    grupo: str = "estrategia") -> plt.Figure:
    """Exp 1. Painel 2x2 com as quatro métricas de dano e uma legenda única embaixo.

    Componentes e alcance em fração dos nós da rede intacta; a eficiência, normalizada
    pelo número original de pares (metricas.eficiencia_global_dirigida).
    """
    estilo()
    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA, 5.4), sharex=True)
    letras = "abcd"
    for ax, metrica, letra in zip(eixos.flat, metricas, letras):
        _desenhar_curvas(ax, curvas, metrica, grupo)
        ax.set_title(f"({letra}) {TITULOS_ROBUSTEZ.get(metrica, metrica)}")
    for ax in eixos[1]:
        ax.set_xlabel(ROTULO_FRACAO_REMOVIDA)
    handles, labels = eixos.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def indices_por_rede(indices: pd.DataFrame, coluna: str = "R_fraco") -> plt.Figure:
    """Exp 4. Índice R por rede (eco, bsu, mge, nulos) e estratégia.

    Barras agrupadas por rede, uma por estratégia. Nas redes nulas (várias
    linhas por rede, coluna "indice"), a barra é a média entre as redes
    sorteadas e o traço é ±1 desvio padrão entre elas. coluna: qual índice
    mostrar (R_fraco, R_forte, metade_fraco, metade_forte...).
    """
    estilo()
    redes = _ordenar(indices["rede"], REDES)
    estrategias = _ordenar(indices["estrategia"], ESTRATEGIAS)
    resumo = indices.groupby(["rede", "estrategia"])[coluna].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(LARGURA, 3.0))
    largura = 0.8 / max(len(estrategias), 1)
    x = np.arange(len(redes))
    for j, estrategia in enumerate(estrategias):
        rotulo, cor, _ = _cor_e_estilo("estrategia", estrategia, j)
        medias = [resumo.loc[(r, estrategia), "mean"] if (r, estrategia) in resumo.index else np.nan for r in redes]
        erros = [resumo.loc[(r, estrategia), "std"] if (r, estrategia) in resumo.index
                 and resumo.loc[(r, estrategia), "count"] > 1 else 0 for r in redes]
        posicoes = x - 0.4 + largura * (j + 0.5)
        ax.bar(posicoes, medias, width=largura * 0.88, color=cor, label=rotulo, zorder=3)
        if any(e > 0 for e in erros):
            ax.errorbar(posicoes, medias, yerr=erros, fmt="none", ecolor=TINTA_SECUNDARIA,
                        elinewidth=0.8, capsize=2, zorder=4)
    ax.set_xticks(x, [_cor_e_estilo("rede", r, i)[0] for i, r in enumerate(redes)])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(coluna.replace("_", " "))
    _eixos_lineares_com_virgula(ax, x=False)
    ax.legend(loc="upper right", ncol=min(len(estrategias), 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exp 2
# ---------------------------------------------------------------------------
def curvas_roc(rocs: dict[str, pd.DataFrame], aucs: dict[str, float]) -> plt.Figure:
    """Exp 2. Uma curva ROC por ranking, com o AUC na legenda e a diagonal de referência.

    rocs: ranking -> tabela com colunas fpr, tpr (validacao.curva_roc).
    """
    estilo()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=CINZA, linewidth=0.8, label="Ao acaso (AUC = 0,5)", zorder=2)
    for i, nome in enumerate(_ordenar(rocs, RANKINGS_EXP2)):
        rotulo = RANKINGS_EXP2.get(nome, nome)
        if nome in aucs:
            rotulo += f" (AUC = {num(aucs[nome])})"
        roc = rocs[nome]
        ax.plot(roc["fpr"], roc["tpr"], color=CORES[i % len(CORES)], label=rotulo, zorder=3,
                drawstyle="steps-post")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    _eixos_lineares_com_virgula(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exp 3
# ---------------------------------------------------------------------------
_POSICOES_ROTULO = ((5, 3, "left"), (5, -10, "left"), (-5, 3, "right"), (-5, -10, "right"),
                    (0, 9, "center"), (0, -14, "center"))  # deslocamento em pontos e alinhamento


def _rotular_sem_sobrepor(ax: plt.Axes, pontos: list[tuple[float, float, str]]) -> None:
    """Escreve cada nome ao lado do seu ponto sem que os nomes se cruzem ou saiam do gráfico.

    Mede a caixa de cada nome em cada posição de _POSICOES_ROTULO e escolhe a
    combinação com menos sobreposições, depois menos nomes para fora, depois as
    posições mais à frente na lista. Com até 8 nomes testa todas as
    combinações; acima disso, escolhe nome a nome.
    """
    if not pontos:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    area = ax.get_window_extent(renderer)

    def anotar(x, y, texto, posicao):
        dx, dy, alinhamento = _POSICOES_ROTULO[posicao]
        return ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points",
                           ha=alinhamento, fontsize=7.5, color=TINTA_SECUNDARIA, zorder=4)

    caixas = []  # caixas[i][j]: caixa do nome i na posição j
    for x, y, texto in pontos:
        linha = []
        for j in range(len(_POSICOES_ROTULO)):
            rotulo = anotar(x, y, texto, j)
            linha.append(rotulo.get_window_extent(renderer).expanded(1.04, 1.12))
            rotulo.remove()
        caixas.append(linha)

    def fora(caixa) -> bool:
        return caixa.x0 < area.x0 or caixa.x1 > area.x1 or caixa.y0 < area.y0 or caixa.y1 > area.y1

    def custo(escolha) -> tuple[int, int, int]:
        escolhidas = [caixas[i][j] for i, j in enumerate(escolha)]
        cruzam = sum(a.overlaps(b) for k, a in enumerate(escolhidas) for b in escolhidas[k + 1:])
        return cruzam, sum(map(fora, escolhidas)), sum(escolha)

    opcoes = range(len(_POSICOES_ROTULO))
    if len(pontos) <= 8:
        melhor = min(itertools.product(opcoes, repeat=len(pontos)), key=custo)
    else:
        melhor = []
        for i in range(len(pontos)):
            def custo_do_nome(j, i=i):
                cruzam = sum(caixas[i][j].overlaps(caixas[k][melhor[k]]) for k in range(i))
                return cruzam, fora(caixas[i][j]), j
            melhor.append(min(opcoes, key=custo_do_nome))
    for (x, y, texto), posicao in zip(pontos, melhor):
        anotar(x, y, texto, posicao)


def avalanche_vs_grau(avalanches: pd.DataFrame, todos: pd.DataFrame | None = None) -> plt.Figure:
    """Exp 3. Tamanho da avalanche x grau das substâncias escolhidas e dos controles.

    avalanches: colunas grupo (chaves de GRUPOS_EXP3), metabolito, nome, grau, avalanche.
    todos: opcional, colunas grau e avalanche de todos os metabólitos (tabela
    exp3_avalanches), desenhados em cinza ao fundo para dar o contexto da rede.
    As substâncias escolhidas levam o nome ao lado do ponto.
    """
    estilo()
    fig, ax = plt.subplots(figsize=(LARGURA * 0.75, 3.6))
    if todos is not None and len(todos):
        ax.scatter(todos["grau"], todos["avalanche"], s=9, color=CINZA, alpha=0.35, linewidths=0,
                   label="Todos os metabólitos", zorder=1)
    for i, grupo in enumerate(_ordenar(avalanches["grupo"], GRUPOS_EXP3)):
        parte = avalanches[avalanches["grupo"] == grupo]
        rotulo, cor, marcador = GRUPOS_EXP3.get(grupo, (grupo, CORES[i % len(CORES)], "o"))
        ax.scatter(parte["grau"], parte["avalanche"], s=36, color=cor, marker=marcador,
                   edgecolors="white", linewidths=0.8, label=rotulo, zorder=3)
    ax.set_xlabel("Grau total do metabólito removido")
    ax.set_ylabel("Tamanho da avalanche\n(metabólitos perdidos)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08, y=0.08)
    _eixos_lineares_com_virgula(ax)
    ax.legend(loc="center right")
    fig.tight_layout()
    escolhidas = avalanches[avalanches["grupo"] == "substancia"]
    _rotular_sem_sobrepor(ax, list(zip(escolhidas["grau"], escolhidas["avalanche"], escolhidas["nome"])))
    return fig


def distribuicao_avalanches(tamanhos: pd.Series, ccdf: pd.DataFrame | None = None,
                            ajuste: dict | None = None) -> plt.Figure:
    """Exp 3. Distribuição dos tamanhos de avalanche de todos os metabólitos (log-log).

    CCDF P(S >= s) das avalanches de tamanho >= 1, como na distribuição de
    graus. Com ccdf (metricas.ccdf_com_ajustes das avalanches >= 1) e ajuste
    (metricas.ajuste_lei_potencia), sobrepõe os ajustes de lei de potência e
    log-normal. O título diz quantos metabólitos não derrubam nenhum outro
    (avalanche 0), que não aparecem no log-log.
    """
    estilo()
    valores = np.asarray(pd.Series(tamanhos).dropna(), dtype=int)
    positivos = np.sort(valores[valores > 0])
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 3.0))
    if ccdf is not None:
        _desenhar_ccdf(ax, ccdf, ajuste, "Dados (avalanches ≥ 1)")
    elif len(positivos):
        s = np.unique(positivos)
        empirica = 1.0 - np.searchsorted(positivos, s, side="left") / len(positivos)
        ax.plot(s, empirica, linestyle="none", marker="o", color=CORES[0], markeredgecolor="white",
                markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
    zeros = int((valores == 0).sum())
    ax.set_title(f"{num(zeros, 0)} de {num(len(valores), 0)} metabólitos "
                 f"({num(100 * zeros / max(len(valores), 1), 0)}%) sem avalanche",
                 fontweight="normal", color=TINTA_SECUNDARIA)
    ax.set_xlabel("Tamanho da avalanche s")
    ax.set_ylabel("P(S ≥ s)")
    fig.tight_layout()
    return fig
