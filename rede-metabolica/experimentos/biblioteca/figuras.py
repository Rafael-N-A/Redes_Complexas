"""Figuras do artigo, com o mesmo estilo em todas.

Cada função recebe as tabelas que os scripts gravaram em resultados/tabelas/
e devolve uma figura do matplotlib. Quem grava é o script, com
config.arquivo_figura(...) e salvar(); a biblioteca não decide nada sobre arquivos.

    distribuicao_graus           Figura 2 (Exp 0): distribuição de graus; o painel (b),
                                 a CCDF com os ajustes, é o que vai para o artigo
    curvas_robustez              Figura 3 (Exp 1): maior componente fraco ao remover nós
    barras_essenciais_por_faixa  Figura 4 (Exp 2): % de essenciais por faixa de dano

Convenções (iguais em todas as figuras):
  - largura de uma coluna do modelo SBC (~16 cm); texto em português, com
    vírgula decimal nos eixos lineares;
  - cores categóricas numa ordem fixa, testada para daltonismo (CORES); cada
    entidade tem sempre a mesma cor: a estratégia aleatória é sempre azul;
  - estratégias adaptativas em linha não contínua, cada uma com o seu traço,
    além da cor, para a figura continuar legível impressa em preto e branco (e
    para separar o par aqua x magenta, que fica perto sob deuteranopia);
  - grade e eixos em cinza claro, sem bordas de cima e da direita; legenda
    sempre que há mais de uma série.

As cores que aparecem juntas na Figura 3 (a aleatória e as três adaptativas:
CORES[0], [2], [4], [6]) foram conferidas com o validador de paletas (deltaE em
OKLab sob protanopia e deuteranopia); o PageRank fica no violeta porque o
vermelho, ao lado do magenta, reprova até para visão normal.
"""
from __future__ import annotations

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
CINZA = "#898781"          # referências
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

LARGURA = 6.3              # polegadas (~16 cm)

ESTRATEGIAS = {  # nome -> (rótulo, cor, estilo de linha); as adaptativas, cada uma com o seu traço
    "aleatoria": ("Aleatória", CORES[0], "-"),
    "grau": ("Grau", CORES[1], "-"),
    "grau_adaptativo": ("Grau (adaptativo)", CORES[2], "--"),
    "betweenness": ("Betweenness", CORES[3], "-"),
    "betweenness_adaptativo": ("Betweenness (adaptativo)", CORES[4], "-."),
    "pagerank": ("PageRank", CORES[6], "-"),
    "pagerank_adaptativo": ("PageRank (adaptativo)", CORES[6], (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6))),
}

METRICAS_ROBUSTEZ = {  # coluna das curvas do Exp 1 -> rótulo do eixo y
    "maior_fraco": "Maior componente fraco\n(fração dos nós)",
    "maior_forte": "Maior componente forte\n(fração dos nós)",
    "alcance": "Alcance topológico a partir\ndo meio mínimo (fração)",
}

FAIXAS_EXP2 = {  # Figura 4: medida -> (rótulo, cor); a ordem é a das barras em cada faixa
    "dano_forte": ("Quebrar a rede\n(saem do núcleo)", CORES[1]),
    "dano": ("Deixar de produzir\n(cascata)", CORES[0]),
}

ROTULO_FRACAO_REMOVIDA = "Fração de nós removidos"


def _virgula(valor: float, _posicao=None) -> str:
    """Número curto com vírgula decimal: 0.25 -> '0,25', 1.0 -> '1'."""
    return f"{valor:g}".replace(".", ",")


def num(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal, ponto de milhar e sinal de menos (−), para legendas e anotações."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")


def _eixos_lineares_com_virgula(ax: plt.Axes, x: bool = True, y: bool = True) -> None:
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_virgula))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_virgula))


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
# Figura 2 (Exp 0)
# ---------------------------------------------------------------------------
_MARCADORES_GRAU = {"entrada": ("Entrada", CORES[0], "o"),
                    "saida": ("Saída", CORES[1], "s"),
                    "total": ("Total", CORES[2], "^")}


def distribuicao_graus(dist: pd.DataFrame, ccdf: pd.DataFrame | None = None,
                       ajuste: dict | None = None) -> plt.Figure:
    """Exp 0. Entrada, saída e total em log-log (tabela de metricas.distribuicao_graus).

    Com ccdf (tabela de metricas.ccdf_com_ajustes), um segundo painel com a CCDF
    do grau total e os ajustes de lei de potência e log-normal sobrepostos a
    partir de xmin, como em Clauset et al. (2009); é o painel da Figura 2 do
    artigo. ajuste (uma linha de metricas.ajuste_lei_potencia) acrescenta alpha,
    R e p na anotação. Graus 0 não aparecem no log-log.
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
# Figura 3 (Exp 1)
# ---------------------------------------------------------------------------
def _desenhar_curvas(ax: plt.Axes, curvas: pd.DataFrame, metrica: str) -> None:
    """Uma curva por estratégia (cor e traço de ESTRATEGIAS); na aleatória, a faixa de ±1 dp."""
    for i, estrategia in enumerate(_ordenar(curvas["estrategia"], ESTRATEGIAS)):
        curva = curvas[curvas["estrategia"] == estrategia].sort_values("fracao_removida")
        curva = curva.dropna(subset=[metrica])
        if curva.empty:
            continue
        rotulo, cor, estilo_linha = ESTRATEGIAS.get(estrategia, (str(estrategia), CORES[i % len(CORES)], "-"))
        ax.plot(curva["fracao_removida"], curva[metrica], color=cor, linestyle=estilo_linha,
                label=rotulo, zorder=3)
        coluna_dp = f"{metrica}_dp"
        if coluna_dp in curva and curva[coluna_dp].notna().any():
            dp = curva[coluna_dp].fillna(0)
            # Todas as métricas são frações (de 0 a 1), e a faixa não passa disso.
            ax.fill_between(curva["fracao_removida"], (curva[metrica] - dp).clip(lower=0),
                            (curva[metrica] + dp).clip(upper=1), color=cor, alpha=0.15, linewidth=0, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _eixos_lineares_com_virgula(ax)


def curvas_robustez(curvas: pd.DataFrame, metrica: str = "maior_fraco") -> plt.Figure:
    """Exp 1. Uma curva por estratégia: metrica x fracao_removida, a curva inteira.

    curvas: a tabela exp1_curvas, já filtrada nas estratégias que devem aparecer.
    Na estratégia aleatória, a faixa de ±1 desvio padrão (<metrica>_dp). É a
    curva clássica de Albert, Jeong e Barabási (2000).
    """
    estilo()
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 3.0))
    _desenhar_curvas(ax, curvas, metrica)
    ax.set_xlabel(ROTULO_FRACAO_REMOVIDA)
    ax.set_ylabel(METRICAS_ROBUSTEZ.get(metrica, metrica))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 4 (Exp 2)
# ---------------------------------------------------------------------------
def barras_essenciais_por_faixa(faixas: pd.DataFrame) -> plt.Figure:
    """Exp 2. % de genes essenciais por faixa de dano, por medida.

    faixas: a tabela exp2_essenciais_por_faixa (medida, faixa, genes, essenciais,
    taxa). Barras lado a lado em cada faixa, uma por medida (cores e rótulos de
    FAIXAS_EXP2), com o valor em cima, e a linha tracejada da média (a fração de
    essenciais entre todos os genes da tabela, a mesma em toda medida).
    """
    estilo()
    medidas = _ordenar(faixas["medida"], FAIXAS_EXP2)
    rotulos = list(dict.fromkeys(faixas["faixa"]))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.arange(len(rotulos))
    largura = 0.38
    for i, medida in enumerate(medidas):
        parte = faixas[faixas["medida"] == medida].set_index("faixa").reindex(rotulos)
        nome, cor = FAIXAS_EXP2.get(medida, (medida, CORES[i % len(CORES)]))
        taxa = 100 * parte["taxa"].to_numpy(dtype=float)
        posicoes = x + (i - (len(medidas) - 1) / 2) * largura
        ax.bar(posicoes, taxa, largura, label=nome, color=cor, zorder=3)
        for xi, v in zip(posicoes, taxa):
            if not np.isnan(v):
                ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    primeira = faixas[faixas["medida"] == medidas[0]]
    base = 100 * primeira["essenciais"].sum() / primeira["genes"].sum()
    ax.axhline(base, color="0.4", ls="--", lw=0.8, zorder=2)
    ax.text(1.0, base + 1.5, f"média ({base:.0f}%)", ha="center", va="bottom", fontsize=7, color="0.3")
    ax.set_xticks(x, rotulos)
    ax.set_xlabel("Tamanho do dano (nº de metabólitos)")
    ax.set_ylabel("Genes essenciais na Keio (%)")
    ax.set_ylim(0, max(80.0, float(np.nanmax(100 * faixas["taxa"].to_numpy(dtype=float))) + 10))
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    return fig
