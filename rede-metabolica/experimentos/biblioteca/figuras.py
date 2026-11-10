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
  - estratégias adaptativas em linha não contínua, cada uma com o seu traço,
    além da cor, para a figura continuar legível impressa em preto e branco (e
    para separar o par aqua x magenta, que fica perto sob deuteranopia); a
    harmônica e o PageRank usam a mesma cor nas versões estática e adaptativa,
    que só se distinguem pelo traço;
  - grade e eixos em cinza claro, sem bordas de cima e da direita; legenda
    sempre que há mais de uma série.

As combinações de cores que aparecem juntas foram conferidas com o validador
de paletas (deltaE em OKLab sob protanopia e deuteranopia): aleatória e as
quatro adaptativas (CORES[0], [2], [4], [6], [7]; o par aqua x magenta fica na
faixa de 6 a 8 e por isso as adaptativas têm traços diferentes); as estratégias
dos modelos clássicos (CORES[0:4], [6], [7]); e os modelos clássicos (CORES[0:7],
na ordem da paleta).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
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

ESTRATEGIAS = {  # nome -> (rótulo, cor, estilo de linha); as adaptativas, cada uma com o seu traço
    "aleatoria": ("Aleatória", CORES[0], "-"),
    "grau": ("Grau", CORES[1], "-"),
    "grau_adaptativo": ("Grau (adaptativo)", CORES[2], "--"),
    "betweenness": ("Betweenness", CORES[3], "-"),
    "betweenness_adaptativo": ("Betweenness (adaptativo)", CORES[4], "-."),
    "harmonica": ("Harmônica", CORES[6], "-"),
    "harmonica_adaptativo": ("Harmônica (adaptativo)", CORES[6], (0, (1.2, 1.6))),
    "pagerank": ("PageRank", CORES[7], "-"),
    "pagerank_adaptativo": ("PageRank (adaptativo)", CORES[7], (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6))),
}

MODELOS = {  # Exp 4, modelos clássicos (não dirigidos): nome -> (rótulo, cor, estilo); cores na ordem da paleta
    "eco": ("E. coli", CORES[0], "-"),
    "configuracao": ("Configuração", CORES[1], "--"),
    "erdos_renyi": ("Erdős–Rényi", CORES[2], "--"),
    "ba_misto": ("BA misto (m = 1 ou 2)", CORES[3], "-."),
    "ba_m1": ("BA m = 1 (árvore)", CORES[4], ":"),
    "newman_watts": ("Newman–Watts (k = 2)", CORES[5], "-."),
    "watts_strogatz": ("Watts–Strogatz (k = 4)", CORES[6], ":"),
}

ROTULOS_CURTOS_MODELOS = {  # eixo x das barras dos modelos clássicos
    "configuracao": "Configu-\nração",
    "erdos_renyi": "Erdős–\nRényi",
    "ba_misto": "BA misto\n(m = 1 ou 2)",
    "ba_m1": "BA m = 1\n(árvore)",
    "newman_watts": "Newman–\nWatts (k = 2)",
    "watts_strogatz": "WS (k = 4,\np = 0,1)",
}

REDES = {  # Exp 4: nome -> (rótulo, cor, estilo de linha); os modelos nulos tracejados
    "eco": ("E. coli", CORES[0], "-"),
    "bsu": ("B. subtilis", CORES[1], "-"),
    "mge": ("M. genitalium", CORES[2], "-"),
    "rewiring": ("Rewiring (mesmos graus)", CORES[3], "--"),
    "rewiring_reciproco": ("Rewiring (graus e reciprocidade)", CORES[5], "-."),
    "erdos_renyi": ("Erdős–Rényi", CORES[4], ":"),
}
# As cores que aparecem juntas foram validadas entre todos os pares: os três
# organismos (CORES[0:3]) e a E. coli com os três nulos (CORES[0], [3], [5], [4]).
# M. genitalium e Erdős–Rényi nunca dividem uma figura de linhas.

ROTULOS_CURTOS_REDES = {  # eixo x das barras do Exp 4, onde não cabe o rótulo inteiro
    "rewiring": "Rewiring\n(graus)",
    "rewiring_reciproco": "Rewiring\n(graus e\nreciprocidade)",
}

ROTULOS_INDICES = {  # coluna da tabela de índices do Exp 4 -> rótulo do eixo y
    "R_fraco": "Índice R\n(componente fraco)",
    "R_forte": "Índice R\n(componente forte)",
    "R_fraco_norm": "R normalizado\n(componente fraco)",
    "R_forte_norm": "R normalizado\n(componente forte)",
    "metade_fraco": "Fração removida até o\ncomponente fraco cair à metade",
    "metade_forte": "Fração removida até o\ncomponente forte cair à metade",
    "R_alcance_norm": "R normalizado\n(alcance topológico)",
    # modelos clássicos (não dirigidos: um componente só)
    "R": "Índice R\n(maior componente)",
    "R_norm": "R normalizado\n(maior componente)",
    "metade": "Fração removida até o\nmaior componente cair à metade",
}

METRICAS_ROBUSTEZ = {  # coluna das curvas -> rótulo do eixo y
    "maior_fraco": "Maior componente fraco\n(fração dos nós)",
    "maior_forte": "Maior componente forte\n(fração dos nós)",
    "eficiencia": "Eficiência global",
    "alcance": "Alcance topológico a partir\ndo meio mínimo (fração)",
    # Exp 4: divididas pelo valor na rede intacta, para comparar redes que já começam partidas
    "maior_fraco_rel": "Maior componente fraco\n(relativo à rede intacta)",
    "maior_forte_rel": "Maior componente forte\n(relativo à rede intacta)",
    "alcance_rel": "Alcance topológico\n(relativo à rede intacta)",
    # modelos clássicos (não dirigidos)
    "maior_componente": "Maior componente\n(fração dos nós)",
    "maior_componente_rel": "Maior componente\n(relativo à rede intacta)",
}

TITULOS_ROBUSTEZ = {  # títulos curtos dos painéis
    "maior_fraco": "Maior componente fraco",
    "maior_forte": "Maior componente forte",
    "eficiencia": "Eficiência global",
    "alcance": "Alcance topológico",
    "maior_fraco_rel": "Maior componente fraco",
    "maior_forte_rel": "Maior componente forte",
    "alcance_rel": "Alcance topológico",
    "maior_componente": "Maior componente",
    "maior_componente_rel": "Maior componente",
}

GRUPOS_EXP3 = {  # coluna "grupo" da tabela exp3_substancias -> (rótulo, cor, marcador)
    "substancia": ("Substâncias escolhidas", CORES[0], "o"),
    "hub": ("Controle: maior grau", CORES[1], "s"),
    "baixo_grau": ("Controle: baixo grau", CORES[2], "^"),
}

RANKINGS_EXP2 = {  # Exp 2: nome do ranking -> rótulo (a ordem é a das figuras)
    "dano": "Dano na cascata",
    "dano_forte": "Queda do núcleo",
    "dano_fraco": "Queda do componente fraco",
    "dano_eficiencia": "Queda da eficiência",
    "grau": "Grau",
    "betweenness": "Betweenness",
    "harmonica": "Harmônica",
    "pagerank": "PageRank",
    "dano_meio_minimo": "Cascata (meio mínimo)",
    "reacoes_perdidas": "Reações perdidas",
}

SUBCONJUNTOS_EXP2 = {  # Exp 2: coluna subconjunto -> (rótulo, cor)
    "todos": ("Todos os genes da rede", CORES[0]),
    "perde_reacao": ("Só os que perdem alguma reação", CORES[1]),
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


_REFERENCIAS = {"estrategia": ESTRATEGIAS, "rede": REDES, "modelo": MODELOS}
_ROTULOS_CURTOS = {"rede": ROTULOS_CURTOS_REDES, "modelo": ROTULOS_CURTOS_MODELOS}


def _cor_e_estilo(grupo: str, valor: str, posicao: int) -> tuple[str, str, str]:
    """(rótulo, cor, estilo de linha) de uma série; cai na ordem de CORES se o valor for novo.

    grupo: "estrategia" (ESTRATEGIAS), "rede" (REDES, Exp 4) ou "modelo" (MODELOS, modelos clássicos).
    """
    referencia = _REFERENCIAS.get(grupo, {})
    if valor in referencia:
        return referencia[valor]
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
    ordem = _ordenar(curvas[grupo], _REFERENCIAS.get(grupo, {}))
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
            # Todas as métricas são frações (de 0 a 1), e a faixa não passa disso.
            ax.fill_between(curva["fracao_removida"], (curva[metrica] - dp).clip(lower=0),
                            (curva[metrica] + dp).clip(upper=1), color=cor, alpha=0.15, linewidth=0, zorder=2)
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


def estatica_vs_adaptativa(curvas: pd.DataFrame, centralidades: list[str],
                           metrica: str = "maior_fraco", ate: float = 0.25) -> plt.Figure:
    """Exp 1. Um painel por centralidade: a versão estática x a adaptativa, com a aleatória de referência.

    curvas: a tabela exp1_curvas (todas as estratégias). Em cada painel, a
    aleatória (com a faixa de ±1 dp), a estática (linha contínua) e a adaptativa
    (com o traço dela), na cor de cada estratégia. ate: fim do eixo x (os ataques
    derrubam a rede nos primeiros nós). Legenda em cada painel.
    """
    estilo()
    colunas = 2 if len(centralidades) > 1 else 1
    linhas_painel = math.ceil(len(centralidades) / colunas)
    fig, eixos = plt.subplots(linhas_painel, colunas, figsize=(LARGURA, 2.5 * linhas_painel + 0.4),
                              sharex=True, sharey=True, squeeze=False)
    for ax, centralidade, letra in zip(eixos.flat, centralidades, "abcdefgh"):
        nomes = ["aleatoria", centralidade, f"{centralidade}_adaptativo"]
        _desenhar_curvas(ax, curvas[curvas["estrategia"].isin(nomes)], metrica, "estrategia")
        ax.set_xlim(0, ate)
        ax.set_title(f"({letra}) {ESTRATEGIAS.get(centralidade, (centralidade,))[0]}")
        ax.legend(loc="upper right", fontsize=7.5)
    for ax in eixos.flat[len(centralidades):]:
        ax.set_visible(False)
    for ax in eixos[-1]:
        ax.set_xlabel(ROTULO_FRACAO_REMOVIDA)
    for ax in eixos[:, 0]:
        ax.set_ylabel(METRICAS_ROBUSTEZ.get(metrica, metrica))
    fig.tight_layout()
    return fig


def _desenhar_indices(ax: plt.Axes, indices: pd.DataFrame, coluna: str, grupo: str = "rede") -> None:
    """Barras agrupadas por rede, uma por estratégia; nas redes nulas, média e ±1 dp entre as sorteadas.

    grupo: a coluna das redes ("rede" no Exp 4, "modelo" nos modelos clássicos).
    """
    redes = _ordenar(indices[grupo], _REFERENCIAS.get(grupo, {}))
    estrategias = _ordenar(indices["estrategia"], ESTRATEGIAS)
    resumo = indices.groupby([grupo, "estrategia"])[coluna].agg(["mean", "std", "count"])
    largura = 0.8 / max(len(estrategias), 1)
    x = np.arange(len(redes))
    for j, estrategia in enumerate(estrategias):
        rotulo, cor, _ = _cor_e_estilo("estrategia", estrategia, j)
        medias = [resumo.loc[(r, estrategia), "mean"] if (r, estrategia) in resumo.index else np.nan for r in redes]
        erros = [resumo.loc[(r, estrategia), "std"] if (r, estrategia) in resumo.index
                 and resumo.loc[(r, estrategia), "count"] > 1 else 0 for r in redes]
        posicoes = x - 0.4 + largura * (j + 0.5)
        ax.bar(posicoes, medias, width=largura * 0.88, color=cor, label=rotulo, zorder=3)
        com_erro = [i for i, e in enumerate(erros) if e > 0]  # só nas redes nulas: sem traço vazio nas outras
        if com_erro:
            ax.errorbar(posicoes[com_erro], np.array(medias)[com_erro], yerr=np.array(erros)[com_erro],
                        fmt="none", ecolor=TINTA_SECUNDARIA, elinewidth=0.8, capsize=2, zorder=4)
    curtos = _ROTULOS_CURTOS.get(grupo, {})
    ax.set_xticks(x, [curtos.get(r, _cor_e_estilo(grupo, r, i)[0]) for i, r in enumerate(redes)])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ROTULOS_INDICES.get(coluna, coluna.replace("_", " ")))
    ax.set_ylim(bottom=0)
    _eixos_lineares_com_virgula(ax, x=False)


def indices_por_rede(indices: pd.DataFrame, coluna: str = "R_fraco") -> plt.Figure:
    """Exp 4. Índice R por rede (eco, bsu, mge, nulos) e estratégia.

    Barras agrupadas por rede, uma por estratégia. Nas redes nulas (várias
    linhas por rede, coluna "indice"), a barra é a média entre as redes
    sorteadas e o traço é ±1 desvio padrão entre elas. coluna: qual índice
    mostrar (R_fraco, R_forte, metade_fraco, metade_forte...).
    """
    estilo()
    fig, ax = plt.subplots(figsize=(LARGURA, 3.0))
    _desenhar_indices(ax, indices, coluna)
    ax.legend(loc="upper right", ncol=min(indices["estrategia"].nunique(), 3))
    fig.tight_layout()
    return fig


def painel_indices_redes(indices: pd.DataFrame,
                         colunas: tuple[str, ...] = ("R_fraco_norm", "metade_fraco"),
                         grupo: str = "rede") -> plt.Figure:
    """Exp 4. Um painel por índice, um embaixo do outro, com as mesmas redes no eixo x.

    Mesmo desenho de indices_por_rede; a legenda das estratégias fica uma vez só, em cima.
    grupo: "rede" (organismos e nulos) ou "modelo" (modelos clássicos, colunas R_norm e metade).
    """
    estilo()
    fig, eixos = plt.subplots(len(colunas), 1, figsize=(LARGURA, 2.3 * len(colunas) + 0.6),
                              sharex=True, squeeze=False)
    for ax, coluna, letra in zip(eixos[:, 0], colunas, "abcdef"):
        _desenhar_indices(ax, indices, coluna, grupo)
        ax.set_title(f"({letra})")
    handles, labels = eixos[0, 0].get_legend_handles_labels()
    ncol = len(labels) if len(labels) <= 4 else math.ceil(len(labels) / 2)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(labels) <= 4 else 0.92))
    return fig


def painel_curvas_redes(curvas: pd.DataFrame,
                        metricas: tuple[str, ...] = ("maior_fraco_rel", "maior_forte_rel"),
                        estrategias: tuple[str, ...] = ("aleatoria", "grau_adaptativo"),
                        limites_x: dict[str, float] | None = None,
                        grupo: str = "rede") -> plt.Figure:
    """Exp 4. Grade de curvas: uma linha por métrica, uma coluna por estratégia, uma curva por rede.

    curvas: a tabela exp4_curvas (coluna "rede"), já filtrada nas redes que
    devem aparecer. O título de cada painel é a estratégia; a métrica fica no eixo y. A faixa é ±1 desvio padrão (<metrica>_dp): entre as
    repetições na aleatória dos organismos e entre as redes sorteadas nos
    modelos nulos. limites_x: estratégia -> fim do eixo x (os ataques
    derrubam a rede nos primeiros nós; o padrão corta o grau adaptativo em 0,4).
    grupo: "rede" ou "modelo" (a tabela exp4_classicos_curvas).
    """
    estilo()
    limites_x = {"grau_adaptativo": 0.4, "grau": 0.4} if limites_x is None else limites_x
    fig, eixos = plt.subplots(len(metricas), len(estrategias),
                              figsize=(LARGURA, 2.4 * len(metricas) + 0.7), squeeze=False)
    letras = iter("abcdefghij")
    for i, metrica in enumerate(metricas):
        for j, estrategia in enumerate(estrategias):
            ax = eixos[i, j]
            _desenhar_curvas(ax, curvas[curvas["estrategia"] == estrategia], metrica, grupo)
            ax.set_xlim(0, limites_x.get(estrategia, 1))
            ax.set_title(f"({next(letras)}) {ESTRATEGIAS.get(estrategia, (estrategia,))[0]}")
            if j == 0:
                ax.set_ylabel(METRICAS_ROBUSTEZ.get(metrica, metrica))
            if i == len(metricas) - 1:
                ax.set_xlabel(ROTULO_FRACAO_REMOVIDA)
    handles, labels = eixos[0, 0].get_legend_handles_labels()
    ncol = min(len(labels), 4) if len(labels) <= 4 else math.ceil(len(labels) / 2)
    linhas_legenda = math.ceil(len(labels) / max(ncol, 1))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01),
               fontsize=8 if linhas_legenda == 1 else 7.5)
    fig.tight_layout(rect=(0, 0.06 + 0.035 * max(linhas_legenda - 1, 0), 1, 1))
    return fig


def ccdf_por_rede(ccdf: pd.DataFrame, grupo: str = "rede") -> plt.Figure:
    """Exp 4. CCDF do grau total de cada rede (colunas rede, grau, n_nos, ccdf_empirica), em log-log.

    Só os graus observados, como no Exp 0. Mostra se a cauda pesada da E. coli
    se repete nos outros organismos (Jeong et al., 2000). grupo="modelo": a
    tabela exp4_classicos_ccdf (a E. coli não dirigida e os modelos clássicos,
    com os graus das redes sorteadas de cada modelo juntos). O modelo de
    configuração tem por construção os mesmos graus da rede real, e a curva dele
    ficaria escondida embaixo da dela: não é desenhado, e o rótulo da rede real
    diz isso. Graus 0 (nós isolados dos modelos) não vão ao log-log.
    """
    estilo()
    classicos = grupo == "modelo"
    fig, ax = plt.subplots(figsize=(LARGURA * (0.8 if classicos else 0.62), 3.3 if classicos else 3.0))
    marcadores = itertools.cycle("os^Dv<>")
    redes = _ordenar(ccdf[grupo], _REFERENCIAS.get(grupo, {}))
    com_configuracao = classicos and "configuracao" in redes
    if classicos:  # a rede real (o que não é modelo) primeiro, com a cor da E. coli
        redes = [r for r in redes if r not in config.MODELOS_CLASSICOS] + [r for r in redes if r in config.MODELOS_CLASSICOS]
    for i, rede in enumerate(redes):
        if com_configuracao and rede == "configuracao":
            continue
        parte = ccdf[(ccdf[grupo] == rede) & (ccdf["n_nos"] > 0) & (ccdf["ccdf_empirica"] > 0) & (ccdf["grau"] > 0)]
        rotulo, cor, estilo_linha = _cor_e_estilo(grupo, rede, i)
        if com_configuracao and rede not in config.MODELOS_CLASSICOS:
            rotulo += " (= configuração)"
        ax.plot(parte["grau"], parte["ccdf_empirica"], color=cor, linestyle=estilo_linha, linewidth=1,
                marker=next(marcadores), markersize=3.5 if classicos else 4.5,
                markeredgecolor="white", markeredgewidth=0.5, label=rotulo)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grau k" if classicos else "Grau total k")
    ax.set_ylabel("P(K ≥ k)")
    if classicos:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    else:
        ax.legend(loc="lower left")
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
    fig, ax = plt.subplots(figsize=(3.6, 4.3))
    ax.plot([0, 1], [0, 1], color=CINZA, linewidth=0.8, label="Ao acaso (AUC = 0,5)", zorder=2)
    for i, nome in enumerate(_ordenar(rocs, RANKINGS_EXP2)):
        rotulo = RANKINGS_EXP2.get(nome, nome)
        if nome in aucs:
            rotulo += f" (AUC = {num(aucs[nome])})"
        roc = rocs[nome]
        ax.plot(roc["fpr"], roc["tpr"], color=CORES[i % len(CORES)], label=rotulo, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    _eixos_lineares_com_virgula(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)  # embaixo: não cobre as curvas
    fig.tight_layout()
    return fig


_EIXOS_DISPERSAO_EXP2 = {  # medida estrutural no eixo x -> rótulo
    "dano_forte": "Metabólitos que deixam o núcleo\n(queda do maior componente forte)",
    "dano_eficiencia": "Queda relativa da eficiência global",
}


def dispersao_dano_genes(genes: pd.DataFrame, rotulos: dict[str, list[str]] | None = None,
                         correlacao: pd.DataFrame | None = None,
                         estruturais: tuple[str, ...] = ("dano_forte", "dano_eficiencia")) -> plt.Figure:
    """Exp 2 (b). Dano estrutural (eixo x) x dano funcional (a cascata, eixo y) de cada gene.

    genes: a tabela exp2_genes (colunas essencial, dano e as de estruturais). Um
    painel por medida estrutural; os genes não essenciais em cinza, ao fundo, e os
    essenciais da Keio destacados. rotulos: texto -> símbolos dos genes; o texto vai
    ao lado do ponto médio desses genes (ex.: "aroA–aroD" para os quatro da via do
    chiquimato, que caem quase no mesmo lugar). correlacao: a tabela
    exp2_correlacoes; o Spearman de cada medida com a cascata, nos genes que
    perdem reação, vai numa nota embaixo da legenda.
    """
    estilo()
    fig, eixos = plt.subplots(1, len(estruturais), figsize=(LARGURA, 3.5), squeeze=False)
    essencial = genes["essencial"].astype(bool)
    por_simbolo = genes.set_index("simbolo")
    rho = {}
    if correlacao is not None:
        perde = correlacao[correlacao["subconjunto"] == "perde_reacao"]
        rho = {linha["medida_y"]: linha["spearman"] for linha in perde.to_dict("records") if linha["medida_x"] == "dano"}
    for ax, coluna, letra in zip(eixos[0], estruturais, "abcd"):
        ax.scatter(genes.loc[~essencial, coluna], genes.loc[~essencial, "dano"], s=12, color=CINZA, alpha=0.45,
                   linewidths=0, label="Não essencial", zorder=2)
        ax.scatter(genes.loc[essencial, coluna], genes.loc[essencial, "dano"], s=22, color=CORES[1],
                   edgecolors="white", linewidths=0.6, label="Essencial (Keio)", zorder=3)
        ax.set_xlabel(_EIXOS_DISPERSAO_EXP2.get(coluna, coluna))
        ax.set_title(f"({letra}) {RANKINGS_EXP2.get(coluna, coluna)}")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.06, y=0.06)
        _eixos_lineares_com_virgula(ax)
    eixos[0, 0].set_ylabel("Metabólitos que deixam de ser\nproduzidos (cascata)")
    handles, labels = eixos[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.035))
    nota = " · ".join(f"{RANKINGS_EXP2.get(c, c).lower()} {num(rho[c])}" for c in estruturais if c in rho)
    if nota:
        fig.text(0.5, 0.0, f"Spearman com a cascata nos genes que perdem alguma reação: {nota}",
                 ha="center", va="bottom", fontsize=7.5, color=TINTA_SECUNDARIA)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ax, coluna in zip(eixos[0], estruturais):
        pontos = []
        for texto, simbolos in (rotulos or {}).items():
            presentes = [s for s in simbolos if s in por_simbolo.index]
            if presentes:
                pontos.append((float(por_simbolo.loc[presentes, coluna].mean()),
                               float(por_simbolo.loc[presentes, "dano"].mean()), texto))
        _rotular_sem_sobrepor(ax, pontos)
    return fig


def barras_auc(avaliacao: pd.DataFrame, rankings: list[str] | None = None) -> plt.Figure:
    """Exp 2 (c). AUC de cada ranking contra a Keio, com o intervalo de 95%, nos dois subconjuntos.

    avaliacao: a tabela exp2_rankings (colunas subconjunto, ranking, auc,
    auc_ic_inf, auc_ic_sup). Barras horizontais, uma por subconjunto em cada
    ranking (todos os genes x só os que perdem alguma reação: a checagem das
    isoenzimas), e a linha do acaso em 0,5. rankings: quais e em que ordem (de
    cima para baixo); padrão: os da tabela, na ordem de RANKINGS_EXP2.
    """
    estilo()
    rankings = rankings or _ordenar(avaliacao["ranking"], RANKINGS_EXP2)
    subconjuntos = _ordenar(avaliacao["subconjunto"], SUBCONJUNTOS_EXP2)
    tabela = avaliacao.set_index(["subconjunto", "ranking"])
    fig, ax = plt.subplots(figsize=(LARGURA * 0.8, 0.42 * len(rankings) + 1.1))
    y = np.arange(len(rankings))[::-1]
    altura = 0.8 / max(len(subconjuntos), 1)
    for j, subconjunto in enumerate(subconjuntos):
        rotulo, cor = SUBCONJUNTOS_EXP2.get(subconjunto, (subconjunto, CORES[j % len(CORES)]))
        linhas = [tabela.loc[(subconjunto, r)] if (subconjunto, r) in tabela.index else None for r in rankings]
        valores = np.array([l["auc"] if l is not None else np.nan for l in linhas])
        inferior = np.array([l["auc"] - l["auc_ic_inf"] if l is not None else 0 for l in linhas])
        superior = np.array([l["auc_ic_sup"] - l["auc"] if l is not None else 0 for l in linhas])
        posicoes = y + 0.4 - altura * (j + 0.5)
        ax.barh(posicoes, valores, height=altura * 0.86, color=cor, label=rotulo, zorder=3)
        ax.errorbar(valores, posicoes, xerr=[inferior, superior], fmt="none", ecolor=TINTA_SECUNDARIA,
                    elinewidth=0.8, capsize=2, zorder=4)
    ax.axvline(0.5, color=CINZA, linewidth=1, zorder=2)
    ax.text(0.5, 1.0, " acaso", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, color=TINTA_SECUNDARIA)
    ax.set_yticks(y, [RANKINGS_EXP2.get(r, r) for r in rankings])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC contra a Keio (com IC de 95%)")
    _eixos_lineares_com_virgula(ax, y=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Exp 3
# ---------------------------------------------------------------------------
_POSICOES_ROTULO = ((5, 3, "left"), (5, -10, "left"), (-5, 3, "right"), (-5, -10, "right"),
                    (0, 9, "center"), (0, -14, "center"))  # deslocamento em pontos e alinhamento


def _rotular_sem_sobrepor(ax: plt.Axes, pontos: list[tuple[float, float, str]],
                          obstaculos: np.ndarray | None = None) -> None:
    """Escreve cada nome ao lado do seu ponto sem que os nomes se cruzem ou saiam do gráfico.

    Mede a caixa de cada nome em cada posição de _POSICOES_ROTULO e escolhe a
    combinação com menos sobreposições, depois menos nomes para fora, depois
    menos pontos de obstaculos cobertos (coordenadas de dados, uma linha por
    ponto; ex.: os outros pontos de uma dispersão), depois as posições mais à
    frente na lista. Com até 6 nomes testa todas as combinações; acima disso
    (6^7 combinações já levam vários segundos), escolhe nome a nome e depois
    revê cada escolha com as outras fixas, até nada melhorar (no máximo 5 rodadas).
    """
    if not pontos:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    area = ax.get_window_extent(renderer)
    tela = (ax.transData.transform(np.asarray(obstaculos, dtype=float))
            if obstaculos is not None and len(obstaculos) else np.empty((0, 2)))

    def cobre(caixa) -> int:
        dentro = ((tela[:, 0] >= caixa.x0) & (tela[:, 0] <= caixa.x1)
                  & (tela[:, 1] >= caixa.y0) & (tela[:, 1] <= caixa.y1))
        return int(dentro.sum())

    def anotar(x, y, texto, posicao):
        dx, dy, alinhamento = _POSICOES_ROTULO[posicao]
        return ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points",
                           ha=alinhamento, fontsize=7.5, color=TINTA_SECUNDARIA, zorder=4,
                           path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")])

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

    cobertos = [[cobre(caixa) for caixa in linha] for linha in caixas]

    def custo(escolha) -> tuple[int, int, int, int]:
        escolhidas = [caixas[i][j] for i, j in enumerate(escolha)]
        cruzam = sum(a.overlaps(b) for k, a in enumerate(escolhidas) for b in escolhidas[k + 1:])
        return cruzam, sum(map(fora, escolhidas)), sum(cobertos[i][j] for i, j in enumerate(escolha)), sum(escolha)

    opcoes = range(len(_POSICOES_ROTULO))
    if len(pontos) <= 6:
        melhor = min(itertools.product(opcoes, repeat=len(pontos)), key=custo)
    else:
        melhor = []
        for i in range(len(pontos)):
            def custo_do_nome(j, i=i):
                cruzam = sum(caixas[i][j].overlaps(caixas[k][melhor[k]]) for k in range(i))
                return cruzam, fora(caixas[i][j]), cobertos[i][j], j
            melhor.append(min(opcoes, key=custo_do_nome))
        for _ in range(5):  # revê cada nome com os outros fixos
            antes = list(melhor)
            for i in range(len(pontos)):
                melhor[i] = min(opcoes, key=lambda j, i=i: custo([*melhor[:i], j, *melhor[i + 1:]]))
            if melhor == antes:
                break
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


# Rampa sequencial azul da paleta (passos 250 a 700): o passo mais claro ainda aparece no fundo branco.
AZUL_SEQUENCIAL = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
                   "#1c5cab", "#184f95", "#104281", "#0d366b")


def dano_vs_avalanche(dano: pd.DataFrame, exemplos: pd.DataFrame | None = None,
                      correlacoes: pd.DataFrame | None = None, separacao_rotulos: float = 0.05) -> plt.Figure:
    """Exp 3. Dano estrutural (queda do núcleo) x dano funcional (avalanche) de cada metabólito.

    dano: a tabela exp3_dano (colunas metabolito, grau, dano_forte, avalanche);
    só entram os metabólitos produzidos (com avalanche). A cor é o grau, numa
    rampa sequencial de um tom só em escala log (a distribuição de graus tem
    cauda pesada); os de grau maior ficam por cima. exemplos: a tabela
    exp3_exemplos (metabolito, nome_curto): esses levam o nome ao lado do ponto
    (os casos da discussão e os mais discordantes entre as duas medidas), na
    ordem da tabela; um nome cujo ponto fica a menos de separacao_rotulos (em
    fração dos eixos) de um ponto já nomeado fica de fora, para os nomes não se
    amontoarem (continua na tabela). correlacoes: a tabela exp3_correlacoes; o
    Spearman do dano estrutural com a avalanche vai numa nota embaixo do gráfico.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    estilo()
    pontos = dano.dropna(subset=["avalanche"]).sort_values("grau")
    fig, ax = plt.subplots(figsize=(LARGURA * 0.85, 3.9))
    mapa = LinearSegmentedColormap.from_list("azul", AZUL_SEQUENCIAL)
    graus = pontos["grau"].clip(lower=1)
    maior_grau = max(float(graus.max()), 2.0)
    dispersao = ax.scatter(pontos["dano_forte"], pontos["avalanche"], c=graus, cmap=mapa,
                           norm=LogNorm(vmin=1, vmax=maior_grau), s=16,
                           edgecolors="white", linewidths=0.4, zorder=3)
    barra = fig.colorbar(dispersao, ax=ax, pad=0.02, fraction=0.05)
    barra.set_label("Grau total (escala log)")
    barra.outline.set_visible(False)
    marcas = [v for v in (1, 2, 5, 10, 20, 50, 100) if v <= maior_grau]
    barra.set_ticks(marcas, labels=[str(v) for v in marcas])
    barra.ax.minorticks_off()
    ax.set_xlabel("Metabólitos que deixam o núcleo (queda do maior componente forte)")
    ax.set_ylabel("Tamanho da avalanche\n(metabólitos que deixam de ser produzidos)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.06, y=0.06)
    _eixos_lineares_com_virgula(ax)
    if correlacoes is not None and len(correlacoes):
        rho = correlacoes.set_index(["medida", "outra"])["spearman"]
        partes = [f"{texto} {num(rho[(medida, 'avalanche')])}"
                  for medida, texto in (("dano_forte", "queda do núcleo"), ("dano_eficiencia", "queda da eficiência"))
                  if (medida, "avalanche") in rho.index]
        if partes:
            fig.text(0.02, 0.0, "Spearman com a avalanche: " + " · ".join(partes), ha="left", va="bottom",
                     fontsize=7.5, color=TINTA_SECUNDARIA)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if exemplos is not None and len(exemplos):
        por_codigo = pontos.set_index("metabolito")
        escala_x = max(float(pontos["dano_forte"].max()), 1.0)
        escala_y = max(float(pontos["avalanche"].max()), 1.0)
        rotulos: list[tuple[float, float, str]] = []
        for m, nome in zip(exemplos["metabolito"], exemplos["nome_curto"]):
            if m not in por_codigo.index:
                continue
            x, y = float(por_codigo.loc[m, "dano_forte"]), float(por_codigo.loc[m, "avalanche"])
            if all(math.hypot((x - a) / escala_x, (y - b) / escala_y) >= separacao_rotulos for a, b, _ in rotulos):
                rotulos.append((x, y, str(nome)))
        _rotular_sem_sobrepor(ax, rotulos, pontos[["dano_forte", "avalanche"]].to_numpy())
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
