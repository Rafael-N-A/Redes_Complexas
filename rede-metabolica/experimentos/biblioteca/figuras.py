"""Figuras do artigo, com o mesmo estilo em todas.

Cada função recebe as tabelas que os scripts gravaram em resultados/tabelas/
e devolve uma figura do matplotlib. Quem grava é o script, com
config.arquivo_figura(...) e salvar(); a biblioteca não decide nada sobre arquivos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402


def estilo() -> None:
    """Ajusta o rcParams uma vez: fonte, tamanho, sem bordas de cima e da direita."""
    raise NotImplementedError


def salvar(fig: plt.Figure, destino: Path) -> Path:
    """Grava fig em destino (dpi = config.DPI_FIGURA, bbox justo), fecha a figura
    e devolve o caminho. Cria a pasta se faltar."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Uma função por figura do artigo
# ---------------------------------------------------------------------------
def distribuicao_graus(dist: pd.DataFrame) -> plt.Figure:
    """Exp 0. Entrada, saída e total em log-log (tabela de metricas.distribuicao_graus)."""
    raise NotImplementedError


def curvas_robustez(curvas: pd.DataFrame, metrica: str = "maior_fraco") -> plt.Figure:
    """Exp 1 e 4. Uma curva por estratégia: metrica x fracao_removida.

    Na estratégia aleatória, a faixa de ±1 desvio padrão (<metrica>_dp).
    """
    raise NotImplementedError


def avalanche_vs_grau(avalanches: pd.DataFrame) -> plt.Figure:
    """Exp 3. Tamanho da avalanche x grau das substâncias escolhidas e dos controles."""
    raise NotImplementedError


def distribuicao_avalanches(tamanhos: pd.Series) -> plt.Figure:
    """Exp 3. Distribuição dos tamanhos de avalanche de todos os metabólitos (log-log)."""
    raise NotImplementedError


def curvas_roc(rocs: dict[str, pd.DataFrame], aucs: dict[str, float]) -> plt.Figure:
    """Exp 2. Uma curva ROC por ranking, com o AUC na legenda e a diagonal de referência."""
    raise NotImplementedError


def indices_por_rede(indices: pd.DataFrame) -> plt.Figure:
    """Exp 4. Índice R por rede (eco, bsu, mge, nulos) e estratégia."""
    raise NotImplementedError
