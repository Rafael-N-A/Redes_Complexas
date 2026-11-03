"""Cascata de falhas na rede bipartida (Smart, Amaral e Ottino, 2008).

Regras:
    - uma reação falha quando perde qualquer um de seus substratos (E);
    - um metabólito falha quando todas as reações que o produzem falharam (OU);
    - metabólitos sem produtor dentro da rede são sempre disponíveis.

Implementação: expansão a partir das fontes.
Em vez de partir de "tudo vivo" e ir derrubando, parta só das fontes e vá
ativando as reações que têm todos os substratos disponíveis, até nada mudar.
O dano é o que estava disponível na rede intacta e deixa de estar.

Por quê: toda reação reversível A <-> B cria um ciclo em que A "produz" B e
B "produz" A. Partindo de tudo vivo, esse ciclo se sustenta sozinho mesmo
quando A e B perderam todas as entradas de fora, e a cascata para antes da
hora. A expansão não tem esse problema. Os testes em testes/test_regras.py
cobrem exatamente esse caso.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd

import config


def fontes_externas(B: nx.DiGraph) -> set[str]:
    """Fontes sempre disponíveis: metabólitos sem nenhuma reação que os produza,
    mais os metabólitos-moeda (config.METABOLITOS_MOEDA) presentes em B.

    Por que a moeda entra: na versão com moeda, ATP e ADP
    só se produzem um a partir do outro. Sem uma entrada externa, a expansão
    não chega a nenhum dos dois e trava já na rede intacta. Na versão sem
    moeda eles nem estão em B, então a regra não muda nada.
    """
    raise NotImplementedError


def disponiveis(B: nx.DiGraph, fontes: Iterable[str],
                metabolitos_removidos: Iterable[str] = (),
                reacoes_removidas: Iterable[str] = ()) -> tuple[set[str], set[str]]:
    """Expansão: metabólitos produzíveis a partir das fontes e nós de reação ativos.

    metabolitos_removidos: C##### que nunca ficam disponíveis (mesmo se forem fontes)
    reacoes_removidas:     códigos R##### que nunca ativam (os dois sentidos)
    Retorna (metabolitos_disponiveis, nos_de_reacao_ativos).

    Implementação (uma passada pelas arestas, porque os Exps 2 e 3
    chamam esta função ~1.500 vezes):
      - para cada nó de reação, um contador de substratos ainda faltando;
      - uma fila com os metabólitos que acabaram de ficar disponíveis;
      - ao tirar um metabólito da fila, decremente o contador das reações que
        o consomem; a reação que chegar a zero ativa e põe seus produtos na fila.
    Reação sem nenhum substrato (por exemplo, só moeda, na versão sem moeda)
    ativa logo no início.
    """
    raise NotImplementedError


def cascata(B: nx.DiGraph, metabolitos_removidos: Iterable[str] = (),
            reacoes_removidas: Iterable[str] = (),
            fontes: Iterable[str] | None = None) -> set[str]:
    """Metabólitos perdidos pela remoção (o tamanho da avalanche é o len).

    Perdidos = disponíveis na rede intacta - disponíveis após a remoção,
    sem contar os próprios metabólitos removidos. Sem fontes, usa fontes_externas(B).
    """
    raise NotImplementedError


def avalanches_todos(B: nx.DiGraph, fontes: Iterable[str] | None = None) -> pd.Series:
    """Tamanho da avalanche ao remover cada metabólito sozinho (índice = C#####).

    Base da distribuição de tamanhos de avalanche do exp3.
    """
    raise NotImplementedError
