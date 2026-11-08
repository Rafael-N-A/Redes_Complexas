"""Cascata de falhas na rede bipartida (Smart, Amaral e Ottino, 2008).

Regras:
    - uma reação falha quando perde qualquer um de seus substratos (E);
    - um metabólito falha quando todas as reações que o produzem falharam (OU);
    - as fontes externas (ver fontes_externas) estão sempre disponíveis, a
      não ser que sejam elas as removidas.

Implementação: expansão a partir das fontes.
Em vez de partir de "tudo vivo" e ir derrubando, parte só das fontes e vai
ativando as reações que têm todos os substratos disponíveis, até nada mudar.
O dano é o que estava disponível na rede intacta e deixa de estar.

Por quê: toda reação reversível A <-> B cria um ciclo em que A "produz" B e
B "produz" A. Partindo de tudo vivo, esse ciclo se sustenta sozinho mesmo
quando A e B perderam todas as entradas de fora, e a cascata para antes da
hora. A expansão não tem esse problema. Os testes em testes/test_regras.py
cobrem exatamente esse caso.

As fontes são calculadas sempre na rede intacta: uma remoção nunca cria
fontes novas. É isso que impede o ciclo A <-> B de virar fonte depois de
perder a entrada.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import networkx as nx
import pandas as pd

import config


def _metabolitos(B: nx.DiGraph) -> set[str]:
    return {n for n, tipo in B.nodes(data="tipo") if tipo == "metabolito"}


def fontes_externas(B: nx.DiGraph) -> set[str]:
    """Fontes sempre disponíveis: o que a rede não consegue produzir a partir do
    resto dela, mais os metabólitos-moeda (config.METABOLITOS_MOEDA) presentes em B.

    "O que a rede não consegue produzir" são os metabólitos dos componentes
    fortemente conexos de origem de B, isto é, os que não recebem nenhuma
    aresta de fora do próprio componente. Isso inclui dois casos:
      - metabólito sem nenhuma reação que o produza (componente de um nó só);
      - grupo de metabólitos que só se produzem uns aos outros, como
        nitrato <-> nitrito ou sulfato <-> APS <-> PAPS. São nutrientes que
        entram por uma primeira reação reversível: sem um produtor de fora, a
        expansão nunca chegaria a eles, embora venham do meio.
    Um grupo que recebe alguma entrada de fora (o A <-> B alimentado por X do
    teste do ciclo reversível) não é fonte: ele depende dessa entrada.

    Por que a moeda entra: na versão com moeda, ATP e ADP só se produzem um a
    partir do outro e o ciclo tem entrada de fora (os substratos das reações
    que gastam ATP), então não é um grupo de origem. Sem a moeda entre as
    fontes, a expansão não chega a nenhum dos dois e trava já na rede intacta.
    Na versão sem moeda eles nem estão em B, então a regra não muda nada.

    Os grupos de origem são procurados sem os nós de moeda: como ela está
    sempre disponível, uma aresta vinda dela não é dependência de nada. Assim
    as fontes (fora a própria moeda) são as mesmas nas duas versões, e a
    cascata também (o que config.VERSAO_CASCATA pressupõe).
    """
    moeda = set(config.METABOLITOS_MOEDA)
    sem_a_moeda = B.subgraph(n for n in B if n not in moeda)
    condensado = nx.condensation(sem_a_moeda)
    fontes: set[str] = set()
    for componente in condensado:
        if condensado.in_degree(componente) == 0:
            fontes |= {n for n in condensado.nodes[componente]["members"]
                       if B.nodes[n].get("tipo") == "metabolito"}
    return fontes | (moeda & _metabolitos(B))


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
      - ao tirar um metabólito da fila, decrementa o contador das reações que
        o consomem; a reação que chega a zero ativa e põe seus produtos na fila.
    Reação sem nenhum substrato (por exemplo, só moeda, na versão sem moeda)
    ativa logo no início. Fontes que não são metabólitos de B são ignoradas.
    """
    removidos = set(metabolitos_removidos)
    reacoes_fora = set(reacoes_removidas)

    faltando: dict[str, int] = {}
    for no, dados in B.nodes(data=True):
        if dados.get("tipo") == "reacao" and dados["reacao"] not in reacoes_fora:
            faltando[no] = B.in_degree(no)

    disponivel: set[str] = set()
    ativas: set[str] = set()
    fila: deque[str] = deque()

    def liberar(metabolito: str) -> None:
        if metabolito not in disponivel and metabolito not in removidos:
            disponivel.add(metabolito)
            fila.append(metabolito)

    def ativar(no_reacao: str) -> None:
        ativas.add(no_reacao)
        for produto in B.successors(no_reacao):
            liberar(produto)

    for fonte in fontes:
        if fonte in B and B.nodes[fonte].get("tipo") == "metabolito":
            liberar(fonte)
    for no_reacao, n in faltando.items():
        if n == 0:
            ativar(no_reacao)

    while fila:
        metabolito = fila.popleft()
        for no_reacao in B.successors(metabolito):
            if no_reacao in faltando:
                faltando[no_reacao] -= 1
                if faltando[no_reacao] == 0:
                    ativar(no_reacao)
    return disponivel, ativas


def _perdidos(B: nx.DiGraph, fontes: set[str], intactos: set[str],
              metabolitos_removidos: Iterable[str], reacoes_removidas: Iterable[str]) -> set[str]:
    removidos = set(metabolitos_removidos)
    restantes, _ = disponiveis(B, fontes, removidos, reacoes_removidas)
    return intactos - restantes - removidos


def cascata(B: nx.DiGraph, metabolitos_removidos: Iterable[str] = (),
            reacoes_removidas: Iterable[str] = (),
            fontes: Iterable[str] | None = None) -> set[str]:
    """Metabólitos perdidos pela remoção (o tamanho da avalanche é o len).

    Perdidos = disponíveis na rede intacta - disponíveis após a remoção,
    sem contar os próprios metabólitos removidos. Sem fontes, usa fontes_externas(B).
    O que já não era produzido na rede intacta nunca conta como perdido.
    """
    fontes = fontes_externas(B) if fontes is None else set(fontes)
    intactos, _ = disponiveis(B, fontes)
    return _perdidos(B, fontes, intactos, metabolitos_removidos, reacoes_removidas)


def avalanches_todos(B: nx.DiGraph, fontes: Iterable[str] | None = None) -> pd.Series:
    """Tamanho da avalanche ao remover cada metabólito sozinho (índice = C#####).

    Base da distribuição de tamanhos de avalanche do exp3. Só entram os
    metabólitos disponíveis na rede intacta: remover um que já não era
    produzido não muda nada e só encheria a distribuição de zeros.
    A rede intacta é calculada uma vez só, e não a cada remoção.
    """
    fontes = fontes_externas(B) if fontes is None else set(fontes)
    intactos, _ = disponiveis(B, fontes)
    tamanhos = {m: len(_perdidos(B, fontes, intactos, [m], ())) for m in sorted(intactos)}
    serie = pd.Series(tamanhos, name="avalanche", dtype=int)
    serie.index.name = "metabolito"
    return serie
