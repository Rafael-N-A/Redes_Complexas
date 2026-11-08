"""Testes da cascata (experimentos/biblioteca/cascata.py).

Três tipos de teste:
  1. redes de brinquedo com a resposta feita à mão;
  2. comparação com uma implementação independente e ingênua (ponto fixo
     por rodadas, direto da tabela de reações, sem passar pelo
     grafo_bipartido nem pela fila com contadores), em centenas de redes
     sorteadas;
  3. propriedades que valem em qualquer rede: monotonia, as duas versões
     (com e sem moeda) darem o mesmo resultado, avalanches_todos bater com
     a cascata metabólito a metabólito.

As regras básicas (E dos substratos, OU das produtoras, ciclo reversível,
moeda como fonte) estão em test_regras.py.
"""
from __future__ import annotations

import random

import networkx as nx
import pandas as pd

import config
import contrato
from experimentos.biblioteca import cascata, grafo

ATP, ADP, NAD = "C00002", "C00008", "C00003"
ACP = "C00229"
MOEDA_SORTEIO = [ATP, ADP, NAD]


def tabela_reacoes(*reacoes) -> pd.DataFrame:
    """Tabela "reacoes" do contrato. Cada item: (codigo, substratos, produtos, reversivel)."""
    linhas = [
        {"reacao": codigo, "substratos": list(s), "produtos": list(p), "reversivel": rev,
         "genes": ["g_" + codigo], "vias": ["eco00000"], "n_ocorrencias": 1,
         "conflito_reversibilidade": False, "orientacao_invertida": False, "lados_variam": False}
        for codigo, s, p, rev in reacoes
    ]
    return pd.DataFrame(linhas, columns=contrato.COLUNAS["reacoes"])


# ---------------------------------------------------------------------------
# 1. Redes de brinquedo
# ---------------------------------------------------------------------------
def test_cadeia_linear():
    """X -> A -> B -> C: remover um elo derruba tudo o que vem depois dele."""
    B = grafo.grafo_bipartido(tabela_reacoes(
        ("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], False), ("R3", ["B"], ["C"], False)))
    assert cascata.fontes_externas(B) == {"X"}
    assert cascata.cascata(B, ["A"]) == {"B", "C"}
    assert cascata.cascata(B, ["X"]) == {"A", "B", "C"}
    assert cascata.cascata(B, ["C"]) == set()
    assert cascata.cascata(B, reacoes_removidas=["R2"]) == {"B", "C"}


def test_removidos_nao_entram_na_avalanche():
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], False)))
    assert "A" not in cascata.cascata(B, ["A"])
    assert "X" not in cascata.cascata(B, ["X"])


def test_reacao_sem_substrato_ativa_no_inicio():
    """R0 só gasta moeda (some na versão sem moeda): ativa sozinha e produz A."""
    reacoes = tabela_reacoes(("R0", [ATP], ["A"], False), ("R1", ["A"], ["B"], False))
    B = grafo.grafo_bipartido(reacoes)
    disponiveis, ativas = cascata.disponiveis(B, cascata.fontes_externas(B))
    assert {"A", "B"} <= disponiveis
    assert {"R0:ida", "R1:ida"} <= ativas
    assert cascata.cascata(B, reacoes_removidas=["R0"]) == {"A", "B"}


def test_remover_reacao_reversivel_tira_os_dois_sentidos():
    """X -> A <-> B -> C: sem R2, B e C se perdem; nenhum sentido de R2 ativa."""
    B = grafo.grafo_bipartido(tabela_reacoes(
        ("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], True), ("R3", ["B"], ["C"], False)))
    assert cascata.cascata(B, reacoes_removidas=["R2"]) == {"B", "C"}
    _, ativas = cascata.disponiveis(B, cascata.fontes_externas(B), reacoes_removidas=["R2"])
    assert not {"R2:ida", "R2:volta"} & ativas


def test_grupo_de_origem_e_fonte():
    """N <-> M -> P, sem nada que produza N ou M de fora: os dois são fontes.

    É o caso de nitrato <-> nitrito e de sulfato <-> APS na E. coli: nutrientes
    que entram por uma primeira reação reversível. Consequência (documentada):
    remover a reação entre eles não derruba M, porque M também é fonte.
    """
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["N"], ["M"], True), ("R2", ["M"], ["P"], False)))
    assert cascata.fontes_externas(B) == {"N", "M"}
    assert cascata.cascata(B, reacoes_removidas=["R2"]) == {"P"}
    assert cascata.cascata(B, reacoes_removidas=["R1"]) == set()
    assert cascata.cascata(B, ["M"]) == {"P"}


def test_grupo_com_entrada_de_fora_nao_e_fonte():
    """X -> A <-> B: o grupo {A, B} depende de X, então só X é fonte."""
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], True)))
    assert cascata.fontes_externas(B) == {"X"}


def test_carregador_em_ciclo_fechado_nunca_liga():
    """A + Q -> B e B -> Q: Q só sai de B, B só sai de Q (com A, que vem de fora).

    Q e B nunca ficam disponíveis, então não contam como perdidos. Era o caso
    do ACP antes de ele entrar na lista de moeda (ver o teste seguinte).
    """
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["A", "Q"], ["B"], False), ("R2", ["B"], ["Q"], False)))
    disponiveis, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
    assert disponiveis == {"A"}
    assert cascata.cascata(B, ["A"]) == set()


def test_acp_e_moeda_e_a_sintese_de_acidos_graxos_liga():
    """Acetil-CoA -> malonil-CoA; malonil-CoA + ACP -> malonil-ACP; malonil-ACP -> acil-ACP + ACP.

    Com o ACP na lista de moeda, a síntese liga nas duas versões, e remover a
    acetil-CoA derruba a cadeia toda.
    """
    assert ACP in config.METABOLITOS_MOEDA
    reacoes = tabela_reacoes(
        ("R00742", ["C00024"], ["C00083"], False),
        ("R01626", ["C00083", ACP], ["C01209"], False),
        ("R99999", ["C01209"], ["C00173", ACP], False),
    )
    for remover_moeda in (True, False):
        B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
        disponiveis, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
        assert {"C01209", "C00173"} <= disponiveis, f"remover_moeda={remover_moeda}"
        assert cascata.cascata(B, ["C00024"]) == {"C00083", "C01209", "C00173"}


def test_moeda_nao_quebra_grupo_de_origem():
    """N + ATP <-> M: na versão com moeda, a aresta vinda do ATP não conta como
    entrada de fora, e N e M continuam fontes, como na versão sem moeda."""
    reacoes = tabela_reacoes(("R1", ["N", ATP], ["M", ADP], True), ("R2", [ADP], [ATP], False))
    fontes = {}
    for remover_moeda in (True, False):
        B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
        fontes[remover_moeda] = cascata.fontes_externas(B) - set(config.METABOLITOS_MOEDA)
    assert fontes[True] == fontes[False] == {"N", "M"}


def test_fontes_explicitas():
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], False)))
    disponiveis, _ = cascata.disponiveis(B, {"A"})
    assert disponiveis == {"A", "B"}
    assert cascata.cascata(B, ["A"], fontes={"A"}) == {"B"}
    assert cascata.cascata(B, ["X"], fontes={"A"}) == set()  # X nem estava disponível


def test_remover_o_que_nao_existe_nao_muda_nada():
    B = grafo.grafo_bipartido(tabela_reacoes(("R1", ["X"], ["A"], False)))
    assert cascata.cascata(B, ["C99999"]) == set()
    assert cascata.cascata(B, reacoes_removidas=["R99999"]) == set()


def test_nao_altera_o_grafo():
    B = grafo.grafo_bipartido(tabela_reacoes(
        ("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], True), ("R3", ["A", "B"], ["C"], False)))
    nos, arestas = dict(B.nodes(data=True)), set(B.edges)
    cascata.cascata(B, ["A"], ["R2"])
    cascata.avalanches_todos(B)
    assert dict(B.nodes(data=True)) == nos and set(B.edges) == arestas


def test_avalanches_todos_formato():
    """Uma linha por metabólito disponível na rede intacta.

    Q e D formam um ciclo fechado (Q só sai de D, D precisa de Q) e nunca
    ficam disponíveis, então não entram na série.
    """
    B = grafo.grafo_bipartido(tabela_reacoes(
        ("R1", ["X"], ["A"], False), ("R2", ["A"], ["B"], False),
        ("R3", ["A", "Q"], ["D"], False), ("R4", ["D"], ["Q"], False)))
    serie = cascata.avalanches_todos(B)
    assert serie.name == "avalanche" and serie.index.name == "metabolito"
    assert serie.to_dict() == {"A": 1, "B": 0, "X": 2}
    assert pd.api.types.is_integer_dtype(serie)


# ---------------------------------------------------------------------------
# 2. Implementação independente
# ---------------------------------------------------------------------------
def rede_sorteada(semente: int, n_metabolitos: int = 12, n_reacoes: int = 14) -> pd.DataFrame:
    """Rede pequena e bagunçada: 0 a 3 substratos (raramente 0), 1 a 3 produtos,
    40% reversíveis, moeda às vezes, e às vezes o mesmo metabólito dos dois lados.

    As proporções foram escolhidas para que boa parte das redes tenha grupos
    de origem (fontes com produtor), ciclos que nunca ligam e reações sem
    substrato: os casos em que uma implementação errada se denunciaria.
    """
    rng = random.Random(semente)
    pool = [f"M{i:02d}" for i in range(n_metabolitos)] + MOEDA_SORTEIO
    reacoes = []
    for i in range(n_reacoes):
        substratos = rng.sample(pool, rng.choices([0, 1, 2, 3], weights=[5, 50, 30, 15])[0])
        produtos = rng.sample(pool, rng.choices([1, 2, 3], weights=[60, 30, 10])[0])
        reacoes.append((f"R{i:03d}", substratos, produtos, rng.random() < 0.4))
    return tabela_reacoes(*reacoes)


def _sentidos(reacoes: pd.DataFrame, remover_moeda: bool) -> list[tuple[str, set[str], set[str]]]:
    moeda = set(config.METABOLITOS_MOEDA) if remover_moeda else set()
    sentidos = []
    for r in reacoes.to_dict("records"):
        s, p = set(r["substratos"]) - moeda, set(r["produtos"]) - moeda
        sentidos.append((r["reacao"], s, p))
        if r["reversivel"]:
            sentidos.append((r["reacao"], p, s))
    return sentidos


def disponiveis_ingenuo(reacoes, remover_moeda, fontes, metabolitos_removidos=(), reacoes_removidas=()):
    """Ponto fixo por rodadas: repete "toda reação com todos os substratos produz"
    até nada mudar. Lento, mas sem nenhuma estrutura em comum com a fila."""
    removidos, fora = set(metabolitos_removidos), set(reacoes_removidas)
    sentidos = _sentidos(reacoes, remover_moeda)
    disponivel = set(fontes) - removidos
    mudou = True
    while mudou:
        mudou = False
        for codigo, s, p in sentidos:
            if codigo not in fora and s <= disponivel:
                novos = p - disponivel - removidos
                if novos:
                    disponivel |= novos
                    mudou = True
    return disponivel


def fontes_ingenuas(B: nx.DiGraph) -> set[str]:
    """Outra caracterização do componente forte de origem: m está num deles
    se todo nó que chega em m também é alcançável a partir de m. Calculada
    sem a moeda, que entra à parte."""
    moeda = set(config.METABOLITOS_MOEDA)
    H = B.subgraph(n for n in B if n not in moeda)
    fontes = {m for m, tipo in H.nodes(data="tipo")
              if tipo == "metabolito" and nx.ancestors(H, m) <= nx.descendants(H, m)}
    return fontes | {m for m in moeda if m in B}


def test_fontes_iguais_a_definicao_independente():
    for semente in range(300):
        reacoes = rede_sorteada(semente)
        for remover_moeda in (True, False):
            B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
            assert cascata.fontes_externas(B) == fontes_ingenuas(B), f"semente={semente}"


def test_disponiveis_e_cascata_iguais_ao_ponto_fixo_ingenuo():
    for semente in range(300):
        rng = random.Random(10_000 + semente)
        reacoes = rede_sorteada(semente)
        for remover_moeda in (True, False):
            B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
            fontes = cascata.fontes_externas(B)
            intactos = disponiveis_ingenuo(reacoes, remover_moeda, fontes)
            assert cascata.disponiveis(B, fontes)[0] == intactos, f"semente={semente}"
            metabolitos = sorted(grafo.nos_metabolito(B))
            codigos = sorted(reacoes["reacao"])
            for _ in range(5):
                mets = rng.sample(metabolitos, rng.randint(0, min(3, len(metabolitos))))
                reacs = rng.sample(codigos, rng.randint(0, 3))
                esperado = intactos - disponiveis_ingenuo(reacoes, remover_moeda, fontes, mets, reacs) - set(mets)
                assert cascata.cascata(B, mets, reacs) == esperado, f"semente={semente}, {mets}, {reacs}"


def test_reacoes_ativas_sao_as_que_tem_todos_os_substratos():
    for semente in range(100):
        B = grafo.grafo_bipartido(rede_sorteada(semente))
        disponiveis, ativas = cascata.disponiveis(B, cascata.fontes_externas(B))
        esperadas = {r for r in grafo.nos_reacao(B) if set(B.predecessors(r)) <= disponiveis}
        assert ativas == esperadas, f"semente={semente}"
        assert {p for r in ativas for p in B.successors(r)} <= disponiveis


# ---------------------------------------------------------------------------
# 3. Propriedades
# ---------------------------------------------------------------------------
def test_monotonia():
    """Remover mais nunca recupera nada: perdidos(S) está contido em perdidos(S + T)."""
    for semente in range(200):
        rng = random.Random(20_000 + semente)
        B = grafo.grafo_bipartido(rede_sorteada(semente))
        metabolitos = sorted(grafo.nos_metabolito(B))
        codigos = sorted({d["reacao"] for _, d in B.nodes(data=True) if d["tipo"] == "reacao"})
        s, t = rng.sample(metabolitos, 1), rng.sample(metabolitos, 2)
        r, q = rng.sample(codigos, 1), rng.sample(codigos, 2)
        assert cascata.cascata(B, s, r) - set(t) <= cascata.cascata(B, s + t, r + q), f"semente={semente}"


def test_mesmo_resultado_com_e_sem_moeda():
    """A premissa de config.VERSAO_CASCATA: fora a própria moeda, as duas versões
    têm as mesmas fontes, os mesmos disponíveis e as mesmas avalanches."""
    moeda = set(config.METABOLITOS_MOEDA)
    for semente in range(300):
        reacoes = rede_sorteada(semente)
        resultado = {}
        for remover_moeda in (True, False):
            B = grafo.grafo_bipartido(reacoes, remover_moeda=remover_moeda)
            fontes = cascata.fontes_externas(B)
            avalanches = cascata.avalanches_todos(B)
            resultado[remover_moeda] = (
                fontes - moeda,
                cascata.disponiveis(B, fontes)[0] - moeda,
                avalanches[[m for m in avalanches.index if m not in moeda]].to_dict(),
                {c: cascata.cascata(B, reacoes_removidas=[c]) - moeda for c in reacoes["reacao"]},
            )
        assert resultado[True] == resultado[False], f"semente={semente}"


def test_avalanches_todos_igual_a_cascata_um_a_um():
    for semente in range(100):
        B = grafo.grafo_bipartido(rede_sorteada(semente))
        intactos, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
        esperado = {m: len(cascata.cascata(B, [m])) for m in sorted(intactos)}
        assert cascata.avalanches_todos(B).to_dict() == esperado, f"semente={semente}"


def test_perdidos_estavam_disponiveis():
    for semente in range(100):
        rng = random.Random(30_000 + semente)
        B = grafo.grafo_bipartido(rede_sorteada(semente))
        intactos, _ = cascata.disponiveis(B, cascata.fontes_externas(B))
        mets = rng.sample(sorted(grafo.nos_metabolito(B)), 2)
        perdidos = cascata.cascata(B, mets)
        assert perdidos <= intactos and not perdidos & set(mets), f"semente={semente}"
