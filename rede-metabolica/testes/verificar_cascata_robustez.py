"""Verifica a cascata e a robustez nos dados reais de um organismo.

Os testes (pytest) usam redes pequenas e sorteadas. Este script faz o mesmo
tipo de conferência na rede real, em blocos:

  1. Fontes e rede intacta: quem é fonte, quanto da rede a expansão alcança.
  2. Cascata contra uma implementação ingênua (ponto fixo por rodadas, direto
     da tabela de reações). FALHOU aqui é erro no código.
  3. Propriedades da cascata: as duas versões (com e sem moeda) dão o mesmo
     resultado, monotonia, o grafo não muda. FALHOU aqui é erro no código.
  4. Plausibilidade biológica da cascata (só para a E. coli) e o contraste
     com a abordagem "tudo vivo". ATENÇÃO aqui pede investigação.
  5. Robustez contra uma implementação ingênua (cada passo refeito do zero
     com a NetworkX). FALHOU aqui é erro no código.
  6. Robustez completa (todas as estratégias), com o tempo e o que a teoria
     prevê (Albert, Jeong e Barabási, 2000). ATENÇÃO aqui pede investigação.

As implementações ingênuas são as mesmas dos testes (testes/test_cascata.py
e testes/test_robustez.py).

Uso, na raiz do projeto:
    python -m testes.verificar_cascata_robustez --org eco
    python -m testes.verificar_cascata_robustez --org eco --rep 100   (Exp 1 de verdade, ~2 min a mais)
Sai com código 1 se algum item dos blocos 2, 3 ou 5 falhar.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import cascata, grafo, metricas, robustez
from testes.test_cascata import disponiveis_ingenuo, fontes_ingenuas
from testes.test_robustez import simulacao_ingenua
from testes.verificar_grafo import Placar

GLUTAMATO, HEME, OXALOACETATO, ASPARTATO = "C00025", "C00032", "C00036", "C00049"
ACETIL_COA, MALONIL_COA, PALMITOIL_ACP = "C00024", "C00083", "C05764"
CORISMATO, TRIPTOFANO, FENILALANINA, TIROSINA = "C00251", "C00078", "C00079", "C00082"
D_FENILALANINA = "C02265"
ACETIL_COA_CARBOXILASE, GLUTAMIL_TRNA_REDUTASE = "R00742", "R04109"


def _rotulo(m: str, nomes: dict[str, str]) -> str:
    return f"{m} {nomes.get(m, '?')}"


def _cronometro(inicio: float) -> str:
    return f"({time.perf_counter() - inicio:.1f} s)"


# ---------------------------------------------------------------------------
# Cascata
# ---------------------------------------------------------------------------
def fontes_e_rede_intacta(placar: Placar, B: nx.DiGraph, nomes: dict[str, str]) -> tuple[set, set]:
    print("\n== 1. Fontes e rede intacta (sem moeda)")
    inicio = time.perf_counter()
    fontes = cascata.fontes_externas(B)
    moeda = set(config.METABOLITOS_MOEDA)
    sem_produtor = {m for m in fontes - moeda if B.in_degree(m) == 0}
    em_grupo = fontes - moeda - sem_produtor
    print(f"  [info] {len(fontes)} fontes: {len(sem_produtor)} sem produtor, {len(em_grupo)} em grupos "
          f"de origem, {len(fontes & moeda)} de moeda")
    print(f"  [info] em grupos de origem: {', '.join(sorted(nomes.get(m, m) for m in em_grupo)[:12])}, ...")
    placar.checar("fontes = definição independente (tudo que chega nelas sai delas)",
                  fontes == fontes_ingenuas(B))
    intactos, ativas = cascata.disponiveis(B, fontes)
    metabolitos, reacoes = grafo.nos_metabolito(B), grafo.nos_reacao(B)
    cobertura = len(intactos) / len(metabolitos)
    placar.avisar("a rede intacta produz pelo menos 95% dos metabólitos", cobertura >= 0.95,
                  f"{len(intactos)} de {len(metabolitos)}, {cobertura:.1%}; "
                  f"{len(ativas)} de {len(reacoes)} nós de reação ativos")
    nunca = sorted(metabolitos - intactos)
    print(f"  [info] nunca disponíveis: {', '.join(nomes.get(m, m) for m in nunca[:10])}"
          + (", ..." if len(nunca) > 10 else "") + f" {_cronometro(inicio)}")
    return fontes, intactos


def contra_ingenua(placar: Placar, B: nx.DiGraph, reacoes: pd.DataFrame, fontes: set, intactos: set,
                   amostra: int) -> pd.Series:
    print("\n== 2. Cascata contra a implementação ingênua (sem moeda)")
    inicio = time.perf_counter()
    rng = random.Random(config.SEMENTE)
    placar.checar("rede intacta", intactos == disponiveis_ingenuo(reacoes, True, fontes))

    def esperado(mets=(), reacs=()):
        return intactos - disponiveis_ingenuo(reacoes, True, fontes, mets, reacs) - set(mets)

    erradas = [m for m in config.SUBSTANCIAS_EXP3 if cascata.cascata(B, [m]) != esperado([m])]
    placar.checar("remoção de cada substância do Exp 3", not erradas, ", ".join(erradas))
    # Redes pequenas (mge: 160 metabólitos produzidos, 139 reações) têm menos que a amostra.
    mets = rng.sample(sorted(intactos), min(amostra, len(intactos)))
    erradas = [m for m in mets if cascata.cascata(B, [m]) != esperado([m])]
    placar.checar(f"{len(mets)} metabólitos sorteados, um de cada vez", not erradas, ", ".join(erradas[:5]))
    codigos = sorted(reacoes["reacao"])
    reacs = rng.sample(codigos, min(amostra, len(codigos)))
    erradas = [r for r in reacs if cascata.cascata(B, reacoes_removidas=[r]) != esperado(reacs=[r])]
    placar.checar(f"{len(reacs)} reações sorteadas, uma de cada vez", not erradas, ", ".join(erradas[:5]))
    conjuntos = [rng.sample(codigos, 5) for _ in range(amostra // 4)]
    erradas = [c for c in conjuntos if cascata.cascata(B, reacoes_removidas=c) != esperado(reacs=c)]
    placar.checar(f"{len(conjuntos)} conjuntos de 5 reações (como um nocaute de vários genes)", not erradas)

    avalanches = cascata.avalanches_todos(B, fontes)
    um_a_um = {m: len(cascata.cascata(B, [m], fontes=fontes)) for m in sorted(intactos)}
    placar.checar("avalanches_todos = cascata metabólito a metabólito (todos)",
                  avalanches.to_dict() == um_a_um, f"{len(avalanches)} metabólitos")
    print(f"  {_cronometro(inicio)}")
    return avalanches


def propriedades(placar: Placar, reacoes: pd.DataFrame, B: nx.DiGraph, fontes: set, intactos: set,
                 avalanches: pd.Series, amostra: int) -> None:
    print("\n== 3. Propriedades da cascata")
    inicio = time.perf_counter()
    moeda = set(config.METABOLITOS_MOEDA)
    Bc = grafo.grafo_bipartido(reacoes, remover_moeda=False)
    fontes_c = cascata.fontes_externas(Bc)
    intactos_c, _ = cascata.disponiveis(Bc, fontes_c)
    placar.checar("com e sem moeda: as mesmas fontes (fora a moeda)", fontes - moeda == fontes_c - moeda)
    placar.checar("com e sem moeda: os mesmos disponíveis na rede intacta", intactos - moeda == intactos_c - moeda)
    av_c = cascata.avalanches_todos(Bc, fontes_c)
    placar.checar("com e sem moeda: as mesmas avalanches",
                  avalanches.to_dict() == av_c.drop([m for m in av_c.index if m in moeda]).to_dict())
    rng = random.Random(config.SEMENTE + 1)
    codigos = sorted(reacoes["reacao"])
    reacs = rng.sample(codigos, min(amostra, len(codigos)))
    erradas = [r for r in reacs if cascata.cascata(B, reacoes_removidas=[r])
               != cascata.cascata(Bc, reacoes_removidas=[r]) - moeda]
    placar.checar(f"com e sem moeda: o mesmo dano em {len(reacs)} reações sorteadas", not erradas,
                  ", ".join(erradas[:5]))

    mets = sorted(intactos)
    quebras = []
    for _ in range(amostra):
        s, t = rng.sample(mets, 1), rng.sample(mets, 2)
        r, q = rng.sample(codigos, 1), rng.sample(codigos, 2)
        if not cascata.cascata(B, s, r) - set(t) <= cascata.cascata(B, s + t, r + q):
            quebras.append((s, t, r, q))
    placar.checar(f"monotonia: remover mais nunca recupera nada ({amostra} pares)", not quebras)
    nos, arestas = B.number_of_nodes(), B.number_of_edges()
    cascata.cascata(B, mets[:20], codigos[:20])
    placar.checar("a cascata não altera o grafo", (B.number_of_nodes(), B.number_of_edges()) == (nos, arestas))
    print(f"  {_cronometro(inicio)}")


def tudo_vivo(B: nx.DiGraph, fontes: set, removidos: set) -> set[str]:
    """A abordagem descartada: parte de tudo vivo e derruba até estabilizar."""
    vivos = grafo.nos_metabolito(B) - removidos
    ativas = {r for r in grafo.nos_reacao(B)}
    mudou = True
    while mudou:
        mudou = False
        for r in list(ativas):
            if not set(B.predecessors(r)) <= vivos:
                ativas.discard(r)
                mudou = True
        for m in list(vivos):
            if m not in fontes and not set(B.predecessors(m)) & ativas:
                vivos.discard(m)
                mudou = True
    return vivos


def plausibilidade(placar: Placar, B: nx.DiGraph, fontes: set, intactos: set, avalanches: pd.Series,
                   nomes: dict[str, str]) -> None:
    print("\n== 4. Plausibilidade biológica da cascata (sem moeda)")
    inicio = time.perf_counter()
    casos = [
        ("sem glutamato, perde-se o heme (via C5)", [GLUTAMATO], [], {HEME}),
        ("sem a glutamil-tRNA redutase (HemA), perde-se o heme", [], [GLUTAMIL_TRNA_REDUTASE], {HEME}),
        ("sem acetil-CoA, perdem-se malonil-CoA e palmitoil-ACP", [ACETIL_COA], [], {MALONIL_COA, PALMITOIL_ACP}),
        ("sem a acetil-CoA carboxilase, perdem-se malonil-CoA e palmitoil-ACP", [],
         [ACETIL_COA_CARBOXILASE], {MALONIL_COA, PALMITOIL_ACP}),
        ("sem oxaloacetato, perde-se o aspartato", [OXALOACETATO], [], {ASPARTATO}),
        ("sem corismato, perde-se o triptofano", [CORISMATO], [], {TRIPTOFANO}),
    ]
    for descricao, mets, reacs, esperados in casos:
        perdidos = cascata.cascata(B, mets, reacs, fontes=fontes)
        faltam = esperados - perdidos
        placar.avisar(descricao, not faltam,
                      f"avalanche de {len(perdidos)}" + (f"; não se perdem: {', '.join(nomes.get(m, m) for m in faltam)}"
                                                        if faltam else ""))
    # Fenilalanina e tirosina sobrevivem sem corismato, e não é erro: a
    # D-fenilalanina não tem produtor (é fonte, vem do meio), a DadA a leva a
    # fenilpiruvato, e o KEGG desenha a prefenato desidratase (R01373) como
    # reversível, então o fenilpiruvato volta a prefenato e dele sai a tirosina.
    sem_corismato = cascata.cascata(B, [CORISMATO], fontes=fontes)
    sobrevivem = [nomes.get(m, m) for m in (FENILALANINA, TIROSINA) if m in intactos and m not in sem_corismato]
    if sobrevivem:
        print(f"  [info] sem corismato, {' e '.join(sobrevivem)} continuam: a D-fenilalanina "
              f"{'é' if D_FENILALANINA in fontes else 'não é'} fonte (vem do meio) e dá fenilpiruvato, "
              "e a prefenato desidratase é reversível no KEGG")

    print("  Avalanches das substâncias do Exp 3 (expansão contra 'tudo vivo'):")
    for m, nome in config.SUBSTANCIAS_EXP3.items():
        vivos = tudo_vivo(B, fontes, {m})
        perdidos_tv = intactos - vivos - {m}
        print(f"    {nome:20s} expansão {int(avalanches.get(m, 0)):3d}   tudo vivo {len(perdidos_tv):3d}")
    rng = random.Random(config.SEMENTE + 2)
    amostra = rng.sample(sorted(intactos), 200)
    menores = sum(len(intactos - tudo_vivo(B, fontes, {m}) - {m}) < avalanches[m] for m in amostra)
    print(f"  [info] em {menores} de 200 metabólitos sorteados, 'tudo vivo' dá uma avalanche menor "
          "(ciclos reversíveis que se sustentam sozinhos)")
    print("  Maiores avalanches:")
    for m, v in avalanches.sort_values(ascending=False).head(8).items():
        print(f"    {v:3d}  {_rotulo(m, nomes)}")
    print(f"  [info] distribuição: {int((avalanches == 0).sum())} de {len(avalanches)} sem dano nenhum, "
          f"mediana {avalanches.median():.0f}, média {avalanches.mean():.2f}, máximo {avalanches.max()} "
          f"{_cronometro(inicio)}")


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------
def robustez_contra_ingenua(placar: Placar, G: nx.DiGraph) -> None:
    print("\n== 5. Robustez contra a implementação ingênua (sem moeda)")
    nutrientes = list(config.NUTRIENTES)
    casos = [("aleatoria", None), ("grau", None), ("grau_adaptativo", None),
             ("betweenness", None), ("betweenness_adaptativo", 100)]
    for estrategia, ate in casos:
        inicio = time.perf_counter()
        rng = (lambda: np.random.default_rng([config.SEMENTE, 0])) if estrategia == "aleatoria" else (lambda: None)
        curva = robustez.simular_remocao(G, estrategia, rng=rng(), nutrientes=nutrientes)
        esperada = simulacao_ingenua(G, estrategia, nutrientes, config.PASSO_EFICIENCIA,
                                     config.PASSO_BETWEENNESS, rng(), ate=ate)
        curva = curva.iloc[:len(esperada)].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(curva, esperada, check_dtype=False, rtol=1e-12)
            ok, detalhe = True, ""
        except AssertionError as erro:
            ok, detalhe = False, str(erro).splitlines()[0]
        trecho = "curva inteira" if ate is None else f"primeiros {ate} passos"
        placar.checar(f"{estrategia}: {trecho}", ok, detalhe or _cronometro(inicio).strip("()"))
    curva = robustez.simular_remocao(G, "grau", nutrientes=nutrientes)
    n = G.number_of_nodes()
    inicial = curva.iloc[0]
    placar.checar("passo 0 = rede intacta (núcleo, alcance e eficiência de metricas.py)",
                  np.isclose(inicial["maior_forte"], len(metricas.maior_componente_forte(G)) / n)
                  and np.isclose(inicial["alcance"], metricas.alcance(G, nutrientes, n))
                  and np.isclose(inicial["eficiencia"], metricas.eficiencia_global_dirigida(G, n)))


def robustez_completa(placar: Placar, G: nx.DiGraph, n_rep: int) -> None:
    print(f"\n== 6. Robustez completa (sem moeda, {n_rep} repetições na aleatória)")
    curvas, indices = [], []
    tempos = {}
    for estrategia in config.ESTRATEGIAS:
        inicio = time.perf_counter()
        c, i = robustez.comparar_estrategias(G, [estrategia], n_rep=n_rep, nutrientes=config.NUTRIENTES)
        tempos[estrategia] = time.perf_counter() - inicio
        curvas.append(c)
        indices.append(i)
    curvas, indices = pd.concat(curvas, ignore_index=True), pd.concat(indices, ignore_index=True)
    print(f"  {'estratégia':24s} {'R_fraco':>8s} {'R_forte':>8s} {'metade em':>10s} {'alcance 5%':>11s} {'tempo':>7s}")
    for linha in indices.to_dict("records"):
        curva = curvas[curvas["estrategia"] == linha["estrategia"]]
        alcance_5 = curva.loc[curva["fracao_removida"] >= 0.05, "alcance"].iloc[0]
        dp = f" ± {linha['R_fraco_dp']:.4f}" if linha["n_rep"] > 1 else ""
        print(f"  {linha['estrategia']:24s} {linha['R_fraco']:8.4f} {linha['R_forte']:8.4f} "
              f"{linha['metade_fraco']:10.1%} {alcance_5:11.3f} {tempos[linha['estrategia']]:6.1f}s{dp}")
    print("  (metade em: robustez.fracao_ate_metade, fração removida até o maior componente fraco cair"
          " à metade do inicial, média das repetições na aleatória; alcance 5%: depois de remover 5% da rede)")

    R = indices.set_index("estrategia")["R_fraco"]
    direcionadas = [e for e in R.index if e != "aleatoria"]
    placar.avisar("a remoção aleatória é a mais tolerada (R maior que o de toda direcionada)",
                  all(R["aleatoria"] > R[e] for e in direcionadas),
                  f"aleatória {R['aleatoria']:.3f}, direcionadas de {R[direcionadas].min():.3f} a {R[direcionadas].max():.3f}")
    metade = indices.set_index("estrategia")["metade_fraco"]
    placar.avisar("a remoção aleatória é a que mais demora a derrubar o maior componente à metade",
                  all(metade["aleatoria"] > metade[e] for e in direcionadas),
                  f"aleatória {metade['aleatoria']:.1%}, direcionadas de {metade[direcionadas].min():.1%}"
                  f" a {metade[direcionadas].max():.1%}")
    placar.avisar("recalcular o grau causa pelo menos o mesmo dano", R["grau_adaptativo"] <= R["grau"] + 1e-12)
    placar.avisar("recalcular a betweenness causa pelo menos o mesmo dano",
                  R["betweenness_adaptativo"] <= R["betweenness"] + 1e-12)
    por_rep = tempos["aleatoria"] / n_rep
    estimativa = sum(tempos.values()) - tempos["aleatoria"] + por_rep * config.N_REPETICOES_ALEATORIAS
    print(f"  [info] {por_rep:.2f} s por repetição aleatória; o Exp 1 com "
          f"{config.N_REPETICOES_ALEATORIAS} repetições deve levar ~{estimativa / 60:.1f} min por versão")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica cascata.py e robustez.py nos dados reais.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--rep", type=int, default=20, help="repetições da remoção aleatória no bloco 6")
    parser.add_argument("--amostra", type=int, default=200, help="sorteios nos blocos 2 e 3")
    args = parser.parse_args()

    tabelas = grafo.carregar_tabelas(args.org)
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    B = grafo.grafo_bipartido(reacoes)
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes))
    print(f"{args.org}: {len(reacoes)} reações; bipartida sem moeda com {len(grafo.nos_metabolito(B))} "
          f"metabólitos e {len(grafo.nos_reacao(B))} nós de reação; grafo simples com {G.number_of_nodes()} nós")

    placar = Placar()
    fontes, intactos = fontes_e_rede_intacta(placar, B, nomes)
    avalanches = contra_ingenua(placar, B, reacoes, fontes, intactos, args.amostra)
    propriedades(placar, reacoes, B, fontes, intactos, avalanches, args.amostra)
    if args.org == "eco":
        plausibilidade(placar, B, fontes, intactos, avalanches, nomes)
    robustez_contra_ingenua(placar, G)
    robustez_completa(placar, G, args.rep)

    print(f"\nResultado: {placar.falhas} falha(s), {placar.atencoes} atenção(ões)")
    sys.exit(1 if placar.falhas else 0)


if __name__ == "__main__":
    main()
