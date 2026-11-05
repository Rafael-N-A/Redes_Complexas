"""Verifica o grafo e as métricas nos dados reais de um organismo.

Os testes (pytest) usam redes pequenas com resposta conhecida. Este script
faz o mesmo tipo de conferência na rede real, em três blocos:

  1. Invariantes: propriedades que valem em qualquer rede bem construída.
     FALHOU aqui é erro no código.
  2. Comparação com implementações independentes (NetworkX), na rede real.
     FALHOU aqui também é erro no código.
  3. Plausibilidade biológica (só para a E. coli): caminhos e ciclos que a
     bioquímica garante que existem. ATENÇÃO aqui pede investigação, mas
     pode ser só um detalhe de como o KEGG desenha a via.

No fim, uma prévia da Tabela 1 nas duas versões (com e sem moeda) e o
efeito da regra de reversibilidade ("vale o reversível" contra "vale o
irreversível") no núcleo e no ciclo de Krebs.

Uso, na raiz do projeto:
    python -m testes.verificar_grafo --org eco
Sai com código 1 se algum item dos blocos 1 ou 2 falhar.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import networkx as nx

import config
import contrato
from experimentos.biblioteca import grafo, metricas

GLICOSE, PIRUVATO, ACETIL_COA = "C00031", "C00022", "C00024"
GLUTAMATO, OXOGLUTARATO = "C00025", "C00026"
CICLO_DE_KREBS = {
    "C00158": "citrato", "C00311": "isocitrato", "C00026": "2-oxoglutarato",
    "C00091": "succinil-CoA", "C00042": "succinato", "C00122": "fumarato",
    "C00149": "malato", "C00036": "oxaloacetato",
}


class Placar:
    def __init__(self):
        self.falhas = 0
        self.atencoes = 0

    def checar(self, descricao: str, ok: bool, detalhe: str = "") -> None:
        self.falhas += not ok
        print(f"  [{'OK' if ok else 'FALHOU'}] {descricao}" + (f" ({detalhe})" if detalhe else ""))

    def avisar(self, descricao: str, ok: bool, detalhe: str = "") -> None:
        self.atencoes += not ok
        print(f"  [{'OK' if ok else 'ATENÇÃO'}] {descricao}" + (f" ({detalhe})" if detalhe else ""))


def _lados(r: dict, remover: bool) -> tuple[list[str], list[str]]:
    return grafo.sem_moeda(r["substratos"], remover), grafo.sem_moeda(r["produtos"], remover)


def arestas_esperadas(reacoes, remover: bool) -> int:
    """Conta as arestas direto da tabela, sem passar pelo grafo.py."""
    total = 0
    for r in reacoes.to_dict("records"):
        s, p = _lados(r, remover)
        pares = {(a, b) for a in s for b in p if a != b}
        if r["reversivel"]:
            pares |= {(b, a) for a, b in pares}
        total += len(pares)
    return total


def _invertida(s, p, ref_s, ref_p) -> bool:
    s, p, ref_s, ref_p = set(s), set(p), set(ref_s), set(ref_p)
    return len(s & ref_p) + len(p & ref_s) > len(s & ref_s) + len(p & ref_p)


def regra_de_reversibilidade(placar: Placar, reacoes, brutas) -> None:
    """Confere a regra 4 da fusão direto na camada bruta, sem passar pelo montar_tabelas."""
    print("\n== 1. Regra de reversibilidade (vale o reversível)")
    conflito_irreversivel = reacoes[reacoes["conflito_reversibilidade"] & ~reacoes["reversivel"]]
    placar.checar("toda reação com conflito entre as vias ficou reversível", conflito_irreversivel.empty,
                  ", ".join(conflito_irreversivel["reacao"].head(5)))
    ocorrencias = {}
    for b in brutas.to_dict("records"):
        for codigo in b["reacoes"]:
            ocorrencias.setdefault(codigo, []).append(b)
    erradas = []
    for r in reacoes[~reacoes["reversivel"]].to_dict("records"):
        for b in ocorrencias.get(r["reacao"], []):
            if b["tipo"] == "reversible" or _invertida(b["substratos"], b["produtos"],
                                                        r["substratos"], r["produtos"]):
                erradas.append(r["reacao"])
                break
    placar.checar("toda irreversível é irreversível em todas as vias e no mesmo sentido",
                  not erradas, ", ".join(erradas[:5]))


def invariantes(placar: Placar, reacoes, metabolitos, versao: str) -> nx.DiGraph:
    remover = config.VERSOES_MOEDA[versao]
    print(f"\n== 1. Invariantes ({versao})")
    M = grafo.grafo_metabolitos(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(M)
    U = grafo.grafo_nao_dirigido(G)
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)

    esperado = arestas_esperadas(reacoes, remover)
    placar.checar("arestas do multigrafo = contagem direta na tabela", M.number_of_edges() == esperado,
                  f"{M.number_of_edges()} x {esperado}")
    moeda_no_grafo = set(M) & set(config.METABOLITOS_MOEDA)
    if remover:
        placar.checar("nenhum metabólito-moeda no grafo", not moeda_no_grafo, ", ".join(sorted(moeda_no_grafo)))
    else:
        print(f"  [info] metabólitos-moeda presentes: {len(moeda_no_grafo)} de {len(config.METABOLITOS_MOEDA)}")
    placar.checar("nenhum laço (aresta de um nó para ele mesmo)", nx.number_of_selfloops(M) == 0)
    sem_volta = [k for u, v, k, rev in M.edges(keys=True, data="reversivel") if rev and not M.has_edge(v, u, key=k)]
    placar.checar("toda aresta de reação reversível tem a volta", not sem_volta, ", ".join(sorted(set(sem_volta))[:5]))
    em_reacao = set(metabolitos.loc[metabolitos["em_reacao"], "metabolito"])
    placar.checar("todo nó é um metabólito que aparece em reação", set(M) <= em_reacao)
    placar.checar("peso total do grafo simples = arestas do multigrafo",
                  sum(w for _, _, w in G.edges(data="weight")) == M.number_of_edges())
    placar.checar("arestas do grafo simples = pares distintos do multigrafo",
                  G.number_of_edges() == len(set(M.edges())))
    placar.checar("arestas do não dirigido = pares não ordenados",
                  U.number_of_edges() == len({frozenset(e) for e in G.edges()}))
    tabela_graus = metricas.graus(G)
    placar.checar("soma dos graus de entrada = de saída = arestas",
                  tabela_graus["entrada"].sum() == tabela_graus["saida"].sum() == G.number_of_edges())
    placar.checar("as quatro partes do bow-tie somam todos os nós",
                  sum(metricas.bow_tie(G).values()) == G.number_of_nodes())

    n_reversiveis = int(reacoes["reversivel"].sum())
    placar.checar("bipartido: nós de reação = reações + reversíveis",
                  len(grafo.nos_reacao(B)) == len(reacoes) + n_reversiveis,
                  f"{len(grafo.nos_reacao(B))} x {len(reacoes) + n_reversiveis}")
    tipos = dict(B.nodes(data="tipo"))
    placar.checar("bipartido: toda aresta liga metabólito e reação", all(tipos[u] != tipos[v] for u, v in B.edges()))
    erradas = []
    for r in reacoes.to_dict("records"):
        s, p = _lados(r, remover)
        sentidos = [("ida", s, p)] + ([("volta", p, s)] if r["reversivel"] else [])
        for sentido, entram, saem in sentidos:
            no = grafo.no_de_reacao(r["reacao"], sentido)
            if set(B.predecessors(no)) != set(entram) or set(B.successors(no)) != set(saem):
                erradas.append(no)
    placar.checar("bipartido: substratos e produtos de cada nó de reação batem com a tabela",
                  not erradas, ", ".join(erradas[:5]))
    placar.checar("choke points são reações da tabela",
                  metricas.choke_points(reacoes, remover) <= set(reacoes["reacao"]))
    return G


def comparacoes(placar: Placar, G: nx.DiGraph) -> None:
    print("\n== 2. Comparação com a NetworkX (sem moeda)")
    U = grafo.grafo_nao_dirigido(G)
    simetrica = U.to_directed()
    nossa = metricas.eficiencia_global_dirigida(simetrica, U.number_of_nodes())
    da_networkx = nx.global_efficiency(U)
    placar.checar("eficiência na versão simétrica = nx.global_efficiency", math.isclose(nossa, da_networkx),
                  f"{nossa:.6f} x {da_networkx:.6f}")

    nucleo = metricas.maior_componente_forte(G)
    H = G.subgraph(nucleo)
    caminho, diametro = metricas.caminho_medio_e_diametro(G)
    placar.checar("caminho médio no núcleo = nx.average_shortest_path_length",
                  math.isclose(caminho, nx.average_shortest_path_length(H)))
    placar.checar("diâmetro do núcleo = nx.diameter", diametro == nx.diameter(H))

    condensado = nx.condensation(G)
    componente_do_nucleo = condensado.graph["mapping"][next(iter(nucleo))]
    membros = nx.get_node_attributes(condensado, "members")
    entrada = set().union(*(membros[c] for c in nx.ancestors(condensado, componente_do_nucleo)))
    saida = set().union(*(membros[c] for c in nx.descendants(condensado, componente_do_nucleo)))
    partes = metricas.componentes_bow_tie(G)
    placar.checar("bow-tie = o mesmo calculado pelo grafo condensado",
                  partes["entrada"] == entrada and partes["saida"] == saida)

    fontes = [f for f in config.NUTRIENTES if f in G]
    por_descendentes = set(fontes).union(*(nx.descendants(G, f) for f in fontes)) if fontes else set()
    placar.checar("alcance = união dos nx.descendants das fontes",
                  metricas.alcancaveis(G, config.NUTRIENTES) == por_descendentes,
                  f"{len(por_descendentes)} nós a partir de {len(fontes)} fontes")


def _caminho(G: nx.DiGraph, origem: str, destino: str, nomes: dict[str, str]) -> tuple[bool, str]:
    if origem not in G or destino not in G:
        return False, "um dos dois não está no grafo"
    try:
        caminho = nx.shortest_path(G, origem, destino)
    except nx.NetworkXNoPath:
        return False, "sem caminho"
    return True, f"{len(caminho) - 1} passos: " + " -> ".join(nomes.get(m, m) for m in caminho)


def plausibilidade(placar: Placar, G: nx.DiGraph, nomes: dict[str, str]) -> None:
    print("\n== 3. Plausibilidade biológica (sem moeda)")
    for origem, destino, descricao in [
        (GLICOSE, PIRUVATO, "glicólise: glicose chega ao piruvato"),
        (PIRUVATO, ACETIL_COA, "piruvato chega à acetil-CoA"),
        (OXOGLUTARATO, GLUTAMATO, "2-oxoglutarato chega ao glutamato"),
        (GLUTAMATO, OXOGLUTARATO, "glutamato volta ao 2-oxoglutarato"),
    ]:
        ok, detalhe = _caminho(G, origem, destino, nomes)
        placar.avisar(descricao, ok, detalhe)

    ausentes = [nome for codigo, nome in CICLO_DE_KREBS.items() if codigo not in G]
    placar.avisar("os 8 intermediários do ciclo de Krebs estão no grafo", not ausentes, ", ".join(ausentes))
    presentes = [c for c in CICLO_DE_KREBS if c in G]
    if presentes:
        nucleo = metricas.maior_componente_forte(G)
        fora = [CICLO_DE_KREBS[c] for c in presentes if c not in nucleo]
        placar.avisar("o ciclo de Krebs fecha: todos no núcleo (maior componente forte)", not fora,
                      "fora do núcleo: " + ", ".join(fora) if fora else f"núcleo com {len(nucleo)} nós")


def previa(G: nx.DiGraph, nomes: dict[str, str], reacoes, versao: str) -> None:
    print(f"\n== Prévia da Tabela 1 ({versao})")
    inicio = time.perf_counter()
    linha = metricas.tabela_descritiva(G)
    for chave, valor in linha.items():
        print(f"  {chave:20s} {valor:.4f}" if isinstance(valor, float) else f"  {chave:20s} {valor}")
    ajuste = metricas.ajuste_lei_potencia(metricas.graus(G)["total"])
    print(f"  lei de potência      alpha={ajuste['alpha']:.2f}, xmin={ajuste['xmin']:.0f}, "
          f"cauda={ajuste['n_cauda']} nós, R x log-normal={ajuste['R_vs_lognormal']:.2f} (p={ajuste['p']:.3f})")
    choke = metricas.choke_points(reacoes, config.VERSOES_MOEDA[versao])
    print(f"  choke points         {len(choke)} de {len(reacoes)} reações")
    print("  10 maiores graus:")
    for r in metricas.top_por_grau(G, 10, nomes).to_dict("records"):
        marca = " (moeda)" if r["moeda"] else ""
        print(f"    {r['total']:4d}  {r['metabolito']}  {r['nome']}{marca}")
    if versao == "sem_moeda":
        print("  5 maiores betweenness:")
        for m, b in metricas.betweenness(G).nlargest(5).items():
            print(f"    {b:.4f}  {m}  {nomes.get(m, '')}")
    print(f"  ({time.perf_counter() - inicio:.1f} s)")


def sensibilidade(reacoes, nomes: dict[str, str]) -> None:
    print("\n== Sensibilidade à regra de reversibilidade (sem moeda)")
    for rotulo, alternativa in [("vale o reversível (padrão)", False), ("vale o irreversível", True)]:
        G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, conflito_como_irreversivel=alternativa))
        n_rev = sum(grafo.reversivel(r, alternativa) for r in reacoes.to_dict("records"))
        nucleo = metricas.maior_componente_forte(G)
        caminho, diametro = metricas.caminho_medio_e_diametro(G)
        fora = [n for c, n in CICLO_DE_KREBS.items() if c in G and c not in nucleo]
        print(f"  {rotulo:28s} reversíveis={n_rev}, núcleo={len(nucleo)}, caminho médio={caminho:.2f}, "
              f"diâmetro={diametro}, Krebs fora do núcleo: {', '.join(fora) or 'nenhum'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica grafo.py e metricas.py nos dados reais.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    args = parser.parse_args()

    tabelas = grafo.carregar_tabelas(args.org)
    reacoes, metabolitos = tabelas["reacoes"], tabelas["metabolitos"]
    nomes = grafo.nomes_metabolitos(metabolitos)
    print(f"{args.org}: {len(reacoes)} reações, {int(reacoes['reversivel'].sum())} reversíveis")

    placar = Placar()
    regra_de_reversibilidade(placar, reacoes, contrato.carregar(args.org, "reacoes_por_via"))
    grafos = {versao: invariantes(placar, reacoes, metabolitos, versao) for versao in config.VERSOES_MOEDA}
    comparacoes(placar, grafos["sem_moeda"])
    if args.org == "eco":
        plausibilidade(placar, grafos["sem_moeda"], nomes)
    for versao, G in grafos.items():
        previa(G, nomes, reacoes, versao)
    sensibilidade(reacoes, nomes)

    print(f"\nResultado: {placar.falhas} falha(s), {placar.atencoes} atenção(ões)")
    sys.exit(1 if placar.falhas else 0)


if __name__ == "__main__":
    main()
