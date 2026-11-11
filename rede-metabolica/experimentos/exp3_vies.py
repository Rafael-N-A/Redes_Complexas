"""Experimento 3 — Dano individual dos metabólitos (P2 para metabólitos, sem gabarito).

Pergunta: o metabólito cuja remoção mais danifica a rede é o mais central? E
as duas réguas de dano concordam? Remove cada metabólito sozinho e mede o
dano das mesmas duas formas do Exp 2 (genes):
  estrutural  quantos metabólitos deixam o maior componente forte (o núcleo) e o
              maior componente fraco, e a queda relativa da eficiência global
              (metricas.dano_remocao_individual, no grafo metabólito-cêntrico simples);
  funcional   a avalanche da cascata: quantos metabólitos que a rede intacta produz
              deixam de ser produzidos (cascata.avalanches_todos, na rede bipartida).
Não há gabarito experimental de essencialidade para metabólitos, então a
comparação é entre as medidas: Spearman de cada uma com o grau, a betweenness
e o PageRank (os nós que o Exp 1 remove primeiro são os que, sozinhos, mais
danificam a rede?), com a harmônica como centralidade de referência, e entre
o dano estrutural e o funcional.
Cobre também o item "spreading" do roteiro (a avalanche se espalha até onde?).

Roda numa versão só (config.VERSAO_CASCATA): com a moeda sempre disponível,
a cascata dá o mesmo resultado com ou sem moeda.

As 6 substâncias escolhidas e os controles continuam, como exemplos
(config.SUBSTANCIAS_EXP3 e config.N_CONTROLES_EXP3):
  substancia  as 6 escolhidas por relevância biológica
  hub         controle: os 6 de maior grau fora das escolhidas
  baixo_grau  controle: 6 sorteados entre os de menor grau
E os casos da discussão (config.EXEMPLOS_DANO_EXP3) com os mais discordantes
entre as duas réguas (discordantes).

Lê:    dados/<org>/reacoes.csv, metabolitos.csv
Grava: resultados/tabelas/exp3_substancias_<org>.csv  (6 escolhidas + 12 controles, com os dois danos)
       resultados/tabelas/exp3_avalanches_<org>.csv   (avalanche de todos os metabólitos produzidos)
       resultados/tabelas/exp3_dano_<org>.csv         (uma linha por metabólito do grafo: centralidades,
                                                       dano estrutural e avalanche)
       resultados/tabelas/exp3_correlacoes_<org>.csv  (Spearman entre as medidas de dano e as centralidades)
       resultados/tabelas/exp3_exemplos_<org>.csv     (os casos da discussão e os mais discordantes)
       resultados/tabelas/exp3_resumo_<org>.csv       (distribuição, ajuste, correlações e a rede intacta)
       resultados/tabelas/exp3_ccdf_<org>.csv         (CCDF das avalanches >= 1 e ajustes)
       resultados/figuras/exp3_avalanche_vs_grau_<org>.png
       resultados/figuras/exp3_distribuicao_<org>.png
       resultados/figuras/exp3_dano_vs_avalanche_<org>.png (queda do núcleo x avalanche, cor pelo grau)

Uso, na raiz:
    python -m experimentos.exp3_vies --org eco
    python -m experimentos.exp3_vies --so-figuras
"""
from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from experimentos.biblioteca import cascata, figuras, grafo, metricas

COLUNAS_SUBSTANCIAS = ["grupo", "metabolito", "nome", "grau", "betweenness", "avalanche", "fracao",
                       "profundidade", "posicao_avalanche", "posicao_grau",
                       "dano_forte", "dano_fraco", "dano_eficiencia", "posicao_dano_forte"]
COLUNAS_AVALANCHES = ["metabolito", "nome", "grau", "betweenness", "avalanche", "fracao"]
CENTRALIDADES = ("grau", "betweenness", "harmonica", "pagerank")
MEDIDAS_DANO = ("dano_forte", "dano_fraco", "dano_eficiencia", "avalanche")
COLUNAS_DANO = ["metabolito", "nome", "produzido", *CENTRALIDADES, *metricas.COLUNAS_DANO_ESTRUTURAL, "avalanche"]
COLUNAS_EXEMPLOS = ["grupo", "metabolito", "nome", "nome_curto", "grau", "dano_forte", "dano_eficiencia",
                    "avalanche", "posicao_grau", "posicao_dano_forte", "posicao_avalanche", "discordancia", "motivo"]


def escolher_controles(G: nx.DiGraph, excluir: set[str], n: int, rng: np.random.Generator,
                       candidatos: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Os dois grupos de controle.

    Retorna (hubs, baixo_grau):
      hubs:       os n nós de maior grau total, fora de excluir; empates
                  desfeitos pelo código, como em metricas.top_por_grau
      baixo_grau: n nós sorteados com rng entre a fração config.QUANTIL_BAIXO_GRAU
                  de menor grau, fora de excluir (todos, se forem menos de n)
    candidatos: se dado, os dois grupos saem só dele (no exp3, os metabólitos
    que a rede intacta produz: remover um que nunca é produzido daria avalanche
    0 por definição, e o controle não diria nada). O quantil de grau é sempre
    calculado sobre todos os nós de G.
    """
    graus = metricas.graus(G)["total"]
    limite = float(np.quantile(graus, config.QUANTIL_BAIXO_GRAU))
    permitidos = set(graus.index) - set(excluir)
    if candidatos is not None:
        permitidos &= set(candidatos)
    tabela = graus[graus.index.isin(permitidos)].rename("total").reset_index()
    tabela = tabela.sort_values(["total", "metabolito"], ascending=[False, True])
    hubs = tabela["metabolito"].head(n).tolist()
    baixos = sorted(tabela.loc[tabela["total"] <= limite, "metabolito"])
    baixo_grau = sorted(rng.choice(baixos, size=min(n, len(baixos)), replace=False).tolist()) if baixos else []
    return hubs, baixo_grau


def _posicao(serie: pd.Series) -> pd.Series:
    """Posição de cada valor do maior para o menor (1 = maior); empates ficam com a melhor posição."""
    return serie.rank(ascending=False, method="min").astype(int)


def profundidade(G: nx.DiGraph, removido: str, perdidos: set[str]) -> int:
    """Até onde a avalanche chega: a maior distância (em reações) de removido a um perdido.

    Todo metabólito perdido descende do removido no grafo metabólito-cêntrico
    (só perde a produção quem dependia dele), então a distância sempre existe.
    0 se não perdeu nada.
    """
    if not perdidos or removido not in G:
        return 0
    distancias = nx.single_source_shortest_path_length(G, removido)
    faltando = perdidos - set(distancias)
    if faltando:
        raise ValueError(f"metabólitos perdidos sem caminho a partir de {removido}: {sorted(faltando)[:5]}")
    return max(distancias[m] for m in perdidos)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    resultado = spearmanr(x, y)
    return float(resultado.statistic), float(resultado.pvalue)


def tabela_dano(G: nx.DiGraph, estrutural: pd.DataFrame, avalanches: pd.Series,
                nomes: dict[str, str]) -> pd.DataFrame:
    """Uma linha por metabólito de G: centralidades, dano estrutural e avalanche.

    estrutural: metricas.dano_remocao_individual(G); avalanches: cascata.avalanches_todos
    (só os produzidos na rede intacta; avalanche NaN e produzido=False para os outros).
    Centralidades no grafo simples: as do Exp 1 (grau total, betweenness e PageRank,
    nx.pagerank) e a harmônica (metricas.harmonica), que não entra na remoção.
    Ordenada pela queda do núcleo, depois pela da eficiência (maiores primeiro) e
    pelo código.
    """
    tabela = estrutural.set_index("metabolito").reindex(sorted(G.nodes))
    tabela.index.name = "metabolito"
    tabela["nome"] = [nomes.get(m, "") for m in tabela.index]
    tabela["produzido"] = tabela.index.isin(avalanches.index)
    tabela["grau"] = metricas.graus(G)["total"].reindex(tabela.index)
    tabela["betweenness"] = metricas.betweenness(G).reindex(tabela.index)
    tabela["harmonica"] = metricas.harmonica(G).reindex(tabela.index)
    tabela["pagerank"] = pd.Series(nx.pagerank(G), dtype=float).reindex(tabela.index)
    tabela["avalanche"] = avalanches.reindex(tabela.index).astype(float)
    tabela = tabela.reset_index()[COLUNAS_DANO]
    return tabela.sort_values(["dano_forte", "dano_eficiencia", "metabolito"],
                              ascending=[False, False, True]).reset_index(drop=True)


def correlacoes_dano(dano: pd.DataFrame) -> pd.DataFrame:
    """Spearman de cada medida de dano com cada centralidade, e do dano estrutural com o funcional.

    Cada par usa os metabólitos com as duas medidas definidas: todos os do grafo
    para dano estrutural x centralidade; só os produzidos (com avalanche) quando
    a avalanche entra. Colunas: medida, outra, spearman, p, n.
    """
    pares = [(medida, centralidade) for medida in MEDIDAS_DANO for centralidade in CENTRALIDADES]
    pares += [(medida, "avalanche") for medida in MEDIDAS_DANO if medida != "avalanche"]
    linhas = []
    for medida, outra in pares:
        parte = dano[[medida, outra]].dropna()
        rho, p = _spearman(parte[medida], parte[outra])
        linhas.append({"medida": medida, "outra": outra, "spearman": rho, "p": p, "n": len(parte)})
    return pd.DataFrame(linhas)


def discordantes(dano: pd.DataFrame, k: int = config.N_ROTULOS_EXP3) -> pd.DataFrame:
    """Os k metabólitos em que o dano estrutural mais supera o funcional, e os k do contrário.

    Só os produzidos (com avalanche). Discordância = percentil da queda do núcleo
    menos o percentil da avalanche (posto médio nos empates): perto de +1, quebra
    o núcleo e não interrompe produção; perto de -1, o contrário. Empates na
    discordância: o maior dano do lado em questão primeiro, depois o código.
    Coluna grupo: "estrutural_sem_funcional" ou "funcional_sem_estrutural".
    """
    tabela = dano.dropna(subset=["avalanche"]).copy()
    tabela["discordancia"] = tabela["dano_forte"].rank(pct=True) - tabela["avalanche"].rank(pct=True)
    estrutural = tabela.sort_values(["discordancia", "dano_forte", "metabolito"], ascending=[False, False, True])
    funcional = tabela.sort_values(["discordancia", "avalanche", "metabolito"], ascending=[True, False, True])
    return pd.concat([estrutural.head(k).assign(grupo="estrutural_sem_funcional"),
                      funcional.head(k).assign(grupo="funcional_sem_estrutural")], ignore_index=True)


def exemplos_dano(dano: pd.DataFrame, exemplos: dict[str, str] = config.EXEMPLOS_DANO_EXP3,
                  k: int = config.N_ROTULOS_EXP3) -> pd.DataFrame:
    """Os casos da discussão (grupo "exemplo", de config.EXEMPLOS_DANO_EXP3) e os discordantes.

    Posições entre os metabólitos do grafo (1 = maior); a da avalanche, só entre
    os produzidos. nome_curto: config.NOMES_CURTOS_EXP3 ou, sem ele, o nome do KEGG.
    Códigos de exemplo fora do grafo são avisados e ficam de fora.
    """
    tabela = dano.copy()
    tabela["posicao_grau"] = _posicao(tabela["grau"])
    tabela["posicao_dano_forte"] = _posicao(tabela["dano_forte"])
    tabela["posicao_avalanche"] = tabela["avalanche"].rank(ascending=False, method="min").astype("Int64")
    discordancia = discordantes(tabela, k)
    por_codigo = tabela.set_index("metabolito")
    faltando = [m for m in exemplos if m not in por_codigo.index]
    if faltando:
        print(f"aviso: exemplos fora do grafo: {', '.join(faltando)}")
    curados = tabela[tabela["metabolito"].isin(list(exemplos))].assign(grupo="exemplo")
    curados = curados.set_index("metabolito").reindex([m for m in exemplos if m in por_codigo.index]).reset_index()
    curados["motivo"] = curados["metabolito"].map(exemplos)
    valores = discordancia.set_index("metabolito")["discordancia"]
    juntos = pd.concat([curados, discordancia.drop(columns="discordancia").assign(
        motivo=lambda t: t["grupo"].map({"estrutural_sem_funcional": "quebra o núcleo sem interromper produção",
                                         "funcional_sem_estrutural": "interrompe produção sem quebrar o núcleo"}))],
                       ignore_index=True)
    juntos["discordancia"] = juntos["metabolito"].map(valores)
    juntos["nome_curto"] = [config.NOMES_CURTOS_EXP3.get(m, n) for m, n in zip(juntos["metabolito"], juntos["nome"])]
    return juntos[COLUNAS_EXEMPLOS]


def rodar(org: str) -> pd.DataFrame:
    """Calcula as avalanches e o dano estrutural, e grava as tabelas. Devolve a tabela das substâncias e controles.

    Passos:
      1. tabelas = grafo.carregar_tabelas(org); remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
      2. B = grafo.grafo_bipartido(tabelas["reacoes"], remover_moeda=remover)
         G = grafo.grafo_simples(grafo.grafo_metabolitos(tabelas["reacoes"], remover_moeda=remover))
      3. Confira se os códigos de config.SUBSTANCIAS_EXP3 estão em B; avise os que faltarem.
      4. hubs, baixo = escolher_controles(G, set(config.SUBSTANCIAS_EXP3), config.N_CONTROLES_EXP3,
                                          np.random.default_rng(config.SEMENTE), candidatos=produzidos)
      5. Para cada metabólito dos três grupos: grupo, metabolito, nome, grau, betweenness,
         avalanche = len(cascata.cascata(B, metabolitos_removidos=[m])), fracao (do que a
         rede intacta produz), profundidade, as posições entre todos os produzidos e o dano estrutural.
      6. todos = cascata.avalanches_todos(B), junto com o grau e a betweenness de cada metabólito.
      7. Dano estrutural: metricas.dano_remocao_individual(G), junto com as centralidades e a
         avalanche (tabela_dano); Spearman entre as medidas (correlacoes_dano); exemplos.
      8. Resumo: quantos não derrubam nenhum outro, média e máximo, ajuste de lei de potência
         das avalanches >= 1 (Clauset et al.), correlação de Spearman da avalanche com o grau
         e com a betweenness (sobre todos os produzidos, com grau 0 para os que estão fora do
         grafo) e a rede intacta.
      9. Grave em config.arquivo_tabela("exp3", <conteudo>, org).

    O grau e a betweenness são os do grafo metabólito-cêntrico simples (os mesmos do Exp 1);
    metabólitos que só aparecem na rede bipartida (sem aresta no grafo) ficam com 0.
    """
    inicio = time.perf_counter()
    tabelas = grafo.carregar_tabelas(org)
    remover = config.VERSOES_MOEDA[config.VERSAO_CASCATA]
    reacoes = tabelas["reacoes"]
    nomes = grafo.nomes_metabolitos(tabelas["metabolitos"])
    B = grafo.grafo_bipartido(reacoes, remover_moeda=remover)
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=remover))

    fontes = cascata.fontes_externas(B)
    produzidos, _ = cascata.disponiveis(B, fontes)
    substancias = [m for m in config.SUBSTANCIAS_EXP3 if m in B]
    faltando = [f"{m} ({config.SUBSTANCIAS_EXP3[m]})" for m in config.SUBSTANCIAS_EXP3 if m not in B]
    if faltando:
        print(f"aviso: fora da rede de {org}, ficam de fora: {', '.join(faltando)}")

    graus = metricas.graus(G)["total"]
    entre = metricas.betweenness(G)

    # Avalanche de todos os metabólitos produzidos, com a centralidade de cada um.
    avalanches = cascata.avalanches_todos(B, fontes)
    todos = pd.DataFrame({"metabolito": avalanches.index,
                          "nome": [nomes.get(m, "") for m in avalanches.index],
                          "grau": graus.reindex(avalanches.index).fillna(0).astype(int).to_numpy(),
                          "betweenness": entre.reindex(avalanches.index).fillna(0.0).to_numpy(),
                          "avalanche": avalanches.to_numpy()})
    todos["fracao"] = todos["avalanche"] / len(produzidos)
    todos["posicao_avalanche"] = _posicao(todos["avalanche"])
    todos["posicao_grau"] = _posicao(todos["grau"])
    todos = todos.sort_values(["avalanche", "metabolito"], ascending=[False, True]).reset_index(drop=True)
    por_codigo = todos.set_index("metabolito")

    # Dano estrutural de cada metabólito do grafo, removido sozinho.
    estrutural = metricas.dano_remocao_individual(G)
    dano = tabela_dano(G, estrutural, avalanches, nomes)
    dano_por_codigo = dano.set_index("metabolito")
    posicao_forte = _posicao(dano_por_codigo["dano_forte"])

    hubs, baixo = escolher_controles(G, set(config.SUBSTANCIAS_EXP3), config.N_CONTROLES_EXP3,
                                     np.random.default_rng(config.SEMENTE), candidatos=produzidos)
    linhas = []
    for grupo, membros in (("substancia", substancias), ("hub", hubs), ("baixo_grau", baixo)):
        for m in membros:
            perdidos = cascata.cascata(B, metabolitos_removidos=[m], fontes=fontes)
            produzido = m in por_codigo.index
            no_grafo = m in dano_por_codigo.index
            linhas.append({
                "grupo": grupo, "metabolito": m,
                "nome": config.SUBSTANCIAS_EXP3.get(m) or nomes.get(m, ""),
                "grau": int(graus.get(m, 0)), "betweenness": float(entre.get(m, 0.0)),
                "avalanche": len(perdidos), "fracao": len(perdidos) / len(produzidos),
                "profundidade": profundidade(G, m, perdidos),
                "posicao_avalanche": int(por_codigo.loc[m, "posicao_avalanche"]) if produzido else pd.NA,
                "posicao_grau": int(por_codigo.loc[m, "posicao_grau"]) if produzido else pd.NA,
                **{c: dano_por_codigo.loc[m, c] if no_grafo else pd.NA
                   for c in ("dano_forte", "dano_fraco", "dano_eficiencia")},
                "posicao_dano_forte": int(posicao_forte[m]) if no_grafo else pd.NA,
            })
    escolhidos = pd.DataFrame(linhas, columns=COLUNAS_SUBSTANCIAS)

    correlacoes = correlacoes_dano(dano)
    rho = correlacoes.set_index(["medida", "outra"])["spearman"]
    antes = metricas.medidas_estruturais(G, G.number_of_nodes())
    positivos = todos.loc[todos["avalanche"] > 0, "avalanche"]
    ajuste = metricas.ajuste_lei_potencia(positivos)
    rho_grau, p_grau = _spearman(todos["grau"], todos["avalanche"])
    rho_bet, p_bet = _spearman(todos["betweenness"], todos["avalanche"])
    resumo = pd.DataFrame([{
        "org": org, "versao": config.VERSAO_CASCATA,
        "metabolitos": len(grafo.nos_metabolito(B)), "produzidos": len(produzidos),
        "sem_avalanche": int((todos["avalanche"] == 0).sum()),
        "fracao_sem_avalanche": float((todos["avalanche"] == 0).mean()),
        "avalanche_media": float(todos["avalanche"].mean()),
        "avalanche_max": int(todos["avalanche"].max()), "metabolito_max": todos["metabolito"].iloc[0],
        "spearman_grau": rho_grau, "p_grau": p_grau,
        "spearman_betweenness": rho_bet, "p_betweenness": p_bet,
        "lp_alpha": ajuste["alpha"], "lp_xmin": ajuste["xmin"], "lp_n_cauda": ajuste["n_cauda"],
        "lp_R_vs_lognormal": ajuste["R_vs_lognormal"], "lp_p": ajuste["p"],
        "metabolitos_grafo": G.number_of_nodes(), "nucleo_intacto": antes["maior_forte"],
        "maior_fraco_intacto": antes["maior_fraco"], "eficiencia_intacta": antes["eficiencia"],
        # Tirar um nó do núcleo sempre o diminui em pelo menos 1 (o próprio nó): o que
        # interessa são os que levam outros metabólitos junto (queda >= 2).
        "nucleo_quebra": int((dano["dano_forte"] >= 2).sum()),
        "dano_forte_max": int(dano["dano_forte"].max()), "metabolito_dano_forte_max": dano["metabolito"].iloc[0],
        "spearman_forte_avalanche": rho[("dano_forte", "avalanche")],
        "spearman_eficiencia_avalanche": rho[("dano_eficiencia", "avalanche")],
    }])

    for conteudo, tabela in (("substancias", escolhidos), ("avalanches", todos[COLUNAS_AVALANCHES]),
                             ("resumo", resumo), ("ccdf", metricas.ccdf_com_ajustes(positivos)),
                             ("dano", dano), ("correlacoes", correlacoes), ("exemplos", exemplos_dano(dano))):
        destino = config.arquivo_tabela("exp3", conteudo, org)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(destino, index=False)
    print(f"{org}: avalanches e remoções individuais de {G.number_of_nodes()} metabólitos "
          f"({time.perf_counter() - inicio:.0f} s)")
    return escolhidos


def plotar(org: str) -> None:
    """Lê as tabelas e grava avalanche x grau, a distribuição dos tamanhos e dano estrutural x avalanche.

    Avalanche x grau: os três grupos, com todos os metabólitos em cinza ao
    fundo. Distribuição: CCDF das avalanches >= 1 com os ajustes de lei de
    potência e log-normal, como a distribuição de graus do Exp 0. Dano
    estrutural x avalanche: a queda do núcleo e a avalanche de cada metabólito
    produzido, com a cor pelo grau e os nomes dos casos da tabela de exemplos.
    """
    figuras.estilo()
    escolhidos = pd.read_csv(config.arquivo_tabela("exp3", "substancias", org))
    todos = pd.read_csv(config.arquivo_tabela("exp3", "avalanches", org))
    ccdf = pd.read_csv(config.arquivo_tabela("exp3", "ccdf", org))
    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", org)).iloc[0]
    ajuste = {chave: resumo[f"lp_{chave}"] for chave in ("alpha", "xmin", "R_vs_lognormal", "p")}
    figuras.salvar(figuras.avalanche_vs_grau(escolhidos, todos),
                   config.arquivo_figura("exp3", "avalanche_vs_grau", org))
    figuras.salvar(figuras.distribuicao_avalanches(todos["avalanche"], ccdf, ajuste),
                   config.arquivo_figura("exp3", "distribuicao", org))
    dano = pd.read_csv(config.arquivo_tabela("exp3", "dano", org))
    exemplos = pd.read_csv(config.arquivo_tabela("exp3", "exemplos", org))
    correlacoes = pd.read_csv(config.arquivo_tabela("exp3", "correlacoes", org))
    figuras.salvar(figuras.dano_vs_avalanche(dano, exemplos.drop_duplicates("metabolito"), correlacoes),
                   config.arquivo_figura("exp3", "dano_vs_avalanche", org))


def _resumo(escolhidos: pd.DataFrame, org: str) -> str:
    resumo = pd.read_csv(config.arquivo_tabela("exp3", "resumo", org)).iloc[0]
    correlacoes = pd.read_csv(config.arquivo_tabela("exp3", "correlacoes", org)).set_index(["medida", "outra"])
    exemplos = pd.read_csv(config.arquivo_tabela("exp3", "exemplos", org))
    rho = correlacoes["spearman"]
    linhas = [f"{org}: {resumo['produzidos']} de {resumo['metabolitos']} metabólitos produzidos; "
              f"{resumo['sem_avalanche']} ({resumo['fracao_sem_avalanche']:.0%}) sem avalanche; "
              f"maior {resumo['avalanche_max']} ({resumo['metabolito_max']})",
              f"  Spearman avalanche x grau = {resumo['spearman_grau']:.2f}, "
              f"x betweenness = {resumo['spearman_betweenness']:.2f} (todos os produzidos); "
              f"lei de potência x log-normal R = {resumo['lp_R_vs_lognormal']:.2f} (p = {resumo['lp_p']:.3f})",
              f"  dano estrutural (cada um dos {resumo['metabolitos_grafo']} metabólitos removido sozinho; núcleo "
              f"intacto = {resumo['nucleo_intacto']}, fraco = {resumo['maior_fraco_intacto']}, eficiência = "
              f"{resumo['eficiencia_intacta']:.4f}): {resumo['nucleo_quebra']} dos {resumo['nucleo_intacto']} do "
              f"núcleo levam outros metabólitos junto; no máximo {resumo['dano_forte_max']} "
              f"({resumo['metabolito_dano_forte_max']})",
              f"  {'Spearman':16s} " + " ".join(f"{c:>11s}" for c in (*CENTRALIDADES, "avalanche"))]
    for medida in MEDIDAS_DANO:
        valores = [rho.get((medida, outra), np.nan) for outra in (*CENTRALIDADES, "avalanche")]
        linhas.append(f"  {medida:16s} " + " ".join(f"{v:11.2f}" if np.isfinite(v) else f"{'':11s}" for v in valores))
    linhas.append(f"  {'exemplo':26s} {'metabólito':18s} {'grau':>4s} {'núcleo':>6s} {'eficiência':>10s} {'aval.':>5s}")
    for linha in exemplos.to_dict("records"):
        avalanche = f"{linha['avalanche']:5.0f}" if pd.notna(linha["avalanche"]) else f"{'-':>5s}"
        linhas.append(f"  {linha['grupo']:26s} {str(linha['nome_curto'])[:18]:18s} {linha['grau']:4.0f} "
                      f"{linha['dano_forte']:6.0f} {linha['dano_eficiencia']:10.3f} {avalanche}")
    linhas.append(f"  {'grupo':11s} {'metabólito':28s} {'grau':>4s} {'aval.':>5s} {'prof.':>5s} {'posição':>7s}"
                  f" {'núcleo':>6s}")
    for linha in escolhidos.to_dict("records"):
        nucleo = f"{linha['dano_forte']:6.0f}" if pd.notna(linha["dano_forte"]) else f"{'-':>6s}"
        linhas.append(f"  {linha['grupo']:11s} {str(linha['nome'])[:28]:28s} {linha['grau']:4d} "
                      f"{linha['avalanche']:5d} {linha['profundidade']:5d} {str(linha['posicao_avalanche']):>7s}"
                      f" {nucleo}")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 3: dano individual dos metabólitos (estrutural e cascata).")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        print(_resumo(rodar(args.org), args.org))
    plotar(args.org)


if __name__ == "__main__":
    main()
