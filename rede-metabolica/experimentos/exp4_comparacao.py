"""Experimento 4 — Comparação com outras redes (P3: a rede se parece com Erdős–Rényi, Watts–Strogatz ou Barabási–Albert?).

Pergunta: os padrões da E. coli se repetem em outros organismos e diferem de
redes aleatórias que preservam parte da sua estrutura? E de qual modelo
clássico de rede ela fica mais perto?

Três camadas, com as mesmas funções da base (exp0) e do Exp 1:
  - outros organismos (config.ORGANISMOS_COMPARACAO), pelo mesmo pipeline de coleta;
  - modelos nulos da E. coli (modelos_nulos.TIPOS), dirigidos: rewiring recíproco
    (mesmos graus e pares recíprocos), rewiring (mesmos graus) e Erdős–Rényi (mesmos n e m);
  - modelos clássicos (config.MODELOS_CLASSICOS), na versão NÃO dirigida, porque os
    modelos clássicos são não dirigidos: configuração, Erdős–Rényi G(N, M), Barabási–Albert
    misto (m = 1 ou 2) e com m = 1, Newman–Watts (k = 2) e Watts–Strogatz (k = 4), todos
    com o N e o M da E. coli não dirigida ou o mais perto possível (ver modelos_classicos.py).
    Robustez pelo maior componente conexo, por percolação reversa (union-find), com as
    estratégias de config.ESTRATEGIAS_CLASSICOS; caminho médio no maior componente de
    cada rede; escore z da E. coli contra cada modelo e o sigma de mundo pequeno
    (C/C_ER)/(L/L_ER), com C_ER e L_ER as médias das redes Erdős–Rényi.
Nos modelos nulos e clássicos só rodam métricas e robustez: eles não têm reações.

Comparar redes de tamanhos e conectividades diferentes. O índice R de
Schneider et al. (2011) é a área sob a curva do maior componente em fração dos
nós, e por isso já começa limitado pelo tamanho do maior componente na rede
intacta: a M. genitalium começa com 31% dos nós no componente fraco, a E. coli
com 80% e o rewiring com 96%. Para separar "começa partida" de "parte mais
rápido", o Exp 4 relata também:
  - R normalizado: a mesma área com a curva dividida pelo valor na rede intacta
    (S(q)/S(0), como nas curvas relativas de Albert, Jeong e Barabási, 2000);
  - fração removida até o maior componente cair à metade (já relativa ao início).
Contra os modelos nulos, cada métrica da E. coli vira um escore z,
z = (E. coli − média dos nulos) / desvio padrão dos nulos (Maslov e Sneppen, 2002;
Milo et al., 2002): |z| > 2 indica uma diferença que o modelo nulo não explica.

Roda numa versão só (config.VERSAO_COMPARACAO) e, por padrão, sem medir a
eficiência (config.MEDIR_EFICIENCIA_EXP4), para caber no tempo (~10 min).

Lê:    dados/<org>/reacoes.csv de cada organismo (rode a coleta de todos antes)
Grava em resultados/tabelas/ (<v> = config.VERSAO_COMPARACAO):
       exp4_descritiva_redes_<v>.csv  uma linha por rede: Tabela 1, reciprocidade,
                                      trocas do rewiring e lei de potência (organismos)
       exp4_indices_redes_<v>.csv     uma linha por rede e estratégia: R, R normalizado,
                                      metade e alcance
       exp4_curvas_redes_<v>.csv      organismos inteiros; nulos já como média por tipo
                                      (os _dp dos nulos são entre as redes sorteadas)
       exp4_nulos_redes_<v>.csv       E. coli x cada modelo nulo: média, dp e escore z
       exp4_ccdf_redes_<v>.csv        CCDF do grau total de cada organismo
       exp4_classicos_descritiva_redes_<v>.csv  modelos clássicos, uma linha por rede: M, grau médio e
                                      máximo, clustering, maior componente, caminho médio
                                      (no maior componente), assortatividade, sigma
       exp4_classicos_indices_redes_<v>.csv     uma linha por rede e estratégia: S0, R, R normalizado, metade
       exp4_classicos_curvas_redes_<v>.csv      curvas do maior componente; nos modelos, média das redes
       exp4_classicos_z_redes_<v>.csv           E. coli x cada modelo clássico: média, dp e escore z
       exp4_classicos_ccdf_redes_<v>.csv        CCDF do grau (não dirigido) da E. coli e de cada modelo
e em resultados/figuras/:
       exp4_indices_redes_<v>.png     R normalizado e metade, por rede e estratégia
       exp4_curvas_nulos_<v>.png      E. coli x modelos nulos
       exp4_curvas_organismos_<v>.png E. coli x B. subtilis x M. genitalium
       exp4_graus_redes_<v>.png       CCDF do grau dos organismos
       exp4_classicos_indices_redes_<v>.png  R normalizado e metade, por modelo clássico e estratégia
       exp4_classicos_curvas_redes_<v>.png   remoção aleatória x ataque por grau, E. coli e modelos
       exp4_classicos_graus_redes_<v>.png    CCDF dos graus, E. coli e modelos sobrepostos

Uso, na raiz:
    python -m experimentos.exp4_comparacao
    python -m experimentos.exp4_comparacao --orgs eco bsu --sem-nulos --sem-classicos
    python -m experimentos.exp4_comparacao --so-figuras
"""
from __future__ import annotations

import argparse
import math
import time

import networkx as nx
import numpy as np
import pandas as pd

import config
from experimentos.biblioteca import figuras, grafo, metricas, modelos_classicos, modelos_nulos, robustez

TIPOS_NULOS = modelos_nulos.TIPOS
ESTRATEGIAS_FIGURA = ("aleatoria", "grau_adaptativo")  # aleatória x o ataque mais forte

# Métricas que viram relativas à rede intacta (coluna nova: <metrica>_rel).
RELATIVAS = ("maior_fraco", "maior_forte", "alcance")

COLUNAS_INDICES = ("rede", "indice", "estrategia", "n_rep",
                   "S0_fraco", "S0_forte", "alcance0",
                   "R_fraco", "R_fraco_dp", "R_forte", "R_forte_dp", "R_fraco_norm", "R_forte_norm",
                   "metade_fraco", "metade_fraco_dp", "metade_forte", "metade_forte_dp",
                   "R_alcance", "R_alcance_norm")

# O que entra no confronto com os nulos. Ficam de fora nós, arestas, densidade e
# grau médio, que os três modelos preservam por construção.
DESCRITIVAS_NULOS = ("maior_fraco", "componentes_fracos", "nucleo", "entrada", "saida", "restante",
                     "caminho_medio", "diametro", "clustering_medio", "assortatividade", "reciprocidade")
INDICES_NULOS = ("R_fraco_norm", "R_forte_norm", "metade_fraco", "metade_forte", "R_alcance_norm")

COLUNAS_CCDF = ("rede", "grau", "n_nos", "ccdf_empirica")

# Modelos clássicos: o que entra no confronto (escores z) e nas figuras.
DESCRITIVAS_CLASSICOS = ("arestas", "grau_medio", "grau_max", "clustering_medio", "maior_componente",
                         "caminho_medio", "diametro", "assortatividade", "sigma")
INDICES_CLASSICOS = ("R_norm", "R", "metade")
ESTRATEGIAS_FIGURA_CLASSICOS = ("aleatoria", "grau")  # remoção aleatória x ataque por grau


# ---------------------------------------------------------------------------
# Medidas de uma rede
# ---------------------------------------------------------------------------
def grafo_do_organismo(org: str, versao: str = config.VERSAO_COMPARACAO) -> nx.DiGraph:
    """O grafo simples metabólito-cêntrico de org, na versão pedida (o mesmo do Exp 1)."""
    reacoes = grafo.carregar_tabelas(org)["reacoes"]
    return grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, remover_moeda=config.VERSOES_MOEDA[versao]))


def relativas(curvas: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta <metrica>_rel e <metrica>_rel_dp: a curva dividida pelo valor na rede intacta.

    Por estratégia; NaN se o valor inicial for 0 ou NaN (o alcance sem nutrientes).
    """
    curvas = curvas.copy()
    for metrica in RELATIVAS:
        inicial = curvas.loc[curvas["n_removidos"] == 0].set_index("estrategia")[metrica]
        divisor = curvas["estrategia"].map(inicial).where(lambda v: v > 0)
        curvas[f"{metrica}_rel"] = curvas[metrica] / divisor
        curvas[f"{metrica}_rel_dp"] = curvas[f"{metrica}_dp"] / divisor
    return curvas


def indices_completos(curvas: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """Os índices de robustez.comparar_estrategias mais os valores iniciais e os normalizados.

    curvas precisa das colunas _rel (relativas()). R normalizado = R / S0, que é
    a mesma área da curva relativa. R_alcance é o índice R do alcance topológico
    (média depois de cada remoção), e o normalizado divide pelo alcance inicial.
    """
    linhas = []
    for linha in indices.to_dict("records"):
        curva = curvas[curvas["estrategia"] == linha["estrategia"]].sort_values("n_removidos")
        inicio = curva.iloc[0]
        depois = curva[curva["n_removidos"] > 0]
        s0_fraco, s0_forte, alcance0 = inicio["maior_fraco"], inicio["maior_forte"], inicio["alcance"]
        linhas.append({
            **linha,
            "S0_fraco": s0_fraco, "S0_forte": s0_forte, "alcance0": alcance0,
            "R_fraco_norm": linha["R_fraco"] / s0_fraco if s0_fraco > 0 else math.nan,
            "R_forte_norm": linha["R_forte"] / s0_forte if s0_forte > 0 else math.nan,
            "R_alcance": metricas.indice_robustez(depois["alcance"]),
            "R_alcance_norm": metricas.indice_robustez(depois["alcance_rel"]),
        })
    return pd.DataFrame(linhas).reindex(columns=[c for c in COLUNAS_INDICES if c not in ("rede", "indice")])


def medir_rede(G: nx.DiGraph, nutrientes, n_rep: int, semente: int,
               progresso: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Descritivas, curvas (com as relativas) e índices de uma rede, com as estratégias do Exp 4."""
    descritiva = metricas.tabela_descritiva(G)
    curvas, indices = robustez.comparar_estrategias(
        G, config.ESTRATEGIAS_COMPARACAO, n_rep=n_rep, semente=semente, nutrientes=nutrientes,
        medir_eficiencia=config.MEDIR_EFICIENCIA_EXP4, progresso=progresso)
    curvas = relativas(curvas)
    return descritiva, curvas, indices_completos(curvas, indices)


def media_das_redes(curvas: list[pd.DataFrame]) -> pd.DataFrame:
    """Curva média de várias redes do mesmo modelo nulo, por estratégia e passo.

    Os *_dp passam a ser o desvio padrão entre as redes (das curvas de cada uma;
    na aleatória, da média das repetições de cada rede). Todas têm o mesmo
    número de nós, então os passos coincidem. Coluna n_redes.
    """
    juntas = pd.concat(curvas, ignore_index=True)
    metricas_curva = [*robustez.METRICAS, *(f"{m}_rel" for m in RELATIVAS)]
    grupos = juntas.groupby(["estrategia", "n_removidos"], sort=False)
    media = grupos[["fracao_removida", *metricas_curva]].mean()
    desvio = grupos[metricas_curva].std(ddof=1)
    desvio.columns = [f"{c}_dp" for c in metricas_curva]
    resultado = pd.concat([media, desvio], axis=1).reset_index()
    resultado["n_redes"] = len(curvas)
    return resultado


def _ccdf_empirica(G: nx.DiGraph) -> pd.DataFrame:
    """CCDF do grau total só nos graus observados: grau, n_nos, ccdf_empirica = P(K >= grau)."""
    contagem = pd.Series([d for _, d in G.degree()]).value_counts().sort_index()
    ccdf = contagem[::-1].cumsum()[::-1] / contagem.sum()
    return pd.DataFrame({"grau": contagem.index, "n_nos": contagem.to_numpy(), "ccdf_empirica": ccdf.to_numpy()})


# ---------------------------------------------------------------------------
# Confronto com os nulos
# ---------------------------------------------------------------------------
def escore_z(real: float, valores_nulos) -> tuple[float, float, float]:
    """(média, desvio padrão amostral, z) dos valores nulos.

    z é NaN se o dp for indefinido ou zero, o que inclui o resto de
    arredondamento (~1e-16) do dp de valores iguais: é o caso da reciprocidade
    no rewiring recíproco, que o modelo preserva por construção.
    """
    valores = np.asarray(pd.Series(valores_nulos, dtype=float).dropna(), dtype=float)
    if len(valores) == 0:
        return math.nan, math.nan, math.nan
    media = float(valores.mean())
    dp = float(valores.std(ddof=1)) if len(valores) > 1 else math.nan
    if dp <= 1e-12 * max(1.0, abs(media)):
        dp = 0.0 if not math.isnan(dp) else dp
        return media, dp, math.nan
    return media, dp, (real - media) / dp if not math.isnan(real) else math.nan


def confronto_nulos(descritiva: pd.DataFrame, indices: pd.DataFrame, real: str) -> pd.DataFrame:
    """Uma linha por métrica (e estratégia, nos índices): o valor da rede real e, para cada
    modelo nulo presente, média, dp e escore z (colunas <tipo>_media, <tipo>_dp, <tipo>_z)."""
    tipos = [t for t in TIPOS_NULOS if t in set(descritiva["rede"])]
    linhas = []
    for metrica in DESCRITIVAS_NULOS:
        valor = float(descritiva.loc[descritiva["rede"] == real, metrica].iloc[0])
        linha = {"rede_real": real, "metrica": metrica, "estrategia": "", "real": valor}
        for tipo in tipos:
            nulos = descritiva.loc[descritiva["rede"] == tipo, metrica]
            linha[f"{tipo}_media"], linha[f"{tipo}_dp"], linha[f"{tipo}_z"] = escore_z(valor, nulos)
        linhas.append(linha)
    for estrategia in _em_ordem(indices["estrategia"], config.ESTRATEGIAS_COMPARACAO):
        da_estrategia = indices[indices["estrategia"] == estrategia]
        for metrica in INDICES_NULOS:
            valor = float(da_estrategia.loc[da_estrategia["rede"] == real, metrica].iloc[0])
            linha = {"rede_real": real, "metrica": metrica, "estrategia": estrategia, "real": valor}
            for tipo in tipos:
                nulos = da_estrategia.loc[da_estrategia["rede"] == tipo, metrica]
                linha[f"{tipo}_media"], linha[f"{tipo}_dp"], linha[f"{tipo}_z"] = escore_z(valor, nulos)
            linhas.append(linha)
    return pd.DataFrame(linhas)


def _em_ordem(valores, referencia) -> list:
    vistos = list(dict.fromkeys(valores))
    return [v for v in referencia if v in vistos] + [v for v in vistos if v not in referencia]


# ---------------------------------------------------------------------------
# Modelos clássicos (não dirigidos)
# ---------------------------------------------------------------------------
def acrescentar_sigma(descritiva: pd.DataFrame) -> pd.DataFrame:
    """Coluna sigma: (C/C_ER)/(L/L_ER) de cada rede, com C_ER e L_ER as médias das redes erdos_renyi.

    NaN em tudo se não houver redes Erdős–Rényi (modelos_classicos.sigma_mundo_pequeno).
    """
    descritiva = descritiva.copy()
    er = descritiva[descritiva["modelo"] == "erdos_renyi"]
    c_er = float(er["clustering_medio"].mean()) if len(er) else math.nan
    l_er = float(er["caminho_medio"].mean()) if len(er) else math.nan
    descritiva["sigma"] = [modelos_classicos.sigma_mundo_pequeno(c, l, c_er, l_er)
                           for c, l in zip(descritiva["clustering_medio"], descritiva["caminho_medio"])]
    return descritiva


def confronto_classicos(descritiva: pd.DataFrame, indices: pd.DataFrame, real: str) -> pd.DataFrame:
    """Como confronto_nulos, para os modelos clássicos: uma linha por métrica (e estratégia, nos
    índices), com o valor da rede real e, para cada modelo, <modelo>_media, <modelo>_dp e <modelo>_z.

    O z é o de escore_z: NaN quando o modelo preserva a métrica por construção
    (as arestas no Erdős–Rényi e na configuração, o grau máximo na configuração).
    """
    modelos = [m for m in config.MODELOS_CLASSICOS if m in set(descritiva["modelo"])]
    linhas = []

    def linha_de(metrica: str, estrategia: str, tabela: pd.DataFrame) -> dict:
        valor = float(tabela.loc[tabela["modelo"] == real, metrica].iloc[0])
        linha = {"rede_real": real, "metrica": metrica, "estrategia": estrategia, "real": valor}
        for modelo in modelos:
            amostra = tabela.loc[tabela["modelo"] == modelo, metrica]
            linha[f"{modelo}_media"], linha[f"{modelo}_dp"], linha[f"{modelo}_z"] = escore_z(valor, amostra)
        return linha

    for metrica in DESCRITIVAS_CLASSICOS:
        linhas.append(linha_de(metrica, "", descritiva))
    for estrategia in _em_ordem(indices["estrategia"], config.ESTRATEGIAS_CLASSICOS):
        da_estrategia = indices[indices["estrategia"] == estrategia]
        for metrica in INDICES_CLASSICOS:
            linhas.append(linha_de(metrica, estrategia, da_estrategia))
    return pd.DataFrame(linhas)


def rodar_classicos(org: str) -> pd.DataFrame:
    """A E. coli (não dirigida) contra os modelos clássicos. Grava as tabelas exp4_classicos_*.

    Passos:
      1. U = modelos_classicos.rede_nao_dirigida(grafo_do_organismo(org)).
      2. Na rede real: modelos_classicos.descritiva e robustez_rede com
         config.N_REPETICOES_ALEATORIAS repetições da aleatória (as mesmas ordens do
         Exp 1: o maior componente não dirigido é o maior componente fraco, então o R da
         aleatória é o R_fraco do Exp 1 sem proteger os nutrientes).
      3. Para cada modelo de config.MODELOS_CLASSICOS, config.N_MODELOS_NULOS redes
         (modelos_classicos.gerar_modelos, semente config.SEMENTE * 1000 + i), cada uma com
         config.N_REPETICOES_NULOS repetições da aleatória e a semente da própria rede.
      4. sigma de mundo pequeno (acrescentar_sigma) e o confronto (escores z).
    Devolve a tabela descritiva.
    """
    inicio = time.perf_counter()
    U = modelos_classicos.rede_nao_dirigida(grafo_do_organismo(org))
    descritivas, indices, curvas, ccdfs = [], [], [], []

    def medir(rede: nx.Graph, modelo: str, indice, n_rep: int, semente: int) -> pd.DataFrame:
        descritivas.append({"modelo": modelo, "indice": indice,
                            "parametros": rede.graph.get("parametros", "rede real"),
                            **modelos_classicos.descritiva(rede)})
        curva, ind = modelos_classicos.robustez_rede(rede, config.ESTRATEGIAS_CLASSICOS, n_rep, semente)
        indices.append(ind.assign(modelo=modelo, indice=indice))
        return curva

    curvas.append(medir(U, org, pd.NA, config.N_REPETICOES_ALEATORIAS, config.SEMENTE).assign(modelo=org, n_redes=1))
    ccdfs.append(modelos_classicos.ccdf_graus([U]).assign(modelo=org))
    from tqdm import tqdm  # import tardio, como no robustez.py
    for modelo in config.MODELOS_CLASSICOS:
        redes = list(modelos_classicos.gerar_modelos(U, modelo, config.N_MODELOS_NULOS, config.SEMENTE))
        curvas_modelo = [medir(rede, modelo, i, config.N_REPETICOES_NULOS, config.SEMENTE * 1000 + i)
                         for i, rede in enumerate(tqdm(redes, desc=modelo, unit="rede"))]
        curvas.append(modelos_classicos.media_das_redes(curvas_modelo).assign(modelo=modelo))
        ccdfs.append(modelos_classicos.ccdf_graus(redes).assign(modelo=modelo))

    descritiva = acrescentar_sigma(pd.DataFrame(descritivas))
    tabela_indices = pd.concat(indices, ignore_index=True)
    tabela_indices = tabela_indices[["modelo", "indice", *modelos_classicos.COLUNAS_INDICES]]
    tabela_curvas = pd.concat(curvas, ignore_index=True)
    tabela_curvas = tabela_curvas[["modelo", "n_redes", *modelos_classicos.COLUNAS_CURVA]]
    _gravar(descritiva, "classicos_descritiva")
    _gravar(tabela_indices, "classicos_indices")
    _gravar(tabela_curvas, "classicos_curvas")
    _gravar(confronto_classicos(descritiva, tabela_indices, org), "classicos_z")
    _gravar(pd.concat(ccdfs, ignore_index=True)[["modelo", "grau", "n_nos", "ccdf_empirica"]], "classicos_ccdf")
    print(f"Exp 4, modelos clássicos: {len(descritiva)} redes em {time.perf_counter() - inicio:.0f} s")
    return descritiva


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def _gravar(tabela: pd.DataFrame, conteudo: str) -> None:
    destino = config.arquivo_tabela("exp4", conteudo, "redes", config.VERSAO_COMPARACAO)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)


def _ler(conteudo: str) -> pd.DataFrame | None:
    caminho = config.arquivo_tabela("exp4", conteudo, "redes", config.VERSAO_COMPARACAO)
    return pd.read_csv(caminho, keep_default_na=True) if caminho.exists() else None


def rodar(orgs: list[str], com_nulos: bool = True, com_classicos: bool = True) -> pd.DataFrame:
    """Calcula métricas e robustez de cada rede e grava as tabelas. Devolve a descritiva.

    com_classicos: no fim, rodar_classicos(orgs[0]) (a seção dos modelos clássicos).
    Passos:
      1. Para cada org: G = grafo_do_organismo(org); medir_rede(G, config.NUTRIENTES,
         config.N_REPETICOES_ALEATORIAS, config.SEMENTE) (as mesmas repetições do Exp 1,
         então a E. coli dá os mesmos índices dele) e o ajuste de lei de potência do grau total.
      2. Se com_nulos, a partir do G do primeiro org, para cada tipo em TIPOS_NULOS:
           para cada rede i de modelos_nulos.gerar_nulos(G_ref, tipo):
             medir_rede com n_rep=config.N_REPETICOES_NULOS e semente = config.SEMENTE * 1000 + i
             (a mesma da rede); nutrientes só nos rewirings, que mantêm os nomes dos nós.
         As curvas dos nulos são guardadas como média por tipo (media_das_redes).
      3. Coluna "rede" em todas as tabelas ("eco", "bsu", "mge", "rewiring", ...);
         coluna "indice" para as redes nulas (0 a N_MODELOS_NULOS - 1), vazia nos organismos.
      4. Com os nulos, o confronto do primeiro org com cada tipo (confronto_nulos).
    """
    inicio = time.perf_counter()
    descritivas, indices, curvas, ccdfs = [], [], [], []
    grafos = {}
    for org in orgs:
        G = grafos[org] = grafo_do_organismo(org)
        print(f"{org}: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        desc, curva, ind = medir_rede(G, config.NUTRIENTES, config.N_REPETICOES_ALEATORIAS,
                                      config.SEMENTE, progresso=True)
        lp = metricas.ajuste_lei_potencia(d for _, d in G.degree())
        descritivas.append({"rede": org, "indice": pd.NA, **desc, "trocas": pd.NA,
                            "lp_alpha": lp["alpha"], "lp_xmin": lp["xmin"], "lp_n_cauda": lp["n_cauda"],
                            "lp_R_vs_lognormal": lp["R_vs_lognormal"], "lp_p": lp["p"]})
        indices.append(ind.assign(rede=org, indice=pd.NA))
        curvas.append(curva.assign(rede=org, n_redes=1))
        ccdfs.append(_ccdf_empirica(G).assign(rede=org))

    if com_nulos:
        from tqdm import tqdm  # import tardio, como no robustez.py
        G_ref = grafos[orgs[0]]
        for tipo in TIPOS_NULOS:
            nutrientes = () if tipo == "erdos_renyi" else config.NUTRIENTES
            curvas_tipo = []
            redes = modelos_nulos.gerar_nulos(G_ref, tipo, config.N_MODELOS_NULOS, config.SEMENTE)
            for i, N in enumerate(tqdm(redes, total=config.N_MODELOS_NULOS, desc=tipo, unit="rede")):
                desc, curva, ind = medir_rede(N, nutrientes, config.N_REPETICOES_NULOS,
                                              config.SEMENTE * 1000 + i)
                descritivas.append({"rede": tipo, "indice": i, **desc, "trocas": N.graph.get("trocas", pd.NA)})
                indices.append(ind.assign(rede=tipo, indice=i))
                curvas_tipo.append(curva)
            curvas.append(media_das_redes(curvas_tipo).assign(rede=tipo))

    descritiva = pd.DataFrame(descritivas)
    descritiva = descritiva[["rede", "indice", *[c for c in descritiva.columns if c not in ("rede", "indice")]]]
    tabela_indices = pd.concat(indices, ignore_index=True).reindex(columns=list(COLUNAS_INDICES))
    tabela_curvas = pd.concat(curvas, ignore_index=True)
    primeiras = ["rede", "estrategia", "n_redes"]
    tabela_curvas = tabela_curvas[[*primeiras, *[c for c in tabela_curvas.columns if c not in primeiras]]]

    _gravar(descritiva, "descritiva")
    _gravar(tabela_indices, "indices")
    _gravar(tabela_curvas, "curvas")
    _gravar(pd.concat(ccdfs, ignore_index=True).reindex(columns=list(COLUNAS_CCDF)), "ccdf")
    if com_nulos:
        _gravar(confronto_nulos(descritiva, tabela_indices, orgs[0]), "nulos")
    print(f"Exp 4: {len(descritiva)} redes em {time.perf_counter() - inicio:.0f} s")
    if com_classicos:
        rodar_classicos(orgs[0])
    return descritiva


def plotar() -> None:
    """Lê as tabelas e grava as figuras: as quatro das redes (a dos nulos só se eles foram rodados)
    e as três dos modelos clássicos, se as tabelas deles existirem."""
    figuras.estilo()
    descritiva, indices, curvas, ccdf = (_ler(c) for c in ("descritiva", "indices", "curvas", "ccdf"))
    if descritiva is None or indices is None or curvas is None or ccdf is None:
        raise FileNotFoundError("faltam as tabelas do Exp 4; rode antes sem --so-figuras")
    organismos = [r for r in dict.fromkeys(descritiva["rede"]) if r not in TIPOS_NULOS]
    nulos = [t for t in TIPOS_NULOS if t in set(descritiva["rede"])]
    versao = config.VERSAO_COMPARACAO

    figuras.salvar(figuras.painel_indices_redes(indices),
                   config.arquivo_figura("exp4", "indices", "redes", versao))
    figuras.salvar(figuras.painel_curvas_redes(curvas[curvas["rede"].isin(organismos)],
                                               estrategias=ESTRATEGIAS_FIGURA),
                   config.arquivo_figura("exp4", "curvas", "organismos", versao))
    if nulos:
        figuras.salvar(figuras.painel_curvas_redes(curvas[curvas["rede"].isin([organismos[0], *nulos])],
                                                   estrategias=ESTRATEGIAS_FIGURA),
                       config.arquivo_figura("exp4", "curvas", "nulos", versao))
    figuras.salvar(figuras.ccdf_por_rede(ccdf), config.arquivo_figura("exp4", "graus", "redes", versao))

    classicos = {c: _ler(f"classicos_{c}") for c in ("indices", "curvas", "ccdf")}
    if all(tabela is not None for tabela in classicos.values()):
        figuras.salvar(figuras.painel_indices_redes(classicos["indices"], ("R_norm", "metade"), grupo="modelo"),
                       config.arquivo_figura("exp4", "classicos_indices", "redes", versao))
        figuras.salvar(figuras.painel_curvas_redes(classicos["curvas"], ("maior_componente", "maior_componente_rel"),
                                                   ESTRATEGIAS_FIGURA_CLASSICOS, limites_x={"grau": 0.5},
                                                   grupo="modelo"),
                       config.arquivo_figura("exp4", "classicos_curvas", "redes", versao))
        figuras.salvar(figuras.ccdf_por_rede(classicos["ccdf"], grupo="modelo"),
                       config.arquivo_figura("exp4", "classicos_graus", "redes", versao))


def _resumo() -> str:
    descritiva, indices, confronto = _ler("descritiva"), _ler("indices"), _ler("nulos")
    linhas = [f"{'rede':19s} {'S0':>5s} {'núcleo':>6s} {'recipr.':>7s} "
              + " ".join(f"{e[:10]:>17s}" for e in config.ESTRATEGIAS_COMPARACAO),
              f"{'':19s} {'':5s} {'':6s} {'':7s} " + " ".join(f"{'Rnorm  metade':>17s}"
                                                          for _ in config.ESTRATEGIAS_COMPARACAO)]
    media = descritiva.groupby("rede", sort=False)[["maior_fraco", "nucleo", "reciprocidade"]].mean()
    por_rede = indices.groupby(["rede", "estrategia"], sort=False)[["R_fraco_norm", "metade_fraco"]].mean()
    for rede, d in media.iterrows():
        partes = [f"{por_rede.loc[(rede, e), 'R_fraco_norm']:7.3f} {por_rede.loc[(rede, e), 'metade_fraco']:7.1%}"
                  if (rede, e) in por_rede.index else f"{'':17s}" for e in config.ESTRATEGIAS_COMPARACAO]
        linhas.append(f"{rede:19s} {d['maior_fraco']:5.2f} {d['nucleo']:6.0f} {d['reciprocidade']:7.3f} "
                      + " ".join(f"{p:>17s}" for p in partes))
    if confronto is not None:
        tipos = [t for t in TIPOS_NULOS if f"{t}_z" in confronto]
        linhas.append(f"\nescores z de {confronto['rede_real'].iloc[0]} contra os nulos (|z| > 2: o nulo não explica)")
        linhas.append(f"  {'métrica':32s} {'real':>8s} " + " ".join(f"{t[:18]:>18s}" for t in tipos))
        for linha in confronto.fillna({"estrategia": ""}).to_dict("records"):
            nome = linha["metrica"] + (f" ({linha['estrategia']})" if linha["estrategia"] else "")
            linhas.append(f"  {nome:32s} {linha['real']:8.3f} "
                          + " ".join(f"{linha[f'{t}_z']:18.1f}" for t in tipos))
    classicos = _resumo_classicos()
    if classicos:
        linhas.append("\n" + classicos)
    return "\n".join(linhas)


def _resumo_classicos() -> str:
    """A tabela dos modelos clássicos (médias das redes de cada modelo) e os escores z da rede real."""
    descritiva, indices, confronto = (_ler(f"classicos_{c}") for c in ("descritiva", "indices", "z"))
    if descritiva is None or indices is None:
        return ""
    media = descritiva.groupby("modelo", sort=False)[
        ["arestas", "grau_max", "clustering_medio", "maior_componente", "caminho_medio", "assortatividade",
         "sigma"]].mean()
    por = indices.groupby(["modelo", "estrategia"], sort=False)[["R", "R_norm", "metade"]].mean()
    linhas = ["modelos clássicos (não dirigidos; média das redes de cada modelo; R sem normalizar e normalizado)",
              f"  {'rede':15s} {'M':>6s} {'kmax':>5s} {'C':>6s} {'%comp':>6s} {'L':>6s} {'assort':>7s} {'sigma':>6s}"
              f" {'R aleat':>8s} {'Rn aleat':>8s} {'metade':>7s} {'R grau':>7s} {'Rn grau':>7s} {'metade':>7s}"
              f" {'R betw':>7s}"]
    for modelo, d in media.iterrows():
        def valor(estrategia, coluna):
            return por.loc[(modelo, estrategia), coluna] if (modelo, estrategia) in por.index else math.nan
        linhas.append(f"  {str(modelo):15s} {d['arestas']:6.0f} {d['grau_max']:5.1f} {d['clustering_medio']:6.3f}"
                      f" {100 * d['maior_componente']:6.1f} {d['caminho_medio']:6.2f} {d['assortatividade']:7.3f}"
                      f" {d['sigma']:6.1f} {valor('aleatoria', 'R'):8.3f} {valor('aleatoria', 'R_norm'):8.3f}"
                      f" {valor('aleatoria', 'metade'):7.3f} {valor('grau', 'R'):7.3f} {valor('grau', 'R_norm'):7.3f}"
                      f" {valor('grau', 'metade'):7.3f} {valor('betweenness', 'R'):7.3f}")
    if confronto is not None:
        modelos = [m for m in config.MODELOS_CLASSICOS if f"{m}_z" in confronto]
        linhas.append(f"  escores z de {confronto['rede_real'].iloc[0]} (R normalizado e descritivas)")
        linhas.append(f"    {'métrica':28s} {'real':>8s} " + " ".join(f"{m[:14]:>14s}" for m in modelos))
        escolhidas = confronto[(confronto["metrica"] == "R_norm")
                               | confronto["estrategia"].fillna("").eq("")]
        for linha in escolhidas.fillna({"estrategia": ""}).to_dict("records"):
            nome = linha["metrica"] + (f" ({linha['estrategia']})" if linha["estrategia"] else "")
            linhas.append(f"    {nome:28s} {linha['real']:8.3f} "
                          + " ".join(f"{linha[f'{m}_z']:14.1f}" for m in modelos))
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 4: comparação com outros organismos, modelos nulos "
                                                 "e modelos clássicos.")
    parser.add_argument("--orgs", nargs="+",
                        default=[config.ORGANISMO_PRINCIPAL, *config.ORGANISMOS_COMPARACAO],
                        help="códigos KEGG; o primeiro é a referência dos modelos nulos e clássicos")
    parser.add_argument("--sem-nulos", action="store_true", help="pula os modelos nulos (teste rápido)")
    parser.add_argument("--sem-classicos", action="store_true", help="pula os modelos clássicos")
    parser.add_argument("--so-figuras", action="store_true", help="refaz só as figuras, sem recalcular")
    args = parser.parse_args()

    if not args.so_figuras:
        rodar(args.orgs, com_nulos=not args.sem_nulos, com_classicos=not args.sem_classicos)
        print(_resumo())
    plotar()


if __name__ == "__main__":
    main()
