"""Configuração central do projeto.

Tudo que é parâmetro (caminhos, organismo, listas de metabólitos, semente,
número de repetições) fica aqui, para que a coleta e os experimentos usem
os mesmos valores. O formato das tabelas fica em `contrato.py`.

Rode tudo a partir da raiz do projeto, como módulo:

    python -m coleta.baixar_kgml --org eco
    python -m experimentos.exp1_robustez
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"

PASTA_TABELAS = RESULTADOS / "tabelas"
PASTA_FIGURAS = RESULTADOS / "figuras"

# Gabarito de genes essenciais (baixado à mão; ver dados/keio/LEIAME.md)
ARQ_KEIO = DADOS / "keio" / "essenciais.csv"

# Respostas brutas da API que não dependem de organismo
ARQ_COMPOSTOS = DADOS / "kgml" / "_compostos.tsv"   # list/compound
ARQ_GLICANOS = DADOS / "kgml" / "_glicanos.tsv"     # list/glycan


def pasta_brutos(org: str) -> Path:
    """Cache dos downloads de um organismo (KGML de cada via + listas)."""
    return DADOS / "kgml" / org


def pasta_processados(org: str) -> Path:
    """Tabelas montadas pela coleta, lidas pelos experimentos."""
    return DADOS / org


def caminho_legivel(caminho: Path) -> str:
    """Caminho relativo à raiz do projeto, para mensagens; o absoluto se estiver fora dela."""
    try:
        return str(caminho.relative_to(RAIZ))
    except ValueError:
        return str(caminho)


def _nome_resultado(exp: str, conteudo: str, org: str, versao: str | None) -> str:
    partes = [exp, conteudo, org] + ([versao] if versao else [])
    return "_".join(partes)


def arquivo_tabela(exp: str, conteudo: str, org: str, versao: str | None = None) -> Path:
    """resultados/tabelas/<exp>_<conteudo>_<org>[_<versao>].csv

    Ex.: arquivo_tabela("exp1", "curvas", "eco", "sem_moeda")
         -> resultados/tabelas/exp1_curvas_eco_sem_moeda.csv
    """
    return PASTA_TABELAS / f"{_nome_resultado(exp, conteudo, org, versao)}.csv"


def arquivo_figura(exp: str, conteudo: str, org: str, versao: str | None = None) -> Path:
    """resultados/figuras/<exp>_<conteudo>_<org>[_<versao>].<FORMATO_FIGURA>"""
    return PASTA_FIGURAS / f"{_nome_resultado(exp, conteudo, org, versao)}.{FORMATO_FIGURA}"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
FORMATO_FIGURA = "png"
DPI_FIGURA = 200


# ---------------------------------------------------------------------------
# KEGG
# ---------------------------------------------------------------------------
KEGG_URL = "https://rest.kegg.jp"
PAUSA_ENTRE_REQUISICOES = 0.5   # segundos; com o cache, cada arquivo é baixado uma vez
TENTATIVAS_REQUISICAO = 3       # tentativas para erros temporários (403, 429, 5xx, conexão)
TIMEOUT_REQUISICAO = 30         # segundos de espera por resposta
LIMITE_VIAS_REGULARES = 1100    # mantém vias com número < 01100 (descarta 011xx e 012xx)

# Vias abaixo de 01100 que o KEGG classifica fora de Metabolismo: não entram na rede.
VIAS_EXCLUIDAS = {
    "00970": "Aminoacyl-tRNA biosynthesis (KEGG: Genetic Information Processing > Translation)",
}

# Fração de entradas de gene sem <reaction> acima da qual o relatório de
# extração marca a rede para reparo com o mapa global eco01100.
LIMITE_LACUNA_GENES = 0.10

# ---------------------------------------------------------------------------
# Organismo
# ---------------------------------------------------------------------------
ORGANISMO_PRINCIPAL = "eco"            # E. coli K-12 MG1655

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
SEMENTE = 42

# ---------------------------------------------------------------------------
# Metabólitos-moeda (17): saem da rede principal. Os metabólitos mais frequentes
# nas reações ficam no relatorio_extracao.json da coleta, para conferência.
# ---------------------------------------------------------------------------
METABOLITOS_MOEDA = {
    "C00001": "H2O",
    "C00002": "ATP",
    "C00003": "NAD+",
    "C00004": "NADH",
    "C00005": "NADPH",
    "C00006": "NADP+",
    "C00007": "O2",
    "C00008": "ADP",
    "C00009": "Ortofosfato",
    "C00010": "CoA",
    "C00011": "CO2",
    "C00013": "Difosfato",
    "C00014": "NH3",
    "C00020": "AMP",
    "C00080": "H+",
    # Carregador de unidades de um carbono, como a CoA é de grupos acila: sai só
    # a forma livre. As formas carregadas (5,10-metileno-THF, 10-formil-THF,
    # 5-metil-THF) ficam, como a acetil-CoA fica, porque levam o carbono adiante.
    "C00101": "THF (tetra-hidrofolato)",
    # Carregador de grupos acila na síntese de ácidos graxos, como a CoA é fora dela:
    # sai só a forma livre (holo-ACP); os acil-ACPs ficam. Sem isso, ACP e apo-ACP
    # só se produzem um do outro (a origem real, a tradução do gene acpP, não é
    # reação do KEGG) e a cascata nunca chegaria aos ácidos graxos e ao lipídio A.
    "C00229": "ACP (proteína carregadora de acila)",
}

# As duas versões da rede: nome -> remover metabólitos-moeda?
VERSOES_MOEDA = {"sem_moeda": True, "com_moeda": False}

# A versão de todas as análises (Exps 0 a 4). A com moeda entra só na
# comparação da Tabela 1 (Exp 0). Na cascata, os metabólitos-moeda são sempre
# fontes, então o resultado é o mesmo nas duas versões.
VERSAO_PRINCIPAL = "sem_moeda"

# ---------------------------------------------------------------------------
# Meio mínimo (Exp 1: alcance a partir de nutrientes; checagem do chiquimato)
# Atenção: NH3 e ortofosfato também são metabólitos-moeda. Na versão
# "sem_moeda" eles saem do grafo e o alcance parte só da glicose e do sulfato.
# ---------------------------------------------------------------------------
NUTRIENTES = {
    "C00031": "D-Glicose",
    "C00014": "NH3",
    "C00009": "Ortofosfato",
    "C00059": "Sulfato",
}

# ---------------------------------------------------------------------------
# Experimento 1 — Remoção de nós por centralidade (P1)
# Cada centralidade nas versões estática (calculada uma vez, na rede intacta) e
# adaptativa (recalculada na rede que sobrou). O grau é o total (entrada + saída)
# no grafo simples. Nas remoções, os nutrientes (fontes do alcance) vão para o
# fim da ordem (robustez.simular_remocao).
# ---------------------------------------------------------------------------
ESTRATEGIAS = ("aleatoria", "grau", "grau_adaptativo", "betweenness", "betweenness_adaptativo",
               "pagerank", "pagerank_adaptativo")
N_REPETICOES_ALEATORIAS = 100
PASSO_BETWEENNESS = 10      # nas adaptativas de betweenness e PageRank, recalcula a cada k remoções

# ---------------------------------------------------------------------------
# Experimento 2 — Dano x essencialidade dos genes (P2; gabarito: Keio)
# ---------------------------------------------------------------------------
K_PRECISAO = (10, 25, 50)   # precisão nos k primeiros do ranking

# Como o grau, a betweenness e o PageRank dos metabólitos viram a pontuação de um
# gene: "max" (o metabólito mais central das reações do gene) ou "soma".
AGREGACAO_CENTRALIDADE_GENE = "max"

# Medida de dano funcional (cascata) de um nocaute:
#   "avalanche" -> metabólitos perdidos partindo das fontes externas (o "meio rico")
#   "alcance"   -> metabólitos perdidos partindo só do meio mínimo (NUTRIENTES)
DANO_NOCAUTE = "avalanche"

# Casos para a discussão (tabela exp2_exemplos_dano): grupo -> (símbolos dos genes, o que ilustram).
EXEMPLOS_DANO_EXP2 = {
    "estrutural_alto_nao_essencial": (
        ("aroA", "aroB", "aroC", "aroD", "purH", "thyA"),
        "alto dano estrutural e não essencial: aroA-aroD são da via do chiquimato, e o meio LB "
        "já fornece os aminoácidos aromáticos; purH e thyA também tiram dezenas de metabólitos "
        "do núcleo sem interromper quase nenhuma produção"),
    "funcional_alto_essencial": (
        ("fabG", "fabD", "fabI", "lpxC"),
        "alto dano funcional e essencial: síntese de ácidos graxos (fabG, fabD, fabI) e do "
        "lipídio A (lpxC); a cascata vê a perda do que está a jusante, e a Keio confirma"),
    "essencial_dano_zero": (
        ("accA", "accB", "accC", "accD", "eno", "pgk"),
        "essenciais com dano zero na cascata e no núcleo: accA-accD formam um complexo que a regra "
        "OU trata como isoenzimas (nenhum gene sozinho derruba a reação); eno e pgk perdem a sua "
        "reação, mas a rede tem um desvio entre os mesmos metabólitos, sem garantia de fluxo"),
}

# Faixas de dano da Figura 4 (exp2_essenciais_por_faixa): rótulo -> maior dano da
# faixa (inclusive); a primeira começa em 0 e a última não tem teto.
# Só os genes que perdem pelo menos uma reação (os outros têm dano 0 em tudo).
FAIXAS_DANO_EXP2 = (("0", 0), ("1–2", 2), ("3–9", 9), ("10 ou mais", float("inf")))

# ---------------------------------------------------------------------------
# Experimento 4 — Comparação com os modelos clássicos (P3), na versão NÃO dirigida
# da rede: mesmo N e mesmo M da E. coli (ou o mais perto possível),
# N_REDES_POR_MODELO redes de cada modelo, com a semente SEMENTE * 1000 + i.
# Ver experimentos/biblioteca/modelos_classicos.py.
# ---------------------------------------------------------------------------
MODELOS_CLASSICOS = ("configuracao", "erdos_renyi", "ba_misto", "ba_m1", "newman_watts", "watts_strogatz")
# Estratégias de remoção (maior componente conexo): a aleatória e o ataque por
# grau, por percolação reversa (union-find).
ESTRATEGIAS_CLASSICOS = ("aleatoria", "grau")
N_REDES_POR_MODELO = 10     # redes sorteadas de cada modelo
N_REPETICOES_MODELOS = 20   # repetições da remoção aleatória em cada rede sorteada
TROCAS_POR_ARESTA = 10      # modelo de configuração: TROCAS_POR_ARESTA * M trocas de arestas
K_WATTS_STROGATZ = 4        # WS de clustering alto: com k = 4 tem o dobro das arestas da E. coli
P_WATTS_STROGATZ = 0.1

# ---------------------------------------------------------------------------
# Análises complementares da discussão (experimentos/checar_inferencias.py)
# Só na E. coli: a parte dos genes usa a tabela do Exp 2 (gabarito da Keio).
# ---------------------------------------------------------------------------
# Os primeiros k metabólitos de cada ranking, com k = round(fração * N): 3,5% (na
# E. coli, os 39 que a remoção por grau leva para derrubar o maior componente fraco
# à metade, metade_fraco do Exp 1) e 10% (111 de 1111).
TOPS_INFERENCIAS = (0.035, 0.10)
N_PRIMEIROS_INFERENCIAS = 10     # quantos nomes do topo de cada ranking vão para o resumo
MINIMOS_LADO_INFERENCIAS = (1, 5)  # genes que tiram pelo menos n metabólitos do maior fraco
# Via do chiquimato: os genes conferidos e os aminoácidos aromáticos acompanhados.
GENES_CHIQUIMATO = ("aroA", "aroB", "aroC", "aroD")
AROMATICOS = {"C00079": "Phe", "C00082": "Tyr", "C00078": "Trp"}
