"""Configuração central do projeto.

Tudo que é parâmetro (caminhos, organismos, listas de metabólitos, semente,
número de repetições) fica aqui, para que a coleta e os experimentos usem
os mesmos valores. O formato das tabelas fica em `contrato.py`.

Rode tudo a partir da raiz do projeto, como módulo:

    python -m coleta.baixar_kgml --org eco
    python -m experimentos.exp1_robustez --org eco
    python -m pytest testes/
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
# Organismos
# ---------------------------------------------------------------------------
ORGANISMO_PRINCIPAL = "eco"            # E. coli K-12 MG1655
ORGANISMOS_COMPARACAO = ["bsu", "mge"]  # alternativa: "sce"

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
SEMENTE = 42

# ---------------------------------------------------------------------------
# Metabólitos-moeda
# Conferir contra os nós de maior grau na primeira execução do exp0.
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

# As duas versões de toda análise: nome -> remover metabólitos-moeda?
VERSOES_MOEDA = {"sem_moeda": True, "com_moeda": False}

# Versão usada pelos Exps 2 e 3. Na cascata, os metabólitos-moeda são sempre
# fontes, então o resultado é o mesmo nas duas versões; os hubs de controle
# do Exp 3 saem desta versão do grafo.
VERSAO_CASCATA = "sem_moeda"

# Versão padrão do Exp 1. A com moeda só repetiria a mesma conclusão com o
# dobro de figuras; continua disponível com --versao com_moeda ou ambas.
VERSAO_ROBUSTEZ = "sem_moeda"

# ---------------------------------------------------------------------------
# Meio mínimo (Exp 1: alcance a partir de nutrientes)
# Atenção: NH3 e ortofosfato também são metabólitos-moeda. Na versão
# "sem_moeda" eles saem do grafo e o alcance parte só dos que sobraram.
# Conferir os códigos na extração (a glicose pode aparecer como C00267).
# ---------------------------------------------------------------------------
NUTRIENTES = {
    "C00031": "D-Glicose",
    "C00014": "NH3",
    "C00009": "Ortofosfato",
    "C00059": "Sulfato",
}

# ---------------------------------------------------------------------------
# Experimento 1 — Robustez estrutural
# Cada centralidade nas versões estática (calculada uma vez, na rede intacta) e
# adaptativa (recalculada na rede que sobrou). O grau é o total (entrada + saída)
# no grafo simples. Nas remoções, os nutrientes (fontes do alcance) vão para o
# fim da ordem (robustez.simular_remocao, proteger_fontes=True).
# ---------------------------------------------------------------------------
ESTRATEGIAS = ("aleatoria", "grau", "grau_adaptativo", "betweenness", "betweenness_adaptativo",
               "harmonica", "harmonica_adaptativo", "pagerank", "pagerank_adaptativo")
N_REPETICOES_ALEATORIAS = 100
PASSO_EFICIENCIA = 0.05     # mede a eficiência a cada 5% da rede removida
PASSO_BETWEENNESS = 10      # nas adaptativas de betweenness, harmônica e PageRank, recalcula a cada k remoções

# ---------------------------------------------------------------------------
# Experimento 2 — Dano x essencialidade dos genes (gabarito: Keio)
# ---------------------------------------------------------------------------
K_PRECISAO = (10, 25, 50)   # precisão nos k primeiros do ranking

# Como o grau e a betweenness dos metabólitos viram a pontuação de um gene:
# "max" (o metabólito mais central das reações do gene) ou "soma".
AGREGACAO_CENTRALIDADE_GENE = "max"

# Medida de dano funcional (cascata) de um nocaute:
#   "avalanche" -> metabólitos perdidos com as fontes externas (reaproveita o Exp 3)
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

# ---------------------------------------------------------------------------
# Experimento 3 — Dano individual dos metabólitos (não há gabarito para metabólitos)
# As 6 substâncias e os 12 controles continuam nas tabelas, como exemplos.
# ---------------------------------------------------------------------------
SUBSTANCIAS_EXP3 = {
    "C00022": "Piruvato",
    "C00024": "Acetil-CoA",
    "C00036": "Oxaloacetato",
    "C00026": "2-Oxoglutarato",
    "C00025": "L-Glutamato",
    "C00092": "Glicose-6-fosfato",
}
N_CONTROLES_EXP3 = 6        # 6 hubs de maior grau + 6 sorteados de baixo grau
QUANTIL_BAIXO_GRAU = 0.25   # os de baixo grau são sorteados entre os 25% de menor grau
N_ROTULOS_EXP3 = 3          # na dispersão dano estrutural x avalanche: os k mais discordantes de cada lado

# Casos para a discussão (tabela exp3_exemplos): código -> o que ilustram.
# O C04734 é o FAICAR (formil-AICAR, 5-formamido-imidazol-4-carboxamida ribotídeo);
# o AICAR propriamente dito é o C04677, o substrato da reação anterior (PurH).
EXEMPLOS_DANO_EXP3 = {
    "C04734": "síntese de purinas: tira dezenas de metabólitos do núcleo e não interrompe nenhuma produção",
    "C00130": "tira dezenas de metabólitos do núcleo e interrompe só uma produção",
    "C00083": "grau baixo, mas uma das maiores avalanches (síntese de ácidos graxos)",
    "C00024": "alta nas duas medidas",
}

# Nomes curtos para os rótulos das figuras do Exp 3 (o nome do KEGG é longo demais
# para rótulo); os que não estão aqui aparecem com o nome do KEGG.
NOMES_CURTOS_EXP3 = {
    "C04734": "FAICAR", "C00130": "IMP", "C00083": "Malonil-CoA", "C00024": "Acetil-CoA",
    "C00655": "XMP", "C00365": "dUMP", "C01269": "EPSP", "C04501": "GlcNAc-1P", "C00043": "UDP-GlcNAc",
    "C05753": "3-Oxodecanoil-ACP", "C04619": "3-Hidroxidecanoil-ACP", "C00144": "GMP",
    "C00022": "Piruvato", "C00025": "L-Glutamato", "C00036": "Oxaloacetato",
}

# ---------------------------------------------------------------------------
# Experimento 4 — Comparação com outras redes
# Estratégias mais baratas para não multiplicar o custo do Exp 1.
# ---------------------------------------------------------------------------
ESTRATEGIAS_COMPARACAO = ("aleatoria", "grau", "grau_adaptativo")
VERSAO_COMPARACAO = "sem_moeda"  # o Exp 4 roda numa versão só, para caber no tempo
# Tipos de modelo nulo: modelos_nulos.TIPOS (rewiring recíproco, rewiring, Erdős–Rényi).
N_MODELOS_NULOS = 10        # redes sorteadas por tipo de modelo nulo (30 no total, ~14 s cada)
N_REPETICOES_NULOS = 20     # repetições da remoção aleatória em cada rede nula
TROCAS_POR_ARESTA = 10      # rewiring: TROCAS_POR_ARESTA * número de arestas trocas aceitas
MEDIR_EFICIENCIA_EXP4 = False  # o índice R não depende dela; sem ela, o Exp 4 leva ~9 min

# Modelos clássicos (seção nova do Exp 4, na versão NÃO dirigida da rede): mesmo N e
# mesmo M da E. coli (ou o mais perto possível), config.N_MODELOS_NULOS redes de cada,
# com a semente SEMENTE * 1000 + i. Ver experimentos/biblioteca/modelos_classicos.py.
MODELOS_CLASSICOS = ("configuracao", "erdos_renyi", "ba_misto", "ba_m1", "newman_watts", "watts_strogatz")
# Estratégias de remoção nos modelos clássicos (maior componente conexo). As estáticas
# e a aleatória saem por percolação reversa (union-find); o grau adaptativo, da
# sequência calculada passo a passo.
ESTRATEGIAS_CLASSICOS = ("aleatoria", "grau", "grau_adaptativo", "betweenness", "harmonica", "pagerank")
K_WATTS_STROGATZ = 4        # WS de clustering alto: com k = 4 tem o dobro das arestas da E. coli
P_WATTS_STROGATZ = 0.1
