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
}

# As duas versões de toda análise: nome -> remover metabólitos-moeda?
VERSOES_MOEDA = {"sem_moeda": True, "com_moeda": False}

# Versão usada pelos Exps 2 e 3. Na cascata, os metabólitos-moeda são sempre
# fontes, então o resultado é o mesmo nas duas versões; os hubs de controle
# do Exp 3 saem desta versão do grafo.
VERSAO_CASCATA = "sem_moeda"

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
# ---------------------------------------------------------------------------
ESTRATEGIAS = ("aleatoria", "grau", "grau_adaptativo", "betweenness", "betweenness_adaptativo")
N_REPETICOES_ALEATORIAS = 100
PASSO_EFICIENCIA = 0.05     # mede a eficiência a cada 5% da rede removida
PASSO_BETWEENNESS = 10      # na versão adaptativa, recalcula a cada k remoções

# ---------------------------------------------------------------------------
# Experimento 2 — Topologia vs. biologia
# ---------------------------------------------------------------------------
K_PRECISAO = (10, 25, 50)   # precisão nos k primeiros do ranking

# Como o grau e a betweenness dos metabólitos viram a pontuação de um gene:
# "max" (o metabólito mais central das reações do gene) ou "soma".
AGREGACAO_CENTRALIDADE_GENE = "max"

# Medida de dano de um nocaute:
#   "avalanche" -> metabólitos perdidos com as fontes externas (reaproveita o Exp 3)
#   "alcance"   -> metabólitos perdidos partindo só do meio mínimo (NUTRIENTES)
DANO_NOCAUTE = "avalanche"

# ---------------------------------------------------------------------------
# Experimento 3 — Viés por substância
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

# ---------------------------------------------------------------------------
# Experimento 4 — Comparação com outras redes
# Estratégias mais baratas para não multiplicar o custo do Exp 1.
# ---------------------------------------------------------------------------
ESTRATEGIAS_COMPARACAO = ("aleatoria", "grau", "grau_adaptativo")
VERSAO_COMPARACAO = "sem_moeda"  # o Exp 4 roda numa versão só, para caber no tempo
N_MODELOS_NULOS = 10        # redes sorteadas por tipo de modelo nulo
N_REPETICOES_NULOS = 20     # repetições da remoção aleatória em cada rede nula
TROCAS_POR_ARESTA = 10      # rewiring: nswap = TROCAS_POR_ARESTA * número de arestas
MEDIR_EFICIENCIA_EXP4 = False  # corta o Exp 4 de ~75 para ~20 min; o índice R não depende dela
