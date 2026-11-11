# Estudo sobre a rede metabólica da *Escherichia coli*

Código, dados e resultados do projeto final de Redes Complexas (Ciência da Computação, UFSJ).
A rede metabólica da *E. coli* K-12 MG1655 é extraída do KEGG e analisada com as ferramentas
clássicas de robustez de redes; o que a rede aponta como importante é comparado com a
essencialidade medida em laboratório (coleção Keio). O trabalho responde a três perguntas:

| Pergunta | Script | No artigo |
|---|---|---|
| **P1.** Removendo metabólitos por grau, intermediação e outras centralidades, como a rede se desfaz? | `experimentos/exp1_robustez.py` | seção B.1, Tabela 2, Figura 3 |
| **P2.** O gene cuja deleção causa mais dano na rede é essencial? | `experimentos/exp2_biologia.py` | seção B.2, Tabela 3, Figura 4 |
| **P3.** A rede se parece com Erdős–Rényi, Watts–Strogatz ou Barabási–Albert? | `experimentos/exp4_comparacao.py` | seção B.3, Tabela 4 |

A análise descritiva (seção A, Tabela 1 e Figura 2) fica em `experimentos/exp0_descritiva.py`,
e `experimentos/checar_inferencias.py` calcula os números da discussão que não estão nas tabelas.

## Como rodar

Requisitos: Linux ou macOS, com bash e Python 3.12 ou mais novo (as versões travadas em
`requirements-lock.txt` exigem pelo menos o 3.12). Na pasta `rede-metabolica/`:

```bash
bash setup.sh                       # cria o .venv e instala as versões de requirements-lock.txt
bash executar_tudo.sh               # coleta (a partir do cache), Exps 0, 1, 2 e 4 e as análises complementares
bash executar_tudo.sh --so-figuras  # refaz só as figuras, a partir de resultados/tabelas/
```

O pipeline completo leva cerca de 3 minutos. Cada etapa também roda sozinha, como módulo:

```bash
source .venv/bin/activate
python -m coleta.baixar_kgml --org eco      # KGML das vias (só baixa o que falta em dados/kgml/eco/)
python -m coleta.montar_tabelas --org eco   # tabelas da rede em dados/eco/
python -m experimentos.exp0_descritiva      # Tabela 1 e Figura 2              (~5 s)
python -m experimentos.exp1_robustez        # P1: Tabela 2 e Figura 3          (~1,5 min)
python -m experimentos.exp2_biologia        # P2: Tabela 3 e Figura 4          (~40 s)
python -m experimentos.exp4_comparacao      # P3: Tabela 4                     (~15 s)
python -m experimentos.checar_inferencias   # números da discussão; depois do Exp 2 (~10 s)
```

Tudo é determinístico (semente `config.SEMENTE = 42`). Os parâmetros ficam em `config.py` e o
formato das tabelas de dados, em `contrato.py`. Os Exps 0, 1, 2 e 4 imprimem no fim um resumo
com os números das tabelas do artigo. A coleta não baixa de novo o que já está no cache (use
`--forcar` para baixar tudo), mas precisa de internet para consultar a versão da base no KEGG,
que fica registrada em `dados/kgml/eco/_metadados.json`.

## Estrutura

```
rede-metabolica/
├── config.py, contrato.py        parâmetros e formato das tabelas de dados
├── setup.sh, executar_tudo.sh    ambiente virtual e pipeline completo
├── coleta/
│   ├── baixar_kgml.py            download da API REST do KEGG, com cache em dados/kgml/
│   ├── montar_tabelas.py         parser próprio do KGML e fusão das vias -> dados/eco/
│   └── converter_keio.py         tabela suplementar da Keio -> dados/keio/essenciais.csv
├── experimentos/
│   ├── exp0_descritiva.py        análise descritiva
│   ├── exp1_robustez.py          P1
│   ├── exp2_biologia.py          P2
│   ├── exp4_comparacao.py        P3
│   ├── checar_inferencias.py     análises complementares da discussão
│   └── biblioteca/
│       ├── grafo.py              visões da rede: metabólitos, simples, não dirigida e bipartida
│       ├── metricas.py           descritivas, componentes, índice R, alcance, eficiência, intermediação
│       ├── robustez.py           ordens de remoção (estáticas e adaptativas) e curvas de robustez
│       ├── cascata.py            cascata de falhas na rede bipartida
│       ├── nocaute.py            nocaute de genes: dano funcional e estrutural
│       ├── validacao.py          gabarito da Keio, AUC com IC de 95% e precisão nos k primeiros
│       ├── modelos_classicos.py  geradores, descritivas e robustez (union-find) dos modelos clássicos
│       └── figuras.py            as figuras do artigo, com o mesmo estilo
├── dados/                        dados de entrada (abaixo)
└── resultados/
    ├── tabelas/                  CSVs gerados pelos experimentos
    └── figuras/                  PNGs das Figuras 2, 3 e 4
```

## Dados

| Onde | O que é |
|---|---|
| `dados/kgml/eco/` | cache da API do KEGG para a *E. coli* K-12 MG1655 (`eco`), baixado em 4/11/2026: o KGML de cada uma das 97 vias metabólicas, as listas de vias e de genes e `_metadados.json` (data e versão da base) |
| `dados/kgml/_compostos.tsv`, `_glicanos.tsv` | nomes de todos os compostos e glicanos do KEGG |
| `dados/eco/` | a rede montada pela coleta: `reacoes.csv` (1.244 reações, com substratos, produtos, sentido, genes e vias), `metabolitos.csv`, `genes.csv`, a camada bruta, sem fusão (`reacoes_por_via.csv`, `genes_por_via.csv`) e `relatorio_extracao.json` (contagens e conferências da extração) |
| `dados/keio/essenciais.csv` | os 300 genes essenciais da coleção Keio (Baba et al., 2006, Tabela Suplementar 6); a procedência está em `dados/keio/LEIAME.md` |

O cache fixa a versão do KEGG usada no artigo (a base muda continuamente), por isso os dados
ficam no repositório. Para refazer a lista da Keio a partir do arquivo original
(`msb4100050-s8.xls`): `pip install xlrd` e
`python -m coleta.converter_keio caminho/do/arquivo.xls --forcar` (a tabela tem 300 genes, e o texto
de Baba et al. fala em 303 candidatos; ver o `LEIAME.md`).

## Metodologia

**Extração.** Da API REST do KEGG, para a *E. coli* K-12 (código `eco`): as 97 vias metabólicas,
sem os mapas globais (011xx e 012xx), que repetem as demais vias, e sem a via de aminoacil-tRNA
(tradução). Cada via vem num arquivo KGML, lido por um parser próprio (`coleta/montar_tabelas.py`),
porque o Biopython perdia a ordem dos substratos. As vias são fundidas numa tabela com uma linha por
reação; quando duas vias discordam sobre a reversibilidade (77 reações), vale a reversível. Entram os
metabólitos, as reações, com seu sentido, e os 952 genes que as catalisam; ficam de fora fluxos,
compartimentos e regulação.

**Modelagem** (`experimentos/biblioteca/grafo.py`). Cada vértice é um metabólito e cada aresta liga
um substrato a um produto da mesma reação, nos dois sentidos se ela for reversível. Os 17
metabólitos-moeda de `config.METABOLITOS_MOEDA` (ATP, água, NAD+...) saem da rede: a rede final tem
1.111 metabólitos e 1.716 arestas (a versão com moeda só aparece na comparação da Tabela 1). Um gene
não é vértice: deletá-lo remove as reações que só ele catalisa (se há isoenzima, a reação continua).
Há também a versão não dirigida (1.277 arestas), para os modelos clássicos, e a bipartida
metabólito–reação, para a cascata de falhas.

**Análise descritiva** (`exp0_descritiva.py`). Densidade, graus, agrupamento, maior componente fraco,
gravata-borboleta (núcleo = maior componente forte, entrada e saída), caminho médio e diâmetro no
núcleo, e o teste de lei de potência contra log-normal da distribuição de graus (Clauset et al., 2009,
pacote `powerlaw`).

**P1, remoção de nós** (`exp1_robustez.py`). Os metabólitos são removidos um a um ao acaso (100
repetições) e por grau, intermediação e PageRank, cada centralidade calculada uma vez (estática) ou
recalculada durante as remoções (adaptativa; intermediação e PageRank a cada 10 remoções). A cada passo
mede-se o maior componente fraco, o núcleo e o alcance a partir da glicose e do sulfato (que vão para o
fim da ordem de remoção). Cada curva vira o índice R de Schneider et al. (2011), normalizado pelo maior
componente da rede intacta, e a fração removida até o maior componente cair à metade.

**P2, dano e essencialidade** (`exp2_biologia.py`). Para cada um dos 952 genes, simula-se a deleção e
mede-se o dano estrutural (metabólitos que saem do núcleo e do maior componente fraco), o funcional
(metabólitos que a rede intacta produz e deixam de ser produzidos, pela cascata de falhas de Smart et
al., 2008, partindo de tudo o que a rede não fabrica sozinha, o "meio rico") e a queda da eficiência
global; cada gene recebe também a centralidade do metabólito mais central das suas reações. Cada
medida ordena os genes, e a ordem é comparada com a Keio pela AUC (com IC de 95%) e pela precisão nos
10 primeiros, em todos os genes e só nos 574 que perdem ao menos uma reação.

**P3, modelos clássicos** (`exp4_comparacao.py`). Na versão não dirigida, 10 redes de cada modelo com o
mesmo número de vértices e, quando possível, de arestas: configuração (mesmos graus), Erdős–Rényi,
Barabási–Albert com 1 ou 2 arestas por vértice novo (misto) ou só 1 (árvore), Newman–Watts com a
densidade da *E. coli* e Watts–Strogatz com 4 vizinhos. Compara-se agrupamento, caminho médio, grau
máximo e o R normalizado (remoção aleatória e ataque por grau, por percolação reversa), com o escore z
da *E. coli* contra cada modelo.

**Análises complementares** (`checar_inferencias.py`). Onde ficam, na gravata-borboleta, os primeiros
metabólitos removidos por cada centralidade; que lado (entrada ou saída do núcleo) cada nocaute
desconecta; o nocaute de *aroA* a *aroD* no meio mínimo (só glicose e sulfato); e as pontes da rede não
dirigida contra o modelo de configuração.

## Onde está cada resultado do artigo

Caminhos relativos a `resultados/`.

| No artigo | Arquivo |
|---|---|
| Tabela 1 | `tabelas/exp0_tabela1_eco.csv`, coluna `sem_moeda` (a `com_moeda` é a comparação do texto: núcleo de 497) |
| Figura 2 | `figuras/exp0_graus_eco_sem_moeda.png`, painel (b) |
| Hubs (piruvato, acetil-CoA, frutose-6-fosfato) | `tabelas/exp0_top10_eco_sem_moeda.csv` |
| Núcleo, entrada e saída de cada metabólito | `tabelas/exp0_bowtie_eco_sem_moeda.csv` |
| Distribuição de graus e ajustes | `tabelas/exp0_graus_eco_sem_moeda.csv`, `exp0_lei_potencia_eco_sem_moeda.csv`, `exp0_ccdf_eco_sem_moeda.csv` |
| Tabela 2 | `tabelas/exp1_indices_eco_sem_moeda.csv` (`R_fraco_norm` e `metade_fraco`; o ± da aleatória é `R_fraco_dp / S0_fraco` e `metade_fraco_dp`; `metade_forte` é a do núcleo) |
| Figura 3 | `figuras/exp1_maior_fraco_inteira_eco_sem_moeda.png` |
| Alcance a partir da glicose ao longo das remoções | `tabelas/exp1_curvas_eco_sem_moeda.csv`, coluna `alcance` |
| Tabela 3 | `tabelas/exp2_rankings_eco.csv` (`auc`, `auc_ic_inf`, `auc_ic_sup`, `precisao_10`; subconjunto `todos` = 952 genes, `perde_reacao` = 574) |
| Figura 4 | `figuras/exp2_essenciais_por_faixa_eco.png` (dados em `tabelas/exp2_essenciais_por_faixa_eco.csv`) |
| Correlação entre dano estrutural e funcional | `tabelas/exp2_correlacoes_eco.csv` |
| Genes citados (*aroA*–*aroD*, *purH*, *fabD*, *fabG*, *fabI*, *lpxC*) | `tabelas/exp2_exemplos_dano_eco.csv`; todos os genes em `tabelas/exp2_genes_eco.csv` |
| Contagens da P2 (952 genes, 103 essenciais, 574 perdem reação) | `tabelas/exp2_resumo_eco.csv` |
| Tabela 4 | média por `modelo` de `tabelas/exp4_classicos_descritiva_redes_sem_moeda.csv` e `exp4_classicos_indices_redes_sem_moeda.csv` (o Exp 4 imprime essa tabela no fim) |
| Escores z (ex.: z = −3,5 contra a configuração) | `tabelas/exp4_classicos_z_redes_sem_moeda.csv` |
| Os 39 primeiros removidos por centralidade | `tabelas/inferencias_rankings_eco.csv`, `inferencias_primeiros_eco.csv` |
| Ramos de saída × de entrada (*hemA*–*hemD*, *melA*) | `tabelas/inferencias_lados_eco.csv`, `inferencias_genes_eco.csv` |
| Via do chiquimato no meio mínimo | `tabelas/inferencias_chiquimato_eco.csv` |
| Pontes: *E. coli* × modelo de configuração | `tabelas/inferencias_pontes_eco.csv` |
| Contagens da extração (vias, reações, genes, conflitos de reversibilidade) | `../dados/eco/relatorio_extracao.json` |

## Regra de reversibilidade

A regra da coleta ("vale a reversível") pode ser comparada com a contrária sem refazer a coleta.
Com a contrária, succinato e succinil-CoA saem do núcleo, e o ciclo de Krebs deixa de fechar
(na pasta `rede-metabolica/`, com o `.venv` ativo):

```bash
python -c "
from experimentos.biblioteca import grafo, metricas
reacoes = grafo.carregar_tabelas('eco')['reacoes']
for regra in (False, True):
    G = grafo.grafo_simples(grafo.grafo_metabolitos(reacoes, conflito_como_irreversivel=regra))
    nucleo = metricas.maior_componente_forte(G)
    print('vale o irreversível:' if regra else 'vale o reversível: ', 'núcleo', len(nucleo),
          '| succinato e succinil-CoA no núcleo:', {'C00042', 'C00091'} <= nucleo)
"
```

```
vale o reversível:  núcleo 415 | succinato e succinil-CoA no núcleo: True
vale o irreversível: núcleo 368 | succinato e succinil-CoA no núcleo: False
```

## Limitações

A rede é estática: não tem fluxos, regulação nem compartimentos. O KGML não diz se os genes de uma
reação são isoenzimas ou partes de um complexo, e todos são tratados como isoenzimas (regra OU), o
que subestima o dano de genes de complexos como *accA*–*accD*. O "meio rico" da cascata não é o LB da
Keio. Os modelos clássicos têm parâmetros inteiros, então o Barabási–Albert puro e o Watts–Strogatz
não casam ao mesmo tempo o número de vértices e o de arestas da rede (daí o BA misto e as duas versões
de Watts–Strogatz).
