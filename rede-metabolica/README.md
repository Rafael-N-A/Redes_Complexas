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
