# Rede metabólica da *E. coli*: robustez, dano × essencialidade e modelos clássicos

Rede metabólito-cêntrica da *E. coli* K-12 MG1655 montada a partir do KEGG
(organismo `eco`, 97 vias metabólicas fundidas). A versão principal é sem
metabólitos-moeda (`sem_moeda`): um grafo dirigido simples com 1.111
metabólitos e 1.716 arestas. A cascata de falhas usa a visão bipartida
metabólito–reação. Os experimentos respondem a três perguntas sobre robustez:

| Pergunta | Experimento |
|---|---|
| **P1.** Removendo nós por grau, betweenness e PageRank, como a rede se desfaz? | Exp 1 |
| **P2.** O nó (gene) que causa mais dano na rede é essencial? Depende de como o dano é medido. | Exp 2 (genes × Keio) e Exp 3 (metabólitos, sem gabarito) |
| **P3.** A rede se parece com Erdős–Rényi, Watts–Strogatz ou Barabási–Albert? | Exp 4 |

O Exp 0 é a análise descritiva (Tabela 1 e distribuição de graus). A checagem das
inferências do artigo (`experimentos/checar_inferencias.py`) confere nos dados
afirmações da discussão, com os grafos e as funções dos Exps 1, 2 e 4.

## Como rodar

Na raiz do projeto (`rede-metabolica/`):

```bash
bash setup.sh                       # cria o .venv e instala as versões de requirements-lock.txt
bash executar_tudo.sh               # testes, coleta (do cache em dados/kgml), Exps 0 a 4 e a checagem (~30 min)
bash executar_tudo.sh --sem-testes
bash executar_tudo.sh --so-figuras  # refaz só as figuras, a partir de resultados/tabelas/
```

Cada experimento roda sozinho como módulo, com `rodar()`, `plotar()` e a opção `--so-figuras`:

```bash
python -m experimentos.exp1_robustez --org eco
python -m experimentos.exp2_biologia
python -m experimentos.exp3_vies --org eco
python -m experimentos.exp4_comparacao --sem-nulos --sem-classicos   # teste rápido
python -m experimentos.checar_inferencias                           # depois do Exp 2 (~10 s)
python -m pytest testes/                                             # ~90 s
python -m testes.verificar_cascata_robustez --org eco                # conferência na rede real
```

Tudo é determinístico pela semente (`config.SEMENTE = 42`); constantes ficam em
`config.py` e o formato das tabelas em `contrato.py`. Os nomes das saídas vêm de
`config.arquivo_tabela(exp, conteudo, org, versao)` e `config.arquivo_figura(...)`.

## Estrutura

```
config.py, contrato.py        parâmetros e formato das tabelas
coleta/                       download do KEGG (com cache) e montagem das tabelas
experimentos/biblioteca/
    grafo.py                  visões da rede (metabólito-cêntrica, simples, não dirigida, bipartida)
    metricas.py               descritivas, dano (componentes, índice R, eficiência, alcance),
                              harmônica (Exps 2 e 3), eficiência rápida (scipy) e dano estrutural de uma remoção
    robustez.py               estratégias de remoção (estáticas e adaptativas) e curvas de robustez
    cascata.py                cascata de falhas na rede bipartida (avalanche)
    nocaute.py                nocaute de genes: dano funcional (cascata) e estrutural
    validacao.py              gabarito da Keio, AUC com IC, precisão@k, ROC
    modelos_nulos.py          nulos dirigidos (rewiring, rewiring recíproco, Erdős–Rényi)
    modelos_classicos.py      modelos clássicos não dirigidos e percolação reversa (union-find)
    figuras.py                todas as figuras, com o mesmo estilo
experimentos/exp0_descritiva.py ... exp4_comparacao.py
experimentos/checar_inferencias.py   checagem das inferências do artigo (só tabelas)
testes/                       pytest (302 testes) e os scripts verificar_*.py
```

## Métodos em uma página

- **Robustez (Exp 1 e 4).** Cada curva de remoção vira três números: o índice R
  de Schneider et al. (2011), a área sob a curva do maior componente; o **R
  normalizado**, R dividido pelo maior componente na rede intacta (S0); e a
  fração removida até o maior componente cair à metade.
- **Estratégias (Exp 1).** Aleatória (100 repetições) e três centralidades,
  cada uma estática e adaptativa: grau total (entrada + saída, no grafo simples:
  um vizinho ligado por reação reversível conta duas vezes; a adaptativa recalcula
  a cada remoção), betweenness e PageRank (as adaptativas recalculam a
  cada `config.PASSO_BETWEENNESS = 10` remoções). Empates pelo código do nó. A
  centralidade de autovetor fica de fora (não é bem definida numa rede dirigida
  que não é fortemente conexa). A harmônica não entra na remoção de nós: continua
  só como centralidade de referência nos rankings do Exp 2 e nas correlações do Exp 3.
- **Fontes protegidas.** Os nutrientes de `config.NUTRIENTES` vão para o fim da
  ordem em todas as estratégias (`robustez.simular_remocao(..., proteger_fontes=True)`,
  o padrão): sem isso o ataque por grau removia a glicose e o alcance caía a zero
  por isso, e não pelo dano à rede. São 2 dos 1.111 nós (glicose e sulfato).
- **Dano de um nocaute (Exp 2) e de um metabólito (Exp 3).** Duas medidas principais:
  *estrutural* — quantos metabólitos deixam o maior componente forte (o núcleo) e o
  maior componente fraco (tamanho antes − depois; também como fração de N); e
  *funcional* — quantos metabólitos que a rede intacta produz deixam de ser
  produzidos (a avalanche da cascata, meio rico). Complementar: a queda relativa da
  eficiência global dirigida, (E0 − E)/E0. O nocaute remove as reações que só aquele
  gene catalisa (regra OU): remove arestas, nunca metabólitos; os 1.111 nós continuam em N.
- **Checagem das isoenzimas (Exp 2).** A avaliação contra a Keio se repete só com os
  574 genes que perdem pelo menos uma reação (`subconjunto = todos | perde_reacao`).
- **Modelos clássicos (Exp 4).** Na versão não dirigida (N = 1.111, M = 1.277):
  configuração (mesmos graus), Erdős–Rényi G(N, M), Barabási–Albert com m misto
  (`dual_barabasi_albert_graph(N, 1, 2, p)`, p = 2 − M/N ≈ 0,85), BA com m = 1 (árvore,
  referência), Newman–Watts (k = 2, p = M/N − 1 ≈ 0,15: mesma densidade, clustering ≈ 0)
  e Watts–Strogatz (k = 4, p = 0,1: clustering alto, mas o dobro de arestas), 10 redes
  de cada (semente `SEMENTE*1000 + i`). Robustez pelo maior componente conexo, por
  percolação reversa com union-find; escores z da *E. coli* contra cada modelo e o
  sigma de mundo pequeno (C/C_ER)/(L/L_ER).

## Saídas

Tabelas em `resultados/tabelas/`, figuras em `resultados/figuras/`. As marcadas com
**(novo)** foram acrescentadas nesta versão; as outras continuam com o mesmo nome.

### Exp 0 — descritiva (`exp0_descritiva.py`)
`exp0_tabela1_eco[_<versao>].csv`, `exp0_top10_*`, `exp0_graus_*`, `exp0_lei_potencia_*`,
`exp0_ccdf_*`, `exp0_bowtie_*`; figura `exp0_graus_eco_<versao>.png`.

### Exp 1 — robustez: remoção por centralidade (`exp1_robustez.py`)
| Saída | Conteúdo |
|---|---|
| `exp1_curvas_eco_sem_moeda.csv` | curvas das 7 estratégias (maior componente fraco e forte, alcance, eficiência) |
| `exp1_indices_eco_sem_moeda.csv` | por estratégia: R, **R normalizado e S0 (novo)**, metade; componentes fraco e forte |
| `exp1_<metrica>_eco_sem_moeda.png` | por métrica: aleatória + a versão adaptativa de cada centralidade, com detalhe dos 10% iniciais |
| `exp1_painel_eco_sem_moeda.png` | as quatro métricas num painel 2×2, com as mesmas curvas |
| `exp1_maior_fraco_inteira_eco_sem_moeda.png` **(novo)** | a figura do artigo: maior componente fraco, as mesmas curvas, a curva inteira sem o detalhe |
| `exp1_estatica_adaptativa_eco_sem_moeda.png` **(novo)** | para cada centralidade, estática × adaptativa (maior componente fraco) |

### Exp 2 — dano × essencialidade dos genes (`exp2_biologia.py`)
| Saída | Conteúdo |
|---|---|
| `exp2_genes_eco.csv` | por gene: essencial, dano na cascata, **dano estrutural (núcleo, fraco, eficiência; novo)**, grau, betweenness, **harmônica e PageRank (novo)**, choke |
| `exp2_rankings_eco.csv` | AUC com IC 95%, p e precisão@k de cada ranking, **por subconjunto (todos / perde_reacao; novo)** |
| `exp2_roc_eco.csv` | pontos das curvas ROC (por subconjunto) |
| `exp2_correlacoes_eco.csv` **(novo)** | Spearman entre o dano estrutural e o funcional |
| `exp2_exemplos_dano_eco.csv` **(novo)** | os três grupos de casos de `config.EXEMPLOS_DANO_EXP2` |
| `exp2_essenciais_por_faixa_eco.csv` **(novo)** | genes e essenciais por faixa de dano (0, 1–2, 3–9, 10 ou mais), estrutural (núcleo) e funcional (cascata), nos 574 que perdem reação |
| `exp2_exemplos_eco.csv`, `exp2_choke_eco.csv`, `exp2_resumo_eco.csv` | essenciais de baixa centralidade, choke points, cobertura e rede intacta |
| `exp2_roc_eco.png` | (a) ROC da cascata, da queda do núcleo, da eficiência e do grau |
| `exp2_dispersao_eco.png` **(novo)** | (b) dano estrutural × funcional por gene, essenciais destacados |
| `exp2_auc_eco.png` **(novo)** | (c) AUC de cada ranking, todos os genes × os 574 que perdem reação |
| `exp2_essenciais_por_faixa_eco.png` **(novo)** | (d) a Figura 4 do artigo: % de essenciais por faixa de dano, quebrar a rede × deixar de produzir |

### Exp 3 — dano individual dos metabólitos (`exp3_vies.py`)
| Saída | Conteúdo |
|---|---|
| `exp3_dano_eco.csv` **(novo)** | por metabólito do grafo: centralidades, dano estrutural e avalanche |
| `exp3_correlacoes_eco.csv` **(novo)** | Spearman de cada medida de dano com as centralidades e entre si |
| `exp3_exemplos_eco.csv` **(novo)** | casos da discussão e os mais discordantes entre as duas medidas |
| `exp3_substancias_eco.csv` | as 6 substâncias e os 12 controles, agora com o dano estrutural |
| `exp3_avalanches_eco.csv`, `exp3_resumo_eco.csv`, `exp3_ccdf_eco.csv` | avalanches, resumo e CCDF |
| `exp3_dano_vs_avalanche_eco.png` **(novo)** | queda do núcleo × avalanche, cor pelo grau, nomes onde as medidas discordam |
| `exp3_avalanche_vs_grau_eco.png`, `exp3_distribuicao_eco.png` | como antes |

### Exp 4 — comparação com outras redes (`exp4_comparacao.py`)
Organismos e nulos dirigidos, como antes: `exp4_{descritiva,indices,curvas,nulos,ccdf}_redes_sem_moeda.csv`
e as figuras `exp4_indices_redes`, `exp4_curvas_{organismos,nulos}`, `exp4_graus_redes`. Modelos clássicos **(novo)**:

| Saída | Conteúdo |
|---|---|
| `exp4_classicos_descritiva_redes_sem_moeda.csv` | por rede: M, grau médio e máximo, clustering, maior componente, caminho médio e diâmetro no maior componente, assortatividade, sigma |
| `exp4_classicos_indices_redes_sem_moeda.csv` | por rede e estratégia: S0, R, R normalizado, metade |
| `exp4_classicos_curvas_redes_sem_moeda.csv` | curvas do maior componente (absoluto e relativo), média das redes de cada modelo |
| `exp4_classicos_z_redes_sem_moeda.csv` | escore z da *E. coli* contra cada modelo |
| `exp4_classicos_ccdf_redes_sem_moeda.csv` | CCDF dos graus (não dirigidos) |
| `exp4_classicos_indices_redes_sem_moeda.png` | R normalizado e metade, por modelo e estratégia |
| `exp4_classicos_curvas_redes_sem_moeda.png` | remoção aleatória × ataque por grau |
| `exp4_classicos_graus_redes_sem_moeda.png` | CCDF dos graus, *E. coli* e modelos sobrepostos |

### Checagem das inferências do artigo (`checar_inferencias.py`) **(novo)**
Só tabelas, só na *E. coli*; lê a `exp2_genes_eco.csv` (rode antes o Exp 2).

| Saída | Conteúdo |
|---|---|
| `inferencias_rankings_eco.csv` | (1) os primeiros 39 e 111 (3,5% e 10%) que o Exp 1 remove por grau, betweenness e PageRank: frações no núcleo, entrada, saída e resto da gravata-borboleta, graus médios de entrada e de saída, sumidouros |
| `inferencias_primeiros_eco.csv` | (1) os 10 primeiros de cada ranking, com a parte da gravata e os graus |
| `inferencias_genes_eco.csv` | (2) genes cujo nocaute tira metabólitos do maior componente fraco, pelo lado da gravata do que sai (entrada × saída), com a essencialidade da Keio |
| `inferencias_lados_eco.csv` | (2) genes e taxa de essenciais por lado (todos e só os que tiram ≥ 5) |
| `inferencias_chiquimato_eco.csv` | (3) aroA–aroD: o que sai do núcleo e o que deixa de ser produzido no meio rico e no mínimo (Phe, Tyr, Trp) |
| `inferencias_pontes_eco.csv` | (4) pontes, pontes internas e pontos de articulação na versão não dirigida: *E. coli* e as 10 redes de configuração do Exp 4 |

## Limitações (vão para o texto, não para o código)

- A essencialidade só existe para genes, e só 103 dos 300 genes essenciais da Keio estão na rede.
- O "meio rico" do modelo não é o LB; a rede não tem fluxos (sem FBA).
- Complexos enzimáticos são tratados como isoenzimas (regra OU): é o caso de accA–accD.
- Os modelos clássicos são não dirigidos e têm parâmetros inteiros (o BA puro e o WS não
  casam N e M ao mesmo tempo com a rede; daí o BA misto e as duas versões do WS).
