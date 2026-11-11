#!/usr/bin/env bash
# Roda o pipeline completo: coleta e experimentos.
# Uso, na raiz do projeto:
#   bash executar_tudo.sh                # coleta (do cache), Exps 0 a 4 e as análises complementares
#   bash executar_tudo.sh --so-figuras   # refaz só as figuras, a partir de resultados/tabelas/
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Ambiente virtual não encontrado. Rode antes: bash setup.sh"
  exit 1
fi

FIGURAS=()
for arg in "$@"; do
  case "$arg" in
    --so-figuras) FIGURAS=(--so-figuras) ;;
    *) echo "Opção desconhecida: $arg"; exit 1 ;;
  esac
done

etapa() { echo; echo "=== $1"; }

# 1. Coleta (o código do organismo vem do config.py; o KGML vem do cache em dados/kgml/)
if [ ${#FIGURAS[@]} -eq 0 ]; then
  ORG=$("$PY" -c "import config; print(config.ORGANISMO_PRINCIPAL)")
  etapa "Coleta: $ORG"
  "$PY" -m coleta.baixar_kgml --org "$ORG"
  "$PY" -m coleta.montar_tabelas --org "$ORG"
fi

# 2. Experimentos
etapa "Exp 0 — análise descritiva (Tabela 1, Figura 2)"
"$PY" -m experimentos.exp0_descritiva "${FIGURAS[@]}"

etapa "Exp 1 — P1: remoção de nós por centralidade (Tabela 2, Figura 3)"
"$PY" -m experimentos.exp1_robustez "${FIGURAS[@]}"

etapa "Exp 2 — P2: dano x essencialidade dos genes (Tabela 3, Figura 4)"
"$PY" -m experimentos.exp2_biologia "${FIGURAS[@]}"

# 3. Só tabelas: o Exp 4 e as análises complementares (que precisam da tabela do Exp 2)
if [ ${#FIGURAS[@]} -eq 0 ]; then
  etapa "Exp 4 — P3: comparação com os modelos clássicos (Tabela 4)"
  "$PY" -m experimentos.exp4_comparacao

  etapa "Análises complementares da discussão"
  "$PY" -m experimentos.checar_inferencias
fi

echo
echo "Pronto. Resultados em resultados/tabelas/ e resultados/figuras/."
