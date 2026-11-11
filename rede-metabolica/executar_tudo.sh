#!/usr/bin/env bash
# Roda o pipeline completo: testes, coleta e experimentos.
# Uso, na raiz do projeto:
#   bash executar_tudo.sh                # testes, coleta, exp0 a exp4 e a checagem das inferências
#   bash executar_tudo.sh --sem-testes   # pula o pytest do início
#   bash executar_tudo.sh --so-figuras   # refaz só as figuras, a partir de resultados/tabelas/
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Ambiente virtual não encontrado. Rode antes: bash setup.sh"
  exit 1
fi

TESTES=1
FIGURAS=()
for arg in "$@"; do
  case "$arg" in
    --sem-testes) TESTES=0 ;;
    --so-figuras) FIGURAS=(--so-figuras); TESTES=0 ;;
    *) echo "Opção desconhecida: $arg"; exit 1 ;;
  esac
done

etapa() { echo; echo "=== $1"; }

# 1. Testes das regras: se falharem, não vale a pena rodar o resto
if [ "$TESTES" = 1 ]; then
  etapa "Testes das regras"
  "$PY" -m pytest -q testes/
fi

# 2. Coleta de todos os organismos (os códigos vêm do config.py)
if [ ${#FIGURAS[@]} -eq 0 ]; then
  ORGS=$("$PY" -c "import config; print(' '.join([config.ORGANISMO_PRINCIPAL, *config.ORGANISMOS_COMPARACAO]))")
  for org in $ORGS; do
    etapa "Coleta: $org"
    "$PY" -m coleta.baixar_kgml --org "$org"
    "$PY" -m coleta.montar_tabelas --org "$org"
  done
fi

# 3. Experimentos
etapa "Exp 0 — análise descritiva"
"$PY" -m experimentos.exp0_descritiva --versao ambas "${FIGURAS[@]}"

etapa "Exp 1 — robustez: remoção por centralidade (só sem moeda; --versao ambas para as duas)"
"$PY" -m experimentos.exp1_robustez "${FIGURAS[@]}"

etapa "Exp 2 — dano x essencialidade dos genes (Keio)"
"$PY" -m experimentos.exp2_biologia "${FIGURAS[@]}"

etapa "Exp 3 — dano individual dos metabólitos"
"$PY" -m experimentos.exp3_vies "${FIGURAS[@]}"

etapa "Exp 4 — comparação com outras redes, modelos nulos e modelos clássicos"
"$PY" -m experimentos.exp4_comparacao "${FIGURAS[@]}"

# 4. Checagem das inferências do artigo (só tabelas; precisa da tabela do Exp 2)
if [ ${#FIGURAS[@]} -eq 0 ]; then
  etapa "Checagem das inferências do artigo"
  "$PY" -m experimentos.checar_inferencias
fi

echo
echo "Pronto. Resultados em resultados/tabelas/ e resultados/figuras/."
