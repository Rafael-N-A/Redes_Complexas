#!/usr/bin/env bash
# Cria o ambiente virtual (.venv) e instala as dependências.
# Uso, na raiz do projeto:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. Ambiente virtual (no Ubuntu, o pip fora de um venv é bloqueado)
if [ ! -x .venv/bin/python ]; then
  echo "Criando o ambiente virtual em .venv ..."
  if ! python3 -m venv .venv; then
    rm -rf .venv
    echo
    echo "Não foi possível criar o ambiente virtual."
    echo "No Ubuntu, instale o pacote do venv e rode de novo:"
    echo "  sudo apt install python3-venv"
    echo "(se a mensagem acima sugerir outro nome, como python3.12-venv, use esse)"
    exit 1
  fi
fi

PY=.venv/bin/python
"$PY" -m pip install --upgrade pip --quiet

# 2. Dependências: versões travadas se o lock existir; senão, instala e cria o lock
if [ -f requirements-lock.txt ]; then
  echo "Instalando as versões travadas de requirements-lock.txt ..."
  "$PY" -m pip install -r requirements-lock.txt
else
  echo "Instalando requirements.txt ..."
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip freeze > requirements-lock.txt
  echo "Versões gravadas em requirements-lock.txt (versione esse arquivo)."
fi

# 3. Pastas de dados e resultados
mkdir -p dados/kgml dados/keio resultados/tabelas resultados/figuras

# 4. Conferência rápida
"$PY" -c "import networkx, pandas, requests, sklearn, matplotlib; print('Dependências OK')"

echo
echo "Pronto. Para usar:"
echo "  source .venv/bin/activate"
echo "  python -m pytest testes/                 # confere as regras"
echo "  python -m coleta.baixar_kgml --org eco   # um passo por vez"
echo "  bash executar_tudo.sh                    # ou o pipeline completo"
