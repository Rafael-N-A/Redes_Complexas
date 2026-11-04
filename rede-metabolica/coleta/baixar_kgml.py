"""Coleta, passo 1: baixa da API REST do KEGG tudo que a rede precisa.

Tudo vai para o cache em dados/kgml/. Um arquivo que já está no cache
não é baixado de novo (use --forcar para baixar tudo outra vez).

Saídas:
    dados/kgml/<org>/<via>.xml       KGML de cada via metabólica regular
    dados/kgml/<org>/_vias.tsv       resposta de list/pathway/<org>
    dados/kgml/<org>/_genes.tsv      resposta de list/<org> (gene -> símbolo e descrição)
    dados/kgml/<org>/_metadados.json data do download, vias baixadas, falhas, versão da base
    dados/kgml/_compostos.tsv        resposta de list/compound (todos os organismos)
    dados/kgml/_glicanos.tsv         resposta de list/glycan

Uso, na raiz do projeto:
    python -m coleta.baixar_kgml --org eco
    python -m coleta.baixar_kgml --org eco --forcar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

import config

_sessao: requests.Session | None = None


class ErroKEGG(RuntimeError):
    """Resposta da API que não deu para usar. status = código HTTP, ou None se nem houve resposta."""

    def __init__(self, mensagem: str, status: int | None = None):
        super().__init__(mensagem)
        self.status = status


def _obter_sessao() -> requests.Session:
    global _sessao
    if _sessao is None:
        _sessao = requests.Session()
        _sessao.headers["User-Agent"] = "rede-metabolica/1.0 (trabalho academico; python-requests)"
    return _sessao


def requisitar(caminho_api: str) -> str:
    """Faz GET em {config.KEGG_URL}/{caminho_api} e devolve o texto.

    Espera config.PAUSA_ENTRE_REQUISICOES depois de cada chamada, para não
    sobrecarregar a API. Erros temporários (403, 429, 5xx, falha de conexão)
    são tentados de novo até config.TENTATIVAS_REQUISICAO vezes, com espera
    crescente. Outras respostas diferentes de 200 (como 404) levantam ErroKEGG
    na hora, para a função chamadora decidir se registra a falha e segue.
    """
    url = f"{config.KEGG_URL}/{caminho_api}"
    ultimo_erro: ErroKEGG | None = None

    for tentativa in range(1, config.TENTATIVAS_REQUISICAO + 1):
        try:
            resposta = _obter_sessao().get(url, timeout=config.TIMEOUT_REQUISICAO)
        except requests.RequestException as erro:
            ultimo_erro = ErroKEGG(f"{url}: falha de conexão ({erro.__class__.__name__}: {erro})")
        else:
            time.sleep(config.PAUSA_ENTRE_REQUISICOES)
            if resposta.status_code == 200:
                return resposta.text
            erro = ErroKEGG(f"{url}: HTTP {resposta.status_code}", status=resposta.status_code)
            if resposta.status_code not in (403, 429, 500, 502, 503, 504):
                raise erro
            ultimo_erro = erro

        if tentativa < config.TENTATIVAS_REQUISICAO:
            time.sleep(config.PAUSA_ENTRE_REQUISICOES * 4 ** tentativa)

    assert ultimo_erro is not None
    raise ultimo_erro


def baixar_com_cache(caminho_api: str, destino: Path, forcar: bool = False,
                     validar: Callable[[str], bool] | None = None) -> Path:
    """Baixa caminho_api para destino, a menos que destino já exista.

    É o único lugar que escreve no cache: todas as outras funções passam
    por aqui. Cria as pastas que faltarem. Grava primeiro num arquivo
    temporário e só depois renomeia, para um download interrompido não
    deixar um arquivo pela metade no cache. Se validar recusar o texto,
    nada é gravado e levanta ErroKEGG.
    """
    if destino.exists() and destino.stat().st_size > 0 and not forcar:
        return destino

    texto = requisitar(caminho_api)
    if validar is not None and not validar(texto):
        raise ErroKEGG(f"{config.KEGG_URL}/{caminho_api}: resposta em formato inesperado "
                       f"({len(texto)} caracteres)")

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".parcial")
    temporario.write_text(texto, encoding="utf-8")
    temporario.replace(destino)
    return destino


def _parece_kgml(texto: str) -> bool:
    return "<pathway" in texto[:5000]


def _parece_lista(texto: str) -> bool:
    return "\t" in texto[:5000]


def caminho_kgml(org: str, via: str) -> Path:
    return config.pasta_brutos(org) / f"{via}.xml"


def listar_vias(org: str, forcar: bool = False) -> list[tuple[str, str]]:
    """Lista as vias do organismo: [(codigo, nome), ...], ex. ("eco00020", "Citrate cycle").

    Fonte: list/pathway/<org>, guardada em _vias.tsv. Cada linha tem o
    código e o nome separados por tab; o prefixo "path:" é removido se aparecer.
    """
    arquivo = baixar_com_cache(f"list/pathway/{org}", config.pasta_brutos(org) / "_vias.tsv",
                               forcar, validar=_parece_lista)
    vias = []
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        codigo, _, nome = linha.partition("\t")
        vias.append((codigo.strip().split(":")[-1], nome.strip()))
    return vias


def numero_da_via(codigo: str) -> str:
    """Os 5 últimos dígitos do código: "eco00970" -> "00970"."""
    return codigo[-5:]


def filtrar_vias_regulares(vias: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Mantém só as vias metabólicas regulares (número < config.LIMITE_VIAS_REGULARES).

    Descarta os mapas globais e de visão geral (011xx, 012xx), como o eco01100,
    e as vias de config.VIAS_EXCLUIDAS (abaixo de 01100, mas fora de
    Metabolismo, como a 00970). O número é formado pelos 5 últimos dígitos
    do código.
    """
    regulares = []
    for codigo, nome in vias:
        numero = numero_da_via(codigo)
        if not numero.isdigit():
            continue
        if int(numero) < config.LIMITE_VIAS_REGULARES and numero not in config.VIAS_EXCLUIDAS:
            regulares.append((codigo, nome))
    return regulares


def baixar_kgml(org: str, via: str, forcar: bool = False) -> Path | None:
    """Baixa get/<via>/kgml para dados/kgml/<org>/<via>.xml.

    Devolve None se a via não tiver KGML (a API responde 400 ou 404); quem
    chama registra a falha nos metadados e segue para a próxima via. Outros
    erros (rede, formato inesperado) levantam ErroKEGG.
    """
    try:
        return baixar_com_cache(f"get/{via}/kgml", caminho_kgml(org, via), forcar,
                                validar=_parece_kgml)
    except ErroKEGG as erro:
        if erro.status in (400, 404):
            return None
        raise


def baixar_listas_globais(forcar: bool = False) -> None:
    """Baixa os nomes de todos os compostos e glicanos numa chamada cada.

    list/compound -> config.ARQ_COMPOSTOS, list/glycan -> config.ARQ_GLICANOS.
    Servem a todos os organismos, então ficam fora da pasta do organismo.
    """
    baixar_com_cache("list/compound", config.ARQ_COMPOSTOS, forcar, validar=_parece_lista)
    baixar_com_cache("list/glycan", config.ARQ_GLICANOS, forcar, validar=_parece_lista)


def baixar_genes(org: str, forcar: bool = False) -> Path:
    """Baixa list/<org> (genes do organismo) para dados/kgml/<org>/_genes.tsv."""
    return baixar_com_cache(f"list/{org}", config.pasta_brutos(org) / "_genes.tsv", forcar,
                            validar=_parece_lista)


def registrar_metadados(org: str, vias: list[tuple[str, str]], falhas: list[dict],
                        baixadas: int, do_cache: int) -> Path:
    """Grava dados/kgml/<org>/_metadados.json.

    Conteúdo: data e hora do download, vias baixadas, vias que falharam e a
    resposta de info/<org> (versão da base). A data fica registrada porque
    o KEGG é atualizado continuamente.

    data_download só muda quando esta execução baixou algum KGML; se tudo veio
    do cache, fica a data do download anterior. ultima_execucao muda sempre.
    "vias" é a lista que o montar_tabelas lê: só as vias com KGML no cache.
    """
    destino = config.pasta_brutos(org) / "_metadados.json"
    anterior = {}
    if destino.exists():
        try:
            anterior = json.loads(destino.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            anterior = {}

    agora = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        info = requisitar(f"info/{org}").strip()
    except ErroKEGG as erro:
        info = f"não foi possível consultar info/{org}: {erro}"

    falhou = {f["via"] for f in falhas}
    metadados = {
        "org": org,
        "fonte": config.KEGG_URL,
        "data_download": agora if baixadas or not anterior.get("data_download") else anterior["data_download"],
        "ultima_execucao": agora,
        "info_kegg": info,
        "criterio": {
            "limite_vias_regulares": f"{config.LIMITE_VIAS_REGULARES:05d}",
            "vias_excluidas": config.VIAS_EXCLUIDAS,
        },
        "n_vias_regulares": len(vias),
        "n_baixadas_nesta_execucao": baixadas,
        "n_reaproveitadas_do_cache": do_cache,
        "vias": [{"codigo": c, "nome": n} for c, n in vias if c not in falhou],
        "falhas": falhas,
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os dados do KEGG para um organismo.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--forcar", action="store_true", help="baixa de novo mesmo o que está no cache")
    args = parser.parse_args()
    org = args.org

    try:
        todas = listar_vias(org, args.forcar)
    except ErroKEGG as erro:
        sys.exit(f"Não foi possível listar as vias de {org}: {erro}")
    if not todas:
        sys.exit(f"A API não devolveu nenhuma via para '{org}'. O código do organismo está certo?")

    vias = filtrar_vias_regulares(todas)
    excluidas = [c for c, _ in todas if numero_da_via(c) in config.VIAS_EXCLUIDAS]
    print(f"{org}: {len(todas)} vias no KEGG, {len(vias)} metabólicas regulares "
          f"(excluídas pela lista: {', '.join(excluidas) or 'nenhuma'})")

    falhas: list[dict] = []
    baixadas = do_cache = 0
    for codigo, _nome in tqdm(vias, desc=f"KGML {org}", unit="via"):
        arquivo = caminho_kgml(org, codigo)
        ja_tinha = arquivo.exists() and arquivo.stat().st_size > 0 and not args.forcar
        try:
            caminho = baixar_kgml(org, codigo, args.forcar)
        except ErroKEGG as erro:
            falhas.append({"via": codigo, "motivo": str(erro)})
            continue
        if caminho is None:
            falhas.append({"via": codigo, "motivo": "a API não tem KGML para esta via"})
        elif ja_tinha:
            do_cache += 1
        else:
            baixadas += 1

    try:
        baixar_listas_globais(args.forcar)
        baixar_genes(org, args.forcar)
    except ErroKEGG as erro:
        sys.exit(f"Não foi possível baixar as listas de nomes: {erro}")

    destino = registrar_metadados(org, vias, falhas, baixadas, do_cache)
    print(f"Concluído: {baixadas} baixadas agora, {do_cache} do cache, {len(falhas)} falhas.")
    for falha in falhas:
        print(f"  - {falha['via']}: {falha['motivo']}")
    print(f"Metadados em {config.caminho_legivel(destino)}")


if __name__ == "__main__":
    main()
