"""Coleta, passo 1: baixa da API REST do KEGG tudo que a rede precisa.

Tudo vai para o cache em dados/kgml/. Um arquivo que já está no cache
não é baixado de novo (use --forcar para baixar tudo outra vez).

Saídas:
    dados/kgml/<org>/<via>.xml      KGML de cada via metabólica regular
    dados/kgml/<org>/_vias.tsv      resposta de list/pathway/<org>
    dados/kgml/<org>/_genes.tsv     resposta de list/<org> (b-number -> símbolo)
    dados/kgml/<org>/_metadados.json data do download, vias baixadas, falhas
    dados/kgml/_compostos.tsv       resposta de list/compound (todos os organismos)
    dados/kgml/_glicanos.tsv        resposta de list/glycan

Uso:
    python -m coleta.baixar_kgml --org eco
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config


def requisitar(caminho_api: str) -> str:
    """Faz GET em {config.KEGG_URL}/{caminho_api} e devolve o texto.

    Espera config.PAUSA_ENTRE_REQUISICOES depois de cada chamada, para não
    sobrecarregar a API. Levanta erro em resposta diferente de 200, para a
    função chamadora decidir se registra a falha e segue.
    """
    raise NotImplementedError("TODO: requests.get + time.sleep")


def baixar_com_cache(caminho_api: str, destino: Path, forcar: bool = False) -> Path:
    """Baixa caminho_api para destino, a menos que destino já exista.

    É o único lugar que escreve no cache: todas as outras funções passam
    por aqui. Cria as pastas que faltarem.
    """
    raise NotImplementedError("TODO: se destino existe e não forcar, retorna; senão requisitar e gravar")


def listar_vias(org: str, forcar: bool = False) -> list[tuple[str, str]]:
    """Lista as vias do organismo: [(codigo, nome), ...], ex. ("eco00020", "Citrate cycle").

    Fonte: list/pathway/<org>, guardada em _vias.tsv. Cada linha tem o
    código e o nome separados por tab; remova o prefixo "path:" se aparecer.
    """
    raise NotImplementedError


def filtrar_vias_regulares(vias: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Mantém só as vias metabólicas regulares (número < config.LIMITE_VIAS_REGULARES).

    Descarta os mapas globais e de visão geral (011xx, 012xx), como o eco01100,
    e as vias de config.VIAS_EXCLUIDAS (abaixo de 01100, mas fora de
    Metabolismo, como a 00970). O número é formado pelos 5 últimos dígitos
    do código.
    """
    raise NotImplementedError


def baixar_kgml(org: str, via: str, forcar: bool = False) -> Path | None:
    """Baixa get/<via>/kgml para dados/kgml/<org>/<via>.xml.

    Devolve None se a via não tiver KGML (a API responde com erro); quem
    chama registra a falha nos metadados e segue para a próxima via.
    """
    raise NotImplementedError


def baixar_listas_globais(forcar: bool = False) -> None:
    """Baixa os nomes de todos os compostos e glicanos numa chamada cada.

    list/compound -> config.ARQ_COMPOSTOS, list/glycan -> config.ARQ_GLICANOS.
    Servem a todos os organismos, então ficam fora da pasta do organismo.
    """
    raise NotImplementedError


def baixar_genes(org: str, forcar: bool = False) -> Path:
    """Baixa list/<org> (genes do organismo) para dados/kgml/<org>/_genes.tsv."""
    raise NotImplementedError


def registrar_metadados(org: str, vias: list[tuple[str, str]], falhas: list[str]) -> Path:
    """Grava dados/kgml/<org>/_metadados.json.

    Conteúdo: data e hora do download, vias baixadas, vias que falharam e a
    resposta de info/<org> (versão da base). A data fica registrada porque
    o KEGG é atualizado continuamente.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os dados do KEGG para um organismo.")
    parser.add_argument("--org", default=config.ORGANISMO_PRINCIPAL, help="código KEGG do organismo")
    parser.add_argument("--forcar", action="store_true", help="baixa de novo mesmo o que está no cache")
    args = parser.parse_args()

    vias = filtrar_vias_regulares(listar_vias(args.org, args.forcar))
    print(f"{args.org}: {len(vias)} vias metabólicas regulares")

    falhas = []
    for codigo, _nome in vias:
        if baixar_kgml(args.org, codigo, args.forcar) is None:
            falhas.append(codigo)

    baixar_listas_globais(args.forcar)
    baixar_genes(args.org, args.forcar)
    registrar_metadados(args.org, vias, falhas)
    print(f"Concluído. Vias sem KGML: {len(falhas)}")


if __name__ == "__main__":
    main()
