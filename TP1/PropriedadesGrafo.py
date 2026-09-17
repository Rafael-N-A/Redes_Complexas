import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from CriacaoGrafo import GrafoErdosRenyi

NUM_VERTICES = [100, 1000, 10000]
# ordem preservada: primeiro 0-4, depois 5-8, depois 9-11
PROBABILIDADE = [0.005, 0.01, 0.02, 0.1, 0.0005, 0.001, 0.003, 0.01, 0.00005, 0.0001, 0.0004, 0.002]

def main():
    g = GrafoErdosRenyi(NUM_VERTICES, PROBABILIDADE)

    #Grau Maximo
    grau_maximo = max(dict(g.Grafo.degree()).values())
    print("Grau Maximo:", grau_maximo)

    #Grau Minimo
    grau_minimo = min(dict(g.Grafo.degree()).values())
    print("Grau Minimo:", grau_minimo)

    #Distancia Media
    #Grafo Conectado
    if nx.is_connected(g.Grafo):
        distancia_media = nx.average_shortest_path_length(g.Grafo)
        print("Distancia Media:", distancia_media)
    #Grafo Desconectado
    else:
        for C in (g.Grafo.subgraph(c).copy() for c in nx.connected_components(g.Grafo)):
            distancia_media = -1

    #Coeficiente de Clustering
    #Grafo Conectado
    if nx.is_connected(g.Grafo):
        coeficiente_clustering = nx.average_clustering(g.Grafo)
        print("Coeficiente de Clustering:", coeficiente_clustering)
    #Grafo Desconectado
    else:
        for C in (g.Grafo.subgraph(c).copy() for c in nx.connected_components(g.Grafo)):
            coeficiente_clustering = -1

    # Informações armazenadas em arquivos de cada grafo
    with open("ErdosRenyi.txt", "a", encoding="utf-8") as arquivo:
        arquivo.write(f"Numero de Vertices: {NUM_VERTICES}\n")
        arquivo.write(f"Probabilidade: {PROBABILIDADE}\n")
        arquivo.write(f"Grau Maximo: {grau_maximo}\n")
        arquivo.write(f"Grau Minimo: {grau_minimo}\n")
        arquivo.write(f"Distancia Media: {distancia_media}\n")
        arquivo.write(f"Coeficiente de Clustering: {coeficiente_clustering}\n")

    #Distribuicao de Graus
    distribuicao_graus = list(dict(g.Grafo.degree()).values())
    print("Distribuicao de Graus:", distribuicao_graus)

    contagem = Counter(distribuicao_graus)

    grau_val = sorted(contagem.keys())
    prob_vals = [contagem[grau] / NUM_VERTICES for grau in grau_val]

    prob_acumulada = np.cumsum(prob_vals[::-1])[::-1]

    plt.figure(figsize=(8, 5))
    plt.plot(grau_val, prob_acumulada, marker='o', linestyle='-', color='darkgreen')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Grau k")
    plt.ylabel("P(K >= k)")
    plt.title("Distribuição cumulativa de graus")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig("distribuicao_graus_CCDF100.png")
    plt.close()

    # Plotar o grafo
    #plt.figure(figsize=(8, 5))
    #nx.draw(g.Grafo, with_labels=True)
    #plt.savefig("grafo_erdos_renyi100.png")
    #plt.close()

if __name__ == "__main__":
    main()