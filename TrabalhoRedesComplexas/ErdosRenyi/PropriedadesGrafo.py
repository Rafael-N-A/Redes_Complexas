import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from GrafoErdosRenyi import GrafoErdosRenyi

NUM_VERTICES = 100
PROBABILIDADE = 0.1

def main():
    g = GrafoErdosRenyi(NUM_VERTICES, PROBABILIDADE)

    #Grau Maximo
    grau_maximo = max(dict(g.Grafo.degree()).values())
    print("Grau Maximo:", grau_maximo)

    #Grau Minimo
    grau_minimo = min(dict(g.Grafo.degree()).values())
    print("Grau Minimo:", grau_minimo)

    #Distancia Media
    if nx.is_connected(g.Grafo):
        distancia_media = nx.average_shortest_path_length(g.Grafo)
        print("Distancia Media:", distancia_media)
    else:
        for C in (g.Grafo.subgraph(c).copy() for c in nx.connected_components(g.Grafo)):
            distancia_media = nx.average_shortest_path_length(C)
            print("Distancia Media do Componente Conectado:", distancia_media)

    #Coeficiente de Clustering
    if nx.is_connected(g.Grafo):
        coeficiente_clustering = nx.average_clustering(g.Grafo)
        print("Coeficiente de Clustering:", coeficiente_clustering)
    else:
        for C in (g.Grafo.subgraph(c).copy() for c in nx.connected_components(g.Grafo)):
            coeficiente_clustering = nx.average_clustering(C)
            print("Coeficiente de Clustering do Componente Conectado:", coeficiente_clustering)

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
    plt.savefig("distribuicao_graus_CCDF.png")
    plt.close()

    # Plotar o grafo
    #plt.figure(figsize=(8, 5))
    #nx.draw(g.Grafo, with_labels=True)
    #plt.savefig("grafo_erdos_renyi.png")
    #plt.close()

if __name__ == "__main__":
    main()