import networkx as nx

class GrafoErdosRenyi:
    def __init__(self, n, p):
        self.n = n  # Numero de vertices
        self.p = p  # Probabilidade de conexao
        self.Grafo = nx.erdos_renyi_graph(n, p, 100)  # Gerar grafo

class GrafoWattsStrogatz:
    def __init__(self, n, k, p):
        self.n = n  # Numero de vertices
        self.k = k  # Numero de vizinhos mais proximos
        self.p = p  # Probabilidade de reconexao
        self.Grafo = nx.watts_strogatz_graph(n, k, p, 100)  # Gerar grafo

class GrafoBarabasiAlbert:
    def __init__(self, n, m):
        self.n = n  # Numero de vertices
        self.m = m  # Numero de arestas a serem adicionadas para cada novo vertice
        self.Grafo = nx.barabasi_albert_graph(n, m, 100)  # Gerar grafo