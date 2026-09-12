import networkx as nx

class GrafoErdosRenyi:
    def __init__(self, n, p):
        self.n = n  # Numero de vertices
        self.p = p  # Probabilidade de conexao
        self.Grafo = nx.erdos_renyi_graph(n, p, 100)  # Gerar grafo