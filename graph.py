"""
Représentation d'un graphe simple avec matrice d'adjacence et positions.
"""
import torch

class Graph:
    """Graphe simple avec matrice d'adjacence et positions des nœuds."""
    def __init__(self, adj_matrix, pos=None):
        """
        Args:
            adj_matrix: torch.Tensor de taille (N, N) binaire (0/1)
            pos: torch.Tensor de taille (N, 2) dans [0,1] (optionnel)
        """
        self.N = adj_matrix.shape[0]
        self.adj = adj_matrix
        if pos is None:
            # positions aléatoires initiales
            self.pos = torch.rand(self.N, 2, requires_grad=True)
        else:
            self.pos = pos

    def get_edges(self):
        """Retourne la liste des arêtes (paires d'indices)."""
        edges = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.adj[i, j] == 1:
                    edges.append((i, j))
        return edges