"""
Fonctions de coût (losses) pour l'optimisation du dessin de graphe.
"""
import torch
import numpy as np
from renderer import DifferentiableRenderer



class GraphLayoutLosses:
    """Collection de pertes pour l'optimisation du dessin."""
    def __init__(self, renderer):
        self.renderer = renderer



    def stress_loss(self, pos, adj):
        """
        Perte de stress classique basée sur les distances dans le graphe.
        """
        N = pos.shape[0]
        # Calcul des distances dans le graphe (Floyd-Warshall)
        adj_np = adj.detach().cpu().numpy()
        dist_graph = np.full((N, N), np.inf)
        np.fill_diagonal(dist_graph, 0)
        for i in range(N):
            for j in range(N):
                if adj_np[i, j] == 1:
                    dist_graph[i, j] = 1
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if dist_graph[i, k] + dist_graph[k, j] < dist_graph[i, j]:
                        dist_graph[i, j] = dist_graph[i, k] + dist_graph[k, j]
        dist_graph = torch.tensor(dist_graph, dtype=torch.float32, device=pos.device)
        max_dist = torch.max(dist_graph[torch.isfinite(dist_graph)])
        dist_graph[~torch.isfinite(dist_graph)] = max_dist * 2
        # Distances euclidiennes entre positions
        pos_i = pos.unsqueeze(1)
        pos_j = pos.unsqueeze(0)
        dist_eucl = torch.sqrt(((pos_i - pos_j)**2).sum(dim=2) + 1e-8)

        weight = 1.0 / (dist_graph + 1e-8)
        loss = weight * (dist_eucl - dist_graph * 0.1)**2
        return loss.mean()

    def node_overlap_loss(self, pos):
        """Pénalise les chevauchements de nœuds via l'image rendue."""
        node_img = self.renderer.render_nodes(pos)
        return (node_img**2).mean()
    

    def edge_crossing_penalty(self, pos, edges):
        """Pénalise les croisements d'arêtes via le Laplacien de l'image."""
        edge_img = self.renderer.render_edges(pos, edges)
        laplacian = torch.zeros_like(edge_img)
        laplacian[1:-1, 1:-1] = (edge_img[1:-1, 1:-1] * -4 +
                                  edge_img[:-2, 1:-1] + edge_img[2:, 1:-1] +
                                  edge_img[1:-1, :-2] + edge_img[1:-1, 2:])
        return torch.abs(laplacian).mean()

    def total_loss(self, pos, adj, edges, weights=(1.0, 0.1, 0.01)):
        """Combinaison linéaire des pertes."""
        loss_stress = self.stress_loss(pos, adj)
        loss_overlap = self.node_overlap_loss(pos)
        loss_cross = self.edge_crossing_penalty(pos, edges)
        return (weights[0] * loss_stress +
                weights[1] * loss_overlap +
                weights[2] * loss_cross)