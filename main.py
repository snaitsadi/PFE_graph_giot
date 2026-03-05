"""
Démonstration de dessin de graphe différentiable (optimisation directe).
"""
import torch
from graph import Graph
from renderer import DifferentiableRenderer
from losses import GraphLayoutLosses
from optimizer import optimize_graph
from visualization import plot_graph

if __name__ == "__main__":
    print("=== Démonstration de dessin de graphe différentiable (optimisation directe) ===")

    # Paramètres
    H, W = 64, 64
    N = 8  # nombre de nœuds

    # Création d'un graphe test (un cycle + quelques arêtes)
    adj = torch.zeros(N, N)
    for i in range(N):
        adj[i, (i+1)%N] = 1
        adj[(i+1)%N, i] = 1
    # Ajout de quelques arêtes supplémentaires
    adj[0, 3] = 1
    adj[3, 0] = 1
    g = Graph(adj)

    # Initialisation du rendu et des pertes
    renderer = DifferentiableRenderer(height=H, width=W, sigma_node=0.05, sigma_edge=0.02)
    losses = GraphLayoutLosses(renderer)

    # Affichage initial
    print("Graphe initial (positions aléatoires)")
    plot_graph(g, renderer, "Initial")

    # Optimisation directe
    print("\nOptimisation directe...")
    opt_pos = optimize_graph(g, renderer, losses, n_iter=500, lr=0.01)
    g_opt = Graph(g.adj, opt_pos)
    plot_graph(g_opt, renderer, "Après optimisation directe")

    print("Démonstration terminée.")