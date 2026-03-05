"""
Optimisation directe des positions d'un graphe par descente de gradient.
"""
import torch
import torch.optim as optim

def optimize_graph(graph, renderer, losses, n_iter=500, lr=0.01):
    """Optimise les positions d'un graphe par descente de gradient."""
    pos = graph.pos.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([pos], lr=lr)
    edges = graph.get_edges()
    adj = graph.adj

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = losses.total_loss(pos, adj, edges)
        loss.backward()
        optimizer.step()
        # Clipping pour rester dans [0,1]
        with torch.no_grad():
            pos.clamp_(0, 1)
        if i % 100 == 0:
            print(f"Iter {i}, loss = {loss.item():.4f}")
    return pos