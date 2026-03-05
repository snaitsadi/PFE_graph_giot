"""
Rendu différentiable d'un graphe en image (grille de pixels).
"""
import torch

class DifferentiableRenderer:
      """Rendu d'un graphe sur une grille de pixels de façon différentiable."""
    def __init__(self, height=64, width=64, sigma_node=0.05, sigma_edge=0.02):
        self.H = height
        self.W = width
        self.sigma_node = sigma_node
        self.sigma_edge = sigma_edge
        # Grille de pixels (coordonnées normalisées entre 0 et 1)
        y = torch.linspace(0, 1, self.H)
        x = torch.linspace(0, 1, self.W)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')
        # Forme (H, W) pour les coordonnées
