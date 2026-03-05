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
