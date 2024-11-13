import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Metric(nn.Module):

    @staticmethod
    def to_one_hot(y: np.ndarray, dtype, device) -> torch.Tensor:
        """
        Convert a 1D numpy array to one-hot encoding.
        """
        # check if y is a numpy array
        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a numpy array')
        c = np.unique(y)
        I = torch.eye(len(c), dtype=dtype, device=device)
        y_onehot = torch.zeros((len(y), len(c)), dtype=dtype, device=device)
        for i, j in enumerate(c):
            y_onehot[y == j] = I[i]
        return y_onehot
    
    @staticmethod
    def within_between(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the within and between class variance.
        x: NxD embedding matrix
        y: NxC one-hot encoded matrix
        """

        x_c = torch.einsum('nd,nc->cd', x, y) # CxD
        between = torch.norm(x_c - x_c.mean(dim=0, keepdim=True), p=2).mean()
        
        x_c = torch.einsum('cd,nc->nd', x_c, y) # NxD
        within = torch.norm(x - x_c, p=2).mean()

        return (within, between)
    
    def forward(self, Z: torch.Tensor, y: dict):
        # ...