import numpy as np
import torch
from abc import ABC, abstractmethod
from . import distances


class Metric(ABC):

    @abstractmethod
    def compute(self, Z_layer, **kwargs):
        pass

    @staticmethod
    def to_one_hot(x: np.ndarray) -> np.ndarray:
        """
        Convert a 1D numpy array to one-hot encoding.
        """
        c = np.unique(x)
        I = torch.eye(len(c), dtype=torch.int32)
        M = torch.zeros((len(x), len(c)), dtype=torch.int32)
        for i, j in enumerate(c):
            M[x == j] = I[i]
        return M
    
    @staticmethod
    def CL_SNR(D, mask):
        """
        Compute the Signal-to-Noise Ratio across classes.
        D: 
        """
        within_dist = torch.masked_select(D, mask).mean()
        between_dist = torch.masked_select(D, ~mask).mean()
        return between_dist / (within_dist + between_dist)
    
    @staticmethod
    def label2mask(x: np.ndarray) -> np.ndarray:
        """
        Convert a one hot encoded array to a mask.
        x: NxC array
        mask: Nx(N-1)/2 x 1 boolean array of within class instances
        """
        N, C = x.shape
        r, c = torch.triu_indices(N, N, offset=1)
        mask = (C@C.T)[r, c].bool()
        return mask

# HERE GOES YOUR CODE