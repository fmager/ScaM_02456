import torch
import torch.nn.functional as F

class DistFn:
    # Z is a matrix, with observations as rows and features as columns
    # Should return the pairwise distances between observations
    # If input size is NxD, output size should be Nx(N-1)/2

    @staticmethod
    def euclidean(Z):
        return DistFn.l2(Z)
    
    @staticmethod
    def l2(Z):
        return F.pdist(Z, p=2)
    
    @staticmethod
    def l1(Z):
        return F.pdist(Z, p=1)
    
    @staticmethod
    def cosine(Z):
        Z = Z / torch.norm(Z, dim=1, keepdim=True)
        Z = torch.abs(Z @ Z.T)
        rows, cols = torch.triu_indices(Z.size(0), Z.size(1), offset=1)
        return Z[rows, cols]

