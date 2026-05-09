"""
    DRONE: Data-aware Low-rank Compression for Large NLP Models
"""

import os
os.system('clear')
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

import numpy as np
from scipy.linalg import svd
from numpy.linalg import pinv, norm
import torch

def drone_projection(W, X, k, eps=1e-10):
    """
        PyTorch version of optimal rank-k projection matrix M* minimizing ||WX - MX||_F^2
        Parameters:
            W: (m x n) torch tensor
            X: (n x p) torch tensor
            k: desired rank of M*
            eps: numerical stability threshold for rank estimation
        Returns:
            M_star: (m x n) torch tensor
    """

    # Full SVD of W
    U_w, S_w, V_w_T = torch.linalg.svd(W, full_matrices=False)
    r = (S_w > eps).sum().item()
    U_w_r = U_w[:, :r]
    S_w_r = S_w[:r]
    V_w_r = V_w_T[:r, :]

    # Full SVD of X.T
    U_x, S_x, V_x_T = torch.linalg.svd(X.T, full_matrices=False)
    t = (S_x > eps).sum().item()
    U_x_t = U_x[:, :t]
    S_x_t = S_x[:t]
    V_x_t = V_x_T[:t, :]

    # Z = S_w_r @ V_w_r @ V_x_t.T @ S_x_t
    S_w_r_diag = torch.diag(S_w_r)
    S_x_t_diag = torch.diag(S_x_t)
    Z = S_w_r_diag @ V_w_r @ V_x_t.T @ S_x_t_diag

    # Truncated SVD of Z
    U_z, S_z, V_z_T = torch.linalg.svd(Z, full_matrices=False)
    U_z_k = U_z[:, :k]
    S_z_k = torch.diag(S_z[:k])
    V_z_k_T = V_z_T[:k, :]
    Z_k = U_z_k @ S_z_k @ V_z_k_T

    # M* = V_w_r.T @ S_w_r^-1 @ Z_k @ S_x_t^-1 @ V_x_t
    S_w_r_inv = torch.diag(1.0 / S_w_r)
    S_x_t_inv = torch.diag(1.0 / S_x_t)
    M_star = V_w_r.T @ S_w_r_inv @ Z_k @ S_x_t_inv @ V_x_t

    return M_star


if __name__=="__main__":

    # Set reproducibility
    torch.manual_seed(42)

    # Parameters
    m, n, p = 20, 15, 25   # W: [m x n], X: [n x p]
    k = 15                # Desired low rank (can be ≤ min(m, n, p))

    # Generate random test matrices as torch tensors
    W = torch.randn(m, n)
    X = torch.randn(n, p)

    # Compute optimal low-rank projection
    M_star = drone_projection(W, X, k)

    # Evaluate Frobenius norm of (WX - M*X)
    WX = W @ X
    WMX = W @ M_star @ X
    loss = torch.norm(WX - WMX, p='fro')

    print(f"Frobenius norm of WX - M*X: {loss.item():.6f}")