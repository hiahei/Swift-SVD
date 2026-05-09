import os
os.system('clear')
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import pickle as pk
import time

def SVD_compress(data, r):
    """
        Perform SVD compression and return the compressed result.
    """

    U, S, V_T = torch.linalg.svd(data, full_matrices=False) # Perform SVD
    # Truncate to rank r (specified)
    U_r = U[:, :r]
    S_r = S[:r]
    V_r_T = V_T[:r, :]
    V_r = V_r_T.T

    # Convert to diagonal matrix (sigma) using top-r singular values
    S_r = torch.diag(S_r)

    data_recon = data @ V_r @ V_r_T # Method derived in the paper
    data_svd = U_r @ S_r @ V_r_T # Standard SVD reconstruction
    return data_recon, data_svd

# Algorithm 1
class IncrementalSVD:
    def __init__(self, dim: int, dtype=torch.float32, device='cuda', name='IncrementalSVD_module'):
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.device = device
        # Keep on GPU; use float32 to avoid accumulation overflow
        self.XtX = torch.zeros((dim, dim), dtype=torch.float32, device='cuda')
        self.total_samples = 0
        self.V = None
        self.S = None

    def feed(self, x: torch.Tensor):
        # Accumulate on the fly; no buffer storage
        x = x.detach()
        x = x.view(-1, x.size(-1)).to(torch.float32)
        if x.device != self.XtX.device:
            x = x.to(self.XtX.device)
        # In-place accumulation for efficiency

        # Divide by token count before each accumulation to avoid overflow
        # self.current_sample = x.size(0)
        # self.XtX /= self.current_sample

        self.XtX.addmm_(x.T, x)
        self.total_samples += x.size(0)

    def finalize(self):
        # Divide by total token count to avoid overflow
        # self.XtX /= self.total_samples

        # Perform eigendecomposition on GPU
        eigvals, eigvecs = torch.linalg.eigh(self.XtX)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        S = torch.sqrt(eigvals.clamp(min=0))
        # Move results to CPU to save GPU memory
        self.V = eigvecs.to('cpu')
        self.S = S.to('cpu')

        # Release XtX to free GPU memory
        # del self.XtX
        self.XtX = None # Safe release
        torch.cuda.empty_cache()
        return self.V, self.S

    def evaluate_projection(self, X: torch.Tensor, r: int):
        """
            Project X onto top-r components and compute reconstruction RMSE.

            Args:
                X: original data [B, D]
                r: number of components to keep

            Returns:
                X_recon: reconstructed data [B, D]
                rmse: scalar reconstruction error
        """
        if self.V is None:
            raise RuntimeError("Call finalize() before evaluate_projection().")

        device = self.V.device
        X = X.to(device)
        X = X.to(torch.float32)

        V_r = self.V[:, :r]  # [D, r] top-r V_k

        X_low = X @ V_r
        X_recon = X_low @ V_r.T
        
        # Compressed W_k = X_recon
        loss = torch.norm(X - X_recon, p='fro') # Compute error
        return X_recon, loss


if __name__=="__main__":

    # Generate toy data
    data = torch.randn(20000, 32).repeat(1, 4)  # [20000, 128]
    print(data.shape)  # torch.Size([20000, 128])
    # print(data[:10, :10])

    # Initialize IncrementalSVD with feed/flush logic
    svd = IncrementalSVD(dim=128, device='cpu', buffer_size=2048)

    # Create a list of start/end indices to simulate variable-length batches
    indices = list(range(0, len(data)))
    random.shuffle(indices)  # just for realism

    start = 0
    while start < len(indices):
        # Choose random batch length between 100 and 800
        length = random.randint(100, 800)
        end = min(start + length, len(indices))
        batch_indices = indices[start:end]
        batch = data[batch_indices]
        print(batch.shape)
        svd.feed(batch)
        start = end

    # Finalize and get singular vectors/values
    V, S = svd.finalize()

    # Reconstruct and evaluate
    data_recon, loss = svd.evaluate_projection(data, r=32)
    print(f"Incremental: {loss:.4f}")

