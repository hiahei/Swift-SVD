
import matplotlib.pyplot as plt
def plot_tensor_1d(tensor, title='1D Tensor Plot', logscale=False, save_path=False):
    """
    Plot a 1D PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): A 1D tensor to plot.
        title (str): Title of the plot.
        logscale (bool): Whether to use log scale for y-axis.
    """
    x = tensor.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    if logscale:
        plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.close()  # prevent inline display in notebooks
