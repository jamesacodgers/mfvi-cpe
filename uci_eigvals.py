# this script calculates and plots the eigen spectrum 
# of the input Gramm matrix of variuous UCI data sets

import os
import torch
import matplotlib.pyplot as plt

from src.UCI_data import load_dataset


dtype = torch.float64
    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

if __name__ == "__main__":

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, dataset in zip(axes, datasets):
        print("="*50)
        print(dataset.capitalize())
        X_df, _ = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype)
        n, d = X.shape

        X = (X - X.mean(0)) / (X.std(0) + 1e-12)

        XX = X.T@X

        eigvals = torch.linalg.eigvalsh(XX).numpy()

        print(eigvals.shape)

        ax.scatter(range(d), eigvals[::-1], label="Eigenvalues")

        ax.set_title(dataset)
        ax.set_xlabel("index")
        ax.set_ylabel("value")

        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="outside lower center",
            ncol=2,
            frameon=False)
    
    fig.suptitle("Eigenvalues of Gramm matrix")

    outpath = f"figs/uci/uci_eigenspectrum.pdf"
    os.makedirs("figs/uci/", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    
    print(f"Saved figure to {outpath}")

