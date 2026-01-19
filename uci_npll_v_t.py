# this script creates plots of the NPLL as a function of posterior temperature
# for various UCI data sets
import os
import torch
from math import log 
import matplotlib.pyplot as plt

from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import compute_mfvi_analytic, compute_exact_posterior, post_pred_mean_var
from src.basis_functions import apply_basis

dtype = torch.float64

O_NOISE = .1
P_PRSCISN = 1
TR_FRC = 0.7

BASIS = "rbf" # | "identity" | "polynomial"
BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 5000 }  # {} | {"degree": 5} 

set_seeds(42)

def npll(X, y, Ts):

    # calculate NPLL
    # Note: we return the mean to make numbers comparable acorss data set size

    n, d = X.shape

    # via expectation under true input distribution

    # pre-process
    tt_split_ind = int(TR_FRC * n)
    perm = torch.randperm(n)
    X = X[perm]
    X_tr = X[:tt_split_ind]
    X_te = X[tt_split_ind:]
    y = y[perm]
    y_tr = y[:tt_split_ind]
    y_te = y[tt_split_ind:]
    m_X_tr = X_tr.mean(0)
    std_X_tr = X_tr.std(0)
    m_y_tr = y_tr.mean()
    X_tr = (X_tr - m_X_tr) / (std_X_tr + 1e-8)
    X_te = (X_te - m_X_tr) / (std_X_tr + 1e-8)

    if BASIS == "rbf":
        # centers = X_tr[:BASIS_KWARGS["m"]] # choose random center points
        # BASIS_KWARGS["centers"] = centers
        centers = torch.randn((BASIS_KWARGS["m"], d))
        BASIS_KWARGS["centers"] = centers

    X_tr = apply_basis(X=X_tr, basis_type=BASIS, **BASIS_KWARGS)
    X_te = apply_basis(X=X_te, basis_type=BASIS, **BASIS_KWARGS)

    y_tr -= m_y_tr
    y_te -= m_y_tr

    mu, Sigma = compute_exact_posterior(train_X=X_tr, train_y=y_tr, noise_std=O_NOISE, prior_precision=P_PRSCISN)

    m_opt, S_opt = compute_mfvi_analytic(train_X=X_tr, train_y=y_tr, noise_std=O_NOISE, prior_precision=P_PRSCISN)

    true_post_mean, true_post_var = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=O_NOISE)
    
    mfvi_post_mean, mfvi_post_var = post_pred_mean_var(test_X=X_te, post_mean=m_opt, post_cov=torch.diag(S_opt), 
                                                       noise_std=O_NOISE, temp=Ts)
    
    true_npll = log(2*torch.pi) / 2 + \
                    torch.log(true_post_var) / 2 + \
                    (y_te - true_post_mean)[:, None]**2 / true_post_var / 2
    mfvi_nplls = log(2*torch.pi) / 2 + \
                    torch.log(mfvi_post_var) / 2 + \
                    (y_te - mfvi_post_mean)[:, None]**2 / mfvi_post_var / 2

    return torch.mean(true_npll, dim = 0), torch.mean(mfvi_nplls, dim = 0)
    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]



if __name__ == "__main__":
    Ts = 10 ** torch.linspace(-4, 0, 100, dtype=dtype) 
    log10Ts = torch.log10(Ts)

    p = BASIS_KWARGS["m"]
    l = BASIS_KWARGS["lengthscale"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()

    eigvals_fig, eigvals_axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    eigvals_axes = eigvals_axes.ravel()

    for ax, eigvals_ax, dataset in zip(axes, eigvals_axes, datasets):
        print("="*50)
        print(dataset.capitalize())
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype)

        true_npll, mfvi_nplls = npll(X, y, Ts[:, None])  
        true_npll = true_npll.numpy()
        mfvi_nplls = mfvi_nplls.numpy()

        # MFVI curve
        ax.plot(log10Ts, mfvi_nplls, label="MFVI")

        # True posterior horizontal line (constant across temperatures)
        ax.axhline(true_npll, linestyle="--", label="True posterior")

        ax.set_title(dataset)
        ax.set_xlabel(r"$\log_{10}(T)$")
        ax.set_ylabel("Mean Test NLL")

        ax.grid(True, alpha=0.3)
        m_X = X.mean(0)
        std_X = X.std(0)
        X = (X - m_X) / (std_X + 1e-8)

        Phi = apply_basis(X=X, basis_type=BASIS, **BASIS_KWARGS)
        PhiTPhi = Phi.T @ Phi
        eigvals = torch.linalg.eigvalsh(PhiTPhi).numpy()

        eigvals_ax.scatter(range(p), eigvals[::-1], label="Eigenvalues")

        eigvals_ax.set_title(dataset)
        eigvals_ax.set_xlabel("index")
        eigvals_ax.set_ylabel("value")

        eigvals_ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="outside lower center",
            ncol=2,
            frameon=False)
    
    fig.suptitle(f"basis: {BASIS}")


    outpath = f"figs/uci/uci_lin_npll_test_{BASIS}_p={p}_l={l}.pdf"
    os.makedirs("figs/uci/", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

    handles, labels = eigvals_axes[0].get_legend_handles_labels()
    eigvals_fig.legend(handles, labels,
            loc="outside lower center",
            ncol=2,
            frameon=False)

    eigvals_fig.suptitle("Eigenvalues of Feature Gramm matrix")
    eigvals_outpath = f"figs/uci/uci_eigvals_{BASIS}_p={p}_l={l}.pdf"
    eigvals_fig.savefig(eigvals_outpath, bbox_inches="tight")
    
    print(f"Saved figure to {outpath}")
    
    print(f"Saved figure to {eigvals_outpath}")






    
