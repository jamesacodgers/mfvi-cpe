# this script creates plots of the NPLL as a function of posterior temperature
# for various UCI data sets
import os
import torch
from math import log 
import matplotlib.pyplot as plt

from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import compute_mfvi_analytic, compute_exact_posterior, post_pred_mean_var, opt_sigma_l
from src.basis_functions import apply_basis


dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

O_NOISE = .1 # overridden if LEARN_NOISE_L == True
LEARN_NOISE_L = False
P_PRSCISN = 1
TR_FRC = 0.7

BASIS = "rbf" # | "identity" | "polynomial"
BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 499 }  # {} | {"degree": 5} 

# BASIS = "identity"
# BASIS_KWARGS = {}

set_seeds(42)

def npll(X, y, Ts):

    # calculate NPLL
    # Note: we return the mean to make numbers comparable acorss data set size

    n, d = X.shape
    
    # via expectation under true input distribution

    # pre-process
    tt_split_ind = int(TR_FRC * n)
    perm = torch.randperm(n, device=X.device)
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

        # sample center from Gaussian with emprical covariance
        eps = torch.randn((BASIS_KWARGS["m"], d), dtype=dtype, device=X.device)
        n_tr, d = X_tr.shape
        X_trX_tr = X_tr.T @ X_tr / n_tr
        L_tr, _ = torch.linalg.cholesky_ex(X_trX_tr + 1e-6 * torch.eye(d).to(X))
        centers = L_tr[None, ...] @ eps[..., None] 
        BASIS_KWARGS["centers"] = centers.squeeze(-1)

    y_tr -= m_y_tr
    y_te -= m_y_tr

    noise_std = O_NOISE
    if LEARN_NOISE_L:
        sigma, l,  _ = opt_sigma_l(X_tr, y_tr, prior_precision=P_PRSCISN, basis_kwargs=BASIS_KWARGS, basis_type=BASIS,
                                   init_sigma=0.1, init_l=1, verbose=True, steps=1000)
        noise_std = sigma
        BASIS_KWARGS["lengthscale"] = l

    X_tr = apply_basis(X=X_tr, basis_type=BASIS, **BASIS_KWARGS); print(X_tr.shape)
    X_te = apply_basis(X=X_te, basis_type=BASIS, **BASIS_KWARGS)

    mu, Sigma = compute_exact_posterior(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=P_PRSCISN)

    m_opt, S_opt = compute_mfvi_analytic(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=P_PRSCISN)

    true_post_mean, true_post_var = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std, temp=Ts)
    
    mfvi_post_mean, mfvi_post_var = post_pred_mean_var(test_X=X_te, post_mean=m_opt, post_cov=torch.diag(S_opt), 
                                                       noise_std=noise_std, temp=Ts)
    
    true_npll = log(2*torch.pi) / 2 + \
                    torch.log(true_post_var) / 2 + \
                    (y_te - true_post_mean)[:, None]**2 / true_post_var / 2
    mfvi_nplls = log(2*torch.pi) / 2 + \
                    torch.log(mfvi_post_var) / 2 + \
                    (y_te - mfvi_post_mean)[:, None]**2 / mfvi_post_var / 2
    

    true_T1_mean, true_T1_var = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std)
    
    true_T1_npll = log(2*torch.pi) / 2 + \
                        torch.log(true_T1_var) / 2 + \
                        (y_te - true_T1_mean)[:, None]**2 / true_T1_var / 2
    kls = 0.5 * torch.log(mfvi_post_var / true_T1_var) + 0.5 * true_T1_var / mfvi_post_var - 0.5

    return torch.mean(true_T1_npll, dim = 0), torch.mean(true_npll, dim = 0), torch.mean(mfvi_nplls, dim = 0), torch.mean(kls, dim=0)
    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

if __name__ == "__main__":
    Ts = 10 ** torch.linspace(-3, 3, 100, dtype=dtype, device=device) 
    log10Ts = torch.log10(Ts)

    if BASIS == "rbf":
        p = BASIS_KWARGS["m"]
        l = BASIS_KWARGS["lengthscale"] if not LEARN_NOISE_L else "learned"
    
    if BASIS == "polynomial":
        deg = BASIS_KWARGS["degree"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()

    eigvals_fig, eigvals_axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    eigvals_axes = eigvals_axes.ravel()

    log10Ts_np = log10Ts.detach().cpu().numpy()

    ax2_first = None

    for ax, eigvals_ax, dataset in zip(axes, eigvals_axes, datasets):
        print("="*50)
        print(dataset.capitalize())
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype, device=device)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype, device=device)

        true_t1_npll, true_npll, mfvi_nplls, kl = npll(X, y, Ts[:, None])
        true_t1_npll = true_t1_npll.detach().cpu().numpy()
        true_npll = true_npll.detach().cpu().numpy()
        mfvi_nplls = mfvi_nplls.detach().cpu().numpy()
        kl = kl.detach().cpu().numpy()

        ax.plot(log10Ts_np, mfvi_nplls, label="MFVI", color = "orange")
        ax.plot(log10Ts_np, true_npll, label="Full Covariance", color = "blue")
        ax.axhline(true_t1_npll, linestyle="--", label="True posterior", color = "blue")

        ax2 = ax.twinx()
        if ax2_first is None:
            ax2_first = ax2
        ax2.plot(log10Ts_np, kl, label="post pred KL(true(T=1) || T-mfvi) ", color="green")
        ax2.set_ylabel("KL")

        ax.set_title(dataset)
        ax.set_xlabel(r"$\log_{10}(T)$")
        ax.set_ylabel("Mean Test NLL")
        ax.grid(True, alpha=0.3)

        m_X = X.mean(0)
        std_X = X.std(0)
        X = (X - m_X) / (std_X + 1e-8)

        Phi = apply_basis(X=X, basis_type=BASIS, **BASIS_KWARGS)
        PhiTPhi = Phi.T @ Phi
        eigvals = torch.linalg.eigvalsh(PhiTPhi).detach().cpu().numpy()

        eigvals_ax.scatter(range(len(eigvals)), eigvals[::-1], label="Eigenvalues")
        eigvals_ax.set_title(dataset)
        eigvals_ax.set_xlabel("index")
        eigvals_ax.set_ylabel("value")
        eigvals_ax.grid(True, alpha=0.3)

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax2_first.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
            loc="outside lower center",
            ncol=2,
            frameon=False)

    fig.suptitle(f"basis: {BASIS}")


    if BASIS == "rbf":
        outpath = f"figs/uci/uci_lin_npll_test_{BASIS}_p={p}_l={l}.pdf"
        eigvals_outpath = f"figs/uci/uci_eigvals_{BASIS}_p={p}_l={l}.pdf"

    elif BASIS == "polynomial":
        outpath = f"figs/uci/uci_lin_npll_test_{BASIS}_deg={deg}.pdf"
        eigvals_outpath = f"figs/uci/uci_eigvals_{BASIS}_deg={deg}.pdf"

    else:
        outpath = f"figs/uci/uci_lin_npll_test_{BASIS}.pdf"
        eigvals_outpath = f"figs/uci/uci_eigvals_{BASIS}.pdf"

    os.makedirs("figs/uci/", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

    handles, labels = eigvals_axes[0].get_legend_handles_labels()
    eigvals_fig.legend(handles, labels,
            loc="outside lower center",
            ncol=2,
            frameon=False)

    eigvals_fig.suptitle("Eigenvalues of Feature Gramm matrix")
    eigvals_fig.savefig(eigvals_outpath, bbox_inches="tight")
    
    print(f"Saved figure to {outpath}")
    
    print(f"Saved figure to {eigvals_outpath}")
