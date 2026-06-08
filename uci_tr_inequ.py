# This script creates a LaTex table allowing us to verify the trace ineqality 
# in Thm 4.5 on various UCI data sets (loading takes a sec)

import os
import torch

from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import compute_exact_posterior, compute_mfvi_analytic, post_pred_mean_var

from src.basis_functions import apply_basis

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

O_NOISE = .1
P_PRSCISN = 1
TR_FRC = 0.7

# BASIS = "rbf" # | "identity" | "polynomial"
# BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 50 }  # {} | {"degree": 5} 

BASIS = "identity"
BASIS_KWARGS = {}

set_seeds(42)

def tr_exp(X, y):

    # verifying Thm 4.5

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
    X_tr = (X_tr - m_X_tr) / (std_X_tr + 1e-12)
    X_te = (X_te - m_X_tr) / (std_X_tr + 1e-12)
    y_tr -= m_y_tr
    y_te -= m_y_tr

    if BASIS == "rbf":
        # centers = X_tr[:BASIS_KWARGS["m"]] # choose random center points
        # BASIS_KWARGS["centers"] = centers

        # sample center from Gaussian with emprical covariance
        eps = torch.randn((BASIS_KWARGS["m"], d)).to(X_tr)
        n_tr, _ = X_tr.shape
        X_trX_tr = X_tr.T @ X_tr / n_tr
        L_tr, _ = torch.linalg.cholesky_ex(X_trX_tr)
        centers = L_tr[None, ...] @ eps[..., None] 
 
        BASIS_KWARGS["centers"] = centers.squeeze(-1)

    X_tr = apply_basis(X=X_tr, basis_type=BASIS, **BASIS_KWARGS)
    X_te = apply_basis(X=X_te, basis_type=BASIS, **BASIS_KWARGS)

    
    # mu, Sigma = full_cov_post(train_X=X_tr, train_y=y_tr, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)
    mu, Sigma = compute_exact_posterior(train_X=X_tr, train_y=y_tr, noise_std=O_NOISE, prior_precision=P_PRSCISN)

    # m_opt, S_opt = mfvi_post_analytic(train_X=X_tr, train_y=y_tr, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)
    m_opt, S_opt = compute_mfvi_analytic(train_X=X_tr, train_y=y_tr, noise_std=O_NOISE, prior_precision=P_PRSCISN)

    _, train_true_post_var = post_pred_mean_var(test_X=X_tr, post_mean=mu, post_cov=Sigma, noise_std=None)
    _, train_mfvi_post_var = post_pred_mean_var(test_X=X_tr, post_mean=m_opt, post_cov=torch.diag(S_opt), noise_std=None)

    _, test_true_post_var = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, noise_std=None)
    _, test_mfvi_post_var = post_pred_mean_var(test_X=X_te, post_mean=m_opt, post_cov=torch.diag(S_opt), noise_std=None)

    avg_test_true_post_var = test_true_post_var.mean()
    avg_test_mfvi_post_var = test_mfvi_post_var.mean()

    avg_train_true_post_var = train_true_post_var.mean()
    avg_train_mfvi_post_var = train_mfvi_post_var.mean()

    # via trace

    # pre-process
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)

    if BASIS == "rbf":
        # centers = X_tr[:BASIS_KWARGS["m"]] # choose random center points
        # BASIS_KWARGS["centers"] = centers

        # sample center from Gaussian with emprical covariance
        eps = torch.randn((BASIS_KWARGS["m"], d)).to(X)
        n, _ = X.shape
        XX = X.T @ X / n
        L, _ = torch.linalg.cholesky_ex(XX)
        centers = L[None, ...] @ eps[..., None] 
 
        BASIS_KWARGS["centers"] = centers.squeeze(-1)

    X = apply_basis(X=X, basis_type=BASIS, **BASIS_KWARGS)

    y -= y.mean()
    _, Sigma = compute_exact_posterior(train_X=X, train_y=y, noise_std=O_NOISE, prior_precision=P_PRSCISN)
    _, S_opt = compute_mfvi_analytic(train_X=X, train_y=y, noise_std=O_NOISE, prior_precision=P_PRSCISN)

    trace_Sigma = torch.diag(Sigma).sum()
    trace_S_opt = torch.diag(S_opt).sum()

    return avg_train_mfvi_post_var, avg_train_true_post_var, avg_test_mfvi_post_var, avg_test_true_post_var, trace_S_opt, trace_Sigma


datasets = ["boston" , "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

def _fmt(x: float, *, sig: int = 4) -> str:
    # LaTeX-friendly formatting (uses scientific notation when needed)
    return f"{x:.{sig}g}"

if __name__ == "__main__":
    results = []

    for dataset in datasets:
        print("="*50)
        print(f"{dataset}".capitalize())
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype, device=device)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype, device=device)

        avg_train_mfvi_post_var, avg_train_true_post_var, avg_test_mfvi_post_var, avg_test_true_post_var, trace_S_opt, trace_Sigma = tr_exp(X, y)

        results.append({
            "dataset": dataset,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "avg_train_mfvi_post_var": float(avg_train_mfvi_post_var.item()),
            "avg_train_true_post_var": float(avg_train_true_post_var.item()),
            "avg_test_mfvi_post_var": float(avg_test_mfvi_post_var.item()),
            "avg_test_true_post_var": float(avg_test_true_post_var.item()),
            "trace_S_opt": float(trace_S_opt.item()),
            "trace_Sigma": float(trace_Sigma.item()),
        })
        for key in results[-1]:
            print(key, results[-1][key])
    cols = ["n", "d", "avg_train_mfvi_post_var", "avg_train_true_post_var","avg_test_mfvi_post_var", "avg_test_true_post_var", "trace_S_opt", "trace_Sigma"]

    os.makedirs("figs/uci/", exist_ok=True)
    output_path = "figs/uci/tr_inequ_table.txt"
    with open(output_path, "w") as f:
        print(r"\begin{table}[t]", file=f)
        print(r"\centering", file=f)
        print(r"\begin{tabular}{l" + "r" * len(cols) + r"}", file=f)
        print(r"\toprule", file=f)
        print(
            "Dataset & " + " & ".join([c.replace("_", r"\_") for c in cols]) + r" \\",
            file=f
        )
        print(r"\midrule", file=f)

        for row in results:
            line = [row["dataset"]]
            for c in cols:
                v = row[c]
                if c in ("n", "d"):
                    line.append(str(v))
                else:
                    line.append(_fmt(v, sig=5))
            print(" & ".join(line) + r" \\", file=f)

        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)
        BASIS_KWARGS.pop("centers", None)
        print(fr"\caption{{Results across datasets for BASIS={BASIS}, BASIS\_KWARGS={BASIS_KWARGS}.}}", file=f)
        print(r"\label{tab:results}", file=f)
        print(r"\end{table}", file=f)

    print(f"Wrote LaTeX table to {output_path}")


