# This script creates a LaTex table allowing us to verify the trace ineqality 
# in Thm 4.5 on various UCI data sets (loading ta)

import os
from uci_data import load_dataset
import torch
dtype = torch.float64

O_NOISE = .1
P_PRSCISN = 1
TR_FRC = 0.7

def full_cov_post(train_X, train_y, obs_noise, prior_prscisn):

    # Mean and covariance of true posterior under a Normal-Normal linear model 

    n, d = train_X.shape
    Sigma_inv = train_X.T @ train_X / obs_noise**2 + prior_prscisn * torch.eye(d, dtype=dtype)
    Sigma = torch.linalg.inv(Sigma_inv)

    mu = Sigma @ train_X.T @ train_y / obs_noise**2

    return mu, Sigma

def mfvi_post_analytic(train_X, train_y, obs_noise, prior_prscisn):

    # Optimal (in KL) mean and covariance of MFVI posterior under a Normal-Normal linear model
    # Note: Here we compute them analytically, which is usually what we want to avoid 

    n, d = train_X.shape
    Sigma_inv = train_X.T @ train_X / obs_noise**2 + prior_prscisn * torch.eye(d, dtype=dtype)
    Sigma_true = torch.linalg.inv(Sigma_inv)
    Sigma = torch.diag(1 / torch.diag(Sigma_inv))

    mu = Sigma_true @ train_X.T @ train_y / obs_noise**2

    return mu, Sigma


def post_pred_mean_var(test_X, post_mean, post_cov, obs_noise = None, temp = torch.ones((1,1), dtype=dtype)):
    
    # given (estimates of) the posterior parameters, 
    # compute the posterior predictive mean and variance
    # optional: temper the posterior with array shape (n_temps, 1)

    mean = test_X @ post_mean
    var = test_X.unsqueeze(1) @ post_cov.unsqueeze(0) @ test_X.unsqueeze(1).swapaxes(1,2) 
    t_var = var.squeeze(1) @ temp.T # (n, n_temps)

    return (mean, t_var) if obs_noise is None else (mean, t_var + obs_noise**2)

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

    mu, Sigma = full_cov_post(train_X=X_tr, train_y=y_tr, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)

    m_opt, S_opt = mfvi_post_analytic(train_X=X_tr, train_y=y_tr, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)

    _, true_post_var = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, obs_noise=None)
    _, mfvi_post_var = post_pred_mean_var(test_X=X_te, post_mean=m_opt, post_cov=S_opt, obs_noise=None)

    avg_true_post_var = true_post_var.mean()
    avg_mfvi_post_var = mfvi_post_var.mean()

    # via trace

    # pre-process
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y -= y.mean()
    _, Sigma = full_cov_post(train_X=X, train_y=y, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)
    _, S_opt = mfvi_post_analytic(train_X=X, train_y=y, obs_noise=O_NOISE, prior_prscisn=P_PRSCISN)

    trace_Sigma = torch.diag(Sigma).sum()
    trace_S_opt = torch.diag(S_opt).sum()

    return avg_mfvi_post_var, avg_true_post_var, trace_S_opt, trace_Sigma

datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

def _fmt(x: float, *, sig: int = 4) -> str:
    # LaTeX-friendly formatting (uses scientific notation when needed)
    return f"{x:.{sig}g}"

if __name__ == "__main__":
    results = []

    for dataset in datasets:
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype)

        avg_mfvi_post_var, avg_true_post_var, trace_S_opt, trace_Sigma = tr_exp(X, y)

        results.append({
            "dataset": dataset,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "avg_mfvi_post_var": float(avg_mfvi_post_var.item()),
            "avg_true_post_var": float(avg_true_post_var.item()),
            "trace_S_opt": float(trace_S_opt.item()),
            "trace_Sigma": float(trace_Sigma.item()),
        })

    cols = ["n", "d", "avg_mfvi_post_var", "avg_true_post_var", "trace_S_opt", "trace_Sigma"]

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
        print(r"\caption{Results across datasets.}", file=f)
        print(r"\label{tab:results}", file=f)
        print(r"\end{table}", file=f)

    print(f"Wrote LaTeX table to {output_path}")


