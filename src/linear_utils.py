import torch
import numpy as np
from torch.distributions import Normal
import math

from src.basis_functions import apply_basis
from src.utils import check_bad

dtype = torch.float64

# Set default dtype to float64 for numerical stability
# torch.set_default_dtype(torch.float64)


def post_pred_mean_var(test_X, post_mean, post_cov, noise_std = None, temp = torch.ones((1,1))):
    
    # given (estimates of) the posterior parameters, 
    # compute the posterior predictive mean and variance
    # optional: temper the posterior with array shape (n_temps, 1)

    mean = test_X @ post_mean
    var = test_X.unsqueeze(1) @ post_cov.unsqueeze(0) @ test_X.unsqueeze(1).swapaxes(1,2) 
    t_var = var.squeeze(1) @ temp.T.to(test_X) # (n, n_temps)

    return (mean, t_var) if noise_std is None else (mean, t_var + noise_std**2)

def compute_mfvi_analytic(train_X: torch.Tensor, train_y: torch.Tensor, temperature: float = 1.0, prior_precision = 1.0,
                          noise_std: float = 1.0):
    """
    Compute closed-form mean-field VI solution for Cold Posterior Bayesian linear regression.
    Both likelihood and prior are scaled by 1/T.
    """
    XTX = train_X.T @ train_X
    XTy = train_X.T @ train_y
    
    # Posterior precision (Cold: 1/T * (Likelihood Precision + Prior Precision))
    # Total precision = 1/T * (X^T X / sigma^2 + I)
    precision = (XTX / (noise_std**2) + prior_precision*torch.eye(train_X.shape[1]).to(train_X)) / temperature
    
    # Posterior mean (Independent of T)
    mu = torch.linalg.solve(precision, XTy / (temperature * noise_std**2))
    
    # Posterior variances (diagonal only for mean-field)
    XTX_diag = torch.diag(XTX)
    sigma_sq = (temperature * noise_std**2) / (XTX_diag + prior_precision*noise_std**2)
    return mu, sigma_sq

def compute_exact_posterior(train_X, train_y, noise_std, prior_precision):

    # Mean and covariance of true posterior under a Normal-Normal linear model 

    n, d = train_X.shape
    Precision_matrix = train_X.T @ train_X / noise_std**2 + prior_precision * torch.eye(d).to(train_X)
    L = torch.linalg.cholesky(Precision_matrix)
    Sigma = torch.cholesky_inverse(L)

    mu = Sigma @ train_X.T @ train_y / noise_std**2

    return mu, check_bad(Sigma)

def test_nll_mfvi(X_test: torch.Tensor, y_test: torch.Tensor, 
                  mu: torch.Tensor, sigma: torch.Tensor,
                  noise_std: float = 1.0, n_samples: int = 500):
    """Monte Carlo estimation of NLL for MFVI posterior."""
    eps = torch.randn(n_samples, len(mu))
    weights = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll

def test_nll_exact(X_test: torch.Tensor, y_test: torch.Tensor, 
                   mu_post: torch.Tensor, Sigma_post: torch.Tensor,
                   noise_std: float = 1.0, n_samples: int = 500):
    """Monte Carlo estimation of NLL for Exact posterior."""
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll

def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42, diagonal_input: bool = True):
    """Generate linear regression data with a specific input structure."""
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    if diagonal_input:
        # All features are copies of the first one (high redundancy)
        x1 = torch.randn(n_samples) * input_std
        X = x1.unsqueeze(1).repeat(1, n_dims)
    else:
        # Standard independent features
        X = torch.randn(n_samples, n_dims) * input_std
        
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights


def compute_analytic_nll_approximations(mu: torch.Tensor, sigma: torch.Tensor, 
                                      true_weights: torch.Tensor, Sigma_exact: torch.Tensor,
                                      input_std: float, noise_std: float):
    """
    Computes 1st and 2nd order Delta Method approximations of the Expected NLL.
    Optimized for the specific case where Input Covariance Sigma = input_std^2 * Ones_Matrix.
    """
    D = len(mu)
    var_x = input_std**2
    S_diag = sigma
    
    tr_S_Sigma = var_x * torch.sum(S_diag)
    tr_S_Sigma_sq = (var_x**2) * (torch.sum(S_diag))**2
    K = tr_S_Sigma + noise_std**2
    
    def compute_nll_internal(N, correction_term):
        nll_1 = 0.5 * (N / K + torch.log(2 * torch.tensor(np.pi) * K))
        corr = ((K - 2*N) / (2 * K**3)) * tr_S_Sigma_sq + (1 / K**2) * correction_term
        nll_2 = nll_1 - corr
        return nll_1.item(), nll_2.item()

    # 1. True Weights
    beta_diff = true_weights - mu
    tr_M_Sigma_true = var_x * (torch.sum(beta_diff))**2
    tr_M_Sigma_S_Sigma_true = (var_x**2) * (torch.sum(beta_diff)**2) * torch.sum(S_diag)
    nll_true = compute_nll_internal(tr_M_Sigma_true + noise_std**2, tr_M_Sigma_S_Sigma_true)

    # 2. Approx Posterior
    tr_M_Sigma_approx = tr_S_Sigma
    tr_M_Sigma_S_Sigma_approx = tr_S_Sigma_sq
    nll_approx = compute_nll_internal(tr_M_Sigma_approx + noise_std**2, tr_M_Sigma_S_Sigma_approx)

    # 3. Exact Posterior
    tr_M_Sigma_exact = var_x * torch.sum(Sigma_exact)
    tr_M_Sigma_S_Sigma_exact = (var_x**2) * torch.sum(S_diag) * torch.sum(Sigma_exact)
    nll_exact = compute_nll_internal(tr_M_Sigma_exact + noise_std**2, tr_M_Sigma_S_Sigma_exact)
    
    return nll_true, nll_approx, nll_exact


import math
import torch

def opt_sigma(
    X: torch.Tensor,
    y: torch.Tensor,
    prior_precision: float,
    *,
    lr: float = 1e-2,
    steps: int = 500,
    init_sigma: float = 0.1,
    jitter: float = 1e-8,
    verbose: bool = False,
):
    # assume X is already standardized and y is centered

    device = X.device
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)

    n, d = X.shape
    a = torch.as_tensor(prior_precision, dtype=dtype, device=device)

    log_sigma2 = torch.tensor(
        math.log(init_sigma**2), dtype=dtype, device=device, requires_grad=True
    )
    opt = torch.optim.Adam([log_sigma2], lr=lr)

    I_n = torch.eye(n, dtype=dtype, device=device)
    I_d = torch.eye(d, dtype=dtype, device=device)

    use_primal = (n > d)  # primal (dxd) if n>d, dual (nxn) if n<=d

    if use_primal:
        XX = X.T @ X
        Xy = X.T @ y
    else:
        K = X @ X.T

    hist = []
    log2pi = math.log(2.0 * math.pi)

    for t in range(steps):
        opt.zero_grad()

        sigma2 = torch.exp(log_sigma2).clamp_min(1e-12)

        if use_primal:
            A = I_d + (1.0 / (a * sigma2)) * XX
            A = A + jitter * I_d
            L, _ = torch.linalg.cholesky_ex(A)

            logdetA = 2.0 * torch.log(torch.diag(L)).sum()
            logdetC = n * torch.log(sigma2) + logdetA

            v = torch.cholesky_solve(Xy.unsqueeze(1), L).squeeze(1)
            v = (1.0 / a) * v

            Cinv_y = (y / sigma2) - (X @ v) / (sigma2 * sigma2)
            quad = y.dot(Cinv_y)

        else:
            C = sigma2 * I_n + (1.0 / a) * K
            C = C + jitter * I_n
            L, _ = torch.linalg.cholesky_ex(C)

            logdetC = 2.0 * torch.log(torch.diag(L)).sum()

            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
            quad = y.dot(alpha)

        logp = -0.5 * (logdetC + quad + n * log2pi)

        loss = -logp
        loss.backward()
        opt.step()

        hist.append(float(logp.detach().cpu()))

        if verbose and (t % max(1, steps // 10) == 0 or t == steps - 1):
            print(f"[{t:4d}] sigma={float(torch.sqrt(sigma2).detach().cpu()):.6g}, logp={hist[-1]:.6g}")

    sigma_hat = float(torch.sqrt(torch.exp(log_sigma2)).detach().cpu())
    return sigma_hat, hist

def opt_sigma_l(
    X: torch.Tensor,
    y: torch.Tensor,
    prior_precision: float,
    basis_kwargs: dict,
    basis_type: str,
    *,
    lr: float = 1e-2,
    steps: int = 500,
    init_sigma: float = 0.1,
    init_l: float = 1.,
    jitter: float = 1e-8,
    verbose: bool = False,
                        ):  
    # assume X is already standardized and y is centered

    device = X.device
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)

    n, d = X.shape
    m = basis_kwargs["m"]
    a = torch.as_tensor(prior_precision, dtype=dtype, device=device)

    log_sigma2 = torch.tensor(
        math.log(init_sigma**2), dtype=dtype, device=device, requires_grad=True
    )
    log_l = torch.tensor(
        math.log(init_l), dtype=dtype, device=device, requires_grad=True
    )
    opt = torch.optim.Adam([log_sigma2, log_l], lr=lr)

    I_n = torch.eye(n, dtype=dtype, device=device)
    I_d = torch.eye(m, dtype=dtype, device=device)

    
    use_primal = (n > m)  # primal (mxm) if n>m, dual (nxn) if n<=m

    hist = []
    log2pi = math.log(2.0 * math.pi)

    for t in range(steps):
        opt.zero_grad()

        sigma2 = torch.exp(log_sigma2).clamp_min(1e-12)
        l = torch.exp(log_l).clamp_min(1e-12)

        basis_kwargs["lengthscale"] = l

        if use_primal:
            Phi = apply_basis(X, basis_type=basis_type, **basis_kwargs)

            PP = Phi.T @ Phi
            Phiy = Phi.T @ y
            A = I_d + (1.0 / (a * sigma2)) * PP
            A = A + jitter * I_d
            L, _ = torch.linalg.cholesky_ex(A)

            logdetA = 2.0 * torch.log(torch.diag(L)).sum()
            logdetC = n * torch.log(sigma2) + logdetA

            v = torch.cholesky_solve(Phiy.unsqueeze(1), L).squeeze(1)
            v = (1.0 / a) * v

            Cinv_y = (y / sigma2) - (Phi @ v) / (sigma2 * sigma2)
            quad = y.dot(Cinv_y)

        else:
            Phi = apply_basis(X, basis_type=basis_type, **basis_kwargs)

            K = Phi @ Phi.T
            C = sigma2 * I_n + (1.0 / a) * K
            C = C + jitter * I_n
            L, _ = torch.linalg.cholesky_ex(C)

            logdetC = 2.0 * torch.log(torch.diag(L)).sum()

            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
            quad = y.dot(alpha)

        logp = -0.5 * (logdetC + quad + n * log2pi)

        loss = -logp
        loss.backward()
        opt.step()

        hist.append(float(logp.detach().cpu()))

        if verbose and (t % max(1, steps // 10) == 0 or t == steps - 1):
            print(f"[{t:4d}] sigma={float(torch.sqrt(sigma2).detach().cpu()):.6g}, l={float(l.detach().cpu()):.6g} , logp={hist[-1]:.6g}")

    sigma_hat = float(torch.sqrt(torch.exp(log_sigma2)).detach().cpu())
    l_hat = float(torch.exp(log_l).detach().cpu())
    return sigma_hat, l_hat, hist


def opt_sigma_l_alpha(
    X: torch.Tensor,
    y: torch.Tensor,
    basis_kwargs: dict,
    basis_type: str,
    *,
    lr: float = 1e-2,
    steps: int = 500,
    init_sigma: float = 0.1,
    init_l: float = 1.,
    init_prior_precision: float = 1.,
    jitter: float = 1e-8,
    verbose: bool = False,
                        ):  
    # assume X is already standardized and y is centered

    device = X.device
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)

    n, d = X.shape
    m = basis_kwargs["m"]

    log_a = torch.tensor(
        math.log(init_prior_precision), dtype=dtype, device=device, requires_grad=True
    )

    log_sigma2 = torch.tensor(
        math.log(init_sigma**2), dtype=dtype, device=device, requires_grad=True
    )
    log_l = torch.tensor(
        math.log(init_l), dtype=dtype, device=device, requires_grad=True
    )
    opt = torch.optim.Adam([
    {"params": [log_sigma2, log_l], "lr": lr},
    {"params": [log_a], "lr": lr * 0.1},
])


    I_n = torch.eye(n, dtype=dtype, device=device)
    I_d = torch.eye(m, dtype=dtype, device=device)

    
    use_primal = (n > m)  # primal (mxm) if n>m, dual (nxn) if n<=m

    hist = []
    log2pi = math.log(2.0 * math.pi)

    for t in range(steps):
        opt.zero_grad()

        sigma2 = torch.exp(log_sigma2).clamp_min(1e-12)
        l = torch.exp(log_l).clamp_min(1e-12)
        a = torch.exp(log_a).clamp_min(1e-12)

        basis_kwargs["lengthscale"] = l

        if use_primal:
            Phi = apply_basis(X, basis_type=basis_type, **basis_kwargs)

            PP = Phi.T @ Phi
            Phiy = Phi.T @ y
            A = I_d + (1.0 / (a * sigma2)) * PP
            A = A + jitter * I_d
            L, _ = torch.linalg.cholesky_ex(A)

            logdetA = 2.0 * torch.log(torch.diag(L)).sum()
            logdetC = n * torch.log(sigma2) + logdetA

            v = torch.cholesky_solve(Phiy.unsqueeze(1), L).squeeze(1)
            v = (1.0 / a) * v

            Cinv_y = (y / sigma2) - (Phi @ v) / (sigma2 * sigma2)
            quad = y.dot(Cinv_y)

        else:
            Phi = apply_basis(X, basis_type=basis_type, **basis_kwargs)

            K = Phi @ Phi.T
            C = sigma2 * I_n + (1.0 / a) * K
            C = C + jitter * I_n
            L, _ = torch.linalg.cholesky_ex(C)

            logdetC = 2.0 * torch.log(torch.diag(L)).sum()

            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
            quad = y.dot(alpha)

        logp = -0.5 * (logdetC + quad + n * log2pi)

        loss = -logp
        loss.backward()
        opt.step()

        hist.append(float(logp.detach().cpu()))

        if verbose and (t % max(1, steps // 10) == 0 or t == steps - 1):
            print(f"[{t:4d}] sigma={float(torch.sqrt(sigma2).detach().cpu()):.6g}, l={float(l.detach().cpu()):.6g}, a={float(a.detach().cpu()):.6g} , logp={hist[-1]:.6g}")

    sigma_hat = float(torch.sqrt(torch.exp(log_sigma2)).detach().cpu())
    l_hat = float(torch.exp(log_l).detach().cpu())
    a_hat = float(torch.exp(log_a).detach().cpu())
    return sigma_hat, l_hat, a_hat, hist

def pca_ood_trte_split(X, y, n_tr):

    # pick train points to be most aligned with 1st PC
    # could also use more than the first ofc

    _X = (X-X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

    _, _, V = torch.linalg.svd(_X, full_matrices=False)
    pc1 = V[0] # (d, )
    sim_score = torch.abs(_X @ pc1)

    sort_inds = torch.argsort(sim_score, descending=True)
    X = X[sort_inds]
    y = y[sort_inds]

    X_tr = X[:n_tr]
    X_te = X[n_tr:]

    y_tr = y[:n_tr]
    y_te = y[n_tr:]

    return X_tr, X_te, y_tr, y_te

def feat_ood_trte_split(X, y, n_tr):

    # make train test split based on raw features
    # randomly pick feature
    # split on this feature train_frac quantile

    n, d = X.shape

    p = torch.randint(low=0, high=d, size=(1,), device=X.device).item()
    x_p = X[:, p]
    sort_inds = torch.argsort(x_p)

    X = X[sort_inds]
    y = y[sort_inds]

    X_tr = X[:n_tr]
    X_te = X[n_tr:]

    y_tr = y[:n_tr]
    y_te = y[n_tr:]

    return X_tr, X_te, y_tr, y_te

def rand_trte_split(X, y, n_tr):

    X_tr = X[:n_tr]
    X_te = X[n_tr:]

    y_tr = y[:n_tr]
    y_te = y[n_tr:]

    return X_tr, X_te, y_tr, y_te

import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_temp_gap_violins(results, include_nll=False):
    """
    For each divergence family, create a figure with one column per dataset.
    Each dataset-column contains violin plots of:
        Δ = argmin_T(metric_test) - argmin_T(metric_train)
    computed per repetition, using log10(T).

    If a family has both marginal and joint versions, their violins are shown side-by-side.

    Parameters
    ----------
    results : dict
        The dict saved by your script (torch.save(results, ...)).
    include_nll : bool
        If True, also plots MFVI NLL and "true tempered" NLL as additional families.

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping from family name to the created figure.
    """
    datasets = results.get("meta", {}).get("datasets", list(results.get("data", {}).keys()))
    log10Ts = np.asarray(results["temps"]["log10Ts"])

    # Masks (use the same ranges you used for plotting)
    mask_kl    = np.asarray(results["temps"]["mask_kl"])
    mask_alpha = np.asarray(results["temps"]["mask_alpha"])
    mask_wass  = np.asarray(results["temps"]["mask_wass"])
    mask_diff  = np.asarray(results["temps"]["mask_diff"])
    mask_nll   = np.asarray(results["temps"]["mask_nll"])

    # Define "divergence families" and which (train/test) keys belong to each
    families = [
        ("Forward KL",     [("fwd_kls", "marg"), ("fwd_joint_kl", "joint")], mask_kl),
        ("Reverse KL",     [("rev_kls", "marg"), ("rev_joint_kl", "joint")], mask_kl),
        ("Alpha (Hellinger)", [("alpha", "marg"), ("joint_alpha", "joint")], mask_alpha),
        ("Wasserstein-2",  [("wass2", "marg"), ("joint_wass2", "joint")], mask_wass),
        ("VarDiff^2",      [("sq_diff_post_var", "marg")], mask_diff),
    ]
    if include_nll:
        families += [
            ("NLL (MFVI)",   [("mfvi_nplls", "marg")], mask_nll),
            ("NLL (True tempered)", [("true_t_npll", "marg")], mask_nll),
        ]

    def _temp_gap_per_rep(reps_te, reps_tr, mask):
        """
        reps_te/reps_tr: arrays of shape (n_reps, n_temps)
        return: array shape (n_reps,) of Δ log10T = log10T_te* - log10T_tr*
        """
        reps_te = np.asarray(reps_te)
        reps_tr = np.asarray(reps_tr)

        # Restrict to the intended temperature window for this metric
        te_m = reps_te[:, mask]
        tr_m = reps_tr[:, mask]
        log10_m = log10Ts[mask]

        # Argmin per rep
        te_idx = np.argmin(te_m, axis=1)
        tr_idx = np.argmin(tr_m, axis=1)

        te_star = log10_m[te_idx]
        tr_star = log10_m[tr_idx]
        return te_star - tr_star

    figs = {}
    n_ds = len(datasets)

    for fam_name, variants, mask in families:
        fig, axes = plt.subplots(
            1, n_ds, figsize=(2.6 * n_ds, 3.2), sharey=True, constrained_layout=True
        )
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            reps_dict = results["data"][ds]["reps"]

            deltas = []
            vlabels = []
            for base_key, vlab in variants:
                k_te = f"{base_key}_te"
                k_tr = f"{base_key}_tr"
                if (k_te not in reps_dict) or (k_tr not in reps_dict):
                    continue

                gap = _temp_gap_per_rep(reps_dict[k_te], reps_dict[k_tr], mask)
                deltas.append(gap)
                vlabels.append(vlab)

            if len(deltas) == 0:
                ax.set_title(ds)
                ax.axhline(0.0, linewidth=1)
                ax.set_xticks([])
                ax.grid(True, axis="y", alpha=0.3)
                continue

            # Positions for side-by-side violins (marg/joint)
            positions = np.arange(1, len(deltas) + 1, dtype=float)
            vp = ax.violinplot(deltas, positions=positions, widths=0.7, showmeans=True)

            ax.set_title(ds)
            ax.set_xticks(positions)
            ax.set_xticklabels(vlabels, rotation=0)
            ax.axhline(0.0, linewidth=1)
            ax.grid(True, axis="y", alpha=0.3)

        axes[0].set_ylabel(r"$\Delta \log_{10}(T) = \log_{10}(T^*_{test}) - \log_{10}(T^*_{train})$")
        fig.suptitle(fam_name)
        figs[fam_name] = fig

    return figs





