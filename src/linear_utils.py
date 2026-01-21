import torch
import numpy as np
from torch.distributions import Normal
import math

dtype = torch.float64

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


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
    Sigma = torch.linalg.inv(Precision_matrix)

    mu = Sigma @ train_X.T @ train_y / noise_std**2

    return mu, Sigma

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
    lr: float = 5e-2,
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

        sigma2 = torch.exp(log_sigma2)

        if use_primal:
            A = I_d + (1.0 / (a * sigma2)) * XX
            A = A + jitter * I_d
            L = torch.linalg.cholesky_ex(A)

            logdetA = 2.0 * torch.log(torch.diag(L)).sum()
            logdetC = n * torch.log(sigma2) + logdetA

            v = torch.cholesky_solve(Xy.unsqueeze(1), L).squeeze(1)
            v = (1.0 / a) * v

            Cinv_y = (y / sigma2) - (X @ v) / (sigma2 * sigma2)
            quad = y.dot(Cinv_y)

        else:
            C = sigma2 * I_n + (1.0 / a) * K
            C = C + jitter * I_n
            L = torch.linalg.cholesky(C)

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
