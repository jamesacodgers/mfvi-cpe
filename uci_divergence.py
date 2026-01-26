# this scripts calculates and plots various diveregences between true and T-MFVI
# posterior predictive at a range of temperatures, for a number of UCI data sets

import os
import torch
from math import log 
import matplotlib.pyplot as plt
import numpy as np

from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import compute_mfvi_analytic, compute_exact_posterior, post_pred_mean_var, opt_sigma_l_alpha, pca_ood_trte_split, rand_trte_split
from src.basis_functions import apply_basis
from src.utils import check_bad

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set up #

O_NOISE = .1; P_PRSCISN = 1 # overridden if LEARN_NOISE_L_ALPHA == True
LEARN_NOISE_L_ALPHA = False

TR_FRC = 0.7
N_MAX = 1000 # max tr + te points (subsampling, lower is faster ...)
OOD_TRTE = True

N_REPS = 2

T_RANGE_KL = (-1, .5)
T_RANGE_ALPHA = (-2, 1)
T_RANGE_WASS = (-1, .5)
T_RANGE_DIFF = (-1.5, .5)
T_RANGE_NLL = (-2, 2) # extended range to check warm posteriors

T_POINTS = 10

BASIS = "rbf" # | "identity" | "polynomial"
BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 499 }  # {} | {"degree": 5} 

# BASIS = "identity"
# BASIS_KWARGS = {}

set_seeds(42)

def divergences(X, y, Ts, n_max = 1000):

    # Ts shape (n_temps, 1)

    n, d = X.shape

    n_eff = min(n, n_max)

    # pre-process
    tt_split_ind = int(TR_FRC * n_eff)
    perm = torch.randperm(n, device=X.device)
    X = X[perm][:n_eff]
    y = y[perm][:n_eff]

    if OOD_TRTE:
        X_tr, X_te, y_tr, y_te = pca_ood_trte_split(X, y, tt_split_ind)

    else:
        X_tr, X_te, y_tr, y_te = rand_trte_split(X, y, tt_split_ind)

    m_X_tr = X_tr.mean(0)
    std_X_tr = X_tr.std(0)
    m_y_tr = y_tr.mean()
    X_tr = (X_tr - m_X_tr) / (std_X_tr + 1e-8)
    X_te = (X_te - m_X_tr) / (std_X_tr + 1e-8)

    y_tr -= m_y_tr
    y_te -= m_y_tr

    n_te = X_te.shape[0]
    n_tr = X_tr.shape[0]

    if BASIS == "rbf":
        # centers = X_tr[:BASIS_KWARGS["m"]] # choose random center points
        # BASIS_KWARGS["centers"] = centers

        # sample center from Gaussian with emprical covariance
        eps = torch.randn((BASIS_KWARGS["m"], d), dtype=dtype, device=X.device)
        X_trX_tr = X_tr.T @ X_tr / n_tr
        L_tr, _ = torch.linalg.cholesky_ex(X_trX_tr + 1e-6 * torch.eye(d).to(X))
        centers = L_tr[None, ...] @ eps[..., None] 
        BASIS_KWARGS["centers"] = centers.squeeze(-1)

    

    noise_std = O_NOISE
    prior_precision = P_PRSCISN
    if LEARN_NOISE_L_ALPHA:
        sigma, l, alpha,  _ = opt_sigma_l_alpha(X_tr, y_tr, basis_kwargs=BASIS_KWARGS, basis_type=BASIS,
                                   init_sigma=0.1, init_l=1, init_prior_precision=0.5,
                                   verbose=True, steps=2000)
        noise_std = sigma
        BASIS_KWARGS["lengthscale"] = l
        prior_precision = alpha

    X_tr = apply_basis(X=X_tr, basis_type=BASIS, **BASIS_KWARGS); print("train", X_tr.shape)
    X_te = apply_basis(X=X_te, basis_type=BASIS, **BASIS_KWARGS); print("test", X_te.shape)

    mu, Sigma = compute_exact_posterior(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=prior_precision)

    m_opt, S_opt = compute_mfvi_analytic(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=prior_precision)

    true_post_mean_te, true_post_var_te = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std)
    
    mfvi_post_mean_te, mfvi_post_var_te = post_pred_mean_var(test_X=X_te, post_mean=m_opt, post_cov=torch.diag(S_opt), 
                                                       noise_std=noise_std, temp=Ts)
    
    true_post_mean_tr, true_post_var_tr = post_pred_mean_var(test_X=X_tr, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std)
    
    mfvi_post_mean_tr, mfvi_post_var_tr = post_pred_mean_var(test_X=X_tr, post_mean=m_opt, post_cov=torch.diag(S_opt), 
                                                       noise_std=noise_std, temp=Ts)

    metrix = {}

    # marginal kls
    fwd_kls_te = 0.5 * torch.log(mfvi_post_var_te / true_post_var_te) + 0.5 * true_post_var_te / mfvi_post_var_te - 0.5
    rev_kls_te = 0.5 * torch.log(true_post_var_te / mfvi_post_var_te) + 0.5 * mfvi_post_var_te / true_post_var_te - 0.5
    fwd_kls_tr = 0.5 * torch.log(mfvi_post_var_tr / true_post_var_tr) + 0.5 * true_post_var_tr / mfvi_post_var_tr - 0.5
    rev_kls_tr = 0.5 * torch.log(true_post_var_tr / mfvi_post_var_tr) + 0.5 * mfvi_post_var_tr / true_post_var_tr - 0.5

    metrix["fwd_kls_te"] = check_bad(torch.mean(fwd_kls_te, dim=0)).detach().cpu().numpy()
    metrix["rev_kls_te"] = check_bad(torch.mean(rev_kls_te, dim=0)).detach().cpu().numpy()
    metrix["fwd_kls_tr"] = check_bad(torch.mean(fwd_kls_tr, dim=0)).detach().cpu().numpy()
    metrix["rev_kls_tr"] = check_bad(torch.mean(rev_kls_tr, dim=0)).detach().cpu().numpy()

    # joint kl
    sig_true_te = X_te @ Sigma @ X_te.T + noise_std**2 * torch.eye(n_te).to(X)
    sig_mfvi_te = Ts[..., None] * (X_te @ torch.diag(S_opt) @ X_te.T)[None, ...] + noise_std**2 * torch.eye(n_te)[None, ...].to(X)
    sig_true_tr = X_tr @ Sigma @ X_tr.T + noise_std**2 * torch.eye(n_tr).to(X)
    sig_mfvi_tr = Ts[..., None] * (X_tr @ torch.diag(S_opt) @ X_tr.T)[None, ...] + noise_std**2 * torch.eye(n_tr)[None, ...].to(X)

    # log det A = 2 * log det L(A), det tri is prod of diag
    sig_true_te_L = torch.linalg.cholesky(sig_true_te); sig_true_te_logdet = 2.0 * torch.log(torch.diagonal(sig_true_te_L, dim1=-2, dim2=-1)).sum(-1)
    sig_true_tr_L = torch.linalg.cholesky(sig_true_tr); sig_true_tr_logdet = 2.0 * torch.log(torch.diagonal(sig_true_tr_L, dim1=-2, dim2=-1)).sum(-1)

    sig_mfvi_te_L = torch.linalg.cholesky(sig_mfvi_te); sig_mfvi_te_logdet = 2.0 * torch.log(torch.diagonal(sig_mfvi_te_L, dim1=-2, dim2=-1)).sum(-1)
    sig_mfvi_tr_L = torch.linalg.cholesky(sig_mfvi_tr); sig_mfvi_tr_logdet = 2.0 * torch.log(torch.diagonal(sig_mfvi_tr_L, dim1=-2, dim2=-1)).sum(-1)

    # LL'X=B, solve for X = arg(tr)
    fwd_joint_kl_te = 0.5 * (torch.diagonal(torch.cholesky_solve(sig_true_te[None, ...], sig_mfvi_te_L), dim1=-2, dim2=-1).sum(-1) - n_te + (sig_mfvi_te_logdet - sig_true_te_logdet))
    rev_joint_kl_te = 0.5 * (torch.diagonal(torch.cholesky_solve(sig_mfvi_te, sig_true_te_L[None, ...]), dim1=-2, dim2=-1).sum(-1) - n_te + (sig_true_te_logdet - sig_mfvi_te_logdet))

    fwd_joint_kl_tr = 0.5 * (torch.diagonal(torch.cholesky_solve(sig_true_tr[None, ...], sig_mfvi_tr_L), dim1=-2, dim2=-1).sum(-1) - n_tr + (sig_mfvi_tr_logdet - sig_true_tr_logdet))
    rev_joint_kl_tr = 0.5 * (torch.diagonal(torch.cholesky_solve(sig_mfvi_tr, sig_true_tr_L[None, ...]), dim1=-2, dim2=-1).sum(-1) - n_tr + (sig_true_tr_logdet - sig_mfvi_tr_logdet))

    metrix["fwd_joint_kl_te"] = check_bad(fwd_joint_kl_te).detach().cpu().numpy()
    metrix["rev_joint_kl_te"] = check_bad(rev_joint_kl_te).detach().cpu().numpy()
    metrix["fwd_joint_kl_tr"] = check_bad(fwd_joint_kl_tr).detach().cpu().numpy()
    metrix["rev_joint_kl_tr"] = check_bad(rev_joint_kl_tr).detach().cpu().numpy()


    # alpha div @ α = 0.5 (= 4 sq Hellinger and symmetric) 

    # marginals

    hell_sq_tr = 1 - (true_post_var_tr * mfvi_post_var_tr) ** 0.25 / ((true_post_var_tr + mfvi_post_var_tr) / 2) ** 0.5
    hell_sq_te = 1 - (true_post_var_te * mfvi_post_var_te) ** 0.25 / ((true_post_var_te + mfvi_post_var_te) / 2) ** 0.5

    metrix["alpha_tr"] = check_bad(torch.mean(4 * hell_sq_tr, dim=0)).detach().cpu().numpy()
    metrix["alpha_te"] = check_bad(torch.mean(4 * hell_sq_te, dim=0)).detach().cpu().numpy()

    # hybrid marginally (do I want this or do I actually want the average accross data, not joint indep?)

    # marg_sig_true_te_logdet = torch.log(true_post_var_te).sum(dim=0)
    # marg_sig_true_tr_logdet = torch.log(true_post_var_tr).sum(dim=0)
    # marg_sig_mfvi_te_logdet = torch.log(mfvi_post_var_te).sum(dim=0)
    # marg_sig_mfvi_tr_logdet = torch.log(mfvi_post_var_tr).sum(dim=0)

    # marg_sig_sum_tr_logdet = torch.log((true_post_var_tr + mfvi_post_var_tr) / 2).sum(dim=0)
    # marg_sig_sum_te_logdet = torch.log((true_post_var_te + mfvi_post_var_te) / 2).sum(dim=0)

    # hell_sq_tr = 1 - torch.exp(0.25 * (marg_sig_true_tr_logdet + marg_sig_mfvi_tr_logdet) - 0.5 * marg_sig_sum_tr_logdet)
    # hell_sq_te = 1 - torch.exp(0.25 * (marg_sig_true_te_logdet + marg_sig_mfvi_te_logdet) - 0.5 * marg_sig_sum_te_logdet)

    # metrix["alpha_tr"] = check_bad(4 * hell_sq_tr).detach().cpu().numpy()
    # metrix["alpha_te"] = check_bad(4 * hell_sq_te).detach().cpu().numpy()

    # jointly (can re-use dets from KLs) 

    sig_sum_tr_L = torch.linalg.cholesky((sig_true_tr + sig_mfvi_tr) / 2); sig_sum_tr_logdet = 2 * torch.log(torch.diagonal(sig_sum_tr_L, dim1=1, dim2=2)).sum(dim=-1)
    sig_sum_te_L = torch.linalg.cholesky((sig_true_te + sig_mfvi_te) / 2); sig_sum_te_logdet = 2 * torch.log(torch.diagonal(sig_sum_te_L, dim1=1, dim2=2)).sum(dim=-1)

    joint_hell_sq_tr = 1 - torch.exp(0.25 * (sig_true_tr_logdet + sig_mfvi_tr_logdet) - 0.5 * sig_sum_tr_logdet)
    joint_hell_sq_te = 1 - torch.exp(0.25 * (sig_true_te_logdet + sig_mfvi_te_logdet) - 0.5 * sig_sum_te_logdet)

    metrix["joint_alpha_tr"] = check_bad(4 * joint_hell_sq_tr).detach().cpu().numpy()
    metrix["joint_alpha_te"] = check_bad(4 * joint_hell_sq_te).detach().cpu().numpy()

    # 2-Wasserstein

    # marginals (avgerage)

    wass2_tr = true_post_var_tr + mfvi_post_var_tr - 2 * (true_post_var_tr * mfvi_post_var_tr) ** 0.5
    wass2_te = true_post_var_te + mfvi_post_var_te - 2 * (true_post_var_te * mfvi_post_var_te) ** 0.5

    metrix["wass2_tr"] = check_bad(torch.mean(wass2_tr, dim=0)).detach().cpu().numpy()
    metrix["wass2_te"] = check_bad(torch.mean(wass2_te, dim=0)).detach().cpu().numpy()

    # joint

    D, V = torch.linalg.eigh(sig_true_te)
    sig_true_te_sqrt = V @ torch.diag_embed(D).sqrt() @ torch.transpose(V, -1, -2)
    D, V = torch.linalg.eigh(sig_true_tr)
    sig_true_tr_sqrt = V @ torch.diag_embed(D).sqrt() @ torch.transpose(V, -1, -2)


    A = sig_true_te_sqrt @ sig_mfvi_te @ sig_true_te_sqrt
    D, V = torch.linalg.eigh(A)
    A_sqrt = V @ torch.diag_embed(D).sqrt() @ torch.transpose(V, -1, -2)
    joint_wass2_te = torch.diagonal(sig_true_te + sig_mfvi_te - 2 *  A_sqrt, dim1=1, dim2=2).sum(dim=-1)

    A = sig_true_tr_sqrt @ sig_mfvi_tr @ sig_true_tr_sqrt
    D, V = torch.linalg.eigh(A)
    A_sqrt = V @ torch.diag_embed(D).sqrt() @ torch.transpose(V, -1, -2)
    joint_wass2_tr = torch.diagonal(sig_true_tr + sig_mfvi_tr - 2 *  A_sqrt, dim1=1, dim2=2).sum(dim=-1)

    metrix["joint_wass2_te"] = check_bad(joint_wass2_te).detach().cpu().numpy()
    metrix["joint_wass2_tr"] = check_bad(joint_wass2_tr).detach().cpu().numpy()

    # mean predictive variance

    diff_post_var_te = true_post_var_te.mean(dim=0) - mfvi_post_var_te.mean(dim=0)
    diff_post_var_tr = true_post_var_tr.mean(dim=0) - mfvi_post_var_tr.mean(dim=0)

    metrix["sq_diff_post_var_te"] = check_bad(diff_post_var_te**2).detach().cpu().numpy()
    metrix["sq_diff_post_var_tr"] = check_bad(diff_post_var_tr**2).detach().cpu().numpy()


    # NLL (negative predictive log-likelihood)

    # temper true post for these plots
    true_t_post_mean_te, true_t_post_var_te = post_pred_mean_var(test_X=X_te, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std, temp=Ts)
    true_t_post_mean_tr, true_t_post_var_tr = post_pred_mean_var(test_X=X_tr, post_mean=mu, post_cov=Sigma, 
                                                       noise_std=noise_std, temp=Ts)

    true_t_npll_te = log(2*torch.pi) / 2 + \
                    torch.log(true_t_post_var_te) / 2 + \
                    (y_te - true_t_post_mean_te)[:, None]**2 / true_t_post_var_te / 2
    true_t_npll_tr = log(2*torch.pi) / 2 + \
                    torch.log(true_t_post_var_tr) / 2 + \
                    (y_tr - true_t_post_mean_tr)[:, None]**2 / true_t_post_var_tr / 2
    
    mfvi_nplls_te = log(2*torch.pi) / 2 + \
                    torch.log(mfvi_post_var_te) / 2 + \
                    (y_te - mfvi_post_mean_te)[:, None]**2 / mfvi_post_var_te / 2
    mfvi_nplls_tr = log(2*torch.pi) / 2 + \
                    torch.log(mfvi_post_var_tr) / 2 + \
                    (y_tr - mfvi_post_mean_tr)[:, None]**2 / mfvi_post_var_tr / 2
    
    metrix["true_t_npll_te"] = check_bad(true_t_npll_te.mean(dim=0)).detach().cpu().numpy()
    metrix["true_t_npll_tr"] = check_bad(true_t_npll_tr.mean(dim=0)).detach().cpu().numpy()

    metrix["mfvi_nplls_te"] = check_bad(mfvi_nplls_te.mean(dim=0)).detach().cpu().numpy()
    metrix["mfvi_nplls_tr"] = check_bad(mfvi_nplls_tr.mean(dim=0)).detach().cpu().numpy()

    return metrix
    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

if __name__ == "__main__":

    from tueplots.bundles import icml2024
    plt.rcParams.update(icml2024(nrows=3, ncols=3))

    # for now until latex works on cluster
    plt.rcParams.update({
        "text.usetex": False,        
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    tmin = min(T_RANGE_KL[0], T_RANGE_ALPHA[0], T_RANGE_WASS[0], T_RANGE_DIFF[0], T_RANGE_NLL[0])
    tmax = max(T_RANGE_KL[1], T_RANGE_ALPHA[1], T_RANGE_WASS[1], T_RANGE_DIFF[1], T_RANGE_NLL[1])

    Ts = 10 ** torch.linspace(tmin, tmax, T_POINTS, dtype=dtype, device=device)
    log10Ts = torch.log10(Ts)

    if BASIS == "rbf":
        p = BASIS_KWARGS["m"]
        l = BASIS_KWARGS["lengthscale"] if not LEARN_NOISE_L_ALPHA else "learned"

    if BASIS == "polynomial":
        deg = BASIS_KWARGS["degree"]

    # color structure: te one color, tr one color (everywhere)
    COL_TE = "tab:blue"
    COL_TR = "tab:orange"

    # std line style
    STD_LS = ":"
    STD_ALPHA = 0.6

    fig_kl_fwd, axes_kl_fwd = plt.subplots(3, 3,
                                          figsize=(11, 11),
                                          constrained_layout=True)
    fig_kl_rev, axes_kl_rev = plt.subplots(3, 3,
                                          figsize=(11, 11),
                                          constrained_layout=True)
    fig_alpha, axes_mid = plt.subplots(3, 3,
                                       figsize=(11, 11),
                                       constrained_layout=True)
    fig_wass, axes_bot = plt.subplots(3, 3,
                                      figsize=(11, 11),
                                      constrained_layout=True)
    fig_diff, axes_last = plt.subplots(3, 3,
                                       figsize=(11, 11),
                                       constrained_layout=True)
    fig_nll, axes_nll = plt.subplots(3, 3,
                                     figsize=(11, 11),
                                     constrained_layout=True)

    axes_kl_fwd = axes_kl_fwd.ravel()
    axes_kl_rev = axes_kl_rev.ravel()
    axes_mid = axes_mid.ravel()
    axes_bot = axes_bot.ravel()
    axes_last = axes_last.ravel()
    axes_nll = axes_nll.ravel()

    log10Ts_np = log10Ts.detach().cpu().numpy()

    mask_kl = (log10Ts_np >= T_RANGE_KL[0]) & (log10Ts_np <= T_RANGE_KL[1])
    mask_alpha = (log10Ts_np >= T_RANGE_ALPHA[0]) & (log10Ts_np <= T_RANGE_ALPHA[1])
    mask_wass = (log10Ts_np >= T_RANGE_WASS[0]) & (log10Ts_np <= T_RANGE_WASS[1])
    mask_diff = (log10Ts_np >= T_RANGE_DIFF[0]) & (log10Ts_np <= T_RANGE_DIFF[1])
    mask_nll = (log10Ts_np >= T_RANGE_NLL[0]) & (log10Ts_np <= T_RANGE_NLL[1])

    x_kl = log10Ts_np[mask_kl]
    x_alpha = log10Ts_np[mask_alpha]
    x_wass = log10Ts_np[mask_wass]
    x_diff = log10Ts_np[mask_diff]
    x_nll = log10Ts_np[mask_nll]

    ax2_first_fwd = None
    ax2_first_rev = None
    ax4_first = None
    ax6_first = None

    for axf, axr, axb, axc, axd, axn, dataset in zip(
            axes_kl_fwd, axes_kl_rev, axes_mid, axes_bot, axes_last, axes_nll, datasets):
        print("="*50)
        print(dataset.capitalize())
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype, device=device)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype, device=device)

        keys = [
            "fwd_kls_te", "rev_kls_te", "fwd_kls_tr", "rev_kls_tr",
            "fwd_joint_kl_te", "rev_joint_kl_te", "fwd_joint_kl_tr", "rev_joint_kl_tr",
            "alpha_te", "alpha_tr", "joint_alpha_te", "joint_alpha_tr",
            "wass2_te", "wass2_tr", "joint_wass2_te", "joint_wass2_tr",
            "sq_diff_post_var_te", "sq_diff_post_var_tr",
            "true_t_npll_te", "true_t_npll_tr",
            "mfvi_nplls_te", "mfvi_nplls_tr",
        ]

        reps = {k: [] for k in keys}
        for _ in range(N_REPS):
            met = divergences(X, y, Ts[:, None], N_MAX)
            for k in keys:
                reps[k].append(met[k])

        metrix = {}
        metrix2 = {}
        for k in keys:
            arr = np.stack(reps[k], axis=0)
            metrix[k] = arr.mean(axis=0)
            metrix2[k] = 2.0 * arr.std(axis=0)

        # ---------- KL (forward) ----------
        for k, lab, col in [("fwd_kls_te", "marg fwd KL te", COL_TE),
                            ("fwd_kls_tr", "marg fwd KL tr", COL_TR)]:
            y0 = metrix[k][mask_kl]
            s0 = metrix2[k][mask_kl]
            ln = axf.plot(x_kl, y0, label=lab, linestyle="--", color=col)[0]
            axf.plot(x_kl, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axf.plot(x_kl, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axf.axvline(x_kl[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axf2 = axf.twinx()
        if ax2_first_fwd is None:
            ax2_first_fwd = axf2

        for k, lab, col in [("fwd_joint_kl_te", "joint fwd KL te", COL_TE),
                            ("fwd_joint_kl_tr", "joint fwd KL tr", COL_TR)]:
            y0 = metrix[k][mask_kl]
            s0 = metrix2[k][mask_kl]
            ln = axf2.plot(x_kl, y0, label=lab, color=col)[0]
            axf2.plot(x_kl, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axf2.plot(x_kl, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axf2.axvline(x_kl[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axf.set_title(dataset)
        axf.set_xlabel(r"$\log_{10}(T)$")
        axf.set_ylabel("Marginal KL")
        axf.grid(True, alpha=0.3)
        axf2.set_ylabel("Joint KL")

        # ---------- KL (reverse) ----------
        for k, lab, col in [("rev_kls_te", "marg rev KL te", COL_TE),
                            ("rev_kls_tr", "marg rev KL tr", COL_TR)]:
            y0 = metrix[k][mask_kl]
            s0 = metrix2[k][mask_kl]
            ln = axr.plot(x_kl, y0, label=lab, linestyle="--", color=col)[0]
            axr.plot(x_kl, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axr.plot(x_kl, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axr.axvline(x_kl[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axr2 = axr.twinx()
        if ax2_first_rev is None:
            ax2_first_rev = axr2

        for k, lab, col in [("rev_joint_kl_te", "joint rev KL te", COL_TE),
                            ("rev_joint_kl_tr", "joint rev KL tr", COL_TR)]:
            y0 = metrix[k][mask_kl]
            s0 = metrix2[k][mask_kl]
            ln = axr2.plot(x_kl, y0, label=lab, color=col)[0]
            axr2.plot(x_kl, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axr2.plot(x_kl, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axr2.axvline(x_kl[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axr.set_title(dataset)
        axr.set_xlabel(r"$\log_{10}(T)$")
        axr.set_ylabel("Marginal KL")
        axr.grid(True, alpha=0.3)
        axr2.set_ylabel("Joint KL")

        # ---------- Alpha ----------
        for k, lab, col in [("alpha_te", "marg alpha te", COL_TE),
                            ("alpha_tr", "marg alpha tr", COL_TR)]:
            y0 = metrix[k][mask_alpha]
            s0 = metrix2[k][mask_alpha]
            ln = axb.plot(x_alpha, y0, label=lab, linestyle="--", color=col)[0]
            axb.plot(x_alpha, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axb.plot(x_alpha, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axb.axvline(x_alpha[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axb2 = axb.twinx()
        if ax4_first is None:
            ax4_first = axb2

        for k, lab, col in [("joint_alpha_te", "joint alpha te", COL_TE),
                            ("joint_alpha_tr", "joint alpha tr", COL_TR)]:
            y0 = metrix[k][mask_alpha]
            s0 = metrix2[k][mask_alpha]
            ln = axb2.plot(x_alpha, y0, label=lab, color=col)[0]
            axb2.plot(x_alpha, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axb2.plot(x_alpha, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axb2.axvline(x_alpha[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axb.set_title(dataset)
        axb.set_xlabel(r"$\log_{10}(T)$")
        axb.set_ylabel("Marginal alpha")
        axb.grid(True, alpha=0.3)
        axb2.set_ylabel("Joint alpha")

        # ---------- Wasserstein ----------
        for k, lab, col in [("wass2_te", "marg wass2 te", COL_TE),
                            ("wass2_tr", "marg wass2 tr", COL_TR)]:
            y0 = metrix[k][mask_wass]
            s0 = metrix2[k][mask_wass]
            ln = axc.plot(x_wass, y0, label=lab, linestyle="--", color=col)[0]
            axc.plot(x_wass, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axc.plot(x_wass, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axc.axvline(x_wass[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axc2 = axc.twinx()
        if ax6_first is None:
            ax6_first = axc2

        for k, lab, col in [("joint_wass2_te", "joint wass2 te", COL_TE),
                            ("joint_wass2_tr", "joint wass2 tr", COL_TR)]:
            y0 = metrix[k][mask_wass]
            s0 = metrix2[k][mask_wass]
            ln = axc2.plot(x_wass, y0, label=lab, color=col)[0]
            axc2.plot(x_wass, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axc2.plot(x_wass, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axc2.axvline(x_wass[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axc.set_title(dataset)
        axc.set_xlabel(r"$\log_{10}(T)$")
        axc.set_ylabel("Marginal wass2")
        axc.grid(True, alpha=0.3)
        axc2.set_ylabel("Joint wass2")

        # ---------- Diff ----------
        for k, lab, col in [("sq_diff_post_var_te", "diff² post var te", COL_TE),
                            ("sq_diff_post_var_tr", "diff² post var tr", COL_TR)]:
            y0 = metrix[k][mask_diff]
            s0 = metrix2[k][mask_diff]
            ln = axd.plot(x_diff, y0, label=lab, linestyle="--", color=col)[0]
            axd.plot(x_diff, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axd.plot(x_diff, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axd.axvline(x_diff[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axd.set_title(dataset)
        axd.set_xlabel(r"$\log_{10}(T)$")
        axd.set_ylabel("Diff² mean post var")
        axd.grid(True, alpha=0.3)

        # ---------- NLL ----------
        for k_true, k_mfvi, lab_true, lab_mfvi, col in [
            ("true_t_npll_te", "mfvi_nplls_te", "true T-NLL te", "mfvi T-NLL te", COL_TE),
            ("true_t_npll_tr", "mfvi_nplls_tr", "true T-NLL tr", "mfvi T-NLL tr", COL_TR),
        ]:
            y0 = metrix[k_true][mask_nll]
            s0 = metrix2[k_true][mask_nll]
            ln = axn.plot(x_nll, y0, label=lab_true, linestyle="-", color=col)[0]
            axn.plot(x_nll, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axn.plot(x_nll, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axn.axvline(x_nll[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

            y0 = metrix[k_mfvi][mask_nll]
            s0 = metrix2[k_mfvi][mask_nll]
            ln = axn.plot(x_nll, y0, label=lab_mfvi, linestyle="--", color=col)[0]
            axn.plot(x_nll, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axn.plot(x_nll, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
            axn.axvline(x_nll[np.argmin(y0)], color=ln.get_color(), alpha=0.4)

        axn.set_title(dataset)
        axn.set_xlabel(r"$\log_{10}(T)$")
        axn.set_ylabel("NLL")
        axn.grid(True, alpha=0.3)

    # ---------- legends ----------
    handles1, labels1 = axes_kl_fwd[0].get_legend_handles_labels()
    handles2, labels2 = ax2_first_fwd.get_legend_handles_labels()
    handles1r, labels1r = axes_kl_rev[0].get_legend_handles_labels()
    handles2r, labels2r = ax2_first_rev.get_legend_handles_labels()

    handles3, labels3 = axes_mid[0].get_legend_handles_labels()
    handles4, labels4 = ax4_first.get_legend_handles_labels()
    handles5, labels5 = axes_bot[0].get_legend_handles_labels()
    handles6, labels6 = ax6_first.get_legend_handles_labels()
    handles7, labels7 = axes_last[0].get_legend_handles_labels()
    handles8, labels8 = axes_nll[0].get_legend_handles_labels()


    fig_kl_fwd.legend(handles1 + handles2,
                      labels1 + labels2,
                      loc="outside lower center",
                      ncol=4,
                      frameon=False)

    fig_kl_rev.legend(handles1r + handles2r,
                      labels1r + labels2r,
                      loc="outside lower center",
                      ncol=4,
                      frameon=False)

    fig_alpha.legend(handles3 + handles4,
                     labels3 + labels4,
                     loc="outside lower center",
                     ncol=4,
                     frameon=False)

    fig_wass.legend(handles5 + handles6,
                    labels5 + labels6,
                    loc="outside lower center",
                    ncol=4,
                    frameon=False)

    fig_diff.legend(handles7,
                    labels7,
                    loc="outside lower center",
                    ncol=4,
                    frameon=False)

    fig_nll.legend(handles8,
                   labels8,
                   loc="outside lower center",
                   ncol=4,
                   frameon=False)

    fig_kl_fwd.suptitle(f"basis: {BASIS} | fwd KL | OOD test == {OOD_TRTE}")
    fig_kl_rev.suptitle(f"basis: {BASIS} | rev KL| OOD test == {OOD_TRTE}")
    fig_alpha.suptitle(f"basis: {BASIS} | alpha| OOD test == {OOD_TRTE}")
    fig_wass.suptitle(f"basis: {BASIS} | wass2| OOD test == {OOD_TRTE}")
    fig_diff.suptitle(f"basis: {BASIS} | var diff| OOD test == {OOD_TRTE}")
    fig_nll.suptitle(f"basis: {BASIS} | NLL| OOD test == {OOD_TRTE}")

    if BASIS == "rbf":
        base = f"figs/uci/uci_divs_{BASIS}_p={p}_l={l}"
    elif BASIS == "polynomial":
        base = f"figs/uci/uci_divs_{BASIS}_deg={deg}"
    else:
        base = f"figs/uci/uci_divs_{BASIS}"

    outpath_kl_fwd = base + "_kl_fwd.pdf"
    outpath_kl_rev = base + "_kl_rev.pdf"
    outpath_alpha = base + "_alpha.pdf"
    outpath_wass = base + "_wass.pdf"
    outpath_diff = base + "_diff.pdf"
    outpath_nll = base + "_nll.pdf"

    os.makedirs("figs/uci/", exist_ok=True)
    fig_kl_fwd.savefig(outpath_kl_fwd, bbox_inches="tight")
    fig_kl_rev.savefig(outpath_kl_rev, bbox_inches="tight")
    fig_alpha.savefig(outpath_alpha, bbox_inches="tight")
    fig_wass.savefig(outpath_wass, bbox_inches="tight")
    fig_diff.savefig(outpath_diff, bbox_inches="tight")
    fig_nll.savefig(outpath_nll, bbox_inches="tight")

    print(f"Saved figure to {outpath_kl_fwd}")
    print(f"Saved figure to {outpath_kl_rev}")
    print(f"Saved figure to {outpath_alpha}")
    print(f"Saved figure to {outpath_wass}")
    print(f"Saved figure to {outpath_diff}")
    print(f"Saved figure to {outpath_nll}")



