# this scripts calculates and plots various diveregences between true and T-MFVI
# posterior predictive at a range of temperatures, for a number of UCI data sets

import os
import torch
from math import log 
import matplotlib.pyplot as plt
import numpy as np

from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import (compute_mfvi_analytic, 
                              compute_exact_posterior, 
                              post_pred_mean_var, 
                              opt_sigma_l_alpha,
                              tr_idte_oodte_feat

)
from src.basis_functions import apply_basis
from src.utils import check_bad
import copy

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set up #
FOLDER = "results2"

O_NOISE = .1; P_PRSCISN = 1 # overridden if LEARN_NOISE_L_ALPHA == True
LEARN_NOISE_L_ALPHA = True

TR_FRC = 0.6
ID_FRC = 0.2
N_MAX = 1000 # max tr + te points (subsampling, lower is faster ...)
PCA = False # OOD split by pca or feature 

N_REPS = 20

T_RANGE_KL = (-1, .5)
T_RANGE_ALPHA = (-2, 1)
T_RANGE_WASS = (-1, .5)
T_RANGE_DIFF = (-1.5, .5)
T_RANGE_NLL = (-3, 2) # extended range to check warm posteriors

T_POINTS = 50

BASIS = "rbf" # | "identity" | "polynomial"
BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 500 }  # {} | {"degree": 5} 

# BASIS = "identity"
# BASIS_KWARGS = {}

set_seeds(42)

def divergences(X, y, Ts, n_max=1000):
    """
    Returns:
      metrix: dict of metrics with suffixes _tr, _id, _ood
      hypers: dict of shared hypers used for all three splits
    """

    n, d = X.shape
    n_eff = min(n, n_max)

    # subsample + permute once
    perm = torch.randperm(n, device=X.device)
    X = X[perm][:n_eff]
    y = y[perm][:n_eff]

    X_tr, y_tr, X_id, y_id, X_ood, y_ood = tr_idte_oodte_feat(X, y, tr_frac=TR_FRC, id_frac=ID_FRC) 

    print("train: ", X_tr.shape, "\n id test: ", X_id.shape, "\n ood test: ", X_ood.shape)

    m_X_tr = X_tr.mean(0)
    std_X_tr = X_tr.std(0)
    m_y_tr = y_tr.mean()

    def standardize(Xs, ys):
        Xs = (Xs - m_X_tr) / (std_X_tr + 1e-8)
        ys = ys - m_y_tr
        return Xs, ys

    X_tr, y_tr = standardize(X_tr, y_tr)
    X_id, y_id = standardize(X_id, y_id)
    X_ood, y_ood = standardize(X_ood, y_ood)

    n_tr = X_tr.shape[0]
    n_id = X_id.shape[0]
    n_ood = X_ood.shape[0]

    basis_kwargs = copy.deepcopy(BASIS_KWARGS)

    if BASIS == "rbf":
        eps = torch.randn((basis_kwargs["m"], d), dtype=dtype, device=X.device)
        X_trX_tr = X_tr.T @ X_tr / n_tr
        L_tr, _ = torch.linalg.cholesky_ex(X_trX_tr + 1e-6 * torch.eye(d, device=X.device, dtype=dtype))
        centers = (L_tr[None, ...] @ eps[..., None]).squeeze(-1)  # (m, d)
        basis_kwargs["centers"] = centers

    noise_std = O_NOISE
    prior_precision = P_PRSCISN

    if LEARN_NOISE_L_ALPHA:
        sigma, l, alpha, _ = opt_sigma_l_alpha(
            X_tr, y_tr,
            basis_kwargs=basis_kwargs,
            basis_type=BASIS,
            init_sigma=0.1, init_l=1, init_prior_precision=0.5,
            verbose=True, steps=2000
        )
        noise_std = sigma
        basis_kwargs["lengthscale"] = l
        prior_precision = alpha

    Phi_tr  = apply_basis(X=X_tr,  basis_type=BASIS, **basis_kwargs)
    Phi_id  = apply_basis(X=X_id,  basis_type=BASIS, **basis_kwargs)
    Phi_ood = apply_basis(X=X_ood, basis_type=BASIS, **basis_kwargs)

    mu, Sigma = compute_exact_posterior(
        train_X=Phi_tr, train_y=y_tr,
        noise_std=noise_std, prior_precision=prior_precision
    )
    m_opt, S_opt = compute_mfvi_analytic(
        train_X=Phi_tr, train_y=y_tr,
        noise_std=noise_std, prior_precision=prior_precision
    )
    S_mfvi = torch.diag(S_opt)

    # compute all metrics for one split given (Phi, y)
    def metrics_for_split(Phi, y_split):
        n_s = Phi.shape[0]

        # predictive mean/var (true and MFVI)
        true_mean, true_var = post_pred_mean_var(
            test_X=Phi, post_mean=mu, post_cov=Sigma, noise_std=noise_std
        )
        mfvi_mean, mfvi_var = post_pred_mean_var(
            test_X=Phi, post_mean=m_opt, post_cov=S_mfvi, noise_std=noise_std, temp=Ts
        )

        out = {}

        # marginal KLs
        fwd_kls = 0.5 * torch.log(mfvi_var / true_var) + 0.5 * true_var / mfvi_var - 0.5
        rev_kls = 0.5 * torch.log(true_var / mfvi_var) + 0.5 * mfvi_var / true_var - 0.5
        out["fwd_kls"] = check_bad(torch.mean(fwd_kls, dim=0)).detach().cpu().numpy()
        out["rev_kls"] = check_bad(torch.mean(rev_kls, dim=0)).detach().cpu().numpy()

        # joint KLs
        sig_true = Phi @ Sigma @ Phi.T + noise_std**2 * torch.eye(n_s, device=Phi.device, dtype=Phi.dtype)
        sig_mfvi = Ts[..., None] * (Phi @ S_mfvi @ Phi.T)[None, ...] + noise_std**2 * torch.eye(n_s, device=Phi.device, dtype=Phi.dtype)[None, ...]

        sig_true_L = torch.linalg.cholesky(sig_true)
        sig_true_logdet = 2.0 * torch.log(torch.diagonal(sig_true_L, dim1=-2, dim2=-1)).sum(-1)

        sig_mfvi_L = torch.linalg.cholesky(sig_mfvi)
        sig_mfvi_logdet = 2.0 * torch.log(torch.diagonal(sig_mfvi_L, dim1=-2, dim2=-1)).sum(-1)

        fwd_joint_kl = 0.5 * (
            torch.diagonal(torch.cholesky_solve(sig_true[None, ...], sig_mfvi_L), dim1=-2, dim2=-1).sum(-1)
            - n_s
            + (sig_mfvi_logdet - sig_true_logdet)
        )
        rev_joint_kl = 0.5 * (
            torch.diagonal(torch.cholesky_solve(sig_mfvi, sig_true_L[None, ...]), dim1=-2, dim2=-1).sum(-1)
            - n_s
            + (sig_true_logdet - sig_mfvi_logdet)
        )

        out["fwd_joint_kl"] = check_bad(fwd_joint_kl).detach().cpu().numpy()
        out["rev_joint_kl"] = check_bad(rev_joint_kl).detach().cpu().numpy()

        # alpha (Hellinger @ alpha=0.5) marginal
        hell_sq = 1 - (true_var * mfvi_var) ** 0.25 / ((true_var + mfvi_var) / 2) ** 0.5
        out["alpha"] = check_bad(torch.mean(4 * hell_sq, dim=0)).detach().cpu().numpy()

        # alpha joint
        sig_sum_L = torch.linalg.cholesky((sig_true + sig_mfvi) / 2)
        sig_sum_logdet = 2.0 * torch.log(torch.diagonal(sig_sum_L, dim1=1, dim2=2)).sum(dim=-1)

        joint_hell_sq = 1 - torch.exp(0.25 * (sig_true_logdet + sig_mfvi_logdet) - 0.5 * sig_sum_logdet)
        out["joint_alpha"] = check_bad(4 * joint_hell_sq).detach().cpu().numpy()

        # wasserstein-2 marginal
        wass2 = true_var + mfvi_var - 2 * (true_var * mfvi_var) ** 0.5
        out["wass2"] = check_bad(torch.mean(wass2, dim=0)).detach().cpu().numpy()

        # wasserstein-2 joint
        D, V = torch.linalg.eigh(sig_true)
        sig_true_sqrt = V @ torch.diag_embed(D).sqrt() @ torch.transpose(V, -1, -2)
        A = sig_true_sqrt @ sig_mfvi @ sig_true_sqrt
        D2, V2 = torch.linalg.eigh(A)
        A_sqrt = V2 @ torch.diag_embed(D2).sqrt() @ torch.transpose(V2, -1, -2)
        joint_wass2 = torch.diagonal(sig_true + sig_mfvi - 2 * A_sqrt, dim1=1, dim2=2).sum(dim=-1)
        out["joint_wass2"] = check_bad(joint_wass2).detach().cpu().numpy()

        # squared diff of mean predictive variance
        diff_post_var = true_var.mean(dim=0) - mfvi_var.mean(dim=0)
        out["sq_diff_post_var"] = check_bad(diff_post_var**2).detach().cpu().numpy()

        # NLL (true tempered + mfvi)
        true_t_mean, true_t_var = post_pred_mean_var(
            test_X=Phi, post_mean=mu, post_cov=Sigma, noise_std=noise_std, temp=Ts
        )
        true_t_npll = log(2*torch.pi) / 2 + torch.log(true_t_var) / 2 + (y_split - true_t_mean)[:, None]**2 / true_t_var / 2
        mfvi_npll  = log(2*torch.pi) / 2 + torch.log(mfvi_var)    / 2 + (y_split - mfvi_mean)[:, None]**2 / mfvi_var  / 2

        out["true_t_npll"] = check_bad(true_t_npll.mean(dim=0)).detach().cpu().numpy()
        out["mfvi_nplls"]  = check_bad(mfvi_npll.mean(dim=0)).detach().cpu().numpy()

        return out

    # compute for each split (same hypers/model)
    m_tr  = metrics_for_split(Phi_tr,  y_tr)
    m_id  = metrics_for_split(Phi_id,  y_id)
    m_ood = metrics_for_split(Phi_ood, y_ood)

    # flatten into your old "metrix" style with suffixes
    metrix = {}
    for base in m_tr.keys():
        metrix[f"{base}_tr"]  = m_tr[base]
        metrix[f"{base}_id"]  = m_id[base]
        metrix[f"{base}_ood"] = m_ood[base]

    hypers = {
        "noise_std": noise_std,
        "lengthscale": basis_kwargs.get("lengthscale", None),
        "prior_precision": prior_precision,

    }
    return metrix, hypers

    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

if __name__ == "__main__":
    import time
    from tueplots.bundles import icml2024
    plt.rcParams.update(icml2024(nrows=3, ncols=3))

    t0 = time.perf_counter()

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


    variants = [("id", "_id"), ("ood", "_ood")]
 

    figs = {}
    axes = {}
    ax_first = {}

    for te_split, tag in variants:

        fig_kl_fwd, axes_kl_fwd = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_kl_rev, axes_kl_rev = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_alpha, axes_mid = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_wass, axes_bot = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_diff, axes_last = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_nll, axes_nll = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)

        figs[tag] = {
            "kl_fwd": fig_kl_fwd,
            "kl_rev": fig_kl_rev,
            "alpha": fig_alpha,
            "wass": fig_wass,
            "diff": fig_diff,
            "nll": fig_nll,
        }
        axes[tag] = {
            "kl_fwd": axes_kl_fwd.ravel(),
            "kl_rev": axes_kl_rev.ravel(),
            "alpha": axes_mid.ravel(),
            "wass": axes_bot.ravel(),
            "diff": axes_last.ravel(),
            "nll": axes_nll.ravel(),
        }
        ax_first[tag] = {"fwd": None, "rev": None, "alpha": None, "wass": None}

    # save results and meta data
    results = {}
    results["meta"] = {
        "datasets": datasets,
        "dtype": str(dtype),
        "device": str(device),
        "O_NOISE": O_NOISE,
        "P_PRSCISN": P_PRSCISN,
        "LEARN_NOISE_L_ALPHA": LEARN_NOISE_L_ALPHA,
        "TR_FRC": TR_FRC,
        "ID_FRC": ID_FRC,
        "N_MAX": N_MAX,
        "PCA": PCA,
        "N_REPS": N_REPS,
        "T_RANGE_KL": T_RANGE_KL,
        "T_RANGE_ALPHA": T_RANGE_ALPHA,
        "T_RANGE_WASS": T_RANGE_WASS,
        "T_RANGE_DIFF": T_RANGE_DIFF,
        "T_RANGE_NLL": T_RANGE_NLL,
        "T_POINTS": T_POINTS,
        "BASIS": BASIS,
        "BASIS_KWARGS": dict(BASIS_KWARGS),
    }
    results["temps"] = {
        "Ts": Ts.detach().cpu().numpy(),
        "log10Ts": log10Ts_np,
        "mask_kl": mask_kl,
        "mask_alpha": mask_alpha,
        "mask_wass": mask_wass,
        "mask_diff": mask_diff,
        "mask_nll": mask_nll,
        "x_kl": x_kl,
        "x_alpha": x_alpha,
        "x_wass": x_wass,
        "x_diff": x_diff,
        "x_nll": x_nll,
    }
    results["data"] = {}

    bases = [
        "fwd_kls", "rev_kls",
        "fwd_joint_kl", "rev_joint_kl",
        "alpha", "joint_alpha",
        "wass2", "joint_wass2",
        "sq_diff_post_var",
        "true_t_npll",
        "mfvi_nplls",
    ]
    splits = ["tr", "id", "ood"]
    keys = [f"{b}_{s}" for b in bases for s in splits]

    def pick(d, base, split):
        return d[f"{base}_{split}"]

    for dataset in datasets:
        print("=" * 50)
        print(dataset.capitalize())
        X_df, y_df = load_dataset(dataset)

        X = torch.tensor(X_df.values, dtype=dtype, device=device)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype, device=device)

        reps = {k: [] for k in keys}
        hypers = {}
        for i in range(N_REPS):
            met, hyp = divergences(X, y, Ts[:, None], N_MAX)
            hypers[f"rep{i}"] = hyp
            for k in keys:
                reps[k].append(met[k])

        metrix = {}
        metrix2 = {}
        for k in keys:
            arr = np.stack(reps[k], axis=0)
            metrix[k] = arr.mean(axis=0)
            metrix2[k] = 2.0 * arr.std(axis=0)

        # save results
        reps_np = {}
        for k in keys:
            reps_np[k] = np.stack(reps[k], axis=0)

        results["data"][dataset] = {
            "keys": list(keys),
            "reps": reps_np,
            "mean": dict(metrix),
            "std2": dict(metrix2),
            "hypers": hypers
        }

        # plot into each variant's figure set
        for te_split, tag in variants:

            axf, axr, axb, axc, axd, axn = (
                axes[tag]["kl_fwd"][datasets.index(dataset)],
                axes[tag]["kl_rev"][datasets.index(dataset)],
                axes[tag]["alpha"][datasets.index(dataset)],
                axes[tag]["wass"][datasets.index(dataset)],
                axes[tag]["diff"][datasets.index(dataset)],
                axes[tag]["nll"][datasets.index(dataset)],
            )

            # ---------- KL (forward) ----------
            for base, lab_te, lab_tr, ls_te, ls_tr, ax_ in [
                ("fwd_kls", f"marg fwd KL te", "marg fwd KL tr", "--", "--", axf),
            ]:
                y_te = pick(metrix, base, te_split)[mask_kl]
                s_te = pick(metrix2, base, te_split)[mask_kl]
                y_tr = pick(metrix, base, "tr")[mask_kl]
                s_tr = pick(metrix2, base, "tr")[mask_kl]

                ln = ax_.plot(x_kl, y_te, label=lab_te, linestyle=ls_te, color=COL_TE)[0]
                ax_.plot(x_kl, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_kl, y_tr, label=lab_tr, linestyle=ls_tr, color=COL_TR)[0]
                ax_.plot(x_kl, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axf2 = axf.twinx()
            if ax_first[tag]["fwd"] is None:
                ax_first[tag]["fwd"] = axf2

            for base, lab_te, lab_tr, ax_ in [
                ("fwd_joint_kl", "joint fwd KL te", "joint fwd KL tr", axf2),
            ]:
                y_te = pick(metrix, base, te_split)[mask_kl]
                s_te = pick(metrix2, base, te_split)[mask_kl]
                y_tr = pick(metrix, base, "tr")[mask_kl]
                s_tr = pick(metrix2, base, "tr")[mask_kl]

                ln = ax_.plot(x_kl, y_te, label=lab_te, color=COL_TE)[0]
                ax_.plot(x_kl, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_kl, y_tr, label=lab_tr, color=COL_TR)[0]
                ax_.plot(x_kl, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axf.set_title(dataset)
            axf.set_xlabel(r"$\log_{10}(T)$")
            axf.set_ylabel("Marginal KL")
            axf.grid(True, alpha=0.3)
            axf2.set_ylabel("Joint KL")

            # ---------- KL (reverse) ----------
            for base, lab_te, lab_tr, ls_te, ls_tr, ax_ in [
                ("rev_kls", f"marg rev KL te", "marg rev KL tr", "--", "--", axr),
            ]:
                y_te = pick(metrix, base, te_split)[mask_kl]
                s_te = pick(metrix2, base, te_split)[mask_kl]
                y_tr = pick(metrix, base, "tr")[mask_kl]
                s_tr = pick(metrix2, base, "tr")[mask_kl]

                ln = ax_.plot(x_kl, y_te, label=lab_te, linestyle=ls_te, color=COL_TE)[0]
                ax_.plot(x_kl, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_kl, y_tr, label=lab_tr, linestyle=ls_tr, color=COL_TR)[0]
                ax_.plot(x_kl, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axr2 = axr.twinx()
            if ax_first[tag]["rev"] is None:
                ax_first[tag]["rev"] = axr2

            for base, lab_te, lab_tr, ax_ in [
                ("rev_joint_kl", "joint rev KL te", "joint rev KL tr", axr2),
            ]:
                y_te = pick(metrix, base, te_split)[mask_kl]
                s_te = pick(metrix2, base, te_split)[mask_kl]
                y_tr = pick(metrix, base, "tr")[mask_kl]
                s_tr = pick(metrix2, base, "tr")[mask_kl]

                ln = ax_.plot(x_kl, y_te, label=lab_te, color=COL_TE)[0]
                ax_.plot(x_kl, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_kl, y_tr, label=lab_tr, color=COL_TR)[0]
                ax_.plot(x_kl, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_kl, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_kl[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axr.set_title(dataset)
            axr.set_xlabel(r"$\log_{10}(T)$")
            axr.set_ylabel("Marginal KL")
            axr.grid(True, alpha=0.3)
            axr2.set_ylabel("Joint KL")

            # ---------- Alpha ----------
            for base, lab_te, lab_tr, ls_te, ls_tr, ax_ in [
                ("alpha", f"marg alpha te", "marg alpha tr", "--", "--", axb),
            ]:
                y_te = pick(metrix, base, te_split)[mask_alpha]
                s_te = pick(metrix2, base, te_split)[mask_alpha]
                y_tr = pick(metrix, base, "tr")[mask_alpha]
                s_tr = pick(metrix2, base, "tr")[mask_alpha]

                ln = ax_.plot(x_alpha, y_te, label=lab_te, linestyle=ls_te, color=COL_TE)[0]
                ax_.plot(x_alpha, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_alpha, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_alpha[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_alpha, y_tr, label=lab_tr, linestyle=ls_tr, color=COL_TR)[0]
                ax_.plot(x_alpha, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_alpha, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_alpha[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axb2 = axb.twinx()
            if ax_first[tag]["alpha"] is None:
                ax_first[tag]["alpha"] = axb2

            for base, lab_te, lab_tr, ax_ in [
                ("joint_alpha", "joint alpha te", "joint alpha tr", axb2),
            ]:
                y_te = pick(metrix, base, te_split)[mask_alpha]
                s_te = pick(metrix2, base, te_split)[mask_alpha]
                y_tr = pick(metrix, base, "tr")[mask_alpha]
                s_tr = pick(metrix2, base, "tr")[mask_alpha]

                ln = ax_.plot(x_alpha, y_te, label=lab_te, color=COL_TE)[0]
                ax_.plot(x_alpha, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_alpha, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_alpha[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_alpha, y_tr, label=lab_tr, color=COL_TR)[0]
                ax_.plot(x_alpha, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_alpha, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_alpha[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axb.set_title(dataset)
            axb.set_xlabel(r"$\log_{10}(T)$")
            axb.set_ylabel("Marginal alpha")
            axb.grid(True, alpha=0.3)
            axb2.set_ylabel("Joint alpha")

            # ---------- Wasserstein ----------
            for base, lab_te, lab_tr, ls_te, ls_tr, ax_ in [
                ("wass2", f"marg wass2 te", "marg wass2 tr", "--", "--", axc),
            ]:
                y_te = pick(metrix, base, te_split)[mask_wass]
                s_te = pick(metrix2, base, te_split)[mask_wass]
                y_tr = pick(metrix, base, "tr")[mask_wass]
                s_tr = pick(metrix2, base, "tr")[mask_wass]

                ln = ax_.plot(x_wass, y_te, label=lab_te, linestyle=ls_te, color=COL_TE)[0]
                ax_.plot(x_wass, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_wass, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_wass[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_wass, y_tr, label=lab_tr, linestyle=ls_tr, color=COL_TR)[0]
                ax_.plot(x_wass, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_wass, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_wass[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axc2 = axc.twinx()
            if ax_first[tag]["wass"] is None:
                ax_first[tag]["wass"] = axc2

            for base, lab_te, lab_tr, ax_ in [
                ("joint_wass2", "joint wass2 te", "joint wass2 tr", axc2),
            ]:
                y_te = pick(metrix, base, te_split)[mask_wass]
                s_te = pick(metrix2, base, te_split)[mask_wass]
                y_tr = pick(metrix, base, "tr")[mask_wass]
                s_tr = pick(metrix2, base, "tr")[mask_wass]

                ln = ax_.plot(x_wass, y_te, label=lab_te, color=COL_TE)[0]
                ax_.plot(x_wass, y_te + s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.plot(x_wass, y_te - s_te, linestyle=STD_LS, color=COL_TE, alpha=STD_ALPHA)
                ax_.axvline(x_wass[np.argmin(y_te)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                ln = ax_.plot(x_wass, y_tr, label=lab_tr, color=COL_TR)[0]
                ax_.plot(x_wass, y_tr + s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.plot(x_wass, y_tr - s_tr, linestyle=STD_LS, color=COL_TR, alpha=STD_ALPHA)
                ax_.axvline(x_wass[np.argmin(y_tr)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axc.set_title(dataset)
            axc.set_xlabel(r"$\log_{10}(T)$")
            axc.set_ylabel("Marginal wass2")
            axc.grid(True, alpha=0.3)
            axc2.set_ylabel("Joint wass2")

            # ---------- Diff ----------
            for split, lab, col in [(te_split, "diff² post var te", COL_TE),
                                    ("tr", "diff² post var tr", COL_TR)]:
                y0 = pick(metrix, "sq_diff_post_var", split)[mask_diff]
                s0 = pick(metrix2, "sq_diff_post_var", split)[mask_diff]
                ln = axd.plot(x_diff, y0, label=lab, linestyle="--", color=col)[0]
                axd.plot(x_diff, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axd.plot(x_diff, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axd.axvline(x_diff[np.argmin(y0)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axd.set_title(dataset)
            axd.set_xlabel(r"$\log_{10}(T)$")
            axd.set_ylabel("Diff² mean post var")
            axd.grid(True, alpha=0.3)

            # ---------- NLL ----------
            for split, col, tag2 in [(te_split, COL_TE, "te"), ("tr", COL_TR, "tr")]:
                y0 = pick(metrix, "true_t_npll", split)[mask_nll]
                s0 = pick(metrix2, "true_t_npll", split)[mask_nll]
                ln = axn.plot(x_nll, y0, label=f"true T-NLL {tag2}", linestyle="-", color=col)[0]
                axn.plot(x_nll, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axn.plot(x_nll, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axn.axvline(x_nll[np.argmin(y0)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                y0 = pick(metrix, "mfvi_nplls", split)[mask_nll]
                s0 = pick(metrix2, "mfvi_nplls", split)[mask_nll]
                ln = axn.plot(x_nll, y0, label=f"mfvi T-NLL {tag2}", linestyle="--", color=col)[0]
                axn.plot(x_nll, y0 + s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axn.plot(x_nll, y0 - s0, linestyle=STD_LS, color=col, alpha=STD_ALPHA)
                axn.axvline(x_nll[np.argmin(y0)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axn.set_title(dataset)
            axn.set_xlabel(r"$\log_{10}(T)$")
            axn.set_ylabel("NLL")
            axn.grid(True, alpha=0.3)

    # ---------- legends + saving ----------
    if BASIS == "rbf":
        base = f"figs/uci/{FOLDER}/uci_divs_{BASIS}_p={p}_l={l}"
    elif BASIS == "polynomial":
        base = f"figs/uci/{FOLDER}/uci_divs_{BASIS}_deg={deg}"
    else:
        base = f"figs/uci/{FOLDER}/uci_divs_{BASIS}"

    ood_pth = ""

    if PCA:
        ood_pth = "_pca"
    else:
        ood_pth = "_feat"

    os.makedirs(f"figs/uci/{FOLDER}/", exist_ok=True)

    for te_split, tag in variants:

        fig_kl_fwd = figs[tag]["kl_fwd"]
        fig_kl_rev = figs[tag]["kl_rev"]
        fig_alpha = figs[tag]["alpha"]
        fig_wass = figs[tag]["wass"]
        fig_diff = figs[tag]["diff"]
        fig_nll = figs[tag]["nll"]

        handles1, labels1 = axes[tag]["kl_fwd"][0].get_legend_handles_labels()
        handles2, labels2 = ax_first[tag]["fwd"].get_legend_handles_labels()
        handles1r, labels1r = axes[tag]["kl_rev"][0].get_legend_handles_labels()
        handles2r, labels2r = ax_first[tag]["rev"].get_legend_handles_labels()

        handles3, labels3 = axes[tag]["alpha"][0].get_legend_handles_labels()
        handles4, labels4 = ax_first[tag]["alpha"].get_legend_handles_labels()
        handles5, labels5 = axes[tag]["wass"][0].get_legend_handles_labels()
        handles6, labels6 = ax_first[tag]["wass"].get_legend_handles_labels()
        handles7, labels7 = axes[tag]["diff"][0].get_legend_handles_labels()
        handles8, labels8 = axes[tag]["nll"][0].get_legend_handles_labels()

        fig_kl_fwd.legend(handles1 + handles2, labels1 + labels2,
                          loc="outside lower center", ncol=4, frameon=False)
        fig_kl_rev.legend(handles1r + handles2r, labels1r + labels2r,
                          loc="outside lower center", ncol=4, frameon=False)
        fig_alpha.legend(handles3 + handles4, labels3 + labels4,
                         loc="outside lower center", ncol=4, frameon=False)
        fig_wass.legend(handles5 + handles6, labels5 + labels6,
                        loc="outside lower center", ncol=4, frameon=False)
        fig_diff.legend(handles7, labels7,
                        loc="outside lower center", ncol=4, frameon=False)
        fig_nll.legend(handles8, labels8,
                       loc="outside lower center", ncol=4, frameon=False)

        te_name = "id" if te_split == "id" else "ood"
        fig_kl_fwd.suptitle(f"basis: {BASIS} | fwd KL | te == {te_name}")
        fig_kl_rev.suptitle(f"basis: {BASIS} | rev KL | te == {te_name}")
        fig_alpha.suptitle(f"basis: {BASIS} | alpha | te == {te_name}")
        fig_wass.suptitle(f"basis: {BASIS} | wass2 | te == {te_name}")
        fig_diff.suptitle(f"basis: {BASIS} | var diff| te == {te_name}")
        fig_nll.suptitle(f"basis: {BASIS} | NLL | te == {te_name}")

        outpath_kl_fwd = base + ood_pth + tag + "_kl_fwd.pdf"
        outpath_kl_rev = base + ood_pth + tag + "_kl_rev.pdf"
        outpath_alpha = base + ood_pth + tag + "_alpha.pdf"
        outpath_wass = base + ood_pth + tag + "_wass.pdf"
        outpath_diff = base + ood_pth + tag + "_diff.pdf"
        outpath_nll = base + ood_pth + tag + "_nll.pdf"

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

    outpath_results = base + ood_pth + "_results.pt"
    torch.save(results, outpath_results)
    print(f"\nSaved results dict to {outpath_results}")

    t1 = time.perf_counter()

    print(f"This took {t1-t0:.2f} seconds.")




