# this scripts calculates and plots various diveregences between true and T-MFVI
# posterior predictive at a range of temperatures, for a number of UCI data sets

import os
import torch
from math import log 
import matplotlib.pyplot as plt
import numpy as np


from src.utils import set_seeds
from src.UCI_data import load_dataset
from src.linear_utils import compute_mfvi_analytic, compute_exact_posterior, post_pred_mean_var, opt_sigma_l
from src.basis_functions import apply_basis
from src.utils import check_bad


dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

O_NOISE = .1 # overridden if LEARN_NOISE_L == True
LEARN_NOISE_L = False
P_PRSCISN = 1
TR_FRC = 0.7

BASIS = "rbf" # | "identity" | "polynomial"
BASIS_KWARGS = {"centers": None, "lengthscale": 1, "m" : 500 }  # {} | {"degree": 5} 

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
    X = X[perm]
    X_tr = X[:tt_split_ind]
    X_te = X[tt_split_ind:n_eff]
    y = y[perm]
    y_tr = y[:tt_split_ind]
    y_te = y[tt_split_ind:n_eff]
    m_X_tr = X_tr.mean(0)
    std_X_tr = X_tr.std(0)
    m_y_tr = y_tr.mean()
    X_tr = (X_tr - m_X_tr) / (std_X_tr + 1e-8)
    X_te = (X_te - m_X_tr) / (std_X_tr + 1e-8)

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

    y_tr -= m_y_tr
    y_te -= m_y_tr

    noise_std = O_NOISE
    if LEARN_NOISE_L:
        sigma, l,  _ = opt_sigma_l(X_tr, y_tr, prior_precision=P_PRSCISN, basis_kwargs=BASIS_KWARGS, basis_type=BASIS,
                                   init_sigma=0.1, init_l=1, verbose=True, steps=1000)
        noise_std = sigma
        BASIS_KWARGS["lengthscale"] = l

    X_tr = apply_basis(X=X_tr, basis_type=BASIS, **BASIS_KWARGS); print("train", X_tr.shape)
    X_te = apply_basis(X=X_te, basis_type=BASIS, **BASIS_KWARGS); print("test", X_te.shape)

    mu, Sigma = compute_exact_posterior(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=P_PRSCISN)

    m_opt, S_opt = compute_mfvi_analytic(train_X=X_tr, train_y=y_tr, noise_std=noise_std, prior_precision=P_PRSCISN)

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

    metrix["diff_post_var_te"] = check_bad(diff_post_var_te).detach().cpu().numpy()
    metrix["diff_post_var_tr"] = check_bad(diff_post_var_tr).detach().cpu().numpy()

    return metrix
    
datasets = ["boston", "energy", "concrete", "yacht", "wine", "protein", "kin8nm", "power", "naval"]

if __name__ == "__main__":
    n_reps = 10

    T_RANGE_KL = (-1, .5)
    T_RANGE_ALPHA = (-2, 1)
    T_RANGE_WASS = (-1, .5)
    T_RANGE_DIFF = (-1.5, .5)

    tmin = min(T_RANGE_KL[0], T_RANGE_ALPHA[0], T_RANGE_WASS[0], T_RANGE_DIFF[0])
    tmax = max(T_RANGE_KL[1], T_RANGE_ALPHA[1], T_RANGE_WASS[1], T_RANGE_DIFF[1])

    Ts = 10 ** torch.linspace(tmin, tmax, 50, dtype=dtype, device=device)
    log10Ts = torch.log10(Ts)

    if BASIS == "rbf":
        p = BASIS_KWARGS["m"]
        l = BASIS_KWARGS["lengthscale"] if not LEARN_NOISE_L else "learned"

    if BASIS == "polynomial":
        deg = BASIS_KWARGS["degree"]

    fig, axes = plt.subplots(12, 3, figsize=(11, 36), constrained_layout=True)
    axes = axes.ravel()
    axes_top = axes[:9]
    axes_mid = axes[9:18]
    axes_bot = axes[18:27]
    axes_last = axes[27:]

    log10Ts_np = log10Ts.detach().cpu().numpy()

    mask_kl = (log10Ts_np >= T_RANGE_KL[0]) & (log10Ts_np <= T_RANGE_KL[1])
    mask_alpha = (log10Ts_np >= T_RANGE_ALPHA[0]) & (log10Ts_np <= T_RANGE_ALPHA[1])
    mask_wass = (log10Ts_np >= T_RANGE_WASS[0]) & (log10Ts_np <= T_RANGE_WASS[1])
    mask_diff = (log10Ts_np >= T_RANGE_DIFF[0]) & (log10Ts_np <= T_RANGE_DIFF[1])

    x_kl = log10Ts_np[mask_kl]
    x_alpha = log10Ts_np[mask_alpha]
    x_wass = log10Ts_np[mask_wass]
    x_diff = log10Ts_np[mask_diff]

    ax2_first = None
    ax4_first = None
    ax6_first = None

    for ax, axb, axc, axd, dataset in zip(axes_top, axes_mid, axes_bot, axes_last, datasets):
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
            "diff_post_var_te", "diff_post_var_tr",
        ]

        reps = {k: [] for k in keys}
        for _ in range(n_reps):
            met = divergences(X, y, Ts[:, None])
            for k in keys:
                reps[k].append(met[k])

        metrix = {}
        metrix2 = {}
        for k in keys:
            arr = np.stack(reps[k], axis=0)
            metrix[k] = arr.mean(axis=0)
            metrix2[k] = 2.0 * arr.std(axis=0)

        for k, lab in [("fwd_kls_te", "marg fwd KL te"), ("rev_kls_te", "marg rev KL te"),
                       ("fwd_kls_tr", "marg fwd KL tr"), ("rev_kls_tr", "marg rev KL tr")]:
            y0 = metrix[k]; e0 = metrix2[k]
            ln = ax.plot(x_kl, y0[mask_kl], label=lab, linestyle="--")[0]
            c0 = ln.get_color()
            ax.plot(x_kl, (y0 + e0)[mask_kl], color=c0, linestyle=":", linewidth=1, label="_nolegend_")
            ax.plot(x_kl, (y0 - e0)[mask_kl], color=c0, linestyle=":", linewidth=1, label="_nolegend_")

        ax2 = ax.twinx()
        if ax2_first is None:
            ax2_first = ax2
        for k, lab, col in [("fwd_joint_kl_te", "joint fwd KL te", "green"),
                            ("rev_joint_kl_te", "joint rev KL te", "olive"),
                            ("fwd_joint_kl_tr", "joint fwd KL tr", "darkgreen"),
                            ("rev_joint_kl_tr", "joint rev KL tr", "seagreen")]:
            y0 = metrix[k]; e0 = metrix2[k]
            ax2.plot(x_kl, y0[mask_kl], label=lab, color=col)
            ax2.plot(x_kl, (y0 + e0)[mask_kl], color=col, linestyle=":", linewidth=1, label="_nolegend_")
            ax2.plot(x_kl, (y0 - e0)[mask_kl], color=col, linestyle=":", linewidth=1, label="_nolegend_")
        ax2.set_ylabel("Joint KL")

        ax.set_title(dataset)
        ax.set_xlabel(r"$\log_{10}(T)$")
        ax.set_ylabel("Marginal KL")
        ax.grid(True, alpha=0.3)

        for k, lab in [("alpha_te", "marg alpha te"), ("alpha_tr", "marg alpha tr")]:
            y0 = metrix[k]; e0 = metrix2[k]
            ln = axb.plot(x_alpha, y0[mask_alpha], label=lab, linestyle="--")[0]
            c0 = ln.get_color()
            axb.plot(x_alpha, (y0 + e0)[mask_alpha], color=c0, linestyle=":", linewidth=1, label="_nolegend_")
            axb.plot(x_alpha, (y0 - e0)[mask_alpha], color=c0, linestyle=":", linewidth=1, label="_nolegend_")

        axb2 = axb.twinx()
        if ax4_first is None:
            ax4_first = axb2
        for k, lab, col in [("joint_alpha_te", "joint alpha te", "green"),
                            ("joint_alpha_tr", "joint alpha tr", "darkgreen")]:
            y0 = metrix[k]; e0 = metrix2[k]
            axb2.plot(x_alpha, y0[mask_alpha], label=lab, color=col)
            axb2.plot(x_alpha, (y0 + e0)[mask_alpha], color=col, linestyle=":", linewidth=1, label="_nolegend_")
            axb2.plot(x_alpha, (y0 - e0)[mask_alpha], color=col, linestyle=":", linewidth=1, label="_nolegend_")
        axb2.set_ylabel("Joint alpha")

        axb.set_title(dataset)
        axb.set_xlabel(r"$\log_{10}(T)$")
        axb.set_ylabel("Marginal alpha")
        axb.grid(True, alpha=0.3)

        for k, lab in [("wass2_te", "marg wass2 te"), ("wass2_tr", "marg wass2 tr")]:
            y0 = metrix[k]; e0 = metrix2[k]
            ln = axc.plot(x_wass, y0[mask_wass], label=lab, linestyle="--")[0]
            c0 = ln.get_color()
            axc.plot(x_wass, (y0 + e0)[mask_wass], color=c0, linestyle=":", linewidth=1, label="_nolegend_")
            axc.plot(x_wass, (y0 - e0)[mask_wass], color=c0, linestyle=":", linewidth=1, label="_nolegend_")

        axc2 = axc.twinx()
        if ax6_first is None:
            ax6_first = axc2
        for k, lab, col in [("joint_wass2_te", "joint wass2 te", "green"),
                            ("joint_wass2_tr", "joint wass2 tr", "darkgreen")]:
            y0 = metrix[k]; e0 = metrix2[k]
            axc2.plot(x_wass, y0[mask_wass], label=lab, color=col)
            axc2.plot(x_wass, (y0 + e0)[mask_wass], color=col, linestyle=":", linewidth=1, label="_nolegend_")
            axc2.plot(x_wass, (y0 - e0)[mask_wass], color=col, linestyle=":", linewidth=1, label="_nolegend_")
        axc2.set_ylabel("Joint wass2")

        axc.set_title(dataset)
        axc.set_xlabel(r"$\log_{10}(T)$")
        axc.set_ylabel("Marginal wass2")
        axc.grid(True, alpha=0.3)

        for k, lab in [("diff_post_var_te", "diff post var te"), ("diff_post_var_tr", "diff post var tr")]:
            y0 = metrix[k]; e0 = metrix2[k]
            ln = axd.plot(x_diff, y0[mask_diff], label=lab, linestyle="--")[0]
            c0 = ln.get_color()
            axd.plot(x_diff, (y0 + e0)[mask_diff], color=c0, linestyle=":", linewidth=1, label="_nolegend_")
            axd.plot(x_diff, (y0 - e0)[mask_diff], color=c0, linestyle=":", linewidth=1, label="_nolegend_")

        axd.set_title(dataset)
        axd.set_xlabel(r"$\log_{10}(T)$")
        axd.set_ylabel("Diff mean post var")
        axd.grid(True, alpha=0.3)

    handles1, labels1 = axes_top[0].get_legend_handles_labels()
    handles2, labels2 = ax2_first.get_legend_handles_labels()
    handles3, labels3 = axes_mid[0].get_legend_handles_labels()
    handles4, labels4 = ax4_first.get_legend_handles_labels()
    handles5, labels5 = axes_bot[0].get_legend_handles_labels()
    handles6, labels6 = ax6_first.get_legend_handles_labels()
    handles7, labels7 = axes_last[0].get_legend_handles_labels()
    fig.legend(handles1 + handles2 + handles3 + handles4 + handles5 + handles6 + handles7, labels1 + labels2 + labels3 + labels4 + labels5 + labels6 + labels7,
            loc="outside lower center",
            ncol=4,
            frameon=False)





    fig.suptitle(f"basis: {BASIS}")

    if BASIS == "rbf":
        outpath = f"figs/uci/uci_divs_{BASIS}_p={p}_l={l}.pdf"
    elif BASIS == "polynomial":
        outpath = f"figs/uci/uci_divs_{BASIS}_deg={deg}.pdf"
    else:
        outpath = f"figs/uci/uci_divs_{BASIS}.pdf"

    os.makedirs("figs/uci/", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

    print(f"Saved figure to {outpath}")

