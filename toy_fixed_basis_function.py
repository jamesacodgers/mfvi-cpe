import torch 
import numpy as np

import matplotlib.pyplot as plt

# ICML/LaTeX Formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# ICML Formatting Constants
LBL_FS = 18
TTL_FS = 20
TICK_FS = 14
LEG_FS = 14
LW_LINE = 2.5

from src.basis_functions import polynomial_basis, rbf_basis
from src.linear_utils import compute_exact_posterior, compute_mfvi_analytic

N=32
N_TEST=100

NOISE_SIGMA=0.1

N_CENTERS=[16,1024]
RBF_SIGMA=0.3
torch.manual_seed(42) 

# x = torch.linspace(-3,3, N).reshape(-1,1)
x = torch.randn(N).reshape(-1,1)*2
y = torch.sinc(x.squeeze(-1)) + torch.randn(N)*NOISE_SIGMA
x = x/3
x_test = torch.linspace(-4.5,4.5,N_TEST).reshape(-1,1)
f_test = torch.sinc(x_test.squeeze(-1)) 
x_test = x_test/3

TEMPERATURES = [0.001, 0.01, 0.1, 1.0]
res = {}

for n_centers in N_CENTERS:
    rbf_centers = torch.linspace(-2.5,2.5,n_centers).reshape(-1,1)
    rbf_length_scale = torch.tensor(RBF_SIGMA)

    phi = rbf_basis(x, centers=rbf_centers, lengthscale=rbf_length_scale)
    print(phi.shape)

    mu, Sigma = compute_exact_posterior(phi, y, NOISE_SIGMA, prior_precision=1.0)
    phi_test = rbf_basis(x_test, centers=rbf_centers,lengthscale=rbf_length_scale)

    mu_preds = mu@phi_test.T
    Sigma_preds = phi_test@Sigma@phi_test.T
    sigma_diag = torch.sqrt(torch.diag(Sigma_preds))

    res[n_centers] = {
        "mu_preds": mu_preds,
        "sigma_diag": sigma_diag,
        "mfvi": {}
    }

    # Store eigenvalues for the Exact Posterior
    evals_exact = torch.linalg.eigvalsh(Sigma)
    res[n_centers]["evals_exact"] = evals_exact.sort(descending=False)[0]

    for temp in TEMPERATURES:
        m, S = compute_mfvi_analytic(phi, y, temp, noise_std=NOISE_SIGMA)
        m_preds = m@phi_test.T
        S_preds = phi_test@torch.diag(S)@phi_test.T
        s_diag = torch.sqrt(torch.diag(S_preds))
        
        res[n_centers]["mfvi"][temp] = {
            "m_preds": m_preds,
            "s_diag": s_diag,
            "S": S
        }


# --- ICML Formatting Constants ---
WIDTH_1COL = 3.25
LBL_FS = 8
TTL_FS = 9
TICK_FS = 7
LEG_FS = 6

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

with torch.no_grad():
    # Colors for different models
    exact_color_main = "black"
    p_colors_spectrum = {16: "#1f77b4", 1024: "#d62728"}
    p_ls_spectrum = {16: "-", 1024: "--"}
    mfvi_cmap = plt.get_cmap("YlOrRd")

    # Legend handles
    h_mean = Line2D([0], [0], color='black', linewidth=1.2, label='Exact mean')
    h_exact_ci = Patch(facecolor='black', alpha=0.1, label='Exact CI')
    h_mfvi_t1 = Line2D([0], [0], color='black', ls='--', linewidth=1.2, label='MFVI ($T=1$)')
    h_data = Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=4, label='Data')

    style_handles = [h_mean, h_exact_ci, h_mfvi_t1, h_data]
    temp_handles = []

    # 1. Predictive Plots (Separate for each P)
    for n_centers, data in res.items():
        fig, ax = plt.subplots(figsize=(WIDTH_1COL, 1.4), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
        
        # Exact Posterior
        ax.plot(x_test, data["mu_preds"], color=exact_color_main, linewidth=1.2, zorder=5)
        ax.fill_between(x_test.squeeze(-1), 
                            data["mu_preds"]+2*(data["sigma_diag"]), 
                            data["mu_preds"]-2*(data["sigma_diag"]), 
                            alpha=0.1, color=exact_color_main, zorder=4)
        
        # MFVI Posterior for different temperatures
        for j, temp in enumerate(TEMPERATURES):
            mfvi_data = data["mfvi"][temp]
            t_label = f"$T=10^{{{int(np.log10(temp))}}}$" if temp < 1 else "$T=1$"

            if temp == 1.0:
                ax.plot(x_test, mfvi_data["m_preds"] + 2*mfvi_data["s_diag"], color='black', ls='--', linewidth=1.2, alpha=0.9, zorder=6)
                ax.plot(x_test, mfvi_data["m_preds"] - 2*mfvi_data["s_diag"], color='black', ls='--', linewidth=1.2, alpha=0.9, zorder=6)
            else:
                color_val = (np.log10(temp) - np.log10(min(TEMPERATURES))) / \
                            (np.log10(max(TEMPERATURES)) - np.log10(min(TEMPERATURES)))
                color = mfvi_cmap(color_val * 0.7 + 0.3)
                ax.plot(x_test, mfvi_data["m_preds"] + 2*mfvi_data["s_diag"], color=color, ls='-', linewidth=0.8, alpha=1.0, zorder=3)
                ax.plot(x_test, mfvi_data["m_preds"] - 2*mfvi_data["s_diag"], color=color, ls='-', linewidth=0.8, alpha=1.0, zorder=3)

                if n_centers == 1024:
                    temp_handles.append(Line2D([0], [0], color=color, linewidth=1.2, label=t_label))

        # Training Data
        ax.scatter(x, y, s=10, c="black", marker='o', zorder=10)
        ax.set_xlabel("$x$", fontsize=LBL_FS, labelpad=-1)
        ax.set_ylabel("$y$", fontsize=LBL_FS, labelpad=-1)
        ax.set_ylim(-1.5,1.5)
        ax.tick_params(labelsize=TICK_FS, pad=0, length=2)
        ax.grid(True, alpha=0.2)
        
        fig.savefig(f"figs/toy_fixed_basis_function/fixed_basis_P{n_centers}_v2.pdf", bbox_inches='tight')
        plt.close(fig)

    # Save legend as separate figure
    all_handles = style_handles + temp_handles
    fig_leg, ax_leg = plt.subplots(figsize=(WIDTH_1COL * 2, 0.5))
    ax_leg.axis('off')
    leg = ax_leg.legend(handles=all_handles, fontsize=9, loc='center', ncol=4,
                        frameon=False, handlelength=1.2, columnspacing=0.8)
    fig_leg.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.set_size_inches(bbox.width, bbox.height)
    fig_leg.savefig("figs/toy_fixed_basis_function/fixed_basis_legend_v2.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_leg)

    # 2. Exact Posterior Spectrum plot (Separate)
    fig_spec, ax_spec = plt.subplots(figsize=(WIDTH_1COL, 1.4), constrained_layout=True)
    fig_spec.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
    
    for n_centers, data in res.items():
        evals = 1/data["evals_exact"]
        x_frac = torch.linspace(0, 1, len(evals))
        ax_spec.plot(x_frac, evals, color=p_colors_spectrum[n_centers], ls=p_ls_spectrum[n_centers], label=f"$P={n_centers}$", linewidth=1.2)
    
    ax_spec.set_yscale('log')
    ax_spec.set_xlabel("$p/P$", fontsize=LBL_FS, labelpad=-1)
    ax_spec.set_ylabel(r"$1/\delta_p$", fontsize=LBL_FS, labelpad=-1)
    ax_spec.tick_params(labelsize=TICK_FS, pad=0, length=2)
    ax_spec.grid(True, alpha=0.2, which='both')
    # Legend moved below axis
    ax_spec.legend(fontsize=LEG_FS, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=2, frameon=False, borderaxespad=0.1)
    
    fig_spec.savefig("figs/toy_fixed_basis_function/fixed_basis_spectrum_v2.pdf", bbox_inches='tight')
    plt.close(fig_spec)
    # plt.show()

