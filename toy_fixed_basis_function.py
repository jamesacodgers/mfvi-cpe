import torch 
import numpy 

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

from src.linear_utils import polynomial_basis, rbf_basis, compute_exact_posterior, compute_mfvi_analytic

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

    mu, Sigma = compute_exact_posterior(phi, y, NOISE_SIGMA)
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
        m, S = compute_mfvi_analytic(phi, y, temp, NOISE_SIGMA)
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

    # Legend handles structure (2x4 row-major)
    # Row 1: Style-based
    h_mean = Line2D([0], [0], color='black', linewidth=1.2, label='Mean')
    h_exact_ci = Patch(facecolor='black', alpha=0.1, label='Exact CI')
    h_mfvi_ci_style = Line2D([0], [0], color='black', ls='--', linewidth=0.8, label='$T$-MFVI CI')
    h_data = Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=4, label='Data')
    
    style_handles = [h_mean, h_exact_ci, h_mfvi_ci_style, h_data]
    temp_handles = []

    # 1. Predictive Plots (Separate for each P)
    for n_centers, data in res.items():
        fig, ax = plt.subplots(figsize=(WIDTH_1COL, 1.8), constrained_layout=True)
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
            color_val = (torch.tensor(temp).log10() - torch.tensor(TEMPERATURES[0]).log10()) / \
                        (torch.tensor(TEMPERATURES[-1]).log10() - torch.tensor(TEMPERATURES[0]).log10())
            color = mfvi_cmap(color_val.item() * 0.7 + 0.3)
            
            # Label for T handles
            t_label = f"$T=10^{{{int(torch.tensor(temp).log10().item())}}}$" if temp < 1 else "$T=1$"
            
            ax.plot(x_test, mfvi_data["m_preds"] + 2*mfvi_data["s_diag"], color=color, ls="--", linewidth=0.8, alpha=0.8)
            ax.plot(x_test, mfvi_data["m_preds"] - 2*mfvi_data["s_diag"], color=color, ls="--", linewidth=0.8, alpha=0.8)
            
            if n_centers == 1024:
                temp_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=t_label))

        # Training Data
        ax.scatter(x, y, s=10, c="black", marker='o', zorder=10)
        ax.set_xlabel("$x$", fontsize=LBL_FS, labelpad=-1)
        ax.set_ylabel("$y$", fontsize=LBL_FS, labelpad=-1)
        ax.set_ylim(-1.5,1.5)
        ax.tick_params(labelsize=TICK_FS, pad=0, length=2)
        ax.grid(True, alpha=0.2)
        
        if n_centers == 1024:
            # Combine handles for 2x4 grid
            all_handles = style_handles + temp_handles
            ax.legend(handles=all_handles, fontsize=LEG_FS-1, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, 
                       borderaxespad=0.1, handlelength=1.2, columnspacing=0.8)
        
        fig.savefig(f"figs/toy_fixed_basis_function/fixed_basis_P{n_centers}.pdf", bbox_inches='tight')
        plt.close(fig)

    # 2. Exact Posterior Spectrum plot (Separate)
    fig_spec, ax_spec = plt.subplots(figsize=(WIDTH_1COL, 1.7), constrained_layout=True)
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
    
    fig_spec.savefig("figs/toy_fixed_basis_function/fixed_basis_spectrum.pdf", bbox_inches='tight')
    plt.close(fig_spec)
    # plt.show()

