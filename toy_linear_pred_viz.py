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
LBL_FS = 26
TTL_FS = 28
TICK_FS = 22
LEG_FS = 14
LW_EXACT = 3
LW_MFVI = 2.5
LW_MAP = 2
LW_PRIOR = 2.5
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


def compute_mfvi_analytic(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int,
                          temperature: float = 1.0, noise_std: float = 1.0):
    r"""
    Compute closed-form mean-field VI solution for the COLD Bayesian linear regression posterior.
    """
    D = n_dims
    noise_var = noise_std**2
    norm_sq = torch.sum(x_vec**2)
    
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu = torch.full((D,), mu_val)
    
    sigma_sq_val = temperature * noise_var / (norm_sq + noise_var)
    sigma = torch.full((D,),sigma_sq_val)
    
    return mu, sigma


def compute_exact_posterior(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int, 
                           noise_std: float = 1.0):
    """
    Computes exact posterior mean and sum of all elements of covariance.
    """
    D = n_dims
    noise_var = noise_std ** 2
    norm_sq = torch.sum(x_vec**2)
    
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu_post = torch.full((D,), mu_val)
    
    sum_Sigma_post = (D * noise_var) / (noise_var + D * norm_sq)
    
    return mu_post, sum_Sigma_post.item()


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    y = x1 * true_weights.sum() + torch.randn(n_samples) * noise_std
    return x1, y, true_weights


def generate_test_data(true_weights: torch.Tensor, v: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    y_test = x1 * torch.dot(v, true_weights) + torch.randn(n_samples) * noise_std
    return x1, y_test


# Main experiment
def run_experiment(temperatures, NOISE_STD, N_SAMPLES, n_dims_list, mode='iid'):
    results_list = []
    
    for n_dims in n_dims_list:
        print(f"\n{'='*60}")
        print(f"MODE = {mode.upper()}, N_DIMS = {n_dims}")
        print('='*60)
        
        input_std = 1.0 / np.sqrt(n_dims)
        
        if mode == 'iid':
            v_vec = torch.ones(n_dims)
        else: # ood (orthogonal)
            if n_dims == 2:
                v_vec = torch.tensor([1.0, -1.0])
            else:
                v_vec = torch.zeros(n_dims)
                half = n_dims // 2
                v_vec[:half] = 1.0
                v_vec[half:] = -1.0

        seed_train = 42
        X_train, y_train, _ = generate_data(n_samples=N_SAMPLES, n_dims=n_dims, input_std=input_std, noise_std=NOISE_STD, seed=seed_train)
        
        _, sum_Sigma_exact_1 = compute_exact_posterior(X_train, y_train, n_dims, noise_std=NOISE_STD)
        if mode == 'iid':
            v_T_Sigma_exact_v = sum_Sigma_exact_1
        else:
            v_T_Sigma_exact_v = float(n_dims)

        results = {
            'temperatures': temperatures, 
            'v_T_Sigma_mfvi_v': [compute_mfvi_analytic(X_train, y_train, n_dims, temperature=t, noise_std=NOISE_STD)[1][0].item() * n_dims for t in temperatures],
            'v_T_Sigma_exact_v': v_T_Sigma_exact_v,
            'n_dims': n_dims,
            'mode': mode
        }
        results_list.append(results)
    return results_list

def plot_combined_predictive_results(results_list, filename, NOISE_STD):
    n_cols = len(results_list)
    # Sharey=True to compare densities on same scale
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 5), sharey=True)
    
    if n_cols == 1:
        axes = [axes]
        
    for i, (ax, res) in enumerate(zip(axes, results_list)):
        var_scale = 1.0 / res['n_dims']
        raw_var_exact = res['v_T_Sigma_exact_v']
        raw_var_mfvi = res['v_T_Sigma_mfvi_v']
        
        var_pred_prior = 1.0 
        var_pred_exact_t1 = raw_var_exact * var_scale
        var_pred_mfvi = [v * var_scale for v in raw_var_mfvi]
        
        max_var = max(max(var_pred_mfvi), var_pred_exact_t1, var_pred_prior) + NOISE_STD**2
        max_std = np.sqrt(max_var)
        x_range = np.linspace(-4 * max_std, 4 * max_std, 5000)
        
        prior_std = np.sqrt(var_pred_prior + NOISE_STD**2)
        prior_pdf = (1 / (prior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / prior_std)**2)
        ax.plot(x_range, prior_pdf, color='#1f77b4', linewidth=LW_PRIOR, label='Prior Pred')

        exact_std_t1 = np.sqrt(var_pred_exact_t1 + NOISE_STD**2)
        exact_pdf_t1 = (1 / (exact_std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / exact_std_t1)**2)
        ax.plot(x_range, exact_pdf_t1, color='#2ca02c', linewidth=LW_EXACT, linestyle='-', label='Exact Post ($T=1$)')
        
        map_std = NOISE_STD
        map_pdf = (1 / (map_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / map_std)**2)
        ax.plot(x_range, map_pdf, color='#d62728', linewidth=LW_MAP, label='MAP (Noise)', zorder=20)
        
        t_array = np.array(res['temperatures'])
        if res['mode'] == 'iid':
            cmap = plt.get_cmap('coolwarm')
            norm = plt.Normalize(vmin=np.log10(t_array.min()), vmax=0.0)
        else:
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0.0, vmax=2.0)
        
        for t_idx, temp in enumerate(t_array):
            v_p = var_pred_mfvi[t_idx]
            std = np.sqrt(v_p + NOISE_STD**2)
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std)**2)
            color = cmap(norm(np.log10(temp)))
            ax.plot(x_range, pdf, color=color, alpha=0.2, linewidth=1.0, zorder=1)
            
        t1_idx = np.argmin(np.abs(t_array - 1.0))
        v_p_t1 = var_pred_mfvi[t1_idx]
        std_t1 = np.sqrt(v_p_t1 + NOISE_STD**2)
        pdf_t1 = (1 / (std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std_t1)**2)
        ax.plot(x_range, pdf_t1, color='black', linestyle='--', linewidth=2, label='MFVI ($T=1$)', alpha=0.9, zorder=2)
            
        # No x-label as per request
        # ax.set_xlabel('Predictive $y^*$', fontsize=LBL_FS)
        if i == 0:
            ax.set_ylabel('Predictive Density', fontsize=LBL_FS)
        
        tick_vals = np.array([-6, -3, 0, 3, 6])
        tick_locs = tick_vals * NOISE_STD
        tick_labels = [r'$\mu^\top x-6\sigma$', r'$\mu^\top x-3\sigma$', r'$\mu^\top x$', r'$\mu^\top x+3\sigma$', r'$\mu^\top x+6\sigma$']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=30, ha='right')
        ax.set_xlim([-7 * NOISE_STD, 7 * NOISE_STD])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=TICK_FS)
        
        # Sub-figure label
        sub_label = f"({chr(97+i)}) $P = {res['n_dims']}$"
        ax.text(0.5, -0.35, sub_label, transform=ax.transAxes, ha='center', fontsize=LBL_FS)
    
    # Single Colorbar on the right
    fig.subplots_adjust(right=0.85, top=0.88)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Log10($T$)', fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    # Legend for the whole figure - reduced bbox_to_anchor for smaller gap
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=LEG_FS, ncol=4)
    
    # Don't use tight_layout here as it might conflict with the manual colorbar/subplots_adjust
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    n_dims_list = [2**1, 2**10]
    NOISE_STD = 0.3
    N_SAMPLES = 8
    
    # IID: T in [10^-4, 1]
    temps_iid = np.logspace(-4, 0, 100)
    print("Running IID Experiment (T <= 1)...")
    iid_results = run_experiment(temps_iid, NOISE_STD, N_SAMPLES, n_dims_list, mode='iid')
    plot_combined_predictive_results(iid_results, "figs/toy_example/prediction_viz_iid_combined.pdf", NOISE_STD)
    
    # OOD: T in [1, 10^2]
    temps_ood = np.logspace(0, 2, 100)
    print("\nRunning OOD (Orthogonal) Experiment (T >= 1)...")
    ood_results = run_experiment(temps_ood, NOISE_STD, N_SAMPLES, n_dims_list, mode='ood')
    plot_combined_predictive_results(ood_results, "figs/toy_example/prediction_viz_ood_combined.pdf", NOISE_STD)
    
    print("\nAll experiments complete.")
