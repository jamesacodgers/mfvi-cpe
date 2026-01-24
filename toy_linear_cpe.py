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
    p_T(w) \propto [p(y|w) p(w)]^(1/T)
    """
    D = n_dims
    noise_var = noise_std**2
    norm_sq = torch.sum(x_vec**2)
    
    # Cold Posterior Mean is same as standard MAP mean (T=1)
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu = torch.full((D,), mu_val)
    
    # Cold MFVI Variance: v_i = T * sigma^2 / (||x_i||^2 + sigma^2)
    sigma_sq_val = temperature * noise_var / (norm_sq + noise_var)
    sigma = torch.full((D,),sigma_sq_val)
    
    return mu, sigma





def compute_exact_posterior(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int, 
                           noise_std: float = 1.0):
    """
    Computes exact posterior mean and sum of all elements of covariance.
    Uses symmetry and Woodbury identity for O(N) computation.
    """
    N = len(x_vec)
    D = n_dims
    noise_var = noise_std ** 2
    
    # All columns of X are identical: X = x_vec @ 1.T
    norm_sq = torch.sum(x_vec**2)
    
    # mu = (x^T y) / (D ||x||^2 + noise_var) * 1
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu_post = torch.full((D,), mu_val)
    
    # sum(Sigma) = 1.T (I - \lambda/(1 + \lambda D) 1 1.T) 1  where \lambda = ||x||^2 / noise_var
    #            = D / (1 + \lambda D) = D * noise_var / (noise_var + D ||x||^2)
    sum_Sigma_post = (D * noise_var) / (noise_var + D * norm_sq)
    
    return mu_post, sum_Sigma_post.item()


def test_nll_analytic(x_val: torch.Tensor, y_test: torch.Tensor, 
                      mu: torch.Tensor, v: torch.Tensor,
                      v_T_Sigma_v: float,
                      noise_std: float = 1.0):
    """
    Analytical Test NLL for linear Gaussian model.
    x_test = x_val * v
    v_T_Sigma_v is v^T Sigma v.
    """
    M = x_val * torch.dot(v, mu)
    V = (x_val**2) * v_T_Sigma_v + noise_std**2
    
    nll = 0.5 * torch.log(2 * np.pi * V) + 0.5 * (y_test - M)**2 / V
    return nll.mean().item()


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    # X = x1.unsqueeze(1).repeat(1, n_dims)
    # Optimized: only return the representative vector
    y = x1 * true_weights.sum() + torch.randn(n_samples) * noise_std
    return x1, y, true_weights


def generate_test_data(true_weights: torch.Tensor, v: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    # x_test = x1 * v
    y_test = x1 * torch.dot(v, true_weights) + torch.randn(n_samples) * noise_std
    return x1, y_test


# Main experiment
def run_experiment(all_results, temperatures, NOISE_STD, N_SAMPLES, N_REPEATS, n_dims_list, mode='iid'):
    results_list = []
    
    for n_dims in n_dims_list:
        print(f"\n{'='*60}")
        print(f"MODE = {mode.upper()}, N_DIMS = {n_dims}")
        print('='*60)
        
        input_std = 1.0 / np.sqrt(n_dims)
        
        repeat_mfvi_nlls = []
        repeat_exact_nlls_t1 = []
        repeat_map_nlls = []
        
        last_run_X_train = None
        last_run_y_train = None

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

        for run_idx in range(N_REPEATS):
            seed_train = 42 + run_idx
            seed_test = 123 + run_idx
            X_train, y_train, true_weights = generate_data(n_samples=N_SAMPLES, n_dims=n_dims, input_std=input_std, noise_std=NOISE_STD, seed=seed_train)
            X_test, y_test = generate_test_data(true_weights, v_vec, n_samples=1_000, input_std=input_std, noise_std=NOISE_STD, seed=seed_test)
            
            mu_exact, sum_Sigma_exact_1 = compute_exact_posterior(X_train, y_train, n_dims, noise_std=NOISE_STD)
            
            if mode == 'iid':
                # v^T Sigma_exact v = 1^T Sigma 1
                v_T_Sigma_exact_v = sum_Sigma_exact_1
            else:
                # v^T Sigma_exact v = v^T (I - ...) v = v^T v = P (for orthogonal v)
                v_T_Sigma_exact_v = float(n_dims)
                
            exact_nll_t1 = test_nll_analytic(X_test, y_test, mu_exact, v_vec, v_T_Sigma_exact_v, noise_std=NOISE_STD)
            map_nll = test_nll_analytic(X_test, y_test, mu_exact, v_vec, 0.0, noise_std=NOISE_STD)
            
            run_mfvi_nlls = []
            for temp in temperatures:
                mu, sigma = compute_mfvi_analytic(X_train, y_train, n_dims, temperature=temp, noise_std=NOISE_STD)
                # v^T Sigma_mfvi v = sum(v_i^2 * sigma_i^2) = sigma_sq_val * ||v||^2 = sigma_sq_val * P
                v_T_Sigma_mfvi_v = sigma[0].item() * n_dims
                test_nll = test_nll_analytic(X_test, y_test, mu, v_vec, v_T_Sigma_mfvi_v, noise_std=NOISE_STD)
                run_mfvi_nlls.append(test_nll)
            
            repeat_mfvi_nlls.append(run_mfvi_nlls)
            repeat_exact_nlls_t1.append(exact_nll_t1)
            repeat_map_nlls.append(map_nll)
            if (run_idx + 1) % 100 == 0:
                print(f"Run {run_idx+1}/{N_REPEATS} complete. (D={n_dims})")

            if run_idx == N_REPEATS - 1:
                last_run_X_train = X_train
                last_run_y_train = y_train

        mfvi_nlls = np.array(repeat_mfvi_nlls)
        results = {
            'temperatures': temperatures, 
            'mean_mfvi_nll': np.mean(mfvi_nlls, axis=0),
            'sem_mfvi_nll': np.std(mfvi_nlls, axis=0) / np.sqrt(N_REPEATS),
            'mean_exact_nll': np.mean(repeat_exact_nlls_t1),
            'sem_exact_nll': np.std(repeat_exact_nlls_t1) / np.sqrt(N_REPEATS),
            'mean_map_nll': np.mean(repeat_map_nlls),
            'sem_map_nll': np.std(repeat_map_nlls) / np.sqrt(N_REPEATS),
            'v_T_Sigma_mfvi_v': [compute_mfvi_analytic(last_run_X_train, last_run_y_train, n_dims, temperature=t, noise_std=NOISE_STD)[1][0].item() * n_dims for t in temperatures],
            'v_T_Sigma_exact_v': v_T_Sigma_exact_v, # For the last run
            'n_dims': n_dims,
            'mode': mode
        }
        results_list.append(results)
    return results_list

def plot_experiment_results(results_list, filename, title_suffix):
    n_total_rows = len(results_list)
    fig = plt.figure(figsize=(16, 5 * n_total_rows))
    
    for i, res in enumerate(results_list):
        # 1. NLL performance
        ax1 = plt.subplot(n_total_rows, 2, i * 2 + 1)
        N_val = float(N_SAMPLES)
        D_val = float(res['n_dims'])
        t_crit = (1.0 + (D_val - 1.0) * (1.0 + N_val / (NOISE_STD**2))**(-1))/D_val
        
        ax1.plot(res['temperatures'], res['mean_mfvi_nll'], '-', color='black', linewidth=LW_MFVI, label='MFVI NLL', alpha=0.9)
        ax1.fill_between(res['temperatures'], res['mean_mfvi_nll'] - 1.96 * res['sem_mfvi_nll'], res['mean_mfvi_nll'] + 1.96 * res['sem_mfvi_nll'], color='black', alpha=0.15, label=r'95\% CI')
        ax1.axhline(y=res['mean_exact_nll'], color='#2ca02c', linestyle='--', linewidth=LW_EXACT, alpha=0.8, label='Exact Post ($T=1$)')
        ax1.axhspan(res['mean_exact_nll'] - 1.96 * res['sem_exact_nll'],
                    res['mean_exact_nll'] + 1.96 * res['sem_exact_nll'],
                    color='#2ca02c', alpha=0.1)
        
        ax1.axhline(y=res['mean_map_nll'], color='#d62728', linestyle='--', linewidth=LW_MAP, alpha=0.8, label='MAP NLL')
        ax1.axhspan(res['mean_map_nll'] - 1.96 * res['sem_map_nll'],
                    res['mean_map_nll'] + 1.96 * res['sem_map_nll'],
                    color='#d62728', alpha=0.1)
        ax1.axvline(x=1.0, color='grey', linestyle='--', alpha=0.5, label='$T=1$')
        if res['mode'] == 'iid':
            ax1.axvline(x=t_crit, color='blue', linestyle=':', linewidth=2, alpha=0.8, label=r'$T_{\mathrm{crit}}$')
        
        ax1.set_xlabel('Temperature $T$', fontsize=LBL_FS)
        ax1.set_ylabel('Test NLL', fontsize=LBL_FS)
        ax1.set_xscale('log')
        ax1.set_title(f'NLL vs T ($P={res["n_dims"]}, {title_suffix}$)', fontsize=TTL_FS)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=LEG_FS, ncol=2)
        ax1.tick_params(labelsize=TICK_FS)

        # 2. Predictive Densities
        ax3 = plt.subplot(n_total_rows, 2, i * 2 + 2)
        
        # evaluation point x^* = x_val * v
        # We use x_val = input_std = 1/sqrt(P) to represent a "typical" test point magnitude.
        # Variance = x_val^2 * (v^T Sigma v) + noise
        # Scale factor = 1/P
        var_scale = 1.0 / res['n_dims']
        
        # Unscaled variances from results
        raw_var_exact = res['v_T_Sigma_exact_v']
        raw_var_mfvi = res['v_T_Sigma_mfvi_v']
        
        # Prior Variance: v_iid^T I v_iid = P. v_ood^T I v_ood = P.
        # Scaled Prior Variance = (1/P) * P = 1.0
        var_pred_prior = 1.0 
        
        var_pred_exact_t1 = raw_var_exact * var_scale
        var_pred_mfvi = [v * var_scale for v in raw_var_mfvi]
        
        # Debug print
        if i == 0:
            print(f"Mode: {res['mode'].upper()}, D={res['n_dims']}")
            print(f"  Scaled Prior Var: {var_pred_prior}")
            print(f"  Scaled Exact Var: {var_pred_exact_t1}")
            print(f"  Scaled MFVI Var (first/last): {var_pred_mfvi[0]:.4f} / {var_pred_mfvi[-1]:.4f}")

        max_var = max(max(var_pred_mfvi), var_pred_exact_t1, var_pred_prior) + NOISE_STD**2
        max_std = np.sqrt(max_var)
        x_range = np.linspace(-4 * max_std, 4 * max_std, 5000) # Increased resolution
        
        prior_std = np.sqrt(var_pred_prior + NOISE_STD**2)
        prior_pdf = (1 / (prior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / prior_std)**2)
        ax3.plot(x_range, prior_pdf, color='#1f77b4', linewidth=LW_PRIOR, label='Prior Predictive')

        exact_std_t1 = np.sqrt(var_pred_exact_t1 + NOISE_STD**2)
        exact_pdf_t1 = (1 / (exact_std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / exact_std_t1)**2)
        # Use linestyle for Exact Post to allow Prior to be visible underneath if they are identical
        ax3.plot(x_range, exact_pdf_t1, color='#2ca02c', linewidth=LW_EXACT, linestyle='-', label='Exact Post ($T=1$)')
        
        map_std = NOISE_STD
        map_pdf = (1 / (map_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / map_std)**2)
        ax3.plot(x_range, map_pdf, color='#d62728', linewidth=LW_MAP, label='MAP (Noise Only)', zorder=20)
        t_array = np.array(res['temperatures'])
        
        if res['mode'] == 'iid':
            cmap = plt.get_cmap('coolwarm')
            # Temperatures are in [10^-4, 1] - log10 is [-4, 0]
            norm = plt.Normalize(vmin=np.log10(t_array.min()), vmax=0.0)
        else:
            cmap = plt.get_cmap('viridis')
            # Temperatures are in [1, 10^2] - log10 is [0, 2]
            norm = plt.Normalize(vmin=0.0, vmax=2.0)
        
        for t_idx, temp in enumerate(t_array):
            v_p = var_pred_mfvi[t_idx]
            std = np.sqrt(v_p + NOISE_STD**2)
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std)**2)
            color = cmap(norm(np.log10(temp)))
            ax3.plot(x_range, pdf, color=color, alpha=0.3, linewidth=1.0, zorder=1)
            
        t1_idx = np.argmin(np.abs(t_array - 1.0))
        v_p_t1 = var_pred_mfvi[t1_idx]
        std_t1 = np.sqrt(v_p_t1 + NOISE_STD**2)
        pdf_t1 = (1 / (std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std_t1)**2)
        ax3.plot(x_range, pdf_t1, color='black', linestyle='--', linewidth=2, label='MFVI ($T=1$)', alpha=0.9, zorder=2)
            
        ax3.set_xlabel('Predictive $y^*$', fontsize=LBL_FS)
        ax3.set_ylabel('Density', fontsize=LBL_FS)
        if res['mode'] == 'iid':
            ax3.set_title(r'Post Pred ($x^* \propto \mathbf{1}$)', fontsize=TTL_FS)
        else:
            ax3.set_title(r'Post Pred ($x^* \perp \mathbf{1}$)', fontsize=TTL_FS)
        
        tick_vals = np.array([-6, -4, -2, 0, 2, 4, 6])
        tick_locs = tick_vals * NOISE_STD
        tick_labels = [r'$\mu-6\sigma^2$', r'$\mu-4\sigma^2$', r'$\mu-2\sigma^2$', r'$\mu$', r'$\mu+2\sigma^2$', r'$\mu+4\sigma^2$', r'$\mu+6\sigma^2$']
        ax3.set_xticks(tick_locs)
        ax3.set_xticklabels(tick_labels)
        ax3.set_xlim([-7 * NOISE_STD, 7 * NOISE_STD])
        ax3.legend(fontsize=LEG_FS)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(labelsize=TICK_FS)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3)
        cbar.set_label('Log10(Temperature)')
    
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    n_dims_list = [2**1, 2**10]
    N_REPEATS = 10000 
    NOISE_STD = 0.3
    N_SAMPLES = 8
    
    # IID: T in [10^-4, 1]
    temps_iid = np.logspace(-4, 0, 100)
    print("Running IID Experiment (T <= 1)...")
    iid_results = run_experiment([], temps_iid, NOISE_STD, N_SAMPLES, N_REPEATS, n_dims_list, mode='iid')
    plot_experiment_results(iid_results, "figs/toy_example/prediction_analysis_iid.pdf", "IID")
    
    # OOD: T in [1, 10^2]
    temps_ood = np.logspace(0, 2, 100)
    print("\nRunning OOD (Orthogonal) Experiment (T >= 1)...")
    ood_results = run_experiment([], temps_ood, NOISE_STD, N_SAMPLES, N_REPEATS, n_dims_list, mode='ood')
    plot_experiment_results(ood_results, "figs/toy_example/prediction_analysis_ood.pdf", "OOD Ortho")
    
    print("\nAll experiments complete.")