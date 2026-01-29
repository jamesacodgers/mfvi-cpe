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
                v_T_Sigma_exact_v = sum_Sigma_exact_1
            else:
                v_T_Sigma_exact_v = float(n_dims)
                
            exact_nll_t1 = test_nll_analytic(X_test, y_test, mu_exact, v_vec, v_T_Sigma_exact_v, noise_std=NOISE_STD)
            map_nll = test_nll_analytic(X_test, y_test, mu_exact, v_vec, 0.0, noise_std=NOISE_STD)
            
            run_mfvi_nlls = []
            for temp in temperatures:
                mu, sigma = compute_mfvi_analytic(X_train, y_train, n_dims, temperature=temp, noise_std=NOISE_STD)
                v_T_Sigma_mfvi_v = sigma[0].item() * n_dims
                test_nll = test_nll_analytic(X_test, y_test, mu, v_vec, v_T_Sigma_mfvi_v, noise_std=NOISE_STD)
                run_mfvi_nlls.append(test_nll)
            
            repeat_mfvi_nlls.append(run_mfvi_nlls)
            repeat_exact_nlls_t1.append(exact_nll_t1)
            repeat_map_nlls.append(map_nll)
            if (run_idx + 1) % 100 == 0:
                print(f"Run {run_idx+1}/{N_REPEATS} complete. (D={n_dims})")

        mfvi_nlls = np.array(repeat_mfvi_nlls)
        results = {
            'temperatures': temperatures, 
            'mean_mfvi_nll': np.mean(mfvi_nlls, axis=0),
            'sem_mfvi_nll': np.std(mfvi_nlls, axis=0) / np.sqrt(N_REPEATS),
            'mean_exact_nll': np.mean(repeat_exact_nlls_t1),
            'sem_exact_nll': np.std(repeat_exact_nlls_t1) / np.sqrt(N_REPEATS),
            'mean_map_nll': np.mean(repeat_map_nlls),
            'sem_map_nll': np.std(repeat_map_nlls) / np.sqrt(N_REPEATS),
            'n_dims': n_dims,
            'mode': mode
        }
        results_list.append(results)
    return results_list

def plot_combined_nll_results(results_list, filename, NOISE_STD):
    # Two panels in one row
    n_cols = len(results_list)
    # sharey=False as requested to allow different y values between plots
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 5), sharey=False)
    
    if n_cols == 1:
        axes = [axes]
        
    for i, (ax, res) in enumerate(zip(axes, results_list)):
        ax.plot(res['temperatures'], res['mean_mfvi_nll'], '-', color='black', linewidth=LW_MFVI, label='MFVI NLL', alpha=0.9)
        ax.fill_between(res['temperatures'], res['mean_mfvi_nll'] - 1.96 * res['sem_mfvi_nll'], res['mean_mfvi_nll'] + 1.96 * res['sem_mfvi_nll'], color='black', alpha=0.15, label=r'95\% CI')
        ax.axhline(y=res['mean_exact_nll'], color='#2ca02c', linestyle='--', linewidth=LW_EXACT, alpha=0.8, label='Exact Post ($T=1$)')
        ax.axhspan(res['mean_exact_nll'] - 1.96 * res['sem_exact_nll'],
                    res['mean_exact_nll'] + 1.96 * res['sem_exact_nll'],
                    color='#2ca02c', alpha=0.1)
        
        ax.axhline(y=res['mean_map_nll'], color='#d62728', linestyle='--', linewidth=LW_MAP, alpha=0.8, label='MAP NLL')
        ax.axhspan(res['mean_map_nll'] - 1.96 * res['sem_map_nll'],
                    res['mean_map_nll'] + 1.96 * res['sem_map_nll'],
                    color='#d62728', alpha=0.1)
        ax.axvline(x=1.0, color='grey', linestyle='--', alpha=0.5, label='$T=1$')
        
        ax.set_xlabel('Temperature $T$', fontsize=LBL_FS)
        if i == 0:
            ax.set_ylabel('Test NLL', fontsize=LBL_FS)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=TICK_FS)
        
        # Add sub-figure labels below the plot
        label = f"({chr(97+i)}) $P = {res['n_dims']}$"
        ax.text(0.5, -0.3, label, transform=ax.transAxes, ha='center', fontsize=LBL_FS)
    
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=LEG_FS, ncol=3)
    
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
    plot_combined_nll_results(iid_results, "figs/toy_example/nll_analysis_iid_combined.pdf", NOISE_STD)
    
    # OOD: T in [1, 10^2]
    temps_ood = np.logspace(0, 2, 100)
    print("\nRunning OOD (Orthogonal) Experiment (T >= 1)...")
    ood_results = run_experiment([], temps_ood, NOISE_STD, N_SAMPLES, N_REPEATS, n_dims_list, mode='ood')
    plot_combined_nll_results(ood_results, "figs/toy_example/nll_analysis_ood_combined.pdf", NOISE_STD)
    
    print("\nAll experiments complete.")
