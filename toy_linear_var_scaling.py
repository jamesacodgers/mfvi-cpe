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
LBL_FS = 22
TTL_FS = 24
TICK_FS = 18
LEG_FS = 14
LW = 2.5

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)

def compute_mfvi_analytic(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int,
                          temperature: float = 1.0, noise_std: float = 1.0):
    D = n_dims
    noise_var = noise_std**2
    norm_sq = torch.sum(x_vec**2)
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu = torch.full((D,), mu_val)
    sigma_sq_val = temperature * noise_var / (norm_sq + noise_var)
    sigma = torch.full((D,), sigma_sq_val)
    return mu, sigma

def compute_exact_posterior(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int, 
                           noise_std: float = 1.0):
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
    return x1, y

def run_scaling_sweep(n_dims_list, NOISE_STD, N_SAMPLES, N_REPEATS):
    temperatures = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = {
        'n_dims': n_dims_list,
        'temperatures': temperatures,
        'exact': {'mean': [], 'sem': []},
        'mfvi': {t: {'mean': [], 'sem': []} for t in temperatures},
        'map': [],
        'prior': []
    }
    
    for n_dims in n_dims_list:
        print(f"Sweep P = {n_dims}...")
        input_std = 1.0 / np.sqrt(n_dims)
        var_scale = 1.0 / n_dims
        noise_var = NOISE_STD**2
        
        # MAP and Prior are constant for a given P (no randomness in their formula here)
        results['prior'].append(1.0 + noise_var)
        results['map'].append(noise_var)
        
        run_exact_vars = []
        run_mfvi_vars = {t: [] for t in temperatures}
        
        for run_idx in range(N_REPEATS):
            seed_train = 42 + run_idx
            X_train, y_train = generate_data(n_samples=N_SAMPLES, n_dims=n_dims, input_std=input_std, noise_std=NOISE_STD, seed=seed_train)
            
            # Exact
            _, sum_Sigma_exact = compute_exact_posterior(X_train, y_train, n_dims, noise_std=NOISE_STD)
            run_exact_vars.append(sum_Sigma_exact * var_scale + noise_var)
            
            # MFVI for each temp
            for t in temperatures:
                _, sigma_mfvi = compute_mfvi_analytic(X_train, y_train, n_dims, temperature=t, noise_std=NOISE_STD)
                run_mfvi_vars[t].append(sigma_mfvi[0].item() * n_dims * var_scale + noise_var)
        
        # Statistics
        results['exact']['mean'].append(np.mean(run_exact_vars))
        results['exact']['sem'].append(np.std(run_exact_vars) / np.sqrt(N_REPEATS))
        
        for t in temperatures:
            results['mfvi'][t]['mean'].append(np.mean(run_mfvi_vars[t]))
            results['mfvi'][t]['sem'].append(np.std(run_mfvi_vars[t]) / np.sqrt(N_REPEATS))
            
    return results

def plot_scaling_results(res, filename):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    P = np.array(res['n_dims'])
    
    # Static lines
    ax1.plot(P, res['prior'], 'k--', label='Prior Pred', alpha=0.4)
    ax1.plot(P, res['map'], color='#d62728', linestyle='--', label='MAP (Noise Only)', alpha=0.6)
    
    # Exact with CI
    mean_exact = np.array(res['exact']['mean'])
    sem_exact = np.array(res['exact']['sem'])
    ax1.plot(P, mean_exact, color='#2ca02c', linewidth=LW+1, label='Exact Post ($T=1$)')
    ax1.fill_between(P, mean_exact - 1.96 * sem_exact, mean_exact + 1.96 * sem_exact, color='#2ca02c', alpha=0.15)
    
    # MFVI with CI for different temps
    # Use a colormap for temperatures
    cmap = plt.get_cmap('coolwarm')
    temps = res['temperatures']
    norm = plt.Normalize(vmin=np.log10(min(temps)), vmax=np.log10(max(temps)))
    
    for t in temps:
        mean_v = np.array(res['mfvi'][t]['mean'])
        sem_v = np.array(res['mfvi'][t]['sem'])
        color = cmap(norm(np.log10(t)))
        label = f'MFVI ($T={t}$)'
        if t == 1.0:
            ax1.plot(P, mean_v, color='black', linewidth=LW, label=label, zorder=10)
        else:
            ax1.plot(P, mean_v, color=color, linewidth=LW-0.5, label=label, alpha=0.8)
        
        ax1.fill_between(P, mean_v - 1.96 * sem_v, mean_v + 1.96 * sem_v, color=color, alpha=0.1)
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.08, top=1.2)
    ax1.set_xlabel('Number of Dimensions $P$', fontsize=LBL_FS)
    ax1.set_ylabel('Predictive Variance', fontsize=LBL_FS)
    ax1.set_title(r'IID Variance Scaling ($x^* \propto \mathbf{1}$, $N=8$)', fontsize=TTL_FS)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=LEG_FS, ncol=2)
    ax1.tick_params(labelsize=TICK_FS)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    n_dims_sweep = [2**i for i in range(1, 11)]
    NOISE_STD = 0.3
    N_SAMPLES = 8
    N_REPEATS = 1000 # Increased for smoother CIs
    
    print(f"Running Variance Scaling Sweep (P={n_dims_sweep}, repeats={N_REPEATS})...")
    results = run_scaling_sweep(n_dims_sweep, NOISE_STD, N_SAMPLES, N_REPEATS)
    plot_scaling_results(results, "figs/toy_example/variance_scaling_sweep.pdf")
    print("Optimization Analysis complete.")
