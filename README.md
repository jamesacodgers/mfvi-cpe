# MFVI-CPE

Code for the paper's experiments on mean-field variational inference (MFVI) and
the cold-posterior effect (CPE). This repository reproduces both the **UCI
experiments** and the **toy / synthetic paper figures**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available; otherwise everything runs on CPU.
The linear-model toy experiments are CPU-friendly; only `bnn_ivon_fig.py` (a real
neural-network training run) benefits meaningfully from a GPU.

### LaTeX (required for the figures)

The figure scripts render text with `matplotlib`'s `text.usetex=True`, so a
working **LaTeX distribution plus `dvipng`** must be on your `PATH` to produce the
PDFs (the Computer Modern fonts, e.g. `cmr*.tfm`, are needed). Without it the
scripts fail at render time with a missing-font error such as
`FileNotFoundError: ... 'cmr7.tfm'`.

```bash
# macOS (Homebrew): BasicTeX is enough
brew install --cask basictex
sudo tlmgr update --self && sudo tlmgr install dvipng cm-super type1cm underscore

# Debian/Ubuntu
sudo apt-get install texlive-latex-extra texlive-fonts-recommended cm-super dvipng
```

The UCI scripts and `play.ipynb` do not require LaTeX.

> **Run all scripts from the repository root.** Output paths (`figs/...`), the
> `src.` imports, and the `uci_data/` cache are all resolved relative to the
> current working directory. The scripts write into existing `figs/<name>/`
> directories — these are tracked in the repo, so no setup is needed.

## Repository layout

```
src/
  UCI_data.py        load_dataset(key, target=None) -> (X, y) DataFrames, with caching
  linear_utils.py    core math: exact & tempered-MFVI posteriors, posterior predictive,
                     divergences/NLL, type-II ML hyperparameter fitting, OOD splits, plots
  basis_functions.py identity / polynomial / RBF feature maps
  utils.py           set_seeds, check_bad (NaN/Inf guard)

# --- paper figures (toy / synthetic) ---
toy_linear_cpe.py            CPE prediction analysis for the linear model
toy_fixed_basis_function.py  fixed-basis (poly / RBF) predictive + spectrum figures
experiment_2d_viz.py         2D data / parameter / precision visualisation
bnn_ivon_fig.py              BNN-vs-linear comparison trained with IVON
play.ipynb                   UCI violin plots from saved *_results.pt tensors

# --- UCI experiments ---
uci_divergence.py        divergences (KL, alpha, Wasserstein, var-diff, NLL) vs temperature
uci_divergence_3way.py   as above, with a 3-way train / in-dist-test / OOD-test split
uci_npll_v_t.py          NPLL vs temperature + feature-Gram eigenspectra (3x3 grid)
uci_eigvals.py           eigenspectrum of the feature Gram matrix (3x3 grid)
uci_hypers.py            optimal observation-noise estimate via type-II ML (naval)
uci_tr_inequ.py          verifies the trace inequality (Thm 4.5) -> LaTeX table
plots.py                 standalone replotting helpers for saved UCI results

figs/                    output directories (tracked); generated figures land here
figs/uci/results1/       archived outputs of uci_divergence.py (p=5000 and p=500)
figs/uci/results2/       archived outputs of uci_divergence_3way.py (3-way "feat" split)
```

## Reproducing the paper figures

Each command below regenerates one figure (or small set of figures). All toy
scripts are configured by editing the constants near the top of the file — there
is no command-line interface except for `bnn_ivon_fig.py`.

### 1. Fixed-basis function figures — `toy_fixed_basis_function.py`

```bash
python toy_fixed_basis_function.py
```

Outputs (to `figs/toy_fixed_basis_function/`):

- `fixed_basis_P16_v2.pdf`, `fixed_basis_P1024_v2.pdf` — predictive fits at two
  feature counts
- `fixed_basis_legend_v2.pdf` — shared legend
- `fixed_basis_spectrum_v2.pdf` — feature-Gram eigenspectrum (not included in paper)

### 2. 2D linear visualisation — `experiment_2d_viz.py`

```bash
python experiment_2d_viz.py
```

Outputs (to `figs/2d_linear/`):

- `2d_data_viz.pdf` — data + predictive in data space
- `2d_parameter_viz.pdf` — exact vs MFVI posterior in parameter space
- `2d_precision_viz.pdf` — precision / covariance visualisation

### 3. Linear-model CPE prediction analysis — `toy_linear_cpe.py`

```bash
python toy_linear_cpe.py
```

Output: `figs/toy_example/prediction_analysis_icml.pdf`

### 4. BNN-vs-linear comparison (IVON) — `bnn_ivon_fig.py`

This one *does* train a small Bayesian neural network with the IVON optimizer, so
it is the slowest figure and the only script with a CLI. Defaults reproduce the
paper's `large` architecture:

```bash
python bnn_ivon_fig.py                       # uses --name large, --epochs 100000
```

Output: `figs/ivon/<name>_bnn_ivon_arch_comparison.pdf`

Key flags (see `--help` for the full list): `--name`, `--epochs`, `--lr`,
`--temperature`, `--prior_std`, `--noise_sigma`, `--train_samples`,
`--test_samples`.

### 5. UCI violin plots — `play.ipynb`

The notebook reads a saved results tensor and renders one violin plot per
dataset. The required tensor,
`figs/uci/results2/uci_divs_rbf_p=500_l=learned_feat_results.pt`, is already
committed, so the notebook runs out of the box:

```bash
jupyter notebook play.ipynb     # run all cells
```

Outputs: `figs/uci/paper_figs/<dataset>_violin.pdf`.

To regenerate the underlying tensor from scratch instead of using the committed
one, run `python uci_divergence_3way.py` (writes to `figs/uci/results2/`) and
re-run the notebook.

## Reproducing the UCI results

```bash
python uci_divergence.py        # main divergence-vs-temperature figures + results .pt
python uci_divergence_3way.py   # 3-way (ID / OOD) split variant
python uci_npll_v_t.py          # NPLL vs temperature + eigenspectra
python uci_eigvals.py           # Gram-matrix eigenspectra
python uci_tr_inequ.py          # trace-inequality LaTeX table
python uci_hypers.py            # type-II ML noise estimate (quick sanity check)
```

### Data

The UCI datasets are downloaded on first use and cached as pickles in a local
`uci_data/` directory (git-ignored). No manual download is needed — the first
run of any UCI script fetches:

`boston, energy, concrete, yacht, wine, protein, kin8nm, power, naval`

via `ucimlrepo` / `openml` / direct UCI URLs. Subsequent runs read from the
cache. To force a re-download, delete `uci_data/`.

### Configuration

Each UCI script is configured by editing the constants near the top of the file.
The most important ones (in `uci_divergence.py` / `uci_divergence_3way.py`):

| Constant | Meaning |
|---|---|
| `BASIS`, `BASIS_KWARGS` | feature map; `"rbf"` with `m` = number of RBF features (e.g. 5000 / 500) |
| `LEARN_NOISE_L_ALPHA` | learn observation noise, lengthscale and prior precision by type-II ML |
| `OOD_TRTE`, `PCA` | out-of-distribution split (by PCA or by feature) vs random split |
| `N_REPS` | number of train/test resamples averaged over |
| `TR_FRC`, `N_MAX` | train fraction and max subsampled points (lower = faster) |
| `T_RANGE_*`, `T_POINTS` | temperature sweep range (log10) and resolution per metric |

### Outputs

Figures and result tensors are written to `figs/uci/`. Filenames encode the
configuration, e.g.:

```
figs/uci/uci_divs_rbf_p=5000_l=learned_ood_feat_nll.pdf
figs/uci/uci_divs_rbf_p=5000_l=learned_ood_feat_results.pt   # torch.save'd results dict
```

The `figs/uci/results1/` and `figs/uci/results2/` folders contain the archived
figures and `*_results.pt` tensors used in the paper. A saved `*_results.pt` can
be reloaded with `torch.load(...)` (or via `plots.py` / `play.ipynb`) to
regenerate plots without rerunning the sweep.
