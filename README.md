# MFVI-CPE

Code for the paper's experiments on mean-field variational inference (MFVI) and
the cold-posterior effect (CPE). This README focuses on **reproducing the UCI
experiments**; the toy / synthetic experiments live in the `toy_*.py`,
`experiment_2d_viz.py`, and `bnn_ivon_fig.py` scripts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available; otherwise everything runs on CPU
(the linear-model experiments are CPU-friendly).

## Data

The UCI datasets are downloaded on first use and cached as pickles in a local
`uci_data/` directory (git-ignored). No manual download is needed — the first
run of any UCI script will fetch:

`boston, energy, concrete, yacht, wine, protein, kin8nm, power, naval`

via `ucimlrepo` / `openml` / direct UCI URLs. Subsequent runs read from the
cache. To force a re-download, delete `uci_data/`.

## Repository layout

```
src/
  UCI_data.py        load_dataset(key, target=None) -> (X, y) DataFrames, with caching
  linear_utils.py    core math: exact & tempered-MFVI posteriors, posterior predictive,
                     divergences/NLL, type-II ML hyperparameter fitting, OOD splits, plots
  basis_functions.py identity / polynomial / RBF feature maps
  utils.py           set_seeds, check_bad (NaN/Inf guard)

uci_divergence.py        divergences (KL, alpha, Wasserstein, var-diff, NLL) vs temperature
uci_divergence_3way.py   as above, with a 3-way train / in-dist-test / OOD-test split
uci_npll_v_t.py          NPLL vs temperature + feature-Gram eigenspectra (3x3 grid)
uci_eigvals.py           eigenspectrum of the feature Gram matrix (3x3 grid)
uci_hypers.py            optimal observation-noise estimate via type-II ML (naval)
uci_tr_inequ.py          verifies the trace inequality (Thm 4.5) -> LaTeX table

figs/uci/                generated figures and *_results.pt result tensors
figs/uci/results1/       archived outputs of uci_divergence.py (p=5000 and p=500)
figs/uci/results2/       archived outputs of uci_divergence_3way.py (3-way "feat" split)
```

## Reproducing the UCI results

> **Run all scripts from the repository root** — output paths (`figs/uci/...`),
> the `src.` imports, and the `uci_data/` cache are all resolved relative to the
> current working directory.

```bash
python uci_divergence.py        # main divergence-vs-temperature figures + results .pt
python uci_divergence_3way.py   # 3-way (ID / OOD) split variant
python uci_npll_v_t.py          # NPLL vs temperature + eigenspectra
python uci_eigvals.py           # Gram-matrix eigenspectra
python uci_tr_inequ.py          # trace-inequality LaTeX table
python uci_hypers.py            # type-II ML noise estimate (quick sanity check)
```

### Configuration

There is **no command-line interface** — each script is configured by editing the
constants near the top of the file. The most important ones (in
`uci_divergence.py` / `uci_divergence_3way.py`):

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
be reloaded with `torch.load(...)` to regenerate plots without rerunning the
sweep.
