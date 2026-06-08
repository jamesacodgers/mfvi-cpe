import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def id_ood_violin_plots(results_id, results_ood, marg_or_joint=True, include_nll=True):
    """
    For each dataset, creates one figure.

    For each divergence on the x-axis, shows TWO violins:
        - ID:  Δ log10T = log10(T*_test) - log10(T*_train) from results_id
        - OOD: Δ log10T = log10(T*_test) - log10(T*_train) from results_ood

    T* is the argmin over temperatures for that metric, per repetition/run.

    Parameters
    ----------
    results_id : dict
        Results dict from the script for in-distribution (ID) test setting.
    results_ood : dict
        Results dict from the script for out-of-distribution (OOD) test setting.
    marg_or_joint : bool
        If True: include only marginal divergences (where applicable).
        If False: include only joint divergences (where applicable).
        Note: metrics that have no joint version are included only in marginal mode.
    include_nll : bool
        If True, also appends two additional x-axis categories:
            - "true NLL"  (keys: true_t_npll_{te,tr})
            - "mfvi NLL"  (keys: mfvi_nplls_{te,tr})

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping dataset -> figure
    """

    def _get_meta(res):
        datasets = res.get("meta", {}).get("datasets", list(res.get("data", {}).keys()))
        log10Ts = np.asarray(res["temps"]["log10Ts"])
        masks = {
            "kl":    np.asarray(res["temps"]["mask_kl"]),
            "alpha": np.asarray(res["temps"]["mask_alpha"]),
            "wass":  np.asarray(res["temps"]["mask_wass"]),
            "diff":  np.asarray(res["temps"]["mask_diff"]),
            "nll":   np.asarray(res["temps"]["mask_nll"]),
        }
        return datasets, log10Ts, masks

    ds_id, log10_id, masks_id = _get_meta(results_id)
    ds_ood, log10_ood, masks_ood = _get_meta(results_ood)

    datasets = [d for d in ds_id if d in set(ds_ood)]

    # Candidate list: (display_name, base_key, mask_name, kind)
    all_divs = [
        ("fwd KL",          "fwd_kls",          "kl",    "marg"),
        ("fwd KL",          "fwd_joint_kl",     "kl",    "joint"),
        ("rev KL",          "rev_kls",          "kl",    "marg"),
        ("rev KL",          "rev_joint_kl",     "kl",    "joint"),
        ("alpha",           "alpha",            "alpha", "marg"),
        ("alpha",           "joint_alpha",      "alpha", "joint"),
        ("wass2",           "wass2",            "wass",  "marg"),
        ("wass2",           "joint_wass2",      "wass",  "joint"),
        ("var diff²",       "sq_diff_post_var", "diff",  "marg"),  # no joint version
    ]

    want_kind = "marg" if marg_or_joint else "joint"
    divs = [d for d in all_divs if d[3] == want_kind]

    if include_nll:
        divs += [
            ("true NLL", "true_t_npll", "nll", "marg"),
            ("mfvi NLL", "mfvi_nplls",  "nll", "marg"),
        ]

    def _temp_gap_per_rep(res, dataset, base_key, mask, log10Ts):
        reps = res["data"][dataset]["reps"]
        k_te = f"{base_key}_te"
        k_tr = f"{base_key}_tr"
        if (k_te not in reps) or (k_tr not in reps):
            return None

        te = np.asarray(reps[k_te])  # (n_reps, n_temps)
        tr = np.asarray(reps[k_tr])  # (n_reps, n_temps)

        te_m = te[:, mask]
        tr_m = tr[:, mask]
        log10_m = log10Ts[mask]

        te_idx = np.argmin(te_m, axis=1)
        tr_idx = np.argmin(tr_m, axis=1)

        return log10_m[te_idx] - log10_m[tr_idx]  # (n_reps,)

    figs = {}

    for ds in datasets:
        names, deltas_id, deltas_ood = [], [], []

        for disp, base_key, mask_name, _kind in divs:
            gap_id = _temp_gap_per_rep(results_id, ds, base_key, masks_id[mask_name], log10_id)
            gap_ood = _temp_gap_per_rep(results_ood, ds, base_key, masks_ood[mask_name], log10_ood)
            if gap_id is None or gap_ood is None:
                continue

            names.append(disp)
            deltas_id.append(gap_id)
            deltas_ood.append(gap_ood)

        n = len(names)
        if n == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(8, 1.2 * n), 3.8), constrained_layout=True)

        x = np.arange(1, n + 1, dtype=float)
        offset = 0.18
        pos_id = x - offset
        pos_ood = x + offset

        vp_id = ax.violinplot(deltas_id, positions=pos_id, widths=0.32, showmeans=True)
        vp_ood = ax.violinplot(deltas_ood, positions=pos_ood, widths=0.32, showmeans=True)

        for b in vp_id["bodies"]:
            b.set_facecolor("tab:blue"); b.set_alpha(0.55)
        for b in vp_ood["bodies"]:
            b.set_facecolor("tab:orange"); b.set_alpha(0.55)

        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(r"\log_{10}(T^*_{test}) - \log_{10}(T^*_{train})$")
        kind_title = "marginal" if marg_or_joint else "joint"
        ax.set_title(f"{ds}: ID vs OOD train–test optimal temperature gap ({kind_title})")
        ax.grid(True, axis="y", alpha=0.3)

        ax.legend(
            handles=[Patch(facecolor="tab:blue", alpha=0.55, label="ID"),
                     Patch(facecolor="tab:orange", alpha=0.55, label="OOD")],
            frameon=False,
            loc="upper right",
        )

        figs[ds] = fig

    return figs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def raw_id_ood_violin_plots(
    results,
    marg_or_joint=True,
    two_rows_marg_top_joint_bottom=False,
):
    """
    Violin plots of optimal temperatures T* (argmin over temperatures), per divergence.

    Modes
    -----
    - two_rows_marg_top_joint_bottom=False:
        * marg_or_joint=True  -> plot marginal divergences (incl. sq_diff_post_var)
        * marg_or_joint=False -> plot joint divergences (incl. sq_fro_diff_post_cov)
    - two_rows_marg_top_joint_bottom=True:
        plot 2 rows per dataset:
            top    = marginal divergences
            bottom = joint divergences
        Columns are aligned by keeping only divergences that exist for BOTH marg & joint.

    Notes
    -----
    - For the joint "diff" we use the temperature that minimizes `sq_fro_diff_post_cov`.
      (Argmin is identical to minimizing Frobenius norm vs squared Frobenius norm.)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Patch
    from tueplots.bundles import icml2024

    plt.rcParams.update(icml2024(nrows=2 if two_rows_marg_top_joint_bottom else 1, ncols=1))

    def _get_meta(res):
        datasets = res.get("meta", {}).get("datasets", list(res.get("data", {}).keys()))
        log10Ts = np.asarray(res["temps"]["log10Ts"])
        masks = {
            "kl":    np.asarray(res["temps"]["mask_kl"]),
            "alpha": np.asarray(res["temps"]["mask_alpha"]),
            "wass":  np.asarray(res["temps"]["mask_wass"]),
            "diff":  np.asarray(res["temps"]["mask_diff"]),
        }
        return datasets, log10Ts, masks

    datasets, log10Ts, masks = _get_meta(results)

    # One canonical set of 5 "columns" with both marg & joint keys -> stacks nicely.
    # Each entry: (mask_name, marg_key, joint_key, xlabel_marg, xlabel_joint)
    col_specs = [
        ("kl",    "fwd_kls",           "fwd_joint_kl",
         r"${D}_{KL}(p \mid q_T)$",     r"${D}_{KL}(p \mid q_T)$"),
        ("kl",    "rev_kls",           "rev_joint_kl",
         r"${D}_{KL}(q_T \mid p)$",     r"${D}_{KL}(q_T \mid p)$"),
        ("alpha", "alpha",             "joint_alpha",
         r"${D}_\alpha(p \mid q_T)$",   r"${D}_\alpha(p \mid q_T)$"),
        ("wass",  "wass2",             "joint_wass2",
         r"$W_2(p, q_T)$",              r"$W_2(p, q_T)$"),
        ("diff",  "sq_diff_post_var",  "sq_fro_diff_post_cov",
         r"$(\sigma_p^2 - \sigma_{q_T}^2)^2$",  r"$\|\Sigma_p - \Sigma_{q_T}\|_F^2$"),
    ]

    def _argmin_log10T_per_rep(res, dataset, base_key, split_suffix, mask, log10Ts):
        reps = res["data"][dataset]["reps"]
        k = f"{base_key}_{split_suffix}"
        if k not in reps:
            return None

        arr = np.asarray(reps[k])  # (n_reps, n_temps) [or legacy (n_reps, n_temps, 1)]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        arr_m = arr[:, mask]
        log10_m = log10Ts[mask]
        idx = np.argmin(arr_m, axis=1)
        return log10_m[idx]  # (n_reps,)

    def _exists_all_splits(res, ds, base_key):
        reps = res["data"][ds]["reps"]
        return (
            f"{base_key}_tr" in reps
            and f"{base_key}_id" in reps
            and f"{base_key}_ood" in reps
        )

    def _collect(ds, kind, aligned_specs):
        """
        kind: "marg" or "joint"
        returns: xlabels, vals_train, vals_id, vals_ood
        """
        xlabels = []
        vals_train, vals_id, vals_ood = [], [], []

        for mask_name, marg_key, joint_key, xl_m, xl_j in aligned_specs:
            base_key = marg_key if kind == "marg" else joint_key
            m = masks[mask_name]

            tr_star  = _argmin_log10T_per_rep(results, ds, base_key, "tr",  m, log10Ts)
            id_star  = _argmin_log10T_per_rep(results, ds, base_key, "id",  m, log10Ts)
            ood_star = _argmin_log10T_per_rep(results, ds, base_key, "ood", m, log10Ts)

            if tr_star is None or id_star is None or ood_star is None:
                continue

            xlabels.append(xl_m if kind == "marg" else xl_j)
            vals_train.append(tr_star)
            vals_id.append(id_star)
            vals_ood.append(ood_star)

        return xlabels, vals_train, vals_id, vals_ood

    def _plot_one_row(ax, fig, xlabels, vals_train, vals_id, vals_ood, title, show_x, show_legend):
        n = len(xlabels)
        if n == 0:
            ax.set_axis_off()
            return

        x = np.arange(1, n + 1, dtype=float)
        off = 0.24
        pos_train = x - off
        pos_id    = x
        pos_ood   = x + off

        vp_train = ax.violinplot(vals_train, positions=pos_train, widths=0.25, showmeans=True)
        vp_id    = ax.violinplot(vals_id,    positions=pos_id,    widths=0.25, showmeans=True)
        vp_ood   = ax.violinplot(vals_ood,   positions=pos_ood,   widths=0.25, showmeans=True)

        for b in vp_train["bodies"]:
            b.set_facecolor("tab:blue");   b.set_alpha(0.55)
        for b in vp_id["bodies"]:
            b.set_facecolor("tab:orange"); b.set_alpha(0.55)
        for b in vp_ood["bodies"]:
            b.set_facecolor("tab:green");  b.set_alpha(0.55)

        ax.set_ylabel(r"$\log_{10}(T^*)$")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        if show_x:
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=15, ha="right")

            dx = 20 / 72
            txt_offset = mtransforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
            for lab in ax.get_xticklabels():
                lab.set_transform(lab.get_transform() + txt_offset)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        if show_legend:
            ax.legend(
                handles=[
                    Patch(facecolor="tab:blue",   alpha=0.55, label="Train"),
                    Patch(facecolor="tab:orange", alpha=0.55, label="ID Test"),
                    Patch(facecolor="tab:green",  alpha=0.55, label="OOD Test"),
                ],
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=3,
            )

    figs = {}

    for ds in datasets:
        if two_rows_marg_top_joint_bottom:
            # Keep only columns that exist for BOTH marg and joint (so stacking aligns)
            aligned_specs = []
            for spec in col_specs:
                _, marg_key, joint_key, _, _ = spec
                if _exists_all_splits(results, ds, marg_key) and _exists_all_splits(results, ds, joint_key):
                    aligned_specs.append(spec)

            if len(aligned_specs) == 0:
                continue

            xlab_m, tr_m, id_m, ood_m = _collect(ds, "marg", aligned_specs)
            xlab_j, tr_j, id_j, ood_j = _collect(ds, "joint", aligned_specs)

            # If something odd happens and collection drops items, align by intersection length
            n = min(len(xlab_m), len(xlab_j))
            if n == 0:
                continue
            xlab_m, tr_m, id_m, ood_m = xlab_m[:n], tr_m[:n], id_m[:n], ood_m[:n]
            xlab_j, tr_j, id_j, ood_j = xlab_j[:n], tr_j[:n], id_j[:n], ood_j[:n]

            fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)  # no sharey

            _plot_one_row(
                axes[0], fig, xlab_m, tr_m, id_m, ood_m,
                title="Marginal divergences",
                show_x=False,          # remove x ticks/labels on top row
                show_legend=False,
            )
            _plot_one_row(
                axes[1], fig, xlab_j, tr_j, id_j, ood_j,
                title="Joint divergences",
                show_x=True,
                show_legend=True,
            )

            fig.suptitle(ds)

            figs[ds] = fig


        else:
            # single row: either marginal or joint
            want_kind = "marg" if marg_or_joint else "joint"

            # Keep only columns that exist for that kind
            specs = []
            for spec in col_specs:
                _, marg_key, joint_key, _, _ = spec
                base_key = marg_key if want_kind == "marg" else joint_key
                if _exists_all_splits(results, ds, base_key):
                    specs.append(spec)

            if len(specs) == 0:
                continue

            xlab, tr_v, id_v, ood_v = _collect(ds, want_kind, specs)
            if len(xlab) == 0:
                continue

            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
            _plot_one_row(
                ax, fig, xlab, tr_v, id_v, ood_v,
                title="Marginal divergences" if want_kind == "marg" else "Joint divergences",
                show_x=True,
                show_legend=True,
            )

            figs[ds] = fig

    return figs


import numpy as np
import matplotlib.pyplot as plt

def divergence_grid_plots(results, apply_icml_style=True):
    """
    Recreate the 3x3 grid divergence plots (KL fwd/rev, alpha, wass, diff, nll)
    for both variants: te_split in {id, ood}.

    Changes vs previous version:
      - Uncertainty band is IQR (Q1..Q3) from per-rep "reps" (not +-2std).
      - Keep the mean line.
      - For twinx plots, align y=0 at the same vertical level on both y-axes.

    Returns
    -------
    figs : dict[str, dict[str, matplotlib.figure.Figure]]
        figs["_id" or "_ood"][metric_name] -> Figure
        metric_name in {"kl_fwd","kl_rev","alpha","wass","diff","nll"}
    """
    # Optional style (kept inside, as requested)
    if apply_icml_style:
        try:
            from tueplots.bundles import icml2024
            plt.rcParams.update(icml2024(nrows=3, ncols=3))
        except Exception:
            pass
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        })

    datasets = results.get("meta", {}).get("datasets", list(results.get("data", {}).keys()))
    temps = results["temps"]

    log10Ts = np.asarray(temps["log10Ts"])

    mask_kl    = np.asarray(temps.get("mask_kl",    np.ones_like(log10Ts, dtype=bool)))
    mask_alpha = np.asarray(temps.get("mask_alpha", np.ones_like(log10Ts, dtype=bool)))
    mask_wass  = np.asarray(temps.get("mask_wass",  np.ones_like(log10Ts, dtype=bool)))
    mask_diff  = np.asarray(temps.get("mask_diff",  np.ones_like(log10Ts, dtype=bool)))
    mask_nll   = np.asarray(temps.get("mask_nll",   np.ones_like(log10Ts, dtype=bool)))

    x_kl    = np.asarray(temps.get("x_kl",    log10Ts[mask_kl]))
    x_alpha = np.asarray(temps.get("x_alpha", log10Ts[mask_alpha]))
    x_wass  = np.asarray(temps.get("x_wass",  log10Ts[mask_wass]))
    x_diff  = np.asarray(temps.get("x_diff",  log10Ts[mask_diff]))
    x_nll   = np.asarray(temps.get("x_nll",   log10Ts[mask_nll]))

    # colors / styling (same defaults as your script)
    COL_TE = "tab:blue"
    COL_TR = "tab:orange"
    fill_alf = 0.15

    variants = [("id", "_id"), ("ood", "_ood")]

    # Helper: squeeze legacy saved arrays (e.g., (T,1) instead of (T,))
    def _vec(a):
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[-1] == 1:
            a = a[:, 0]
        return a

    def pick(mean_dict, base, split):
        return _vec(mean_dict[f"{base}_{split}"])

    def _mean_and_iqr(data_block, base, split):
        """
        Returns (mean, q1, q3) arrays of shape (n_temps,)
        mean from stored 'mean'; q1/q3 computed from stored 'reps'.
        """
        mean_dict = data_block["mean"]
        reps_dict = data_block["reps"]

        k = f"{base}_{split}"

        mu = pick(mean_dict, base, split)

        if k not in reps_dict:
            # fall back: no band
            return mu, None, None

        arr = np.asarray(reps_dict[k])  # (n_reps, n_temps) or legacy (n_reps, n_temps, 1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        q1 = np.quantile(arr, 0.25, axis=0)
        q3 = np.quantile(arr, 0.75, axis=0)
        return _vec(mu), _vec(q1), _vec(q3)

    def _align_zero(ax1, ax2):
        """
        Force 0 to appear at the same vertical position on ax1 and ax2.
        Ensures 0 is within both y-limits, then expands ax2 limits to match ax1's zero position.
        """
        y1min, y1max = ax1.get_ylim()
        y2min, y2max = ax2.get_ylim()

        # ensure 0 is included on both
        y1min = min(y1min, 0.0); y1max = max(y1max, 0.0)
        y2min = min(y2min, 0.0); y2max = max(y2max, 0.0)

        # guard degenerate
        if np.isclose(y1max - y1min, 0.0):
            y1max = y1min + 1.0
        if np.isclose(y2max - y2min, 0.0):
            y2max = y2min + 1.0

        ax1.set_ylim(y1min, y1max)
        ax2.set_ylim(y2min, y2max)

        # fraction of axis height where 0 lies on ax1
        p = (0.0 - y1min) / (y1max - y1min)

        # avoid pathological p exactly 0 or 1
        eps = 1e-9
        p = float(np.clip(p, eps, 1.0 - eps))

        # choose range R for ax2 so that its [y2min', y2max'] covers current [y2min, y2max]
        # with 0 at the same fraction p
        need_R1 = y2max / (1.0 - p)  # ensures y2max' >= y2max
        need_R2 = (-y2min) / p       # ensures y2min' <= y2min
        R = max(need_R1, need_R2, 1e-6)

        y2min_new = -p * R
        y2max_new = (1.0 - p) * R
        ax2.set_ylim(y2min_new, y2max_new)

    # create figures/axes containers
    figs = {}
    axes = {}
    ax_first = {}

    for te_split, tag in variants:
        fig_kl_fwd, axes_kl_fwd = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_kl_rev, axes_kl_rev = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_alpha,  axes_mid    = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_wass,   axes_bot    = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_diff,   axes_last   = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        fig_nll,    axes_nll    = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)

        figs[tag] = {
            "kl_fwd": fig_kl_fwd,
            "kl_rev": fig_kl_rev,
            "alpha":  fig_alpha,
            "wass":   fig_wass,
            "diff":   fig_diff,
            "nll":    fig_nll,
        }
        axes[tag] = {
            "kl_fwd": axes_kl_fwd.ravel(),
            "kl_rev": axes_kl_rev.ravel(),
            "alpha":  axes_mid.ravel(),
            "wass":   axes_bot.ravel(),
            "diff":   axes_last.ravel(),
            "nll":    axes_nll.ravel(),
        }
        ax_first[tag] = {"fwd": None, "rev": None, "alpha": None, "wass": None}

    # plot each dataset into the grids
    for dataset in datasets:
        if dataset not in results["data"]:
            continue

        data_block = results["data"][dataset]
        ds_idx = datasets.index(dataset)

        for te_split, tag in variants:
            axf, axr, axb, axc, axd, axn = (
                axes[tag]["kl_fwd"][ds_idx],
                axes[tag]["kl_rev"][ds_idx],
                axes[tag]["alpha"][ds_idx],
                axes[tag]["wass"][ds_idx],
                axes[tag]["diff"][ds_idx],
                axes[tag]["nll"][ds_idx],
            )

            # ---------- KL (forward) marginal ----------
            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "fwd_kls", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "fwd_kls", "tr")

            y_te_m = y_te[mask_kl]
            y_tr_m = y_tr[mask_kl]
            ln = axf.plot(x_kl, y_te_m, label="marg fwd KL te", linestyle="--", color=COL_TE)[0]
            if q1_te is not None:
                axf.fill_between(x_kl, q1_te[mask_kl], q3_te[mask_kl], color=COL_TE, alpha=fill_alf)
            axf.axvline(x_kl[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axf.plot(x_kl, y_tr_m, label="marg fwd KL tr", linestyle="--", color=COL_TR)[0]
            if q1_tr is not None:
                axf.fill_between(x_kl, q1_tr[mask_kl], q3_tr[mask_kl], color=COL_TR, alpha=fill_alf)
            axf.axvline(x_kl[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            # ---------- KL (forward) joint ----------
            axf2 = axf.twinx()
            if ax_first[tag]["fwd"] is None:
                ax_first[tag]["fwd"] = axf2

            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "fwd_joint_kl", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "fwd_joint_kl", "tr")

            y_te_m = y_te[mask_kl]
            y_tr_m = y_tr[mask_kl]
            ln = axf2.plot(x_kl, y_te_m, label="joint fwd KL te", color=COL_TE)[0]
            if q1_te is not None:
                axf2.fill_between(x_kl, q1_te[mask_kl], q3_te[mask_kl], color=COL_TE, alpha=fill_alf)
            axf2.axvline(x_kl[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axf2.plot(x_kl, y_tr_m, label="joint fwd KL tr", color=COL_TR)[0]
            if q1_tr is not None:
                axf2.fill_between(x_kl, q1_tr[mask_kl], q3_tr[mask_kl], color=COL_TR, alpha=fill_alf)
            axf2.axvline(x_kl[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axf.set_title(dataset)
            axf.set_xlabel(r"$\log_{10}(T)$")
            axf.set_ylabel("Marginal KL")
            axf.grid(True, alpha=0.3)
            axf2.set_ylabel("Joint KL")
            _align_zero(axf, axf2)

            # ---------- KL (reverse) marginal ----------
            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "rev_kls", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "rev_kls", "tr")

            y_te_m = y_te[mask_kl]
            y_tr_m = y_tr[mask_kl]
            ln = axr.plot(x_kl, y_te_m, label="marg rev KL te", linestyle="--", color=COL_TE)[0]
            if q1_te is not None:
                axr.fill_between(x_kl, q1_te[mask_kl], q3_te[mask_kl], color=COL_TE, alpha=fill_alf)
            axr.axvline(x_kl[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axr.plot(x_kl, y_tr_m, label="marg rev KL tr", linestyle="--", color=COL_TR)[0]
            if q1_tr is not None:
                axr.fill_between(x_kl, q1_tr[mask_kl], q3_tr[mask_kl], color=COL_TR, alpha=fill_alf)
            axr.axvline(x_kl[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            # ---------- KL (reverse) joint ----------
            axr2 = axr.twinx()
            if ax_first[tag]["rev"] is None:
                ax_first[tag]["rev"] = axr2

            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "rev_joint_kl", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "rev_joint_kl", "tr")

            y_te_m = y_te[mask_kl]
            y_tr_m = y_tr[mask_kl]
            ln = axr2.plot(x_kl, y_te_m, label="joint rev KL te", color=COL_TE)[0]
            if q1_te is not None:
                axr2.fill_between(x_kl, q1_te[mask_kl], q3_te[mask_kl], color=COL_TE, alpha=fill_alf)
            axr2.axvline(x_kl[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axr2.plot(x_kl, y_tr_m, label="joint rev KL tr", color=COL_TR)[0]
            if q1_tr is not None:
                axr2.fill_between(x_kl, q1_tr[mask_kl], q3_tr[mask_kl], color=COL_TR, alpha=fill_alf)
            axr2.axvline(x_kl[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axr.set_title(dataset)
            axr.set_xlabel(r"$\log_{10}(T)$")
            axr.set_ylabel("Marginal KL")
            axr.grid(True, alpha=0.3)
            axr2.set_ylabel("Joint KL")
            _align_zero(axr, axr2)

            # ---------- Alpha marginal ----------
            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "alpha", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "alpha", "tr")

            y_te_m = y_te[mask_alpha]
            y_tr_m = y_tr[mask_alpha]
            ln = axb.plot(x_alpha, y_te_m, label="marg alpha te", linestyle="--", color=COL_TE)[0]
            if q1_te is not None:
                axb.fill_between(x_alpha, q1_te[mask_alpha], q3_te[mask_alpha], color=COL_TE, alpha=fill_alf)
            axb.axvline(x_alpha[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axb.plot(x_alpha, y_tr_m, label="marg alpha tr", linestyle="--", color=COL_TR)[0]
            if q1_tr is not None:
                axb.fill_between(x_alpha, q1_tr[mask_alpha], q3_tr[mask_alpha], color=COL_TR, alpha=fill_alf)
            axb.axvline(x_alpha[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            # ---------- Alpha joint ----------
            axb2 = axb.twinx()
            if ax_first[tag]["alpha"] is None:
                ax_first[tag]["alpha"] = axb2

            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "joint_alpha", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "joint_alpha", "tr")

            y_te_m = y_te[mask_alpha]
            y_tr_m = y_tr[mask_alpha]
            ln = axb2.plot(x_alpha, y_te_m, label="joint alpha te", color=COL_TE)[0]
            if q1_te is not None:
                axb2.fill_between(x_alpha, q1_te[mask_alpha], q3_te[mask_alpha], color=COL_TE, alpha=fill_alf)
            axb2.axvline(x_alpha[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axb2.plot(x_alpha, y_tr_m, label="joint alpha tr", color=COL_TR)[0]
            if q1_tr is not None:
                axb2.fill_between(x_alpha, q1_tr[mask_alpha], q3_tr[mask_alpha], color=COL_TR, alpha=fill_alf)
            axb2.axvline(x_alpha[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axb.set_title(dataset)
            axb.set_xlabel(r"$\log_{10}(T)$")
            axb.set_ylabel("Marginal alpha")
            axb.grid(True, alpha=0.3)
            axb2.set_ylabel("Joint alpha")
            _align_zero(axb, axb2)

            # ---------- Wasserstein marginal ----------
            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "wass2", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "wass2", "tr")

            y_te_m = y_te[mask_wass]
            y_tr_m = y_tr[mask_wass]
            ln = axc.plot(x_wass, y_te_m, label="marg wass2 te", linestyle="--", color=COL_TE)[0]
            if q1_te is not None:
                axc.fill_between(x_wass, q1_te[mask_wass], q3_te[mask_wass], color=COL_TE, alpha=fill_alf)
            axc.axvline(x_wass[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axc.plot(x_wass, y_tr_m, label="marg wass2 tr", linestyle="--", color=COL_TR)[0]
            if q1_tr is not None:
                axc.fill_between(x_wass, q1_tr[mask_wass], q3_tr[mask_wass], color=COL_TR, alpha=fill_alf)
            axc.axvline(x_wass[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            # ---------- Wasserstein joint ----------
            axc2 = axc.twinx()
            if ax_first[tag]["wass"] is None:
                ax_first[tag]["wass"] = axc2

            y_te, q1_te, q3_te = _mean_and_iqr(data_block, "joint_wass2", te_split)
            y_tr, q1_tr, q3_tr = _mean_and_iqr(data_block, "joint_wass2", "tr")

            y_te_m = y_te[mask_wass]
            y_tr_m = y_tr[mask_wass]
            ln = axc2.plot(x_wass, y_te_m, label="joint wass2 te", color=COL_TE)[0]
            if q1_te is not None:
                axc2.fill_between(x_wass, q1_te[mask_wass], q3_te[mask_wass], color=COL_TE, alpha=fill_alf)
            axc2.axvline(x_wass[np.argmin(y_te_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            ln = axc2.plot(x_wass, y_tr_m, label="joint wass2 tr", color=COL_TR)[0]
            if q1_tr is not None:
                axc2.fill_between(x_wass, q1_tr[mask_wass], q3_tr[mask_wass], color=COL_TR, alpha=fill_alf)
            axc2.axvline(x_wass[np.argmin(y_tr_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axc.set_title(dataset)
            axc.set_xlabel(r"$\log_{10}(T)$")
            axc.set_ylabel("Marginal wass2")
            axc.grid(True, alpha=0.3)
            axc2.set_ylabel("Joint wass2")
            _align_zero(axc, axc2)

            # ---------- Diff ----------
            for split, lab, col in [(te_split, "diff² post var te", COL_TE),
                                    ("tr", "diff² post var tr", COL_TR)]:
                y0, q1, q3 = _mean_and_iqr(data_block, "sq_diff_post_var", split)
                y0_m = y0[mask_diff]
                ln = axd.plot(x_diff, y0_m, label=lab, linestyle="--", color=col)[0]
                if q1 is not None:
                    axd.fill_between(x_diff, q1[mask_diff], q3[mask_diff], color=col, alpha=fill_alf)
                axd.axvline(x_diff[np.argmin(y0_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axd.set_title(dataset)
            axd.set_xlabel(r"$\log_{10}(T)$")
            axd.set_ylabel("Diff² mean post var")
            axd.grid(True, alpha=0.3)

            # ---------- NLL ----------
            for split, col, tag2 in [(te_split, COL_TE, "te"), ("tr", COL_TR, "tr")]:
                y0, q1, q3 = _mean_and_iqr(data_block, "true_t_npll", split)
                y0_m = y0[mask_nll]
                ln = axn.plot(x_nll, y0_m, label=f"true T-NLL {tag2}", linestyle="-", color=col)[0]
                if q1 is not None:
                    axn.fill_between(x_nll, q1[mask_nll], q3[mask_nll], color=col, alpha=fill_alf)
                axn.axvline(x_nll[np.argmin(y0_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

                y0, q1, q3 = _mean_and_iqr(data_block, "mfvi_nplls", split)
                y0_m = y0[mask_nll]
                ln = axn.plot(x_nll, y0_m, label=f"mfvi T-NLL {tag2}", linestyle="--", color=col)[0]
                if q1 is not None:
                    axn.fill_between(x_nll, q1[mask_nll], q3[mask_nll], color=col, alpha=fill_alf)
                axn.axvline(x_nll[np.argmin(y0_m)], color=ln.get_color(), linestyle=ln.get_linestyle(), alpha=0.4)

            axn.set_title(dataset)
            axn.set_xlabel(r"$\log_{10}(T)$")
            axn.set_ylabel("NLL")
            axn.grid(True, alpha=0.3)

    # ---------- legends + suptitles inside the function ----------
    BASIS = results.get("meta", {}).get("BASIS", "unknown")

    for te_split, tag in variants:
        fig_kl_fwd = figs[tag]["kl_fwd"]
        fig_kl_rev = figs[tag]["kl_rev"]
        fig_alpha  = figs[tag]["alpha"]
        fig_wass   = figs[tag]["wass"]
        fig_diff   = figs[tag]["diff"]
        fig_nll    = figs[tag]["nll"]

        # KL fwd legends
        h1, l1 = axes[tag]["kl_fwd"][0].get_legend_handles_labels()
        h2, l2 = ax_first[tag]["fwd"].get_legend_handles_labels()
        fig_kl_fwd.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=4, frameon=False)

        # KL rev legends
        h1r, l1r = axes[tag]["kl_rev"][0].get_legend_handles_labels()
        h2r, l2r = ax_first[tag]["rev"].get_legend_handles_labels()
        fig_kl_rev.legend(h1r + h2r, l1r + l2r, loc="outside lower center", ncol=4, frameon=False)

        # Alpha legends
        h3, l3 = axes[tag]["alpha"][0].get_legend_handles_labels()
        h4, l4 = ax_first[tag]["alpha"].get_legend_handles_labels()
        fig_alpha.legend(h3 + h4, l3 + l4, loc="outside lower center", ncol=4, frameon=False)

        # Wass legends
        h5, l5 = axes[tag]["wass"][0].get_legend_handles_labels()
        h6, l6 = ax_first[tag]["wass"].get_legend_handles_labels()
        fig_wass.legend(h5 + h6, l5 + l6, loc="outside lower center", ncol=4, frameon=False)

        # Diff + NLL legends
        h7, l7 = axes[tag]["diff"][0].get_legend_handles_labels()
        fig_diff.legend(h7, l7, loc="outside lower center", ncol=4, frameon=False)

        h8, l8 = axes[tag]["nll"][0].get_legend_handles_labels()
        fig_nll.legend(h8, l8, loc="outside lower center", ncol=4, frameon=False)

        te_name = "id" if te_split == "id" else "ood"
        fig_kl_fwd.suptitle(f"basis: {BASIS} | fwd KL | te == {te_name}")
        fig_kl_rev.suptitle(f"basis: {BASIS} | rev KL | te == {te_name}")
        fig_alpha.suptitle(f"basis: {BASIS} | alpha | te == {te_name}")
        fig_wass.suptitle(f"basis: {BASIS} | wass2 | te == {te_name}")
        fig_diff.suptitle(f"basis: {BASIS} | var diff | te == {te_name}")
        fig_nll.suptitle(f"basis: {BASIS} | NLL | te == {te_name}")

    return figs
