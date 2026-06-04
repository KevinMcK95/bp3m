"""
Diagnostic plots for star influence on image transformation parameters.

Produced after BP3M convergence from compute_star_influence() output.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_influence_diagnostics(
    influence_df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 20,
) -> None:
    """
    Generate diagnostic figures from compute_star_influence() output.

    Writes PNG files to output_dir:
        plots_influence_residual_maps.png   — per-image Cook's D maps
        plots_influence_cooks_d.png         — Cook's D distributions and rankings
        plots_influence_param_<p>.png       — per-image parameter influence maps (one per param)
    """
    output_dir = Path(output_dir)
    df = influence_df.copy()
    image_names = sorted(df["image_name"].unique())
    n_images    = len(image_names)

    _plot_residual_maps(df, image_names, n_images, output_dir)
    _plot_cooks_d(df, image_names, top_n, output_dir)
    _plot_param_influence(df, image_names, n_images, output_dir)


def _n_cols(n_images: int) -> tuple[int, int]:
    """Return (ncols, nrows) for a grid of n_images panels."""
    ncols = min(4, n_images)
    nrows = (n_images + ncols - 1) // ncols
    return ncols, nrows


def _img_label(img: str) -> str:
    return img.replace("_hi", " hi").replace("_lo", " lo")


def _plot_residual_maps(df, image_names, n_images, output_dir):
    ncols, nrows = _n_cols(n_images)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4.0 * nrows + 0.6),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    d_max = float(np.nanpercentile(df["cooks_d"].values, 99))
    d_max = max(d_max, 1e-10)
    norm  = mcolors.LogNorm(vmin=1e-6, vmax=d_max)

    for ax_idx, img in enumerate(image_names):
        ax  = axes_flat[ax_idx]
        sub = df[df["image_name"] == img]
        use = sub["use_for_fit"].values
        x   = sub["X_c"].values
        y   = sub["Y_c"].values
        cd  = sub["cooks_d"].values

        sc = ax.scatter(x[use], y[use], c=cd[use],
                        cmap="hot_r", norm=norm,
                        s=10, zorder=2, rasterized=True)
        if (~use).any():
            ax.scatter(x[~use], y[~use], c="dodgerblue",
                       marker="x", s=25, linewidths=0.8,
                       zorder=3, label="clipped", alpha=0.8)

        ax.set_title(_img_label(img), fontsize=10)
        ax.set_xlabel("X [px]", fontsize=9)
        ax.set_ylabel("Y [px]", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[n_images:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="hot_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes_flat[:n_images], shrink=0.6, pad=0.02,
                      aspect=30, location="right")
    cb.set_label("Cook's D", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Per-image Cook's D  (hot = high influence;  × = clipped)",
                 fontsize=11, y=1.01)
    _save(fig, output_dir / "plots_influence_residual_maps.png")


def _plot_cooks_d(df, image_names, top_n, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # ── Panel 1: Cook's D histogram ───────────────────────────────────────
    ax = axes[0]
    used    = df[df["use_for_fit"]]
    clipped = df[~df["use_for_fit"]]
    cd_pos  = df["cooks_d"].values
    cd_pos  = cd_pos[cd_pos > 0]
    if len(cd_pos) > 1:
        bins = np.logspace(np.log10(cd_pos.min()), np.log10(cd_pos.max() + 1e-12), 50)
    else:
        bins = 30
    ax.hist(used["cooks_d"].values, bins=bins, color="steelblue",
            alpha=0.75, label=f"used ({len(used)})", density=True)
    if len(clipped):
        ax.hist(clipped["cooks_d"].values, bins=bins, color="tomato",
                alpha=0.75, label=f"clipped ({len(clipped)})", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Cook's D", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Cook's D distribution", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    # ── Panel 2: Cook's D vs magnitude ───────────────────────────────────
    ax = axes[1]
    if "mag" in df.columns:
        m  = used["mag"].values
        cd = used["cooks_d"].values
        sr = used["sigma_resid"].values
        ok = np.isfinite(m) & np.isfinite(cd) & (cd > 0)
        if ok.any():
            sc = ax.scatter(m[ok], np.log10(cd[ok]),
                            c=np.log10(np.maximum(sr[ok], 1e-9)),
                            cmap="RdYlGn_r", s=8, alpha=0.6, vmin=-1, vmax=1,
                            rasterized=True)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("log σ_resid", fontsize=9)
            cb.ax.tick_params(labelsize=8)
        ax.set_xlabel("HST magnitude", fontsize=10)
        ax.set_ylabel("log₁₀(Cook's D)", fontsize=10)
        ax.set_title("Cook's D vs magnitude\n(colour = log σ_resid)", fontsize=11)
        ax.tick_params(labelsize=9)
    else:
        ax.text(0.5, 0.5, "No magnitude column", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)

    # ── Panel 3: Top-N ranked Cook's D ───────────────────────────────────
    ax = axes[2]
    top = (df.groupby("Gaia_id")["cooks_d"].max()
             .sort_values(ascending=False)
             .head(top_n))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.values, color="steelblue", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(gid)[-8:] for gid in top.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("max Cook's D across images", fontsize=10)
    ax.set_title(f"Top {top_n} most influential stars\n(last 8 digits of Gaia ID)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Star influence diagnostics", fontsize=12)
    _save(fig, output_dir / "plots_influence_cooks_d.png")


def _plot_param_influence(df, image_names, n_images, output_dir):
    infl_cols = [c for c in df.columns if c.startswith("infl_") and
                 c[5:] in ("a", "b", "c", "d", "w", "z")]
    if not infl_cols:
        return

    ncols, nrows = _n_cols(n_images)

    for col in infl_cols:
        pname    = col[5:]
        used_df  = df[df["use_for_fit"]]
        vals     = used_df[col].values
        fin_vals = np.abs(vals[np.isfinite(vals)])
        vmax     = float(np.nanpercentile(fin_vals, 99)) if len(fin_vals) else 1e-10
        vmax     = max(vmax, 1e-10)
        norm     = mcolors.Normalize(vmin=-vmax, vmax=vmax)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 4.0 * nrows + 0.6),
            constrained_layout=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for ax_idx, img in enumerate(image_names):
            ax  = axes_flat[ax_idx]
            sub = df[df["image_name"] == img]
            use = sub["use_for_fit"].values

            sc = ax.scatter(
                sub["X_c"].values[use], sub["Y_c"].values[use],
                c=sub[col].values[use],
                cmap="RdBu_r", norm=norm,
                s=12, zorder=2, rasterized=True)
            if (~use).any():
                ax.scatter(sub["X_c"].values[~use], sub["Y_c"].values[~use],
                           c="gray", marker="x", s=20, linewidths=0.8,
                           zorder=3, alpha=0.6)

            ax.set_title(_img_label(img), fontsize=10)
            ax.set_xlabel("X [px]", fontsize=9)
            ax.set_ylabel("Y [px]", fontsize=9)
            ax.tick_params(labelsize=8)

        for ax in axes_flat[n_images:]:
            ax.set_visible(False)

        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes_flat[:n_images], shrink=0.6, pad=0.02,
                          aspect=30, location="right")
        cb.set_label(f"infl_{pname}  (δr)", fontsize=10)
        cb.ax.tick_params(labelsize=8)

        fig.suptitle(f"Per-image influence on parameter '{pname}'  (red/blue = ±δr)",
                     fontsize=11, y=1.01)
        _save(fig, output_dir / f"plots_influence_param_{pname}.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
