"""
Save and restore the minimal fitting inputs and posterior outputs for BP3M.

Saving inputs allows a fit to be repeated at any point without re-running the
full data-loading and geometry-precomputation pipeline.  Saving results preserves
the fitted posterior for downstream analysis without re-fitting.

Layout of a saved checkpoint directory
---------------------------------------
<checkpoint_dir>/
    metadata.json           — image_names list + per-image metadata dicts
    gaia_catalog.csv        — Gaia catalog rows for stars observed in ≥1 image
    hst_sources/
        <img_name>.csv      — per-image HST source table (Gaia_id, X, Y, errors,
                              use_for_alignment flag)
    results/                — only present after save_results() is called
        r_hat.npy           — (n_r,) posterior image transformation vector
        C_r.npy             — (n_r, n_r) posterior covariance of r
        v_hat.npy           — (n_stars, 5) conditional stellar astrometry mean
        v_mean.npy          — (n_stars, 5) marginalised stellar astrometry mean
        v_cov.npy           — (n_stars, 5, 5) marginalised covariance (r-prop.)
        C_vT.npy            — (n_stars, 5, 5) conditional astrometry covariance
        image_names.json    — ordered list of image names (matches r ordering)
        K_matrices.npz      — {img: (n, N_V, N_R)} K matrices from final pass
        use_for_fit.npz     — {img: (n,) bool} stars used in final alignment solve
        star_indices.npz    — {img: (n,) int} global indices into stellar catalog

Public API
----------
save_inputs(solver, path)
    Write inputs from a fitted (or just-initialised) solver instance.

load_inputs(path)
    Reload inputs; returns the five arguments needed by BP3MSolver.__init__:
    (images, stars_per_image, gaia_catalog, star_id_to_idx, image_names, star_in_image)

save_results(r_hat, C_r, v_hat, v_mean, v_cov, C_vT, K_img, solver, path)
    Append posterior arrays under <path>/results/.

load_results(path)
    Load posterior arrays; returns
    (r_hat, C_r, v_hat, v_mean, v_cov, C_vT, image_names)

Example
-------
    from bp3m.checkpointing import save_inputs, load_inputs, save_results, load_results
    from bp3m.solver import BP3MSolver

    # After running the fit:
    save_inputs(solver, "checkpoints/fornax")
    save_results(r_hat, C_r, v_hat, v_mean, v_cov, C_vT, K_img, solver,
                 "checkpoints/fornax")

    # To repeat the fit later:
    images, spi, gcat, s2i, inames, sii = load_inputs("checkpoints/fornax")
    solver2 = BP3MSolver(images, spi, gcat, s2i, inames, sii)
    r_hat2, C_r2, *_ = solver2.fit()

    # Or just load the previous results:
    r_hat, C_r, v_hat, v_mean, v_cov, C_vT, image_names = load_results(
        "checkpoints/fornax")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

_HST_COLUMNS = [
    "Gaia_id", "X", "Y", "Y_orig",
    "x_hst_err", "y_hst_err", "xy_hst_corr",
    "use_for_alignment",
]
"""Columns from stars_per_image that are needed to reconstruct geometry.
Y_orig is included so that split_images_by_ccd can be applied to a loaded
checkpoint without needing to reload from the original CSVs."""

# Characters that are unsafe in file names (Windows + POSIX)
_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _safe_name(img_name: str) -> str:
    """Convert an image name to a safe file-system stem."""
    return _UNSAFE_RE.sub("_", img_name)


def _image_meta_to_json(meta: dict) -> dict:
    """Convert image metadata dict to a JSON-serialisable form."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _image_meta_from_json(meta: dict) -> dict:
    """Restore plain Python types from JSON-loaded image metadata."""
    return {k: v for k, v in meta.items()}


# ── Saving ────────────────────────────────────────────────────────────────────

def save_inputs(solver, path: str | Path) -> None:
    """
    Save the minimal fitting inputs from *solver* to *path*.

    The solver can be in any state (just initialised, mid-fit, or
    post-fit) — only the static inputs are written.

    Parameters
    ----------
    solver : BP3MSolver (or BP3MSolverSparse)
        A solver instance (must have `images`, `stars_per_image`,
        `gaia_cat`, `star_id_to_idx`, `image_names` attributes).
    path : str or Path
        Destination directory (created if it does not exist).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    hst_dir = path / "hst_sources"
    hst_dir.mkdir(exist_ok=True)

    # ── 1. Image metadata + image_names ──────────────────────────────────────
    payload = {
        "image_names": list(solver.image_names),
        "images": {
            img: _image_meta_to_json(solver.images[img])
            for img in solver.image_names
        },
    }
    with open(path / "metadata.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    # ── 2. Gaia catalog (only rows for observed stars) ────────────────────────
    observed_ids: set = set()
    for img in solver.image_names:
        df_img = solver.stars_per_image[img]
        observed_ids.update(df_img["Gaia_id"].values.tolist())

    gaia_mask = solver.gaia_cat["Gaia_id"].isin(observed_ids)
    solver.gaia_cat[gaia_mask].to_csv(path / "gaia_catalog.csv", index=False)

    # ── 3. Per-image HST source tables ────────────────────────────────────────
    for img in solver.image_names:
        df_img = solver.stars_per_image[img]
        # Keep only the columns actually needed; tolerate missing optional ones
        cols = [c for c in _HST_COLUMNS if c in df_img.columns]
        safe = _safe_name(img)
        df_img[cols].to_csv(hst_dir / f"{safe}.csv", index=False)

    print(f"  Inputs saved to '{path}' "
          f"({len(solver.image_names)} images, "
          f"{gaia_mask.sum()} Gaia sources)")


def save_results(
    r_hat: np.ndarray,
    C_r: np.ndarray,
    v_hat: np.ndarray,
    v_mean: np.ndarray,
    v_cov: np.ndarray,
    C_vT: np.ndarray,
    K_img: dict,
    solver,
    path: str | Path,
) -> None:
    """
    Save posterior fitting results to *path*/results/.

    Parameters
    ----------
    r_hat   : (n_r,)          posterior image transformation vector
    C_r     : (n_r, n_r)      posterior covariance of r
    v_hat   : (n_stars, 5)    conditional stellar astrometry (at r_hat)
    v_mean  : (n_stars, 5)    marginalised stellar astrometry mean
    v_cov   : (n_stars, 5, 5) marginalised covariance (r-propagation only)
    C_vT    : (n_stars, 5, 5) conditional astrometry covariance
    K_img   : dict            {image_name: (n, N_V, N_R) ndarray or None}
                              K matrices from the final solver pass
    solver  : BP3MSolver       (used to record image_names and use_for_fit)
    path    : str or Path      checkpoint directory (need not exist yet)
    """
    res_dir = Path(path) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    np.save(res_dir / "r_hat.npy",  r_hat)
    np.save(res_dir / "C_r.npy",    C_r)
    np.save(res_dir / "v_hat.npy",  v_hat)
    np.save(res_dir / "v_mean.npy", v_mean)
    np.save(res_dir / "v_cov.npy",  v_cov)
    np.save(res_dir / "C_vT.npy",   C_vT)

    with open(res_dir / "image_names.json", "w") as fh:
        json.dump(list(solver.image_names), fh, indent=2)

    # K matrices, per-image star indices, and use_for_fit flags
    _k_data   = {}
    _use_data = {}
    _idx_data = {}
    for img in solver.image_names:
        if K_img.get(img) is None:
            continue
        d = solver._img_data[img]
        _k_data[img]   = K_img[img]
        _use_data[img] = d["use_for_fit"]
        _idx_data[img] = d["sidx"]

    np.savez(res_dir / "K_matrices.npz",   **_k_data)
    np.savez(res_dir / "use_for_fit.npz",  **_use_data)
    np.savez(res_dir / "star_indices.npz", **_idx_data)

    print(f"  Results saved to '{res_dir}'")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_inputs(path: str | Path):
    """
    Reload fitting inputs from a checkpoint directory.

    Returns
    -------
    images : dict
        {image_name: dict of metadata}
    stars_per_image : dict
        {image_name: pd.DataFrame with HST source columns}
    gaia_catalog : pd.DataFrame
    star_id_to_idx : dict
        {Gaia_id: integer index into gaia_catalog}
    image_names : list[str]
    star_in_image : dict
        {Gaia_id: list of image names}  (rebuilt from HST source tables)
    """
    path = Path(path)

    # ── metadata ─────────────────────────────────────────────────────────────
    with open(path / "metadata.json") as fh:
        payload = json.load(fh)

    image_names: list[str] = payload["image_names"]
    images: dict = {
        img: _image_meta_from_json(meta)
        for img, meta in payload["images"].items()
    }

    # ── Gaia catalog ──────────────────────────────────────────────────────────
    gaia_catalog = pd.read_csv(path / "gaia_catalog.csv")

    # Rebuild star_id_to_idx in the same order as the saved catalog
    star_id_to_idx: dict = {
        int(gid): i
        for i, gid in enumerate(gaia_catalog["Gaia_id"])
    }

    # ── Per-image HST source tables ───────────────────────────────────────────
    hst_dir = path / "hst_sources"
    stars_per_image: dict = {}
    star_in_image: dict[int, list] = {}

    for img in image_names:
        safe = _safe_name(img)
        csv_path = hst_dir / f"{safe}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"HST source table not found: {csv_path}\n"
                f"(checkpoint may be incomplete or image name mapping changed)"
            )
        df = pd.read_csv(csv_path)
        stars_per_image[img] = df
        for gid in df["Gaia_id"].values:
            star_in_image.setdefault(int(gid), []).append(img)

    print(f"  Inputs loaded from '{path}' "
          f"({len(image_names)} images, "
          f"{len(gaia_catalog)} Gaia sources)")

    return images, stars_per_image, gaia_catalog, star_id_to_idx, image_names, star_in_image


def load_results(path: str | Path):
    """
    Load posterior fitting results from a checkpoint directory.

    Returns
    -------
    r_hat        : (n_r,) ndarray
    C_r          : (n_r, n_r) ndarray
    v_hat        : (n_stars, 5) ndarray
    v_mean       : (n_stars, 5) ndarray
    v_cov        : (n_stars, 5, 5) ndarray
    C_vT         : (n_stars, 5, 5) ndarray
    image_names  : list[str]  (ordered, matches r_hat block layout)
    """
    res_dir = Path(path) / "results"
    if not res_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {res_dir}\n"
            "Run save_results() first."
        )

    r_hat       = np.load(res_dir / "r_hat.npy")
    C_r         = np.load(res_dir / "C_r.npy")
    v_hat       = np.load(res_dir / "v_hat.npy")
    v_mean      = np.load(res_dir / "v_mean.npy")
    v_cov       = np.load(res_dir / "v_cov.npy")
    C_vT        = np.load(res_dir / "C_vT.npy")

    with open(res_dir / "image_names.json") as fh:
        image_names = json.load(fh)

    print(f"  Results loaded from '{res_dir}' "
          f"({len(image_names)} images, "
          f"n_r={len(r_hat)}, n_stars={len(v_hat)})")

    return r_hat, C_r, v_hat, v_mean, v_cov, C_vT, image_names
