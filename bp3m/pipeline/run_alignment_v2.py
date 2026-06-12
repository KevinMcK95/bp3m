"""
run_alignment_v2.py  —  BP3M v2 alignment using HST cross-match master catalog.

Reads master_combined_v2.csv (output of hst_catalog_crossmatch Phase 5) and
runs a BP3M solve that includes both Gaia-matched stars and HST-only stars.

HST-only sources are introduced in a phased manner:
  Pre-inclusion phase (outer iterations 1 .. hst_enable_iter - 1):
    use_for_fit = False.  Gaia sources establish the transformation posterior.
    Per-detection residuals for HST-only sources are computed and soft-flagged
    if they exceed outlier_sigma.

  Transition (after iteration hst_enable_iter - 1):
    Soft-flagged detections (flagged in >= 1 iteration) are permanently removed.
    Remaining HST-only sources with n_detect_fit >= 2 and sigma_pm < hst_max_pm_unc
    have use_for_fit flipped to True.

  Post-inclusion (iteration hst_enable_iter onwards):
    HST-only and Gaia sources are treated identically by the solver.

Results are written to {output_dir}/{field}/BP3M_v2_results/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

def _ensure_bp3m(bp3m_dir=None):
    pass  # bp3m is installed as a package; no sys.path manipulation needed


# ── V2AlignmentCallback ───────────────────────────────────────────────────────

class V2AlignmentCallback:
    """
    Per-iteration callback for phased inclusion of HST-only sources in BP3M v2.

    Usage
    -----
    Pass an instance as the ``per_iter_callback`` argument to solver.fit().
    The callback is called as ``callback(solver, it_outer)`` at the end of
    each outer EM iteration (1-based).

    Pre-inclusion phase (it_outer < hst_enable_iter):
        Computes raw pixel residuals for HST-only detections against the current
        transformation.  Detections with residual > outlier_sigma * pixel_scale
        are soft-flagged (recorded but not yet removed).

    Transition (it_outer == hst_enable_iter):
        Permanently removes detections that were soft-flagged in any prior
        iteration.  Sources that still have n_detect_fit >= 2 have use_for_fit
        flipped to True.

    Post-inclusion (it_outer > hst_enable_iter):
        No action — the solver handles HST-only and Gaia sources identically.

    Parameters
    ----------
    hst_star_mask : (n_stars,) bool
        True for HST-only rows in the gaia_catalog (i.e. hst_only_mask from
        load_master_v2).
    hst_enable_iter : int
        1-based outer iteration at which HST-only sources are enabled.
        Default: 5.
    outlier_sigma : float
        Residual threshold in pixels (relative to the per-image pixel scale)
        for soft-flagging HST-only detections during the pre-inclusion phase.
        Default: 5.
    """

    def __init__(self, hst_star_mask: np.ndarray,
                 hst_enable_iter: int = 5,
                 outlier_sigma: float = 5.0,
                 pm_init: np.ndarray | None = None):
        self.hst_star_mask   = np.asarray(hst_star_mask, dtype=bool)
        self.hst_enable_iter = hst_enable_iter
        self.outlier_sigma   = outlier_sigma
        # pm_init: (n_stars, 2) array of (pmra_xmatch, pmdec_xmatch) for all stars;
        # only the HST-only rows matter.  NaN entries are skipped at transition.
        self.pm_init = pm_init  # may be None → fall back to v_survey as-is

        # soft_flags[img][k] — per-image per-detection soft-flag count
        # Set during __call__ for images containing HST-only detections.
        self._soft_flags: dict[str, np.ndarray] = {}
        self._enabled = False  # True once HST-only sources have been enabled

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_soft_flags(self, img: str, n: int) -> np.ndarray:
        """Return (or create) the soft-flag array for image `img` of length n."""
        if img not in self._soft_flags:
            self._soft_flags[img] = np.zeros(n, dtype=int)
        return self._soft_flags[img]

    def _compute_pixel_residuals(self, solver, img: str) -> np.ndarray | None:
        """
        Compute per-detection raw pixel residuals for image `img`.

        Returns (n,) residual magnitudes in pixels, or None if image has no data.
        """
        d = solver._img_data.get(img)
        if d is None:
            return None
        r_hat = solver._r_hat_current
        nr = solver.N_R
        j_idx = solver.image_names.index(img)
        r_j = r_hat[j_idx * nr: (j_idx + 1) * nr]

        X_mat = d["X_mat"]   # (n, 2, N_R)
        JU    = d["JU"]      # (n, 2, N_V)
        xys   = d["xys"]     # (n, 2) — observed Gaia-frame positions
        sidx  = d["sidx"]    # (n,) star indices

        # Current stellar astrometry estimate from solver
        # v_hat ≈ a_arr (set by _update_geometry in _inner_converge)
        # Access via the solver's cached geometry (xs, ys are the reference positions)
        # For residual purposes: x_pred = X_mat @ r_j - JU @ v_hat[sidx]
        # But v_hat isn't stored on solver directly.  Instead use a_arr which
        # is effectively v_hat after the last inner convergence.
        # We access it from the solver's last-computed h-vector implicitly:
        # Simplification: use xs/ys from _img_data as the reference, ignoring v_hat.
        # This is equivalent to the initial residual screening but using current r_j.
        x_pred = np.einsum("nkl,l->nk", X_mat, r_j)  # (n, 2)
        x_resid = xys - x_pred                         # (n, 2)

        resid_px = np.hypot(x_resid[:, 0], x_resid[:, 1])
        return resid_px

    # ── Main callback ─────────────────────────────────────────────────────────

    def __call__(self, solver, it_outer: int) -> None:
        """Called by solver.fit() at the end of each outer EM iteration."""
        if self._enabled:
            return  # post-inclusion: nothing to do

        hst_star_idx = np.where(self.hst_star_mask)[0]
        if len(hst_star_idx) == 0:
            return

        hst_star_set = set(hst_star_idx.tolist())

        # ── Pre-inclusion: soft-flag outlier detections ───────────────────────
        if it_outer < self.hst_enable_iter:
            n_flagged = 0
            for img in solver.image_names:
                d = solver._img_data.get(img)
                if d is None:
                    continue
                sidx = d["sidx"]  # (n,) — indices into gaia_catalog
                # Find HST-only detections in this image
                hst_mask = np.array([s in hst_star_set for s in sidx])
                if not hst_mask.any():
                    continue

                resid_px = self._compute_pixel_residuals(solver, img)
                if resid_px is None:
                    continue

                soft = self._get_soft_flags(img, len(sidx))
                # Flag detections with large residuals
                pscale = solver.images[img].get("orig_pixel_scale", 50.0)  # mas/pix
                # Residuals are in Gaia pseudo-image pixels (same pixel scale)
                threshold_px = self.outlier_sigma  # dimensionless in pixel units
                bad = hst_mask & (resid_px > threshold_px)
                soft[bad] += 1
                n_flagged += int(bad.sum())

            print(f"  [V2Callback] iter {it_outer}: soft-flagged {n_flagged} "
                  f"HST-only detections across all images")

        # ── Transition: permanently remove soft-flagged, enable qualifying sources
        elif it_outer == self.hst_enable_iter:
            print(f"  [V2Callback] iter {it_outer}: enabling HST-only sources")

            # Count how many valid detections each HST-only source retains
            # after permanently removing soft-flagged ones.
            hst_detect_count = {int(i): 0 for i in hst_star_idx}

            for img in solver.image_names:
                d = solver._img_data.get(img)
                if d is None:
                    continue
                sidx = d["sidx"]
                use_fit = d["use_for_fit"]     # current (n,) bool
                n = len(sidx)

                hst_mask = np.array([s in hst_star_set for s in sidx])
                if not hst_mask.any():
                    continue

                soft = self._soft_flags.get(img, np.zeros(n, dtype=int))

                # Permanently remove detections that were soft-flagged >= 1 time
                permanently_bad = hst_mask & (soft > 0)
                if permanently_bad.any():
                    # Mark them as excluded by setting use_for_fit to False
                    # (they start False anyway; this ensures _update_use_for_fit
                    # can't re-admit them by setting a hard ceiling in influence_excl)
                    excl = d.get("influence_excl")
                    if excl is None:
                        excl = np.zeros(n, dtype=bool)
                        d["influence_excl"] = excl
                    excl |= permanently_bad

                # Count valid remaining detections for each HST-only source
                valid_hst = hst_mask & ~permanently_bad
                for k in np.where(valid_hst)[0]:
                    star_idx = int(sidx[k])
                    if star_idx in hst_detect_count:
                        hst_detect_count[star_idx] += 1

            # Enable sources meeting the detection threshold — astrometry tier only.
            # HST-only sources get stellar PM estimates from the Gaia-constrained
            # transformation but do NOT enter use_for_fit (alignment tier).
            #
            # WHY NOT use_for_fit: HST-only have only a diffuse PM prior, so their
            # MAP PM (a_align) is data-driven and noisy. Unlike Gaia stars where the
            # tight PM prior guarantees per-image Schur complement cancellation, noisy
            # a_align for HST-only produces small per-iteration transformation biases.
            # With 2800 HST-only vs 100 Gaia stars (20:1 in Pal5), these biases
            # accumulate and destabilize the Gaia-constrained solution over many
            # iterations. This design is field-agnostic: dense GCs (E3), streams
            # (Pal5), and pure field-star fields all work correctly.
            n_enabled = 0
            for img in solver.image_names:
                d = solver._img_data.get(img)
                if d is None:
                    continue
                sidx = d["sidx"]
                n = len(sidx)

                hst_mask = np.array([s in hst_star_set for s in sidx])
                if not hst_mask.any():
                    continue

                soft = self._soft_flags.get(img, np.zeros(n, dtype=int))
                excl = d.get("influence_excl", np.zeros(n, dtype=bool))

                use_astrom = d.get("use_for_astrom",
                                   d["use_for_fit"].copy())
                for k in np.where(hst_mask)[0]:
                    star_idx = int(sidx[k])
                    if (hst_detect_count.get(star_idx, 0) >= 2 and
                            not excl[k] and soft[k] == 0):
                        # use_for_fit stays False → no transformation influence
                        use_astrom[k] = True
                        n_enabled += 1

                d["use_for_astrom"] = use_astrom
                # use_for_fit intentionally NOT modified here

            # Seed v_survey PM for newly-enabled HST-only sources from their
            # crossmatch PM estimates.  Without this, all HST-only stars start
            # at pmra=pmdec=0, creating a bulk systematic residual that pulls the
            # transformation away from the Gaia-constrained solution.
            if self.pm_init is not None:
                n_seeded = 0
                seeded_stars: set[int] = set()
                for idx in hst_detect_count:
                    if hst_detect_count[idx] < 2:
                        continue
                    pmra_seed  = float(self.pm_init[idx, 0])
                    pmdec_seed = float(self.pm_init[idx, 1])
                    if np.isfinite(pmra_seed) and np.isfinite(pmdec_seed):
                        solver.v_survey[idx, 2] = pmra_seed
                        solver.v_survey[idx, 3] = pmdec_seed
                        # Keep v_survey-derived quantities in sync
                        seeded_stars.add(idx)
                        n_seeded += 1
                if n_seeded > 0:
                    # Recompute the cached C_survey_inv_dot_v for seeded rows so
                    # the solver's next update starts from the correct prior term.
                    seeded = np.array(sorted(seeded_stars), dtype=int)
                    solver.C_survey_inv_dot_v[seeded] = np.einsum(
                        'nij,nj->ni',
                        solver.C_survey_inv[seeded],
                        solver.v_survey[seeded],
                    )
                    print(f"  [V2Callback] Seeded PM from xmatch for {n_seeded} "
                          f"HST-only sources")

            self._enabled = True
            total_hst_enabled = sum(1 for cnt in hst_detect_count.values() if cnt >= 2)
            print(f"  [V2Callback] Enabled {n_enabled} HST-only detections "
                  f"across {total_hst_enabled} sources")


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def _plot_soft_weights(z_weights_out, solver, plot_dir):
    """
    Two-panel diagnostic for soft-weight IRLS results.

    Left:  histogram of z_det values (0-1), split by Gaia-matched vs HST-only.
    Right: bar chart of N_eff (sum of z) per image, with N_total overlaid.

    Saved to plot_dir / 'soft_weights_diagnostic.png'.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_z     = []
    all_gaia  = []   # True for detections from Gaia-matched stars
    neff_per  = {}   # {img: (n_eff, n_total)}

    for img, z in z_weights_out.items():
        if z is None:
            continue
        d         = solver._img_data[img]
        gaia_flag = d.get("use_for_align_init", np.ones(len(z), dtype=bool))
        survivors = d["use_for_fit_max"]
        z_surv    = z[survivors]
        g_surv    = np.asarray(gaia_flag)[survivors]
        all_z.extend(z_surv.tolist())
        all_gaia.extend(g_surv.tolist())
        neff_per[img] = (float(z_surv.sum()), int(survivors.sum()))

    all_z    = np.array(all_z)
    all_gaia = np.array(all_gaia, dtype=bool)

    total_det = len(all_z)
    total_eff = float(all_z.sum())
    pct       = 100.0 * total_eff / max(total_det, 1)

    # Sort images by N_eff descending for the bar chart
    imgs_sorted = sorted(neff_per, key=lambda i: neff_per[i][0], reverse=True)
    neff_vals   = [neff_per[i][0] for i in imgs_sorted]
    ntot_vals   = [neff_per[i][1] for i in imgs_sorted]

    # Infer nu from solver if available; otherwise fall back to a generic label
    nu = getattr(solver, '_soft_nu_used', '?')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: z histogram ────────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    if all_gaia.any():
        ax.hist(all_z[all_gaia],  bins=bins, alpha=0.6, label='Gaia-matched', color='steelblue')
    if (~all_gaia).any():
        ax.hist(all_z[~all_gaia], bins=bins, alpha=0.6, label='HST-only',     color='darkorange')
    ax.axvline(0.5, color='red', lw=1, ls='--', label='z=0.5')
    ax.set_xlabel('z_det  (Student-t weight)')
    ax.set_ylabel('N detections')
    ax.set_title('Detection weight distribution')
    ax.legend(fontsize=9)

    # ── Right: N_eff per image ───────────────────────────────────────────────
    ax2 = axes[1]
    x   = np.arange(len(imgs_sorted))
    ax2.bar(x, ntot_vals, color='lightgrey', label='N_total')
    ax2.bar(x, neff_vals, color='steelblue', alpha=0.8, label='N_eff')
    ax2.set_xticks(x)
    ax2.set_xticklabels([i.split('_')[0] for i in imgs_sorted],
                        rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Detections')
    ax2.set_title('N_eff per image (sorted)')
    ax2.legend(fontsize=9)

    fig.suptitle(
        f'Soft-weight IRLS: detection weights (ν={nu})\n'
        f'N_det={total_det}, N_eff={total_eff:.0f} ({pct:.1f}%)',
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(plot_dir) / 'soft_weights_diagnostic.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Full-catalog residuals ─────────────────────────────────────────────────────

def _save_full_catalog_residuals(output_bp3m, solver, image_names, r_hat,
                                  data_root, field_name):
    """
    Compute and save per-image GDC-frame residuals for ALL master_combined_v2 stars.

    Unlike detections.npz (which only covers BP3M solver stars), this covers the
    full ~120k-star master catalog — including faint HST-only sources where CTE
    effects are strongest.

    Saves: detections_catalog.npz
        {img}_X_c       : (n,) centered GDC x pixel  (x_gdc - 2048)
        {img}_Y_c       : (n,) centered GDC y pixel  (y_gdc - 2048)
        {img}_dx_gdc    : (n,) x residual in GDC frame [pixels]
        {img}_dy_gdc    : (n,) y residual in GDC frame [pixels]
        {img}_mag_inst  : (n,) instrumental mag (mag_gdc from detections_F814W.csv)
        {img}_in_bp3m   : (n,) bool — star in BP3M solver (Gaia-matched alignment star)
    """
    import pandas as pd
    from astropy.time import Time
    from bp3m.astro_utils import (
        plane_project, plane_project_jacobian, plane_project_tangent_derivs,
        get_tele_position, get_parallax_factors, compute_poly_jacobian,
    )

    xmatch_dir = data_root / field_name / "hst_xmatch"
    det_path   = xmatch_dir / "detections_F814W.csv"
    mcat_path  = xmatch_dir / "master_combined_v2.csv"

    if not det_path.exists() or not mcat_path.exists():
        print("  _save_full_catalog_residuals: required files not found — skipping")
        return

    print("\n  Computing full-catalog GDC residuals (all master_combined_v2 stars)...")

    # ── Load ────────────────────────────────────────────────────────────────────
    det  = pd.read_csv(det_path,  dtype={'gaia_source_id': np.int64})
    mcat = pd.read_csv(mcat_path, dtype={'gaia_source_id': np.int64}, low_memory=False)

    star_cols = ['ra_xmatch', 'dec_xmatch', 'pmra_xmatch', 'pmdec_xmatch',
                 'parallax_xmatch', 'epoch_ref_xmatch']

    # ── Merge detections with master catalog ────────────────────────────────────
    # Part 1: Gaia-matched detections (gaia_source_id != 0)
    mcat_gaia = (mcat[mcat['gaia_source_id'] != 0]
                 [['gaia_source_id'] + star_cols].copy())
    det_gaia  = det[det['gaia_source_id'].to_numpy(np.int64) != 0][
                    ['sub_name', 'gaia_source_id', 'catalog_index',
                     'x_gdc', 'y_gdc', 'mag_gdc']].copy()
    det_gaia_m = det_gaia.merge(mcat_gaia, on='gaia_source_id', how='inner')

    # Part 2: HST-only detections (gaia_source_id == 0)
    # Build reverse index: (sub_name, catalog_index) → master catalog row
    # Parse hst_indices_F814W column
    print("    Parsing hst_indices_F814W for HST-only lookup...")
    mcat_hst_src = mcat[mcat['hst_indices_F814W'].notna()][
                       ['hst_indices_F814W'] + star_cols].copy()
    mcat_hst_src = mcat_hst_src.reset_index(drop=True)

    # Explode "sub_name:catalog_index" entries — separator may be ',' or ';'
    mcat_hst_src['_entries'] = (mcat_hst_src['hst_indices_F814W']
                                .str.replace(';', ',', regex=False)
                                .str.split(','))
    mcat_exploded = mcat_hst_src.explode('_entries')
    mcat_exploded = mcat_exploded[mcat_exploded['_entries'].str.contains(':', na=False)]
    entry_parts = mcat_exploded['_entries'].str.split(':', expand=True)
    mcat_exploded = mcat_exploded.copy()
    mcat_exploded['sub_name']       = entry_parts[0]
    mcat_exploded['catalog_index']  = entry_parts[1].astype(np.int64)
    rev_idx = mcat_exploded[['sub_name', 'catalog_index'] + star_cols].reset_index(drop=True)

    det_hst = det[det['gaia_source_id'].to_numpy(np.int64) == 0][
                  ['sub_name', 'gaia_source_id', 'catalog_index',
                   'x_gdc', 'y_gdc', 'mag_gdc']].copy()
    det_hst['catalog_index'] = det_hst['catalog_index'].astype(np.int64)
    det_hst_m = det_hst.merge(rev_idx, on=['sub_name', 'catalog_index'], how='inner')

    # Combine and sort by sub_name for per-image grouping
    det_all = pd.concat([det_gaia_m, det_hst_m], ignore_index=True)
    del det_gaia_m, det_hst_m, mcat_exploded, rev_idx  # free memory

    n_matched = len(det_all)
    n_gaia_m  = (det_all['gaia_source_id'].to_numpy(np.int64) != 0).sum()
    print(f"    Matched {n_matched:,} detections "
          f"({n_gaia_m:,} Gaia-matched + {n_matched - n_gaia_m:,} HST-only)")

    # BP3M solver Gaia IDs (for in_bp3m flag on Gaia-matched detections)
    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)

    # ── Per-image residual computation ─────────────────────────────────────────
    out_arrays = {}
    n_r = solver.N_R
    poly_order = solver.poly_order

    det_by_img = det_all.groupby('sub_name', sort=False)

    for j_idx, img in enumerate(image_names):
        if img not in det_by_img.groups:
            continue
        meta = solver.images.get(img)
        if meta is None:
            continue

        img_df = det_by_img.get_group(img)

        # Image geometry
        ra0    = float(meta['ra0'])
        dec0   = float(meta['dec0'])
        pscale = float(meta['orig_pixel_scale'])   # mas/pixel
        hst_time = Time(float(meta['hst_time_mjd']), format='mjd')
        hst_yr   = float(hst_time.jyear)
        tele_xyz = meta.get('tele_XYZ') or get_tele_position(hst_time, curr_id='earth')

        # r_j for this image
        cs  = j_idx * n_r
        r_j = r_hat[cs : cs + n_r]

        # Per-detection arrays
        x_gdc    = img_df['x_gdc'].to_numpy(float)
        y_gdc    = img_df['y_gdc'].to_numpy(float)
        mag_arr  = img_df['mag_gdc'].to_numpy(float)
        ra_arr   = img_df['ra_xmatch'].to_numpy(float)
        dec_arr  = img_df['dec_xmatch'].to_numpy(float)
        pmra_arr = img_df['pmra_xmatch'].to_numpy(float)
        pmdec_arr= img_df['pmdec_xmatch'].to_numpy(float)
        plx_arr  = img_df['parallax_xmatch'].to_numpy(float)
        epoch_arr= img_df['epoch_ref_xmatch'].to_numpy(float)  # Julian year
        gids_arr = img_df['gaia_source_id'].to_numpy(np.int64)

        n = len(img_df)

        # Centered GDC pixel positions
        X_c = x_gdc - 2048.0
        Y_c = y_gdc - 2048.0

        # Gaia reference position in pseudo-image frame (pix)
        xs, ys = plane_project(ra_arr, dec_arr, ra0, dec0, pscale)
        xys = np.stack([xs, ys], axis=1)   # (n, 2)

        # Jacobian J: (n, 2, 2) in pix/mas
        J_arr = plane_project_jacobian(ra_arr, dec_arr, ra0, dec0, pscale)

        # Tangent-point derivatives (pscale/1000 → arcsec units, matching solver)
        dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = plane_project_tangent_derivs(
            ra_arr, dec_arr, ra0, dec0, pscale / 1000.0)

        # Parallax factors
        plx_ra_arr, plx_dec_arr = get_parallax_factors(ra_arr, dec_arr, tele_xyz)

        # Time baseline: HST epoch minus star reference epoch (Julian years)
        dt_arr = hst_yr - epoch_arr

        # X_mat: (n, 2, n_r) design matrix — vectorized for poly_order=1
        if poly_order == 1:
            X_mat = np.zeros((n, 2, n_r))
            X_mat[:, 0, 0] = X_c;         X_mat[:, 0, 1] = Y_c
            X_mat[:, 0, 4] = 1.0
            X_mat[:, 0, 6] = dxs_dra0;    X_mat[:, 0, 7] = dxs_ddec0
            X_mat[:, 1, 2] = X_c;         X_mat[:, 1, 3] = Y_c
            X_mat[:, 1, 5] = 1.0
            X_mat[:, 1, 6] = dys_dra0;    X_mat[:, 1, 7] = dys_ddec0
        else:
            from bp3m.astro_utils import build_X_matrix
            X_mat = np.array([
                build_X_matrix(X_c[k], Y_c[k],
                               dxs_dra0[k], dxs_ddec0[k],
                               dys_dra0[k], dys_ddec0[k], poly_order)
                for k in range(n)])

        # U matrix: (n, 2, 5) — stellar motion time-evolution
        U_arr = np.zeros((n, 2, 5))
        U_arr[:, 0, 0] = 1.0;          U_arr[:, 1, 1] = 1.0
        U_arr[:, 0, 2] = dt_arr;       U_arr[:, 1, 3] = dt_arr
        U_arr[:, 0, 4] = plx_ra_arr;   U_arr[:, 1, 4] = plx_dec_arr

        # JU = J @ U: (n, 2, 5)
        JU = np.einsum('nij,njk->nik', J_arr, U_arr)

        # Approximate stellar motion vector: Δα*=0, Δδ=0 since ra/dec_xmatch
        # is the MAP position; only PM and parallax contribute.
        v_approx = np.zeros((n, 5))
        v_approx[:, 2] = pmra_arr
        v_approx[:, 3] = pmdec_arr
        v_approx[:, 4] = plx_arr

        # Predicted pseudo-image position and residual
        pred = (np.einsum('nij,j->ni', X_mat, r_j)
                - np.einsum('nij,nj->ni', JU, v_approx))
        resid_pseudo = xys - pred   # (n, 2)

        # Back-project residual to GDC frame
        if poly_order == 1:
            J_inv = np.linalg.inv(solver.R[img])   # (2, 2)
            dxy   = resid_pseudo @ J_inv.T          # (n, 2)
        else:
            J_loc = compute_poly_jacobian(r_j, X_c, Y_c, poly_order)
            J_inv = np.linalg.inv(J_loc)            # (n, 2, 2)
            dxy   = np.einsum('nij,nj->ni', J_inv, resid_pseudo)

        # In-BP3M flag: True for Gaia-matched stars used in BP3M alignment
        in_bp3m = np.array([int(g) in bp3m_gaia_ids for g in gids_arr.tolist()],
                           dtype=bool)

        out_arrays[f'{img}_X_c']      = X_c.astype(np.float32)
        out_arrays[f'{img}_Y_c']      = Y_c.astype(np.float32)
        out_arrays[f'{img}_dx_gdc']   = dxy[:, 0].astype(np.float32)
        out_arrays[f'{img}_dy_gdc']   = dxy[:, 1].astype(np.float32)
        out_arrays[f'{img}_mag_inst'] = mag_arr.astype(np.float32)
        out_arrays[f'{img}_in_bp3m']  = in_bp3m

    if not out_arrays:
        print("  WARNING: no detections matched — detections_catalog.npz not saved")
        return

    out_path = output_bp3m / 'detections_catalog.npz'
    np.savez(out_path, **out_arrays)
    n_imgs  = sum(1 for k in out_arrays if k.endswith('_X_c'))
    n_total = sum(len(v) for k, v in out_arrays.items() if k.endswith('_X_c'))
    print(f"\n  Saved detections_catalog.npz: {n_imgs} images, "
          f"{n_total:,} detections → {out_path}")


# ── Main run function ─────────────────────────────────────────────────────────

def run_alignment_v2(
    output_dir: Path,
    field_name: str,
    n_iter: int = 20,
    n_samples: int = 1000,
    clip_sigma: float = 4.5,
    poly_order: int = 1,
    use_sparse: bool = False,
    no_prefilter: bool = False,
    no_plots: bool = False,
    hst_enable_iter: int = 5,
    hst_max_pm_unc: float = 5.0,
    hst_max_per_image: int = 1000,
    outlier_sigma: float = 5.0,
    use_influence_clip: bool = True,
    influence_d_thresh: float = 1.0,   # same as V1; auto-scaled by V1/V2 C_r ratio at runtime
    influence_sigma_min: float = 2.0,
    hst_pm_sigma_diffuse: float = 100.0,
    bp3m_dir: Path | None = None,
    pos_err_floor: float = 5e-3,
    det_chi2_threshold: float | None = None,
    use_soft_weights: bool = False,
    student_t_nu: float = 50.0,
) -> Path:
    """
    Run BP3M v2 alignment using the master_combined_v2.csv cross-match catalog.

    Parameters
    ----------
    output_dir    : pipeline root directory
    field_name    : field subdirectory name
    n_iter        : maximum EM outer iterations
    n_samples     : posterior samples for marginalisation
    clip_sigma    : MAD sigma for outlier rejection (0 = disabled)
    poly_order    : polynomial order for image transformation
    use_sparse    : use sparse Schur-complement solver
    no_prefilter  : skip Phase-0 pre-filter pass
    no_plots      : skip diagnostic plot generation
    hst_enable_iter : outer iteration at which HST-only sources are enabled
    hst_max_pm_unc  : global PM uncertainty cut for HST-only eligibility (mas/yr)
    hst_max_per_image : per-image cap on HST-only source count
    outlier_sigma   : residual threshold (pixels) for soft-flagging HST-only dets
    use_influence_clip : enable test-4 Cook's D influence clipping
    influence_d_thresh : Cook's D threshold
    influence_sigma_min : minimum sigma_resid for influence flagging
    bp3m_dir      : override default bp3m location
    pos_err_floor : minimum positional uncertainty in pixels
    det_chi2_threshold : if set, exclude (star, image) pairs whose per-detection
        chi2 from Phase 4 exceeds this value.  Requires det_chi2 column in
        master_combined_v2.csv.  Suggested: 9.0 (3σ).

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_v2_results/)
    """
    _ensure_bp3m(bp3m_dir)

    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    import pandas as pd

    from .data_loader_master import load_master_v2

    data_root   = Path(output_dir)
    output_bp3m = data_root / field_name / "BP3M_v2_results"
    output_bp3m.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 50)
    print("BP3M v2: alignment with HST-only sources")
    print("─" * 50)
    print(f"  n_iter={n_iter}  clip_sigma={clip_sigma}  poly_order={poly_order}")
    print(f"  hst_enable_iter={hst_enable_iter}  hst_max_pm_unc={hst_max_pm_unc}  "
          f"hst_max_per_image={hst_max_per_image}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading v2 master catalog data for '{field_name}'...")
    images, stars_per_image, gaia_catalog, hst_only_mask = load_master_v2(
        data_root, field_name,
        hst_max_pm_unc=hst_max_pm_unc,
        hst_max_per_image=hst_max_per_image,
        pos_err_floor=pos_err_floor,
        det_chi2_threshold=det_chi2_threshold,
    )

    if not images:
        raise RuntimeError(
            f"No usable images found for '{field_name}'. "
            "Check that master_combined_v2.csv exists and has detections."
        )

    # ── Build index maps ──────────────────────────────────────────────────────
    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    imgs = {n: images[n] for n in image_names if n in images}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    print(f"  Stars: {len(gaia_catalog)} "
          f"({int((~hst_only_mask).sum())} Gaia + {int(hst_only_mask.sum())} HST-only)"
          f"   Images: {len(image_names)}")

    # ── Inject v1 BP3M transformation + alpha as initialization ──────────────
    # Load converged (a,b,c,d,w,z) and alpha from the previous v1 BP3M run so
    # that Phase 0 uses those posteriors for outlier screening rather than the
    # rough fast_cross_match solution from transformation.csv.
    v1_bp3m_dir   = data_root / field_name / "BP3M_results"
    v1_xform_path = v1_bp3m_dir / "image_transformations.csv"
    v1_abcdwz: dict[str, np.ndarray] = {}
    v1_alpha:  dict[str, float]      = {}
    # v1 stellar astrometry (MAP conditional posteriors) used for Phase 0 chi2 validation
    v1_stellar_astrom: pd.DataFrame | None = None
    if v1_xform_path.exists():
        v1_df = pd.read_csv(v1_xform_path)
        for _, row in v1_df.iterrows():
            img_key = str(row["image_name"])
            v1_abcdwz[img_key] = np.array([
                float(row["a"]), float(row["b"]),
                float(row["c"]), float(row["d"]),
                float(row["w"]), float(row["z"]),
            ])
            v1_alpha[img_key] = float(row["alpha"]) if "alpha" in row.index else 1.0
        n_matched = sum(1 for k in imgs if k in v1_abcdwz)
        print(f"  Loaded v1 BP3M results: {len(v1_abcdwz)} images, "
              f"{n_matched}/{len(imgs)} matched to current image list.")
        # Deep-copy each meta dict so per-sub-name overrides don't bleed across.
        imgs = {
            sub: dict(meta) for sub, meta in imgs.items()
        }
        for sub, meta in imgs.items():
            if sub in v1_abcdwz:
                meta["fcm_abcdwz"] = v1_abcdwz[sub]

        # Load v1 MAP stellar astrometry for chi2 validation in Phase 0
        _v1_astrom_path = v1_bp3m_dir / "stellar_astrometry.csv"
        if _v1_astrom_path.exists():
            _load_cols = ['Gaia_id', 'pmra_bp3m_cond', 'pmdec_bp3m_cond',
                          'parallax_bp3m_cond', 'chi2_hst', 'n_det_chi2']
            v1_stellar_astrom = pd.read_csv(
                _v1_astrom_path,
                usecols=lambda c: c in _load_cols,
                dtype={'Gaia_id': np.int64},
            )
        # Load V1 C_r to estimate typical transformation uncertainty scale.
        # V2 starts from V1's converged solution so its C_r is smaller than V1's
        # early-iteration C_r.  We scale influence_d_thresh so that the effective
        # absolute-shift threshold is the same as in V1.
        _v1_cr_path = v1_bp3m_dir / "C_r.npy"
        _v1_cr_scale: float = 1.0
        if _v1_cr_path.exists() and influence_d_thresh == 3.0:
            # Only auto-scale if the user hasn't overridden d_thresh manually
            # (default 3.0 means "auto-scale from V1 C_r").
            try:
                _v1_cr = np.load(_v1_cr_path)
                _n_r_per = _v1_cr.shape[0] // max(len(v1_abcdwz), 1)
                # Typical w-parameter (translation) uncertainty = median sqrt(C_r[4,4])
                _cr_w_vals = []
                for _j in range(len(v1_abcdwz)):
                    _cs = _j * _n_r_per
                    _cr_j = _v1_cr[_cs:_cs+_n_r_per, _cs:_cs+_n_r_per]
                    if _cr_j.shape[0] > 4:
                        _cr_w_vals.append(float(np.sqrt(max(_cr_j[4, 4], 0.0))))
                if _cr_w_vals:
                    _v1_cr_scale = float(np.median(_cr_w_vals))
                    print(f"  V1 C_r scale (median σ_w): {_v1_cr_scale:.4e} px  "
                          f"→ influence_d_thresh auto-scaled to {_v1_cr_scale / 1e-3:.1f}×1e-3")
            except Exception as _exc:
                print(f"  Warning: could not load V1 C_r for d_thresh scaling: {_exc}")
    else:
        print(f"  Note: no v1 BP3M results found at {v1_xform_path}; "
              "using transformation.csv initialization.")

    # ── Initialise solver ─────────────────────────────────────────────────────
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(
        imgs, filtered_spi, gaia_catalog,
        star_id_to_idx, image_names, star_in_image,
        poly_order=poly_order,
    )

    # ── Override diffuse PM prior for HST-only stars ──────────────────────────
    # The solver is initialised with _SIGMA_PM=100 mas/yr for all 2p/HST-only
    # stars.  For the v2 alignment HST-only sources we use a wider prior so
    # their MAP PM is determined primarily by their HST detections rather than
    # being pulled toward the 0-centred prior.  This does NOT affect the
    # original BP3M v1 run or the master catalogue building.
    if hst_pm_sigma_diffuse != 100.0:
        hst_star_indices = np.where(hst_only_mask)[0]
        if len(hst_star_indices) > 0:
            sigma_pm_inv2 = float(hst_pm_sigma_diffuse) ** -2
            solver._C_VG_inv_per_star[hst_star_indices, 2] = sigma_pm_inv2
            solver._C_VG_inv_per_star[hst_star_indices, 3] = sigma_pm_inv2
            solver._sigma_diff_per_star[hst_star_indices, 2] = float(hst_pm_sigma_diffuse)
            solver._sigma_diff_per_star[hst_star_indices, 3] = float(hst_pm_sigma_diffuse)
            print(f"  HST-only PM diffuse prior: σ = {hst_pm_sigma_diffuse:.0f} mas/yr "
                  f"(default 100) for {len(hst_star_indices)} stars")

    # ── Apply saved v1 alpha inflation to solver._img_data ───────────────────
    # Sets the starting alpha_applied so the EM loop continues from where v1
    # left off rather than resetting to alpha=1.
    if v1_alpha:
        n_alpha_set = 0
        for img in image_names:
            if img in v1_alpha and img in solver._img_data and solver._img_data[img] is not None:
                a0 = max(1.0, v1_alpha[img])
                solver._img_data[img]["alpha_applied"] = a0
                solver._img_data[img]["C_hst"] = (
                    a0**2 * solver._img_data[img]["C_hst_orig"]
                )
                n_alpha_set += 1
        if n_alpha_set:
            print(f"  Applied v1 alpha inflation to {n_alpha_set} images "
                  f"(median α={np.median([v1_alpha[i] for i in image_names if i in v1_alpha]):.3f}).")

    # ── Callback ──────────────────────────────────────────────────────────────
    # Build pm_init from xmatch PM values stored in gaia_catalog for HST-only
    # rows (pmra_xmatch / pmdec_xmatch columns added by data_loader_master).
    # These seed solver.v_survey at the transition so HST-only sources don't
    # start at PM=0, which would pull the transformation away from the Gaia
    # solution.  NaN entries are skipped (v_survey left at 0 for those stars).
    _n_stars = len(gaia_catalog)
    pm_init = np.full((_n_stars, 2), np.nan)
    if "pmra_xmatch" in gaia_catalog.columns:
        pm_init[:, 0] = pd.to_numeric(
            gaia_catalog["pmra_xmatch"],  errors='coerce').fillna(np.nan).values
        pm_init[:, 1] = pd.to_numeric(
            gaia_catalog["pmdec_xmatch"], errors='coerce').fillna(np.nan).values
    n_pm_seeds = int(np.isfinite(pm_init[:, 0]).sum())
    print(f"  PM init seeds from xmatch: {n_pm_seeds}/{int(hst_only_mask.sum())} "
          f"HST-only sources have finite pmra_xmatch")

    # hst_only_mask is aligned to gaia_catalog; solver uses the same ordering
    callback = V2AlignmentCallback(
        hst_star_mask=hst_only_mask,
        hst_enable_iter=hst_enable_iter,
        outlier_sigma=outlier_sigma,
        pm_init=pm_init,
    )

    # ── Phase 0 (fixed-transformation pre-filter) ─────────────────────────────
    # When v1 BP3M results are available, use the converged transformation
    # (already in r_init) WITHOUT updating it to screen bad detections.
    # This verifies the data is loaded correctly (residuals should be small)
    # and establishes a clean star set before Phase 1 solves for any updates.
    #
    # Residuals: resid = xys - X_mat @ r_init + JU @ v_survey_{pm,plx}
    # Outlier threshold: _PHASE0_SIGMA_THRESH × MAD-sigma, floored at 0.3 px.
    # When no v1 results exist, fall back to the standard solver prefilter.
    _PHASE0_SIGMA_THRESH = 5.0
    _run_solver_prefilter = not no_prefilter

    if v1_abcdwz and not no_prefilter:
        _run_solver_prefilter = False   # we handle Phase 0 ourselves below

        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)
        # Rebuild xys and JU to be consistent with the v1 transformation.
        # __init__ computed xys at the fast_cross_match initialization; without
        # this call, _solve_one_pass in Phase 1 sees residuals of size
        # (v1_transform − fcm_transform) rather than the true HST residuals,
        # producing a large spurious epoch-dependent drift (~0.5 px) that
        # reverses the v1 PM correction and corrupts the output PMs.
        solver._update_geometry(r_init_hat, solver.v_survey)

        print("\n Phase 0: fixed-transformation pre-filter (v1 BP3M posterior)")

        n_flagged_total = 0
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            j_idx = image_names.index(img)
            r_j   = r_init_hat[j_idx * solver.N_R:(j_idx + 1) * solver.N_R]

            X_mat = d["X_mat"]
            xys   = d["xys"]
            JU    = d["JU"]
            sidx  = d["sidx"]
            use   = d["use_for_fit"].copy()

            _v_pm = np.zeros_like(solver.v_survey[sidx])
            _v_pm[:, 2:] = solver.v_survey[sidx, 2:]
            motion    = np.einsum("nij,nj->ni", JU, _v_pm)
            x_pred    = np.einsum("nkl,l->nk", X_mat, r_j) - motion
            resid_mag = np.hypot(*(xys - x_pred).T)   # (n,)

            if use.any():
                r_align    = resid_mag[use]
                mad_sigma  = np.median(np.abs(r_align - np.median(r_align))) / 0.6745
                thresh     = max(_PHASE0_SIGMA_THRESH * mad_sigma, 0.3)
                bad        = use & (resid_mag > thresh)

                n_flag = int(bad.sum())
                n_flagged_total += n_flag
                if n_flag:
                    d["use_for_fit"][bad]     = False
                    d["use_for_fit_max"][bad] = False
                    # Keep use_for_astrom in sync: detections removed from
                    # alignment must also be removed from astrometry so that
                    # H_vv (which uses use_any = fit | astrom) is consistent
                    # with h_align. Outliers in use_for_astrom inflate H_vv
                    # and corrupt the Schur complement cancellation.
                    if "use_for_astrom" in d:
                        d["use_for_astrom"][bad] = False

                print(f"  {img}: {int(use.sum())-n_flag}/{int(use.sum())} kept  "
                      f"med={np.median(r_align):.4f}px  "
                      f"σ={mad_sigma:.4f}px  thresh={thresh:.4f}px  "
                      f"flagged={n_flag}")

        print(f"  Phase 0 total flagged: {n_flagged_total} detections")


        # ── Phase 0 astrometry validation ─────────────────────────────────────
        # Apply r_init (= v1 BP3M posterior) to all detections and solve for
        # per-star astrometry via MAP in pixel space using the JU Jacobian,
        # with the same Gaia/diffuse prior as Phase 4 of hst_catalog_crossmatch.
        # The solution must be identical to the master_combined_v2 pmra_xmatch
        # values (both built from the same transformation + same detections).
        _master_v2_path = (data_root / field_name / "hst_xmatch"
                           / "master_combined_v2.csv")
        if _master_v2_path.exists():
            try:
                _mv2 = pd.read_csv(
                    _master_v2_path,
                    usecols=lambda c: c in ('gaia_source_id', 'pmra_xmatch',
                                             'pmdec_xmatch', 'sigma_pmra_xmatch',
                                             'sigma_pmdec_xmatch'),
                    dtype={'gaia_source_id': np.int64},
                    low_memory=False,
                )
                _gaia_rows = _mv2[_mv2['gaia_source_id'] > 0].copy()
                _gaia_rows['_gid'] = _gaia_rows['gaia_source_id'].astype(np.int64)
                _gaia_rows = _gaia_rows[_gaia_rows['_gid'] > 0].set_index('_gid')

                # Accumulate per-star MAP normal equations in pixel space.
                # Model: b_j = JU_j @ v_i + noise, where
                #   b_j  = X_mat_j @ r_k − xys_j  (transformation residual, (2,))
                #   JU_j = (2,5) Jacobian maps [Δα0,Δδ0,pmra,pmdec,plx] → pixels
                # Same formulation as Phase 4 of hst_catalog_crossmatch.
                _n_s   = solver.n_stars
                _N_V   = 5
                _AtCA  = np.zeros((_n_s, _N_V, _N_V))
                _AtCb  = np.zeros((_n_s, _N_V))
                _n_wls = np.zeros(_n_s, dtype=int)

                for _ji, _img in enumerate(image_names):
                    _d = solver._img_data.get(_img)
                    if _d is None:
                        continue
                    _use = _d.get('use_for_fit',
                                  np.ones(len(_d['sidx']), dtype=bool))
                    if not _use.any():
                        continue
                    _r_j   = r_init_hat[_ji * solver.N_R:(_ji + 1) * solver.N_R]
                    _sidx  = _d['sidx'][_use]
                    _xys   = _d['xys'][_use]
                    _X_mat = _d['X_mat'][_use]
                    _JU    = _d['JU'][_use]        # (n, 2, 5)
                    _C_inv = np.linalg.inv(_d['C_hst'][_use])  # (n, 2, 2)
                    _b     = np.einsum('nkl,l->nk', _X_mat, _r_j) - _xys  # (n, 2)
                    # JU^T @ C_inv: (n, 5, 2)
                    _JtCi  = np.einsum('nki,nkj->nij', _JU, _C_inv)
                    # JU^T @ C_inv @ JU: (n, 5, 5)
                    _JtCiJ = np.einsum('nik,nkj->nij', _JtCi, _JU)
                    # JU^T @ C_inv @ b: (n, 5)
                    _JtCib = np.einsum('nik,nk->ni', _JtCi, _b)
                    np.add.at(_AtCA, _sidx, _JtCiJ)
                    np.add.at(_AtCb, _sidx, _JtCib)
                    np.add.at(_n_wls, _sidx, 1)

                # Add the same priors as Phase 4 of hst_catalog_crossmatch:
                #   5p/6p Gaia: full Gaia covariance inverse (solver.C_survey_inv)
                #   2p/HST-only: Gaia position prior + diffuse PM prior
                # For HST-only stars the diffuse sigma may differ from the 100 mas/yr
                # used for 2p Gaia stars (hst_pm_sigma_diffuse parameter).
                _diffuse_pm_inv_gaia = (1.0 / 100.0) ** 2   # Gaia 2p: always 100 mas/yr
                _diffuse_pm_inv_hst  = (1.0 / hst_pm_sigma_diffuse) ** 2  # HST-only
                for _i in range(_n_s):
                    if _n_wls[_i] < 1:
                        continue
                    # Gaia prior (full 5×5 for 5p/6p; position-only for 2p)
                    _AtCA[_i] += solver.C_survey_inv[_i]
                    _AtCb[_i] += solver.C_survey_inv_dot_v[_i]
                    # For non-5p stars: add diffuse PM prior (prior mean = 0)
                    # HST-only stars use hst_pm_sigma_diffuse; Gaia 2p use 100 mas/yr.
                    if not solver.full_gaia_astrometry[_i]:
                        _pm_inv = (_diffuse_pm_inv_hst if hst_only_mask[_i]
                                   else _diffuse_pm_inv_gaia)
                        _AtCA[_i, 2, 2] += _pm_inv
                        _AtCA[_i, 3, 3] += _pm_inv

                # Chi2 thresholds for astrometric outlier flagging.
                # Stars exceeding these are removed from Phase 1 (use_for_fit=False).
                _PHASE0_CHI2_THRESH_5P = 20.5  # 5p/6p df=5: chi2(5, p=0.001) = 20.52
                _PHASE0_CHI2_THRESH_2P = 13.8  # 2p    df=2: chi2(2, p=0.001) = 13.82

                gc_ids = gaia_catalog['Gaia_id'].values.astype(np.int64)
                _fga   = solver.full_gaia_astrometry   # bool (n_stars,)

                # Per-star chi2 containers (for printing + flagging)
                _chi2_5p: list[float] = []   # 5p/6p: full 5-param vs Gaia prior (df=5)
                _chi2_2p: list[float] = []   # 2p:    PM vs diffuse prior         (df=2)
                _outlier_star_idxs: set[int] = set()

                # Catalogue sanity-check containers (5p/6p diagnostic)
                _pmra_wls  = []
                _pmdec_wls = []
                _pmra_cat  = []
                _pmdec_cat = []
                _sig_wls_r = []
                _sig_wls_d = []

                # v1 BP3M vs Gaia prior containers (5p/6p diagnostic)
                _chi2_v1_5: list[float] = []
                # Build v1 posterior lookup (vectorised — iterrows corrupts int64)
                # Tuple: (pmra_cond, pmdec_cond, plx_cond, dra_cond, ddec_cond)
                # Position offsets: use delta_racosdec_bp3m / delta_dec_bp3m
                # (unconditional, but position is tightly constrained so ≈ conditional)
                _v1_pm_lookup: dict[int, tuple[float, float, float, float, float]] = {}
                if v1_stellar_astrom is not None and len(v1_stellar_astrom) > 0:
                    _v1_ids   = v1_stellar_astrom['Gaia_id'].values
                    _v1_pmra  = pd.to_numeric(v1_stellar_astrom['pmra_bp3m_cond'],
                                              errors='coerce').values
                    _v1_pmdec = pd.to_numeric(v1_stellar_astrom['pmdec_bp3m_cond'],
                                              errors='coerce').values
                    _v1_plx = (pd.to_numeric(v1_stellar_astrom['parallax_bp3m_cond'],
                                             errors='coerce').values
                               if 'parallax_bp3m_cond' in v1_stellar_astrom.columns
                               else np.full(len(_v1_ids), np.nan))
                    _v1_dra = (pd.to_numeric(v1_stellar_astrom['delta_racosdec_bp3m'],
                                             errors='coerce').values
                               if 'delta_racosdec_bp3m' in v1_stellar_astrom.columns
                               else np.zeros(len(_v1_ids)))
                    _v1_ddec = (pd.to_numeric(v1_stellar_astrom['delta_dec_bp3m'],
                                              errors='coerce').values
                                if 'delta_dec_bp3m' in v1_stellar_astrom.columns
                                else np.zeros(len(_v1_ids)))
                    for _k in range(len(_v1_ids)):
                        _vid = int(_v1_ids[_k])
                        if _vid > 0 and np.isfinite(_v1_pmra[_k]) and np.isfinite(_v1_pmdec[_k]):
                            _v1_pm_lookup[_vid] = (
                                float(_v1_pmra[_k]),
                                float(_v1_pmdec[_k]),
                                float(_v1_plx[_k]),
                                float(_v1_dra[_k])  if np.isfinite(_v1_dra[_k])  else 0.0,
                                float(_v1_ddec[_k]) if np.isfinite(_v1_ddec[_k]) else 0.0,
                            )

                for _i, _gid in enumerate(gc_ids):
                    if _n_wls[_i] < 1 or _gid <= 0:
                        continue
                    try:
                        _v_sol = np.linalg.solve(_AtCA[_i], _AtCb[_i])
                        _C_sol = np.linalg.inv(_AtCA[_i])
                    except np.linalg.LinAlgError:
                        continue

                    # ── Chi2 vs prior (outlier flagging for both 5p/6p and 2p) ──
                    if _fga[_i]:
                        # 5p/6p: full df=5 chi2 using Gaia 5-param covariance inverse.
                        # C_survey_inv[i] is the full 5×5 Gaia covariance inverse for
                        # [Δα₀cosδ, Δδ₀, pmra, pmdec, plx]; v_survey[:2]=0 (position
                        # prior centred at Gaia position, so offset = 0).
                        _dv5 = _v_sol - solver.v_survey[_i]
                        try:
                            _c2 = float(_dv5 @ solver.C_survey_inv[_i] @ _dv5)
                        except Exception:
                            _c2 = np.nan
                        if np.isfinite(_c2):
                            _chi2_5p.append(_c2)
                            if _c2 > _PHASE0_CHI2_THRESH_5P:
                                _outlier_star_idxs.add(_i)
                    else:
                        # 2p: chi2 = (pmra^2 + pmdec^2) / sigma_diffuse^2  (df=2)
                        # Position chi2 is ≈0 (tight prior dominates), so only
                        # the PM component is informative for outlier detection.
                        _c2 = float(_v_sol[2]**2 + _v_sol[3]**2) * _diffuse_pm_inv_gaia
                        _chi2_2p.append(_c2)
                        if np.isfinite(_c2) and _c2 > _PHASE0_CHI2_THRESH_2P:
                            _outlier_star_idxs.add(_i)

                    # ── Catalogue sanity check (5p/6p diagnostic) ─────────────
                    if _fga[_i] and _gid in _gaia_rows.index:
                        _row = _gaia_rows.loc[_gid]
                        _pmra_c  = float(_row.get('pmra_xmatch',  np.nan) or np.nan)
                        _pmdec_c = float(_row.get('pmdec_xmatch', np.nan) or np.nan)
                        _sig_c_r = float(_row.get('sigma_pmra_xmatch',  np.nan) or np.nan)
                        _sig_c_d = float(_row.get('sigma_pmdec_xmatch', np.nan) or np.nan)
                        if np.isfinite(_pmra_c) and np.isfinite(_pmdec_c):
                            _pmra_wls.append(_v_sol[2])
                            _pmdec_wls.append(_v_sol[3])
                            _pmra_cat.append(_pmra_c)
                            _pmdec_cat.append(_pmdec_c)
                            _sig_wls_r.append(float(np.sqrt(max(_C_sol[2, 2], 0.0))))
                            _sig_wls_d.append(float(np.sqrt(max(_C_sol[3, 3], 0.0))))

                    # ── v1 BP3M posterior vs Gaia prior (5p/6p diagnostic) ────
                    # Full df=5 chi2: use Gaia C_survey_inv with v1 5-param solution
                    # (dra/ddec from unconditional BP3M offsets; PM/plx from conditional).
                    if _fga[_i] and int(_gid) in _v1_pm_lookup:
                        _v1t = _v1_pm_lookup[int(_gid)]  # (pmra, pmdec, plx, dra, ddec)
                        _dv5_v1 = np.array([
                            _v1t[3] - float(solver.v_survey[_i, 0]),  # Δα₀cosδ
                            _v1t[4] - float(solver.v_survey[_i, 1]),  # Δδ₀
                            _v1t[0] - float(solver.v_survey[_i, 2]),  # pmra
                            _v1t[1] - float(solver.v_survey[_i, 3]),  # pmdec
                            (float(_v1t[2]) - float(solver.v_survey[_i, 4]))
                            if np.isfinite(_v1t[2]) else 0.0,          # plx (0 if 2p in v1)
                        ])
                        try:
                            _chi2_v1_5.append(float(
                                _dv5_v1 @ solver.C_survey_inv[_i] @ _dv5_v1))
                        except Exception:
                            pass

                # ── Chi2 outlier summary ───────────────────────────────────────
                print()
                if _chi2_5p:
                    _c2_5p = np.array([x for x in _chi2_5p if np.isfinite(x)])
                    _n_flag_5p = sum(1 for _i in _outlier_star_idxs if _fga[_i])
                    print(f"  Phase 0 chi2 (5p/6p, df=5 vs Gaia prior, "
                          f"{len(_c2_5p)} stars):")
                    print(f"    med={np.median(_c2_5p):.2f}  "
                          f"p95={np.percentile(_c2_5p, 95):.2f}  "
                          f"frac>{_PHASE0_CHI2_THRESH_5P:.0f}="
                          f"{float((_c2_5p > _PHASE0_CHI2_THRESH_5P).sum())/max(len(_c2_5p),1):.3f}  "
                          f"[expected med≈3.36  p95≈11.07]  flagged={_n_flag_5p}")
                if _chi2_2p:
                    _c2_2p = np.array([x for x in _chi2_2p if np.isfinite(x)])
                    _n_flag_2p = sum(1 for _i in _outlier_star_idxs if not _fga[_i])
                    print(f"  Phase 0 chi2 (2p, df=2 PM vs diffuse 100 mas/yr prior, "
                          f"{len(_c2_2p)} stars):")
                    print(f"    med={np.median(_c2_2p):.4f}  "
                          f"p95={np.percentile(_c2_2p, 95):.4f}  "
                          f"frac>{_PHASE0_CHI2_THRESH_2P:.0f}="
                          f"{float((_c2_2p > _PHASE0_CHI2_THRESH_2P).sum())/max(len(_c2_2p),1):.3f}  "
                          f"[expected med≈1.39  p95≈5.99]  flagged={_n_flag_2p}")

                # ── Flag outlier stars for Phase 1+ ───────────────────────────
                # Stars with chi2 > threshold have their detections permanently
                # removed so they cannot bias the Phase 1 transformation update.
                _n_det_removed = 0
                for _img in image_names:
                    _d = solver._img_data.get(_img)
                    if _d is None:
                        continue
                    for _k, _si in enumerate(_d["sidx"]):
                        if _si in _outlier_star_idxs and _d["use_for_fit"][_k]:
                            _d["use_for_fit"][_k] = False
                            _d["use_for_fit_max"][_k] = False
                            if "use_for_astrom" in _d:
                                _d["use_for_astrom"][_k] = False
                            _n_det_removed += 1
                if _outlier_star_idxs:
                    print(f"  Phase 0 astrometric outliers flagged: "
                          f"{len(_outlier_star_idxs)} stars, "
                          f"{_n_det_removed} detections removed from Phase 1+")
                else:
                    print(f"  Phase 0 astrometric outliers: none (all stars consistent "
                          f"with prior at chi2 ≤ {_PHASE0_CHI2_THRESH:.0f})")

                # ── Catalogue sanity check (diagnostic) ───────────────────────
                if _pmra_wls:
                    _pmra_wls  = np.array(_pmra_wls)
                    _pmdec_wls = np.array(_pmdec_wls)
                    _pmra_cat  = np.array(_pmra_cat)
                    _pmdec_cat = np.array(_pmdec_cat)
                    _sig_wls_r = np.array(_sig_wls_r)
                    _sig_wls_d = np.array(_sig_wls_d)
                    _dpmra  = _pmra_wls  - _pmra_cat
                    _dpmdec = _pmdec_wls - _pmdec_cat
                    print(f"\n  Phase 0 MAP vs master_combined_v2 (5p/6p, "
                          f"{len(_pmra_wls)} stars):")
                    print(f"    pmra:  MAP med={np.nanmedian(_pmra_wls):+.3f}  "
                          f"cat med={np.nanmedian(_pmra_cat):+.3f}  "
                          f"Δmed={np.nanmedian(_dpmra):+.4f}  "
                          f"σ_wls med={np.nanmedian(_sig_wls_r):.4f} mas/yr")
                    print(f"    pmdec: MAP med={np.nanmedian(_pmdec_wls):+.3f}  "
                          f"cat med={np.nanmedian(_pmdec_cat):+.3f}  "
                          f"Δmed={np.nanmedian(_dpmdec):+.4f}  "
                          f"σ_wls med={np.nanmedian(_sig_wls_d):.4f} mas/yr")
                    if abs(np.nanmedian(_dpmra)) > 0.1 or abs(np.nanmedian(_dpmdec)) > 0.1:
                        print("    WARNING: MAP PM offset from master catalogue "
                              "> 0.1 mas/yr — check prior setup or detection set.")
                    else:
                        print("    OK: MAP consistent with master catalogue.")

                # ── v1 BP3M vs Gaia prior diagnostic ──────────────────────────
                if _chi2_v1_5:
                    _c2_v1 = np.array([x for x in _chi2_v1_5 if np.isfinite(x)])
                    print(f"\n  Phase 0 v1 BP3M posterior vs Gaia prior "
                          f"(5p/6p df=5, {len(_c2_v1)} stars):")
                    print(f"    chi2:  "
                          f"med={np.median(_c2_v1):.2f}  "
                          f"p95={np.percentile(_c2_v1, 95):.2f}  "
                          f"frac>11={float((_c2_v1 > 11.07).sum())/max(len(_c2_v1), 1):.3f}  "
                          f"[expected med≈3.36  p95≈11.07]")

            except Exception as _val_exc:
                import traceback
                print(f"  Phase 0 astrometry validation skipped: {_val_exc}")
                traceback.print_exc()
            print()

    # ── Scale influence_d_thresh by V1/V2 C_r ratio ──────────────────────────
    # Cook's D = (X^T Cs^{-1} resid)^T C_r (X^T Cs^{-1} resid) / N_R.
    # V1 starts from a rough transformation (large C_r); V2 starts from the
    # converged V1 solution (small C_r).  The same D_thresh therefore removes
    # different detections: V1 rarely exceeds D>1 because C_r is large, while
    # V2 can exceed D>1 for perfectly fine detections.
    # Fix: scale D_thresh so that the ABSOLUTE shift threshold (in pixels) is
    # the same as V1's, i.e.  D_thresh_V2 = D_thresh × (C_r_V1 / C_r_V2).
    _infl_d_thresh_scaled = influence_d_thresh
    if v1_abcdwz and n_iter > 0:
        v1_cr_path = data_root / field_name / "BP3M_results" / "C_r.npy"
        if v1_cr_path.exists():
            try:
                _v1_cr = np.load(v1_cr_path)
                _nr    = solver.N_R
                _n_img = len(image_names)
                # Median σ_w (sqrt of C_r[4,4] per image) as scale indicator
                _v1_sigma_w = float(np.median([
                    np.sqrt(max(_v1_cr[j*_nr+4, j*_nr+4], 0.0))
                    for j in range(min(_n_img, _v1_cr.shape[0] // _nr))
                ]))
                # One solve pass at V1 init to get current C_r
                _r_init_for_cr = np.concatenate([
                    solver._img_data[img]["r_init"] for img in image_names])
                _, _C_r_v2, _, _, _ = solver._solve_one_pass(_r_init_for_cr)
                _v2_sigma_w = float(np.median([
                    np.sqrt(max(_C_r_v2[j*_nr+4, j*_nr+4], 0.0))
                    for j in range(_n_img)
                ]))
                if _v2_sigma_w > 0 and _v1_sigma_w > 0:
                    # D = (X Cs^{-1} resid)^T C_r (X Cs^{-1} resid) / N_R.
                    # Larger C_r → larger D for the same physical resid.
                    # To apply the same physical shift threshold as V1 (D_thresh_V1 × σ_w_V1),
                    # V2 needs D_thresh_V2 = D_thresh_V1 × (σ_w_V2 / σ_w_V1).
                    _cr_ratio = _v2_sigma_w / _v1_sigma_w
                    _infl_d_thresh_scaled = influence_d_thresh * _cr_ratio
                    print(f"  Influence clipping: σ_w(V1)={_v1_sigma_w:.4e}  "
                          f"σ_w(V2)={_v2_sigma_w:.4e}  "
                          f"C_r ratio(V2/V1)={_cr_ratio:.2f}  "
                          f"→ influence_d_thresh={_infl_d_thresh_scaled:.2f} "
                          f"(base={influence_d_thresh:.1f})")
            except Exception as _exc:
                print(f"  Warning: C_r ratio scaling failed ({_exc}); "
                      f"using influence_d_thresh={influence_d_thresh}")

    # ── Enable HST-only sources for n_iter=0 ─────────────────────────────────
    # When n_iter=0, Phase 2 outer iterations never run, so the V2AlignmentCallback
    # transition (which sets use_for_astrom=True and seeds v_survey PM) never fires.
    # Trigger it manually so the single _solve_one_pass computes HST-only astrometry
    # from the fixed v1 transformation rather than returning PM=0 (prior mean).
    if n_iter == 0 and callback is not None:
        print("  n_iter=0: enabling HST-only sources for astrometry before solve...")
        callback(solver, callback.hst_enable_iter)

    # ── Build z_init from Phase-6 chi2 values (soft-weight warm start) ────────
    # If soft weights are enabled AND the per-image DataFrames contain a
    # det_chi2 column (written by data_loader_master when master_combined_v2.csv
    # has the det_chi2 column), pre-compute initial z values from those chi2
    # values.  This warm-starts the IRLS so that detections already identified
    # as poor fits during catalogue building start with low z, rather than
    # starting at z=1 and waiting for the seed solve to detect them.
    _z_init: dict | None = None
    if use_soft_weights:
        _z_init = {}
        for _img, _df_raw in solver.stars_per_image.items():
            if _img not in solver._img_data or solver._img_data[_img] is None:
                continue
            if "det_chi2" not in _df_raw.columns:
                _z_init = None   # column absent — fall back to seed-solve weights
                break
            # Apply the same Gaia_id mask that setup_images uses so the row
            # count matches _img_data[img]["n"].
            _df = _df_raw[_df_raw["Gaia_id"].isin(solver.star_id_to_idx)].reset_index(drop=True)
            _chi2 = _df["det_chi2"].to_numpy(float)
            _nu   = student_t_nu
            _z    = np.minimum(1.0, (_nu + 2.0) / (_nu + np.where(np.isfinite(_chi2), _chi2, 0.0)))
            # Detections without Phase-6 chi2 (HST-only and catalogue gaps) are
            # treated as good fits: z=1.0.  They will be updated after the seed
            # solve by _update_soft_weights once the callback has seeded their PMs.
            _z[~np.isfinite(_chi2)] = 1.0
            # Use use_for_fit_max (Phase-0 hard floor, includes HST-only) rather
            # than use_for_astrom (which is still False for HST-only before the
            # callback fires inside solver.fit).  Phase-0-rejected detections are
            # in use_for_fit_max but will be overridden to z=0 by
            # _update_soft_weights after the seed solve — their small contribution
            # to Δz at iter 1 is acceptable.
            _d = solver._img_data[_img]
            _mask = _d["use_for_fit_max"].astype(float)
            _z_init[_img] = _z * _mask
        if _z_init is not None:
            _n_imgs_with_chi2 = sum(1 for v in _z_init.values() if v is not None)
            _total_det = sum(int(v.sum()) for v in _z_init.values() if v is not None)
            print(f"  Soft-weight warm start: Phase-6 chi2 available for "
                  f"{_n_imgs_with_chi2} images ({_total_det} detections)")

    # ── Fit ───────────────────────────────────────────────────────────────────
    clip = clip_sigma if clip_sigma > 0 else None
    t0 = time.time()
    # When HST-only stars are enabled mid-run (n_iter >= hst_enable_iter), require
    # at least hst_enable_iter+3 outer iterations before allowing early stopping.
    # This ensures the EM has time to converge after the new sources are added.
    _min_outer = max(hst_enable_iter + 3, 4) if n_iter >= hst_enable_iter else 4

    r_hat, C_r, v_hat, C_vT, a_arr, K_img, z_weights_out = solver.fit(
        n_iter=n_iter,
        clip_sigma=clip,
        inflate_hst_errors=True,   # alpha computed from data, same as v1
        inflate_from_iter=0,       # v1 alpha is pre-validated: allow decrease from iter 0
        min_outer_iters=_min_outer,
        hst_fit_sigma_mult=0.5,    # HST-only must have tighter residuals to stay in alignment
        prefilter=_run_solver_prefilter,
        use_influence_clip=use_influence_clip,
        influence_d_thresh=_infl_d_thresh_scaled,
        influence_sigma_min=influence_sigma_min,
        use_two_tier=True,         # enables use_for_astrom tracking
        per_iter_callback=callback,
        use_soft_weights=use_soft_weights,
        student_t_nu=student_t_nu,
        z_init=_z_init,
    )
    print(f"  Fit completed in {time.time()-t0:.1f}s")

    # ── Soft-weight output ────────────────────────────────────────────────────
    if use_soft_weights and z_weights_out is not None:
        import pandas as _pd_sw
        rows = []
        for img, z in z_weights_out.items():
            if z is None:
                continue
            d = solver._img_data[img]
            for k in range(len(z)):
                rows.append({
                    'image':    img,
                    'star_idx': int(d['sidx'][k]),
                    'z_det':    float(z[k]),
                })
        zdf = _pd_sw.DataFrame(rows)
        zdf.to_csv(output_bp3m / 'soft_weights.csv', index=False)
        print(f"  Saved: soft_weights.csv  ({len(zdf)} detection weights)")
        _plot_dir = output_bp3m / 'plots'
        _plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_soft_weights(z_weights_out, solver, _plot_dir)
        except Exception as _exc:
            print(f"  WARNING: soft_weights_diagnostic plot failed — {_exc}")

    # ── Sample posteriors ─────────────────────────────────────────────────────
    print(f"  Drawing {n_samples} posterior samples...")
    r_samp, v_mean, v_cov = solver.sample_posteriors(
        r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)

    # ── Save results ──────────────────────────────────────────────────────────
    from .run_alignment import _save_results
    _save_results(
        output_bp3m, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            "n_iter":            n_iter,
            "n_samples":         n_samples,
            "clip_sigma":        clip_sigma,
            "poly_order":        poly_order,
            "hst_enable_iter":   hst_enable_iter,
            "hst_max_pm_unc":    hst_max_pm_unc,
            "hst_max_per_image": hst_max_per_image,
        },
    )

    # ── Full-catalog residuals (all master_combined_v2 stars) ────────────────
    try:
        _save_full_catalog_residuals(
            output_bp3m, solver, image_names, r_hat, data_root, field_name)
    except Exception as _exc:
        import traceback
        print(f"  WARNING: _save_full_catalog_residuals failed — {_exc}")
        traceback.print_exc()

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_bp3m)
        except Exception as exc:
            print(f"  WARNING: plots failed — {exc}")

    print(f"\n  Results written to: {output_bp3m}")
    return output_bp3m
