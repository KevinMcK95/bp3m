"""
run_alignment_cte.py  —  Joint CTE + astrometry model for HST ACS/WFC.

Extends BP3M v2 alignment to simultaneously fit per-image transformations (r_j),
stellar astrometry (v_i), and a parametric CTE model θ_CTE.

See docs/cte_joint_model.md for full mathematical derivation and design decisions.

CTE model summary
-----------------
For chip c (hi/lo), detection k in image j (epoch t_j):

  δCTE_x_k = (t_j − t_launch) · f1(mag_k) · b(xt_k, yt_k) · γ_x_c
  δCTE_y_k = (t_j − t_launch) · f1(mag_k) · b(xt_k, yt_k) · γ_y_c

  f1(mag) = 10^{0.4·(mag − mag_ref)}   (fixed, no free parameters)
  b = yt × P(xt, yt)  where P is a complete polynomial of degree `spatial_order`
  xt = (x_raw − 2048) / 2048  (normalised x)
  yt = |y_raw − y_readout_raw| / 2048  (normalised distance from readout, ∈ [0,1])
  Default (spatial_order=2): b = [yt, yt², xt·yt, yt³, xt·yt², xt²·yt]  (6 terms)

n_spatial = (spatial_order+1)*(spatial_order+2)//2 terms in full basis.
nb = n_spatial*(mag_poly_order+1) − 1 coefficients per chip per direction
  (−1 drops the degenerate constant-mag × yt term, absorbed by per-image y-scale).

Note on y_raw coordinate system: py1pass stores pixel y in a unified global frame
(0..~4096). Each chip reads toward its OUTER edge (away from the central gap):
  _hi images (WFC1/SCI,2): y_raw ∈ [2057, 4087], readout at outer top y≈4096.
  _lo images (WFC2/SCI,1): y_raw ∈ [8, 2039],    readout at outer bottom y≈0.
CTE trails toward the gap: _hi trails toward LOW y_raw, _lo toward HIGH y_raw.
yt = |y_raw − y_readout_raw| / 2048 ∈ [0,1]; yt=0 at readout, yt=1 near gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm as _tqdm

# ── ACS/WFC chip geometry constants ──────────────────────────────────────────
# py1pass unified y frame (0..~4096):
#   _hi images (WFC1/SCI,2): y_raw ∈ [2057, 4087] — readout at OUTER TOP edge y≈4096
#   _lo images (WFC2/SCI,1): y_raw ∈ [8, 2039]    — readout at OUTER BOTTOM edge y≈0
# Each chip reads toward its outer edge (away from the central gap):
#   _hi (WFC1): transfer direction upward (+y), trail toward LOW y_raw (toward gap).
#   _lo (WFC2): transfer direction downward (−y), trail toward HIGH y_raw (toward gap).
# yt = |y_raw − y_readout_raw| / 2048 ∈ [0, 1]; yt=0 at readout, yt=1 near gap.
# GDC-frame: Y_c = y_gdc − 2048.  Readout positions in GDC frame:
_HI_Y_READOUT =  2048.0   # _hi chip readout: y_raw≈4096 → Y_c = 4096−2048 = 2048
_LO_Y_READOUT = -2048.0   # _lo chip readout: y_raw≈0    → Y_c = 0−2048 = −2048
_HI_Y_READOUT_RAW = 4096.0   # _hi images (y∈[2057,4087]): yt = (4096−y)/2048 ∈ [0,1]
_LO_Y_READOUT_RAW = 0.0      # _lo images (y∈[8,2039]):    yt = y/2048 ∈ [0,1]
_ACS_LAUNCH_YR = 2002.165   # ACS launch 2002-03-01; used as CTE time origin
_MAG_REF      = -15.0   # just below the brightest instrumental mag (~-14.5)


# ── CTE parameter dataclass ───────────────────────────────────────────────────

@dataclass
class CTEChipParams:
    chip: str
    y_readout_raw: float      # raw chip-local readout Y (used for CTE basis)
    x0: float = 2048.0
    gamma_x: np.ndarray = field(default_factory=lambda: np.zeros(5))
    gamma_y: np.ndarray = field(default_factory=lambda: np.zeros(5))
    mag_poly_order: int  = 0          # magnitude polynomial order
    spatial_order:  int  = 2          # inner spatial polynomial order; nb = _cte_n_spatial(spatial_order)*(mag_poly_order+1)-1
    mag_norm_ref:   float = 0.0       # magnitude normalisation centre
    mag_norm_scale: float = 1.0       # magnitude normalisation scale

    def copy(self) -> 'CTEChipParams':
        return CTEChipParams(chip=self.chip,
                             y_readout_raw=self.y_readout_raw,
                             x0=float(self.x0),
                             gamma_x=self.gamma_x.copy(),
                             gamma_y=self.gamma_y.copy(),
                             mag_poly_order=self.mag_poly_order,
                             spatial_order=self.spatial_order,
                             mag_norm_ref=float(self.mag_norm_ref),
                             mag_norm_scale=float(self.mag_norm_scale))


def default_cte_params(mag_poly_order: int = 0,
                       mag_norm_ref: float = 0.0,
                       mag_norm_scale: float = 1.0,
                       spatial_order: int = 2) -> dict[str, CTEChipParams]:
    # nb = _cte_n_spatial(spatial_order)*(mag_poly_order+1)-1; the degenerate 1×yt term is excluded
    n = _cte_n_spatial(spatial_order) * (mag_poly_order + 1) - 1
    return {
        'hi': CTEChipParams(chip='hi',
                            y_readout_raw=_HI_Y_READOUT_RAW,
                            gamma_x=np.zeros(n), gamma_y=np.zeros(n),
                            mag_poly_order=mag_poly_order,
                            spatial_order=spatial_order,
                            mag_norm_ref=mag_norm_ref,
                            mag_norm_scale=mag_norm_scale),
        'lo': CTEChipParams(chip='lo',
                            y_readout_raw=_LO_Y_READOUT_RAW,
                            gamma_x=np.zeros(n), gamma_y=np.zeros(n),
                            mag_poly_order=mag_poly_order,
                            spatial_order=spatial_order,
                            mag_norm_ref=mag_norm_ref,
                            mag_norm_scale=mag_norm_scale),
    }


# ── Chip classification ────────────────────────────────────────────────────────

def _chip_from_image(img: str) -> str | None:
    """Return 'hi', 'lo', or None for merged (unsplit) images."""
    if img.endswith('_hi'):
        return 'hi'
    if img.endswith('_lo'):
        return 'lo'
    return None   # merged image — no CTE model applies


# ── Magnitude weighting ───────────────────────────────────────────────────────

def func1_mag(mag: np.ndarray, mag_ref: float = _MAG_REF) -> np.ndarray:
    """Fixed magnitude weighting: 10^{0.4*(mag - mag_ref)}. No free parameters.
    Used only by the non-joint (iterative) CTE pipeline."""
    return np.power(10.0, 0.4 * (np.asarray(mag, dtype=float) - mag_ref))


def mag_poly_basis(mag: np.ndarray, order: int,
                   mag_norm_ref: float = 0.0,
                   mag_norm_scale: float = 1.0) -> np.ndarray:
    """Polynomial basis in normalised magnitude: returns (n, order+1) array.

    m_norm = (mag - mag_norm_ref) / mag_norm_scale
    Columns: [1, m_norm, m_norm², ..., m_norm^order]
    """
    m = (np.asarray(mag, dtype=float) - mag_norm_ref) / mag_norm_scale
    return np.stack([m**k for k in range(order + 1)], axis=1)


# ── CTE basis function ────────────────────────────────────────────────────────

def _cte_n_spatial(spatial_order: int) -> int:
    """Number of terms in the full CTE basis for a given inner polynomial order."""
    return (spatial_order + 1) * (spatial_order + 2) // 2


def _cte_basis_labels(spatial_order: int) -> list:
    """Labels for each column of cte_basis(spatial_order), index 0 = degenerate yt."""
    _sup = {'1': '', '2': '²', '3': '³', '4': '⁴'}
    def _yt(p): return 'yt' + _sup.get(str(p), f'^{p}')
    def _xt(p): return ('xt' if p == 1 else 'xt' + _sup.get(str(p), f'^{p}'))
    labels = ['yt']
    for total_deg in range(1, spatial_order + 1):
        labels.append(_yt(total_deg + 1))
        for xp in range(1, total_deg + 1):
            labels.append(f'{_xt(xp)}·{_yt(total_deg - xp + 1)}')
    return labels


def cte_basis(xt: np.ndarray, yt: np.ndarray, spatial_order: int = 2) -> np.ndarray:
    """CTE basis: yt × [complete polynomial of degree `spatial_order` in (xt, yt)].

    Returns (n, n_spatial) where n_spatial = (spatial_order+1)*(spatial_order+2)//2.
    Column 0 is always yt (degenerate with y-plate-scale); callers drop it with [:, 1:].
    Column 1 is always yt² (primary CTE term — γ[0] after dropping col 0).

    spatial_order=1: [yt, yt², xt·yt]
    spatial_order=2: [yt, yt², xt·yt, yt³, xt·yt², xt²·yt]  (default)
    spatial_order=3: above + [yt⁴, xt·yt³, xt²·yt², xt³·yt]
    """
    terms = [yt]
    for total_deg in range(1, spatial_order + 1):
        terms.append(yt**(total_deg + 1))                   # pure yt^(d+1)
        for xp in range(1, total_deg + 1):                  # mixed terms
            terms.append(xt**xp * yt**(total_deg - xp + 1))
    return np.stack(terms, axis=1)


# ── GDC Jacobian: raw pixel → GDC pixel ──────────────────────────────────────

def _fit_gdc_jacobian_coeffs(spi_df) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fit a 2nd-order bivariate polynomial from raw pixel → GDC pixel for one image.

    Uses all stars in spi_df that have valid X_orig, Y_orig (raw) and X, Y (GDC).

    Returns (coeffs_x, coeffs_y) each shape (6,) for the polynomial:
        x_gdc ≈ c[0] + c[1]*x + c[2]*y + c[3]*x² + c[4]*x*y + c[5]*y²
    where (x, y) = (X_orig, Y_orig).  Returns None if unavailable.
    """
    if spi_df is None:
        return None
    for col in ('X_orig', 'Y_orig', 'X', 'Y'):
        if col not in spi_df.columns:
            return None
    x_raw = spi_df['X_orig'].to_numpy(float)
    y_raw = spi_df['Y_orig'].to_numpy(float)
    x_gdc = spi_df['X'].to_numpy(float)
    y_gdc = spi_df['Y'].to_numpy(float)
    ok = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(x_gdc) & np.isfinite(y_gdc)
    if ok.sum() < 6:
        return None
    xr, yr = x_raw[ok], y_raw[ok]
    A = np.column_stack([np.ones(ok.sum()), xr, yr, xr**2, xr*yr, yr**2])
    cx, _, _, _ = np.linalg.lstsq(A, x_gdc[ok], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, y_gdc[ok], rcond=None)
    return cx, cy


def _eval_gdc_jacobian(coeffs, x_raw: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """
    Evaluate per-star GDC Jacobian from polynomial coefficients.

    coeffs : (coeffs_x, coeffs_y) from _fit_gdc_jacobian_coeffs, or None
    Returns (n, 2, 2) array:
        J[i] = [[∂x_gdc/∂x_raw, ∂x_gdc/∂y_raw],
                 [∂y_gdc/∂x_raw, ∂y_gdc/∂y_raw]]
    Falls back to identity if coeffs is None.
    """
    n = len(x_raw)
    J = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    if coeffs is None:
        return J
    cx, cy = coeffs
    # Derivatives of c[0]+c[1]*x+c[2]*y+c[3]*x²+c[4]*x*y+c[5]*y²
    J[:, 0, 0] = cx[1] + 2.0*cx[3]*x_raw + cx[4]*y_raw   # ∂x_gdc/∂x_raw
    J[:, 0, 1] = cx[2] + cx[4]*x_raw + 2.0*cx[5]*y_raw   # ∂x_gdc/∂y_raw
    J[:, 1, 0] = cy[1] + 2.0*cy[3]*x_raw + cy[4]*y_raw   # ∂y_gdc/∂x_raw
    J[:, 1, 1] = cy[2] + cy[4]*x_raw + 2.0*cy[5]*y_raw   # ∂y_gdc/∂y_raw
    return J


# ── CTE displacement computation ──────────────────────────────────────────────

def compute_cte_displacement(
    x_raw_c: np.ndarray, y_raw: np.ndarray,
    mag: np.ndarray, dt: np.ndarray,
    chip_params: CTEChipParams,
) -> np.ndarray:
    """
    CTE displacement in raw pixel frame.

    x_raw_c : raw x coordinate centred at chip centre [px] (= x_raw − 2048)
    y_raw   : raw chip-local y coordinate [px]
    dt      : years since ACS launch (t_obs − t_launch)

    Returns (n, 2) array of (δCTE_x, δCTE_y) in raw pixel units.
    Apply J_gdc to convert to GDC pixels before applying to residuals.
    """
    xt  = x_raw_c / 2048.0
    yt  = np.abs(y_raw - chip_params.y_readout_raw) / 2048.0
    B   = cte_basis(xt, yt, chip_params.spatial_order)  # (n, n_spatial) spatial basis
    MP  = mag_poly_basis(mag, chip_params.mag_poly_order,
                         chip_params.mag_norm_ref,
                         chip_params.mag_norm_scale)   # (n, order+1) mag basis
    # Combined basis: drop 1×yt column (degenerate with per-image y-plate-scale)
    _full = (MP[:, :, None] * B[:, None, :]).reshape(len(mag), -1)
    PsiB = dt[:, None] * _full[:, 1:]
    return np.stack([PsiB @ chip_params.gamma_x,
                     PsiB @ chip_params.gamma_y], axis=1)


# ── mag_inst injection ────────────────────────────────────────────────────────

def _inject_mag_inst(solver, image_names: list, filtered_spi: dict,
                     gaia_catalog) -> None:
    """
    Inject per-detection instrumental mag into solver._img_data[img]['mag_inst'].

    Uses filtered_spi[img]['mag'] (from FITS catalog, already loaded by
    data_loader_master).  Maps via Gaia_id lookup — works for both positive
    Gaia-matched stars and negative HST-only stars.
    """
    gc_ids = gaia_catalog['Gaia_id'].to_numpy(dtype=np.int64)

    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        spi_df = filtered_spi.get(img)
        if spi_df is None:
            continue
        mag_col = ('mag_gdc' if 'mag_gdc' in spi_df.columns
                   else 'mag' if 'mag' in spi_df.columns
                   else None)
        if mag_col is None:
            continue

        # Build Gaia_id → mag lookup (vectorized, no iterrows)
        _gids = spi_df['Gaia_id'].to_numpy(dtype=np.int64)
        _mags = spi_df[mag_col].to_numpy(dtype=float)
        gid_to_mag = {int(_gids[k]): float(_mags[k]) for k in range(len(spi_df))}

        sidx = d['sidx']
        mag_arr = np.array([gid_to_mag.get(int(gc_ids[s]), np.nan) for s in sidx])
        d['mag_inst'] = mag_arr


# ── Solver CTE correction ─────────────────────────────────────────────────────

def apply_cte_to_solver(
    solver,
    image_names: list[str],
    cte_params: dict[str, CTEChipParams],
    t_launch_yr: float,
    filtered_spi: dict | None = None,
    subtract: bool = False,
) -> None:
    """
    Apply CTE correction to solver._img_data[img]['xys'] for all images.

    Stores 'xys_orig' on first call.  Each subsequent call recomputes from
    xys_orig so corrections don't accumulate.

    filtered_spi : stars_per_image dict (same object passed to the solver).
        If provided, raw chip-local y ('Y_orig') is used for the CTE y-basis.
        If None, falls back to an approximation from GDC Y_c.

    Skips images where chip cannot be determined (merged/unsplit images) or
    where mag_inst is missing / entirely NaN.
    """
    from astropy.time import Time

    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue

        if 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

        chip = _chip_from_image(img)
        if chip is None or chip not in cte_params:
            continue

        mag = d.get('mag_inst')
        if mag is None:
            continue
        ok = np.isfinite(mag)
        if not ok.any():
            continue

        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt_scalar = hst_yr - t_launch_yr
        dt = np.full(len(mag), dt_scalar)

        X_c = d['X_c']   # GDC-centred x

        # Raw chip-local y for CTE y-basis
        if filtered_spi is not None and img in filtered_spi:
            spi_df = filtered_spi[img]
            if 'Y_orig' in spi_df.columns:
                y_raw = spi_df['Y_orig'].to_numpy(float)
            else:
                # Fallback: approximate from GDC (Y_c = y_raw - 2048 for both chips)
                Y_c = d['Y_c']
                y_raw = Y_c + 2048.0
        else:
            Y_c = d['Y_c']
            y_raw = Y_c + 2048.0

        # Guard against length mismatch (shouldn't happen but be safe)
        if len(y_raw) != len(mag):
            Y_c = d['Y_c']
            y_raw = Y_c + 2048.0

        # x_raw: use X_orig if available, fall back to X_c + 2048 (≈ raw x)
        spi_df = (filtered_spi[img] if filtered_spi is not None and img in filtered_spi
                  else None)
        if spi_df is not None and 'X_orig' in spi_df.columns and len(spi_df) == len(mag):
            x_raw_for_jac = spi_df['X_orig'].to_numpy(float)
        else:
            x_raw_for_jac = X_c + 2048.0

        delta_cte_raw = np.zeros((len(mag), 2))
        delta_cte_raw[ok] = compute_cte_displacement(
            (x_raw_for_jac[ok] - 2048.0), y_raw[ok], mag[ok], dt[ok], cte_params[chip])

        # Propagate raw-pixel CTE displacement through GDC Jacobian → GDC pixels,
        # then through the plate solution R_j → pseudo-image arcseconds.
        jac_coeffs = _fit_gdc_jacobian_coeffs(spi_df)
        J_gdc = _eval_gdc_jacobian(jac_coeffs, x_raw_for_jac, y_raw)  # (n, 2, 2)
        delta_cte_gdc = np.einsum('nij,nj->ni', J_gdc, delta_cte_raw)  # (n, 2)

        R_j = solver.R[img]                                    # (2, 2)
        delta_cte_pseudo = delta_cte_gdc @ R_j.T               # (n, 2)
        if subtract:
            d['xys'] = d['xys_orig'] - delta_cte_pseudo
        else:
            d['xys'] = d['xys_orig'] + delta_cte_pseudo


# ── Multi-filter catalog loader ───────────────────────────────────────────────

def _load_full_catalog_df_all_filters(data_root, field_name: str):
    """
    Load and merge all detections_{filter}.csv + master_combined_v2.csv.

    Replaces run_alignment_v2._load_full_catalog_df so that residual maps are
    generated for every image in the solver, not just F814W images.
    master_combined_v2 has hst_indices_{filter} for every filter present, with
    the same 'sub_name:catalog_index,...' format, so the same HST-only join logic
    works for each filter.

    Returns dict {sub_name: DataFrame} or None if required files are missing.
    """
    import pandas as pd
    from pathlib import Path as _Path

    xmatch_dir = _Path(data_root) / field_name / 'hst_xmatch'
    mcat_path  = xmatch_dir / 'master_combined_v2.csv'
    det_files  = sorted(xmatch_dir.glob('detections_*.csv'))

    if not mcat_path.exists() or not det_files:
        return None

    mcat = pd.read_csv(mcat_path, dtype={'gaia_source_id': np.int64}, low_memory=False)

    star_cols = ['ra_xmatch', 'dec_xmatch', 'pmra_xmatch', 'pmdec_xmatch',
                 'parallax_xmatch', 'epoch_ref_xmatch']

    mcat_gaia = (mcat[mcat['gaia_source_id'] != 0]
                 [['gaia_source_id'] + star_cols].copy())

    all_dfs = []
    for det_path in det_files:
        filt = det_path.stem.replace('detections_', '')   # e.g. 'F814W'
        hst_idx_col = f'hst_indices_{filt}'

        det = pd.read_csv(det_path, dtype={'gaia_source_id': np.int64})

        # Columns to keep from detection file (filter/instrument/detector are
        # constant per sub_name so safe to carry through for title annotation)
        det_pos_cols = ['sub_name', 'gaia_source_id', 'catalog_index',
                        'x_gdc', 'y_gdc', 'mag_gdc']
        for _col in ('filter', 'instrument', 'detector'):
            if _col in det.columns:
                det_pos_cols.append(_col)

        # Gaia-matched detections — astrometry is filter-independent
        det_gaia = det[det['gaia_source_id'].to_numpy(np.int64) != 0][
                       det_pos_cols].copy()
        det_gaia_m = det_gaia.merge(mcat_gaia, on='gaia_source_id', how='inner')
        all_dfs.append(det_gaia_m)

        # HST-only detections — use this filter's index column in master
        if hst_idx_col not in mcat.columns:
            continue

        mcat_hst_src = mcat[mcat[hst_idx_col].notna()][
                           [hst_idx_col] + star_cols].copy().reset_index(drop=True)
        mcat_hst_src['_entries'] = (mcat_hst_src[hst_idx_col]
                                    .str.replace(';', ',', regex=False)
                                    .str.split(','))
        mcat_exploded = mcat_hst_src.explode('_entries')
        mcat_exploded = mcat_exploded[
            mcat_exploded['_entries'].str.contains(':', na=False)].copy()
        entry_parts = mcat_exploded['_entries'].str.split(':', expand=True)
        mcat_exploded['sub_name']      = entry_parts[0]
        mcat_exploded['catalog_index'] = entry_parts[1].astype(np.int64)
        rev_idx = (mcat_exploded[['sub_name', 'catalog_index'] + star_cols]
                   .reset_index(drop=True))

        det_hst = det[det['gaia_source_id'].to_numpy(np.int64) == 0][
                      det_pos_cols].copy()
        det_hst['catalog_index'] = det_hst['catalog_index'].astype(np.int64)
        det_hst_m = det_hst.merge(rev_idx, on=['sub_name', 'catalog_index'], how='inner')
        all_dfs.append(det_hst_m)

    if not all_dfs:
        return None

    det_all = pd.concat(all_dfs, ignore_index=True)
    n_total = len(det_all)
    n_gaia  = (det_all['gaia_source_id'].to_numpy(np.int64) != 0).sum()
    n_imgs  = det_all['sub_name'].nunique()
    print(f"    Loaded {n_total:,} detections across {n_imgs} images "
          f"({n_gaia:,} Gaia-matched + {n_total - n_gaia:,} HST-only) "
          f"from {len(det_files)} filter(s)")
    return {img: grp.reset_index(drop=True)
            for img, grp in det_all.groupby('sub_name', sort=False)}


# ── Residual collection ───────────────────────────────────────────────────────

def collect_cte_residuals(
    img_to_df: dict,
    solver,
    image_names: list[str],
    r_hat: np.ndarray,
    t_launch_yr: float,
    field_mean_pm: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """
    Collect per-chip GDC-frame residuals for updating CTE parameters.

    Uses the full master-catalog detection set (all ~127k stars via img_to_df),
    not just the solver's BP3M alignment stars.  img_to_df is the output of
    _load_full_catalog_df and should be cached across CTE iterations.

    Parameters
    ----------
    field_mean_pm : (pmra_mean, pmdec_mean) in mas/yr, or None.
        If provided, ALL stars use this common PM for residual computation.
        This removes the CTE absorbed into individual HST-derived PMs for
        faint/HST-only stars, revealing the true CTE signal.
        If None, the per-star pmra_xmatch/pmdec_xmatch from the master catalog
        are used (accurate for Gaia-matched stars; CTE-contaminated for HST-only).

    Returns
    -------
    residuals : {'hi': {...}, 'lo': {...}} each with arrays:
        dx, dy : (n,) GDC residuals [pixels]
        X_c : (n,) GDC-centred x [px]  (for X-direction basis and cross-terms)
        y_raw : (n,) raw chip-local y [px]  (for Y-direction CTE basis)
        mag, dt, z : per-detection geometry/weighting
    """
    from astropy.time import Time
    from .run_alignment_v2 import _compute_full_catalog_residuals_from_df

    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    out_arrays = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat,
        global_pm_override=field_mean_pm)

    residuals = {c: {'dx': [], 'dy': [], 'X_c': [], 'y_raw': [],
                     'mag': [], 'dt': [], 'z': []}
                 for c in ('hi', 'lo')}

    for img in image_names:
        if f'{img}_X_c' not in out_arrays:
            continue
        chip = _chip_from_image(img)
        if chip is None:
            continue

        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                             format='mjd').jyear)
        dt = hst_yr - t_launch_yr

        X_c  = out_arrays[f'{img}_X_c'].astype(float)
        dx   = out_arrays[f'{img}_dx_gdc'].astype(float)
        dy   = out_arrays[f'{img}_dy_gdc'].astype(float)
        mag  = out_arrays[f'{img}_mag_inst'].astype(float)

        # Use raw chip-local y for CTE y-basis (physically correct readout direction)
        if f'{img}_y_raw' in out_arrays:
            y_raw = out_arrays[f'{img}_y_raw'].astype(float)
        else:
            # Fallback: approximate raw y from GDC (Y_c = y_raw - 2048 for both chips)
            Y_c = out_arrays[f'{img}_Y_c'].astype(float)
            y_raw = Y_c + 2048.0

        ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & np.isfinite(y_raw)

        # PM quality filter: only use stars with converged v2 PM
        # (sigma_pmra < 1 mas/yr, sigma_pmdec < 1 mas/yr, |pmra| < 3 mas/yr).
        # This excludes HST-only stars with no Gaia anchor, faint stars with
        # noisy cross-match PMs, and non-members with large PMs.
        # Gaia-matched stars (in_bp3m=True) are always included — they have
        # Gaia priors and their residuals are CTE-free for alignment purposes.
        in_bp3m_raw = out_arrays.get(f'{img}_in_bp3m')
        in_bp3m_flag = (in_bp3m_raw.astype(bool)
                        if in_bp3m_raw is not None
                        else np.zeros(len(dx), dtype=bool))

        sig_pmra  = out_arrays.get(f'{img}_sigma_pmra')
        sig_pmdec = out_arrays.get(f'{img}_sigma_pmdec')
        pmra_arr  = out_arrays.get(f'{img}_pmra_xmatch')
        pmdec_arr = out_arrays.get(f'{img}_pmdec_xmatch')
        if sig_pmra is not None:
            sig_pmra  = sig_pmra.astype(float)
            sig_pmdec = sig_pmdec.astype(float)
            pmra_arr  = pmra_arr.astype(float)
            pmdec_arr = pmdec_arr.astype(float)
            pm_quality = (np.isfinite(sig_pmra) & (sig_pmra < 1.0) &
                          np.isfinite(sig_pmdec) & (sig_pmdec < 1.0) &
                          np.isfinite(pmra_arr)  & (np.abs(pmra_arr)  < 3.0) &
                          np.isfinite(pmdec_arr) & (np.abs(pmdec_arr) < 3.0))
            ok = ok & (in_bp3m_flag | pm_quality)
        else:
            ok = ok & in_bp3m_flag
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue

        residuals[chip]['dx'].append(dx[ok])
        residuals[chip]['dy'].append(dy[ok])
        residuals[chip]['X_c'].append(X_c[ok])
        residuals[chip]['y_raw'].append(y_raw[ok])
        residuals[chip]['mag'].append(mag[ok])
        residuals[chip]['dt'].append(np.full(n_ok, dt))
        residuals[chip]['z'].append(np.ones(n_ok))

    for chip in ('hi', 'lo'):
        for key in residuals[chip]:
            arr = residuals[chip][key]
            residuals[chip][key] = np.concatenate(arr) if arr else np.array([])
        n = len(residuals[chip]['dx'])
        print(f"    {chip}: {n:,} detections")

    return residuals


# ── CTE parameter update ──────────────────────────────────────────────────────

def update_cte_params(
    residuals_by_chip: dict[str, dict],
    cte_params: dict[str, CTEChipParams],
    regularize: float = 1e-8,
    **_kwargs,   # absorb legacy n_inner / delta_tol kwargs silently
) -> tuple[dict[str, CTEChipParams], dict]:
    """
    Update CTE parameters (γ_x, γ_y) from GDC-frame residuals via linear WLS.

    With func1_mag fixed, both x and y updates are independent linear solves —
    no nonlinear inner loop needed.
    """
    new_params = {c: cte_params[c].copy() for c in ('hi', 'lo')}
    info = {}

    for chip in ('hi', 'lo'):
        res = residuals_by_chip[chip]
        if len(res['dx']) == 0:
            info[chip] = {'converged': True, 'n_det': 0}
            continue

        dx    = res['dx'].astype(float)
        dy    = res['dy'].astype(float)
        X_c   = res['X_c'].astype(float)
        y_raw = res['y_raw'].astype(float)
        mag   = res['mag'].astype(float)
        dt    = res['dt'].astype(float)
        z     = res['z'].astype(float)

        ok = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
              & np.isfinite(dt) & np.isfinite(y_raw) & (np.abs(dt) > 0))
        dx, dy, X_c, y_raw, mag, dt, z = (arr[ok] for arr in
                                            (dx, dy, X_c, y_raw, mag, dt, z))
        n = len(dx)

        if n < 10:
            info[chip] = {'converged': True, 'n_det': n}
            continue

        p   = new_params[chip]
        xt  = X_c / 2048.0
        yt  = np.abs(y_raw - p.y_readout_raw) / 2048.0
        f1  = func1_mag(mag)
        Psi = dt * f1                        # (n,)
        B   = cte_basis(xt, yt, p.spatial_order)[:, 1:]   # drop degenerate 1×yt
        A   = Psi[:, None] * B              # (n, 4) design matrix

        col_scale = np.std(A, axis=0).clip(min=1e-30)
        A_s = A / col_scale

        AtWA = (A_s * z[:, None]).T @ A_s + regularize * np.eye(4)
        try:
            AtWr_y = (A_s * z[:, None]).T @ (-dy)
            p.gamma_y = np.linalg.solve(AtWA, AtWr_y) / col_scale
        except np.linalg.LinAlgError:
            pass
        try:
            AtWr_x = (A_s * z[:, None]).T @ (-dx)
            p.gamma_x = np.linalg.solve(AtWA, AtWr_x) / col_scale
        except np.linalg.LinAlgError:
            pass

        rms_y = float(np.sqrt(np.mean((-dy - A @ p.gamma_y) ** 2)))
        rms_x = float(np.sqrt(np.mean((-dx - A @ p.gamma_x) ** 2)))

        info[chip] = {'n_det': n, 'rms_x': rms_x, 'rms_y': rms_y}
        print(f"    {chip}: "
              f"|γ_y|={np.linalg.norm(p.gamma_y):.4e}  "
              f"|γ_x|={np.linalg.norm(p.gamma_x):.4e}  "
              f"rms_y={rms_y:.4f}px  n={n:,}")

    return new_params, info


# ── Per-image alpha (uncertainty inflation) update ───────────────────────────

def _update_image_alpha(
    solver,
    image_names: list[str],
    r_hat: np.ndarray,
    a_arr: np.ndarray,
) -> None:
    """
    Re-estimate the per-image HST position-uncertainty inflation factor (alpha)
    from the current joint-solve residuals and update C_hst = alpha^2 * C_hst_orig.

    Must be called AFTER apply_cte_to_solver (subtract=True) so that
    solver._img_data[img]['xys'] contains CTE-corrected positions, and AFTER
    solver._update_R(r_hat) so that solver.R[img] is current.

    alpha is measured against C_hst_orig (the un-inflated covariance stored at
    solver initialisation), so it is an absolute factor — no multiplication with
    the previous alpha is needed.  Values below 1.0 are clamped to 1.0.
    """
    _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)   # median of chi2(2 dof) distribution
    nr = solver.N_R
    for j_idx, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None:
            continue
        C_hst_orig = d.get("C_hst_orig")
        if C_hst_orig is None:
            continue
        use_fit = np.asarray(d["use_for_fit"], bool)
        if use_fit.sum() < 4:
            continue

        cs    = j_idx * nr
        r_j   = r_hat[cs:cs + nr]
        sidx  = d["sidx"]
        xys   = d["xys"]       # CTE-corrected positions
        X_mat = d["X_mat"]
        JU    = d["JU"]

        # resid = x_obs - (X r_j - JU v)  [same sign convention as solver]
        pred  = (np.einsum('nij,j->ni', X_mat, r_j)
                 - np.einsum('nij,nj->ni', JU, a_arr[sidx]))
        resid = xys - pred                  # (n, 2)

        # chi2 per detection vs C_hst_orig (un-inflated)
        R_j      = solver.R[img]            # (2, 2) current rotation
        C_s_orig = np.einsum('ij,njk,lk->nil', R_j, C_hst_orig, R_j)
        C_s_inv  = np.linalg.inv(C_s_orig[use_fit])
        chi2     = np.einsum('ni,nij,nj->n',
                             resid[use_fit], C_s_inv, resid[use_fit])

        alpha_j = float(max(1.0, np.sqrt(float(np.median(chi2)) / _MEDIAN_CHI2_2)))
        d["alpha_applied"] = alpha_j
        d["C_hst"]         = alpha_j**2 * C_hst_orig


def _apply_cte_to_residual_arrays(
    out_arrays: dict,
    image_names: list[str],
    cte_params: dict,
    t_launch_yr: float,
    solver,
    filtered_spi: dict | None = None,
) -> dict:
    """
    Subtract CTE displacement from pre-computed GDC residual arrays.

    For a linear transformation (poly_order=1):
        dy_gdc_corrected = dy_gdc_before - delta_CTE_y
    This converts "residuals with joint r_hat, original positions" into
    "residuals with joint r_hat, CTE-corrected positions".

    Returns a new dict; does not modify the input.
    """
    from astropy.time import Time
    result = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in out_arrays.items()}

    for img in image_names:
        chip = _chip_from_image(img)
        if chip is None or chip not in cte_params:
            continue
        X_c_key = f'{img}_X_c'; dx_key = f'{img}_dx_gdc'; dy_key = f'{img}_dy_gdc'
        mag_key = f'{img}_mag_inst'; Y_c_key = f'{img}_Y_c'
        if dx_key not in result:
            continue
        X_c = result[X_c_key].astype(float)
        Y_c = result[Y_c_key].astype(float)
        mag = result[mag_key].astype(float)
        # y_raw: prefer stored value, then filtered_spi, then GDC approximation
        y_raw_key = f'{img}_y_raw'
        if y_raw_key in result:
            y_raw = result[y_raw_key].astype(float)
        elif filtered_spi is not None and img in filtered_spi:
            spi_df = filtered_spi[img]
            if 'Y_orig' in spi_df.columns and len(spi_df) == len(X_c):
                y_raw = spi_df['Y_orig'].to_numpy(float)
            else:
                y_raw = Y_c + 2048.0   # Y_c = y_raw − 2048 for both chips
        else:
            y_raw = Y_c + 2048.0       # Y_c = y_raw − 2048 for both chips

        hst_yr    = float(Time(float(solver.images[img]['hst_time_mjd']), format='mjd').jyear)
        dt_scalar = hst_yr - t_launch_yr
        dt        = np.full(len(mag), dt_scalar)
        ok = np.isfinite(mag) & np.isfinite(y_raw)
        if not ok.any():
            continue
        # x_raw: use X_orig if available, fall back to X_c + 2048 (≈ raw x)
        spi_for_jac = (filtered_spi.get(img) if filtered_spi is not None else None)
        x_raw_for_jac = (spi_for_jac['X_orig'].to_numpy(float)
                         if spi_for_jac is not None and 'X_orig' in spi_for_jac.columns
                            and len(spi_for_jac) == len(X_c)
                         else X_c + 2048.0)
        delta_cte_raw = np.zeros((len(mag), 2))
        delta_cte_raw[ok] = compute_cte_displacement(
            (x_raw_for_jac[ok] - 2048.0), y_raw[ok], mag[ok], dt[ok], cte_params[chip])
        # Propagate raw-pixel CTE displacement → GDC pixels via J_gdc.
        # dx_gdc/dy_gdc are in GDC pixel units (from resid_pseudo @ R_j^{-T}),
        # so we subtract the GDC-frame correction without any further R_j rotation.
        jac_coeffs = _fit_gdc_jacobian_coeffs(spi_for_jac)
        J_gdc = _eval_gdc_jacobian(jac_coeffs, x_raw_for_jac, y_raw)   # (n, 2, 2)
        delta_cte_gdc = np.einsum('nij,nj->ni', J_gdc, delta_cte_raw)  # (n, 2) GDC pixels
        result[dx_key] = (result[dx_key].astype(float) - delta_cte_gdc[:, 0]).astype(np.float32)
        result[dy_key] = (result[dy_key].astype(float) - delta_cte_gdc[:, 1]).astype(np.float32)
    return result


# ── Joint Schur-complement solve for (r, γ_CTE, μ_pop) ───────────────────────

def _joint_solve_cte(
    solver,
    image_names: list[str],
    cte_params: dict[str, CTEChipParams],
    t_launch_yr: float,
    filtered_spi: dict | None,
    member_sidx: np.ndarray,
    sigma_pm: float,
    plx_pop: float,
    sigma_plx_tot: float,
    mu_pop_current: np.ndarray,
    mu_pop_prior: np.ndarray,
    C_pop_prior_inv: np.ndarray,
    r_current: np.ndarray,
    regularize_gamma: float = 1e-8,
    gamma_prior: np.ndarray | None = None,
    hst_prior_sidx: np.ndarray | None = None,
    fit_cte_x: bool = True,
) -> tuple:
    """
    Joint Schur-complement solve for (r, γ_CTE, μ_pop) after marginalising
    stellar astrometry {v_i}, mirroring solver._solve_one_pass.

    member_sidx     : Gaia-matched member stars — drive μ_pop AND receive the
                      population prior.
    hst_prior_sidx  : HST-only member stars — receive the population prior
                      (PM regularisation) but do NOT contribute to the μ_pop
                      Schur correction.  Pass the fixed HST-only member set
                      from run_alignment_joint_cte so their PMs are
                      regularised without biasing the population mean.

    Parameters
    ----------
    solver           : BP3M Solver (read-only)
    image_names      : images to process; r_current is indexed by position here
    cte_params       : CTEChipParams keyed by 'hi' / 'lo'
    t_launch_yr      : CTE time origin in Julian years (_ACS_LAUNCH_YR)
    filtered_spi     : stars_per_image dict for exact y_raw (may be None)
    member_sidx      : (n_mem,) global star indices of likely cluster members
    sigma_pm         : LVD-derived intrinsic PM dispersion (mas/yr)
    plx_pop          : LVD mean parallax (mas)
    sigma_plx_tot    : total LVD parallax uncertainty (mas, including depth)
    mu_pop_current   : (2,) current iterate of population mean PM (mas/yr)
    mu_pop_prior     : (2,) empirical prior mean for mu_pop (mas/yr)
    C_pop_prior_inv  : (2,2) prior precision on mu_pop (mas/yr)^{-2}
    r_current        : (n_r,) current image-parameter iterate, stacked over
                       image_names in order
    regularize_gamma : small ridge added to H_γγ
    gamma_prior      : if provided, regularise toward this value instead of
                       zero (prevents r–γ oscillation in degenerate directions)

    Returns
    -------
    r_hat       : (n_r,)               updated image parameters
    C_r         : (n_r, n_r)           marginal posterior covariance of r
    gamma_hat   : (20,)                CTE parameters [hi_x, hi_y, lo_x, lo_y]
    mu_pop_hat  : (2,)                 updated population mean PM (mas/yr)
    C_shared    : (n_shared, n_shared) full joint posterior covariance
    a_arr       : (n_stars, 5)         stellar astrometry posterior mean
    K_img       : dict{img -> (n,5,N_R)}
    C_vT        : (n_stars, 5, 5)      stellar astrometry posterior covariance
    """
    from astropy.time import Time

    nr       = solver.N_R
    n_images = len(image_names)
    n_r      = nr * n_images
    # Read poly orders from cte_params template (both chips must agree)
    _ref_chip = cte_params.get('hi', cte_params.get('lo'))
    mag_poly_order = _ref_chip.mag_poly_order
    spatial_order  = _ref_chip.spatial_order
    mag_norm_ref   = _ref_chip.mag_norm_ref
    mag_norm_scale = _ref_chip.mag_norm_scale
    nb       = _cte_n_spatial(spatial_order) * (mag_poly_order + 1) - 1  # per chip per direction
    n_gamma  = 4 * nb                      # hi_x, hi_y, lo_x, lo_y
    N_V      = 5
    n_shared = n_r + n_gamma + 2

    idx_r   = slice(0, n_r)
    idx_gam = slice(n_r, n_r + n_gamma)
    idx_mu  = slice(n_r + n_gamma, n_shared)

    # Population mean couples to PM components of stellar astrometry
    M = np.zeros((N_V, 2))
    M[2, 0] = 1.0   # μ_α* row
    M[3, 1] = 1.0   # μ_δ row

    sigma_pm_inv_sq  = 1.0 / sigma_pm**2
    sigma_plx_inv_sq = 1.0 / sigma_plx_tot**2

    n_stars     = len(solver.C_survey_inv)
    member_mask = np.zeros(n_stars, dtype=bool)
    member_mask[member_sidx] = True

    # ── Precision matrices and information vectors ────────────────────────────
    # Start from solver's Gaia prior; add diagonal _C_VG_inv_per_star (HST-only
    # PM dispersion 100 mas/yr) then add population and parallax priors for members.
    H_vv = solver.C_survey_inv.copy()
    H_vv[:, np.arange(N_V), np.arange(N_V)] += solver._C_VG_inv_per_star

    # All stars that receive the population prior (Gaia members + fixed HST-only).
    # member_sidx alone drives μ_pop (Schur correction below).
    if hst_prior_sidx is not None and len(hst_prior_sidx) > 0:
        _all_prior = np.concatenate([member_sidx, hst_prior_sidx])
    else:
        _all_prior = member_sidx

    H_vv[_all_prior, 2, 2] += sigma_pm_inv_sq    # population PM prior (μ_α*)
    H_vv[_all_prior, 3, 3] += sigma_pm_inv_sq    # population PM prior (μ_δ)
    H_vv[_all_prior, 4, 4] += sigma_plx_inv_sq   # LVD parallax prior

    h_align = solver.C_survey_inv_dot_v.copy()
    h_all   = solver.C_survey_inv_dot_v.copy()

    # Prior information term: σ_pm^{-2} M μ_pop_current for all prior stars
    h_align[_all_prior, 2] += sigma_pm_inv_sq * mu_pop_current[0]
    h_align[_all_prior, 3] += sigma_pm_inv_sq * mu_pop_current[1]
    h_all[_all_prior, 2]   += sigma_pm_inv_sq * mu_pop_current[0]
    h_all[_all_prior, 3]   += sigma_pm_inv_sq * mu_pop_current[1]

    # LVD parallax prior information: σ_plx^{-2} * plx_pop
    h_align[_all_prior, 4] += sigma_plx_inv_sq * plx_pop
    h_all[_all_prior, 4]   += sigma_plx_inv_sq * plx_pop

    # ── Shared-parameter precision and data accumulations ─────────────────────
    H_rr        = np.zeros((n_r, n_r))
    H_gamma     = np.zeros((n_gamma, n_gamma))
    P_rg        = np.zeros((n_r, n_gamma))      # X^T Cs^{-1} G, all fitting stars
    GCs_xresid  = np.zeros(n_gamma)             # G^T Cs^{-1} x_resid, all active stars
    # Q_total_mem: summed JUT_Cs @ G for member stars only (μ_pop Schur coupling)
    # Q_total_all: summed JUT_Cs @ G for ALL active stars ((γ,γ) and (r,γ) Schur)
    Q_total_mem = np.zeros((n_stars, N_V, n_gamma))
    Q_total_all = np.zeros((n_stars, N_V, n_gamma))
    active_glob = np.zeros(n_stars, dtype=bool)  # tracks which stars were active

    K_img      = {}
    Q_img      = {}
    XCs_xresid = {}

    # ── Per-image first pass: H_vv, h, K, G, Q, H_rr, H_gamma ───────────────
    for j_idx, img in enumerate(_tqdm(image_names, desc="  joint_solve", unit="img",
                                      ncols=90, leave=False)):
        d = solver._img_data.get(img)
        if d is None:
            K_img[img] = None
            Q_img[img] = None
            continue

        sidx         = d["sidx"]
        # All fitting stars (no member restriction) — constrain r and γ
        use_fit      = d["use_for_fit"]
        use_astrom_f = (d.get("use_for_astrom", d["use_for_fit"])
                        if getattr(solver, '_use_two_tier', False) else d["use_for_fit"])
        use_any      = use_fit | use_astrom_f          # all active (Gaia + HST-only)
        # Member stars get tight population PM prior
        use_member   = use_any & member_mask[sidx]

        sidx_any = sidx[use_any]
        sidx_fit = sidx[use_fit]
        active_glob[sidx_any] = True

        JU  = d["JU"]
        X   = d["X_mat"]
        xys = d.get("xys_orig", d["xys"])   # use raw (pre-CTE) positions

        cs  = j_idx * nr
        r_j = r_current[cs:cs + nr]

        Cs     = solver._compute_Cs(img, r_j)
        Cs_inv = np.linalg.inv(Cs)

        x_pred  = np.einsum('nkl,l->nk', X, r_j)
        x_resid = xys - x_pred

        JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)

        # Stellar precision/information (mirrors solver._solve_one_pass)
        np.add.at(H_vv, sidx_any,
                  np.einsum('nik,nkj->nij', JUT_Cs[use_any], JU[use_any]))
        np.subtract.at(h_all, sidx_any,
                       np.einsum('nik,nk->ni', JUT_Cs[use_any], x_resid[use_any]))
        np.subtract.at(h_align, sidx_fit,
                       np.einsum('nik,nk->ni', JUT_Cs[use_fit], x_resid[use_fit]))

        K = np.einsum('nik,nkl->nil', JUT_Cs, X)   # (n, 5, N_R)
        K_img[img] = K

        # H_rr and XCs_xresid use all fitting stars (not just members)
        XCsX = np.einsum('nki,nkl,nlj->ij',
                         X[use_fit], Cs_inv[use_fit], X[use_fit])
        H_rr[cs:cs+nr, cs:cs+nr] += XCsX + d["C_r_prior_inv"]
        XCs_xresid[img] = np.einsum('nki,nkl,nl->ni',
                                    X[use_fit], Cs_inv[use_fit], x_resid[use_fit])

        # ── CTE design matrix G ────────────────────────────────────────────────
        chip = _chip_from_image(img)
        if chip is None or chip not in cte_params:
            Q_img[img] = None
            continue

        mag = d.get('mag_inst')
        if mag is None:
            Q_img[img] = None
            continue

        Y_c    = d['Y_c']
        spi_df = filtered_spi.get(img) if filtered_spi else None
        if (spi_df is not None and 'Y_orig' in spi_df.columns
                and len(spi_df) == len(mag)):
            y_raw = spi_df['Y_orig'].to_numpy(float)
        else:
            y_raw = Y_c + 2048.0  # Y_c = y_raw - 2048 for both chips

        hst_yr    = float(Time(float(solver.images[img]['hst_time_mjd']),
                               format='mjd').jyear)
        dt_scalar = hst_yr - t_launch_yr

        p   = cte_params[chip]
        n   = len(sidx)
        ok  = np.isfinite(mag) & np.isfinite(y_raw)

        cx, cy = (0, nb) if chip == 'hi' else (2*nb, 3*nb)
        R_j    = r_j[:4].reshape(2, 2)   # [[a, b], [c, d]]

        # Include GDC Jacobian: raw-pixel CTE → GDC pixels → pseudo-image arcseconds.
        # R_eff[i] = R_j @ J_gdc[i] maps raw-pixel displacement to pseudo-image.
        if spi_df is not None and 'X_orig' in spi_df.columns and len(spi_df) == n:
            x_raw_jac = spi_df['X_orig'].to_numpy(float)
        else:
            x_raw_jac = d['X_c'] + 2048.0

        # Use raw coordinates for the CTE spatial basis
        xt  = (x_raw_jac - 2048.0) / 2048.0
        yt  = np.abs(y_raw - p.y_readout_raw) / 2048.0
        # Combined basis: dt * mag_poly ⊗ spatial, drop 1×yt → (n, nb)
        MP  = np.where(ok[:, None],
                       mag_poly_basis(mag, mag_poly_order,
                                      mag_norm_ref, mag_norm_scale), 0.0)
        PsiB = (dt_scalar * (MP[:, :, None] * cte_basis(xt, yt, spatial_order)[:, None, :])
                ).reshape(n, nb + 1)[:, 1:]   # (n, nb); drop degenerate 1×yt
        jac_coeffs = _fit_gdc_jacobian_coeffs(spi_df)
        J_gdc  = _eval_gdc_jacobian(jac_coeffs, x_raw_jac, y_raw)    # (n, 2, 2)
        R_eff  = np.einsum('ij,njk->nik', R_j, J_gdc)                 # (n, 2, 2)

        G = np.zeros((n, 2, n_gamma))
        G[ok, :, cx:cx+nb] = R_eff[ok, :, 0:1] * PsiB[ok, None, :]  # raw-x CTE
        G[ok, :, cy:cy+nb] = R_eff[ok, :, 1:2] * PsiB[ok, None, :]  # raw-y CTE

        Q = np.einsum('nik,nkl->nil', JUT_Cs, G)   # (n, 5, n_gamma)
        Q_img[img] = Q

        # H_gamma and GCs_xresid accumulate over ALL active stars (not just members)
        if use_any.any():
            H_gamma    += np.einsum('nki,nkl,nlj->ij',
                                    G[use_any], Cs_inv[use_any], G[use_any])
            GCs_xresid += np.einsum('nki,nkl,nl->i',
                                    G[use_any], Cs_inv[use_any], x_resid[use_any])
            np.add.at(Q_total_all, sidx[use_any], Q[use_any])

        # Q_total_mem only for member stars (needed for (γ,μ) Schur coupling)
        if use_member.any():
            np.add.at(Q_total_mem, sidx[use_member], Q[use_member])

        # P_rg couples r and γ for all fitting stars
        if use_fit.any():
            P_rg[cs:cs+nr] += np.einsum('nki,nkl,nlj->ij',
                                         X[use_fit], Cs_inv[use_fit], G[use_fit])

    # ── Invert H_vv → C_vT, stellar posteriors ───────────────────────────────
    C_vT    = np.linalg.inv(H_vv)
    a_align = np.einsum('nij,nj->ni', C_vT, h_align)
    a       = np.einsum('nij,nj->ni', C_vT, h_all)

    # ── Assemble full Schur complement system ─────────────────────────────────
    H_gamma += regularize_gamma * np.eye(n_gamma)

    # H_μμ direct: prior precision + sum_members σ_pm^{-2} I₂ (M^T M = I₂)
    H_mu  = C_pop_prior_inv.copy()
    H_mu += sigma_pm_inv_sq * float(len(member_sidx)) * np.eye(2)

    Lambda = np.zeros((n_shared, n_shared))
    Lambda[idx_r,   idx_r]   = H_rr
    Lambda[idx_gam, idx_gam] = H_gamma
    Lambda[idx_mu,  idx_mu]  = H_mu
    Lambda[idx_r,   idx_gam] = P_rg        # direct H_rγ (positive)
    Lambda[idx_gam, idx_r]   = P_rg.T

    # RHS initialisation
    rhs = np.zeros(n_shared)
    for j_idx, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None:
            continue
        cs = j_idx * nr
        rhs[cs:cs+nr] += d["C_r_prior_inv"] @ (d["r_prior"] - r_current[cs:cs+nr])
        if img in XCs_xresid:
            rhs[cs:cs+nr] += XCs_xresid[img].sum(axis=0)
    rhs[idx_gam] = GCs_xresid
    # γ prior: regularise toward gamma_prior (warmstart) instead of zero.
    # This prevents the r–γ degeneracy from causing gamma to oscillate away
    # from the warmstart value across Newton iterations.
    if gamma_prior is not None:
        rhs[idx_gam] += regularize_gamma * gamma_prior
    # μ_pop rhs: prior gradient + linearisation of population prior at v=0.
    # The population prior ½σ⁻²(v−μ_pop_current−Δμ)² has gradient w.r.t. Δμ of
    # −σ⁻²(v−μ_pop_current−Δμ) which at v=0, Δμ=0 gives +σ⁻²·μ_pop_current.
    # h_μ = −gradient → C_pop_prior_inv(μ_prior−μ_current) − σ⁻²·n_mem·μ_current.
    # The Schur correction (+σ⁻² Σ a[members,2:4]) then cancels this at convergence.
    rhs[idx_mu]  = C_pop_prior_inv @ (mu_pop_prior - mu_pop_current)
    rhs[idx_mu] -= sigma_pm_inv_sq * float(len(member_sidx)) * mu_pop_current

    # ── Global Schur corrections for γ block (all active stars) ─────────────
    all_active_sidx = np.where(active_glob)[0]
    if len(all_active_sidx) > 0:
        Qt_all    = Q_total_all[all_active_sidx]               # (n_act, 5, n_gamma)
        Cv_all    = C_vT[all_active_sidx]                      # (n_act, 5, 5)
        CvT_Q_all = np.einsum('nij,njk->nik', Cv_all, Qt_all) # (n_act, 5, n_gamma)

        # (γ, γ) Schur: -Q_all^T C_vT Q_all  (over all active stars)
        Lambda[idx_gam, idx_gam] -= np.einsum('nji,njk->ik', Qt_all, CvT_Q_all)
        # γ rhs: +Q_all^T a  (over all active stars)
        rhs[idx_gam]             += np.einsum('nji,nj->i',   Qt_all, a[all_active_sidx])

    # ── Global Schur corrections for μ block (member stars only) ─────────────
    if len(member_sidx) > 0:
        Qt_m     = Q_total_mem[member_sidx]                        # (n_mem, 5, n_gamma)
        Cv_m     = C_vT[member_sidx]                               # (n_mem, 5, 5)

        # (γ, μ) Schur: -σ^{-2} Q_mem^T C_vT M
        CvT_M_m  = Cv_m @ M                                        # (n_mem, 5, 2)
        QT_CvT_M = np.einsum('nji,njk->ik', Qt_m, CvT_M_m)        # (n_gamma, 2)
        Lambda[idx_gam, idx_mu] -= sigma_pm_inv_sq * QT_CvT_M
        Lambda[idx_mu, idx_gam] -= sigma_pm_inv_sq * QT_CvT_M.T

        # (μ, μ) Schur: -σ^{-4} Σ_i C_vT_i[2:4, 2:4]
        Lambda[idx_mu, idx_mu] -= (sigma_pm_inv_sq**2
                                   * Cv_m[:, 2:4, 2:4].sum(axis=0))

        # μ rhs: +σ^{-2} Σ_i a[member_i, 2:4]
        rhs[idx_mu] += sigma_pm_inv_sq * a[member_sidx, 2:4].sum(axis=0)

    # ── Per-image Schur corrections (r, r), (r, γ), (r, μ), cross-image ──────
    for j_idx, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None or K_img.get(img) is None:
            continue

        cs       = j_idx * nr
        sidx     = d["sidx"]
        # All fitting stars — no member restriction for (r,r) and (r,γ)
        use_fit  = d["use_for_fit"]
        use_fmem = use_fit & member_mask[sidx]   # fitting ∩ member for (r,μ)

        sidx_fit  = sidx[use_fit]
        K_fit     = K_img[img][use_fit]          # (n_fit, 5, N_R)
        Cv_fit    = C_vT[sidx_fit]               # (n_fit, 5, 5)

        # (r, r) diagonal Schur block — all fitting stars
        CvT_K_fit = np.einsum('nij,njk->nik', Cv_fit, K_fit)
        Lambda[cs:cs+nr, cs:cs+nr] -= np.einsum('nji,njk->ik', K_fit, CvT_K_fit)
        rhs[cs:cs+nr]              += np.einsum('nji,nj->i',   K_fit, a_align[sidx_fit])

        # (r, γ) Schur: K_fit^T C_vT Q_total_all — all fitting stars
        Qt_fit    = Q_total_all[sidx_fit]
        CvT_Q_fit = np.einsum('nij,njk->nik', Cv_fit, Qt_fit)
        KT_CvT_Q  = np.einsum('nji,njk->ik', K_fit, CvT_Q_fit)   # (N_R, n_gamma)
        Lambda[cs:cs+nr, idx_gam] -= KT_CvT_Q
        Lambda[idx_gam, cs:cs+nr] -= KT_CvT_Q.T

        # (r, μ) Schur: -σ^{-2} K_{fit∩member}^T C_vT M
        if use_fmem.any():
            sidx_fmem = sidx[use_fmem]
            K_fmem    = K_img[img][use_fmem]
            CvT_M_fm  = C_vT[sidx_fmem] @ M                        # (n_fm, 5, 2)
            KT_CvT_M  = np.einsum('nji,njk->ik', K_fmem, CvT_M_fm)  # (N_R, 2)
            Lambda[cs:cs+nr, idx_mu] -= sigma_pm_inv_sq * KT_CvT_M
            Lambda[idx_mu, cs:cs+nr] -= sigma_pm_inv_sq * KT_CvT_M.T

        # (r, r) cross-image coupling — all fitting stars (no member restriction)
        for j2_idx, img2 in enumerate(image_names):
            if j2_idx <= j_idx:
                continue
            d2 = solver._img_data.get(img2)
            if d2 is None or K_img.get(img2) is None:
                continue
            sidx_d2  = d2["sidx"]
            use2     = d2["use_for_fit"]    # all fitting, no member restriction
            sidx2    = sidx_d2[use2]
            K2       = K_img[img2][use2]

            common, ix1, ix2 = np.intersect1d(sidx_fit, sidx2,
                                               return_indices=True)
            if len(common) == 0:
                continue

            CvT_K2 = np.einsum('nij,njk->nik', C_vT[common], K2[ix2])
            block  = np.einsum('nji,njk->ik', K_fit[ix1], CvT_K2)
            cs2    = j2_idx * nr
            Lambda[cs:cs+nr,   cs2:cs2+nr] -= block
            Lambda[cs2:cs2+nr, cs:cs+nr]   -= block.T

    # ── Solve (Δr, γ, Δμ) with diagonal preconditioning ─────────────────────
    d_diag    = np.sqrt(np.maximum(np.abs(np.diag(Lambda)), 1e-30))
    d_inv     = 1.0 / d_diag
    Lambda_sc = d_inv[:, None] * Lambda * d_inv[None, :]
    try:
        C_shared_sc = np.linalg.inv(Lambda_sc)
    except np.linalg.LinAlgError:
        C_shared_sc = np.linalg.pinv(Lambda_sc)
    C_shared = d_inv[:, None] * C_shared_sc * d_inv[None, :]
    delta    = C_shared @ rhs

    r_hat      = r_current + delta[idx_r]
    gamma_hat  = delta[idx_gam]
    if not fit_cte_x:
        # gamma layout: [γx_hi(0:nb), γy_hi(nb:2nb), γx_lo(2nb:3nb), γy_lo(3nb:4nb)]
        gamma_hat[0:nb]       = 0.0
        gamma_hat[2*nb:3*nb]  = 0.0
    mu_pop_hat = mu_pop_current + delta[idx_mu]
    C_r        = C_shared[idx_r, idx_r]

    # ── Schur coupling corrections for stellar astrometry ─────────────────────
    # BP3M stores JU = -J (opposite sign Jacobian), so the loss is
    #   ½‖x_resid − J v − G γ‖² = ½‖x_resid + JU v − G γ‖²
    # and h_all = prior − JU^T C⁻¹ x_resid = prior + J^T C⁻¹ x_resid (correct RHS).
    # The coupling corrections that account for the solved (γ_hat, Δr, Δμ) are:
    #
    #   v_posterior = a + C_vT @ Q @ γ_hat          (γ: Q = JU^T C⁻¹ G = -J^T C⁻¹ G < 0)
    #               + C_vT @ K @ Δr                  (Δr: K = JU^T C⁻¹ X = -J^T C⁻¹ X < 0)
    #               + σ_pm⁻² C_vT[:,2:4] @ Δμ        (Δμ: prior mean update, sign +)
    #
    # All three use PLUS.  Q and K are negative (because JU = -J), so +C_vT@Q@γ
    # still gives a negative correction for positive γ (correct: CTE reduces PMra).

    # 1. Gamma correction (all active stars)
    if len(all_active_sidx) > 0:
        Qt = Q_total_all[all_active_sidx]          # (n_act, 5, n_gamma)
        Cv = C_vT[all_active_sidx]                 # (n_act, 5, 5)
        a[all_active_sidx] += np.einsum('nij,njk,k->ni', Cv, Qt, gamma_hat)

    # 2. Per-image transform correction (all active stars)
    for j_idx, img in enumerate(image_names):
        d_img = solver._img_data.get(img)
        if d_img is None or K_img.get(img) is None:
            continue
        cs = j_idx * nr
        delta_r_j = delta[cs:cs + nr]
        if np.allclose(delta_r_j, 0):
            continue
        K_j     = K_img[img]                          # (n_in_img, 5, N_R)
        sidx    = d_img['sidx']
        use_any = d_img.get('use_for_astrom', d_img['use_for_fit'])
        if not use_any.any():
            continue
        s_idx = sidx[use_any]
        np.add.at(a, s_idx,
                  np.einsum('nij,njk,k->ni',
                            C_vT[s_idx], K_j[use_any], delta_r_j))

    # 3. Population-PM prior correction (all prior stars: Gaia members + HST-only members)
    # h_all was built with mu_pop_current; the prior update from delta_mu must be
    # propagated through C_vT to keep stellar PMs consistent with mu_pop_hat.
    delta_mu = delta[idx_mu]
    if len(_all_prior) > 0 and np.any(delta_mu != 0):
        # a[i, :] += sigma_pm_inv_sq * C_vT[i, :, 2:4] @ delta_mu
        a[_all_prior] += sigma_pm_inv_sq * np.einsum(
            'nij,j->ni', C_vT[_all_prior, :, 2:4], delta_mu)

    # ── Full marginalisation over all shared parameters (r, γ, μ_pop) ─────────
    # C_vT = H_vv^{-1} is conditioned on fixed (r, γ, μ_pop).  The marginal
    # posterior including uncertainty in every shared parameter is:
    #
    #   C_v_marginal[i] = C_vT[i] + P[i] @ C_shared @ P[i]^T
    #
    # where P[i] = C_vT[i] @ J_i and J_i = [J_r[i] | J_γ[i] | J_μ[i]]
    #   J_r[i]     = Σ_j K_j[i]             at columns idx_r    (alignment)
    #   J_γ[i]     = Q_total_all[i]          at columns idx_gam  (CTE)
    #   J_μ[i]     = −σ_pm⁻² M              at columns idx_mu   (pop-PM prior)
    #
    # Processing in batches of BATCH stars to keep peak memory ≲ 200 MB.
    if len(all_active_sidx) > 0:
        C_vT = C_vT.copy()

        # Map global star index → position in all_active_sidx (−1 if inactive)
        _act_map = np.full(n_stars, -1, dtype=int)
        _act_map[all_active_sidx] = np.arange(len(all_active_sidx))
        n_act_total = len(all_active_sidx)

        BATCH = 500
        for b0 in range(0, n_act_total, BATCH):
            b1    = min(b0 + BATCH, n_act_total)
            b_act = all_active_sidx[b0:b1]   # global star indices in this batch
            n_b   = b1 - b0

            # Assemble P_b[i] = C_vT[i] @ J_i  for each star in batch
            P_b = np.zeros((n_b, 5, n_shared))

            # (a) Alignment (r): C_vT[i] @ K_j[i], accumulated over all images
            for j_idx, img in enumerate(image_names):
                d_img = solver._img_data.get(img)
                if d_img is None or K_img.get(img) is None:
                    continue
                cs   = j_idx * nr
                sidx = d_img['sidx']
                use  = d_img.get('use_for_astrom', d_img['use_for_fit'])
                s_g  = sidx[use]           # global indices, active in this image
                li   = _act_map[s_g]       # position in all_active_sidx
                in_b = (li >= b0) & (li < b1)
                if not in_b.any():
                    continue
                li_b = li[in_b] - b0       # index within batch
                K_j  = K_img[img][use][in_b]            # (m, 5, N_R)
                CK   = np.einsum('nij,njk->nik',
                                 C_vT[s_g[in_b]], K_j)  # (m, 5, N_R)
                np.add.at(P_b, (li_b, slice(None), slice(cs, cs + nr)), CK)

            # (b) CTE (γ): C_vT[i] @ Q_total[i]
            P_b[:, :, idx_gam] = np.einsum(
                'nij,njk->nik', C_vT[b_act], Q_total_all[b_act])

            # (c) Pop-PM prior (μ): −σ⁻² C_vT[i, :, 2:4]  (member stars only)
            _ip_raw = _act_map[_all_prior]           # positions in active array
            _ip = _ip_raw[(_ip_raw >= b0) & (_ip_raw < b1)]  # in this batch
            if len(_ip) > 0:
                _lb = _ip - b0
                P_b[_lb, :, idx_mu] = (
                    -sigma_pm_inv_sq * C_vT[all_active_sidx[_ip], :, 2:4])

            # C_v_extra = P_b @ C_shared @ P_b^T  (n_b, 5, 5)
            PC = (P_b.reshape(n_b * 5, n_shared) @ C_shared
                  ).reshape(n_b, 5, n_shared)
            C_vT[b_act] += np.einsum('nij,nkj->nik', PC, P_b)

    return r_hat, C_r, gamma_hat, mu_pop_hat, C_shared, a, K_img, C_vT


def _gamma_to_cte_params(gamma_hat: np.ndarray,
                         template: dict) -> dict:
    """
    Convert the gamma_hat vector from _joint_solve_cte into CTEChipParams.

    Layout (must match _joint_solve_cte):
      hi_x[0:nb], hi_y[nb:2nb], lo_x[2nb:3nb], lo_y[3nb:4nb]
    where nb = _cte_n_spatial(spatial_order) * (mag_poly_order + 1) - 1.
    """
    import copy
    params = copy.deepcopy(template)
    nb = _cte_n_spatial(params['hi'].spatial_order) * (params['hi'].mag_poly_order + 1) - 1
    params['hi'].gamma_x = gamma_hat[0*nb:1*nb].copy()
    params['hi'].gamma_y = gamma_hat[1*nb:2*nb].copy()
    params['lo'].gamma_x = gamma_hat[2*nb:3*nb].copy()
    params['lo'].gamma_y = gamma_hat[3*nb:4*nb].copy()
    return params


# ── Field mean PM via iterative sigma-clipping ────────────────────────────────

def _compute_warmstart_field_pm(
    data_root,
    field_name: str,
    n_sigma: float = 3.0,
    n_iter: int = 10,
) -> tuple[float, float] | None:
    """
    Estimate field bulk PM from Gaia-matched stars' pmra_xmatch in master_combined_v2.csv.

    Reads directly from master_combined_v2.csv (where pmra_xmatch is populated for
    all stars including Gaia-matched ones) rather than gaia_catalog (where the
    Gaia-matched rows have NaN pmra_xmatch in memory).  Excludes HST-only stars
    (gaia_source_id == 0) so their CTE-biased within-HST PMs don't inflate γ₀.

    Returns (pmra_mean, pmdec_mean) in mas/yr, or None if insufficient data.
    """
    import pandas as _pd
    mcat_path = (Path(data_root) / field_name / 'hst_xmatch'
                 / 'master_combined_v2.csv')
    if not mcat_path.exists():
        return None

    mcat = _pd.read_csv(mcat_path,
                        usecols=['gaia_source_id', 'pmra_xmatch', 'pmdec_xmatch'],
                        dtype={'gaia_source_id': np.int64},
                        low_memory=False)
    # Gaia-matched rows have gaia_source_id > 0; HST-only rows have 0
    gaia_rows = mcat[mcat['gaia_source_id'] > 0]
    pmra  = gaia_rows['pmra_xmatch'].to_numpy(float)
    pmdec = gaia_rows['pmdec_xmatch'].to_numpy(float)

    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra, pmdec = pmra[finite], pmdec[finite]

    if len(pmra) < 5:
        return None

    keep = np.ones(len(pmra), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra   = float(np.median(pmra[keep]))
        med_dec  = float(np.median(pmdec[keep]))
        dra, ddec = pmra[keep] - med_ra, pmdec[keep] - med_dec
        sigma    = max(float(np.median(np.hypot(dra, ddec))) / 0.6745, 0.01)
        new_keep = np.hypot(pmra - med_ra, pmdec - med_dec) < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    pmra_mean  = float(np.mean(pmra[keep]))
    pmdec_mean = float(np.mean(pmdec[keep]))
    print(f"  Warm-start field mean PM (Gaia xmatch, n={keep.sum()}/{len(pmra)}): "
          f"({pmra_mean:+.3f}, {pmdec_mean:+.3f}) mas/yr")
    return (pmra_mean, pmdec_mean)


def _compute_field_mean_pm(
    solver,
    gaia_catalog,
    hst_only_mask: np.ndarray,
    v_hat: np.ndarray,
    n_sigma: float = 3.0,
    n_iter: int = 10,
) -> tuple[float, float]:
    """
    Estimate the field bulk PM from Gaia-matched member stars.

    Uses the BP3M-fitted stellar PMs (v_hat) for Gaia-matched stars that have
    at least one detection contributing to the astrometry solution (use_for_astrom
    or use_for_fit).  Stars with no HST detections default to their prior (v_survey),
    which is 0 for 2p Gaia stars and would bias the mean.

    Applies iterative 2D sigma-clipping to find the member locus.  No prior on
    the mean PM is assumed — the locus is inferred from the data.

    Returns (pmra_mean, pmdec_mean) in mas/yr.
    """
    # v_hat shape: (n_stars, 5) — [Δα, Δδ, μ_α, μ_δ, plx]
    n_stars = v_hat.shape[0]

    # Build per-star "has_hst" mask: True if any detection with use_for_astrom or
    # use_for_fit=True.  Stars excluded from all detections have v_hat = v_survey
    # (0 for 2p) and must NOT be included in the field mean.
    n_hst = np.zeros(n_stars, dtype=int)
    for img in solver.image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx = d['sidx']
        use_any = d.get('use_for_astrom', d['use_for_fit'])
        np.add.at(n_hst, sidx[use_any], 1)

    # Gaia-matched stars with at least one contributing detection
    gaia_mask = (~hst_only_mask) & (n_hst >= 1)

    pmra  = v_hat[gaia_mask, 2].copy()   # mas/yr
    pmdec = v_hat[gaia_mask, 3].copy()   # mas/yr

    finite = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra  = pmra[finite]
    pmdec = pmdec[finite]

    if len(pmra) < 5:
        return (0.0, 0.0)

    # Iterative sigma-clipping in 2D PM space
    keep = np.ones(len(pmra), dtype=bool)
    for _ in range(n_iter):
        if keep.sum() < 5:
            break
        med_ra  = float(np.median(pmra[keep]))
        med_dec = float(np.median(pmdec[keep]))
        # Median absolute deviation → robust sigma
        dra   = pmra[keep]  - med_ra
        ddec  = pmdec[keep] - med_dec
        sigma = float(np.median(np.hypot(dra, ddec))) / 0.6745
        sigma = max(sigma, 0.01)   # avoid zero
        dist  = np.hypot(pmra - med_ra, pmdec - med_dec)
        new_keep = dist < n_sigma * sigma
        if new_keep.sum() == keep.sum():
            break   # converged
        keep = new_keep

    n_members = int(keep.sum())

    # Precision-weighted mean: use Gaia PM uncertainty (pmra_error) as weight proxy.
    # For Gaia 5p/6p stars: pmra_error is O(0.05-1) mas/yr → high weight.
    # For Gaia 2p stars: pmra_error is NaN → use diffuse prior σ=100 mas/yr → low weight.
    # Note: solver._C_VG_inv_per_star[:, 2] is 0 for 5p/6p stars (Gaia covariance is
    # stored separately), so we use the catalog pmra_error directly.
    try:
        if 'pmra_error' not in gaia_catalog.columns:
            raise KeyError('no pmra_error column')
        pmra_err = gaia_catalog['pmra_error'].to_numpy(float)[gaia_mask][finite]
        # Fill NaN (Gaia 2p / no Gaia PM) with diffuse prior σ = 100 mas/yr
        pmra_err = np.where(np.isfinite(pmra_err) & (pmra_err > 0), pmra_err, 100.0)
        prec = 1.0 / np.maximum(pmra_err, 0.01) ** 2
        w = prec[keep]
        w_sum = float(w.sum())
        if w_sum > 0 and np.ptp(w) > 0:
            pmra_mean  = float(np.sum(w * pmra[keep]) / w_sum)
            pmdec_mean = float(np.sum(w * pmdec[keep]) / w_sum)
        else:
            pmra_mean  = float(np.mean(pmra[keep]))
            pmdec_mean = float(np.mean(pmdec[keep]))
        weight_label = 'Gaia-precision-weighted'
    except Exception:
        pmra_mean  = float(np.mean(pmra[keep]))
        pmdec_mean = float(np.mean(pmdec[keep]))
        weight_label = 'unweighted'

    sigma_final = float(np.median(np.hypot(
        pmra[keep] - pmra_mean, pmdec[keep] - pmdec_mean)) / 0.6745)

    print(f"  Field mean PM ({weight_label}): ({pmra_mean:+.3f}, {pmdec_mean:+.3f}) mas/yr  "
          f"σ={sigma_final:.3f}  n_members={n_members}/{len(pmra)}")
    return (pmra_mean, pmdec_mean)


def _select_members_from_a(
    a_arr: np.ndarray,
    mu_pop: np.ndarray,
    hst_only_mask: np.ndarray,
    n_hst: np.ndarray,
    sigma_clip: float = 3.0,
    n_iter: int = 5,
    min_members: int = 5,
    init_window_masyr: float = 2.0,
    pm_sys_floor: float = 0.2,
    eligible_sidx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sigma-clip on PM distance from mu_pop to identify likely cluster members.

    Only Gaia-matched stars with at least one HST detection contributing to
    the astrometry solution are eligible (HST-only stars have unconstrained PMs
    and should not drive the population prior).

    Parameters
    ----------
    a_arr         : (n_stars, 5) posterior stellar astrometry from _joint_solve_cte
    mu_pop        : (2,) current population mean PM (mas/yr)
    hst_only_mask : (n_stars,) True for HST-only stars
    n_hst         : (n_stars,) count of contributing HST detections per star
    sigma_clip    : number of robust sigmas for membership cut
    n_iter        : max sigma-clipping iterations
    min_members   : minimum members to keep; returns all eligible if below
    init_window_masyr : initial PM distance window to seed sigma estimate
    eligible_sidx : if provided, restrict candidates to this index set (can
                    only drop members, never add stars not in the initial set)

    Returns
    -------
    (n_mem,) array of global star indices into the solver star array
    """
    eligible = (~hst_only_mask) & (n_hst >= 1)
    if eligible_sidx is not None:
        _init_mask = np.zeros(len(hst_only_mask), bool)
        _init_mask[eligible_sidx] = True
        eligible = eligible & _init_mask
    eidx        = np.where(eligible)[0]
    if len(eidx) < min_members:
        return eidx

    pmra  = a_arr[eidx, 2]
    pmdec = a_arr[eidx, 3]
    dist  = np.hypot(pmra - mu_pop[0], pmdec - mu_pop[1])

    # Bootstrap from a narrow window around mu_pop (same as warm_start_cte)
    keep = np.isfinite(dist) & (dist < init_window_masyr)
    if keep.sum() < min_members:
        keep = np.isfinite(dist)   # fall back to all finite

    for _ in range(n_iter):
        if keep.sum() < min_members:
            break
        sigma    = float(np.median(dist[keep])) / 0.6745
        sigma    = max(sigma, pm_sys_floor)
        new_keep = np.isfinite(dist) & (dist < sigma_clip * sigma)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    return eidx[keep]


# ── Warm start ────────────────────────────────────────────────────────────────

def _plot_warmstart_cte(
    solver, image_names, filtered_spi, t_launch_yr,
    gamma_warm, member_sidx, r_init, output_dir,
    cte_template=None,
) -> None:
    """
    Plot detection y-residuals (tangent-plane) vs y_raw before and after
    applying gamma_warm, for member stars.  Saved to output_dir/plots/.
    """
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.time import Time

    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    nr = solver.N_R
    n_stars = solver.C_survey_inv.shape[0]
    member_mask = np.zeros(n_stars, dtype=bool)
    member_mask[member_sidx] = True
    _tmpl = cte_template if cte_template is not None else default_cte_params()
    cte_w = _gamma_to_cte_params(gamma_warm, _tmpl)
    _nb   = _cte_n_spatial(_tmpl['hi'].spatial_order) * (_tmpl['hi'].mag_poly_order + 1) - 1

    rec = {c: {'y_raw': [], 'dy_b': [], 'dy_a': [], 'mag': []}
           for c in ('hi', 'lo')}

    for j_idx, img in enumerate(_tqdm(image_names, desc="  warmstart plot",
                                      unit="img", ncols=90, leave=False)):
        chip = _chip_from_image(img)
        if chip is None:
            continue
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx = d['sidx']
        use_f = d['use_for_fit'] & member_mask[sidx]
        use_a = ((d.get('use_for_astrom', d['use_for_fit'])
                  if getattr(solver, '_use_two_tier', False) else d['use_for_fit'])
                 & member_mask[sidx])
        use = use_f | use_a
        if not use.any():
            continue

        r_j  = r_init[j_idx * nr:(j_idx + 1) * nr]
        xys  = d.get('xys_orig', d['xys'])
        X    = d['X_mat']
        x_pred = np.einsum('nkl,l->nk', X, r_j)

        mag = d.get('mag_inst')
        if mag is None:
            continue
        Y_c = d['Y_c']
        p   = cte_w[chip]
        sdf = filtered_spi.get(img) if filtered_spi else None
        if sdf is not None and 'Y_orig' in sdf.columns and len(sdf) == len(mag):
            y_raw = sdf['Y_orig'].to_numpy(float)
        else:
            y_raw = Y_c + 2048.0  # Y_c = y_raw - 2048 for both chips

        hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                            format='mjd').jyear)
        dt   = hst_yr - t_launch_yr
        _x_raw_plt = (sdf['X_orig'].to_numpy(float) if sdf is not None and 'X_orig' in sdf.columns
                      and len(sdf) == len(mag) else d['X_c'] + 2048.0)
        xt   = (_x_raw_plt - 2048.0) / 2048.0
        yt   = np.abs(y_raw - p.y_readout_raw) / 2048.0
        ok   = np.isfinite(mag) & np.isfinite(y_raw)
        MP   = np.where(ok[:, None],
                        mag_poly_basis(mag, _tmpl['hi'].mag_poly_order,
                                       _tmpl['hi'].mag_norm_ref,
                                       _tmpl['hi'].mag_norm_scale), 0.0)
        PsiB = (dt * (MP[:, :, None] * cte_basis(xt, yt, _tmpl['hi'].spatial_order)[:, None, :])
                ).reshape(len(mag), _nb + 1)[:, 1:]   # drop degenerate 1×yt

        R_j    = r_j[:4].reshape(2, 2)
        cy     = _nb if chip == 'hi' else 3 * _nb
        dcte_y = (PsiB @ gamma_warm[cy:cy + _nb]) * float(R_j[1, 1])

        ui = np.where(use)[0]
        ok_u = ok[ui]
        ui   = ui[ok_u]
        if len(ui) == 0:
            continue

        rec[chip]['y_raw'].append(y_raw[ui])
        rec[chip]['dy_b'].append(xys[ui, 1] - x_pred[ui, 1])
        rec[chip]['dy_a'].append(xys[ui, 1] - x_pred[ui, 1] - dcte_y[ui])
        rec[chip]['mag'].append(mag[ui])

    for chip in ('hi', 'lo'):
        for k in rec[chip]:
            arr = rec[chip][k]
            rec[chip][k] = np.concatenate(arr) if arr else np.array([])

    def _binned(x, y, n_bins=15):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < n_bins:
            return np.array([]), np.array([])
        xo, yo = x[ok], y[ok]
        order  = np.argsort(xo)
        chunks = np.array_split(order, n_bins)
        xm, ym = [], []
        for ch in chunks:
            if len(ch) < 3:
                continue
            xm.append(xo[ch].mean()); ym.append(yo[ch].mean())
        return np.array(xm), np.array(ym)

    chip_labels = {'hi': 'hi chip (ext=1)', 'lo': 'lo chip (ext=4)'}
    try:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for row, chip in enumerate(('hi', 'lo')):
            y_r = rec[chip]['y_raw']
            m   = rec[chip]['mag']
            for col, (dy, panel_lbl) in enumerate(
                    [(rec[chip]['dy_b'], 'before CTE warm-start'),
                     (rec[chip]['dy_a'], 'after CTE warm-start')]):
                ax = axes[row, col]
                if len(y_r) == 0:
                    ax.set_visible(False)
                    continue
                ok = np.isfinite(y_r) & np.isfinite(dy) & np.isfinite(m)
                if ok.sum() == 0:
                    continue
                pcts = np.nanpercentile(m[ok], [0, 50, 100])
                for mlo, mhi, c, bl in [
                        (pcts[0], pcts[1], 'steelblue', 'bright'),
                        (pcts[1], pcts[2], 'firebrick', 'faint')]:
                    mask = ok & (m >= mlo) & (m < mhi)
                    if mask.sum() < 5:
                        continue
                    ax.scatter(y_r[mask], dy[mask], s=1, alpha=0.08,
                               color=c, linewidths=0, rasterized=True)
                    xm, ym = _binned(y_r[mask], dy[mask])
                    if len(xm):
                        ax.plot(xm, ym, color=c, lw=2,
                                label=f'{bl} ({mask.sum():,})')
                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
                ax.set_xlabel('y_raw (detector px)', fontsize=10)
                ax.set_ylabel('y residual (tangent-plane px)', fontsize=10)
                ax.set_title(f'{chip_labels[chip]} — {panel_lbl}', fontsize=10)
                ax.legend(fontsize=8)
                clip_v = np.nanpercentile(np.abs(dy[ok]), 97) if ok.any() else 1.0
                ax.set_ylim(-clip_v * 1.5, clip_v * 1.5)
        gy_hi0 = float(gamma_warm[_nb])
        fig.suptitle(
            f'CTE warm-start diagnostic  '
            f'γ_y_hi[0](yt²)={gy_hi0:+.4e}  '
            f'|γ_y_hi|={float(np.linalg.norm(gamma_warm[_nb:2*_nb])):.3e}  '
            f'|γ_y_lo|={float(np.linalg.norm(gamma_warm[3*_nb:4*_nb])):.3e}',
            fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / 'cte_warmstart_before_after.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: plots/cte_warmstart_before_after.png")
    except Exception as exc:
        import traceback
        print(f"  WARNING: cte_warmstart_before_after.png failed — {exc}")
        traceback.print_exc()


def _diagnose_cte_by_magbin(
    solver,
    image_names,
    filtered_spi,
    t_launch_yr,
    member_sidx,
    r_current,
    mag_bins=None,
    regularize_gamma=1e-8,
    output_dir=None,
    label='warmstart',
    cte_template=None,
):
    """
    Direct (no Schur) gamma regression per magnitude bin with fixed r.

    Fixes r = r_current and marginalises stellar positions analytically, then
    for each magnitude bin solves: H_gamma_bin @ gamma = GCs_xresid_bin.
    This bypasses the full Schur complement and directly shows whether the CTE
    signal is present and magnitude-dependent.

    Returns dict {(mag_lo, mag_hi): gamma_20vec}.
    """
    from astropy.time import Time as _ATime

    n_stars = solver.C_survey_inv.shape[0]
    member_mask = np.zeros(n_stars, dtype=bool)
    member_mask[member_sidx] = True

    nr      = solver.N_R
    # Diagnostic always uses K=0: fit 4 spatial coefficients per bin.
    # Polynomial magnitude dependence within a bin is meaningless because
    # stars in a narrow bin all have similar magnitudes (collinear columns).
    # nb=4 because the degenerate 1×yt term is excluded (see default_cte_params).
    _tmpl   = default_cte_params(0)
    nb      = 4
    n_gamma = 16
    cte_zero = _tmpl
    _use_two_tier = getattr(solver, '_use_two_tier', False)

    # Auto-range magnitude bins from actual data if not provided
    if mag_bins is None:
        _all_mags = []
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            mag = d.get('mag_inst')
            if mag is None:
                continue
            sidx = d['sidx']
            use_al = d['use_for_fit'] & member_mask[sidx]
            if _use_two_tier:
                use_ast = d.get('use_for_astrom', d['use_for_fit']) & member_mask[sidx]
            else:
                use_ast = use_al
            use_m = use_al | use_ast
            mf = mag[use_m]
            _all_mags.append(mf[np.isfinite(mf)])
        if _all_mags:
            _all_mags = np.concatenate(_all_mags)
        if len(_all_mags) > 2:
            lo, hi = np.percentile(_all_mags, [2, 98])
            edges = np.linspace(lo, hi, 7)   # 6 bins
            mag_bins = [(float(edges[i]), float(edges[i+1])) for i in range(len(edges)-1)]
            print(f"  Auto mag bins: {lo:.1f} – {hi:.1f}  ({len(mag_bins)} bins)")
        else:
            mag_bins = [(14, 18), (18, 20), (20, 22), (22, 24), (24, 28)]

    H_bins = {b: np.zeros((n_gamma, n_gamma)) for b in mag_bins}
    b_bins = {b: np.zeros(n_gamma) for b in mag_bins}
    n_bins = {b: 0 for b in mag_bins}

    print(f"  Accumulating CTE signal across {len(image_names)} images "
          f"({len(mag_bins)} mag bins, {len(member_sidx)} member stars)...")
    for j_idx, img in enumerate(_tqdm(image_names, desc="  mag-bin diag", unit="img",
                                      ncols=90, leave=False)):
        d = solver._img_data.get(img)
        if d is None:
            continue

        sidx      = d['sidx']
        use_align = d['use_for_fit'] & member_mask[sidx]
        if _use_two_tier:
            use_astrom_fl = d.get('use_for_astrom', d['use_for_fit'])
        else:
            use_astrom_fl = d['use_for_fit']
        use_mem = use_align | (use_astrom_fl & member_mask[sidx])
        if not use_mem.any():
            continue

        chip = _chip_from_image(img)
        if chip is None:
            continue

        mag = d.get('mag_inst')
        if mag is None:
            continue

        p    = cte_zero[chip]
        Y_c  = d['Y_c']
        spi_df = filtered_spi.get(img) if filtered_spi else None
        if (spi_df is not None and 'Y_orig' in spi_df.columns
                and len(spi_df) == len(mag)):
            y_raw = spi_df['Y_orig'].to_numpy(float)
        else:
            y_raw = Y_c + 2048.0  # Y_c = y_raw - 2048 for both chips

        cs  = j_idx * nr
        r_j = r_current[cs:cs + nr]
        X   = d['X_mat']
        xys = d.get('xys_orig', d['xys'])

        x_pred  = np.einsum('nkl,l->nk', X, r_j)
        x_resid = xys - x_pred

        Cs     = solver._compute_Cs(img, r_j)
        Cs_inv = np.linalg.inv(Cs)

        hst_yr    = float(_ATime(float(solver.images[img]['hst_time_mjd']),
                                  format='mjd').jyear)
        dt_scalar = hst_yr - t_launch_yr

        ok  = np.isfinite(mag) & np.isfinite(y_raw)
        cx, cy = (0, nb) if chip == 'hi' else (2*nb, 3*nb)
        R_j    = r_j[:4].reshape(2, 2)
        n_det  = len(sidx)
        if spi_df is not None and 'X_orig' in spi_df.columns and len(spi_df) == n_det:
            x_raw_jac = spi_df['X_orig'].to_numpy(float)
        else:
            x_raw_jac = d['X_c'] + 2048.0
        # Use raw coordinates for the CTE spatial basis
        xt  = (x_raw_jac - 2048.0) / 2048.0
        yt  = np.abs(y_raw - p.y_readout_raw) / 2048.0
        MP  = np.where(ok[:, None],
                       mag_poly_basis(mag, _tmpl['hi'].mag_poly_order,
                                      _tmpl['hi'].mag_norm_ref,
                                      _tmpl['hi'].mag_norm_scale), 0.0)
        PsiB = (dt_scalar * (MP[:, :, None] * cte_basis(xt, yt, _tmpl['hi'].spatial_order)[:, None, :])
                ).reshape(len(sidx), nb + 1)[:, 1:]  # (n, nb); drop degenerate 1×yt
        _jac_coeffs = _fit_gdc_jacobian_coeffs(spi_df)
        _J_gdc  = _eval_gdc_jacobian(_jac_coeffs, x_raw_jac, y_raw)  # (n, 2, 2)
        _R_eff  = np.einsum('ij,njk->nik', R_j, _J_gdc)              # (n, 2, 2)
        G = np.zeros((n_det, 2, n_gamma))
        G[ok, :, cx:cx+nb] = _R_eff[ok, :, 0:1] * PsiB[ok, None, :]
        G[ok, :, cy:cy+nb] = _R_eff[ok, :, 1:2] * PsiB[ok, None, :]

        for b in mag_bins:
            mag_lo, mag_hi = b
            in_bin = use_mem & (mag >= mag_lo) & (mag < mag_hi) & ok
            if not in_bin.any():
                continue
            H_bins[b] += np.einsum('nki,nkl,nlj->ij',
                                   G[in_bin], Cs_inv[in_bin], G[in_bin])
            b_bins[b] += np.einsum('nki,nkl,nl->i',
                                   G[in_bin], Cs_inv[in_bin], x_resid[in_bin])
            n_bins[b] += int(in_bin.sum())

    gamma_bins = {}
    for b in mag_bins:
        H = H_bins[b]
        if np.abs(np.diag(H)).max() < 1e-30:
            gamma_bins[b] = np.zeros(n_gamma)
            continue
        H_reg = H + regularize_gamma * np.eye(n_gamma)
        d_sc  = np.sqrt(np.maximum(np.abs(np.diag(H_reg)), 1e-30))
        d_inv = 1.0 / d_sc
        gamma_bins[b] = np.linalg.solve(
            d_inv[:, None] * H_reg * d_inv[None, :],
            d_inv * b_bins[b]) * d_inv

    # Print spatial coefficients per bin (1×yt excluded as degenerate)
    _basis_names = ['yt²', 'xt·yt', 'xt²·yt', 'xt·yt²']
    print(f"  CTE γ per mag bin (direct regression, n_mem={len(member_sidx)}, nb={nb}):")
    print(f"  {'bin':>10}  {'n':>8}  "
          + "  ".join(f"γ_yhi[{i}]({_basis_names[i]})" for i in range(4))
          + "  |γ_yhi|"
          + "  " + "  ".join(f"γ_ylo[{i}]" for i in range(4))
          + "  |γ_ylo|")
    for b in mag_bins:
        g = gamma_bins[b]
        vals_hi = "  ".join(f"{g[nb+i]:+.3e}" for i in range(4))
        vals_lo = "  ".join(f"{g[3*nb+i]:+.3e}" for i in range(4))
        norm_hi = float(np.linalg.norm(g[nb:2*nb]))
        norm_lo = float(np.linalg.norm(g[3*nb:4*nb]))
        print(f"  mag {b[0]:5.1f}-{b[1]:<5.1f}: n={n_bins[b]:7d}"
              f"  {vals_hi}  {norm_hi:.3e}"
              f"  {vals_lo}  {norm_lo:.3e}")

    if output_dir is not None:
        _plot_gamma_vs_magbin(gamma_bins, mag_bins, n_bins, output_dir, label, nb=nb)

    return gamma_bins


def _plot_gamma_vs_magbin(gamma_bins, mag_bins, n_bins, output_dir, label='warmstart', nb=4):
    """Plot γ_y per magnitude bin for CTE diagnostic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # basis_names matches the non-degenerate basis (yt excluded)
    _all_basis_names = ['yt²', 'xt·yt', 'xt²·yt', 'xt·yt²']
    basis_names = _all_basis_names[:nb]
    colors      = ['tomato', 'forestgreen', 'darkorchid', 'darkorange'][:nb]
    centers = [0.5 * (b[0] + b[1]) for b in mag_bins]
    counts  = [n_bins[b] for b in mag_bins]

    # (2 chips) × (nb basis functions + 1 norm) → 2 rows, 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # hi y-block starts at nb; lo y-block starts at 3*nb
    for col, (chip_label, y_off) in enumerate([('hi', nb), ('lo', 3*nb)]):
        # Top row: nb y-direction coefficients (constant poly term) per bin
        ax_coef = axes[0, col]
        for i in range(nb):
            vals = [float(gamma_bins[b][y_off + i]) for b in mag_bins]
            ax_coef.plot(centers, vals, 'o-', color=colors[i], lw=1.8, ms=7,
                         label=f'γ_y[{i}] ({basis_names[i]})')
        ax_coef.axhline(0, color='k', lw=0.8, ls='--')
        ax_coef.set_xlabel('F814W magnitude')
        ax_coef.set_ylabel('γ_y coefficient (const-poly term)')
        ax_coef.set_title(f'γ_y_{chip_label}: {nb} spatial coefficients vs mag bin')
        ax_coef.legend(fontsize=8)

        # Bottom row: |γ_y| norm (all nb coefficients) + annotation of n per bin
        ax_norm = axes[1, col]
        norms = [float(np.linalg.norm(gamma_bins[b][y_off:y_off + nb]))
                 for b in mag_bins]
        ax_norm.plot(centers, norms, 's-', color='navy', lw=2, ms=8)
        for xc, yv, nc in zip(centers, norms, counts):
            ax_norm.text(xc, yv, f'  {nc//1000:.0f}k', fontsize=8, va='bottom')
        ax_norm.axhline(0, color='k', lw=0.8, ls='--')
        ax_norm.set_xlabel('F814W magnitude')
        ax_norm.set_ylabel(f'|γ_y_{chip_label}|')
        ax_norm.set_title(f'|γ_y_{chip_label}| norm vs mag bin')

    fig.suptitle(f'CTE γ_y by magnitude bin — direct regression ({label})\n'
                 f'basis: [{", ".join(basis_names)}]',
                 fontsize=11)
    fig.tight_layout()
    fname = f'cte_gamma_by_magbin_{label}.png'
    fig.savefig(plot_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plots/{fname}")


def warm_start_cte(
    solver,
    image_names: list[str],
    filtered_spi: dict | None,
    t_launch_yr: float,
    member_sidx_gaia: np.ndarray,
    member_sidx_hst: np.ndarray,
    sigma_pm: float,
    plx_pop: float,
    sigma_plx_tot: float,
    mu_pop_prior: np.ndarray,
    C_pop_prior_inv: np.ndarray,
    r_init: np.ndarray,
    cte_template: dict | None = None,
    regularize_gamma: float = 1e-8,
    output_dir: Path | None = None,
    n_gaia_warmstart_iters: int = 3,
    fit_cte_x: bool = True,
) -> tuple[dict[str, CTEChipParams], np.ndarray]:
    """
    Warm-start CTE in four phases:

    [1/4] K=0 magnitude-bin diagnostic (all members, direct regression).
    [2/4] Gaia-only joint (r, μ_pop) refinement — isolates the reference
          frame from CTE-distorted HST-only PMs.  Updates solver._r_hat_current
          so the full joint loop inherits the cleaner alignment.
    [3/4] Full-member CTE warmstart — Gaia drives μ_pop, HST-only stars
          receive the population prior for PM regularisation but do not
          contribute to the μ_pop Schur correction.  Extracts γ_warm.
    [4/4] Diagnostic plots.

    Returns
    -------
    (cte_warm, mu_pop_warm, r_ws, a_arr_ws)
        cte_warm    : dict[str, CTEChipParams] with warm-started γ coefficients
        mu_pop_warm : (2,) population mean PM refined from Gaia members
        r_ws        : (n_r,) image parameters from phase [2/4] Gaia-only refinement
        a_arr_ws    : (n_stars, 5) stellar astrometry from phase [3/4] joint solve
    """
    if cte_template is None:
        cte_template = default_cte_params()

    member_sidx_init = np.concatenate([member_sidx_gaia, member_sidx_hst])
    _mag_order  = cte_template['hi'].mag_poly_order
    _spat_order = cte_template['hi'].spatial_order
    nb = _cte_n_spatial(_spat_order) * (_mag_order + 1) - 1
    print(f"\n  {'─'*56}")
    print(f"  CTE warm start  "
          f"(mag_poly_order={_mag_order}  spatial_order={_spat_order}  nb={nb}  n_images={len(image_names)})")
    print(f"  {'─'*56}")
    print(f"  μ_pop_prior = ({mu_pop_prior[0]:+.4f}, {mu_pop_prior[1]:+.4f}) mas/yr  "
          f"n_gaia={len(member_sidx_gaia)}  n_hst={len(member_sidx_hst)}")

    # ── Phase 1: per-magnitude-bin diagnostic (bypasses Schur complement) ──────
    print(f"\n  [1/4] Magnitude-bin CTE diagnostic (K=0 direct regression)...")
    import time as _wtime
    _t0 = _wtime.time()
    gamma_bins = _diagnose_cte_by_magbin(
        solver, image_names, filtered_spi, t_launch_yr,
        member_sidx_init, r_init,
        regularize_gamma=regularize_gamma,
        output_dir=output_dir,
        label='warmstart',
        cte_template=cte_template,
    )
    print(f"  [1/4] done ({_wtime.time()-_t0:.1f}s)")

    # ── Phase 2: Gaia-only (r, μ_pop) refinement ─────────────────────────────
    # Use only Gaia-matched members (well-constrained individual PMs) to refine
    # the reference frame before the CTE fit.  HST-only stars are excluded here
    # because their PMs are derived from HST positions, which are CTE-distorted.
    print(f"\n  [2/4] Gaia-only (r, μ_pop) refinement "
          f"({n_gaia_warmstart_iters} iter, {len(member_sidx_gaia)} stars)...")
    _t0 = _wtime.time()
    r_ws  = r_init.copy()
    mu_ws = mu_pop_prior.copy()
    _r_prev_ws  = r_ws.copy()
    _mu_prev_ws = mu_ws.copy()
    for _ws_it in range(n_gaia_warmstart_iters):
        _result_g = _joint_solve_cte(
            solver, image_names, cte_template, t_launch_yr, filtered_spi,
            member_sidx_gaia, sigma_pm, plx_pop, sigma_plx_tot,
            mu_ws, mu_pop_prior, C_pop_prior_inv, r_ws,
            regularize_gamma=regularize_gamma,
            hst_prior_sidx=None,
            fit_cte_x=fit_cte_x,
        )
        r_ws, _, _, mu_ws, _C_shared_ws, _, _, _ = _result_g
        solver._update_R(r_ws)
        solver._update_geometry(r_ws, solver.v_survey)
        _dr_ws  = float(np.max(np.abs(r_ws  - _r_prev_ws)))
        _dmu_ws = float(np.max(np.abs(mu_ws - _mu_prev_ws)))
        _C_mu_ws   = _C_shared_ws[-2:, -2:]
        _sig_ra_ws = float(np.sqrt(_C_mu_ws[0, 0]))
        _sig_de_ws = float(np.sqrt(_C_mu_ws[1, 1]))
        _rho_ws    = float(_C_mu_ws[0, 1] / (_sig_ra_ws * _sig_de_ws + 1e-30))
        print(f"    iter {_ws_it+1}/{n_gaia_warmstart_iters}: "
              f"Δr={_dr_ws:.3e}  Δμ={_dmu_ws:.4f}  "
              f"μ_pop=({mu_ws[0]:+.4f}±{_sig_ra_ws:.4f}, "
              f"{mu_ws[1]:+.4f}±{_sig_de_ws:.4f}) mas/yr  ρ={_rho_ws:+.3f}")
        _r_prev_ws  = r_ws.copy()
        _mu_prev_ws = mu_ws.copy()
    print(f"  [2/4] done ({_wtime.time()-_t0:.1f}s)")

    # ── Phase 3: full-member CTE warmstart (two-tier) ─────────────────────────
    # Gaia members drive μ_pop; HST-only members get the population prior
    # (PM regularisation) but do not pull μ_pop.
    print(f"\n  [3/4] Joint CTE warmstart ({len(member_sidx_gaia)} Gaia + "
          f"{len(member_sidx_hst)} HST-only, two-tier)...")
    _t0 = _wtime.time()
    result = _joint_solve_cte(
        solver, image_names, cte_template, t_launch_yr, filtered_spi,
        member_sidx_gaia, sigma_pm, plx_pop, sigma_plx_tot,
        mu_ws, mu_pop_prior, C_pop_prior_inv, r_ws,
        regularize_gamma=regularize_gamma,
        hst_prior_sidx=member_sidx_hst,
        fit_cte_x=fit_cte_x,
    )
    _, _, gamma_warm, _, _, a_arr_ws, _, _ = result
    _dt = _wtime.time() - _t0

    cte_warm = _gamma_to_cte_params(gamma_warm, cte_template)

    gy_hi   = float(np.linalg.norm(gamma_warm[nb:2*nb]))
    gy_lo   = float(np.linalg.norm(gamma_warm[3*nb:4*nb]))
    gyx0    = float(gamma_warm[nb])
    gyx0_lo = float(gamma_warm[3*nb])
    print(f"  [3/4] done ({_dt:.1f}s)")
    print(f"  γ_y_hi[0](yt²) = {gyx0:+.4e}   |γ_y_hi| = {gy_hi:.3e}")
    print(f"  γ_y_lo[0](yt²) = {gyx0_lo:+.4e}   |γ_y_lo| = {gy_lo:.3e}")

    # ── Phase 4: diagnostic plots ──────────────────────────────────────────────
    if output_dir is not None:
        print(f"\n  [4/4] Saving warmstart diagnostic plots...")
        _t0 = _wtime.time()
        _plot_warmstart_cte(solver, image_names, filtered_spi, t_launch_yr,
                            gamma_warm, member_sidx_init, r_ws, output_dir,
                            cte_template=cte_template)
        print(f"  [4/4] done ({_wtime.time()-_t0:.1f}s)")

    print(f"\n  {'─'*56}")
    print(f"  CTE warm start complete.")
    print(f"  μ_pop_warm = ({mu_ws[0]:+.4f}±{_sig_ra_ws:.4f}, "
          f"{mu_ws[1]:+.4f}±{_sig_de_ws:.4f}) mas/yr  ρ={_rho_ws:+.3f}")
    print(f"  {'─'*56}\n")
    return cte_warm, mu_ws, r_ws, a_arr_ws


def _warm_start_cte_residuals(
    img_to_df: dict,
    solver,
    image_names: list[str],
    r_hat_init: np.ndarray,
    t_epoch0_yr: float,
    field_mean_pm: tuple[float, float] | None = None,
    data_root: Path | None = None,
    field_name: str | None = None,
) -> dict[str, CTEChipParams]:
    """
    Estimate initial γ_y by fitting per-star PM residuals to the CTE model.

    Since v2 BP3M absorbed CTE into individual star PMs, the PM residual
    (pm_i − field_mean) directly encodes the CTE-induced proper motion bias:

        Δpm_dec_i  ≈  pscale · d · φ(mag_i) · B_y(y_raw_i, X_c_i) @ γ_y

    WLS with 1/σ²_pm weights recovers γ_y from the per-star PM pattern.
    An intercept column is included to absorb the mean CTE level (which was
    also absorbed into the v2 field-mean PM and thus cancels out of residuals
    only up to sampling noise).

    Uses sigma_pmdec from master_combined_v2.csv (BP3M v2 fit uncertainty) for
    ALL stars including HST-only, which gives proper weights to faint stars
    that carry the strongest CTE signal.  The per-star key is the rounded
    (pmra_xmatch, pmdec_xmatch) pair, which has near-zero collision rate.

    field_mean_pm : (pmra_mean, pmdec_mean) in mas/yr.  Should be the true
        field mean, e.g. from Gaia-bright member cross-match.
    data_root, field_name : if provided, master_combined_v2.csv is read to
        build the sigma_pmdec lookup.  Required for proper HST-only weights.
    """
    import pandas as _pd

    print("  Warm-starting CTE from v2 PM residuals...")

    mean_pmra  = float(field_mean_pm[0]) if field_mean_pm else 0.0
    mean_pmdec = float(field_mean_pm[1]) if field_mean_pm else 0.0

    params = default_cte_params()
    n_r    = solver.N_R

    # ── Build sigma_pmdec lookup from master catalog ──────────────────────────
    # master_combined_v2.csv has sigma_pmdec (v2 fit uncertainty) for ALL stars,
    # including HST-only stars that lack Gaia priors.  Key: rounded PM pair.
    sigma_lookup: dict[str, float] = {}
    if data_root is not None and field_name is not None:
        mcat_path = (Path(data_root) / field_name / 'hst_xmatch'
                     / 'master_combined_v2.csv')
        if mcat_path.exists():
            _mc = _pd.read_csv(mcat_path,
                               usecols=['pmra_xmatch', 'pmdec_xmatch', 'sigma_pmdec'],
                               low_memory=False)
            fin = _mc['pmra_xmatch'].notna() & _mc['sigma_pmdec'].notna()
            _mc = _mc[fin]
            _keys = (_mc['pmra_xmatch'].round(6).astype(str) + '_'
                     + _mc['pmdec_xmatch'].round(6).astype(str))
            sigma_lookup = dict(zip(_keys, _mc['sigma_pmdec']))
            print(f"    Loaded sigma_pmdec lookup: {len(sigma_lookup):,} stars")

    # ── Mean transformation coefficients and pixel scale per chip ────────────
    # r_j = [a, b, c, d, w, z, ...] — R = [[a,b],[c,d]], so:
    #   a = R[0,0]: x-pixel → RA sky (x-CTE warm start)
    #   d = R[1,1]: y-pixel → Dec sky (y-CTE warm start)
    chip_a      = {'hi': [], 'lo': []}
    chip_d      = {'hi': [], 'lo': []}
    chip_pscale = {'hi': [], 'lo': []}
    for j_idx, img in enumerate(image_names):
        chip = _chip_from_image(img)
        if chip is None:
            continue
        r_j    = r_hat_init[j_idx * n_r : (j_idx + 1) * n_r]
        chip_a[chip].append(float(r_j[0]))
        chip_d[chip].append(float(r_j[3]))
        chip_pscale[chip].append(float(solver.images[img]['orig_pixel_scale']))

    mean_a      = {c: float(np.mean(v)) if v else 1.0  for c, v in chip_a.items()}
    mean_d      = {c: float(np.mean(v)) if v else 1.0  for c, v in chip_d.items()}
    mean_pscale = {c: float(np.mean(v)) if v else 50.0 for c, v in chip_pscale.items()}

    # ── Collect per-detection data from img_to_df ─────────────────────────────
    _pm_window = 2.0   # mas/yr half-width membership pre-selection

    rows = []
    for img in _tqdm(image_names, desc="  collecting detections", unit="img",
                     ncols=90, leave=False):
        chip = _chip_from_image(img)
        if chip is None or img not in img_to_df:
            continue
        df = img_to_df[img]
        # Accept either canonical or field-specific column names
        y_col   = ('y_raw'   if 'y_raw'   in df.columns else
                   'Y_orig'  if 'Y_orig'  in df.columns else None)
        mag_col = ('mag_gdc' if 'mag_gdc' in df.columns else
                   'mag'     if 'mag'     in df.columns else None)
        needed_base = {'x_gdc', 'pmra_xmatch', 'pmdec_xmatch'}
        if y_col is None or mag_col is None or not needed_base.issubset(df.columns):
            continue

        pmra_arr  = df['pmra_xmatch'].to_numpy(float)
        pmdec_arr = df['pmdec_xmatch'].to_numpy(float)
        member_mask = (
            np.isfinite(pmra_arr)  & (np.abs(pmra_arr  - mean_pmra)  < _pm_window)
            & np.isfinite(pmdec_arr) & (np.abs(pmdec_arr - mean_pmdec) < _pm_window)
            & df[y_col].notna().to_numpy()
            & df[mag_col].notna().to_numpy()
        )
        if not member_mask.any():
            continue

        idx = df.index[member_mask]
        tmp = df.loc[idx, [y_col, 'x_gdc', mag_col,
                            'pmra_xmatch', 'pmdec_xmatch']].copy()
        # Normalise to canonical column names for the aggregation step below
        if y_col != 'y_raw':
            tmp = tmp.rename(columns={y_col: 'y_raw'})
        if mag_col != 'mag_gdc':
            tmp = tmp.rename(columns={mag_col: 'mag_gdc'})
        tmp['chip']    = chip

        # Build sigma_pmdec: prefer master-catalog v2 fit uncertainty,
        # fall back to Gaia xmatch sigma, then use 5.0 as last resort.
        pm_keys = (tmp['pmra_xmatch'].round(6).astype(str) + '_'
                   + tmp['pmdec_xmatch'].round(6).astype(str))
        if sigma_lookup:
            sigma_from_master = pm_keys.map(sigma_lookup)
        else:
            sigma_from_master = _pd.Series(np.nan, index=tmp.index)

        if 'sigma_pmdec_xmatch' in df.columns:
            sigma_xmatch = df.loc[idx, 'sigma_pmdec_xmatch'].values
        else:
            sigma_xmatch = np.full(len(tmp), np.nan)

        # Prefer master-catalog sigma; fall back to xmatch sigma; then 5.0
        sigma_vals = np.where(
            np.isfinite(sigma_from_master.values), sigma_from_master.values,
            np.where(np.isfinite(sigma_xmatch), sigma_xmatch, 5.0))
        tmp['sigma_pmdec'] = sigma_vals

        # Per-star key: rounded (pmra, pmdec) pair — near-zero collision rate
        tmp['star_key'] = pm_keys.values
        rows.append(tmp)

    if not rows:
        print("  WARNING: no member detections found — returning zero γ_y")
        return params

    all_dets = _pd.concat(rows, ignore_index=True)

    # Aggregate per (star_key, chip): median position/mag, first PM/sigma
    agg = (all_dets
           .groupby(['star_key', 'chip'], sort=False)
           .agg(y_raw        = ('y_raw',        'median'),
                x_gdc        = ('x_gdc',         'median'),
                mag_gdc      = ('mag_gdc',        'median'),
                pmra_xmatch  = ('pmra_xmatch',    'first'),
                pmdec_xmatch = ('pmdec_xmatch',   'first'),
                sigma_pmdec  = ('sigma_pmdec',    'first'),
                n_det        = ('y_raw',           'count'))
           .reset_index())

    n_hi = int((agg['chip'] == 'hi').sum())
    n_lo = int((agg['chip'] == 'lo').sum())
    print(f"    Member stars (window ±{_pm_window} mas/yr): hi={n_hi:,}  lo={n_lo:,}")

    # ── Per-chip weighted least squares for y-CTE and x-CTE ──────────────────
    for chip in ('hi', 'lo'):
        p       = params[chip]
        pscale  = mean_pscale[chip]   # mas/px
        d_coef  = mean_d[chip]        # ≈ R[1,1], maps y-px → sky-Dec
        a_coef  = mean_a[chip]        # ≈ R[0,0], maps x-px → sky-RA

        sub = agg[agg['chip'] == chip].reset_index(drop=True)

        ok = (sub['pmdec_xmatch'].notna()
              & sub['pmra_xmatch'].notna()
              & sub['y_raw'].notna()
              & sub['mag_gdc'].notna()
              & sub['sigma_pmdec'].notna()
              & (sub['sigma_pmdec'] > 0)
              & (sub['sigma_pmdec'] < 50.0))   # exclude completely unconstrained (2p prior ~100)
        sub = sub[ok].reset_index(drop=True)

        if len(sub) < 20:
            print(f"    {chip}: only {len(sub)} stars — skipping")
            continue

        y_raw  = sub['y_raw'].to_numpy(float)
        X_c    = sub['x_gdc'].to_numpy(float) - 2048.0
        mag    = sub['mag_gdc'].to_numpy(float)
        dpm_dec = sub['pmdec_xmatch'].to_numpy(float) - mean_pmdec   # mas/yr
        dpm_ra  = sub['pmra_xmatch'].to_numpy(float)  - mean_pmra    # mas/yr
        sigma   = sub['sigma_pmdec'].to_numpy(float).clip(0.01)

        phi = func1_mag(mag)
        xt  = X_c / 2048.0   # approx: x_raw ≈ x_gdc; x_raw unavailable in PM-agg data
        yt  = np.abs(y_raw - p.y_readout_raw) / 2048.0
        B   = cte_basis(xt, yt, p.spatial_order)[:, 1:]   # drop degenerate 1×yt

        w   = 1.0 / sigma**2
        sqw = np.sqrt(w)

        # y-CTE: dpm_dec ≈ pscale·d·phi·B @ gamma_y (+ intercept)
        Ay_phys = pscale * d_coef * phi[:, None] * B   # (n, 4) physical units (mas/yr)
        col_scale_y = np.std(Ay_phys, axis=0).clip(min=1e-30)
        A_y = np.column_stack([np.ones(len(sub)), Ay_phys / col_scale_y])
        try:
            coeffs_y, _, _, _ = np.linalg.lstsq(
                A_y * sqw[:, None], dpm_dec * sqw, rcond=None)
            params[chip].gamma_y = coeffs_y[1:] / col_scale_y
        except np.linalg.LinAlgError:
            print(f"    {chip}: y-CTE WLS failed")
            coeffs_y = np.zeros(5)

        pred_y = A_y @ coeffs_y
        rms_y  = float(np.sqrt(np.average((dpm_dec - pred_y)**2, weights=w)))
        print(f"    {chip} y-CTE: γ_y[0](yt²)={params[chip].gamma_y[0]:.3e}  "
              f"intercept={coeffs_y[0]:+.3f} mas/yr  "
              f"rms={rms_y:.3f} mas/yr  n={len(sub):,}")

        # x-CTE: dpm_ra ≈ pscale·a·phi·B @ gamma_x (+ intercept)
        Ax_phys = pscale * a_coef * phi[:, None] * B   # (n, 4) physical units (mas/yr)
        col_scale_x = np.std(Ax_phys, axis=0).clip(min=1e-30)
        A_x = np.column_stack([np.ones(len(sub)), Ax_phys / col_scale_x])
        try:
            coeffs_x, _, _, _ = np.linalg.lstsq(
                A_x * sqw[:, None], dpm_ra * sqw, rcond=None)
            params[chip].gamma_x = coeffs_x[1:] / col_scale_x
        except np.linalg.LinAlgError:
            print(f"    {chip}: x-CTE WLS failed")
            coeffs_x = np.zeros(5)

        pred_x = A_x @ coeffs_x
        rms_x  = float(np.sqrt(np.average((dpm_ra - pred_x)**2, weights=w)))
        print(f"    {chip} x-CTE: γ_x[0](yt²)={params[chip].gamma_x[0]:.3e}  "
              f"intercept={coeffs_x[0]:+.3f} mas/yr  "
              f"rms={rms_x:.3f} mas/yr")

    # ── Cross-chip fallback for wrong-sign γ_y0 ──────────────────────────────
    # apply_cte_to_solver adds Φ·B@γ to observed positions.  For y-CTE:
    # CTE shifts stars away from readout (+Y' direction), so correction needs
    # δCTE_y < 0 → γ_y[0] < 0.  If v2 over-absorbed the CTE for one chip,
    # the PM residuals can give the wrong sign.  Cross-seed from the other chip.
    other = {'hi': 'lo', 'lo': 'hi'}
    for chip in ('hi', 'lo'):
        gam = params[chip].gamma_y
        if gam[0] > 0:
            fb = params[other[chip]].gamma_y
            if fb[0] < 0:
                params[chip].gamma_y = fb.copy()
                print(f"    {chip}: γ_y[0]={gam[0]:.3e} (wrong sign) "
                      f"→ seeded from {other[chip]}: {fb[0]:.3e}")
            else:
                params[chip].gamma_y = -np.abs(gam)
                print(f"    {chip}: γ_y[0]={gam[0]:.3e} (both wrong sign) → negated")

    return params


# ── Convergence CSV ────────────────────────────────────────────────────────────

def _save_cte_convergence(output_dir: Path, outer_iter: int,
                          cte_params: dict, info: dict) -> None:
    import csv
    csv_path = output_dir / 'cte_convergence.csv'
    _gy_names = [f'gamma_y{k}' for k in range(5)]
    _gx_names = [f'gamma_x{k}' for k in range(5)]
    fieldnames = ['iter', 'chip'] + _gy_names + _gx_names + ['rms_y', 'rms_x', 'n_det']
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for chip in ('hi', 'lo'):
            p  = cte_params[chip]
            ci = info.get(chip, {})
            row = {
                'iter':  outer_iter,
                'chip':  chip,
                'rms_y': f'{ci.get("rms_y", float("nan")):.6f}',
                'rms_x': f'{ci.get("rms_x", float("nan")):.6f}',
                'n_det': ci.get('n_det', 0),
            }
            for k in range(5):
                row[f'gamma_y{k}'] = f'{p.gamma_y[k]:.6e}'
                row[f'gamma_x{k}'] = f'{p.gamma_x[k]:.6e}'
            writer.writerow(row)


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def _plot_cte_diagnostics(output_dir: Path, cte_params: dict,
                          before_npz: Path, after_npz: Path,
                          image_names: list[str], solver,
                          file_prefix: str = '',
                          astrom_csv: Path | None = None) -> None:
    """
    Create CTE diagnostic figures:
      1. cte_amplitude.png   — δCTE_y amplitude vs Y_c for mag bins, per chip
      2. cte_before_after.png — dy_gdc vs Y_c before/after correction, per chip × epoch
      3. cte_slope_vs_mag.png — slope d(dy_gdc)/d(Y_c) vs mag, before/after, both chips
      4. cte_convergence.png  — δ and |γ_y| vs outer iteration (from convergence CSV)
      5. cte_2d_map.png       — 2D detector maps of dy_gdc before/after
    """
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import pandas as pd
    from astropy.time import Time

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load residual arrays ──────────────────────────────────────────────────
    def _load_npz(path):
        if path is None or not path.exists():
            return None
        return np.load(path, allow_pickle=True)

    npz_before = _load_npz(before_npz)
    npz_after  = _load_npz(after_npz)

    _has_residuals = (npz_before is not None or npz_after is not None)
    _has_astrom    = (astrom_csv is not None and Path(astrom_csv).exists())
    if not _has_residuals and not _has_astrom:
        print("  _plot_cte_diagnostics: no residual arrays or stellar astrometry found — skipping")
        return

    # Build combined per-chip arrays from npz, annotated with chip/epoch/dt
    def _collect_from_npz(npz, image_names, solver):
        if npz is None:
            return None
        recs = {c: {'X_c': [], 'Y_c': [], 'y_raw': [], 'dx': [], 'dy': [],
                    'mag': [], 'dt': [], 'epoch_yr': []} for c in ('hi', 'lo')}
        for img in image_names:
            if f'{img}_X_c' not in npz:
                continue
            chip = _chip_from_image(img)
            if chip is None:
                continue
            hst_yr = float(Time(float(solver.images[img]['hst_time_mjd']),
                                 format='mjd').jyear)
            X_c = npz[f'{img}_X_c'].astype(float)
            Y_c = npz[f'{img}_Y_c'].astype(float)
            dx  = npz[f'{img}_dx_gdc'].astype(float)
            dy  = npz[f'{img}_dy_gdc'].astype(float)
            mag = npz[f'{img}_mag_inst'].astype(float)
            # Raw chip-local y — prefer stored value, fall back to GDC approximation
            if f'{img}_y_raw' in npz:
                y_raw = npz[f'{img}_y_raw'].astype(float)
            else:
                y_raw = Y_c + 2048.0  # Y_c = y_raw - 2048 for both chips
            ok  = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & np.isfinite(y_raw)
            n   = ok.sum()
            if n == 0:
                continue
            recs[chip]['X_c'].append(X_c[ok])
            recs[chip]['Y_c'].append(Y_c[ok])
            recs[chip]['y_raw'].append(y_raw[ok])
            recs[chip]['dx'].append(dx[ok])
            recs[chip]['dy'].append(dy[ok])
            recs[chip]['mag'].append(mag[ok])
            recs[chip]['dt'].append(np.full(n, hst_yr))
            recs[chip]['epoch_yr'].append(np.full(n, round(hst_yr)))
        for chip in ('hi', 'lo'):
            for k in recs[chip]:
                arr = recs[chip][k]
                recs[chip][k] = np.concatenate(arr) if arr else np.array([])
        return recs

    data_before = _collect_from_npz(npz_before, image_names, solver)
    data_after  = _collect_from_npz(npz_after,  image_names, solver)

    chip_colors = {'hi': 'steelblue', 'lo': 'darkorange'}
    mag_ref = _MAG_REF

    # ── Figure 1: CTE amplitude vs raw y for mag bins ────────────────────────
    # Row 0 (always): δCTE_y — hi chip | lo chip
    # Row 1 (if fit_cte_x): δCTE_x — hi chip | lo chip
    try:
        _has_cte_x = any(
            np.any(cte_params[c].gamma_x != 0) for c in ('hi', 'lo')
            if c in cte_params
        )
        n_rows = 2 if _has_cte_x else 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows),
                                 squeeze=False)

        all_mjds  = [float(solver.images[img]['hst_time_mjd'])
                     for img in image_names if img in solver.images]
        t0_mjd    = min(all_mjds)
        t1_mjd    = max(all_mjds)
        dt_max    = (t1_mjd - t0_mjd) / 365.25

        # Collect actual mag_inst percentiles from after-CTE npz for labelling
        all_mag_inst = []
        npz_src = npz_after if npz_after is not None else npz_before
        if npz_src is not None:
            for img in image_names:
                key = f'{img}_mag_inst'
                if key in npz_src:
                    m = npz_src[key].astype(float)
                    all_mag_inst.append(m[np.isfinite(m)])
        if all_mag_inst:
            all_mag_inst = np.concatenate(all_mag_inst)
            mag_p10, mag_p50, mag_p90 = np.percentile(all_mag_inst, [10, 50, 90])
        else:
            mag_p10, mag_p50, mag_p90 = -9.5, -7.8, -6.8  # fallback

        mag_bins = [(mag_p10, 'steelblue', f'bright (p10={mag_p10:.1f})'),
                    (mag_p50, 'green',     f'med   (p50={mag_p50:.1f})'),
                    (mag_p90, 'firebrick', f'faint (p90={mag_p90:.1f})')]

        # Rows: 0 = y-CTE, 1 = x-CTE (if fit)
        components = [('gamma_y', 'δCTE_y')]
        if _has_cte_x:
            components.append(('gamma_x', 'δCTE_x'))

        for row_i, (gamma_attr, cte_lbl) in enumerate(components):
            for col_i, chip in enumerate(('hi', 'lo')):
                ax = axes[row_i, col_i]
                p  = cte_params[chip]
                if p.y_readout_raw > 2000:   # hi chip
                    y_raw_grid = np.linspace(2048.0, 4096.0, 500)
                else:                        # lo chip
                    y_raw_grid = np.linspace(0.0, 2048.0, 500)
                X_c_grid = np.zeros_like(y_raw_grid)

                _nb  = len(p.gamma_y)
                _n   = len(y_raw_grid)
                xt_g = X_c_grid / 2048.0
                yt_g = np.abs(y_raw_grid - p.y_readout_raw) / 2048.0
                B_g  = cte_basis(xt_g, yt_g, p.spatial_order)
                gamma_vec = getattr(p, gamma_attr)
                for mag_v, col, lbl in mag_bins:
                    MP_v = mag_poly_basis(np.full(_n, mag_v), p.mag_poly_order,
                                          p.mag_norm_ref, p.mag_norm_scale)
                    PsiB = (MP_v[:, :, None] * B_g[:, None, :]).reshape(_n, _nb + 1)[:, 1:]
                    dcte = dt_max * (PsiB @ gamma_vec)
                    ax.plot(y_raw_grid, dcte, color=col, lw=1.8, label=lbl)

                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
                ax.axvline(p.y_readout_raw, color='k', lw=0.8, ls=':', alpha=0.5,
                           label=f'readout (y={p.y_readout_raw:.0f})')
                ax.set_xlabel('Raw global y (px, py1pass frame)')
                ax.set_ylabel(f'{cte_lbl} (px)' if col_i == 0 else '')
                ax.set_title(f'_{chip} chip — {cte_lbl} amplitude (Δt={dt_max:.1f} yr, X_c=0)')
                if row_i == 0 or col_i == 0:
                    ax.legend(fontsize=9)

        fig.suptitle('CTE correction amplitude (converged parameters, raw detector frame)',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / f'{file_prefix}cte_amplitude.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: plots/{file_prefix}cte_amplitude.png")
    except Exception as exc:
        print(f"  WARNING: cte_amplitude.png failed — {exc}")

    # ── Helper: binned mean ───────────────────────────────────────────────────
    def _binned(x, y, n_bins=10):
        order = np.argsort(x)
        xo, yo = x[order], y[order]
        edges = np.array_split(np.arange(len(xo)), n_bins)
        xm, ym, ye = [], [], []
        for idx in edges:
            if len(idx) < 3:
                continue
            xm.append(xo[idx].mean())
            ym.append(yo[idx].mean())
            ye.append(yo[idx].std() / np.sqrt(len(idx)))
        return np.array(xm), np.array(ym), np.array(ye)

    def _slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            return np.nan, np.nan
        x, y = x[ok], y[ok]
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        b  = np.dot(dx, y - ym) / max(np.dot(dx, dx), 1e-30)
        r  = y - (ym + b * dx)
        var_b = r.var() / max(np.dot(dx, dx), 1e-30)
        return b, np.sqrt(max(var_b, 0))

    # ── Figure 2: dy_gdc vs y_raw before/after — CTE fingerprint ─────────────
    # Layout: 2 rows (hi/lo chip) × 2 cols (dy/dx), before (solid) and after
    # (dashed) overlaid on each axis.  Legend uses proxy artists so the same
    # 5-entry key (3 mag bins + 2 line styles) works for every subplot.
    if data_before is not None or data_after is not None:
        try:
            from matplotlib.lines import Line2D

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            all_mags = []
            for src in [data_before, data_after]:
                if src is not None:
                    for chip in ('hi', 'lo'):
                        m = src[chip]['mag']
                        if len(m):
                            all_mags.append(m)
            if all_mags:
                all_mags = np.concatenate(all_mags)
                pcts = np.nanpercentile(all_mags, [0, 33, 67, 100])
            else:
                pcts = [17, 19, 21, 23]

            bin_cols   = ['steelblue', 'darkorange', 'firebrick']
            bin_labels = [f'bright  {pcts[0]:.1f}–{pcts[1]:.1f}',
                          f'mid  {pcts[1]:.1f}–{pcts[2]:.1f}',
                          f'faint  {pcts[2]:.1f}–{pcts[3]:.1f}']

            # Proxy-artist legend: 3 mag-bin colors + 2 line styles (before/after)
            proxy_handles = (
                [Line2D([0], [0], color=c, lw=2.5, label=lbl)
                 for c, lbl in zip(bin_cols, bin_labels)]
                + [Line2D([0], [0], color='gray', lw=2, ls='-',  label='before CTE'),
                   Line2D([0], [0], color='gray', lw=2, ls='--', label='after CTE')]
            )

            for row_i, chip in enumerate(('hi', 'lo')):
                for col_i, comp in enumerate(('dy', 'dx')):
                    ax = axes[row_i, col_i]

                    for src, ls, alpha_fill in [
                            (data_before, '-',  0.18),
                            (data_after,  '--', 0.10)]:
                        if src is None:
                            continue
                        cd = src[chip]
                        if len(cd.get(comp, [])) == 0 or len(cd.get('y_raw', [])) == 0:
                            continue
                        for bi, col in enumerate(bin_cols):
                            m = cd['mag']
                            mask = (np.isfinite(m) & (m >= pcts[bi]) & (m < pcts[bi+1])
                                    & np.isfinite(cd[comp]) & np.isfinite(cd['y_raw']))
                            if mask.sum() < 10:
                                continue
                            xm, ym, ye = _binned(cd['y_raw'][mask], cd[comp][mask])
                            ax.plot(xm, ym, ls=ls, color=col, lw=1.8, zorder=4)
                            ax.fill_between(xm, ym-ye, ym+ye, color=col,
                                            alpha=alpha_fill, zorder=3)

                    p = cte_params[chip]
                    ax.axvline(p.y_readout_raw, color='k', lw=0.9, ls=':',
                               label=f'readout y={p.y_readout_raw:.0f}')
                    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
                    comp_lbl = (r'$\delta y_\mathrm{GDC}$ (px)' if comp == 'dy'
                                else r'$\delta x_\mathrm{GDC}$ (px)')
                    ax.set_xlabel('Raw chip-local y (px)', fontsize=10)
                    ax.set_ylabel(comp_lbl, fontsize=10)
                    ax.set_title(f'chip {chip} — {comp} residual vs y_raw', fontsize=10)
                    ax.legend(handles=proxy_handles, fontsize=9, loc='best',
                              framealpha=0.85)

            fig.suptitle('CTE fingerprint: GDC residual vs raw y by magnitude\n'
                         'solid = before CTE correction   dashed = after CTE correction',
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(plot_dir / f'{file_prefix}cte_before_after.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/{file_prefix}cte_before_after.png")
        except Exception as exc:
            print(f"  WARNING: cte_before_after.png failed — {exc}")

    # ── Figure 3: CTE slope vs magnitude before/after (raw y) ────────────────
    # Slope d(residual)/d(y_raw) vs magnitude — the key CTE fingerprint:
    # CTE signal gets steeper (larger |slope|) for fainter stars.
    if data_before is not None or data_after is not None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            N_MAG_BINS = 8

            all_mags = []
            for src in [data_before, data_after]:
                if src is None:
                    continue
                for chip in ('hi', 'lo'):
                    m = src[chip]['mag']
                    if len(m):
                        all_mags.append(m)
            if not all_mags:
                plt.close(fig)
                raise ValueError("no data for slope plot")
            all_mags   = np.concatenate(all_mags)
            mag_edges  = np.nanpercentile(all_mags, np.linspace(0, 100, N_MAG_BINS+1))
            mag_mids   = 0.5 * (mag_edges[:-1] + mag_edges[1:])

            # (chip, when) → (fmt, color, markerfacecolor, label)
            styles = [
                (('hi', 'before'), 'o-',  'steelblue',  'steelblue',  'hi before'),
                (('hi', 'after'),  'o--', 'steelblue',  'none',       'hi after'),
                (('lo', 'before'), 's-',  'darkorange', 'darkorange', 'lo before'),
                (('lo', 'after'),  's--', 'darkorange', 'none',       'lo after'),
            ]

            for col_i, comp in enumerate(('dy', 'dx')):
                ax = axes[col_i]
                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
                for (chip, when), fmt, color, mfc, label in styles:
                    src = data_before if when == 'before' else data_after
                    if src is None:
                        continue
                    cd = src[chip]
                    if len(cd.get(comp, [])) == 0 or len(cd.get('y_raw', [])) == 0:
                        continue
                    slopes, errs = [], []
                    for bi in range(N_MAG_BINS):
                        m = cd['mag']
                        mask = (np.isfinite(m) & (m >= mag_edges[bi])
                                & (m < mag_edges[bi+1])
                                & np.isfinite(cd[comp]) & np.isfinite(cd['y_raw']))
                        if mask.sum() < 8:
                            slopes.append(np.nan); errs.append(np.nan)
                            continue
                        sl, sl_e = _slope(cd['y_raw'][mask], cd[comp][mask])
                        slopes.append(sl * 1e3); errs.append(sl_e * 1e3)
                    slopes = np.array(slopes); errs = np.array(errs)
                    ok = np.isfinite(slopes)
                    if ok.any():
                        ax.errorbar(mag_mids[ok], slopes[ok], yerr=errs[ok],
                                    fmt=fmt, color=color, markerfacecolor=mfc,
                                    ms=6, capsize=3, lw=1.5, label=label)

                comp_lbl = (r'slope $d(\delta y_\mathrm{GDC})/d(y_\mathrm{raw})$ (mpx/px)'
                            if comp == 'dy'
                            else r'slope $d(\delta x_\mathrm{GDC})/d(y_\mathrm{raw})$ (mpx/px)')
                ax.set_xlabel('Instrumental magnitude', fontsize=10)
                ax.set_ylabel(comp_lbl, fontsize=10)
                ax.set_title(f'CTE slope vs magnitude ({comp})', fontsize=11)
                ax.legend(fontsize=9, framealpha=0.85)

            fig.suptitle(r'CTE fingerprint: slope $d(\delta\mathrm{GDC})/d(y_\mathrm{raw})$'
                         ' before vs after correction\n'
                         r'CTE signal: |slope| increases with magnitude, sign set by readout direction',
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(plot_dir / f'{file_prefix}cte_slope_vs_mag.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/{file_prefix}cte_slope_vs_mag.png")
        except Exception as exc:
            print(f"  WARNING: cte_slope_vs_mag.png failed — {exc}")

    # ── Figure 4: convergence ─────────────────────────────────────────────────
    try:
        conv_path = output_dir / 'cte_convergence.csv'
        if conv_path.exists():
            import pandas as _pd
            cdf = _pd.read_csv(conv_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            for chip, col in [('hi', 'steelblue'), ('lo', 'darkorange')]:
                sub = cdf[cdf['chip'] == chip]
                if len(sub) == 0:
                    continue
                gy_cols = [c for c in sub.columns if c.startswith('gamma_y')]
                gx_cols = [c for c in sub.columns if c.startswith('gamma_x')]
                gy_norm = np.sqrt((sub[gy_cols] ** 2).sum(axis=1))
                gx_norm = np.sqrt((sub[gx_cols] ** 2).sum(axis=1))
                axes[0].plot(sub['iter'], gy_norm, 'o-', color=col,
                             label=f'|γ_y|_{chip}', lw=1.8)
                axes[1].plot(sub['iter'], gx_norm, 's--', color=col,
                             label=f'|γ_x|_{chip}', lw=1.8)
            axes[0].set_xlabel('CTE outer iteration')
            axes[0].set_ylabel('|γ_y| norm')
            axes[0].set_title('CTE |γ_y| convergence')
            axes[0].legend(fontsize=9)
            axes[1].set_xlabel('CTE outer iteration')
            axes[1].set_ylabel('|γ_x| norm')
            axes[1].set_title('CTE |γ_x| convergence')
            axes[1].legend(fontsize=9)
            fig.suptitle('CTE parameter convergence', fontsize=11)
            fig.tight_layout()
            fig.savefig(plot_dir / 'cte_convergence.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/cte_convergence.png")
    except Exception as exc:
        print(f"  WARNING: cte_convergence.png failed — {exc}")

    # ── Figure 5: 2D detector map before/after ─────────────────────────────────
    for label, npz_data in [('before', data_before), ('after', data_after)]:
        if npz_data is None:
            continue
        try:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            clip = 0.05
            norm = TwoSlopeNorm(vcenter=0, vmin=-clip, vmax=clip)
            for col_i, chip in enumerate(('hi', 'lo')):
                ax   = axes[col_i]
                cd   = npz_data[chip]
                if len(cd['dy']) == 0:
                    continue
                ok   = np.isfinite(cd['dy'])
                sc   = ax.scatter(cd['X_c'][ok], cd['Y_c'][ok], c=cd['dy'][ok],
                                  cmap='RdBu_r', norm=norm, s=2, alpha=0.5,
                                  linewidths=0, rasterized=True)
                cb   = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(r'$\delta y_\mathrm{GDC}$ (px)', fontsize=9)
                ax.set_xlabel('X_c (px)')
                ax.set_ylabel('Y_c (px)')
                ax.set_title(f'_{chip} chip — {label} CTE correction')
            fig.suptitle(f'2D detector map of dy_gdc ({label} CTE)', fontsize=11)
            fig.tight_layout()
            out_path = plot_dir / f'{file_prefix}cte_2d_map_{label}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/{file_prefix}cte_2d_map_{label}.png")
        except Exception as exc:
            print(f"  WARNING: cte_2d_map_{label}.png failed — {exc}")

    # ── Figure 6: PM residuals of CTE fitting members vs raw detector position ──
    # "Before" = pmra_xmatch from master_combined_v2.csv (uncorrected v2 PM)
    # "After"  = pmra_bp3m  from stellar_astrometry.csv (CTE-corrected PM)
    # Membership is defined by which stars appear in the stellar astrometry CSV —
    # only stars that were actually used in the CTE fit are shown.
    try:
        import pandas as _pd
        from astropy.io import fits as _afits
        from matplotlib.lines import Line2D as _Line2D
        from matplotlib.colors import TwoSlopeNorm as _TSN
        from collections import defaultdict as _dd
        from scipy.spatial import cKDTree as _KDT

        field_dir  = output_dir.parent
        hst_root   = field_dir / 'HST' / 'mastDownload' / 'HST'
        master_csv = field_dir / 'hst_xmatch' / 'master_combined_v2.csv'
        if astrom_csv is None:
            astrom_csv = output_dir / 'stellar_astrometry.csv'
        if not master_csv.exists():
            raise FileNotFoundError(f'master_combined_v2.csv not found: {master_csv}')
        if not Path(astrom_csv).exists():
            print(f"  pm_vs_detector: {Path(astrom_csv).name} not found — skipping")
            raise RuntimeError("astrom_csv_missing")

        # ── Load stellar astrometry CSV (defines CTE fitting membership) ──────
        astrom = _pd.read_csv(astrom_csv)
        _has_bp3m = (astrom['pmra_bp3m'].notna() & astrom['pmdec_bp3m'].notna() &
                     astrom['sigma_pmra_bp3m'].notna())
        astrom = astrom[_has_bp3m].copy().reset_index(drop=True)
        if len(astrom) < 10:
            print(f"  pm_vs_detector: too few CTE members ({len(astrom)}) — skipping")
            raise RuntimeError("astrom_csv_missing")

        _astrom_gids  = astrom['Gaia_id'].to_numpy(np.int64)
        _pmra_a       = astrom['pmra_bp3m'].to_numpy(float)
        _pmdec_a      = astrom['pmdec_bp3m'].to_numpy(float)
        _sig_ra_a     = np.maximum(astrom['sigma_pmra_bp3m'].to_numpy(float),  0.001)
        _sig_dec_a    = np.maximum(astrom['sigma_pmdec_bp3m'].to_numpy(float), 0.001)
        _a_ra_astrom  = astrom['ra'].to_numpy(float) if 'ra' in astrom.columns \
                        else np.full(len(astrom), np.nan)
        _a_dec_astrom = astrom['dec'].to_numpy(float) if 'dec' in astrom.columns \
                        else np.full(len(astrom), np.nan)

        # ── Load master catalog ───────────────────────────────────────────────
        master = _pd.read_csv(master_csv, low_memory=False)

        # ── ACS/WFC filter: keep only rows observed in ACS images (j-prefix) ─
        _sub_cols = [c for c in master.columns if c.startswith('sub_names_')]
        _acs = _pd.Series(False, index=master.index)
        for _c in _sub_cols:
            _acs |= master[_c].fillna('').str.contains(r'\bj', regex=True)
        master = master[_acs].copy().reset_index(drop=True)

        # ── Keep only stars with valid xmatch PMs and uncertainties ──────────
        _ok = (master['pmra_xmatch'].notna() & master['pmdec_xmatch'].notna() &
               master['sigma_pmra_xmatch'].notna() & master['sigma_pmdec_xmatch'].notna() &
               master['corr_pmra_pmdec_xmatch'].notna())
        master = master[_ok].copy().reset_index(drop=True)

        # ── Match master to CTE fitting members (astrom_csv defines membership) ─
        _master_gids = master['gaia_source_id'].to_numpy(np.int64) \
            if 'gaia_source_id' in master.columns \
            else np.zeros(len(master), dtype=np.int64)
        _astrom_gid_to_row = {int(_astrom_gids[i]): i
                              for i in range(len(_astrom_gids))
                              if int(_astrom_gids[i]) > 0}
        _in_cte = np.zeros(len(master), dtype=bool)
        _astrom_row = np.full(len(master), -1, dtype=int)

        # Step 1: Gaia ID match
        for mi in range(len(master)):
            gid = int(_master_gids[mi])
            if gid > 0 and gid in _astrom_gid_to_row:
                _in_cte[mi]    = True
                _astrom_row[mi] = _astrom_gid_to_row[gid]

        # Step 2: positional cross-match for HST-only master rows
        _unmatched = ~_in_cte
        _a_ra_ok = np.isfinite(_a_ra_astrom) & np.isfinite(_a_dec_astrom)
        if _unmatched.any() and _a_ra_ok.any():
            _tree = _KDT(np.column_stack([_a_ra_astrom[_a_ra_ok],
                                          _a_dec_astrom[_a_ra_ok]]))
            _m_ra  = master['ra_xmatch'].to_numpy(float) \
                if 'ra_xmatch' in master.columns else np.full(len(master), np.nan)
            _m_dec = master['dec_xmatch'].to_numpy(float) \
                if 'dec_xmatch' in master.columns else np.full(len(master), np.nan)
            _q_ok = _unmatched & np.isfinite(_m_ra) & np.isfinite(_m_dec)
            if _q_ok.any():
                _dists, _near = _tree.query(
                    np.column_stack([_m_ra[_q_ok], _m_dec[_q_ok]]))
                _a_ok_idx = np.where(_a_ra_ok)[0]
                _tol = 0.5 / 3600.0
                for mi, dist, ni in zip(np.where(_q_ok)[0], _dists, _near):
                    if dist < _tol:
                        ai = int(_a_ok_idx[ni])
                        _in_cte[mi]    = True
                        _astrom_row[mi] = ai

        # ── Restrict master to CTE fitting members only ───────────────────────
        master      = master[_in_cte].copy().reset_index(drop=True)
        _astrom_row = _astrom_row[_in_cte]
        print(f"  Figure 6: {len(master):,} CTE fitting members (from stellar_astrometry)")

        # ── Extract before/after PM data ──────────────────────────────────────
        pmra_b  = master['pmra_xmatch'].to_numpy(float)
        pmdec_b = master['pmdec_xmatch'].to_numpy(float)
        s_ra    = master['sigma_pmra_xmatch'].to_numpy(float)
        s_dec   = master['sigma_pmdec_xmatch'].to_numpy(float)
        rho     = master['corr_pmra_pmdec_xmatch'].to_numpy(float)
        cov_det = s_ra**2 * s_dec**2 * (1.0 - rho**2)

        # All CTE members have "after" data (they're all in astrom_csv)
        has_after = np.ones(len(master), dtype=bool)
        pmra_a_m  = _pmra_a[_astrom_row]
        pmdec_a_m = _pmdec_a[_astrom_row]
        w_ra_a    = 1.0 / _sig_ra_a[_astrom_row]**2
        w_dec_a   = 1.0 / _sig_dec_a[_astrom_row]**2

        # ── Precision-weighted field mean for "before" ────────────────────────
        _W00  = s_dec**2 / np.where(cov_det > 0, cov_det, 1.0)
        _W11  = s_ra**2  / np.where(cov_det > 0, cov_det, 1.0)
        _W01  = -rho * s_ra * s_dec / np.where(cov_det > 0, cov_det, 1.0)
        _ok_w = cov_det > 0
        if _ok_w.sum() >= 5:
            _S00 = _W00[_ok_w].sum(); _S11 = _W11[_ok_w].sum(); _S01 = _W01[_ok_w].sum()
            _detS = _S00*_S11 - _S01**2
            if abs(_detS) > 0:
                _rhs0 = (_W00[_ok_w]*pmra_b[_ok_w] + _W01[_ok_w]*pmdec_b[_ok_w]).sum()
                _rhs1 = (_W01[_ok_w]*pmra_b[_ok_w] + _W11[_ok_w]*pmdec_b[_ok_w]).sum()
                mean_pmra_b  = float((_S11*_rhs0 - _S01*_rhs1) / _detS)
                mean_pmdec_b = float((-_S01*_rhs0 + _S00*_rhs1) / _detS)
            else:
                mean_pmra_b  = float(np.nanmean(pmra_b))
                mean_pmdec_b = float(np.nanmean(pmdec_b))
        else:
            mean_pmra_b  = float(np.nanmean(pmra_b))
            mean_pmdec_b = float(np.nanmean(pmdec_b))
        res_ra_b  = pmra_b  - mean_pmra_b
        res_dec_b = pmdec_b - mean_pmdec_b
        w_ra_b  = _W11
        w_dec_b = _W00
        print(f"  Before field mean: μ_α*={mean_pmra_b:+.4f}  μ_δ={mean_pmdec_b:+.4f} mas/yr")

        # ── Precision-weighted field mean for "after" ─────────────────────────
        _wa = w_ra_a
        mean_pmra_a  = float(np.sum(_wa * pmra_a_m)  / np.sum(_wa)) if np.sum(_wa) > 0 \
                       else mean_pmra_b
        mean_pmdec_a = float(np.sum(w_dec_a * pmdec_a_m) / np.sum(w_dec_a)) \
                       if np.sum(w_dec_a) > 0 else mean_pmdec_b
        res_ra_a  = pmra_a_m  - mean_pmra_a
        res_dec_a = pmdec_a_m - mean_pmdec_a
        print(f"  After  field mean: μ_α*={mean_pmra_a:+.4f}  "
              f"μ_δ={mean_pmdec_a:+.4f} mas/yr")

        # ── Parse hst_indices to find best image ──────────────────────────────
        _hst_idx_cols = [c for c in master.columns if c.startswith('hst_indices_')]
        img_member_count = _dd(int)
        star_img_idx = [{} for _ in range(len(master))]
        for _col in _hst_idx_cols:
            for _mi, _val in enumerate(master[_col]):
                if _pd.isna(_val):
                    continue
                for _entry in str(_val).split(','):
                    _entry = _entry.strip()
                    if ':' not in _entry:
                        continue
                    _img, _idx = _entry.rsplit(':', 1)
                    _img = _img.strip()
                    img_member_count[_img] += 1
                    if _img not in star_img_idx[_mi]:
                        star_img_idx[_mi][_img] = int(_idx.strip())

        if not img_member_count:
            raise ValueError('no hst_indices found for member stars')

        rootname_count = _dd(int)
        for _img_name, _cnt in img_member_count.items():
            _root = _img_name.replace('_hi', '').replace('_lo', '')
            rootname_count[_root] += _cnt
        best_root = max(rootname_count, key=rootname_count.get)
        best_variants = [v for v in [best_root+'_hi', best_root+'_lo']
                         if v in img_member_count]
        print(f"  Best image: {best_root}  ({rootname_count[best_root]:,} member detections)")

        # ── Load catalog.fits ─────────────────────────────────────────────────
        cat_path = hst_root / best_root / f'{best_root}_flc_catalog.fits'
        if not cat_path.exists():
            raise FileNotFoundError(f'catalog.fits not found: {cat_path}')
        with _afits.open(str(cat_path)) as hdul:
            _cat   = hdul[1].data
            cat_x  = _cat['x'].copy().astype(float)
            cat_y  = _cat['y'].copy().astype(float)
            cat_chip = _cat['chip_ext'].copy()
            cat_mag  = _cat['mag_st'].copy().astype(float)

        # ── Collect member stars found in best_root ────────────────────────
        rows_with_pos, cat_indices = [], []
        seen_rows: set = set()
        for mi in range(len(master)):
            for variant in best_variants:
                if variant in star_img_idx[mi] and mi not in seen_rows:
                    seen_rows.add(mi)
                    rows_with_pos.append(mi)
                    cat_indices.append(star_img_idx[mi][variant])
                    break
        rows_with_pos = np.array(rows_with_pos, dtype=int)
        cat_indices   = np.array(cat_indices,   dtype=int)
        print(f"  Stars with detector positions: {len(rows_with_pos):,}")

        x_raw  = cat_x[cat_indices]
        y_raw  = cat_y[cat_indices]
        chip   = cat_chip[cat_indices]
        mag    = cat_mag[cat_indices]

        # Per-star arrays aligned to rows_with_pos
        res_ra_b_s   = res_ra_b[rows_with_pos]
        res_dec_b_s  = res_dec_b[rows_with_pos]
        w_ra_b_s     = w_ra_b[rows_with_pos]
        w_dec_b_s    = w_dec_b[rows_with_pos]
        res_ra_a_s   = res_ra_a[rows_with_pos]
        res_dec_a_s  = res_dec_a[rows_with_pos]
        w_ra_a_s     = w_ra_a[rows_with_pos]
        w_dec_a_s    = w_dec_a[rows_with_pos]
        has_after_s  = has_after[rows_with_pos]

        # ── Helpers (same as cte_diagnostic_leo_i.py) ────────────────────────
        def _wmean_binned(x, y, w, n_bins=14):
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
            if ok.sum() < 3:
                return np.array([]), np.array([]), np.array([])
            x, y, w = x[ok], y[ok], w[ok]
            order = np.argsort(x)
            x, y, w = x[order], y[order], w[order]
            xm, ym, ye = [], [], []
            for _idx in np.array_split(np.arange(len(x)), n_bins):
                if len(_idx) < 3:
                    continue
                wi = w[_idx]; wt = wi.sum()
                if wt <= 0:
                    continue
                xm.append(x[_idx].mean())
                ym.append(float(np.sum(wi * y[_idx]) / wt))
                ye.append(float(np.sqrt(1.0 / wt)))
            return np.array(xm), np.array(ym), np.array(ye)

        def _wls_slope(x, y, w):
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
            if ok.sum() < 5:
                return np.nan, np.nan
            x, y, w = x[ok], y[ok], w[ok]
            xm = np.sum(w*x)/np.sum(w);  ym = np.sum(w*y)/np.sum(w)
            dx = x - xm
            denom = np.sum(w * dx**2)
            if denom <= 0:
                return np.nan, np.nan
            b = np.sum(w * dx * (y - ym)) / denom
            resid = y - (ym + b*dx)
            var_b = max(np.sum(w*resid**2) / max(len(x)-2, 1) / denom, 0.0)
            return float(b), float(np.sqrt(var_b))

        # ── Figure layout ─────────────────────────────────────────────────────
        chip_vals   = np.unique(chip)
        chip_masks  = [chip == v for v in chip_vals]
        chip_titles = [f'chip_ext={v}' for v in chip_vals]
        n_chips     = len(chip_vals)

        mag_pcts   = np.nanpercentile(mag, [0, 33, 67, 100])
        bin_colors = ['steelblue', 'darkorange', 'firebrick']
        bin_labels = [
            f'bright  ({mag_pcts[0]:.1f}–{mag_pcts[1]:.1f})',
            f'mid     ({mag_pcts[1]:.1f}–{mag_pcts[2]:.1f})',
            f'faint   ({mag_pcts[2]:.1f}–{mag_pcts[3]:.1f})',
        ]

        fig = plt.figure(figsize=(14, 27))
        gs  = fig.add_gridspec(5, 2, hspace=0.40, wspace=0.32,
                               left=0.08, right=0.96, top=0.95, bottom=0.04)
        clip_pm = 0.35  # mas/yr

        # Proxy artist legend for rows 2–3
        proxy_handles = (
            [_Line2D([0], [0], color=c, lw=2.5, label=lbl)
             for c, lbl in zip(bin_colors, bin_labels)]
            + [_Line2D([0], [0], color='gray', lw=2, ls='-',  label=f'before CTE  N={len(rows_with_pos):,}'),
               _Line2D([0], [0], color='gray', lw=2, ls='--', label=f'after CTE   N={has_after_s.sum():,}')]
        )

        # ── Row 0: 2D detector map coloured by Δpmra, before (left) and after (right) ─
        for col_i, (lbl, res_ra_s, n_lbl) in enumerate([
                ('before CTE', res_ra_b_s, len(rows_with_pos)),
                ('after CTE',  np.where(has_after_s, res_ra_a_s, np.nan),
                 int(has_after_s.sum())),
        ]):
            ax = fig.add_subplot(gs[0, col_i])
            norm = _TSN(vcenter=0, vmin=-clip_pm, vmax=clip_pm)
            for cmask, marker, clbl in zip(chip_masks, ['o', 's'], chip_titles):
                sc = ax.scatter(x_raw[cmask], y_raw[cmask], c=res_ra_s[cmask],
                                cmap='RdBu_r', norm=norm, s=4, alpha=0.6,
                                linewidths=0, rasterized=True, marker=marker, label=clbl)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r'$\Delta\mu_{\alpha*}$ (mas yr$^{-1}$)', fontsize=9)
            ax.set_xlabel('Raw detector X (px)', fontsize=11)
            ax.set_ylabel('Raw detector Y (px)', fontsize=11)
            ax.set_title(f'{lbl}  —  Δpmra  (N={n_lbl:,})', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)

        # ── Row 1: 2D detector map coloured by Δpmdec, before (left) and after (right) ─
        for col_i, (lbl, res_dec_s, n_lbl) in enumerate([
                ('before CTE', res_dec_b_s, len(rows_with_pos)),
                ('after CTE',  np.where(has_after_s, res_dec_a_s, np.nan),
                 int(has_after_s.sum())),
        ]):
            ax = fig.add_subplot(gs[1, col_i])
            norm = _TSN(vcenter=0, vmin=-clip_pm, vmax=clip_pm)
            for cmask, marker, clbl in zip(chip_masks, ['o', 's'], chip_titles):
                sc = ax.scatter(x_raw[cmask], y_raw[cmask], c=res_dec_s[cmask],
                                cmap='RdBu_r', norm=norm, s=4, alpha=0.6,
                                linewidths=0, rasterized=True, marker=marker, label=clbl)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r'$\Delta\mu_\delta$ (mas yr$^{-1}$)', fontsize=9)
            ax.set_xlabel('Raw detector X (px)', fontsize=11)
            ax.set_ylabel('Raw detector Y (px)', fontsize=11)
            ax.set_title(f'{lbl}  —  {best_root}  (N={n_lbl:,})', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)

        # ── Rows 1–n_chips: PM residuals vs y_raw per chip ───────────────────
        def _plot_pm_vs_y(ax, cmask, res_b, w_b, res_a, w_a, has_a, lbl, chip_lbl):
            slopes_b = []
            for bi, (color, blbl) in enumerate(zip(bin_colors, bin_labels)):
                m_b = (np.isfinite(mag) & (mag >= mag_pcts[bi])
                       & (mag < mag_pcts[bi+1]) & cmask)
                m_a = m_b & has_a

                for res, w, m, ls in [(res_b, w_b, m_b, '-'), (res_a, w_a, m_a, '--')]:
                    if m.sum() < 5:
                        continue
                    ax.scatter(y_raw[m], res[m], s=2, alpha=0.15,
                               color=color, linewidths=0, rasterized=True, zorder=2)
                    xm, ym, ye = _wmean_binned(y_raw[m], res[m], w[m])
                    if len(xm):
                        ax.plot(xm, ym, ls=ls, color=color, lw=2.0, zorder=4)
                        ax.fill_between(xm, ym-ye, ym+ye, color=color,
                                        alpha=0.20, zorder=3)
                        sl, sl_e = _wls_slope(y_raw[m], res[m], w[m])
                        if np.isfinite(sl):
                            yfit = np.linspace(y_raw[m].min(), y_raw[m].max(), 200)
                            ymn = np.average(res[m], weights=w[m])
                            xmn = np.average(y_raw[m], weights=w[m])
                            ax.plot(yfit, ymn + sl*(yfit-xmn),
                                    ls=ls, color=color, lw=1.2, zorder=5,
                                    label=f'  slope={sl*1e3:+.2f} μas/yr/px' if ls=='-' else None)
                    if ls == '-' and m.sum() >= 5:
                        sl, _ = _wls_slope(y_raw[m], res[m], w[m])
                        if np.isfinite(sl):
                            slopes_b.append(sl)

            ax.axhline(0, color='k', lw=0.8, alpha=0.5)
            ax.set_xlabel('Raw detector Y (px)', fontsize=11)
            ax.set_ylabel(lbl, fontsize=11)
            ax.set_title(f'{chip_lbl} — {lbl} vs raw Y', fontsize=11)
            ax.set_ylim(-clip_pm*1.5, clip_pm*1.5)
            ax.legend(handles=proxy_handles, fontsize=7, loc='upper left', ncol=2)

            # Annotate readout direction
            if slopes_b:
                mean_sl = float(np.mean(slopes_b))
                ylo, yhi = y_raw[cmask].min(), y_raw[cmask].max()
                side, ha = (ylo, 'left') if mean_sl < 0 else (yhi, 'right')
                ax.annotate('readout', xy=(side, ax.get_ylim()[0]*0.85),
                            fontsize=7, color='0.4', style='italic', ha=ha)

        for chip_row, (cmask, chip_lbl) in enumerate(zip(chip_masks, chip_titles)):
            row = 2 + chip_row
            for col_i, (res_b, w_b, res_a, w_a, pm_lbl) in enumerate([
                    (res_ra_b_s,  w_ra_b_s,  res_ra_a_s,  w_ra_a_s,
                     r'$\Delta\mu_{\alpha*}$ (mas yr$^{-1}$)'),
                    (res_dec_b_s, w_dec_b_s, res_dec_a_s, w_dec_a_s,
                     r'$\Delta\mu_\delta$ (mas yr$^{-1}$)'),
            ]):
                ax = fig.add_subplot(gs[row, col_i])
                _plot_pm_vs_y(ax, cmask, res_b, w_b, res_a, w_a, has_after_s, pm_lbl, chip_lbl)

        # ── Row 3: slope vs magnitude, before and after, both chips ──────────
        N_MAG_BINS = 8
        mag_edges  = np.nanpercentile(mag, np.linspace(0, 100, N_MAG_BINS+1))
        mag_mids   = 0.5 * (mag_edges[:-1] + mag_edges[1:])

        chip_styles = [
            dict(fmt='o-',  fmt_a='o--', color='steelblue',  mfc_b='steelblue',  mfc_a='none', label=chip_titles[0]),
            dict(fmt='s-',  fmt_a='s--', color='darkorange', mfc_b='darkorange', mfc_a='none',
                 label=chip_titles[1] if len(chip_titles) > 1 else ''),
        ]

        for col_i, (res_b, w_b, res_a, w_a, pm_lbl) in enumerate([
                (res_ra_b_s,  w_ra_b_s,  res_ra_a_s,  w_ra_a_s,
                 r'slope $\Delta\mu_{\alpha*}$ (μas yr$^{-1}$ px$^{-1}$)'),
                (res_dec_b_s, w_dec_b_s, res_dec_a_s, w_dec_a_s,
                 r'slope $\Delta\mu_\delta$ (μas yr$^{-1}$ px$^{-1}$)'),
        ]):
            ax = fig.add_subplot(gs[4, col_i])
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)

            for cmask, style in zip(chip_masks, chip_styles):
                if cmask.sum() < 5:
                    continue
                for (res, w, use_mask, fmt, mfc, lbl_sfx) in [
                        (res_b, w_b, np.ones(len(chip), dtype=bool),
                         style['fmt'],   style['mfc_b'], ' before'),
                        (res_a, w_a, has_after_s,
                         style['fmt_a'], style['mfc_a'], ' after'),
                ]:
                    slopes, errs, ns = [], [], []
                    for k in range(N_MAG_BINS):
                        m = (np.isfinite(mag) & (mag >= mag_edges[k])
                             & (mag < mag_edges[k+1]) & cmask & use_mask)
                        if m.sum() < 10:
                            slopes.append(np.nan); errs.append(np.nan); ns.append(0)
                            continue
                        sl, sl_e = _wls_slope(y_raw[m], res[m], w[m])
                        slopes.append(sl * 1e3); errs.append(sl_e * 1e3); ns.append(m.sum())
                    slopes = np.array(slopes); errs = np.array(errs)
                    ok_sl  = np.isfinite(slopes) & (np.array(ns) >= 10)
                    if ok_sl.any():
                        ax.errorbar(mag_mids[ok_sl], slopes[ok_sl], yerr=errs[ok_sl],
                                    fmt=fmt, color=style['color'], markerfacecolor=mfc,
                                    ms=6, capsize=4, lw=1.5, zorder=4,
                                    label=style['label'] + lbl_sfx)
                    for k in np.where(ok_sl)[0]:
                        ax.text(mag_mids[k], slopes[k] + errs[k] + 0.005,
                                str(ns[k]), ha='center', va='bottom',
                                fontsize=6, color='0.5')

            ax.set_xlabel('HST ST magnitude', fontsize=11)
            ax.set_ylabel(pm_lbl, fontsize=11)
            ax.set_title('CTE fingerprint: gradient slope vs magnitude (per chip)', fontsize=11)
            ax.legend(fontsize=9)
            ax.text(0.97, 0.05,
                    'CTE: |slope| grows with magnitude\n'
                    'Sign set by readout direction\n'
                    '(opposite between chips — expected)',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color='0.45', style='italic')

        fig.suptitle(
            f'Leo I — CTE diagnostic  ({best_root},  N={len(rows_with_pos):,} CTE members)',
            fontsize=13, y=0.97)
        fig.savefig(plot_dir / f'{file_prefix}cte_pm_vs_detector.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: plots/{file_prefix}cte_pm_vs_detector.png")
    except RuntimeError as exc:
        if "astrom_csv_missing" not in str(exc):
            import traceback; traceback.print_exc()
            print(f"  WARNING: cte_pm_vs_detector.png failed — {exc}")
    except Exception as exc:
        import traceback
        print(f"  WARNING: cte_pm_vs_detector.png failed — {exc}")
        traceback.print_exc()


# ── Joint CTE outer loop ──────────────────────────────────────────────────────

def _run_joint_cte_loop(
    solver,
    image_names: list[str],
    cte_params: dict,
    t_launch_yr: float,
    filtered_spi: dict | None,
    hst_only_mask: np.ndarray,
    sigma_pm: float,
    plx_pop: float,
    sigma_plx_tot: float,
    mu_pop_prior: np.ndarray,
    C_pop_prior_inv: np.ndarray,
    n_iter: int = 20,
    member_sigma_clip: float = 3.0,
    regularize_gamma: float = 1e-8,
    pm_sys_floor: float = 0.2,
    gaia_catalog=None,
    init_pm_window: float = 2.0,
    member_sidx_init: np.ndarray | None = None,
    mu_pop_init: np.ndarray | None = None,
    fit_cte_x: bool = True,
) -> tuple:
    """
    Outer Gauss-Newton loop for the joint (r, γ_CTE, μ_pop) solve.

    Replaces the alternating solver.fit() + update_cte_params() loop with a
    single coherent joint optimisation.  Each iteration calls _joint_solve_cte
    once, updates r_current / mu_pop_current, then re-selects member stars from
    the updated posterior PMs.

    Parameters
    ----------
    solver          : BP3MSolver with _img_data, C_survey_inv, etc.
    image_names     : ordered list of image keys
    cte_params      : initial CTEChipParams dict (gamma values ignored —
                      they are solved for from scratch each iteration)
    t_launch_yr     : ACS launch year (for CTE dt calculation)
    filtered_spi    : per-image SPI DataFrames (for Y_orig lookup); may be None
    hst_only_mask   : (n_stars,) True for HST-only stars (excluded from members)
    sigma_pm        : LVD intrinsic PM dispersion (mas/yr)
    plx_pop         : LVD mean parallax (mas)
    sigma_plx_tot   : LVD total parallax uncertainty (mas)
    mu_pop_prior    : (2,) empirical prior mean for population PM (mas/yr)
    C_pop_prior_inv : (2,2) prior precision on mu_pop (mas/yr)^{-2}
    n_iter          : maximum Gauss-Newton iterations
    member_sigma_clip : sigma threshold for membership selection
    regularize_gamma : diagonal regularisation on H_gamma

    Returns
    -------
    r_hat, C_r, gamma_hat, mu_pop_hat, C_shared, a_arr, K_img, C_vT, cte_params
    """
    n_images  = len(image_names)
    n_r_total = solver.N_R * n_images

    # r_current from the most recent _update_R call (set by caller before entry)
    r_current = solver._r_hat_current.copy()

    # mu_pop_init overrides mu_pop_prior as the starting point (warmstart provides it)
    mu_pop_current = mu_pop_init.copy() if mu_pop_init is not None else mu_pop_prior.copy()

    # Count contributing HST detections per star for member eligibility
    n_stars = solver.C_survey_inv.shape[0]
    n_hst   = np.zeros(n_stars, dtype=int)
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        sidx    = d['sidx']
        use_any = d.get('use_for_astrom', d['use_for_fit'])
        np.add.at(n_hst, sidx[use_any], 1)

    # Split initial member set into:
    #   mu_member_sidx  — Gaia-only: drive μ_pop, re-selected each iteration
    #                     (can only lose stars, never gain new ones)
    #   hst_prior_sidx  — HST-only: fixed, receive population prior for PM
    #                     regularisation but do NOT contribute to μ_pop Schur
    if member_sidx_init is not None:
        _hst_init_mask  = hst_only_mask[member_sidx_init]
        hst_prior_sidx  = member_sidx_init[_hst_init_mask]    # fixed throughout
        mu_member_sidx  = member_sidx_init[~_hst_init_mask]   # Gaia, re-selected
    else:
        hst_prior_sidx = np.array([], dtype=int)
        mu_member_sidx = np.where(~hst_only_mask)[0]
    # Keep the initial Gaia set as the ceiling — no new stars can join
    _mu_member_sidx_init = mu_member_sidx.copy()
    print(f"  Initial members: {len(mu_member_sidx)} Gaia (drives μ_pop) + "
          f"{len(hst_prior_sidx)} HST-only (fixed prior)")

    # Output variables (initialised to sensible defaults)
    r_hat      = r_current.copy()
    C_r        = None
    _p0  = cte_params.get('hi', cte_params.get('lo'))
    _nb0 = _cte_n_spatial(_p0.spatial_order) * (_p0.mag_poly_order + 1) - 1
    # Initialise gamma from warmstart cte_params (not zeros) and use as prior
    # to prevent the r–γ degeneracy from causing oscillatory blow-up.
    _p = cte_params.get('hi', cte_params.get('lo', None))
    if _p is not None and len(getattr(_p, 'gamma_x', [])) == _nb0:
        gamma_hat = np.concatenate([
            cte_params['hi'].gamma_x if 'hi' in cte_params else np.zeros(_nb0),
            cte_params['hi'].gamma_y if 'hi' in cte_params else np.zeros(_nb0),
            cte_params['lo'].gamma_x if 'lo' in cte_params else np.zeros(_nb0),
            cte_params['lo'].gamma_y if 'lo' in cte_params else np.zeros(_nb0),
        ])
    else:
        gamma_hat = np.zeros(4 * _nb0)
    gamma_prior = gamma_hat.copy()   # warmstart gamma as regularisation anchor
    mu_pop_hat = mu_pop_prior.copy()
    C_shared   = None
    a_arr      = None
    K_img      = {}
    C_vT       = None

    solver._update_R(r_current)
    solver._update_geometry(r_current, solver.v_survey)

    gamma_history   = [gamma_hat.copy()]       # index 0 = warmstart
    mu_pop_history  = [mu_pop_current.copy()]
    C_mu_history    = [np.zeros((2, 2))]       # warmstart has no posterior C_shared yet

    import time as _jtime
    for it in range(n_iter):
        r_prev      = r_current.copy()
        gamma_prev  = gamma_hat.copy()
        mu_pop_prev = mu_pop_current.copy()

        _n_total = len(mu_member_sidx) + len(hst_prior_sidx)
        print(f"\n  ── Joint iter {it+1}/{n_iter}  "
              f"n_gaia={len(mu_member_sidx)}  n_hst_prior={len(hst_prior_sidx)}  "
              f"μ=({mu_pop_current[0]:+.4f},{mu_pop_current[1]:+.4f}) ──")
        _t_iter = _jtime.time()
        result = _joint_solve_cte(
            solver, image_names, cte_params, t_launch_yr, filtered_spi,
            mu_member_sidx, sigma_pm, plx_pop, sigma_plx_tot,
            mu_pop_current, mu_pop_prior, C_pop_prior_inv, r_current,
            regularize_gamma=regularize_gamma,
            gamma_prior=gamma_prior,
            hst_prior_sidx=hst_prior_sidx,
            fit_cte_x=fit_cte_x,
        )
        r_hat, C_r, gamma_hat, mu_pop_hat, C_shared, a_arr, K_img, C_vT = result

        # Update state
        cte_params     = _gamma_to_cte_params(gamma_hat, cte_params)
        mu_pop_current = mu_pop_hat
        r_current      = r_hat
        gamma_history.append(gamma_hat.copy())
        mu_pop_history.append(mu_pop_hat.copy())
        C_mu_history.append(C_shared[-2:, -2:].copy() if C_shared is not None else np.zeros((2, 2)))

        # Update solver geometry for next iteration
        solver._update_R(r_hat)
        solver._update_geometry(r_hat, solver.v_survey)

        # Apply current CTE model.  Alpha (per-image uncertainty inflation) is
        # intentionally NOT updated here: keeping Cs constant across Newton
        # iterations makes the effective problem linear → converges in 1-2 steps.
        # Alpha is updated once after the loop with the final converged solution.
        apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr,
                            filtered_spi=filtered_spi, subtract=True)

        # Refresh n_hst counts (use_for_astrom flags may have changed)
        n_hst[:] = 0
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            sidx    = d['sidx']
            use_any = d.get('use_for_astrom', d['use_for_fit'])
            np.add.at(n_hst, sidx[use_any], 1)

        # Re-select Gaia members from posterior PMs.
        # Restricted to the initial Gaia set: can only drop members, never add.
        print(f"  Refining Gaia member selection...")
        mu_member_sidx = _select_members_from_a(
            a_arr, mu_pop_current, hst_only_mask, n_hst,
            sigma_clip=member_sigma_clip,
            pm_sys_floor=pm_sys_floor,
            eligible_sidx=_mu_member_sidx_init)

        # Convergence diagnostics
        dr   = float(np.max(np.abs(r_hat - r_prev)))
        dg   = float(np.max(np.abs(gamma_hat - gamma_prev)))
        dmu  = float(np.max(np.abs(mu_pop_hat - mu_pop_prev)))
        _pc  = cte_params.get('hi', cte_params.get('lo'))
        _nb  = _cte_n_spatial(_pc.spatial_order) * (_pc.mag_poly_order + 1) - 1
        gy_hi = float(np.linalg.norm(gamma_hat[_nb:2*_nb]))
        gy_lo = float(np.linalg.norm(gamma_hat[3*_nb:4*_nb]))
        gx_hi = float(np.linalg.norm(gamma_hat[:_nb]))
        gx_lo = float(np.linalg.norm(gamma_hat[2*_nb:3*_nb]))
        _C_mu_it   = C_shared[-2:, -2:]
        _sig_ra_it = float(np.sqrt(_C_mu_it[0, 0]))
        _sig_de_it = float(np.sqrt(_C_mu_it[1, 1]))
        _rho_it    = float(_C_mu_it[0, 1] / (_sig_ra_it * _sig_de_it + 1e-30))
        print(f"  → Δr={dr:.3e}  Δγ={dg:.3e}  Δμ={dmu:.4f}  "
              f"({_jtime.time()-_t_iter:.1f}s)")
        print(f"  → μ_pop=({mu_pop_hat[0]:+.4f}±{_sig_ra_it:.4f}, "
              f"{mu_pop_hat[1]:+.4f}±{_sig_de_it:.4f}) mas/yr  ρ={_rho_it:+.3f}  "
              f"n_gaia_new={len(mu_member_sidx)}  (HST fixed: {len(hst_prior_sidx)})")
        print(f"  → |γ_y| hi={gy_hi:.3e} lo={gy_lo:.3e}  "
              f"|γ_x| hi={gx_hi:.3e} lo={gx_lo:.3e}")

        if it >= 2 and dr < 1e-6 and dg < 1e-8 and dmu < 1e-4:
            print(f"  Converged at iteration {it + 1}")
            break

    # Single post-convergence alpha update with the final CTE+transform solution.
    _update_image_alpha(solver, image_names, r_hat, a_arr)

    return r_hat, C_r, gamma_hat, mu_pop_hat, C_shared, a_arr, K_img, C_vT, cte_params, gamma_history, mu_pop_history, C_mu_history


def _plot_joint_convergence(
    gamma_history: list,
    mu_pop_history: list,
    cte_template: dict,
    output_dir: Path,
    file_prefix: str = '',
    C_mu_history: list | None = None,
) -> None:
    """
    Per-iteration evolution of ALL CTE parameters and μ_pop.

    Layout (rows × 2 cols):
      Row 0: μ_pop value (with ±1σ if C_mu_history given) | convergence delta norms
      Row 1: γy_hi all nb coefficients | γy_lo all nb coefficients
      Row 2 (if x-CTE active): γx_hi | γx_lo

    Index 0 = warmstart; subsequent indices = joint iterations.
    """
    import warnings; warnings.filterwarnings('ignore')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    _ref    = cte_template['hi']
    _nb     = _cte_n_spatial(_ref.spatial_order) * (_ref.mag_poly_order + 1) - 1
    n_iter  = len(gamma_history) - 1
    iters   = np.arange(len(gamma_history))
    xlbls   = ['ws'] + [str(i+1) for i in range(n_iter)]

    gamma_arr = np.array(gamma_history)   # (n_iter+1, 4*nb)
    mu_arr    = np.array(mu_pop_history)  # (n_iter+1, 2)

    _has_x = np.any(gamma_arr[:, 0:_nb] != 0) or np.any(gamma_arr[:, 2*_nb:3*_nb] != 0)
    n_rows  = 3 if _has_x else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows), squeeze=False)

    # ── Build coefficient labels for the nb-long block ────────────────────────
    sp_lbls = _cte_basis_labels(_ref.spatial_order)   # n_spatial labels, index 0 = degenerate yt
    coef_labels = []
    # mag_order=0: skip first spatial term (degenerate)
    coef_labels += [f'1·{s}' for s in sp_lbls[1:]]
    # mag_order=1..order: all spatial terms
    for k in range(1, _ref.mag_poly_order + 1):
        mp = f'm^{k}' if k > 1 else 'm'
        coef_labels += [f'{mp}·{s}' for s in sp_lbls]

    # Build colormap: cycle through tab20 for nb coefficients
    cmap    = cm.get_cmap('tab20', max(_nb, 1))
    colors  = [cmap(i % 20) for i in range(_nb)]
    # Linestyles by magnitude order (first n_spatial-1 terms are m^0)
    n_sp    = _cte_n_spatial(_ref.spatial_order)
    lstyles = []
    for i in range(_nb):
        if i < n_sp - 1:               # mag_order=0
            lstyles.append('solid')
        elif i < 2 * n_sp - 1:         # mag_order=1
            lstyles.append('dashed')
        elif i < 3 * n_sp - 1:         # mag_order=2
            lstyles.append('dotted')
        else:                           # mag_order=3+
            lstyles.append((0, (3, 1, 1, 1)))

    def _plot_gamma_block(ax, block, title):
        for k in range(_nb):
            ax.plot(iters, block[:, k], color=colors[k], ls=lstyles[k],
                    lw=1.2, alpha=0.85, label=coef_labels[k] if _nb <= 12 else None)
        ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.4)
        ax.axhline(0, color='k', lw=0.7, ls=':', alpha=0.35)
        ax.set_xticks(iters); ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
        if _nb <= 12:
            ax.legend(fontsize=6, ncol=2, loc='best')

    # ── Row 0 left: μ_pop ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(iters, mu_arr[:, 0], 'o-', color='steelblue', lw=1.5, label=r'$\mu_{\alpha*}$')
    ax.plot(iters, mu_arr[:, 1], 's-', color='firebrick',  lw=1.5, label=r'$\mu_\delta$')
    if C_mu_history is not None:
        C_mu_arr = np.array(C_mu_history)   # (n_iter+1, 2, 2)
        sig_ra  = np.sqrt(np.clip(C_mu_arr[:, 0, 0], 0, None))
        sig_dec = np.sqrt(np.clip(C_mu_arr[:, 1, 1], 0, None))
        ax.fill_between(iters, mu_arr[:, 0] - sig_ra,  mu_arr[:, 0] + sig_ra,
                        color='steelblue', alpha=0.15)
        ax.fill_between(iters, mu_arr[:, 1] - sig_dec, mu_arr[:, 1] + sig_dec,
                        color='firebrick',  alpha=0.15)
    ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('mas/yr'); ax.set_title(r'$\mu_{\rm pop}$ (±1σ shaded)', fontsize=9)
    ax.set_xticks(iters); ax.set_xticklabels(xlbls, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 0 right: convergence delta norms ─────────────────────────────────
    ax = axes[0, 1]
    dmu = np.max(np.abs(np.diff(mu_arr, axis=0)), axis=1)
    dgam = np.max(np.abs(np.diff(gamma_arr, axis=0)), axis=1)
    ax.semilogy(iters[1:], dmu,  'o-', color='firebrick',  lw=1.5, label='Δμ_pop (max)')
    ax.semilogy(iters[1:], dgam, 's-', color='steelblue',  lw=1.5, label='Δγ (max)')
    ax.axhline(1e-4, color='gray', lw=0.8, ls=':', alpha=0.6, label='Δμ tol')
    ax.axhline(1e-8, color='gray', lw=0.8, ls='--', alpha=0.6, label='Δγ tol')
    ax.set_xlabel('Iteration'); ax.set_ylabel('max |Δ|')
    ax.set_title('Convergence delta norms', fontsize=9)
    ax.set_xticks(iters[1:]); ax.set_xticklabels(xlbls[1:], fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Rows 1+ : all γ coefficients per chip×direction ──────────────────────
    _plot_gamma_block(axes[1, 0], gamma_arr[:, 1*_nb:2*_nb], f'γy hi — all {_nb} coeffs')
    _plot_gamma_block(axes[1, 1], gamma_arr[:, 3*_nb:4*_nb], f'γy lo — all {_nb} coeffs')
    axes[1, 0].set_ylabel('coefficient value')
    if _has_x:
        _plot_gamma_block(axes[2, 0], gamma_arr[:, 0*_nb:1*_nb], f'γx hi — all {_nb} coeffs')
        _plot_gamma_block(axes[2, 1], gamma_arr[:, 2*_nb:3*_nb], f'γx lo — all {_nb} coeffs')
        axes[2, 0].set_ylabel('coefficient value')
        axes[2, 0].set_xlabel('Iteration (0=ws)')
        axes[2, 1].set_xlabel('Iteration (0=ws)')
    axes[1, 0].set_xlabel('Iteration (0=ws)')
    axes[1, 1].set_xlabel('Iteration (0=ws)')

    fig.suptitle(
        f'Joint CTE convergence — spatial_order={_ref.spatial_order}  '
        f'mag_poly_order={_ref.mag_poly_order}  nb={_nb}  (iter 0 = warmstart)',
        fontsize=11)
    fig.tight_layout()
    out_name = f'{file_prefix}cte_convergence_history.png'
    fig.savefig(plot_dir / out_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plots/{out_name}")


def _plot_per_image_detector_residuals(
    output_dir,
    image_names: list[str],
    solver,
    filtered_spi: dict,
    arrays_stage1: dict,
    arrays_stage2: dict,
    arrays_stage3: dict,
    stage_labels: tuple = ("bp3m v2", "post-r/μ_pop", "post-CTE"),
    prefix: str = 'warmstart',
    vclip: float | None = None,
    img_meta: dict | None = None,
) -> None:
    """
    Per-visit 3×2 detector residual maps combining both chips (hi+lo).

    Rows: 3 stages (stage1=v2, stage2=post-r/mu_pop, stage3=post-CTE).
    Cols: dx_gdc (col 0) and dy_gdc (col 1).
    Axes: raw pixel coordinates covering the full detector (y_raw 0–4096).
    Color: GDC-frame residual in pixels.

    Images are grouped by visit rootname (stripping _hi/_lo suffix) so both
    chips appear in the same figure.  Saves one PNG per visit to
    output_dir/f'{prefix}_{root}.png'.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.time import Time as _Time
    from pathlib import Path as _Path
    from collections import defaultdict as _defaultdict

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Group images by visit rootname (strip _hi / _lo) ─────────────────────
    visit_groups = _defaultdict(list)
    for img in image_names:
        if img.endswith('_hi') or img.endswith('_lo'):
            root = img[:-3]
        else:
            root = img
        visit_groups[root].append(img)

    stages_arrs = [arrays_stage1, arrays_stage2, arrays_stage3]
    saved = 0

    for root, imgs in visit_groups.items():
        # ── Accumulate data from all chips in this visit ──────────────────────
        xr_by_stage  = [[] for _ in range(3)]
        yr_by_stage  = [[] for _ in range(3)]
        dx_by_stage  = [[] for _ in range(3)]
        dy_by_stage  = [[] for _ in range(3)]
        total_n = 0
        years   = []

        for img in imgs:
            spi = filtered_spi.get(img)
            if spi is None or len(spi) == 0:
                continue
            xc_key = f'{img}_X_c'
            if xc_key not in arrays_stage1:
                continue

            xc_all = arrays_stage1[xc_key].astype(float)
            yc_all = arrays_stage1[f'{img}_Y_c'].astype(float)
            xc_spi = spi['X'].to_numpy(float) - 2048.0
            yc_spi = spi['Y'].to_numpy(float) - 2048.0

            pos_dict = {(round(float(x), 2), round(float(y), 2)): k
                        for k, (x, y) in enumerate(zip(xc_all, yc_all))}
            match_idx = np.array([
                pos_dict.get((round(float(x), 2), round(float(y), 2)), -1)
                for x, y in zip(xc_spi, yc_spi)
            ])
            valid = match_idx >= 0
            if not valid.any():
                continue

            m_idx = match_idx[valid]
            x_raw = (spi['X_orig'].to_numpy(float)[valid]
                     if 'X_orig' in spi.columns else xc_spi[valid] + 2048.0)
            y_raw = (spi['Y_orig'].to_numpy(float)[valid]
                     if 'Y_orig' in spi.columns else yc_spi[valid] + 2048.0)

            for si, arr in enumerate(stages_arrs):
                dxk = f'{img}_dx_gdc'
                dyk = f'{img}_dy_gdc'
                dx = arr[dxk][m_idx].astype(float) if dxk in arr else np.zeros(m_idx.size)
                dy = arr[dyk][m_idx].astype(float) if dyk in arr else np.zeros(m_idx.size)
                xr_by_stage[si].append(x_raw)
                yr_by_stage[si].append(y_raw)
                dx_by_stage[si].append(dx)
                dy_by_stage[si].append(dy)

            total_n += int(valid.sum())
            hst_yr = float(_Time(float(solver.images[img]['hst_time_mjd']),
                                 format='mjd').jyear)
            years.append(hst_yr)

        if total_n == 0:
            continue

        # Concatenate across chips
        for si in range(3):
            xr_by_stage[si] = np.concatenate(xr_by_stage[si]) if xr_by_stage[si] else np.array([])
            yr_by_stage[si] = np.concatenate(yr_by_stage[si]) if yr_by_stage[si] else np.array([])
            dx_by_stage[si] = np.concatenate(dx_by_stage[si]) if dx_by_stage[si] else np.array([])
            dy_by_stage[si] = np.concatenate(dy_by_stage[si]) if dy_by_stage[si] else np.array([])

        # ── Colour limit from stage-1 residuals ───────────────────────────────
        if vclip is None:
            _vals1 = np.concatenate([np.abs(dx_by_stage[0]), np.abs(dy_by_stage[0])])
            _finite = _vals1[np.isfinite(_vals1)]
            _vc = float(np.percentile(_finite, 97)) if len(_finite) > 0 else 0.3
            _vc = max(_vc, 0.05)
        else:
            _vc = float(vclip)

        # ── Build figure ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 2, figsize=(12, 9),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})

        for row_i, stage_lbl in enumerate(stage_labels):
            x_d = xr_by_stage[row_i]
            y_d = yr_by_stage[row_i]
            for col_i, (vals, clbl) in enumerate(
                    zip([dx_by_stage[row_i], dy_by_stage[row_i]],
                        ['dx_gdc (px)', 'dy_gdc (px)'])):
                ax = axes[row_i, col_i]
                sc = ax.scatter(x_d, y_d, c=vals,
                                cmap='RdBu_r', vmin=-_vc, vmax=_vc,
                                s=1.5, alpha=0.6, linewidths=0,
                                rasterized=True)
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                if row_i == 0:
                    ax.set_title(clbl, fontsize=10, pad=4)
                ax.text(0.02, 0.97, stage_lbl, transform=ax.transAxes,
                        va='top', ha='left', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.75,
                                  pad=2, edgecolor='none'))
                ax.tick_params(labelsize=8)
                if col_i == 0:
                    ax.set_ylabel('y_raw (px)', fontsize=8)
                if row_i == 2:
                    ax.set_xlabel('x_raw (px)', fontsize=8)

        unique_yrs = sorted(set(round(y, 3) for y in years))
        yr_str = '/'.join(f'{y:.3f}' for y in unique_yrs)

        # Collect filter/instrument/detector across chips in this visit
        _inst_parts = []
        for _img in imgs:
            _m = (img_meta or {}).get(_img)
            if _m:
                _inst_parts.append(
                    f"{_m.get('filter','?')} {_m.get('instrument','?')}/{_m.get('detector','?')}"
                )
        _inst_str = '  |  '.join(dict.fromkeys(_inst_parts))   # unique, order-preserving

        _title = f'{root}   obs {yr_str} yr   n={total_n} stars   colour ±{_vc:.3f} px'
        if _inst_str:
            _title = f'{_inst_str}\n{_title}'
        fig.suptitle(_title, fontsize=10)

        out_path = output_dir / f'{prefix}_{root}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    n_visits = len(visit_groups)
    print(f"  Saved {saved}/{n_visits} per-visit residual maps → {output_dir}")


def _save_warmstart_stellar_astrometry(
    a_arr_ws: np.ndarray,
    solver,
    gaia_catalog: 'pd.DataFrame',
    output_dir: Path,
) -> None:
    """
    Save warmstart stellar astrometry as stellar_astrometry_warmstart.csv.
    Uses uniform 1 mas/yr uncertainties since C_vT is not returned from warmstart.
    """
    import pandas as _pd

    n_stars = a_arr_ws.shape[0]
    # Build star ID → row index mapping
    star_ids  = np.array(list(solver.star_id_to_idx.keys()), dtype=np.int64)
    star_rows = np.array(list(solver.star_id_to_idx.values()), dtype=int)
    order     = np.argsort(star_rows)
    star_ids  = star_ids[order]   # sorted by internal solver row index

    pmra   = a_arr_ws[star_rows[order], 2]
    pmdec  = a_arr_ws[star_rows[order], 3]

    # Get ra, dec from gaia_catalog where available
    ra_arr  = np.full(len(star_ids), np.nan)
    dec_arr = np.full(len(star_ids), np.nan)
    if gaia_catalog is not None and 'ra' in gaia_catalog.columns:
        _gc_gids = gaia_catalog['Gaia_id'].to_numpy(np.int64) \
            if 'Gaia_id' in gaia_catalog.columns else np.zeros(len(gaia_catalog), np.int64)
        _gc_ra   = gaia_catalog['ra'].to_numpy(float)
        _gc_dec  = gaia_catalog['dec'].to_numpy(float)
        _gmap    = {int(_gc_gids[i]): i for i in range(len(_gc_gids))}
        for k, gid in enumerate(star_ids):
            if int(gid) in _gmap:
                i = _gmap[int(gid)]
                ra_arr[k]  = _gc_ra[i]
                dec_arr[k] = _gc_dec[i]

    df = _pd.DataFrame({
        'Gaia_id':         star_ids,
        'ra':              ra_arr,
        'dec':             dec_arr,
        'pmra_bp3m':       pmra,
        'pmdec_bp3m':      pmdec,
        'sigma_pmra_bp3m': np.ones(len(star_ids)),   # uniform uncertainty
        'sigma_pmdec_bp3m': np.ones(len(star_ids)),
    })
    out_path = Path(output_dir) / 'stellar_astrometry_warmstart.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved: stellar_astrometry_warmstart.csv  ({len(df):,} stars)")


# ── Main function ──────────────────────────────────────────────────────────────

def run_alignment_cte(
    output_dir: Path,
    field_name: str,
    n_iter_bp3m: int = 10,
    n_iter_cte: int = 8,
    n_samples: int = 1000,
    mcmc_posteriors: bool = False,
    clip_sigma: float = 4.5,
    poly_order: int = 1,
    use_sparse: bool = False,
    no_plots: bool = False,
    plot_residuals: bool = False,
    hst_enable_iter: int = 5,
    hst_max_pm_unc: float = 5.0,
    hst_max_per_image: int = 100_000,
    outlier_sigma: float = 5.0,
    use_influence_clip: bool = True,
    influence_d_thresh: float = 1.0,
    influence_sigma_min: float = 2.0,
    hst_pm_sigma_diffuse: float = 100.0,
    pos_err_floor: float = 5e-3,
    det_chi2_threshold: float | None = None,
    use_soft_weights: bool = False,
    student_t_nu: float = 50.0,
    cte_delta_tol: float = 1e-3,
    cte_gamma_rtol: float = 1e-3,
    n_inner_delta: int = 5,
    bp3m_dir: Path | None = None,
) -> Path:
    """
    Joint CTE + astrometry alignment using master_combined_v2.csv catalog.

    Outer Gauss-Newton loop:
      1. Apply current CTE model → modify solver._img_data[img]['xys'].
      2. Run BP3M v2 solve (r_j, v_i) on CTE-corrected positions.
      3. Collect full-catalog GDC residuals (~127k stars) using current r_hat.
      4. Update CTE parameters (γ_c, δ_c) via Gauss-Newton.
      5. Check convergence; stop early if ‖Δγ‖/‖γ‖ < cte_gamma_rtol.

    Parameters
    ----------
    n_iter_bp3m : BP3M EM iterations per CTE outer step
    n_iter_cte  : max CTE Gauss-Newton outer iterations
    (remaining params identical to run_alignment_v2)

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_cte_results/)
    """
    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    from astropy.time import Time
    import pandas as pd

    from .data_loader_master import load_master_v2
    from .run_alignment_v2 import (
        V2AlignmentCallback,
        _compute_full_catalog_residuals_from_df,
        _plot_soft_weights,
    )
    from .run_alignment import _save_results

    data_root  = Path(output_dir)
    output_cte = data_root / field_name / "BP3M_cte_results"
    output_cte.mkdir(parents=True, exist_ok=True)

    # Remove stale convergence CSV so each run starts fresh
    _conv_csv = output_cte / 'cte_convergence.csv'
    if _conv_csv.exists():
        _conv_csv.unlink()

    print("\n" + "─" * 60)
    print("BP3M CTE: joint CTE + astrometry alignment")
    print("─" * 60)
    print(f"  n_iter_bp3m={n_iter_bp3m}  n_iter_cte={n_iter_cte}  "
          f"poly_order={poly_order}  clip_sigma={clip_sigma}")

    # ── Load data (same as v2) ─────────────────────────────────────────────────
    print(f"\n  Loading v2 master catalog for '{field_name}'...")
    images, stars_per_image, gaia_catalog, hst_only_mask = load_master_v2(
        data_root, field_name,
        hst_max_pm_unc=hst_max_pm_unc,
        hst_max_per_image=hst_max_per_image,
        pos_err_floor=pos_err_floor,
        det_chi2_threshold=det_chi2_threshold,
    )

    if not images:
        raise RuntimeError(f"No usable images for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)
    imgs         = {n: images[n] for n in image_names if n in images}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    print(f"  Stars: {len(gaia_catalog)} "
          f"({int((~hst_only_mask).sum())} Gaia + {int(hst_only_mask.sum())} HST-only)  "
          f"Images: {len(image_names)}")

    # ── Load full-catalog detection DataFrame (cached for all CTE iterations) ──
    print("\n  Loading full-catalog detection data (all filters, cached for CTE iterations)...")
    img_to_df = _load_full_catalog_df_all_filters(data_root, field_name)
    if img_to_df is None:
        raise RuntimeError(
            "No detections_*.csv or master_combined_v2.csv not found. "
            "Run hst_catalog_crossmatch first.")

    # ── Reference epoch: first exposure ───────────────────────────────────────
    all_mjds     = [float(images[img]['hst_time_mjd']) for img in image_names
                    if img in images]
    t_epoch0_mjd = float(min(all_mjds))
    t_epoch0_yr  = float(Time(t_epoch0_mjd, format='mjd').jyear)
    print(f"  t_epoch0 = {t_epoch0_yr:.4f} yr  "
          f"({Time(t_epoch0_mjd, format='mjd').isot[:10]})")
    t_launch_yr = _ACS_LAUNCH_YR
    print(f"  t_launch = {t_launch_yr:.4f} yr (ACS launch 2002-03-01)")

    # ── Inject v1/v2 BP3M transformation as warm start ────────────────────────
    v1_bp3m_dir   = data_root / field_name / "BP3M_results"
    v1_xform_path = v1_bp3m_dir / "image_transformations.csv"
    v1_abcdwz: dict[str, np.ndarray] = {}
    if v1_xform_path.exists():
        v1_df = pd.read_csv(v1_xform_path)
        for _, row in v1_df.iterrows():
            img_key = str(row["image_name"])
            v1_abcdwz[img_key] = np.array([
                float(row["a"]), float(row["b"]),
                float(row["c"]), float(row["d"]),
                float(row["w"]), float(row["z"]),
            ])
        imgs = {sub: dict(meta) for sub, meta in imgs.items()}
        for sub, meta in imgs.items():
            if sub in v1_abcdwz:
                meta["fcm_abcdwz"] = v1_abcdwz[sub]
        print(f"  Loaded v1 BP3M: {len(v1_abcdwz)} images as initialization")

    # ── Initialise solver ──────────────────────────────────────────────────────
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(
        imgs, filtered_spi, gaia_catalog,
        star_id_to_idx, image_names, star_in_image,
        poly_order=poly_order,
    )

    # ── Inject mag_inst into solver._img_data ─────────────────────────────────
    # Must be done before apply_cte_to_solver is called.
    _inject_mag_inst(solver, image_names, filtered_spi, gaia_catalog)

    # ── HST-only diffuse PM prior ──────────────────────────────────────────────
    if hst_pm_sigma_diffuse != 100.0:
        hst_star_indices = np.where(hst_only_mask)[0]
        if len(hst_star_indices) > 0:
            sigma_pm_inv2 = float(hst_pm_sigma_diffuse) ** -2
            solver._C_VG_inv_per_star[hst_star_indices, 2] = sigma_pm_inv2
            solver._C_VG_inv_per_star[hst_star_indices, 3] = sigma_pm_inv2

    # ── PM seeds for HST-only (callback) ──────────────────────────────────────
    _n_stars = len(gaia_catalog)
    pm_init  = np.full((_n_stars, 2), np.nan)
    if "pmra_xmatch" in gaia_catalog.columns:
        pm_init[:, 0] = pd.to_numeric(
            gaia_catalog["pmra_xmatch"], errors='coerce').fillna(np.nan).values
        pm_init[:, 1] = pd.to_numeric(
            gaia_catalog["pmdec_xmatch"], errors='coerce').fillna(np.nan).values

    callback = V2AlignmentCallback(
        hst_star_mask=hst_only_mask,
        hst_enable_iter=hst_enable_iter,
        outlier_sigma=outlier_sigma,
        pm_init=pm_init,
    )

    # ── Phase 0: fixed-transformation pre-filter ───────────────────────────────
    r_init_hat = None
    if v1_abcdwz:
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)
        solver._update_geometry(r_init_hat, solver.v_survey)
        print("\n  Phase 0: fixed-transform pre-filter (v1 BP3M posterior)")

        n_flag0 = 0
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            j_idx = image_names.index(img)
            r_j   = r_init_hat[j_idx * solver.N_R:(j_idx + 1) * solver.N_R]
            use   = d["use_for_fit"].copy()
            _v_pm = np.zeros_like(solver.v_survey[d["sidx"]])
            _v_pm[:, 2:] = solver.v_survey[d["sidx"], 2:]
            motion    = np.einsum("nij,nj->ni", d["JU"], _v_pm)
            x_pred    = np.einsum("nkl,l->nk", d["X_mat"], r_j) - motion
            resid_mag = np.hypot(*(d["xys"] - x_pred).T)
            if use.any():
                r_align    = resid_mag[use]
                mad_sigma  = np.median(np.abs(r_align - np.median(r_align))) / 0.6745
                thresh     = max(5.0 * mad_sigma, 0.3)
                bad        = use & (resid_mag > thresh)
                n_flag0   += int(bad.sum())
                if bad.any():
                    d["use_for_fit"][bad]     = False
                    d["use_for_fit_max"][bad] = False
                    if "use_for_astrom" in d:
                        d["use_for_astrom"][bad] = False
        print(f"  Phase 0: {n_flag0} detections flagged")
    else:
        # No v1 results: use the fast_cross_match initialisation
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)

    # ── Store xys_orig for all images ──────────────────────────────────────────
    for img in image_names:
        d = solver._img_data.get(img)
        if d is not None and 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

    # ── CTE warm start ─────────────────────────────────────────────────────────
    # Compute field mean PM from Gaia-matched (CTE-free) xmatch PMs before the
    # warm start. Without this, HST-only stars' CTE-contaminated pmra_xmatch
    # inflate γ₀, and subsequent iterations converge to this biased warm start.
    _ws_field_pm = _compute_warmstart_field_pm(data_root, field_name)
    print(f"\n  CTE warm start (field_mean_pm={_ws_field_pm})...")
    cte_params = _warm_start_cte_residuals(
        img_to_df, solver, image_names, r_init_hat, t_launch_yr,
        field_mean_pm=_ws_field_pm)

    # ── Outer CTE + BP3M Gauss-Newton loop ────────────────────────────────────
    clip      = clip_sigma if clip_sigma > 0 else None
    _min_outer = max(hst_enable_iter + 3, 4) if n_iter_bp3m >= hst_enable_iter else 4

    r_hat = C_r = v_hat = C_vT = a_arr = K_img = z_weights_out = None
    print(f"\n  Starting CTE outer loop ({n_iter_cte} iterations)...")

    for cte_iter in range(n_iter_cte):
        print(f"\n  ─── CTE iteration {cte_iter + 1}/{n_iter_cte} ───")
        t_iter = time.time()

        # Step 1: Apply CTE correction to solver xys
        apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr,
                            filtered_spi=filtered_spi)

        # Step 2: BP3M solve
        print(f"  BP3M solve ({n_iter_bp3m} EM iterations)...")
        (r_hat, C_r, v_hat, C_vT,
         a_arr, K_img, z_weights_out) = solver.fit(
            n_iter=n_iter_bp3m,
            clip_sigma=clip,
            inflate_hst_errors=True,
            inflate_from_iter=0,
            min_outer_iters=_min_outer,
            # Only run solver prefilter on first iteration (Phase 0 already did it)
            prefilter=False,
            use_influence_clip=use_influence_clip,
            influence_d_thresh=influence_d_thresh,
            influence_sigma_min=influence_sigma_min,
            use_two_tier=True,
            # Fire callback only on iter 0 (enables HST-only sources)
            per_iter_callback=callback if cte_iter == 0 else None,
            use_soft_weights=use_soft_weights,
            student_t_nu=student_t_nu,
        )
        print(f"  BP3M done ({time.time() - t_iter:.1f}s)")

        # Update solver.R for the new r_hat so subsequent CTE computation is correct
        solver._update_R(r_hat)

        # Step 3a: Compute field mean PM from Gaia-matched member stars
        field_mean_pm = _compute_field_mean_pm(
            solver, gaia_catalog, hst_only_mask, v_hat)

        # Step 3b: Collect full-catalog residuals with current r_hat
        print(f"  Collecting residuals ({len(img_to_df)} images)  "
              f"[field PM: {field_mean_pm[0]:.2f}, {field_mean_pm[1]:.2f} mas/yr]...")
        residuals = collect_cte_residuals(
            img_to_df, solver, image_names, r_hat, t_launch_yr,
            field_mean_pm=field_mean_pm)

        # Step 4: Update CTE parameters
        gamma_before = {c: cte_params[c].gamma_y.copy() for c in ('hi', 'lo')}
        cte_params, info = update_cte_params(
            residuals, cte_params,
            n_inner=n_inner_delta,
            delta_tol=cte_delta_tol,
        )
        _save_cte_convergence(output_cte, cte_iter + 1, cte_params, info)

        # Step 5: Convergence check (require at least 2 iterations)
        gamma_norms  = [np.linalg.norm(cte_params[c].gamma_y) for c in ('hi', 'lo')]
        gamma_deltas = [np.linalg.norm(cte_params[c].gamma_y - gamma_before[c])
                        for c in ('hi', 'lo')]
        gamma_rchg = max(
            d / max(n, 1e-10) for d, n in zip(gamma_deltas, gamma_norms))

        print(f"  γ_y relative change = {gamma_rchg:.4e}")

        if gamma_rchg < cte_gamma_rtol and cte_iter >= 1:
            print(f"  CTE converged at iteration {cte_iter + 1}")
            break

    # ── Final BP3M solve with converged CTE ───────────────────────────────────
    print("\n  Final BP3M solve with converged CTE parameters...")
    apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr,
                        filtered_spi=filtered_spi)
    (r_hat, C_r, v_hat, C_vT,
     a_arr, K_img, z_weights_out) = solver.fit(
        n_iter=n_iter_bp3m,
        clip_sigma=clip,
        inflate_hst_errors=True,
        inflate_from_iter=0,
        min_outer_iters=_min_outer,
        prefilter=False,
        use_influence_clip=use_influence_clip,
        influence_d_thresh=influence_d_thresh,
        influence_sigma_min=influence_sigma_min,
        use_two_tier=True,
        per_iter_callback=None,
        use_soft_weights=use_soft_weights,
        student_t_nu=student_t_nu,
    )
    solver._update_R(r_hat)

    # ── Save converged CTE parameters ─────────────────────────────────────────
    cte_out = {}
    for chip in ('hi', 'lo'):
        p = cte_params[chip]
        cte_out[f'{chip}_gamma_x']       = p.gamma_x
        cte_out[f'{chip}_gamma_y']       = p.gamma_y
        cte_out[f'{chip}_y_readout_raw'] = np.array([p.y_readout_raw])
        cte_out[f'{chip}_x0']            = np.array([p.x0])
    cte_out['t_epoch0_yr']  = np.array([t_epoch0_yr])
    cte_out['t_launch_yr']  = np.array([t_launch_yr])
    np.savez(output_cte / 'cte_params.npz', **cte_out)
    print(f"  Saved: cte_params.npz  "
          f"(|γ_y_hi|={np.linalg.norm(cte_params['hi'].gamma_y):.4e}, "
          f"|γ_y_lo|={np.linalg.norm(cte_params['lo'].gamma_y):.4e})")

    # ── Posteriors ────────────────────────────────────────────────────────────
    if mcmc_posteriors:
        print(f"  Drawing {n_samples} posterior samples (MCMC marginalisation)...")
        _, v_mean, v_cov = solver.sample_posteriors(
            r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)
    else:
        print(f"  Computing analytic marginalised posteriors...")
        v_mean, v_cov = solver.compute_analytic_posteriors(r_hat, C_r, a_arr, K_img, C_vT)

    # ── Save results (same format as v2) ──────────────────────────────────────
    _save_results(
        output_cte, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            "n_iter_bp3m":   n_iter_bp3m,
            "n_iter_cte":    n_iter_cte,
            "n_samples":     n_samples,
            "clip_sigma":    clip_sigma,
            "poly_order":    poly_order,
            "t_epoch0_yr":   t_epoch0_yr,
            "t_launch_yr":   t_launch_yr,
        },
    )

    # ── Star influence ────────────────────────────────────────────────────────
    print("  Computing star influence metrics...")
    try:
        import pandas as _pd
        influence_df = solver.compute_star_influence(r_hat, C_r, a_arr)
        influence_df.to_csv(output_cte / "star_influence.csv", index=False)
        print(f"  Saved: star_influence.csv  ({len(influence_df)} star-image pairs)")
    except Exception as _exc:
        print(f"  WARNING: star influence computation failed — {_exc}")
        import traceback; traceback.print_exc()

    # ── Save post-CTE full-catalog residuals ──────────────────────────────────
    print("\n  Saving post-CTE full-catalog residuals...")
    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    out_arrays_cte = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat)
    if out_arrays_cte:
        np.savez(output_cte / 'detections_catalog_cte.npz', **out_arrays_cte)
        n_imgs  = sum(1 for k in out_arrays_cte if k.endswith('_X_c'))
        n_total = sum(len(v) for k, v in out_arrays_cte.items() if k.endswith('_X_c'))
        print(f"  Saved detections_catalog_cte.npz: {n_imgs} images, "
              f"{n_total:,} detections")

    # ── Soft-weight output ─────────────────────────────────────────────────────
    if use_soft_weights and z_weights_out is not None:
        import csv as _csv
        rows = []
        for img, z in z_weights_out.items():
            if z is None:
                continue
            d = solver._img_data[img]
            for k in range(len(z)):
                rows.append({'image': img, 'star_idx': int(d['sidx'][k]),
                             'z_det': float(z[k])})
        zcsv = output_cte / 'soft_weights.csv'
        with open(zcsv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=['image', 'star_idx', 'z_det'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: soft_weights.csv  ({len(rows)} detection weights)")
        _plot_dir = output_cte / 'plots'
        _plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_soft_weights(z_weights_out, solver, _plot_dir)
        except Exception as exc:
            print(f"  WARNING: soft_weights_diagnostic plot failed — {exc}")

    # ── Standard BP3M diagnostic plots ────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating standard BP3M diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_cte,
                       plot_residuals=plot_residuals)
        except Exception as exc:
            print(f"  WARNING: standard plots failed — {exc}")

        # ── CTE-specific diagnostic plots ─────────────────────────────────────
        before_npz = data_root / field_name / "BP3M_v2_results" / "detections_catalog.npz"
        after_npz  = output_cte / "detections_catalog_cte.npz"
        try:
            _plot_cte_diagnostics(
                output_cte, cte_params,
                before_npz=before_npz,
                after_npz=after_npz,
                image_names=image_names,
                solver=solver,
            )
        except Exception as exc:
            import traceback
            print(f"  WARNING: CTE diagnostic plots failed — {exc}")
            traceback.print_exc()

    print(f"\n  CTE results written to: {output_cte}")
    return output_cte


def _compute_mag_normalization(solver, image_names) -> tuple[float, float]:
    """Return (mag_norm_ref, mag_norm_scale) from all finite instrumental mags.

    ref   = median of 2nd-98th percentile values
    scale = (98th - 2nd percentile) / 2  (half-range, robust to outliers)
    """
    mags = []
    for img in image_names:
        d = solver._img_data.get(img)
        if d is None:
            continue
        m = d.get('mag_inst')
        if m is None:
            continue
        mf = m[np.isfinite(m)]
        if len(mf):
            mags.append(mf)
    if not mags:
        return 0.0, 1.0
    all_m = np.concatenate(mags)
    p2, p98 = np.percentile(all_m, [2, 98])
    ref   = float(0.5 * (p2 + p98))
    scale = float(max((p98 - p2) / 2.0, 1e-6))
    return ref, scale


# ── Joint CTE + population entry point ────────────────────────────────────────

def run_alignment_joint_cte(
    output_dir: Path,
    field_name: str,
    # Population / LVD priors
    sigma_pm: float = 0.01,
    plx_pop: float = 0.004,
    sigma_plx_tot: float = 1e-4,
    mu_pop_prior_sigma: float = 0.5,
    # Iteration / convergence
    n_iter_joint: int = 20,
    member_sigma_clip: float = 3.0,
    regularize_gamma: float = 1e-8,
    pm_sys_floor: float = 0.2,
    # CTE polynomial orders
    mag_poly_order: int = 3,
    spatial_order: int = 2,
    # Standard alignment options (same defaults as run_alignment_cte)
    poly_order: int = 1,
    use_sparse: bool = False,
    no_plots: bool = False,
    plot_residuals: bool = True,
    hst_max_pm_unc: float = 5.0,
    hst_max_per_image: int = 100_000,
    hst_pm_sigma_diffuse: float = 100.0,
    pos_err_floor: float = 5e-3,
    det_chi2_threshold: float | None = None,
    bp3m_dir: Path | None = None,
    warmstart_only: bool = False,
    fit_cte_x: bool = True,
) -> Path:
    """
    Joint CTE + population mean PM alignment.

    Replaces the alternating solver.fit() + update_cte_params() loop with
    _run_joint_cte_loop, which simultaneously fits image transformations (r),
    CTE coefficients (γ, 20 params), and population mean PM (μ_pop) after
    analytically marginalising stellar astrometry {v_i}.

    LVD priors for the target field:
      sigma_pm        : intrinsic PM dispersion (mas/yr).
                        Approx. σ_LOS / (4.74047 × d_kpc).
      plx_pop         : mean parallax (mas) = 1000 / d_kpc.
      sigma_plx_tot   : total parallax uncertainty (mas).
      mu_pop_prior_sigma : width of the Gaussian prior on μ_pop (mas/yr).

    The population mean PM prior is warm-started from the field Gaia cross-match
    (same as warm_start_cte).

    Parameters
    ----------
    output_dir  : root data directory (same as run_alignment_cte)
    field_name  : subdirectory name (e.g. 'Leo_I')
    sigma_pm    : LVD intrinsic PM dispersion σ_PM (mas/yr)
    plx_pop     : LVD mean parallax (mas)
    sigma_plx_tot : LVD total parallax uncertainty (mas, incl. depth)
    mu_pop_prior_sigma : Gaussian prior half-width on μ_pop (mas/yr)
    n_iter_joint : maximum joint Gauss-Newton iterations
    member_sigma_clip : sigma for PM-distance membership cut

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_joint_cte_results/)
    """
    import time as _time
    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    from astropy.time import Time
    import pandas as pd

    from .data_loader_master import load_master_v2
    from .run_alignment_v2 import (
        V2AlignmentCallback,
        _compute_full_catalog_residuals_from_df,
    )
    from .run_alignment import _save_results

    data_root  = Path(output_dir)
    output_dir_joint = data_root / field_name / "BP3M_joint_cte_results"
    output_dir_joint.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 60)
    print("BP3M joint CTE + population: joint (r, γ, μ_pop) solve")
    print("─" * 60)
    print(f"  sigma_pm={sigma_pm:.4f} mas/yr  plx_pop={plx_pop:.5f} mas  "
          f"mu_prior_sigma={mu_pop_prior_sigma:.2f} mas/yr")
    print(f"  n_iter_joint={n_iter_joint}  member_sigma_clip={member_sigma_clip}")

    # ── Load data (identical to run_alignment_cte) ────────────────────────────
    print(f"\n  Loading v2 master catalog for '{field_name}'...")
    images, stars_per_image, gaia_catalog, hst_only_mask = load_master_v2(
        data_root, field_name,
        hst_max_pm_unc=hst_max_pm_unc,
        hst_max_per_image=hst_max_per_image,
        pos_err_floor=pos_err_floor,
        det_chi2_threshold=det_chi2_threshold,
    )
    if not images:
        raise RuntimeError(f"No usable images for '{field_name}'.")

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)
    imgs         = {n: images[n] for n in image_names if n in images}
    filtered_spi = {n: stars_per_image[n] for n in image_names}

    print(f"  Stars: {len(gaia_catalog)} "
          f"({int((~hst_only_mask).sum())} Gaia + {int(hst_only_mask.sum())} HST-only)  "
          f"Images: {len(image_names)}")

    img_to_df = _load_full_catalog_df_all_filters(data_root, field_name)
    if img_to_df is None:
        raise RuntimeError(
            "No detections_*.csv or master_combined_v2.csv not found. "
            "Run hst_catalog_crossmatch first.")

    # Per-image filter/instrument/detector metadata for residual figure titles
    _img_meta = {}
    for _img, _df in img_to_df.items():
        if len(_df) == 0:
            continue
        _r = _df.iloc[0]
        _img_meta[_img] = {
            'filter':     str(_r['filter'])     if 'filter'     in _df.columns else '?',
            'instrument': str(_r['instrument']) if 'instrument' in _df.columns else '?',
            'detector':   str(_r['detector'])   if 'detector'   in _df.columns else '?',
        }

    all_mjds     = [float(images[img]['hst_time_mjd']) for img in image_names
                    if img in images]
    t_epoch0_mjd = float(min(all_mjds))
    t_epoch0_yr  = float(Time(t_epoch0_mjd, format='mjd').jyear)
    t_launch_yr  = _ACS_LAUNCH_YR
    print(f"  t_epoch0 = {t_epoch0_yr:.4f} yr")

    # ── Warm start from v1/v2 transformations ─────────────────────────────────
    v1_bp3m_dir   = data_root / field_name / "BP3M_results"
    v1_xform_path = v1_bp3m_dir / "image_transformations.csv"
    v1_abcdwz: dict[str, np.ndarray] = {}
    if v1_xform_path.exists():
        v1_df = pd.read_csv(v1_xform_path)
        for _, row in v1_df.iterrows():
            img_key = str(row["image_name"])
            v1_abcdwz[img_key] = np.array([
                float(row["a"]), float(row["b"]),
                float(row["c"]), float(row["d"]),
                float(row["w"]), float(row["z"]),
            ])
        imgs = {sub: dict(meta) for sub, meta in imgs.items()}
        for sub, meta in imgs.items():
            if sub in v1_abcdwz:
                meta["fcm_abcdwz"] = v1_abcdwz[sub]
        print(f"  Loaded v1 BP3M: {len(v1_abcdwz)} images as initialization")

    # ── Initialise solver ──────────────────────────────────────────────────────
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(
        imgs, filtered_spi, gaia_catalog,
        star_id_to_idx, image_names, star_in_image,
        poly_order=poly_order,
    )

    _inject_mag_inst(solver, image_names, filtered_spi, gaia_catalog)

    # ── CTE template (magnitude polynomial normalization) ─────────────────────
    _mag_norm_ref, _mag_norm_scale = _compute_mag_normalization(solver, image_names)
    cte_template = default_cte_params(mag_poly_order, _mag_norm_ref, _mag_norm_scale, spatial_order)
    _nb = _cte_n_spatial(spatial_order) * (mag_poly_order + 1) - 1
    print(f"  CTE spatial_order={spatial_order}  mag_poly_order={mag_poly_order}  nb={_nb}  "
          f"mag_norm: ref={_mag_norm_ref:.2f}  scale={_mag_norm_scale:.2f}")

    if hst_pm_sigma_diffuse != 100.0:
        hst_star_indices = np.where(hst_only_mask)[0]
        if len(hst_star_indices) > 0:
            sigma_pm_inv2 = float(hst_pm_sigma_diffuse) ** -2
            solver._C_VG_inv_per_star[hst_star_indices, 2] = sigma_pm_inv2
            solver._C_VG_inv_per_star[hst_star_indices, 3] = sigma_pm_inv2

    # ── Phase 0: fixed-transform pre-filter ──────────────────────────────────
    r_init_hat = None
    if v1_abcdwz:
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)
        solver._update_geometry(r_init_hat, solver.v_survey)
        print("\n  Phase 0: fixed-transform pre-filter")
        n_flag0 = 0
        for img in image_names:
            d = solver._img_data.get(img)
            if d is None:
                continue
            j_idx  = image_names.index(img)
            r_j    = r_init_hat[j_idx * solver.N_R:(j_idx + 1) * solver.N_R]
            use    = d["use_for_fit"].copy()
            _v_pm  = np.zeros_like(solver.v_survey[d["sidx"]])
            _v_pm[:, 2:] = solver.v_survey[d["sidx"], 2:]
            motion    = np.einsum("nij,nj->ni", d["JU"], _v_pm)
            x_pred    = np.einsum("nkl,l->nk", d["X_mat"], r_j) - motion
            resid_mag = np.hypot(*(d["xys"] - x_pred).T)
            if use.any():
                r_align   = resid_mag[use]
                mad_sigma = np.median(np.abs(r_align - np.median(r_align))) / 0.6745
                thresh    = max(5.0 * mad_sigma, 0.3)
                bad       = use & (resid_mag > thresh)
                n_flag0  += int(bad.sum())
                if bad.any():
                    d["use_for_fit"][bad]     = False
                    d["use_for_fit_max"][bad] = False
                    if "use_for_astrom" in d:
                        d["use_for_astrom"][bad] = False
        print(f"  Phase 0: {n_flag0} detections flagged")
    else:
        r_init_hat = np.concatenate([solver._img_data[img]["r_init"]
                                      for img in image_names])
        solver._update_R(r_init_hat)

    # Store xys_orig (required by _joint_solve_cte)
    for img in image_names:
        d = solver._img_data.get(img)
        if d is not None and 'xys_orig' not in d:
            d['xys_orig'] = d['xys'].copy()

    # ── Pre-enable HST-only stars for astrometry (use_for_astrom only) ────────
    # HST-only sources start with use_for_alignment=False so the solver never
    # enables them via the standard quality filter.  Here we do the equivalent
    # of V2AlignmentCallback's transition step: enable sources with ≥ 2
    # detections as use_for_astrom=True (NOT use_for_fit) and seed their
    # v_survey PMs from the crossmatch values in the catalog.
    print("  Pre-enabling HST-only stars for astrometry...")
    _pm_seed = None
    if 'pmra_xmatch' in gaia_catalog.columns and 'pmdec_xmatch' in gaia_catalog.columns:
        _pm_seed = gaia_catalog[['pmra_xmatch', 'pmdec_xmatch']].to_numpy(float)
    _v2cb = V2AlignmentCallback(
        hst_star_mask=hst_only_mask,
        hst_enable_iter=1,
        pm_init=_pm_seed,
    )
    solver._r_hat_current = r_init_hat
    _v2cb(solver, it_outer=1)   # transition: enables hst-only stars with n_det >= 2
    solver._use_two_tier = True  # honour use_for_astrom in _joint_solve_cte

    # ── Population mean PM prior from Gaia cross-match ────────────────────────
    _ws_field_pm = _compute_warmstart_field_pm(data_root, field_name)
    if _ws_field_pm is not None:
        mu_pop_prior = np.array([_ws_field_pm[0], _ws_field_pm[1]])
    else:
        print("  WARNING: could not estimate field PM — using (0, 0) as prior")
        mu_pop_prior = np.zeros(2)
    C_pop_prior_inv = np.eye(2) / mu_pop_prior_sigma**2

    print(f"  μ_pop prior: ({mu_pop_prior[0]:+.3f}, {mu_pop_prior[1]:+.3f}) ± "
          f"{mu_pop_prior_sigma:.2f} mas/yr")

    # ── Initial member selection from catalog PMs ──────────────────────────────
    # Gaia rows: read pmra_xmatch from master CSV (NaN in gaia_catalog in memory).
    # HST-only rows: pmra_xmatch is already in gaia_catalog; read sigma_pmdec_xmatch
    # from master CSV via rounded-PM key lookup for the quality cut.
    _mcat_path   = data_root / field_name / 'hst_xmatch' / 'master_combined_v2.csv'
    _init_radius = member_sigma_clip * max(sigma_pm, pm_sys_floor)
    _mu_ra, _mu_dec = float(mu_pop_prior[0]), float(mu_pop_prior[1])
    if _mcat_path.exists():
        _want_cols = {'gaia_source_id', 'pmra_xmatch', 'pmdec_xmatch',
                      'sigma_pmdec_xmatch'}
        _mcat = pd.read_csv(_mcat_path,
                            usecols=lambda c: c in _want_cols,
                            dtype={'gaia_source_id': np.int64},
                            low_memory=False)

        # ── Gaia members (pmra_xmatch from CSV keyed by gaia_source_id) ──────
        _mcat_gaia = (_mcat[_mcat['gaia_source_id'] > 0]
                      .drop_duplicates('gaia_source_id')
                      .set_index('gaia_source_id'))
        _gaia_idxs = np.where(~hst_only_mask)[0]
        _gaia_ids  = gaia_catalog.iloc[_gaia_idxs]['Gaia_id'].to_numpy(np.int64)
        _g_pmra  = np.array([float(_mcat_gaia.loc[g, 'pmra_xmatch'])
                             if g in _mcat_gaia.index else np.nan for g in _gaia_ids])
        _g_pmdec = np.array([float(_mcat_gaia.loc[g, 'pmdec_xmatch'])
                             if g in _mcat_gaia.index else np.nan for g in _gaia_ids])
        _ok_gaia = (np.isfinite(_g_pmra) & np.isfinite(_g_pmdec)
                    & (np.hypot(_g_pmra - _mu_ra, _g_pmdec - _mu_dec) < _init_radius))
        member_sidx_gaia = _gaia_idxs[_ok_gaia]

        # ── HST-only members ─────────────────────────────────────────────────
        # pmra_xmatch is populated in gaia_catalog for HST-only rows.
        # sigma_pmdec_xmatch: build lookup from CSV HST rows (gaia_source_id==0)
        # keyed by rounded (pmra, pmdec) pair — same approach as _warm_start_cte_residuals.
        _hst_idxs  = np.where(hst_only_mask)[0]
        _h_pmra  = gaia_catalog.iloc[_hst_idxs]['pmra_xmatch'].to_numpy(float)
        _h_pmdec = gaia_catalog.iloc[_hst_idxs]['pmdec_xmatch'].to_numpy(float)

        _h_sigma = np.full(len(_hst_idxs), np.nan)
        if 'sigma_pmdec_xmatch' in _mcat.columns:
            _mhst = _mcat[(_mcat['gaia_source_id'] == 0)
                          & _mcat['pmra_xmatch'].notna()
                          & _mcat['sigma_pmdec_xmatch'].notna()].copy()
            _sig_keys = (_mhst['pmra_xmatch'].round(6).astype(str) + '_'
                         + _mhst['pmdec_xmatch'].round(6).astype(str))
            _sig_lookup = dict(zip(_sig_keys, _mhst['sigma_pmdec_xmatch']))
            _hkeys = (pd.Series(_h_pmra).round(6).astype(str) + '_'
                      + pd.Series(_h_pmdec).round(6).astype(str))
            _h_sigma = np.array([_sig_lookup.get(k, np.nan) for k in _hkeys])

        _ok_hst = (
            np.isfinite(_h_pmra) & np.isfinite(_h_pmdec)
            & (np.hypot(_h_pmra - _mu_ra, _h_pmdec - _mu_dec) < _init_radius)
            & (np.hypot(_h_pmra, _h_pmdec) < 3.0)       # |PM| < 3 mas/yr
            & np.isfinite(_h_sigma) & (_h_sigma < 1.0)  # σ_PM < 1 mas/yr
        )
        member_sidx_hst  = _hst_idxs[_ok_hst]
        member_sidx_init = np.concatenate([member_sidx_gaia, member_sidx_hst])
    else:
        member_sidx_gaia = np.where(~hst_only_mask)[0]
        member_sidx_hst  = np.array([], dtype=int)
        member_sidx_init = member_sidx_gaia
    print(f"  Initial members ({member_sigma_clip}σ={_init_radius:.3f} mas/yr): "
          f"{len(member_sidx_gaia)} Gaia + {len(member_sidx_hst)} HST-only "
          f"= {len(member_sidx_init)} total")

    # ── bp3m v2 residuals (BEFORE warm start, using r_init and current solver.R) ──
    _ws_v2 = None
    if not no_plots and img_to_df is not None:
        _bp3m_gids_pre = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
        try:
            _ws_v2 = _compute_full_catalog_residuals_from_df(
                img_to_df, _bp3m_gids_pre, solver, image_names, r_init_hat)
        except Exception as _exc_v2:
            print(f"  WARNING: v2 pre-warmstart residuals failed — {_exc_v2}")

    # ── CTE warm start ─────────────────────────────────────────────────────────
    print("\n  CTE warm start...")
    cte_params, mu_pop_warm, r_ws_diag, a_arr_ws = warm_start_cte(
        solver, image_names, filtered_spi, t_launch_yr,
        member_sidx_gaia, member_sidx_hst,
        sigma_pm, plx_pop, sigma_plx_tot,
        mu_pop_prior, C_pop_prior_inv, r_init_hat,
        cte_template=cte_template,
        regularize_gamma=regularize_gamma,
        output_dir=output_dir_joint,
        fit_cte_x=fit_cte_x,
    )

    # ── Warmstart CTE diagnostic plots ────────────────────────────────────────
    if not no_plots and img_to_df is not None:
        print("\n  Computing warmstart CTE diagnostic residuals...")
        _bp3m_gids_ws = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
        _ws_before = _compute_full_catalog_residuals_from_df(
            img_to_df, _bp3m_gids_ws, solver, image_names, r_ws_diag)
        _ws_after  = _apply_cte_to_residual_arrays(
            _ws_before, image_names, cte_params, t_launch_yr, solver, filtered_spi)
        np.savez(output_dir_joint / 'detections_catalog_ws_r.npz', **_ws_before)
        np.savez(output_dir_joint / 'detections_catalog_ws_cte.npz', **_ws_after)
        try:
            _plot_cte_diagnostics(
                output_dir_joint, cte_params,
                before_npz=output_dir_joint / 'detections_catalog_ws_r.npz',
                after_npz=output_dir_joint / 'detections_catalog_ws_cte.npz',
                image_names=image_names,
                solver=solver,
                file_prefix='warmstart_',
            )
        except Exception as _exc:
            print(f"  WARNING: warmstart CTE diagnostics failed — {_exc}")
        # Warmstart cte_pm_vs_detector using warmstart stellar astrometry
        try:
            _save_warmstart_stellar_astrometry(
                a_arr_ws, solver, gaia_catalog, output_dir_joint)
            _plot_cte_diagnostics(
                output_dir_joint, cte_params,
                before_npz=None,
                after_npz=None,
                image_names=image_names,
                solver=solver,
                file_prefix='warmstart_',
                astrom_csv=output_dir_joint / 'stellar_astrometry_warmstart.csv',
            )
        except Exception as _exc:
            print(f"  WARNING: warmstart pm_vs_detector failed — {_exc}")

        # ── Per-image detector residual maps (3 stages × dx/dy) ──────────────
        if _ws_v2 is not None:
            try:
                _plot_per_image_detector_residuals(
                    output_dir_joint / 'plots' / 'residuals',
                    image_names, solver, filtered_spi,
                    arrays_stage1=_ws_v2,
                    arrays_stage2=_ws_before,
                    arrays_stage3=_ws_after,
                    stage_labels=("bp3m v2",
                                  "post-r/μ_pop warmstart",
                                  "post-CTE warmstart"),
                    prefix='warmstart',
                    img_meta=_img_meta,
                )
            except Exception as _exc:
                import traceback
                print(f"  WARNING: per-image residual maps failed — {_exc}")
                traceback.print_exc()

    if warmstart_only:
        print("\n  warmstart_only=True — stopping after warm start. "
              f"Results in: {output_dir_joint}")
        return output_dir_joint

    # ── Joint solve loop ───────────────────────────────────────────────────────
    print(f"\n  Starting joint (r, γ, μ_pop) loop ({n_iter_joint} iterations)...")
    t0 = _time.time()

    (r_hat, C_r, gamma_hat, mu_pop_hat, C_shared,
     a_arr, K_img, C_vT, cte_params,
     _gamma_hist, _mu_pop_hist, _C_mu_hist) = _run_joint_cte_loop(
        solver, image_names, cte_params, t_launch_yr, filtered_spi,
        hst_only_mask,
        sigma_pm, plx_pop, sigma_plx_tot,
        mu_pop_prior, C_pop_prior_inv,
        n_iter=n_iter_joint,
        member_sigma_clip=member_sigma_clip,
        regularize_gamma=regularize_gamma,
        pm_sys_floor=pm_sys_floor,
        gaia_catalog=gaia_catalog,
        member_sidx_init=member_sidx_init,
        mu_pop_init=mu_pop_warm,
        fit_cte_x=fit_cte_x,
    )
    print(f"  Joint loop done ({_time.time() - t0:.1f}s)")
    if C_shared is not None:
        _C_mu    = C_shared[-2:, -2:]
        _sig_ra  = float(np.sqrt(_C_mu[0, 0]))
        _sig_dec = float(np.sqrt(_C_mu[1, 1]))
        _rho_fin = float(_C_mu[0, 1] / (_sig_ra * _sig_dec + 1e-30))
        print(f"  Final μ_pop = ({mu_pop_hat[0]:+.4f}±{_sig_ra:.4f}, "
              f"{mu_pop_hat[1]:+.4f}±{_sig_dec:.4f}) mas/yr  ρ={_rho_fin:+.3f}")
    else:
        print(f"  Final μ_pop = ({mu_pop_hat[0]:+.4f}, {mu_pop_hat[1]:+.4f}) mas/yr")
    print(f"  Final |γ_y_hi| = {np.linalg.norm(gamma_hat[_nb:2*_nb]):.4e}  "
          f"|γ_y_lo| = {np.linalg.norm(gamma_hat[3*_nb:4*_nb]):.4e}")

    # Use a_arr as the working stellar astrometry estimate (matches solver.fit convention)
    v_hat = a_arr

    # ── Save converged CTE parameters ─────────────────────────────────────────
    cte_out = {}
    for chip in ('hi', 'lo'):
        p = cte_params[chip]
        cte_out[f'{chip}_gamma_x']       = p.gamma_x
        cte_out[f'{chip}_gamma_y']       = p.gamma_y
        cte_out[f'{chip}_y_readout_raw'] = np.array([p.y_readout_raw])
        cte_out[f'{chip}_x0']            = np.array([p.x0])
    cte_out['t_epoch0_yr']       = np.array([t_epoch0_yr])
    cte_out['t_launch_yr']       = np.array([t_launch_yr])
    cte_out['mu_pop_hat']        = mu_pop_hat
    cte_out['gamma_hat']         = gamma_hat
    cte_out['mu_pop_prior']      = mu_pop_prior
    cte_out['mu_pop_prior_sigma'] = np.array([mu_pop_prior_sigma])
    cte_out['mag_poly_order']    = np.array([mag_poly_order])
    cte_out['spatial_order']     = np.array([spatial_order])
    cte_out['mag_norm_ref']      = np.array([_mag_norm_ref])
    cte_out['mag_norm_scale']    = np.array([_mag_norm_scale])
    if C_shared is not None:
        _n_gamma = 4 * _nb
        n_r = C_shared.shape[0] - _n_gamma - 2  # n_shared = n_r + n_gamma + n_mu(2)
        cte_out['C_mu_pop'] = C_shared[-2:, -2:]
        cte_out['C_gamma']  = C_shared[n_r:n_r + _n_gamma, n_r:n_r + _n_gamma]
    np.savez(output_dir_joint / 'cte_params.npz', **cte_out)
    print(f"  Saved: cte_params.npz")

    # ── Save per-iteration history ────────────────────────────────────────────
    _sp_lbls = _cte_basis_labels(spatial_order)
    _coef_lbls = [f'1·{s}' for s in _sp_lbls[1:]]
    for k in range(1, mag_poly_order + 1):
        _coef_lbls += [f'm^{k}·{s}' if k > 1 else f'm·{s}' for s in _sp_lbls]
    np.savez(output_dir_joint / 'cte_iteration_history.npz',
             gamma_history  = np.array(_gamma_hist),    # (n_iter+1, 4*nb)
             mu_pop_history = np.array(_mu_pop_hist),   # (n_iter+1, 2)
             C_mu_history   = np.array(_C_mu_hist),     # (n_iter+1, 2, 2)
             coef_labels    = np.array(_coef_lbls),     # (nb,) labels for one chip-direction block
             spatial_order  = np.array([spatial_order]),
             mag_poly_order = np.array([mag_poly_order]),
             nb             = np.array([_nb]))
    print(f"  Saved: cte_iteration_history.npz  ({len(_gamma_hist)} iters, nb={_nb})")

    # ── Analytic marginalised posteriors ──────────────────────────────────────
    print("  Computing analytic marginalised posteriors...")
    v_mean, v_cov = solver.compute_analytic_posteriors(r_hat, C_r, a_arr, K_img, C_vT)

    # ── Save results ──────────────────────────────────────────────────────────
    _save_results(
        output_dir_joint, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            "solver":          "joint_cte",
            "sigma_pm":        sigma_pm,
            "plx_pop":         plx_pop,
            "sigma_plx_tot":   sigma_plx_tot,
            "mu_pop_prior":        mu_pop_prior.tolist(),
            "mu_pop_prior_sigma":  mu_pop_prior_sigma,
            "mu_pop_hat":          mu_pop_hat.tolist(),
            "sigma_mu_pop":        (np.sqrt(np.diag(C_shared[-2:, -2:])).tolist()
                                    if C_shared is not None else None),
            "n_iter_joint":        n_iter_joint,
            "poly_order":      poly_order,
            "t_epoch0_yr":     t_epoch0_yr,
            "t_launch_yr":     t_launch_yr,
        },
    )

    # ── Star influence ────────────────────────────────────────────────────────
    try:
        import pandas as _pd
        influence_df = solver.compute_star_influence(r_hat, C_r, a_arr)
        influence_df.to_csv(output_dir_joint / "star_influence.csv", index=False)
        print(f"  Saved: star_influence.csv  ({len(influence_df)} star-image pairs)")
    except Exception as _exc:
        print(f"  WARNING: star influence computation failed — {_exc}")

    # ── Post-CTE full-catalog residuals ───────────────────────────────────────
    # Compute "before CTE" (joint r_hat, original positions) and
    # "after CTE" (joint r_hat, CTE-corrected positions) for proper comparison.
    apply_cte_to_solver(solver, image_names, cte_params, t_launch_yr,
                        filtered_spi=filtered_spi, subtract=True)
    bp3m_gaia_ids = set(int(g) for g in solver.star_id_to_idx.keys() if int(g) > 0)
    print("\n  Saving post-CTE full-catalog residuals...")
    _before_arrays = _compute_full_catalog_residuals_from_df(
        img_to_df, bp3m_gaia_ids, solver, image_names, r_hat)
    _after_arrays  = _apply_cte_to_residual_arrays(
        _before_arrays, image_names, cte_params, t_launch_yr, solver, filtered_spi)
    out_arrays = _after_arrays  # backwards compat for any downstream uses
    if _before_arrays:
        np.savez(output_dir_joint / 'detections_catalog_joint_r.npz', **_before_arrays)
        np.savez(output_dir_joint / 'detections_catalog_cte.npz', **_after_arrays)
        n_imgs  = sum(1 for k in _before_arrays if k.endswith('_X_c'))
        n_total = sum(len(v) for k, v in _before_arrays.items() if k.endswith('_X_c'))
        print(f"  Saved detections_catalog_joint_r.npz + detections_catalog_cte.npz: "
              f"{n_imgs} images, {n_total:,} detections")

    # ── Standard BP3M diagnostic plots ────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating standard BP3M diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_dir_joint,
                       plot_residuals=plot_residuals)
        except Exception as exc:
            print(f"  WARNING: standard plots failed — {exc}")

        before_npz = output_dir_joint / "detections_catalog_joint_r.npz"
        after_npz  = output_dir_joint / "detections_catalog_cte.npz"
        try:
            _plot_cte_diagnostics(
                output_dir_joint, cte_params,
                before_npz=before_npz,
                after_npz=after_npz,
                image_names=image_names,
                solver=solver,
            )
        except Exception as exc:
            print(f"  WARNING: CTE diagnostic plots failed — {exc}")

        try:
            _plot_joint_convergence(_gamma_hist, _mu_pop_hist, cte_template, output_dir_joint,
                                   C_mu_history=_C_mu_hist)
        except Exception as exc:
            print(f"  WARNING: convergence history plot failed — {exc}")

        # ── Per-image detector residual maps: v2 / post-joint-r / post-CTE ────
        if _ws_v2 is not None and _before_arrays and _after_arrays:
            try:
                _plot_per_image_detector_residuals(
                    output_dir_joint / 'plots' / 'residuals',
                    image_names, solver, filtered_spi,
                    arrays_stage1=_ws_v2,
                    arrays_stage2=_before_arrays,
                    arrays_stage3=_after_arrays,
                    stage_labels=("bp3m v2",
                                  "post-joint r/μ_pop",
                                  "post-CTE (final)"),
                    prefix='final',
                    img_meta=_img_meta,
                )
            except Exception as exc:
                print(f"  WARNING: final per-image residual maps failed — {exc}")

    print(f"\n  Joint CTE results written to: {output_dir_joint}")
    return output_dir_joint
