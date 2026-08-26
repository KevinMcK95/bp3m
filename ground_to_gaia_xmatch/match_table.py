"""
Match-table and transformation-table schema — one definition for all instruments.

Two files per image, both written here and read here so the writer and reader
cannot disagree:

  matched_gaia.csv    one row per matched (source, Gaia) pair
  transformation.csv  the fitted transform, long format (parameter,value)

On the covariance columns
-------------------------
`src_cov_xx/yy/xy` are the raw MEASUREMENT covariance of the source, in mas^2,
and nothing else.  `sigma_xi_mas`/`sigma_eta_mas` are the TOTAL match covariance
(Gaia + projected source + model + residual floor) and exist only for
diagnostics and the sigma cut.

The alignment solver must read `src_cov_*`.  Reading `sigma_*` instead
double-counts the Gaia covariance that the solver already applies as its prior —
for 2p stars, whose propagated Gaia position covariance is ~1000 mas, that made
the measurement look ~75x worse than it is and collapsed their proper motions
onto the flat prior.  Keep these columns distinct.

On transformation.csv
---------------------
Historically CFHT wrote a wide table pivoted on extension while LSST wrote
long-format parameter/value rows, so the two align scripts needed different
readers.  The canonical per-image format here is LONG (parameter,value):
self-describing and extensible without breaking readers.  The exposure-level
roll-up is WIDE (one row per detector) because that is the convenient shape for
analysis.  `read_transformation` accepts either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns written for every matched pair, in order.
MATCH_COLUMNS = [
    'src_index', 'gaia_source_id',
    'src_x', 'src_y', 'src_ra', 'src_dec',
    'src_xi_mas', 'src_eta_mas',
    'src_mag', 'src_magerr', 'src_is_star',
    # Raw measurement covariance [mas^2] — the solver reads THESE.
    'src_cov_xx', 'src_cov_yy', 'src_cov_xy',
    'has_gaia_pms', 'gaia_is_clean',
    'gaia_xi_mas', 'gaia_eta_mas', 'gaia_ra_prop', 'gaia_dec_prop',
    'gaia_gmag', 'gaia_bprp', 'gaia_pmra', 'gaia_pmdec',
    'residual_xi_mas', 'residual_eta_mas',
    # Total match covariance [mas] — diagnostics only, NOT the source error.
    'sigma_xi_mas', 'sigma_eta_mas',
    'residual_sigma', 'residual_mag', 'zp',
]


def build_match_table(cat, gaia, rows, zp) -> pd.DataFrame:
    """
    Assemble the match table for one image.

    `rows` is the scored/deduplicated DataFrame from refinement.final_pass,
    carrying columns h, g, s, dx, dy, md, cxx, cyy.
    """
    h = rows['h'].values
    g = rows['g'].values

    def _gx(name, default=np.nan):
        arr = gaia.extra.get(name)
        return arr[g] if arr is not None else np.full(len(g), default)

    df = pd.DataFrame({
        'src_index': h,
        'gaia_source_id': gaia.extra['source_id'][g].astype(np.int64),
        'src_x': cat.x[h], 'src_y': cat.y[h],
        'src_ra': cat.ra[h], 'src_dec': cat.dec[h],
        'src_xi_mas': cat.xi[h], 'src_eta_mas': cat.eta[h],
        'src_mag': cat.mag[h], 'src_magerr': cat.magerr[h],
        'src_is_star': cat.is_star[h],
        'src_cov_xx': cat.C_src[h, 0, 0],
        'src_cov_yy': cat.C_src[h, 1, 1],
        'src_cov_xy': cat.C_src[h, 0, 1],
        'has_gaia_pms': gaia.has_pms[g],
        'gaia_is_clean': gaia.clean[g],
        'gaia_xi_mas': gaia.xi[g], 'gaia_eta_mas': gaia.eta[g],
        'gaia_ra_prop': _gx('ra_prop'), 'gaia_dec_prop': _gx('dec_prop'),
        'gaia_gmag': gaia.mag[g], 'gaia_bprp': _gx('bp_rp'),
        'gaia_pmra': _gx('pmra'), 'gaia_pmdec': _gx('pmdec'),
        'residual_xi_mas': rows['dx'].values,
        'residual_eta_mas': rows['dy'].values,
        'sigma_xi_mas': np.sqrt(rows['cxx'].values),
        'sigma_eta_mas': np.sqrt(rows['cyy'].values),
        'residual_sigma': rows['s'].values,
        'residual_mag': rows['md'].values - zp,
        'zp': zp,
    })
    return df[MATCH_COLUMNS]


def check_one_to_one(df: pd.DataFrame, label: str = '') -> None:
    """
    Assert strict one-to-one matching.  Cheap, and catches the class of bug
    that silently double-weights stars in the alignment likelihood.
    """
    n_g = df['gaia_source_id'].duplicated().sum()
    n_h = df['src_index'].duplicated().sum()
    if n_g or n_h:
        raise ValueError(
            f'{label}: match table is not one-to-one — '
            f'{n_g} duplicate Gaia ids, {n_h} duplicate source indices')


# ── transformation.csv ───────────────────────────────────────────────────────

def transformation_dict(meta, fit, best, n_matches, resid_xi, resid_eta) -> dict:
    """Flat parameter dict for one image's fitted transform."""
    A, B, C, D = fit['A'], fit['B'], fit['C'], fit['D']
    out = {
        'A': A, 'B': B, 'C': C, 'D': D,
        'xs_o': fit['xs_o'], 'ys_o': fit['ys_o'],
        'xt_o': fit['xt_o'], 'yt_o': fit['yt_o'],
        'scale': np.sqrt(A * D - B * C),
        'rotation_deg': np.degrees(np.arctan2(B - C, A + D)),
        'on_skew': 0.5 * (A - D),
        'off_skew': 0.5 * (B + C),
        'zp': fit['zp'],
        'ra0': meta.ra0, 'dec0': meta.dec0, 'mjd': meta.mjd,
        'pixel_scale': meta.pixel_scale,
        'instrument': meta.instrument,
        'exposure': meta.exposure,
        'detector': meta.detector,
        'band': meta.band if meta.band is not None else '',
        'image_id': meta.image_id,
        'resid_xi_rms_mas': resid_xi,
        'resid_eta_rms_mas': resid_eta,
        'init_resid_xi_mas': fit['init_resid_xi'],
        'init_resid_eta_mas': fit['init_resid_eta'],
        'disc_tier': best.get('tier', ''),
        'disc_gaia_tier': best.get('gaia_tier', ''),
        'n_matches': n_matches,
    }
    out.update(meta.key)
    return out


def write_transformation(path: Path, params: dict) -> None:
    """Write one image's transform in long (parameter,value) format."""
    pd.DataFrame(list(params.items()), columns=['parameter', 'value']) \
      .to_csv(path, index=False)


def read_transformation(path: Path) -> dict:
    """
    Read a transformation table, accepting either the canonical long format or
    a legacy single-row wide table.
    """
    df = pd.read_csv(path)
    if set(df.columns) >= {'parameter', 'value'}:
        return dict(zip(df['parameter'], df['value']))
    if len(df) == 1:
        return df.iloc[0].to_dict()
    raise ValueError(f'{path}: unrecognised transformation table format '
                     f'(columns={list(df.columns)}, {len(df)} rows)')


def write_all_transformations(path: Path, params_list: list[dict]) -> None:
    """Exposure-level roll-up: wide, one row per detector."""
    pd.DataFrame(params_list).to_csv(path, index=False)


__all__ = ['MATCH_COLUMNS', 'build_match_table', 'check_one_to_one',
           'transformation_dict', 'write_transformation', 'read_transformation',
           'write_all_transformations']
