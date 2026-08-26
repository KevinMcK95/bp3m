"""
Build the per-image Gaia reference field.

Propagates Gaia to the image epoch, projects onto the image tangent plane, and
trims to the detector footprint.

The one rule that matters here: nothing in this module removes a Gaia source on
quality grounds.  `clean_label` is carried through as a boolean column on the
field and used downstream ONLY to seed discovery tiers.  It is RUWE-derived, so
every 2p source has clean_label=False; filtering on it deletes the entire 2p
population before matching begins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaia_cross_match.cross_match import propagate_gaia_with_cov

from .discovery import GaiaField
from .geometry import gnomonic
from .instruments.base import ImageMeta


def clean_mask(gaia_df: pd.DataFrame) -> np.ndarray:
    """
    RUWE-based quality flag, as a mask.

    Sources without RUWE (all 2p) are reported as False here — matching the
    stored clean_label — which is precisely why this must never be used as a
    filter.
    """
    if 'clean_label' in gaia_df.columns:
        return gaia_df['clean_label'].values.astype(bool)
    if 'ruwe' in gaia_df.columns:
        ruwe = pd.to_numeric(gaia_df['ruwe'], errors='coerce').values
        return np.isfinite(ruwe) & (ruwe <= 1.4)
    return np.ones(len(gaia_df), dtype=bool)


def build_gaia_field(gaia_df: pd.DataFrame, meta: ImageMeta,
                     xi_src=None, eta_src=None, margin=3500.0) -> GaiaField:
    """
    Propagate, project, and trim Gaia for one image.

    Parameters
    ----------
    gaia_df : DataFrame
        FULL Gaia catalogue for the field — not quality-filtered.
    meta : ImageMeta
        Supplies the epoch and tangent point.
    xi_src, eta_src : ndarray, optional
        Source positions [mas]; when given, Gaia is trimmed to their bounding
        box plus `margin`.
    margin : float
        Footprint padding [mas].
    """
    ra_prop, dec_prop, Ct = propagate_gaia_with_cov(gaia_df, meta.mjd)
    C = Ct[:, 0:2, 0:2].copy()
    err = np.power(np.maximum(np.linalg.det(C), 1e-30), 0.25)
    xi, eta = gnomonic(ra_prop, dec_prop, meta.ra0, meta.dec0)

    keep = np.ones(len(gaia_df), dtype=bool)
    if xi_src is not None and len(xi_src):
        keep = ((xi >= xi_src.min() - margin) & (xi <= xi_src.max() + margin)
                & (eta >= eta_src.min() - margin) & (eta <= eta_src.max() + margin))

    def col(name, default=np.nan):
        if name in gaia_df.columns:
            return pd.to_numeric(gaia_df[name], errors='coerce').to_numpy(float)[keep]
        return np.full(int(keep.sum()), default)

    return GaiaField(
        xi=xi[keep], eta=eta[keep], C=C[keep], err=err[keep],
        mag=col('gmag', 20.0),
        has_pms=np.isfinite(gaia_df['pmra'].values)[keep],
        clean=clean_mask(gaia_df)[keep],
        extra={
            'source_id': gaia_df['source_id'].to_numpy(dtype=np.int64)[keep],
            'ra_prop': ra_prop[keep], 'dec_prop': dec_prop[keep],
            'pmra': col('pmra'), 'pmdec': col('pmdec'), 'bp_rp': col('bp_rp'),
            'index': np.where(keep)[0],
        },
    )


__all__ = ['build_gaia_field', 'clean_mask']
