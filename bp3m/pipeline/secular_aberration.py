"""Secular (Galactic) aberration apparent proper motion for extragalactic sources.

The acceleration of the Solar System Barycentre (SSB) toward the Galactic
centre produces an apparent proper motion of all extragalactic sources
(QSOs, background galaxies) of ~5 µas/yr directed toward the projection of
the Galactic centre on the sky.  This is a systematic offset that should be
used as the PM prior mean for QSO anchors instead of zero.

Derivation: the aberration vector A = κ × n̂_GC is projected perpendicular
to the line of sight n̂_source:

    μ_α cos δ = (A · ê_α)  =  κ cos(δ_GC) sin(α_GC − α)
    μ_δ       = (A · ê_δ)  =  κ [−sin δ cos δ_GC cos(α_GC − α) + cos δ sin δ_GC]

Reference: Klioner et al. 2021, A&A 649, A9 (Gaia EDR3 frame tie); κ = 5.05 µas/yr.
"""

from __future__ import annotations

import numpy as np

# Galactic centre direction in ICRS (J2000) equatorial coordinates (IAU).
_RA_GC_DEG  = 266.40499  # degrees
_DEC_GC_DEG = -28.93617  # degrees

# Secular aberration amplitude: Klioner et al. 2021 (Gaia EDR3).
# Alternative well-measured values: MacMillan et al. 2019 (VLBI) = 5.28 µas/yr.
KAPPA_DEFAULT_UAS_YR = 5.05  # µas/yr


def secular_aberration_pm(
    ra: "np.ndarray | float",
    dec: "np.ndarray | float",
    kappa_uas_yr: float = KAPPA_DEFAULT_UAS_YR,
) -> "tuple[np.ndarray, np.ndarray]":
    """Secular aberration apparent proper motion in µas/yr.

    Returns the predicted (pmra_cos_dec, pmdec) in µas/yr for extragalactic
    sources at the given sky positions.  This should replace the default
    zero-PM prior for QSO anchors.

    Parameters
    ----------
    ra, dec : float or array-like
        Source coordinates in degrees (ICRS).
    kappa_uas_yr : float
        Secular aberration amplitude in µas/yr (default 5.05, Klioner 2021).

    Returns
    -------
    pmra_cos_dec : ndarray
        RA proper motion × cos(dec), in µas/yr.
    pmdec : ndarray
        Dec proper motion in µas/yr.
    """
    ra_rad  = np.deg2rad(np.asarray(ra,  dtype=float))
    dec_rad = np.deg2rad(np.asarray(dec, dtype=float))

    a_gc = np.deg2rad(_RA_GC_DEG)
    d_gc = np.deg2rad(_DEC_GC_DEG)

    pmra_cos_dec = kappa_uas_yr * np.cos(d_gc) * np.sin(a_gc - ra_rad)

    pmdec = kappa_uas_yr * (
        -np.sin(dec_rad) * np.cos(d_gc) * np.cos(a_gc - ra_rad)
        + np.cos(dec_rad) * np.sin(d_gc)
    )

    return pmra_cos_dec, pmdec
