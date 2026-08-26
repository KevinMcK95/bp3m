"""
Tangent-plane geometry shared by every instrument.

All cross-matching happens on a gnomonic tangent plane in milliarcseconds about
an image's (ra0, dec0).  Adapters call gnomonic() to get there; nothing
downstream needs to know about sky coordinates again.
"""

from __future__ import annotations

import numpy as np

DEG2MAS = 3.6e6
MAS2DEG = 1.0 / DEG2MAS


def gnomonic(ra, dec, ra0, dec0):
    """
    Gnomonic (tangent-plane) projection.

    Parameters
    ----------
    ra, dec : array_like
        Sky coordinates [deg].
    ra0, dec0 : float
        Tangent point [deg].

    Returns
    -------
    xi, eta : ndarray
        Tangent-plane coordinates [mas].  xi increases with RA, eta with Dec.
    """
    r2d = np.pi / 180.0
    ra_r, dec_r = np.asarray(ra) * r2d, np.asarray(dec) * r2d
    ra0_r, dec0_r = ra0 * r2d, dec0 * r2d

    cos_dec, sin_dec = np.cos(dec_r), np.sin(dec_r)
    cos_dec0, sin_dec0 = np.cos(dec0_r), np.sin(dec0_r)
    cos_dra = np.cos(ra_r - ra0_r)

    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    xi = (cos_dec * np.sin(ra_r - ra0_r) / denom) / r2d * DEG2MAS
    eta = ((cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom) / r2d * DEG2MAS
    return xi, eta


def inverse_gnomonic(xi, eta, ra0, dec0):
    """Inverse of gnomonic(): tangent-plane [mas] -> sky [deg]."""
    r2d = np.pi / 180.0
    x = np.asarray(xi) * MAS2DEG * r2d
    y = np.asarray(eta) * MAS2DEG * r2d
    ra0_r, dec0_r = ra0 * r2d, dec0 * r2d

    rho = np.sqrt(x**2 + y**2)
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)
    # rho == 0 is the tangent point itself; avoid 0/0.
    safe_rho = np.where(rho > 0, rho, 1.0)

    dec = np.arcsin(np.where(rho > 0,
                             cos_c * np.sin(dec0_r) + y * sin_c * np.cos(dec0_r) / safe_rho,
                             np.sin(dec0_r)))
    ra = ra0_r + np.arctan2(x * sin_c,
                            rho * np.cos(dec0_r) * cos_c - y * np.sin(dec0_r) * sin_c)
    return np.degrees(ra) % 360.0, np.degrees(dec)


def covariance_from_sigmas(sigma_a, sigma_b, cov_ab=None, floor=0.0):
    """
    Build an (n, 2, 2) covariance from 1-sigma errors plus an optional
    off-diagonal, with a systematic floor added in quadrature on the diagonal.

    The floor is an independent additive error, so it goes on the diagonal only;
    the measured off-diagonal is preserved and clipped to keep the result
    positive-definite.

    All inputs and the result are in consistent units (mas and mas^2 here).
    """
    sigma_a = np.asarray(sigma_a, dtype=float)
    sigma_b = np.asarray(sigma_b, dtype=float)
    n = len(sigma_a)

    C = np.zeros((n, 2, 2))
    C[:, 0, 0] = sigma_a**2 + floor**2
    C[:, 1, 1] = sigma_b**2 + floor**2
    if cov_ab is not None:
        cov_ab = np.asarray(cov_ab, dtype=float)
        cov_ab = np.where(np.isfinite(cov_ab), cov_ab, 0.0)
        cov_max = 0.99 * np.sqrt(C[:, 0, 0] * C[:, 1, 1])
        C[:, 0, 1] = C[:, 1, 0] = np.clip(cov_ab, -cov_max, cov_max)
    return C


__all__ = ['gnomonic', 'inverse_gnomonic', 'covariance_from_sigmas', 'DEG2MAS', 'MAS2DEG']
