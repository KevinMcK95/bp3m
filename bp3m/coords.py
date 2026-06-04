"""
Coordinate transformation helpers: HST pixel → (RA, Dec).

Given the posterior image transformation r_hat and its covariance C_r from
BP3MSolver, convert arbitrary HST (X, Y) positions in a given image to
equatorial coordinates, propagating both the r-vector uncertainty and (optionally)
the HST measurement noise into the output (RA, Dec) uncertainty.

Main public functions
---------------------
plane_project_inverse(x, y, ra0, dec0, pixel_scale)
    Inverse gnomonic projection: pseudo-image pixels → (RA, Dec) degrees.

hst_to_radec(X, Y, img_name, solver, r_hat, C_r,
             x_hst_err=None, y_hst_err=None, xy_hst_corr=None)
    Convert HST pixel coordinates to (RA, Dec) with full uncertainty budget.
"""

import numpy as np
from .astro_utils import (
    plane_project_jacobian,
    build_X_matrix,
    compute_poly_jacobian,
    hst_position_cov,
    DEG2RAD, RAD2MAS,
)


# ── Inverse gnomonic projection ───────────────────────────────────────────────

def plane_project_inverse(x, y, ra0, dec0, pixel_scale):
    """
    Inverse of plane_project: pseudo-image pixel coordinates → (RA, Dec).

    Uses the standard gnomonic (TAN) inverse formulae. The sign convention
    matches plane_project: x increases to the West (−RA direction),
    y increases to the North (+Dec direction).

    Parameters
    ----------
    x, y       : float or array  [pseudo-image pixels]
    ra0, dec0  : float  [degrees]  tangent-point (image pointing)
    pixel_scale: float  [mas/pixel]

    Returns
    -------
    ra, dec : float or array  [degrees]
    """
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))

    # Convert pixels to dimensionless gnomonic plate coordinates (radians)
    # ξ  = -x * pscale / RAD2MAS  (West convention → negative RA direction)
    # η  =  y * pscale / RAD2MAS
    xi  = -x * pixel_scale / RAD2MAS   # radians
    eta =  y * pixel_scale / RAD2MAS   # radians

    dec0_r = dec0 * DEG2RAD
    ra0_r  = ra0  * DEG2RAD

    p = np.sqrt(xi**2 + eta**2)   # angular distance from tangent point (tan of angle)

    # Gnomonic inverse (Calabretta & Greisen 2002)
    # For p == 0: ra = ra0, dec = dec0
    with np.errstate(invalid='ignore', divide='ignore'):
        dec_r = np.where(
            p < 1e-15,
            dec0_r,
            np.arctan2(
                np.cos(np.arctan(p)) * np.sin(dec0_r) + eta * np.sin(np.arctan(p)) * np.cos(dec0_r) / p,
                np.sqrt(
                    (np.cos(np.arctan(p)) * np.cos(dec0_r) - eta * np.sin(np.arctan(p)) * np.sin(dec0_r) / p)**2
                    + (xi * np.sin(np.arctan(p)) / p)**2
                )
            )
        )
        ra_r = np.where(
            p < 1e-15,
            ra0_r,
            ra0_r + np.arctan2(
                xi * np.sin(np.arctan(p)) / p,
                np.cos(np.arctan(p)) * np.cos(dec0_r)
                - eta * np.sin(np.arctan(p)) * np.sin(dec0_r) / p
            )
        )

    ra  = ra_r  / DEG2RAD
    dec = dec_r / DEG2RAD

    if scalar:
        return float(ra[0]), float(dec[0])
    return ra, dec


# ── Main coordinate transform with uncertainty ────────────────────────────────

def hst_to_radec(X, Y, img_name, solver, r_hat, C_r,
                 x_hst_err=None, y_hst_err=None, xy_hst_corr=None):
    """
    Convert HST detector pixel positions to (RA, Dec) using the posterior
    image transformation, with full uncertainty propagation.

    The model:
        x_gaia = X_mat(X_c, Y_c) @ r_j          (pseudo-image pixels)
        (ra, dec) = plane_project_inverse(x_gaia, ra0, dec0, pscale)

    Uncertainty budget (added in quadrature in pseudo-image pixel space):
        C_xy = X_mat @ C_r_j @ X_mat.T           (r-vector uncertainty)
             + R_j @ C_hst @ R_j.T               (HST measurement noise, optional)
        C_radec = J^{-1} @ C_xy @ J^{-T}         (propagated to mas²)

    where J = plane_project_jacobian [pix/mas], so J^{-1} is [mas/pix].

    Parameters
    ----------
    X, Y         : float or array  [raw HST pixels]
    img_name     : str  (must be in solver.image_names)
    solver       : fitted BP3MSolver instance
    r_hat        : (n_r,) posterior image transformation vector
    C_r          : (n_r, n_r) posterior covariance of r
    x_hst_err    : float or array [HST pixels], optional
        1-sigma uncertainty in X.  Pass None to omit HST noise.
    y_hst_err    : float or array [HST pixels], optional
        1-sigma uncertainty in Y.
    xy_hst_corr  : float or array, optional
        Correlation between X and Y errors (default 0).

    Returns
    -------
    ra           : float or array  [degrees]
    dec          : float or array  [degrees]
    sigma_ra_star: float or array  [mas]  uncertainty in α·cos(δ)
    sigma_dec    : float or array  [mas]  uncertainty in δ
    cov_radec    : (…, 2, 2) ndarray  [mas²]  full covariance matrix
                   [[C_αα, C_αδ], [C_δα, C_δδ]]
    """
    j_idx     = solver.image_names.index(img_name)
    meta      = solver.images[img_name]
    nr        = solver.N_R
    poly_order = solver.poly_order

    ra0    = meta["ra0"]
    dec0   = meta["dec0"]
    pscale = meta["orig_pixel_scale"]   # mas/pix
    Xo = Yo = 2048.0

    # ── Centered HST positions ────────────────────────────────────────────────
    scalar = np.ndim(X) == 0
    X = np.atleast_1d(np.asarray(X, float))
    Y = np.atleast_1d(np.asarray(Y, float))
    n = len(X)

    X_c = X - Xo
    Y_c = Y - Yo

    # ── Extract r_j and its covariance block ──────────────────────────────────
    cs    = j_idx * nr
    r_j   = r_hat[cs:cs + nr]
    C_r_j = C_r[cs:cs + nr, cs:cs + nr]

    # ── Build X_mat for each input point (tangent-point derivatives = 0 here) ─
    X_mats = np.zeros((n, 2, nr))
    for k in range(n):
        X_mats[k] = build_X_matrix(X_c[k], Y_c[k], 0., 0., 0., 0.,
                                   poly_order=poly_order)

    # ── Predicted pseudo-image position: x_gaia = X_mat @ r_j ────────────────
    x_gaia = np.einsum('nkl,l->nk', X_mats, r_j)   # (n, 2): [x_pix, y_pix]

    # ── Convert pseudo-image position to (RA, Dec) ────────────────────────────
    ra, dec = plane_project_inverse(x_gaia[:, 0], x_gaia[:, 1], ra0, dec0, pscale)

    # ── Propagate uncertainties ───────────────────────────────────────────────
    J     = plane_project_jacobian(ra, dec, ra0, dec0, pscale)   # (n, 2, 2)
    J_inv = np.linalg.inv(J)

    # C_xy from r-vector uncertainty: X_mat @ C_r_j @ X_mat.T  (n, 2, 2) pix²
    XCr  = np.einsum('nij,jk->nik', X_mats, C_r_j)
    C_xy = np.einsum('nik,njk->nij', XCr, X_mats)

    # Optionally add HST measurement noise: J_k @ C_hst_k @ J_k^T
    # where J_k is the position-dependent transformation Jacobian.
    if x_hst_err is not None:
        x_hst_err  = np.broadcast_to(np.asarray(x_hst_err,  float), (n,))
        y_hst_err  = np.broadcast_to(np.asarray(y_hst_err,  float), (n,))
        if xy_hst_corr is None:
            xy_hst_corr = np.zeros(n)
        else:
            xy_hst_corr = np.broadcast_to(np.asarray(xy_hst_corr, float), (n,))

        # Full position-dependent Jacobian J_trans (reduces to R=[[a,b],[c,d]] for poly_order=1)
        J_trans = compute_poly_jacobian(r_j, X_c, Y_c, poly_order)   # (n, 2, 2)

        for k in range(n):
            C_hst_k = hst_position_cov(x_hst_err[k], y_hst_err[k], xy_hst_corr[k])
            C_xy[k] += J_trans[k] @ C_hst_k @ J_trans[k].T

    # Propagate C_xy → C_radec: J^{-1} @ C_xy @ J^{-T}  [mas²]
    JiCxy    = np.einsum('nij,njk->nik', J_inv, C_xy)           # (n, 2, 2)
    cov_radec = np.einsum('nij,nkj->nik', JiCxy, J_inv)          # (n, 2, 2) mas²

    sigma_ra_star = np.sqrt(np.maximum(cov_radec[:, 0, 0], 0.))  # (n,) mas
    sigma_dec     = np.sqrt(np.maximum(cov_radec[:, 1, 1], 0.))  # (n,) mas

    if scalar:
        return (float(ra[0]), float(dec[0]),
                float(sigma_ra_star[0]), float(sigma_dec[0]),
                cov_radec[0])
    return ra, dec, sigma_ra_star, sigma_dec, cov_radec
