"""
Astrometric utility functions: plane projection, parallax factors, Jacobians.

All angles in degrees unless noted. Positions in mas, pixel scales in mas/pixel.
rad2mas = 180 * 3600 * 1000 / pi
"""

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, ICRS, SkyCoord
import astropy.units as u

RAD2MAS = 180.0 * 3600.0 * 1000.0 / np.pi  # radians -> milliarcseconds
DEG2RAD = np.pi / 180.0

#account for systematics in Gaia data
#amount to inflate uncertainties by
#might want to change to function of magnitude in the future
GAIA_SYS_DICT = {
    'mult_6p':1.22,
    'mult_5p':1.05,
    'mult_2p':1.00,
    'parallax_sys_err':0.011, #mas, from E. Vasiliev and H. Baumgardt 2021, MNRAS 505, 5978–6002
    'pm_sys_err':0.026, #mas/yr, from E. Vasiliev and H. Baumgardt 2021, MNRAS 505, 5978–6002
}

def michalik_sigma_plx_prior(ra_deg, dec_deg, g_mag):
    """
    Michalik et al. (2015) magnitude- and direction-dependent parallax prior width.

    Returns the 1-sigma prior on parallax in mas:
        sigma_plx_prior = 10 * sigma_plx_F90(l, b, G)
    where sigma_plx_F90 is the 90th-percentile parallax at that magnitude and
    direction, following Eqs. 15-18 of Michalik et al. (2015, A&A 583, A68):
        log10(sigma_F90) = s0 + s1*|sin b| + s2*cos(b)*cos(l)
        s0(G) = 2.187 - 0.2547*G + 0.006382*G^2
        s1(G) = 0.114 - 0.0579*G + 0.01369*G^2 - 0.000506*G^3
        s2(G) = 0.031 - 0.0062*G
    The model is defined for G in [6, 20]; G is capped at 20 before evaluation.

    Parameters
    ----------
    ra_deg, dec_deg : float or array-like
        ICRS coordinates in degrees.
    g_mag : float or array-like
        Gaia G magnitude (or proxy). Values above 20 are clamped to 20.

    Returns
    -------
    sigma_plx_prior : float or ndarray, mas
        1-sigma prior width = 10 * sigma_F90.
    """
    ra_arr  = np.atleast_1d(np.asarray(ra_deg,  dtype=float))
    dec_arr = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    g_arr   = np.atleast_1d(np.asarray(g_mag,   dtype=float))
    g_arr   = np.clip(g_arr, 6.0, 20.0)

    sc = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame='icrs')
    gc = sc.galactic
    l  = gc.l.rad
    b  = gc.b.rad

    s0 =  2.187  - 0.2547  * g_arr + 0.006382 * g_arr**2
    s1 =  0.114  - 0.0579  * g_arr + 0.01369  * g_arr**2 - 0.000506 * g_arr**3
    s2 =  0.031  - 0.0062  * g_arr

    log_sig = s0 + s1 * np.abs(np.sin(b)) + s2 * np.cos(b) * np.cos(l)
    sigma_prior = 10.0 * 10.0**log_sig   # mas

    if sigma_prior.size == 1:
        return float(sigma_prior[0])
    return sigma_prior


def plane_project(ra, dec, ra0, dec0, pixel_scale):
    """
    Gnomonic (tangent-plane) projection of (ra, dec) onto a pseudo-image
    centered at (ra0, dec0) with the given pixel scale (mas/pixel).

    Returns (x, y) in pixels, with x along -RA and y along +Dec.

    Parameters
    ----------
    ra, dec : float or array, degrees
    ra0, dec0 : float, degrees (tangent point / image center)
    pixel_scale : float, mas/pixel

    Returns
    -------
    x, y : float or array, pixels
    """
    ra_rad = ra * DEG2RAD
    dec_rad = dec * DEG2RAD
    ra0_rad = ra0 * DEG2RAD
    dec0_rad = dec0 * DEG2RAD

    dra_rad = ra_rad - ra0_rad
    r2 = (np.sin(dec0_rad) * np.sin(dec_rad)
          + np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    x = -RAD2MAS * np.cos(dec_rad) * np.sin(dra_rad) / (pixel_scale * r2)
    y = (RAD2MAS * (np.cos(dec0_rad) * np.sin(dec_rad)
                    - np.sin(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))
         / (pixel_scale * r2))
    return x, y


def plane_project_jacobian(ra, dec, ra0, dec0, pixel_scale):
    """
    2×2 Jacobian of plane_project w.r.t. (α*, δ) in mas.
    J[0,0] = dx/d(α*_mas), J[0,1] = dx/d(δ_mas), etc.

    (α* = α·cos(δ), so d/dα* = (1/cos(δ)) d/dα)
    Returns shape (..., 2, 2)
    """
    ra_rad = ra * DEG2RAD
    dec_rad = dec * DEG2RAD
    ra0_rad = ra0 * DEG2RAD
    dec0_rad = dec0 * DEG2RAD
    dra_rad = ra_rad - ra0_rad

    r2 = (np.sin(dec0_rad) * np.sin(dec_rad)
          + np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    # Convert pixel coordinates to mas then express derivatives in mas^{-1} * pix
    # x = -RAD2MAS * cos(dec) * sin(dra) / (pix_scale * r2)
    # y = RAD2MAS * (cos(dec0)*sin(dec) - sin(dec0)*cos(dec)*cos(dra)) / (pix_scale * r2)

    # dr2/d(ra) = -cos(dec0)*cos(dec)*sin(dra)
    # dr2/d(dec) =  cos(dec0)*cos(dec)*sin(dec0)*... actually:
    # r2 = sin(dec0)*sin(dec) + cos(dec0)*cos(dec)*cos(dra)
    # dr2/d(dec) = sin(dec0)*cos(dec) - cos(dec0)*sin(dec)*cos(dra)
    dr2_dra  = -np.cos(dec0_rad) * np.cos(dec_rad) * np.sin(dra_rad)
    dr2_ddec = (np.sin(dec0_rad) * np.cos(dec_rad)
                - np.cos(dec0_rad) * np.sin(dec_rad) * np.cos(dra_rad))

    # dx/dra  (radians)
    # x = -C * cos(dec)*sin(dra) / r2   where C = RAD2MAS/pix_scale
    C = RAD2MAS / pixel_scale
    Ndx = -np.cos(dec_rad) * np.sin(dra_rad)   # numerator of x/C
    Ndy_num = (np.cos(dec0_rad) * np.sin(dec_rad)
               - np.sin(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    # d(Ndx)/dra = -cos(dec)*cos(dra)
    dNdx_dra  = -np.cos(dec_rad) * np.cos(dra_rad)
    # d(Ndx)/ddec = sin(dec)*sin(dra)
    dNdx_ddec =  np.sin(dec_rad) * np.sin(dra_rad)

    # d(Ndy)/dra = sin(dec0)*cos(dec)*sin(dra)
    dNdy_dra  =  np.sin(dec0_rad) * np.cos(dec_rad) * np.sin(dra_rad)
    # d(Ndy)/ddec = cos(dec0)*cos(dec) + sin(dec0)*sin(dec)*cos(dra)
    dNdy_ddec = (np.cos(dec0_rad) * np.cos(dec_rad)
                 + np.sin(dec0_rad) * np.sin(dec_rad) * np.cos(dra_rad))

    r2_sq = r2 ** 2
    # dx/dra (pix/rad), dy/dra (pix/rad)
    dx_dra  = C * (dNdx_dra  * r2 - Ndx  * dr2_dra)  / r2_sq
    dx_ddec = C * (dNdx_ddec * r2 - Ndx  * dr2_ddec) / r2_sq
    dy_dra  = C * (dNdy_dra  * r2 - Ndy_num * dr2_dra)  / r2_sq
    dy_ddec = C * (dNdy_ddec * r2 - Ndy_num * dr2_ddec) / r2_sq

    # Convert from pix/rad to pix/mas: divide by RAD2MAS
    # Also convert from d/dα to d/dα*: multiply by 1/cos(dec) for x
    # J[:,0] = d/d(α*_mas) = (1/cos(dec)) * d/dα_rad / RAD2MAS
    cos_dec = np.cos(dec_rad)
    J00 = dx_dra  / (cos_dec * RAD2MAS)   # dx/d(α*_mas)   [pix/mas]
    J01 = dx_ddec / RAD2MAS                # dx/d(δ_mas)    [pix/mas]
    J10 = dy_dra  / (cos_dec * RAD2MAS)   # dy/d(α*_mas)   [pix/mas]
    J11 = dy_ddec / RAD2MAS               # dy/d(δ_mas)    [pix/mas]

    # Stack: shape (..., 2, 2)
    J = np.stack([np.stack([J00, J01], axis=-1),
                  np.stack([J10, J11], axis=-1)], axis=-2)
    return J


def plane_project_tangent_derivs(ra, dec, ra0, dec0, pixel_scale):
    """
    Derivatives of plane_project w.r.t. the tangent point (ra0, dec0),
    returned in units of pix/mas (since ra0, dec0 offsets are in degrees,
    we convert).

    Returns (dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0) each shape like ra.
    Units: pixels per mas of tangent-point shift.
    """
    ra_rad = ra * DEG2RAD
    dec_rad = dec * DEG2RAD
    ra0_rad = ra0 * DEG2RAD
    dec0_rad = dec0 * DEG2RAD
    dra_rad = ra_rad - ra0_rad

    r2 = (np.sin(dec0_rad) * np.sin(dec_rad)
          + np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    C = RAD2MAS / pixel_scale
    Ndx = -np.cos(dec_rad) * np.sin(dra_rad)
    Ndy = (np.cos(dec0_rad) * np.sin(dec_rad)
           - np.sin(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    r2_sq = r2 ** 2

    # dr2/dra0 = cos(dec0)*cos(dec)*sin(dra)  (opposite sign from dr2/dra)
    dr2_dra0  =  np.cos(dec0_rad) * np.cos(dec_rad) * np.sin(dra_rad)
    # dr2/ddec0 = cos(dec0)*sin(dec) - sin(dec0)*cos(dec)*cos(dra)
    dr2_ddec0 = (np.cos(dec0_rad) * np.sin(dec_rad)
                 - np.sin(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    # dNdx/dra0 = -cos(dec)*cos(dra)  ... wait, dra = ra-ra0, so d(dra)/dra0 = -1
    # d(Ndx)/dra0 = d[-cos(dec)*sin(dra)]/dra0 = cos(dec)*cos(dra)  (chain rule: -cos(dra)*(-1))
    dNdx_dra0  =  np.cos(dec_rad) * np.cos(dra_rad)
    # dNdx/ddec0 = 0 (Ndx doesn't depend on dec0)
    dNdx_ddec0 = np.zeros_like(ra_rad)

    # dNdy/dra0: d[cos(dec0)*sin(dec) - sin(dec0)*cos(dec)*cos(dra)]/dra0
    #   = -sin(dec0)*cos(dec)*sin(dra)*(-1) = sin(dec0)*cos(dec)*sin(dra) ... wait
    #   Actually d(dra)/dra0 = -1, so:
    #   dNdy/dra0 = -sin(dec0)*cos(dec)*(-sin(dra))*(-1) = -sin(dec0)*cos(dec)*sin(dra)
    dNdy_dra0  = -np.sin(dec0_rad) * np.cos(dec_rad) * np.sin(dra_rad)
    # dNdy/ddec0 = -sin(dec0)*sin(dec) - cos(dec0)*cos(dec)*cos(dra)
    dNdy_ddec0 = (-np.sin(dec0_rad) * np.sin(dec_rad)
                  - np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(dra_rad))

    dx_dra0  = C * (dNdx_dra0  * r2 - Ndx * dr2_dra0)  / r2_sq
    dx_ddec0 = C * (dNdx_ddec0 * r2 - Ndx * dr2_ddec0) / r2_sq
    dy_dra0  = C * (dNdy_dra0  * r2 - Ndy * dr2_dra0)  / r2_sq
    dy_ddec0 = C * (dNdy_ddec0 * r2 - Ndy * dr2_ddec0) / r2_sq

    # Convert from pix/rad to pix/mas
    scale = 1.0 / RAD2MAS

    # #turn off RA0,Dec0 fitting for now
    # scale = 0
    return (dx_dra0 * scale, dx_ddec0 * scale,
            dy_dra0 * scale, dy_ddec0 * scale)


def get_tele_position(time,curr_id='hst'):
    """
    Uses astropy to get telescopes's position relative to the Sun.

    Returns (X_au,Y_au,Z_au) at the provided time
    """
    # telescopes's position relative to barycentre in AU (ICRS)
    earth = get_body_barycentric(curr_id, time)
    X_au = earth.x.to(u.au).value
    Y_au = earth.y.to(u.au).value
    Z_au = earth.z.to(u.au).value

    return np.array([X_au,Y_au,Z_au])

def get_parallax_factors(ra_deg, dec_deg, tele_xyz):
    """
    Compute the parallax displacement factors (plxα*, plxδ) for a source at
    (ra, dec) at time t (MJD). These give the displacement in (α*, δ) that a
    source with parallax = 1 mas would have due to annual parallax.

    Returns (plx_ra_star, plx_dec) both in mas per mas-of-parallax.
    """
    # telescopes's position relative to barycentre in AU (ICRS)
    X_au,Y_au,Z_au = tele_xyz

    ra_rad = ra_deg * DEG2RAD
    dec_rad = dec_deg * DEG2RAD

    # Standard parallax factor formula (apparent - barycentric displacement per
    # unit parallax, verified against direct vector geometry):
    #   plx_ra* = +(X_au * sin(ra) - Y_au * cos(ra))
    #   plx_dec = +(X_au * cos(ra)*sin(dec) + Y_au * sin(ra)*sin(dec) - Z_au * cos(dec))
    # NOTE: before 2026-09 these were negated, which sign-flipped the parallax
    # column of the U design matrix everywhere (solver, v2, xmatch Phase 6).
    plx_ra_star = X_au * np.sin(ra_rad) - Y_au * np.cos(ra_rad)
    plx_dec = (X_au * np.cos(ra_rad) * np.sin(dec_rad)
               + Y_au * np.sin(ra_rad) * np.sin(dec_rad)
               - Z_au * np.cos(dec_rad))
    return plx_ra_star, plx_dec


def epoch_distortion_n_shape(order):
    """Number of scalar shape functions for the epoch-distortion basis."""
    return sum(deg + 1 for deg in range(2, order + 1))


def epoch_distortion_pairs(order):
    """(i, j) Legendre-degree pairs, total degree 2..order, in fixed order."""
    return [(deg - j, j) for deg in range(2, order + 1) for j in range(deg + 1)]


def epoch_distortion_basis(X_c, Y_c, order=3, half_x=2048.0, half_y=1024.0):
    """
    Shared epoch-distortion basis: products of Legendre polynomials
    P_i(x)P_j(y) with total degree i+j in [2, order], evaluated on
    chip-normalised coordinates (X_c/half_x, Y_c/half_y).

    Degrees 0 and 1 are deliberately EXCLUDED: constant and linear terms are
    exactly degenerate with coherent shifts of the per-image alignment
    parameters (a,b,c,d,dRA0,dDec0). Legendre products are used instead of raw
    monomials so the retained shapes are orthogonal to {1, x, y} over the
    uniform chip domain — near-zero statistical coupling to the alignment
    under roughly uniform star coverage, and coefficients with a stable
    meaning across fields (pixels of displacement at basis amplitude ~1).

    Returns
    -------
    B : (n, 2, K) with K = 2 * n_shape — columns 0..n_shape-1 displace x,
        columns n_shape..K-1 displace y, in PIXELS per unit coefficient.
    """
    xt = np.asarray(X_c, float) / half_x
    yt = np.asarray(Y_c, float) / half_y

    def _leg(t, k):
        if k == 0:
            return np.ones_like(t)
        if k == 1:
            return t
        if k == 2:
            return 1.5 * t * t - 0.5
        if k == 3:
            return 2.5 * t ** 3 - 1.5 * t
        # generic recurrence for k >= 4
        pm2, pm1 = _leg(t, k - 2), _leg(t, k - 1)
        return ((2 * k - 1) * t * pm1 - (k - 1) * pm2) / k

    pairs = epoch_distortion_pairs(order)
    n_shape = len(pairs)
    n = len(xt)
    B = np.zeros((n, 2, 2 * n_shape))
    for k, (i, j) in enumerate(pairs):
        phi = _leg(xt, i) * _leg(yt, j)
        B[:, 0, k] = phi
        B[:, 1, n_shape + k] = phi
    return B


def propagate_gaia_positions(ra_deg, dec_deg, pmra_masyr, pmdec_masyr,
                             parallax_mas, dt_yr, tele_xyz):
    """
    Propagate barycentric catalogue positions by proper motion and annual
    parallax to the epoch whose observer position is *tele_xyz* (AU,
    barycentric — from get_tele_position).

    THE single implementation of this propagation. Do not re-derive the
    parallax factors or the mas->deg offsets inline anywhere else: a
    frame-swap sign bug lived undetected in a duplicated copy of this physics
    from 2026-06 to 2026-09 (see get_parallax_factors).

    Parameters: ra/dec in degrees; pmra = mu_alpha* (mas/yr); parallax in mas;
    dt_yr = t_target − t_catalogue in Julian years (scalar or array).
    Non-finite pm/parallax values are treated as 0.

    Returns (ra_prop_deg, dec_prop_deg).
    """
    ra_deg  = np.asarray(ra_deg,  dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    pmra  = np.where(np.isfinite(pmra_masyr),  pmra_masyr,  0.0)
    pmdec = np.where(np.isfinite(pmdec_masyr), pmdec_masyr, 0.0)
    plx   = np.where(np.isfinite(parallax_mas), parallax_mas, 0.0)

    f_ra, f_dec = get_parallax_factors(ra_deg, dec_deg, tele_xyz)
    ra_off_mas  = pmra  * dt_yr + plx * f_ra
    dec_off_mas = pmdec * dt_yr + plx * f_dec

    ra_prop  = ra_deg  + (ra_off_mas / 3.6e6) / np.cos(dec_deg * DEG2RAD)
    dec_prop = dec_deg +  dec_off_mas / 3.6e6
    return ra_prop, dec_prop


def build_U_matrix(dt_yr, plx_ra_star, plx_dec):
    """
    Build the 2×5 time-evolution matrix U for a single star/image pair.

    v_T,i = (Δα*, Δδ, μα*, μδ, ϖ)
    θ_T,i,j = θ_s,i + diag(1/cos(δ), 1) · U · v_T,i   (in deg)

    U = [[1, 0, dt, 0,  plx_ra_star],
         [0, 1, 0,  dt, plx_dec    ]]

    Parameters
    ----------
    dt_yr : float, time difference (HST epoch - Gaia epoch) in years
    plx_ra_star : float, parallax factor in RA* direction (mas/mas)
    plx_dec : float, parallax factor in Dec direction (mas/mas)

    Returns
    -------
    U : (2, 5) array
    """
    return np.array([[1., 0., dt_yr, 0.,    plx_ra_star],
                     [0., 1., 0.,    dt_yr, plx_dec    ]])


def n_r_from_poly_order(poly_order):
    """
    Number of image-transformation parameters for a given polynomial order.

    Layout:
      Degree 0–1 (linear + tangent point): 6 parameters — fixed.
      Each additional degree k ≥ 2 contributes 2*(k+1) new parameters:
        (k+1) for the x-equation block + (k+1) for the y-equation block.

    Formula: N_R(p) = (p+1)*(p+2)

      p=1 →  6  (linear: a,b,c,d,Δα0,Δδ0)
      p=2 → 12  (adds 3 quadratic terms per equation)
      p=3 → 20  (adds 4 cubic terms per equation)

    Note: the w/z pixel-translation terms have been removed.  The tangent-point
    shift parameters Δα0/Δδ0 (indices 4,5) capture the same bulk-offset effect
    more accurately via per-star Jacobians, and the two sets of parameters were
    degenerate when both were free.
    """
    return (poly_order + 1) * (poly_order + 2)


def build_X_matrix(x_hst, y_hst, dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0,
                   poly_order=1):
    """
    Build the 2×N_R(poly_order) design matrix X for a single star/image pair.

    X · r_j = observed position, where

      r_j = (a, b, c, d, Δα0, Δδ0,
             [x² coeff (x eq), xy coeff (x eq), y² coeff (x eq),
              x² coeff (y eq), xy coeff (y eq), y² coeff (y eq),
              ... higher-degree blocks ...])

    The w/z pixel-translation terms have been removed.  Bulk image offsets are
    now captured by Δα0/Δδ0 (indices 4,5) via per-star Jacobians, which is more
    accurate than a constant pixel translation.

    Column layout for degree k ≥ 2 (new terms beyond the 6 base parameters):
      x-equation block (k+1 cols): coeff of x^(k-j)·y^j, j=0..k  → row 0 only
      y-equation block (k+1 cols): same monomials                  → row 1 only

    Parameters
    ----------
    x_hst, y_hst : float   centered HST pixel positions (X-Xo, Y-Yo)
    dxs_dra0, … : float   tangent-point derivatives (from plane_project_tangent_derivs)
    poly_order   : int     polynomial order (default 1 = linear)

    Returns
    -------
    X : (2, N_R) ndarray
    """
    n_r = n_r_from_poly_order(poly_order)
    X = np.zeros((2, n_r))

    # ── Linear and tangent-point terms (positions 0-5) ────────────────────────
    X[0, 0] = x_hst;    X[0, 1] = y_hst
    X[0, 4] = dxs_dra0; X[0, 5] = dxs_ddec0
    X[1, 2] = x_hst;    X[1, 3] = y_hst
    X[1, 4] = dys_dra0; X[1, 5] = dys_ddec0

    # ── Higher-order polynomial terms ─────────────────────────────────────────
    # Scale each degree-k monomial by 1/2048^(k-1) so that the basis functions
    # remain O(2048) — the same scale as the linear x_hst / y_hst terms.
    # This keeps all polynomial r-vector coefficients (a, b, … and higher-order)
    # at a similar order of magnitude, improving numerical conditioning.
    _S = 2048.0
    col = 6
    for deg in range(2, poly_order + 1):
        scale = _S ** (deg - 1)
        for j in range(deg + 1):
            X[0, col] = x_hst ** (deg - j) * y_hst ** j / scale   # x-equation
            col += 1
        for j in range(deg + 1):
            X[1, col] = x_hst ** (deg - j) * y_hst ** j / scale   # y-equation
            col += 1

    return X


def compute_poly_jacobian(r_j, X_c, Y_c, poly_order):
    """
    Position-dependent 2×2 Jacobian of (x_pred, y_pred) w.r.t. (X_c, Y_c)
    for the polynomial image transformation.

    Used to propagate the per-star HST position covariance C_hst into the
    pseudo-image frame:  C_s,k = J_k @ C_hst_k @ J_k^T.

    For poly_order=1 this reduces to  J_k = [[a, b], [c, d]]  (the R matrix).

    Parameters
    ----------
    r_j       : (N_R,) current image-transformation vector for image j
    X_c, Y_c  : (n,)  centered HST pixel positions for all stars in image j
    poly_order: int

    Returns
    -------
    J : (n, 2, 2) ndarray
    """
    n = len(X_c)
    J = np.zeros((n, 2, 2))

    # Linear part: J = [[a, b], [c, d]] at every position
    J[:, 0, 0] = r_j[0]   # dx/dXc = a
    J[:, 0, 1] = r_j[1]   # dx/dYc = b
    J[:, 1, 0] = r_j[2]   # dy/dXc = c
    J[:, 1, 1] = r_j[3]   # dy/dYc = d

    # Same 1/2048^(deg-1) scaling as in build_X_matrix — derivatives carry
    # the same scale factor so that the Jacobian is consistent with X_mat.
    _S = 2048.0
    # Quadratic+ coefficients start at column 6: (a,b,c,d,dRA0,dDec0) then the
    # per-degree blocks — same layout as build_X_matrix. (col=8 was the stale
    # pre-reparameterization layout with w/z; it made poly_order>=2 crash with
    # IndexError after the w/z removal, and would silently misread for
    # poly_order>=3.)
    col = 6
    for deg in range(2, poly_order + 1):
        scale = _S ** (deg - 1)
        # x-equation block
        for j in range(deg + 1):
            coeff = r_j[col]
            k = deg - j          # power of X_c in this monomial
            if k > 0:
                J[:, 0, 0] += coeff * k * X_c ** (k - 1) * Y_c ** j / scale
            if j > 0:
                J[:, 0, 1] += coeff * j * X_c ** k * Y_c ** (j - 1) / scale
            col += 1
        # y-equation block
        for j in range(deg + 1):
            coeff = r_j[col]
            k = deg - j
            if k > 0:
                J[:, 1, 0] += coeff * k * X_c ** (k - 1) * Y_c ** j / scale
            if j > 0:
                J[:, 1, 1] += coeff * j * X_c ** k * Y_c ** (j - 1) / scale
            col += 1

    return J


def gaia_cov_to_survey_cov(ra_error_mas, dec_error_mas, pmra_error, pmdec_error,
                            parallax_error,
                            ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
                            dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
                            parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr):
    """
    Build 5×5 covariance matrix for v_s,i = (Δα*, Δδ, μα*, μδ, ϖ).

    Gaia reports ra_error, dec_error in mas. The (Δα*, Δδ) entries correspond
    to the position uncertainty – but since we set Δα*=Δδ=0 in v_s,i (the
    survey already centred), those rows/cols just carry the position uncertainty
    for the Bayesian prior update (they are added to C_s,i for the data term).

    Order: (Δα*, Δδ, μα*, μδ, ϖ)
    """
    # Gaia errors: ra_error is already in mas (as σ_α* = σ_α·cos(δ))
    sigmas = np.array([ra_error_mas, dec_error_mas, pmra_error, pmdec_error, parallax_error])

    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = ra_dec_corr
    corr[0, 2] = corr[2, 0] = ra_pmra_corr
    corr[0, 3] = corr[3, 0] = ra_pmdec_corr
    corr[0, 4] = corr[4, 0] = ra_parallax_corr
    corr[1, 2] = corr[2, 1] = dec_pmra_corr
    corr[1, 3] = corr[3, 1] = dec_pmdec_corr
    corr[1, 4] = corr[4, 1] = dec_parallax_corr
    corr[2, 3] = corr[3, 2] = pmra_pmdec_corr
    corr[2, 4] = corr[4, 2] = parallax_pmra_corr
    corr[3, 4] = corr[4, 3] = parallax_pmdec_corr

    C = sigmas[:, None] * corr * sigmas[None, :]
    return C


def hst_position_cov(x_err_pix, y_err_pix, xy_corr, floor_err=0.001):
    """
    Build 2×2 HST position covariance matrix from per-star error estimates.
    x_err, y_err in pixels; xy_corr is the correlation coefficient.
    floor_err in pixels: uncertainty floor added in quadrature.  Default is
    0.001 px — effectively negligible since py1pass already computes a
    systematic floor from the PSF-fit residuals.
    """
    cov_xy = x_err_pix * y_err_pix * xy_corr
    return np.array([[x_err_pix**2, cov_xy],
                     [cov_xy,       y_err_pix**2]])+np.eye(2)*floor_err**2


def rotation_matrix_from_abcd(a, b, c, d):
    """
    Given (a,b,c,d) linear transformation matrix, return R = [[a,b],[c,d]].
    This is used to rotate the HST position covariance into the survey frame:
    C_s,i,j = R · C_i,j · R^T
    """
    return np.array([[a, b], [c, d]])


def abcd_from_rotation_pixscale_skew(rotation_deg, pixel_scale_ratio, on_skew, off_skew):
    """
    Convert human-readable image parameters to (a,b,c,d).
    pixel_scale_ratio is relative to the nominal pixel scale.
    rotation_deg is the position angle in degrees.
    """
    rot_rad = rotation_deg * DEG2RAD
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)
    s = pixel_scale_ratio
    a = s * cos_r + on_skew
    b =  s * sin_r + off_skew
    c = -s * sin_r + off_skew
    d = s * cos_r - on_skew
    return a, b, c, d


def field_typical_astrometry(pmra, pmdec, plx=None,
                             bin_masyr=0.1, smooth_bins=2.0, min_n_mode=30):
    """Field-typical (pmra, pmdec, parallax) for propagating stars WITHOUT
    their own astrometric solution (Gaia 2p, HST-only) during cross-matching.

    In dense fields with large systemic PM, propagating 2p stars at 0 PM
    matches them to whatever source sits nearest the un-propagated position
    (Omega_Cen 2026-09-02: 23% of 2p "stars" at PM~0 vs 2% at the systemic
    PM).  The fix is to centre the search at the PM of the DOMINANT field
    population:

      - n >= min_n_mode: smoothed 2D histogram mode of (pmra, pmdec) —
        a median can land between cluster and field populations, the mode
        locks onto the dominant one.  Parallax: 1D mode the same way.
      - n <  min_n_mode: component-wise median (mode too noisy).
      - n == 0: zeros (the legacy behaviour).

    Returns dict(pmra, pmdec, plx, method, n).
    """
    import numpy as _np

    pmra = _np.asarray(pmra, float)
    pmdec = _np.asarray(pmdec, float)
    fin = _np.isfinite(pmra) & _np.isfinite(pmdec)
    n = int(fin.sum())
    out = {"pmra": 0.0, "pmdec": 0.0, "plx": 0.0, "method": "none", "n": n}
    if n == 0:
        return out

    pr, pd_ = pmra[fin], pmdec[fin]
    if n < min_n_mode:
        out.update(pmra=float(_np.median(pr)), pmdec=float(_np.median(pd_)),
                   method="median")
    else:
        from scipy.ndimage import gaussian_filter
        med = _np.array([_np.median(pr), _np.median(pd_)])
        mad = _np.maximum(1.4826 * _np.array(
            [_np.median(_np.abs(pr - med[0])),
             _np.median(_np.abs(pd_ - med[1]))]), bin_masyr)
        half = _np.clip(10.0 * mad, 2.0, 50.0)
        bins = [_np.arange(med[k] - half[k], med[k] + half[k] + bin_masyr,
                           bin_masyr) for k in (0, 1)]
        H, ex, ey = _np.histogram2d(pr, pd_, bins=bins)
        Hs = gaussian_filter(H, smooth_bins)
        i, j = _np.unravel_index(_np.argmax(Hs), Hs.shape)
        out.update(pmra=float(0.5 * (ex[i] + ex[i + 1])),
                   pmdec=float(0.5 * (ey[j] + ey[j + 1])),
                   method="mode")

    # Robust PM dispersion about the adopted centre — the physically
    # meaningful search-window scale for stars without their own PMs.
    # (Filling 2p PM errors with 100 mas/yr makes every candidate within
    # ~2 px "consistent", so mismatches survive any centring fix.)
    dr = _np.hypot(pr - out["pmra"], pd_ - out["pmdec"])
    out["pm_sig"] = float(max(1.4826 * _np.median(dr), 0.3))

    if plx is not None:
        plx = _np.asarray(plx, float)
        pfin = _np.isfinite(plx)
        if pfin.sum() >= min_n_mode and out["method"] == "mode":
            from scipy.ndimage import gaussian_filter1d
            pv = plx[pfin]
            medp = float(_np.median(pv))
            madp = max(1.4826 * float(_np.median(_np.abs(pv - medp))), 0.02)
            b = _np.arange(medp - 10 * madp, medp + 10 * madp + 0.02, 0.02)
            h, e = _np.histogram(pv, bins=b)
            hs = gaussian_filter1d(h.astype(float), smooth_bins)
            out["plx"] = float(0.5 * (e[_np.argmax(hs)] + e[_np.argmax(hs) + 1]))
        elif pfin.any():
            out["plx"] = float(_np.median(plx[pfin]))
        if pfin.any():
            _pv = plx[pfin]
            out["plx_sig"] = float(max(
                1.4826 * _np.median(_np.abs(_pv - out["plx"])), 0.05))
    out.setdefault("plx_sig", 0.5)
    return out
