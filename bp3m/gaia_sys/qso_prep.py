"""QSO conditioning-sample steps of the pop_fit production for the Gaia field modes (handoff 3.2 / 3.2a / 3.4).
VERBATIM functions: inducing_points from pop_fit_dev/gp_sys.py (sha256 2ac26998d353b4a8ada161b46889809f8382c40285923f3e76389f109a77089b), crossmatch_milliquas and
measure_pm_systematic from pop_fit_dev/data.py (sha256 3e60df3a41125ac67ad3f2c80c209c7934ca2a7b882a91f6566ef00eaf678808), copied 2026-10-29.  The MILLIQUAS default path is
replaced by an explicit argument (bp3m's lib_dir copy, byte-identical to pop_fit's milliquas.fits 10-29).
Re-vendor (never edit) when pop_fit announces a change."""
import os
import numpy as np
import pandas as pd

DEG = np.pi / 180
MILLIQUAS_PATH = None    # VENDOR NOTE: always passed explicitly by bp3m (crossmatch_milliquas raises on None)


# VERBATIM pop_fit_dev/gp_sys.py
def inducing_points(ra0, dec0, extent_deg, n_rings=3, min_spacing_deg=0.1):
    """
    Hexagonal set of inducing points: the centre plus n_rings rings (6, 12, 18 points),
    spacing extent/n_rings (≥ min_spacing_deg).  Returns (ra, dec) arrays.
    """
    sp = max(extent_deg / max(n_rings, 1), min_spacing_deg)
    xs, ys = [0.0], [0.0]
    for k in range(1, n_rings + 1):
        for j in range(6 * k):
            a = 2 * np.pi * j / (6 * k)
            xs.append(k * sp * np.cos(a)); ys.append(k * sp * np.sin(a))
    xs, ys = np.array(xs), np.array(ys)
    return ra0 + xs / np.cos(dec0 * DEG), dec0 + ys


# VERBATIM pop_fit_dev/data.py
def crossmatch_milliquas(gaia_ra, gaia_dec, milliquas_path=None,
                          match_radius_arcsec=1.5):
    """
    Cross-match Gaia positions with the MILLIQUAS catalog using a 3-D
    unit-vector KD-tree (exact great-circle distances for small angles).

    Parameters
    ----------
    gaia_ra, gaia_dec   : (N,) degrees
    milliquas_path      : path to milliquas.fits; defaults to MILLIQUAS_PATH
    match_radius_arcsec : matching radius (default 1.5 arcsec)

    Returns
    -------
    matched : (N,) bool — True where a MILLIQUAS counterpart is found
    """
    from astropy.io import fits
    from scipy.spatial import cKDTree

    if milliquas_path is None:
        milliquas_path = MILLIQUAS_PATH

    print(f'  Loading MILLIQUAS from {os.path.basename(milliquas_path)}...')
    with fits.open(milliquas_path) as hdul:
        mq_ra  = hdul[1].data['RA'].astype(np.float64)
        mq_dec = hdul[1].data['DEC'].astype(np.float64)
    print(f'  {len(mq_ra):,} MILLIQUAS sources loaded')

    def _xyz(ra_deg, dec_deg):
        ra  = np.deg2rad(np.asarray(ra_deg,  dtype=np.float64))
        dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
        cd  = np.cos(dec)
        return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])

    gaia_xyz = _xyz(gaia_ra,  gaia_dec)
    mq_xyz   = _xyz(mq_ra,    mq_dec)

    tree    = cKDTree(mq_xyz)
    r_rad   = np.deg2rad(match_radius_arcsec / 3600.0)
    r_chord = 2.0 * np.sin(r_rad / 2.0)   # chord length ≈ θ for small angles

    dists, _ = tree.query(gaia_xyz, k=1, distance_upper_bound=r_chord * 1.01)
    return dists <= r_chord


# VERBATIM pop_fit_dev/data.py
def measure_pm_systematic(qso_df, min_qsos=20,
                           parallax_nsigma=3.0, pm_nsigma=3.0, n_iter=10):
    """
    Inverse-variance weighted mean PM of QSO candidates, with two-stage cleaning.

    Stage 1 — parallax cut: removes sources with |parallax| / parallax_error >
    parallax_nsigma.  True QSOs are at cosmological distances (parallax ≈ 0);
    stars with significantly non-zero parallax are rejected here.

    Stage 2 — iterative PM sigma-clipping: compute the inverse-variance
    weighted mean (mu_ra, mu_dec), remove sources where
    |pmra − mu_ra| / pmra_error > pm_nsigma or
    |pmdec − mu_dec| / pmdec_error > pm_nsigma, then repeat until convergence.
    Clipping relative to the running mean avoids biasing the result toward zero.

    Returns
    -------
    dict with keys:
        delta_pmra, delta_pmdec : float or None (None if < min_qsos survived)
        sigma_pmra, sigma_pmdec : float or None
        n_kept                  : int  — QSOs surviving all cleaning
        n_raw                   : int  — total rows in qso_df
        mask_valid              : (N,) bool — finite PM + error
        mask_plx_kept           : (N,) bool — survived parallax cut
        mask_kept               : (N,) bool — survived all cleaning
        parallax_nsigma, pm_nsigma : floats (echoed for plot labels)
    """
    N     = len(qso_df)
    pmra  = qso_df['pmra'].to_numpy(dtype=float)
    pmdec = qso_df['pmdec'].to_numpy(dtype=float)
    e_ra  = qso_df['pmra_error'].to_numpy(dtype=float)
    e_dec = qso_df['pmdec_error'].to_numpy(dtype=float)

    ok = (np.isfinite(pmra) & np.isfinite(pmdec)
          & np.isfinite(e_ra) & np.isfinite(e_dec)
          & (e_ra > 0) & (e_dec > 0))

    # Stage 1: parallax consistency cut
    ok_plx = ok.copy()
    if 'parallax' in qso_df.columns and 'parallax_error' in qso_df.columns:
        plx    = qso_df['parallax'].to_numpy(dtype=float)
        e_plx  = qso_df['parallax_error'].to_numpy(dtype=float)
        plx_ok = np.isfinite(plx) & np.isfinite(e_plx) & (e_plx > 0)
        ok_plx = ok & (~plx_ok | (np.abs(plx) < parallax_nsigma * e_plx))

    def _no_result(mask_kept):
        return dict(
            delta_pmra=None, delta_pmdec=None,
            sigma_pmra=None, sigma_pmdec=None,
            n_kept=int(mask_kept.sum()), n_raw=N,
            mask_valid=ok, mask_plx_kept=ok_plx, mask_kept=mask_kept,
            parallax_nsigma=parallax_nsigma, pm_nsigma=pm_nsigma,
        )

    if ok_plx.sum() < min_qsos:
        return _no_result(ok_plx)

    # Stage 2: iterative PM sigma-clipping around the running mean
    mask = ok_plx.copy()
    for _ in range(n_iter):
        w_ra  = 1.0 / e_ra[mask]**2
        w_dec = 1.0 / e_dec[mask]**2
        mu_ra  = np.sum(w_ra  * pmra[mask])  / np.sum(w_ra)
        mu_dec = np.sum(w_dec * pmdec[mask]) / np.sum(w_dec)

        chi_ra  = np.abs(pmra  - mu_ra)  / e_ra
        chi_dec = np.abs(pmdec - mu_dec) / e_dec
        new_mask = ok_plx & (chi_ra < pm_nsigma) & (chi_dec < pm_nsigma)
        if new_mask.sum() == mask.sum():
            break
        if new_mask.sum() < min_qsos:
            break
        mask = new_mask

    if mask.sum() < min_qsos:
        return _no_result(mask)

    w_ra  = 1.0 / e_ra[mask]**2
    w_dec = 1.0 / e_dec[mask]**2
    delta_pmra  = float(np.sum(w_ra  * pmra[mask])  / np.sum(w_ra))
    delta_pmdec = float(np.sum(w_dec * pmdec[mask]) / np.sum(w_dec))
    sigma_pmra  = float(np.sqrt(1.0 / np.sum(w_ra)))
    sigma_pmdec = float(np.sqrt(1.0 / np.sum(w_dec)))

    return dict(
        delta_pmra=delta_pmra, delta_pmdec=delta_pmdec,
        sigma_pmra=sigma_pmra, sigma_pmdec=sigma_pmdec,
        n_kept=int(mask.sum()), n_raw=N,
        mask_valid=ok, mask_plx_kept=ok_plx, mask_kept=mask,
        parallax_nsigma=parallax_nsigma, pm_nsigma=pm_nsigma,
    )
