"""STDPSF loader, FITS image loader, PSF auto-detection, GDC/WCS corrections, and catalog writer."""

import glob
import os
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

# py1pass intentionally writes long keyword names (e.g. CHIP1_CRPIX1_GDC) as
# HIERARCH cards, which are valid FITS and read back correctly.  Suppress the
# astropy VerifyWarning that fires whenever a keyword name exceeds 8 characters.
warnings.filterwarnings(
    'ignore',
    message=r'.*greater than 8 characters.*HIERARCH.*',
    category=fits.verify.VerifyWarning,
)


# ---------------------------------------------------------------------------
# PSF auto-detection
# ---------------------------------------------------------------------------

# Instrument/detector → prefix used in STDPSF filenames
_DETECTOR_PREFIX = {
    ('ACS',  'WFC'):  'ACSWFC',
    ('ACS',  'HRC'):  'ACSHRC',
    ('ACS',  'SBC'):  'ACSSBC',
    ('WFC3', 'UVIS'): 'WFC3UV',
    ('WFC3', 'IR'):   'WFC3IR',
}

# Science and DQ extension pairs, and PSF-grid y-offsets, per chip
# (sci_ext, dq_ext, y_offset_for_psf_grid)
# Used only as a fallback when the FITS file cannot be opened to read CCDCHIP.
_CHIP_CONFIG = {
    ('ACS',  'WFC'):  [(1, 3, 0.0), (4, 6, 2048.0)],
    ('ACS',  'HRC'):  [(1, 2, 0.0)],
    ('ACS',  'SBC'):  [(1, 2, 0.0)],
    ('WFC3', 'UVIS'): [(1, 3, 0.0), (4, 6, 2051.0)],
    ('WFC3', 'IR'):   [(1, 2, 0.0)],
}

# y_offset (rows of bottom chip) to add to image-y for PSF grid / combined-frame lookup,
# keyed by (instrume, detector, ccdchip_value).
# For two-chip cameras chip2 is always the bottom (y_offset=0); chip1 is the top.
_CCDCHIP_Y_OFFSET = {
    ('ACS',  'WFC',  1): 2048.0,
    ('ACS',  'WFC',  2): 0.0,
    ('ACS',  'HRC',  1): 0.0,
    ('ACS',  'SBC',  1): 0.0,
    ('WFC3', 'UVIS', 1): 2051.0,
    ('WFC3', 'UVIS', 2): 0.0,
    ('WFC3', 'IR',   1): 0.0,
}

# MJD boundaries for ACS/WFC PSF servicing-mission era selection
_SM3B_MJD = 52346.0   # 2002-03-12: start of SM3B era
_SM4_MJD  = 54975.0   # 2009-05-24: start of SM4 era


def _sm_suffix(mjd_obs, instrume, detector):
    """Return PSF filename SM-era suffix for ACS/WFC; '' for all other detectors."""
    if (instrume.upper(), detector.upper()) != ('ACS', 'WFC'):
        return ''
    if mjd_obs < _SM3B_MJD:
        return ''
    elif mjd_obs < _SM4_MJD:
        return '_SM3'
    else:
        return '_SM4'


def _filter_candidates(filt):
    """Return filter name variants to try when looking up library files.

    STScI names some filters differently from the FITS header keyword:
    - ``F850LP`` (FITS header) → ``F850L`` (STScI filename)
    - ``F350LP`` (FITS header) → ``F350L`` (STScI filename)
    The original name is always tried first so that user-renamed local
    libraries keep working.
    """
    names = [filt]
    if filt.endswith('LP'):
        names.append(filt[:-1])   # strip trailing P: F850LP → F850L
    return names


def find_psf(psf_dir, header):
    """Find the best-matching STDPSF file for a FITS image header.

    For ACS/WFC, selects the SM-era PSF based on the MJD-OBS keyword:
    - MJD < 52346 (before 2002-03-12 SM3B): no suffix
    - 52346 ≤ MJD < 54975 (before 2009-05-24 SM4): ``_SM3`` suffix
    - MJD ≥ 54975 (after SM4): ``_SM4`` suffix

    Falls back across SM suffixes, filter name variants (e.g. F850LP→F850L),
    and performs a final case-insensitive glob so the function is resilient to
    minor naming differences in locally-downloaded library files.

    Raises FileNotFoundError if no match is found.
    """
    instrume = header.get('INSTRUME', '').strip().upper()
    detector = header.get('DETECTOR', '').strip().upper()
    filt = _extract_filter(header, instrume)

    det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
    if det_prefix is None:
        raise ValueError(f"Unknown instrument/detector: {instrume}/{detector}. "
                         f"Supported: {list(_DETECTOR_PREFIX.keys())}")

    # HST FLC/FLT files use EXPSTART (MJD) rather than MJD-OBS
    for _mjd_key in ('MJD-OBS', 'EXPSTART', 'EXPEND'):
        _mjd_val = header.get(_mjd_key, None)
        if _mjd_val is not None:
            mjd = float(_mjd_val)
            break
    else:
        mjd = 0.0
    sm = _sm_suffix(mjd, instrume, detector)

    # SM suffix fallback order: preferred → next older → base
    # e.g. SM4 images: try _SM4, then _SM3, then no suffix
    if sm == '_SM4':
        sm_order = ['_SM4', '_SM3', '']
    elif sm == '_SM3':
        sm_order = ['_SM3', '']
    else:
        sm_order = ['']

    filt_names = _filter_candidates(filt)

    for f in filt_names:
        for suffix in sm_order:
            candidate = os.path.join(psf_dir, f'STDPSF_{det_prefix}_{f}{suffix}.fits')
            if os.path.exists(candidate):
                return candidate
            # case-insensitive fallback via glob
            import glob as _glob
            matches = _glob.glob(os.path.join(
                psf_dir, f'[Ss][Tt][Dd][Pp][Ss][Ff]_{det_prefix}_{f}{suffix}.fits'))
            if not matches:
                matches = _glob.glob(os.path.join(
                    psf_dir, f'STDPSF_{det_prefix}_{f.lower()}{suffix.lower()}.fits'))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"No PSF file found for {instrume}/{detector} {filt} (SM suffix='{sm}') "
        f"in {psf_dir}.\n"
        f"  Tried filter names: {filt_names}, SM suffixes: {sm_order}\n"
        f"  Available: {os.listdir(psf_dir)}"
    )


def find_gdc(gdc_dir, header):
    """Find the GDC FITS file for a FITS image header.

    Searches *gdc_dir* for the best-matching GDC file.  Preference order:
    1. ``STDGDC_OFFICIAL_JFRAME_{det_prefix}_{filt}.fits`` — the J-frame
       corrected version produced by Anderson, which maps all filters into
       the F814W-based J-frame.  This is the correct file to use for
       cross-filter and cross-epoch astrometry.
    2. ``STDGDC_{det_prefix}_{filt}.fits`` — the raw per-filter GDC.
       For some filters (e.g. ACS/WFC F435W) this file encodes distortion
       relative to a different internal frame and will produce large
       systematic offsets (~60–100 px) when used naively.

    Returns the path of the first match, or None if neither file exists.
    """
    instrume = header.get('INSTRUME', '').strip().upper()
    detector = header.get('DETECTOR', '').strip().upper()
    det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
    if det_prefix is None:
        return None
    try:
        filt = _extract_filter(header, instrume)
    except ValueError:
        return None
    for f in _filter_candidates(filt):
        for name in (f'STDGDC_OFFICIAL_JFRAME_{det_prefix}_{f}.fits',
                     f'STDGDC_{det_prefix}_{f}.fits'):
            path = os.path.join(gdc_dir, name)
            if os.path.exists(path):
                return path
    return None


def _extract_filter(header, instrume):
    """Extract the science filter name from a FITS primary header."""
    # ACS uses FILTER1 / FILTER2; pick the non-CLEAR one
    if instrume in ('ACS', ''):
        for key in ('FILTER1', 'FILTER2'):
            val = header.get(key, '').strip().upper()
            if val and not val.startswith('CLEAR'):
                return val
    # WFC3 and most other instruments use FILTER
    for key in ('FILTER', 'FILTNAM1', 'FILTNAM2'):
        val = header.get(key, '').strip().upper()
        if val and not val.startswith('CLEAR'):
            return val
    raise ValueError("Cannot determine science filter from FITS header. "
                     "Set FILTER, FILTER1, or FILTER2.")


def get_chip_config(instrume, detector):
    """Return per-chip configuration for a two-chip instrument.

    Parameters
    ----------
    instrume : str  e.g. 'ACS'
    detector : str  e.g. 'WFC'

    Returns
    -------
    list of (sci_ext, dq_ext, y_offset) tuples, one per chip.
    y_offset is added to image-y to get detector-y for PSF grid lookup.
    """
    key = (instrume.strip().upper(), detector.strip().upper())
    config = _CHIP_CONFIG.get(key)
    if config is None:
        warnings.warn(f"Unknown instrument/detector {key}; assuming single chip at ext 1.")
        return [(1, 2, 0.0)]
    return config


def get_chip_config_from_fits(image_path, instrume, detector):
    """Determine chip configuration by reading CCDCHIP from each SCI extension.

    Unlike get_chip_config, which hard-codes the assumption that SCI chip 2 is
    always in extension 1, this function reads the CCDCHIP keyword from the
    actual FITS headers so it works even when the extension ordering differs.

    DQ extension is assumed to be sci_ext + 2 (standard calacs/calwf3 layout).

    Returns
    -------
    list of (sci_ext, dq_ext, y_offset) sorted by y_offset ascending (bottom chip first).
    Falls back to get_chip_config if the file cannot be read or has no CCDCHIP keywords.
    """
    instr  = instrume.strip().upper()
    det    = detector.strip().upper()
    try:
        with fits.open(image_path) as hdul:
            chips = []
            for i, hdu in enumerate(hdul):
                if not hasattr(hdu, 'header'):
                    continue
                if hdu.header.get('EXTNAME', '').strip().upper() != 'SCI':
                    continue
                ccdchip_raw = hdu.header.get('CCDCHIP', None)
                if ccdchip_raw is None:
                    continue
                try:
                    ccdchip = int(ccdchip_raw)
                except (ValueError, TypeError):
                    continue
                y_off = _CCDCHIP_Y_OFFSET.get((instr, det, ccdchip), 0.0)
                dq_ext = i + 2   # SCI→ERR→DQ is the standard triplet layout
                chips.append((i, dq_ext, y_off))
            if chips:
                return sorted(chips, key=lambda t: t[2])   # bottom chip first
    except Exception:
        pass
    return get_chip_config(instrume, detector)


# ---------------------------------------------------------------------------
# PSF loading
# ---------------------------------------------------------------------------

def load_stdpsf(path):
    """Load a STDPSF FITS file (Anderson & King format).

    Returns
    -------
    psf_cube  : (n_psf, size, size) float64 array
    xs        : 1D float64 array of PSF grid x detector coordinates (length nx_g)
    ys        : 1D float64 array of PSF grid y detector coordinates (length ny_g)
    psf_scale : supersampling factor (int)
    grid_shape: (ny_g, nx_g) tuple
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        psf_raw = np.array(hdul[0].data, dtype=np.float64)

    # WFC3/UVIS STDPSFs use -0.1 as a sentinel for "no valid calibration data"
    # at boundary pixels (e.g. off-chip or unsampled regions).  These are not
    # PSF values — they are flag markers.  Without this fix, spline_filter sees
    # the sharp 0 → -0.1 discontinuity at the array edge and produces Gibbs-
    # like ringing in the spline coefficients.  The ringing decays rapidly
    # (~0.268 per pixel for cubic order-3 splines) and is negligible at the
    # distances involved in fitting (~38 oversampled pixels from the boundary),
    # so the sentinel values do not affect PSF fitting accuracy — but they do
    # show up visibly in diagnostic plots that render the raw PSF array.
    #
    # Empirical-fit negatives (~-5e-5, 0.03% of PSF peak) are left intact:
    # they represent the PSF model's fitting uncertainty near the wings and
    # have no measurable impact on spline interpolation or photometry.
    #
    # Threshold of -0.001 cleanly separates the two populations (there are no
    # values between -0.001 and -0.0001 in any WFC3/UVIS STDPSF file).
    psf_raw = np.where(psf_raw < -0.001, 0.0, psf_raw)

    if psf_raw.ndim == 2:
        psf_raw = psf_raw[np.newaxis]

    n_psf = psf_raw.shape[0]
    psf_scale = float(hdr.get('PSFSCALE', hdr.get('OVERSAMP', 4.0)))

    nx_g = hdr.get('NXPSFS', hdr.get('NXPSF', None))
    ny_g = hdr.get('NYPSFS', hdr.get('NYPSF', None))

    if nx_g is not None and ny_g is not None and n_psf > 1:
        nx_g = int(nx_g)
        ny_g = int(ny_g)
        xs = _read_psf_positions(hdr, 'IPSFX', nx_g)
        ys = _read_psf_positions(hdr, 'JPSFY', ny_g)
        if len(xs) == nx_g and len(ys) == ny_g:
            return (psf_raw,
                    np.array(xs, dtype=np.float64),
                    np.array(ys, dtype=np.float64),
                    int(psf_scale), (ny_g, nx_g))
        warnings.warn("Could not read all PSF grid positions; using regular grid.")

    if n_psf == 1:
        return psf_raw, np.array([0.0]), np.array([0.0]), int(psf_scale), (1, 1)

    nx_g = int(np.ceil(np.sqrt(n_psf)))
    ny_g = int(np.ceil(n_psf / nx_g))
    xs_arr = np.linspace(0.0, 4096.0, nx_g)
    ys_arr = np.linspace(0.0, 4096.0, ny_g)
    warnings.warn(f"PSF grid positions not found; placing {n_psf} PSFs on "
                  f"a regular {ny_g}×{nx_g} grid.")
    return psf_raw[:ny_g * nx_g], xs_arr, ys_arr, int(psf_scale), (ny_g, nx_g)


def _read_psf_positions(hdr, prefix, n):
    """Read IPSFX01..IPSFXnn or JPSFY01..JPSFYnn keywords."""
    vals = []
    for i in range(1, n + 1):
        for fmt in (f'{prefix}{i:02d}', f'{prefix}{i:04d}'):
            if fmt in hdr:
                vals.append(float(hdr[fmt]))
                break
    return vals


# ---------------------------------------------------------------------------
# GDC loading and application
# ---------------------------------------------------------------------------

def load_stdgdc(path):
    """Load a STDGDC FITS file (Anderson geometric distortion correction).

    The file contains five extensions:
      [1] XGC : forward raw→corrected x map, shape (NDIM_YGC, NDIM_XGC)
      [2] YGC : forward raw→corrected y map, shape (NDIM_YGC, NDIM_XGC)
      [3] MGC : pixel-area magnitude correction, shape (NDIM_YGC, NDIM_XGC)
      [4] XCG : reverse corrected→raw x map
      [5] YCG : reverse corrected→raw y map

    Arrays are stored as scaled integers in the raw FITS file; astropy applies
    BSCALE/BZERO automatically, returning float32.  The returned arrays are
    cast to float64 for double-precision interpolation.

    Returns a dict with keys: xgc, ygc, mgc, ndim_xgc, ndim_ygc, xgc_0, ygc_0.
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        ndim_xgc = int(hdr['NDIM_XGC'])
        ndim_ygc = int(hdr['NDIM_YGC'])
        xgc_0 = int(hdr.get('XGC_0', 0))
        ygc_0 = int(hdr.get('YGC_0', 0))
        xgc = np.array(hdul[1].data, dtype=np.float64)
        ygc = np.array(hdul[2].data, dtype=np.float64)
        mgc = np.array(hdul[3].data, dtype=np.float64)
    return {
        'xgc': xgc, 'ygc': ygc, 'mgc': mgc,
        'ndim_xgc': ndim_xgc, 'ndim_ygc': ndim_ygc,
        'xgc_0': xgc_0, 'ygc_0': ygc_0,
    }


def apply_gdc(x_raw, y_raw, gdc):
    """Apply GDC forward correction to 0-indexed combined-frame positions.

    Implements the Fortran ``extract_stdgdc`` SENSE=1 (forward) and SENSE=0
    (magnitude correction) logic.  Input ``x_raw, y_raw`` are 0-indexed
    combined-frame detector coordinates; the bilinear interpolation is
    identical to the Fortran 1-indexed version because the fractional parts
    are the same.

    Parameters
    ----------
    x_raw, y_raw : float or array-like
        0-indexed combined-frame pixel positions.
    gdc : dict
        As returned by ``load_stdgdc``.

    Returns
    -------
    x_corr, y_corr : ndarray
        J-frame corrected positions (NaN for out-of-bounds points).
    mc : ndarray
        Pixel-area magnitude correction (0.0 for out-of-bounds points).
    """
    xgc = gdc['xgc']
    ygc = gdc['ygc']
    mgc = gdc['mgc']
    nx = gdc['ndim_xgc']
    ny = gdc['ndim_ygc']

    x = np.atleast_1d(np.asarray(x_raw, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y_raw, dtype=np.float64))

    # Fortran SENSE=1 bounds: ipix in [0-1, nx+1] (loose), clamp to [0, nx-2]
    valid = (x >= -1.0) & (x <= nx + 1.0) & (y >= -1.0) & (y <= ny + 1.0)

    ix = np.clip(np.floor(x).astype(int), 0, nx - 2)
    iy = np.clip(np.floor(y).astype(int), 0, ny - 2)
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    def _bilin(arr):
        return ((1 - fx) * (1 - fy) * arr[iy,     ix    ] +
                (1 - fx) *      fy  * arr[iy + 1,  ix    ] +
                     fx  * (1 - fy) * arr[iy,      ix + 1] +
                     fx  *      fy  * arr[iy + 1,  ix + 1])

    x_corr = np.where(valid, _bilin(xgc), np.nan)
    y_corr = np.where(valid, _bilin(ygc), np.nan)

    # MGC: stricter bounds (no clamping, return 0 for OOB)
    ix0 = np.floor(x).astype(int)
    iy0 = np.floor(y).astype(int)
    valid_mgc = (ix0 >= 0) & (ix0 <= nx - 2) & (iy0 >= 0) & (iy0 <= ny - 2)
    mc = np.where(valid_mgc, _bilin(mgc), 0.0)

    return x_corr, y_corr, mc


def _gdc_jacobian_batch(x_raw, y_raw, gdc, step=0.1):
    """Compute GDC Jacobian d(x_gdc,y_gdc)/d(x,y) via central differences.

    Returns J of shape (N, 2, 2) where J[i] = [[dxc/dx, dxc/dy],[dyc/dx, dyc/dy]].
    """
    xp, yp, _ = apply_gdc(x_raw + step, y_raw, gdc)
    xm, ym, _ = apply_gdc(x_raw - step, y_raw, gdc)
    xu, yu, _ = apply_gdc(x_raw, y_raw + step, gdc)
    xd, yd, _ = apply_gdc(x_raw, y_raw - step, gdc)

    two_step = 2.0 * step
    N = len(np.atleast_1d(x_raw))
    J = np.zeros((N, 2, 2))
    J[:, 0, 0] = (xp - xm) / two_step   # dxc/dx
    J[:, 0, 1] = (xu - xd) / two_step   # dxc/dy
    J[:, 1, 0] = (yp - ym) / two_step   # dyc/dx
    J[:, 1, 1] = (yu - yd) / two_step   # dyc/dy
    return J


# ---------------------------------------------------------------------------
# WCS helpers
# ---------------------------------------------------------------------------

def _wcs_transform_batch(x_chip, y_chip, wcs_obj):
    """Apply per-chip WCS to 0-indexed chip coordinates, returning (ra, dec) in degrees."""
    from astropy.wcs import WCS
    xy = np.column_stack([np.asarray(x_chip, dtype=float),
                          np.asarray(y_chip, dtype=float)]) + 1 #convert to 1-index
    # sky = wcs_obj.all_pix2world(xy, 0)  # origin=0 for 0-indexed
    sky = wcs_obj.all_pix2world(xy,1)  # origin=1 for 1-indexed
    return sky[:, 0], sky[:, 1]         # ra, dec in degrees


def _wcs_jacobian_batch(x_chip, y_chip, wcs_obj, step=0.1):
    """Compute d(RA,Dec)/d(x,y) Jacobian in deg/pix via central differences.

    Returns J of shape (N, 2, 2) where J[i] = [[dra/dx, dra/dy],[ddec/dx, ddec/dy]].
    """
    ra_xp, dec_xp = _wcs_transform_batch(x_chip + step, y_chip, wcs_obj)
    ra_xm, dec_xm = _wcs_transform_batch(x_chip - step, y_chip, wcs_obj)
    ra_yp, dec_yp = _wcs_transform_batch(x_chip, y_chip + step, wcs_obj)
    ra_ym, dec_ym = _wcs_transform_batch(x_chip, y_chip - step, wcs_obj)

    two_step = 2.0 * step
    N = len(np.atleast_1d(x_chip))
    J = np.zeros((N, 2, 2))
    J[:, 0, 0] = (ra_xp  - ra_xm)  / two_step   # dRA/dx
    J[:, 0, 1] = (ra_yp  - ra_ym)  / two_step   # dRA/dy
    J[:, 1, 0] = (dec_xp - dec_xm) / two_step   # dDec/dx
    J[:, 1, 1] = (dec_yp - dec_ym) / two_step   # dDec/dy
    return J


def _propagate_cov_2x2(cov_xy, J_batch):
    """Propagate 2×2 xy covariance through Jacobian J: Cov_out = J @ Cov_xy @ J.T.

    Parameters
    ----------
    cov_xy : (N, 2, 2) array — raw position covariance in (x, y) order
    J_batch : (N, 2, 2) array — Jacobian for each star

    Returns (N, 2, 2) array of propagated covariances.
    """
    return J_batch @ cov_xy @ J_batch.transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# GDC + WCS post-processing for a list of StarRecords
# ---------------------------------------------------------------------------

def xy2radec(x,y,r0,d0):
    '''
    x,y are RA,Dec offsets, in degrees
    r0,d0 are the central RA0,Dec0, in degrees
    '''

    xrad = np.radians(x)
    yrad = np.radians(y)
    r0rad = np.radians(r0)
    d0rad = np.radians(d0)

    cosd0 = np.cos(d0rad)
    sind0 = np.sin(d0rad)

    dr = np.arctan2(xrad,(cosd0-yrad*sind0))
    ra = r0 + np.degrees(dr)

    tande = np.arctan2(np.cos(dr)*(sind0+yrad*cosd0),(cosd0-yrad*sind0))

    dec = np.degrees(tande)
    
    return ra,dec
    


def _apply_gdc_wcs(records, gdc, image_path, chips, instrume, detector, verbose=False):
    """Apply GDC and WCS corrections in-place to a list of StarRecords.

    Stores results as private attributes on each record:
      _x_gdc, _y_gdc, _mc, _cov_gdc (2×2 ndarray or None)
      _ra, _dec, _cov_radec (2×2 ndarray or None)
    """
    from astropy.wcs import WCS as _WCS

    n = len(records)
    if n == 0:
        return

    # --- WCS (per chip) ---
    chip_wcs = {}
    chip_wcs_vals = {}
    wcs_keys = ['CRPIX1','CRPIX2','CRVAL1','CRVAL2',\
                'CTYPE1','CTYPE2','CD1_1','CD1_2','CD2_1','CD2_2']
    #need to convert the XY reference coordinate to GDC frame
    try:
        with fits.open(image_path) as hdul:
            for sci_ext, _dq, _yoff in chips:
                try:
                    chip_wcs[sci_ext] = _WCS(hdul[sci_ext].header,
                                             hdul, relax=True)
                    chip_wcs_vals[sci_ext] = {}
                    for wcs_key in wcs_keys:
                        chip_wcs_vals[sci_ext][wcs_key] = hdul[sci_ext].header[wcs_key]

                    # CRPIX from FITS is 1-indexed. Star catalog x,y are 0-indexed.
                    # Convert to 0-indexed before combining with y_offset so that
                    # CRPIX_COMBINED and CRPIX_GDC are in the same frame as catalog x,y.
                    _crpix1_0 = chip_wcs_vals[sci_ext]['CRPIX1'] - 1.0
                    _crpix2_0 = chip_wcs_vals[sci_ext]['CRPIX2'] - 1.0
                    chip_wcs_vals[sci_ext]['Y_OFFSET']        = _yoff
                    chip_wcs_vals[sci_ext]['CRPIX2_COMBINED'] = _crpix2_0 + _yoff

                    if gdc is not None:
                        #transform ref coordinate to GDC frame (0-indexed input)
                        _rx, _ry, _ = apply_gdc(_crpix1_0,
                                                 _crpix2_0 + _yoff,
                                                 gdc)
                        chip_wcs_vals[sci_ext]['CRPIX1_GDC'] = float(_rx[0])
                        chip_wcs_vals[sci_ext]['CRPIX2_GDC'] = float(_ry[0])

                except Exception:
                    chip_wcs[sci_ext] = None
    except Exception:
        pass

    # --- GDC ---
    if gdc is not None:
        x_comb = np.array([r.x + getattr(r, '_x_offset', 0.0) for r in records])
        y_comb = np.array([r.y + getattr(r, '_y_offset', 0.0) for r in records])

        x_gdc, y_gdc, mc = apply_gdc(x_comb, y_comb, gdc)
        J_gdc = _gdc_jacobian_batch(x_comb, y_comb, gdc)

        for i, r in enumerate(records):
            cov_xy = r.cov[1:3, 1:3]   # 2×2 block: cov(x,y)
            cov_g = J_gdc[i] @ cov_xy @ J_gdc[i].T
            r._x_gdc = float(x_gdc[i])
            r._y_gdc = float(y_gdc[i])
            r._mc    = float(mc[i])
            r._cov_gdc = cov_g
    else:
        for r in records:
            r._x_gdc = np.nan
            r._y_gdc = np.nan
            r._mc    = 0.0
            r._cov_gdc = None

    # Group records by chip for batch WCS transforms
    from collections import defaultdict
    chip_idx = defaultdict(list)
    for i, r in enumerate(records):
        chip_idx[getattr(r, '_chip_ext', 0)].append(i)

    ra_all  = np.full(n, np.nan)
    dec_all = np.full(n, np.nan)
    cov_radec_all = [None] * n

    for sci_ext, idxs in chip_idx.items():
        wcs_obj = chip_wcs.get(sci_ext)
        if wcs_obj is None:
            continue

        x_chip = np.array([records[i].x for i in idxs])
        y_chip = np.array([records[i].y for i in idxs])

        # x_chip = np.array([records[i]._x_gdc for i in idxs])
        # y_chip = np.array([records[i]._y_gdc for i in idxs])
        # CRPIX1 = chip_wcs_vals[sci_ext]['CRPIX1_GDC']
        # CRPIX2 = chip_wcs_vals[sci_ext]['CRPIX2_GDC']
        # CRVAL1 = chip_wcs_vals[sci_ext]['CRVAL1']
        # CRVAL2 = chip_wcs_vals[sci_ext]['CRVAL2']
        # CD1_1,CD1_2,CD2_1,CD2_2 = chip_wcs_vals[sci_ext]['CD1_1'],\
        #                           chip_wcs_vals[sci_ext]['CD1_2'],\
        #                           chip_wcs_vals[sci_ext]['CD2_1'],\
        #                           chip_wcs_vals[sci_ext]['CD2_2']

        # dxgc = x_chip-CRPIX1
        # dygc = y_chip-CRPIX2
        # dRA = dxgc*CD1_1 + dygc*CD1_2
        # dDE = dxgc*CD2_1 + dygc*CD2_2

        # ra,dec = xy2radec(dRA,dDE,CRVAL1,CRVAL2)

        # J_wcs = np.array([[CD1_1,CD1_2],
        #                   [CD2_1,CD2_2]])

        # for k, i in enumerate(idxs):
        #     ra_all[i]  = ra[k]
        #     dec_all[i] = dec[k]
        #     cov_radec_all[i] = J_wcs @ records[i]._cov_gdc @ J_wcs.T

        try:
            ra, dec = _wcs_transform_batch(x_chip, y_chip, wcs_obj)
            J_wcs   = _wcs_jacobian_batch(x_chip, y_chip, wcs_obj)
            for k, i in enumerate(idxs):
                ra_all[i]  = ra[k]
                dec_all[i] = dec[k]
                cov_xy = records[i].cov[1:3, 1:3]
                cov_radec_all[i] = J_wcs[k] @ cov_xy @ J_wcs[k].T
        except Exception as exc:
            if verbose:
                warnings.warn(f"WCS transform failed for ext {sci_ext}: {exc}")

    for i, r in enumerate(records):
        r._ra  = float(ra_all[i])
        r._dec = float(dec_all[i])
        r._cov_radec = cov_radec_all[i]
        r._chip_wcs  = chip_wcs_vals.get(getattr(r, '_chip_ext', None), {})


# ---------------------------------------------------------------------------
# Image loading (single chip)
# ---------------------------------------------------------------------------

def load_image(path, sci_ext=1, dq_ext=None, dq_flags=None):
    """Load one science chip from a FITS file.

    For ACS/WFC and WFC3/UVIS FLC/FLT files the routine automatically
    determines the detector-coordinate y_offset needed for PSF grid lookup:
    - ACS/WFC ext 1 (chip 2, bottom): y_offset = 0
    - ACS/WFC ext 4 (chip 1, top):    y_offset = 2048
    - WFC3/UVIS ext 1 (chip 2, bottom): y_offset = 0
    - WFC3/UVIS ext 4 (chip 1, top):    y_offset = 2051

    Parameters
    ----------
    path     : path to FITS file
    sci_ext  : science extension number (default 1)
    dq_ext   : DQ extension number, or None to skip masking
    dq_flags : list of DQ flag values to treat as bad (None = any non-zero)

    Returns
    -------
    data       : 2D float64 science array
    gain       : float, detector gain in e-/DN
    read_noise : float, detector read noise in e-
    mask       : 2D bool array (True = bad pixel), or None
    x_offset   : float, detector x offset for PSF grid lookup
    y_offset   : float, detector y offset for PSF grid lookup
    """
    with fits.open(path) as hdul:
        primary_hdr = hdul[0].header
        sci_hdr = hdul[sci_ext].header
        data = np.array(hdul[sci_ext].data, dtype=np.float64)
        instrume = primary_hdr.get('INSTRUME', '').strip()
        detector = primary_hdr.get('DETECTOR', '').strip()
        noise_info = _get_noise_info(primary_hdr, sci_hdr, instrume)
        mask = _load_dq_mask(hdul, dq_ext, dq_flags)

    x_offset, y_offset = _detector_offsets(instrume, detector, sci_ext)
    return (data, noise_info['effective_gain'], noise_info['read_noise'],
            mask, x_offset, y_offset)


def _load_dq_mask(hdul, dq_ext, dq_flags):
    if dq_ext is None:
        return None
    try:
        dq = np.array(hdul[dq_ext].data, dtype=np.int32)
        if dq_flags is None:
            return dq != 0
        mask = np.zeros(dq.shape, dtype=bool)
        for f in dq_flags:
            mask |= (dq & int(f)) != 0
        return mask
    except (IndexError, KeyError):
        warnings.warn(f"Could not load DQ extension {dq_ext}.")
        return None


def _get_noise_info(primary_hdr, sci_hdr, instrume):
    """Return a dict with full noise parameter provenance.

    Keys
    ----
    hardware_gain   : e-/DN as read from the header (informational)
    effective_gain  : gain to use in the noise model — 1.0 when data is already
                      in electrons (BUNIT = ELECTRONS), hardware_gain otherwise
    read_noise      : e-  (always in electrons regardless of image units)
    bunit           : BUNIT string from the science extension header
    data_in_electrons : True when the pipeline has already applied the gain
    gain_from_header  : True when hardware_gain came from a header keyword
    rn_from_header    : True when read_noise came from a header keyword
    """
    instrume = instrume.strip().upper()
    bunit = (sci_hdr.get('BUNIT', '') if sci_hdr is not None else '').strip().upper()
    # FLC/FLT files from calacs/calwf3 have BUNIT = 'ELECTRONS'; the gain
    # conversion has already been applied so the noise model must use gain = 1.
    _electron_bunits = {'ELECTRONS', 'ELECTRONS/S', 'ELECTRON', 'E-', 'E'}
    data_in_electrons = bunit in _electron_bunits

    if instrume in ('ACS', 'WFC3'):
        gain_a_raw = primary_hdr.get('ATODGNA', None)
        gain_b_raw = primary_hdr.get('ATODGNB', None)
        gain_from_header = gain_a_raw is not None
        gain_a = float(gain_a_raw) if gain_a_raw is not None \
                 else float(primary_hdr.get('CCDGAIN', 2.0))
        gain_b = float(gain_b_raw) if gain_b_raw is not None else gain_a
        hardware_gain = (gain_a + gain_b) / 2.0

        rn_a_raw = primary_hdr.get('READNSEA', None)
        rn_b_raw = primary_hdr.get('READNSEB', None)
        rn_from_header = rn_a_raw is not None
        rn_a = float(rn_a_raw) if rn_a_raw is not None else 5.0
        rn_b = float(rn_b_raw) if rn_b_raw is not None else rn_a
        read_noise = (rn_a + rn_b) / 2.0
    else:
        gain_raw = primary_hdr.get('CCDGAIN', primary_hdr.get('GAIN', None))
        gain_from_header = gain_raw is not None
        hardware_gain = float(gain_raw) if gain_raw is not None else 1.0

        rn_raw = primary_hdr.get('RDNOISE', primary_hdr.get('READNOISE', None))
        rn_from_header = rn_raw is not None
        read_noise = float(rn_raw) if rn_raw is not None else 5.0

    effective_gain = 1.0 if data_in_electrons else hardware_gain

    return {
        'hardware_gain':    hardware_gain,
        'effective_gain':   effective_gain,
        'read_noise':       read_noise,
        'bunit':            bunit or 'UNKNOWN',
        'data_in_electrons': data_in_electrons,
        'gain_from_header': gain_from_header,
        'rn_from_header':   rn_from_header,
    }


def _get_noise_params(hdr, instrume):
    """Thin wrapper — returns (effective_gain, read_noise) for backward compat."""
    info = _get_noise_info(hdr, None, instrume)
    return info['effective_gain'], info['read_noise']


def _detector_offsets(instrume, detector, sci_ext):
    instrume = instrume.upper()
    detector = detector.upper()
    key = (instrume, detector)
    config = _CHIP_CONFIG.get(key)
    if config is None:
        return 0.0, 0.0
    for (sext, _dext, y_off) in config:
        if sext == sci_ext:
            return 0.0, y_off
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# High-level convenience: run on all chips at once
# ---------------------------------------------------------------------------

def run_photometry_fits(
    image_path,
    psf_path,
    psf_scale=None,
    half_width=3,
    sky_inner=4,
    sky_outer=8,
    hmin=4,
    fmin_thresh=70.0,
    max_iter_fit=5,
    tol=1e-4,
    n_passes=1,
    gain=None,
    read_noise=None,
    zero_point=0.0,
    dq_flags=None,
    sat_threshold=None,
    verbose=False,
    sigma_clip=True,
    sigma_clip_sigma=4.0,
    sigma_clip_iter=2,
    return_residual=False,
    gdc_path=None,
    lib_dir=None,
    backend='auto',
    conc_limit=0.9,
    psf_delta=None,
    mag_st_max=28.0,
    **kwargs,
):
    """Run PSF-fitting photometry on all science chips of a FITS image.

    Parameters
    ----------
    image_path : path to science FITS file (FLC, FLT, etc.)
    psf_path   : path to STDPSF FITS file, OR a directory in which the
                 appropriate PSF will be auto-detected from the image header.
                 For ACS/WFC the SM-era suffix (_SM3/_SM4) is selected
                 automatically from the MJD-OBS header keyword.
    psf_scale  : override PSF supersampling factor (auto-read from PSF file if None)
    dq_flags   : list of DQ flag values to mask (None = all non-zero flags)
    gdc_path   : path to STDGDC FITS file, OR a directory in which the GDC
                 file will be auto-detected.  If None and lib_dir is given,
                 looks in ``lib_dir/STDGDCs/{det_prefix}/``.  If None and no
                 lib_dir, GDC correction is skipped.
    lib_dir    : root library directory (e.g. ``./lib``).  When set, PSFs are
                 found in ``lib_dir/STDPSFs/{det_prefix}/`` and GDCs in
                 ``lib_dir/STDGDCs/{det_prefix}/`` if not overridden by
                 *psf_path*/*gdc_path*.
    psf_delta  : (psf_size, psf_size) ndarray or None.  If given, added to
                 every PSF in the cube (psf_cube += delta[np.newaxis]) before
                 fitting.  Produced by measure_psf_perturbation / psf_delta.npy.
    fmin_thresh: float.  Hard lower bound on the detection flux threshold in
                 electrons (default 40).  The effective fmin is
                 max(fmin_from_mag(mag_st_max), fmin_thresh).
    mag_st_max : float.  Faint ST-magnitude limit (default 28).  Converted to a
                 flux threshold per chip using PHOTFLAM and EXPTIME; floored at
                 fmin_thresh.  When PHOTFLAM/EXPTIME are missing, fmin_thresh is
                 used directly.
    **kwargs   : passed to run_photometry (noise_map, mask, etc.)

    Returns
    -------
    records : list of StarRecord (all chips combined)
    If return_residual is True: (records, {sci_ext: residual_array}, {sci_ext: var_array})

    Notes
    -----
    When a valid GDC file is found, each StarRecord gains private attributes
    ``_x_gdc``, ``_y_gdc``, ``_mc``, ``_cov_gdc``, ``_ra``, ``_dec``,
    ``_cov_radec`` used by catalog_to_table to populate the extra columns.
    """
    from .core import run_photometry

    # --- Read instrument/detector from primary header ---
    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header
        instrume = primary_hdr.get('INSTRUME', '').strip()
        detector = primary_hdr.get('DETECTOR', '').strip()
        _primary_hdr_saved = primary_hdr

    det_prefix = _DETECTOR_PREFIX.get((instrume.upper(), detector.upper()), '')

    # --- Resolve PSF path (with lib_dir fallback) ---
    _psf_path = psf_path
    if _psf_path is None and lib_dir is not None:
        _psf_path = os.path.join(lib_dir, 'STDPSFs', det_prefix)
    if _psf_path is None:
        raise ValueError("Must supply psf_path or lib_dir.")

    if os.path.isdir(_psf_path):
        _psf_path = find_psf(_psf_path, primary_hdr)
        if verbose:
            try:
                filt = _extract_filter(primary_hdr, instrume)
            except ValueError:
                filt = 'unknown'
            print(f"Instrument: {instrume}/{detector}, Filter: {filt}")
            print(f"Auto-selected PSF: {os.path.basename(_psf_path)}")
    elif verbose:
        try:
            filt = _extract_filter(primary_hdr, instrume)
        except ValueError:
            filt = 'unknown'
        print(f"Instrument: {instrume}/{detector}, Filter: {filt}")

    psf_cube, xs, ys, psf_scale_file, grid_shape = load_stdpsf(_psf_path)
    if psf_scale is None:
        psf_scale = psf_scale_file

    if psf_delta is not None:
        psf_cube = psf_cube + psf_delta[np.newaxis, :, :]

    # --- Resolve GDC path ---
    gdc = None
    _gdc_path = gdc_path
    if _gdc_path is None and lib_dir is not None:
        _gdc_dir = os.path.join(lib_dir, 'STDGDCs', det_prefix)
        _gdc_path = find_gdc(_gdc_dir, primary_hdr) if os.path.isdir(_gdc_dir) else None
    elif _gdc_path is not None and os.path.isdir(_gdc_path):
        _gdc_path = find_gdc(_gdc_path, primary_hdr)

    if _gdc_path is not None and os.path.exists(_gdc_path):
        try:
            gdc = load_stdgdc(_gdc_path)
            if verbose:
                print(f"GDC file: {os.path.basename(_gdc_path)}")
        except Exception as exc:
            warnings.warn(f"Failed to load GDC {_gdc_path}: {exc}")
    elif verbose and (gdc_path is not None or lib_dir is not None):
        warnings.warn("No GDC file found; skipping geometric distortion correction.")

    chips = get_chip_config_from_fits(image_path, instrume, detector)

    all_records = []
    residuals   = {}
    var_images  = {}
    for sci_ext, dq_ext, y_offset in chips:
        with fits.open(image_path) as _hdul:
            _sci_hdr = _hdul[sci_ext].header
        _noise_info = _get_noise_info(_primary_hdr_saved, _sci_hdr, instrume)

        # Per-chip gain: respect user override, otherwise use effective_gain from
        # the science extension header (1.0 for BUNIT=ELECTRONS, hardware_gain for COUNTS).
        gain_use = gain if gain is not None else _noise_info['effective_gain']
        rn_use   = read_noise if read_noise is not None else _noise_info['read_noise']

        # Photometric calibration zero-points (chip-specific, from science extension header).
        # PHOTFLAM [erg/cm²/Å/e⁻] and PHOTZPT define STMAG; PHOTPLAM [Å] adds ABMAG.
        # EXPTIME is in the primary header.  All quantities may be absent (e.g. drizzled
        # mosaics), in which case the ZP attributes stay NaN and the calibrated mag
        # columns in the output table are filled with NaN.
        _photflam = _sci_hdr.get('PHOTFLAM', None)
        _photzpt  = _sci_hdr.get('PHOTZPT',  -21.10)
        _photplam = _sci_hdr.get('PHOTPLAM', None)
        _exptime  = float(_primary_hdr_saved.get('EXPTIME', 0.0))
        if _photflam is not None and _exptime > 0.0:
            # ZP such that m_ST = -2.5*log10(flux_electrons) + _zp_st
            # (i.e. flux_electrons is total over the exposure, not per second)
            _zp_st = -2.5 * np.log10(_photflam) + _photzpt + 2.5 * np.log10(_exptime)
            if _photplam is not None:
                _c_ang = 2.998e18   # speed of light in Å/s
                _zp_ab = (-2.5 * np.log10(_photflam * _photplam**2 / _c_ang)
                          - 48.6 + 2.5 * np.log10(_exptime))
            else:
                _zp_ab = np.nan
        else:
            _zp_st = np.nan
            _zp_ab = np.nan

        # Effective fmin = max(fmin_from_mag(mag_st_max), fmin_thresh).
        # mag_st_max sets the target depth; fmin_thresh is the hard floor.
        if np.isfinite(_zp_st):
            _fmin_from_mag  = 10 ** ((_zp_st - mag_st_max) / 2.5)
            _fmin_effective = max(_fmin_from_mag, fmin_thresh)
        else:
            _fmin_from_mag  = None
            _fmin_effective = fmin_thresh

        if verbose:
            _g_src = 'header' if _noise_info['gain_from_header'] else 'default'
            _rn_src = 'header' if _noise_info['rn_from_header'] else 'default'
            _g_note = (' [image in electrons — noise gain = 1.0]'
                       if _noise_info['data_in_electrons'] else ' e-/DN')
            print(f"\nChip: sci_ext={sci_ext}, dq_ext={dq_ext}, y_offset={y_offset}")
            print(f"  BUNIT         : {_noise_info['bunit']}")
            print(f"  Hardware gain : {_noise_info['hardware_gain']:.4f} e-/DN  [{_g_src}]")
            print(f"  Noise gain    : {_noise_info['effective_gain']:.4f}{_g_note}")
            print(f"  Read noise    : {_noise_info['read_noise']:.2f} e-  [{_rn_src}]")
            if not np.isfinite(_zp_st):
                print(f"  fmin          : {_fmin_effective:.1f} e-  "
                      f"(fmin_thresh; mag_st_max={mag_st_max:.2f} ignored — "
                      f"PHOTFLAM/EXPTIME missing)")
            elif _fmin_from_mag >= fmin_thresh:
                print(f"  fmin          : {_fmin_effective:.1f} e-  "
                      f"(mag_st_max={mag_st_max:.2f})")
            else:
                print(f"  fmin          : {_fmin_effective:.1f} e-  "
                      f"(fmin_thresh floor; mag_st_max={mag_st_max:.2f} → "
                      f"{_fmin_from_mag:.1f} e- in {_exptime:.0f}s)")

        data, _g, _rn, mask, x_offset, _ = load_image(
            image_path, sci_ext=sci_ext, dq_ext=dq_ext, dq_flags=dq_flags)

        # Load raw DQ integer array for per-star DQ stats and to rebuild the
        # boolean mask excluding warm pixels (bit 4), which are mild enough to
        # use in peak detection, sky estimation, and PSF fitting.
        _dq_array = None
        if dq_ext is not None:
            try:
                with fits.open(image_path) as _dqh:
                    _dq_array = np.array(_dqh[dq_ext].data, dtype=np.int32)
                mask = (_dq_array & ~np.int32(4)) != 0
            except Exception:
                pass
        _peak_mask = mask

        result = run_photometry(
            data=data,
            psf_models=psf_cube,
            psf_positions=(xs, ys),
            psf_scale=psf_scale,
            half_width=half_width,
            sky_inner=sky_inner,
            sky_outer=sky_outer,
            hmin=hmin,
            fmin=_fmin_effective,
            max_iter_fit=max_iter_fit,
            tol=tol,
            n_passes=n_passes,
            gain=gain_use,
            read_noise=rn_use,
            zero_point=zero_point,
            mask=mask,
            peak_mask=_peak_mask,
            verbose=verbose,
            x_offset=x_offset,
            y_offset=y_offset,
            sat_threshold=(sat_threshold if sat_threshold is not None else np.inf),
            sigma_clip=sigma_clip,
            sigma_clip_sigma=sigma_clip_sigma,
            sigma_clip_iter=sigma_clip_iter,
            return_residual=return_residual,
            _apply_chi2_inflation=False,
            _classify=False,
            backend=backend,
            conc_limit=conc_limit,
            **kwargs,
        )
        if return_residual:
            records, chip_residual, chip_var = result
            residuals[sci_ext] = chip_residual
            var_images[sci_ext] = chip_var
        else:
            records = result

        for r in records:
            r._chip_ext  = sci_ext
            r._x_offset  = x_offset
            r._y_offset  = y_offset
            r._zp_st     = _zp_st
            r._zp_ab     = _zp_ab
            r._exptime   = _exptime

        n_conv = sum(r.converged for r in records)
        all_records.extend(records)
        if verbose:
            n_chip = len(records)
            nc_str = (f" ({n_chip - n_conv} non-converged excluded)"
                      if n_chip > n_conv else "")
            print(f"  -> {n_conv} converged stars on chip ext={sci_ext}{nc_str}")

    # Drop non-converged stars before classification and chi² scaling so that
    # bad fits do not pollute the concentration locus or the chi² magnitude bins.
    n_before = len(all_records)
    all_records = [r for r in all_records if r.converged]
    n_removed = n_before - len(all_records)
    if verbose and n_removed:
        print(f"  Removed {n_removed} non-converged star(s) before classification "
              f"(hit max_iter without convergence)")

    # Classify stars once on the full multi-chip catalogue so the concentration
    # locus is built from both detectors together, giving a better-sampled
    # magnitude-adaptive boundary especially at the bright end.
    from .core import classify_stars as _classify_stars
    n_star_cand = _classify_stars(all_records, conc_lo=conc_limit)
    if verbose:
        n_tot = len(all_records)
        print(f"  Star classification (combined catalogue): "
              f"{n_star_cand}/{n_tot} sources classified as likely stars "
              f"({100.0 * n_star_cand / max(n_tot, 1):.1f}%)")

    # Apply chi²-inflation once across the combined catalogue
    from .core import inflate_chi2
    inflate_chi2(all_records, zero_point, verbose=verbose)

    # Compute per-star DQ flag summaries (1×1, 2×2, 3×3 windows).
    # Uses the raw DQ integer array so all flag bits are preserved.
    from .core import compute_dq_stats
    for sci_ext, dq_ext_i, _y_off in chips:
        _chip_recs = [r for r in all_records
                      if getattr(r, '_chip_ext', sci_ext) == sci_ext]
        if _chip_recs and _dq_array is not None:
            compute_dq_stats(_chip_recs, _dq_array, x_offset=x_offset)

    # Apply GDC and WCS corrections
    _apply_gdc_wcs(all_records, gdc, image_path, chips, instrume, detector,
                   verbose=verbose)

    if return_residual:
        return all_records, residuals, var_images, _psf_path, _gdc_path
    return all_records, _psf_path, _gdc_path


# ---------------------------------------------------------------------------
# Catalog output
# ---------------------------------------------------------------------------

def catalog_to_table(records, zero_point=0.0,
                     sigma_floor_x=0.0, sigma_floor_y=0.0, eps_flux=0.0,
                     floor_params=None):
    """Convert a list of StarRecord to an astropy Table.

    Includes raw pixel positions (x, y), GDC-corrected J-frame positions
    (x_gdc, y_gdc), photometrically corrected magnitude (mag_gdc), propagated
    GDC covariance, WCS sky coordinates (ra, dec), and propagated RA/Dec
    covariance.  GDC/WCS columns contain NaN when the correction was not
    applied (e.g. when called via the Python API without a GDC file).
    """
    _cols = ['x', 'y', 'flux', 'flux_err', 'sky', 'sky_err', 'mag', 'mag_err',
             'qfit', 'chi2', 'central_res', 'n_sat', 'psf_frac', 'psf_peak',
             'peak', 'pass_number',
             'n_neighbors', 'dist_nearest', 'dist_nearest_brighter',
             'cov_ff', 'cov_xx', 'cov_yy', 'cov_ss',
             'cov_fx', 'cov_fy', 'cov_fs', 'cov_xy', 'cov_xs', 'cov_ys',
             'n_iter', 'converged', 'delta_max', 'chi2_scale', 'eps_psf',
             'concentration', 'concentration_2x2', 'concentration_3x3',
             'n_conc_1x1', 'n_conc_2x2', 'n_conc_3x3',
             'is_star_candidate',
             'sigma_x_model', 'sigma_y_model', 'sigma_f_model', 'chip_ext',
             'x_gdc', 'y_gdc', 'mag_gdc', 'mag_err_gdc',
             'cov_xx_gdc', 'cov_yy_gdc', 'cov_xy_gdc',
             'ra', 'dec', 'ra_err', 'dec_err',
             'cov_ra_ra', 'cov_dec_dec', 'cov_ra_dec',
             'mag_st', 'mag_ab', 'mag_st_gdc']

    if not records:
        dtypes = ([float] * 8 +          # x y flux flux_err sky sky_err mag mag_err
                  [float, float, float] + # qfit chi2 central_res
                  [int, float, float] +   # n_sat psf_frac psf_peak
                  [float, int] +          # peak pass_number
                  [int, float, float] +   # n_neighbors dist_nearest dist_nearest_brighter
                  [float] * 10 +          # cov_ff..cov_ys
                  [int, bool, float, float, float] +  # n_iter converged delta_max chi2_scale eps_psf
                  [float, float, float] +           # concentration concentration_2x2 concentration_3x3
                  [int, int, int, bool] +           # n_conc_1x1 n_conc_2x2 n_conc_3x3 is_star_candidate
                  [float, float, float, int] +  # sigma_x/y/f_model chip_ext
                  [float] * 14 +          # x_gdc y_gdc mag_gdc mag_err_gdc cov_gdc ra dec ra_err dec_err cov_radec
                  [float] * 3)            # mag_st mag_ab mag_st_gdc
        return Table(names=_cols, dtype=dtypes)

    def _c(i, j):
        return np.array([r.cov[i, j] for r in records])

    def _gattr(name, default=np.nan):
        return np.array([getattr(r, name, default) for r in records])

    def _cov_gdc(i, j):
        out = np.full(len(records), np.nan)
        for k, r in enumerate(records):
            cg = getattr(r, '_cov_gdc', None)
            if cg is not None and np.isfinite(cg).all():
                out[k] = cg[i, j]
        return out

    def _cov_radec(i, j):
        out = np.full(len(records), np.nan)
        for k, r in enumerate(records):
            cr = getattr(r, '_cov_radec', None)
            if cr is not None and np.isfinite(cr).all():
                out[k] = cr[i, j]
        return out

    # --- Systematic floor: additive variance on top of chi²-scaled cov -------
    _flux_arr = np.array([r.flux for r in records])
    _floor_ff = (eps_flux * np.where(_flux_arr > 0, _flux_arr, 1.0)) ** 2
    _floor_xx = sigma_floor_x ** 2
    _floor_yy = sigma_floor_y ** 2
    # Recompute flux_err and mag_err with floor applied.
    _cov_ff_floor = _c(0, 0) + _floor_ff
    _cov_xx_floor = _c(1, 1) + _floor_xx
    _cov_yy_floor = _c(2, 2) + _floor_yy
    _flux_err_floor = np.sqrt(np.maximum(_cov_ff_floor, 0.0))
    _log10e = 2.5 / np.log(10.0)
    _mag_err_floor = np.where(
        _flux_arr > 0, _log10e * _flux_err_floor / _flux_arr, np.nan)

    # Per-star 3-component noise model sigma values from the fitted floor parameters.
    # sigma = sqrt( floor² + (A·10^(0.2·m))² + (C·10^(0.4·m))² )
    _mag_arr = np.array([r.mag for r in records])
    _popt_x = _popt_y = _popt_f = None
    if floor_params is not None and floor_params.get('fit_A_ok'):
        _popt_x = floor_params.get('popt_x')
        _popt_y = floor_params.get('popt_y')
        _popt_f = floor_params.get('popt_f')

    def _eval_noise_model(mag, popt):
        if popt is None:
            return np.full(len(mag), np.nan)
        log10_floor, log10_A, log10_C = popt
        return np.sqrt(
            (10.0 ** log10_floor) ** 2
            + (10.0 ** log10_A * 10.0 ** (0.2 * mag)) ** 2
            + (10.0 ** log10_C * 10.0 ** (0.4 * mag)) ** 2
        )

    _sigma_x_model = _eval_noise_model(_mag_arr, _popt_x)
    _sigma_y_model = _eval_noise_model(_mag_arr, _popt_y)
    _sigma_f_model = _eval_noise_model(_mag_arr, _popt_f)

    # Photometrically corrected mag using MGC
    mag_gdc = np.array([r.mag + getattr(r, '_mc', 0.0) for r in records])
    mag_err_gdc = _mag_err_floor

    # Calibrated magnitudes: STMAG and ABMAG.
    # ZP computed from PHOTFLAM/PHOTZPT/PHOTPLAM/EXPTIME in the science extension header.
    # m_ST = -2.5*log10(flux_electrons) + zp_st  (where zp_st folds in EXPTIME)
    # mag_st_gdc adds the GDC pixel-area correction (mc) on top.
    # All three share the same flux uncertainty as mag/mag_err.
    # NaN when calibration keywords are absent (e.g. Python API, drizzled images).
    _instr_mag = -2.5 * np.log10(np.where(_flux_arr > 0, _flux_arr, np.nan))
    _zp_st_arr = np.array([getattr(r, '_zp_st', np.nan) for r in records])
    _zp_ab_arr = np.array([getattr(r, '_zp_ab', np.nan) for r in records])
    _mc_arr    = np.array([getattr(r, '_mc',    0.0)    for r in records])
    mag_st     = np.where(np.isfinite(_zp_st_arr), _instr_mag + _zp_st_arr, np.nan)
    mag_ab     = np.where(np.isfinite(_zp_ab_arr), _instr_mag + _zp_ab_arr, np.nan)
    mag_st_gdc = np.where(np.isfinite(_zp_st_arr), mag_st + _mc_arr,        np.nan)

    # RA/Dec uncertainties in arcsec: sqrt(var) * 3600
    cov_ra_ra   = _cov_radec(0, 0)
    cov_dec_dec = _cov_radec(1, 1)
    cov_ra_dec  = _cov_radec(0, 1)
    ra_err  = np.where(np.isfinite(cov_ra_ra),
                       np.sqrt(np.maximum(cov_ra_ra,  0.0)) * 3600.0, np.nan)
    dec_err = np.where(np.isfinite(cov_dec_dec),
                       np.sqrt(np.maximum(cov_dec_dec, 0.0)) * 3600.0, np.nan)

    t = Table({
        'x':            np.array([r.x + getattr(r, '_x_offset', 0.0) for r in records]),
        'y':            np.array([r.y + getattr(r, '_y_offset', 0.0) for r in records]),
        'flux':         _flux_arr,
        'flux_err':     _flux_err_floor,
        'sky':          np.array([r.sky for r in records]),
        'sky_err':      np.array([r.sky_err for r in records]),
        'mag':          np.array([r.mag for r in records]),
        'mag_err':      _mag_err_floor,
        'qfit':         np.array([r.qfit for r in records]),
        'chi2':         np.array([r.chi2 for r in records]),
        'central_res':  np.array([r.central_res for r in records]),
        'n_sat':        np.array([r.n_sat for r in records], dtype=int),
        'psf_frac':     np.array([r.psf_frac for r in records]),
        'psf_peak':     np.array([r.psf_peak for r in records]),
        'peak':         np.array([r.peak for r in records]),
        'pass_number':  np.array([r.pass_number for r in records], dtype=int),
        'n_neighbors':           np.array([r.n_neighbors for r in records], dtype=int),
        'dist_nearest':          np.array([r.dist_nearest for r in records]),
        'dist_nearest_brighter': np.array([r.dist_nearest_brighter for r in records]),
        'cov_ff':       _cov_ff_floor,
        'cov_xx':       _cov_xx_floor,
        'cov_yy':       _cov_yy_floor,
        'cov_ss':       _c(3, 3),
        'cov_fx':       _c(0, 1),
        'cov_fy':       _c(0, 2),
        'cov_fs':       _c(0, 3),
        'cov_xy':       _c(1, 2),
        'cov_xs':       _c(1, 3),
        'cov_ys':       _c(2, 3),
        'n_iter':       np.array([getattr(r, 'n_iter',    0)    for r in records], dtype=int),
        'converged':    np.array([getattr(r, 'converged', True) for r in records], dtype=bool),
        'delta_max':    np.array([getattr(r, 'delta_max',   0.0) for r in records]),
        'chi2_scale':   np.array([getattr(r, 'chi2_scale',  1.0) for r in records]),
        'eps_psf':        np.array([getattr(r, 'eps_psf', 0.0) for r in records]),
        'concentration':      np.array([getattr(r, 'concentration',      np.nan) for r in records]),
        'concentration_2x2': np.array([getattr(r, 'concentration_2x2',  np.nan) for r in records]),
        'concentration_3x3': np.array([getattr(r, 'concentration_3x3',  np.nan) for r in records]),
        'n_conc_1x1': np.array([getattr(r, 'n_conc_1x1', 0) for r in records], dtype=int),
        'n_conc_2x2': np.array([getattr(r, 'n_conc_2x2', 0) for r in records], dtype=int),
        'n_conc_3x3': np.array([getattr(r, 'n_conc_3x3', 0) for r in records], dtype=int),
        'is_star_candidate': np.array([getattr(r, 'is_star_candidate', True) for r in records],
                                      dtype=bool),
        'dq_1x1': np.array([getattr(r, 'dq_1x1', 0) for r in records], dtype=np.int32),
        'dq_2x2': np.array([getattr(r, 'dq_2x2', 0) for r in records], dtype=np.int32),
        'dq_3x3': np.array([getattr(r, 'dq_3x3', 0) for r in records], dtype=np.int32),
        'sigma_x_model':  _sigma_x_model,
        'sigma_y_model':  _sigma_y_model,
        'sigma_f_model':  _sigma_f_model,
        'chip_ext':       np.array([getattr(r, '_chip_ext', 0) for r in records], dtype=int),
        # GDC-corrected positions
        'x_gdc':        _gattr('_x_gdc'),
        'y_gdc':        _gattr('_y_gdc'),
        'mag_gdc':      mag_gdc,
        'mag_err_gdc':  mag_err_gdc,
        'cov_xx_gdc':   _cov_gdc(0, 0) + _floor_xx,
        'cov_yy_gdc':   _cov_gdc(1, 1) + _floor_yy,
        'cov_xy_gdc':   _cov_gdc(0, 1),
        # WCS sky coordinates
        'ra':           _gattr('_ra'),
        'dec':          _gattr('_dec'),
        'ra_err':       ra_err,
        'dec_err':      dec_err,
        'cov_ra_ra':    cov_ra_ra,
        'cov_dec_dec':  cov_dec_dec,
        'cov_ra_dec':   cov_ra_dec,
        # Calibrated magnitudes (NaN when PHOTFLAM/EXPTIME not available)
        'mag_st':       mag_st,
        'mag_ab':       mag_ab,
        'mag_st_gdc':   mag_st_gdc,
    })
    t.meta['ZP']            = zero_point
    t.meta['SIGMA_FLOOR_X'] = sigma_floor_x
    t.meta['SIGMA_FLOOR_Y'] = sigma_floor_y
    t.meta['EPS_FLUX']      = eps_flux

    # Per-chip WCS reference pixel metadata.
    # Keys stored: CRPIX1, CRPIX2 (raw FITS 1-indexed); CRPIX2_COMBINED (= CRPIX2 + y_offset);
    # Y_OFFSET; CRPIX1_GDC, CRPIX2_GDC (GDC-corrected, only when GDC was applied);
    # CRVAL1, CRVAL2 (RA/Dec at reference pixel); CD matrix elements.
    _chip_wcs_seen = {}
    for r in records:
        _cext = getattr(r, '_chip_ext', 0)
        if _cext not in _chip_wcs_seen:
            _chip_wcs_seen[_cext] = getattr(r, '_chip_wcs', {})
    _wcs_meta_keys = ('CRPIX1', 'CRPIX2', 'CRPIX2_COMBINED', 'Y_OFFSET',
                      'CRPIX1_GDC', 'CRPIX2_GDC',
                      'CRVAL1', 'CRVAL2',
                      'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
    # Collect per-chip photometric calibration from records for metadata.
    _chip_phot_seen = {}
    for r in records:
        _cext = getattr(r, '_chip_ext', 0)
        if _cext not in _chip_phot_seen:
            _chip_phot_seen[_cext] = {
                'ZP_ST':    getattr(r, '_zp_st',  np.nan),
                'ZP_AB':    getattr(r, '_zp_ab',  np.nan),
                'EXPTIME':  getattr(r, '_exptime', np.nan),
            }

    for _cext, _wv in sorted(_chip_wcs_seen.items()):
        _pfx = f'CHIP{_cext}_'
        for _k in _wcs_meta_keys:
            if _k in _wv:
                t.meta[_pfx + _k] = _wv[_k]
        # Photometric calibration metadata for this chip
        _ph = _chip_phot_seen.get(_cext, {})
        for _k, _v in _ph.items():
            if np.isfinite(_v):
                t.meta[_pfx + _k] = float(_v)

    # Units
    for col in ('x', 'y', 'dist_nearest', 'dist_nearest_brighter', 'x_gdc', 'y_gdc'):
        t[col].unit = 'pix'
    for col in ('cov_xx', 'cov_yy', 'cov_xy', 'cov_xs', 'cov_ys',
                'cov_xx_gdc', 'cov_yy_gdc', 'cov_xy_gdc'):
        t[col].unit = 'pix2'
    for col in ('mag', 'mag_err', 'mag_gdc', 'mag_err_gdc', 'mag_st', 'mag_ab', 'mag_st_gdc'):
        t[col].unit = 'mag'
    for col in ('ra', 'dec'):
        t[col].unit = 'deg'
    for col in ('ra_err', 'dec_err'):
        t[col].unit = 'arcsec'
    for col in ('cov_ra_ra', 'cov_dec_dec', 'cov_ra_dec'):
        t[col].unit = 'deg2'

    _desc = {
        'x':                    'Sub-pixel x position (0-based detector coords)',
        'y':                    'Sub-pixel y position (0-based combined-frame; y_offset added for multi-chip)',
        'flux':                 'Fitted source flux',
        'flux_err':             '1-sigma flux uncertainty from covariance',
        'sky':                  'Local sky background per pixel (fitted simultaneously)',
        'sky_err':              '1-sigma sky uncertainty from covariance',
        'mag':                  'Instrumental magnitude = ZP - 2.5*log10(flux)',
        'mag_err':              '1-sigma magnitude uncertainty',
        'qfit':                 'Quality of fit: sum|res|/sum|data-sky|; <0.1 excellent, >0.5 poor',
        'chi2':                 'Scaled reduced chi-sq: sqrt(sum(r^2/var)/(N-4)); ~1 ideal',
        'central_res':          'Normalised central-pixel residual (Fortran C/cc)',
        'n_sat':                'Pixels in fit window above sat_threshold (Fortran n)',
        'psf_frac':             'PSF value at fitted sub-pixel position (Fortran f)',
        'psf_peak':             'PSF value at perfect centre (Fortran F)',
        'peak':                 'Peak pixel value above sky at rounded position',
        'pass_number':          'Photometry pass in which this star was detected',
        'n_neighbors':          'Other detected stars within hw pixels',
        'dist_nearest':         'Distance to nearest other detected star',
        'dist_nearest_brighter':'Distance to nearest detected star with higher flux',
        'cov_ff':               'Covariance: var(flux)',
        'cov_xx':               'Covariance: var(x) in raw pixel coords',
        'cov_yy':               'Covariance: var(y) in raw pixel coords',
        'cov_ss':               'Covariance: var(sky)',
        'cov_fx':               'Covariance: cov(flux, x)',
        'cov_fy':               'Covariance: cov(flux, y)',
        'cov_fs':               'Covariance: cov(flux, sky)',
        'cov_xy':               'Covariance: cov(x, y) in raw pixel coords',
        'cov_xs':               'Covariance: cov(x, sky)',
        'cov_ys':               'Covariance: cov(y, sky)',
        'n_iter':               'Newton iterations taken to converge',
        'converged':            'True if Newton loop converged within max_iter',
        'delta_max':            'Maximum position step |δx| or |δy| at the final Newton iteration',
        'chi2_scale':           'Chi²-scaling factor applied to covariance',
        'eps_psf':              'Per-star implied fractional PSF model error: chi2 / sqrt(flux * psf_frac * gain)',
        'concentration':        '1×1 concentration: peak pixel / (flux * psf_frac); stars ~1.0, CRs > 1, galaxies < 1',
        'concentration_2x2':   '2×2 concentration: 4-pixel sum / (flux * PSF sum); stars ~1.0, more robust than 1×1',
        'concentration_3x3':   '3×3 concentration: 9-pixel sum / (flux * PSF sum); stars ~1.0, most robust morphological metric',
        'n_conc_1x1':          'Number of unmasked pixels used in the 1×1 concentration (0 or 1)',
        'n_conc_2x2':          'Number of unmasked pixels used in the 2×2 concentration (0–4); <2 gives NaN',
        'n_conc_3x3':          'Number of unmasked pixels used in the 3×3 concentration (0–9); <5 gives NaN',
        'is_star_candidate':    'True if source morphology and fit quality are consistent with a point source (star)',
        'dq_1x1':               'Bitwise OR of raw DQ flag values at the fitted (x,y) pixel',
        'dq_2x2':               'Bitwise OR of raw DQ flags in the 2×2 region at (x,y),(x+1,y),(x,y+1),(x+1,y+1)',
        'dq_3x3':               'Bitwise OR of raw DQ flags in the 3×3 region centred on fitted (x,y)',
        'sigma_x_model':        'Model σ_x at this star magnitude from 3-component noise fit (NaN if fit unavailable)',
        'sigma_y_model':        'Model σ_y at this star magnitude from 3-component noise fit (NaN if fit unavailable)',
        'sigma_f_model':        'Model fractional flux error at this star magnitude from 3-component noise fit (NaN if fit unavailable)',
        'chip_ext':             'FITS SCI extension (chip) this star was measured on',
        'x_gdc':                'GDC-corrected x position in Anderson J-frame',
        'y_gdc':                'GDC-corrected y position in Anderson J-frame',
        'mag_gdc':              'Pixel-area corrected magnitude: mag + MGC (distortion-corrected)',
        'mag_err_gdc':          '1-sigma uncertainty of mag_gdc',
        'cov_xx_gdc':           'GDC-propagated var(x_gdc)',
        'cov_yy_gdc':           'GDC-propagated var(y_gdc)',
        'cov_xy_gdc':           'GDC-propagated cov(x_gdc, y_gdc)',
        'ra':                   'Right ascension from per-chip WCS (J2000)',
        'dec':                  'Declination from per-chip WCS (J2000)',
        'ra_err':               '1-sigma RA uncertainty propagated from position covariance',
        'dec_err':              '1-sigma Dec uncertainty propagated from position covariance',
        'cov_ra_ra':            'WCS-propagated var(RA)',
        'cov_dec_dec':          'WCS-propagated var(Dec)',
        'cov_ra_dec':           'WCS-propagated cov(RA, Dec)',
        'mag_st':               'STMAG calibrated magnitude: -2.5*log10(PHOTFLAM*flux/EXPTIME)+PHOTZPT; NaN if header keywords absent',
        'mag_ab':               'ABMAG calibrated magnitude: computed from PHOTFLAM, PHOTPLAM, EXPTIME; NaN if header keywords absent',
        'mag_st_gdc':           'STMAG + GDC pixel-area correction (mag_st + MGC); most physically complete magnitude',
    }
    for col, desc in _desc.items():
        if col in t.colnames:
            t[col].description = desc

    return t
