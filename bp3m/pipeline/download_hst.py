"""
Step 2: Search MAST and download HST (or future JWST) images for a sky region.

Uses astroquery.mast to find FLC exposures, presents the observations table
for review, and downloads selected products to:
    {output_dir}/{field_name}/HST/mastDownload/HST/{obs_id}/{obs_id}_flc.fits

The directory layout produced here is exactly what bp3m's data_loader_flc and
the cross-matcher expect.

Extension note
--------------
JWST support will be added by passing telescope='JWST' and the appropriate
instrument list. The download path will then use:
    {output_dir}/{field_name}/JWST/mastDownload/...
The psf_fitting and cross_match modules accept a ``telescope`` argument that
selects instrument-specific behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astroquery.mast import Observations
from tqdm import tqdm   # module level: the force-redownload path used it before the local import (UnboundLocalError)


# Instruments supported per telescope (MAST instrument_name values).
# JWST support adapted from Liwen Chen's bp3m fork (download_jwst.py / download_hst.py,
# 2026-06..09): instrument names, the mode-qualified MAST query names, the post-Gaia
# time-window logic and the JWST failed-observation header rules come from there.
_INSTRUMENTS = {
    'HST':  ['ACS/WFC', 'WFC3/UVIS', 'ACS/HRC'],
    'JWST': ['NIRCAM', 'NIRISS', 'MIRI'],
}

# MAST's instrument_name for JWST imaging is mode-qualified ('NIRCAM/IMAGE'); the bare
# name matches nothing in query_criteria (Liwen Chen: instrument_name=['MIRI'] -> 0 rows,
# ['MIRI/IMAGE'] -> thousands).  Only the query uses these; the returned rows are
# normalised back to the bare name by _bare_instrument().
_JWST_INSTRUMENT_NAME_MAST = {
    'NIRCAM': 'NIRCAM/IMAGE',
    'NIRISS': 'NIRISS/IMAGE',
    'MIRI':   'MIRI/IMAGE',
}

# Default image product per telescope: HST calibrated, CTE-corrected exposures; JWST
# stage-2 calibrated per-detector exposures (one _cal.fits per detector per exposure).
_DEFAULT_IM_TYPE = {'HST': '_flc', 'JWST': '_cal'}

# Map MAST instrument_name → STDPSFs/STDGDCs subdirectory name (bp3m-setup layout:
# one directory per instrument, JWST files flattened into NIRCAM/NIRISS/MIRI).
_INST_TO_LIBDIR = {
    'ACS/WFC':   'ACSWFC',
    'WFC3/UVIS': 'WFC3UV',
    'ACS/HRC':   'ACSHRC',
    'NIRCAM':    'NIRCAM',
    'NIRISS':    'NIRISS',
    'MIRI':      'MIRI',
}


def _bare_instrument(name) -> str:
    """'NIRCAM/IMAGE' -> 'NIRCAM'; HST names ('ACS/WFC') are returned unchanged."""
    s = str(name)
    return s.split('/')[0] if s.endswith('/IMAGE') else s

# Default Gaia DR3 reference epoch as MJD (2017-05-28)
_GAIA_DR3_MJD = Time('2017-05-28').mjd

# Normalise filter names from PSF/GDC filenames to MAST canonical names.
# STScI PSF files occasionally use abbreviated names that differ from MAST.
_PSF_FILTER_NORM = {
    'F850L':  'F850LP',   # STDPSF_ACSWFC_F850L.fits → MAST F850LP
    'F475Wx': 'F475W',    # STScI ships WFC3/UVIS F475W as STDPSF_WFC3UV_F475Wx.fits; without this
                          # mapping UVIS F475W was never a PSF+GDC combo and its images were
                          # excluded from download/fitting (NGC_55: 15 images, found 2026-09-30)
}

def _normalise_filter(name: str) -> str:
    """Map a PSF/GDC filename filter token to the MAST canonical name."""
    return _PSF_FILTER_NORM.get(name, name)


def _clean_mast_filter(raw: str) -> str:
    """Return the science filter from a MAST filter string, dropping CLEAR entries.

    MAST returns paired-filter strings like 'F814W;CLEAR1L' or 'CLEAR2L;F606W'.
    We split on ';', drop any token that is empty or starts with 'CLEAR', and
    return the first remaining token.  Falls back to the raw string if nothing
    survives the filter.

    JWST uses the same rule: NIRCam strings are 'FILTER;PUPIL' and the PSF/GDC
    library is keyed on the filter wheel (first token; a filter+pupil pair such as
    'F150W2;F162M' has no library entry and is dropped by the PSF+GDC gate),
    NIRISS strings are 'CLEAR;F200W' (filter wheel CLEAR, the pupil carries the
    band, which is what the library uses), MIRI strings are a single token.
    """
    tokens = [t.strip() for t in raw.split(';')]
    science = [t for t in tokens if t and not t.upper().startswith('CLEAR')]
    return science[0] if science else raw.strip()


def _query_params_sidecar(hst_dir: Path, field_name: str) -> Path:
    return hst_dir / f"{field_name}_obs_params.json"


def _make_query_params(
    ra, dec, search_width, search_height,
    hst_filters, t_exptime_min, t_exptime_max,
    time_baseline_days, date_second_epoch_mjd,
    obs_date_min, obs_date_max, im_type, telescope, instruments,
    lib_dir, target_name=None,
) -> dict:
    return {
        "ra":                    ra,
        "dec":                   dec,
        "search_width":          search_width,
        "search_height":         search_height,
        "hst_filters":           sorted(hst_filters) if hst_filters else None,
        "t_exptime_min":         t_exptime_min,
        "t_exptime_max":         float(t_exptime_max) if np.isfinite(t_exptime_max) else None,
        "time_baseline_days":    time_baseline_days,
        "date_second_epoch_mjd": date_second_epoch_mjd,
        "obs_date_min":          obs_date_min,
        "obs_date_max":          obs_date_max,
        "im_type":               im_type,
        "telescope":             telescope.upper(),
        "instruments":           sorted(instruments) if instruments else None,
        "lib_dir":               str(lib_dir) if lib_dir else None,
        "download_aux":          True,
        "target_name":           (sorted(target_name) if isinstance(target_name, (list, tuple))
                                  else target_name) if target_name else None,
    }


def get_available_psf_gdc_combos(lib_dir: str | Path) -> dict[str, set[str]]:
    """
    Scan a lib/ directory to find instrument+filter combinations that
    have BOTH a STDPSF and a STDGDC file.

    Parameters
    ----------
    lib_dir : path to lib/ directory containing STDPSFs/ and STDGDCs/

    Returns
    -------
    dict mapping MAST instrument_name → set of filter strings that have both
    PSF and GDC.  E.g. ``{'ACS/WFC': {'F606W', 'F814W', ...}, ...}``
    """
    lib_dir = Path(lib_dir)
    psf_root = lib_dir / "STDPSFs"
    gdc_root = lib_dir / "STDGDCs"

    if not psf_root.exists() or not gdc_root.exists():
        return {}

    # Build reverse map: libdir name → MAST instrument name
    libdir_to_inst = {v: k for k, v in _INST_TO_LIBDIR.items()}

    result: dict[str, set[str]] = {}

    for psf_sub in sorted(psf_root.iterdir()):
        if not psf_sub.is_dir():
            continue
        det = psf_sub.name                              # e.g. 'ACSWFC'
        inst = libdir_to_inst.get(det)
        if inst is None:
            continue

        gdc_sub = gdc_root / det
        if not gdc_sub.exists():
            continue

        # Collect filters with a PSF file (ignore _SM3/_SM4 variants and 'vintage')
        psf_filters: set[str] = set()
        for f in psf_sub.glob("STDPSF_*.fits"):
            parts = f.stem.split('_')
            # STDPSF_ACSWFC_F814W, STDPSF_ACSWFC_F814W_SM4, STDPSF_ACSWFC_F850L_SM3
            # SM-suffixed files are valid PSFs (e.g. F850L only exists as SM3).
            # Using a set means multiple variants for the same filter don't double-count.
            if len(parts) >= 3 and parts[-1] != 'vintage':
                psf_filters.add(_normalise_filter(parts[2]))

        # Collect filters with a GDC file: plain STDGDC_<det>_<filt> and the
        # STDGDC_OFFICIAL_JFRAME_<det>_<filt> tables pypass.io.find_gdc prefers.
        # (Until 2026-09-25 the OFFICIAL names were skipped here, so once the ACS/WFC
        # library held only OFFICIAL tables the MAST query asked for ACS/WFC F775W
        # only -- the one filter still carrying a plain-named table.)
        gdc_filters: set[str] = set()
        for f in gdc_sub.glob("STDGDC_*.fits"):
            parts = f.stem.split('_')
            if 'VFRAME' in parts or parts[-1] == 'vintage':
                continue
            if f.stat().st_size < 2880:
                # an HTML error page saved under a .fits name (seen in a JWST library
                # copy) must not count as availability
                continue
            if len(parts) >= 3:
                gdc_filters.add(_normalise_filter(parts[-1]))

        common = psf_filters & gdc_filters
        if common:
            result[inst] = common

    return result


def search_mast(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    hst_filters: list[str] | None = None,
    project: list[str] | None = None,
    t_exptime_min: float = 2.0,
    t_exptime_max: float = np.inf,
    time_baseline_days: float | None = None,
    date_second_epoch_mjd: float = _GAIA_DR3_MJD,
    obs_date_min: str | None = None,
    obs_date_max: str | None = None,
    im_type: str | None = None,
    telescope: str = 'HST',
    instruments: list[str] | None = None,
    available_combos: dict[str, set[str]] | None = None,
    include_recent: bool = False,
    target_name: list[str] | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query MAST for science images of a sky region (one telescope per call).

    include_recent: keep observations taken within the last year (normally cut, since
    they are usually still exclusive-access); for proprietary data with a MAST login.
    target_name   : keep only observations whose MAST target_name contains one of these
                    substrings (case-insensitive); e.g. the LMC calibration fields.

    Telescope differences (JWST logic from Liwen Chen's fork): JWST queries use the
    mode-qualified instrument names, keep only the CAL product (no DRZ/SPT/JIT/JIF),
    count exposures per detector-independent exposure stem, and — because every JWST
    image postdates the Gaia DR3 epoch — a time_baseline_days requirement raises the
    earliest allowed date instead of lowering the latest one; t_baseline is reported
    as a positive number of years after the Gaia epoch.

    Parameters
    ----------
    im_type         : product suffix; None selects the telescope default
                      (_DEFAULT_IM_TYPE: '_flc' for HST, '_cal' for JWST).
    time_baseline_days: minimum HST–Gaia time baseline in days. None means no
                        minimum (all images up to date_second_epoch_mjd are kept).
    obs_date_min    : earliest observation date to include (ISO string, e.g. '2005-01-01').
                      None means no lower bound.
    obs_date_max    : latest observation date to include (ISO string). None means no upper bound.
    instruments     : list of MAST instrument_name values to restrict to
                      (e.g. ['ACS/WFC']). None uses all supported instruments.
    available_combos: output of get_available_psf_gdc_combos(); if provided,
                      filters to only instrument+filter combos that have PSF+GDC.

    Returns
    -------
    obs_table          : one row per observation set (with i_exptime, t_baseline)
    data_products_table: one row per image file (with URI, parent_obsid, …)
    """
    tel = telescope.upper()
    is_jwst = tel == 'JWST'
    if im_type is None:
        im_type = _DEFAULT_IM_TYPE.get(tel, '_flc')
    allowed_inst = _INSTRUMENTS.get(tel)
    if allowed_inst is None:
        raise ValueError(f"Unsupported telescope '{telescope}'. "
                         f"Choose from {list(_INSTRUMENTS)}")

    # Apply user instrument filter
    if instruments is not None:
        requested = [i.upper() for i in instruments]
        allowed_inst = [i for i in allowed_inst
                        if i.replace('/', '').upper() in requested
                        or i.upper() in requested]
        if not allowed_inst:
            raise ValueError(f"No supported instruments match {instruments}. "
                             f"Available: {_INSTRUMENTS[telescope.upper()]}")

    # Apply PSF+GDC availability filter
    if available_combos:
        allowed_inst = [i for i in allowed_inst if i in available_combos]
        if not allowed_inst:
            print("  WARNING: No instruments have both PSF and GDC files in lib_dir.")

    if hst_filters is None or hst_filters == ['any']:
        if available_combos:
            # Union of all filters available across all allowed instruments
            hst_filters = sorted(set.union(*[available_combos[i]
                                              for i in allowed_inst
                                              if i in available_combos]) or set())
        if not hst_filters:
            if is_jwst:
                # filters with both a PSF and a GDC in the STScI JWST1PASS library (2026-10)
                hst_filters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F158M', 'F182M',
                               'F200W', 'F210M', 'F212N', 'F277W', 'F356W', 'F380M', 'F430M',
                               'F444W', 'F480M', 'F560W', 'F770W', 'F1000W']
            else:
                hst_filters = ['F435W', 'F475W', 'F555W', 'F606W',
                               'F625W', 'F658N', 'F775W', 'F814W', 'F850LP']
    if project is None:
        project = [tel]
    # MAST needs the mode-qualified JWST instrument names in the query
    query_inst = [_JWST_INSTRUMENT_NAME_MAST.get(i, i) for i in allowed_inst] if is_jwst else allowed_inst

    # Shrink box slightly to avoid edge artefacts
    cos_dec = np.cos(np.deg2rad(dec))
    margin_ra  = 0.056 / cos_dec
    margin_dec = 0.056
    ra1  = ra  - search_width  / 2 + margin_ra
    ra2  = ra  + search_width  / 2 - margin_ra
    dec1 = dec - search_height / 2 + margin_dec
    dec2 = dec + search_height / 2 - margin_dec

    # Build MAST time bounds
    t_min_bound = 0
    if obs_date_min is not None:
        t_min_bound = Time(obs_date_min).mjd

    # Latest date: the last year is exclusive-access unless include_recent.  For HST a
    # time-baseline requirement lowers the latest date (images predate Gaia DR3); for
    # JWST (all images after the Gaia epoch) it raises the earliest date instead.
    if include_recent:
        t_max_mjd = Time.now().mjd + 1.0
    else:
        t_max_mjd = (Time.now()-366*u.day).mjd
    if time_baseline_days is not None:
        if is_jwst:
            t_min_bound = max(t_min_bound, _GAIA_DR3_MJD + time_baseline_days)
        else:
            t_max_mjd = _GAIA_DR3_MJD - time_baseline_days

    print(f"  Querying MAST (this can take a minute)...")
    import time as _time
    _mast_retries = 5
    _mast_delay   = 10  # seconds between retries
    for _attempt in range(_mast_retries):
        try:
            obs_raw = Observations.query_criteria(
                dataproduct_type=['image'],
                obs_collection=[tel],
                s_ra=[ra1, ra2],
                s_dec=[dec1, dec2],
                instrument_name=query_inst,
                t_max=[t_min_bound, t_max_mjd],
                filters=hst_filters,
                project=project,
            )
            break
        except Exception as _e:
            if _attempt < _mast_retries - 1:
                print(f"  MAST query failed (attempt {_attempt+1}/{_mast_retries}): {_e}")
                print(f"  Retrying in {_mast_delay}s ...")
                _time.sleep(_mast_delay)
                _mast_delay *= 2  # exponential back-off
            else:
                raise

    if len(obs_raw) == 0:
        return pd.DataFrame(), pd.DataFrame()

    _delay2 = 10
    for _attempt in range(_mast_retries):
        try:
            prod_raw = Observations.get_product_list(obs_raw)
            break
        except Exception as _e:
            if _attempt < _mast_retries - 1:
                print(f"  MAST get_product_list failed (attempt {_attempt+1}/{_mast_retries}): {_e}")
                print(f"  Retrying in {_delay2}s ...")
                _time.sleep(_delay2)
                _delay2 *= 2
            else:
                raise
    im_sub   = im_type[1:].upper()   # '_flc' → 'FLC', '_cal' → 'CAL'
    _AUX_TYPES = {'SPT', 'JIT', 'JIF'} if not is_jwst else set()
    _sub = prod_raw['productSubGroupDescription']
    _aux_mask = (_sub == 'SPT') | (_sub == 'JIT') | (_sub == 'JIF')
    if is_jwst:
        # JWST: the per-detector CAL exposures only (no DRZ equivalent, no HST support files)
        mask = (_sub == im_sub) & (prod_raw['obs_collection'] == tel)
    else:
        mask = (
            ((_sub == im_sub) | (_sub == 'DRZ') | _aux_mask) &
            (prod_raw['obs_collection'] == tel)
        )
    prod_df = prod_raw[mask].to_pandas()
    obs_df  = obs_raw.to_pandas()
    # 'NIRCAM/IMAGE' -> 'NIRCAM' so library combos, tables and --instruments agree
    if is_jwst and 'instrument_name' in obs_df.columns:
        obs_df['instrument_name'] = obs_df['instrument_name'].map(_bare_instrument)

    # Drop HAP pipeline products
    prod_df = prod_df[~prod_df['project'].str.contains('HAP', na=False)]

    # Count exposures per observation to compute individual exposure time.
    # HST: one FLC per exposure, MAST t_exptime = total over the exposures.
    # JWST: one CAL file per DETECTOR per exposure (NIRCam: 8-10 files per dither) while
    # MAST t_exptime is the total per detector over the dithers (Draco 04513: 601 s =
    # 4 x EFFEXPTM 150 s), so count distinct exposure stems (file name without the
    # detector suffix), not files.
    _im_rows = prod_df[prod_df['productSubGroupDescription'] == im_sub]
    if is_jwst:
        _stem = (_im_rows['productFilename'].astype(str)
                 .str.replace(r'_[a-z0-9]+_cal\.fits$', '', regex=True))
        n_exp = (_im_rows.assign(_stem=_stem)
                 .groupby('parent_obsid')['_stem'].nunique().rename('n_exp'))
    else:
        n_exp = (_im_rows.groupby('parent_obsid')['parent_obsid']
                 .count()
                 .rename('n_exp'))
    obs_df['obsid'] = obs_df['obsid'].astype(str)
    n_exp.index = n_exp.index.astype(str)
    obs_df = obs_df.merge(n_exp.rename_axis('obsid'), on='obsid', how='inner')
    obs_df['i_exptime'] = obs_df['t_exptime'] / obs_df['n_exp']

    obs_time = Time(obs_df['t_max'].values, format='mjd')
    obs_time.format = 'iso'; obs_time.out_subfmt = 'date'
    obs_df['obs_time']   = obs_time.value
    # years between the image and the Gaia epoch, positive for both telescopes
    _sign = -1.0 if is_jwst else 1.0
    obs_df['t_baseline'] = np.round(
        _sign * (date_second_epoch_mjd - obs_df['t_max'].values) / 365.2422, 2)
    obs_df['filters'] = obs_df['filters'].apply(_clean_mast_filter)

    if target_name:
        _names = [target_name] if isinstance(target_name, str) else list(target_name)
        _tgt = obs_df['target_name'].astype(str)
        _keep = np.zeros(len(obs_df), dtype=bool)
        for _n in _names:
            _keep |= _tgt.str.contains(_n, case=False, na=False, regex=False).to_numpy()
        print(f"  target_name filter {_names}: {int(_keep.sum())}/{len(obs_df)} observations kept")
        obs_df = obs_df[_keep]

    # Without a MAST login, exclusive-access products cannot be downloaded: drop them
    # here so they do not count as selected images (Liwen Chen's fork; with a token the
    # 366-day rule / --include_proprietary decides instead).
    if not _MAST_LOGGED_IN and 'dataRights' in prod_df.columns:
        _priv = prod_df['dataRights'].astype(str).str.upper().ne('PUBLIC')
        if _priv.any():
            _priv_obs = set(prod_df.loc[_priv, 'parent_obsid'].astype(str))
            print(f"  {int(_priv.sum())} exclusive-access product(s) in {len(_priv_obs)} observation(s) "
                  f"dropped (no MAST login)")
            prod_df = prod_df[~_priv]
            obs_df = obs_df[~obs_df['obsid'].astype(str).isin(_priv_obs)]

    # Merge exposure-time info into products table
    meta = obs_df[['obsid', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']]
    prod_df['parent_obsid'] = prod_df['parent_obsid'].astype(str)
    prod_df = prod_df.merge(
        meta.rename(columns={'obsid': 'parent_obsid'}), on='parent_obsid', how='left')

    # Filter by exposure time and (optionally) time baseline.
    # Auxiliary products (SPT/JIT/JIF) have no i_exptime — keep them unconditionally.
    t_base_yr = time_baseline_days / 365.2422 if time_baseline_days is not None else -np.inf
    obs_df = obs_df[
        (obs_df['i_exptime'] >= t_exptime_min) &
        (obs_df['i_exptime'] <= t_exptime_max) &
        (obs_df['t_baseline'] >= t_base_yr)
    ]
    _prod_is_aux = prod_df['productSubGroupDescription'].isin(_AUX_TYPES)
    prod_df = prod_df[
        _prod_is_aux | (
            (prod_df['i_exptime'] >= t_exptime_min) &
            (prod_df['i_exptime'] <= t_exptime_max) &
            (prod_df['t_baseline'] >= t_base_yr)
        )
    ]

    # Post-query date filter for obs_date_max
    if obs_date_max is not None:
        t_max_iso = Time(obs_date_max).mjd
        obs_df  = obs_df[obs_df['t_max'] <= t_max_iso]
        keep_ids = set(obs_df['obsid'].astype(str))
        # Aux products (SPT/JIT/JIF) follow the science observation filter.
        _prod_is_aux2 = prod_df['productSubGroupDescription'].isin(_AUX_TYPES)
        prod_df = prod_df[
            (_prod_is_aux2 & prod_df['parent_obsid'].isin(keep_ids)) |
            (~_prod_is_aux2 & prod_df['t_baseline'].notna() &
             prod_df['parent_obsid'].isin(keep_ids))
        ]

    # Filter products to only PSF+GDC-available instrument+filter combos
    if available_combos:
        def _combo_ok(row):
            inst_name = row.get('instrument_name', '')
            filt_name = row.get('filters', '')
            if inst_name not in available_combos:
                return False
            return filt_name in available_combos[inst_name]

        if 'instrument_name' in obs_df.columns and 'filters' in obs_df.columns:
            mask_obs = obs_df.apply(_combo_ok, axis=1).astype(bool)
            dropped = obs_df[~mask_obs]
            if not dropped.empty:
                dropped_combos = sorted(set(
                    f"{r['instrument_name']}/{r['filters']}"
                    for _, r in dropped.iterrows()
                ))
                print(f"  WARNING: {len(dropped)} observation(s) dropped — no PSF+GDC "
                      f"in lib_dir for: {', '.join(dropped_combos)}")
            obs_df  = obs_df[mask_obs]
            keep_ids = set(obs_df['obsid'].astype(str))
            prod_df = prod_df[prod_df['parent_obsid'].isin(keep_ids)]

    return obs_df.reset_index(drop=True), prod_df.reset_index(drop=True)


_MAST_LOGGED_IN = False


def mast_login() -> bool:
    """Log in to MAST for exclusive-access (proprietary) data, if a token is available.

    Token source, first found wins: $MAST_API_TOKEN, $BP3M_HOME/mast_token, ~/.mast_token
    (create at https://auth.mast.stsci.edu/token with the 'mast:exclusive_access' scope;
    keep the file chmod 600).  The token is never printed.  Without a token, public data
    download as before and exclusive-access products fail with an authorisation error.
    """
    global _MAST_LOGGED_IN
    if _MAST_LOGGED_IN:
        return True
    import os
    tok = os.environ.get('MAST_API_TOKEN', '').strip()
    src = 'MAST_API_TOKEN'
    if not tok:
        home = Path(os.environ['BP3M_HOME']) if 'BP3M_HOME' in os.environ else Path.home() / '.bp3m'
        for p in (home / 'mast_token', Path.home() / '.mast_token'):
            if p.exists():
                tok, src = p.read_text().strip(), str(p)
                break
    if not tok:
        return False
    try:
        Observations.login(token=tok)
        _MAST_LOGGED_IN = True
        print(f"  MAST: logged in for exclusive-access data (token from {src})")
    except Exception as e:
        print(f"  WARNING: MAST login failed ({type(e).__name__}) — exclusive-access products will not download")
    return _MAST_LOGGED_IN


def download_hst_images(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    output_dir: Path,
    field_name: str,
    hst_filters: list[str] | None = None,
    project: list[str] | None = None,
    t_exptime_min: float = 2.0,
    t_exptime_max: float = np.inf,
    time_baseline_days: float | None = None,
    date_second_epoch_mjd: float = _GAIA_DR3_MJD,
    obs_date_min: str | None = None,
    obs_date_max: str | None = None,
    im_type: str | None = None,
    telescope: str = 'HST',
    instruments: list[str] | None = None,
    lib_dir: str | Path | None = None,
    gaia_df: 'pd.DataFrame | None' = None,
    n_processes: int = 4,
    field_ids: list[int] | None = None,
    quiet: bool = False,
    force_redownload: bool = True,
    mast_refresh_days: int | None = 30,
    skip_mast_download: bool = False,
    extra_pointings: "list[tuple[float, float, float, float]] | None" = None,
    delve_csv_path: 'str | Path | None' = None,
    include_recent: bool = False,
    target_name: list[str] | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Search MAST for images of a field and download them (one telescope per call;
    bp3m_run loops over --telescope HST JWST so each telescope keeps its own root).

    Creates:
        {output_dir}/{field_name}/{telescope}/mastDownload/{telescope}/{obs_id}/...

    Parameters
    ----------
    obs_date_min    : earliest observation date (ISO string, e.g. '2005-01-01'). None = no limit.
    obs_date_max    : latest observation date (ISO string). None = no limit.
    instruments     : restrict to these instrument/detector names (None = all supported).
    lib_dir         : path to lib/ containing STDPSFs/ and STDGDCs/. If provided, only
                      instruments+filters with both PSF and GDC files are kept.
    gaia_df         : Gaia catalogue DataFrame (used to count stars per footprint). Optional.
    field_ids       : list of 1-based field IDs to download (integers shown in the table).
                      None = all. Pass [0] to skip download and just return the table.
    force_redownload    : if True, re-query MAST and re-download even if cached files exist.
    mast_refresh_days   : re-query MAST if cached obs table is older than this many days
                          (default 30).  Set to None to disable age-based refresh.
    skip_mast_download  : if True, always use the cached obs table regardless of age or
                          query-param changes (overrides mast_refresh_days and force_redownload).
    extra_pointings     : additional (ra, dec, search_width, search_height) tuples to
                          query and merge into the combined obs/products tables.
                          The footprint plot shows all pointings together.

    Returns
    -------
    obs_table, data_products_table  (both pd.DataFrame)
    """
    mast_login()
    tel_upper  = telescope.upper()
    if im_type is None:
        im_type = _DEFAULT_IM_TYPE.get(tel_upper, '_flc')
    hst_dir    = Path(output_dir) / field_name / tel_upper
    hst_dir.mkdir(parents=True, exist_ok=True)
    obs_csv    = hst_dir / f"{field_name}_obs.csv"
    prod_csv   = hst_dir / f"{field_name}_data_products.csv"

    print("\n" + "─"*50)
    print(f"Step 2: Searching MAST for {tel_upper} images")
    print("─"*50)

    # Load PSF+GDC availability if lib_dir provided
    available_combos: dict[str, set[str]] | None = None
    if lib_dir is not None:
        available_combos = get_available_psf_gdc_combos(lib_dir)
        if available_combos:
            total = sum(len(v) for v in available_combos.values())
            print(f"  PSF+GDC availability: {total} instrument+filter combos in lib_dir")
        else:
            print(f"  WARNING: No PSF+GDC combos found in {lib_dir}")

    # Build the full list of (ra, dec, sw, sh) tuples for this run
    all_pointings = [(ra, dec, search_width, search_height)] + list(extra_pointings or [])
    n_pointings   = len(all_pointings)

    # Per-pointing query params — used for cache validation
    def _pparams(ra_i, dec_i, sw_i, sh_i):
        return _make_query_params(
            ra_i, dec_i, sw_i, sh_i,
            hst_filters, t_exptime_min, t_exptime_max,
            time_baseline_days, date_second_epoch_mjd,
            obs_date_min, obs_date_max, im_type, telescope, instruments, lib_dir,
            target_name=target_name,
        )

    current_params_list = [_pparams(*p) for p in all_pointings]
    params_sidecar = _query_params_sidecar(hst_dir, field_name)

    if skip_mast_download:
        use_cache = obs_csv.exists() and prod_csv.exists()
        if not use_cache:
            print("  WARNING: --skip_mast_download set but no cached obs table found; "
                  "will query MAST.")
    else:
        use_cache = (
            not force_redownload
            and obs_csv.exists() and prod_csv.exists()
            and params_sidecar.exists()
        )

    if use_cache and not skip_mast_download:
        stored = json.loads(params_sidecar.read_text())
        # Sidecar may be a legacy dict (single-pointing) or a list (multi)
        stored_list = stored if isinstance(stored, list) else [stored]
        if len(stored_list) != n_pointings:
            print(f"  Number of pointings changed ({len(stored_list)} → {n_pointings}) "
                  f"— re-querying MAST.")
            use_cache = False
        else:
            for i, (cur, sto) in enumerate(zip(current_params_list, stored_list)):
                diffs = [f"    [{i}] {k}: {sto.get(k)!r} → {cur[k]!r}"
                         for k in cur if cur[k] != sto.get(k)]
                if diffs:
                    lbl = f"pointing {i+1}" if n_pointings > 1 else "query"
                    print(f"  {lbl} params changed — re-querying MAST:")
                    for d in diffs:
                        print(d)
                    use_cache = False
                    break

    if use_cache and not skip_mast_download and mast_refresh_days is not None:
        import time as _time_mod
        cache_age_days = (_time_mod.time() - obs_csv.stat().st_mtime) / 86400
        if cache_age_days > mast_refresh_days:
            print(f"  Cached obs table is {cache_age_days:.1f} days old "
                  f"(threshold {mast_refresh_days} d) — re-querying MAST.")
            use_cache = False

    if use_cache:
        print(f"  Loading cached observation table from {hst_dir}")
        try:
            obs_df  = pd.read_csv(obs_csv)
            prod_df = pd.read_csv(prod_csv)
        except pd.errors.EmptyDataError:
            if skip_mast_download:
                print("  Cached observation table is empty — no HST images found for this field.")
                manifest = hst_dir / f"{field_name}_selected_obsids.json"
                manifest.write_text("[]")
                return
            print("  Cached observation table is empty — re-querying MAST.")
            use_cache = False
    if not use_cache:
        def _do_search(ra_i, dec_i, sw_i, sh_i):
            return search_mast(
                ra_i, dec_i, sw_i, sh_i,
                hst_filters=hst_filters, project=project,
                t_exptime_min=t_exptime_min, t_exptime_max=t_exptime_max,
                time_baseline_days=time_baseline_days,
                date_second_epoch_mjd=date_second_epoch_mjd,
                obs_date_min=obs_date_min, obs_date_max=obs_date_max,
                im_type=im_type, telescope=telescope,
                instruments=instruments,
                available_combos=available_combos,
                include_recent=include_recent,
                target_name=target_name,
            )

        if n_pointings == 1:
            obs_df, prod_df = _do_search(*all_pointings[0])
        else:
            obs_parts, prod_parts = [], []
            for i, (ra_i, dec_i, sw_i, sh_i) in enumerate(all_pointings):
                print(f"  Querying MAST for pointing {i+1}/{n_pointings} "
                      f"(RA={ra_i:.4f}, Dec={dec_i:+.4f}) ...")
                o_i, p_i = _do_search(ra_i, dec_i, sw_i, sh_i)
                obs_parts.append(o_i)
                prod_parts.append(p_i)
            obs_df  = (pd.concat(obs_parts,  ignore_index=True)
                       .drop_duplicates(subset='obsid').reset_index(drop=True))
            prod_df = (pd.concat(prod_parts, ignore_index=True)
                       .drop_duplicates(subset=['parent_obsid', 'productFilename'])
                       .reset_index(drop=True))

        obs_df.to_csv(obs_csv,   index=False)
        prod_df.to_csv(prod_csv, index=False)
        # Sidecar: list when multi-pointing, plain dict for single (legacy compat)
        sidecar_data = (current_params_list[0] if n_pointings == 1
                        else current_params_list)
        params_sidecar.write_text(json.dumps(sidecar_data, indent=2))

    if obs_df.empty:
        print("  No observations found matching the criteria.")
        # Write empty manifest so downstream steps know to exit gracefully.
        manifest = hst_dir / f"{field_name}_selected_obsids.json"
        manifest.write_text("[]")
        return obs_df, prod_df

    # Attach field_id (1-based sequential index shown to the user)
    obs_df = obs_df.reset_index(drop=True)
    obs_df.insert(0, 'field_id', np.arange(1, len(obs_df) + 1))

    # Count Gaia stars in each footprint if catalog is available
    if gaia_df is not None and 's_region' in obs_df.columns:
        obs_df['n_gaia'] = _count_gaia_in_footprints(obs_df, gaia_df)
    elif 'n_gaia' not in obs_df.columns:
        obs_df['n_gaia'] = -1   # unknown

    # Propagate n_gaia and field_id to products table via obsid
    obs_df['obsid'] = obs_df['obsid'].astype(str)
    prod_df['parent_obsid'] = prod_df['parent_obsid'].astype(str)
    id_map = obs_df.set_index('obsid')[['field_id']].rename_axis('parent_obsid')
    prod_df = prod_df.merge(id_map.reset_index(), on='parent_obsid', how='left')

    print(f"\n  Found {len(obs_df)} observation(s):")
    _print_obs_table(obs_df)

    # Save footprint plot — load qso_candidates + vetted qso_anchors if available
    footprint_png = hst_dir / f"{field_name}_footprints.png"
    _qso_df     = None
    _anchor_dfs = []
    _gaia_dir   = Path(output_dir) / field_name / "Gaia"

    # Collect QSO candidates from first pointing (fallback: most-recent file)
    _qso_csv = (_gaia_dir /
        f"{field_name}_ra{ra:.4f}_dec{dec:.4f}"
        f"_w{search_width:.4f}_h{search_height:.4f}_qso_candidates.csv")
    if not _qso_csv.exists():
        _qso_csvs = sorted(_gaia_dir.glob("*_qso_candidates.csv"),
                           key=lambda p: p.stat().st_mtime)
        _qso_csv = _qso_csvs[-1] if _qso_csvs else None
    if _qso_csv is not None and _qso_csv.exists():
        try:
            _qso_df = pd.read_csv(_qso_csv)
        except Exception:
            pass

    # Collect QSO anchors from all pointings
    from bp3m.pipeline.qso_vetting import find_qso_anchors as _fqa
    for _ra_i, _dec_i, _sw_i, _sh_i in all_pointings:
        _anchor_csv = _fqa(_gaia_dir, field_name, _ra_i, _dec_i, _sw_i, _sh_i)
        if _anchor_csv is not None and _anchor_csv.exists():
            try:
                _adf = pd.read_csv(_anchor_csv, dtype={'source_id': 'int64'})
                _adf = _adf[_adf['is_qso_anchor'].fillna(False)]
                if len(_adf) > 0:
                    _anchor_dfs.append(_adf)
            except Exception:
                pass
    _anchor_df = None
    if _anchor_dfs:
        _anchor_df = (pd.concat(_anchor_dfs, ignore_index=True)
                      .drop_duplicates('source_id'))

    # Footprint plot with all search boxes
    _search_boxes = all_pointings if n_pointings > 1 else None
    try:
        plot_footprints(obs_df, footprint_png,
                        gaia_df=gaia_df, qso_df=_qso_df, anchor_df=_anchor_df,
                        field_name=field_name,
                        ra=ra, dec=dec,
                        search_width=search_width,
                        search_height=search_height,
                        search_boxes=_search_boxes)
    except Exception as _e:
        print(f"  WARNING: footprint plot failed — {_e}")

    if delve_csv_path is not None and Path(delve_csv_path).exists():
        _delve_png = Path(output_dir) / field_name / "DELVE" / f"{field_name}_footprints.png"
        try:
            plot_delve_footprint(
                delve_csv=delve_csv_path,
                obs_df=obs_df,
                save_path=_delve_png,
                field_name=field_name,
                ra=ra, dec=dec,
                search_width=search_width,
                search_height=search_height,
                search_boxes=_search_boxes,
            )
        except Exception as _e:
            print(f"  WARNING: DELVE footprint plot failed — {_e}")

    # Select which observations to download
    if field_ids == 'all':
        print("  Downloading all observations.")

    elif field_ids is not None:
        if 0 in field_ids:
            print("  Skipping download (field_id 0).")
            return obs_df, prod_df
        selected_obsids = set(
            obs_df.loc[obs_df['field_id'].isin(field_ids), 'obsid'].astype(str)
        )
        obs_df  = obs_df[obs_df['obsid'].astype(str).isin(selected_obsids)]
        prod_df = prod_df[prod_df['parent_obsid'].isin(selected_obsids)]

    elif not quiet:
        choice = input(
            "\n  Enter field IDs to download (space-separated, e.g. '1 3 5'), "
            "or 'y' for all, 'n' to skip: "
        ).strip()
        if choice.lower() == 'n':
            return obs_df, prod_df
        elif choice.lower() not in ('y', ''):
            try:
                ids = [int(x) for x in choice.split()]
            except ValueError:
                print("  Invalid input — downloading all.")
                ids = list(obs_df['field_id'])
            selected_obsids = set(
                obs_df.loc[obs_df['field_id'].isin(ids), 'obsid'].astype(str)
            )
            obs_df  = obs_df[obs_df['obsid'].astype(str).isin(selected_obsids)]
            prod_df = prod_df[prod_df['parent_obsid'].isin(selected_obsids)]

    # Download FLC products only
    flc_sub = im_type[1:].upper()
    to_dl   = prod_df[prod_df['productSubGroupDescription'] == flc_sub].copy()
    if to_dl.empty:
        print("  No FLC products to download.")
        return obs_df, prod_df

    # Skip already-downloaded files unless force_redownload
    failed_obsids: dict[str, str] = {}  # obs_id → reason (kept on disk, skipped downstream)
    if not force_redownload and 'dataURI' in to_dl.columns:
        from astropy.io import fits
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # tqdm: module-level import (a local import here made the name local to the whole
        # function, so the force-redownload path raised UnboundLocalError, 2026-10-01)

        mast_root  = hst_dir / "mastDownload" / tel_upper
        _cache_path = hst_dir / ".verify_cache.json"

        # mtime-based verification cache: {obs_id/fname: [size, mtime_ns, fail_reason]}
        # Files whose size and mtime match the cache skip the FITS open entirely.
        _vcache: dict = {}
        if _cache_path.exists():
            try:
                _vcache = json.loads(_cache_path.read_text())
            except Exception:
                _vcache = {}

        # Build per-file work items once so threads get plain values (no pandas ops).
        _file_specs = []
        for _idx, _row in to_dl.iterrows():
            _fname    = Path(_row['dataURI']).name
            _obs_id   = _row.get('obs_id', '')
            _dest     = mast_root / _obs_id / _fname
            _exp_size = _row.get('size', None)
            _cache_key = f"{_obs_id}/{_fname}"
            _file_specs.append((_dest, _exp_size, _cache_key, _idx, _obs_id, _fname))

        def _verify_one(spec):
            dest, exp_size, cache_key, row_idx, obs_id, fname = spec

            if not dest.exists():
                return spec, 'missing', None, None

            st        = dest.stat()
            disk_size = st.st_size

            # Fast size check: catches empty files and truncated downloads.
            if disk_size == 0 or (exp_size and disk_size != exp_size):
                reason = ("empty" if disk_size == 0
                          else f"size {disk_size} != expected {exp_size}")
                return spec, 'broken', reason, None

            # mtime cache hit: file unchanged since last verification — skip FITS open.
            cached = _vcache.get(cache_key)
            if cached and cached[0] == disk_size and cached[1] == st.st_mtime_ns:
                if len(cached) >= 4 and cached[3] == _VERIFY_RULES:
                    return spec, 'cached', cached[2], None
                # Verdict made under older rules (no-HDRLET rule, no calibration-exposure
                # rule; 2026-10-01): re-judge from the primary header only (the FITS
                # structure was already verified for this size+mtime).
                _cr = _check_exptime(dest, tel_upper)
                return spec, 'verified', _cr, [disk_size, st.st_mtime_ns, _cr, _VERIFY_RULES]

            # Full FITS verify + failed-observation check in one open.
            # memmap=True avoids loading pixel data into RAM.
            try:
                with fits.open(dest, memmap=True) as hdul:
                    hdul.verify('exception')
                    fail_reason = _failed_reason(hdul[0].header, tel_upper)
            except Exception as e:
                return spec, 'broken', f"FITS error: {e}", None

            new_entry = [disk_size, st.st_mtime_ns, fail_reason, _VERIFY_RULES]
            return spec, 'verified', fail_reason, new_entry

        already        = []
        broken         = []
        _cache_updates = {}

        n_threads = min(n_processes, len(_file_specs))
        with ThreadPoolExecutor(max_workers=n_threads) as _pool:
            _futures = {_pool.submit(_verify_one, s): s for s in _file_specs}
            with tqdm(total=len(_file_specs), desc="  Verifying cached files",
                      unit="file", dynamic_ncols=True) as _pbar:
                for _fut in as_completed(_futures):
                    _pbar.update(1)
                    spec, status, detail, new_entry = _fut.result()
                    dest, _, cache_key, row_idx, obs_id, fname = spec

                    if status == 'missing':
                        pass  # not on disk — will be downloaded

                    elif status == 'broken':
                        tqdm.write(f"  WARNING: {fname} is corrupt ({detail}) "
                                   f"— will re-download.")
                        if dest.exists():
                            dest.unlink()
                        _invalidate_psf_cache(dest)
                        broken.append(fname)

                    elif status in ('cached', 'verified'):
                        if new_entry is not None:
                            _cache_updates[cache_key] = new_entry
                        if detail:   # detail == fail_reason
                            tqdm.write(f"  WARNING: {fname} is a failed observation "
                                       f"({detail}) — skipping all downstream steps.")
                            _invalidate_psf_cache(dest)
                            failed_obsids[obs_id] = detail
                        already.append(row_idx)

        # Persist updated cache entries for future runs.
        _vcache.update(_cache_updates)
        try:
            _cache_path.write_text(json.dumps(_vcache))
        except Exception:
            pass

        if already:
            n_valid = len(already) - len(failed_obsids)
            print(f"  {n_valid} file(s) already cached and verified; skipping re-download.")
        if broken:
            print(f"  {len(broken)} broken file(s) removed; will re-download.")
        to_dl = to_dl.drop(index=already)

    import time as _time

    if to_dl.empty:
        print("  All files already downloaded.")
        if failed_obsids:
            print(f"  NOTE: {len(failed_obsids)} failed observation(s) excluded from processing: "
                  + ", ".join(sorted(failed_obsids)))
        _write_selected_obsids(prod_df, hst_dir, field_name, im_type, failed_obsids)

        # Check whether any aux files are missing before deciding to return early.
        _aux_types_check = {'SPT', 'JIT', 'JIF'}
        _aux_df_check = prod_df[prod_df['productSubGroupDescription'].isin(_aux_types_check)]
        _mast_root_check = hst_dir / "mastDownload" / tel_upper
        _aux_missing = (
            not _aux_df_check.empty
            and 'dataURI' in _aux_df_check.columns
            and any(
                not (_mast_root_check / row.get('obs_id', '') / Path(row['dataURI']).name).exists()
                for _, row in _aux_df_check.iterrows()
            )
        )
        if not _aux_missing:
            return obs_df, prod_df  # FLCs and aux all on disk — nothing to do.
        # Some aux files are missing; fall through to download them without
        # touching PSF caches (FLCs did not change).
    else:
        # force_redownload: FLCs are re-fetched but pypass/cross-match products are
        # KEPT (user 2026-09-25).  The PSF-fit cache compares each FLC's content
        # fingerprint (size + DATE/PROCTIME/CAL_VER) with the one recorded at fit
        # time, so a changed delivery is refit automatically and an identical one
        # is reused.  (Corrupt files and failed observations are still invalidated
        # individually above/below.)
        pass

        print(f"\n  Downloading {len(to_dl)} {im_type} file(s) to {hst_dir}...")
        _dl_delay = 10
        for _dl_attempt in range(5):
            try:
                # cache=False under force_redownload: astroquery otherwise keeps any local file of the
                # "expected size", and a reprocessed FLC usually has the SAME size (HVS3 ibsi03 3.7.2 -> 3.7.3)
                try:
                    Observations.download_products(
                        Table.from_pandas(to_dl), download_dir=str(hst_dir), cache=not force_redownload)
                except Exception:
                    Observations.download_products(to_dl, download_dir=str(hst_dir), cache=not force_redownload)
                break
            except Exception as _e:
                if _dl_attempt < 4:
                    print(f"  Download failed (attempt {_dl_attempt+1}/5): {_e}")
                    print(f"  Retrying in {_dl_delay}s ...")
                    _time.sleep(_dl_delay)
                    _dl_delay *= 2
                else:
                    raise

        print("  Download complete.")

        # Validate newly downloaded files for failed observations (EXPTIME=0).
        if 'dataURI' in to_dl.columns:
            mast_root_nd = hst_dir / "mastDownload" / tel_upper
            for _, row in tqdm(to_dl.iterrows(), total=len(to_dl),
                               desc="  Validating downloaded files", unit="file",
                               dynamic_ncols=True):
                obs_id = row.get('obs_id', '')
                if obs_id in failed_obsids:
                    continue
                fname = Path(row['dataURI']).name
                dest = mast_root_nd / obs_id / fname
                if not dest.exists():
                    continue
                fail_reason = _check_exptime(dest, tel_upper)
                if fail_reason:
                    print(f"  WARNING: {fname} is a failed observation ({fail_reason}) — "
                          f"skipping all downstream steps.")
                    _invalidate_psf_cache(dest)
                    failed_obsids[obs_id] = fail_reason

        if failed_obsids:
            print(f"  NOTE: {len(failed_obsids)} failed observation(s) excluded from processing: "
                  + ", ".join(sorted(failed_obsids)))
        _write_selected_obsids(prod_df, hst_dir, field_name, im_type, failed_obsids)

    if tel_upper != 'HST':
        # SPT/JIT/JIF support files, jitter summaries and guide-star tables are HST
        # products; JWST pointing/guiding diagnostics are not used yet.
        return obs_df, prod_df

    # Download auxiliary products (SPT/JIT/JIF) — no PSF cache invalidation.
    _aux_types = {'SPT', 'JIT', 'JIF'}
    aux_df = prod_df[prod_df['productSubGroupDescription'].isin(_aux_types)].copy()
    if not aux_df.empty and 'dataURI' in aux_df.columns:
        mast_root_aux = hst_dir / "mastDownload" / tel_upper
        need_dl = []
        for _, row in aux_df.iterrows():
            fname  = Path(row['dataURI']).name
            obs_id = row.get('obs_id', '')
            dest   = mast_root_aux / obs_id / fname
            if not dest.exists() or (force_redownload and dest.exists()):
                need_dl.append(row.name)
        if need_dl:
            aux_to_dl = aux_df.loc[need_dl]
            print(f"\n  Downloading {len(aux_to_dl)} auxiliary (SPT/JIT/JIF) file(s)...")
            _aux_delay = 10
            for _aux_attempt in range(5):
                try:
                    try:
                        Observations.download_products(
                            Table.from_pandas(aux_to_dl), download_dir=str(hst_dir))
                    except Exception:
                        Observations.download_products(aux_to_dl, download_dir=str(hst_dir))
                    break
                except Exception as _e:
                    if _aux_attempt < 4:
                        print(f"  Aux download failed (attempt {_aux_attempt+1}/5): {_e}")
                        print(f"  Retrying in {_aux_delay}s ...")
                        _time.sleep(_aux_delay)
                        _aux_delay *= 2
                    else:
                        print(f"  WARNING: auxiliary download failed — {_e}")
                        break
            else:
                print("  Auxiliary download complete.")
        else:
            n_aux = len(aux_df)
            print(f"  {n_aux} auxiliary (SPT/JIT/JIF) file(s) already on disk.")

    # Summarise per-FLC jitter from JIT/JIF files (skips FLCs that already
    # have a jitter_summary.json).
    from bp3m.pipeline.jitter_summary import summarise_jitter
    summarise_jitter(hst_dir)

    # Resolve guide star sky positions and cross-match to Gaia (skips if
    # the CSV already exists).
    from bp3m.pipeline.guide_stars import download_guide_stars
    download_guide_stars(hst_dir, field_name=field_name)

    return obs_df, prod_df


# Filter → display colour mapping (approximate true-colour ordering)
_FILTER_COLORS = {
    'F220W': '#9b59b6', 'F225W': '#8e44ad', 'F275W': '#6c3483',
    'F330W': '#4a235a', 'F336W': '#7d3c98', 'F350LP': '#aab7b8',
    'F390M': '#2471a3', 'F390W': '#2e86c1', 'F435W': '#1a5276',
    'F438W': '#154360', 'F467M': '#1f618d', 'F475W': '#2980b9',
    'F502N': '#148f77', 'F547M': '#1e8449', 'F550M': '#1d8348',
    'F555W': '#27ae60', 'F600LP': '#d4ac0d', 'F606W': '#f1c40f',
    'F621M': '#e67e22', 'F625W': '#d35400', 'F658N': '#cb4335',
    'F660N': '#c0392b', 'F775W': '#e74c3c', 'F814W': '#922b21',
    'F850LP': '#641e16', 'F850L': '#641e16',
}
_DEFAULT_COLOR = '#95a5a6'


def plot_footprints(
    obs_df: pd.DataFrame,
    save_path: str | Path,
    gaia_df: 'pd.DataFrame | None' = None,
    qso_df: 'pd.DataFrame | None' = None,
    anchor_df: 'pd.DataFrame | None' = None,
    field_name: str = '',
    ra: float | None = None,
    dec: float | None = None,
    search_width: float | None = None,
    search_height: float | None = None,
    search_boxes: "list[tuple[float,float,float,float]] | None" = None,
) -> None:
    """
    Plot HST image footprints on the sky with Gaia stars in the background.

    Footprints are coloured by filter, labelled with their field_id, and
    a legend shows which filter maps to which colour.  Saves a PNG to
    ``save_path``.

    Parameters
    ----------
    obs_df        : observations DataFrame with columns field_id, s_region, filters,
                    proposal_id, instrument_name, obs_time.
    save_path     : output PNG path.
    gaia_df       : optional Gaia catalogue; plotted as background scatter.
    qso_df        : optional Gaia qso_candidates catalogue (all Gaia-flagged
                    possible QSOs, before external catalog vetting); plotted as
                    small orange circles.
    anchor_df     : optional vetted QSO anchors (survived Quaia/MILLIQUAS
                    cross-match + astrometric cut); plotted as larger gold stars
                    on top of the raw candidates.
    field_name    : used in the figure title.
    ra, dec       : primary field centre (degrees).  When provided together with
                    search_width / search_height, the axes are fixed to the
                    user-specified search box.
    search_width, search_height : primary search box full-width in degrees.
    search_boxes  : for multi-pointing: list of (ra, dec, sw, sh) tuples, one
                    per pointing.  When given, all boxes are drawn and the plot
                    cutout encompasses the union of all boxes.  Overrides the
                    single ra/dec/search_width/search_height for guard/cutout.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── Background Gaia stars ─────────────────────────────────────────────────
    if gaia_df is not None and len(gaia_df) > 0:
        gmag = gaia_df['gmag'].values if 'gmag' in gaia_df.columns else None
        ax.scatter(gaia_df['ra'].values, gaia_df['dec'].values,
                   c=gmag, cmap='Greys', vmin=16, vmax=22,
                   s=2, alpha=0.5, rasterized=True, zorder=1)

    # ── QSO candidates (all Gaia-flagged, before external vetting) ───────────
    if qso_df is not None and len(qso_df) > 0 and 'ra' in qso_df.columns:
        # If we also have anchors, show raw candidates as lighter background markers
        _qso_alpha = 0.45 if anchor_df is not None else 0.8
        ax.scatter(qso_df['ra'].values, qso_df['dec'].values,
                   marker='o', s=18, color='#f4a261', alpha=_qso_alpha,
                   linewidths=0.2, zorder=3,
                   label=f'Gaia QSO candidates ({len(qso_df)})')

    # ── Vetted QSO anchors (Quaia/MILLIQUAS + astrometric cut) ───────────────
    if anchor_df is not None and len(anchor_df) > 0:
        _acol = 'ra' if 'ra' in anchor_df.columns else None
        _dcol = 'dec' if 'dec' in anchor_df.columns else None
        if _acol and _dcol:
            ax.scatter(anchor_df[_acol].values, anchor_df[_dcol].values,
                       marker='*', s=90, color='gold', edgecolors='darkorange',
                       linewidths=0.6, alpha=0.95, zorder=5,
                       label=f'Vetted QSO anchors ({len(anchor_df)})')

    # ── Footprint polygons ────────────────────────────────────────────────────
    filter_patches: dict[str, mpatches.Patch] = {}   # for legend

    # Normalise search_boxes: use multi-pointing list if given, else build from
    # single (ra, dec, sw, sh).  None → no guard / cutout.
    _boxes: list[tuple[float, float, float, float]] | None = None
    if search_boxes is not None and len(search_boxes) > 0:
        _boxes = list(search_boxes)
    elif ra is not None and dec is not None and search_width and search_height:
        _boxes = [(ra, dec, search_width, search_height)]

    # Draw all search boxes as dashed rectangles
    if _boxes is not None:
        for _bi, (_bra, _bdec, _bsw, _bsh) in enumerate(_boxes):
            _rect_ra  = [_bra - _bsw/2, _bra + _bsw/2,
                         _bra + _bsw/2, _bra - _bsw/2, _bra - _bsw/2]
            _rect_dec = [_bdec - _bsh/2, _bdec - _bsh/2,
                         _bdec + _bsh/2, _bdec + _bsh/2, _bdec - _bsh/2]
            ax.plot(_rect_ra, _rect_dec, 'k--', lw=1.0, alpha=0.6, zorder=1,
                    label='Search box' if _bi == 0 else None)

    # Build search-box guard: footprints with centroids outside the union of all
    # boxes (×3 margin) are silently skipped — prevents corrupt WCS s_region
    # entries from ballooning the plot axes.
    if _boxes is not None:
        _guard_ra_lo  = min(b[0] - b[2]/2 for b in _boxes) * 1  # expand below
        _guard_ra_hi  = max(b[0] + b[2]/2 for b in _boxes)
        _guard_dec_lo = min(b[1] - b[3]/2 for b in _boxes)
        _guard_dec_hi = max(b[1] + b[3]/2 for b in _boxes)
        # Widen guard to 3× the full span so only truly anomalous entries are skipped
        _span_ra  = _guard_ra_hi  - _guard_ra_lo
        _span_dec = _guard_dec_hi - _guard_dec_lo
        _cen_ra   = (_guard_ra_lo  + _guard_ra_hi)  / 2
        _cen_dec  = (_guard_dec_lo + _guard_dec_hi) / 2
        _guard_ra_lo  = _cen_ra  - _span_ra  * 1.5
        _guard_ra_hi  = _cen_ra  + _span_ra  * 1.5
        _guard_dec_lo = _cen_dec - _span_dec * 1.5
        _guard_dec_hi = _cen_dec + _span_dec * 1.5
    else:
        _guard_ra_lo = _guard_ra_hi = _guard_dec_lo = _guard_dec_hi = None

    for _, row in obs_df.iterrows():
        filt    = str(row.get('filters', '')).strip().upper()
        fid     = int(row.get('field_id', 0))
        color   = _FILTER_COLORS.get(filt, _DEFAULT_COLOR)
        s_region = str(row.get('s_region', ''))

        polygons = _parse_polygons(s_region)
        if not polygons:
            continue

        # Skip polygons whose centroid lies far outside the search box.
        if _guard_ra_lo is not None:
            cx = np.mean(polygons[0][:, 0])
            cy = np.mean(polygons[0][:, 1])
            if not (_guard_ra_lo <= cx <= _guard_ra_hi and
                    _guard_dec_lo <= cy <= _guard_dec_hi):
                continue

        for poly_verts in polygons:
            patch = MplPolygon(poly_verts, closed=True,
                               facecolor='none', edgecolor=color,
                               lw=1.5, zorder=2)
            ax.add_patch(patch)

        # Label at centroid of first polygon
        verts = polygons[0]
        cx = np.mean(verts[:, 0])
        cy = np.mean(verts[:, 1])
        ax.text(cx, cy, str(fid), ha='center', va='center',
                fontsize=7, fontweight='bold', color=color,
                zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.6))

        if filt not in filter_patches:
            filter_patches[filt] = mpatches.Patch(
                facecolor='none', edgecolor=color, lw=1.5,
                label=filt)

    # ── Axes limits ───────────────────────────────────────────────────────────
    # Strategy: use data-derived limits (zoom in to where HST actually points)
    # but clamp to the user's cutout so that wildly-offset WCS entries cannot
    # zoom the plot out beyond the requested field of view.
    #
    # When the search box is available:
    #   - Only footprint centroids that fall inside the cutout contribute to
    #     the data bounds (filters out corrupt WCS entries).
    #   - Final limits = data bounds with 8 % padding, clamped to cutout.
    # When the search box is not available: use raw data bounds as before.

    pad_factor = 0.08

    if _boxes is not None:
        # Cutout = union of all search boxes
        cut_ra_lo  = min(b[0] - b[2]/2 for b in _boxes)
        cut_ra_hi  = max(b[0] + b[2]/2 for b in _boxes)
        cut_dec_lo = min(b[1] - b[3]/2 for b in _boxes)
        cut_dec_hi = max(b[1] + b[3]/2 for b in _boxes)

        # Collect bounds from footprints whose centroid lies inside the cutout.
        # Gaia stars are intentionally excluded — they span the full search box
        # and would zoom the plot out far beyond the HST images.
        all_ra, all_dec = [], []
        for _, row in obs_df.iterrows():
            bbox = _footprint_bbox(str(row.get('s_region', '')))
            if not bbox:
                continue
            cra  = (bbox[0] + bbox[1]) / 2
            cdec = (bbox[2] + bbox[3]) / 2
            if cut_ra_lo <= cra <= cut_ra_hi and cut_dec_lo <= cdec <= cut_dec_hi:
                all_ra  += [bbox[0], bbox[1]]
                all_dec += [bbox[2], bbox[3]]

        if all_ra:
            span_ra  = max(all_ra)  - min(all_ra)
            span_dec = max(all_dec) - min(all_dec)
            data_ra_lo  = min(all_ra)  - span_ra  * pad_factor
            data_ra_hi  = max(all_ra)  + span_ra  * pad_factor
            data_dec_lo = min(all_dec) - span_dec * pad_factor
            data_dec_hi = max(all_dec) + span_dec * pad_factor
            # Clamp: zoom in freely, but never exceed the cutout.
            ra_lo  = max(cut_ra_lo,  data_ra_lo)
            ra_hi  = min(cut_ra_hi,  data_ra_hi)
            dec_lo = max(cut_dec_lo, data_dec_lo)
            dec_hi = min(cut_dec_hi, data_dec_hi)
        else:
            # No footprints inside cutout — show full cutout.
            ra_lo, ra_hi   = cut_ra_lo,  cut_ra_hi
            dec_lo, dec_hi = cut_dec_lo, cut_dec_hi

        center_dec = (dec_lo + dec_hi) / 2
    else:
        all_ra, all_dec = [], []
        for _, row in obs_df.iterrows():
            bbox = _footprint_bbox(str(row.get('s_region', '')))
            if bbox:
                all_ra  += [bbox[0], bbox[1]]
                all_dec += [bbox[2], bbox[3]]
        if not all_ra:
            plt.close(fig)
            return
        pad_ra  = (max(all_ra)  - min(all_ra))  * pad_factor
        pad_dec = (max(all_dec) - min(all_dec)) * pad_factor
        ra_lo, ra_hi   = min(all_ra) - pad_ra, max(all_ra) + pad_ra
        dec_lo, dec_hi = min(all_dec) - pad_dec, max(all_dec) + pad_dec
        center_dec = (dec_lo + dec_hi) / 2

    ax.set_xlim(ra_hi, ra_lo)   # RA right-to-left
    ax.set_ylim(dec_lo, dec_hi)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(center_dec)), adjustable='box')

    # ── Legend, labels, title ─────────────────────────────────────────────────
    import matplotlib.lines as mlines
    legend_handles = list(filter_patches.values())
    if qso_df is not None and len(qso_df) > 0 and 'ra' in qso_df.columns:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', color='#f4a261',
                          markersize=6, linestyle='None', alpha=0.7,
                          label=f'Gaia QSO candidates ({len(qso_df)})'))
    if anchor_df is not None and len(anchor_df) > 0:
        legend_handles.append(
            mlines.Line2D([], [], marker='*', color='gold',
                          markeredgecolor='darkorange', markeredgewidth=0.6,
                          markersize=10, linestyle='None',
                          label=f'Vetted QSO anchors ({len(anchor_df)})'))
    if legend_handles:
        ax.legend(handles=legend_handles,
                  title='Filter / sources', fontsize=8, title_fontsize=8,
                  loc='best', framealpha=0.8)

    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    title = f'{field_name} — HST footprints' if field_name else 'HST footprints'
    ax.set_title(title, fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Footprint plot saved: {save_path}")


def plot_delve_footprint(
    delve_csv: 'str | Path',
    obs_df: pd.DataFrame,
    save_path: 'str | Path',
    field_name: str = '',
    ra: float | None = None,
    dec: float | None = None,
    search_width: float | None = None,
    search_height: float | None = None,
    search_boxes: 'list[tuple[float,float,float,float]] | None' = None,
) -> None:
    """
    Plot DELVE sources coloured by magnitude with HST footprints overlaid.

    Axis limits are derived from the HST footprints in obs_df using the same
    logic as plot_footprints, so the zoom level matches the HST field of view.
    Saves a PNG to save_path (in the DELVE subdirectory).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    delve_df = pd.read_csv(delve_csv)

    # Pick best available magnitude column (prefer r, then g, i, z)
    mag_col = next((c for c in ('r_mag', 'g_mag', 'i_mag', 'z_mag')
                    if c in delve_df.columns and delve_df[c].notna().any()), None)
    filter_label = mag_col.split('_')[0].upper() if mag_col else ''

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── DELVE sources coloured by magnitude ──────────────────────────────────
    if mag_col:
        mag = delve_df[mag_col].values
        sc = ax.scatter(delve_df['ra'].values, delve_df['dec'].values,
                        c=mag, cmap='plasma_r', vmin=17, vmax=24,
                        s=2, alpha=0.5, rasterized=True, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label(f'{filter_label}-band mag', fontsize=9)
    else:
        ax.scatter(delve_df['ra'].values, delve_df['dec'].values,
                   s=2, alpha=0.5, color='steelblue', rasterized=True, zorder=1)

    # ── Normalise search boxes ────────────────────────────────────────────────
    _boxes: 'list[tuple[float,float,float,float]] | None' = None
    if search_boxes is not None and len(search_boxes) > 0:
        _boxes = list(search_boxes)
    elif ra is not None and dec is not None and search_width and search_height:
        _boxes = [(ra, dec, search_width, search_height)]

    if _boxes is not None:
        for _bi, (_bra, _bdec, _bsw, _bsh) in enumerate(_boxes):
            _rect_ra  = [_bra - _bsw/2, _bra + _bsw/2,
                         _bra + _bsw/2, _bra - _bsw/2, _bra - _bsw/2]
            _rect_dec = [_bdec - _bsh/2, _bdec - _bsh/2,
                         _bdec + _bsh/2, _bdec + _bsh/2, _bdec - _bsh/2]
            ax.plot(_rect_ra, _rect_dec, 'k--', lw=1.0, alpha=0.6, zorder=2,
                    label='Search box' if _bi == 0 else None)

    # ── HST footprints overlaid for reference ─────────────────────────────────
    filter_patches: dict[str, mpatches.Patch] = {}
    if obs_df is not None and len(obs_df) > 0:
        _guard_ra_lo = _guard_ra_hi = _guard_dec_lo = _guard_dec_hi = None
        if _boxes is not None:
            _cen_ra   = (_boxes[0][0] if len(_boxes) == 1
                         else sum(b[0] for b in _boxes) / len(_boxes))
            _cen_dec  = (_boxes[0][1] if len(_boxes) == 1
                         else sum(b[1] for b in _boxes) / len(_boxes))
            _span_ra  = max(b[0]+b[2]/2 for b in _boxes) - min(b[0]-b[2]/2 for b in _boxes)
            _span_dec = max(b[1]+b[3]/2 for b in _boxes) - min(b[1]-b[3]/2 for b in _boxes)
            _guard_ra_lo  = _cen_ra  - _span_ra  * 1.5
            _guard_ra_hi  = _cen_ra  + _span_ra  * 1.5
            _guard_dec_lo = _cen_dec - _span_dec * 1.5
            _guard_dec_hi = _cen_dec + _span_dec * 1.5

        for _, row in obs_df.iterrows():
            filt  = str(row.get('filters', '')).strip().upper()
            fid   = int(row.get('field_id', 0))
            color = _FILTER_COLORS.get(filt, _DEFAULT_COLOR)
            polys = _parse_polygons(str(row.get('s_region', '')))
            if not polys:
                continue
            cx = np.mean(polys[0][:, 0])
            cy = np.mean(polys[0][:, 1])
            if (_guard_ra_lo is not None and
                    not (_guard_ra_lo <= cx <= _guard_ra_hi
                         and _guard_dec_lo <= cy <= _guard_dec_hi)):
                continue
            for verts in polys:
                patch = plt.Polygon(verts, closed=True,
                                    facecolor='none', edgecolor=color,
                                    lw=1.5, zorder=3)
                ax.add_patch(patch)
            ax.text(cx, cy, str(fid), ha='center', va='center',
                    fontsize=7, fontweight='bold', color=color, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))
            if filt not in filter_patches:
                filter_patches[filt] = mpatches.Patch(
                    facecolor='none', edgecolor=color, lw=1.5, label=filt)

    # ── Axis limits: same logic as plot_footprints (driven by HST footprints) ─
    pad_factor = 0.08
    if _boxes is not None:
        cut_ra_lo  = min(b[0] - b[2]/2 for b in _boxes)
        cut_ra_hi  = max(b[0] + b[2]/2 for b in _boxes)
        cut_dec_lo = min(b[1] - b[3]/2 for b in _boxes)
        cut_dec_hi = max(b[1] + b[3]/2 for b in _boxes)

        all_ra, all_dec = [], []
        if obs_df is not None:
            for _, row in obs_df.iterrows():
                bbox = _footprint_bbox(str(row.get('s_region', '')))
                if not bbox:
                    continue
                cra  = (bbox[0] + bbox[1]) / 2
                cdec = (bbox[2] + bbox[3]) / 2
                if cut_ra_lo <= cra <= cut_ra_hi and cut_dec_lo <= cdec <= cut_dec_hi:
                    all_ra  += [bbox[0], bbox[1]]
                    all_dec += [bbox[2], bbox[3]]

        if all_ra:
            span_ra  = max(all_ra)  - min(all_ra)
            span_dec = max(all_dec) - min(all_dec)
            ra_lo  = max(cut_ra_lo,  min(all_ra)  - span_ra  * pad_factor)
            ra_hi  = min(cut_ra_hi,  max(all_ra)  + span_ra  * pad_factor)
            dec_lo = max(cut_dec_lo, min(all_dec) - span_dec * pad_factor)
            dec_hi = min(cut_dec_hi, max(all_dec) + span_dec * pad_factor)
        else:
            ra_lo, ra_hi   = cut_ra_lo, cut_ra_hi
            dec_lo, dec_hi = cut_dec_lo, cut_dec_hi
        center_dec = (dec_lo + dec_hi) / 2
    else:
        margin = 0.05
        ra_lo  = delve_df['ra'].min()  - margin
        ra_hi  = delve_df['ra'].max()  + margin
        dec_lo = delve_df['dec'].min() - margin
        dec_hi = delve_df['dec'].max() + margin
        center_dec = (dec_lo + dec_hi) / 2

    ax.set_xlim(ra_hi, ra_lo)   # RA right-to-left
    ax.set_ylim(dec_lo, dec_hi)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(center_dec)), adjustable='box')

    # ── Legend, labels ────────────────────────────────────────────────────────
    legend_handles = list(filter_patches.values())
    if legend_handles:
        ax.legend(handles=legend_handles, title='HST filter',
                  fontsize=8, title_fontsize=8, loc='best', framealpha=0.8)

    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    n_delve = len(delve_df)
    title = (f'{field_name} — DELVE sources ({n_delve:,}) + HST footprints'
             if field_name else f'DELVE sources ({n_delve:,})')
    ax.set_title(title, fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  DELVE footprint plot saved: {save_path}")


def _parse_polygons(s_region: str) -> list[np.ndarray]:
    """
    Parse a MAST s_region string into a list of (N,2) vertex arrays.
    Handles strings with one or more POLYGON blocks and ignores non-numeric
    tokens such as the ICRS frame identifier (e.g. "POLYGON ICRS ra dec ...").
    """
    polys = []
    parts = s_region.upper().split('POLYGON')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        nums = []
        for tok in part.split():
            try:
                nums.append(float(tok))
            except (ValueError, TypeError):
                continue
        if len(nums) < 6:
            continue
        verts = np.array(nums).reshape(-1, 2)
        verts[:, 0] = verts[:, 0] % 360   # normalise RA from (−180,180] → [0,360)
        polys.append(verts)
    return polys


def find_flc_images(output_dir: Path, field_name: str,
                    telescope: str = 'HST', im_type: str | None = None) -> list[Path]:
    """
    Return sorted list of downloaded science image paths for a field.

    Expected structure:
        {output_dir}/{field}/{telescope}/mastDownload/{telescope}/{obs_id}/{obs_id}_flc.fits
    (JWST: ..._cal.fits).  im_type None selects the telescope default (_DEFAULT_IM_TYPE).
    """
    if im_type is None:
        im_type = _DEFAULT_IM_TYPE.get(telescope.upper(), '_flc')
    root = Path(output_dir) / field_name / telescope.upper() / "mastDownload" / telescope.upper()
    suffix = f"{im_type}.fits"
    found = sorted(root.rglob(f"*{suffix}")) if root.exists() else []
    return found


def _footprint_bbox(s_region: str) -> tuple[float, float, float, float] | None:
    """
    Parse a MAST s_region string (one or more POLYGON vertices) and return
    (ra_min, ra_max, dec_min, dec_max).  Returns None if unparseable.
    Ignores non-numeric tokens such as ICRS (e.g. "POLYGON ICRS ra dec ...").
    """
    try:
        tokens = s_region.upper().replace('POLYGON', ' ').split()
        coords = []
        for t in tokens:
            try:
                coords.append(float(t))
            except (ValueError, TypeError):
                pass
        if len(coords) < 4:
            return None
        ras  = [r % 360 for r in coords[0::2]]   # normalise RA from (−180,180] → [0,360)
        decs = coords[1::2]
        return min(ras), max(ras), min(decs), max(decs)
    except Exception:
        return None


def _count_gaia_in_footprints(obs_df: pd.DataFrame,
                               gaia_df: pd.DataFrame) -> pd.Series:
    """
    For each row in obs_df, count Gaia stars whose (ra, dec) falls within
    the image footprint bounding box derived from s_region.
    Returns a Series of integer counts aligned to obs_df's index.
    """
    ra_g  = gaia_df['ra'].values
    dec_g = gaia_df['dec'].values
    counts = []
    for _, row in obs_df.iterrows():
        bbox = _footprint_bbox(str(row.get('s_region', '')))
        if bbox is None:
            counts.append(-1)
            continue
        ra_min, ra_max, dec_min, dec_max = bbox
        n = int(np.sum(
            (ra_g  >= ra_min) & (ra_g  <= ra_max) &
            (dec_g >= dec_min) & (dec_g <= dec_max)
        ))
        counts.append(n)
    return pd.Series(counts, index=obs_df.index)


def _invalidate_psf_cache(flc_path: Path) -> None:
    """Delete PSF and cross-match caches for a given FLC path, if they exist."""
    for p in (flc_path.parent / f"{flc_path.stem}_catalog.fits",
              flc_path.parent / "psf_params.json",
              flc_path.parent / "matched_gaia.csv",
              flc_path.parent / "xmatch_params.json"):
        if p.exists():
            p.unlink()


# Failed-observation rule set recorded in each verify-cache entry; bump on any rule change so
# cached verdicts are re-judged.  2 = 2026-10-01: no-HDRLET rule dropped, calibration rule added.
_VERIFY_RULES = 2


def _failed_reason_jwst(h0) -> str | None:
    """JWST failed-observation rules on a primary header (Liwen Chen's bp3m fork):
    1. EFFEXPTM == 0        — no effective exposure time collected;
    2. ENG_QUAL != 'OK'     — guide-star / engineering problem during the exposure;
    3. DATAPROB == True     — pipeline flagged a data problem;
    4. VISITSTA != 'SUCCESSFUL' — the visit did not complete.
    """
    effexptm = h0.get('EFFEXPTM', None)
    eng_qual = str(h0.get('ENG_QUAL', '') or '').strip()
    dataprob = h0.get('DATAPROB', False)
    visitsta = str(h0.get('VISITSTA', '') or '').strip()
    if effexptm is not None and float(effexptm) == 0.0:
        return "EFFEXPTM=0.0"
    if eng_qual and eng_qual != 'OK':
        return f"ENG_QUAL='{eng_qual}'"
    if dataprob is True or str(dataprob).strip().upper() in ('T', 'TRUE'):
        return "DATAPROB=True"
    if visitsta and visitsta != 'SUCCESSFUL':
        return f"VISITSTA='{visitsta}'"
    return None


def _failed_reason_hst(h0) -> str | None:
    """HST failed-observation rules on a primary header; see _check_exptime."""
    exptime = h0.get('EXPTIME', None)
    expflag = str(h0.get('EXPFLAG', '') or '').strip()
    imagetyp = str(h0.get('IMAGETYP', 'EXT') or 'EXT').strip().upper()
    targname = str(h0.get('TARGNAME', '') or '').strip()
    if imagetyp != 'EXT':
        return f"calibration exposure (IMAGETYP='{imagetyp}', TARGNAME='{targname}')"
    if exptime is not None and float(exptime) == 0.0:
        return f"EXPTIME=0.0 (EXPFLAG='{expflag}')" if expflag else "EXPTIME=0.0"
    if expflag and expflag != 'NORMAL':
        return f"EXPFLAG='{expflag}'"
    return None


def _failed_reason(h0, telescope: str = 'HST') -> str | None:
    return _failed_reason_jwst(h0) if telescope.upper() == 'JWST' else _failed_reason_hst(h0)


def _check_exptime(flc_path: Path, telescope: str = 'HST') -> str | None:
    """Return a failure reason string if the image is a failed observation, else None.

    JWST images use _failed_reason_jwst (EFFEXPTM, ENG_QUAL, DATAPROB, VISITSTA).
    HST checks three conditions in priority order (file is kept on disk in all cases):
    1. EXPTIME == 0 — shutter open but no real sky signal (e.g. EXCESSIVE DOWNTIME).
    2. EXPFLAG != 'NORMAL' — any non-nominal exposure flag indicates compromised data.
       Known values seen in practice:
         'EXCESSIVE DOWNTIME'    — guide-star loss; EXPTIME typically 0
         'TDF-DOWN AT EXPSTART'  — science telemetry unavailable; data may be corrupt
         'INTERRUPTED'           — exposure cut short by HST safing or guide-star loss
       Any other non-NORMAL value is also flagged.
    3. IMAGETYP != 'EXT' — calibration exposure (internal FLAT lamps TUNGSTEN/DEUTERIUM,
       DARK, BIAS; RA_TARG = DEC_TARG = 0).  MAST position queries return some of them;
       the no-HDRLET rule used to drop them as a side effect (calibration frames carry no
       headerlet): 47% of the 2,357 HDRLET-excluded archive images (sample of 600,
       2026-10-01), e.g. Leo_I iblb2qltq (DARK, 900 s) PSF-fitted and then 'No stars in field'.
    A missing HDRLET extension is NOT a failure (rule removed 2026-10-01): headerlets
    only carry MAST's a-posteriori WCS solutions.  BP3M takes CRVAL as the tangent
    point (pointing is fitted), CD/ORIENTAT as the rotation prior centre and applies
    its own GDC, so the header distortion/headerlet solution is never used.
    """
    from astropy.io import fits
    try:
        h0 = fits.getheader(flc_path, 0)
        return _failed_reason(h0, telescope)
    except Exception:
        pass
    return None


def _write_selected_obsids(prod_df: pd.DataFrame, hst_dir: Path,
                            field_name: str, im_type: str,
                            failed_obsids: dict[str, str] | None = None) -> None:
    """Save individual FLC image obs_ids to a JSON manifest.

    These are the per-exposure obs_ids (e.g. 'jbjm03llq') that match the
    directory names under mastDownload/ and the image names used by BP3M —
    not the parent observation obsids.

    failed_obsids, if given, maps obs_id → reason string for images that were
    downloaded but must be skipped (e.g. EXPTIME=0 failed observations).
    These are written to a separate {field}_failed_obsids.json manifest and
    excluded from the selected manifest.
    """
    flc_sub = im_type[1:].upper()   # '_flc' → 'FLC'
    flc_rows = prod_df[prod_df['productSubGroupDescription'] == flc_sub]
    all_obsids = sorted(set(flc_rows['obs_id'].astype(str)))
    bad = set(failed_obsids or {})
    obsids = [o for o in all_obsids if o not in bad]
    manifest = hst_dir / f"{field_name}_selected_obsids.json"
    manifest.write_text(json.dumps(obsids, indent=2))
    failed_manifest = hst_dir / f"{field_name}_failed_obsids.json"
    if failed_obsids:
        failed_manifest.write_text(json.dumps(failed_obsids, indent=2))
    elif failed_manifest.exists():
        failed_manifest.unlink()


def _print_obs_table(obs_df: pd.DataFrame) -> None:
    """Print the observations table with field_id, proposal_id, n_gaia, n_exp."""
    display_cols = {
        'field_id':      'ID',
        'proposal_id':   'PropID',
        'target_name':   'Target',
        'obs_time':      'Date',
        'instrument_name': 'Instrument',
        'filters':       'Filter',
        'i_exptime':     'ExpTime(s)',
        'n_exp':         'N_exp',
        't_baseline':    'Baseline(yr)',
        'n_gaia':        'N_Gaia',
    }
    present = {k: v for k, v in display_cols.items() if k in obs_df.columns}
    disp = obs_df[list(present.keys())].rename(columns=present).copy()
    # Format floats nicely
    for col in ('ExpTime(s)', 'Baseline(yr)'):
        if col in disp.columns:
            disp[col] = disp[col].map(lambda x: f'{x:.1f}' if pd.notna(x) else '?')
    if 'N_Gaia' in disp.columns:
        disp['N_Gaia'] = disp['N_Gaia'].map(lambda x: str(x) if x >= 0 else '?')
    print(disp.to_string(index=False))
