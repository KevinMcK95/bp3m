"""
Extract DELVE proper motion catalog data for a bp3m field.

Finds the relevant nside=32 HEALPix tiles from a local DELVE catalog directory,
loads all mover extensions, applies a star-galaxy cut, deduplicates across tiles,
converts units and rotates the 5×5 covariance from gnomonic to equatorial
coordinates, and saves a Gaia-format-compatible CSV to:

    {output_dir}/{field_name}/DELVE/{field_name}_ra{ra}_dec{dec}_w{w}_h{h}_delve.csv

Column names are chosen to match the Gaia CSV as closely as possible so that the
same cross-matching and alignment infrastructure can consume both catalogs.

Unit conventions (Gaia-compatible):
  positions      : degrees
  position errors: mas   (√c_xx × 1000 / cos(dec) for RA,  √c_yy × 1000 for Dec)
  pmra / pmdec   : mas/yr (DELVE already reports these in mas/yr)
  pmra/pmdec errs: mas/yr (√c_vxvx × 1000,  √c_vyvy × 1000)
  parallax       : mas   (DELVE arcsec × 1000)
  parallax error : mas   (√c_pipi × 1000)
  correlations   : dimensionless (unchanged by the diagonal gnomonic→equatorial rotation)
  ref_epoch      : 2016.0 (MJD 57388.0)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

log = logging.getLogger(__name__)

_NSIDE = 32
_EXTS  = ["modest1_movers", "modest2_movers", "modest3_movers", "fast_movers"]
_EXT_OFFSETS = {"modest1_movers": 0, "modest2_movers": 1,
                "modest3_movers": 2, "fast_movers": 3}
# Unique source_id = healpix_pixel * _ID_TILE_STRIDE + ext_offset * _ID_EXT_STRIDE + row_idx
_ID_TILE_STRIDE = 10 ** 9
_ID_EXT_STRIDE  = 10 ** 6

# Star-galaxy separation: DES spread_model < threshold → point source
_SPREAD_THR  = 0.003
_SPREAD_BANDS = ["r", "i", "g", "z"]  # preference order for the cut


def _cache_stem(field_name: str, ra: float, dec: float,
                search_width: float, search_height: float) -> str:
    return (f"{field_name}_ra{ra:.4f}_dec{dec:+.4f}"
            f"_w{search_width:.4f}_h{search_height:.4f}_delve")


def _find_tiles(ra: float, dec: float,
                search_width: float, search_height: float,
                delve_dir: Path) -> list[Path]:
    """Return existing DELVE tile paths whose footprints overlap the search box."""
    import healpy as hp
    half_diag = np.degrees(np.arctan(
        np.sqrt((np.radians(search_width / 2)) ** 2
                + (np.radians(search_height / 2)) ** 2)
    ))
    # Extra buffer: one tile width (~1.8°) so boundary sources are never missed
    radius = half_diag + 1.8
    vec    = hp.ang2vec(np.radians(90.0 - dec), np.radians(ra))
    pixels = hp.query_disc(_NSIDE, vec, np.radians(radius),
                           nest=False, inclusive=True)
    paths  = [delve_dir / f"PM_hp{p:05d}.fits" for p in pixels]
    return [p for p in paths if p.exists()]


def _star_galaxy_cut(d: np.ndarray) -> np.ndarray:
    """Boolean mask: True = likely point source. Uses best available band."""
    for band in _SPREAD_BANDS:
        n_col  = f"{band}_n"
        sp_col = f"{band}_spread"
        if n_col in d.dtype.names and sp_col in d.dtype.names:
            ok = d[n_col] > 0
            if ok.sum() > 0:
                return ok & (np.abs(d[sp_col]) < _SPREAD_THR)
    return np.ones(len(d), dtype=bool)


def _build_source_id(healpix_pixel: int, extname: str,
                     row_idx: np.ndarray) -> np.ndarray:
    offset = _EXT_OFFSETS.get(extname, 0)
    return (healpix_pixel * _ID_TILE_STRIDE
            + offset * _ID_EXT_STRIDE
            + row_idx.astype(np.int64))


def _corr(cov_ab: np.ndarray,
          cov_aa: np.ndarray,
          cov_bb: np.ndarray) -> np.ndarray:
    denom = np.sqrt(np.abs(cov_aa) * np.abs(cov_bb))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, cov_ab / denom, 0.0)


def _load_tile(path: Path) -> tuple[int, float, float, list[pd.DataFrame]]:
    """Load all mover extensions from one tile. Returns (pixel, RA0, Dec0, frames)."""
    frames: list[pd.DataFrame] = []
    with fits.open(path, memmap=False) as hdul:
        h   = hdul[0].header
        ra0 = float(h["RA0"])
        dec0 = float(h["DEC0"])
        pixel = int(path.stem.split("hp")[1])
        ext_names = {hdu.name for hdu in hdul}
        for extname in _EXTS:
            if extname not in ext_names:
                continue
            d = hdul[extname].data
            mask = _star_galaxy_cut(d)
            d = d[mask]
            if len(d) == 0:
                continue

            dec_rad = np.radians(d["dec"])
            cos_dec = np.cos(dec_rad)

            # ── unit conversions ──────────────────────────────────────────────
            # Position errors: gnomonic xi uncertainty (arcsec) → equatorial RA (mas)
            # σ_RA = √c_xx × 1000 / cos(dec);  σ_Dec = √c_yy × 1000
            sig_x  = np.sqrt(np.abs(d["c_xx"]))    # arcsec
            sig_y  = np.sqrt(np.abs(d["c_yy"]))    # arcsec
            sig_vx = np.sqrt(np.abs(d["c_vxvx"]))  # arcsec/yr
            sig_vy = np.sqrt(np.abs(d["c_vyvy"]))  # arcsec/yr
            sig_pi = np.sqrt(np.abs(d["c_pipi"]))  # arcsec

            ra_error_mas       = sig_x  * 1000.0 / cos_dec
            dec_error_mas      = sig_y  * 1000.0
            pmra_error_masyr   = sig_vx * 1000.0
            pmdec_error_masyr  = sig_vy * 1000.0
            parallax_error_mas = sig_pi * 1000.0
            parallax_mas       = d["parallax"] * 1000.0

            # ── correlations (unchanged by diagonal gnomonic→equatorial rotation)
            rows = {
                # astrometry
                "ra":              d["ra"],
                "dec":             d["dec"],
                "ra_error":        ra_error_mas,
                "dec_error":       dec_error_mas,
                "parallax":        parallax_mas,
                "parallax_error":  parallax_error_mas,
                "pmra":            d["pmra"],      # already mas/yr
                "pmra_error":      pmra_error_masyr,
                "pmdec":           d["pmdec"],     # already mas/yr
                "pmdec_error":     pmdec_error_masyr,
                "ref_epoch":       np.full(len(d), 2016.0),
                # correlations (all 10 off-diagonal pairs)
                "ra_dec_corr":          _corr(d["c_xy"],   d["c_xx"],   d["c_yy"]),
                "ra_parallax_corr":     _corr(d["c_xpi"],  d["c_xx"],   d["c_pipi"]),
                "ra_pmra_corr":         _corr(d["c_xvx"],  d["c_xx"],   d["c_vxvx"]),
                "ra_pmdec_corr":        _corr(d["c_xvy"],  d["c_xx"],   d["c_vyvy"]),
                "dec_parallax_corr":    _corr(d["c_ypi"],  d["c_yy"],   d["c_pipi"]),
                "dec_pmra_corr":        _corr(d["c_yvx"],  d["c_yy"],   d["c_vxvx"]),
                "dec_pmdec_corr":       _corr(d["c_yvy"],  d["c_yy"],   d["c_vyvy"]),
                "parallax_pmra_corr":   _corr(d["c_vxpi"], d["c_pipi"], d["c_vxvx"]),
                "parallax_pmdec_corr":  _corr(d["c_vypi"], d["c_pipi"], d["c_vyvy"]),
                "pmra_pmdec_corr":      _corr(d["c_vxvy"], d["c_vxvx"], d["c_vyvy"]),
                # photometry (DES griz; r_mag is closest to Gaia G)
                "r_mag":           d["r_mag"],
                "g_mag":           d["g_mag"],
                "i_mag":           d["i_mag"],
                "z_mag":           d["z_mag"],
                "r_n":             d["r_n"],
                "g_n":             d["g_n"],
                "i_n":             d["i_n"],
                "z_n":             d["z_n"],
                "r_spread":        d["r_spread"],
                "g_spread":        d["g_spread"],
                "i_spread":        d["i_spread"],
                "z_spread":        d["z_spread"],
                "r_spread_err":    d["r_spread_err"],
                "g_spread_err":    d["g_spread_err"],
                "i_spread_err":    d["i_spread_err"],
                "z_spread_err":    d["z_spread_err"],
                # fit quality
                "pm":              d["pm"],   # total PM magnitude (mas/yr)
                "chisq_total":     d["chisqTotal"],
                "dof":             d["dof"],
                "n_clip":          d["nClip"],
                # provenance
                "mtype":           d["mtype"].astype(str),
                "healpix_pixel":   np.full(len(d), pixel, dtype=np.int32),
                # synthetic unique ID
                "source_id":       _build_source_id(pixel, extname,
                                                    np.arange(len(d), dtype=np.int64)),
            }
            frames.append(pd.DataFrame(rows))
    return pixel, ra0, dec0, frames


def _dedup_by_position(df: pd.DataFrame, radius_deg: float = 2.78e-5) -> pd.DataFrame:
    """Remove duplicate sources within radius_deg (default 0.1 arcsec = 2.78e-5 deg)."""
    if len(df) == 0:
        return df
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    # Greedy keep-first dedup: mark duplicates via self-match
    idx, d2d, _ = coords.match_to_catalog_sky(coords,
                                              nthneighbor=2)
    dup = d2d.deg < radius_deg
    # keep the first occurrence of each matched pair
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if dup[i] and idx[i] < i:
            keep[i] = False
    return df[keep].reset_index(drop=True)


def _filter_to_box(df: pd.DataFrame, ra: float, dec: float,
                   search_width: float, search_height: float,
                   buffer_deg: float = 0.05) -> pd.DataFrame:
    """Keep sources within the search box plus a small buffer."""
    cos_dec = np.cos(np.radians(dec))
    dra  = np.abs((df["ra"].values  - ra + 180) % 360 - 180) * cos_dec
    ddec = np.abs(df["dec"].values - dec)
    mask = (dra <= search_width / 2 + buffer_deg) & \
           (ddec <= search_height / 2 + buffer_deg)
    return df[mask].reset_index(drop=True)


def download_delve(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    output_dir: "str | Path",
    field_name: str,
    delve_dir: "str | Path",
    force_redownload: bool = False,
) -> pd.DataFrame | None:
    """
    Extract DELVE PM catalog data for a field and cache as a Gaia-format CSV.

    Parameters
    ----------
    ra, dec            : field centre (degrees)
    search_width/height: search box size (degrees)
    output_dir         : pipeline root directory
    field_name         : subdirectory name (e.g. 'Fornax_dSph')
    delve_dir          : path to the directory containing PM_hp*.fits files
    force_redownload   : ignore cache and regenerate from FITS files

    Returns
    -------
    pd.DataFrame with Gaia-compatible columns, or None if no DELVE tiles found.
    """
    delve_dir  = Path(delve_dir)
    output_dir = Path(output_dir)
    out_subdir = output_dir / field_name / "DELVE"
    out_subdir.mkdir(parents=True, exist_ok=True)

    stem     = _cache_stem(field_name, ra, dec, search_width, search_height)
    out_path = out_subdir / f"{stem}.csv"
    meta_path = out_subdir / f"{stem}.tiles.json"

    # ── find tiles ────────────────────────────────────────────────────────────
    tile_paths = _find_tiles(ra, dec, search_width, search_height, delve_dir)
    if not tile_paths:
        log.warning("[DELVE] No tiles found for %s  (ra=%.4f, dec=%.4f)",
                    field_name, ra, dec)
        return None

    tile_names = [p.name for p in tile_paths]
    log.info("[DELVE] %d tile(s) for %s: %s", len(tile_paths), field_name,
             ", ".join(tile_names))

    # ── cache check ───────────────────────────────────────────────────────────
    if not force_redownload and out_path.exists() and meta_path.exists():
        cached_tiles = json.loads(meta_path.read_text()).get("tiles", [])
        if set(cached_tiles) == set(tile_names):
            log.info("[DELVE] Loading cached catalogue: %s", out_path)
            return pd.read_csv(out_path)
        log.info("[DELVE] Tile list changed — regenerating.")

    # ── load all tiles ────────────────────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []
    for path in tile_paths:
        _, _, _, frames = _load_tile(path)
        all_frames.extend(frames)

    if not all_frames:
        log.warning("[DELVE] No point sources found after star-galaxy cut.")
        return None

    df = pd.concat(all_frames, ignore_index=True)
    log.info("[DELVE] %d sources from %d tile(s) before dedup/filter",
             len(df), len(tile_paths))

    # ── deduplicate across tile boundaries ────────────────────────────────────
    df = _dedup_by_position(df)
    log.info("[DELVE] %d sources after spatial dedup (0.1\")", len(df))

    # ── filter to field footprint ─────────────────────────────────────────────
    df = _filter_to_box(df, ra, dec, search_width, search_height)
    log.info("[DELVE] %d sources within field footprint (+buffer)", len(df))

    if len(df) == 0:
        log.warning("[DELVE] No sources remain after footprint filter.")
        return None

    # ── save ──────────────────────────────────────────────────────────────────
    df.to_csv(out_path, index=False)
    meta_path.write_text(json.dumps({"field": field_name, "ra": ra, "dec": dec,
                                     "search_width": search_width,
                                     "search_height": search_height,
                                     "tiles": tile_names,
                                     "n_sources": len(df)}, indent=2))
    print(f"[DELVE] Saved {len(df):,} sources → {out_path}")
    return df
