"""
Identify HST guide stars used across all exposures, resolve their sky
positions from the STScI Guide Star Catalog 2.4.2, and cross-match to
Gaia DR3.

Flow:
  1. Scan all *_spt.fits files → collect unique GSC IDs (DGESTAR/SGESTAR),
     mean EXPSTART MJD, and mean V1 telescope pointing per guide star.
  2. Query STScI GSC 2.4.2 web service (cone search around V1 pointing).
     GSC 2.4.2 is the current operational HST guide star catalog; it
     contains entries absent from the older VizieR-hosted GSC 2.3.2, and
     already embeds Gaia DR1/DR2 source IDs.
     Fall back to VizieR I/305 (GSC 2.3.2) or I/254 (GSC 1.x) if the
     star is not found in GSC 2.4.2.
  3. For each matched GSC entry that has a Gaia DR2 source ID: query
     Gaia DR3 directly by source_id to retrieve up-to-date astrometry
     (ra, dec, pmra, pmdec, parallax, radial_velocity).  If no DR2 ID is
     available, fall back to a Gaia DR3 cone search.
  4. Cross-match quality is assessed by propagating the Gaia J2016
     position → J2000 apparent (GCRS, to match the nature of GSC
     photographic plate observations) and comparing to the GSC position.
  5. Save {field_name}_guide_stars.csv in hst_dir.
     Subsequent calls are incremental: new guide stars are resolved and
     appended; existing entries have their obs statistics refreshed.

Per-image positions:
  get_guide_star_position_at_epoch() propagates any entry from the CSV to
  an arbitrary HST observation epoch, returning the barycentric ICRS
  position (no aberration) suitable for comparing to the Gaia reference
  frame used by BP3M.

Usage (standalone):
    python -m bp3m.pipeline.guide_stars <hst_dir> [--field <name>] [--force]
"""

from __future__ import annotations

import argparse
import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, GCRS
from astropy.io import fits
from astropy.time import Time
import astropy.units as u


# Gaia DR3 reference epoch (Julian year, TCB scale)
_GAIA_EPOCH = Time(2016.0, format="jyear", scale="tcb")

# GSC catalog epoch for cross-match comparison
_GSC_EPOCH = Time(2000.0, format="jyear")

# STScI GSC 2.4.2 web service
_GSC242_URL = ("https://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx"
               "?RA={ra:.6f}&DEC={dec:.6f}&SR={sr:.4f}"
               "&FORMAT=VOTable&CAT=GSC242")

# FGS approximate field radius from V1 axis (degrees); 14 arcmin + margin
_FGS_SEARCH_RADIUS_DEG = 0.50


# ── GSC ID parsing ────────────────────────────────────────────────────────────

def _parse_guide_star_entry(raw: str) -> tuple[str, str]:
    """
    Parse DGESTAR/SGESTAR into (gsc_id, fgs_unit).
    e.g. 'N6UH000265F1' → ('N6UH000265', 'F1')
         '0083300313F2' → ('0083300313', 'F2')
         'NONE'         → ('', '')
    """
    raw = raw.strip()
    if not raw or raw.upper() in ("NONE", "N/A", ""):
        return "", ""
    if raw[-2:] in ("F1", "F2", "F3"):
        return raw[:-2], raw[-2:]
    return raw, ""


def _gsc_catalog_for_id(gsc_id: str) -> tuple[str, str]:
    """
    VizieR fallback: (catalog, id_column) for the given GSC ID format.
    Starts with a letter → GSC 2.3.2 (I/305, 'GSC2.3')
    Purely numeric      → GSC 1.x   (I/254, 'GSC')
    """
    if gsc_id and gsc_id[0].isalpha():
        return "I/305", "GSC2.3"
    return "I/254", "GSC"


# ── Step 1: collect IDs from SPT files ───────────────────────────────────────

def collect_guide_star_ids(hst_dir: str | Path) -> pd.DataFrame:
    """
    Scan all *_spt.fits and return a DataFrame of unique guide stars with
    usage counts, mean observation epoch, and mean V1 telescope pointing.

    Columns: gsc_id, fgs_unit, n_dominant, n_subdominant, mean_obs_mjd,
             mean_ra_v1, mean_dec_v1, vizier_catalog, id_column.
    """
    hst_dir = Path(hst_dir)
    mast_root = hst_dir / "mastDownload" / "HST"

    counts: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "n_dominant": 0, "n_subdominant": 0,
        "obs_mjds": [], "ra_v1s": [], "dec_v1s": [],
    })

    for spt_path in sorted(mast_root.rglob("*_spt.fits")):
        with fits.open(spt_path) as hdul:
            h = hdul[0].header
            dom_raw  = h.get("DGESTAR", "") or ""
            sub_raw  = h.get("SGESTAR", "") or ""
            expstart = h.get("EXPSTART")
            ra_v1    = h.get("RA_V1")
            dec_v1   = h.get("DEC_V1")
        dom_id, dom_fgs = _parse_guide_star_entry(dom_raw)
        sub_id, sub_fgs = _parse_guide_star_entry(sub_raw)
        for gid, fgs, role in ((dom_id, dom_fgs, "dominant"),
                                (sub_id, sub_fgs, "subdominant")):
            if not gid:
                continue
            c = counts[(gid, fgs)]
            c[f"n_{role}"] += 1
            if expstart is not None:
                c["obs_mjds"].append(float(expstart))
            if ra_v1 is not None:
                c["ra_v1s"].append(float(ra_v1))
            if dec_v1 is not None:
                c["dec_v1s"].append(float(dec_v1))

    rows = []
    for (gsc_id, fgs_unit), c in sorted(counts.items()):
        cat, col = _gsc_catalog_for_id(gsc_id)
        rows.append({
            "gsc_id":         gsc_id,
            "fgs_unit":       fgs_unit,
            "n_dominant":     c["n_dominant"],
            "n_subdominant":  c["n_subdominant"],
            "mean_obs_mjd":   float(np.mean(c["obs_mjds"]))   if c["obs_mjds"]  else np.nan,
            "mean_ra_v1":     float(np.mean(c["ra_v1s"]))     if c["ra_v1s"]   else np.nan,
            "mean_dec_v1":    float(np.mean(c["dec_v1s"]))    if c["dec_v1s"]  else np.nan,
            "vizier_catalog": cat,
            "id_column":      col,
        })
    return pd.DataFrame(rows)


# ── Step 2: batch catalog fetches (one request per catalog per field) ─────────

def _fetch_gsc242_cone(ra_center: float, dec_center: float,
                       radius_deg: float = _FGS_SEARCH_RADIUS_DEG,
                       retries: int = 3):
    """
    Fetch the full STScI GSC 2.4.2 cone as an astropy Table, or None.
    Called once per field; guide star IDs are matched from the returned table.
    """
    import urllib.request
    from astropy.io.votable import parse_single_table

    url = _GSC242_URL.format(ra=ra_center, dec=dec_center, sr=radius_deg)
    delay = 5
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                data = resp.read()
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    GSC 2.4.2 retry {attempt+1}: {e}")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    WARNING: GSC 2.4.2 cone fetch failed: {e}")
                return None
    try:
        tbl = parse_single_table(io.BytesIO(data)).to_table()
        return tbl if len(tbl) > 0 else None
    except Exception as e:
        print(f"    WARNING: GSC 2.4.2 parse failed: {e}")
        return None


def _fetch_vizier_cone(catalog: str, id_col: str,
                       ra_center: float, dec_center: float,
                       radius_deg: float = _FGS_SEARCH_RADIUS_DEG,
                       retries: int = 3):
    """
    Fetch a VizieR catalog cone as a DataFrame keyed by id_col, or None.
    Called once per catalog per field.
    """
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    V = Vizier(columns=[id_col, "RAJ2000", "DEJ2000",
                        "Vmag", "Fmag", "jmag", "Pmag"],
               row_limit=10000)
    coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)
    delay = 5
    for attempt in range(retries):
        try:
            result = V.query_region(coord, radius=radius_deg * u.deg,
                                    catalog=catalog)
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    VizieR {catalog} retry {attempt+1}: {e}")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    WARNING: VizieR {catalog} cone fetch failed: {e}")
                return None

    if not result or len(result) == 0:
        return None
    return result[0].to_pandas()


def _gsc242_row_to_dict(tbl, row_idx: int) -> dict:
    """Extract position/mag/DR2 ID from a GSC 2.4.2 table row."""
    row = tbl[row_idx]

    def _sf(col):
        if col not in tbl.colnames:
            return np.nan
        try:
            f = float(row[col])
            return f if np.isfinite(f) else np.nan
        except (TypeError, ValueError):
            return np.nan

    dr2_id = None
    for col in ("gaiaDr2SourceID", "gaiaDr1SourceID"):
        if col in tbl.colnames:
            val = str(row[col]).strip()
            if val and val not in ("0", "", "--", "nan"):
                try:
                    dr2_id = int(val)
                    break
                except (ValueError, TypeError):
                    pass

    mag = np.nan
    for col in ("gaiaGMag", "FpgMag", "VpgMag", "JpgMag", "VMag", "BMag"):
        v = _sf(col)
        if np.isfinite(v):
            mag = v
            break

    return {
        "ra": _sf("ra"), "dec": _sf("dec"),
        "gaia_dr2_source_id": dr2_id, "mag": mag,
    }


def _vizier_row_to_tuple(df, id_col: str, match_id: str):
    """Return (ra, dec, mag) for the first row matching match_id, or None."""
    rows = df[df[id_col].astype(str).str.strip() == match_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    ra  = float(row["RAJ2000"])
    dec = float(row["DEJ2000"])
    mag = np.nan
    for mcol in ("Vmag", "jmag", "Fmag", "Pmag"):
        if mcol in row.index:
            val = row[mcol]
            try:
                f = float(val)
                if np.isfinite(f):
                    mag = f
                    break
            except (TypeError, ValueError):
                pass
    return ra, dec, mag


def resolve_gsc_positions(gs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve GSC positions using a three-tier fallback chain:
      1. STScI GSC 2.4.2  (current operational catalog, has Gaia DR2 IDs)
      2. VizieR I/305      (GSC 2.3.2, 2006 public release)
      3. VizieR I/254      (GSC 1.x, photographic plates)

    Each catalog is fetched ONCE for the entire field (cone search around
    the mean V1 telescope pointing), then all guide star IDs are matched
    from the cached tables.  This reduces web requests from N_stars × 3
    to 3 regardless of field size.

    Adds ra_gsc, dec_gsc, mag_gsc, gaia_dr2_source_id, gsc_catalog columns.
    """
    cols = ["ra_gsc", "dec_gsc", "mag_gsc", "gaia_dr2_source_id", "gsc_catalog"]
    for c in cols:
        gs_df[c] = np.nan
    gs_df["gaia_dr2_source_id"] = gs_df["gaia_dr2_source_id"].astype(object)
    gs_df["gsc_catalog"] = ""

    # Mean V1 pointing for this field
    ra_center  = float(gs_df["mean_ra_v1"].dropna().mean())  if gs_df["mean_ra_v1"].notna().any()  else np.nan
    dec_center = float(gs_df["mean_dec_v1"].dropna().mean()) if gs_df["mean_dec_v1"].notna().any() else np.nan
    have_center = np.isfinite(ra_center) and np.isfinite(dec_center)

    # ── Fetch all three catalogs once ─────────────────────────────────────
    gsc242_tbl = None
    gsc23_df   = None
    gsc1_df    = None

    if have_center:
        print(f"  Fetching GSC 2.4.2 cone (center {ra_center:.4f}, "
              f"{dec_center:.4f}, r={_FGS_SEARCH_RADIUS_DEG}°) ...", end=" ")
        gsc242_tbl = _fetch_gsc242_cone(ra_center, dec_center)
        print(f"{len(gsc242_tbl)} sources" if gsc242_tbl is not None else "failed")

        print("  Fetching VizieR I/305 (GSC 2.3.2) cone ...", end=" ")
        gsc23_df = _fetch_vizier_cone("I/305", "GSC2.3", ra_center, dec_center)
        print(f"{len(gsc23_df)} sources" if gsc23_df is not None else "failed")

        print("  Fetching VizieR I/254 (GSC 1.x) cone ...", end=" ")
        gsc1_df = _fetch_vizier_cone("I/254", "GSC", ra_center, dec_center)
        print(f"{len(gsc1_df)} sources" if gsc1_df is not None else "failed")
    else:
        print("  WARNING: no V1 pointing available — catalog fetches skipped")

    # ── Match each guide star from cached tables ──────────────────────────
    for idx, row in gs_df.iterrows():
        gsc_id = row["gsc_id"]

        # 1. GSC 2.4.2
        if gsc242_tbl is not None:
            ids = [str(x).strip() for x in gsc242_tbl["hstID"]]
            matches = [i for i, x in enumerate(ids) if x == gsc_id]
            if matches:
                d = _gsc242_row_to_dict(gsc242_tbl, matches[0])
                ra, dec = d["ra"], d["dec"]
                if np.isfinite(ra) and np.isfinite(dec):
                    print(f"  {gsc_id}: GSC 2.4.2  RA={ra:.5f}  Dec={dec:.5f}  "
                          f"mag={d['mag']:.2f}  gaia_dr2={d['gaia_dr2_source_id']}")
                    gs_df.at[idx, "ra_gsc"]      = ra
                    gs_df.at[idx, "dec_gsc"]     = dec
                    gs_df.at[idx, "mag_gsc"]     = d["mag"]
                    gs_df.at[idx, "gsc_catalog"] = "GSC2.4.2"
                    if d["gaia_dr2_source_id"] is not None:
                        gs_df.at[idx, "gaia_dr2_source_id"] = int(
                            d["gaia_dr2_source_id"])
                    continue

        # 2. VizieR I/305
        if gsc23_df is not None:
            vres = _vizier_row_to_tuple(gsc23_df, "GSC2.3", gsc_id)
            if vres:
                ra, dec, mag = vres
                print(f"  {gsc_id}: I/305  RA={ra:.5f}  Dec={dec:.5f}  mag={mag:.2f}")
                gs_df.at[idx, "ra_gsc"]      = ra
                gs_df.at[idx, "dec_gsc"]     = dec
                gs_df.at[idx, "mag_gsc"]     = mag
                gs_df.at[idx, "gsc_catalog"] = "I/305"
                continue

        # 3. VizieR I/254
        if gsc1_df is not None:
            vres = _vizier_row_to_tuple(gsc1_df, "GSC", gsc_id)
            if vres:
                ra, dec, mag = vres
                print(f"  {gsc_id}: I/254  RA={ra:.5f}  Dec={dec:.5f}  mag={mag:.2f}")
                gs_df.at[idx, "ra_gsc"]      = ra
                gs_df.at[idx, "dec_gsc"]     = dec
                gs_df.at[idx, "mag_gsc"]     = mag
                gs_df.at[idx, "gsc_catalog"] = "I/254"
                continue

        print(f"  {gsc_id}: NOT FOUND in any catalog")

    return gs_df


# ── Astrometric propagation ───────────────────────────────────────────────────

def propagate_gaia_to_epoch(
    ra: float, dec: float,
    pmra: float, pmdec: float,
    parallax: float, radial_velocity: float,
    target_time: Time,
    apparent: bool = False,
) -> tuple[float, float]:
    """
    Propagate a Gaia DR3 source from J2016.0 to target_time.

    apparent=False  (default — ICRS for HST pointing in Gaia frame)
        Returns barycentric ICRS position: proper motion + perspective
        acceleration (vlos).  No aberration.  Use for comparing to Gaia
        catalog positions or recording the HST pointing in the Gaia frame.

    apparent=True   (cross-match against observed catalog positions)
        Returns geocentric apparent position (GCRS): adds annual parallax
        and annual aberration (~20") via ICRS→GCRS transform.  Use for
        comparing against GSC photographic-plate positions (which include
        aberration of the plate epoch) to assess cross-match quality.

    Parameters
    ----------
    ra, dec         : Gaia ICRS position at J2016.0 (degrees)
    pmra, pmdec     : proper motion (mas/yr); pmra is pmra×cos(dec)
    parallax        : mas; ≤0 or NaN → no distance (skip parallax/persp.)
    radial_velocity : km/s; NaN → skip perspective acceleration
    target_time     : astropy Time of the target epoch
    apparent        : if True, return GCRS apparent position
    """
    have_dist = np.isfinite(parallax) and parallax > 0
    have_vlos = have_dist and np.isfinite(radial_velocity)

    kwargs = dict(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=_GAIA_EPOCH,
        frame="icrs",
    )
    if have_dist:
        kwargs["distance"] = (1000.0 / parallax) * u.pc
    if have_vlos:
        kwargs["radial_velocity"] = radial_velocity * u.km / u.s

    coord = SkyCoord(**kwargs)
    coord_t = coord.apply_space_motion(new_obstime=target_time)

    if apparent and have_dist:
        coord_gcrs = coord_t.transform_to(GCRS(obstime=target_time))
        return float(coord_gcrs.ra.deg), float(coord_gcrs.dec.deg)
    else:
        return float(coord_t.ra.deg), float(coord_t.dec.deg)


def get_guide_star_position_at_epoch(
    gaia_source_id: int,
    obs_time: Time,
    guide_stars_csv: str | Path,
) -> tuple[float, float] | None:
    """
    Return the predicted barycentric ICRS (RA, Dec) in degrees for a guide
    star at obs_time, using stored Gaia DR3 astrometric parameters.
    Returns None if the source is not found or has no Gaia data.
    """
    df = pd.read_csv(guide_stars_csv, dtype={"gaia_source_id": "Int64"})
    row = df[df["gaia_source_id"] == gaia_source_id]
    if row.empty:
        return None
    r = row.iloc[0]
    if pd.isna(r.get("ra_gaia")):
        return None
    pmra  = float(r.get("pmra_gaia",  0.0) or 0.0)
    pmdec = float(r.get("pmdec_gaia", 0.0) or 0.0)
    plx   = float(r.get("parallax_gaia", np.nan) or np.nan)
    vlos  = float(r.get("vlos_gaia",   np.nan) or np.nan)
    return propagate_gaia_to_epoch(
        float(r["ra_gaia"]), float(r["dec_gaia"]),
        pmra, pmdec, plx, vlos,
        target_time=obs_time,
        apparent=False,  # ICRS (no aberration): HST pointing in Gaia frame
    )


# ── Step 3: Gaia DR3 lookup ───────────────────────────────────────────────────

def _query_gaia_dr3_batch(dr2_ids: list[int]) -> dict[int, pd.Series]:
    """
    Batch DR2→DR3 cross-match via gaiaedr3.dr2_neighbourhood.
    Fetches all requested DR2 IDs in a single TAP query.
    Returns {dr2_source_id: closest-DR3-row-as-Series}.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    id_list = ", ".join(str(int(x)) for x in dr2_ids)
    adql = f"""
    SELECT g.source_id, n.dr2_source_id, n.dr3_source_id,
           n.angular_distance,
           g.ra, g.dec, g.pmra, g.pmdec, g.parallax,
           g.radial_velocity, g.phot_g_mean_mag,
           g.ra_error, g.dec_error, g.parallax_error,
           g.pmra_error, g.pmdec_error,
           g.ra_dec_corr, g.ra_parallax_corr, g.ra_pmra_corr, g.ra_pmdec_corr,
           g.dec_parallax_corr, g.dec_pmra_corr, g.dec_pmdec_corr,
           g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr
    FROM gaiaedr3.dr2_neighbourhood AS n
    JOIN gaiadr3.gaia_source AS g ON g.source_id = n.dr3_source_id
    WHERE n.dr2_source_id IN ({id_list})
    ORDER BY n.angular_distance ASC
    """
    try:
        job = Gaia.launch_job(adql)
        tbl = job.get_results()
    except Exception as e:
        print(f"    WARNING: batch DR2→DR3 query failed: {e}")
        return {}

    if len(tbl) == 0:
        return {}

    df = tbl.to_pandas()
    df["dr2_source_id"] = df["dr2_source_id"].astype("int64")

    # For each DR2 ID keep only the closest DR3 match
    result: dict[int, pd.Series] = {}
    for dr2_id, group in df.groupby("dr2_source_id"):
        group = group.sort_values("angular_distance")
        if len(group) > 1:
            print(f"    NOTE: DR2 id {dr2_id} maps to {len(group)} DR3 sources; "
                  f"using closest (sep={group.iloc[0]['angular_distance']:.1f} mas)")
        result[int(dr2_id)] = group.iloc[0]
    return result


def _gaia_cone_search(ra: float, dec: float,
                      radius_arcsec: float = 30.0,
                      max_gmag: float = 16.0) -> pd.DataFrame | None:
    """
    Fallback: Gaia DR3 cone search when no DR2 source ID is available.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    adql = f"""
    SELECT source_id, ra, dec, pmra, pmdec, parallax,
           radial_velocity, phot_g_mean_mag,
           ra_error, dec_error, parallax_error, pmra_error, pmdec_error,
           ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
           dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
           parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
           DISTANCE(POINT('ICRS',{ra},{dec}),
                    POINT('ICRS',ra,dec)) * 3600.0 AS sep_arcsec_catalog
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
                     CIRCLE('ICRS',{ra},{dec},{radius_arcsec/3600:.6f}))
      AND phot_g_mean_mag < {max_gmag}
    ORDER BY sep_arcsec_catalog ASC
    """
    try:
        job = Gaia.launch_job(adql)
        tbl = job.get_results()
        if len(tbl) == 0:
            return None
        df = tbl.to_pandas()
        df["source_id"] = df["source_id"].astype("int64")
        return df
    except Exception as e:
        print(f"    WARNING: Gaia cone search failed: {e}")
        return None


# ── Step 4: cross-match ───────────────────────────────────────────────────────

def crossmatch_to_gaia(gs_df: pd.DataFrame,
                       search_radius_arcsec: float = 30.0) -> pd.DataFrame:
    """
    Cross-match each guide star to Gaia DR3.

    Primary path: if GSC 2.4.2 provided a Gaia DR2 source ID, query Gaia
    DR3 directly by source_id — no position-based search needed.

    Fallback path: Gaia DR3 cone search (30" radius) ranked by propagated
    J2000 apparent separation (Gaia J2016 → J2000 GCRS vs GSC J2000) to
    correctly handle high-PM guide stars.

    Adds columns: gaia_source_id, ra_gaia, dec_gaia, pmra_gaia, pmdec_gaia,
    parallax_gaia, vlos_gaia, gmag_gaia, sep_catalog_arcsec,
    sep_J2000_arcsec.
    """
    new_cols = [
        "gaia_source_id",
        "ra_gaia", "dec_gaia",
        "pmra_gaia", "pmdec_gaia",
        "parallax_gaia", "vlos_gaia",
        "gmag_gaia",
        "sep_catalog_arcsec",
        "sep_J2000_arcsec",
        # 5×5 astrometric covariance (errors in mas / mas/yr; correlations dimensionless)
        # ordering: (ra, dec, parallax, pmra, pmdec)
        "ra_error_gaia", "dec_error_gaia", "parallax_error_gaia",
        "pmra_error_gaia", "pmdec_error_gaia",
        "ra_dec_corr_gaia", "ra_parallax_corr_gaia",
        "ra_pmra_corr_gaia", "ra_pmdec_corr_gaia",
        "dec_parallax_corr_gaia", "dec_pmra_corr_gaia", "dec_pmdec_corr_gaia",
        "parallax_pmra_corr_gaia", "parallax_pmdec_corr_gaia",
        "pmra_pmdec_corr_gaia",
    ]
    for col in new_cols:
        gs_df[col] = np.nan
    gs_df["gaia_source_id"] = gs_df["gaia_source_id"].astype(object)

    # ── Batch DR2→DR3 lookup: one TAP query for all stars with a DR2 ID ──────
    def _valid_dr2(val):
        return (val is not None
                and not (isinstance(val, float) and np.isnan(val)))

    dr2_ids = [
        int(row["gaia_dr2_source_id"])
        for _, row in gs_df.iterrows()
        if _valid_dr2(row.get("gaia_dr2_source_id"))
        and not np.isnan(float(row.get("ra_gsc", np.nan) or np.nan))
    ]
    dr3_cache: dict[int, pd.Series] = {}
    if dr2_ids:
        print(f"  Batch Gaia DR3 lookup for {len(dr2_ids)} DR2 IDs ...", end=" ")
        dr3_cache = _query_gaia_dr3_batch(dr2_ids)
        print(f"{len(dr3_cache)} matched")

    for idx, row in gs_df.iterrows():
        ra_gsc  = row["ra_gsc"]
        dec_gsc = row["dec_gsc"]
        if np.isnan(ra_gsc) or np.isnan(dec_gsc):
            continue

        dr2_id  = row.get("gaia_dr2_source_id")
        have_dr2 = _valid_dr2(dr2_id)

        # ── Primary: DR2→DR3 result from batch cache ─────────────────────────
        if have_dr2 and int(dr2_id) in dr3_cache:
            gaia_row = dr3_cache[int(dr2_id)]
            _fill_gaia_row(gs_df, idx, gaia_row, ra_gsc, dec_gsc,
                           sep_cat_col="sep_catalog_arcsec",
                           sep_j2000_col="sep_J2000_arcsec")
            continue
        elif have_dr2:
            print(f"  {row['gsc_id']}: DR2 id {int(dr2_id)} not found in DR3 "
                  f"(falling back to cone search)")

        # ── Fallback: cone search ranked by J2000 apparent sep ───────────
        print(f"  Gaia cone search for {row['gsc_id']} "
              f"at ({ra_gsc:.5f}, {dec_gsc:.5f}) ...", end=" ")
        candidates = _gaia_cone_search(ra_gsc, dec_gsc,
                                       radius_arcsec=search_radius_arcsec,
                                       max_gmag=16.0)
        if candidates is None or candidates.empty:
            print("no candidates")
            continue

        gsc_sky = SkyCoord(ra_gsc * u.deg, dec_gsc * u.deg)
        best_sep_j2000 = np.inf
        best_cand      = None

        for _, cand in candidates.iterrows():
            pmra  = float(cand["pmra"])  if np.isfinite(float(cand["pmra"]  or np.nan)) else 0.0
            pmdec = float(cand["pmdec"]) if np.isfinite(float(cand["pmdec"] or np.nan)) else 0.0
            plx   = float(cand["parallax"])         if np.isfinite(float(cand["parallax"]         or np.nan)) else np.nan
            vlos  = float(cand["radial_velocity"])  if np.isfinite(float(cand["radial_velocity"]  or np.nan)) else np.nan

            ra_j2000, dec_j2000 = propagate_gaia_to_epoch(
                float(cand["ra"]), float(cand["dec"]),
                pmra, pmdec, plx, vlos,
                target_time=_GSC_EPOCH,
                apparent=True,  # GSC is from actual obs → compare apparent-to-apparent
            )
            sep = gsc_sky.separation(
                SkyCoord(ra_j2000 * u.deg, dec_j2000 * u.deg)
            ).arcsec
            if sep < best_sep_j2000:
                best_sep_j2000 = sep
                best_cand      = cand

        if best_cand is None:
            print("no match")
            continue

        best_cand = best_cand.copy()
        best_cand["sep_arcsec_catalog"] = float(
            SkyCoord(ra_gsc * u.deg, dec_gsc * u.deg).separation(
                SkyCoord(float(best_cand["ra"]) * u.deg,
                         float(best_cand["dec"]) * u.deg)
            ).arcsec)
        _fill_gaia_row(gs_df, idx, best_cand, ra_gsc, dec_gsc,
                       sep_cat_col="sep_catalog_arcsec",
                       sep_j2000_col="sep_J2000_arcsec",
                       sep_j2000_val=best_sep_j2000)

    return gs_df


def _fill_gaia_row(gs_df, idx, gaia_row, ra_gsc, dec_gsc,
                   sep_cat_col, sep_j2000_col, sep_j2000_val=None):
    """Populate Gaia columns from a gaia_row Series (DR3 query or cone cand)."""
    def _f(col, default=np.nan):
        v = gaia_row.get(col, default)
        try:
            f = float(v)
            return f if np.isfinite(f) else np.nan
        except (TypeError, ValueError):
            return np.nan

    pmra_val  = _f("pmra")
    pmdec_val = _f("pmdec")
    plx_val   = _f("parallax")
    vlos_val  = _f("radial_velocity")
    ra_g      = _f("ra")
    dec_g     = _f("dec")
    gmag      = _f("phot_g_mean_mag")

    # catalog sep
    if np.isfinite(ra_g) and np.isfinite(dec_g):
        cat_sep = float(SkyCoord(ra_gsc * u.deg, dec_gsc * u.deg).separation(
            SkyCoord(ra_g * u.deg, dec_g * u.deg)).arcsec)
    else:
        cat_sep = gaia_row.get("sep_arcsec_catalog", np.nan)
        if cat_sep is None:
            cat_sep = np.nan
        cat_sep = float(cat_sep)

    # J2000 apparent sep
    if sep_j2000_val is None and np.isfinite(ra_g) and np.isfinite(dec_g):
        ra_j2000, dec_j2000 = propagate_gaia_to_epoch(
            ra_g, dec_g,
            pmra_val  if np.isfinite(pmra_val)  else 0.0,
            pmdec_val if np.isfinite(pmdec_val) else 0.0,
            plx_val, vlos_val,
            target_time=_GSC_EPOCH,
            apparent=True,
        )
        sep_j2000_val = float(SkyCoord(ra_gsc * u.deg, dec_gsc * u.deg).separation(
            SkyCoord(ra_j2000 * u.deg, dec_j2000 * u.deg)).arcsec)

    try:
        src_id = int(gaia_row["source_id"])
    except (KeyError, TypeError, ValueError):
        src_id = None

    gs_df.at[idx, "gaia_source_id"]     = src_id
    gs_df.at[idx, "ra_gaia"]            = ra_g
    gs_df.at[idx, "dec_gaia"]           = dec_g
    gs_df.at[idx, "pmra_gaia"]          = pmra_val
    gs_df.at[idx, "pmdec_gaia"]         = pmdec_val
    gs_df.at[idx, "parallax_gaia"]      = plx_val
    gs_df.at[idx, "vlos_gaia"]          = vlos_val
    gs_df.at[idx, "gmag_gaia"]          = gmag
    gs_df.at[idx, sep_cat_col]          = cat_sep
    gs_df.at[idx, sep_j2000_col]        = sep_j2000_val if sep_j2000_val is not None else np.nan

    # 5×5 astrometric covariance columns
    _cov_cols = [
        ("ra_error_gaia",             "ra_error"),
        ("dec_error_gaia",            "dec_error"),
        ("parallax_error_gaia",       "parallax_error"),
        ("pmra_error_gaia",           "pmra_error"),
        ("pmdec_error_gaia",          "pmdec_error"),
        ("ra_dec_corr_gaia",          "ra_dec_corr"),
        ("ra_parallax_corr_gaia",     "ra_parallax_corr"),
        ("ra_pmra_corr_gaia",         "ra_pmra_corr"),
        ("ra_pmdec_corr_gaia",        "ra_pmdec_corr"),
        ("dec_parallax_corr_gaia",    "dec_parallax_corr"),
        ("dec_pmra_corr_gaia",        "dec_pmra_corr"),
        ("dec_pmdec_corr_gaia",       "dec_pmdec_corr"),
        ("parallax_pmra_corr_gaia",   "parallax_pmra_corr"),
        ("parallax_pmdec_corr_gaia",  "parallax_pmdec_corr"),
        ("pmra_pmdec_corr_gaia",      "pmra_pmdec_corr"),
    ]
    for out_col, src_col in _cov_cols:
        gs_df.at[idx, out_col] = _f(src_col)

    pm_mag = float(np.hypot(pmra_val  if np.isfinite(pmra_val)  else 0.0,
                            pmdec_val if np.isfinite(pmdec_val) else 0.0))
    vlos_str = (f"  vlos={vlos_val:.1f} km/s" if np.isfinite(vlos_val) else "")
    print(f"source_id={src_id}  "
          f"G={gmag:.2f}  "
          f"sep_cat={cat_sep:.3f}\"  "
          f"sep_J2000={sep_j2000_val:.3f}\"  "
          f"PM={pm_mag:.1f} mas/yr{vlos_str}")


# ── Incremental obs-epoch update ──────────────────────────────────────────────

def _update_obs_stats(existing_df: pd.DataFrame,
                      current_scan: pd.DataFrame) -> pd.DataFrame:
    """
    Refresh n_dominant, n_subdominant, mean_obs_mjd, mean_ra_v1, mean_dec_v1
    from the current SPT scan for all matching gsc_ids.
    """
    scan_map = current_scan.set_index("gsc_id")
    for idx, row in existing_df.iterrows():
        gid = row["gsc_id"]
        if gid not in scan_map.index:
            continue
        sr = scan_map.loc[gid]
        for col in ("n_dominant", "n_subdominant",
                    "mean_obs_mjd", "mean_ra_v1", "mean_dec_v1"):
            if col in sr.index:
                existing_df.at[idx, col] = sr[col]
    return existing_df


# ── main entry ────────────────────────────────────────────────────────────────

def download_guide_stars(hst_dir: str | Path,
                         field_name: str | None = None,
                         force: bool = False) -> Path | None:
    """
    Full pipeline: collect IDs → GSC 2.4.2 (+ VizieR fallback) →
    Gaia DR3 (by source_id or cone search) → save CSV.

    Incremental: existing entries have obs statistics refreshed; new guide
    stars are fully resolved and appended.  Pass force=True to regenerate
    from scratch.
    """
    hst_dir = Path(hst_dir)
    if field_name is None:
        field_name = hst_dir.parent.name
    out_csv = hst_dir / f"{field_name}_guide_stars.csv"

    print("  Scanning SPT files for guide star IDs...")
    current_scan = collect_guide_star_ids(hst_dir)
    if current_scan.empty:
        print("  No guide star IDs found in SPT files.")
        return None
    current_ids = set(current_scan["gsc_id"].tolist())

    # ── incremental path ──────────────────────────────────────────────────
    if out_csv.exists() and not force:
        existing_df = pd.read_csv(out_csv)
        existing_ids = set(existing_df["gsc_id"].tolist())
        new_ids = current_ids - existing_ids

        existing_df = _update_obs_stats(existing_df, current_scan)

        if not new_ids:
            print(f"  No new guide stars ({len(existing_ids)} already resolved). "
                  "Obs statistics refreshed.")
            existing_df.to_csv(out_csv, index=False)
            return out_csv

        print(f"  {len(new_ids)} new guide star(s): {', '.join(sorted(new_ids))}")
        new_rows = current_scan[current_scan["gsc_id"].isin(new_ids)].copy()
        new_rows = resolve_gsc_positions(new_rows)
        n_res = new_rows["ra_gsc"].notna().sum()
        print(f"  Resolved {n_res}/{len(new_rows)} new guide star positions.")
        if n_res > 0:
            new_rows = crossmatch_to_gaia(new_rows)
            n_match = new_rows["gaia_source_id"].notna().sum()
            print(f"  Matched {n_match}/{n_res} new guide stars to Gaia DR3.")

        _fix_gaia_id_dtype(new_rows)
        _fix_gaia_id_dtype(existing_df)
        merged = pd.concat([existing_df, new_rows], ignore_index=True)
        merged.to_csv(out_csv, index=False)
        print(f"  Updated: {out_csv.name}")
        return out_csv

    # ── full fresh run ────────────────────────────────────────────────────
    print(f"  Found {len(current_scan)} unique guide star(s): "
          + ", ".join(current_scan["gsc_id"].tolist()))
    print("  Resolving positions from GSC 2.4.2 (VizieR fallback)...")
    gs_df = resolve_gsc_positions(current_scan)
    n_resolved = gs_df["ra_gsc"].notna().sum()
    print(f"  Resolved {n_resolved}/{len(gs_df)} guide star positions.")
    if n_resolved > 0:
        print("  Cross-matching to Gaia DR3...")
        gs_df = crossmatch_to_gaia(gs_df)
        n_matched = gs_df["gaia_source_id"].notna().sum()
        print(f"  Matched {n_matched}/{n_resolved} guide stars to Gaia DR3.")

    _fix_gaia_id_dtype(gs_df)
    gs_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")
    return out_csv


def _fix_gaia_id_dtype(df: pd.DataFrame) -> None:
    """Store gaia_source_id as int64 where not NaN (in-place)."""
    if "gaia_source_id" not in df.columns:
        return
    mask = df["gaia_source_id"].notna()
    if mask.any():
        df.loc[mask, "gaia_source_id"] = (
            df.loc[mask, "gaia_source_id"].astype("int64"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hst_dir",
                        help="Directory containing mastDownload/HST/")
    parser.add_argument("--field", default=None,
                        help="Field name for output filename")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate guide_stars.csv from scratch")
    args = parser.parse_args()
    download_guide_stars(args.hst_dir, field_name=args.field, force=args.force)


if __name__ == "__main__":
    main()
