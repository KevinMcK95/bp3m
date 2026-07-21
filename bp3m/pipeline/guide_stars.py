"""
Identify HST guide stars used across all exposures, resolve their sky
positions from GSC catalogs, and cross-match to Gaia DR3.

Flow:
  1. Scan all *_spt.fits files → collect unique GSC IDs (DGESTAR/SGESTAR)
     and per-exposure observation times.
  2. Query VizieR for each ID to get RA/Dec at J2000.0:
       N6UH###### → GSC 2.3.2  (VizieR I/305, column GSC2.3)
       ##########  → GSC 1.x   (VizieR I/254, column GSC)
  3. Gaia DR3 TAP cone search (r=30") around each GSC position.
     For each candidate, propagate from Gaia J2016.0 → J2000.0 using
     proper motion, perspective acceleration (vlos), and annual parallax
     (GCRS transform).  Best match = smallest propagated sep vs GSC J2000.
  4. Save {field_name}_guide_stars.csv in hst_dir.
     Subsequent calls are incremental: new guide stars are resolved and
     appended; existing entries are updated with current obs statistics and
     re-propagated positions.

Per-image positions:
  get_guide_star_position_at_epoch() propagates any entry from the CSV to
  an arbitrary HST observation epoch, accounting for PM + parallax + vlos.
  This is called separately for each FLC (e.g. from jitter_summary.py) to
  obtain the guide star's sky position at the time of that image.

Usage (standalone):
    python -m bp3m.pipeline.guide_stars <hst_dir> [--field <name>] [--force]
"""

from __future__ import annotations

import argparse
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

# GSC catalog epoch (J2000.0 for both GSC 1.x and GSC 2.3.2 coordinate system)
_GSC_EPOCH = Time(2000.0, format="jyear")


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
    (vizier_catalog, id_column) for the given GSC ID format.
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
    usage counts and mean observation epoch.

    Columns: gsc_id, fgs_unit, n_dominant, n_subdominant, mean_obs_mjd,
             vizier_catalog, id_column.
    """
    hst_dir = Path(hst_dir)
    mast_root = hst_dir / "mastDownload" / "HST"

    counts: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_dominant": 0, "n_subdominant": 0, "obs_mjds": []})

    for spt_path in sorted(mast_root.rglob("*_spt.fits")):
        with fits.open(spt_path) as hdul:
            h = hdul[0].header
            dom_raw  = h.get("DGESTAR", "") or ""
            sub_raw  = h.get("SGESTAR", "") or ""
            expstart = h.get("EXPSTART")       # MJD
        dom_id, dom_fgs = _parse_guide_star_entry(dom_raw)
        sub_id, sub_fgs = _parse_guide_star_entry(sub_raw)
        if dom_id:
            counts[(dom_id, dom_fgs)]["n_dominant"] += 1
            if expstart is not None:
                counts[(dom_id, dom_fgs)]["obs_mjds"].append(float(expstart))
        if sub_id:
            counts[(sub_id, sub_fgs)]["n_subdominant"] += 1
            if expstart is not None:
                counts[(sub_id, sub_fgs)]["obs_mjds"].append(float(expstart))

    rows = []
    for (gsc_id, fgs_unit), c in sorted(counts.items()):
        cat, col = _gsc_catalog_for_id(gsc_id)
        mjds = c["obs_mjds"]
        rows.append({
            "gsc_id":         gsc_id,
            "fgs_unit":       fgs_unit,
            "n_dominant":     c["n_dominant"],
            "n_subdominant":  c["n_subdominant"],
            "mean_obs_mjd":   float(np.mean(mjds)) if mjds else np.nan,
            "vizier_catalog": cat,
            "id_column":      col,
        })
    return pd.DataFrame(rows)


# ── Step 2: VizieR GSC position lookup ───────────────────────────────────────

def _query_vizier_by_id(gsc_id: str, catalog: str, id_col: str,
                        retries: int = 3) -> tuple[float, float, float] | None:
    """Return (ra_deg, dec_deg, mag) from VizieR, or None."""
    from astroquery.vizier import Vizier

    V = Vizier(columns=[id_col, "RAJ2000", "DEJ2000",
                        "Vmag", "Fmag", "jmag", "Pmag"],
               row_limit=5)
    delay = 5
    for attempt in range(retries):
        try:
            result = V.query_constraints(catalog=catalog, **{id_col: gsc_id})
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    VizieR retry {attempt+1}: {e}")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    WARNING: VizieR failed for {gsc_id}: {e}")
                return None

    if not result or len(result) == 0 or len(result[0]) == 0:
        return None

    row = result[0][0]
    ra  = float(row["RAJ2000"])
    dec = float(row["DEJ2000"])
    mag = np.nan
    for mcol in ("Vmag", "jmag", "Fmag", "Pmag"):
        if mcol in row.colnames:
            val = row[mcol]
            if val is not None and not (hasattr(val, "mask") and val.mask):
                try:
                    mag = float(val)
                    break
                except (TypeError, ValueError):
                    pass
    return ra, dec, mag


def resolve_gsc_positions(gs_df: pd.DataFrame) -> pd.DataFrame:
    """Add ra_gsc, dec_gsc, mag_gsc columns from VizieR."""
    ras, decs, mags = [], [], []
    for _, row in gs_df.iterrows():
        print(f"  Querying {row['vizier_catalog']} for {row['gsc_id']} ...",
              end=" ")
        result = _query_vizier_by_id(row["gsc_id"], row["vizier_catalog"],
                                     row["id_column"])
        if result:
            ra, dec, mag = result
            print(f"RA={ra:.5f}  Dec={dec:.5f}  mag={mag:.2f}")
        else:
            ra, dec, mag = np.nan, np.nan, np.nan
            print("NOT FOUND")
        ras.append(ra); decs.append(dec); mags.append(mag)

    out = gs_df.copy()
    out["ra_gsc"]  = ras
    out["dec_gsc"] = decs
    out["mag_gsc"] = mags
    return out


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

    Two modes controlled by `apparent`:

    apparent=False  (default — catalog / cross-match comparison)
        Returns the barycentric ICRS position: proper motion +
        perspective acceleration (vlos).  No aberration.  Use this to
        compare against GSC catalog positions (which are also barycentric
        ICRS astrometric coordinates, not apparent coordinates).

    apparent=True   (actual observation position)
        Returns the geocentric apparent position in GCRS: adds annual
        parallax factors (Earth's orbital offset) and annual aberration
        (~20") via an ICRS→GCRS frame transform.  Use this for computing
        where a star actually appears in an HST image or in the FGS field.

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
        # GCRS: annual parallax + aberration (actual apparent sky position)
        coord_gcrs = coord_t.transform_to(GCRS(obstime=target_time))
        return float(coord_gcrs.ra.deg), float(coord_gcrs.dec.deg)
    else:
        # Barycentric ICRS after proper motion + perspective acceleration
        return float(coord_t.ra.deg), float(coord_t.dec.deg)


def get_guide_star_position_at_epoch(
    gaia_source_id: int,
    obs_time: Time,
    guide_stars_csv: str | Path,
) -> tuple[float, float] | None:
    """
    Return the predicted (RA, Dec) in degrees for a guide star at obs_time,
    using stored Gaia astrometric parameters.  Returns None if the source
    is not found in the CSV or has no Gaia data.
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
        apparent=True,   # actual apparent position for image/FGS use
    )


# ── Step 3: Gaia cone search + cross-match ────────────────────────────────────

def _gaia_cone_search(ra: float, dec: float,
                      radius_arcsec: float = 30.0,
                      max_gmag: float = 16.0) -> pd.DataFrame | None:
    """
    Query Gaia DR3 via TAP for sources within radius_arcsec of (ra, dec)
    brighter than max_gmag.  Returns DataFrame or None.
    Includes radial_velocity for perspective-acceleration propagation.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    adql = f"""
    SELECT source_id, ra, dec, pmra, pmdec, parallax,
           radial_velocity,
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
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
        print(f"    WARNING: Gaia TAP query failed: {e}")
        return None


def crossmatch_to_gaia(gs_df: pd.DataFrame,
                       search_radius_arcsec: float = 30.0) -> pd.DataFrame:
    """
    Cross-match each guide star to Gaia DR3.

    For each candidate Gaia source in the search cone, propagate its
    position from J2016.0 → J2000.0 — including proper motion, perspective
    acceleration, and annual parallax factors — then rank by angular
    separation from the GSC J2000.0 position.  The closest propagated
    position wins.

    The large search radius (default 30") is needed because high-PM guide
    stars can move >10" between J2016 and J2000; ranking by propagated
    sep instead of catalog sep correctly identifies the guide star regardless
    of its proper motion.

    Adds columns:
      gaia_source_id         Gaia DR3 source_id (int64)
      ra_gaia / dec_gaia     Gaia J2016.0 ICRS position (deg)
      pmra_gaia / pmdec_gaia proper motion (mas/yr, pmra×cos δ)
      parallax_gaia          Gaia parallax (mas)
      vlos_gaia              Gaia radial velocity (km/s)
      gmag_gaia              Gaia G magnitude
      sep_catalog_arcsec     Gaia J2016 vs GSC J2000 separation (")
      sep_J2000_arcsec       Gaia propagated to J2000 vs GSC J2000 (")
    """
    new_cols = [
        "gaia_source_id",
        "ra_gaia", "dec_gaia",
        "pmra_gaia", "pmdec_gaia",
        "parallax_gaia", "vlos_gaia",
        "gmag_gaia",
        "sep_catalog_arcsec",
        "sep_J2000_arcsec",
    ]
    for col in new_cols:
        gs_df[col] = np.nan
    gs_df["gaia_source_id"] = gs_df["gaia_source_id"].astype(object)

    for idx, row in gs_df.iterrows():
        ra_gsc, dec_gsc = row["ra_gsc"], row["dec_gsc"]
        if np.isnan(ra_gsc) or np.isnan(dec_gsc):
            continue

        print(f"  Gaia search for {row['gsc_id']} "
              f"at ({ra_gsc:.5f}, {dec_gsc:.5f}) ...", end=" ")

        candidates = _gaia_cone_search(ra_gsc, dec_gsc,
                                       radius_arcsec=search_radius_arcsec,
                                       max_gmag=16.0)
        if candidates is None or candidates.empty:
            print("no candidates")
            continue

        gsc_sky = SkyCoord(ra_gsc * u.deg, dec_gsc * u.deg)

        best_sep_j2000 = np.inf
        best_row       = None
        best_cat_sep   = np.nan

        for _, cand in candidates.iterrows():
            pmra  = float(cand["pmra"])  if np.isfinite(float(cand["pmra"]  or np.nan)) else 0.0
            pmdec = float(cand["pmdec"]) if np.isfinite(float(cand["pmdec"] or np.nan)) else 0.0
            plx   = float(cand["parallax"]) if np.isfinite(float(cand["parallax"] or np.nan)) else np.nan
            vlos  = float(cand["radial_velocity"]) if np.isfinite(
                float(cand["radial_velocity"] or np.nan)) else np.nan

            ra_j2000, dec_j2000 = propagate_gaia_to_epoch(
                float(cand["ra"]), float(cand["dec"]),
                pmra, pmdec, plx, vlos,
                target_time=_GSC_EPOCH,
                apparent=False,  # catalog comparison: no aberration
            )
            sep_j2000 = gsc_sky.separation(
                SkyCoord(ra_j2000 * u.deg, dec_j2000 * u.deg)
            ).arcsec

            if sep_j2000 < best_sep_j2000:
                best_sep_j2000 = sep_j2000
                best_row       = cand
                best_cat_sep   = float(cand["sep_arcsec_catalog"])

        if best_row is None:
            print("no match")
            continue

        vlos_val = float(best_row["radial_velocity"] or np.nan)
        if not np.isfinite(vlos_val):
            vlos_val = np.nan

        pmra_val  = float(best_row["pmra"]  or np.nan)
        pmdec_val = float(best_row["pmdec"] or np.nan)
        plx_val   = float(best_row["parallax"] or np.nan)

        gs_df.at[idx, "gaia_source_id"]     = int(best_row["source_id"])
        gs_df.at[idx, "ra_gaia"]            = float(best_row["ra"])
        gs_df.at[idx, "dec_gaia"]           = float(best_row["dec"])
        gs_df.at[idx, "pmra_gaia"]          = pmra_val if np.isfinite(pmra_val)  else np.nan
        gs_df.at[idx, "pmdec_gaia"]         = pmdec_val if np.isfinite(pmdec_val) else np.nan
        gs_df.at[idx, "parallax_gaia"]      = plx_val if np.isfinite(plx_val)   else np.nan
        gs_df.at[idx, "vlos_gaia"]          = vlos_val
        gs_df.at[idx, "gmag_gaia"]          = float(best_row["phot_g_mean_mag"])
        gs_df.at[idx, "sep_catalog_arcsec"] = best_cat_sep
        gs_df.at[idx, "sep_J2000_arcsec"]   = float(best_sep_j2000)

        pm_mag = float(np.hypot(pmra_val  if np.isfinite(pmra_val)  else 0.0,
                                pmdec_val if np.isfinite(pmdec_val) else 0.0))
        vlos_str = (f"  vlos={vlos_val:.1f} km/s"
                    if np.isfinite(vlos_val) else "")
        print(f"source_id={int(best_row['source_id'])}  "
              f"G={float(best_row['phot_g_mean_mag']):.2f}  "
              f"sep_cat={best_cat_sep:.3f}\"  sep_J2000={best_sep_j2000:.3f}\"  "
              f"PM={pm_mag:.1f} mas/yr{vlos_str}")

    return gs_df


# ── Incremental obs-epoch update ──────────────────────────────────────────────

def _update_obs_stats(existing_df: pd.DataFrame,
                      current_scan: pd.DataFrame) -> pd.DataFrame:
    """
    Update n_dominant, n_subdominant, mean_obs_mjd from current_scan for
    all matching gsc_ids in existing_df.  Re-propagates ra_gaia_obs /
    dec_gaia_obs using the (unchanged) Gaia astrometric parameters.
    """
    scan_map = current_scan.set_index("gsc_id")

    for idx, row in existing_df.iterrows():
        gid = row["gsc_id"]
        if gid not in scan_map.index:
            continue
        sr = scan_map.loc[gid]
        existing_df.at[idx, "n_dominant"]    = int(sr["n_dominant"])
        existing_df.at[idx, "n_subdominant"] = int(sr["n_subdominant"])
        existing_df.at[idx, "mean_obs_mjd"]  = float(sr["mean_obs_mjd"])

    return existing_df


# ── main entry ────────────────────────────────────────────────────────────────

def download_guide_stars(hst_dir: str | Path,
                         field_name: str | None = None,
                         force: bool = False) -> Path | None:
    """
    Full pipeline: collect IDs → VizieR → Gaia (with J2000 propagation for
    cross-match ranking) → save CSV.

    Incremental mode (default): if the CSV already exists, only new guide
    stars (not yet in the CSV) are resolved and appended; existing entries
    have their obs statistics refreshed.  Pass force=True to regenerate
    from scratch.
    """
    hst_dir = Path(hst_dir)

    if field_name is None:
        field_name = hst_dir.parent.name

    out_csv = hst_dir / f"{field_name}_guide_stars.csv"

    # Always scan SPT files (fast, local only)
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

        # Refresh obs stats for all existing entries
        existing_df = _update_obs_stats(existing_df, current_scan)

        if not new_ids:
            print(f"  No new guide stars ({len(existing_ids)} already resolved). "
                  "Obs statistics refreshed.")
            existing_df.to_csv(out_csv, index=False)
            return out_csv

        print(f"  {len(new_ids)} new guide star(s): {', '.join(sorted(new_ids))}")
        new_rows_scan = current_scan[current_scan["gsc_id"].isin(new_ids)].copy()

        print("  Resolving new positions from GSC catalogs (VizieR)...")
        new_rows = resolve_gsc_positions(new_rows_scan)
        n_res = new_rows["ra_gsc"].notna().sum()
        print(f"  Resolved {n_res}/{len(new_rows)} new guide star positions.")

        if n_res > 0:
            print("  Cross-matching new guide stars to Gaia DR3 "
                  "(Gaia J2016 → J2000 propagation)...")
            new_rows = crossmatch_to_gaia(new_rows)
            n_match = new_rows["gaia_source_id"].notna().sum()
            print(f"  Matched {n_match}/{n_res} new guide stars to Gaia DR3.")

        # Normalise gaia_source_id dtype before concat
        _fix_gaia_id_dtype(new_rows)
        _fix_gaia_id_dtype(existing_df)

        merged = pd.concat([existing_df, new_rows], ignore_index=True)
        merged.to_csv(out_csv, index=False)
        print(f"  Updated: {out_csv.name}")
        return out_csv

    # ── full fresh run ────────────────────────────────────────────────────
    print(f"  Found {len(current_scan)} unique guide star(s): "
          + ", ".join(current_scan["gsc_id"].tolist()))

    print("  Resolving positions from GSC catalogs (VizieR)...")
    gs_df = resolve_gsc_positions(current_scan)
    n_resolved = gs_df["ra_gsc"].notna().sum()
    print(f"  Resolved {n_resolved}/{len(gs_df)} guide star positions.")

    if n_resolved > 0:
        print("  Cross-matching to Gaia DR3 "
              "(Gaia J2016 → J2000 propagation for ranking)...")
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
