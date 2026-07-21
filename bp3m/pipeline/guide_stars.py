"""
Identify HST guide stars used across all exposures, resolve their sky
positions from GSC catalogs, and cross-match to Gaia EDR3.

Flow:
  1. Scan all SPT files → collect unique GSC IDs (from DGESTAR / SGESTAR).
  2. Query VizieR for each ID to get RA/Dec:
       N6UH###### format → GSC 2.3.2  (VizieR I/305, column GSC2.3)
       ##########  format → GSC 1.x   (VizieR I/254, column GSC)
  3. Targeted Gaia EDR3 TAP cone search (r=5") per guide star → unique match
     for bright, isolated stars (G~8-14).
  4. Save {field_name}_guide_stars.csv in hst_dir.

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
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


# ── GSC ID parsing ────────────────────────────────────────────────────────────

def _parse_guide_star_entry(raw: str) -> tuple[str, str]:
    """
    Parse a DGESTAR/SGESTAR keyword value into (gsc_id, fgs_unit).
    e.g. 'N6UH000265F1' → ('N6UH000265', 'F1')
         '0083300313F2' → ('0083300313', 'F2')
         'NONE'         → ('', '')
    """
    raw = raw.strip()
    if not raw or raw.upper() in ('NONE', 'N/A', ''):
        return '', ''
    if raw[-2:] in ('F1', 'F2', 'F3'):
        return raw[:-2], raw[-2:]
    return raw, ''


def _gsc_catalog_for_id(gsc_id: str) -> tuple[str, str]:
    """
    Return (vizier_catalog, id_column) appropriate for this GSC ID format.
    N6UH###### → GSC 2.3.2  (I/305, 'GSC2.3')  — starts with a letter
    ########## → GSC 1.x    (I/254, 'GSC')       — purely numeric
    """
    if gsc_id and gsc_id[0].isalpha():
        return 'I/305', 'GSC2.3'
    return 'I/254', 'GSC'


# ── Step 1: collect IDs from SPT files ───────────────────────────────────────

def collect_guide_star_ids(hst_dir: str | Path) -> pd.DataFrame:
    """
    Scan all *_spt.fits files and return a DataFrame of unique guide stars:
      gsc_id, fgs_unit, n_dominant, n_subdominant, vizier_catalog, id_column
    """
    hst_dir = Path(hst_dir)
    mast_root = hst_dir / "mastDownload" / "HST"

    counts: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_dominant": 0, "n_subdominant": 0})

    for spt_path in sorted(mast_root.rglob("*_spt.fits")):
        with fits.open(spt_path) as hdul:
            h = hdul[0].header
            dom_raw = h.get("DGESTAR", "") or ""
            sub_raw = h.get("SGESTAR", "") or ""
        dom_id, dom_fgs = _parse_guide_star_entry(dom_raw)
        sub_id, sub_fgs = _parse_guide_star_entry(sub_raw)
        if dom_id:
            counts[(dom_id, dom_fgs)]["n_dominant"] += 1
        if sub_id:
            counts[(sub_id, sub_fgs)]["n_subdominant"] += 1

    rows = []
    for (gsc_id, fgs_unit), c in sorted(counts.items()):
        cat, col = _gsc_catalog_for_id(gsc_id)
        rows.append({
            "gsc_id":        gsc_id,
            "fgs_unit":      fgs_unit,
            "n_dominant":    c["n_dominant"],
            "n_subdominant": c["n_subdominant"],
            "vizier_catalog": cat,
            "id_column":     col,
        })
    return pd.DataFrame(rows)


# ── Step 2: query VizieR for GSC positions ───────────────────────────────────

def _query_vizier_by_id(gsc_id: str, catalog: str, id_col: str,
                        retries: int = 3) -> tuple[float, float, float] | None:
    """
    Return (ra_deg, dec_deg, mag) from VizieR, or None if not found.
    Tries the given catalog; on failure retries with exponential backoff.
    """
    from astroquery.vizier import Vizier

    V = Vizier(columns=[id_col, "RAJ2000", "DEJ2000", "Vmag", "Fmag",
                        "jmag", "Pmag"],
               row_limit=5)
    delay = 5
    for attempt in range(retries):
        try:
            result = V.query_constraints(catalog=catalog,
                                         **{id_col: gsc_id})
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    VizieR query failed (attempt {attempt+1}): {e} — "
                      f"retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    WARNING: VizieR query failed for {gsc_id}: {e}")
                return None

    if not result or len(result) == 0 or len(result[0]) == 0:
        return None

    row = result[0][0]
    ra  = float(row["RAJ2000"])
    dec = float(row["DEJ2000"])
    # Pick whichever magnitude column has a value
    mag = np.nan
    for mcol in ("Vmag", "jmag", "Fmag", "Pmag"):
        if mcol in row.colnames:
            val = row[mcol]
            if val is not None and not (hasattr(val, 'mask') and val.mask):
                try:
                    mag = float(val)
                    break
                except (TypeError, ValueError):
                    pass
    return ra, dec, mag


def resolve_gsc_positions(gs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ra_gsc, dec_gsc, mag_gsc columns by querying VizieR for each ID.
    """
    ras, decs, mags = [], [], []
    for _, row in gs_df.iterrows():
        print(f"  Querying {row['vizier_catalog']} for {row['gsc_id']} ...", end=" ")
        result = _query_vizier_by_id(row["gsc_id"], row["vizier_catalog"],
                                     row["id_column"])
        if result:
            ra, dec, mag = result
            print(f"RA={ra:.5f}  Dec={dec:.5f}  mag={mag:.2f}")
        else:
            ra, dec, mag = np.nan, np.nan, np.nan
            print("NOT FOUND")
        ras.append(ra)
        decs.append(dec)
        mags.append(mag)

    out = gs_df.copy()
    out["ra_gsc"]  = ras
    out["dec_gsc"] = decs
    out["mag_gsc"] = mags
    return out


# ── Step 3: cross-match to Gaia EDR3 ─────────────────────────────────────────

def _gaia_cone_search(ra: float, dec: float, radius_arcsec: float = 5.0,
                      max_gmag: float = 16.0) -> pd.DataFrame | None:
    """
    Query Gaia EDR3 via TAP for sources within radius_arcsec of (ra, dec)
    brighter than max_gmag.  Returns DataFrame sorted by separation, or None.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    adql = f"""
    SELECT source_id, ra, dec, pmra, pmdec, parallax,
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
           DISTANCE(POINT('ICRS',{ra},{dec}),
                    POINT('ICRS',ra,dec)) * 3600.0 AS sep_arcsec
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
                     CIRCLE('ICRS',{ra},{dec},{radius_arcsec/3600:.6f}))
      AND phot_g_mean_mag < {max_gmag}
    ORDER BY sep_arcsec ASC
    """
    try:
        job = Gaia.launch_job(adql)
        tbl = job.get_results()
        if len(tbl) == 0:
            return None
        df = tbl.to_pandas()
        # Ensure source_id is int64 (Gaia IDs > 2^53 corrupt as float64)
        df["source_id"] = df["source_id"].astype("int64")
        return df
    except Exception as e:
        print(f"    WARNING: Gaia TAP query failed: {e}")
        return None


def crossmatch_to_gaia(gs_df: pd.DataFrame,
                       search_radius_arcsec: float = 5.0) -> pd.DataFrame:
    """
    For each guide star with a valid GSC position, find the nearest Gaia
    source within search_radius_arcsec.  Adds Gaia columns to gs_df.
    """
    gaia_cols = ["gaia_source_id", "ra_gaia", "dec_gaia",
                 "pmra_gaia", "pmdec_gaia", "parallax_gaia",
                 "gmag_gaia", "match_sep_arcsec"]
    for col in gaia_cols:
        gs_df[col] = np.nan
    gs_df["gaia_source_id"] = gs_df["gaia_source_id"].astype(object)

    for idx, row in gs_df.iterrows():
        ra, dec = row["ra_gsc"], row["dec_gsc"]
        if np.isnan(ra) or np.isnan(dec):
            continue
        print(f"  Gaia TAP search for {row['gsc_id']} "
              f"at ({ra:.5f}, {dec:.5f}) ...", end=" ")
        result = _gaia_cone_search(ra, dec,
                                   radius_arcsec=search_radius_arcsec,
                                   max_gmag=16.0)
        if result is None or result.empty:
            print("no match")
            continue
        best = result.iloc[0]
        gs_df.at[idx, "gaia_source_id"]   = int(best["source_id"])
        gs_df.at[idx, "ra_gaia"]          = float(best["ra"])
        gs_df.at[idx, "dec_gaia"]         = float(best["dec"])
        gs_df.at[idx, "pmra_gaia"]        = float(best["pmra"]) if best["pmra"] is not None else np.nan
        gs_df.at[idx, "pmdec_gaia"]       = float(best["pmdec"]) if best["pmdec"] is not None else np.nan
        gs_df.at[idx, "parallax_gaia"]    = float(best["parallax"]) if best["parallax"] is not None else np.nan
        gs_df.at[idx, "gmag_gaia"]        = float(best["phot_g_mean_mag"])
        gs_df.at[idx, "match_sep_arcsec"] = float(best["sep_arcsec"])
        print(f"source_id={int(best['source_id'])}  "
              f"G={float(best['phot_g_mean_mag']):.2f}  "
              f"sep={float(best['sep_arcsec']):.3f}\"")

    # Warn if separation is suspiciously large
    matched = gs_df["match_sep_arcsec"].notna()
    large_sep = gs_df.loc[matched & (gs_df["match_sep_arcsec"] > 2.0)]
    if not large_sep.empty:
        print(f"  WARNING: {len(large_sep)} guide star(s) matched with "
              f"sep > 2\": check GSC positions")

    return gs_df


# ── main entry ────────────────────────────────────────────────────────────────

def download_guide_stars(hst_dir: str | Path,
                         field_name: str | None = None,
                         force: bool = False) -> Path | None:
    """
    Full pipeline: collect IDs → VizieR → Gaia → save CSV.
    Returns path to the written CSV, or None if nothing to do.
    """
    hst_dir   = Path(hst_dir)
    mast_root = hst_dir / "mastDownload" / "HST"

    if field_name is None:
        field_name = hst_dir.parent.name

    out_csv = hst_dir / f"{field_name}_guide_stars.csv"
    if out_csv.exists() and not force:
        print(f"  Guide star catalog already exists ({out_csv.name}) — skipping.")
        return out_csv

    print("  Collecting guide star IDs from SPT files...")
    gs_df = collect_guide_star_ids(hst_dir)
    if gs_df.empty:
        print("  No guide star IDs found in SPT files.")
        return None
    print(f"  Found {len(gs_df)} unique guide star(s): "
          + ", ".join(gs_df["gsc_id"].tolist()))

    print("  Resolving positions from GSC catalogs (VizieR)...")
    gs_df = resolve_gsc_positions(gs_df)

    n_resolved = gs_df["ra_gsc"].notna().sum()
    print(f"  Resolved {n_resolved}/{len(gs_df)} guide star positions.")

    if n_resolved > 0:
        print("  Cross-matching to Gaia EDR3...")
        gs_df = crossmatch_to_gaia(gs_df)
        n_matched = gs_df["gaia_source_id"].notna().sum()
        print(f"  Matched {n_matched}/{n_resolved} guide stars to Gaia.")

    # Ensure int64 for Gaia source IDs
    mask = gs_df["gaia_source_id"].notna()
    gs_df.loc[mask, "gaia_source_id"] = (
        gs_df.loc[mask, "gaia_source_id"].astype("int64"))

    gs_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hst_dir",
                        help="Directory containing mastDownload/HST/")
    parser.add_argument("--field", default=None,
                        help="Field name for output filename (default: parent dir name)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing guide_stars.csv")
    args = parser.parse_args()
    download_guide_stars(args.hst_dir, field_name=args.field, force=args.force)


if __name__ == "__main__":
    main()
