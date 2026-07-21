"""
Identify HST guide stars used across all exposures, resolve their sky
positions from GSC catalogs, and cross-match to Gaia DR3.

Flow:
  1. Scan all *_spt.fits files → collect unique GSC IDs (DGESTAR/SGESTAR).
  2. Query VizieR for each ID to get RA/Dec/epoch:
       N6UH###### → GSC 2.3.2  (VizieR I/305, column GSC2.3)
       ##########  → GSC 1.x   (VizieR I/254, column GSC)
  3. Gaia DR3 TAP cone search (r=10") around each GSC position.
     For each candidate, propagate from Gaia J2016.0 → GSC catalog epoch
     using proper motion, perspective acceleration (vlos), and annual
     parallax (parallax factors via GCRS transform).  Best match is the
     candidate with the smallest separation after propagation.
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
from astropy.coordinates import SkyCoord, GCRS
from astropy.io import fits
from astropy.time import Time
import astropy.units as u


# Gaia DR3 reference epoch (Julian year, TCB scale)
_GAIA_EPOCH = Time(2016.0, format="jyear", scale="tcb")

# GSC catalog epoch (J2000.0 for both GSC 1.x and GSC 2.3.2)
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
    usage counts.
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
            "gsc_id":         gsc_id,
            "fgs_unit":       fgs_unit,
            "n_dominant":     c["n_dominant"],
            "n_subdominant":  c["n_subdominant"],
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


# ── Step 3: Gaia propagation + cross-match ────────────────────────────────────

def propagate_gaia_to_epoch(
    ra: float, dec: float,
    pmra: float, pmdec: float,
    parallax: float, radial_velocity: float,
    target_time: Time,
) -> tuple[float, float]:
    """
    Propagate a Gaia DR3 source from J2016.0 to target_time and return the
    apparent geocentric (RA, Dec) in degrees.

    Accounts for:
      - Proper motion  (linear barycentric propagation via apply_space_motion)
      - Perspective acceleration  (change in angular velocity due to radial
        motion toward/away from observer; only when vlos and parallax known)
      - Annual parallax factors  (Earth's orbital offset from barycenter;
        via GCRS frame transform, requires positive parallax)

    Parameters
    ----------
    ra, dec         : Gaia ICRS position at J2016.0 (degrees)
    pmra, pmdec     : proper motion (mas/yr); pmra is pmra×cos(dec)
    parallax        : mas; ≤0 or NaN → no distance, skip parallax/persp.
    radial_velocity : km/s; NaN → skip perspective acceleration
    target_time     : astropy Time of the observation
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

    # Propagate proper motion (+ perspective acceleration if vlos & dist known)
    coord_t = coord.apply_space_motion(new_obstime=target_time)

    if have_dist:
        # Transform to GCRS: applies annual parallax via Earth's orbital offset
        coord_gcrs = coord_t.transform_to(GCRS(obstime=target_time))
        return float(coord_gcrs.ra.deg), float(coord_gcrs.dec.deg)
    else:
        return float(coord_t.ra.deg), float(coord_t.dec.deg)


def _gaia_cone_search(ra: float, dec: float,
                      radius_arcsec: float = 10.0,
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
                       search_radius_arcsec: float = 10.0,
                       gsc_epoch: Time = _GSC_EPOCH) -> pd.DataFrame:
    """
    Cross-match each guide star to Gaia DR3.

    For each candidate Gaia source in the search cone, propagate its
    position from J2016.0 to gsc_epoch (J2000.0) — including proper
    motion, perspective acceleration, and parallax factors — then rank
    by angular separation from the GSC position.  The closest propagated
    position wins.

    Adds columns: gaia_source_id, ra_gaia, dec_gaia, pmra_gaia, pmdec_gaia,
    parallax_gaia, vlos_gaia, gmag_gaia,
    sep_catalog_arcsec (raw catalog distance to GSC position),
    sep_propagated_arcsec (after propagation to GSC epoch).
    """
    new_cols = [
        "gaia_source_id",
        "ra_gaia", "dec_gaia",
        "pmra_gaia", "pmdec_gaia",
        "parallax_gaia", "vlos_gaia",
        "gmag_gaia",
        "sep_catalog_arcsec",
        "sep_propagated_arcsec",
    ]
    for col in new_cols:
        gs_df[col] = np.nan
    gs_df["gaia_source_id"] = gs_df["gaia_source_id"].astype(object)

    gsc_coord_ref = None  # will hold SkyCoord of current GSC position

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

        best_sep  = np.inf
        best_row  = None

        for _, cand in candidates.iterrows():
            pmra  = cand["pmra"]  if (cand["pmra"]  is not None
                                      and np.isfinite(cand["pmra"]))  else 0.0
            pmdec = cand["pmdec"] if (cand["pmdec"] is not None
                                      and np.isfinite(cand["pmdec"])) else 0.0
            plx   = cand["parallax"] if (cand["parallax"] is not None
                                         and np.isfinite(cand["parallax"])) else np.nan
            vlos  = cand["radial_velocity"] if (
                cand["radial_velocity"] is not None
                and np.isfinite(cand["radial_velocity"])) else np.nan

            ra_prop, dec_prop = propagate_gaia_to_epoch(
                float(cand["ra"]), float(cand["dec"]),
                float(pmra), float(pmdec),
                float(plx), float(vlos),
                target_time=gsc_epoch,
            )
            sep = gsc_sky.separation(
                SkyCoord(ra_prop * u.deg, dec_prop * u.deg)
            ).arcsec

            if sep < best_sep:
                best_sep = sep
                best_row = cand
                best_ra_prop, best_dec_prop = ra_prop, dec_prop

        if best_row is None:
            print("no match after propagation")
            continue

        vlos_val = best_row["radial_velocity"]
        if vlos_val is None or not np.isfinite(float(vlos_val)):
            vlos_val = np.nan

        gs_df.at[idx, "gaia_source_id"]        = int(best_row["source_id"])
        gs_df.at[idx, "ra_gaia"]               = float(best_row["ra"])
        gs_df.at[idx, "dec_gaia"]              = float(best_row["dec"])
        gs_df.at[idx, "pmra_gaia"]             = float(best_row["pmra"]) if np.isfinite(float(best_row["pmra"] or np.nan)) else np.nan
        gs_df.at[idx, "pmdec_gaia"]            = float(best_row["pmdec"]) if np.isfinite(float(best_row["pmdec"] or np.nan)) else np.nan
        gs_df.at[idx, "parallax_gaia"]         = float(best_row["parallax"]) if np.isfinite(float(best_row["parallax"] or np.nan)) else np.nan
        gs_df.at[idx, "vlos_gaia"]             = float(vlos_val) if np.isfinite(float(vlos_val)) else np.nan
        gs_df.at[idx, "gmag_gaia"]             = float(best_row["phot_g_mean_mag"])
        gs_df.at[idx, "sep_catalog_arcsec"]    = float(best_row["sep_arcsec_catalog"])
        gs_df.at[idx, "sep_propagated_arcsec"] = float(best_sep)

        vlos_str = f"  vlos={float(vlos_val):.1f} km/s" if np.isfinite(float(vlos_val)) else ""
        print(f"source_id={int(best_row['source_id'])}  "
              f"G={float(best_row['phot_g_mean_mag']):.2f}  "
              f"sep_cat={float(best_row['sep_arcsec_catalog']):.3f}\"  "
              f"sep_prop={best_sep:.3f}\"{vlos_str}")

    # Warn about large propagated separations
    matched = gs_df["sep_propagated_arcsec"].notna()
    large = gs_df.loc[matched & (gs_df["sep_propagated_arcsec"] > 1.0)]
    if not large.empty:
        print(f"  WARNING: {len(large)} guide star(s) have propagated "
              f"sep > 1\" — verify GSC positions and Gaia cross-match")

    return gs_df


# ── main entry ────────────────────────────────────────────────────────────────

def download_guide_stars(hst_dir: str | Path,
                         field_name: str | None = None,
                         force: bool = False) -> Path | None:
    """
    Full pipeline: collect IDs → VizieR → Gaia (with propagation) → save CSV.
    """
    hst_dir = Path(hst_dir)

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
        print("  Cross-matching to Gaia DR3 (with proper motion + "
              "parallax factor + vlos propagation)...")
        gs_df = crossmatch_to_gaia(gs_df)
        n_matched = gs_df["gaia_source_id"].notna().sum()
        print(f"  Matched {n_matched}/{n_resolved} guide stars to Gaia DR3.")

    # Ensure Gaia source IDs are stored as int64
    mask = gs_df["gaia_source_id"].notna()
    if mask.any():
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
                        help="Field name for output filename")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing guide_stars.csv")
    args = parser.parse_args()
    download_guide_stars(args.hst_dir, field_name=args.field, force=args.force)


if __name__ == "__main__":
    main()
