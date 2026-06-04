"""
data_loader_master.py  —  BP3M v2 data loader from master_combined_v2.csv.

Reads the master HST cross-match catalog and builds BP3M input arrays that
include both Gaia-matched stars (with full Gaia prior) and HST-only stars
(with zero Gaia inverse covariance, Michalik parallax prior, 100 mas/yr PM
prior).

Intended for use with run_alignment_v2.py / V2AlignmentCallback.

Interface
---------
(images, stars_per_image, gaia_catalog, hst_only_mask) = load_master_v2(...)

This is the same (images, stars_per_image, gaia_catalog) triple expected by
BP3MSolver, augmented with hst_only_mask — a boolean array of length n_stars
that is True for synthetic HST-only rows in gaia_catalog.

HST-only source treatment
--------------------------
- Gaia_id : sequential negative int64 (e.g. -1, -2, ...) — never conflicts
  with positive Gaia source IDs.
- gaia_catalog row : ra/dec from ra_xmatch/dec_xmatch; pmra=pmdec=parallax=NaN;
  pseudocolour=NaN → solver classifies as gaia_2p → Michalik+100-mas/yr prior.
  ra_error=dec_error=1e6 → effectively flat position prior.
- use_for_alignment = False in stars_per_image → solver initialises use_for_fit=False.
  V2AlignmentCallback flips use_for_fit=True at iteration ≥ hst_enable_iter.

Detection set construction
---------------------------
For each source row in master_combined_v2.csv:
  - If pass2_hst_indices is populated: use it as the complete detection set
    (it supersedes the per-filter hst_indices_* columns, which are a subset).
  - Otherwise: union the hst_indices_* columns.
  - Remove any (sub_name, catalog_index) pair where sub_name appears in the
    outlier_images column.

Quality cuts applied at load time
-----------------------------------
HST-only sources must satisfy:
  sigma_pmra_xmatch < hst_max_pm_unc  (default 5 mas/yr)
  n_detect_fit >= hst_min_detect       (default 2)

Per-image cap:
  For each image, rank HST-only sources by sigma_pmra_xmatch and retain at
  most hst_max_per_image (default 1000).  Sources eliminated from an image
  lose that detection; if the remaining count drops below hst_min_detect the
  source is excluded entirely.

Detection uniqueness:
  Every (sub_name, catalog_index) pair must appear in at most one source row.
  An exception is raised if this invariant is violated.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

def _ensure_bp3m():
    pass  # bp3m is installed as a package; no sys.path manipulation needed


# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_SAT_FRAC      = 0.25   # matches data_loader_flc._MAX_SAT_FRAC
_MIN_POS_ERR_PX    = 5e-3   # minimum position error floor in pixels

# HST-only eligibility defaults
_HST_MAX_PM_UNC    = 5.0    # mas/yr — global quality cut
_HST_MIN_DETECT    = 2      # minimum detections after outlier removal + per-image cap
_HST_MAX_PER_IMAGE = 1000   # maximum HST-only sources contributing to any single image

# Gaia catalog columns to load
_GAIA_COLS = [
    "source_id", "ra", "dec", "ra_error", "dec_error", "ra_dec_corr",
    "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax", "parallax_error", "parallax_pmra_corr", "parallax_pmdec_corr",
    "pmra", "pmra_error", "pmra_pmdec_corr", "pmdec", "pmdec_error",
    "ref_epoch", "ruwe", "pseudocolour", "gmag", "gmag_error",
    "bpmag", "bpmag_error", "rpmag", "rpmag_error", "bp_rp", "bp_rp_error",
]

# Correlation and error columns that solver expects; filled with NaN if absent
_EXTRA_COLS = [
    "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
    "pmra_error", "pmdec_error", "parallax_error",
    "gmag_error", "bpmag", "bpmag_error", "rpmag", "rpmag_error",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _sub_name_to_base(sub_name: str) -> str:
    """Strip _hi/_lo CCD suffix to get the base image name."""
    for sfx in ("_lo", "_hi"):
        if sub_name.endswith(sfx):
            return sub_name[: -len(sfx)]
    return sub_name


def _parse_detections_column(val) -> list[tuple[str, int]]:
    """Parse 'sname:idx,sname:idx,...' into list of (sname, idx) pairs.

    The master catalog uses comma as the primary separator, but some rows
    also contain semicolons as a secondary separator (from the crossmatch
    pipeline writing inconsistently).  Both are handled here.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    # Normalise: replace semicolons with commas, then split
    normalised = str(val).replace(";", ",")
    result = []
    for token in normalised.split(","):
        token = token.strip()
        if ":" not in token:
            continue
        sname, cidx_str = token.rsplit(":", 1)
        try:
            result.append((sname.strip(), int(cidx_str.strip())))
        except ValueError:
            continue
    return result


def _parse_outlier_images(val) -> set[str]:
    """Parse outlier_images column (comma- or semicolon-separated sub_names)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    # Support both comma (current writer) and semicolon (legacy) separators.
    normalised = str(val).replace(";", ",")
    return {s.strip() for s in normalised.split(",") if s.strip()}


def _parse_det_chi2(val) -> dict[str, float]:
    """Parse det_chi2 column ('sname:chi2,...') into {sname: chi2} dict."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return {}
    result: dict[str, float] = {}
    for token in str(val).split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            continue
        try:
            result[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return result


def _estimate_gmag(row: pd.Series,
                   color_offsets: dict[str, float] | None = None) -> float:
    """Estimate G magnitude from available HST magnitudes.

    color_offsets maps filter name → median(G − filter_mag) computed from
    Gaia-matched sources.  When provided the G estimate is filter_mag +
    offset; without it the HST magnitude is used directly (cap at 20).
    """
    _FILTER_COLS = (
        ("F606W", "mag_wmean_F606W"),
        ("F475W", "mag_wmean_F475W"),
        ("F435W", "mag_wmean_F435W"),
        ("F814W", "mag_wmean_F814W"),
        ("F555W", "mag_wmean_F555W"),
    )
    for filt, col in _FILTER_COLS:
        val = row.get(col, np.nan)
        if isinstance(val, (int, float)) and np.isfinite(float(val)):
            fmag = float(val)
            if color_offsets and filt in color_offsets:
                return fmag + color_offsets[filt]
            return min(fmag, 20.0)
    return 20.0


def _load_fits_catalog(cat_path: Path) -> dict | None:
    """Load a _flc_catalog.fits file into numpy arrays. Returns None if missing."""
    if not cat_path.exists():
        return None
    with fits.open(cat_path, memmap=False) as h:
        tbl = h[1].data
        return {
            "x":      tbl["x"].astype(float),
            "y":      tbl["y"].astype(float),
            "x_gdc":  tbl["x_gdc"].astype(float),
            "y_gdc":  tbl["y_gdc"].astype(float),
            "cov_xx": tbl["cov_xx_gdc"].astype(float),
            "cov_yy": tbl["cov_yy_gdc"].astype(float),
            "cov_xy": tbl["cov_xy_gdc"].astype(float),
            "mag":    tbl["mag"].astype(float),
            "qfit":   tbl["qfit"].astype(float),
            "n_sat":  tbl["n_sat"].astype(int),
        }


# ── Public entry point ────────────────────────────────────────────────────────

def load_master_v2(
    data_root,
    field_name: str,
    hst_max_pm_unc: float = _HST_MAX_PM_UNC,
    hst_min_detect: int = _HST_MIN_DETECT,
    hst_max_per_image: int = _HST_MAX_PER_IMAGE,
    pos_err_floor: float = _MIN_POS_ERR_PX,
    det_chi2_threshold: float | None = None,
) -> tuple[dict, dict, pd.DataFrame, np.ndarray]:
    """
    Load BP3M v2 inputs from {field_dir}/hst_xmatch/master_combined_v2.csv.

    Parameters
    ----------
    data_root      : root output directory
    field_name     : field subdirectory name
    hst_max_pm_unc : global quality cut on sigma_pmra_xmatch for HST-only sources
    hst_min_detect : minimum detections after outlier removal + per-image cap
    hst_max_per_image : per-image cap on HST-only source count (ranked by sigma_pmra)
    pos_err_floor  : minimum positional uncertainty in pixels
    det_chi2_threshold : if set, exclude individual (star, image) detections whose
        per-detection chi2 from the Phase 4 fit exceeds this value.  Requires a
        ``det_chi2`` column in master_combined_v2.csv (written by
        hst_catalog_crossmatch.py ≥ current version).  Detections missing from
        det_chi2 (old catalogues) are kept.  Suggested value: 9.0 (≈3σ).

    Returns
    -------
    images : dict[sub_name -> meta dict]
        Image metadata dicts as returned by data_loader_flc._read_image_meta.
        Keys include _hi/_lo suffixes exactly as they appear in sub_names.
    stars_per_image : dict[sub_name -> pd.DataFrame]
        Per-image source tables.  HST-only sources have use_for_alignment=False.
    gaia_catalog : pd.DataFrame
        Rows 0..N_gaia-1 : real Gaia sources observed in these images.
        Rows N_gaia..N_gaia+N_hst-1 : synthetic HST-only rows (gaia_2p=True).
    hst_only_mask : (n_stars,) bool
        True for the synthetic HST-only rows in gaia_catalog.
    """
    _ensure_bp3m()
    from bp3m.data_loader_flc import _read_image_meta  # private but stable

    data_root  = Path(data_root)
    field_dir  = data_root / field_name
    xmatch_dir = field_dir / "hst_xmatch"
    hst_root   = field_dir / "HST" / "mastDownload" / "HST"
    gaia_dir   = field_dir / "Gaia"

    # ── Master catalog ────────────────────────────────────────────────────────
    master_path = xmatch_dir / "master_combined_v2.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"master_combined_v2.csv not found: {master_path}")

    print(f"  Loading master catalog: {master_path}")
    master = pd.read_csv(master_path, low_memory=False)
    # Ensure gaia_source_id is int64 without float64 precision loss.
    # New files (written after the int64 fix) store 0 for missing and have no NaN
    # in this column, so direct int64 conversion is safe.  Old files stored NaN
    # for HST-only rows, which forces float64.  In either case we convert via
    # the float→int64 lookup built from the Gaia catalog (built below), so we
    # defer this conversion until gaia_float_to_int64 is available.
    hst_idx_cols = [c for c in master.columns if c.startswith("hst_indices_")]

    # ── Real Gaia catalog ─────────────────────────────────────────────────────
    gaia_files = sorted(glob.glob(str(gaia_dir / "*_gaia.csv")))
    if not gaia_files:
        raise FileNotFoundError(f"No Gaia catalog files found in {gaia_dir}")

    # Read source_id as int64 for both column-name conventions.
    # Specifying both keys is safe: pandas silently ignores dtype entries
    # for columns that don't exist in the file.
    _gaia_id_dtype = {"source_id": np.int64, "SOURCE_ID": np.int64}
    gaia_raw = pd.concat(
        [pd.read_csv(f, dtype=_gaia_id_dtype).rename(columns={"SOURCE_ID": "source_id"})
         for f in gaia_files],
        ignore_index=True,
    ).drop_duplicates("source_id")

    gaia_real = (
        gaia_raw[[c for c in _GAIA_COLS if c in gaia_raw.columns]]
        .dropna(subset=["ra", "dec"])
        .sort_values("source_id")
        .drop_duplicates("source_id", keep="first")
        .reset_index(drop=True)
    )
    gaia_real.rename(columns={"source_id": "Gaia_id", "ref_epoch": "Gaia_time"}, inplace=True)

    # float64 → int64 lookup (Gaia IDs may have been stored as float in CSV)
    gaia_float_to_int64: dict[float, np.int64] = {
        float(gid): np.int64(gid) for gid in gaia_real["Gaia_id"].values
    }
    gaia_id_set: set = set(gaia_real["Gaia_id"].values)

    # Now that gaia_float_to_int64 is built, fix gaia_source_id column precision.
    # Old master files have float64 (NaN for HST-only); new files have int64 (0).
    # We use the float→int64 lookup to recover exact IDs for Gaia-matched rows.
    if "gaia_source_id" in master.columns:
        _raw_gids = pd.to_numeric(master["gaia_source_id"], errors="coerce").fillna(0.0)
        _corrected = np.zeros(len(master), dtype=np.int64)
        for _k, _fval in enumerate(_raw_gids.values):
            if _fval == 0.0:
                _corrected[_k] = 0
            elif int(_fval) in gaia_id_set:
                _corrected[_k] = int(_fval)
            else:
                # Try float→int64 lookup to recover precision
                _corrected[_k] = int(gaia_float_to_int64.get(float(_fval), int(_fval)))
        master["gaia_source_id"] = _corrected

    # ── Compute color offsets G − filter_mag from Gaia-matched rows ───────────
    # Used to convert HST magnitudes to an estimated G magnitude for HST-only
    # sources, giving a realistic trend instead of a pileup at G=20.
    color_offsets: dict[str, float] = {}
    if "gaia_source_id" in master.columns and "gmag" in gaia_real.columns:
        _gaia_gmag = gaia_real.set_index("Gaia_id")["gmag"]
        _matched = master[master.get("has_gaia_match", pd.Series(False)).astype(bool)].copy()
        if len(_matched) == 0:
            # Fallback: rows with a valid gaia_source_id
            _matched = master[pd.to_numeric(
                master["gaia_source_id"], errors='coerce').notna()].copy()
        _matched_gid = pd.to_numeric(
            _matched["gaia_source_id"], errors='coerce').fillna(0).astype(np.int64)
        _matched_gmag = _matched_gid.map(_gaia_gmag)

        for _filt, _col in (("F606W", "mag_wmean_F606W"), ("F475W", "mag_wmean_F475W"),
                             ("F435W", "mag_wmean_F435W"), ("F814W", "mag_wmean_F814W"),
                             ("F555W", "mag_wmean_F555W")):
            if _col not in _matched.columns:
                continue
            _hst_mag = pd.to_numeric(_matched[_col], errors='coerce')
            _valid = np.isfinite(_hst_mag.values) & np.isfinite(_matched_gmag.values)
            if _valid.sum() >= 5:
                color_offsets[_filt] = float(np.median(
                    (_matched_gmag.values - _hst_mag.values)[_valid]))

    # ── Parse detections and classify sources ─────────────────────────────────
    print(f"  Parsing {len(master)} source rows...")

    source_records: list[dict] = []

    for row_i, row in master.iterrows():
        # Select detection set: pass2_hst_indices supersedes hst_indices_*
        pass2 = _parse_detections_column(row.get("pass2_hst_indices", ""))
        if pass2:
            pairs = pass2
        else:
            pairs = []
            for col in hst_idx_cols:
                pairs.extend(_parse_detections_column(row.get(col, "")))
            # deduplicate while preserving order
            seen: set = set()
            deduped = []
            for p in pairs:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            pairs = deduped

        # Phase 4 outliers: keep them in a SEPARATE list so they enter the
        # solver with use_for_alignment=False but are never discarded.
        # The BP3M EM loop can re-enable them (test-3 re-admission) if their
        # residuals are acceptable at the current transformation.
        _outlier_subs = set(_parse_outlier_images(row.get("outlier_images", "")))
        outlier_pairs = [(s, c) for s, c in pairs if s in _outlier_subs]
        pairs         = [(s, c) for s, c in pairs if s not in _outlier_subs]
        # Skip only if there are no detections at all (primary OR outlier).
        if not pairs and not outlier_pairs:
            continue

        # Remove detections whose Phase 4 per-detection chi2 exceeds threshold
        if det_chi2_threshold is not None:
            _det_chi2_map = _parse_det_chi2(row.get("det_chi2", ""))
            if _det_chi2_map:
                pairs = [
                    (s, c) for s, c in pairs
                    if _det_chi2_map.get(s, 0.0) <= det_chi2_threshold
                ]
            if not pairs:
                continue

        # Classify: has_gaia / gaia_id
        has_gaia = bool(row.get("has_gaia_match", False))
        gaia_id: np.int64 | None = None
        if has_gaia:
            raw_id = row.get("gaia_source_id", 0)
            try:
                # gaia_source_id column was corrected to int64 above, so raw_id
                # is already an exact int64 value (0 for missing).
                gaia_id = np.int64(int(raw_id))
                if gaia_id == 0:
                    has_gaia = False
                elif gaia_id not in gaia_id_set:
                    print(f"  Warning: gaia_source_id {gaia_id} (row {row_i}) "
                          f"not in Gaia catalog — skipping source")
                    continue
            except (ValueError, TypeError):
                has_gaia = False

        source_records.append({
            "has_gaia":       has_gaia,
            "gaia_id":        gaia_id,
            "ra":             float(row["ra_xmatch"]),
            "dec":            float(row["dec_xmatch"]),
            "gmag":           _estimate_gmag(row, color_offsets=color_offsets),
            "sigma_pmra":     float(row.get("sigma_pmra_xmatch", np.inf)),
            "n_detect_fit":   int(row.get("n_detect_fit", len(pairs))),
            "detections":     pairs,           # primary (non-outlier) detections
            "outlier_detections": outlier_pairs,  # Phase-6 flagged; start inactive
            # Crossmatch PM estimates — used to seed v_survey when HST-only sources
            # are first enabled (avoids bulk-PM=0 bias pulling the transformation)
            "pmra_xmatch":    float(row.get("pmra_xmatch",  np.nan) or np.nan),
            "pmdec_xmatch":   float(row.get("pmdec_xmatch", np.nan) or np.nan),
            # PM magnitude and parallax from crossmatch (for HST-only quality cuts)
            "pm_abs_masyr":   float(np.hypot(
                float(row.get("pmra_xmatch",  0) or 0),
                float(row.get("pmdec_xmatch", 0) or 0))),
            "parallax_xmatch_abs": abs(float(row.get("parallax_xmatch", 0) or 0)),
            # Per-image Phase-6 chi2 values: {sub_name: chi2}.  Used by the
            # soft-weight IRLS to warm-start z weights without an extra solve.
            "det_chi2_by_img": _parse_det_chi2(row.get("det_chi2", "")),
        })

    n_gaia_srcs  = sum(1 for r in source_records if r["has_gaia"])
    n_hst_srcs   = sum(1 for r in source_records if not r["has_gaia"])
    print(f"  Parsed: {n_gaia_srcs} Gaia, {n_hst_srcs} HST-only sources")

    # ── Global quality cut for HST-only ──────────────────────────────────────
    # Thresholds for excluding implausibly large PM/parallax HST-only sources.
    # Large PM/plx are likely measurement artefacts (bad cross-matches, blends,
    # cosmic rays) that would torque the transformation if included.
    _HST_MAX_PM_ABS    = 30.0   # mas/yr  — total PM magnitude
    _HST_MAX_PLX_ABS   =  5.0   # mas     — absolute parallax
    eligible: list[dict] = []
    for rec in source_records:
        if rec["has_gaia"]:
            eligible.append(rec)
        elif (rec["sigma_pmra"] < hst_max_pm_unc and
              rec["n_detect_fit"] >= hst_min_detect and
              rec.get("pm_abs_masyr", 0.0) <= _HST_MAX_PM_ABS and
              rec.get("parallax_xmatch_abs", 0.0) <= _HST_MAX_PLX_ABS):
            eligible.append(rec)

    n_elig_hst = sum(1 for r in eligible if not r["has_gaia"])
    print(f"  After global quality cut (σ_PM<{hst_max_pm_unc}, n≥{hst_min_detect}, "
          f"|PM|<{_HST_MAX_PM_ABS}, |plx|<{_HST_MAX_PLX_ABS}): "
          f"{n_elig_hst} HST-only eligible")

    # ── Per-image top-N cap for HST-only ──────────────────────────────────────
    # Build: sub_name → [(sigma_pmra, src_idx_in_eligible), ...]
    img_hst_candidates: dict[str, list[tuple[float, int]]] = {}
    for src_i, rec in enumerate(eligible):
        if not rec["has_gaia"]:
            for sub_name, _ in rec["detections"]:
                img_hst_candidates.setdefault(sub_name, []).append(
                    (rec["sigma_pmra"], src_i)
                )

    # For each image, retain only top hst_max_per_image HST-only sources
    image_include: dict[str, set[int]] = {}
    for sub_name, candidates in img_hst_candidates.items():
        candidates.sort(key=lambda t: t[0])
        image_include[sub_name] = {src_i for _, src_i in candidates[:hst_max_per_image]}

    # Apply cap: rebuild detection lists for HST-only sources
    valid_recs: list[dict] = []
    for src_i, rec in enumerate(eligible):
        if rec["has_gaia"]:
            valid_recs.append(rec)
            continue
        # Keep only detections in images where this source is in the top-N
        kept = [(s, c) for s, c in rec["detections"]
                if src_i in image_include.get(s, set())]
        if len(kept) >= hst_min_detect:
            new_rec = dict(rec)
            new_rec["detections"] = kept
            valid_recs.append(new_rec)

    n_valid_gaia = sum(1 for r in valid_recs if r["has_gaia"])
    n_valid_hst  = sum(1 for r in valid_recs if not r["has_gaia"])
    print(f"  After per-image cap (top-{hst_max_per_image}): "
          f"{n_valid_gaia} Gaia, {n_valid_hst} HST-only sources")

    # ── Detection uniqueness: deduplicate (first-in-wins by quality order) ────
    # Sort valid_recs so that higher-quality sources claim their detections first.
    # Gaia sources first, then by n_detect_fit desc, then sigma_pmra asc.
    # This ensures that when the master catalog has degenerate duplicate rows
    # (sigma_pmra=NaN, n_detect_fit=0), the valid source keeps its detections.
    def _quality_key(r):
        gaia_first = 0 if r["has_gaia"] else 1
        ndet = -r.get("n_detect_fit", 0)      # higher n_detect first
        sigma = r.get("sigma_pmra", np.inf)    # lower sigma first
        return (gaia_first, ndet, sigma)

    valid_recs.sort(key=_quality_key)

    # Assign Gaia_ids in sorted order (used for dedup tracking)
    next_hst_id = np.int64(-1)
    for rec in valid_recs:
        if rec["has_gaia"]:
            rec["Gaia_id"] = np.int64(rec["gaia_id"])
        else:
            rec["Gaia_id"] = next_hst_id
            next_hst_id -= 1

    detection_owner: dict[tuple[str, int], np.int64] = {}
    dedup_recs: list[dict] = []
    n_stripped = 0
    for rec in valid_recs:
        clean_dets = []
        for det in rec["detections"]:
            if det not in detection_owner:
                detection_owner[det] = rec["Gaia_id"]
                clean_dets.append(det)
            else:
                n_stripped += 1
        if not rec["has_gaia"] and len(clean_dets) < hst_min_detect:
            continue  # source lost too many detections to conflicts
        new_rec = dict(rec)
        new_rec["detections"] = clean_dets
        dedup_recs.append(new_rec)

    if n_stripped > 0:
        print(f"  Deduplication: removed {n_stripped} conflicting detections "
              f"({len(valid_recs) - len(dedup_recs)} sources dropped below threshold)")
    valid_recs = dedup_recs

    # ── Load image metadata ───────────────────────────────────────────────────
    all_sub_names: set[str] = {s for rec in valid_recs for s, _ in rec["detections"]}

    images: dict[str, dict] = {}
    skipped_meta: list[str] = []
    base_meta_cache: dict[str, dict | None] = {}

    for sub_name in sorted(all_sub_names):
        base = _sub_name_to_base(sub_name)
        if base not in base_meta_cache:
            img_dir = hst_root / base
            base_meta_cache[base] = _read_image_meta(img_dir, base)
        meta = base_meta_cache[base]
        if meta is None:
            skipped_meta.append(sub_name)
        else:
            images[sub_name] = meta   # same dict for _lo and _hi (identical metadata)

    if skipped_meta:
        print(f"  Warning: metadata missing for {len(skipped_meta)} sub-images "
              f"(skipped): {skipped_meta[:5]}{'...' if len(skipped_meta)>5 else ''}")

    valid_sub_names = set(images.keys())

    # ── Load FITS catalogs (cached per base image) ────────────────────────────
    fits_cache: dict[str, dict | None] = {}
    psf_hw_cache: dict[str, int] = {}

    def _get_fits(sub_name: str) -> tuple[dict | None, int]:
        base = _sub_name_to_base(sub_name)
        if base not in fits_cache:
            cat_path = hst_root / base / f"{base}_flc_catalog.fits"
            fits_cache[base] = _load_fits_catalog(cat_path)
            psf_path = hst_root / base / "psf_params.json"
            if psf_path.exists():
                with open(psf_path) as f:
                    hw = int(json.load(f).get("half_width", 3))
            else:
                hw = 3
            psf_hw_cache[base] = hw
        return fits_cache[base], psf_hw_cache[base]

    # ── Group detections by sub_name ──────────────────────────────────────────
    img_records: dict[str, list[dict]] = {}
    for rec in valid_recs:
        _chi2_by_img = rec.get("det_chi2_by_img", {})
        # Primary (non-outlier) detections
        for sub_name, cat_idx in rec["detections"]:
            if sub_name not in valid_sub_names:
                continue
            img_records.setdefault(sub_name, []).append({
                "Gaia_id":      rec["Gaia_id"],
                "cat_idx":      cat_idx,
                "is_gaia":      rec["has_gaia"],
                "is_outlier":   False,
                "det_chi2_val": _chi2_by_img.get(sub_name, np.nan),
            })
        # Outlier detections: included but flagged inactive at startup
        for sub_name, cat_idx in rec.get("outlier_detections", []):
            if sub_name not in valid_sub_names:
                continue
            img_records.setdefault(sub_name, []).append({
                "Gaia_id":      rec["Gaia_id"],
                "cat_idx":      cat_idx,
                "is_gaia":      rec["has_gaia"],
                "is_outlier":   True,
                "det_chi2_val": _chi2_by_img.get(sub_name, np.nan),
            })

    # ── Build per-image DataFrames ────────────────────────────────────────────
    stars_per_image: dict[str, pd.DataFrame] = {}
    skipped_fits: list[str] = []

    for sub_name in sorted(img_records.keys()):
        recs_img = img_records[sub_name]
        fits_data, half_width = _get_fits(sub_name)
        if fits_data is None:
            skipped_fits.append(sub_name)
            continue

        n = len(recs_img)
        window_area = (2 * half_width + 1) ** 2
        n_cat = len(fits_data["x"])

        gaia_ids    = np.empty(n, dtype=np.int64)
        X           = np.full(n, np.nan)
        Y           = np.full(n, np.nan)
        X_orig      = np.full(n, np.nan)
        Y_orig      = np.full(n, np.nan)
        x_err       = np.ones(n)
        y_err       = np.ones(n)
        xy_cor      = np.zeros(n)
        qfit_arr    = np.full(n, -1.0)
        mag_arr     = np.full(n, np.nan)
        is_gaia     = np.zeros(n, dtype=bool)
        ok_sat      = np.zeros(n, dtype=bool)
        is_outlier  = np.zeros(n, dtype=bool)   # Phase 6-flagged outlier
        det_chi2_arr = np.full(n, np.nan)        # Phase 6 per-detection chi2

        for k, r in enumerate(recs_img):
            ci = r["cat_idx"]
            gaia_ids[k]   = r["Gaia_id"]
            is_gaia[k]       = r["is_gaia"]
            is_outlier[k]    = r.get("is_outlier", False)
            det_chi2_arr[k]  = r.get("det_chi2_val", np.nan)

            if ci < 0 or ci >= n_cat:
                continue  # leaves NaN X/Y → filtered out below

            cxx = fits_data["cov_xx"][ci]
            cyy = fits_data["cov_yy"][ci]
            cxy = fits_data["cov_xy"][ci]
            sx = max(np.sqrt(max(cxx, 0.0)), pos_err_floor)
            sy = max(np.sqrt(max(cyy, 0.0)), pos_err_floor)
            denom = sx * sy
            rho = float(np.clip(cxy / denom if denom > 0 else 0.0, -0.9999, 0.9999))

            X[k]         = fits_data["x_gdc"][ci]
            Y[k]         = fits_data["y_gdc"][ci]
            X_orig[k]    = fits_data["x"][ci]
            Y_orig[k]    = fits_data["y"][ci]
            x_err[k]     = sx
            y_err[k]     = sy
            xy_cor[k]    = rho
            qfit_arr[k]  = fits_data["qfit"][ci]
            mag_arr[k]   = fits_data["mag"][ci]
            ok_sat[k]    = (fits_data["n_sat"][ci] / window_area) < _MAX_SAT_FRAC

        ok_pos = np.isfinite(X) & np.isfinite(Y)

        # use_for_alignment: True for Gaia sources that pass position/saturation
        # quality cuts AND were NOT flagged as Phase 6 astrometric outliers.
        # use_for_align_init_flag: True even for Phase-6 outliers — they are real
        # Gaia detections, just temporarily inactive.  The BP3M EM loop can
        # re-enable them via test-3 re-admission if their residuals improve.
        use_for_align_full = is_gaia & ok_pos & ok_sat          # without outlier filter
        use_for_align      = use_for_align_full & ~is_outlier   # initial fit flag

        df = pd.DataFrame({
            "Gaia_id":                 gaia_ids,
            "X":                       X,
            "Y":                       Y,
            "X_orig":                  X_orig,
            "Y_orig":                  Y_orig,
            "x_hst_err":               x_err,
            "y_hst_err":               y_err,
            "xy_hst_corr":             xy_cor,
            "q_hst":                   qfit_arr,
            "mag":                     mag_arr,
            "use_for_alignment":       use_for_align,
            "use_for_align_init_flag": use_for_align_full,
            "use_for_fit":             use_for_align,
            "det_chi2":                det_chi2_arr,
        })

        df = df[ok_pos].reset_index(drop=True)
        if len(df) > 0:
            stars_per_image[sub_name] = df

    if skipped_fits:
        print(f"  Warning: FITS catalog missing for {len(skipped_fits)} sub-images "
              f"(skipped): {skipped_fits[:5]}{'...' if len(skipped_fits)>5 else ''}")

    # ── Merge _hi/_lo pairs with too few Gaia alignment stars ────────────────
    # A chip split is only kept when BOTH halves have ≥ _MIN_GAIA_PER_CHIP
    # Gaia-matched (use_for_alignment=True) stars.  Chips with fewer stars are
    # combined into a single unsplit image so the transformation is not under-
    # constrained.  HST-only stars do NOT count toward this threshold.
    _MIN_GAIA_PER_CHIP = 20
    bases_with_split = set()
    for sname in list(stars_per_image.keys()):
        if sname.endswith("_hi") or sname.endswith("_lo"):
            bases_with_split.add(sname[:-3])

    n_merged = 0
    for base in sorted(bases_with_split):
        hi_name = base + "_hi"
        lo_name = base + "_lo"
        df_hi = stars_per_image.get(hi_name)
        df_lo = stars_per_image.get(lo_name)

        # Count Gaia alignment stars in each half
        n_gaia_hi = int(df_hi["use_for_alignment"].sum()) if df_hi is not None else 0
        n_gaia_lo = int(df_lo["use_for_alignment"].sum()) if df_lo is not None else 0

        if n_gaia_hi < _MIN_GAIA_PER_CHIP or n_gaia_lo < _MIN_GAIA_PER_CHIP:
            # Merge: combine both halves under the base image name
            parts = [d for d in [df_hi, df_lo] if d is not None]
            if not parts:
                continue
            merged_df = pd.concat(parts, ignore_index=True)
            # Remove duplicate Gaia_ids (keep first; both halves share the same
            # FITS catalog so there should be no genuine duplicates here)
            merged_df = merged_df.drop_duplicates(subset="Gaia_id", keep="first")
            stars_per_image[base] = merged_df
            # Use the base meta (same dict for both halves)
            if base not in images and hi_name in images:
                images[base] = images[hi_name]
            for sname in [hi_name, lo_name]:
                stars_per_image.pop(sname, None)
                images.pop(sname, None)
            n_merged += 1

    if n_merged > 0:
        print(f"  Merged {n_merged} _hi/_lo pair(s) with <{_MIN_GAIA_PER_CHIP} "
              f"Gaia alignment stars per chip into unsplit images.")

    # ── Build gaia_catalog ────────────────────────────────────────────────────
    # Determine which Gaia IDs are actually used (some may have been lost due to
    # missing FITS catalogs or metadata)
    used_gaia_ids: set[np.int64] = set()
    for df in stars_per_image.values():
        used_gaia_ids.update(df["Gaia_id"].values)

    gaia_used_ids_real = used_gaia_ids & gaia_id_set
    gaia_real_subset = (
        gaia_real[gaia_real["Gaia_id"].isin(gaia_used_ids_real)]
        .copy()
        .reset_index(drop=True)
    )

    # Collect surviving HST-only records (Gaia_id < 0 and present in some image)
    hst_only_survivors = [
        rec for rec in valid_recs
        if not rec["has_gaia"] and rec["Gaia_id"] in used_gaia_ids
    ]

    n_gaia_final = len(gaia_real_subset)
    n_hst_final  = len(hst_only_survivors)

    if n_hst_final > 0:
        hst_rows = pd.DataFrame({
            "Gaia_id":      [r["Gaia_id"] for r in hst_only_survivors],
            "ra":           [r["ra"]      for r in hst_only_survivors],
            "dec":          [r["dec"]     for r in hst_only_survivors],
            "Gaia_time":    [2016.0] * n_hst_final,
            "gmag":         [r["gmag"]    for r in hst_only_survivors],
            # NaN PM/parallax → solver classifies as gaia_2p (flat PM prior)
            "pmra":         [np.nan] * n_hst_final,
            "pmdec":        [np.nan] * n_hst_final,
            "parallax":     [np.nan] * n_hst_final,
            # Large position errors → flat position prior
            "ra_error":     [1e6] * n_hst_final,
            "dec_error":    [1e6] * n_hst_final,
            # NaN ruwe → treated as trustworthy in _cache_gaia
            "ruwe":         [np.nan] * n_hst_final,
            # NaN pseudocolour → not 6p
            "pseudocolour": [np.nan] * n_hst_final,
            # Crossmatch PM seed — used to init v_survey when these sources are enabled
            "pmra_xmatch":  [r.get("pmra_xmatch",  np.nan) for r in hst_only_survivors],
            "pmdec_xmatch": [r.get("pmdec_xmatch", np.nan) for r in hst_only_survivors],
        })
        gaia_catalog = pd.concat([gaia_real_subset, hst_rows], ignore_index=True)
    else:
        gaia_catalog = gaia_real_subset.copy()

    # Fill in any missing columns that the solver expects
    for col in _EXTRA_COLS:
        if col not in gaia_catalog.columns:
            gaia_catalog[col] = np.nan

    if "Gaia_time" not in gaia_catalog.columns:
        gaia_catalog["Gaia_time"] = 2016.0

    # hst_only_mask aligned to gaia_catalog index
    hst_only_mask = np.zeros(len(gaia_catalog), dtype=bool)
    hst_only_mask[n_gaia_final:] = True

    # ── Final summary ─────────────────────────────────────────────────────────
    n_images  = len(stars_per_image)
    n_dets    = sum(len(df) for df in stars_per_image.values())
    n_gaia_dets = sum(
        int(df["use_for_alignment"].sum()) for df in stars_per_image.values()
    )
    print(f"\n  v2 data loaded:")
    print(f"    images:           {n_images}")
    print(f"    gaia_catalog:     {n_gaia_final} Gaia + {n_hst_final} HST-only "
          f"= {len(gaia_catalog)} total")
    print(f"    total detections: {n_dets}  "
          f"({n_gaia_dets} Gaia alignment + {n_dets-n_gaia_dets} HST-only)")

    return images, stars_per_image, gaia_catalog, hst_only_mask
