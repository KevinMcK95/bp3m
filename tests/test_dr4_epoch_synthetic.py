"""
End-to-end test: synthetic Gaia DR4 epoch astrometry added to Leo I.

Uses existing Leo_I/synthetic_nosplit/ (already has truth + HST-only BP3M_results)
as the baseline.  Creates Leo_I/synthetic_nosplit_dr4/ as a parallel tree with
the same HST/Gaia data but adds synthetic DR4 epoch observations.

Workflow
--------
1. Read truth + Gaia catalog from synthetic_nosplit
2. Generate synthetic DR4 epoch data with truth signal injected
3. Create synthetic_nosplit_dr4 directory (symlinks to same HST/Gaia tree)
4. Run alignment with epoch data on synthetic_nosplit_dr4
5. Compare: print HST-only vs HST+epoch pulls side by side
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

from bp3m.pipeline.run_alignment import run_alignment
from bp3m.pipeline.synthetic import compare_synthetic_results
from bp3m.pipeline.download_gaia_epoch import (
    generate_synthetic_epoch_data,
    compute_epoch_catalog_solutions,
    prepare_epoch_obs_for_solver,
    DR4_REF_EPOCH_JYEAR,
    _NS_PER_JYEAR,
)

DATA_ROOT  = Path("/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results")
FIELD      = "Leo_I"
SYN_BASE        = "synthetic_nosplit"            # existing HST-only synthetic
SYN_EPOCH       = "synthetic_nosplit_dr4"        # HST + epoch, 2p excluded from align
SYN_HSTCAT      = "synthetic_nosplit_dr4_hstcat" # HST-only + epoch cat, 2p excluded
SYN_EPOCH_INC   = "synthetic_nosplit_dr4_inc"       # HST + epoch, 2p included in align
SYN_HSTCAT_INC  = "synthetic_nosplit_dr4_hstcat_inc"# HST-only + epoch cat, 2p included
FIELD_DIR       = DATA_ROOT / FIELD
BASE_DIR        = FIELD_DIR / SYN_BASE
EPOCH_DIR       = FIELD_DIR / SYN_EPOCH
HSTCAT_DIR      = FIELD_DIR / SYN_HSTCAT
EPOCH_INC_DIR   = FIELD_DIR / SYN_EPOCH_INC
HSTCAT_INC_DIR  = FIELD_DIR / SYN_HSTCAT_INC

# ── Step 1: Read existing truth + Gaia catalog ────────────────────────────────
print("=" * 60)
print(f"Reading truth from {SYN_BASE}")
print("=" * 60)

truth_df = pd.read_csv(BASE_DIR / "truth" / "stellar_truth.csv")
truth_df["gaia_source_id"] = truth_df["gaia_source_id"].astype(np.int64)
print(f"  {len(truth_df)} stars in truth table")

gaia_df = pd.read_csv(BASE_DIR / "Gaia" / "Leo_I_synthetic_gaia.csv")
gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
print(f"  {len(gaia_df)} stars in synthetic Gaia catalog")

# ── Step 2: Generate synthetic DR4 epoch data ─────────────────────────────────
print("\n" + "=" * 60)
print("Generating synthetic Gaia DR4 epoch astrometry")
print("=" * 60)

# Sources to give epoch data: all cross-matched stars (both 5p and 2p).
# Gaia DR4 reports epoch detections for all sources; 2p stars simply lack measured
# PM/parallax in the summary catalog but still have individual transit observations.
epoch_sids = set(truth_df["gaia_source_id"].values.tolist())
epoch_sources_df = gaia_df[gaia_df["source_id"].isin(epoch_sids)].copy()
n_5p = epoch_sources_df["pmra"].notna().sum()
n_2p = epoch_sources_df["pmra"].isna().sum()
print(f"  {n_5p} 5p + {n_2p} 2p stars will get epoch data "
      f"(out of {len(epoch_sids)} cross-matched stars)")

epoch_data_raw = generate_synthetic_epoch_data(
    source_df=epoch_sources_df,
    n_transits_per_source=80,
    n_ccd_per_transit=9,
    ref_epoch_jyear=DR4_REF_EPOCH_JYEAR,
    seed=12345,
    # No excess noise: σ_AL from gmag only.  Catalog values will be derived
    # from the epoch solve below to guarantee exact consistency.
)

# Build truth lookup: source_id → [Δα*, Δδ, μα*_true, μδ_true, ϖ_true]
# Use vectorized access (not iterrows) to avoid float64 coercion of large int64 IDs.
_truth_cols = ["true_delta_racosdec", "true_delta_dec",
               "true_pmra", "true_pmdec", "true_parallax"]
_truth_arr = truth_df[_truth_cols].fillna(0.0).values          # (N, 5) float64
_truth_sids = truth_df["gaia_source_id"].values                # (N,) int64 — no float roundtrip
truth_lookup = {int(sid): _truth_arr[i] for i, sid in enumerate(_truth_sids)}

# Inject truth signal: centroid_pos_al += a_k · dv
# dv[0:2] = Δα*, Δδ (position offsets from catalog reference, mas)
# dv[2:5] = μα*_true, μδ_true, ϖ_true as OFFSETS from catalog AGIS values
#           (i.e., true_abs − v_AGIS_catalog), which are exactly the AGIS residuals.
# No further subtraction of catalog values is needed here.
n_injected = 0
for sid, df in epoch_data_raw.items():
    if sid not in truth_lookup:
        continue
    dv = truth_lookup[sid]             # [Δα*, Δδ, μα*_off, μδ_off, ϖ_off]
    obs_tcb  = df["obs_time_tcb"].values.astype(np.float64)
    t_jyear  = 2010.0 + obs_tcb / _NS_PER_JYEAR
    dt       = t_jyear - DR4_REF_EPOCH_JYEAR
    theta    = np.radians(df["scan_pos_angle"].values.astype(np.float64))
    p_al     = df["parallax_factor_al"].values.astype(np.float64)
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    signal   = (dv[0] * sin_th + dv[1] * cos_th
                + dv[2] * dt * sin_th
                + dv[3] * dt * cos_th
                + dv[4] * p_al)
    df = df.copy()
    df["centroid_pos_al"] += signal
    epoch_data_raw[sid] = df
    n_injected += 1
print(f"  Truth signal injected into {n_injected} sources")

# ── Step 2b: Derive exact Gaia catalog values from epoch data ─────────────
# Solve the 5-parameter AGIS normal equations for each star using the
# truth-injected epoch observations.  The catalog (pmra, pmdec, parallax,
# errors, correlations) is replaced by exactly what the epoch solve gives.
# centroid_pos_al is rebaselined to the epoch-derived AGIS solution so that
# prepare_epoch_obs_for_solver reconstructs the same measurements.  This
# guarantees: epoch H/h ≡ Gaia catalog prior (exact, no approximation).
print("\n  Deriving exact catalog solutions from epoch data ...")
epoch_data_solved, gaia_df = compute_epoch_catalog_solutions(
    epoch_data_raw, gaia_df, ref_epoch_jyear=DR4_REF_EPOCH_JYEAR
)

# Consistency check: epoch-derived pmra_error should now EXACTLY match
# the updated catalog pmra_error for a few 5p stars.
print("\n  Consistency check (should be exactly 1.000):")
_check_sids = epoch_sources_df[epoch_sources_df["pmra"].notna()]["source_id"].values[:5]
for _cs in _check_sids:
    _ep = epoch_data_solved.get(int(_cs))
    if _ep is None:
        continue
    _active = _ep[_ep["used_by_agis_al"]]
    _obs_tcb = _active["obs_time_tcb"].values.astype(np.float64)
    _t_yr = 2010.0 + _obs_tcb / _NS_PER_JYEAR
    _dt = _t_yr - DR4_REF_EPOCH_JYEAR
    _theta = np.radians(_active["scan_pos_angle"].values.astype(np.float64))
    _sigma_al = _active["centroid_pos_error_al"].values
    _sigma_exc = float(_active["agis_source_excess_noise"].iloc[0])
    _w = 1.0 / (_sigma_al**2 + _sigma_exc**2)
    _sin_th = np.sin(_theta)
    _cos_th = np.cos(_theta)
    # Full 5×5 H — check pmra channel (index 2)
    _A = np.column_stack([_sin_th, _cos_th, _dt*_sin_th, _dt*_cos_th,
                          _active["parallax_factor_al"].values.astype(np.float64)])
    _H = (_A * _w[:, None]).T @ _A
    try:
        _C = np.linalg.inv(_H)
        _epoch_pmra_err = float(np.sqrt(max(_C[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        _epoch_pmra_err = np.nan
    _cat_row = gaia_df[gaia_df["source_id"] == _cs]
    _cat_pmra_err = float(_cat_row["pmra_error"].iloc[0]) if len(_cat_row) else np.nan
    _ratio = _epoch_pmra_err / _cat_pmra_err if _cat_pmra_err > 0 else np.nan
    print(f"    sid={_cs}: epoch σ_pmra={_epoch_pmra_err:.5f}, "
          f"catalog={_cat_pmra_err:.5f}, ratio={_ratio:.6f}")

# Preprocess into normal-equation contributions using the updated gaia_df
epoch_obs = prepare_epoch_obs_for_solver(
    epoch_data_solved, gaia_df, min_transits=5
)
print(f"  Preprocessed {len(epoch_obs)} sources with ≥5 CCD observations")
total_ccd = sum(d["n_transits"] for d in epoch_obs.values())
print(f"  Total CCD observations: {total_ccd}")

# ── Step 3: directories are set up inside Step 4 below ───────────────────────

# ── Step 4: Run all four alignment scenarios ──────────────────────────────────
# Scenarios A/B use --exclude_2p_from_alignment (only 5p stars drive the image
# transformation).  Scenarios C/D include 2p stars in the alignment.
# Comparing A vs C and B vs D isolates the effect of the flag.

def _setup_symlinks(target_dir, base_dir, epoch_cat_df=None):
    """Create/refresh HST, Gaia (or write epoch cat), and truth symlinks."""
    target_dir.mkdir(exist_ok=True)
    # Gaia: write epoch catalog if provided, else symlink from base
    gaia_link = target_dir / "Gaia"
    if gaia_link.exists() or gaia_link.is_symlink():
        if gaia_link.is_symlink():
            gaia_link.unlink()
        elif gaia_link.is_dir():
            import shutil; shutil.rmtree(gaia_link)
    if epoch_cat_df is not None:
        gaia_link.mkdir(exist_ok=True)
        epoch_cat_df.to_csv(gaia_link / "Leo_I_synthetic_gaia.csv", index=False)
    else:
        gaia_link.symlink_to(base_dir / "Gaia")
    # HST and truth: always symlink
    for subdir in ("HST", "truth"):
        lnk = target_dir / subdir
        if lnk.exists() or lnk.is_symlink():
            lnk.unlink()
        lnk.symlink_to(base_dir / subdir)

# Scenario A: HST + epoch, 2p excluded from alignment
print("\n" + "=" * 60)
print(f"Setting up {SYN_EPOCH} (epoch, 2p excluded)")
print("=" * 60)
_setup_symlinks(EPOCH_DIR, BASE_DIR)
print(f"Running alignment: {SYN_EPOCH}")
run_alignment(output_dir=FIELD_DIR, field_name=SYN_EPOCH,
              gaia_epoch_obs=epoch_obs, split_ccd=False,
              exclude_2p_from_alignment=True)

# Scenario B: HST-only + epoch-derived catalog, 2p excluded from alignment
print("\n" + "=" * 60)
print(f"Setting up {SYN_HSTCAT} (epoch cat, 2p excluded)")
print("=" * 60)
_setup_symlinks(HSTCAT_DIR, BASE_DIR, epoch_cat_df=gaia_df)
print(f"Running alignment: {SYN_HSTCAT}")
run_alignment(output_dir=FIELD_DIR, field_name=SYN_HSTCAT,
              split_ccd=False, exclude_2p_from_alignment=True)

# Scenario C: HST + epoch, 2p included in alignment
print("\n" + "=" * 60)
print(f"Setting up {SYN_EPOCH_INC} (epoch, 2p included)")
print("=" * 60)
_setup_symlinks(EPOCH_INC_DIR, BASE_DIR)
print(f"Running alignment: {SYN_EPOCH_INC}")
run_alignment(output_dir=FIELD_DIR, field_name=SYN_EPOCH_INC,
              gaia_epoch_obs=epoch_obs, split_ccd=False,
              exclude_2p_from_alignment=False)

# Scenario D: HST-only + epoch-derived catalog, 2p included in alignment
print("\n" + "=" * 60)
print(f"Setting up {SYN_HSTCAT_INC} (epoch cat, 2p included)")
print("=" * 60)
_setup_symlinks(HSTCAT_INC_DIR, BASE_DIR, epoch_cat_df=gaia_df)
print(f"Running alignment: {SYN_HSTCAT_INC}")
run_alignment(output_dir=FIELD_DIR, field_name=SYN_HSTCAT_INC,
              split_ccd=False, exclude_2p_from_alignment=False)


# ── Step 5: Compare pulls ─────────────────────────────────────────────────────
# Build source-id sets for 5p and 2p stars (based on original catalog).
orig_cat = pd.read_csv(BASE_DIR / "Gaia" / "Leo_I_synthetic_gaia.csv")
orig_cat["source_id"] = orig_cat["source_id"].astype(np.int64)
orig_5p_sids = set(orig_cat.loc[orig_cat["pmra"].notna(), "source_id"].values.tolist())
orig_2p_sids = set(orig_cat.loc[orig_cat["pmra"].isna(),  "source_id"].values.tolist())
orig_pmra_map  = dict(zip(orig_cat["source_id"], orig_cat["pmra"].fillna(0.0)))
orig_pmdec_map = dict(zip(orig_cat["source_id"], orig_cat["pmdec"].fillna(0.0)))
orig_plx_map   = dict(zip(orig_cat["source_id"], orig_cat["parallax"].fillna(0.0)))

print(f"\n  Original catalog: {len(orig_5p_sids)} 5p stars, {len(orig_2p_sids)} 2p stars")


def _compute_pulls(res_dir: "Path", label: str,
                   ep_gaia_df: pd.DataFrame,
                   use_epoch_cat_ref: bool = False) -> pd.DataFrame:
    """Compute pull statistics from a BP3M result directory.

    For position: truth is true_delta_racosdec/true_delta_dec relative to the
    Gaia position in stellar_astrometry (which is the epoch-derived position for
    both EPOCH and HSTCAT runs).

    For PM/parallax: truth is the original catalog value + the injected offset.
    """
    _res = pd.read_csv(res_dir / "BP3M_results" / "stellar_astrometry.csv")
    _res["Gaia_id"] = _res["Gaia_id"].astype(np.int64)
    _tdf = truth_df.rename(columns={"gaia_source_id": "Gaia_id"})
    _dup = [c for c in _tdf.columns if c in _res.columns and c != "Gaia_id"]
    _tdf = _tdf.drop(columns=_dup)
    df = _res.merge(_tdf, on="Gaia_id", how="inner")
    gids = df["Gaia_id"].values

    # Residuals for PM/parallax: bp3m_estimate - (orig_catalog + truth_offset)
    df["resid_pmra"]     = (df["pmra_bp3m"]
                            - np.array([orig_pmra_map.get(g, 0.0)  for g in gids])
                            - df["true_pmra"])
    df["resid_pmdec"]    = (df["pmdec_bp3m"]
                            - np.array([orig_pmdec_map.get(g, 0.0) for g in gids])
                            - df["true_pmdec"])
    df["resid_parallax"] = (df["parallax_bp3m"]
                            - np.array([orig_plx_map.get(g, 0.0)   for g in gids])
                            - df["true_parallax"])

    # Position residuals: delta_bp3m is measured FROM the epoch-derived ra/dec,
    # so truth = true_delta - (epoch_ra - orig_ra)*cos(dec)*3.6e6 (and similarly dec).
    _ep = ep_gaia_df.copy()
    if "source_id" not in _ep.columns:
        _ep = _ep.reset_index()
    _ep["source_id"] = _ep["source_id"].astype(np.int64)
    _orig_ra_map  = dict(zip(orig_cat["source_id"], orig_cat["ra"]))
    _orig_dec_map = dict(zip(orig_cat["source_id"], orig_cat["dec"]))
    _ep_ra_map    = dict(zip(_ep["source_id"], _ep["ra"]))
    _ep_dec_map   = dict(zip(_ep["source_id"], _ep["dec"]))
    # ep_ra_corr: shift of epoch-derived catalog relative to original catalog (mas).
    # Only apply when the run uses the epoch-derived catalog as its Gaia reference
    # (HSTCAT).  For the EPOCH run the reference IS the original catalog, so the
    # truth is simply true_delta_racosdec with no correction.
    if use_epoch_cat_ref:
        ep_ra_corr  = np.array([
            (_ep_ra_map.get(g, _orig_ra_map.get(g, 0.0)) - _orig_ra_map.get(g, 0.0))
            * np.cos(np.radians(_orig_dec_map.get(g, 0.0))) * 3.6e6
            for g in gids
        ])
        ep_dec_corr = np.array([
            (_ep_dec_map.get(g, _orig_dec_map.get(g, 0.0)) - _orig_dec_map.get(g, 0.0))
            * 3.6e6
            for g in gids
        ])
    else:
        ep_ra_corr  = np.zeros(len(gids))
        ep_dec_corr = np.zeros(len(gids))
    df["resid_delta_racosdec"] = (df["delta_racosdec_bp3m"]
                                  - (df["true_delta_racosdec"] - ep_ra_corr))
    df["resid_delta_dec"]      = (df["delta_dec_bp3m"]
                                  - (df["true_delta_dec"] - ep_dec_corr))

    sig_map = {"pmra":            "sigma_pmra_bp3m",
               "pmdec":           "sigma_pmdec_bp3m",
               "parallax":        "sigma_parallax_bp3m",
               "delta_racosdec":  "sigma_delta_racosdec",
               "delta_dec":       "sigma_delta_dec"}
    for key, scol in sig_map.items():
        df[f"pull_{key}"] = df[f"resid_{key}"] / df[scol].replace(0, np.nan)

    df["is_5p"] = df["Gaia_id"].isin(orig_5p_sids)
    return df


# ── Step 5: Compute pulls for all four scenarios ──────────────────────────────
scenarios = [
    (EPOCH_DIR,      "ep_excl",   False),  # A: epoch,  2p excluded
    (HSTCAT_DIR,     "hc_excl",   True),   # B: hstcat, 2p excluded
    (EPOCH_INC_DIR,  "ep_incl",   False),  # C: epoch,  2p included
    (HSTCAT_INC_DIR, "hc_incl",   True),   # D: hstcat, 2p included
]
cmps = {}
for res_dir, tag, use_ep_ref in scenarios:
    cmps[tag] = _compute_pulls(res_dir, tag, gaia_df,
                               use_epoch_cat_ref=use_ep_ref)

# ── Side-by-side summary ──────────────────────────────────────────────────────
params = [
    ("delta_racosdec", "Δα*"),
    ("delta_dec",      "Δδ"),
    ("pmra",           "Δμα*"),
    ("pmdec",          "Δμδ"),
    ("parallax",       "Δϖ"),
]

tags    = ["ep_excl", "hc_excl", "ep_incl", "hc_incl"]
headers = ["EPOCH_excl", "HSTCAT_excl", "EPOCH_incl", "HSTCAT_incl"]

for subset_label, mask_fn in [("5p stars", lambda df: df["is_5p"]),
                               ("2p stars", lambda df: ~df["is_5p"])]:
    print("\n" + "=" * 76)
    print(f"Summary ({subset_label})  "
          f"  excl=2p excluded from alignment  incl=2p included")
    print("=" * 76)
    hdr = f"  {'Param':<6}"
    for h in headers:
        hdr += f"  {'μ_'+h[:7]:>9} {'σ_'+h[:7]:>9}"
    print(hdr)
    for key, label in params:
        col = f"pull_{key}"
        row = f"  {label:<6}"
        for tag in tags:
            df_ = cmps[tag]
            vals = df_.loc[mask_fn(df_), col].dropna()
            mu   = vals.mean() if len(vals) else float("nan")
            sig  = vals.std()  if len(vals) else float("nan")
            row += f"  {mu:>9.3f} {sig:>9.3f}"
        print(row)
