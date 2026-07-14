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
    prepare_epoch_obs_for_solver,
    DR4_REF_EPOCH_JYEAR,
    _NS_PER_JYEAR,
)

DATA_ROOT  = Path("/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results")
FIELD      = "Leo_I"
SYN_BASE   = "synthetic_nosplit"        # existing HST-only synthetic
SYN_EPOCH  = "synthetic_nosplit_dr4"    # new HST + epoch run
FIELD_DIR  = DATA_ROOT / FIELD
BASE_DIR   = FIELD_DIR / SYN_BASE
EPOCH_DIR  = FIELD_DIR / SYN_EPOCH

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

# Sources to give epoch data: all 5p stars in the cross-match (have truth entries)
epoch_sids = set(truth_df["gaia_source_id"].values.tolist())
epoch_sources_df = gaia_df[gaia_df["source_id"].isin(epoch_sids)].copy()
# Only give epoch data to 5p stars (those with measured pmra in the Gaia catalog)
epoch_sources_5p = epoch_sources_df[epoch_sources_df["pmra"].notna()].copy()
print(f"  {len(epoch_sources_5p)} 5p stars will get epoch data "
      f"(out of {len(epoch_sids)} cross-matched stars)")

epoch_data_raw = generate_synthetic_epoch_data(
    source_df=epoch_sources_5p,
    n_transits_per_source=80,
    n_ccd_per_transit=9,
    ref_epoch_jyear=DR4_REF_EPOCH_JYEAR,
    seed=12345,
)

# Build truth lookup: source_id → [Δα*, Δδ, Δμα*, Δμδ, Δϖ]
# Use vectorized access (not iterrows) to avoid float64 coercion of large int64 IDs.
_truth_cols = ["true_delta_racosdec", "true_delta_dec",
               "true_pmra", "true_pmdec", "true_parallax"]
_truth_arr = truth_df[_truth_cols].fillna(0.0).values          # (N, 5) float64
_truth_sids = truth_df["gaia_source_id"].values                # (N,) int64 — no float roundtrip
truth_lookup = {int(sid): _truth_arr[i] for i, sid in enumerate(_truth_sids)}

# Inject truth signal: centroid_pos_al += a_k · v_true
# centroid_pos_al = a_k · (v_true - v_AGIS) + noise = a_k · delta_v + noise
n_injected = 0
for sid, df in epoch_data_raw.items():
    if sid not in truth_lookup:
        continue
    dv = truth_lookup[sid]
    obs_tcb  = df["obs_time_tcb"].values.astype(np.float64)
    t_jyear  = 2010.0 + obs_tcb / _NS_PER_JYEAR
    dt       = t_jyear - DR4_REF_EPOCH_JYEAR
    theta    = np.radians(df["scan_pos_angle"].values.astype(np.float64))
    p_al     = df["parallax_factor_al"].values.astype(np.float64)
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    signal   = (dv[0] * sin_th + dv[1] * cos_th
                + dv[2] * dt * sin_th + dv[3] * dt * cos_th
                + dv[4] * p_al)
    df = df.copy()
    df["centroid_pos_al"] += signal
    epoch_data_raw[sid] = df
    n_injected += 1
print(f"  Truth signal injected into {n_injected} sources")

# Preprocess into normal-equation contributions
epoch_obs = prepare_epoch_obs_for_solver(
    epoch_data_raw, gaia_df, min_transits=5
)
print(f"  Preprocessed {len(epoch_obs)} sources with ≥5 CCD observations")
total_ccd = sum(d["n_transits"] for d in epoch_obs.values())
print(f"  Total CCD observations: {total_ccd}")

# ── Step 3: Create synthetic_nosplit_dr4 directory ────────────────────────────
print("\n" + "=" * 60)
print(f"Setting up {SYN_EPOCH} directory")
print("=" * 60)

EPOCH_DIR.mkdir(exist_ok=True)

# Symlink HST and Gaia from the base synthetic
for subdir in ("HST", "Gaia"):
    link = EPOCH_DIR / subdir
    target = BASE_DIR / subdir
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    print(f"  Linked {link} -> {target}")

# Symlink truth so compare_synthetic_results can find it
truth_link = EPOCH_DIR / "truth"
truth_target = BASE_DIR / "truth"
if truth_link.exists() or truth_link.is_symlink():
    truth_link.unlink()
truth_link.symlink_to(truth_target)
print(f"  Linked {truth_link} -> {truth_target}")

# ── Step 4: Run alignment with epoch data ─────────────────────────────────────
print("\n" + "=" * 60)
print(f"Running alignment: HST + DR4 epoch ({SYN_EPOCH})")
print("=" * 60)

run_alignment(
    output_dir=FIELD_DIR,
    field_name=SYN_EPOCH,
    gaia_epoch_obs=epoch_obs,
)

# ── Step 5: Compare pulls ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"HST-only pulls ({SYN_BASE})")
print("=" * 60)
cmp_hst = compare_synthetic_results(
    output_dir=DATA_ROOT, field_name=FIELD, syn_name=SYN_BASE
)

print("\n" + "=" * 60)
print(f"HST + DR4 epoch pulls ({SYN_EPOCH})")
print("=" * 60)
cmp_ep = compare_synthetic_results(
    output_dir=DATA_ROOT, field_name=FIELD, syn_name=SYN_EPOCH
)

# ── Side-by-side improvement summary ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary: pull width (σ) — lower is better after adding epoch data")
print("=" * 60)
params = [
    ("delta_racosdec", "Δα*"),
    ("delta_dec",      "Δδ"),
    ("pmra",           "Δμα*"),
    ("pmdec",          "Δμδ"),
    ("parallax",       "Δϖ"),
]
print(f"  {'Param':<8} {'pull_μ_HST':>11} {'pull_σ_HST':>11} "
      f"{'pull_μ_ep':>11} {'pull_σ_ep':>11} {'σ_ratio':>9}")
for key, label in params:
    col = f"pull_{key}"
    if col not in cmp_hst.columns or col not in cmp_ep.columns:
        continue
    ph = cmp_hst[col].dropna()
    pe = cmp_ep[col].dropna()
    ratio = pe.std() / ph.std() if ph.std() > 0 else np.nan
    print(f"  {label:<8} {ph.mean():>11.3f} {ph.std():>11.3f} "
          f"{pe.mean():>11.3f} {pe.std():>11.3f} {ratio:>9.3f}")
