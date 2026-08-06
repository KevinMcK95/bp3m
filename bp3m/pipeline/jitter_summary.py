"""
Summarise HST jitter per FLC exposure and write jitter_summary.json
into each FLC's directory.

Two sources are combined:
  JIF (jitter summary image):  pre-computed exposure-level stats in the
    extension header — guide-star-frame V2/V3 RMS and peak-to-peak (mas),
    lock status, loss-of-lock count, mean pointing.
  JIT (jitter time series):    3 Hz SI-frame pointing corrections
    (SI_V2_RMS, SI_V3_RMS, etc. in arcsec) — we compute median/max over
    all samples that fall within the exposure.

Matching: FLC rootname[:8] == JIT/JIF EXPNAME[:8].
The JIF/JIT files live in mastDownload/HST/<asn_id_lower>/.
ASN_ID is read from the FLC primary header keyword ASN_ID.

Usage (standalone):
    python -m bp3m.pipeline.jitter_summary <hst_dir> [--force]

  hst_dir : path that contains mastDownload/HST/
  --force  : overwrite existing jitter_summary.json files
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_jit_jif(hst_mast_root: Path, asn_id: str):
    """Return (jit_path, jif_path) for an association, or (None, None)."""
    asn_dir = hst_mast_root / asn_id.lower()
    jit = asn_dir / f"{asn_id.lower()}_jit.fits"
    jif = asn_dir / f"{asn_id.lower()}_jif.fits"
    return (jit if jit.exists() else None,
            jif if jif.exists() else None)


def _match_ext(hdul, rootname: str):
    """Return the extension index whose EXPNAME[:8] matches rootname[:8]."""
    prefix = rootname[:8].lower()
    for i in range(1, len(hdul)):
        expname = hdul[i].header.get("EXPNAME", "")
        if expname[:8].lower() == prefix:
            return i
    return None


def _jif_stats(jif_path: Path, rootname: str) -> dict:
    """Extract per-exposure summary stats from the JIF header."""
    out = {}
    with fits.open(jif_path) as hdul:
        idx = _match_ext(hdul, rootname)
        if idx is None:
            return out
        h = hdul[idx].header
        for key in ("V2_RMS", "V3_RMS", "V2_P2P", "V3_P2P",
                    "RA_AVG", "DEC_AVG", "ROLL_AVG",
                    "GUIDEACT", "GSACQ", "ACTGSSEP", "GSSEPRMS",
                    "NLOSSES", "LOCKLOSS", "NRECENT", "RECENTR"):
            val = h.get(key)
            if val is not None:
                out[key] = val
    # combined RMS in the guide-star frame (mas)
    if "V2_RMS" in out and "V3_RMS" in out:
        out["V23_RMS_mas"] = float(np.hypot(out["V2_RMS"], out["V3_RMS"]))
    return out


def _jit_stats(jit_path: Path, rootname: str) -> dict:
    """Compute per-exposure SI-frame jitter stats from the JIT time series."""
    out = {}
    with fits.open(jit_path) as hdul:
        idx = _match_ext(hdul, rootname)
        if idx is None:
            return out
        d = hdul[idx].data
        n = len(d)
        out["n_samples"] = int(n)
        if n == 0:
            return out

        # Per-sample short-window RMS/P2P (high-frequency jitter only).
        for col in ("SI_V2_RMS", "SI_V2_P2P", "SI_V3_RMS", "SI_V3_P2P"):
            if col in d.names:
                arr = d[col].astype(float)
                out[f"{col}_median_arcsec"] = float(np.median(arr))
                out[f"{col}_max_arcsec"]    = float(np.max(arr))

        # Total SI RMS = quadrature sum of slow drift (std of SI_*_AVG over the
        # full exposure) and high-frequency jitter (median of SI_*_RMS per sample).
        # This is the quantity directly relevant to per-star positional smearing.
        for axis, avg_col, rms_col in (
            ("V2", "SI_V2_AVG", "SI_V2_RMS"),
            ("V3", "SI_V3_AVG", "SI_V3_RMS"),
        ):
            if avg_col in d.names and rms_col in d.names:
                drift_rms  = float(d[avg_col].astype(float).std())
                hf_rms_med = float(np.median(d[rms_col].astype(float)))
                out[f"SI_{axis}_total_RMS_arcsec"] = float(
                    np.hypot(drift_rms, hf_rms_med))

        # Combined (V2+V3) total SI RMS.
        if "SI_V2_total_RMS_arcsec" in out and "SI_V3_total_RMS_arcsec" in out:
            out["SI_combined_total_RMS_arcsec"] = float(
                np.hypot(out["SI_V2_total_RMS_arcsec"],
                         out["SI_V3_total_RMS_arcsec"]))

        # roll jitter (deg → mas): RMS and P2P of deviation from mean roll
        if "Roll" in d.names:
            roll = d["Roll"].astype(float)
            roll_drift_mas = (roll - roll.mean()) * 3_600_000.0
            out["Roll_RMS_mas"] = float(roll_drift_mas.std())
            out["Roll_P2P_mas"] = float(roll_drift_mas.max() - roll_drift_mas.min())

        # quality: fraction of samples with any flag set
        flag_cols = ("FGS_flags", "Recenter", "SlewFlag")
        flagged = np.zeros(n, dtype=bool)
        for col in flag_cols:
            if col in d.names:
                flagged |= (d[col].astype(int) != 0)
        out["n_flagged_samples"] = int(flagged.sum())

    return out


# ── main entry ────────────────────────────────────────────────────────────────

def summarise_jitter(hst_dir: str | Path, force: bool = False) -> list[Path]:
    """
    Walk mastDownload/HST/ under hst_dir, find every *_flc.fits, build a
    jitter_summary.json in the same directory, and return the list of
    written paths.
    """
    hst_dir = Path(hst_dir)
    mast_root = hst_dir / "mastDownload" / "HST"
    if not mast_root.exists():
        raise FileNotFoundError(f"mastDownload/HST not found under {hst_dir}")

    flc_files = sorted(mast_root.rglob("*_flc.fits"))
    written = []
    skipped = 0

    for flc_path in flc_files:
        out_path = flc_path.parent / "jitter_summary.json"
        if out_path.exists() and not force:
            skipped += 1
            continue

        with fits.open(flc_path) as hdul:
            h0 = hdul[0].header
            rootname = h0.get("ROOTNAME", flc_path.stem.replace("_flc", ""))
            asn_id   = h0.get("ASN_ID", "")
            expstart = h0.get("EXPSTART")
            exptime  = h0.get("EXPTIME")
            fgslock  = h0.get("FGSLOCK", "")
            instrume = h0.get("INSTRUME", "")

        summary = {
            "rootname": rootname,
            "asn_id":   asn_id,
            "instrume": instrume,
            "fgslock":  fgslock,
            "expstart_mjd": float(expstart) if expstart is not None else None,
            "exptime_s":    float(exptime)  if exptime  is not None else None,
        }

        if asn_id:
            jit_path, jif_path = _find_jit_jif(mast_root, asn_id)
            if jif_path:
                summary["jif"] = _jif_stats(jif_path, rootname)
            if jit_path:
                summary["jit"] = _jit_stats(jit_path, rootname)

        out_path.write_text(json.dumps(summary, indent=2))
        written.append(out_path)

    print(f"  Wrote {len(written)} jitter_summary.json file(s)"
          + (f" ({skipped} already existed, skipped)" if skipped else ""))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hst_dir", help="Directory containing mastDownload/HST/")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing jitter_summary.json files")
    args = parser.parse_args()
    summarise_jitter(args.hst_dir, force=args.force)


if __name__ == "__main__":
    main()
