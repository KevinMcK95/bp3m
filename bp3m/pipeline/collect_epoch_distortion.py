"""
bp3m-collect-distortion: gather per-field epoch-distortion measurements into
one archive table.

Scans <root>/*/<results_name>/epoch_distortion.csv (and the companion
epoch_distortion_linear.csv) across every field under a GaiaHub_results-style
root, tags each row with its field and run settings, and writes combined
tables for the archive-wide distortion program:

    <out>/epoch_distortion_all.csv         — all D coefficients (deg >= 2)
    <out>/epoch_distortion_linear_all.csv  — per-group linear deviations

Run from (or point --root at) the GaiaHub_results directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Collect epoch_distortion.csv tables across fields")
    ap.add_argument("--root", default=".",
                    help="GaiaHub_results-style root (default: cwd)")
    ap.add_argument("--results_name", default="BP3M_results",
                    help="results subdirectory name to read in each field "
                         "(default: BP3M_results)")
    ap.add_argument("--out", default=None,
                    help="output directory "
                         "(default: <root>/epoch_distortion_archive)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.out) if args.out else root / "epoch_distortion_archive"

    coeff_parts, lin_parts = [], []
    n_fields = 0
    for ed_path in sorted(root.glob(f"*/{args.results_name}/epoch_distortion.csv")):
        rdir = ed_path.parent
        field = rdir.parent.name
        cfg = {}
        cfg_path = rdir / "run_config.json"
        if cfg_path.exists():
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                pass
        tag = dict(field=field,
                   epoch_dist_groupby=cfg.get("epoch_dist_groupby", "full"),
                   epoch_dist_sigma=cfg.get("epoch_dist_sigma_mas",
                                            cfg.get("epoch_dist_sigma", np.nan)),
                   results_dir=str(rdir))
        df = pd.read_csv(ed_path)
        for k, v in reversed(tag.items()):
            df.insert(0, k, v)
        coeff_parts.append(df)
        lin_path = rdir / "epoch_distortion_linear.csv"
        if lin_path.exists():
            dl = pd.read_csv(lin_path)
            for k, v in reversed(tag.items()):
                dl.insert(0, k, v)
            lin_parts.append(dl)
        n_fields += 1

    if not coeff_parts:
        print(f"No {args.results_name}/epoch_distortion.csv found under {root}")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    allc = pd.concat(coeff_parts, ignore_index=True)
    allc.to_csv(out / "epoch_distortion_all.csv", index=False)
    print(f"Collected {n_fields} fields, "
          f"{allc.groupby(['field', 'group']).ngroups} chip-groups, "
          f"{len(allc)} coefficients -> {out / 'epoch_distortion_all.csv'}")

    if lin_parts:
        alll = pd.concat(lin_parts, ignore_index=True)
        alll.to_csv(out / "epoch_distortion_linear_all.csv", index=False)
        print(f"Collected {len(alll)} linear-deviation rows "
              f"-> {out / 'epoch_distortion_linear_all.csv'}")

    # Quick census: max |D| amplitude per instrument/detector/filter
    allc["amp_mas"] = allc.coeff_px.abs() * allc.pscale_mas
    cen = (allc.groupby(["instrument", "detector", "filter"])
               .agg(n_groups=("group", "nunique"),
                    n_fields=("field", "nunique"),
                    mjd_min=("mean_mjd", "min"), mjd_max=("mean_mjd", "max"),
                    max_amp_mas=("amp_mas", "max")))
    print("\nArchive census (per-coefficient amplitude, mas):")
    print(cen.to_string(float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
