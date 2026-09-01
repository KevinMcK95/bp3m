"""
Load and apply the shared epoch-distortion correction D fitted by
BP3MSolver(fit_epoch_distortion=True).

The fitted model predicts pseudo-image positions as  X r + B d  (see
solver Eq. 8 and astro_utils.epoch_distortion_basis).  Downstream tools that
model positions as  X r  only (master crossmatch, pop fit, v2) stay exactly
consistent — to linear order in the small displacement — by correcting the
DETECTOR coordinates instead:

    X(x') r = X(x) r + R (x' - x)   with  R = [[a, b], [c, d]]
    =>  x' = x + R^{-1} (B d)

`detector_correction` returns that (dx, dy).  Apply it once at load time and
every downstream formula is automatically D-aware.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bp3m.astro_utils import epoch_distortion_basis


class EpochDistortion:
    """Per-field epoch-distortion correction loaded from BP3M_results/."""

    def __init__(self, groups: dict, group_of: dict):
        self._groups = groups      # g -> dict(order, half_x, half_y, coeff (K,))
        self._group_of = group_of  # image_name -> g (or -1)

    # ── loading ───────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, bp3m_results_dir) -> "EpochDistortion | None":
        """Return an EpochDistortion for the field, or None when the fit did
        not use --fit_epoch_distortion (or the outputs predate it)."""
        d = Path(bp3m_results_dir)
        cfg_path = d / "run_config.json"
        ed_path = d / "epoch_distortion.csv"
        it_path = d / "image_transformations.csv"
        if not (cfg_path.exists() and ed_path.exists() and it_path.exists()):
            return None
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            return None
        if not cfg.get("fit_epoch_distortion"):
            return None

        ed = pd.read_csv(ed_path)
        groups: dict = {}
        for g, sub in ed.groupby("group"):
            sub = sub.reset_index(drop=True)   # rows already in coefficient order
            groups[int(g)] = dict(
                order=int(sub["order"].iloc[0]) if "order" in sub else 3,
                half_x=float(sub["half_x"].iloc[0]) if "half_x" in sub else 2048.0,
                half_y=float(sub["half_y"].iloc[0]) if "half_y" in sub else 1024.0,
                coeff=sub["coeff_px"].to_numpy(float),
            )

        it = pd.read_csv(it_path)
        if "ed_group" not in it.columns:
            print("  WARNING: epoch_distortion.csv present but "
                  "image_transformations.csv lacks ed_group — refit with the "
                  "current bp3m to enable downstream D application.")
            return None
        group_of = {str(r.image_name): int(r.ed_group) for r in it.itertuples()}
        return cls(groups, group_of)

    # ── queries ───────────────────────────────────────────────────────────────
    def has(self, image_name: str) -> bool:
        return self._group_of.get(image_name, -1) >= 0

    def n_groups(self) -> int:
        return len(self._groups)

    def pseudo_disp(self, image_name: str, X_c, Y_c) -> np.ndarray:
        """(n, 2) displacement B @ d in pseudo-image pixels (0 if no group)."""
        g = self._group_of.get(image_name, -1)
        if g < 0 or g not in self._groups:
            return np.zeros((len(np.atleast_1d(X_c)), 2))
        grp = self._groups[g]
        B = epoch_distortion_basis(np.atleast_1d(X_c), np.atleast_1d(Y_c),
                                   grp["order"], half_x=grp["half_x"],
                                   half_y=grp["half_y"])
        return np.einsum("nkl,l->nk", B, grp["coeff"])

    def detector_correction(self, image_name: str, X_c, Y_c, abcd) -> np.ndarray:
        """(n, 2) correction to ADD to detector coordinates so that a D-less
        linear model reproduces the D-full fit:  x' = x + R^{-1} (B d)."""
        disp = self.pseudo_disp(image_name, X_c, Y_c)
        if not disp.any():
            return disp
        a, b, c, d = [float(v) for v in abcd]
        det = a * d - b * c
        if abs(det) < 1e-12:
            return np.zeros_like(disp)
        Rinv = np.array([[d, -b], [-c, a]]) / det
        return disp @ Rinv.T
