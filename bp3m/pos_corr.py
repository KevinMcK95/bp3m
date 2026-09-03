"""Pseudo-GDC centroid corrections (PSF-model-induced position bias).

Loads a correction table produced by stdpsf_builder/make_pseudo_gdc.py and
applies it IN MEMORY to catalog positions at load time — nothing on disk is
modified.  The table holds per-cell (7x7 per chip), per-flux-bin,
per-epoch centroid biases of Anderson-PSF fits plus a sub-pixel-phase
term; the correction is position - bias.

Biases are measured in RAW detector pixels; they are applied directly to
the GDC-frame positions (the GDC Jacobian differs from identity by a few
percent — negligible at the ~0.2 mas bias amplitudes).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class PseudoGDC:
    def __init__(self, path):
        self.path = Path(path)
        z = np.load(self.path, allow_pickle=False)
        self.cell_x = z["cell_x"].astype(float)
        self.cell_y = z["cell_y"].astype(float)
        self.sci_exts = [int(v) for v in z["sci_exts"]]
        self.bias_x = np.nan_to_num(z["bias_x"], nan=0.0)
        self.bias_y = np.nan_to_num(z["bias_y"], nan=0.0)
        self.flux_edges = z["flux_edges"].astype(float)
        self.mjd_sm4 = float(z["mjd_sm4"])
        self.phase_dx = z["phase_dx"].astype(float)
        self.phase_dy = z["phase_dy"].astype(float)
        self.boundary = float(z["chip_boundary"]) if "chip_boundary" in z \
            else 2048.0
        self.instrument = str(z["instrument"])
        self.detector = str(z["detector"])
        self.filter = str(z["filter"])
        # optional epoch-validity window (e.g. GDC-delta tables built from
        # per-group calibration D fits): applies only to images inside it
        self.mjd_window = (tuple(float(v) for v in z["mjd_window"])
                           if "mjd_window" in z else None)
        self.md5 = hashlib.md5(self.path.read_bytes()).hexdigest()

    def matches(self, instrument: str, detector: str, filt: str,
                mjd: float | None = None) -> bool:
        if not (str(instrument).upper() == self.instrument.upper()
                and str(detector).upper() == self.detector.upper()
                and str(filt).upper() == self.filter.upper()):
            return False
        if self.mjd_window is not None and mjd is not None:
            return self.mjd_window[0] <= mjd <= self.mjd_window[1]
        return True

    def _interp_cell(self, grid2d, xc, yc):
        """Bilinear interpolation of a (7y,7x) cell map, edge-clamped."""
        gx = np.interp(xc, self.cell_x, np.arange(self.cell_x.size))
        gy = np.interp(yc, self.cell_y, np.arange(self.cell_y.size))
        x0 = np.clip(np.floor(gx).astype(int), 0, self.cell_x.size - 2)
        y0 = np.clip(np.floor(gy).astype(int), 0, self.cell_y.size - 2)
        fx = np.clip(gx - x0, 0.0, 1.0)
        fy = np.clip(gy - y0, 0.0, 1.0)
        return ((1 - fx) * (1 - fy) * grid2d[y0, x0]
                + fx * (1 - fy) * grid2d[y0, x0 + 1]
                + (1 - fx) * fy * grid2d[y0 + 1, x0]
                + fx * fy * grid2d[y0 + 1, x0 + 1])

    def bias(self, x_raw, y_raw, flux, mjd):
        """Per-detection (bias_x, bias_y) in detector px (raw-frame arrays)."""
        x_raw = np.asarray(x_raw, float)
        y_raw = np.asarray(y_raw, float)
        flux = np.asarray(flux, float)
        n = x_raw.size
        bx = np.zeros(n)
        by = np.zeros(n)
        ei = 0 if mjd < self.mjd_sm4 else 1
        fb = np.digitize(flux, self.flux_edges)
        # chip from mosaic y (ext1 below the boundary, ext4 above)
        ci = (y_raw >= self.boundary).astype(int)
        y_chip = np.where(ci == 1, y_raw - self.boundary, y_raw)
        for c in (0, 1):
            for b in range(self.bias_x.shape[1]):
                m = (ci == c) & (fb == b)
                if not m.any():
                    continue
                bx[m] = self._interp_cell(self.bias_x[ei, b, :, :, c],
                                          x_raw[m], y_chip[m])
                by[m] = self._interp_cell(self.bias_y[ei, b, :, :, c],
                                          x_raw[m], y_chip[m])
        # sub-pixel phase term
        nb = self.phase_dx.shape[0]
        ipx = np.minimum((np.mod(x_raw, 1.0) * nb).astype(int), nb - 1)
        ipy = np.minimum((np.mod(y_raw, 1.0) * nb).astype(int), nb - 1)
        bx = bx + self.phase_dx[ipy, ipx]
        by = by + self.phase_dy[ipy, ipx]
        return bx, by


class PseudoGDCSet:
    """A collection of PseudoGDC tables (one per inst/det/filter).

    Constructed from a comma-separated list of npz paths; per image the
    first matching table is used, others leave the image uncorrected.
    """

    def __init__(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [p for p in str(paths).split(",") if p.strip()]
        self.tables = [PseudoGDC(p.strip()) for p in paths]

    def match(self, instrument: str, detector: str, filt: str,
              mjd: float | None = None):
        """ALL matching tables (corrections are additive: e.g. pseudo-GDC
        PSF-bias + GDC-delta epoch-distortion for the same image)."""
        out = [t for t in self.tables
               if t.matches(instrument, detector, filt, mjd)]
        return out or None

    @property
    def summary(self):
        return ", ".join(f"{t.instrument}/{t.detector}/{t.filter}"
                         f"({t.md5[:6]})" for t in self.tables)
