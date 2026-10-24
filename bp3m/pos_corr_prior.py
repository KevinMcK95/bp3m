"""Learned per-image affine prior centres for the learned GDC correction (option C prototype, 2026-10-13).

The gauge fix (pos_corr_gauge.GaugeFixedPosCorr) removes each image's affine projection from the learned correction
and, with ``gauge_ref``, puts a reference affine part back.  Option B used another model's projection (v4 C_r2) as the
reference.  Option C replaces it by a model of the per-image linear terms themselves, fitted to measured per-image
affine terms relative to the HEADER frame after the non-linear shape (hst_dist_corr ml/v5oct/optionC/fit_prior.py):

    (scale, rot [rad], on_skew, off_skew) = design(year, filter, orbit features) @ beta     per detector

  year     piecewise-linear spline inside each era span (ACS 2002 | 2009.4 SM4 | 2024.45 one-gyro; UVIS 2009.3 | 2024.45)
  filter   one-hot offsets (filters the fit did not see -> 'other')
  orbit    the 9 scaled orbit features of pos_corr_basis.orbit_row (zero without a jitter track, as in training)

The affine part is returned as [1, u, v] coefficients in the GDC frame (u, v = (x_gdc - 2048) / 2048, as in
pos_corr_gauge), with zero offset (pointing is free in every per-image solve) and zero chip offset.  The per-target
intrinsic scatter ``tau`` is the out-of-sample prior width.

Per target the model file chooses (by held-out residual) one of: 'zero' (header), 'P' (the affine projection of the
``base_ref`` model on this image, e.g. v4 gdc_basis_v2:C_r2), 'TFO' (learned) or 'P+TFO' (base projection + learned
residual).  Each target carries its own design spec.

Spec: ``DIR:TAG#gauge=affine6,gauge_ref=prior:/path/prior_model.json``.
"""
from __future__ import annotations
import json
import numpy as np

TARGETS = ('scale', 'rot', 'on', 'off')


def _year(hdr):
    mjd = hdr.get('EXPSTART')
    return 2000.0 + (float(mjd) - 51544.5) / 365.25 if mjd is not None else np.nan


def _det(hdr):
    return f"{str(hdr.get('INSTRUME', '')).strip()}/{str(hdr.get('DETECTOR', '')).strip()}"


def _filter(hdr):
    f = hdr.get('FILTER')
    if not f:
        f1, f2 = str(hdr.get('FILTER1', '')), str(hdr.get('FILTER2', ''))
        f = f1 if (f1.startswith('F') and 'CLEAR' not in f1) else f2
    return str(f).strip()


def design_row(spec, year, filt, orbit_scaled, orbit_names):
    """one design row; identical column order to fit_prior.design (orbit features already scaled)."""
    cols = []
    sp = spec['spans']
    for s, (lo, hi) in enumerate(zip(sp[:-1], sp[1:])):
        seg = float((lo <= year < hi) if s < len(sp) - 2 else (year >= lo))
        cols += [seg, seg * (year - lo)] + [seg * max(year - k, 0.0) for k in spec['knots'][s]]
    if spec['use_filter']:
        f = filt if filt in spec['filters'] else 'other'
        cols += [float(f == fl) for fl in spec['filters'] + ['other']]
    if spec['use_orbit']:
        z = dict(zip(orbit_names, np.asarray(orbit_scaled, float)))
        cols += [float(z.get(o, 0.0)) for o in orbit_names]
    return np.asarray(cols, float)


def linear_to_coef(scale, rot, on, off):
    """(scale, rot, on, off) -> [1, u, v] coefficients (cx, cy) in GDC px (inverse of GaugeFixedPosCorr.info)."""
    jxx, jyy, jyx, jxy = scale + on, scale - on, rot + off, off - rot
    return np.array([0.0, jxx * 2048.0, jxy * 2048.0]), np.array([0.0, jyx * 2048.0, jyy * 2048.0])


def coef_to_linear(cx, cy):
    """[1, u, v] coefficients -> (scale, rot, on, off), the GaugeFixedPosCorr.info convention."""
    jxx, jxy, jyx, jyy = cx[1] / 2048, cx[2] / 2048, cy[1] / 2048, cy[2] / 2048
    return dict(scale=0.5 * (jxx + jyy), rot=0.5 * (jyx - jxy), on=0.5 * (jxx - jyy), off=0.5 * (jxy + jyx))


class PriorCentreRef:
    """gauge reference from a learned prior-centre model (prior_model.json)."""
    is_prior = True

    def __init__(self, path: str):
        self.path = path
        self.m = json.load(open(path))
        self.orbit_names = self.m.get('orbit_features', [])
        self._cache = {}
        self._base = None                      # GaugeFixedPosCorr over base_ref (lazy), for targets with base 'P'

    def _base_linear(self, hdr):
        ref = self.m.get('base_ref')
        if not ref:
            return None
        if self._base is None:
            from bp3m.pos_corr_basis import make_pos_corr
            from bp3m.pos_corr_gauge import GaugeFixedPosCorr
            self._base = GaugeFixedPosCorr(make_pos_corr(ref), 'affine6')
        pr = self._base._projection(hdr)
        return coef_to_linear(pr['cx'], pr['cy']) if pr is not None else None

    @property
    def summary(self):
        return f"learned prior centres {self.m.get('version', '?')} ({self.path})"

    def predict(self, hdr, orbit_scaled=None):
        """dict(scale, rot, on, off, tau_*) for one exposure, or None (detector not modelled / no time)."""
        det = _det(hdr); md = self.m['detectors'].get(det); yr = _year(hdr)
        if md is None or not np.isfinite(yr):
            return None
        z = np.zeros(len(self.orbit_names)) if orbit_scaled is None else orbit_scaled
        out, base = {}, None
        for t in TARGETS:
            mt = md['targets'][t]; spec = mt.get('spec') or md.get('spec')
            v = float(design_row(spec, yr, _filter(hdr), z, self.orbit_names) @ np.asarray(mt['beta'], float))
            if mt.get('base') == 'P':
                if base is None:
                    base = self._base_linear(hdr)
                if base is None:               # base model does not cover this image: no reference for it
                    return None
                v += base[t]
            out[t] = v
            out['tau_' + t] = float(mt['tau'])
        return out

    def coef(self, hdr, base=None, mode='affine6'):
        """[1, u, v] (+ chip column for 8p) coefficients of the predicted affine part, or None."""
        key = hdr.get('_flc_path')
        if key in self._cache:
            return self._cache[key]
        z = None
        if base is not None and hasattr(base, 'orbit_row'):
            try:
                zr, ok = base.orbit_row(hdr)
                z = np.asarray(zr, float) if ok else None
            except Exception:
                z = None
        p = self.predict(hdr, z)
        if p is None:
            out = None
        else:
            cx, cy = linear_to_coef(p['scale'], p['rot'], p['on'], p['off'])
            if mode == 'affine8':
                cx, cy = np.r_[cx, 0.0], np.r_[cy, 0.0]
            out = (cx, cy, p)
        if key is not None:
            self._cache[key] = out
        return out
