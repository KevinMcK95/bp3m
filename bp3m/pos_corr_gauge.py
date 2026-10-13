"""Gauge fix for the learned GDC correction (option B, user 2026-10-13).

Every image in BP3M has its own affine plate solution (a, b, c, d + offsets: 6p, plus an upper-chip translation with
--fit_chip_offset: 8p) acting on the GDC frame (x_gdc, y_gdc).  Any part of the learned correction f that is affine in
that frame is therefore interchangeable with the image's own plate terms: training never pins it, and in the v5oct
models the UVIS per-image linear part is large and unstable (MAD ~0.05 px, p95 ~0.85 px between the full and
no-dense-field models).  Applied with the tight HST per-image priors (skew 1e-5, centred on zero), that linear part
has to be cancelled by the plate terms against the prior, which biased the dwarf PMs (N6 vs N6L, 2026-10-11).

GaugeFixedPosCorr wraps any applier from make_pos_corr and returns  f - A_j,  where A_j is the least-squares affine
(6p: 1, x_gdc, y_gdc per axis; 8p: + upper-chip indicator) projection of f for image j.  The projection is evaluated
on a regular grid over the image's own catalogue footprint (per chip, 1st-99th percentile of the catalogue's raw x/y),
at a reference magnitude (median of the brightest quarter of the catalogue) and the median sky, so it depends only on
the image (header, orbit, filter, footprint), not on which rows a caller happens to pass.  Grid GDC coordinates come
from a per-chip cubic polynomial raw -> GDC fitted to the catalogue (the affine coefficients are <~1e-3, so a 0.1 px
mapping error moves the subtracted term by <~1e-4 px).  Rows that carry x_gdc / y_gdc use them directly.

  pure affine f      -> removed exactly (residual at machine precision)
  non-linear f       -> unchanged apart from its affine projection, which the plate solution would absorb anyway

Reference variant (gauge_ref): measured 2026-10-13 (ml/v5oct/gauge_test/affine_size.csv), the affine part is NOT
pure gauge: the chain's 2p/4p stages pin it against the header (ACS time-dependent skew ~1e-4 vs STDGDC, which the
zero-centred skew prior relies on), and the validated v4 model P carries it for ACS as large as v5oct does.  With
gauge_ref=SPEC the image's affine projection of f is REPLACED by the reference model's projection on the same grid:
    f_out = f - A_f + A_ref     (non-linear shape from f, per-image linear terms from the reference)
Images whose filter the reference does not cover keep f unchanged.

Spec syntax (everything that goes through make_pos_corr, incl. run_config.json, inherits it):
    DIR:TAG#gauge=affine6                       6p gauge (default per-image model)
    DIR:TAG#gauge=affine8                       8p gauge (with --fit_chip_offset)
    DIR:TAG#gauge=affine6,gauge_ref=DIR2:TAG2   affine part taken from the reference model
The '#gauge=none' or no suffix = the model as trained.
"""
from __future__ import annotations
import os
import numpy as np

GAUGE_MODES = ('affine6', 'affine8')
_NGRID_X, _NGRID_Y = 24, 12
_MIN_ROWS_CHIP = 30


def split_spec(spec: str, with_ref: bool = False):
    """'DIR:TAG#gauge=affine6[,gauge_ref=SPEC]' -> ('DIR:TAG', 'affine6'[, SPEC]); no or 'none' gauge -> mode None."""
    s = str(spec)
    base, _, opt = s.partition('#')
    mode, ref = None, None
    for kv in filter(None, opt.split(',')):
        k, _, v = kv.partition('=')
        if k.strip() == 'gauge_ref':
            ref = v.strip() or None
        elif k.strip() == 'gauge':
            v = v.strip().lower()
            if v in ('', 'none'):
                mode = None
            elif v in GAUGE_MODES:
                mode = v
            else:
                raise ValueError(f'pos_corr gauge: unknown mode {v!r} (use one of {GAUGE_MODES} or none)')
    return (base, mode, ref) if with_ref else (base, mode)


def _poly_terms(u, v, deg):
    cols = [np.ones_like(u)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append(u ** (d - i) * v ** i)
    return np.stack(cols, axis=-1)


class _ChipMap:
    """raw (x, y) -> (x_gdc, y_gdc) for one chip: cubic (affine if few rows) least-squares fit to catalogue rows."""

    def __init__(self, x, y, xg, yg):
        self.cx, self.cy = float(np.median(x)), float(np.median(y))
        self.deg = 3 if len(x) >= _MIN_ROWS_CHIP else 1
        A = _poly_terms((x - self.cx) / 2048.0, (y - self.cy) / 2048.0, self.deg)
        self.kx = np.linalg.lstsq(A, xg, rcond=None)[0]
        self.ky = np.linalg.lstsq(A, yg, rcond=None)[0]

    def __call__(self, x, y):
        A = _poly_terms((np.asarray(x, float) - self.cx) / 2048.0, (np.asarray(y, float) - self.cy) / 2048.0, self.deg)
        return A @ self.kx, A @ self.ky


def _chip_of(tbl, y):
    names = tbl.dtype.names or ()
    return np.asarray(tbl['chip_ext'], int) if 'chip_ext' in names else np.where(np.asarray(y, float) >= 2048, 4, 1)


def _hi_of(y):
    """upper chip (the solver's 8p chip_hi side): raw full-frame y >= 2048."""
    return (np.asarray(y, float) >= 2048).astype(float)


def _design(xg, yg, hi, mode):
    u = (np.asarray(xg, float) - 2048.0) / 2048.0; v = (np.asarray(yg, float) - 2048.0) / 2048.0
    cols = [np.ones_like(u), u, v]
    if mode == 'affine8':
        cols.append(np.asarray(hi, float))
    return np.stack(cols, axis=-1)


class GaugeFixedPosCorr:
    """f - (affine projection of f over the image footprint); see the module docstring."""

    def __init__(self, base, mode: str, spec: str = '', ref=None):
        if mode not in GAUGE_MODES:
            raise ValueError(mode)
        self.base, self.mode, self.spec = base, mode, spec
        self.ref = GaugeFixedPosCorr(ref, mode) if ref is not None else None
        self._cache = {}          # flc path -> dict(coef_x, coef_y, maps, info) or None
        self.last_info = None

    def __getattr__(self, name):            # models, dir, match, ... of the wrapped applier
        return getattr(self.base, name)

    @property
    def summary(self):
        how = 'replaced by the reference model\'s' if self.ref is not None else 'removed'
        return f'{self.base.summary} [gauge fix: {self.mode} projection {how} per image]'

    def read_header(self, flc_path) -> dict:
        h = dict(self.base.read_header(flc_path))
        h['_flc_path'] = str(flc_path)
        return h

    # ---- per-image projection ------------------------------------------------------------------------------
    def _catalog_rows(self, flc_path):
        from astropy.io import fits
        cat = str(flc_path).replace('_flc.fits', '_flc_catalog.fits')
        if not os.path.exists(cat):
            return None
        with fits.open(cat, memmap=False) as hd:
            t = np.array(hd[1].data)
        names = t.dtype.names
        need = ('x', 'y', 'x_gdc', 'y_gdc')
        if not all(n in names for n in need):
            return None
        ok = np.isfinite(t['x']) & np.isfinite(t['y']) & np.isfinite(t['x_gdc']) & np.isfinite(t['y_gdc'])
        return t[ok]

    def _grid(self, t):
        """template rows (catalogue dtype) on a regular raw grid per chip; returns (grid, maps)."""
        names = t.dtype.names
        chip = _chip_of(t, t['y'])
        mag = np.asarray(t['mag'], float) if 'mag' in names else None
        m_ok = mag[np.isfinite(mag)] if mag is not None else np.array([])
        mag_ref = float(np.median(np.sort(m_ok)[:max(1, len(m_ok) // 4)])) if len(m_ok) else np.nan
        maps, parts = {}, []
        for c in np.unique(chip):
            r = t[chip == c]
            if len(r) < 3:
                continue
            maps[int(c)] = _ChipMap(r['x'].astype(float), r['y'].astype(float), r['x_gdc'].astype(float), r['y_gdc'].astype(float))
            if len(r) < _MIN_ROWS_CHIP:
                continue
            x0, x1 = np.percentile(r['x'], [1, 99]); y0, y1 = np.percentile(r['y'], [1, 99])
            gx, gy = np.meshgrid(np.linspace(x0, x1, _NGRID_X), np.linspace(y0, y1, _NGRID_Y))
            g = np.zeros(gx.size, dtype=t.dtype)
            for n in names:                    # non-positional fields: catalogue median (numeric) / first value
                col = r[n]
                if np.issubdtype(col.dtype, np.number):
                    fin = col[np.isfinite(col)] if np.issubdtype(col.dtype, np.floating) else col
                    g[n] = np.median(fin) if len(fin) else 0
                else:
                    g[n] = col[0]
            g['x'] = gx.ravel(); g['y'] = gy.ravel()
            g['x_gdc'], g['y_gdc'] = maps[int(c)](g['x'], g['y'])
            if 'chip_ext' in names:
                g['chip_ext'] = c
            if 'mag' in names and np.isfinite(mag_ref):
                g['mag'] = mag_ref
            parts.append(g)
        return (np.concatenate(parts) if parts else None), maps

    def _projection(self, hdr):
        flc = hdr.get('_flc_path')
        if flc is None:
            raise RuntimeError('pos_corr gauge fix needs the header from GaugeFixedPosCorr.read_header (no _flc_path)')
        if flc in self._cache:
            return self._cache[flc]
        out = None
        t = self._catalog_rows(flc)
        if t is not None and len(t) >= 6:
            g, maps = self._grid(t)
            src = 'grid'
            if g is None:                       # too few rows for a grid on any chip: use the catalogue rows
                g, src = t.copy(), 'rows'
                if 'mag' in t.dtype.names:
                    m = np.asarray(t['mag'], float); m = m[np.isfinite(m)]
                    if len(m):
                        g['mag'] = float(np.median(np.sort(m)[:max(1, len(m) // 4)]))
            b = self.base.bias(g, hdr)
            if b is not None:
                bx, by = np.asarray(b[0], float), np.asarray(b[1], float)
                ok = np.isfinite(bx) & np.isfinite(by)
                hi = _hi_of(g['y'])
                mode = self.mode if (self.mode == 'affine6' or (0 < hi[ok].sum() < ok.sum())) else 'affine6'
                A = _design(g['x_gdc'], g['y_gdc'], hi, mode)[ok]
                cx = np.linalg.lstsq(A, bx[ok], rcond=None)[0]
                cy = np.linalg.lstsq(A, by[ok], rcond=None)[0]
                # skew/scale/rotation equivalents of the removed linear part (GDC px per 2048 px -> dimensionless)
                jxx, jxy, jyx, jyy = cx[1] / 2048, cx[2] / 2048, cy[1] / 2048, cy[2] / 2048
                info = dict(n=int(ok.sum()), source=src, mode=mode,
                            dx0=float(cx[0]), dy0=float(cy[0]),
                            scale=float(0.5 * (jxx + jyy)), rot_rad=float(0.5 * (jyx - jxy)),
                            on_skew=float(0.5 * (jxx - jyy)), off_skew=float(0.5 * (jxy + jyx)),
                            chip_dx=float(cx[3]) if mode == 'affine8' else 0.0,
                            chip_dy=float(cy[3]) if mode == 'affine8' else 0.0,
                            rms_removed=float(np.sqrt(np.mean((A @ cx) ** 2 + (A @ cy) ** 2))),
                            rms_before=float(np.sqrt(np.mean(bx[ok] ** 2 + by[ok] ** 2))),
                            rms_after=float(np.sqrt(np.mean((bx[ok] - A @ cx) ** 2 + (by[ok] - A @ cy) ** 2))))
                out = dict(cx=cx, cy=cy, maps=maps, mode=mode, info=info)
        self._cache[flc] = out
        return out

    # ---- applier interface ---------------------------------------------------------------------------------
    def bias(self, tbl, hdr, *args, **kw):
        b = self.base.bias(tbl, hdr, *args, **kw)
        if b is None:
            return None
        p = self._projection(hdr)
        self.last_info = p['info'] if p else None
        if p is None:
            return b
        cx, cy = p['cx'], p['cy']
        if self.ref is not None:
            pr = self.ref._projection(hdr)
            if pr is None:                      # reference does not cover this image: model as trained
                return b
            if len(pr['cx']) != len(cx):        # 8p fell back to 6p in one of them: compare the 6p parts
                cx, cy, rx, ry = cx[:3], cy[:3], pr['cx'][:3], pr['cy'][:3]
            else:
                rx, ry = pr['cx'], pr['cy']
            cx, cy = cx - rx, cy - ry
        names = tbl.dtype.names or ()
        x = np.asarray(tbl['x'], float); y = np.asarray(tbl['y'], float)
        if 'x_gdc' in names and 'y_gdc' in names:
            xg = np.asarray(tbl['x_gdc'], float); yg = np.asarray(tbl['y_gdc'], float)
        else:
            chip = _chip_of(tbl, y); xg = np.full(len(x), np.nan); yg = np.full(len(x), np.nan)
            for c, m in p['maps'].items():
                s = chip == c
                if s.any():
                    xg[s], yg[s] = m(x[s], y[s])
            miss = ~np.isfinite(xg)
            if miss.any() and p['maps']:        # chip without its own map: nearest available map
                m = next(iter(p['maps'].values())); xg[miss], yg[miss] = m(x[miss], y[miss])
        A = _design(xg, yg, _hi_of(y), p['mode'])[:, :len(cx)]
        return np.asarray(b[0], float) - A @ cx, np.asarray(b[1], float) - A @ cy

    def area_mag(self, tbl, hdr, *args, h=4.0, **kw):
        from bp3m.pos_corr_basis import area_mag
        return area_mag(lambda t: self.bias(t, hdr, *args, **kw), tbl, h=h)
