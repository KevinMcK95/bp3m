"""Image-level quality checks run before the Gaia cross-match (user 2026-10-01).

Trailed exposures: guiding can fail while the headers still say FINE lock / EXPFLAG=NORMAL
(47 Tuc j8c002* 2002: every star a ~2:1 streak at the same angle; CR-SPLIT rejection then flagged
the whole stars, pypass kept only faint fragments and the cross-match locked onto chance pairs).

Metric: second moments in memory-mapped cutouts around the brightest unsaturated star candidates
of the pypass catalogue -> median axis ratio and orientation coherence |<exp(2i theta)>|.  Calibrated
on 592 random archive images (ACS/WFC + WFC3/UVIS): blends/galaxies give high axis ratios but
random orientations (coherence 0.06-0.31); the PSF+distortion gives coherent but mild elongation
(~1.1).  The trailed visit sits at axis ratio 1.62-1.94 with coherence 0.82-0.90; no random image
exceeds (1.4, 0.7).  Results are cached in <image>/image_quality.json keyed by file fingerprints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_QUALITY_VERSION = 1
TRAIL_AXIS_RATIO = 1.4
TRAIL_COHERENCE = 0.7
TRAIL_MIN_SOURCES = 8
_NMAX, _HW = 40, 6


def trail_metric(flc, catalog) -> dict:
    from astropy.io import fits
    cat = fits.getdata(catalog)
    names = cat.dtype.names
    ok = np.isfinite(cat['flux']) & (cat['flux'] > 0)
    if 'n_sat' in names:
        ok &= cat['n_sat'] == 0
    if 'is_star_candidate' in names:
        ok &= cat['is_star_candidate'].astype(bool)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-cat['flux'][idx])][:_NMAX]
    q = []
    with fits.open(flc, memmap=True) as h:
        det = str(h[0].header.get('DETECTOR', '')).upper()
        exts = {int(x.ver) * 3 - 2: x for x in h if x.name == 'SCI'}
        yy, xx = np.mgrid[-_HW:_HW + 1, -_HW:_HW + 1]
        for k in idx:
            ext = int(cat['chip_ext'][k]) if 'chip_ext' in names else 1
            hd = exts.get(ext)
            if hd is None:
                continue
            # catalogue x, y are in the combined detector frame: remove the chip offset and the
            # subarray origin (pypass chip_detector_offsets) to get back to this extension's pixels
            yoff = 2048.0 if (hd.header.get('CCDCHIP') == 1 and det in ('WFC', 'UVIS')) else 0.0
            x = float(cat['x'][k]) + float(hd.header.get('LTV1', 0) or 0)
            y = float(cat['y'][k]) - yoff + float(hd.header.get('LTV2', 0) or 0)
            xi, yi = int(round(x)), int(round(y)); ny, nx = hd.shape
            if xi < _HW + 2 or yi < _HW + 2 or xi > nx - _HW - 3 or yi > ny - _HW - 3:
                continue
            c = np.array(hd.section[yi - _HW:yi + _HW + 1, xi - _HW:xi + _HW + 1], float)
            ring = np.array(hd.section[yi - _HW - 2:yi + _HW + 3, xi - _HW - 2:xi + _HW + 3], float)
            sky = np.median(np.concatenate([ring[:2].ravel(), ring[-2:].ravel(), ring[:, :2].ravel(), ring[:, -2:].ravel()]))
            c = np.clip(c - sky, 0, None)
            if c.sum() <= 0:
                continue
            w = c / c.sum(); mx, my = (w * xx).sum(), (w * yy).sum()
            mxx = (w * (xx - mx) ** 2).sum(); myy = (w * (yy - my) ** 2).sum(); mxy = (w * (xx - mx) * (yy - my)).sum()
            l1, l2 = np.linalg.eigvalsh([[mxx, mxy], [mxy, myy]])
            if l1 <= 0:
                continue
            q.append((np.sqrt(l2 / l1), 0.5 * np.arctan2(2 * mxy, mxx - myy)))
    out = {'n_sources': len(q)}
    if len(q) >= TRAIL_MIN_SOURCES:
        q = np.array(q)
        e = np.mean(np.exp(2j * q[:, 1]))
        out.update(axis_ratio=float(np.median(q[:, 0])), pa_coherence=float(abs(e)),
                   pa_deg=float(np.degrees(np.angle(e)) / 2))
    return out


def _fingerprint(p) -> list:
    st = Path(p).stat()
    return [st.st_size, st.st_mtime_ns]


def image_quality(flc, catalog) -> dict:
    """Cached quality record: {'trailed': bool, 'axis_ratio', 'pa_coherence', ...}."""
    root = Path(flc).parent
    side = root / 'image_quality.json'
    key = {'version': IMAGE_QUALITY_VERSION, 'flc': _fingerprint(flc), 'catalog': _fingerprint(catalog),
           'thresholds': [TRAIL_AXIS_RATIO, TRAIL_COHERENCE]}
    if side.exists():
        try:
            rec = json.loads(side.read_text())
            if rec.get('key') == key:
                return rec
        except Exception:
            pass
    try:
        m = trail_metric(flc, catalog)
    except Exception as e:
        m = {'error': f'{type(e).__name__}: {e}'}
    trailed = bool(m.get('axis_ratio', 0) > TRAIL_AXIS_RATIO and m.get('pa_coherence', 0) > TRAIL_COHERENCE)
    rec = {'key': key, 'trailed': trailed, **m}
    try:
        side.write_text(json.dumps(rec, indent=2))
    except Exception:
        pass
    return rec
