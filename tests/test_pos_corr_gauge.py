"""Gauge fix of the learned GDC correction (bp3m/pos_corr_gauge.py).

(a) a correction that is purely affine in the GDC frame is removed to machine precision (6p and 8p);
(b) a non-linear correction keeps its shape: after the fix it equals the input minus that input's own affine
    projection over the footprint, and its non-linear part is untouched;
(c) the spec suffix round-trips through make_pos_corr-style parsing.
Runs with plain python (no pytest needed): python tests/test_pos_corr_gauge.py
"""
import os, sys, tempfile
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bp3m.pos_corr_gauge import GaugeFixedPosCorr, split_spec


def _raw_to_gdc(x, y):
    """mildly non-linear raw -> GDC map (stands in for STDGDC)."""
    u, v = (x - 2048) / 2048, (y - 2048) / 2048
    return x + 3.0 * u * v + 1.5 * u ** 2, y - 2.0 * v ** 2 + 0.8 * u * v


def _catalog(tmp, n=4000, seed=1, two_chips=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(5, 4090, n); y = rng.uniform(5, 4090 if two_chips else 2040, n)
    y = y[(y < 2040) | (y > 2056)]; x = x[:len(y)]
    xg, yg = _raw_to_gdc(x, y)
    cols = [fits.Column(name='x', format='D', array=x), fits.Column(name='y', format='D', array=y),
            fits.Column(name='x_gdc', format='D', array=xg), fits.Column(name='y_gdc', format='D', array=yg),
            fits.Column(name='chip_ext', format='J', array=np.where(y >= 2048, 4, 1)),
            fits.Column(name='mag', format='D', array=rng.uniform(-14, -5, len(x))),
            fits.Column(name='sky', format='D', array=rng.uniform(5, 50, len(x)))]
    flc = os.path.join(tmp, 'jtest01abq_flc.fits')
    fits.BinTableHDU.from_columns(cols).writeto(flc.replace('_flc.fits', '_flc_catalog.fits'), overwrite=True)
    fits.PrimaryHDU().writeto(flc, overwrite=True)
    return flc, fits.getdata(flc.replace('_flc.fits', '_flc_catalog.fits'))


class _Fake:
    """applier with a known bias f(x_gdc, y_gdc, chip) computed from the RAW position through the same map."""
    summary = 'fake'

    def __init__(self, f):
        self.f = f

    @staticmethod
    def read_header(flc):
        return {'INSTRUME': 'ACS', 'DETECTOR': 'WFC'}

    def bias(self, tbl, hdr):
        x = np.asarray(tbl['x'], float); y = np.asarray(tbl['y'], float)
        xg, yg = _raw_to_gdc(x, y)
        return self.f(xg, yg, (y >= 2048).astype(float))


def _affine(xg, yg, hi, k):
    u, v = (xg - 2048) / 2048, (yg - 2048) / 2048
    return (k[0] + k[1] * u + k[2] * v + k[3] * hi, k[4] + k[5] * u + k[6] * v + k[7] * hi)


def test_pure_affine_removed():
    with tempfile.TemporaryDirectory() as tmp:
        flc, cat = _catalog(tmp)
        k6 = [0.3, 0.05, -0.02, 0.0, -0.1, 0.01, 0.04, 0.0]          # ~2.5e-5 skew-sized terms
        g = GaugeFixedPosCorr(_Fake(lambda xg, yg, hi: _affine(xg, yg, hi, k6)), 'affine6')
        bx, by = g.bias(cat, g.read_header(flc))
        assert np.max(np.abs(bx)) < 1e-9 and np.max(np.abs(by)) < 1e-9, (np.max(np.abs(bx)), np.max(np.abs(by)))
        k8 = [0.3, 0.05, -0.02, 0.07, -0.1, 0.01, 0.04, -0.03]       # + upper-chip offset
        g8 = GaugeFixedPosCorr(_Fake(lambda xg, yg, hi: _affine(xg, yg, hi, k8)), 'affine8')
        bx, by = g8.bias(cat, g8.read_header(flc))
        assert np.max(np.abs(bx)) < 1e-9 and np.max(np.abs(by)) < 1e-9
        g6 = GaugeFixedPosCorr(_Fake(lambda xg, yg, hi: _affine(xg, yg, hi, k8)), 'affine6')
        bx, _ = g6.bias(cat, g6.read_header(flc))                     # 6p cannot absorb a chip step: it must remain
        assert np.ptp(bx) > 0.05


def test_nonlinear_shape_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        flc, cat = _catalog(tmp)
        nl = lambda xg, yg: (0.02 * np.sin(xg / 500.0) * np.cos(yg / 700.0), 0.015 * ((xg - 2048) / 2048) ** 3)
        k6 = [0.3, 0.05, -0.02, 0.0, -0.1, 0.01, 0.04, 0.0]
        f = lambda xg, yg, hi: tuple(a + b for a, b in zip(nl(xg, yg), _affine(xg, yg, hi, k6)))
        g = GaugeFixedPosCorr(_Fake(f), 'affine6'); h = g.read_header(flc)
        bx, by = g.bias(cat, h)
        # the fixed correction = non-linear part minus ITS OWN affine projection (the affine k6 part is gone)
        gn = GaugeFixedPosCorr(_Fake(lambda xg, yg, hi: nl(xg, yg)), 'affine6')
        nx, ny = gn.bias(cat, gn.read_header(flc))
        assert np.max(np.abs(bx - nx)) < 1e-6 and np.max(np.abs(by - ny)) < 1e-6
        # and the shape (non-affine content) is unchanged: residual of an affine fit to the output == to the input
        xg, yg = np.asarray(cat['x_gdc']), np.asarray(cat['y_gdc'])
        A = np.stack([np.ones_like(xg), (xg - 2048) / 2048, (yg - 2048) / 2048], 1)
        for out, inp in ((bx, nl(xg, yg)[0]), (by, nl(xg, yg)[1])):
            r_out = out - A @ np.linalg.lstsq(A, out, rcond=None)[0]
            r_in = inp - A @ np.linalg.lstsq(A, inp, rcond=None)[0]
            assert np.max(np.abs(r_out - r_in)) < 1e-4, np.max(np.abs(r_out - r_in))
        # rows without x_gdc (v2 master loader) use the fitted raw->GDC map: same answer to < 1e-3 px
        sub = np.rec.fromarrays([cat['x'][:500], cat['y'][:500], cat['chip_ext'][:500], cat['mag'][:500]], names=['x', 'y', 'chip_ext', 'mag'])
        sx, sy = g.bias(sub, h)
        assert np.max(np.abs(sx - bx[:500])) < 1e-3 and np.max(np.abs(sy - by[:500])) < 1e-3


def test_reference_affine():
    """gauge_ref: output = f - A_f + A_ref -> non-linear shape of f with exactly the reference's affine part."""
    with tempfile.TemporaryDirectory() as tmp:
        flc, cat = _catalog(tmp)
        nl = lambda xg, yg: (0.02 * np.sin(xg / 500.0) * np.cos(yg / 700.0), 0.015 * ((xg - 2048) / 2048) ** 3)
        kA = [0.3, 0.05, -0.02, 0.0, -0.1, 0.01, 0.04, 0.0]; kB = [-0.2, 0.03, 0.01, 0.0, 0.05, -0.02, 0.02, 0.0]
        f = _Fake(lambda xg, yg, hi: tuple(a + b for a, b in zip(nl(xg, yg), _affine(xg, yg, hi, kA))))
        ref = _Fake(lambda xg, yg, hi: _affine(xg, yg, hi, kB))
        g = GaugeFixedPosCorr(f, 'affine6', ref=ref); bx, by = g.bias(cat, g.read_header(flc))
        gn = GaugeFixedPosCorr(_Fake(lambda xg, yg, hi: nl(xg, yg)), 'affine6'); nx, ny = gn.bias(cat, gn.read_header(flc))
        xg, yg = np.asarray(cat['x_gdc']), np.asarray(cat['y_gdc']); ax, ay = _affine(xg, yg, 0 * xg, kB)
        assert np.max(np.abs(bx - (nx + ax))) < 1e-6 and np.max(np.abs(by - (ny + ay))) < 1e-6


def test_spec_parsing():
    assert split_spec('/m/basis:C_r3') == ('/m/basis:C_r3', None)
    assert split_spec('/m/basis:C_r3#gauge=affine6') == ('/m/basis:C_r3', 'affine6')
    assert split_spec('/m/basis:C_r3:noorbit#gauge=affine8') == ('/m/basis:C_r3:noorbit', 'affine8')
    assert split_spec('/m/basis:C_r3#gauge=none') == ('/m/basis:C_r3', None)
    assert split_spec('/m/b:C_r3#gauge=affine6,gauge_ref=/v4/g:C_r2', with_ref=True) == ('/m/b:C_r3', 'affine6', '/v4/g:C_r2')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
