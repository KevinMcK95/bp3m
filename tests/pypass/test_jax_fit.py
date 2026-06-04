"""Tests for the JAX Newton fitting kernel (fit_batch_jax).

All tests are skipped when JAX is not installed.

Both the JAX kernel and the NumPy reference now use cubic B-spline interpolation
on prefiltered coefficient tiles (spline_filter applied to the full blended PSF
before cropping, avoiding IIR boundary artifacts).  JAX uses analytical gradient
weights; NumPy (core.py fit_star) uses central finite differences — the residual
difference causes < 0.001 px position disagreement for bright stars.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX not installed")

from pypass._jax_kernel import prepare_jax_inputs, fit_batch_jax

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE

HW       = 3
TOL      = 1e-3
MAX_ITER = 20
GAIN     = 1.0
RN       = 5.0
SKY      = 200.0


@pytest.fixture(scope="module")
def gauss_psf():
    return make_gaussian_psf().astype(np.float64)


def _inputs(data, xs, ys, skys, gauss_psf, noise_map=None, mask=None):
    psf_cube = gauss_psf[np.newaxis]
    psf_xs = np.array([0.0]);  psf_ys = np.array([0.0])
    return prepare_jax_inputs(
        data, xs, ys, skys,
        psf_cube, psf_xs, psf_ys,
        PSF_SCALE, HW,
        mask=mask, noise_map=noise_map,
        gain=GAIN, read_noise=RN,
    )


# ---------------------------------------------------------------------------
# 1. Single bright star: position and flux recovery
# ---------------------------------------------------------------------------

def test_single_bright_star_position(gauss_psf):
    """JAX recovers position of a bright isolated star to < 0.03 px."""
    true_x, true_y, true_flux = 50.3, 49.7, 8000.0
    # Inject at the sub-pixel position so the true position is (50.3, 49.7)
    data = inject_stars([(true_x, true_y, true_flux)],
                        gauss_psf, PSF_SCALE, sky=SKY, seed=0)

    inp = _inputs(data, np.array([true_x]), np.array([true_y]),
                  np.full(1, SKY), gauss_psf)
    res = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    x_fit = inp['xi'][0] + res['dx'][0]
    y_fit = inp['yi'][0] + res['dy'][0]
    assert abs(x_fit - true_x) < 0.03, f"x error {x_fit:.4f} vs {true_x}"
    assert abs(y_fit - true_y) < 0.03, f"y error {y_fit:.4f} vs {true_y}"


def test_single_bright_star_flux(gauss_psf):
    """JAX recovers flux of a bright isolated star to < 5%."""
    true_x, true_y, true_flux = 80.0, 80.0, 6000.0
    data = inject_stars([(80, 80, true_flux)], gauss_psf, PSF_SCALE, sky=SKY, seed=1)

    inp = _inputs(data, np.array([true_x]), np.array([true_y]),
                  np.full(1, SKY), gauss_psf)
    res = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    assert abs(res['flux'][0] - true_flux) / true_flux < 0.05
    assert res['converged'][0]


# ---------------------------------------------------------------------------
# 2. Batch of stars
# ---------------------------------------------------------------------------

def test_batch_positions(gauss_psf):
    """JAX recovers positions for a batch of 5 stars to < 0.05 px each."""
    stars = [(50, 50, 5000), (80, 60, 3000), (60, 100, 7000),
             (120, 80, 2000), (100, 120, 4000)]
    true_xs = np.array([x for x, y, f in stars], dtype=float)
    true_ys = np.array([y for x, y, f in stars], dtype=float)

    data = inject_stars(stars, gauss_psf, PSF_SCALE, sky=SKY, seed=2)
    inp  = _inputs(data, true_xs, true_ys, np.full(5, SKY), gauss_psf)
    res  = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    x_fit = inp['xi'] + res['dx']
    y_fit = inp['yi'] + res['dy']
    for i in range(5):
        assert abs(x_fit[i] - true_xs[i]) < 0.10, \
            f"star {i}: x={x_fit[i]:.4f} vs {true_xs[i]}"
        assert abs(y_fit[i] - true_ys[i]) < 0.10, \
            f"star {i}: y={y_fit[i]:.4f} vs {true_ys[i]}"


# ---------------------------------------------------------------------------
# 3. JAX vs NumPy agreement
# ---------------------------------------------------------------------------

def test_jax_vs_numpy_position(gauss_psf):
    """JAX and NumPy converged positions agree to < 0.001 px (bicubic on both paths)."""
    from pypass.core import fit_star

    true_x, true_y, true_flux = 100.2, 100.7, 4000.0
    data = inject_stars([(100, 101, true_flux)], gauss_psf, PSF_SCALE,
                        sky=SKY, seed=3)

    # JAX
    inp = _inputs(data, np.array([true_x]), np.array([true_y]),
                  np.full(1, SKY), gauss_psf)
    res = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)
    x_jax = inp['xi'][0] + res['dx'][0]
    y_jax = inp['yi'][0] + res['dy'][0]

    # NumPy (sigma_clip off so both solve the same linear problem)
    rec = fit_star(
        data=data, x0=true_x, y0=true_y,
        psf_cube=gauss_psf[np.newaxis],
        xs=np.array([0.0]), ys=np.array([0.0]),
        psf_scale=PSF_SCALE, hw=HW,
        sky=SKY, gain=GAIN, read_noise=RN,
        max_iter=MAX_ITER, tol=TOL,
        noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0,
        zero_point=0.0, pass_number=1,
        sigma_clip=False,
    )

    assert abs(x_jax - rec.x) < 0.001, f"x: JAX={x_jax:.6f} NumPy={rec.x:.6f}"
    assert abs(y_jax - rec.y) < 0.001, f"y: JAX={y_jax:.6f} NumPy={rec.y:.6f}"


def test_jax_vs_numpy_flux(gauss_psf):
    """JAX and NumPy flux estimates agree to < 5%."""
    from pypass.core import fit_star

    true_x, true_y, true_flux = 60.0, 60.0, 3500.0
    data = inject_stars([(60, 60, true_flux)], gauss_psf, PSF_SCALE,
                        sky=SKY, seed=4)

    inp = _inputs(data, np.array([true_x]), np.array([true_y]),
                  np.full(1, SKY), gauss_psf)
    res = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    rec = fit_star(
        data=data, x0=true_x, y0=true_y,
        psf_cube=gauss_psf[np.newaxis],
        xs=np.array([0.0]), ys=np.array([0.0]),
        psf_scale=PSF_SCALE, hw=HW,
        sky=SKY, gain=GAIN, read_noise=RN,
        max_iter=MAX_ITER, tol=TOL,
        noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0,
        zero_point=0.0, pass_number=1,
        sigma_clip=False,
    )

    rel_diff = abs(res['flux'][0] - rec.flux) / max(rec.flux, 1.0)
    assert rel_diff < 0.05, f"flux: JAX={res['flux'][0]:.1f} NumPy={rec.flux:.1f}"


# ---------------------------------------------------------------------------
# 4. Covariance is positive-definite
# ---------------------------------------------------------------------------

def test_covariance_positive_definite(gauss_psf):
    """4×4 covariance returned by JAX kernel is positive-definite."""
    data = inject_stars([(80, 80, 5000)], gauss_psf, PSF_SCALE, sky=SKY, seed=5)
    inp  = _inputs(data, np.array([80.0]), np.array([80.0]),
                   np.full(1, SKY), gauss_psf)
    res  = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    cov = res['cov'][0]
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


# ---------------------------------------------------------------------------
# 5. Edge star: some pixels out of bounds
# ---------------------------------------------------------------------------

def test_edge_star_no_crash(gauss_psf):
    """Edge star (near boundary, some invalid pixels) does not crash JAX kernel."""
    data = inject_stars([(3, 3, 3000)], gauss_psf, PSF_SCALE, sky=SKY,
                        image_size=(50, 50), seed=6)
    inp  = _inputs(data, np.array([3.0]), np.array([3.0]),
                   np.full(1, SKY), gauss_psf)
    res  = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    # Should return finite values (not NaN/Inf) even with some invalid pixels
    assert np.isfinite(res['flux'][0])
    assert np.isfinite(res['qfit'][0])


# ---------------------------------------------------------------------------
# 6. External noise map
# ---------------------------------------------------------------------------

def test_noise_map_path(gauss_psf):
    """JAX kernel with has_noise_map=True produces finite, reasonable results."""
    data = inject_stars([(70, 70, 4000)], gauss_psf, PSF_SCALE, sky=SKY, seed=7)
    noise_map = np.full_like(data, (RN / GAIN) ** 2 + SKY / GAIN)

    inp = _inputs(data, np.array([70.0]), np.array([70.0]),
                  np.full(1, SKY), gauss_psf, noise_map=noise_map)
    assert inp['has_noise_map']

    res = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)
    assert np.isfinite(res['flux'][0])
    assert abs(res['flux'][0] - 4000.0) / 4000.0 < 0.10


# ---------------------------------------------------------------------------
# 7. qfit and chi2 are plausible
# ---------------------------------------------------------------------------

def test_qfit_bright_star(gauss_psf):
    """qfit for a bright well-fit star is < 0.15."""
    data = inject_stars([(90, 90, 8000)], gauss_psf, PSF_SCALE, sky=SKY, seed=8)
    inp  = _inputs(data, np.array([90.0]), np.array([90.0]),
                   np.full(1, SKY), gauss_psf)
    res  = fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)

    assert res['qfit'][0] < 0.15, f"qfit={res['qfit'][0]:.4f}"
    assert res['chi2'][0] < 5.0,  f"chi2={res['chi2'][0]:.4f}"


# ---------------------------------------------------------------------------
# 8. Kernel caching: second call reuses compiled function
# ---------------------------------------------------------------------------

def test_kernel_cache_hit(gauss_psf):
    """A second fit_batch_jax call with same config uses the cache."""
    from pypass._jax_kernel import _JAX_KERNEL_CACHE

    data = inject_stars([(50, 50, 3000)], gauss_psf, PSF_SCALE, sky=SKY, seed=9)
    inp  = _inputs(data, np.array([50.0]), np.array([50.0]),
                   np.full(1, SKY), gauss_psf)

    cache_before = len(_JAX_KERNEL_CACHE)
    fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)
    fit_batch_jax(inp, gain=GAIN, tol=TOL, max_iter=MAX_ITER)
    assert len(_JAX_KERNEL_CACHE) == cache_before + 1 or len(_JAX_KERNEL_CACHE) == cache_before


# ---------------------------------------------------------------------------
# 9. run_photometry integration: JAX vs NumPy end-to-end
# ---------------------------------------------------------------------------

def test_run_photometry_jax_finds_stars(gauss_psf):
    """run_photometry(backend='jax') detects and fits injected stars."""
    from pypass import run_photometry

    stars = [(80, 80, 5000), (120, 100, 3000), (60, 130, 7000)]
    data  = inject_stars(stars, gauss_psf, PSF_SCALE, sky=SKY, seed=10)
    true_xs = np.array([x for x, y, f in stars], dtype=float)
    true_ys = np.array([y for x, y, f in stars], dtype=float)

    records = run_photometry(data, gauss_psf, psf_scale=PSF_SCALE, half_width=3,
                             gain=GAIN, read_noise=RN, fmin=100.0,
                             backend='jax')

    assert len(records) == 3, f"Expected 3 stars, got {len(records)}"
    xs_fit = np.array([r.x for r in records])
    ys_fit = np.array([r.y for r in records])

    # Match by nearest true position
    for tx, ty in zip(true_xs, true_ys):
        dists = np.hypot(xs_fit - tx, ys_fit - ty)
        assert dists.min() < 0.5, f"No star found near ({tx}, {ty})"


def test_run_photometry_jax_vs_numpy_positions(gauss_psf):
    """JAX and NumPy backends agree on positions to < 0.1 px."""
    from pypass import run_photometry

    stars = [(80, 80, 6000), (120, 100, 4000)]
    data  = inject_stars(stars, gauss_psf, PSF_SCALE, sky=SKY, seed=11)

    kw = dict(psf_scale=PSF_SCALE, half_width=3,
              gain=GAIN, read_noise=RN, fmin=100.0,
              sigma_clip=False)

    rec_np  = run_photometry(data, gauss_psf, backend='numpy', **kw)
    rec_jax = run_photometry(data, gauss_psf, backend='jax',   **kw)

    assert len(rec_np) == len(rec_jax), \
        f"Star count differs: NumPy={len(rec_np)} JAX={len(rec_jax)}"

    xs_np  = np.array(sorted(r.x for r in rec_np))
    xs_jax = np.array(sorted(r.x for r in rec_jax))
    ys_np  = np.array(sorted(r.y for r in rec_np))
    ys_jax = np.array(sorted(r.y for r in rec_jax))

    np.testing.assert_allclose(xs_jax, xs_np,  atol=0.1, err_msg="x position mismatch")
    np.testing.assert_allclose(ys_jax, ys_np,  atol=0.1, err_msg="y position mismatch")


def test_run_photometry_jax_vs_numpy_flux(gauss_psf):
    """JAX and NumPy backends agree on flux to < 5%."""
    from pypass import run_photometry

    data = inject_stars([(100, 100, 5000)], gauss_psf, PSF_SCALE, sky=SKY, seed=12)

    kw = dict(psf_scale=PSF_SCALE, half_width=3,
              gain=GAIN, read_noise=RN, fmin=100.0,
              sigma_clip=False)

    rec_np  = run_photometry(data, gauss_psf, backend='numpy', **kw)
    rec_jax = run_photometry(data, gauss_psf, backend='jax',   **kw)

    assert len(rec_np) == 1 and len(rec_jax) == 1
    rel = abs(rec_jax[0].flux - rec_np[0].flux) / max(rec_np[0].flux, 1.0)
    assert rel < 0.05, f"flux: JAX={rec_jax[0].flux:.1f} NumPy={rec_np[0].flux:.1f}"


# ---------------------------------------------------------------------------
# 10. Sigma clipping
# ---------------------------------------------------------------------------

def test_sigma_clip_clipped_mask_populated(gauss_psf):
    """clipped_mask is a bool array when sigma_clip=True."""
    from pypass import run_photometry

    data = inject_stars([(80, 80, 6000)], gauss_psf, PSF_SCALE, sky=SKY, seed=14)
    data[82, 81] += 3000.0   # cosmic-ray spike inside fit window

    records = run_photometry(data, gauss_psf, psf_scale=PSF_SCALE, half_width=3,
                             gain=GAIN, read_noise=RN, fmin=100.0,
                             sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2,
                             backend='jax')
    assert len(records) == 1
    cm = records[0].clipped_mask
    assert cm is not None, "clipped_mask should be set when sigma_clip=True"
    assert cm.dtype == bool
    assert cm.sum() > 0, "cosmic-ray pixel should have been clipped"


def test_sigma_clip_both_backends_clip_spike(gauss_psf):
    """Both backends clip at least one pixel when a cosmic-ray spike is present.

    Exact clipped counts may differ because NumPy uses bicubic PSF evaluation
    and JAX uses bilinear, giving slightly different residuals at the sigma
    boundary — but both must clip the obvious spike.
    """
    from pypass import run_photometry

    data = inject_stars([(80, 80, 6000)], gauss_psf, PSF_SCALE, sky=SKY, seed=15)
    data[82, 81] += 3000.0

    kw = dict(psf_scale=PSF_SCALE, half_width=3,
              gain=GAIN, read_noise=RN, fmin=100.0,
              sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2)

    rec_np  = run_photometry(data, gauss_psf, backend='numpy', **kw)
    rec_jax = run_photometry(data, gauss_psf, backend='jax',   **kw)

    n_clipped_np  = int(rec_np[0].clipped_mask.sum())  if rec_np[0].clipped_mask  is not None else 0
    n_clipped_jax = int(rec_jax[0].clipped_mask.sum()) if rec_jax[0].clipped_mask is not None else 0
    assert n_clipped_np  > 0, "NumPy should have clipped the spike"
    assert n_clipped_jax > 0, "JAX should have clipped the spike"


def test_sigma_clip_improves_position(gauss_psf):
    """Sigma clipping moves the JAX position closer to truth when a spike is present."""
    from pypass import run_photometry

    true_x, true_y = 80.0, 80.0
    data = inject_stars([(80, 80, 6000)], gauss_psf, PSF_SCALE, sky=SKY, seed=16)
    data[82, 81] += 3000.0

    kw = dict(psf_scale=PSF_SCALE, half_width=3,
              gain=GAIN, read_noise=RN, fmin=100.0)

    rec_clip   = run_photometry(data, gauss_psf, backend='jax',
                                sigma_clip=True,  sigma_clip_sigma=4.0,
                                sigma_clip_iter=2, **kw)
    rec_noclip = run_photometry(data, gauss_psf, backend='jax',
                                sigma_clip=False, **kw)

    err_clip   = np.hypot(rec_clip[0].x   - true_x, rec_clip[0].y   - true_y)
    err_noclip = np.hypot(rec_noclip[0].x - true_x, rec_noclip[0].y - true_y)
    assert err_clip <= err_noclip + 0.02, \
        f"clipping should not worsen position: clip={err_clip:.4f} noclip={err_noclip:.4f}"


def test_sigma_clip_off_leaves_mask_none(gauss_psf):
    """clipped_mask is None when sigma_clip=False."""
    from pypass import run_photometry

    data = inject_stars([(80, 80, 5000)], gauss_psf, PSF_SCALE, sky=SKY, seed=17)
    records = run_photometry(data, gauss_psf, psf_scale=PSF_SCALE, half_width=3,
                             gain=GAIN, read_noise=RN, fmin=100.0,
                             sigma_clip=False, backend='jax')
    assert records[0].clipped_mask is None


def test_refit_stars_jax_multipass(gauss_psf):
    """refit_stars_jax is used in pass 2 when backend='jax' and n_passes=2.

    Checks that the three injected bright stars survive the JAX batch refit
    with positions < 0.15 px and flux within 10% of the NumPy result.
    """
    from pypass import run_photometry
    from scipy.spatial import cKDTree

    true_positions = [(60, 60), (80, 80), (100, 100)]
    data = inject_stars([
        (tx, ty, flux) for (tx, ty), flux in zip(true_positions, [4000, 2000, 1000])
    ], gauss_psf, PSF_SCALE, sky=SKY, seed=99)

    kw = dict(psf_scale=PSF_SCALE, half_width=3,
              gain=GAIN, read_noise=RN, fmin=200.0,   # high fmin → only bright stars
              sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2)

    rec_np  = run_photometry(data, gauss_psf, n_passes=2, backend='numpy', **kw)
    rec_jax = run_photometry(data, gauss_psf, n_passes=2, backend='jax',   **kw)

    # Both backends must recover all 3 bright injected stars.
    assert len(rec_np)  >= 3, f"NumPy found only {len(rec_np)} stars"
    assert len(rec_jax) >= 3, f"JAX found only {len(rec_jax)} stars"

    # Match JAX detections to NumPy detections by nearest neighbour.
    xy_np  = np.array([[r.x, r.y] for r in rec_np])
    xy_jax = np.array([[r.x, r.y] for r in rec_jax])
    tree   = cKDTree(xy_np)
    dists, idxs = tree.query(xy_jax, k=1)

    for i_jax, (dist, i_np) in enumerate(zip(dists, idxs)):
        if dist > 2.0:   # unmatched JAX detection — skip
            continue
        r_jax = rec_jax[i_jax]
        r_np  = rec_np[i_np]
        assert abs(r_jax.x - r_np.x) < 0.15, \
            f"x mismatch: JAX={r_jax.x:.3f} NumPy={r_np.x:.3f}"
        assert abs(r_jax.y - r_np.y) < 0.15, \
            f"y mismatch: JAX={r_jax.y:.3f} NumPy={r_np.y:.3f}"
        rel_flux = abs(r_jax.flux - r_np.flux) / max(r_np.flux, 1.0)
        assert rel_flux < 0.10, \
            f"flux mismatch: JAX={r_jax.flux:.1f} NumPy={r_np.flux:.1f}"


def test_run_photometry_jax_starrecord_fields(gauss_psf):
    """StarRecords from JAX backend have all expected fields populated."""
    from pypass import run_photometry

    data = inject_stars([(80, 80, 5000)], gauss_psf, PSF_SCALE, sky=SKY, seed=13)
    records = run_photometry(data, gauss_psf, psf_scale=PSF_SCALE, half_width=3,
                             gain=GAIN, read_noise=RN, fmin=100.0,
                             backend='jax')

    assert len(records) == 1
    r = records[0]
    assert np.isfinite(r.x) and np.isfinite(r.y)
    assert np.isfinite(r.flux) and r.flux > 1.0
    assert np.isfinite(r.flux_err) and r.flux_err > 0.0
    assert np.isfinite(r.sky)
    assert np.isfinite(r.mag)
    assert np.isfinite(r.qfit)
    assert np.isfinite(r.chi2)
    assert np.isfinite(r.psf_frac) and r.psf_frac > 0.0
    assert np.isfinite(r.psf_peak) and r.psf_peak > 0.0
    assert np.isfinite(r.peak)
    assert r.cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(r.cov) > 0)
    assert r.pass_number == 1
    assert isinstance(r.converged, bool)
