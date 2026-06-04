"""Tests for PSF evaluation, sky estimation, source finding, and fitting."""

import numpy as np
import pytest
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE, PSF_SIZE

from pypass.core import (
    estimate_sky, interpolate_psf, eval_psf_and_grad,
    find_sources, fit_star, StarRecord,
)
from pypass.diagnostics import summarize_catalog


# ---------------------------------------------------------------------------
# 1. PSF normalisation
# ---------------------------------------------------------------------------

def test_psf_normalisation(gauss_psf):
    """sum(P over 5×5 window) / psf_scale² should be ≈ 1 for a centred star."""
    psf_scale = PSF_SCALE
    hw = 2
    diy = np.arange(-hw, hw + 1)[:, np.newaxis]
    dix = np.arange(-hw, hw + 1)[np.newaxis, :]
    P, _, _ = eval_psf_and_grad(gauss_psf, 0.0, 0.0, dix, diy, psf_scale)
    # P sums to ≈ 1 over the window (PSF is normalised so integral ≈ 1)
    assert abs(P.sum() - 1.0) < 0.05, f"PSF window sum = {P.sum():.4f}, expected ≈ 1"


# ---------------------------------------------------------------------------
# 2. Gradient correctness
# ---------------------------------------------------------------------------

def test_gradient_correctness(gauss_psf):
    """Analytical dP/dx from eval_psf_and_grad must agree with numerical FD."""
    psf_scale = PSF_SCALE
    hw = 2
    diy = np.arange(-hw, hw + 1)[:, np.newaxis]
    dix = np.arange(-hw, hw + 1)[np.newaxis, :]

    dx, dy = 0.2, -0.1
    _, dPdx_analytic, dPdy_analytic = eval_psf_and_grad(gauss_psf, dx, dy, dix, diy, psf_scale)

    eps = 1e-4  # detector pixels
    P_xp, _, _ = eval_psf_and_grad(gauss_psf, dx + eps, dy, dix, diy, psf_scale)
    P_xm, _, _ = eval_psf_and_grad(gauss_psf, dx - eps, dy, dix, diy, psf_scale)
    dPdx_num = (P_xp - P_xm) / (2 * eps)

    P_yp, _, _ = eval_psf_and_grad(gauss_psf, dx, dy + eps, dix, diy, psf_scale)
    P_ym, _, _ = eval_psf_and_grad(gauss_psf, dx, dy - eps, dix, diy, psf_scale)
    dPdy_num = (P_yp - P_ym) / (2 * eps)

    # Verify within 10% where gradient is significant (>10% of its peak value).
    # Bicubic spline introduces ~5-10% gradient error at the edges of the window;
    # the fitting tests confirm this is accurate enough for <0.05px recovery.
    for grad_a, grad_n, label in [
        (dPdx_analytic, dPdx_num, 'dP/dx'),
        (dPdy_analytic, dPdy_num, 'dP/dy'),
    ]:
        sig = np.abs(grad_n) > 0.10 * np.abs(grad_n).max()
        if sig.any():
            rel_err = np.abs(grad_a[sig] - grad_n[sig]) / (np.abs(grad_n[sig]) + 1e-30)
            assert rel_err.max() < 0.10, f"max {label} rel error = {rel_err.max():.4f}"


# ---------------------------------------------------------------------------
# 3. Position recovery
# ---------------------------------------------------------------------------

def test_position_recovery(gauss_psf, psf_cube, psf_positions):
    """Recovered position should be within 0.05 px of injected position."""
    x_true, y_true, flux_true = 127.37, 63.82, 5000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 256), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=round(x_true), y0=round(y_true),
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )

    assert abs(rec.x - x_true) < 0.05, f"x error = {abs(rec.x - x_true):.4f} px"
    assert abs(rec.y - y_true) < 0.05, f"y error = {abs(rec.y - y_true):.4f} px"


# ---------------------------------------------------------------------------
# 4. Flux recovery
# ---------------------------------------------------------------------------

def test_flux_recovery(gauss_psf, psf_cube, psf_positions):
    """Recovered flux should be within 5% of injected flux for S/N > 20."""
    x_true, y_true, flux_true = 100.0, 100.0, 10000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(200, 200), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )

    frac_err = abs(rec.flux - flux_true) / flux_true
    assert frac_err < 0.05, f"flux fractional error = {frac_err:.4f}"


# ---------------------------------------------------------------------------
# 5. Covariance positive-definite
# ---------------------------------------------------------------------------

def test_covariance_positive_definite(gauss_psf, psf_cube, psf_positions):
    x_true, y_true, flux_true = 80.5, 80.5, 5000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(160, 160), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )

    eigvals = np.linalg.eigvalsh(rec.cov)
    assert np.all(eigvals > 0), f"Covariance not positive definite: eigvals={eigvals}"


# ---------------------------------------------------------------------------
# 6. flux_err consistency with expected S/N
# ---------------------------------------------------------------------------

def test_flux_err_consistency(gauss_psf, psf_cube, psf_positions):
    """S/N from covariance should agree with Poisson S/N to within factor 2."""
    flux_true = 10000.0
    sky = 200.0
    x_true, y_true = 64.0, 64.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )

    sn_fit = rec.flux / rec.flux_err
    sn_poisson = np.sqrt(flux_true)  # simplified: photon noise on source
    # Allow factor 2 discrepancy
    assert 0.5 * sn_poisson < sn_fit < 2.0 * sn_poisson, \
        f"S/N mismatch: fit={sn_fit:.1f}, poisson={sn_poisson:.1f}"


# ---------------------------------------------------------------------------
# 7. Sky estimation
# ---------------------------------------------------------------------------

def test_sky_estimation():
    """Sky estimate from clean background should match injected sky within 1σ."""
    rng = np.random.default_rng(0)
    sky_true = 150.0
    data = rng.normal(sky_true, 5.0, size=(64, 64))
    sky, sky_sigma = estimate_sky(data, 32, 32, sky_inner=4, sky_outer=8)
    assert abs(sky - sky_true) < 3 * sky_sigma, \
        f"Sky estimate {sky:.2f} too far from true {sky_true}"


# ---------------------------------------------------------------------------
# 8. Edge pixels
# ---------------------------------------------------------------------------

def test_edge_pixels(gauss_psf, psf_cube, psf_positions):
    """Stars within hw pixels of the image boundary must not raise exceptions."""
    image = inject_stars([(1.5, 1.5, 3000.0)], gauss_psf, PSF_SCALE,
                         sky=200.0, image_size=(64, 64), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=1.5, y0=1.5,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=200.0, gain=4.0, read_noise=5.0,
        max_iter=5, tol=1e-4, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )
    assert isinstance(rec, StarRecord)


# ---------------------------------------------------------------------------
# 9. Source finding
# ---------------------------------------------------------------------------

def test_find_sources_detects_star(gauss_psf):
    """A bright injected star should be detected by find_sources."""
    x_true, y_true, flux_true = 64.0, 64.0, 5000.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=200.0, image_size=(128, 128), gain=4.0, read_noise=5.0)

    xs, ys, peaks, skys, sigs = find_sources(
        image, sky_inner=4, sky_outer=8, hmin=3, fmin=0.0)
    assert len(xs) >= 1
    # Brightest should be near injected position
    assert abs(xs[0] - x_true) < 1.5
    assert abs(ys[0] - y_true) < 1.5


# ---------------------------------------------------------------------------
# 10. Convergence tracking
# ---------------------------------------------------------------------------

def test_convergence_tracking(gauss_psf, psf_cube, psf_positions):
    """fit_star should report n_iter and converged on a well-behaved star."""
    x_true, y_true, flux_true = 64.0, 64.0, 8000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )
    assert rec.converged, "Bright isolated star should converge"
    assert 1 <= rec.n_iter <= 15, f"n_iter={rec.n_iter} out of range"


# ---------------------------------------------------------------------------
# 11. Sigma clipping handles cosmic rays
# ---------------------------------------------------------------------------

def test_sigma_clip_cosmic_ray(gauss_psf, psf_cube, psf_positions):
    """Sigma clipping should improve position accuracy when a cosmic ray is present."""
    x_true, y_true, flux_true = 64.0, 64.0, 8000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=4.0, read_noise=5.0)
    # Inject a bright cosmic ray 2 pixels from centre
    image[int(y_true), int(x_true) + 2] += 50000.0

    xs, ys = psf_positions
    kw = dict(psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
              hw=2, sky=sky, gain=4.0, read_noise=5.0,
              max_iter=10, tol=1e-5, noise_map=None, mask=None,
              x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1)

    rec_no  = fit_star(data=image, x0=x_true, y0=y_true, sigma_clip=False, **kw)
    rec_yes = fit_star(data=image, x0=x_true, y0=y_true, sigma_clip=True,  **kw)

    err_no  = abs(rec_no.x  - x_true)
    err_yes = abs(rec_yes.x - x_true)
    # Clipping should either improve or at least not worsen accuracy
    assert err_yes <= err_no + 0.02, \
        f"Sigma clip made position worse: no_clip={err_no:.4f}px, clip={err_yes:.4f}px"
    # And the clipped fit should be close to true position
    assert err_yes < 0.3, f"Clipped position error too large: {err_yes:.4f}px"


# ---------------------------------------------------------------------------
# 12. summarize_catalog
# ---------------------------------------------------------------------------

def test_summarize_catalog(gauss_psf, psf_cube, psf_positions):
    """summarize_catalog should return sane statistics without crashing."""
    x_true, y_true, flux_true = 64.0, 64.0, 5000.0
    sky = 200.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=4.0, read_noise=5.0)

    xs, ys = psf_positions
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=PSF_SCALE,
        hw=2, sky=sky, gain=4.0, read_noise=5.0,
        max_iter=10, tol=1e-5, noise_map=None, mask=None,
        x_offset=0.0, y_offset=0.0, zero_point=0.0, pass_number=1,
    )
    # Give neighbor stats (normally filled by run_photometry)
    rec.n_neighbors = 0; rec.dist_nearest = np.inf; rec.dist_nearest_brighter = np.inf

    stats = summarize_catalog([rec], verbose=False)
    assert stats['n_stars'] == 1
    assert np.isfinite(stats['qfit_median'])
    assert np.isfinite(stats['chi2_median'])
    assert stats['snr_median'] > 0
