"""Diagnostic plots and summary statistics for py1pass catalogues."""

import numpy as np
from .core import _conc_adaptive_bounds, _build_chi2_mag_bins


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_catalog(records, verbose=True, hw=3, floor_params=None):
    """Compute and print summary statistics for a list of StarRecord.

    Parameters
    ----------
    records : list of StarRecord
    verbose : bool — if True, print a formatted summary to stdout
    hw      : int  — fit half-width used; crowding threshold = hw pixels

    Returns
    -------
    dict with summary statistics (always returned regardless of verbose).
    """
    n = len(records)
    if n == 0:
        if verbose:
            print("Catalog is empty.")
        return {'n_stars': 0}

    qfit   = np.array([r.qfit  for r in records])
    chi2   = np.array([r.chi2  for r in records])
    mag    = np.array([r.mag   for r in records])
    flux   = np.array([r.flux  for r in records])
    ferr   = np.array([r.flux_err for r in records])
    snr    = np.where(np.isfinite(ferr) & (ferr > 0), flux / ferr, 0.0)

    finite_q = qfit[np.isfinite(qfit)]
    finite_c = chi2[np.isfinite(chi2)]
    finite_m = mag[np.isfinite(mag)]

    n_sat_any  = sum(r.n_sat > 0 for r in records)
    n_crowded  = sum(r.dist_nearest_brighter < hw for r in records)
    n_conv     = sum(getattr(r, 'converged', True) for r in records)

    passes = sorted({r.pass_number for r in records})
    per_pass = {p: sum(1 for r in records if r.pass_number == p) for p in passes}

    stats = {
        'n_stars':      n,
        'qfit_median':  float(np.median(finite_q)) if len(finite_q) else np.nan,
        'qfit_p90':     float(np.percentile(finite_q, 90)) if len(finite_q) else np.nan,
        'chi2_median':  float(np.median(finite_c)) if len(finite_c) else np.nan,
        'chi2_p90':     float(np.percentile(finite_c, 90)) if len(finite_c) else np.nan,
        'snr_median':   float(np.median(snr)),
        'mag_min':      float(np.min(finite_m)) if len(finite_m) else np.nan,
        'mag_max':      float(np.max(finite_m)) if len(finite_m) else np.nan,
        'n_sat_any':    n_sat_any,
        'frac_crowded': n_crowded / n,
        'n_converged':  n_conv,
        'per_pass':     per_pass,
    }

    if verbose:
        sep = '=' * 52
        print(f"\n{sep}")
        print(f"  Catalog summary  ({n} stars)")
        print(sep)
        if len(finite_m):
            print(f"  Magnitude range : {stats['mag_min']:.2f} → {stats['mag_max']:.2f}")
        print(f"  Pass breakdown  : " +
              ", ".join(f"pass {p}: {c}" for p, c in per_pass.items()))
        print(f"  qfit  : median={stats['qfit_median']:.3f}  "
              f"90th={stats['qfit_p90']:.3f}   (>0.5 = poor)")
        print(f"  chi²  : median={stats['chi2_median']:.3f}  "
              f"90th={stats['chi2_p90']:.3f}   (~1.0 = ideal)")
        print(f"  S/N   : median={stats['snr_median']:.1f}")
        print(f"  Saturated (n_sat > 0)        : "
              f"{n_sat_any} ({100*n_sat_any/n:.1f}%)")
        print(f"  Crowded (brighter nbr <{hw}px) : "
              f"{n_crowded} ({100*n_crowded/n:.1f}%)")
        n_nc = n - n_conv
        if n_nc:
            print(f"  Did not converge (max_iter)   : "
                  f"{n_nc} ({100*n_nc/n:.1f}%)")

        if floor_params is not None:
            fp = floor_params
            if not fp.get('fit_A_ok'):
                ok_tag = '(fit failed — using B as fallback)'
            elif fp.get('floor_A_hit_bound'):
                ok_tag = '(lower-bound limited — no turnover visible in data)'
            else:
                ok_tag = '(turnover detected)'
            print(f"\n  Systematic floor estimates")
            print(f"  Option A – σ=sqrt((A·10^(0.2m))²+floor²) {ok_tag}:")
            print(f"    σ_x_floor = {fp['sigma_x_floor_A']:.4f} px"
                  f"   σ_y_floor = {fp['sigma_y_floor_A']:.4f} px"
                  f"   ε_flux = {fp['eps_flux_A']:.5f}")
            print(f"  Option B – empirical bright-end median  ({fp['n_bright']} stars):")
            print(f"    σ_x_floor = {fp['sigma_x_floor_B']:.4f} px"
                  f"   σ_y_floor = {fp['sigma_y_floor_B']:.4f} px"
                  f"   ε_flux = {fp['eps_flux_B']:.5f}")

        print(sep)

    return stats


# ---------------------------------------------------------------------------
# Systematic floor estimation
# ---------------------------------------------------------------------------

def estimate_systematic_floor(records, bright_frac=0.15, min_stars_bright=20,
                               mag_bin_width=0.5):
    """Estimate pixel-constant position floor and fractional flux floor.

    Uses only converged, well-fit stars (qfit < 0.3, chi² < 3).

    Two complementary methods:

    **Option A** — nonlinear curve fit of the model
        σ = sqrt( (A · 10^(0.2·mag))² + floor² )
    to magnitude-binned median σ values. The fitted ``floor`` is the asymptotic
    uncertainty at infinite S/N.

    **Option B** — empirical: median σ of the brightest ``bright_frac``
    of well-fit stars.

    Parameters
    ----------
    records          : list of StarRecord (chi²-scaled covariances expected)
    bright_frac      : fraction of stars (by flux) used for Option B
    min_stars_bright : minimum bright-star count for Option B
    mag_bin_width    : magnitude bin width for Option A binning

    Returns
    -------
    dict or None (if too few stars)
        sigma_x_floor_A, sigma_y_floor_A, eps_flux_A  — Option A results
        sigma_x_floor_B, sigma_y_floor_B, eps_flux_B  — Option B results
        fit_A_ok    : True if curve_fit converged for all three quantities
        fit_curves  : dict('mag','sigma_x','sigma_y','f_frac') smooth model, or None
        bin_mag, bin_sx, bin_sy, bin_ff : binned medians used for fitting
        n_bright    : number of stars used for Option B
    """
    try:
        from scipy.optimize import curve_fit as _curve_fit
    except ImportError:
        return None

    # Strict filter: well-fit star candidates only — used for Option B.
    # Require n_conc_2x2 >= 3 so stars with heavily masked central pixels
    # (which have inflated position uncertainties) don't bias the floor estimate.
    valid_b = [r for r in records
               if (np.isfinite(r.mag) and np.isfinite(r.qfit)
                   and getattr(r, 'converged', True)
                   and getattr(r, 'is_star_candidate', r.qfit < 0.3)
                   and r.qfit < 0.3 and r.chi2 < 3.0 and r.flux > 1.1
                   and r.cov is not None
                   and getattr(r, 'n_conc_2x2', 4) >= 3)]
    if len(valid_b) < max(10, min_stars_bright):
        return None

    # Generous filter: star candidates with finite values — used for Option A
    # binning across the full magnitude range.  Median per bin is robust to the
    # minority of poor fits even when the qfit cut is relaxed.
    valid_a = [r for r in records
               if (np.isfinite(r.mag) and np.isfinite(r.qfit)
                   and getattr(r, 'converged', True)
                   and getattr(r, 'is_star_candidate', True)
                   and r.flux > 1.1 and r.cov is not None
                   and np.isfinite(r.cov[1, 1]) and np.isfinite(r.cov[2, 2])
                   and getattr(r, 'n_conc_2x2', 4) >= 3)]
    if len(valid_a) < 10:
        valid_a = valid_b   # fallback

    def _arrays(vlist):
        mag_ = np.array([r.mag for r in vlist])
        flux_ = np.array([r.flux for r in vlist])
        sx_  = np.array([np.sqrt(max(float(r.cov[1, 1]), 0.0)) for r in vlist])
        sy_  = np.array([np.sqrt(max(float(r.cov[2, 2]), 0.0)) for r in vlist])
        sf_  = np.array([np.sqrt(max(float(r.cov[0, 0]), 0.0)) for r in vlist])
        ff_  = sf_ / np.where(flux_ > 0, flux_, 1.0)
        return mag_, flux_, sx_, sy_, ff_

    mag_b, flux_b, sx_b, sy_b, ff_b = _arrays(valid_b)
    mag_a, _,      sx_a, sy_a, ff_a = _arrays(valid_a)

    # ---- Option B: empirical bright-end median (strict filter) --------------
    n_bright = min(max(min_stars_bright, int(bright_frac * len(valid_b))), len(valid_b))
    bright_idx = np.argsort(mag_b)[:n_bright]
    sigma_x_B  = float(np.median(sx_b[bright_idx]))
    sigma_y_B  = float(np.median(sy_b[bright_idx]))
    eps_flux_B = float(np.median(ff_b[bright_idx]))

    def _sc_median(vals, sigma_clip=3.5):
        """Sigma-clipped median: median → MAD → reject outliers → re-median.

        Two-pass approach: first pass gets a robust centre estimate; second
        pass rejects stars whose σ deviates by more than sigma_clip × MAD-σ.
        Falls back to the unclipped median if fewer than 3 inliers survive.
        """
        if len(vals) < 3:
            return float(np.median(vals))
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad < 1e-30:
            return med
        thresh = sigma_clip * mad * 1.4826
        keep = np.abs(vals - med) < thresh
        if keep.sum() < 3:
            return med
        return float(np.median(vals[keep]))

    # ---- Bin by magnitude for Option A fit (generous filter) ----------------
    chi2_a = np.array([r.chi2 for r in valid_a])
    mag_min, mag_max = float(np.min(mag_a)), float(np.max(mag_a))
    bin_edges = np.arange(mag_min, mag_max + mag_bin_width, mag_bin_width)
    bin_mag, bin_sx, bin_sy, bin_ff, bin_chi2 = [], [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (mag_a >= lo) & (mag_a < hi)
        if m.sum() < 3:
            continue
        bin_mag.append(float(np.median(mag_a[m])))
        # Sigma-clipped medians for the uncertainty quantities so that a small
        # number of stars with anomalously large σ (e.g. near bad columns or
        # borderline blended fits) do not bias the representative bin value.
        bin_sx.append(_sc_median(sx_a[m]))
        bin_sy.append(_sc_median(sy_a[m]))
        bin_ff.append(_sc_median(ff_a[m]))
        bin_chi2.append(float(np.median(chi2_a[m])))

    bin_mag  = np.array(bin_mag)
    bin_sx   = np.array(bin_sx)
    bin_sy   = np.array(bin_sy)
    bin_ff   = np.array(bin_ff)
    bin_chi2 = np.array(bin_chi2)

    # bin_mag is sorted bright→faint for negative-magnitude convention, but
    # np.arange goes low→high, so bin_mag[0] could be either end depending on
    # zero-point sign.  Sort ascending so index [0] = brightest = smallest value.
    _sort = np.argsort(bin_mag)
    bin_mag  = bin_mag[_sort]
    bin_sx   = bin_sx[_sort]
    bin_sy   = bin_sy[_sort]
    bin_ff   = bin_ff[_sort]
    bin_chi2 = bin_chi2[_sort]

    # ---- Saturation break detection ------------------------------------------
    # Saturated stars inflate both σ and chi² at the bright end.  Use chi² as
    # the primary saturation indicator: a bin is excluded from the fit only when
    # its median chi² is elevated AND its σ is elevated (i.e. the PSF fit is
    # genuinely failing, not just at the systematic floor).
    #
    # Strategy:
    #   1. Find the bright-end contiguous region where median chi² > chi2_sat_thresh.
    #   2. Require that σ is also elevated (> 1.2× the global σ minimum) so that
    #      mild chi² scatter at the bright floor doesn't trigger exclusion.
    #   3. Exclude only those bins from the fit; the model curve is always drawn
    #      from mag_min (absolute brightest bin) to mag_max for full coverage.
    _CHI2_SAT_THRESH = 2.5   # median chi² above which a bin is considered saturated
    _sx_sy_mean = (bin_sx + bin_sy) / 2.0
    _sigma_min  = float(np.min(_sx_sy_mean))
    # A bin is saturated: chi2 elevated AND sigma elevated.
    _sat_flag = (bin_chi2 > _CHI2_SAT_THRESH) & (_sx_sy_mean > _sigma_min * 1.20)
    i_sat_break = 0
    sat_mag_break = None
    # Only look at the bright end (first half of bins).
    n_half = max(1, len(bin_mag) // 2)
    if _sat_flag[:n_half].any():
        # Last (faintest) bin in the bright-end block that is still saturated.
        last_sat = int(np.where(_sat_flag[:n_half])[0][-1])
        i_sat_break   = last_sat + 1   # first non-saturated bin
        sat_mag_break = float(bin_mag[i_sat_break]) if i_sat_break < len(bin_mag) else None

    # Fit-only arrays: start at the saturation break (or bin 0 if none).
    _fit_mag = bin_mag[i_sat_break:]
    _fit_sx  = bin_sx[i_sat_break:]
    _fit_sy  = bin_sy[i_sat_break:]
    _fit_ff  = bin_ff[i_sat_break:]

    def _model(m, log10_floor, log10_A, log10_C):
        """Three-component noise model (all in log10 space → strictly positive):

          σ = sqrt( floor²  +  (A · 10^(0.2m))²  +  (C · 10^(0.4m))² )

        floor : systematic floor (PSF model errors etc.)
        A     : Poisson term — dominates at bright mags, slope 0.2 in log-log
        C     : background/read-noise term — dominates at faint mags, slope 0.4
        """
        return np.sqrt(
            (10.0 ** log10_floor) ** 2
            + (10.0 ** log10_A * 10.0 ** (0.2 * m)) ** 2
            + (10.0 ** log10_C * 10.0 ** (0.4 * m)) ** 2
        )

    def _model_log(m, log10_floor, log10_A, log10_C):
        """log10(_model) — used for fitting in log-space.

        Log-space fitting gives each magnitude bin equal weight per order of
        magnitude in σ.  Without this, the many faint bins (large absolute σ)
        dominate the least-squares residual and the optimizer sacrifices the
        2–3 bright floor-regime bins that anchor the systematic floor.
        """
        return np.log10(np.maximum(
            _model(m, log10_floor, log10_A, log10_C), 1e-12
        ))

    def _init_and_bounds(bsig, bmag):
        """Slope-aware init + data-driven bounds for the 3-component fit.

        Strategy:
        1. Compute log-log slopes (log10 sigma vs mag) between neighbouring bins.
           Apply a 3-point running mean to reduce bin-median noise.
        2. Background term C  : anchor slope-0.4 line on the bin-pair in the faint
           half whose smoothed slope is closest to 0.4.
        3. Poisson term A     : anchor slope-0.2 line on the bin-pair in the bright
           2/3 whose smoothed slope is closest to 0.2.
        4. Floor              : subtract the A and C contributions from the
           bright-end bins; take the median residual as the floor estimate.
        5. Bounds for floor   : if ≥2 consecutive floor-regime pairs (slope < 0.1)
           are visible, tighten the lower bound to init − 0.5 dex (the turnover IS
           in the data — don't let the optimizer escape to floor ≈ 0).  Otherwise
           use the loose init − 1.5 dex bound for the unresolved-floor case.
        """
        n = len(bsig)
        log_sig = np.log10(np.maximum(bsig, 1e-10))

        # Pairwise log-log slopes between adjacent bins
        dmag   = np.diff(bmag)
        dlog   = np.diff(log_sig)
        pslopes = np.where(np.abs(dmag) > 1e-6, dlog / dmag, np.nan)

        # Smooth with a 3-point equal-weight running mean
        n_pairs = len(pslopes)
        finite_s = np.where(np.isfinite(pslopes), pslopes, 0.0)
        if n_pairs >= 3:
            slp = np.convolve(finite_s, [1/3, 1/3, 1/3], mode='same')
        else:
            slp = finite_s

        # ---- estimate C (background term, target slope 0.4) -----------------
        # Search in the faint half of pairs.
        i_faint = max(n_pairs // 2, 1)
        i_C_pair = i_faint + int(np.argmin(np.abs(slp[i_faint:] - 0.4)))
        i_C = min(i_C_pair + 1, n - 1)
        log10_C0 = float(log_sig[i_C] - 0.4 * bmag[i_C])

        # ---- estimate A (Poisson term, target slope 0.2) --------------------
        # Search in the bright 2/3 of pairs.
        i_end_A = max(n_pairs * 2 // 3, 1)
        i_A_pair = int(np.argmin(np.abs(slp[:i_end_A] - 0.2)))
        i_A = min(i_A_pair + 1, n - 1)
        log10_A0 = float(log_sig[i_A] - 0.2 * bmag[i_A])

        # ---- estimate floor from residual at bright-end bins ----------------
        # floor² = max(bsig² − A_term² − C_term², (1e-3)²)
        n_br = min(max(3, n // 5), n)
        A_term = 10**log10_A0 * 10**(0.2 * bmag[:n_br])
        C_term = 10**log10_C0 * 10**(0.4 * bmag[:n_br])
        floor_sq = np.maximum(bsig[:n_br]**2 - A_term**2 - C_term**2, (1e-3)**2)
        log10_f0 = float(np.log10(np.sqrt(float(np.median(floor_sq)))))

        # ---- data-driven lower bound for floor ------------------------------
        # If the floor turnover is visible (≥2 floor-regime pairs with slope<0.1),
        # tighten the bound so the optimizer can't escape to floor ≈ 0.
        n_floor_pairs = int(np.sum(slp[:n_pairs // 2] < 0.1))
        if n_floor_pairs >= 2:
            lb_floor = max(log10_f0 - 0.5, -3.0)   # tight: floor IS in the data
        else:
            lb_floor = max(log10_f0 - 1.5, -3.0)   # loose: floor below bright-end σ

        p0     = [log10_f0, log10_A0, log10_C0]
        bounds = ([lb_floor, -12, -12], [1, 3, 3])
        return p0, bounds

    def _robust_fit(m_data, log_data, p0, bounds, min_bins=4, clip_sigma=3.5):
        """curve_fit in log-space, then one round of bin-level outlier rejection.

        After the initial fit, bins whose log-space residual exceeds
        clip_sigma × MAD-σ are discarded and the model is refit on the
        remaining inliers.  This handles pathological bins (e.g. a 0.5 mag
        window that happens to straddle a transition with very few stars, or
        a faint bin where a single poorly-fit neighbour inflates the median
        σ despite the sigma-clipped median).  The initial fit result is
        returned unchanged if fewer than min_bins would survive clipping.
        """
        popt, _ = _curve_fit(_model_log, m_data, log_data,
                              p0=p0, bounds=bounds, maxfev=10000)
        if len(m_data) <= min_bins + 1:
            return popt  # too few bins — clipping would overfit
        resid = log_data - _model_log(m_data, *popt)
        mad = float(np.median(np.abs(resid)))
        if mad < 1e-10:
            return popt
        thresh = clip_sigma * mad * 1.4826
        keep = np.abs(resid) < thresh
        if keep.sum() < min_bins:
            return popt  # too many outliers — trust the initial fit
        try:
            popt2, _ = _curve_fit(_model_log, m_data[keep], log_data[keep],
                                   p0=popt, bounds=bounds, maxfev=10000)
            return popt2
        except Exception:
            return popt

    fit_A_ok   = False
    sigma_x_A  = sigma_x_B
    sigma_y_A  = sigma_y_B
    eps_flux_A = eps_flux_B
    fit_curves = None
    popt_x = popt_y = popt_f = None

    if len(_fit_mag) >= 4:
        try:
            p0_x, bnd_x = _init_and_bounds(_fit_sx, _fit_mag)
            p0_y, bnd_y = _init_and_bounds(_fit_sy, _fit_mag)
            p0_f, bnd_f = _init_and_bounds(_fit_ff, _fit_mag)
            log_sx = np.log10(np.maximum(_fit_sx, 1e-12))
            log_sy = np.log10(np.maximum(_fit_sy, 1e-12))
            log_ff = np.log10(np.maximum(_fit_ff, 1e-12))
            popt_x = _robust_fit(_fit_mag, log_sx, p0_x, bnd_x)
            popt_y = _robust_fit(_fit_mag, log_sy, p0_y, bnd_y)
            popt_f = _robust_fit(_fit_mag, log_ff, p0_f, bnd_f)
            # Floor is parameter index 0.
            sigma_x_A  = float(10.0 ** popt_x[0])
            sigma_y_A  = float(10.0 ** popt_y[0])
            eps_flux_A = float(10.0 ** popt_f[0])
            fit_A_ok = True
            # Flag when the floor hit its lower bound (no visible turnover in data).
            floor_A_hit_bound = (sigma_x_A < 10.0 ** bnd_x[0][0] * 1.2
                                  or sigma_y_A < 10.0 ** bnd_y[0][0] * 1.2)
            # Draw the model curve over the full magnitude range (including any
            # bright-end bins excluded from the fit due to saturation).  The model
            # is well-defined everywhere and extrapolates cleanly; the saturation
            # break dashed line marks where the fit started.
            m_curve = np.linspace(mag_min, mag_max, 300)
            fit_curves = {
                'mag':             m_curve,
                'sigma_x':         _model(m_curve, *popt_x),
                'sigma_y':         _model(m_curve, *popt_y),
                'f_frac':          _model(m_curve, *popt_f),
                # Individual components for decomposed visualisation.
                'sigma_x_poisson': 10.0**popt_x[1] * 10.0**(0.2 * m_curve),
                'sigma_x_bg':      10.0**popt_x[2] * 10.0**(0.4 * m_curve),
                'sigma_y_poisson': 10.0**popt_y[1] * 10.0**(0.2 * m_curve),
                'sigma_y_bg':      10.0**popt_y[2] * 10.0**(0.4 * m_curve),
            }
        except Exception:
            pass

    if not fit_A_ok:
        floor_A_hit_bound = True

    return {
        'sigma_x_floor_A':  sigma_x_A,
        'sigma_y_floor_A':  sigma_y_A,
        'eps_flux_A':       eps_flux_A,
        'sigma_x_floor_B':  sigma_x_B,
        'sigma_y_floor_B':  sigma_y_B,
        'eps_flux_B':       eps_flux_B,
        'fit_A_ok':           fit_A_ok,
        'floor_A_hit_bound':  floor_A_hit_bound,
        'fit_curves':         fit_curves,
        'bin_mag':            bin_mag,
        'bin_sx':             bin_sx,
        'bin_sy':             bin_sy,
        'bin_ff':             bin_ff,
        'bin_chi2':           bin_chi2,
        'n_bright':           n_bright,
        # Raw fitted parameters (log10 units): [log10_floor, log10_A, log10_C]
        # None when fit_A_ok is False.
        'popt_x':  popt_x if fit_A_ok else None,
        'popt_y':  popt_y if fit_A_ok else None,
        'popt_f':  popt_f if fit_A_ok else None,
        # Saturation break: magnitude of the first bin included in the fit.
        # Bins brighter than this had elevated σ (saturation / PSF breakdown)
        # and were excluded from the curve fit.  None if no break was detected.
        'sat_mag_break': sat_mag_break,
    }


# ---------------------------------------------------------------------------
# Catalog statistics plot
# ---------------------------------------------------------------------------

def plot_catalog_stats(records, output=None, title=None, floor_params=None):
    """Statistical diagnostic plots for a py1pass catalog.

    Creates a multi-panel figure with:
      Row 1 : σ_x vs mag,  σ_y vs mag
      Row 2 : qfit vs mag,  chi² vs mag
      Row 3 : chi² histogram
      Row 4 : x pixel-phase vs x,  y pixel-phase vs y
      Row 5 : 2-D scatter of (x_phase, y_phase)

    Stars are colour-coded by chip (chip_ext attribute) so that per-chip
    sequences in σ_x/σ_y (caused by per-chip chi²-inflation floors) are
    immediately visible.

    Parameters
    ----------
    records : list of StarRecord
    output  : file path to save figure (None → don't save)
    title   : optional figure suptitle

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("matplotlib is required for plot_catalog_stats.")

    if not records:
        raise ValueError("No records to plot.")

    mag        = np.array([r.mag        for r in records])
    flux       = np.array([r.flux       for r in records])
    sx         = np.array([np.sqrt(max(r.cov[1,1], 0.0)) for r in records])
    sy         = np.array([np.sqrt(max(r.cov[2,2], 0.0)) for r in records])
    q          = np.array([r.qfit       for r in records])
    chi2       = np.array([r.chi2       for r in records])
    chi2_scale = np.array([getattr(r, 'chi2_scale', 1.0) for r in records])
    eps_psf    = np.array([getattr(r, 'eps_psf',    0.0) for r in records])
    conc       = np.array([getattr(r, 'concentration',      np.nan) for r in records])
    conc2      = np.array([getattr(r, 'concentration_2x2',  np.nan) for r in records])
    conc3      = np.array([getattr(r, 'concentration_3x3',  np.nan) for r in records])
    # Best available concentration (3×3 preferred)
    conc_best  = np.where(np.isfinite(conc3), conc3,
                 np.where(np.isfinite(conc2), conc2, conc))
    is_star    = np.array([getattr(r, 'is_star_candidate', True) for r in records],
                          dtype=bool)
    x          = np.array([r.x          for r in records])
    y          = np.array([r.y          for r in records])
    conv       = np.array([r.converged  for r in records])
    chip       = np.array([getattr(r, '_chip_ext', 0) for r in records])
    xph        = x % 1.0
    yph        = y % 1.0

    finite = np.isfinite(mag) & np.isfinite(sx) & np.isfinite(sy) \
             & np.isfinite(chi2) & np.isfinite(q)
    keep = (q < 2) & conv & (chi2 < 4) & (flux > 1.1)
    finite &= keep
    mag_f  = mag[finite];  sx_f  = sx[finite];   sy_f  = sy[finite]
    q_f    = q[finite];    chi2_f = chi2[finite]
    chi2_scale_f = chi2_scale[finite]
    eps_psf_f = eps_psf[finite]
    conc_f      = conc[finite]
    conc2_f     = conc2[finite]
    conc3_f     = conc3[finite]
    conc_best_f = conc_best[finite]
    is_star_f   = is_star[finite]
    x_f, y_f, xph_f, yph_f = x[finite], y[finite], xph[finite], yph[finite]
    chip_f = chip[finite]

    # Colour by chip so per-chip sequences are immediately visible.
    unique_chips = sorted(set(chip_f.tolist()))
    _chip_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson',
                    'mediumpurple', 'saddlebrown']
    chip_color = {c: _chip_colors[i % len(_chip_colors)]
                  for i, c in enumerate(unique_chips)}
    multi_chip = len(unique_chips) > 1

    def _scatter_by_chip(ax, xdata, ydata, **kw):
        for c in unique_chips:
            m = chip_f == c
            label = f'chip ext={c}' if multi_chip else None
            ax.scatter(xdata[m], ydata[m], c=chip_color[c], label=label, **kw)
        if multi_chip:
            ax.legend(fontsize=6, markerscale=4, loc='upper left')

    def _scatter_by_qfit(ax, xdata, ydata, **kw):
        sc = ax.scatter(xdata, ydata, c=q_f[:len(xdata)] if len(q_f) != len(xdata)
                        else q_f, cmap='plasma', vmin=0, vmax=0.5, **kw)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='qfit')

    def _scatter_grey_then_stars(ax, ydata, smask, **kw):
        """Non-stars in grey (background), stars coloured by qfit (foreground)."""
        ns = ~smask
        if ns.any():
            ax.scatter(mag_f[ns], ydata[ns], c='#bbbbbb', zorder=1,
                       s=kw.get('s', 2), alpha=kw.get('alpha', 0.3),
                       rasterized=kw.get('rasterized', True))
        if smask.any():
            sc = ax.scatter(mag_f[smask], ydata[smask], c=q_f[smask],
                            cmap='plasma', vmin=0, vmax=0.5, zorder=2, **kw)
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='qfit')

    # chi²_scale curve: recompute the exact pchip used by inflate_chi2 and
    # evaluate on a fine magnitude grid so the red line in the plot matches
    # what was actually applied to every star's covariance.
    def _chi2_scale_curve_pchip():
        """Return (mag_fine, chi2_fine, bin_mags, bin_raw, bin_unc, extrap_mags, extrap_chi2).

        Replicates inflate_chi2's binning logic so the plotted curve matches
        what was actually applied.  Returns empty arrays on failure.
        """
        _empty = (np.array([]),) * 7
        useful = [(r.mag, r.chi2) for r in records
                  if np.isfinite(r.chi2) and r.chi2 < 10.0 and r.flux > 1.1
                  and np.isfinite(r.mag)
                  and getattr(r, 'is_star_candidate', r.qfit < 2.0)]
        if len(useful) < 10:
            return _empty
        mags_u  = np.array([m for m, _ in useful])
        chi2s_u = np.array([c for _, c in useful])

        result = _build_chi2_mag_bins(mags_u, chi2s_u)
        if result is None:
            return _empty

        (bin_mags, bin_smooth, bin_raw, bin_unc,
         extrap_mags, extrap_chi2,
         extrap_faint_mags, extrap_faint_chi2) = result

        try:
            from scipy.interpolate import PchipInterpolator as _Pchip
        except ImportError:
            return _empty

        # Include bright- and faint-end anchor points so the curve extends
        # to the data edge without flat-clamping.
        pchip_mags = bin_mags
        pchip_chi2 = bin_smooth
        if extrap_mags.size:
            pchip_mags = np.concatenate([extrap_mags,       pchip_mags])
            pchip_chi2 = np.concatenate([extrap_chi2,       pchip_chi2])
        if extrap_faint_mags.size:
            pchip_mags = np.concatenate([pchip_mags, extrap_faint_mags])
            pchip_chi2 = np.concatenate([pchip_chi2, extrap_faint_chi2])

        pchip     = _Pchip(pchip_mags, pchip_chi2)
        mag_fine  = np.linspace(float(mags_u.min()), float(mags_u.max()), 300)
        chi2_fine = np.clip(pchip(mag_fine), 0.1, 20.0)
        # Clamp outside the anchor range
        chi2_fine = np.where(mag_fine <= pchip_mags[0],  float(pchip_chi2[0]),  chi2_fine)
        chi2_fine = np.where(mag_fine >= pchip_mags[-1], float(pchip_chi2[-1]), chi2_fine)

        return mag_fine, chi2_fine, bin_mags, bin_raw, bin_unc, extrap_mags, extrap_chi2

    if floor_params is None:
        floor_params = estimate_systematic_floor(records)

    fig = plt.figure(figsize=(14, 25), layout='constrained')
    gs  = gridspec.GridSpec(7, 2, figure=fig)

    kw_sc = dict(s=2, alpha=0.3, rasterized=True)
    _mag_lim = (np.nanmin(mag_f) - 0.5, np.nanmax(mag_f) + 0.5) if mag_f.size else (0, 1)
    _PANEL_BG = '0.90'

    # --- Row 0: position uncertainties vs mag --------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(_PANEL_BG)
    _scatter_grey_then_stars(ax1, sx_f, is_star_f, **kw_sc)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('σ_x  (pix)')
    ax1.set_title('X position uncertainty vs mag  (grey=non-star, colour=qfit)')
    ax1.set_xlim(_mag_lim)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-4, 5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(_PANEL_BG)
    _scatter_grey_then_stars(ax2, sy_f, is_star_f, **kw_sc)
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('σ_y  (pix)')
    ax2.set_title('Y position uncertainty vs mag  (grey=non-star, colour=qfit)')
    ax2.set_xlim(_mag_lim)
    ax2.set_yscale('log')
    ax2.set_ylim(5e-4, 5)

    # Overlay floor estimates on the position-uncertainty panels.
    if floor_params is not None:
        fp = floor_params
        sat_break = fp.get('sat_mag_break', None)
        for _ax, _key_A, _key_B, _bsy in (
                (ax1, 'sigma_x_floor_A', 'sigma_x_floor_B', 'bin_sx'),
                (ax2, 'sigma_y_floor_A', 'sigma_y_floor_B', 'bin_sy')):
            # Binned medians: excluded (saturated) bins drawn as gray ×.
            bm = fp.get('bin_mag', np.array([]))
            bs = fp.get(_bsy, np.array([]))
            if bm.size:
                if sat_break is not None:
                    excl = bm < sat_break
                    if excl.any():
                        _ax.plot(bm[excl], bs[excl], 'x', ms=5,
                                 color='gray', mew=1.2, zorder=6,
                                 label='excluded (saturated)')
                    _ax.plot(bm[~excl], bs[~excl], 'o', ms=4,
                             color='white', mec='black', mew=0.6, zorder=6,
                             label='bin median (fitted)')
                else:
                    _ax.plot(bm, bs, 'o', ms=4, color='white', mec='black',
                             mew=0.6, zorder=6, label='bin median')
            # Vertical dashed line at the saturation break.
            if sat_break is not None:
                _ax.axvline(sat_break, color='gray', lw=1.0, ls='--',
                            zorder=5, label=f'sat. break  m={sat_break:.2f}')
            # Option A fitted curve with decomposed components.
            fc = fp.get('fit_curves')
            if fc is not None:
                _sk  = 'sigma_x' if 'x' in _key_A else 'sigma_y'
                _ax.plot(fc['mag'], fc[_sk], '-', color='tomato', lw=1.8,
                         zorder=7, label=f"fit total (floor={fp[_key_A]:.4f} px)")
                if f'{_sk}_poisson' in fc:
                    _ax.plot(fc['mag'], fc[f'{_sk}_poisson'], '--',
                             color='tomato', lw=1.0, alpha=0.6, zorder=6,
                             label='Poisson  (slope 0.2)')
                if f'{_sk}_bg' in fc:
                    _ax.plot(fc['mag'], fc[f'{_sk}_bg'], ':',
                             color='tomato', lw=1.0, alpha=0.6, zorder=6,
                             label='background  (slope 0.4)')
            # Floor lines.
            _ax.axhline(fp[_key_A], color='tomato', lw=1.2, ls='--', zorder=8,
                        label=f"A floor {fp[_key_A]:.4f} px")
            _ax.axhline(fp[_key_B], color='deepskyblue', lw=1.2, ls=':', zorder=8,
                        label=f"B floor {fp[_key_B]:.4f} px")
            _ax.legend(fontsize=6, markerscale=3, loc='upper left')

    # --- Row 1: qfit and raw chi² -------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(_PANEL_BG)
    # Show star candidates (blue) and non-stars (orange) separately so the
    # two populations (stellar locus vs galaxies/CRs) are immediately visible.
    _kw_nstar = dict(s=3, alpha=0.5, rasterized=True)
    _star_m  = is_star_f
    _nstar_m = ~is_star_f
    if _nstar_m.any():
        ax3.scatter(mag_f[_nstar_m], q_f[_nstar_m],
                    c='darkorange', marker='x',
                    label=f'non-stars ({_nstar_m.sum()})', **_kw_nstar)
    if _star_m.any():
        ax3.scatter(mag_f[_star_m],  q_f[_star_m],
                    c='steelblue', label=f'stars ({_star_m.sum()})', **_kw_nstar)
    ax3.axhline(0.1, color='green', lw=0.8, ls='--', label='excellent (0.1)')
    ax3.axhline(0.5, color='red',   lw=0.8, ls='--', label='poor (0.5)')
    ax3.set_xlabel('Magnitude')
    ax3.set_ylabel('qfit')
    ax3.set_title('Quality of fit vs mag  (blue=star, orange=non-star)')
    ax3.set_yscale('log')
    ax3.set_xlim(_mag_lim)
    ax3.legend(fontsize=7, markerscale=3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(_PANEL_BG)
    _scatter_grey_then_stars(ax4, chi2_f, is_star_f, **kw_sc)
    _bm, _bc, _bin_mags, _bin_raw, _bin_unc, _ex_mags, _ex_chi2 = _chi2_scale_curve_pchip()
    if _bm.size:
        ax4.plot(_bm, _bc, 'r-', lw=1.8, zorder=5, label='chi²_scale curve')
        ax4.errorbar(_bin_mags, _bin_raw, yerr=_bin_unc,
                     fmt='rs', ms=3, lw=0.8, zorder=6, label='bin medians ± SEM')
        if _ex_mags.size:
            ax4.plot(_ex_mags, _ex_chi2, marker='^', ms=5, lw=0.8,
                     color='tomato', ls='--', zorder=6, label='extrapolated')
    ax4.axhline(1.0, color='green', lw=0.8, ls='--', label='ideal (1.0)')
    ax4.set_xlabel('Magnitude')
    ax4.set_ylabel('chi²')
    ax4.set_title('Raw chi² vs mag  (grey=non-star, coloured=star, red=correction curve)')
    ax4.set_xlim(_mag_lim)
    _chi2_fin = chi2_f[np.isfinite(chi2_f)]
    _chi2_top = float(np.percentile(_chi2_fin, 99)) if _chi2_fin.size else 5.0
    ax4.set_ylim(0, _chi2_top * 1.5)
    ax4.legend(fontsize=7, markerscale=3)

    # --- Row 2: chi²_scale vs mag and chi² histogram -------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(_PANEL_BG)
    _scatter_grey_then_stars(ax5, chi2_scale_f, is_star_f, **kw_sc)
    if _bm.size:
        ax5.plot(_bm, _bc, 'r-', lw=1.8, zorder=5)
    ax5.axhline(1.0, color='green', lw=0.8, ls='--', label='no scaling (1.0)')
    ax5.set_xlabel('Magnitude')
    ax5.set_ylabel('chi²_scale')
    ax5.set_title('Applied chi²-scaling vs mag  (grey=non-star)')
    ax5.set_xlim(_mag_lim)
    _cs_fin = chi2_scale_f[np.isfinite(chi2_scale_f)]
    if _cs_fin.size:
        _cs_lo = max(float(np.percentile(_cs_fin, 1)) * 0.9, 0.1)
        _cs_hi = max(float(np.percentile(_cs_fin, 99)),
                     float(np.max(_bc)) if _bm.size else 0.0) * 1.1
        ax5.set_ylim(_cs_lo, _cs_hi)
    ax5.legend(fontsize=7, markerscale=3)

    # Compute scaled chi² once; used in both the histogram and the scatter below.
    _ratio = chi2_f / np.where(chi2_scale_f > 0, chi2_scale_f, 1.0)

    ax6 = fig.add_subplot(gs[2, 1])
    _chi2_plot  = chi2_f[is_star_f & np.isfinite(chi2_f) & (chi2_f < 10)]
    _ratio_plot = _ratio[is_star_f & np.isfinite(_ratio)  & (_ratio  < 10)]
    if _chi2_plot.size:
        # Shared bin edges so the two histograms are directly comparable.
        _bin_max = max(float(np.percentile(_chi2_plot, 99)),
                       float(np.percentile(_ratio_plot, 99)) if _ratio_plot.size else 0,
                       2.0)
        _bins = np.linspace(0, _bin_max, 80)
        ax6.hist(_chi2_plot, bins=_bins, color='steelblue', alpha=0.6,
                 density=True, label='raw chi²')
        _chi2_med = float(np.median(_chi2_plot))
        ax6.axvline(_chi2_med, color='steelblue', lw=1.0, ls=':',
                    label=f'raw median ({_chi2_med:.2f})')
    if _ratio_plot.size:
        ax6.hist(_ratio_plot, bins=_bins if _chi2_plot.size else 80,
                 color='darkorange', alpha=0.6,
                 density=True, label='scaled chi²  (÷ chi²_scale)')
        _ratio_med = float(np.median(_ratio_plot))
        ax6.axvline(_ratio_med, color='darkorange', lw=1.0, ls=':',
                    label=f'scaled median ({_ratio_med:.2f})')
    ax6.axvline(1.0, color='green', lw=1.2, ls='--', label='ideal (1.0)')
    ax6.set_xlabel('Reduced chi²')
    ax6.set_ylabel('Density')
    ax6.set_title('chi² distribution  (stars only; orange = after chi²-scaling)')
    ax6.legend(fontsize=7)

    # --- Row 3: chi²_scale residual (left) + concentration vs mag (right) ---
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.set_facecolor(_PANEL_BG)
    _scatter_grey_then_stars(ax7, _ratio, is_star_f, **kw_sc)
    ax7.axhline(1.0, color='red', lw=1.0, ls='--', label='ratio = 1 (ideal)')
    ax7.set_xlabel('Magnitude')
    ax7.set_ylabel('raw chi² / chi²_scale')
    ax7.set_title('Residual after scaling  (grey=non-star; should be flat at 1)')
    ax7.set_xlim(_mag_lim)
    _r_fin = _ratio[np.isfinite(_ratio)]
    if _r_fin.size:
        ax7.set_ylim(max(0, float(np.percentile(_r_fin, 1)) * 0.8),
                     min(float(np.percentile(_r_fin, 99)) * 1.2, 5.0))
    ax7.legend(fontsize=7, markerscale=3)

    # Concentration panel: 2×2 metric, coloured by is_star_candidate.
    ax_conc = fig.add_subplot(gs[3, 1])
    ax_conc.set_facecolor(_PANEL_BG)
    _ckw_s = dict(s=2, alpha=0.35, rasterized=True)
    _c2_fin = np.isfinite(conc2_f)
    if _c2_fin.any():
        _sm = is_star_f
        _nm = ~is_star_f
        if (_nm & _c2_fin).any():
            ax_conc.scatter(mag_f[_nm & _c2_fin], conc2_f[_nm & _c2_fin],
                            c='darkorange', marker='x', label='non-stars', **_ckw_s)
        if (_sm & _c2_fin).any():
            ax_conc.scatter(mag_f[_sm & _c2_fin], conc2_f[_sm & _c2_fin],
                            c='steelblue', label='stars', **_ckw_s)
        ax_conc.axhline(1.0, color='green', lw=1.0, ls='--', label='ideal (1.0)')
        _c_vals = conc2_f[_c2_fin]
        ax_conc.set_ylim(max(float(np.percentile(_c_vals, 1)) * 0.5, 0.05),
                         min(float(np.percentile(_c_vals, 99)) * 2.0, 6.0))
    ax_conc.set_xlabel('Magnitude')
    ax_conc.set_ylabel('Concentration (2×2)')
    ax_conc.set_title('Concentration (2×2) vs mag  (blue=star, orange=non-star)')
    ax_conc.set_xlim(_mag_lim)
    ax_conc.legend(fontsize=6, markerscale=3)

    # --- Row 4: ε_PSF (left) + qfit coloured by concentration (right) --------
    ax_eps = fig.add_subplot(gs[4, 0])
    ax_eps.set_facecolor(_PANEL_BG)
    _eps_fin = np.isfinite(eps_psf_f) & (eps_psf_f > 0)
    if _eps_fin.sum() > 0:
        _ns_eps = _eps_fin & ~is_star_f
        _ss_eps = _eps_fin &  is_star_f
        if _ns_eps.any():
            ax_eps.scatter(mag_f[_ns_eps], eps_psf_f[_ns_eps], c='#bbbbbb',
                           zorder=1, **kw_sc)
        if _ss_eps.any():
            sc_eps = ax_eps.scatter(mag_f[_ss_eps], eps_psf_f[_ss_eps],
                                    c=q_f[_ss_eps], cmap='plasma',
                                    vmin=0, vmax=0.5, zorder=2, **kw_sc)
            fig.colorbar(sc_eps, ax=ax_eps, fraction=0.03, pad=0.02, label='qfit')
        # Running median — stars only, equal-width magnitude bins so the bright
        # end (few stars) gets its own bins rather than being averaged into a
        # large equal-count bin that starts at a moderately-bright magnitude.
        _ms_e = mag_f[_ss_eps];  _es_e = eps_psf_f[_ss_eps]
        if _ms_e.size >= 3:
            _bin_edges_e = np.arange(float(_ms_e.min()), float(_ms_e.max()) + 0.5, 0.5)
            _bm_e, _be_e = [], []
            for _lo_e, _hi_e in zip(_bin_edges_e[:-1], _bin_edges_e[1:]):
                _m_e = (_ms_e >= _lo_e) & (_ms_e < _hi_e)
                if _m_e.sum() >= 3:
                    _bm_e.append(float(np.median(_ms_e[_m_e])))
                    _be_e.append(float(np.median(_es_e[_m_e])))
            if _bm_e:
                ax_eps.plot(np.array(_bm_e), np.array(_be_e), 'r-', lw=1.8, zorder=5,
                            label='bin median  (0.5 mag bins, stars only)')
        _eps_vals = eps_psf_f[_eps_fin]
        # Show full data range including the bright-star floor — don't clip to
        # 1st percentile because those are the scientifically interesting points.
        ax_eps.set_yscale('log')
        _eps_lo = max(float(np.nanmin(_eps_vals)) * 0.7, 1e-5)
        _eps_hi = float(np.percentile(_eps_vals, 99.5)) * 2.0
        ax_eps.set_ylim(_eps_lo, _eps_hi)
    ax_eps.set_xlabel('Magnitude')
    ax_eps.set_ylabel('ε_PSF')
    ax_eps.set_title('PSF model error  ε = chi² / √(flux · psf_frac · gain)  (grey=non-star)')
    ax_eps.set_xlim(_mag_lim)
    ax_eps.legend(fontsize=7, markerscale=3)

    # qfit vs mag coloured by 2×2 concentration — reveals the two populations:
    # the stellar locus (conc ≈ 1, low qfit) vs extended sources (conc ≠ 1, high qfit).
    ax_qconc = fig.add_subplot(gs[4, 1])
    ax_qconc.set_facecolor(_PANEL_BG)
    _qc_fin = np.isfinite(conc2_f) & np.isfinite(q_f)
    if _qc_fin.any():
        _qc_vals = np.clip(conc2_f[_qc_fin], 0.3, 2.5)
        import matplotlib.cm as _mcm
        import matplotlib.colors as _mco
        _cmap_qc = _mcm.get_cmap('RdYlGn_r')
        _norm_qc = _mco.TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=2.5)
        sc_qc = ax_qconc.scatter(mag_f[_qc_fin], q_f[_qc_fin],
                                 c=_qc_vals, cmap=_cmap_qc, norm=_norm_qc,
                                 s=2, alpha=0.35, rasterized=True)
        fig.colorbar(sc_qc, ax=ax_qconc, fraction=0.03, pad=0.02,
                     label='concentration (2×2)')
        ax_qconc.axhline(0.1, color='green', lw=0.8, ls='--', label='excellent (0.1)')
        ax_qconc.axhline(0.5, color='red',   lw=0.8, ls='--', label='poor (0.5)')
    ax_qconc.set_xlabel('Magnitude')
    ax_qconc.set_ylabel('qfit')
    ax_qconc.set_title('qfit vs mag  (colour = concentration 2×2;  green=1.0=star)')
    ax_qconc.set_yscale('log')
    ax_qconc.set_xlim(_mag_lim)
    ax_qconc.legend(fontsize=7, markerscale=3)

    # --- Rows 5–6: 2-D pixel phase scatter -----------------------------------
    ax8 = fig.add_subplot(gs[5:, :])
    # if xph_f.size > 500 and not multi_chip:
    if xph_f.size > 500:
        h, xe, ye = np.histogram2d(xph_f, yph_f, bins=50,
                                    range=[[0, 1], [0, 1]])
        im = ax8.imshow(h.T, origin='lower', extent=[0, 1, 0, 1],
                        aspect='equal', cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax8, fraction=0.02, pad=0.01, label='Count')
        ax8.set_title('2-D pixel phase distribution  (2D histogram)')
    else:
        ax8.set_facecolor(_PANEL_BG)
        _scatter_by_qfit(ax8, xph_f, yph_f, **kw_sc)
        ax8.set_title('2-D pixel phase distribution  (colour = qfit)')
    ax8.set_xlabel('X phase  (x mod 1)')
    ax8.set_ylabel('Y phase  (y mod 1)')
    ax8.set_xlim(-0.02, 1.02)
    ax8.set_ylim(-0.02, 1.02)
    ax8.set_aspect('equal')

    if title:
        fig.suptitle(title, fontsize=11)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Concentration diagnostic plot
# ---------------------------------------------------------------------------

def plot_concentration_diagnostics(records, output=None, title=None,
                                   conc_limit=0.9, mag_bin_width=0.5,
                                   min_bin_stars=5,
                                   conc_width_factor=4.0, conc_min_width=0.01):
    """Three-panel plot showing 1×1, 2×2, and 3×3 concentration vs magnitude.

    For each panel:
    - Small scatter points for star candidates (blue) and non-stars (orange ×).
    - Binned median (solid line) and 68% region (16th–84th percentile, shaded)
      computed from star candidates only.
    - Horizontal reference lines at 1.0 (ideal) and conc_limit / (1/conc_limit)
      (classification boundaries).

    Parameters
    ----------
    records    : list of StarRecord
    output     : file path to save figure, or None to return the Figure.
    title      : optional suptitle string
    conc_limit : concentration lower boundary (upper = 1/conc_limit)
    mag_bin_width : magnitude bin width for running statistics
    min_bin_stars : minimum star candidates per bin to plot statistics

    Returns
    -------
    matplotlib Figure if output is None, else None.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not records:
        fig, _ = plt.subplots(3, 1, figsize=(8, 12))
        if output:
            fig.savefig(output, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig

    mag  = np.array([r.mag for r in records])
    star = np.array([getattr(r, 'is_star_candidate', True) for r in records], dtype=bool)
    concs = {
        '1×1': np.array([getattr(r, 'concentration',      np.nan) for r in records]),
        '2×2': np.array([getattr(r, 'concentration_2x2',  np.nan) for r in records]),
        '3×3': np.array([getattr(r, 'concentration_3x3',  np.nan) for r in records]),
    }

    conc_lo = conc_limit
    conc_hi = 1.0 / conc_limit

    # Magnitude bins — span finite, classified-star mags.
    star_mags = mag[star & np.isfinite(mag)]
    if star_mags.size:
        mag_min = np.floor(star_mags.min() / mag_bin_width) * mag_bin_width
        mag_max = np.ceil(star_mags.max()  / mag_bin_width) * mag_bin_width
    else:
        mag_min, mag_max = mag[np.isfinite(mag)].min(), mag[np.isfinite(mag)].max()
    bin_edges  = np.arange(mag_min, mag_max + mag_bin_width, mag_bin_width)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    fig.subplots_adjust(hspace=0.08, top=0.93)

    # Y-axis limits: a bit beyond the classification boundaries, centred on 1.
    y_lo = max(0.0, conc_lo - 0.3)
    y_hi = conc_hi + 0.4

    alpha_pt = 0.15
    s_pt = 2

    for ax, (label, conc) in zip(axes, concs.items()):
        finite = np.isfinite(conc)

        # Scatter: non-stars first (underneath), then stars on top.
        ns_mask = ~star & finite
        s_mask  =  star & finite
        ax.scatter(mag[ns_mask], conc[ns_mask], s=s_pt, c='#E07000', alpha=alpha_pt,
                   marker='x', linewidths=0.4, rasterized=True, label='Non-star')
        ax.scatter(mag[s_mask],  conc[s_mask],  s=s_pt, c='#2060C0', alpha=alpha_pt,
                   marker='o', linewidths=0, rasterized=True, label='Star candidate')

        # Binned statistics from final star candidates → 68% band and median.
        # Hard outer limits exclude extreme outliers from biasing the locus.
        _hard_ok = (conc >= 0.5) & (conc <= 2.0)
        bm   = np.full(len(bin_centres), np.nan)
        bp16 = np.full(len(bin_centres), np.nan)
        bp84 = np.full(len(bin_centres), np.nan)
        for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = s_mask & _hard_ok & (mag >= lo) & (mag < hi)
            if mask.sum() >= min_bin_stars:
                v = conc[mask]
                bm[k]   = np.median(v)
                bp16[k] = np.percentile(v, 16)
                bp84[k] = np.percentile(v, 84)

        ok = np.isfinite(bm)
        if ok.any():
            ax.fill_between(bin_centres[ok], bp16[ok], bp84[ok],
                            color='red', alpha=0.20, label='Star 16–84%')
            ax.plot(bin_centres[ok], bm[ok], color='red', lw=1.5,
                    zorder=5, label='Star median')

        # Adaptive classification bounds — computed from the same is_star_candidate
        # set as the 68% band above.  Because classify_stars iterates to convergence,
        # the final star set is self-consistent: these bounds are exactly the ones
        # that selected those stars.
        if ok.any():
            _hw = np.maximum((bp84[ok] - bp16[ok]) / 2, conc_min_width)
            _adp_lo = bm[ok] - conc_width_factor * _hw
            _adp_hi = bm[ok] + conc_width_factor * _hw
            ax.plot(bin_centres[ok], _adp_lo, color='red', lw=1.0, ls='--',
                    zorder=5, label=f'Adaptive ±{conc_width_factor:.0f}σ bounds')
            ax.plot(bin_centres[ok], _adp_hi, color='red', lw=1.0, ls='--',
                    zorder=5)

        # Reference lines: initial fixed window (grey) and ideal star value.
        ax.axhline(1.0,     color='k',       lw=0.8, ls='-',  zorder=3)
        ax.axhline(conc_lo, color='#888888', lw=0.8, ls=':',  zorder=3)
        ax.axhline(conc_hi, color='#888888', lw=0.8, ls=':',  zorder=3)

        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(f'Concentration ({label})', fontsize=10)
        # Bright (more negative mag) on left — conventional CMD direction.
        ax.set_xlim(mag[np.isfinite(mag)].min() - 0.5,
                    mag[np.isfinite(mag)].max() + 0.5)

        # Annotation: count of stars / total in this panel.
        n_s  = int((s_mask  & finite).sum())
        n_ns = int((ns_mask & finite).sum())
        ax.text(0.02, 0.97, f'stars={n_s}  non-stars={n_ns}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8, color='#444444')

    axes[-1].set_xlabel('Magnitude', fontsize=10)

    # Shared legend on top panel.
    handles = [
        mpatches.Patch(color='#2060C0', alpha=0.7, label='Star candidate'),
        mpatches.Patch(color='#E07000', alpha=0.7, label='Non-star'),
        plt.Line2D([0], [0], color='red',     lw=1.5,          label='Star median'),
        plt.Line2D([0], [0], color='red', lw=6, alpha=0.20, label='Star 16–84%'),
        plt.Line2D([0], [0], color='red',     lw=1.0, ls='--',
                   label=f'Adaptive ±{conc_width_factor:.0f}σ bounds'),
        plt.Line2D([0], [0], color='#888888', lw=0.8, ls=':',
                   label=f'Initial seed window ({conc_lo:.2f} / {conc_hi:.2f})'),
    ]
    axes[0].legend(handles=handles, loc='upper right', fontsize=8,
                   framealpha=0.85, ncol=2)

    suptitle = 'Concentration vs magnitude'
    if title:
        suptitle = f'{title}  —  {suptitle}'
    fig.suptitle(suptitle, fontsize=11)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# PSF residual map
# ---------------------------------------------------------------------------

def plot_psf_residual_map(records, data, psf_cube, xs, ys, psf_scale, hw,
                           n_grid=4, output=None, title=None,
                           x_offset=0.0, y_offset=0.0,
                           min_stars=5, gain=1.0, read_noise=5.0,
                           noise_map=None):
    """Show average fractional PSF residuals across a spatial grid of the detector.

    Divides the detector into n_grid × n_grid regions.  For each region with
    ≥ min_stars well-fit stars (qfit < 0.5, converged), stacks the inverse-
    variance-weighted normalised residuals:

        r_norm_k = (data_k - sky - flux·P_k) / (flux · psf_peak)

    so the result is a fractional deviation from the PSF model, independent of
    source brightness.  The mean is displayed as a small image for each tile.

    Parameters
    ----------
    records    : list of StarRecord
    data       : 2D science image (or the star-subtracted residual + this-star model)
    psf_cube   : raw PSF cube
    xs, ys     : PSF grid detector coordinates
    psf_scale  : PSF supersampling factor
    hw         : fit half-width (same as used during photometry)
    n_grid     : number of grid divisions per axis (total n_grid² tiles)
    output     : save path (None = don't save)
    title      : optional figure suptitle
    x_offset, y_offset : detector coordinate offsets
    min_stars  : minimum number of stars per tile to display (grey if fewer)
    gain       : noise model gain (same as run_photometry)
    read_noise : noise model read noise in e-
    noise_map  : optional external variance map

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError("matplotlib is required for plot_psf_residual_map.")

    from scipy.ndimage import spline_filter
    from .core import interpolate_psf, _eval_psf_grad_fast, _window_offsets

    ny, nx = data.shape
    win = 2 * hw + 1  # fit window size in detector pixels

    # Prefilter PSF cube once
    psf_coeffs_cube = np.array([
        spline_filter(p, order=3, output=np.float64) for p in psf_cube
    ])

    # Define tile boundaries
    x_edges = np.linspace(0, nx, n_grid + 1)
    y_edges = np.linspace(0, ny, n_grid + 1)

    # Accumulate weighted residuals per tile: shape (n_grid, n_grid, win, win)
    sum_wr  = np.zeros((n_grid, n_grid, win, win))
    sum_w   = np.zeros((n_grid, n_grid, win, win))
    n_stars = np.zeros((n_grid, n_grid), dtype=int)

    good_recs = [r for r in records
                 if r.qfit < 0.1 and getattr(r, 'converged', True) and r.chi2 < 4
                 and r.flux > 1.1 and np.isfinite(r.mag)
                 and getattr(r, 'is_star_candidate', True)]

    for rec in good_recs:
        # Which tile does this star fall in?
        ix_tile = int(np.clip(np.searchsorted(x_edges[1:], rec.x), 0, n_grid - 1))
        iy_tile = int(np.clip(np.searchsorted(y_edges[1:], rec.y), 0, n_grid - 1))

        xi = int(round(rec.x)); yi = int(round(rec.y))
        dx = rec.x - xi;        dy = rec.y - yi

        y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)
        if (y_hi - y_lo) != win or (x_hi - x_lo) != win:
            continue  # edge star: window clipped, skip to avoid shape mismatch

        d_stamp = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

        local_psf = interpolate_psf(psf_coeffs_cube, xs, ys,
                                     rec.x + x_offset, rec.y + y_offset)
        P, _, _ = _eval_psf_grad_fast(local_psf, dx, dy, dix, diy, psf_scale)

        r_stamp = d_stamp - rec.sky - rec.flux * P   # fit residual

        if noise_map is not None:
            var_stamp = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        else:
            var_stamp = (np.maximum(rec.flux * P, 0.0) + max(rec.sky, 0.0)) / gain \
                        + (read_noise / gain) ** 2
        var_stamp = np.maximum(var_stamp, 1e-10)

        # Fractional residual: normalise by expected PSF amplitude at each pixel
        # Use flux · psf_peak as the normalisation so dim stars count equally
        norm = max(rec.flux * rec.psf_peak, 1e-10)
        r_norm = r_stamp / norm

        # Apply sigma-clipping mask if available
        clip_m = getattr(rec, 'clipped_mask', None)
        bad = np.zeros(d_stamp.shape, dtype=bool)
        if clip_m is not None and clip_m.shape == bad.shape:
            bad |= clip_m

        w = np.where(bad, 0.0, 1.0 / var_stamp)

        sum_wr[iy_tile, ix_tile] += w * r_norm
        sum_w [iy_tile, ix_tile] += w
        n_stars[iy_tile, ix_tile] += 1

    # Average residual per tile (NaN where no weight)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_r = np.where(sum_w > 0, sum_wr / sum_w, np.nan)

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(n_grid * 2.2, n_grid * 2.2),
                             layout='constrained')
    if n_grid == 1:
        axes = np.array([[axes]])

    # Symmetric colour scale across all tiles
    vals = avg_r[np.isfinite(avg_r)]
    vlim = float(np.percentile(np.abs(vals), 98)) if vals.size else 0.05
    vlim = max(vlim, 1e-6)
    norm_r = Normalize(vmin=-vlim, vmax=vlim)

    kw_im = dict(origin='lower', interpolation='nearest', aspect='equal',
                 cmap='RdBu_r', norm=norm_r)

    for iy in range(n_grid):
        for ix in range(n_grid):
            ax = axes[n_grid - 1 - iy, ix]  # y=0 at bottom
            n = n_stars[iy, ix]
            tile_r = avg_r[iy, ix]
            if n >= min_stars and np.isfinite(tile_r).any():
                im = ax.imshow(tile_r, **kw_im)
                ax.set_title(f'N={n}', fontsize=7, pad=1)
            else:
                ax.set_facecolor('#cccccc')
                ax.text(0.5, 0.5, f'N={n}\n(insuf.)', ha='center', va='center',
                        fontsize=7, transform=ax.transAxes, color='#555555')
            # Fit window border
            ax.add_patch(Rectangle((-0.5, -0.5), win, win,
                                   fill=False, edgecolor='cyan',
                                   linewidth=0.5, linestyle='--'))
            ax.set_xticks([]); ax.set_yticks([])

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm_r, cmap='RdBu_r')
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02,
                 label='Fractional PSF residual  (data−model) / (flux·PSF_peak)')

    if title:
        fig.suptitle(title, fontsize=10)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Postage stamp diagnostic plot
# ---------------------------------------------------------------------------

def plot_diagnostics(records, data, psf_cube, xs, ys, psf_scale, hw,
                     output=None, n_stamps=16, title=None,
                     x_offset=0.0, y_offset=0.0,
                     stamp_pad=2, cmap_data='gray', cmap_res='RdBu_r',
                     residual=None, mask=None, noise_map=None):
    """Create a postage-stamp diagnostic figure: data / PSF model / residual.

    Selects a representative sample of stars spanning the full magnitude range
    and image area.  Well-fit stars (qfit < 0.3) are preferred; a small
    fraction of poorly-fit stars are included for diagnostic interest.

    Parameters
    ----------
    records   : list of StarRecord
    data      : 2D science image array (same array passed to run_photometry).
                Used only as a fallback when *residual* is not provided.
    psf_cube  : raw PSF cube, shape (n_psf, ny_psf, nx_psf)
    xs, ys    : 1D arrays of PSF grid detector coordinates
    psf_scale : PSF supersampling factor
    hw        : fit window half-width in pixels
    output    : file path to save figure (PNG, PDF…).  None → don't save.
    n_stamps  : approximate number of star stamps to display
    title     : optional figure suptitle string
    x_offset, y_offset : detector coordinate offsets (same as run_photometry)
    stamp_pad : extra pixels beyond hw to include in each stamp (shows context)
    cmap_data : colormap for data/model panels
    cmap_res  : colormap for residual panel (should be diverging)
    residual  : final residual image returned by run_photometry(return_residual=True).
                When provided, the "Data" panel shows residual + this star's model
                (i.e. the image with all *other* stars already subtracted), so the
                data/model comparison is clean even in crowded fields.
                When None, the raw *data* array is used instead.
    mask      : 2D bool array (True = bad pixel), same as passed to run_photometry.
                Masked pixels are set to NaN in the residual and residual/sigma panels
                so they do not inflate the colour scale.
    noise_map : 2D float64 variance image returned by run_photometry (the star-aware
                variance image).  When provided, the residual/σ panel uses it for
                per-pixel noise, giving a physically correct normalisation that
                accounts for neighbour Poisson noise.  When None, falls back to a
                local sky+read-noise estimate.

    Returns
    -------
    matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, LogNorm, SymLogNorm
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_diagnostics. "
            "Install with:  pip install matplotlib"
        )

    from scipy.ndimage import spline_filter
    from .core import interpolate_psf, _eval_psf_grad_fast

    if not records:
        raise ValueError("No records to plot.")

    # --- Select representative stars ------------------------------------------
    valid = [r for r in records
             if np.isfinite(r.mag) and np.isfinite(r.qfit)
             and r.flux > 1.1 and r.chi2 < 4 and getattr(r, 'converged', True)
             and getattr(r, 'is_star_candidate', True)]
    if not valid:
        raise ValueError("No valid (finite mag/qfit, positive flux) records.")

    good_fits = sorted([r for r in valid if r.qfit < 0.1], key=lambda r: r.mag)
    poor_fits = sorted([r for r in valid if r.qfit >= 0.1], key=lambda r: r.mag)

    # ~80 % good fits, ~20 % poor (for diagnostic interest)
    n_poor = min(len(poor_fits), max(1, n_stamps // 5))
    n_good = min(len(good_fits), n_stamps - n_poor)
    n_poor = n_stamps - n_good  # rebalance if good_fits was short

    def _uniform_sample(lst, k):
        if not lst or k <= 0:
            return []
        if k >= len(lst):
            return list(lst)
        idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
        return [lst[i] for i in idx]

    selected = _uniform_sample(good_fits, n_good) + _uniform_sample(poor_fits, n_poor)
    selected.sort(key=lambda r: r.mag)  # bright → faint for display order
    n_sel = len(selected)

    # --- Layout ---------------------------------------------------------------
    # Groups of 4 columns: data (log) / model (log) / residual / residual/sigma
    # Cap at 3 star-groups per row so individual panels stay readable.
    _NCOLS_PER_STAR = 4
    n_col_groups = max(1, min(3, int(np.ceil(np.sqrt(n_sel / 1.5)))))
    n_rows = int(np.ceil(n_sel / n_col_groups))
    n_cols = n_col_groups * _NCOLS_PER_STAR

    # Each image panel is at least 1.6" wide; colorbars are handled by
    # constrained_layout so no manual width padding needed.
    stamp_size = 2 * (hw + stamp_pad) + 1
    inch_per_stamp = max(1.6, stamp_size / 8.0)
    fig_w = n_cols * inch_per_stamp
    fig_h = n_rows * inch_per_stamp + (0.5 if title else 0.1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False, layout='constrained')

    # --- Prefilter PSF cube once ----------------------------------------------
    psf_coeffs_cube = np.array([
        spline_filter(p, order=3, output=np.float64) for p in psf_cube
    ])
    ny, nx = data.shape

    # --- Stamp rendering ------------------------------------------------------
    for idx, rec in enumerate(selected):
        row = idx // n_col_groups
        cg  = idx %  n_col_groups
        cb  = cg * _NCOLS_PER_STAR
        ax_d, ax_m, ax_r, ax_s = (axes[row, cb], axes[row, cb+1],
                                   axes[row, cb+2], axes[row, cb+3])

        xi = int(round(rec.x)); yi = int(round(rec.y))
        shw = hw + stamp_pad
        y_lo = max(0, yi - shw); y_hi = min(ny, yi + shw + 1)
        x_lo = max(0, xi - shw); x_hi = min(nx, xi + shw + 1)

        # Build PSF model over the stamp window
        dx = rec.x - xi; dy = rec.y - yi
        diy_s = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
        dix_s = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
        local_psf = interpolate_psf(
            psf_coeffs_cube, xs, ys,
            rec.x + x_offset, rec.y + y_offset
        )
        P_s, _, _ = _eval_psf_grad_fast(local_psf, dx, dy, dix_s, diy_s, psf_scale)
        stamp_m = rec.flux * P_s + rec.sky

        # "Data" panel: residual + this star's model so neighbours are removed.
        # residual = original_data - sum(flux_i * P_i for all i), sky not subtracted.
        # Adding back this star's flux*P restores only this star against clean sky.
        if residual is not None:
            stamp_d = (residual[y_lo:y_hi, x_lo:x_hi] +
                       rec.flux * P_s).astype(np.float64)
        else:
            stamp_d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

        stamp_r = stamp_d - stamp_m

        # Per-pixel noise for residual/sigma panel.  Use the star-aware variance
        # image when available (accounts for neighbour Poisson noise); otherwise
        # fall back to a local estimate from the model + sky.
        if noise_map is not None:
            var_s = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
            var_s = np.maximum(var_s, 1e-10)
        else:
            sky_val = max(rec.sky, 0.0)
            var_s = (np.abs(stamp_m) + sky_val) / max(rec.flux, 1.0) + 1.0
        stamp_rs = stamp_r / np.sqrt(var_s)

        # Apply DQ mask: NaN bad pixels in residual panels
        if mask is not None:
            bad = mask[y_lo:y_hi, x_lo:x_hi]
            stamp_r  = np.where(bad, np.nan, stamp_r)
            stamp_rs = np.where(bad, np.nan, stamp_rs)
        above_d_premask = stamp_d - max(rec.sky, 1.0)
        if mask is not None:
            above_d_premask = np.where(mask[y_lo:y_hi, x_lo:x_hi], np.nan, above_d_premask)

        # Apply sigma-clipping mask: mark clipped pixels in residual panels
        clip_m = getattr(rec, 'clipped_mask', None)
        if clip_m is not None and clip_m.shape == stamp_r.shape:
            stamp_r  = np.where(clip_m, np.nan, stamp_r)
            stamp_rs = np.where(clip_m, np.nan, stamp_rs)
            above_d_premask = np.where(clip_m, np.nan, above_d_premask)

        # --- Log stretch for data / model (above sky) -------------------------
        sky_floor = max(rec.sky, 1.0)
        above_d = above_d_premask  # already NaN-masked by DQ and sigma-clip above
        above_m = stamp_m - sky_floor
        # Use SymLogNorm so we capture slightly-negative background gracefully
        peak_signal = max(np.nanmax(above_d) if mask is not None else above_d.max(),
                          above_m.max(), 1.0)
        linthresh = max(0.01 * peak_signal, 1.0)
        norm_log = SymLogNorm(linthresh=linthresh, vmin=-linthresh,
                              vmax=peak_signal, base=10)

        # --- Residual colour scale: symmetric, ignoring masked pixels ---------
        r_valid  = stamp_r[np.isfinite(stamp_r)]
        rs_valid = stamp_rs[np.isfinite(stamp_rs)]
        res_lim  = max(np.abs(r_valid).max(),  1e-10) if r_valid.size  else 1.0
        rs_lim   = max(np.abs(rs_valid).max(), 1.0)   if rs_valid.size else 5.0
        rs_lim   = np.clip(rs_lim, 5.0, 20.0)  # keep scale in ±5–20σ range
        norm_r  = Normalize(vmin=-res_lim, vmax=res_lim)
        norm_rs = Normalize(vmin=-rs_lim,  vmax=rs_lim)

        kw_im = dict(origin='lower', interpolation='nearest', aspect='equal')
        im_d = ax_d.imshow(above_d, norm=norm_log, cmap=cmap_data, **kw_im)
        im_m = ax_m.imshow(above_m, norm=norm_log, cmap=cmap_data, **kw_im)
        im_r = ax_r.imshow(stamp_r,  norm=norm_r,   cmap=cmap_res,  **kw_im)
        im_s = ax_s.imshow(stamp_rs, norm=norm_rs,  cmap=cmap_res,  **kw_im)

        # Colorbars on residual panels (right edge, slim)
        _cb_kw = dict(fraction=0.046, pad=0.04, aspect=12)
        fig.colorbar(im_r, ax=ax_r, **_cb_kw)
        cbar_s = fig.colorbar(im_s, ax=ax_s, **_cb_kw)
        cbar_s.set_label('σ', fontsize=5, labelpad=2)

        for ax in (ax_d, ax_m, ax_r, ax_s):
            ax.set_xticks([]); ax.set_yticks([])
            # Dashed rectangle showing the fit window boundary
            fw_x0 = (xi - hw) - x_lo - 0.5
            fw_y0 = (yi - hw) - y_lo - 0.5
            fw_w  = 2 * hw + 1
            ax.add_patch(Rectangle(
                (fw_x0, fw_y0), fw_w, fw_w,
                fill=False, edgecolor='cyan', linewidth=0.6, linestyle='--'
            ))

        # Star info annotation (upper-left of data panel)
        pass_tag = f"P{rec.pass_number}" if rec.pass_number > 1 else ""
        conv_tag = "" if getattr(rec, 'converged', True) else " !"
        label = f"m={rec.mag:.2f}  q={rec.qfit:.3f}{conv_tag}\n({rec.x:.0f},{rec.y:.0f}) {pass_tag}"
        ax_d.text(0.03, 0.97, label,
                  transform=ax_d.transAxes, fontsize=5, color='white',
                  va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55))

    # Column headers on row 0 of each group
    for cg in range(n_col_groups):
        cb = cg * _NCOLS_PER_STAR
        if n_rows > 0:
            axes[0, cb    ].set_title('Data (log)',    fontsize=7, pad=2)
            axes[0, cb + 1].set_title('Model (log)',   fontsize=7, pad=2)
            axes[0, cb + 2].set_title('Residual',      fontsize=7, pad=2)
            axes[0, cb + 3].set_title('Residual / σ',  fontsize=7, pad=2)

    # Hide unused axes
    for idx in range(n_sel, n_rows * n_col_groups):
        row = idx // n_col_groups
        cg  = idx %  n_col_groups
        cb  = cg * _NCOLS_PER_STAR
        for dc in range(_NCOLS_PER_STAR):
            axes[row, cb + dc].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=9)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# PSF perturbation measurement and visualisation
# ---------------------------------------------------------------------------

def measure_psf_perturbation(
    records,
    residuals_by_chip,
    psf_cube,
    xs, ys,
    psf_scale,
    hw,
    fmin=40.0,
    max_chi2=5.0,
    psf_coeffs_cube=None,
    psf_cache=None,
    clip_sigma=4.0,
    min_stars_clip=20,
    coverage_min_frac=0.05,
    hw_pert=5,
    n_iter_pert=3,
    enforce_constraints=True,
    masks_by_chip=None,
    return_accumulators=False,
):
    """Measure a spatially uniform PSF perturbation from star fit residuals.

    For each converged star candidate, the per-pixel leave-one-out residual
    (normalised by stellar flux) is a noisy estimate of the fractional PSF
    model error at that location.  These are drizzled into the oversampled
    PSF grid with inverse-variance-like weights (∝ flux² × P) and averaged.

    The algorithm runs ``n_iter_pert`` internal refinement iterations.  On
    each pass the currently accumulated δP is subtracted from each star's
    normalised residual (in PSF-frame coordinates) before re-drizzling, so
    successive iterations measure what is *left* after the previous estimate.
    After all iterations, Fortran-style multi-scale smoothing is applied to
    suppress the sub-pixel aliasing that bilinear drizzle introduces, and
    optionally the zero-sum / zero-moment constraints are enforced.

    Smoothing uses separable Savitzky-Golay filters (efficient C
    implementation via scipy.signal.savgol_filter):
      * 3×3 planar background (global trend)
      * 5×7×9-PSF-pixel quadratic fits on the detail (averaged) — captures
        smooth PSF core structure
      * 5-PSF-pixel planar on the detail — used in the PSF wings
      * Radius-dependent blend: core (< 1 det px) uses quadratic; wings
        (> 2 det px) use planar; transition blends both

    Outlier rejection (two-pass, applied before the iteration loop):
      Pass 1 builds the consensus δP.  A scoring pass computes per-star
      weighted RMS of ``(norm_res − bilinear_interp(δP_consensus))``.
      Stars beyond ``median + clip_sigma × 1.4826 × MAD`` are excluded.
      Skipped when fewer than ``min_stars_clip`` qualifying stars found.

    Accumulation uses vectorised ``np.bincount`` (much faster than the
    previous per-star Python loop with ``np.add.at``).

    Parameters
    ----------
    records : list of StarRecord / _FITSRecord
    residuals_by_chip : dict  {chip_ext: 2D ndarray}
        Final residual image per chip (data − Σ flux_k · P_k; sky not
        globally subtracted — each star's sky is stored in r.sky).
    masks_by_chip : dict  {chip_ext: 2D bool ndarray}  or None
        Per-chip DQ mask (True = bad pixel), as returned by load_image.
        When provided, DQ-flagged pixels are excluded from the drizzle.
        Combined with each star's r.clipped_mask (sigma-clipped pixels
        from PSF fitting) so that the perturbation measurement sees exactly
        the same good pixels that were used during fitting.
    psf_cube : 3D ndarray  (n_psf, psf_size, psf_size)
    xs, ys : 1D ndarray  — detector positions of each PSF model
    psf_scale : int  — oversampling factor (typically 4)
    hw : int  — fitting half-width in detector pixels (kept for API compat)
    hw_pert : int  — half-width for residual collection (default 5).  Larger
        than hw to capture PSF wing information; valid because subtract_stars
        uses the full PSF extent so residuals beyond hw are clean.
    n_iter_pert : int  — internal refinement iterations (default 3)
    enforce_constraints : bool  — apply zero-sum and zero-moment constraints
        to the final δP (default True)
    fmin, max_chi2 : quality thresholds
    psf_coeffs_cube : prefiltered PSF coefficients (computed if None)
    psf_cache : dict for interpolate_psf position cache
    clip_sigma : float — MAD sigma threshold for outlier rejection
    min_stars_clip : int — minimum qualifying stars to attempt clipping
    coverage_min_frac : float — PSF pixels below this fraction of peak weight
        are zeroed before smoothing (edge pixels with sparse coverage)

    Returns
    -------
    dict with:
        delta_psf          : (psf_size, psf_size) ndarray — constrained δP
        weight_map         : (psf_size, psf_size) ndarray — accumulated weight
        psf_center         : (psf_size, psf_size) ndarray — central reference PSF
        n_stars            : int — stars used after clipping
        n_outliers_clipped : int — stars removed by sigma-clipping
        constraints_before : dict(sum, mx, my) — moments before enforcement
        constraints_after  : dict(sum, mx, my) — verification after enforcement
    """
    from scipy.ndimage import spline_filter
    from scipy.signal import savgol_filter as _savgol_filter
    from .core import _eval_psf_grad_fast, _window_offsets, interpolate_psf

    psf_size = psf_cube.shape[-1]
    psf_ctr  = psf_size // 2
    n_psf2   = psf_size * psf_size

    if psf_coeffs_cube is None:
        psf_coeffs_cube = np.array([
            spline_filter(p, order=3, output=np.float64) for p in psf_cube
        ])

    # ── PSF-pixel coordinate grids (reused in smoothing and constraints) ──────
    _yg_full, _xg_full = np.mgrid[0:psf_size, 0:psf_size]
    # In PSF pixels, relative to centre (used for constraints)
    xg_c = (_xg_full - psf_ctr).astype(float)
    yg_c = (_yg_full - psf_ctr).astype(float)
    # In detector pixels, relative to centre (used for radius-blend in smoothing)
    r_det = np.sqrt(xg_c**2 + yg_c**2) / float(psf_scale)

    # ── Vectorised accumulation via np.bincount ───────────────────────────────
    # Populated once from the flat concatenated per-star arrays after collection.
    # Closured over n_psf2, psf_size; uses all_iy0/ix0/fy/fx/w from outer scope.

    def _accum_vec(all_iy0, all_ix0, all_fy, all_fx, all_w, all_nr):
        """Bilinear drizzle of all_nr into PSF grid using np.bincount."""
        da_flat = np.zeros(n_psf2)
        wa_flat = np.zeros(n_psf2)
        for dxi, dyi in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ixn = all_ix0 + dxi
            iyn = all_iy0 + dyi
            wx  = all_fx       if dxi else (1.0 - all_fx)
            wy  = all_fy       if dyi else (1.0 - all_fy)
            ww  = all_w * wx * wy
            valid = ((ixn >= 0) & (ixn < psf_size) &
                     (iyn >= 0) & (iyn < psf_size) &
                     (all_w > 0))
            idx = (iyn * psf_size + ixn)[valid]
            da_flat += np.bincount(idx, weights=(ww * all_nr)[valid], minlength=n_psf2)
            wa_flat += np.bincount(idx, weights=ww[valid],            minlength=n_psf2)
        return da_flat.reshape(psf_size, psf_size), wa_flat.reshape(psf_size, psf_size)

    # ── Sigma-clipping helpers (old accumulator, used only for scoring) ───────
    # These operate on the list-of-tuples structure and are only called O(1)
    # times during sigma-clipping before we switch to the fast vectorised path.

    def _accumulate_list(contrib_list):
        da = np.zeros((psf_size, psf_size))
        wa = np.zeros((psf_size, psf_size))
        for iy0_f, ix0_f, fy_f, fx_f, w_f, nr_f, *_ in contrib_list:
            for dxi, wx in ((0, 1.0 - fx_f), (1, fx_f)):
                ixn = ix0_f + dxi
                for dyi, wy in ((0, 1.0 - fy_f), (1, fy_f)):
                    iyn = iy0_f + dyi
                    ww  = w_f * wx * wy
                    valid = ((ixn >= 0) & (ixn < psf_size) &
                             (iyn >= 0) & (iyn < psf_size) &
                             (w_f > 0))
                    np.add.at(da, (iyn[valid], ixn[valid]), (ww * nr_f)[valid])
                    np.add.at(wa, (iyn[valid], ixn[valid]),  ww[valid])
        return da, wa

    def _score_vs_consensus(contrib_list, delta_consensus):
        scores = []
        for iy0_f, ix0_f, fy_f, fx_f, w_f, nr_f, *_ in contrib_list:
            iy0c = np.clip(iy0_f, 0, psf_size - 2)
            ix0c = np.clip(ix0_f, 0, psf_size - 2)
            interp_d = (
                (1 - fy_f) * (1 - fx_f) * delta_consensus[iy0c,   ix0c  ] +
                (1 - fy_f) *      fx_f  * delta_consensus[iy0c,   ix0c+1] +
                     fy_f  * (1 - fx_f) * delta_consensus[iy0c+1, ix0c  ] +
                     fy_f  *      fx_f  * delta_consensus[iy0c+1, ix0c+1]
            )
            W_i = float(w_f.sum())
            s_i = float((w_f * (nr_f - interp_d) ** 2).sum())
            scores.append(np.sqrt(s_i / W_i) if W_i > 0 else 0.0)
        return np.array(scores)

    # ── Collection loop ───────────────────────────────────────────────────────
    star_contribs = []  # (iy0_flat, ix0_flat, fy_flat, fx_flat, w_flat, nr0_flat)

    for r in records:
        if not (getattr(r, 'converged', True)
                and getattr(r, 'is_star_candidate', False)
                and r.qfit < 2.0
                and r.chi2 < max_chi2
                and r.flux >= fmin
                and getattr(r, 'n_conc_2x2', 4) == 4):
            continue

        chip_ext = getattr(r, '_chip_ext', 1)
        x_off    = getattr(r, '_x_offset', 0.0)
        y_off    = getattr(r, '_y_offset', 0.0)
        residual = residuals_by_chip.get(chip_ext)
        if residual is None:
            continue

        ny, nx = residual.shape
        xi = int(round(r.x)); yi = int(round(r.y))
        dx = r.x - xi;        dy = r.y - yi
        y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw_pert, ny, nx)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue

        local_psf = interpolate_psf(psf_coeffs_cube, xs, ys,
                                    r.x + x_off, r.y + y_off,
                                    _cache=psf_cache)
        P, _, _ = _eval_psf_grad_fast(local_psf, dx, dy, dix, diy, psf_scale)
        if P.shape != (y_hi - y_lo, x_hi - x_lo):
            continue

        res_loo  = (residual[y_lo:y_hi, x_lo:x_hi].copy() + r.flux * P) - r.sky
        norm_res = res_loo / r.flux - P

        h_win = y_hi - y_lo
        w_win = x_hi - x_lo

        P_pos = np.maximum(P, 0.0)
        w2d   = (r.flux ** 2) * P_pos

        # Mask bad pixels: DQ-flagged + sigma-clipped during PSF fitting.
        # Sets w2d=0 so masked pixels contribute nothing to numerator or
        # denominator of the drizzle — identical to the pixels excluded in fit.
        chip_dq = (masks_by_chip or {}).get(chip_ext)
        if chip_dq is not None:
            w2d[chip_dq[y_lo:y_hi, x_lo:x_hi]] = 0.0
        clip_m = getattr(r, 'clipped_mask', None)
        if clip_m is not None:
            # clipped_mask covers the fitting window (hw); map it into the
            # (potentially larger) perturbation window (hw_pert).
            y_lo_fit = max(0, yi - hw);  y_hi_fit = min(ny, yi + hw + 1)
            x_lo_fit = max(0, xi - hw);  x_hi_fit = min(nx, xi + hw + 1)
            dy_off = y_lo_fit - y_lo;  dx_off = x_lo_fit - x_lo
            py0 = max(dy_off, 0);       py1 = min(dy_off + (y_hi_fit - y_lo_fit), h_win)
            px0 = max(dx_off, 0);       px1 = min(dx_off + (x_hi_fit - x_lo_fit), w_win)
            fy0 = py0 - dy_off;         fy1 = fy0 + (py1 - py0)
            fx0 = px0 - dx_off;         fx1 = fx0 + (px1 - px0)
            if py1 > py0 and px1 > px0:
                w2d[py0:py1, px0:px1][clip_m[fy0:fy1, fx0:fx1]] = 0.0

        xp  = psf_ctr + (dix - dx) * psf_scale   # (1, w)
        yp  = psf_ctr + (diy - dy) * psf_scale   # (h, 1)
        ix0 = np.floor(xp).astype(int)
        iy0 = np.floor(yp).astype(int)
        fx  = xp - ix0
        fy  = yp - iy0
        star_contribs.append((
            np.broadcast_to(iy0, (h_win, w_win)).ravel().copy(),
            np.broadcast_to(ix0, (h_win, w_win)).ravel().copy(),
            np.broadcast_to(fy,  (h_win, w_win)).ravel().copy(),
            np.broadcast_to(fx,  (h_win, w_win)).ravel().copy(),
            w2d.ravel().copy(),
            norm_res.ravel().copy(),
            chip_ext,   # scalar — used for per-chip accumulation
        ))

    n_stars_pass1 = len(star_contribs)

    # ── Outlier sigma-clipping ────────────────────────────────────────────────
    # Uses the list accumulator (called O(1) times; list is discarded after).
    n_outliers_clipped = 0
    if n_stars_pass1 >= min_stars_clip and clip_sigma > 0:
        da0, wa0 = _accumulate_list(star_contribs)
        good0    = wa0 > 0
        delta_0  = np.where(good0, da0 / np.where(good0, wa0, 1.0), 0.0)
        scores   = _score_vs_consensus(star_contribs, delta_0)
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        threshold = med + clip_sigma * 1.4826 * mad
        keep = scores <= threshold
        n_outliers_clipped = int((~keep).sum())
        if n_outliers_clipped > 0:
            star_contribs = [c for c, k in zip(star_contribs, keep) if k]

    n_stars = len(star_contribs)

    # ── Build flat concatenated arrays (switch to vectorised path) ────────────
    _zero_result = {
        'delta_psf':           np.zeros((psf_size, psf_size)),
        'weight_map':          np.zeros((psf_size, psf_size)),
        'psf_center':          psf_cube[len(psf_cube) // 2].copy(),
        'n_stars':             0,
        'n_stars_initial':     n_stars_pass1,
        'n_outliers_clipped':  n_outliers_clipped,
        'constraints_before':  {'sum': 0.0, 'mx': 0.0, 'my': 0.0},
        'constraints_after':   {'sum': 0.0, 'mx': 0.0, 'my': 0.0},
    }
    if n_stars == 0:
        return _zero_result

    # Collect unique chip IDs before consuming the list
    _all_chips = [c[6] for c in star_contribs]
    _unique_chips = sorted(set(_all_chips))

    all_iy0 = np.concatenate([c[0] for c in star_contribs])
    all_ix0 = np.concatenate([c[1] for c in star_contribs])
    all_fy  = np.concatenate([c[2] for c in star_contribs])
    all_fx  = np.concatenate([c[3] for c in star_contribs])
    all_w   = np.concatenate([c[4] for c in star_contribs])
    all_nr0 = np.concatenate([c[5] for c in star_contribs])

    # Build per-chip index masks (into the concatenated arrays) only if needed.
    # Store as {chip_ext: bool mask over star_contribs entries}.
    if return_accumulators and len(_unique_chips) > 1:
        # Per-star pixel counts for splitting the flat arrays
        _star_lens   = [len(c[0]) for c in star_contribs]
        _star_starts = np.concatenate([[0], np.cumsum(_star_lens)])
        # Map star index → flat-pixel mask
        _chip_flat_mask: dict = {}
        for _chip in _unique_chips:
            _m = np.zeros(len(all_w), dtype=bool)
            for _si, _ce in enumerate(_all_chips):
                if _ce == _chip:
                    _m[_star_starts[_si]:_star_starts[_si+1]] = True
            _chip_flat_mask[_chip] = _m
    else:
        _chip_flat_mask = {}

    # Count stars per chip
    _n_stars_by_chip: dict = {}
    for _chip in _unique_chips:
        _n_stars_by_chip[_chip] = sum(1 for _ce in _all_chips if _ce == _chip)

    del star_contribs  # free memory

    # Clip bilinear base indices to valid interpolation range (clamp, not reject)
    all_iy0c = np.clip(all_iy0, 0, psf_size - 2)
    all_ix0c = np.clip(all_ix0, 0, psf_size - 2)

    # Compute weight accumulation once (same for every iteration).
    _, weight_accum = _accum_vec(all_iy0, all_ix0, all_fy, all_fx, all_w, all_nr0)

    # Smooth taper: ramps linearly from 0 (no coverage) to 1 (at the coverage
    # threshold), then stays at 1 for well-covered pixels.  A hard binary mask
    # creates a step discontinuity in δP at the coverage boundary; when (psf_std
    # + δP) is bilinearly interpolated during PSF evaluation, that step creates
    # position-dependent evaluation artefacts that degrade astrometric quality.
    wmax = weight_accum.max()
    if coverage_min_frac > 0 and wmax > 0:
        coverage_mask  = weight_accum >= (wmax * coverage_min_frac)
        coverage_taper = np.clip(weight_accum / (wmax * coverage_min_frac), 0.0, 1.0)
    else:
        coverage_mask  = np.ones((psf_size, psf_size), dtype=bool)
        coverage_taper = np.ones((psf_size, psf_size), dtype=float)

    # ── Fortran-style multi-scale smoothing ───────────────────────────────────
    def _smooth_delta(delta_raw):
        """Smooth PSF perturbation: planar background + multi-scale quadratic blend.

        The coverage taper is applied AFTER smoothing, not before.  Pre-masking
        zero-pads the array at the coverage boundary, which causes SG filters
        (mode='mirror') to create artefacts that strongly attenuate signals near
        or beyond the boundary.  Smoothing the unmasked drizzle result (which
        naturally decays to zero outside the measured PSF core) and tapering
        afterward gives a smooth transition to zero at the coverage edge.
        """
        def _sg2d(arr, win, poly):
            return _savgol_filter(
                _savgol_filter(arr, win, poly, axis=0, mode='mirror'),
                win, poly, axis=1, mode='mirror',
            )

        bg     = _sg2d(delta_raw, 3, 1)                         # 3×3 planar background
        detail = delta_raw - bg
        quad   = (_sg2d(detail, 5, 2) +                         # 5-, 7-, 9-PSF-px quadratic
                  _sg2d(detail, 7, 2) +
                  _sg2d(detail, 9, 2)) / 3.0
        p5     = _sg2d(detail, 5, 1)                            # 5-PSF-px planar (wings)

        smooth = np.where(r_det < 1.0, bg + quad,               # core: full quadratic
                 np.where(r_det < 2.0, bg + 0.5 * (quad + p5), # transition: blend
                          bg + p5))                              # wings: planar only
        return smooth * coverage_taper                           # smooth taper AFTER smoothing

    # ── Iterative refinement ──────────────────────────────────────────────────
    # Each iteration k:
    #   1. Subtract current cumulative δP from the stored nr0 in PSF-frame coords
    #      (bilinear interpolation of delta_cum at each pixel's PSF position).
    #   2. Re-drizzle the updated residuals.
    #   3. Smooth the per-iteration raw delta and accumulate.
    #
    # This avoids rebuilding the residual images: the per-star nr0 is stored
    # once and updated cheaply via a vectorised bilinear lookup each iteration.
    delta_cum = np.zeros((psf_size, psf_size))

    good_w = weight_accum > 0
    _final_da_k = None   # numerator from the last iteration (for raw accumulator export)
    _final_nr_k = None   # residuals from the last iteration (for per-chip export)

    for _iter in range(n_iter_pert):
        # Bilinear interpolation of delta_cum at every stored PSF-pixel position.
        interp_d = (
            (1.0 - all_fy) * (1.0 - all_fx) * delta_cum[all_iy0c,     all_ix0c    ] +
            (1.0 - all_fy) *       all_fx   * delta_cum[all_iy0c,     all_ix0c + 1] +
                   all_fy  * (1.0 - all_fx) * delta_cum[all_iy0c + 1, all_ix0c    ] +
                   all_fy  *       all_fx   * delta_cum[all_iy0c + 1, all_ix0c + 1]
        )
        all_nr_k = all_nr0 - interp_d

        da_k, _ = _accum_vec(all_iy0, all_ix0, all_fy, all_fx, all_w, all_nr_k)
        delta_raw_k = np.where(good_w, da_k / np.where(good_w, weight_accum, 1.0), 0.0)

        delta_cum += _smooth_delta(delta_raw_k)

        if _iter == n_iter_pert - 1:
            _final_da_k = da_k.copy()    # unsmoothed numerator Σ(w·nr)
            _final_nr_k = all_nr_k       # final-iteration residuals (needed for per-chip)

    # ── Enforce constraints ───────────────────────────────────────────────────
    constraints_before = {
        'sum': float(delta_cum.sum()),
        'mx':  float((xg_c * delta_cum).sum()),
        'my':  float((yg_c * delta_cum).sum()),
    }

    if enforce_constraints:
        delta = delta_cum - delta_cum.mean()
        sum_xg2 = float((xg_c ** 2).sum())
        sum_yg2 = float((yg_c ** 2).sum())
        if sum_xg2 > 0:
            delta -= (float((xg_c * delta).sum()) / sum_xg2) * xg_c
        if sum_yg2 > 0:
            delta -= (float((yg_c * delta).sum()) / sum_yg2) * yg_c
    else:
        delta = delta_cum

    constraints_after = {
        'sum': float(delta.sum()),
        'mx':  float((xg_c * delta).sum()),
        'my':  float((yg_c * delta).sum()),
    }

    result = {
        'delta_psf':           delta,
        'weight_map':          weight_accum,
        'psf_center':          psf_cube[len(psf_cube) // 2].copy(),
        'n_stars':             n_stars,
        'n_stars_initial':     n_stars_pass1,
        'n_outliers_clipped':  n_outliers_clipped,
        'constraints_before':  constraints_before,
        'constraints_after':   constraints_after,
    }

    if return_accumulators:
        _zeros = np.zeros((psf_size, psf_size))
        result['raw_sum_wv']     = _final_da_k if _final_da_k is not None else _zeros.copy()
        result['raw_sum_w']      = weight_accum.copy()
        result['n_stars_by_chip'] = _n_stars_by_chip

        # Per-chip raw accumulators (final-iteration residuals filtered by chip)
        _sum_wv_by_chip: dict = {}
        _sum_w_by_chip:  dict = {}
        for _chip in _unique_chips:
            if _chip in _chip_flat_mask and _final_nr_k is not None:
                _m = _chip_flat_mask[_chip]
                _da_chip, _dw_chip = _accum_vec(
                    all_iy0[_m], all_ix0[_m], all_fy[_m], all_fx[_m],
                    all_w[_m], _final_nr_k[_m])
                _sum_wv_by_chip[_chip] = _da_chip
                _sum_w_by_chip[_chip]  = _dw_chip
            else:
                # Single-chip image: per-chip == combined
                _sum_wv_by_chip[_chip] = result['raw_sum_wv'].copy()
                _sum_w_by_chip[_chip]  = result['raw_sum_w'].copy()
        result['raw_sum_wv_by_chip'] = _sum_wv_by_chip
        result['raw_sum_w_by_chip']  = _sum_w_by_chip

    return result


def plot_psf_perturbation(psf_center, delta_psf, weight_map,
                          output=None, title=""):
    """3D surface plots of the original PSF, δP, and the corrected PSF.

    Layout (4 + 2 panels):
      Top row (3D surfaces) : original P | δP | P + δP | δP/P fractional error
      Bottom row            : weight map (2D) | horizontal cross-section
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

    psf_size = psf_center.shape[0]
    psf_ctr  = psf_size // 2
    psf_corrected = psf_center + delta_psf

    # Zoom 3D surfaces to the region where δP is non-negligible, so the plots
    # show the perturbation structure rather than a tiny feature in a vast flat
    # plain.  Use 1% of |δP|_max as the support threshold; the outer constraint
    # residuals are typically < 0.1% of peak and fall below this.
    abs_max_dp = max(abs(float(delta_psf.min())), abs(float(delta_psf.max())), 1e-10)
    yg_w, xg_w = np.mgrid[0:psf_size, 0:psf_size]
    r_w = np.sqrt((xg_w - psf_ctr) ** 2 + (yg_w - psf_ctr) ** 2)
    significant = np.abs(delta_psf) > 0.01 * abs_max_dp
    cov_r = float(r_w[significant].max()) if significant.any() else psf_ctr // 3
    half_range = int(min(cov_r + 6, psf_ctr))   # coverage + 6-pixel margin
    lo = psf_ctr - half_range
    hi = psf_ctr + half_range + 1

    g_full = np.arange(psf_size) - psf_ctr
    g      = g_full[lo:hi]
    X, Y = np.meshgrid(g, g)

    def _crop(arr):
        return arr[lo:hi, lo:hi]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=10)

    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[3, 1.6])

    _surf_kw = dict(linewidth=0, antialiased=False, rasterized=True, alpha=0.92)
    vmax_psf = float(psf_center.max())
    vmin_psf = float(psf_center.min())

    # --- Original PSF ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot_surface(X, Y, _crop(psf_center), cmap='viridis',
                     vmin=vmin_psf, vmax=vmax_psf, **_surf_kw)
    ax1.set_title("Original PSF", fontsize=9)
    ax1.tick_params(labelsize=7)
    ax1.set_xlabel("PSF px", fontsize=7)
    ax1.set_ylabel("PSF px", fontsize=7)

    # --- δP ---
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    abs_max = max(abs(float(delta_psf.min())), abs(float(delta_psf.max())), 1e-10)
    ax2.plot_surface(X, Y, _crop(delta_psf), cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max, **_surf_kw)
    ax2.set_title(f"δP  (zero-sum, zero-moment)\n"
                  f"max={delta_psf.max():+.4f}  min={delta_psf.min():+.4f}",
                  fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.set_xlabel("PSF px", fontsize=7)
    ax2.set_ylabel("PSF px", fontsize=7)

    # --- P + δP ---
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.plot_surface(X, Y, _crop(psf_corrected), cmap='viridis',
                     vmin=vmin_psf, vmax=vmax_psf, **_surf_kw)
    ax3.set_title("P + δP  (corrected)", fontsize=9)
    ax3.tick_params(labelsize=7)
    ax3.set_xlabel("PSF px", fontsize=7)
    ax3.set_ylabel("PSF px", fontsize=7)

    # --- Fractional error δP / P (masked where PSF is near zero) ---
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    p_thresh = 0.01 * vmax_psf
    p_safe   = np.where(np.abs(psf_center) > p_thresh, psf_center, np.nan)
    frac_err = delta_psf / p_safe
    fabs     = float(np.nanmax(np.abs(frac_err)))
    ax4.plot_surface(X, Y, _crop(np.nan_to_num(frac_err, nan=0.0)), cmap='RdBu_r',
                     vmin=-fabs, vmax=fabs, **_surf_kw)
    ax4.set_title(f"δP / P  (fractional error)\n|peak| ≈ {fabs:.3f}", fontsize=9)
    ax4.tick_params(labelsize=7)
    ax4.set_xlabel("PSF px", fontsize=7)
    ax4.set_ylabel("PSF px", fontsize=7)

    # --- Weight map ---
    ax5 = fig.add_subplot(gs[1, :2])
    wm_plot = weight_map if weight_map is not None else np.zeros((psf_size, psf_size))
    im = ax5.imshow(np.log1p(wm_plot), origin='lower',
                    extent=[-psf_ctr, psf_ctr, -psf_ctr, psf_ctr],
                    cmap='plasma', aspect='equal')
    ax5.set_title("Coverage — log(1 + weight)", fontsize=9)
    ax5.set_xlabel("PSF px", fontsize=8)
    ax5.set_ylabel("PSF px", fontsize=8)
    ax5.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax5, fraction=0.04, pad=0.04)

    scale_dp = 5.0

    # --- Horizontal cross-section through PSF centre (full range) ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(g_full, psf_center[psf_ctr, :],    'b-',  lw=1.4, label="P  (original)")
    ax6.plot(g_full, psf_corrected[psf_ctr, :], 'g--', lw=1.4, label="P + δP")
    ax6.plot(g_full, delta_psf[psf_ctr, :] * scale_dp, 'r-', lw=0.9, alpha=0.8,
             label=f"δP × {scale_dp:.0f}")
    ax6.axhline(0, color='k', lw=0.5, ls=':')
    ax6.set_title("Horizontal cross-section (y = 0)", fontsize=9)
    ax6.set_xlabel("PSF px", fontsize=8)
    ax6.set_ylabel("PSF value", fontsize=8)
    ax6.tick_params(labelsize=7)
    ax6.legend(fontsize=7, loc='upper right')

    # --- Vertical cross-section through PSF centre (full range) ---
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(g_full, psf_center[:, psf_ctr],    'b-',  lw=1.4, label="P  (original)")
    ax7.plot(g_full, psf_corrected[:, psf_ctr], 'g--', lw=1.4, label="P + δP")
    ax7.plot(g_full, delta_psf[:, psf_ctr] * scale_dp, 'r-', lw=0.9, alpha=0.8,
             label=f"δP × {scale_dp:.0f}")
    ax7.axhline(0, color='k', lw=0.5, ls=':')
    ax7.set_title("Vertical cross-section (x = 0)", fontsize=9)
    ax7.set_xlabel("PSF px", fontsize=8)
    ax7.set_ylabel("PSF value", fontsize=8)
    ax7.tick_params(labelsize=7)
    ax7.legend(fontsize=7, loc='upper right')

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
