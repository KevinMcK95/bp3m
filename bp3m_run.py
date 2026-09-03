#!/usr/bin/env python3
"""
bp3m — end-to-end pipeline combining Gaia and HST data to
measure improved proper motions.

Usage examples
--------------
# By target name (Simbad-resolved):
  bp3m --name "Sculptor dSph" --lib_dir ./lib

# By coordinates:
  bp3m --ra 15.039 --dec -33.709 --search_radius 0.3 --lib_dir ./lib

# Resume after cross-match (e.g. to re-run alignment with different params):
  bp3m --name "Sculptor dSph" --lib_dir ./lib \\
      --skip_download --skip_psf --skip_crossmatch \\
      --n_bp3m_iter 30 --bp3m_clip_sigma 3.5

Pipeline steps
--------------
  1  download_gaia    Download Gaia DR3 catalogue
  2  download_hst     Search MAST and download HST FLC images
  3  psf_fitting      PSF-fit each FLC image (py1pass)
  4  cross_match      Cross-match HST ↔ Gaia (fast_cross_match)
  5  alignment        Bayesian alignment + proper motions (BP3M)

Extension note
--------------
JWST support is planned. Pass --telescope JWST once py1pass and
fast_cross_match have been updated for JWST data.
"""

import argparse
import re
import sys
import os
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np


def _config_lib_dir() -> str | None:
    """Read lib_dir from config.toml if it exists (written by bp3m-setup).

    Config location: $BP3M_HOME/config.toml, or ~/.bp3m/config.toml by default.
    """
    import os
    bp3m_home = Path(os.environ["BP3M_HOME"]) if "BP3M_HOME" in os.environ else Path.home() / ".bp3m"
    config = bp3m_home / "config.toml"
    if not config.exists():
        return None
    try:
        text = config.read_text()
        m = re.search(r'^lib_dir\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        return m.group(1) if m else None
    except Exception:
        return None


def _parse_args():
    p = argparse.ArgumentParser(
        prog='bp3m',
        description='Measure proper motions by combining Gaia + HST data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Target ────────────────────────────────────────────────────────────────
    tgt = p.add_argument_group('Target (provide name OR ra+dec)')
    tgt.add_argument('--name', type=str, default=None,
                     help='Target name (resolved via Simbad). '
                          'Used as the field directory name.')
    tgt.add_argument('--ra',  type=float, nargs='+', default=None,
                     help='Centre R.A. (degrees). Multiple values enable multi-pointing mode.')
    tgt.add_argument('--dec', type=float, nargs='+', default=None,
                     help='Centre Dec. (degrees). Must match the number of --ra values.')
    tgt.add_argument('--search_radius', type=float, nargs='+', default=None,
                     help='Circular search radius (degrees). A single value is broadcast '
                          'to all pointings. Converted to an equal-area box per pointing.')
    tgt.add_argument('--search_width',  type=float, default=None,
                     help='Search box width  (degrees, overrides --search_radius; '
                          'applied uniformly to all pointings)')
    tgt.add_argument('--search_height', type=float, default=None,
                     help='Search box height (degrees, overrides --search_radius; '
                          'applied uniformly to all pointings)')

    # ── DELVE ─────────────────────────────────────────────────────────────────
    _DEFAULT_DELVE_DIR = '/bootes_raid6/users/kmckinnon/bp3m/DELVE_ProperMotion/PMCatalog'
    dlv = p.add_argument_group('DELVE options (optional)')
    dlv.add_argument('--use_delve', action='store_true',
                     help='Incorporate DELVE proper-motion priors into the fit. '
                          'If not given, all DELVE steps are silently skipped.')
    dlv.add_argument('--delve_dir', type=str, default=_DEFAULT_DELVE_DIR,
                     help=f'Path to the directory containing DELVE PM_hp*.fits tile files. '
                          f'Default: {_DEFAULT_DELVE_DIR}')
    dlv.add_argument('--delve_use_for_align', action='store_true',
                     help='Allow DELVE-only sources to contribute to image alignment '
                          '(off by default; may be useful when DELVE astrometry improves).')
    dlv.add_argument('--force_redownload_delve', action='store_true',
                     help='Regenerate the DELVE CSV even if a cached one already exists.')
    dlv.add_argument('--skip_delve_crossmatch', action='store_true',
                     help='Skip DELVE cross-matching (use existing matched_delve.csv).')
    dlv.add_argument('--force_rematch_delve', action='store_true',
                     help='Re-run DELVE cross-matching even if matched_delve.csv already exists.')

    # ── Gaia ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Gaia options')
    g.add_argument('--min_gmag', type=float, default=0.0,
                   help='Brightest G magnitude (default 0.0 = no bright limit)')
    g.add_argument('--max_gmag', type=float, default=None,
                   help='Faintest G magnitude (default: no limit)')
    g.add_argument('--source_table', type=str, default='gaiadr3.gaia_source',
                   help='Gaia TAP source table (default gaiadr3.gaia_source)')
    g.add_argument('--sigma_flux_excess', type=float, default=3.0,
                   help='Sigma for flux-excess-factor clipping (default 3.0)')
    g.add_argument('--only_5p', action='store_true',
                   help='Restrict to 5-parameter Gaia solutions only')

    # ── HST ───────────────────────────────────────────────────────────────────
    h = p.add_argument_group('HST / telescope options')
    h.add_argument('--telescope', type=str, default='HST',
                   help='Telescope (default HST; JWST planned)')
    h.add_argument('--hst_filters', type=str, nargs='+', default=None,
                   help='Required filters, e.g. F814W F606W F850LP '
                        '(default: all filters with PSF+GDC in lib_dir). '
                        'Use MAST filter names (e.g. F850LP not F850L).')
    h.add_argument('--hst_im_type', type=str, default='_flc',
                   help='Image type: _flc (default) or _flt')
    h.add_argument('--hst_exptime_min', type=float, default=2.0,
                   help='Minimum average exposure time per image set (s)')
    h.add_argument('--hst_exptime_max', type=float, default=np.inf,
                   help='Maximum average exposure time per image set (s)')
    h.add_argument('--time_baseline', type=float, default=None,
                   help='Minimum HST–Gaia time baseline in days (default: no limit)')
    h.add_argument('--obs_date_min', type=str, default=None,
                   help='Earliest HST observation date to include (ISO, e.g. 2005-01-01). '
                        'Default: no lower limit.')
    h.add_argument('--obs_date_max', type=str, default=None,
                   help='Latest HST observation date to include (ISO, e.g. 2020-12-31). '
                        'Default: no upper limit.')
    h.add_argument('--instruments', type=str, nargs='+', default=None,
                   help='HST instrument/detector combinations to include '
                        '(e.g. ACS/WFC WFC3/UVIS). Default: all supported instruments '
                        'that have PSF and GDC files in lib_dir.')
    h.add_argument('--field_ids', type=str, nargs='+', default=None,
                   help='Field IDs to download. Integers from the table '
                        '(space-separated), "y"/"all" to download everything, '
                        'or "n"/"0" to skip download. Default: interactive prompt.')

    # ── PSF fitting ───────────────────────────────────────────────────────────
    psf = p.add_argument_group('PSF fitting (py1pass)')
    _lib_default = _config_lib_dir() or str(Path.home() / 'GaiaHub-master' / 'lib')
    psf.add_argument('--lib_dir', type=str,
                     default=_lib_default,
                     help='Library directory containing STDPSFs/ and STDGDCs/ '
                          'subdirectories. Defaults to the path set by bp3m-setup '
                          f'(currently: {_lib_default})')
    psf.add_argument('--fmin', type=float, default=None,
                     help='Directly set the pypass flux detection threshold in electrons. '
                          'Overrides both --mag_st_max and --fmin_thresh when given.')
    psf.add_argument('--fmin_thresh', type=float, default=None,
                     help='Hard lower bound on the minimum source flux in electrons '
                          '(default 40). Acts as a floor: fmin will never go below '
                          'this value even when mag_st_max would imply a lower threshold. '
                          'Ignored if --fmin is given.')
    psf.add_argument('--mag_st_max', type=float, default=None,
                     help='Faint ST-magnitude limit used to set the detection threshold '
                          '(default 28). Converted to a flux threshold per image using '
                          'PHOTFLAM and EXPTIME; floored at fmin_thresh. '
                          'Ignored if --fmin is given.')
    psf.add_argument('--hmin', type=int, default=None,
                     help='NMS radius in pixels (default 4)')
    psf.add_argument('--n_passes', type=int, default=None,
                     help='Total PSF fit passes (default 2)')
    psf.add_argument('--n_discovery_passes', type=int, default=None,
                     help='How many of those passes include new-source detection '
                          '(default: n_passes-1, i.e. last pass is refit-only)')
    psf.add_argument('--psf_max_iter', type=int, default=None,
                     help='Max iterations for PSF fit convergence (default 100)')
    psf.add_argument('--conc_limit', type=float, default=None,
                     help='Concentration lower bound for star/non-star classification '
                          '(upper bound = 1/conc_limit, default 0.9)')
    psf.add_argument('--sat_threshold', type=float, default=None,
                     help='Saturation DN threshold (default 60000)')

    # ── Gaia DR4 epoch astrometry (Step 4b) ──────────────────────────────────
    dr4 = p.add_argument_group('Gaia DR4 epoch astrometry (Step 4b, optional)')
    dr4.add_argument('--use_gaia_dr4_epoch', action='store_true',
                     help='Download Gaia DR4 epoch astrometry for cross-matched sources '
                          'and re-solve 5D parameters with gaiasupdate. Requires the '
                          'gaiasupdate package (pip install gaiasupdate) and a working '
                          'ESA DataLink connection (or --dr4_prerelease_votable for '
                          'offline testing with the ESA pre-release sample).')
    dr4.add_argument('--dr4_access', type=str, default='datalink',
                     choices=['datalink', 'prerelease'],
                     help='Epoch data access method: datalink (default, requires ESA '
                          'credentials for non-public releases) or prerelease (local '
                          'VOTable, set --dr4_prerelease_votable).')
    dr4.add_argument('--dr4_prerelease_votable', type=str, default=None,
                     help='Path to the ESA pre-release epoch astrometry VOTable '
                          '(required when --dr4_access=prerelease).')
    dr4.add_argument('--dr4_data_release', type=str, default='Gaia DR4',
                     help='DataLink data-release string (default "Gaia DR4"; use '
                          '"Gaia DR4_INT4" for the internal pre-release).')
    dr4.add_argument('--dr4_credentials', type=str, default=None,
                     help='Path to ESA archive credentials file (for DataLink access).')
    dr4.add_argument('--dr4_model', type=str, default='5p_single_source',
                     choices=['5p_single_source', '3p_single_source_without_offsets',
                              '6p_constrained_colour', '6p_perspective_acceleration'],
                     help='gaiasupdate astrometric model for epoch re-solve '
                          '(default 5p_single_source).')
    dr4.add_argument('--dr4_no_replace_pms', action='store_true',
                     help='Store epoch solutions as extra columns but do NOT replace '
                          'pmra/pmdec/parallax priors in the solver.')
    dr4.add_argument('--force_rerun_dr4_epoch', action='store_true',
                     help='Re-download and re-solve even if epoch cache exists.')

    # ── Cross-matching ────────────────────────────────────────────────────────
    xm = p.add_argument_group('Cross-matching (fast_cross_match)')
    xm.add_argument('--cross_match_pix_floor', type=float, default=0.5,
                    help='HST positional uncertainty floor in pixels applied during cross-matching (default 0.5)')
    xm.add_argument('--min_matches', type=int, default=3,
                    help='Minimum seed matches for 4P discovery (default 3)')
    xm.add_argument('--max_mag_diff', type=float, default=5.0,
                    help='Maximum Gaia–HST magnitude difference (default 5.0)')
    xm.add_argument('--scale_sweep', action='store_true',
                    help='Enable pixel-scale sweep during 4P discovery (slower)')
    xm.add_argument('--discovery_max_offset', type=int, default=50,
                    help='Half-width of the offset histogram search during 4P discovery in pixels (default 50)')
    xm.add_argument('--auto_resid_floor', action='store_true',
                    help='Enable the per-iteration empirical residual covariance floor during affine refinement (default: off)')
    xm.add_argument('--xmatch_init_resid_max', type=float, default=5.0,
                    help='Maximum allowed init 6P residual (px) after 4P discovery before '
                         'declaring the seed spurious and skipping the image (default 5.0). '
                         'Fields with large Gaia scatter or long epoch baselines may need a '
                         'higher value; use 2.0 to restore the old conservative behaviour.')
    xm.add_argument('--no_qso_anchors', action='store_true',
                    help='Skip QSO anchor vetting (Quaia + MILLIQUAS cross-match + '
                         'astrometric cut). By default QSO vetting runs after cross-matching '
                         'and saves {field}/Gaia/{field}_qso_anchors.csv for pop-fit.')

    # ── Alignment (BP3M) ──────────────────────────────────────────────────────
    bp = p.add_argument_group('Bayesian alignment (BP3M)')
    bp.add_argument('--n_bp3m_iter', type=int, default=50,
                    help='Maximum BP3M outer iterations (default 20)')
    bp.add_argument('--n_samples', type=int, default=1000,
                    help='Posterior samples for uncertainty estimation (default 1000). '
                         'Only used when --mcmc_posteriors is set.')
    bp.add_argument('--mcmc_posteriors', action='store_true',
                    help='Use Monte Carlo sampling to marginalise over C_r instead of '
                         'the default exact analytic Big_C approach. The analytic method '
                         'is more accurate when few Gaia alignment stars are available; '
                         'this flag restores the old sampling behaviour for comparison.')
    bp.add_argument('--bp3m_clip_sigma', type=float, default=4.5,
                    help='MAD sigma threshold for outlier rejection (default 4.5; '
                         '0 = disabled)')
    bp.add_argument('--poly_order', type=int, default=1,
                    help='Polynomial order for image transformation (default 1=linear)')
    bp.add_argument('--no_split_ccd', action='store_true',
                    help='Disable per-CCD splitting for ACS/WFC images (default: split enabled)')
    bp.add_argument('--min_stars_split_ccd', type=int, default=20,
                    help='Minimum stars required on each CCD half to allow splitting. '
                         'Images where either half has fewer than N stars are kept unsplit. '
                         'Only applies when --no_split_ccd is not set. (default: 20)')
    bp.add_argument('--two_phase_align', action='store_true',
                    help='Two-phase alignment: converge with alpha OFF first '
                         '(rejection via adaptive thresholds only), then '
                         're-converge with the per-image alpha model enabled, '
                         'warm-started from the Phase A geometry.')
    bp.add_argument('--fit_epoch_distortion', action='store_true',
                    help='Fit a shared low-order distortion correction per '
                         '(instrument, detector, chip, filter, epoch) group, '
                         'Legendre basis of total degree 2..order (degrees 0-1 '
                         'excluded: degenerate with per-image alignment).')
    bp.add_argument('--epoch_dist_order', type=int, default=3,
                    help='Maximum total degree of the epoch-distortion basis')
    bp.add_argument('--epoch_gap_days', type=float, default=180.0,
                    help='Time gap that starts a new epoch group')
    bp.add_argument('--epoch_dist_sigma', type=float, default=10.0,
                    help='Gaussian prior width per epoch-distortion coefficient (mas)')
    bp.add_argument('--epoch_breaks', type=float, nargs='+', default=None,
                    help='Explicit epoch boundaries (decimal years), overrides '
                         'gap clustering at these times')
    bp.add_argument('--epoch_dist_min_images', type=int, default=3,
                    help='Minimum images for a group to receive a correction')
    bp.add_argument('--epoch_dist_groupby', type=str, default='full',
                    choices=['full', 'no_filter', 'no_epoch', 'static'],
                    help='Grouping granularity for the shared distortion: '
                         'full = inst/det/chip/filter x epoch (default); '
                         'no_filter = inst/det/chip x epoch; '
                         'no_epoch = one static D per inst/det/chip/filter; '
                         'static = one static D per inst/det/chip.')
    bp.add_argument('--no_inflate_hst_errors', action='store_true',
                    help='Disable per-image HST error inflation (default: inflation enabled)')
    bp.add_argument('--no_align_prior', action='store_true',
                    help='Disable the alignment parameter prior (a,b,c,d,delta_ra0,delta_dec0). '
                         'Sets the prior precision to zero so posteriors are determined entirely '
                         'by the data. Useful for diagnosing prior-driven biases.')
    bp.add_argument('--test_hysteresis_delta', type=float, default=1.0,
                    help='Width of the reject/re-admit dead-band for tests '
                         '1-3 (adaptive_k + delta for expulsion vs adaptive_k '
                         'for admission). Was effectively 0.1; 1.0 gives a '
                         '~20%% threshold gap that stops borderline '
                         'detections from oscillating.')
    bp.add_argument('--min_align_demote', type=int, default=5,
                    help='Images whose alignment-star count falls below this '
                         'are demoted to astrometry-only with their '
                         'transformation frozen at the last converged value '
                         '(re-promoted at count >= N+3). 0 disables.')
    bp.add_argument('--use_indv_outputs', action='store_true',
                    help='Warm-start the joint fit from BP3M_indv_results/: '
                         'per-image r_init from the indv posterior, indv-'
                         'rejected detections hard-blocked from alignment, '
                         'indv-accepted ones seed the initial mask (still '
                         'refined by the joint tests). Requires indv fits '
                         'run on the SAME cross-match outputs (checked per '
                         'image; mismatches fall back with a warning).')
    bp.add_argument('--bp3m_pos_err_floor', type=float, default=0.05,
                    help='Per-detection positional systematics floor in pixels, added '
                         'IN QUADRATURE to the HST uncertainties before BP3M '
                         '(default 0.05 px = 2.5 mas ACS/WFC, 2.0 mas WFC3/UVIS, '
                         '6.5 mas WFC3/IR; models residual distortion/CTE/cross-filter '
                         'systematics and prevents numerically unstable residuals for '
                         'very bright stars)')
    bp.add_argument('--no_influence_clip', action='store_true',
                    help='Disable test-4 Cook\'s D influence clipping (default: enabled; '
                         'targets moderate-outlier high-leverage detections missed by the sigma threshold)')
    bp.add_argument('--prior_sigma_rot_deg', type=float, default=None,
                    help='Plate rotation prior width in degrees (default 0.05; pre-2026-08-04: 0.1)')
    bp.add_argument('--prior_sigma_scale', type=float, default=None,
                    help='Plate scale prior width, fractional (default 2e-4; pre-2026-08-04: 1.5e-2)')
    bp.add_argument('--prior_sigma_skew', type=float, default=None,
                    help='Plate skew prior width (default 1e-3; pre-2026-08-04: 5e-3)')
    bp.add_argument('--prior_sigma_pointing', type=float, default=None,
                    help='Pointing offset prior width in mas (default 5000.0)')
    bp.add_argument('--inflate_alpha_max', type=float, default=3.0,
                    help='Per-iteration cap on the alpha error-inflation multiplier (default 3.0; '
                         'pre-2026-08-05 default was 10.0)')
    bp.add_argument('--influence_k', type=float, default=5.0,
                    help='Adaptive multiplier k for test-4 sigma_resid and scaled_D thresholds '
                         '(thresh = max(p50 + k*(p50-p16), floor); default 5.0)')
    bp.add_argument('--influence_floor_sr', type=float, default=None,
                    help='Floor for sigma_resid threshold (default: theoretical chi(2) p99 ≈ 3.03)')
    bp.add_argument('--influence_floor_sd', type=float, default=3.0,
                    help='Floor for scaled_D threshold (default 3.0)')
    bp.add_argument('--influence_raw_cooks_d', action='store_true',
                    help='Use raw Cook\'s D instead of null-normalised scaled_D=D*N_R/leverage. '
                         'Raw D is biased against sparse images but matches pre-normalisation behaviour.')
    bp.add_argument('--verbose_tests', action='store_true',
                    help='Print per-iteration breakdown of flagged detections by '
                         'Gaia solution type (2p vs 5p/6p) and chip (_hi/_lo)')
    bp.add_argument('--two_tier', action='store_true',
                    help='Enable two-tier alignment/astrometry system: stars that fail '
                         'ok_gaia can still constrain their own astrometry at 3× the '
                         'alignment threshold (use_for_astrom independent of use_for_fit)')
    bp.add_argument('--sparse', action='store_true',
                    help='Use sparse solver (faster for large mosaics)')
    bp.add_argument('--bp3m_images', type=str, nargs='+', default=None,
                    help='Restrict BP3M to these image names')
    bp.add_argument('--bp3m_all_images', action='store_true',
                    help='Use all available images for BP3M, ignoring the '
                         'field_id selection from the HST download step')
    bp.add_argument('--bp3m_remove_images', type=str, nargs='+', default=None,
                    help='Exclude these images from BP3M')
    bp.add_argument('--restrict_filters', type=str, nargs='+', default=None,
                    help='Keep only images with these filters for BP3M')
    bp.add_argument('--restrict_instdet', type=str, nargs='+', default=None,
                    help='Keep only images from these instrument+detector combinations '
                         'for BP3M (e.g. ACSWFC WFC3UVIS)')
    bp.add_argument('--bp3m_min_stars', type=int, default=0,
                    help='Exclude images with fewer than this many Gaia cross-matched '
                         'stars from BP3M (default: 0 = keep all images)')
    bp.add_argument('--epoch_dist_prior', type=str, default=None,
                    help='Comma-separated epoch_distortion.csv paths from '
                         'calibration fields: recentre matching D-group '
                         'priors on the calibration coefficients (requires '
                         '--fit_epoch_distortion).')
    bp.add_argument('--epoch_dist_prior_inflate', type=float, default=2.0)
    bp.add_argument('--pos_corr_table', type=str, default=None,
                    help='Pseudo-GDC centroid-correction table (npz from '
                         'stdpsf_builder/make_pseudo_gdc.py). Applied IN '
                         'MEMORY at catalog load to matching inst/det/filter '
                         'images; nothing on disk is modified.')
    bp.add_argument('--bp3m_results_suffix', type=str, default=None,
                    help='Write the joint-fit outputs to '
                         'BP3M_results_<suffix> instead of BP3M_results '
                         '(A/B tests without overwriting existing results).')
    bp.add_argument('--force_indv_refit', action='store_true',
                    help='With --fit_indv_images_only: refit every image even '
                         'if cached results are current (same matched_gaia '
                         'md5 and fit parameters).')
    bp.add_argument('--fit_indv_images_only', action='store_true',
                    help='Run BP3M separately on each image and save results in '
                         'BP3M_indv_results/{image_name}/. Skips the joint multi-image fit.')
    bp.add_argument('--exclude_2p_from_alignment', action='store_true',
                    help='Exclude Gaia 2-parameter (position-only) stars from the '
                         'image-transformation alignment; images with too few non-2p '
                         'stars are dropped entirely')

    # ── Synthetic tests ───────────────────────────────────────────────────────
    syn = p.add_argument_group('Synthetic tests (requires completed cross-match, Step 4)')
    syn.add_argument('--test_synthetic', action='store_true',
                     help='Run synthetic data test after cross-match. '
                          'Generates synthetic observations from real data, runs BP3M, '
                          'and compares recovered parameters to ground truth.')
    syn.add_argument('--synthetic_draw_from_prior', action='store_true',
                     help='Draw true stellar parameters from Gaia prior N(v_gaia, C_gaia) '
                          'instead of using MAP values as truth (default: MAP values).')
    syn.add_argument('--synthetic_zero_parallax', action='store_true',
                     help='Set true parallax = 0 for all stars.')
    syn.add_argument('--synthetic_true_gaia', action='store_true',
                     help='Feed true stellar parameters directly as the Gaia prior mean '
                          '(zero Gaia measurement noise). Useful for isolating HST noise.')
    syn.add_argument('--synthetic_jitter_sigma', type=float, default=0.0,
                     help='Std dev of Gaussian perturbation added to true transformation '
                          'parameters (default 0 = no perturbation).')
    syn.add_argument('--synthetic_seed', type=int, default=42,
                     help='Random seed for synthetic data generation (default 42).')
    syn.add_argument('--synthetic_only_5p', action='store_true',
                     help='Exclude 2-param Gaia stars (no measured PM/parallax) from '
                          'the synthetic test. Useful for isolating whether 2-param '
                          'stars affect image parameter estimation.')
    syn.add_argument('--synthetic_all_5p_gaia', action='store_true',
                     help='Give 2-param Gaia stars synthetic 5-param Gaia measurements '
                          '(PM+parallax drawn with median errors from real 5-param stars). '
                          'Tests whether BP3M handles all-5p Gaia data correctly; the '
                          'true PM is still drawn from N(0,10²).')
    syn.add_argument('--synthetic_true_pm', type=float, nargs=2,
                     metavar=('PMRA', 'PMDEC'), default=None,
                     help='Override ALL stars true PM: draw from N((PMRA,PMDEC), width²). '
                          'Generates self-consistent catalog = truth + Gaia noise. '
                          'Example: --synthetic_true_pm 5.0 -5.0')
    syn.add_argument('--synthetic_true_pm_width', type=float, default=0.1,
                     help='1σ width of the true PM distribution (mas/yr, default 0.1).')
    syn.add_argument('--synthetic_true_parallax', type=float, default=None,
                     help='Override ALL stars true parallax: draw from N(VAL, width²). '
                          'Use a positive value for physically meaningful parallaxes. '
                          'Example: --synthetic_true_parallax 5.0')
    syn.add_argument('--synthetic_true_parallax_width', type=float, default=0.1,
                     help='1σ width of the true parallax distribution (mas, default 0.1).')

    # ── Pipeline control ──────────────────────────────────────────────────────
    ctl = p.add_argument_group('Pipeline control')
    ctl.add_argument('--output_dir', type=str, default='.',
                     help='Root output directory (default: current directory)')
    ctl.add_argument('--n_processes', type=int, default=-1,
                     help='Number of cores to use (-1 = all available, default)')
    ctl.add_argument('--skip_download', action='store_true',
                     help='Skip Gaia and HST downloads (use existing files)')
    ctl.add_argument('--force_redownload_gaia', action='store_true',
                     help='Re-query Gaia archive even if local CSV already exists')
    ctl.add_argument('--gaia_timeout', type=int, default=300,
                     help='Per-bin Gaia TAP query timeout in seconds (default 300). '
                          'Increase for large fields with slow archive responses.')
    ctl.add_argument('--force_redownload_hst', action='store_true',
                     help='Re-search MAST and re-download HST files even if cached')
    ctl.add_argument('--mast_refresh_days', type=int, default=30, metavar='N',
                     help='Re-query MAST if the cached obs table is older than N days '
                          '(default 30). Set to 0 to always re-query, or use '
                          '--skip_mast_download to never re-query.')
    ctl.add_argument('--skip_mast_download', action='store_true',
                     help='Use cached MAST obs table as-is, regardless of age. '
                          'Useful when re-running alignment without network access.')
    ctl.add_argument('--force_refit_psf', action='store_true',
                     help='Re-run PSF fitting even if catalogs already exist, starting '
                          'from the bare stdpsf (ignores any stored psf_delta.npy). '
                          'Pass --n_psf_iter 2 alongside this to explicitly apply a '
                          'stored delta in the second iteration.')
    ctl.add_argument('--clean_psf', action='store_true',
                     help='Start PSF fitting from the bare stdpsf, ignoring any stored '
                          'psf_delta.npy. Overrides --apply_psf_delta.')
    ctl.add_argument('--apply_psf_delta', action='store_true',
                     help='Load the stored psf_delta.npy (if present) as the starting PSF '
                          'model for the first fitting iteration. By default the bare stdpsf '
                          'is always used.')
    ctl.add_argument('--n_psf_iter', type=int, default=None,
                     help='Number of iterative PSF fitting passes (default: 1). Pass 2 to '
                          'enable the iterative PSF correction (fit → measure δP → re-fit '
                          'with corrected PSF). WARNING: applying δP in a second pass can '
                          'degrade the 2-D pixel-phase distribution for sparse fields; '
                          'only recommended when many bright stars (≳1000) are available.')
    ctl.add_argument('--reclassify_stars', action='store_true',
                     help='Re-run star classification on existing PSF catalogs using the '
                          'current --conc_limit, without re-fitting PSFs. Regenerates '
                          'concentration plots and invalidates the cross-match cache.')
    ctl.add_argument('--remeasure_psf_perturbation', action='store_true',
                     help='Re-measure PSF perturbation on existing catalogs without re-fitting. '
                          'Reconstructs residual images from catalog star models and regenerates '
                          'psf_delta.npy and psf_perturbation.png for each image.')
    ctl.add_argument('--force_rematch', action='store_true',
                     help='Re-run cross-matching even if matched_gaia.csv already exists')
    ctl.add_argument('--force_validate', action='store_true',
                     help='Re-run cross-image validation (regenerate cross_match_catalog.csv) '
                          'without re-running cross-matching')
    ctl.add_argument('--skip_psf', action='store_true',
                     help='Skip PSF fitting (use existing catalogs)')
    ctl.add_argument('--skip_crossmatch', action='store_true',
                     help='Skip cross-matching (use existing matched_gaia.csv)')
    ctl.add_argument('--skip_alignment', action='store_true',
                     help='Skip BP3M alignment (stop after cross-match)')
    ctl.add_argument('--no_plots', action='store_true',
                     help='Suppress all diagnostic plot generation')
    ctl.add_argument('--plot_residuals', action='store_true',
                     help='Generate per-image HST XY residual maps (slow for large fields; off by default)')
    ctl.add_argument('--plot_influence', action='store_true',
                     help='Generate Cook\'s D influence diagnostic plots (slow; off by default)')
    ctl.add_argument('--quiet', action='store_true',
                     help='Non-interactive mode; use defaults without prompts')
    ctl.add_argument('--checkpoint_dir', type=str, default=None,
                     help='Save BP3M checkpoint to this directory for later re-use')
    ctl.add_argument('--single-image', dest='single_image', action='store_true',
                     help='Process images one at a time (serial), giving each image '
                          '--n_processes cores via JAX pmap.  Default is to process '
                          'multiple images simultaneously with one core each, which '
                          'gives much higher throughput when fitting >10 images.')

    return p.parse_args()


_FIELD_IDS_ALL = 'all'   # sentinel: download all without prompting


def _parse_field_ids(raw: list[str] | None):
    """
    Convert raw --field_ids strings to what download_hst_images expects.

    None (not provided)  → None          (show interactive prompt)
    'y' / 'all' / 'yes'  → _FIELD_IDS_ALL  (download all without prompting)
    'n' / 'no'           → [0]           (skip download)
    integers             → list[int]
    """
    if raw is None:
        return None
    joined = ' '.join(raw).strip().lower()
    if joined in ('y', 'yes', 'all'):
        return _FIELD_IDS_ALL
    if joined in ('n', 'no', '0'):
        return [0]
    try:
        return [int(x) for x in raw]
    except ValueError:
        print(f"  WARNING: could not parse --field_ids {raw!r} — downloading all.")
        return _FIELD_IDS_ALL


def _resolve_pointings(args):
    """
    Resolve target coordinates into a list of (ra, dec, search_width, search_height)
    pointings and store as args.pointings.

    Also sets args.ra/dec/search_radius/search_width/search_height to the FIRST
    pointing's values for backward compatibility with code that uses them directly.
    """
    from bp3m.pipeline.download_gaia import resolve_target

    ras  = list(args.ra)  if args.ra  is not None else None
    decs = list(args.dec) if args.dec is not None else None

    # ── Single-pointing from Simbad ───────────────────────────────────────────
    if ras is None or decs is None:
        if args.name is None:
            print("ERROR: provide --name or both --ra and --dec.", file=sys.stderr)
            sys.exit(1)
        if ras is not None or decs is not None:
            print("ERROR: provide both --ra and --dec (or neither).", file=sys.stderr)
            sys.exit(1)
        print(f"Resolving '{args.name}' via Simbad...")
        auto_r = None
        try:
            ra_s, dec_s, auto_r = resolve_target(args.name)
            ras, decs = [ra_s], [dec_s]
            if args.search_radius is None and auto_r is not None:
                print(f"  Auto search radius from Simbad: {auto_r:.3f} deg")
        except Exception as exc:
            print(f"  Simbad lookup failed: {exc}")
            if not args.quiet:
                ras  = [float(input('  Enter R.A. (degrees): '))]
                decs = [float(input('  Enter Dec. (degrees): '))]
            else:
                print("ERROR: Could not resolve target coordinates.", file=sys.stderr)
                sys.exit(1)
        if args.search_radius is None and auto_r is not None:
            args.search_radius = [auto_r]

    # ── Validate ra/dec lengths ────────────────────────────────────────────────
    if len(ras) != len(decs):
        print(f"ERROR: --ra ({len(ras)} values) and --dec ({len(decs)} values) "
              f"must have the same number of values.", file=sys.stderr)
        sys.exit(1)
    n = len(ras)

    # Multi-pointing requires an explicit --name
    if n > 1 and args.name is None:
        print("ERROR: --name is required when multiple --ra/--dec values are given.",
              file=sys.stderr)
        sys.exit(1)

    # ── Search radius per pointing ────────────────────────────────────────────
    srs_raw = args.search_radius  # None | list[float]
    if srs_raw is None:
        if not args.quiet:
            _sr = float(
                input('Search radius not set. Enter value in degrees [0.25]: ')
                or 0.25)
        else:
            _sr = 0.25
        srs = [_sr] * n
    elif len(srs_raw) == 1:
        srs = list(srs_raw) * n       # broadcast
    elif len(srs_raw) == n:
        srs = list(srs_raw)
    else:
        print(f"ERROR: --search_radius must have 1 or {n} value(s), "
              f"got {len(srs_raw)}.", file=sys.stderr)
        sys.exit(1)

    # ── Build (ra, dec, sw, sh) per pointing ──────────────────────────────────
    pointings: list[tuple[float, float, float, float]] = []
    for ra_i, dec_i, sr_i in zip(ras, decs, srs):
        sw_i = (args.search_width  if args.search_width  is not None
                else 2.0 * sr_i / max(abs(np.cos(np.deg2rad(dec_i))), 0.01))
        sh_i = (args.search_height if args.search_height is not None
                else 2.0 * sr_i)
        pointings.append((float(ra_i), float(dec_i), float(sw_i), float(sh_i)))

    # ── Mutate args for backward compat (primary = first pointing) ────────────
    args.ra            = pointings[0][0]
    args.dec           = pointings[0][1]
    args.search_width  = pointings[0][2]
    args.search_height = pointings[0][3]
    args.search_radius = srs[0]
    args.pointings     = pointings

    if args.name is None:
        args.name = f"ra_{args.ra:.3f}_dec_{args.dec:.3f}"
    args.name = args.name.replace(' ', '_')

    print(f"\n  Field:  {args.name}")
    if n == 1:
        print(f"  Centre: ({args.ra:.5f}, {args.dec:.5f}) deg")
        print(f"  Box:    {args.search_width:.4f} × {args.search_height:.4f} deg")
    else:
        print(f"  Multi-pointing ({n} pointings):")
        for i, (ra_i, dec_i, sw_i, sh_i) in enumerate(pointings):
            print(f"    [{i+1}] RA={ra_i:.5f}  Dec={dec_i:+.5f}  "
                  f"box={sw_i:.4f}×{sh_i:.4f} deg")



def _indv_pool_init():
    """Initializer for the --fit_indv_images_only worker pool: cap BLAS
    threading (each worker solves a small single-image system; n_processes
    workers x multi-threaded BLAS would oversubscribe the box) and force a
    headless matplotlib backend.  Respects explicit user settings."""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


def _run_indv_one(img_name, run_kw, indv_root, extra_cfg):
    """Worker for --fit_indv_images_only.

    Runs a single-image alignment fit with all solver output redirected to
    BP3M_indv_results/{image}/processing_log.txt (mirroring the per-image
    logs written by the PSF-fitting and cross-match steps), so the parent
    can print one summary line per image instead of the full solver output.
    Returns (image, err, n_align_used, n_astrom_used, n_detections, seconds).
    """
    import contextlib
    import time
    import traceback

    t0 = time.time()
    out_dir = Path(indv_root) / img_name
    out_dir.mkdir(parents=True, exist_ok=True)
    err = None
    n_fit = n_astrom = n_det = -1
    with open(out_dir / "processing_log.txt", "w", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from bp3m.pipeline.run_alignment import run_alignment
            run_alignment(images=[img_name], bp3m_dir=out_dir,
                          extra_run_config=extra_cfg,
                          min_align_demote=0,  # never freeze a single-image fit
                          **run_kw)
        except Exception as exc:  # full traceback goes to the log
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
    if err is None:
        try:
            with np.load(out_dir / "use_for_fit.npz") as zf:
                n_fit = int(sum(int(np.sum(zf[k] > 0)) for k in zf.files))
                n_det = int(sum(int(np.asarray(zf[k]).size) for k in zf.files))
            with np.load(out_dir / "use_for_astrom.npz") as za:
                n_astrom = int(sum(int(np.sum(za[k] > 0)) for k in za.files))
        except Exception:
            pass
    return img_name, err, n_fit, n_astrom, n_det, time.time() - t0


def main():
    args = _parse_args()

    # Wire --n_processes to thread limits.
    # In parallel image mode (default): workers set their own limits; the main
    # process only needs env vars for any numpy work it does before spawning.
    # In single-image mode: also configure JAX pmap devices so the single image
    # can use n_processes virtual CPU devices for fit_batch_jax.
    if args.n_processes != -1:
        _n = str(args.n_processes)
        os.environ['OMP_NUM_THREADS']      = _n
        os.environ['OPENBLAS_NUM_THREADS'] = _n
        os.environ['MKL_NUM_THREADS']      = _n
        try:
            import threadpoolctl as _tpc
            _tpc.threadpool_limits(limits=args.n_processes)
        except ImportError:
            pass
        if args.single_image:
            # Only initialize JAX in the main process for single-image mode.
            # In parallel mode workers import JAX themselves with jax_num_cpu_devices=1.
            try:
                import jax as _jax
                _jax.config.update('jax_num_cpu_devices', args.n_processes)
            except Exception:
                pass

    print("=" * 55)
    print("BP3M Analysis")
    print("=" * 55)

    _resolve_pointings(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field = args.name

    # ── Gaia directory and per-pointing CSV paths ─────────────────────────────
    from bp3m.pipeline.download_gaia import _cache_stem
    _gaia_dir = output_dir / field / 'Gaia'

    def _gaia_csv_path_for(ra_i, dec_i, sw_i, sh_i):
        stem = _cache_stem(field, ra_i, dec_i, sw_i, sh_i,
                           args.min_gmag, args.max_gmag)
        p = _gaia_dir / f"{stem}.csv"
        return p if p.exists() else None

    # ── Step 1: Download Gaia (one query per pointing) ────────────────────────
    gaia_df = None
    gaia_csv_paths: list[Path] = []

    if not args.skip_download:
        from bp3m.pipeline.download_gaia import download_gaia
        _n_pt = len(args.pointings)
        for _pi, (_ra_i, _dec_i, _sw_i, _sh_i) in enumerate(args.pointings):
            if _n_pt > 1:
                print(f"\n  [Gaia pointing {_pi+1}/{_n_pt}] "
                      f"RA={_ra_i:.5f}  Dec={_dec_i:+.5f}")
            _gdf_i = download_gaia(
                ra=_ra_i, dec=_dec_i,
                search_width=_sw_i, search_height=_sh_i,
                output_dir=output_dir, field_name=field,
                min_gmag=args.min_gmag, max_gmag=args.max_gmag,
                source_table=args.source_table,
                sigma_flux_excess=args.sigma_flux_excess,
                only_5p=args.only_5p,
                n_processes=args.n_processes,
                query_timeout=args.gaia_timeout,
                force_redownload=args.force_redownload_gaia,
                quiet=args.quiet,
            )
            _p_i = _gaia_csv_path_for(_ra_i, _dec_i, _sw_i, _sh_i)
            if _p_i is not None:
                gaia_csv_paths.append(_p_i)
            if _gdf_i is not None and len(_gdf_i) > 0:
                if gaia_df is None:
                    gaia_df = _gdf_i
                else:
                    import pandas as _pd_gaia
                    gaia_df = (_pd_gaia.concat([gaia_df, _gdf_i], ignore_index=True)
                               .drop_duplicates('source_id')
                               .reset_index(drop=True))

    # Resolve any Gaia CSV paths not yet found (skip_download path)
    for _ra_i, _dec_i, _sw_i, _sh_i in args.pointings:
        _p_i = _gaia_csv_path_for(_ra_i, _dec_i, _sw_i, _sh_i)
        if _p_i is not None and _p_i not in gaia_csv_paths:
            gaia_csv_paths.append(_p_i)

    # Write the Gaia file list sidecar so downstream tools know which files belong
    import json as _json_gaia
    _gaia_dir.mkdir(parents=True, exist_ok=True)
    _gaia_files_json = _gaia_dir / f"{field}_gaia_files.json"
    _gaia_files_json.write_text(
        _json_gaia.dumps([str(p) for p in gaia_csv_paths], indent=2))

    # For alignment etc., pass explicit list (avoids stale-file glob)
    # Single-pointing: pass the one path; multi-pointing: pass the list.
    gaia_csv_path: Path | list | None = (
        gaia_csv_paths[0] if len(gaia_csv_paths) == 1
        else gaia_csv_paths if gaia_csv_paths
        else None
    )

    # ── Step 1b: QSO anchor vetting (one run per pointing) ───────────────────
    from bp3m.pipeline.qso_vetting import qso_anchors_path as _qap
    _qso_anchors_csvs: list[Path] = []
    _any_qso_new = False
    for _ra_i, _dec_i, _sw_i, _sh_i in args.pointings:
        _qso_csv_i = _qap(_gaia_dir, field, _ra_i, _dec_i, _sw_i, _sh_i)
        if not args.no_qso_anchors and not _qso_csv_i.exists():
            if not _any_qso_new:
                from bp3m.pipeline.qso_vetting import vet_qso_candidates
                print("\n" + "─"*50)
                print("Step 1b: QSO anchor vetting")
                print("─"*50)
                _any_qso_new = True
            try:
                vet_qso_candidates(
                    field_name=field, output_dir=output_dir,
                    ra=_ra_i, dec=_dec_i,
                    search_width=_sw_i, search_height=_sh_i,
                    lib_dir=Path(args.lib_dir) if args.lib_dir else None,
                )
            except Exception as _qe:
                print(f"  WARNING: QSO vetting failed — {_qe}")
        if _qso_csv_i.exists():
            _qso_anchors_csvs.append(_qso_csv_i)
    if not _any_qso_new and _qso_anchors_csvs:
        print(f"\n  [Step 1b] QSO anchors already cached "
              f"({len(_qso_anchors_csvs)} pointing(s))")

    # Alignment accepts a single CSV or a list
    _qso_anchors_csv: Path | list | None = (
        _qso_anchors_csvs[0] if len(_qso_anchors_csvs) == 1
        else _qso_anchors_csvs if _qso_anchors_csvs
        else None
    )
    # Helper: True when the qso anchors value is non-empty (Path or non-empty list)
    def _qso_exists(v):
        if v is None:
            return False
        if isinstance(v, list):
            return len(v) > 0
        return v.exists()

    # ── Step 1c: Download DELVE catalogue (one query per pointing) ──────────
    delve_csv_path: "Path | None" = None
    if args.use_delve and not args.skip_download:
        from bp3m.pipeline.download_delve import download_delve as _dl_delve
        print("\n" + "─"*50)
        print("Step 1c: DELVE proper-motion catalogue")
        print("─"*50)
        _delve_frames = []
        for _pi, (_ra_i, _dec_i, _sw_i, _sh_i) in enumerate(args.pointings):
            _ddf = _dl_delve(
                ra=_ra_i, dec=_dec_i,
                search_width=_sw_i, search_height=_sh_i,
                output_dir=output_dir, field_name=field,
                delve_dir=args.delve_dir,
                force_redownload=args.force_redownload_delve,
            )
            if _ddf is not None:
                _delve_frames.append(_ddf)
        if _delve_frames:
            # point downstream steps at the first (or only) DELVE CSV
            from bp3m.pipeline.download_delve import _cache_stem as _dlve_stem
            _delve_dir_out = output_dir / field / "DELVE"
            _delve_csvs = sorted(_delve_dir_out.glob("*_delve.csv"))
            if _delve_csvs:
                delve_csv_path = _delve_csvs[0]
        else:
            print("  No DELVE tiles found for this field — DELVE steps disabled.")
    elif args.use_delve:
        # skip_download path: find any existing DELVE CSV
        _delve_dir_out = output_dir / field / "DELVE"
        _delve_csvs = sorted(_delve_dir_out.glob("*_delve.csv")) if _delve_dir_out.exists() else []
        if _delve_csvs:
            delve_csv_path = _delve_csvs[0]

    # ── Step 2: Download HST ─────────────────────────────────────────────────
    if not args.skip_download:
        from bp3m.pipeline.download_hst import download_hst_images
        # Load Gaia catalog for footprint star counts if not already in memory
        if gaia_df is None and gaia_csv_paths:
            from bp3m.pipeline.explore_utils import load_gaia_catalog
            _frames = []
            for _gp in gaia_csv_paths:
                try:
                    _frames.append(load_gaia_catalog(_gp))
                except Exception:
                    pass
            if _frames:
                import pandas as _pd_g2
                gaia_df = (_pd_g2.concat(_frames, ignore_index=True)
                           .drop_duplicates('source_id').reset_index(drop=True)
                           if len(_frames) > 1 else _frames[0])
        if gaia_df is None:
            # Last-resort fallback: glob any Gaia CSV
            _candidates = sorted(_gaia_dir.glob("*_gaia.csv"))
            if _candidates:
                from bp3m.pipeline.explore_utils import load_gaia_catalog
                gaia_df = load_gaia_catalog(_candidates[0])
        _extra_pt = args.pointings[1:] if len(args.pointings) > 1 else None
        download_hst_images(
            ra=args.ra, dec=args.dec,
            search_width=args.search_width, search_height=args.search_height,
            output_dir=output_dir, field_name=field,
            hst_filters=args.hst_filters,
            t_exptime_min=args.hst_exptime_min,
            t_exptime_max=args.hst_exptime_max,
            time_baseline_days=args.time_baseline,
            obs_date_min=args.obs_date_min,
            obs_date_max=args.obs_date_max,
            im_type=args.hst_im_type,
            telescope=args.telescope,
            instruments=args.instruments,
            lib_dir=Path(args.lib_dir),
            gaia_df=gaia_df,
            field_ids=_parse_field_ids(args.field_ids),
            quiet=args.quiet,
            force_redownload=args.force_redownload_hst,
            mast_refresh_days=args.mast_refresh_days,
            skip_mast_download=args.skip_mast_download,
            n_processes=args.n_processes,
            extra_pointings=_extra_pt,
            delve_csv_path=delve_csv_path,
        )

    # Read manifest of selected obsids written by step 2 (persists across runs)
    import json as _json
    _hst_dir = output_dir / field / args.telescope.upper()
    _manifest = _hst_dir / f"{field}_selected_obsids.json"
    _failed_manifest = _hst_dir / f"{field}_failed_obsids.json"
    _selected_obsids: list[str] | None = None
    if _manifest.exists():
        try:
            _selected_obsids = _json.loads(_manifest.read_text())
        except Exception:
            pass
    # If the failed manifest doesn't exist yet (e.g. step 2 was skipped on this
    # run and the failed-obs check has never been written), scan on-disk FLC
    # files now so downstream steps never accidentally process bad images.
    if not _failed_manifest.exists() and _manifest.exists() and _selected_obsids:
        from bp3m.pipeline.download_hst import _check_exptime as _cet
        _mast_root = _hst_dir / "mastDownload" / args.telescope.upper()
        _scanned_failed: dict[str, str] = {}
        for _oid in list(_selected_obsids):
            _flc = _mast_root / _oid / f"{_oid}_{args.hst_im_type.lstrip('_')}.fits"
            if _flc.exists():
                _reason = _cet(_flc)
                if _reason:
                    _scanned_failed[_oid] = _reason
        if _scanned_failed:
            # Remove from selected list and write both manifests
            _selected_obsids = [o for o in _selected_obsids if o not in _scanned_failed]
            _manifest.write_text(_json.dumps(_selected_obsids, indent=2))
            _failed_manifest.write_text(_json.dumps(_scanned_failed, indent=2))

    if _failed_manifest.exists():
        try:
            _failed = _json.loads(_failed_manifest.read_text())
            if _failed:
                print(f"\nNOTE: {len(_failed)} image(s) are failed observations and will be "
                      f"skipped in all pipeline steps:")
                for _oid, _reason in sorted(_failed.items()):
                    print(f"  {_oid}: {_reason}")
        except Exception:
            pass

    # ── Check that we have images before continuing ───────────────────────────
    if _selected_obsids is not None and len(_selected_obsids) == 0:
        print("\nNo HST images available to process. Check your search "
              "parameters (filters, instruments, search radius, dates).")
        return

    # ── Resolve active image set for steps 3 onwards ─────────────────────────
    # --bp3m_images restricts ALL downstream steps (PSF, cross-match, BP3M),
    # not just the alignment step.  Resolve it now so every step uses the same
    # filtered list.  _selected_obsids is the full set from the download
    # manifest; _bp3m_images is the (possibly narrower) working set.
    _bp3m_images = args.bp3m_images
    if _bp3m_images is None and not args.bp3m_all_images and _selected_obsids is not None:
        _bp3m_images = _selected_obsids
    # _restrict is what we pass as restrict_to_obsids to every step from 3 on.
    _restrict = _bp3m_images  # may be None (→ process all on-disk images)

    # ── Filter by --restrict_instdet for steps 3 onwards ─────────────────────
    # BP3M does its own instdet filtering from image metadata, but steps 3/4
    # only know about obsids — so narrow _restrict here using the cached MAST
    # obs table (data_products CSV joined with obs CSV).  Falls back to reading
    # FITS headers if the CSVs aren't present.
    if args.restrict_instdet and _restrict is not None:
        import pandas as _pd
        _keep_id = {s.upper().replace('/', '') for s in args.restrict_instdet}

        def _instdet_key(name: str) -> str:
            return name.upper().replace('/', '')

        _obsid_to_instdet: dict[str, str] = {}
        _dp_csv = _hst_dir / f"{field}_data_products.csv"
        _obs_csv = _hst_dir / f"{field}_obs.csv"
        if _dp_csv.exists() and _obs_csv.exists():
            try:
                _dp  = _pd.read_csv(str(_dp_csv))
                _obs = _pd.read_csv(str(_obs_csv))
                _flc = _dp[_dp['productFilename'].str.endswith('_flc.fits', na=False)
                           | _dp['productFilename'].str.endswith('_flt.fits', na=False)]
                _merged = _flc.merge(
                    _obs[['obsid', 'instrument_name']],
                    left_on='parent_obsid', right_on='obsid', how='left')
                for _, _row in _merged.iterrows():
                    _oid = str(_row['obs_id'])
                    _inst = str(_row.get('instrument_name', ''))
                    if _inst and _inst != 'nan':
                        _obsid_to_instdet[_oid] = _instdet_key(_inst)
            except Exception as _e:
                print(f"  WARNING: could not read MAST CSVs for instdet filter: {_e}")
        if not _obsid_to_instdet:
            # Fallback: read FITS headers.
            _mast_root = _hst_dir / "mastDownload" / args.telescope.upper()
            _im_suffix = args.hst_im_type.lstrip('_') + '.fits'
            for _oid in _restrict:
                _flc_path = _mast_root / _oid / f"{_oid}_{_im_suffix}"
                if _flc_path.exists():
                    try:
                        from astropy.io import fits as _fits_hdr
                        with _fits_hdr.open(str(_flc_path)) as _h:
                            _inst = _h[0].header.get('INSTRUME', '').strip()
                            _det  = _h[0].header.get('DETECTOR', '').strip()
                        _obsid_to_instdet[_oid] = (_inst + _det).upper()
                    except Exception:
                        pass

        if _obsid_to_instdet:
            _before = len(_restrict)
            _restrict = [o for o in _restrict
                         if _obsid_to_instdet.get(o, '') in _keep_id]
            _bp3m_images = _restrict
            print(f"  --restrict_instdet {args.restrict_instdet}: "
                  f"{_before} → {len(_restrict)} images")
        else:
            print("  WARNING: --restrict_instdet specified but could not determine "
                  "instrument for any obsid — skipping filter for steps 3/4.")

    # ── Step 3: PSF fitting ───────────────────────────────────────────────────
    if not args.skip_psf:
        from bp3m.pipeline.psf_fitting import run_psf_fitting
        run_psf_fitting(
            output_dir=output_dir, field_name=field,
            lib_dir=Path(args.lib_dir),
            telescope=args.telescope,
            im_type=args.hst_im_type,
            n_processes=args.n_processes,
            verbose=not args.quiet,
            force_refit=args.force_refit_psf,
            clean_psf=args.clean_psf,
            apply_psf_delta=args.apply_psf_delta,
            n_psf_iter=args.n_psf_iter,
            parallel=not args.single_image,
            fmin=args.fmin, fmin_thresh=args.fmin_thresh, mag_st_max=args.mag_st_max, hmin=args.hmin,
            n_passes=args.n_passes, n_discovery_passes=args.n_discovery_passes,
            max_iter_fit=args.psf_max_iter,
            sat_threshold=args.sat_threshold, conc_limit=args.conc_limit,
            restrict_to_obsids=_restrict,
        )

    if args.reclassify_stars:
        from bp3m.pipeline.psf_fitting import reclassify_psf_catalogs
        reclassify_psf_catalogs(
            output_dir=output_dir, field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            conc_limit=args.conc_limit,
            restrict_to_obsids=_restrict,
            lib_dir=Path(args.lib_dir) if args.lib_dir else None,
            n_processes=args.n_processes,
            parallel=not args.single_image,
        )

    if args.remeasure_psf_perturbation:
        from bp3m.pipeline.psf_fitting import remeasure_psf_perturbation
        remeasure_psf_perturbation(
            output_dir=output_dir, field_name=field,
            lib_dir=Path(args.lib_dir),
            telescope=args.telescope,
            im_type=args.hst_im_type,
            restrict_to_obsids=_restrict,
            verbose=not args.quiet,
        )

    # ── Step 4: Cross-matching ────────────────────────────────────────────────
    if not args.skip_crossmatch:
        from bp3m.pipeline.cross_match import run_cross_match
        run_cross_match(
            output_dir=output_dir, field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            n_processes=args.n_processes,
            hst_pix_floor=args.cross_match_pix_floor,
            min_matches=args.min_matches,
            max_mag_diff=args.max_mag_diff,
            scale_sweep=args.scale_sweep,
            discovery_max_offset=args.discovery_max_offset,
            use_resid_floor=args.auto_resid_floor,
            force_rematch=args.force_rematch,
            restrict_to_obsids=_restrict,
            lib_dir=Path(args.lib_dir) if args.lib_dir else None,
            run_qso_vetting=False,
            force_validate=getattr(args, 'force_validate', False),
            prior_sigma_rot_deg=args.prior_sigma_rot_deg,
            prior_sigma_scale=args.prior_sigma_scale,
            prior_sigma_skew=args.prior_sigma_skew,
            init_resid_max=args.xmatch_init_resid_max,
        )
    elif getattr(args, 'force_validate', False):
        from bp3m.pipeline.cross_match import _validate_catalog_if_needed
        print("\n" + "─"*50)
        print("Step 4: Cross-image validation (forced)")
        print("─"*50)
        _validate_catalog_if_needed(field, output_dir, force=True)

    # ── Step 4c: DELVE cross-matching (optional) ─────────────────────────────
    if delve_csv_path is not None and not args.skip_delve_crossmatch:
        from bp3m.pipeline.cross_match_delve import run_cross_match_delve
        run_cross_match_delve(
            output_dir=output_dir, field_name=field,
            delve_csv_path=delve_csv_path,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            n_processes=args.n_processes,
            hst_pix_floor=args.cross_match_pix_floor,
            min_matches=args.min_matches,
            max_mag_diff=args.max_mag_diff,
            scale_sweep=args.scale_sweep,
            discovery_max_offset=args.discovery_max_offset,
            use_resid_floor=args.auto_resid_floor,
            force_rematch=args.force_rematch_delve,
            restrict_to_obsids=_restrict,
            lib_dir=Path(args.lib_dir) if args.lib_dir else None,
            prior_sigma_rot_deg=args.prior_sigma_rot_deg,
            prior_sigma_scale=args.prior_sigma_scale,
            prior_sigma_skew=args.prior_sigma_skew,
            init_resid_max=args.xmatch_init_resid_max,
        )

        # Auto-validate: cross_match_catalog.csv must be regenerated after DELVE
        # crossmatch so it includes DELVE PM columns (used by the solver for
        # Gaia+DELVE PM priors and correct trustworthiness counts).  This is
        # needed when switching from a Gaia-only run to --use_delve without
        # --force_validate.
        import pandas as _pd_tmp
        from bp3m.pipeline.cross_match import _validate_catalog_if_needed
        _cat_path = Path(output_dir) / field / "cross_match_catalog.csv"
        _needs_delve_validate = not _cat_path.exists()
        if not _needs_delve_validate:
            _hdr = _pd_tmp.read_csv(_cat_path, nrows=0)
            if 'delve_pmra' not in _hdr.columns:
                print("  cross_match_catalog.csv lacks DELVE columns — "
                      "re-validating to merge DELVE data...")
                _needs_delve_validate = True
        if _needs_delve_validate:
            _validate_catalog_if_needed(field, output_dir, force=True)

    # ── Step 4b: Gaia DR4 epoch astrometry (optional) ────────────────────────
    _gaia_epoch_obs_for_solver: dict | None = None

    if getattr(args, 'use_gaia_dr4_epoch', False):
        from bp3m.pipeline.download_gaia_epoch import (
            collect_matched_source_ids,
            download_epoch_astrometry,
            prepare_epoch_obs_for_solver,
            merge_epoch_solutions_into_catalog,
        )
        import glob as _glob
        import pandas as _pd
        print("\n" + "=" * 55)
        print("Step 4b: Gaia DR4 epoch astrometry")
        print("=" * 55)

        matched_sids = collect_matched_source_ids(output_dir)
        print(f"  Cross-matched sources: {len(matched_sids)}")

        # epoch cache is stored at output_dir/Gaia/epoch (without field),
        # matching the path used by run_gaia_dr4_epoch
        epoch_dir = output_dir / "Gaia" / "epoch"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        epoch_data = download_epoch_astrometry(
            matched_sids,
            epoch_cache_dir=epoch_dir,
            access=args.dr4_access,
            prerelease_votable=args.dr4_prerelease_votable,
            data_release=args.dr4_data_release,
            credentials_file=args.dr4_credentials,
            n_workers=min(args.n_processes, 8) if args.n_processes > 0 else 4,
            force=args.force_rerun_dr4_epoch,
        )
        print(f"  Epoch data available: {len(epoch_data)} sources")

        if epoch_data:
            # Load the Gaia summary catalog to extract AGIS PM+parallax values.
            # Prefer the gaia_files.json sidecar (multi-pointing aware) over glob.
            _gaia_files_json_ep = output_dir / field / "Gaia" / f"{field}_gaia_files.json"
            if _gaia_files_json_ep.exists():
                try:
                    import json as _json_ep
                    _gaia_csvs_ep = json.loads(_gaia_files_json_ep.read_text())
                except Exception:
                    _gaia_csvs_ep = []
            else:
                _gaia_csvs_ep = sorted(_glob.glob(
                    str(output_dir / field / "Gaia" / "*_gaia.csv")))
            if _gaia_csvs_ep:
                _gdf_parts = [_pd.read_csv(p) for p in _gaia_csvs_ep]
                _gdf = (_pd.concat(_gdf_parts, ignore_index=True)
                        .drop_duplicates("source_id") if len(_gdf_parts) > 1
                        else _gdf_parts[0])
                _gdf["source_id"] = _gdf["source_id"].astype("int64")
                _gaia_epoch_obs_for_solver = prepare_epoch_obs_for_solver(
                    epoch_data, _gdf,
                )
                n_ep = len(_gaia_epoch_obs_for_solver)
                print(f"  Precomputed AL obs for {n_ep} sources "
                      f"(will be injected into BP3M solve)")
            else:
                print("  WARNING: no *_gaia.csv found — cannot prepare AL obs")

    # ── Step 5a: Synthetic data generation (optional) ─────────────────────────

    if args.test_synthetic:
        from bp3m.pipeline.synthetic import generate_synthetic_data, compare_synthetic_results

        # Build a unique subdirectory name so different configurations don't
        # overwrite each other (e.g. 'synthetic_only5p_seed43').
        _syn_parts = ["synthetic"]
        if args.synthetic_only_5p:
            _syn_parts.append("only5p")
        if getattr(args, 'synthetic_all_5p_gaia', False):
            _syn_parts.append("all5pgaia")
        _true_pm = getattr(args, 'synthetic_true_pm', None)
        _true_plx = getattr(args, 'synthetic_true_parallax', None)
        if _true_pm is not None:
            _syn_parts.append(f"pm{_true_pm[0]:g}_{_true_pm[1]:g}"
                              .replace('-', 'm').replace('.', 'p'))
        if _true_plx is not None:
            _syn_parts.append(f"plx{_true_plx:g}".replace('-', 'm').replace('.', 'p'))
        if _bp3m_images is not None:
            _syn_parts.append(f"n{len(_bp3m_images)}")
        if args.synthetic_seed != 42:
            _syn_parts.append(f"seed{args.synthetic_seed}")
        syn_name = "_".join(_syn_parts)

        print("\n" + "=" * 55)
        print(f"Synthetic test — generating observations → {syn_name}/")
        print("=" * 55)
        generate_synthetic_data(
            output_dir=output_dir,
            field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            draw_from_prior=args.synthetic_draw_from_prior,
            zero_parallax=args.synthetic_zero_parallax,
            true_gaia=args.synthetic_true_gaia,
            jitter_sigma=args.synthetic_jitter_sigma,
            seed=args.synthetic_seed,
            only_5p=args.synthetic_only_5p,
            all_5p_gaia=getattr(args, 'synthetic_all_5p_gaia', False),
            true_pm_center=(tuple(_true_pm) if _true_pm is not None else None),
            true_pm_width=args.synthetic_true_pm_width,
            true_parallax_center=_true_plx,
            true_parallax_width=args.synthetic_true_parallax_width,
            images=_bp3m_images,
            syn_name=syn_name,
        )

    # ── Step 5: Bayesian alignment ────────────────────────────────────────────
    if not args.skip_alignment:
        from bp3m.pipeline.run_alignment import run_alignment

        if args.fit_indv_images_only:
            # Discover the filtered image list (mirrors run_alignment's filtering).
            from bp3m.data_loader_flc import load_image_data_flc
            from bp3m.data_loader import build_index_maps

            _imgs_all, _spi_all, _gaia_all = load_image_data_flc(
                output_dir, field, pos_err_floor=args.bp3m_pos_err_floor,
                gaia_csv=gaia_csv_path)
            _, _indv_names, _ = build_index_maps(_spi_all, _gaia_all)

            if _bp3m_images is not None:
                _req = set(_bp3m_images)
                _indv_names = [n for n in _indv_names if n in _req]
            if args.bp3m_remove_images:
                _drop = set(args.bp3m_remove_images)
                _indv_names = [n for n in _indv_names if n not in _drop]
            if args.restrict_filters:
                _kf = {f.upper() for f in args.restrict_filters}
                _indv_names = [n for n in _indv_names
                               if _imgs_all[n].get('filter', '').upper() in _kf]
            if args.restrict_instdet:
                _kid = {s.upper() for s in args.restrict_instdet}
                _indv_names = [
                    n for n in _indv_names
                    if (_imgs_all[n].get('instrument', '') +
                        _imgs_all[n].get('detector', '')).upper() in _kid
                ]
            if args.bp3m_min_stars > 0:
                _indv_names = [n for n in _indv_names
                               if len(_spi_all[n]) >= args.bp3m_min_stars]

            _indv_root = output_dir / field / "BP3M_indv_results"

            def _indv_extra_cfg(_img_name):
                """Record exact match provenance in each indv run_config so
                --use_indv_outputs can verify the joint fit sees the same
                cross-match outputs (md5, not mtime heuristics)."""
                import hashlib as _hl
                _m = (output_dir / field / 'HST' / 'mastDownload' / 'HST'
                      / _img_name / 'matched_gaia.csv')
                _cfg = {}
                if _m.exists():
                    _cfg['matched_gaia_md5'] = _hl.md5(_m.read_bytes()).hexdigest()
                    _cfg['matched_gaia_mtime'] = _m.stat().st_mtime
                return _cfg
            _run_kw = dict(
                output_dir=output_dir, field_name=field,
                n_iter=args.n_bp3m_iter,
                n_samples=args.n_samples,
                mcmc_posteriors=args.mcmc_posteriors,
                clip_sigma=args.bp3m_clip_sigma,
                poly_order=args.poly_order,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                two_phase_align=args.two_phase_align,
                fit_epoch_distortion=args.fit_epoch_distortion,
                epoch_dist_order=args.epoch_dist_order,
                epoch_gap_days=args.epoch_gap_days,
                epoch_dist_sigma=args.epoch_dist_sigma,
                epoch_breaks=args.epoch_breaks,
                epoch_dist_min_images=args.epoch_dist_min_images,
                epoch_dist_groupby=args.epoch_dist_groupby,
                use_sparse=args.sparse,
                no_plots=args.no_plots,
                remove_images=None,
                restrict_filters=None,
                restrict_instdet=None,
                bp3m_min_stars=0,
                checkpoint_dir=None,
                use_influence_clip=not args.no_influence_clip,
                prior_sigma_rot_deg=args.prior_sigma_rot_deg,
                prior_sigma_scale=args.prior_sigma_scale,
                prior_sigma_skew=args.prior_sigma_skew,
                prior_sigma_pointing=args.prior_sigma_pointing,
                inflate_alpha_max=args.inflate_alpha_max,
                influence_k=args.influence_k,
                influence_floor_sr=args.influence_floor_sr,
                influence_floor_sd=args.influence_floor_sd,
                influence_raw_cooks_d=args.influence_raw_cooks_d,
                verbose_tests=args.verbose_tests,
                use_two_tier=args.two_tier,
                no_align_prior=args.no_align_prior,
                pos_err_floor=args.bp3m_pos_err_floor,
                plot_residuals=args.plot_residuals,
                plot_influence=args.plot_influence,
                gaia_csv=gaia_csv_path,
                qso_anchors_csv=(_qso_anchors_csv
                                 if _qso_exists(_qso_anchors_csv) else None),
                use_delve=args.use_delve,
                delve_use_for_align=args.delve_use_for_align,
            )

            # ── Resume: skip images whose indv results are current ─────────
            # (same matched_gaia.csv md5 — recorded by _indv_extra_cfg — and
            # the same key fit parameters in run_config.json).
            import json as _json
            _cfg_map = {_img: _indv_extra_cfg(_img) for _img in _indv_names}
            _want_params = {
                'n_iter': args.n_bp3m_iter, 'n_samples': args.n_samples,
                'clip_sigma': args.bp3m_clip_sigma,
                'poly_order': args.poly_order,
                'split_ccd': not args.no_split_ccd,
                'inflate_hst_errors': not args.no_inflate_hst_errors,
                'two_phase_align': args.two_phase_align,
                'pos_err_floor': args.bp3m_pos_err_floor,
            }

            def _indv_cached(_img):
                _d = _indv_root / _img
                if not ((_d / 'run_config.json').exists()
                        and (_d / 'stellar_astrometry.csv').exists()):
                    return False
                try:
                    _cfg = _json.loads((_d / 'run_config.json').read_text())
                except Exception:
                    return False
                _md5 = _cfg_map[_img].get('matched_gaia_md5')
                if _md5 is None or _cfg.get('matched_gaia_md5') != _md5:
                    return False
                return all(_cfg.get(_k) == _v
                           for _k, _v in _want_params.items())

            _n_all = len(_indv_names)
            if not args.force_indv_refit:
                _cached = [n for n in _indv_names if _indv_cached(n)]
                _indv_names = [n for n in _indv_names if n not in set(_cached)]
                if _cached:
                    print(f"  {len(_cached)}/{_n_all} images cached "
                          f"(md5 + fit params match) — skipping; "
                          f"--force_indv_refit to redo")

            from datetime import datetime as _dt
            _n_total = len(_indv_names)
            _n_proc = max(1, min(args.n_processes, max(_n_total, 1)))
            print("\n" + "─"*50)
            print(f"Individual image fitting: {_n_total} images "
                  f"({_n_proc} process{'es' if _n_proc > 1 else ''})")
            print(f"Output: {_indv_root}")
            print("Per-image logs: {image}/processing_log.txt")
            print(f"Start: {_dt.now():%Y-%m-%d %H:%M:%S}")
            print("─"*50, flush=True)

            _n_ok, _n_fail, _n_done = 0, 0, 0

            def _indv_report(_res):
                _img_r, _err, _n_fit, _n_astrom, _n_det, _dt_s = _res
                _im = _imgs_all.get(_img_r, {})
                _tag = "/".join(str(_im.get(_k) or '?')
                                for _k in ('instrument', 'detector', 'filter'))
                _ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
                if _err:
                    print(f"  [{_ts}] {_img_r} ({_n_done}/{_n_total}) {_tag}  "
                          f"FAILED: {_err} — see {_img_r}/processing_log.txt",
                          flush=True)
                    return False
                print(f"  [{_ts}] {_img_r} ({_n_done}/{_n_total}) {_tag}  "
                      f"align {_n_fit}/{_n_det}, astrom {_n_astrom}  "
                      f"[{_dt_s:.1f}s]", flush=True)
                return True

            if _n_proc > 1 and _n_total > 0:
                import multiprocessing as _mp
                from concurrent.futures import (ProcessPoolExecutor,
                                                as_completed)
                # NOTE: no max_tasks_per_child — mid-pool worker respawn under
                # forkserver deadlocks at exactly n_workers x 25 tasks (seen
                # at image 400 on Omega_Cen/47Tuc, same signature as the old
                # xmatch hang).  Memory is recycled by rebuilding the whole
                # pool every _BATCH images instead.
                _BATCH = 300
                for _b0 in range(0, _n_total, _BATCH):
                    _batch = _indv_names[_b0:_b0 + _BATCH]
                    with ProcessPoolExecutor(
                            max_workers=min(_n_proc, len(_batch)),
                            mp_context=_mp.get_context("forkserver"),
                            initializer=_indv_pool_init) as _ex:
                        _futs = {_ex.submit(_run_indv_one, _img, _run_kw,
                                            _indv_root,
                                            _cfg_map[_img]): _img
                                 for _img in _batch}
                        for _fut in as_completed(_futs):
                            try:
                                _res = _fut.result()
                            except Exception as _exc:
                                _res = (_futs[_fut],
                                        f"{type(_exc).__name__}: {_exc}",
                                        -1, -1, -1, 0.0)
                            _n_done += 1
                            if _indv_report(_res):
                                _n_ok += 1
                            else:
                                _n_fail += 1
            else:
                for _img in _indv_names:
                    _res = _run_indv_one(_img, _run_kw, _indv_root,
                                         _cfg_map[_img])
                    _n_done += 1
                    if _indv_report(_res):
                        _n_ok += 1
                    else:
                        _n_fail += 1

            print("─"*50)
            print(f"End: {_dt.now():%Y-%m-%d %H:%M:%S}")
            print(f"\nIndividual fitting complete: {_n_ok} succeeded, {_n_fail} failed")

        elif args.test_synthetic:
            # Run BP3M on the synthetic directory tree.
            # The synthetic data lives at {output_dir}/{field}/{syn_name}/,
            # so we pass output_dir={output_dir}/{field} and field_name=syn_name.
            print("\n" + "=" * 55)
            print(f"Synthetic test — running BP3M on {syn_name}/")
            print("=" * 55)
            run_alignment(
                output_dir=output_dir / field,
                field_name=syn_name,
                n_iter=args.n_bp3m_iter,
                n_samples=args.n_samples,
                mcmc_posteriors=args.mcmc_posteriors,
                clip_sigma=args.bp3m_clip_sigma,
                poly_order=args.poly_order,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                two_phase_align=args.two_phase_align,
                fit_epoch_distortion=args.fit_epoch_distortion,
                epoch_dist_order=args.epoch_dist_order,
                epoch_gap_days=args.epoch_gap_days,
                epoch_dist_sigma=args.epoch_dist_sigma,
                epoch_breaks=args.epoch_breaks,
                epoch_dist_min_images=args.epoch_dist_min_images,
                epoch_dist_groupby=args.epoch_dist_groupby,
                use_sparse=args.sparse,
                no_plots=args.no_plots,
                images=_bp3m_images,
                remove_images=args.bp3m_remove_images,
                restrict_filters=args.restrict_filters,
                restrict_instdet=args.restrict_instdet,
                bp3m_min_stars=args.bp3m_min_stars,
                checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                use_influence_clip=not args.no_influence_clip,
                prior_sigma_rot_deg=args.prior_sigma_rot_deg,
                prior_sigma_scale=args.prior_sigma_scale,
                prior_sigma_skew=args.prior_sigma_skew,
                prior_sigma_pointing=args.prior_sigma_pointing,
                inflate_alpha_max=args.inflate_alpha_max,
                influence_k=args.influence_k,
                influence_floor_sr=args.influence_floor_sr,
                influence_floor_sd=args.influence_floor_sd,
                influence_raw_cooks_d=args.influence_raw_cooks_d,
                verbose_tests=args.verbose_tests,
                use_two_tier=args.two_tier,
                no_align_prior=args.no_align_prior,
                pos_err_floor=args.bp3m_pos_err_floor,
                use_indv_outputs=args.use_indv_outputs,
                test_hysteresis_delta=args.test_hysteresis_delta,
                min_align_demote=args.min_align_demote,
                plot_residuals=args.plot_residuals,
                plot_influence=args.plot_influence,
                use_qso_anchors=not args.no_qso_anchors,
                qso_anchors_csv=_qso_anchors_csv if _qso_exists(_qso_anchors_csv) else None,
                exclude_2p_from_alignment=args.exclude_2p_from_alignment,
                gaia_csv=gaia_csv_path,
                use_delve=args.use_delve,
                delve_use_for_align=args.delve_use_for_align,
            )
            # ── Step 5b: Compare synthetic results to truth ────────────────────
            print("\n" + "=" * 55)
            print("Synthetic test — comparing results to truth")
            print("=" * 55)
            from bp3m.pipeline.synthetic import compare_synthetic_results, run_conditional_solve
            compare_synthetic_results(
                output_dir=output_dir,
                field_name=field,
                syn_name=syn_name,
            )
            # ── Step 5c: Conditional solve with r fixed at r_true ─────────────
            print("\n" + "=" * 55)
            print("Synthetic test — conditional solve (r = r_true)")
            print("=" * 55)
            run_conditional_solve(
                output_dir=output_dir,
                field_name=field,
                syn_name=syn_name,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                poly_order=args.poly_order,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                two_phase_align=args.two_phase_align,
                fit_epoch_distortion=args.fit_epoch_distortion,
                epoch_dist_order=args.epoch_dist_order,
                epoch_gap_days=args.epoch_gap_days,
                epoch_dist_sigma=args.epoch_dist_sigma,
                epoch_breaks=args.epoch_breaks,
                epoch_dist_min_images=args.epoch_dist_min_images,
                epoch_dist_groupby=args.epoch_dist_groupby,
            )
        else:
            _joint_bp3m_dir = None
            if args.bp3m_results_suffix:
                _joint_bp3m_dir = (output_dir / field /
                                   f"BP3M_results_{args.bp3m_results_suffix}")
                print(f"  Joint-fit outputs -> {_joint_bp3m_dir}")
            run_alignment(
                output_dir=output_dir, field_name=field,
                n_iter=args.n_bp3m_iter,
                n_samples=args.n_samples,
                mcmc_posteriors=args.mcmc_posteriors,
                clip_sigma=args.bp3m_clip_sigma,
                poly_order=args.poly_order,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                two_phase_align=args.two_phase_align,
                fit_epoch_distortion=args.fit_epoch_distortion,
                epoch_dist_order=args.epoch_dist_order,
                epoch_gap_days=args.epoch_gap_days,
                epoch_dist_sigma=args.epoch_dist_sigma,
                epoch_breaks=args.epoch_breaks,
                epoch_dist_min_images=args.epoch_dist_min_images,
                epoch_dist_groupby=args.epoch_dist_groupby,
                use_sparse=args.sparse,
                no_plots=args.no_plots,
                images=_bp3m_images,
                remove_images=args.bp3m_remove_images,
                restrict_filters=args.restrict_filters,
                restrict_instdet=args.restrict_instdet,
                bp3m_min_stars=args.bp3m_min_stars,
                checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                use_influence_clip=not args.no_influence_clip,
                prior_sigma_rot_deg=args.prior_sigma_rot_deg,
                prior_sigma_scale=args.prior_sigma_scale,
                prior_sigma_skew=args.prior_sigma_skew,
                prior_sigma_pointing=args.prior_sigma_pointing,
                inflate_alpha_max=args.inflate_alpha_max,
                influence_k=args.influence_k,
                influence_floor_sr=args.influence_floor_sr,
                influence_floor_sd=args.influence_floor_sd,
                influence_raw_cooks_d=args.influence_raw_cooks_d,
                verbose_tests=args.verbose_tests,
                use_two_tier=args.two_tier,
                no_align_prior=args.no_align_prior,
                pos_err_floor=args.bp3m_pos_err_floor,
                use_indv_outputs=args.use_indv_outputs,
                bp3m_dir=_joint_bp3m_dir,
                pos_corr_table=args.pos_corr_table,
                epoch_dist_prior=args.epoch_dist_prior,
                epoch_dist_prior_inflate=args.epoch_dist_prior_inflate,
                test_hysteresis_delta=args.test_hysteresis_delta,
                min_align_demote=args.min_align_demote,
                plot_residuals=args.plot_residuals,
                plot_influence=args.plot_influence,
                use_qso_anchors=not args.no_qso_anchors,
                qso_anchors_csv=_qso_anchors_csv if _qso_exists(_qso_anchors_csv) else None,
                gaia_epoch_obs=_gaia_epoch_obs_for_solver,
                exclude_2p_from_alignment=args.exclude_2p_from_alignment,
                gaia_csv=gaia_csv_path,
                use_delve=args.use_delve,
                delve_use_for_align=args.delve_use_for_align,
            )

    # Save the command only on successful completion so interrupted runs
    # do not overwrite the record of the last successful invocation.
    import shlex as _shlex
    from datetime import datetime as _datetime
    _cmd_file = output_dir / field / 'bp3m_command.txt'
    _cmd_file.parent.mkdir(parents=True, exist_ok=True)
    _cmd_file.write_text(
        f"# {_datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + ' '.join(_shlex.quote(a) for a in sys.argv) + '\n'
    )

    print("\n" + "=" * 55)
    print("Pipeline complete.")
    if args.test_synthetic:
        print(f"Synthetic results: "
              f"{output_dir / field / syn_name / 'BP3M_results' / 'synthetic_comparison.csv'}")
    else:
        print(f"Results: {output_dir / field / 'BP3M_results' / 'stellar_astrometry.csv'}")
    print("=" * 55)


if __name__ == '__main__':
    main()
