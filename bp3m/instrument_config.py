"""
Per-instrument nominal pixel scale, initial scale ratio, and shared fitting
hyperpriors for BP3M.

INSTRUMENT_CONFIG and SIGMA_* are used both in the Gaia cross-matching step
and in the BP3M alignment fit so the two stages always use consistent priors.
Keeping them here ensures any update propagates to both automatically.

initial_scale ratios were derived from the posterior median pixel_scale_mas /
nominal_pixel_scale_mas across all fields with n_stars_alignment > 50
(ACS/WFC: n=3036, WFC3/UVIS: n=1680; 68% half-width ~ 0.0001 for both).

Adding a new instrument/detector:
  1. Add an entry to INSTRUMENT_CONFIG with at least pixel_scale and
     initial_scale.  All sigma_* keys are optional — omit them to inherit
     the defaults in _DEFAULT_CONFIG.
  2. get_instrument_config() merges _DEFAULT_CONFIG with the per-instrument
     overrides, so callers always get a complete dict.
"""

# ── Default hyperpriors (all instruments unless overridden) ───────────────────
# Calibrated from ACS/WFC and WFC3/UVIS posterior scatter (analyze_hyperpriors.py,
# hyperprior_stats.txt, n≈3600 ACS and n≈2200 WFC3/UVIS image halves).
#
# Individual image priors — constrain each chip's transformation:
#   sigma_rot_deg:  rotation prior width.  hi-chip posterior scatter ≈ 0.043°
#                   (ACS) / 0.025° (WFC3) vs 0.10° → prior is comfortably loose.
#   sigma_scale:    pixel scale ratio width.  Posterior scatter ≈ 2-3× smaller
#                   than this; kept loose as a stability guardrail.
#   sigma_skew:     on- and off-axis skew prior width.
#   sigma_pointing: pointing offset width (mas); ~100 ACS pixels — very loose.
#
# Pair priors — constrain the *difference* between the two chips of the same
# exposure (only active when use_pair_prior=True).  Calibrated from
# std(hi−lo) across paired images.  Currently off by default; widths are
# instrument-specific and can be overridden per entry below.
_DEFAULT_CONFIG = {
    # Detector geometry
    "pixel_scale":  0.050,    # arcsec/pix (fallback for unknown instruments)
    "initial_scale": 1.0,     # prior mean for pixel_scale_ratio

    # Individual-image hyperpriors — data-driven (user 2026-09-30) from the Phase D
    # 8p per-image solves of 43,495 archive images against their headers with the
    # basis3f GDC correction applied (hst_dist_corr ml/v4/basis3f/phaseD_image_params.csv;
    # well-constrained images: >=200 stars on both chips).  Widths ~1.5-2x the
    # measured MAD:
    #   rotation vs header  MAD 0.021 deg ACS/WFC, 0.0097 deg WFC3/UVIS (a per-VISIT
    #                       quantity: within-visit 0.001 deg -> visit pooling later)
    #   scale-1 vs header   MAD 5.7e-6 ACS, 5.4e-6 UVIS
    #   skew                MAD 2.6-3.9e-6 ACS, 3.7-4.8e-6 UVIS
    #   upper-chip offset   MAD 7-8 mpx, consistent with posterior noise (<~5 mpx)
    # Pre-2026-09-30 values: 0.10 deg / 5e-4 / 2e-4 (40-80x looser than the data).
    # BUT the Phase D widths are measured around the header WITH the basis GDC
    # correction applied, while BP3M's prior means are initial_scale_ratio (a
    # constant) and zero skew.  Leo I 2026-09-30: without the correction the ACS
    # posteriors sit at skew +5e-5..+8e-5 (uncorrected TDD) and scale -4.5e-5 from
    # the constant; even with the correction the scale scatters 4-8e-5 (VAFACTOR
    # is not in the prior mean).  A 1e-5/5e-6 prior around those means biased
    # mu_pop by 7 sigma.  So: rotation and chip take the Phase D widths; scale
    # is 2e-5 now that the prior mean carries VAFACTOR (Leo I: scale residual = 1.07 x
    # (VAFACTOR-1), r 0.99); skew is 1e-4 without a correction model and
    # sigma_skew_corrected with one (run_alignment / cross-match pick).
    "sigma_rot_deg":  0.03,   # rotation prior width (deg); UVIS override below
    "sigma_scale":    2e-5,   # pixel scale ratio prior width; centre = initial_scale x VAFACTOR (loaders +
                              # cross-match, 2026-09-30), after which the per-image scatter is ~1e-5
    "sigma_skew":     1e-4,   # on- and off-axis skew prior width, no correction model (ACS TDD)
    "sigma_skew_corrected": 1e-5,  # skew prior width when a pos_corr_model is applied (Phase D 3-5e-6; Fornax UVIS 1.2e-5)
    "sigma_pointing": 5000.0, # pointing offset prior width (mas)
    # 8p model (fit_chip_offset): 2-D translation of the upper chip relative to
    # the lower one, in pixels, on top of the shared linear terms.  Phase D
    # (2026-09-30, 11k well-constrained archive images): MAD 7-8 mpx per axis,
    # at the posterior-noise level, no era trend -> intrinsic <~5 mpx.  (The
    # hi-lo pointing scatter of split-CCD fits, 0.08 px, is per-chip linear freedom.)
    "sigma_chip_px":  0.01,

    # Pair-coupling hyperpriors (hi−lo difference)
    # Calibrated: ACS rot 0.044°, WFC3 rot 0.025° → 0.10° conservative round number.
    # Pointing is strongly instrument-dependent (ACS RA 115 mas vs WFC3 15 mas);
    # per-instrument overrides below where data is available.
    "sigma_pair_rot_deg":  0.10,   # expected hi/lo rotation difference (deg)
    "sigma_pair_scale":    5e-4,   # expected hi/lo scale difference
    "sigma_pair_skew":     2e-4,   # expected hi/lo skew difference
    "sigma_pair_pointing": 100.0,  # expected hi/lo pointing difference (mas)
}

# ── Per-instrument config ─────────────────────────────────────────────────────
# Keys are (INSTRUME, DETECTOR) as they appear in the primary FITS header.
# Only include keys that differ from _DEFAULT_CONFIG; the rest are inherited.
INSTRUMENT_CONFIG = {
    # ── HST ──────────────────────────────────────────────────────────────────
    ("ACS",  "WFC"):  {
        "pixel_scale":  0.050,
        "initial_scale": 0.99456,
        # Pair pointing calibrated from 3484 paired images:
        #   RA scatter 115 mas (≈ current 100 mas), Dec scatter 39 mas.
        #   Use 100 mas as a compromise covering both axes.
        "sigma_pair_pointing": 100.0,
        "sigma_chip_px": 0.01,
    },
    ("WFC3", "UVIS"): {
        "pixel_scale":  0.040,
        "initial_scale": 0.99419,
        # Pair pointing calibrated from 2164 paired images:
        #   RA scatter 15 mas, Dec scatter 11 mas — much tighter than ACS.
        "sigma_pair_pointing": 15.0,
        "sigma_chip_px": 0.01,
        # Phase D: UVIS rotation vs header MAD 0.0097 deg (side lobes to +-0.03)
        "sigma_rot_deg": 0.02,
    },
    ("WFC3", "IR"):   {
        "pixel_scale":  0.128,
        "initial_scale": 1.0,
    },
    # ── JWST ─────────────────────────────────────────────────────────────────
    # One entry per detector (each _cal file is one detector).  pixel_scale = the
    # sqrt|det CD| of the stage-2 headers (Liwen Chen's measure_pixel_scale.py over
    # LMC + Draco frames, 2026-08), which reproduces the GDC-frame plate scale of her
    # bp3m fits to 3e-5..1.6e-4, so initial_scale stays 1 and VA_SCALE (the JWST
    # VAFACTOR) centres the prior per exposure.  Prior widths are deliberately loose
    # until calibrated from v1 posteriors as the HST values were (her LMC NIRCam
    # fits scattered 0.29 deg rms in rotation around a 0.1 deg prior with per-channel
    # scales; NIRISS showed a fixed +0.12 deg / -0.67% offset that these per-detector
    # scales and the GDC-frame orientation fit absorb).  Single-chip detectors: no
    # chip/pair terms.
    **{(_i, _d): {"pixel_scale": _ps, "initial_scale": 1.0,
                  "sigma_rot_deg": 0.3, "sigma_scale": 3e-4, "sigma_skew": 1e-4}
       for (_i, _d, _ps) in [
           ("NIRCAM", "NRCA1", 0.031227), ("NIRCAM", "NRCA2", 0.030778),
           ("NIRCAM", "NRCA3", 0.031340), ("NIRCAM", "NRCA4", 0.030900),
           ("NIRCAM", "NRCB1", 0.030746), ("NIRCAM", "NRCB2", 0.031194),
           ("NIRCAM", "NRCB3", 0.030872), ("NIRCAM", "NRCB4", 0.031326),
           ("NIRCAM", "NRCALONG", 0.062906), ("NIRCAM", "NRCBLONG", 0.063001),
           ("NIRISS", "NIS", 0.065567),
           ("MIRI", "MIRIMAGE", 0.110913),
       ]},
}

# ── Fallback for unknown instruments ──────────────────────────────────────────
_UNKNOWN_CONFIG: dict = {}   # inherits everything from _DEFAULT_CONFIG


def get_instrument_config(instrument: str, detector: str) -> dict:
    """Return complete config dict for (instrument, detector).

    All keys from _DEFAULT_CONFIG are always present.  Per-instrument entries
    in INSTRUMENT_CONFIG override individual keys; missing keys fall through to
    the defaults.  Adding a new instrument never requires touching callers.
    """
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(INSTRUMENT_CONFIG.get((instrument, detector), _UNKNOWN_CONFIG))
    return cfg


# ── Module-level aliases (backward compatibility) ─────────────────────────────
# Code that imports these names directly still works; they equal the default
# values.  New code should call get_instrument_config() for per-instrument values.
# The Step-4 cross-match (cross_match.py, cross_match_delve.py) fits raw header ->
# Gaia without any correction model and writes these widths into its cache key,
# so it keeps the pre-2026-09-30 widths (XMATCH_*): tightening them would both
# bias its uncorrected plate fits and re-cross-match the whole archive.
XMATCH_SIGMA_ROT_DEG = 0.10
XMATCH_SIGMA_SCALE   = 5e-4
XMATCH_SIGMA_SKEW    = 2e-4
SIGMA_ROT_DEG        = _DEFAULT_CONFIG["sigma_rot_deg"]
SIGMA_SCALE          = _DEFAULT_CONFIG["sigma_scale"]
SIGMA_SKEW           = _DEFAULT_CONFIG["sigma_skew"]
SIGMA_SKEW_CORRECTED = _DEFAULT_CONFIG["sigma_skew_corrected"]
SIGMA_POINTING       = _DEFAULT_CONFIG["sigma_pointing"]
SIGMA_PAIR_ROT_DEG   = _DEFAULT_CONFIG["sigma_pair_rot_deg"]
SIGMA_PAIR_SCALE     = _DEFAULT_CONFIG["sigma_pair_scale"]
SIGMA_PAIR_SKEW      = _DEFAULT_CONFIG["sigma_pair_skew"]
SIGMA_PAIR_POINTING  = _DEFAULT_CONFIG["sigma_pair_pointing"]
SIGMA_CHIP_PX        = _DEFAULT_CONFIG["sigma_chip_px"]
