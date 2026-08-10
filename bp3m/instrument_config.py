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

    # Individual-image hyperpriors
    "sigma_rot_deg":  0.10,   # rotation prior width (deg)
    "sigma_scale":    5e-4,   # pixel scale ratio prior width (fractional)
    "sigma_skew":     2e-4,   # on- and off-axis skew prior width
    "sigma_pointing": 5000.0, # pointing offset prior width (mas)

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
    },
    ("WFC3", "UVIS"): {
        "pixel_scale":  0.040,
        "initial_scale": 0.99419,
        # Pair pointing calibrated from 2164 paired images:
        #   RA scatter 15 mas, Dec scatter 11 mas — much tighter than ACS.
        "sigma_pair_pointing": 15.0,
    },
    ("WFC3", "IR"):   {
        "pixel_scale":  0.128,
        "initial_scale": 1.0,
    },
    # ── JWST — add entries here as needed, e.g.: ─────────────────────────────
    # ("NIRCAM", "NRCA1"): {"pixel_scale": 0.031, "initial_scale": 1.0},
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
SIGMA_ROT_DEG        = _DEFAULT_CONFIG["sigma_rot_deg"]
SIGMA_SCALE          = _DEFAULT_CONFIG["sigma_scale"]
SIGMA_SKEW           = _DEFAULT_CONFIG["sigma_skew"]
SIGMA_POINTING       = _DEFAULT_CONFIG["sigma_pointing"]
SIGMA_PAIR_ROT_DEG   = _DEFAULT_CONFIG["sigma_pair_rot_deg"]
SIGMA_PAIR_SCALE     = _DEFAULT_CONFIG["sigma_pair_scale"]
SIGMA_PAIR_SKEW      = _DEFAULT_CONFIG["sigma_pair_skew"]
SIGMA_PAIR_POINTING  = _DEFAULT_CONFIG["sigma_pair_pointing"]
