"""
Per-instrument nominal pixel scale, initial scale ratio, and shared fitting
hyperpriors for BP3M.

INSTRUMENT_CONFIG and SIGMA_* are used both in the Gaia cross-matching step
and in the BP3M alignment fit so the two stages always use consistent priors.
Keeping them here ensures any update propagates to both automatically.

initial_scale ratios were derived from the posterior median pixel_scale_mas /
nominal_pixel_scale_mas across all fields with n_stars_alignment > 50
(ACS/WFC: n=3036, WFC3/UVIS: n=1680; 68% half-width ~ 0.0001 for both).
"""

# Keys are (INSTRUME, DETECTOR) as they appear in the primary FITS header.
# Designed to be extended for additional telescopes (e.g. JWST NIRCam, NIRISS)
# by adding entries keyed on their INSTRUME/DETECTOR header values.
INSTRUMENT_CONFIG = {
    # HST
    ("ACS",  "WFC"):  {"pixel_scale": 0.050, "initial_scale": 0.99456},
    ("WFC3", "UVIS"): {"pixel_scale": 0.040, "initial_scale": 0.99419},
    ("WFC3", "IR"):   {"pixel_scale": 0.128, "initial_scale": 1.0},
    # JWST — add entries here as needed, e.g.:
    # ("NIRCAM", "NRCA1"): {"pixel_scale": 0.031, "initial_scale": 1.0},
}

# Fallback for unknown instruments
_DEFAULT_CONFIG = {"pixel_scale": 0.050, "initial_scale": 1.0}

# ---------------------------------------------------------------------------
# Shared fitting hyperpriors
# Used by both the Gaia cross-matching step (catalog_matcher.py) and the
# BP3M alignment solver (solver.py).  Update here to propagate everywhere.
# ---------------------------------------------------------------------------
SIGMA_ROT_DEG  = 0.05     # rotation prior width (degrees)
SIGMA_SCALE    = 5e-4     # pixel scale ratio prior width (fractional)
SIGMA_SKEW     = 2e-4     # on- and off-axis skew prior width
SIGMA_POINTING = 5000.0   # pointing offset prior width (mas); ~100 ACS pixels


def get_instrument_config(instrument: str, detector: str) -> dict:
    """Return config dict for (instrument, detector), falling back to defaults."""
    return INSTRUMENT_CONFIG.get((instrument, detector), _DEFAULT_CONFIG)
