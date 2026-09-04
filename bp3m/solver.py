"""
BP3M Linear Algebra Solver.

Implements the closed-form posterior distributions from Equations 9-11
of McKinnon et al. (in prep), using the Schur complement / information-form
marginalization for efficiency at scale.

Coordinate conventions:
  - HST positions: x_c = X - Xo, y_c = Y - Yo  (centered on image pivot)
  - Gaia pseudo-image: xs = plane_project(ra, dec, ra0, dec0, pscale)
    which gives ~(X_G - Wo) in the detector frame
  - Both are in units of HST pixels (same pixel scale)

The transformation model:
  x_survey_i,j = X_i,j @ r_j - JU_i,j @ v_T,i   (Eq. 8)

where:
  r_j = (a, b, c, d, w, z, Δα0, Δδ0)^T    image transformation (8-dim)
  v_T,i = (Δα*, Δδ, μα*, μδ, ϖ)^T         astrometry update (5-dim)
  X_i,j uses centered HST positions (x_c, y_c)
  JU_i,j = J_i,j @ U_i,j (Jacobian x time-evolution)

The iterative algorithm:
  1. Init R_j from image header rotation/scale
  2. Compute C_s,i,j = R_j @ C_hst_i,j @ R_j^T
  3. Solve for r_hat, C_r (Schur complement of joint Gaussian)
  4. Solve for v_hat_i, C_vT,i (conditional on r_hat)
  5. Update R_j from new (a,b,c,d) in r_hat; repeat from 2
"""

import numpy as np
from scipy import linalg
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from typing import Optional

from .astro_utils import (
    plane_project, plane_project_jacobian, plane_project_tangent_derivs,
    get_parallax_factors, build_U_matrix, build_X_matrix,
    build_U_matrices, build_X_matrices,
    epoch_distortion_basis, epoch_distortion_n_shape, epoch_distortion_pairs,
    hst_position_cov, rotation_matrix_from_abcd, gaia_cov_to_survey_cov,
    abcd_from_rotation_pixscale_skew, n_r_from_poly_order, compute_poly_jacobian,
    get_tele_position, michalik_sigma_plx_prior, RAD2MAS, DEG2RAD, GAIA_SYS_DICT
)
from .instrument_config import (
    SIGMA_ROT_DEG  as _SIGMA_ROT_DEG,
    SIGMA_SCALE    as _SIGMA_SCALE,
    SIGMA_SKEW     as _SIGMA_SKEW,
    SIGMA_POINTING as _SIGMA_POINTING,
    SIGMA_PAIR_ROT_DEG  as _SIGMA_PAIR_ROT_DEG,
    SIGMA_PAIR_SCALE    as _SIGMA_PAIR_SCALE,
    SIGMA_PAIR_SKEW     as _SIGMA_PAIR_SKEW,
    SIGMA_PAIR_POINTING as _SIGMA_PAIR_POINTING,
)

N_R = 6     # r_j dimensions for poly_order=1 (a,b,c,d,Δα0,Δδ0)
N_V = 5     # v_T,i dimensions: (Δα*, Δδ, μα*, μδ, ϖ)

# ── Global astrometry prior ────────────────────────────────────────────────────
# Gaia 5p/6p sources: NO diffuse prior — their Gaia covariance is the sole prior.
# Gaia 2p sources: diffuse PM prior (100 mas/yr) + Michalik et al. (2015)
#   magnitude/direction-dependent parallax prior (10 * sigma_F90).
# HST-only sources (future v2 loader): same as 2p.
# Per-star precision arrays are built in _load_star_data as self._C_VG_inv_per_star.
_SIGMA_POS = 1e6   # effectively flat prior on Δα*, Δδ  (all sources)
_SIGMA_PM  = 100.0  # mas/yr  (2p / HST-only only)

# Image transformation prior uncertainties (1-sigma) are defined in
# instrument_config.py (SIGMA_ROT_DEG, SIGMA_SCALE, SIGMA_SKEW, SIGMA_POINTING)
# and imported above as _SIGMA_* for backward compatibility with pipeline imports.

# Initial residual filter applied in _precompute_geometry.
# Stars whose corrected 2D residual (after removing bulk w,z offset)
# exceeds this threshold are excluded from use_for_fit before the first
# solve pass.  Fornax real matches: p99 ≈ 37 px.  Leo_I false matches:
# p50 ≈ 200 px.  100 px is a clean cut with comfortable margin on both sides.
_INIT_RESID_CLIP_PX = 100.0



def _make_image_prior(meta, poly_order=1,
                      sigma_rot_deg=None, sigma_scale=None,
                      sigma_skew=None, sigma_pointing=None):
    """
    Return (r_prior_j, C_r_prior_inv_j) for image j.

    r_j = (a, b, c, d, Δα0, Δδ0 [, poly terms...])
    Prior:
      (a,b,c,d) — from header rotation/scale (strong prior)
      (Δα0,Δδ0) — sigma = sigma_pointing mas (loose; ~100 ACS WFC pixels)
      poly terms — zero mean, flat prior (determined entirely by data)

    sigma_* override the module-level defaults from instrument_config.py.
    """
    if sigma_rot_deg  is None: sigma_rot_deg  = meta.get("sigma_rot_deg",  _SIGMA_ROT_DEG)
    if sigma_scale    is None: sigma_scale    = meta.get("sigma_scale",    _SIGMA_SCALE)
    if sigma_skew     is None: sigma_skew     = meta.get("sigma_skew",     _SIGMA_SKEW)
    if sigma_pointing is None: sigma_pointing = meta.get("sigma_pointing", _SIGMA_POINTING)
    n_r = n_r_from_poly_order(poly_order)

    rot_rad = meta["orig_rot_deg"] * DEG2RAD
    s = meta.get("initial_scale_ratio", 1.0)
    r_prior = np.zeros(n_r)
    r_prior[:4] = [s*np.cos(rot_rad), s*np.sin(rot_rad), -s*np.sin(rot_rad), s*np.cos(rot_rad)]
    # r_prior[4:] = 0  (Δα0, Δδ0 and all poly terms start at zero)

    # Jacobian ∂(a,b,c,d)/∂(rot_rad, scale_ratio, on_skew, off_skew)
    cr, sr = np.cos(rot_rad), np.sin(rot_rad)
    J = np.array([
        [-s*sr,  cr,  1,  0],
        [ s*cr,  sr,  0,  1],
        [-s*cr, -sr,  0,  1],
        [-s*sr,  cr, -1,  0],
    ])
    sigma = np.array([sigma_rot_deg * DEG2RAD, sigma_scale, sigma_skew, sigma_skew])
    C_abcd = J @ np.diag(sigma**2) @ J.T   # (4, 4)

    # Full n_r × n_r prior precision matrix
    C_r_prior_inv = np.zeros((n_r, n_r))
    try:
        C_r_prior_inv[:4, :4] = np.linalg.inv(C_abcd)
    except np.linalg.LinAlgError:
        C_r_prior_inv[:4, :4] = np.diag(1.0 / np.diag(C_abcd + 1e-30 * np.eye(4)))

    C_r_prior_inv[4, 4] = sigma_pointing ** -2  # Δα0
    C_r_prior_inv[5, 5] = sigma_pointing ** -2  # Δδ0
    # Indices 6+ (poly terms) remain zero — flat prior.

    return r_prior, C_r_prior_inv


def _make_pair_coupling_inv(meta_hi, poly_order=1,
                             sigma_pair_rot_deg=None, sigma_pair_scale=None,
                             sigma_pair_skew=None, sigma_pair_pointing=None):
    """
    Return C_pair_inv (N_R × N_R) for the _hi/_lo chip coupling prior.

    Prior: (param_hi − param_lo) ~ N(0, Σ_pair) for all 6 physical params
    (rotation, scale, on_skew, off_skew, Δα0, Δδ0).  Poly terms (indices 6+)
    are not coupled — higher-order distortions may genuinely differ per chip.

    Caller adds +C_pair_inv to each chip's diagonal block and −C_pair_inv to
    the off-diagonal cross blocks in H_rr.
    """
    if sigma_pair_rot_deg  is None: sigma_pair_rot_deg  = meta_hi.get("sigma_pair_rot_deg",  _SIGMA_PAIR_ROT_DEG)
    if sigma_pair_scale    is None: sigma_pair_scale    = meta_hi.get("sigma_pair_scale",    _SIGMA_PAIR_SCALE)
    if sigma_pair_skew     is None: sigma_pair_skew     = meta_hi.get("sigma_pair_skew",     _SIGMA_PAIR_SKEW)
    if sigma_pair_pointing is None: sigma_pair_pointing = meta_hi.get("sigma_pair_pointing", _SIGMA_PAIR_POINTING)

    n_r = n_r_from_poly_order(poly_order)
    rot_rad = meta_hi["orig_rot_deg"] * DEG2RAD
    s = meta_hi.get("initial_scale_ratio", 1.0)
    cr, sr = np.cos(rot_rad), np.sin(rot_rad)
    J = np.array([
        [-s*sr,  cr,  1,  0],
        [ s*cr,  sr,  0,  1],
        [-s*cr, -sr,  0,  1],
        [-s*sr,  cr, -1,  0],
    ])
    sigma = np.array([sigma_pair_rot_deg * DEG2RAD, sigma_pair_scale,
                      sigma_pair_skew, sigma_pair_skew])
    C_pair_abcd = J @ np.diag(sigma ** 2) @ J.T

    C_pair_inv = np.zeros((n_r, n_r))
    try:
        C_pair_inv[:4, :4] = np.linalg.inv(C_pair_abcd)
    except np.linalg.LinAlgError:
        C_pair_inv[:4, :4] = np.diag(1.0 / np.diag(C_pair_abcd + 1e-30 * np.eye(4)))
    C_pair_inv[4, 4] = sigma_pair_pointing ** -2  # Δα0
    C_pair_inv[5, 5] = sigma_pair_pointing ** -2  # Δδ0
    return C_pair_inv


# ── Option-5 health gate + backstop ceiling on the adaptive thresholds ───────
#
# _adapt_thresh returns p50 + k*(p50-p16), floored but otherwise UNCAPPED, so a
# globally-inconsistent population defines its own "normal" and no star looks
# like an outlier.  Measured failure: an LSST multi-band joint drove the df=5
# threshold to 327 while stars sat 10-67 sigma from their Gaia priors, and none
# of their detections were ever dropped.
#
# The gate encodes *why* adaptivity is licensed.  Adapting upward is justified
# when the quoted uncertainties are mis-scaled -- but the alpha inflation already
# measures and divides out exactly that scale factor.  A centre still far above
# the theoretical median therefore means the MODEL is wrong (uncorrected DCR,
# distortion, ...) rather than the error model, and adapting to it is unjustified
# rather than merely generous.  So when p50 > gate_mult * median(chi2_df), stop
# adapting and fall back to the floor.
#
# Calibration, measured over 418 bp3m v1 HST fields (chi2_gaia reconstructed from
# C_vT.npy + the Gaia prior columns) plus LSST/Fornax:
#   healthy p50 is always BELOW theory: HST 1.94, LSST i-band 1.87, Fornax 0.11,
#   against a theoretical median of 4.35 (df=5).  Upward adaptivity has never
#   been needed to protect a healthy field.  No HST field exceeds p50 = 13.93.
#   GATE_MULT=3.0    -> gate at 13.05 (df=5); gates ~1/418 HST fields.
#   CEILING_MULT=6.0 -> 90.5 (df=5); HST max adaptive is 83.6, so 0/418 bind.
THRESH_GATE_MULT_DEFAULT    = 3.0
THRESH_CEILING_MULT_DEFAULT = 6.0


def _gate_thresh(thresh, p50, df, floor,
                 gate_mult=THRESH_GATE_MULT_DEFAULT,
                 ceiling_mult=THRESH_CEILING_MULT_DEFAULT):
    """
    Apply the health gate and backstop ceiling to one adaptive threshold.

    Returns (thresh, gated, capped).  `gated` means the population was judged
    inconsistent with its own uncertainties and the threshold was pulled back to
    the floor; `capped` means only the backstop ceiling bound.  A NaN p50 (fewer
    than 10 reference points) applies nothing.
    """
    from scipy.stats import chi2 as _c2
    if not np.isfinite(p50):
        return float(thresh), False, False
    if gate_mult is not None and p50 > gate_mult * float(_c2.median(df=df)):
        # Population inconsistent with its own errors. Cap at the CEILING
        # rather than collapsing to the floor: in crowded fields the elevated
        # chi2 median is endemic (blending noise outside the error model), and
        # the floor rejected ~30% of detections that the pre-gate pipeline
        # kept. The ceiling still stops runaway adaptive thresholds.
        ceil = (ceiling_mult if ceiling_mult is not None else 6.0) * float(floor)
        return float(min(thresh, ceil)), True, False
    if ceiling_mult is not None:
        ceil = ceiling_mult * float(floor)
        if thresh > ceil:
            return float(ceil), False, True
    return float(thresh), False, False


class BP3MSolver:
    """
    Simultaneous HST-Gaia astrometric alignment and stellar PM/parallax update.

    Parameters
    ----------
    images : dict  {image_name: dict of metadata}
    stars_per_image : dict  {image_name: pd.DataFrame}
    gaia_catalog : pd.DataFrame  (one row per unique Gaia source)
    star_id_to_idx : dict  {Gaia_id: int index}
    image_names : list[str]
    star_in_image : dict  (unused; we rebuild this from data)
    """

    def __init__(self, images, stars_per_image, gaia_catalog,
                 star_id_to_idx, image_names, star_in_image,
                 poly_order=1, exclude_2p_from_alignment=False,
                 prior_sigma_rot_deg=None, prior_sigma_scale=None,
                 prior_sigma_skew=None, prior_sigma_pointing=None,
                 prior_sigma_pair_rot_deg=None, prior_sigma_pair_scale=None,
                 prior_sigma_pair_skew=None, prior_sigma_pair_pointing=None,
                 use_pair_prior=False,
                 fit_epoch_distortion=False, epoch_dist_order=3,
                 epoch_gap_days=180.0, epoch_dist_sigma_mas=10.0,
                 epoch_breaks=None, epoch_dist_min_images=3,
                 epoch_dist_groupby='full'):
        """
        Parameters
        ----------
        poly_order : int, optional
            Polynomial order for the image transformation.
              1 → linear (a,b,c,d,Δα0,Δδ0) — 6 parameters per image (default)
              2 → adds degree-2 terms — 12 parameters per image
              3 → adds degree-2 and degree-3 terms — 20 parameters per image
            N_R(p) = (p+1)*(p+2)
        prior_sigma_rot_deg, prior_sigma_scale, prior_sigma_skew, prior_sigma_pointing : float or None
            Override the plate prior widths from instrument_config.py.
            None uses the module-level defaults.
        """
        if poly_order < 1:
            raise ValueError(f"poly_order must be ≥ 1, got {poly_order}")
        self.poly_order = poly_order
        self.N_R = n_r_from_poly_order(poly_order)
        self.exclude_2p_from_alignment = exclude_2p_from_alignment
        # Raw CLI overrides (None = use per-instrument value embedded in image meta).
        # _make_image_prior / _make_pair_coupling_inv fall back to meta then to
        # module-level defaults when these are None.
        self._sigma_rot_deg_cli       = prior_sigma_rot_deg
        self._sigma_scale_cli         = prior_sigma_scale
        self._sigma_skew_cli          = prior_sigma_skew
        self._sigma_pointing_cli      = prior_sigma_pointing
        self._sigma_pair_rot_deg_cli  = prior_sigma_pair_rot_deg
        self._sigma_pair_scale_cli    = prior_sigma_pair_scale
        self._sigma_pair_skew_cli     = prior_sigma_pair_skew
        self._sigma_pair_pointing_cli = prior_sigma_pair_pointing
        # Resolved values for logging / run_config.json (CLI override or global default).
        self._prior_sigma_rot_deg  = prior_sigma_rot_deg  if prior_sigma_rot_deg  is not None else _SIGMA_ROT_DEG
        self._prior_sigma_scale    = prior_sigma_scale    if prior_sigma_scale    is not None else _SIGMA_SCALE
        self._prior_sigma_skew     = prior_sigma_skew     if prior_sigma_skew     is not None else _SIGMA_SKEW
        self._prior_sigma_pointing = prior_sigma_pointing if prior_sigma_pointing is not None else _SIGMA_POINTING
        self._use_pair_prior            = use_pair_prior
        self._prior_sigma_pair_rot_deg  = prior_sigma_pair_rot_deg  if prior_sigma_pair_rot_deg  is not None else _SIGMA_PAIR_ROT_DEG
        self._prior_sigma_pair_scale    = prior_sigma_pair_scale    if prior_sigma_pair_scale    is not None else _SIGMA_PAIR_SCALE
        self._prior_sigma_pair_skew     = prior_sigma_pair_skew     if prior_sigma_pair_skew     is not None else _SIGMA_PAIR_SKEW
        self._prior_sigma_pair_pointing = prior_sigma_pair_pointing if prior_sigma_pair_pointing is not None else _SIGMA_PAIR_POINTING

        self.images = images
        self.stars_per_image = stars_per_image
        self.gaia_cat = gaia_catalog.reset_index(drop=True)
        self.star_id_to_idx = star_id_to_idx
        self.image_names = list(image_names)
        self.n_stars = len(gaia_catalog)
        self.n_images = len(image_names)

        # ── Shared epoch-distortion correction D (see epoch_distortion_basis) ─
        self._ed_enabled       = bool(fit_epoch_distortion)
        self._ed_order         = int(epoch_dist_order)
        self._ed_gap_days      = float(epoch_gap_days)
        self._ed_sigma_mas     = float(epoch_dist_sigma_mas)
        self._ed_breaks_jyear  = list(epoch_breaks) if epoch_breaks else []
        self._ed_min_images    = int(epoch_dist_min_images)
        if epoch_dist_groupby not in ('full', 'no_filter', 'no_epoch', 'static'):
            raise ValueError(f"epoch_dist_groupby must be one of "
                             f"full/no_filter/no_epoch/static, got {epoch_dist_groupby!r}")
        self._ed_groupby       = epoch_dist_groupby
        self.ed_groups         = []     # list of dicts (key, imgs, mean_mjd, ...)
        self._ed_gidx          = {}     # img -> group index or -1
        self.ED_K              = 0
        self.n_ed              = 0
        if self._ed_enabled:
            self._build_epoch_distortion_groups()

        self._cache_gaia()
        self._precompute_geometry()
        self._init_transforms()

    # ── Gaia data caching ──────────────────────────────────────────────────────

    def _build_epoch_distortion_groups(self):
        """
        Assign every image to an (instrument, detector, chip, filter, epoch)
        group for the shared epoch-distortion correction.

        Epochs: images of the same (inst, det, chip, filter) are sorted by
        observation MJD and split where the gap exceeds epoch_gap_days, or at
        the explicit epoch_breaks boundaries (decimal years). Groups with
        fewer than epoch_dist_min_images images get no correction (their
        images are flagged -1 and a warning is printed). Single-epoch groups
        ARE fitted: their D is anchored by the Gaia J2016 positions.
        """
        from astropy.time import Time as _Time
        n_shape   = epoch_distortion_n_shape(self._ed_order)
        self.ED_K = 2 * n_shape

        break_mjds = [float(_Time(b, format='jyear').mjd)
                      for b in self._ed_breaks_jyear]

        # Grouping granularity (--epoch_dist_groupby). Chip is ALWAYS kept —
        # the underlying GDC corrections are per chip.
        #   full      : (inst, det, chip, filter) x epoch      [default]
        #   no_filter : (inst, det, chip)         x epoch
        #   no_epoch  : (inst, det, chip, filter), single static D
        #   static    : (inst, det, chip),         single static D
        _use_filter = self._ed_groupby in ('full', 'no_epoch')
        _use_epoch  = self._ed_groupby in ('full', 'no_filter')

        by_key = {}
        for img in self.image_names:
            meta = self.images[img]
            chip = ('hi' if img.endswith('_hi')
                    else 'lo' if img.endswith('_lo') else 'chip')
            key0 = (str(meta.get('instrument', '?')),
                    str(meta.get('detector', '?')),
                    chip,
                    str(meta.get('filter', '?')) if _use_filter else '*')
            by_key.setdefault(key0, []).append((float(meta['hst_time_mjd']), img))

        self.ed_groups = []
        self._ed_gidx = {img: -1 for img in self.image_names}
        n_skipped = 0
        for key0, lst in sorted(by_key.items()):
            lst.sort()
            if not _use_epoch:
                clusters = [lst]           # one static D for the whole key
            else:
                clusters = [[lst[0]]]
                for prev, cur in zip(lst[:-1], lst[1:]):
                    crosses = any(prev[0] < b <= cur[0] for b in break_mjds)
                    if (cur[0] - clusters[-1][-1][0]) > self._ed_gap_days or crosses:
                        clusters.append([cur])
                    else:
                        clusters[-1].append(cur)
            for ep_id, cl in enumerate(clusters):
                imgs = [im for _, im in cl]
                mean_mjd = float(np.mean([m for m, _ in cl]))
                if len(imgs) < self._ed_min_images:
                    n_skipped += len(imgs)
                    print(f"    epoch-distortion: group {key0} epoch {ep_id} "
                          f"has only {len(imgs)} image(s) < "
                          f"{self._ed_min_images} — no correction fitted")
                    continue
                g = len(self.ed_groups)
                for im in imgs:
                    self._ed_gidx[im] = g
                # prior_mean/prior_prec settable via set_epoch_dist_prior()
                self.ed_groups.append(dict(
                    prior_mean=None, prior_prec=None,
                    instrument=key0[0], detector=key0[1], chip=key0[2],
                    filter=key0[3], epoch_id=ep_id, mean_mjd=mean_mjd,
                    images=imgs))
        self.n_ed = self.ED_K * len(self.ed_groups)
        print(f"  epoch-distortion: {len(self.ed_groups)} chip-groups x "
              f"{self.ED_K} coeffs = {self.n_ed} shared parameters "
              f"(groupby={self._ed_groupby}, order {self._ed_order}, "
              f"gap {self._ed_gap_days:.0f} d, "
              f"prior {self._ed_sigma_mas:.1f} mas"
              + (f", {n_skipped} images unfitted" if n_skipped else "") + ")")
        for g, grp in enumerate(self.ed_groups):
            from astropy.time import Time as _T
            print(f"    D[{g}] {grp['instrument']}/{grp['detector']} "
                  f"{grp['filter']} chip={grp['chip']} epoch~"
                  f"{_T(grp['mean_mjd'], format='mjd').jyear:.2f} "
                  f"({len(grp['images'])} images)")

    def set_epoch_dist_prior(self, g, mean, sigma_px):
        """Centre group g's D prior on `mean` (length ED_K, px) with
        per-coefficient widths `sigma_px` (scalar or length ED_K)."""
        mean = np.asarray(mean, float)
        sig = np.broadcast_to(np.asarray(sigma_px, float), mean.shape)
        assert mean.size == self.ED_K
        self.ed_groups[g]['prior_mean'] = mean
        self.ed_groups[g]['prior_prec'] = sig ** -2.0

    def _ed_cols(self, img):
        """Global shared-vector column indices of img's D block (or None)."""
        g = self._ed_gidx.get(img, -1)
        if not self._ed_enabled or g < 0:
            return None
        off = self.N_R * self.n_images + g * self.ED_K
        return np.arange(off, off + self.ED_K)

    def _ed_disp(self, img, shared_vec):
        """(n, 2) epoch-distortion displacement for img at the current shared
        vector, in pixels. Zero when disabled/unfitted/legacy-length vector."""
        cols = self._ed_cols(img)
        d = self._img_data.get(img)
        if cols is None or d is None or shared_vec.size <= cols[0]:
            return 0.0
        return np.einsum('nkl,l->nk', d["B_mat"], shared_vec[cols])

    def _cache_gaia(self):
        g = self.gaia_cat
        ruwe = g["ruwe"].to_numpy(float)
        # NaN RUWE = 2-param stars (no 5D Gaia solution); treat as trustworthy
        # since NaN means "not applicable", not "poor astrometry".
        self.gaia_trustworthy = np.isnan(ruwe) | (ruwe <= 1.4)
        self.gaia_g  = g["gmag"].to_numpy(float)
        self.gaia_n_hst_used = np.zeros(len(self.gaia_g)).astype(int)
        self.gaia_ra  = g["ra"].to_numpy(float)
        self.gaia_dec = g["dec"].to_numpy(float)
        self.gaia_time  = Time(g["Gaia_time"].fillna(2016.0).to_numpy(float),format='jyear',scale='tcb')
        self.gaia_yr  = self.gaia_time.jyear
        self.sigma_from_gaia_prior  = np.zeros(len(self.gaia_g)).astype(float)
        self.ok_star = np.ones(len(self.gaia_g), dtype=bool)

        # Survey astrometry vector: v_s,i = (0, 0, pmra, pmdec, parallax)
        # (Δα*, Δδ) = 0 since Gaia position IS the reference; updates captured in v_T,i
        self.gaia_6p = np.isfinite(g['pseudocolour'])
        # Synthetic rows (negative Gaia_id: DELVE-only, v2 HST-only) are never
        # Gaia solutions.  DELVE-only rows carry their DELVE PM in the bare
        # `pmra` column, which previously made isfinite(pmra) classify them as
        # gaia_5p — contradicting the intended design (the test-1 comment says
        # "For DELVE-only stars (gaia_2p) ..."), skewing the 5p/2p df split and
        # diagnostics, and leaving a VETOED DELVE-only star with no PM prior at
        # all (near-singular H_vv for singly-detected stars).  As gaia_2p they
        # get the standard diffuse treatment, with the DELVE prior replacing the
        # flat PM prior via needs_diffuse_pm exactly like a real 2p+DELVE star.
        # Only the SIGN is needed, so a float64 round-trip of a 19-digit id is
        # harmless here (solver.py has no module-level pandas import).
        _synthetic = (g['Gaia_id'].fillna(0).to_numpy(float) < 0
                      if 'Gaia_id' in g.columns
                      else np.zeros(self.n_stars, dtype=bool))
        self.gaia_5p = np.isfinite(g['pmra']) & ~self.gaia_6p & ~_synthetic
        # 6p sources have full 5D Gaia astrometry (pmra/pmdec/parallax) like 5p;
        # they are NOT 2p just because they also have a pseudocolour measurement.
        self.gaia_2p = np.isfinite(self.gaia_ra) & ~self.gaia_5p & ~self.gaia_6p
        self.full_gaia_astrometry = np.isfinite(g['pmra']) & np.isfinite(g['pmdec']) & np.isfinite(g['parallax'])

        self.v_survey = np.zeros((self.n_stars, N_V))
        for col, idx in [("pmra", 2), ("pmdec", 3), ("parallax", 4)]:
            if col in g.columns:
                self.v_survey[:, idx] = g[col].fillna(0.0).to_numpy(float)

        # Build 5×5 Gaia covariance matrices (units: mas, mas/yr)
        def _get(col, default=0.0):
            return g[col].fillna(default).to_numpy(float) if col in g.columns else \
                   np.full(self.n_stars, default)

        ra_e   = _get("ra_error",    1e6)   # mas
        dec_e  = _get("dec_error",   1e6)
        pmra_e = _get("pmra_error",  1e3)
        pmdec_e= _get("pmdec_error", 1e3)
        plx_e  = _get("parallax_error", 1e3)

        # Correlations
        corr_ra_dec = _get("ra_dec_corr")
        corr_ra_plx = _get("ra_parallax_corr")
        corr_ra_pmra= _get("ra_pmra_corr")
        corr_ra_pmdec= _get("ra_pmdec_corr")
        corr_dec_plx= _get("dec_parallax_corr")
        corr_dec_pmra= _get("dec_pmra_corr")
        corr_dec_pmdec= _get("dec_pmdec_corr")
        corr_plx_pmra= _get("parallax_pmra_corr")
        corr_plx_pmdec= _get("parallax_pmdec_corr")
        corr_pmra_pmdec= _get("pmra_pmdec_corr")

        # Build (n_stars, 5, 5) covariance matrices
        sigmas = np.stack([ra_e, dec_e, pmra_e, pmdec_e, plx_e], axis=1)  # (n, 5)

        corr_mat = np.zeros((self.n_stars, N_V, N_V))
        for i in range(N_V):
            corr_mat[:, i, i] = 1.0
        # Fill off-diagonals (order: Δα*, Δδ, μα*, μδ, ϖ → indices 0,1,2,3,4)
        pairs = [
            (0,1,corr_ra_dec), (0,2,corr_ra_pmra), (0,3,corr_ra_pmdec),
            (0,4,corr_ra_plx), (1,2,corr_dec_pmra), (1,3,corr_dec_pmdec),
            (1,4,corr_dec_plx),(2,3,corr_pmra_pmdec),(2,4,corr_plx_pmra),
            (3,4,corr_plx_pmdec),
        ]
        for i, j, arr in pairs:
            corr_mat[:, i, j] = arr
            corr_mat[:, j, i] = arr

        # C_survey = diag(sigma) @ corr @ diag(sigma)
        self.C_survey = sigmas[:, :, None] * corr_mat * sigmas[:, None, :]

        #account for systematics in Gaia data
        #amount to inflate uncertainties by
        #might want to change to function of magnitude in the future
        # mult_* are SIGMA multipliers (literature error underestimates), so the
        # covariance is inflated by their square — same as gaia_cross_match.
        self.C_survey[self.gaia_6p] *= GAIA_SYS_DICT['mult_6p'] ** 2
        self.C_survey[self.gaia_5p] *= GAIA_SYS_DICT['mult_5p'] ** 2
        self.C_survey[self.gaia_2p] *= GAIA_SYS_DICT['mult_2p']
        self.C_survey += np.diag(np.array([0,0,
                            GAIA_SYS_DICT['pm_sys_err'],GAIA_SYS_DICT['pm_sys_err'],
                            GAIA_SYS_DICT['parallax_sys_err']])**2)

        # Flag stars with full Gaia astrometry
        self.has_full_astro = (
            (g["pmra_error"].notna() if "pmra_error" in g.columns else
             np.zeros(self.n_stars, bool)) &
            (g["parallax_error"].notna() if "parallax_error" in g.columns else
             np.zeros(self.n_stars, bool))
        ).to_numpy(bool)

        # Invert C_survey
        self.C_survey_inv = np.zeros_like(self.C_survey)
        self.C_survey_inv[self.full_gaia_astrometry] = np.linalg.inv(self.C_survey[self.full_gaia_astrometry])
        self.C_survey_inv[~self.full_gaia_astrometry,:2,:2] = np.linalg.inv(self.C_survey[~self.full_gaia_astrometry,:2,:2])

        self.C_survey_inv_dot_v = np.einsum('nij,nj->ni',self.C_survey_inv,self.v_survey)

        # ── Add DELVE PM/parallax precision ───────────────────────────────────
        # For stars with DELVE PM errors in gaia_catalog, add the full DELVE
        # 5×5 covariance inverse to C_survey_inv (information-form combination).
        #
        # Gaia 5p + DELVE: add full DELVE 5×5 precision to C_survey_inv;
        #   consistency veto via full 5D chi2 (threshold 17.7, df=5).
        # Gaia 2p + DELVE: add full DELVE 5×5 precision to C_survey_inv;
        #   consistency veto via 2D position chi2 (threshold 11.8, df=2).
        if 'delve_pmra_error' in g.columns:
            d_pmra_e  = _get('delve_pmra_error',      np.inf)
            d_pmdec_e = _get('delve_pmdec_error',     np.inf)
            d_pmra    = _get('delve_pmra',  0.0)
            d_pmdec   = _get('delve_pmdec', 0.0)

            has_delve_pm = (np.isfinite(d_pmra_e) & (d_pmra_e > 0) &
                            np.isfinite(d_pmdec_e) & (d_pmdec_e > 0))

            if has_delve_pm.any():
                orig_dp_idx = np.where(has_delve_pm)[0]   # pre-veto global indices
                n_dp = len(orig_dp_idx)

                # Build full DELVE 5×5 covariance in Gaia convention
                # (ra, dec, pmra, pmdec, plx) for all has_delve_pm stars.
                # DELVE convention (ra,dec,plx,pmra,pmdec) → permuted to Gaia convention:
                # corr(0,2)=ra_pmra, corr(0,3)=ra_pmdec, corr(0,4)=ra_plx, etc.
                d_sig = np.column_stack([
                    _get('delve_ra_error',       np.inf)[orig_dp_idx],
                    _get('delve_dec_error',      np.inf)[orig_dp_idx],
                    d_pmra_e[orig_dp_idx],
                    d_pmdec_e[orig_dp_idx],
                    _get('delve_parallax_error', np.inf)[orig_dp_idx],
                ])  # (n_dp, 5)
                corr_d = np.zeros((n_dp, 5, 5))
                for k in range(5):
                    corr_d[:, k, k] = 1.0
                def _gc(col):
                    return _get(col, 0.0)[orig_dp_idx]
                corr_d[:, 0, 1] = corr_d[:, 1, 0] = _gc('delve_corr_ra_dec')
                corr_d[:, 0, 2] = corr_d[:, 2, 0] = _gc('delve_corr_ra_pmra')
                corr_d[:, 0, 3] = corr_d[:, 3, 0] = _gc('delve_corr_ra_pmdec')
                corr_d[:, 0, 4] = corr_d[:, 4, 0] = _gc('delve_corr_ra_plx')
                corr_d[:, 1, 2] = corr_d[:, 2, 1] = _gc('delve_corr_dec_pmra')
                corr_d[:, 1, 3] = corr_d[:, 3, 1] = _gc('delve_corr_dec_pmdec')
                corr_d[:, 1, 4] = corr_d[:, 4, 1] = _gc('delve_corr_dec_plx')
                corr_d[:, 2, 3] = corr_d[:, 3, 2] = _gc('delve_corr_pmra_pmdec')
                corr_d[:, 2, 4] = corr_d[:, 4, 2] = _gc('delve_corr_plx_pmra')
                corr_d[:, 3, 4] = corr_d[:, 4, 3] = _gc('delve_corr_plx_pmdec')
                C_delve_dp = d_sig[:, :, None] * corr_d * d_sig[:, None, :]  # (n_dp, 5, 5)
                has_full_5d_dp = (np.isfinite(d_sig).all(axis=1) & (d_sig > 0).all(axis=1))

                # ── Consistency veto for Gaia 5p + DELVE ────────────────────────
                # Full 5D chi2: χ²₅D = Δv^T (C_gaia + C_delve)^{-1} Δv
                # threshold 17.7 = chi2.ppf(0.9973, df=5)
                is_5p_dp = self.full_gaia_astrometry[orig_dp_idx]
                if is_5p_dp.any():
                    tidx_5p = orig_dp_idx[is_5p_dp]
                    cos_dec = np.cos(np.radians(self.gaia_dec[tidx_5p]))
                    ra_d   = _get('delve_ra_cat',  0.0)[tidx_5p]
                    dec_d  = _get('delve_dec_cat', 0.0)[tidx_5p]
                    delta_v5 = np.column_stack([
                        (self.gaia_ra[tidx_5p]  - ra_d)  * cos_dec * 3.6e6,
                        (self.gaia_dec[tidx_5p] - dec_d) * 3.6e6,
                        _get('pmra',     0.0)[tidx_5p] - d_pmra[tidx_5p],
                        _get('pmdec',    0.0)[tidx_5p] - d_pmdec[tidx_5p],
                        _get('parallax', 0.0)[tidx_5p] - _get('delve_parallax', 0.0)[tidx_5p],
                    ])  # (m5, 5)
                    C_comb5 = self.C_survey[tidx_5p] + C_delve_dp[is_5p_dp]
                    m5 = len(tidx_5p)
                    chi2_5d = np.zeros(m5)
                    ok5 = has_full_5d_dp[is_5p_dp] & np.isfinite(delta_v5).all(axis=1)
                    if ok5.any():
                        x5 = np.linalg.solve(C_comb5[ok5],
                                             delta_v5[ok5, :, None]).squeeze(-1)
                        chi2_5d[ok5] = np.einsum('ni,ni->n', delta_v5[ok5], x5)
                    vetoed_5p = chi2_5d > 17.7
                    if vetoed_5p.any():
                        has_delve_pm[tidx_5p[vetoed_5p]] = False
                        print(f'  DELVE prior vetoed for {int(vetoed_5p.sum())} Gaia-5p stars '
                              f'(5D discrepant >3σ; chi2 threshold 17.7)')

                # ── Consistency veto for Gaia 2p + DELVE ────────────────────────
                # Position-only 2D chi2 (Gaia 2p has no PM/plx to compare).
                # threshold 11.8 = chi2.ppf(0.9973, df=2)
                is_2p_dp = (~self.full_gaia_astrometry[orig_dp_idx] &
                             has_full_5d_dp &
                             has_delve_pm[orig_dp_idx])  # exclude already-vetoed
                if is_2p_dp.any():
                    tidx_2p = orig_dp_idx[is_2p_dp]
                    cos_dec = np.cos(np.radians(self.gaia_dec[tidx_2p]))
                    ra_d   = _get('delve_ra_cat',  np.nan)[tidx_2p]
                    dec_d  = _get('delve_dec_cat', np.nan)[tidx_2p]
                    dpos = np.column_stack([
                        (self.gaia_ra[tidx_2p]  - ra_d)  * cos_dec * 3.6e6,
                        (self.gaia_dec[tidx_2p] - dec_d) * 3.6e6,
                    ])  # (m2, 2)
                    C_pos_comb = (self.C_survey[tidx_2p, :2, :2] +
                                  C_delve_dp[is_2p_dp, :2, :2])  # (m2, 2, 2)
                    m2 = len(tidx_2p)
                    chi2_2d = np.zeros(m2)
                    ok2 = np.isfinite(dpos).all(axis=1)
                    if ok2.any():
                        x2 = np.linalg.solve(C_pos_comb[ok2],
                                             dpos[ok2, :, None]).squeeze(-1)
                        chi2_2d[ok2] = np.einsum('ni,ni->n', dpos[ok2], x2)
                    vetoed_2p = chi2_2d > 11.8
                    if vetoed_2p.any():
                        has_delve_pm[tidx_2p[vetoed_2p]] = False
                        print(f'  DELVE prior vetoed for {int(vetoed_2p.sum())} Gaia-2p stars '
                              f'(position offset >3σ; chi2 threshold 11.8)')

                # ── Update C_survey_inv for surviving DELVE stars ────────────────
                # Use has_full_5d_dp survivors in orig_dp_idx after both vetoes.
                surv_in_dp = has_delve_pm[orig_dp_idx]  # (n_dp,) boolean
                if surv_in_dp.any():
                    surv_d_sig    = d_sig[surv_in_dp]         # (n_s, 5)
                    surv_C_delve  = C_delve_dp[surv_in_dp]    # (n_s, 5, 5)
                    surv_idx      = orig_dp_idx[surv_in_dp]   # global indices
                    surv_full5    = self.full_gaia_astrometry[surv_idx]
                    surv_2p       = ~surv_full5

                    # Gaia 5p survivors: add full DELVE 5×5 precision.
                    # DELVE positions (~4 mas) are from an independent DECam fit;
                    # DELVE position uncertainty >> Gaia 5p uncertainty so the
                    # position block contribution is negligible but correctly included.
                    # NOTE: DELVE used Gaia DR3 as its astrometric reference frame,
                    # so the two are not fully independent (minor approximation accepted).
                    if surv_full5.any():
                        idx5 = surv_idx[surv_full5]
                        C_delve_5p = surv_C_delve[surv_full5]     # (n5, 5, 5)
                        delve_full_5d_5p = (np.isfinite(surv_d_sig[surv_full5]).all(axis=1) &
                                    (surv_d_sig[surv_full5] > 0).all(axis=1))
                        # Mirror the 2p branch: a 5p star with an incomplete
                        # DELVE 5x5 never had the prior combined, so revert the
                        # flag — otherwise _has_delve_pm mislabels it Gaia+DELVE
                        # in plots and v_prior/C_prior get pointlessly recomputed.
                        if (~delve_full_5d_5p).any():
                            has_delve_pm[idx5[~delve_full_5d_5p]] = False
                        if delve_full_5d_5p.any():
                            C_delve_inv_5p = np.linalg.inv(C_delve_5p[delve_full_5d_5p])
                            idx5_5d = idx5[delve_full_5d_5p]
                            self.C_survey_inv[idx5_5d] += C_delve_inv_5p
                            cos_d5 = np.cos(np.radians(self.gaia_dec[idx5_5d]))
                            v_delve_5p = np.column_stack([
                                ((_get('delve_ra_cat',  0.0)[idx5_5d] -
                                  self.gaia_ra[idx5_5d]) * cos_d5 * 3.6e6),
                                ((_get('delve_dec_cat', 0.0)[idx5_5d] -
                                  self.gaia_dec[idx5_5d]) * 3.6e6),
                                d_pmra[idx5_5d],
                                d_pmdec[idx5_5d],
                                _get('delve_parallax', 0.0)[idx5_5d],
                            ])  # (n5d, 5); positions as mas offset from Gaia
                            self.C_survey_inv_dot_v[idx5_5d] += np.einsum(
                                'nij,nj->ni', C_delve_inv_5p, v_delve_5p)

                    # Gaia 2p survivors: add full DELVE 5×5 precision.
                    # C_survey_inv already has the Gaia 2p position 2×2 block;
                    # adding full C_delve_inv gives a well-defined 5×5 information matrix.
                    # For 2p stars where any DELVE sigma is invalid (NaN/≤0), we cannot
                    # form a full-rank 5×5 — revert has_delve_pm so the diffuse prior kicks in.
                    if surv_2p.any():
                        idx2 = surv_idx[surv_2p]
                        C_delve_2p = surv_C_delve[surv_2p]     # (n2, 5, 5)
                        delve_full_5d_2p = (np.isfinite(surv_d_sig[surv_2p]).all(axis=1) &
                                            (surv_d_sig[surv_2p] > 0).all(axis=1))
                        # Stars without a full valid 5×5 cannot get the DELVE prior and
                        # also cannot use the diffuse PM prior (already disabled via
                        # has_delve_pm). Revert them so they get the diffuse prior instead.
                        if (~delve_full_5d_2p).any():
                            has_delve_pm[idx2[~delve_full_5d_2p]] = False
                        if delve_full_5d_2p.any():
                            C_delve_inv_2p = np.linalg.inv(C_delve_2p[delve_full_5d_2p])
                            idx2_5d = idx2[delve_full_5d_2p]
                            self.C_survey_inv[idx2_5d] += C_delve_inv_2p
                            # v_delve in solver convention: positions as deviations from
                            # the Gaia catalog position in mas (= 0 when DELVE == Gaia);
                            # PM and parallax in mas/yr and mas.
                            cos_d2 = np.cos(np.radians(self.gaia_dec[idx2_5d]))
                            v_delve_2p = np.column_stack([
                                ((_get('delve_ra_cat',  0.0)[idx2_5d] -
                                  self.gaia_ra[idx2_5d]) * cos_d2 * 3.6e6),
                                ((_get('delve_dec_cat', 0.0)[idx2_5d] -
                                  self.gaia_dec[idx2_5d]) * 3.6e6),
                                d_pmra[idx2_5d],
                                d_pmdec[idx2_5d],
                                _get('delve_parallax', 0.0)[idx2_5d],
                            ])  # (n5d, 5)
                            self.C_survey_inv_dot_v[idx2_5d] += np.einsum(
                                'nij,nj->ni', C_delve_inv_2p, v_delve_2p)

            self._has_delve_pm = has_delve_pm   # stored for use in diffuse-prior block

            # Combined prior mean and covariance for the outlier test:
            #   v_prior[i] = C_prior[i] @ C_survey_inv_dot_v[i]
            #   C_prior[i] = inv(C_survey_inv[i])   (combined Gaia+DELVE)
            # For stars without DELVE this is identical to (v_survey, C_survey).
            # C_survey_inv is now well-defined for both Gaia 5p+DELVE (full Gaia 5D +
            # full DELVE 5D) and Gaia 2p+DELVE (Gaia 2p position + full DELVE 5D).
            self.v_prior = self.v_survey.copy()
            self.C_prior = self.C_survey.copy()
            if has_delve_pm.any():
                idx_dp = np.where(has_delve_pm)[0]
                C_prior_dp = np.linalg.inv(self.C_survey_inv[idx_dp])
                self.C_prior[idx_dp] = C_prior_dp
                self.v_prior[idx_dp] = np.einsum(
                    'nij,nj->ni', C_prior_dp, self.C_survey_inv_dot_v[idx_dp])

        else:
            self._has_delve_pm = np.zeros(self.n_stars, dtype=bool)
            self.v_prior = self.v_survey
            self.C_prior = self.C_survey

        # ── Per-star astrometry prior (Michalik et al. 2015) ──────────────────
        # Gaia 5p/6p: zero precision (no diffuse prior — Gaia covariance suffices).
        # Gaia 2p / HST-only: flat position prior + 100 mas/yr PM prior +
        #   magnitude/direction-dependent parallax prior (10 * sigma_F90).
        needs_diffuse = self.gaia_2p  # 2p; HST-only rows added by v2 loader extend this
        # Stored so the alpha estimator and the test-3 threshold reference can
        # exclude diffuse-prior stars with one definition (ported back from
        # ground_to_gaia_xmatch, where 55% 2p turned alpha_raw into 0.001 and
        # the adaptive threshold into its floor).  _add_gaia_epoch_obs extends
        # this for epoch stars, whose Gaia prior is replaced by the diffuse one.
        self.needs_diffuse = np.asarray(needs_diffuse, dtype=bool).copy()
        sigma_plx_prior = np.full(self.n_stars, np.inf)
        if needs_diffuse.any():
            sigma_plx_prior[needs_diffuse] = michalik_sigma_plx_prior(
                self.gaia_ra[needs_diffuse],
                self.gaia_dec[needs_diffuse],
                self.gaia_g[needs_diffuse],
            )

        # _C_VG_inv_per_star : (n_stars, 5) diagonal precision additions.
        # param order: (Δα*, Δδ, μα*, μδ, ϖ)
        # Stars with DELVE PM have their PM prior supplied via C_survey_inv (above);
        # the flat 100 mas/yr diffuse PM prior is disabled for them.
        self._C_VG_inv_per_star = np.zeros((self.n_stars, N_V), dtype=float)
        # h-side companion to _C_VG_inv_per_star.  The diffuse prior is a
        # DIAGONAL precision added to H_vv; with no matching information-vector
        # term its mean is implicitly zero.  Seeding a non-zero prior mean m
        # (e.g. the xmatch PM/parallax for v2 HST-only stars) therefore requires
        # h += C_VG_inv * m, which is what this array carries — writing the seed
        # into v_survey alone is a NO-OP for the solve, because
        # C_survey_inv_dot_v = C_survey_inv @ v_survey and the PM/plx rows of
        # C_survey_inv are zero for exactly the stars that have a diffuse prior.
        self._C_VG_h_per_star = np.zeros((self.n_stars, N_V), dtype=float)
        self._C_VG_inv_per_star[needs_diffuse, 0] = _SIGMA_POS**-2
        self._C_VG_inv_per_star[needs_diffuse, 1] = _SIGMA_POS**-2
        needs_diffuse_pm = needs_diffuse & ~self._has_delve_pm
        self._C_VG_inv_per_star[needs_diffuse_pm, 2] = _SIGMA_PM**-2
        self._C_VG_inv_per_star[needs_diffuse_pm, 3] = _SIGMA_PM**-2
        finite_plx = needs_diffuse & np.isfinite(sigma_plx_prior)
        self._C_VG_inv_per_star[finite_plx, 4] = sigma_plx_prior[finite_plx]**-2

        # _sigma_diff_per_star : (n_stars, 5) used in the diffuse-prior chi2
        # outlier test.  5p/6p get very large sigmas so the test never triggers.
        self._sigma_diff_per_star = np.full((self.n_stars, N_V), 1e9, dtype=float)
        self._sigma_diff_per_star[needs_diffuse, 0] = 1e4
        self._sigma_diff_per_star[needs_diffuse, 1] = 1e4
        self._sigma_diff_per_star[needs_diffuse_pm, 2] = _SIGMA_PM
        self._sigma_diff_per_star[needs_diffuse_pm, 3] = _SIGMA_PM
        # For DELVE-matched 2p stars, use DELVE PM error for outlier test
        if self._has_delve_pm.any():
            d_pmra_e_stored  = _get('delve_pmra_error',  np.inf)
            d_pmdec_e_stored = _get('delve_pmdec_error', np.inf)
            has_d2p = needs_diffuse & self._has_delve_pm
            self._sigma_diff_per_star[has_d2p, 2] = d_pmra_e_stored[has_d2p]
            self._sigma_diff_per_star[has_d2p, 3] = d_pmdec_e_stored[has_d2p]
        finite_plx = needs_diffuse & np.isfinite(sigma_plx_prior)
        self._sigma_diff_per_star[finite_plx, 4] = sigma_plx_prior[finite_plx]

    # ── Geometry precomputation ────────────────────────────────────────────────

    def _precompute_geometry(self):
        """
        For each image, precompute all star-level geometric quantities that
        don't depend on the current estimate of R_j (i.e., everything except C_s,i,j).
        """
        print("Precomputing geometry...")
        self._img_data = {}

        self.gaia_n_hst_used[:] = 0

        for img in self.image_names:
            meta  = self.images[img]
            df    = self.stars_per_image[img]
            mask  = df["Gaia_id"].isin(self.star_id_to_idx)
            df    = df[mask].copy().reset_index(drop=True)
            if len(df) == 0:
                self._img_data[img] = None
                continue

            n = len(df)
            sidx = np.array([self.star_id_to_idx[gid] for gid in df["Gaia_id"]])

            ra0, dec0 = meta["ra0"], meta["dec0"]
            # Initialize rolling tangent-point accumulator (reset each fit call)
            meta["ra0_current"]  = ra0
            meta["dec0_current"] = dec0
            # pscale    = meta["pixel_scale"]   # mas/pixel
            pscale    = meta["orig_pixel_scale"]   # mas/pixel
            Xo = meta.get("Xo", 2048.0)
            Yo = meta.get("Yo", 2048.0)
            meta['Xo'] = Xo
            meta['Yo'] = Yo
            hst_time  = Time(meta["hst_time_mjd"],format='mjd')
            hst_mjd   = hst_time.mjd
            hst_yr    = hst_time.jyear

            # Gaia positions for these stars
            ra_g  = self.gaia_ra[sidx]
            dec_g = self.gaia_dec[sidx]
            t_g= self.gaia_time[sidx]
            dt_yr = (hst_time - t_g).to(u.year).value   # time offset: negative = Gaia is after HST

            # Gaia pseudo-image positions (x_data): plane project relative to image center
            xs, ys = plane_project(ra_g, dec_g, ra0, dec0, pscale)  # (n,) pixels

            # Jacobian J_i,j: (n, 2, 2) in pix/mas
            J = plane_project_jacobian(ra_g, dec_g, ra0, dec0, pscale)

            # Tangent-point derivatives for Δα0, Δδ0 columns of X_mat (units: px/mas)
            dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = plane_project_tangent_derivs(
                ra_g, dec_g, ra0, dec0, pscale)

            # Parallax factors: difference between HST epoch and Gaia epoch
            #Gaia has already removed the parallax, so no need to subtract plx at J2016
            tele_xyz = get_tele_position(hst_time,curr_id='earth')
            meta['tele_XYZ'] = tele_xyz
            d_plx_ra,  d_plx_dec  = get_parallax_factors(ra_g, dec_g, tele_xyz)

            # U matrix for each star: (n, 2, 5)
            U_arr = build_U_matrices(dt_yr, d_plx_ra, d_plx_dec)

            # JU = J @ U: (n, 2, 5)
            JU = np.einsum('nij,njk->nik', J, U_arr)

            # Centered HST pixel positions: x_c = X - Xo, y_c = Y - Yo
            X_c = df["X"].to_numpy(float) - Xo
            Y_c = df["Y"].to_numpy(float) - Yo
            good_for_fitting = df['use_for_alignment'].to_numpy(bool).copy()
            # Phase-6 outlier flag: Gaia detections that start inactive but are
            # real matches that can be re-enabled by the EM loop if residuals
            # improve.  Separate from use_for_align_init so they don't inflate
            # the adaptive threshold (which uses init_trusted = use_for_align_init).
            _phase6_outlier = (
                (~df['use_for_alignment'].to_numpy(bool))
                & (df['use_for_align_init_flag'].to_numpy(bool))
                if 'use_for_align_init_flag' in df.columns
                else np.zeros(len(df), dtype=bool)
            )

            # #try removing saturated stars for first iteration
            q_hst_ok = df['q_hst'].to_numpy() > 0
            if np.sum(good_for_fitting & q_hst_ok) > 0:
                good_for_fitting &= q_hst_ok
            if np.sum(good_for_fitting & self.gaia_trustworthy[sidx]) > 0:
                good_for_fitting &= self.gaia_trustworthy[sidx]

            # X matrix: (n, 2, N_R)
            X_mat = build_X_matrices(
                X_c, Y_c, dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0,
                poly_order=self.poly_order)

            # HST position covariance C_hst: (n, 2, 2)
            x_err  = df["x_hst_err"].to_numpy(float)
            y_err  = df["y_hst_err"].to_numpy(float)
            xy_cor = df["xy_hst_corr"].fillna(0.).to_numpy(float)
            C_hst = np.zeros((n, 2, 2))
            for k in range(n):
                C_hst[k] = hst_position_cov(x_err[k], y_err[k], xy_cor[k])

            r_prior, C_r_prior_inv = _make_image_prior(
                meta, poly_order=self.poly_order,
                sigma_rot_deg  = self._sigma_rot_deg_cli,
                sigma_scale    = self._sigma_scale_cli,
                sigma_skew     = self._sigma_skew_cli,
                sigma_pointing = self._sigma_pointing_cli,
            )

            # ── Build r_init (initial iterate) ───────────────────────────────
            # When transformation.csv provides (a,b,c,d) from fast_cross_match,
            # use those as the starting point.  The prior (r_prior, C_r_prior_inv)
            # is computed solely from the WCS header and is never modified here.
            # r_init is a copy: changing it never changes the prior.
            fcm_abcd = meta.get("fcm_abcd")
            _n_fcm   = len(fcm_abcd) if fcm_abcd is not None else 0
            r_init = r_prior.copy()
            if _n_fcm:
                r_init[:_n_fcm] = fcm_abcd[:_n_fcm]

            # ── Initial residual screening ────────────────────────────────────
            # Used to permanently block implausible cross-matches (> 100 px after
            # subtracting the median bulk offset).  Without w/z, there is always a
            # bulk offset in r_init predictions, so we always subtract the median.
            # Use only PM and parallax (cols 2-4).  v_survey[:, 0:2] is zero by
            # construction (the Gaia position IS the reference; offsets live in
            # v_T), so slicing 2: just makes that explicit and future-proof.
            _v_pm_plx = np.zeros_like(self.v_survey[sidx])
            _v_pm_plx[:, 2:] = self.v_survey[sidx, 2:]
            ave_motion_offset = np.einsum('nij,nj->ni', JU, _v_pm_plx)
            xys = np.stack([xs, ys], axis=1)
            x_pred_init = np.einsum('nkl,l->nk', X_mat, r_init) - ave_motion_offset
            x_resid_init = xys - x_pred_init

            med_screen = (np.nanmedian(x_resid_init[good_for_fitting], axis=0)
                          if good_for_fitting.any() else np.zeros(2))
            x_resid_corr = x_resid_init - med_screen
            resid_mag = np.hypot(x_resid_corr[:, 0], x_resid_corr[:, 1])

            ok_init = resid_mag <= _INIT_RESID_CLIP_PX  # (n,) hard ceiling mask

            if good_for_fitting.any():
                n_before = int(good_for_fitting.sum())
                good_for_fitting = good_for_fitting & ok_init
                n_rej = n_before - int(good_for_fitting.sum())
                if n_rej > 0:
                    print(f"    {img}: rejected {n_rej}/{n_before} stars with "
                          f"initial residual > {_INIT_RESID_CLIP_PX:.0f} px")

            self.gaia_n_hst_used[sidx[good_for_fitting]] += 1

            self._img_data[img] = {
                "sidx"           : sidx,              # (n,) global star indices
                "n"              : n,
                "xys"            : xys,               # (n, 2) Gaia pseudo-image xy
                "JU"             : JU,                # (n, 2, 5)
                "X_mat"          : X_mat,             # (n, 2, N_R)
                "B_mat"          : (epoch_distortion_basis(
                                        X_c, Y_c, self._ed_order,
                                        half_x=2048.0,
                                        half_y=(507.0 if str(meta.get('detector','')).upper() == 'IR'
                                                else 1024.0))
                                    if (self._ed_enabled and self._ed_gidx.get(img, -1) >= 0)
                                    else None),   # (n, 2, ED_K) epoch-distortion basis
                "ed_sigma_px"    : (self._ed_sigma_mas / meta["orig_pixel_scale"]
                                    if self._ed_enabled else None),
                "C_hst"          : C_hst,             # (n, 2, 2) — may be inflated
                "C_hst_orig"     : C_hst.copy(),      # (n, 2, 2) — original, never modified
                "X_c"            : X_c,               # (n,) centered HST x (needed for poly Jacobian)
                "Y_c"            : Y_c,               # (n,) centered HST y
                "r_prior"        : r_prior,           # (N_R,) prior mean — from WCS header only
                "r_init"         : r_init,            # (N_R,) initial iterate — from transformation.csv when available
                "C_r_prior_inv"  : C_r_prior_inv,     # (N_R, N_R)
                "use_for_fit"    : good_for_fitting,  # (n,) boolean — used for alignment
                # use_for_astrom: alignment stars PLUS sources excluded from alignment
                # (e.g. DELVE-only stars) whose HST positions still constrain their
                # own v_hat via h_all (astrometry-only path in _solve_one_pass).
                "use_for_astrom" : good_for_fitting | df['use_for_fit'].to_numpy(bool),
                "use_for_fit_max": ok_init.copy(),    # hard ceiling: only blocks 100px+ outliers
                # Frozen snapshot of initially-trusted stars (use_for_alignment=True,
                # q_hst>0, gaia_trustworthy, initial residual ≤ 100px).  Used as the
                # reference population for the test-3 adaptive threshold so that
                # sources initially excluded (e.g. non-stars with inflated PSF
                # covariances) cannot bias the threshold even if they are later
                # re-evaluated by the EM loop.
                "use_for_align_init": good_for_fitting.copy(),  # False for Phase-6 outliers (don't inflate threshold)
                "phase6_outlier":     _phase6_outlier.copy(),   # True for Phase-6 Gaia outliers (astrometry-tier re-admission only)
            }

        n_total = sum(d["n"] for d in self._img_data.values() if d)
        print(f"  Done: {n_total} star-image pairs across {self.n_images} images.")

        # Diagnostic: track 5p/6p/2p stars through the three admission gates.
        # Gate 1: star appears in at least one image's sidx (has an HST match).
        # Gate 2: star has use_for_fit=True in at least one image after the
        #         ruwe/q_hst/ok_init filters (gaia_n_hst_used > 0).
        # Gate 3: (future) EM loop re-admission.
        in_images = np.zeros(self.n_stars, bool)
        for d in self._img_data.values():
            if d is not None:
                in_images[d["sidx"]] = True
        admitted = self.gaia_n_hst_used > 0

        pop_labels = [('5p', self.gaia_5p), ('6p', self.gaia_6p),
                      ('2p', self.gaia_2p)]
        rows = []
        for label, mask in pop_labels:
            n_cat   = int(mask.sum())
            n_img   = int((mask & in_images).sum())
            n_admit = int((mask & admitted).sum())
            n_excl  = n_img - n_admit
            rows.append((label, n_cat, n_img, n_admit, n_excl))
        hdr = f"  {'pop':>3}  {'catalog':>7}  {'in_images':>9}  {'admitted':>8}  {'excluded':>8}"
        print(hdr)
        for label, n_cat, n_img, n_admit, n_excl in rows:
            excl_str = f"  ← {n_excl} lost to ruwe/q_hst/resid filter" if n_excl > 0 else ""
            print(f"  {label:>3}  {n_cat:>7}  {n_img:>9}  {n_admit:>8}  {n_excl:>8}{excl_str}")

        # ── Build _hi/_lo chip pair index and precomputed coupling matrices ──────
        self._chip_pairs = []
        self._chip_pair_couplings = {}
        if self._use_pair_prior:
            hi_map, lo_map = {}, {}
            for j_idx, name in enumerate(self.image_names):
                if name.endswith('_hi'):
                    hi_map[name[:-3]] = j_idx
                elif name.endswith('_lo'):
                    lo_map[name[:-3]] = j_idx
            for root, hi_idx in hi_map.items():
                if root not in lo_map:
                    continue
                lo_idx = lo_map[root]
                meta_hi = self.images[self.image_names[hi_idx]]
                C_cp = _make_pair_coupling_inv(
                    meta_hi, self.poly_order,
                    sigma_pair_rot_deg  = self._sigma_pair_rot_deg_cli,
                    sigma_pair_scale    = self._sigma_pair_scale_cli,
                    sigma_pair_skew     = self._sigma_pair_skew_cli,
                    sigma_pair_pointing = self._sigma_pair_pointing_cli,
                )
                self._chip_pairs.append((hi_idx, lo_idx))
                self._chip_pair_couplings[(hi_idx, lo_idx)] = C_cp
            if self._chip_pairs:
                print(f"  Chip-pair coupling prior: {len(self._chip_pairs)} hi/lo pair(s)  "
                      f"σ_pair=(rot={self._prior_sigma_pair_rot_deg}°, "
                      f"scale={self._prior_sigma_pair_scale}, "
                      f"skew={self._prior_sigma_pair_skew}, "
                      f"point={self._prior_sigma_pair_pointing}mas)")

    def _init_transforms(self):
        """Initialise R_j (2×2 rotation matrix) from header info."""
        self.R = {}
        for img in self.image_names:
            meta = self.images[img]
            # a, b, c, d = abcd_from_rotation_pixscale_skew(
            #     meta["rotation_deg"], meta["pixel_scale_ratio"],
            #     meta["on_skew"], meta["off_skew"])
            # a, b, c, d = meta['AG'], meta['BG'], meta['CG'], meta['DG']
            rot_rad = meta["orig_rot_deg"] * DEG2RAD
            s = 1.0
            a = np.cos(rot_rad)
            b = np.sin(rot_rad)
            c = -np.sin(rot_rad)
            d = np.cos(rot_rad)

            self.R[img] = rotation_matrix_from_abcd(a, b, c, d)

    # ── Geometry update (called every fit iteration) ──────────────────────────

    def _update_geometry(self, r_hat, v_hat):
        """
        Recompute per-image geometry (xys, JU, X_mat, X_c, Y_c) using the
        current best estimates of stellar positions and the tangent-point shifts.

        After convergence of a few iterations, the updated stellar positions
        (from v_hat) and the tangent-point corrections (Δα0, Δδ0 in r_hat)
        can shift xys, the Jacobians J, and the parallax factors enough to
        matter.  This method recomputes all position-dependent quantities in
        _img_data except C_hst (which depends only on the HST measurement and
        the transformation Jacobian, the latter handled by _compute_Cs).

        Parameters
        ----------
        r_hat : (n_r,)        current image transformation vector
        v_hat : (n_stars, 5)  current stellar astrometry estimate
                               v_hat[:,0:2] = (Δα*, Δδ) offsets from Gaia [mas]
        """
        import astropy.units as u
        from astropy.time import Time

        self.gaia_n_hst_used[:] = 0
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue

            meta    = self.images[img]
            r_j     = r_hat[j_idx * nr:(j_idx + 1) * nr]
            sidx    = d["sidx"]
            use_align  = d["use_for_fit"]
            use_astrom = d.get("use_for_astrom", use_align)
            n       = d["n"]

            self.gaia_n_hst_used[sidx[use_align | use_astrom]] += 1

            pscale   = meta["orig_pixel_scale"]

            # ── Accumulate tangent-point correction into ra0_current ──────────
            # r_j[4] = Δα0 in mas where Δα0 = (ra0_current - ra0_true)*3.6e6,
            # so ra0_true = ra0_current - Δα0/3.6e6.  Subtract (not add) to
            # move toward ra0_true.
            meta["ra0_current"]  -= r_j[4] / 3_600_000.0   # mas → degrees
            meta["dec0_current"] -= r_j[5] / 3_600_000.0
            meta["ra0_final"]   = meta["ra0_current"]
            meta["dec0_final"]  = meta["dec0_current"]
            # Reset residual in r_hat so next solve starts from Δα0=0 at the new point
            r_hat[j_idx * nr + 4] = 0.0
            r_hat[j_idx * nr + 5] = 0.0

            ra0_tp  = meta["ra0_current"]
            dec0_tp = meta["dec0_current"]

            # ── Updated stellar RA/Dec from v_hat[:,0:2] ─────────────────────
            # v_hat[:,0] = Δα* [mas], v_hat[:,1] = Δδ [mas]
            ra_g_orig  = self.gaia_ra[sidx]
            dec_g_orig = self.gaia_dec[sidx]

            # Δα* = Δα · cos(δ), so Δα = Δα* / cos(δ)
            cos_dec = np.cos(dec_g_orig * DEG2RAD)
            ra_g_up  = ra_g_orig  + v_hat[sidx, 0] / (cos_dec * RAD2MAS)  # degrees
            dec_g_up = dec_g_orig + v_hat[sidx, 1] / RAD2MAS              # degrees

            # dt_yr depends only on the (fixed) image and Gaia epochs — cache
            # it: the astropy Time arithmetic (TDB conversion) costs ~2 erfa
            # dtdb calls per image per solve pass otherwise.
            dt_yr = d.get("dt_yr_cache")
            if dt_yr is None:
                hst_time = Time(meta["hst_time_mjd"], format="mjd")
                t_g   = self.gaia_time[sidx]
                dt_yr = (hst_time - t_g).to(u.year).value
                d["dt_yr_cache"] = dt_yr

            # ── Recompute projected Gaia positions at current tangent point ───
            # xys, J, and tangent-point derivatives are all evaluated at ra0_current
            # (the accumulated best-fit tangent point).  r_j[4:6] are now 0, so
            # X_mat @ r_j contributes nothing from the pointing columns, and the
            # residuals xys - X_mat @ r_j correctly represent the remaining error.
            # Recomputing J at ra0_current keeps the linearisation accurate when
            # the total offset is large.
            xs, ys = plane_project(ra_g_orig, dec_g_orig, ra0_tp, dec0_tp, pscale)
            xys    = np.stack([xs, ys], axis=1)

            # ── Recompute Jacobian J and tangent-point derivatives ────────────
            J = plane_project_jacobian(ra_g_orig, dec_g_orig, ra0_tp, dec0_tp, pscale)
            dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = \
                plane_project_tangent_derivs(ra_g_orig, dec_g_orig, ra0_tp, dec0_tp, pscale)

            # ── Recompute parallax factors ────────────────────────────────────
            #DO use the new best fit RA,Dec positions here
            tele_xyz = meta['tele_XYZ'] 
            d_plx_ra, d_plx_dec = get_parallax_factors(ra_g_up, dec_g_up, tele_xyz)

            # ── Rebuild U and JU ─────────────────────────────────────────────
            U_arr = build_U_matrices(dt_yr, d_plx_ra, d_plx_dec)
            JU = np.einsum('nij,njk->nik', J, U_arr)

            # ── Rebuild X_mat ────────────────────────────────────────────────
            X_c = d["X_c"]   # unchanged — detector positions don't move
            Y_c = d["Y_c"]
            X_mat = build_X_matrices(
                X_c, Y_c, dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0,
                poly_order=self.poly_order)

            d["xys"]  = xys
            d["JU"]   = JU
            d["X_mat"] = X_mat

    # ── Gaia DR4 epoch AL observations (future) ───────────────────────────────
    #
    # When use_gaia_al_obs=True, each Gaia CCD transit contributes a 1-D
    # observation equation to the normal equations alongside the HST detections.
    #
    # For transit k of source i the AL measurement predicts:
    #
    #   ψ_ik = sθ·Δα*_i·cos(δ_i) + cθ·Δδ_i
    #          + sθ·dt_k·μα*_i + cθ·dt_k·μδ_i
    #          + f_al_k·ϖ_i
    #
    # where sθ = sin(scan_pos_angle_k), cθ = cos(scan_pos_angle_k),
    #       f_al_k = parallax_factor_al_k,
    #       dt_k = obs_time_jyear_k − ref_epoch_dr4   (years)
    #
    # The design vector a_ik in the 5-D parameter space (Δα*, Δδ, μα*, μδ, ϖ):
    #   a_ik = [sθ·cos(δ), cθ, sθ·dt_k, cθ·dt_k, f_al_k]
    #
    # Measurement residual (relative to AGIS solution):
    #   r_ik = centroid_pos_al_k − ψ_ik(v_agis_i)
    #        = zeta_k   [pre-stored in the epoch table as 'zeta' if available]
    #
    # Effective per-transit variance:
    #   σ²_ik = centroid_pos_error_al_k² + agis_source_excess_noise_i²
    #
    # Unlike HST images, Gaia has no per-epoch "image transformation" to solve
    # for — scan_pos_angle and parallax_factor_al already encode the full
    # geometry.  The Gaia epoch observations therefore contribute only to H_vv
    # and h_all (not to H_rr or K_img), making them pure stellar-parameter
    # constraints.
    #
    # AGIS down-weighting:  transits with used_by_agis_al=False were rejected
    # by the official AGIS solution (likely due to image parameter quality).
    # These should be down-weighted in the first BP3M iteration, then
    # re-admitted after iterative convergence (analogous to gaiasupdate's
    # huber_downweight / agis_weights scheme).
    #
    # Integration plan:
    #   1. Load epoch DataFrames into the solver via _load_gaia_epoch_obs()
    #   2. In _solve_one_iter(), after the HST loop, add the Gaia AL
    #      contributions with:
    #        H_vv[i] += (a_ik ⊗ a_ik) / σ²_ik   (outer product)
    #        h_all[i] += a_ik * r_ik / σ²_ik
    #   3. Use_for_fit masking already handles which sources are active;
    #      transits of masked sources are skipped.

    def _add_gaia_epoch_obs(self, epoch_obs_preprocessed: dict) -> None:
        """Register precomputed Gaia DR4 AL normal-equation contributions.

        For each star with epoch data:
          - Removes the Gaia 5p summary-solution prior from C_survey_inv /
            C_survey_inv_dot_v (to avoid double-counting the epoch information)
          - Replaces it with the same diffuse prior used for 2p/HST-only stars
            (flat position, 100 mas/yr PM, Michalik parallax prior)

        Parameters
        ----------
        epoch_obs_preprocessed
            Dict source_id (int64) → {'H_contrib': (5,5), 'h_contrib': (5,),
            'n_transits': int, 'n_flagged': int}, as returned by
            bp3m.pipeline.download_gaia_epoch.prepare_epoch_obs_for_solver().
        """
        self._gaia_epoch_contrib = epoch_obs_preprocessed

        epoch_indices = [self.star_id_to_idx[sid] for sid in epoch_obs_preprocessed
                         if sid in self.star_id_to_idx]
        if not epoch_indices:
            print("[Solver] Gaia DR4 epoch obs: no matched sources in catalog")
            return

        idx_ep = np.array(epoch_indices, dtype=int)

        # ── Zero out the Gaia 5p prior for all epoch stars ────────────────────
        # The 5p summary solution is derived from the same epoch transits we are
        # incorporating directly, so using it as a prior would double-count.
        _hd = getattr(self, '_has_delve_pm', None)
        if _hd is not None and bool(_hd[idx_ep].any()):
            _n_wiped = int(_hd[idx_ep].sum())
            print(f"[Solver] WARNING: {_n_wiped} epoch star(s) had a combined "
                  f"DELVE prior; zeroing the Gaia prior for epoch data discards "
                  f"it too (DELVE + epoch-AL combination is not implemented). "
                  f"_has_delve_pm cleared for them so downstream labels stay "
                  f"truthful.")
            _hd[idx_ep] = False
        self.C_survey_inv[idx_ep]       = 0.0
        self.C_survey_inv_dot_v[idx_ep] = 0.0

        # Compute Michalik parallax prior for epoch stars and install diffuse
        # prior into _C_VG_inv_per_star (same treatment as 2p/HST-only stars)
        sigma_plx_ep = michalik_sigma_plx_prior(
            self.gaia_ra[idx_ep], self.gaia_dec[idx_ep], self.gaia_g[idx_ep]
        )
        self._C_VG_inv_per_star[idx_ep, 0] = _SIGMA_POS**-2
        self._C_VG_inv_per_star[idx_ep, 1] = _SIGMA_POS**-2
        self._C_VG_inv_per_star[idx_ep, 2] = _SIGMA_PM**-2
        self._C_VG_inv_per_star[idx_ep, 3] = _SIGMA_PM**-2
        fin_ep = np.isfinite(sigma_plx_ep)
        self._C_VG_inv_per_star[idx_ep[fin_ep], 4] = sigma_plx_ep[fin_ep]**-2
        self._sigma_diff_per_star[idx_ep, 0] = 1e4
        self._sigma_diff_per_star[idx_ep, 1] = 1e4
        self._sigma_diff_per_star[idx_ep, 2] = _SIGMA_PM
        self._sigma_diff_per_star[idx_ep, 3] = _SIGMA_PM
        # Their Gaia prior was just zeroed and replaced by the diffuse prior, so
        # they must also leave the alpha / test-3 reference populations.
        self.needs_diffuse[idx_ep] = True
        self._sigma_diff_per_star[idx_ep[fin_ep], 4] = sigma_plx_ep[fin_ep]

        # ── Zero out the Gaia 2p position prior for all 2p stars ──────────────
        # The 2p position estimate and its uncertainty are also derived from epoch
        # AL observations; using them as priors when epoch data is incorporated
        # directly would double-count the same underlying measurements.
        idx_2p = np.where(self.gaia_2p)[0]
        if len(idx_2p):
            self.C_survey_inv[idx_2p]       = 0.0
            self.C_survey_inv_dot_v[idx_2p] = 0.0
            # _C_VG_inv_per_star already has the diffuse prior for 2p stars
            # (set in _load_star_data), so no further changes needed there.

        n_src     = len(epoch_obs_preprocessed)
        n_matched = len(epoch_indices)
        n_2p      = len(idx_2p)
        print(f"[Solver] Gaia DR4 epoch obs: {n_src} sources, "
              f"{n_matched} matched; Gaia prior zeroed for {n_matched} epoch "
              f"stars + {n_2p} 2p stars → diffuse prior only")

    # ── Core solver ────────────────────────────────────────────────────────────

    def _compute_Cs(self, img, r_j=None):
        """
        Transformed HST covariance: C_s,k = J_k @ C_hst,k @ J_k^T.

        For poly_order=1, J_k = R_j = [[a,b],[c,d]] (constant across stars).
        For poly_order>1, J_k is the full position-dependent Jacobian of the
        transformation evaluated at each star's (X_c, Y_c) position.

        Parameters
        ----------
        img  : str   image name
        r_j  : (N_R,) array or None
            Current r_j for this image. Required for poly_order > 1.
            If None (or poly_order==1), falls back to the cached R matrix.

        Returns
        -------
        C_s : (n, 2, 2) ndarray
        """
        d = self._img_data[img]
        C_hst = d["C_hst"]   # (n, 2, 2)

        if self.poly_order == 1 or r_j is None:
            R = self.R[img]
            return R @ C_hst @ R.T   # broadcasts over n

        # Higher-order: per-star Jacobian
        J = compute_poly_jacobian(r_j, d["X_c"], d["Y_c"], self.poly_order)
        # J: (n, 2, 2),  C_hst: (n, 2, 2)
        return np.einsum('nij,njk,nlk->nil', J, C_hst, J)

    def _solve_one_pass(self, r_current, z_weights=None, need_cov=True):
        """
        Single pass of the linear solver, working in RESIDUAL coordinates to
        avoid catastrophic cancellation.

        We solve for Δr = r - r_current and v_T,i given the residuals
            x_resid_{i,j} = x_data_{i,j} - X_{i,j} @ r_j_current

        which are small (~few pixels) even though absolute coordinates are large
        (~2000 pixels).

        Parameters
        ----------
        z_weights : dict {img: (n,) float} or None
            When provided, soft-weight IRLS mode.  Each entry replaces the
            hard use_for_fit/use_for_astrom flags with use_for_fit_max (Phase-0
            hard floor), and scales Cs_inv by z for each detection.

        Returns
        -------
        r_hat  : (n_r,)          absolute r (= r_current + Δr)
        C_r    : (n_r, n_r)      posterior covariance of r
        a_arr  : (n_stars, 5)    astrometry mean when Δr=0 (i.e., at r=r_current)
        K_img  : dict{img->(n,5,8)}
        C_vT   : (n_stars, 5, 5) astrometry posterior covariance conditional on r
        """
        nr = self.N_R
        n_r = nr * self.n_images
        n_s = n_r + self.n_ed          # shared dim: per-image r blocks + epoch-D
        if self.n_ed and r_current.size == n_r:
            # legacy-length input: append zero epoch-distortion coefficients
            r_current = np.concatenate([r_current, np.zeros(self.n_ed)])

        def _cols_of(j_idx, img):
            base = np.arange(j_idx * nr, j_idx * nr + nr)
            ec = self._ed_cols(img)
            return base if ec is None else np.concatenate([base, ec])

        # ── Precision matrices and information vectors ─────────────────────────
        H_vv = self.C_survey_inv.copy()
        H_vv[:, np.arange(N_V), np.arange(N_V)] += self._C_VG_inv_per_star

        # h_align: prior + alignment contributions only.
        #   Used in the Schur complement rhs so that image-calibration parameters
        #   are driven only by alignment detections.  Prevents slow convergence
        #   caused by astrometry-only residuals (which depend on r_j of other
        #   images) creating indirect cross-image coupling.
        # h_all: prior + alignment + astrometry-only contributions.
        #   Used to compute the returned stellar posteriors so that astrometry-only
        #   detections constrain each star's own v_hat.
        _vg_h  = getattr(self, '_C_VG_h_per_star', None)
        h_base = (self.C_survey_inv_dot_v + _vg_h
                  if _vg_h is not None else self.C_survey_inv_dot_v)
        h_align = h_base.copy()
        h_all   = h_base.copy()

        H_rr = np.zeros((n_s, n_s))

        K_img = {}
        XCs_xresid = {}

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data[img]
            cs = j_idx * nr

            if d is None:
                K_img[img] = None
                continue

            dropped = d.get("_dropped_by_2p_check", False)

            if z_weights is not None:
                z          = z_weights[img]
                use_align  = d["use_for_fit"]
                use_astrom = (d["use_for_fit"]
                              | d.get("use_for_astrom",
                                      d["use_for_fit"]))
            else:
                use_align  = d["use_for_fit"]
                use_astrom = d.get("use_for_astrom", use_align)

            # When exclude_2p_from_alignment is set, 2p stars do not contribute
            # to the image transformation equations (H_rr, h_r).
            if self.exclude_2p_from_alignment:
                sidx_all = d["sidx"]
                not_2p = ~self.gaia_2p[sidx_all]
                use_align = use_align & not_2p
            use_any    = use_align | use_astrom   # for H_vv/h_all (stellar precision)
            sidx_any   = d["sidx"][use_any]
            sidx_align = d["sidx"][use_align]
            JU   = d["JU"]       # (n, 2, 5)
            X    = d["X_mat"]    # (n, 2, N_R)
            xys  = d["xys"]      # (n, 2)

            r_j  = r_current[cs:cs + nr]
            cols = _cols_of(j_idx, img)
            if len(cols) > nr:                      # epoch-distortion active
                X = np.concatenate([X, d["B_mat"]], axis=2)   # (n, 2, nr+K)
            s_j = r_current[cols]

            Cs     = self._compute_Cs(img, r_j)   # (n, 2, 2)
            Cs_inv = np.linalg.inv(Cs)

            if z_weights is not None:
                Cs_inv = Cs_inv * z[:, None, None]

            x_pred  = np.einsum('nkl,l->nk', X, s_j)
            x_resid = xys - x_pred

            JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)

            # Images dropped due to insufficient non-2p alignment stars are
            # excluded entirely: no H_vv, no H_rr data, no Schur correction.
            # Still add the prior so the H_rr block is not zero.
            if dropped:
                H_rr[cs:cs+nr, cs:cs+nr] += d["C_r_prior_inv"]
                K_img[img] = None
                continue

            # H_vv/h_all: stellar astrometry from all used detections
            np.add.at(H_vv, sidx_any, np.einsum('nik,nkj->nij', JUT_Cs[use_any], JU[use_any]))
            np.subtract.at(h_all, sidx_any, np.einsum('nik,nk->ni', JUT_Cs[use_any], x_resid[use_any]))

            # h_align: residual information from alignment detections only
            # (used in Schur complement rhs to avoid cross-image coupling)
            np.subtract.at(h_align, sidx_align, np.einsum('nik,nk->ni', JUT_Cs[use_align], x_resid[use_align]))

            K = np.einsum('nik,nkl->nil', JUT_Cs, X)   # (n, 5, N_R)
            K_img[img] = K

            # H_rr/XCs_xresid: alignment stars only (calibrate image transform)
            XCsX = np.einsum('nki,nkl,nlj->ij', X[use_align], Cs_inv[use_align], X[use_align])
            H_rr[np.ix_(cols, cols)] += XCsX
            XCs_xresid[img] = np.einsum('nki,nkl,nl->ni', X[use_align], Cs_inv[use_align], x_resid[use_align])

            H_rr[cs:cs+nr, cs:cs+nr] += d["C_r_prior_inv"]

        # ── Gaia DR4 epoch AL contributions ───────────────────────────────────
        if getattr(self, '_gaia_epoch_contrib', None):
            for source_id, contrib in self._gaia_epoch_contrib.items():
                if source_id not in self.star_id_to_idx:
                    continue
                i = self.star_id_to_idx[source_id]
                H_vv[i] += contrib['H_contrib']
                h_all[i] += contrib['h_contrib']
                # 2p epoch contributions excluded from alignment when flag is set.
                if not (self.exclude_2p_from_alignment and self.gaia_2p[i]):
                    h_align[i] += contrib['h_contrib']

        # ── Epoch-distortion coefficient prior: diagonal, sigma in pixels ────
        # zero-centred by default; set_epoch_dist_prior() recentres a group on
        # imported calibration coefficients with per-coefficient widths.
        if self.n_ed:
            for g, grp in enumerate(self.ed_groups):
                if grp.get('prior_prec') is not None:
                    prec = grp['prior_prec']
                else:
                    sig_px = self._img_data[grp['images'][0]]["ed_sigma_px"]
                    prec = np.full(self.ED_K, sig_px ** -2)
                off = n_r + g * self.ED_K
                H_rr[np.arange(off, off + self.ED_K),
                     np.arange(off, off + self.ED_K)] += prec

        # ── Hi/lo chip coupling prior: off-diagonal blocks in H_rr ───────────
        for hi_idx, lo_idx in self._chip_pairs:
            C_cp = self._chip_pair_couplings[(hi_idx, lo_idx)]
            hi_cs, lo_cs = hi_idx * nr, lo_idx * nr
            H_rr[hi_cs:hi_cs+nr, hi_cs:hi_cs+nr] += C_cp
            H_rr[lo_cs:lo_cs+nr, lo_cs:lo_cs+nr] += C_cp
            H_rr[hi_cs:hi_cs+nr, lo_cs:lo_cs+nr] -= C_cp
            H_rr[lo_cs:lo_cs+nr, hi_cs:hi_cs+nr] -= C_cp

        # ── Invert H_vv → C_vT ────────────────────────────────────────────────
        C_vT    = np.linalg.inv(H_vv)
        a_align = np.einsum('nij,nj->ni', C_vT, h_align)  # for Schur complement rhs
        a       = np.einsum('nij,nj->ni', C_vT, h_all)    # returned stellar posteriors

        # ── Schur complement for the shared parameters (r blocks + epoch-D) ──
        Cr_inv = H_rr.copy()
        rhs    = np.zeros(n_s)

        # Star-major Schur assembly (default): the diagonal K^T C_vT K and all
        # image-pair fill-in blocks together equal B~^T B~ with B~ the sparse
        # matrix whose 5-row block per star holds the Cholesky-whitened K
        # blocks scattered into each image's columns. One sparse product
        # replaces the O(n_img^2) pair loop with its per-pair intersect1d —
        # mathematically identical. Set BP3M_SCHUR_PAIR_MAJOR=1 to restore the
        # legacy pair-major loop (A/B validation).
        import os as _os
        _pair_major = bool(int(_os.environ.get('BP3M_SCHUR_PAIR_MAJOR', '0')))
        _schur_obs = []   # (sidx, K, cols) per image, star-major path

        # Epoch-D prior rhs: pull toward the prior mean from the current iterate
        if self.n_ed:
            for g, grp in enumerate(self.ed_groups):
                if grp.get('prior_prec') is not None:
                    prec = grp['prior_prec']
                    mu = grp['prior_mean']
                else:
                    sig_px = self._img_data[grp['images'][0]]["ed_sigma_px"]
                    prec = np.full(self.ED_K, sig_px ** -2)
                    mu = 0.0
                off = n_r + g * self.ED_K
                rhs[off:off + self.ED_K] += (mu - r_current[off:off + self.ED_K]) * prec

        for j_idx, img in enumerate(self.image_names):
            r_prior_j      = self._img_data[img]["r_prior"]
            Cr_prior_inv_j = self._img_data[img]["C_r_prior_inv"]
            cs = j_idx * nr
            rhs[cs:cs+nr] += Cr_prior_inv_j @ (r_prior_j - r_current[cs:cs+nr])

            d = self._img_data[img]
            if d is None or K_img[img] is None:
                continue
            cols = _cols_of(j_idx, img)
            use  = d["use_for_fit"]
            # The Schur correction K^T C_v K must use the same star set as
            # H_rr (XCsX).  When 2p stars are excluded from alignment, K must
            # also exclude them so the Schur complement remains consistent.
            if self.exclude_2p_from_alignment:
                use = use & ~self.gaia_2p[d["sidx"]]
            sidx = d["sidx"][use]
            K    = K_img[img][use]

            rhs[cols] += XCs_xresid[img].sum(axis=0)
            rhs[cols] += np.einsum('nji,nj->i', K, a_align[sidx])

            if not _pair_major:
                _schur_obs.append((sidx, K, cols))
                continue

            CvT_K    = np.einsum('nij,njk->nik', C_vT[sidx], K)
            KT_CvT_K = np.einsum('nji,njk->ik',  K, CvT_K)
            Cr_inv[np.ix_(cols, cols)] -= KT_CvT_K

            for j2_idx, img2 in enumerate(self.image_names):
                if j2_idx <= j_idx:
                    continue
                d2 = self._img_data[img2]
                if d2 is None or K_img[img2] is None:
                    continue
                use2 = d2["use_for_fit"]
                if self.exclude_2p_from_alignment:
                    use2 = use2 & ~self.gaia_2p[d2["sidx"]]
                sidx2 = d2["sidx"][use2]
                K2    = K_img[img2][use2]

                common, idx1, idx2 = np.intersect1d(sidx, sidx2,
                                                     return_indices=True)
                if len(common) == 0:
                    continue

                CvT_c  = C_vT[common]
                CvT_K2 = np.einsum('nij,njk->nik', CvT_c, K2[idx2])
                block  = np.einsum('nji,njk->ik', K[idx1], CvT_K2)

                cols2 = _cols_of(j2_idx, img2)
                Cr_inv[np.ix_(cols,  cols2)] -= block
                Cr_inv[np.ix_(cols2, cols)]  -= block.T

        # ── Star-major Schur fill-in: Cr_inv -= B~^T B~ ──────────────────────
        if not _pair_major and _schur_obs:
            from scipy import sparse as _sp
            try:
                L_chol = np.linalg.cholesky(C_vT)          # C_vT = L L^T
            except np.linalg.LinAlgError:
                w_e, Q_e = np.linalg.eigh(C_vT)
                L_chol = Q_e * np.sqrt(np.clip(w_e, 1e-30, None))[:, None, :]
            # Accumulate B~ in bounded chunks (int32 indices, csr-summed) so
            # dense many-image fields don't spike transient memory: one giant
            # COO concat at 200+ dense images costs ~10 GB and risks the
            # per-user OOM kill.
            _shape = (5 * C_vT.shape[0], n_s)
            B_til = None
            data_l, rows_l, cols_l, _budget = [], [], [], 0

            def _flush_B(B_prev):
                B_part = _sp.csr_matrix(
                    (np.concatenate(data_l),
                     (np.concatenate(rows_l), np.concatenate(cols_l))),
                    shape=_shape)
                data_l.clear(); rows_l.clear(); cols_l.clear()
                return B_part if B_prev is None else B_prev + B_part

            for sidx_o, K_o, cols_o in _schur_obs:
                n_o, _, W_o = K_o.shape
                if n_o == 0:
                    continue
                # (L^T K): row block of B~ for each observation
                ltk = np.einsum('nji,njw->niw', L_chol[sidx_o], K_o)
                r_b = (5 * sidx_o.astype(np.int32))[:, None, None] \
                    + np.arange(5, dtype=np.int32)[None, :, None]
                data_l.append(ltk.ravel())
                rows_l.append(np.broadcast_to(r_b, (n_o, 5, W_o))
                              .ravel().astype(np.int32, copy=False))
                cols_l.append(np.broadcast_to(
                    cols_o.astype(np.int32)[None, None, :],
                    (n_o, 5, W_o)).ravel().astype(np.int32, copy=False))
                _budget += n_o * 5 * W_o
                if _budget >= 30_000_000:
                    B_til = _flush_B(B_til)
                    _budget = 0
            if data_l:
                B_til = _flush_B(B_til)
            if B_til is not None:
                Cr_inv -= (B_til.T @ B_til).toarray()
                del B_til

        # ── Hi/lo chip coupling prior: rhs pull toward partner's current iterate ─
        for hi_idx, lo_idx in self._chip_pairs:
            C_cp = self._chip_pair_couplings[(hi_idx, lo_idx)]
            hi_cs, lo_cs = hi_idx * nr, lo_idx * nr
            diff = r_current[hi_cs:hi_cs+nr] - r_current[lo_cs:lo_cs+nr]
            rhs[hi_cs:hi_cs+nr] -= C_cp @ diff
            rhs[lo_cs:lo_cs+nr] += C_cp @ diff

        # ── Solve for Δr, then r_hat = r_current + Δr ─────────────────────────
        # Diagonal preconditioning: the (a,b,c,d) columns have scale ~2048 px
        # while (w,z) columns have scale ~1, giving a ~4e6 condition ratio.
        # Scaling Cr_inv by D^{-1} on both sides (D = sqrt(diag)) reduces the
        # effective condition number to ~1 before inversion.
        # Math: D^{-1} Cr_inv D^{-1} @ D delta_r_tilde = D^{-1} rhs
        #  → C_r = D^{-1} inv(Cr_inv_sc) D^{-1};  delta_r = C_r @ rhs
        d_diag     = np.sqrt(np.maximum(np.abs(np.diag(Cr_inv)), 1e-30))
        d_inv      = 1.0 / d_diag
        Cr_inv_sc  = d_inv[:, None] * Cr_inv * d_inv[None, :]
        if need_cov:
            try:
                C_r_sc = np.linalg.inv(Cr_inv_sc)
            except np.linalg.LinAlgError:
                C_r_sc = np.linalg.pinv(Cr_inv_sc)
            C_r     = d_inv[:, None] * C_r_sc * d_inv[None, :]
            delta_r = C_r @ rhs
        else:
            # Intermediate inner passes only need Δr: a Cholesky solve avoids
            # forming the full (6·n_img + n_D)² inverse. The covariance comes
            # from one need_cov pass at the converged point (_inner_converge).
            C_r = None
            rhs_sc = d_inv * rhs
            try:
                c_fac = linalg.cho_factor(Cr_inv_sc, check_finite=False)
                delta_r = d_inv * linalg.cho_solve(c_fac, rhs_sc,
                                                   check_finite=False)
            except (linalg.LinAlgError, ValueError):
                delta_r = d_inv * (np.linalg.pinv(Cr_inv_sc) @ rhs_sc)
        r_hat   = r_current + delta_r

        # for j_idx, img in enumerate(self.image_names):
        #     meta = self.images[img]
        #     cs = j_idx * N_R
        #     ag,bg,cg,dg = r_hat[cs:cs+N_R][:4]
        #     on_skew = (ag-dg)/2
        #     off_skew = (bg+cg)/2
        #     ratio = np.sqrt(ag*dg-bg*cg)
        #     rot = np.arctan2((bg-cg),(ag+dg))/DEG2RAD
        #     print(img)
        #     print(delta_r[cs:cs+N_R])
        #     print(r_hat[cs:cs+N_R][4:])
        #     print(rot,ratio,on_skew,off_skew)
        #     print(rot-meta['orig_rot_deg'])
        #     print()
        # print()

        return r_hat, C_r, a, K_img, C_vT

    def _r_to_dict(self, r_hat):
        nr = self.N_R
        return {img: r_hat[j*nr:(j+1)*nr]
                for j, img in enumerate(self.image_names)}

    def _update_R(self, r_hat):
        nr = self.N_R
        self._r_hat_current = r_hat.copy()
        for j_idx, img in enumerate(self.image_names):
            r_j = r_hat[j_idx * nr:(j_idx + 1) * nr]
            self.R[img] = rotation_matrix_from_abcd(*r_j[:4])

    # ── Public fit interface ───────────────────────────────────────────────────

    def fit(self, n_iter=20, tol=1e-6, clip_sigma=4.5, inflate_hst_errors=False,
            adaptive_delta=1.0, min_align_demote=5,
            inflate_from_iter=3, inflate_alpha_max=3.0, min_outer_iters=None,
            two_phase_align=False,
            mask_tol_frac=1e-3, mask_tol_iters=3,
            hst_fit_sigma_mult=0.5,
            prefilter=True, chi2_threshold=None, alpha_scale_chi2=False,
            use_influence_clip=True,
            influence_k: float = 5.0,
            influence_floor_sr: float | None = None,
            influence_floor_sd: float = 3.0,
            influence_raw_cooks_d=False,
            verbose_tests=False,
            use_two_tier=False, per_iter_callback=None,
            use_soft_weights: bool = False,
            student_t_nu: float = 50.0,
            z_tol: float = 1.0,
            z_init: dict | None = None,
            no_align_prior: bool = False,
            diag_influence_path=None):
        """
        Iterative BP3M fit with outlier rejection.

        Structure
        ---------
        Phase 1 — initial convergence (full sample, no outlier updates):
          Iterate _solve_one_pass until max|Δr| < tol.

        Phase 2 — EM-style outer/inner loops (when clip_sigma is not None):
          Outer loop (up to n_iter iterations):
            1. Update outliers (_update_use_for_fit) based on current solution.
            2. If use_for_fit did not change, stop — solution is fully converged
               for the accepted star set.
            3. Inner loop: iterate _solve_one_pass until max|Δr| < tol with
               the new (frozen) use_for_fit.  This ensures each outlier update
               starts from the exact MAP for the current accepted set, not a
               partially-converged approximation.
          Stop when the outlier set is stable (no use_for_fit changes).

        This guarantees that the returned r_hat is the exact MAP for the final
        accepted star set — no separate frozen-outlier phase needed.

        Parameters
        ----------
        n_iter     : int,   maximum number of outer (outlier-update) iterations
        tol        : float, inner-loop convergence threshold on max|Δr|
        clip_sigma : float or None
            Kept for API compatibility; actual rejection uses chi2 thresholds.
            Set to None to skip outlier rejection entirely.
        inflate_hst_errors : bool, default False
            If True, apply per-image C_hst inflation (alpha adjustment) starting
            at outer iteration inflate_from_iter.  Can cause oscillation; leave
            False unless you have a specific reason to enable it.
        two_phase_align : bool, default False
            Split the EM into two phases: Phase A iterates with alpha OFF
            (outlier rejection via the adaptive population-relative thresholds
            only, which are rank-based and robust to a mis-scaled error model)
            until the normal convergence criteria fire; Phase B then enables
            the per-image alpha model warm-started from the converged
            geometry, clears the test-4 ratchet so every detection is
            re-judged under the inflated error model, and iterates to
            convergence again. Decouples geometry convergence from error-model
            estimation (alpha is measured at a fixed point instead of chasing
            a moving target). No effect unless inflate_hst_errors is True.
        inflate_from_iter : int, default 3
            First outer iteration (0-based) at which alpha inflation updates fire.
            Default 3 gives outlier rejection time to stabilise before alpha is
            adjusted.  Pass 0 when starting from a pre-validated alpha (e.g. the
            v1 BP3M result) so alpha can decrease from the v1 starting value on
            the very first EM iteration.  The update formula is always
            ``min(max(1.0, alpha_prev * alpha_raw), inflate_alpha_max)`` so
            alpha never drops below 1 relative to C_hst_orig and never exceeds
            the cumulative cap, regardless of this setting.
        inflate_alpha_max : float, default 3.0
            Cap on the CUMULATIVE inflation alpha_applied (and on alpha_raw in
            a single step).  Without the cumulative cap the per-step products
            compound without bound (observed 14.65 with a 3.0 cap).  Images at
            the cap are reported as saturated -- alpha_raw still records the
            per-step demanded factor -- and the test-3 gate carve-out treats
            them as legitimately under-inflated.
        min_outer_iters : int or None, default None
            Minimum number of outer EM iterations before early stopping is
            allowed.  None → 4 if inflate_hst_errors else 2.  Set explicitly
            when HST-only sources are enabled mid-run (e.g.
            ``max(hst_enable_iter + 3, 4)``) so the EM has time to converge
            after the new sources are added.
        hst_fit_sigma_mult : float, default 0.5
            Multiplicative factor applied to the per-image residual threshold
            for detections from stars that were NOT initially in alignment
            (e.g. HST-only sources admitted by V2AlignmentCallback).  Must
            be <= 1.  A value of 0.5 means HST-only detections must have
            sigma_resid < 0.5 × thresh_Gaia to stay in use_for_fit.  Stricter
            than Gaia-matched because HST-only stars can have biased PM priors
            (field-star contamination) that would otherwise pull the
            transformation away from the Gaia-constrained solution.
        prefilter : bool, default True
            Before Phase 1, run one solve pass and apply _update_use_for_fit
            (identical logic to Phase 2) to establish a clean initial star set.
            The updated r_hat from this pass is used as the starting point for
            Phase 1.  Pass False to skip (starts Phase 1 with all stars that
            passed use_for_alignment).
        chi2_threshold : float or None, default None
            If given, replaces the adaptive p50+k*(p84-p50) threshold in test 3
            with a fixed chi2 cut (e.g. 9.21 = chi2(2).ppf(0.99)).
            Expulsion threshold is scaled to chi2_threshold*(1+delta/k) to
            preserve hysteresis.  None → use the standard adaptive threshold.
        alpha_scale_chi2 : bool, default False
            If True, divide each star's HST chi2 by alpha² before applying the
            test-3 threshold, starting at outer iteration 3.  Alpha is estimated
            from the previous iteration's accepted star set for that image, so it
            is available without a chicken-and-egg dependency.  This makes the
            threshold image-independent in units of "sigma given image noise"
            rather than raw chi2, preventing over-rejection of images whose
            formal errors are slightly underestimated.
        use_influence_clip : bool, default True
            If True, apply test-4 influence-based clipping after the EM
            converges (tests 1-3).  Flags detections where sigma_resid >
            thresh_sr AND scaled_D > thresh_sd, where both thresholds are
            set adaptively from the eligible-detection distribution using
            p50 + k*(p50-p16), clamped to theoretical null floors.
            Uses ratchet semantics: flagged pairs are permanently excluded.
        influence_k : float, default 5.0
            Adaptive multiplier k for both sigma_resid and scaled_D thresholds.
            thresh = max(p50 + k*(p50-p16), floor).  Matches the k=5.0 used
            by tests 1-3 (_adapt_thresh).
        influence_floor_sr : float or None, default None
            Floor for the sigma_resid threshold.  None uses the theoretical
            chi(2) 99th percentile: sqrt(chi2(2).ppf(0.99)) ≈ 3.03.
        influence_floor_sd : float, default 3.0
            Floor for the scaled_D threshold.  Empirically calibrated from
            clean converged solutions; approximately the 99th percentile of
            the scaled_D null distribution.
        influence_raw_cooks_d : bool, default False
            If True, compare raw Cook's D against the threshold instead of the
            null-normalised scaled_D = D*N_R/leverage.  Raw D is biased against
            sparse images (high leverage → elevated D even for well-fitting stars),
            but can be useful for testing the effect of the normalisation.
        verbose_tests : bool, default False
            If True, print a per-iteration breakdown of flagged detections by
            Gaia solution type (2p vs 5p/6p) and by chip suffix (_hi/_lo)
            for each of tests 1-2, 3, and 4.  Also prints per-test totals for
            in-use star counts so any systematic bias against a star type or
            chip can be diagnosed.

        per_iter_callback : callable or None, default None
            If provided, called as ``per_iter_callback(solver, it_outer)``
            at the end of each Phase-2 outer iteration (after _inner_converge).
            ``it_outer`` is the 1-based outer iteration number.  The callback
            may read ``solver._r_hat_current`` and ``solver._img_data`` and
            modify ``solver._img_data[img]["use_for_fit"]`` /
            ``solver._img_data[img]["use_for_astrom"]`` in-place (e.g. to
            enable HST-only sources in the v2 phased-inclusion scheme).

        Returns
        -------
        r_hat       : (n_r,)
        C_r         : (n_r, n_r)
        v_hat       : (n_stars, 5)  final astrometry posteriors
        C_vT        : (n_stars, 5, 5)  conditional covariance (given r_hat)
        a_arr       : (n_stars, 5)  astrometry at Δr=0 (= r_hat from last iter)
        K_img       : dict  K matrices per image (for sampling)
        z_weights_out : dict or None  soft weights if use_soft_weights=True, else None
        """
        # Store as instance variable so _solve_one_pass and _update_use_for_fit
        # can access it without signature changes.
        self._use_two_tier = use_two_tier

        # When 2p stars are excluded from the alignment, drop any image that
        # would have fewer non-2p alignment stars than half the transformation
        # DOF (i.e., fewer independent 2D constraints than free parameters).
        if self.exclude_2p_from_alignment:
            min_align = max(4, self.N_R // 2)
            dropped_imgs = []
            for img in self.image_names:
                d = self._img_data.get(img)
                if d is None:
                    continue
                sidx = d["sidx"]
                use_fit = d["use_for_fit"]
                n_non2p = int((use_fit & ~self.gaia_2p[sidx]).sum())
                if n_non2p < min_align:
                    dropped_imgs.append((img, n_non2p))
                    # Mark dropped — _img_data[img] stays alive so r_init is
                    # accessible for r_hat assembly; _solve_one_pass skips it.
                    self._img_data[img]["_dropped_by_2p_check"] = True
            if dropped_imgs:
                print(f"\n  [exclude_2p_from_alignment] Dropping {len(dropped_imgs)} "
                      f"image(s) with fewer than {min_align} non-2p alignment stars:")
                for img, n in dropped_imgs:
                    print(f"    WARNING: {img} dropped — only {n} non-2p alignment "
                          f"stars (need ≥{min_align})")

        r_hat = np.concatenate([self._img_data[img]["r_init"]
                                 for img in self.image_names])
        if self.n_ed:
            r_hat = np.concatenate([r_hat, np.zeros(self.n_ed)])
        self._update_R(r_hat)
        nr  = self.N_R
        C_r = None

        # Parameter names for diagnostic output
        _pnames = ['a', 'b', 'c', 'd', 'Δα0', 'Δδ0']
        if nr > 6:
            _pnames += [f'poly{i}' for i in range(nr - 6)]
        _n_imgs = len(self.image_names)

        def _delta_summary(diff):
            """Return formatted strings: (max_location_str, per_param_stats_str)."""
            n_r_only  = nr * self.n_images
            diff_r    = diff[:n_r_only]
            imax      = int(np.argmax(diff))
            if imax >= n_r_only:                    # epoch-distortion coefficient
                g, k = divmod(imax - n_r_only, self.ED_K)
                max_str = (f"{diff[imax]:.3e}"
                           f"  [epoch-D group {g} / coeff {k}]")
            else:
                img_idx   = imax // nr
                param_idx = imax % nr
                max_str   = (f"{diff[imax]:.3e}"
                             f"  [{self.image_names[img_idx]} / {_pnames[param_idx]}]")

            parts = []
            for p in range(nr):
                vals = diff_r[p::nr]
                med  = float(np.median(vals))
                if _n_imgs > 1:
                    w68 = float(np.percentile(vals, 84) - np.percentile(vals, 16))
                    parts.append(f"{_pnames[p]}: {med:.2e} [{w68:.2e}]")
                else:
                    parts.append(f"{_pnames[p]}: {med:.2e}")
            if len(diff) > n_r_only:
                parts.append(f"epoch-D: {float(np.median(diff[n_r_only:])):.2e} "
                             f"[max {float(np.max(diff[n_r_only:])):.2e}]")
            return max_str, '  '.join(parts)

        # ── Starving-chip demotion (min_align_demote > 0) ────────────────
        # An image whose alignment count collapses below the threshold is
        # demoted to astrometry-only: its detections keep contributing to the
        # star posteriors, but its r is FROZEN at the last converged value
        # (2 stars + a free pointing can swing ~50 mas per iteration and
        # globalise the test-3 limit cycle; a frozen r also removes the
        # ill-conditioned block that made the inner loop crawl).  Promotion
        # back requires min_align_demote + 3 admitted stars (hysteresis).
        self._align_demoted: set = set()
        _img_block = {img: slice(j * self.N_R, (j + 1) * self.N_R)
                      for j, img in enumerate(self.image_names)}

        def _inner_converge(r_hat, label, z_weights=None):
            """Iterate _solve_one_pass until max|Δr| < tol. Returns updated r_hat etc.

            Intermediate passes solve for Δr only (need_cov=False: Cholesky
            solve, no explicit inverse); one covariance-bearing pass runs at
            the converged point so the returned C_r matches r_hat.
            """
            def _final(r_hat):
                r_new, C_r_i, a_i, K_i, CvT_i = self._solve_one_pass(
                    r_hat, z_weights=z_weights)
                for _dimg in self._align_demoted:
                    _bl = _img_block[_dimg]
                    r_new[_bl] = r_hat[_bl]
                self._update_R(r_new)
                self._update_geometry(r_new, a_i)
                return r_new, C_r_i, a_i, K_i, CvT_i

            for it_i in range(500):
                r_new, _, a_i, K_i, CvT_i = self._solve_one_pass(
                    r_hat, z_weights=z_weights, need_cov=False)
                for _dimg in self._align_demoted:
                    _bl = _img_block[_dimg]
                    r_new[_bl] = r_hat[_bl]     # frozen: no update, no Δr
                diff = np.abs(r_new - r_hat)
                delta = np.max(diff)
                r_hat = r_new
                self._update_R(r_hat)
                self._update_geometry(r_hat, a_i)
                if it_i % 10 == 0:
                    max_str, stats_str = _delta_summary(diff)
                    print(f"  {label}: step {it_i+1:3d},  max|Δr| = {max_str}")
                    print(f"    params: {stats_str}")
                if delta < tol:
                    max_str, stats_str = _delta_summary(diff)
                    print(f"  {label}: converged in {it_i+1} inner steps "
                          f"(max|Δr| = {max_str})")
                    print(f"    params: {stats_str}")
                    return _final(r_hat)
            max_str, stats_str = _delta_summary(diff)
            print(f"  {label}: WARNING — did not converge (max|Δr| = {max_str})")
            print(f"    params: {stats_str}")
            return _final(r_hat)

        # ── Disable alignment prior if requested ─────────────────────────────
        if no_align_prior:
            n_r = self.N_R
            for img in self.image_names:
                self._img_data[img]["C_r_prior_inv"] = np.zeros((n_r, n_r))
            print(" no_align_prior=True: alignment priors zeroed out")

        # ── Phase 0: pre-filter using one solve + same outlier rejection as Phase 2
        if prefilter and clip_sigma is not None:
            print(' Phase 0: pre-filter (one solve pass + outlier rejection)')
            r_hat, C_r, a_arr, K_img, C_vT = self._solve_one_pass(r_hat)
            self._update_R(r_hat)
            self._update_geometry(r_hat, a_arr)
            clip_info, _, _ = self._update_use_for_fit(
                r_hat, a_arr, C_r, C_vT, clip_sigma,
                ok_star_prev=None, inflate_errors=False,
                skip_star_tests=True,
                chi2_threshold=chi2_threshold,
                alpha_scale_chi2=False)   # no alpha scaling in pre-filter
            n_in  = sum(n_use for _, n_use, _, _, _, _ in clip_info)
            n_tot = sum(n_t   for _, _,    n_t, _, _, _ in clip_info)
            print(f"  Pre-filter: {n_in}/{n_tot} stars accepted across all images\n")

        # ── Phase 1: initial convergence with filtered sample ─────────────────
        if n_iter == 0:
            print(' Phase 1: skipped (n_iter=0, transformation held fixed at r_init)')
            _, C_r, a_arr, K_img, C_vT = self._solve_one_pass(r_hat)
            self._update_geometry(r_hat, a_arr)
        else:
            print(' Phase 1: convergence with pre-filtered sample')
            r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(r_hat, 'init')

        # ── Phase 2: EM-style outlier rejection ───────────────────────────────
        # Tests 1-3 (chi² / sigma) and test-4 (Cook's D influence) run
        # together in the same outer loop.  Test-4 uses a ratchet: newly
        # flagged detections are added to _img_data[img]["influence_excl"],
        # which _update_use_for_fit treats as a permanent ceiling so they
        # can never be re-admitted by tests 1-3.  The ratchet guarantees
        # monotonic convergence: n_inf_new decreases toward 0.
        _default_min = 4 if inflate_hst_errors else 2
        min_outer = int(min_outer_iters) if min_outer_iters is not None else _default_min

        z_weights_out = None   # set below if use_soft_weights=True

        if use_soft_weights and clip_sigma is not None:
            # TODO(IRLS): this branch never runs the star-level tests, so
            # ok_star stays all-True and sigma_from_gaia_prior stays zero —
            # prior fallback then acts on the logdet checks alone and the
            # diagnostic column is meaningless.  Also, the docstring's claim
            # that z-mode replaces the hard masks with use_for_fit_max is NOT
            # implemented: use_align stays use_for_fit, so soft weighting can
            # down-weight but never re-admit.  Fix both when IRLS is next used.
            print(f'\n Phase 2 (soft-weight IRLS): Student-t downweighting  (ν={student_t_nu})')

            # Seed PM estimates for all stars (including HST-only) before IRLS
            # starts.  Calling the callback at hst_enable_iter triggers the PM
            # seeding step that sets v_survey for HST-only sources from the
            # xmatch catalogue.  Without this, HST-only PMs stay at the diffuse
            # prior (0 mas/yr), giving huge residuals → z≈0 → PMs never improve.
            if per_iter_callback is not None:
                _seed_iter = getattr(per_iter_callback, 'hst_enable_iter', None) or 0
                per_iter_callback(self, _seed_iter)
                # Re-solve once so a_arr reflects the seeded PM estimates before
                # we compute the first set of IRLS weights.
                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, 'soft-w seed', z_weights=None)

            # Initialise weights.  If Phase-6 chi2 values were pre-computed
            # during catalogue building and passed in as z_init, use them
            # directly — they provide a better warm start than the seed-solve
            # residuals for detections the catalogue already flagged as poor.
            # Fall back to computing from the seed-solve residuals otherwise.
            if z_init is not None:
                # Phase-6 chi2 warm start.  Any images missing from z_init fall
                # back to seed-solve residuals (single shared call for efficiency).
                if any(z_init.get(img) is None for img in self.image_names):
                    _z_fb, _, _ = self._update_soft_weights(r_hat, a_arr, student_t_nu)
                    z_weights = {img: (z_init[img] if z_init.get(img) is not None
                                       else _z_fb.get(img))
                                 for img in self.image_names}
                else:
                    z_weights = z_init
            else:
                z_weights, _, _ = self._update_soft_weights(r_hat, a_arr, student_t_nu)

            _n_consec_z_stable = 0
            for it_outer in range(n_iter):
                z_new, n_det, n_eff = self._update_soft_weights(r_hat, a_arr, student_t_nu)

                delta_z = sum(float(np.abs(z_new[img] - z_weights[img]).sum())
                              for img in z_new if z_new[img] is not None)
                z_weights = z_new

                if delta_z < z_tol:
                    _n_consec_z_stable += 1
                else:
                    _n_consec_z_stable = 0

                print(f"\n  Soft-weight iter {it_outer+1}: "
                      f"N_eff={n_eff:.1f}/{n_det} ({100*n_eff/max(n_det,1):.1f}%),  "
                      f"Δz={delta_z:.3f}")

                if _n_consec_z_stable >= 2 and it_outer >= min_outer:
                    print(f"  Weights converged (Δz < {z_tol} for 2 consecutive iters) — stopping.")
                    break

                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, f'soft-w {it_outer+1}', z_weights=z_weights)
            else:
                print(f"  Stopped after {n_iter} IRLS iterations (weights did not converge)")

            z_weights_out = z_weights

        elif clip_sigma is not None:
            print('\n Phase 2: EM-style outlier rejection')

            ok_star_prev = np.ones(self.n_stars, dtype=bool)
            _n_tol_stable = 0
            _n_det_tot = sum(d["n"] for d in self._img_data.values() if d)
            _mask_tol = int(max(1, round(mask_tol_frac * _n_det_tot)))
            _n_consec_stable = 0  # consecutive iters with 0 tests-1/2/3 changes
            # Two-phase alignment state (see docstring). Phase A: alpha off.
            _alpha_phase_active = not (two_phase_align and inflate_hst_errors)
            _eff_from_iter = inflate_from_iter
            _phase_b_start = None
            # Each phase gets the FULL n_iter budget (per CLI --n_bp3m_iter).
            _n_iter_loop = (2 * n_iter) if not _alpha_phase_active else n_iter
            if not _alpha_phase_active:
                print("  two_phase_align: Phase A (α=1) — alpha model deferred "
                      "until geometry converges "
                      f"(up to {n_iter} iters per phase)")

            for it_outer in range(_n_iter_loop):
                if (_phase_b_start is not None
                        and it_outer - _phase_b_start >= n_iter):
                    print(f"  two_phase_align: Phase B budget ({n_iter} iters) "
                          f"exhausted (star set did not fully stabilise)")
                    break
                # Snapshot use_for_fit per image before _update_use_for_fit so
                # we can attribute per-detection changes to test-3 separately.
                if verbose_tests:
                    _snap_use_pre = {
                        img: d["use_for_fit"].copy()
                        for img, d in self._img_data.items() if d is not None}

                clip_info, ok_star_new, n_use_changed = self._update_use_for_fit(
                    r_hat, a_arr, C_r, C_vT, clip_sigma, iteration=it_outer,
                    adaptive_delta=adaptive_delta,
                    ok_star_prev=ok_star_prev,
                    inflate_errors=inflate_hst_errors and _alpha_phase_active,
                    inflate_from_iter=_eff_from_iter,
                    inflate_alpha_max=inflate_alpha_max,
                    hst_fit_sigma_mult=hst_fit_sigma_mult,
                    chi2_threshold=chi2_threshold, alpha_scale_chi2=alpha_scale_chi2)

                # ── demote/promote starving chips (see note above) ────────
                # Decisions only; the masks themselves are managed inside
                # _update_use_for_fit (astrometry follows admissions, the
                # alignment mask empties while demoted, churn stays clean).
                if min_align_demote > 0:
                    for _img, _d in self._img_data.items():
                        if _d is None:
                            continue
                        _n_fit = int(_d.get("n_fit_wouldbe",
                                     np.asarray(_d["use_for_fit"], bool).sum()))
                        if _img in self._align_demoted:
                            if _n_fit >= min_align_demote + 3:
                                self._align_demoted.discard(_img)
                                print(f"    {_img}: re-promoted to alignment "
                                      f"({_n_fit} admitted stars)")
                        elif _n_fit < min_align_demote:
                            self._align_demoted.add(_img)
                            print(f"    {_img}: only {_n_fit} alignment stars "
                                  f"— demoted to astrometry-only (r frozen)")
                            _d["use_for_fit"] = np.zeros_like(
                                np.asarray(_d["use_for_fit"], bool))

                n_global_changed = int(np.sum(ok_star_prev != ok_star_new))
                n_total_changed  = n_global_changed + n_use_changed

                # Track consecutive stability of tests 1-3 (before test-4).
                if n_global_changed == 0 and n_use_changed == 0:
                    _n_consec_stable += 1
                else:
                    _n_consec_stable = 0

                # Test-4: influence clipping.
                # Only runs after min_outer iters so chi² tests remove gross
                # outliers first.  Also suppressed once the EM has been stable
                # for ≥2 consecutive iterations: firing Cook's D on a fully
                # converged solution can perturb sparse fields (few Gaia stars)
                # and trigger a cascade.
                n_inf_new = 0
                _t4_flagged_info = []
                _t4_thresh_sr = _t4_thresh_sd = float('nan')
                if use_influence_clip and it_outer >= min_outer and _n_consec_stable < 2:
                    _diag_rows = [] if diag_influence_path is not None else None
                    n_inf_new, _t4_flagged_info, _t4_thresh_sr, _t4_thresh_sd = \
                        self._apply_influence_clip(
                            r_hat, C_r, a_arr,
                            k_sigma_resid=influence_k,
                            k_scaled_d=influence_k,
                            floor_sigma_resid=influence_floor_sr,
                            floor_scaled_d=influence_floor_sd,
                            raw_cooks_d=influence_raw_cooks_d,
                            diag_rows=_diag_rows,
                            it_outer=it_outer + 1)
                    if diag_influence_path is not None and _diag_rows:
                        import csv, os
                        _write_header = not os.path.exists(diag_influence_path)
                        with open(diag_influence_path, 'a', newline='') as _f:
                            _w = csv.DictWriter(_f, fieldnames=list(_diag_rows[0].keys()))
                            if _write_header:
                                _w.writeheader()
                            _w.writerows(_diag_rows)
                    n_total_changed += n_inf_new

                _t4_str = ""
                if use_influence_clip:
                    _t4_str = f", {n_inf_new} test-4 changes"
                    if it_outer >= min_outer and _n_consec_stable < 2:
                        _t4_str += (f"  [thresh_sr={_t4_thresh_sr:.2f}"
                                    f"  thresh_sd={_t4_thresh_sd:.2f}]")
                print(f"\n  Outer iter {it_outer+1}: "
                      f"{n_global_changed} test-1/2 changes, "
                      f"{n_use_changed} test-3 changes"
                      + _t4_str
                      + f"  ({n_total_changed} total)")
                _alpha_on_now = (inflate_hst_errors and _alpha_phase_active
                                 and it_outer >= _eff_from_iter)
                for img, n_use, n_tot, alpha_applied, alpha_raw, n_astrom_only in clip_info:
                    tags = []
                    if _alpha_on_now:
                        tags.append("α-inflated")
                    if alpha_scale_chi2 and it_outer >= 3:
                        tags.append("α-scaled-chi2")
                    tag_str = f"  [{', '.join(tags)}]" if tags else ""
                    if _alpha_on_now:
                        alpha_str = (f"α_applied={alpha_applied:.3f}  "
                                     f"α_raw={alpha_raw:.3f}")
                    else:
                        alpha_str = f"α={alpha_applied:.3f}"
                    astrom_str = f" (+{n_astrom_only} astrom-only)" if n_astrom_only > 0 else ""
                    print(f"    {img}: {n_use}/{n_tot} align{astrom_str},  {alpha_str}{tag_str}")

                if verbose_tests:
                    self._print_test_type_breakdown(
                        ok_star_prev, ok_star_new,
                        _snap_use_pre, _t4_flagged_info)

                ok_star_prev = ok_star_new.copy()

                # Convergence: tests 1-2 and 4 must be EXACTLY stable; test 3
                # is allowed a small tolerance.  With N detections and a
                # threshold recomputed from the accepted set (and from alpha,
                # which itself depends on that set), a few borderline detections
                # flip forever.  Measured on Leo_I: tests 1-2 settle by iteration
                # 4 and test 4 is always 0, but test 3 flickers 4-32 of ~11000
                # detections for all 50 iterations.  Each flip moves
                # Δα0/Δδ0 by ~residual/N_stars ~ 10 μas -- the tangent point is
                # effectively the per-image residual mean, so it has 1/N leverage
                # that a/b/c/d do not (those move ~1e-8, i.e. 1e5x less).  That
                # shifts every residual and re-creates the flicker: a
                # self-sustaining limit cycle, not a transient.  The jump
                # plateaus (mean of last 10 / previous 10 = 1.02 over 50 iters)
                # and tracks the flip count (Spearman +0.65).  Requiring exactly
                # zero therefore never terminates on a large field.
                _converged_now = False
                if not _alpha_phase_active:
                    # Phase A convergence: without alpha the borderline
                    # test-1/2 chi2 flicker never reaches exactly zero (Leo_I
                    # showed a period-20 limit cycle of 1-2 stars for 50
                    # iterations), so Phase A uses a RELAXED criterion — total
                    # mask churn across tests 1-4 within the test-3 tolerance
                    # for mask_tol_iters consecutive iterations — plus a hard
                    # budget cap so Phase B is always reached. Phase B (below)
                    # keeps the strict criteria and the freeze-mask final
                    # solve, so the reported solution is unaffected.
                    _tot_changed = n_global_changed + n_use_changed + n_inf_new
                    if it_outer >= min_outer and _tot_changed <= _mask_tol:
                        _n_tol_stable += 1
                        if _n_tol_stable >= mask_tol_iters:
                            _converged_now = True
                    else:
                        _n_tol_stable = 0
                    if not _converged_now and it_outer + 1 >= n_iter:
                        print(f"  two_phase_align: Phase A budget "
                              f"({n_iter} iters) reached.")
                        _converged_now = True
                    if _converged_now:
                        print(f"\n  two_phase_align: Phase A (α=1) done at "
                              f"outer iter {it_outer+1} — starting Phase B "
                              f"(per-image α enabled).")
                        _alpha_phase_active = True
                        _eff_from_iter = it_outer + 1
                        _phase_b_start = it_outer + 1
                        _n_tol_stable = 0
                        _n_consec_stable = 0
                        # Re-judge every detection under the inflated model.
                        for _d in self._img_data.values():
                            if _d is not None and "influence_excl" in _d:
                                _d["influence_excl"][:] = False
                else:
                    _stable_124 = (n_global_changed == 0 and n_inf_new == 0)
                    if _stable_124 and n_use_changed == 0 and it_outer >= min_outer:
                        print(f"  Tests 1-4 stable — stopping.")
                        break
                    if _stable_124 and it_outer >= min_outer and n_use_changed <= _mask_tol:
                        _n_tol_stable += 1
                        if _n_tol_stable >= mask_tol_iters:
                            print(f"  Tests 1-2/4 stable and test-3 flicker "
                                  f"{n_use_changed} <= {_mask_tol} for "
                                  f"{mask_tol_iters} iters — stopping.")
                            break
                    else:
                        _n_tol_stable = 0

                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, f'outer {it_outer+1}')

                if per_iter_callback is not None:
                    per_iter_callback(self, it_outer + 1)
            else:
                print(f"  Stopped after {_n_iter_loop} outer iterations "
                      f"(star set did not fully stabilise)")

            # Freeze the detection mask and solve once more.  Nothing below
            # this point calls _update_use_for_fit, so this is a small
            # well-posed problem and its result is a true fixed point of the
            # frozen mask, rather than wherever the test-3 limit cycle happened
            # to be when the loop stopped.
            if n_iter > 0:
                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, 'final (mask frozen)')

            # The loop tests a_arr at the top and re-solves at the bottom, so
            # the a_arr reported below was never itself tested.  Re-run tests
            # 1-2 on it so ok_star / sigma_from_gaia_prior / prior_fallback
            # describe the solution actually being written out.
            ok_star_prev = self._final_star_tests(
                a_arr, C_vT, ok_star_prev=ok_star_prev)

        # Final v_hat = a_arr (Δr = 0 at the last converged r_hat)
        v_hat = a_arr.copy()

        # Store final ok_star so _save_results/_make_plots can apply prior fallback
        # to stars that failed the Gaia-prior chi2 test (Gaia-incompatible).
        self.ok_star = ok_star_prev.copy()

        return r_hat, C_r, v_hat, C_vT, a_arr, K_img, z_weights_out

    def _update_soft_weights(self, r_hat, a_arr, nu=5.0):
        """
        Compute per-detection Student-t IRLS weights from current residuals.

        For each detection k in image j:
            chi2_k = res_k^T Cs_inv_k res_k
            z_k    = min(1, (nu+2) / (nu+chi2_k))
        Phase-0 hard rejections (use_for_fit_max=False) receive z=0.

        Returns
        -------
        z_dict      : {img: (n,) float}
        n_det_total : int   total number of Phase-0-surviving detections
        n_eff_total : float sum of all z values (effective sample size)
        """
        nr = self.N_R
        z_dict = {}
        n_det_total = 0
        n_eff_total = 0.0

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                z_dict[img] = None
                continue

            cs   = j_idx * nr
            r_j  = r_hat[cs:cs + nr]
            sidx = d["sidx"]
            n    = d["n"]

            JU   = d["JU"]      # (n, 2, 5)
            X    = d["X_mat"]   # (n, 2, N_R)
            xys  = d["xys"]     # (n, 2)

            Cs     = self._compute_Cs(img, r_j)
            Cs_inv = np.linalg.inv(Cs)   # (n, 2, 2)

            # Model: xys = X r - JU a + noise  (JU carries the sign; see
            # build_X_matrix / sample_posteriors for the same convention).
            # Residual = xys - (X r - JU a) = xys - X r + JU a.
            pred  = (np.einsum('nij,j->ni',  X,  r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))   # (n, 2)
            _edd = self._ed_disp(img, r_hat)
            if not np.isscalar(_edd):
                pred = pred + _edd
            res   = xys - pred                                        # (n, 2)

            chi2  = np.einsum('ni,nij,nj->n', res, Cs_inv, res)     # (n,)
            z     = np.minimum(1.0, (nu + 2.0) / (nu + chi2))

            # Hard floor: Phase-0-rejected detections always get z=0.
            # All surviving detections (Gaia-matched and HST-only alike) get
            # soft weights from their residuals.  HST-only PMs are seeded from
            # the xmatch catalogue before the first weight computation, so their
            # residuals start small for good detections.
            # Same population as the hard-EM astrometry tier: post-Phase-0 Gaia
            # (use_for_fit) plus callback-enabled HST-only (use_for_astrom).
            # Excludes Phase-0-rejected detections which would otherwise get
            # z > 0 and shift the transformation relative to the hard-EM solution.
            _astrom_mask = d["use_for_fit"] | d.get("use_for_astrom", d["use_for_fit"])
            mask  = _astrom_mask.astype(float)
            z    *= mask

            z_dict[img] = z
            n_det_total += int(mask.sum())
            n_eff_total += float(z.sum())

        return z_dict, n_det_total, n_eff_total

    def _final_star_tests(self, v_hat, C_vT, ok_star_prev=None,
                          adaptive_k=5.0, adaptive_delta=1.0, chi2_pval=0.95,
                          verbose=True):
        """
        Re-run tests 1 and 2 on the FINAL a_arr, after the EM loop has ended.

        Why this exists
        ---------------
        The EM loop tests a_arr at the TOP of each iteration and re-solves at the
        BOTTOM — and since the frozen-mask finalisation ALWAYS re-solves after
        the loop (break path included), the a_arr being reported is newer than
        anything the in-loop tests saw on every path.  These final tests are
        therefore required unconditionally, not only on the exhaustion path.

        If the EM has converged, one extra solve moves nothing and the distinction
        is academic.  If it is still OSCILLATING, the tested and returned states
        can be opposite half-cycles, and then ok_star / sigma_from_gaia_prior /
        prior_fallback describe a solution that is not the one written out:
        Gaia-incompatible stars pass with chi2_gaia ~ 0 while their reported PM is
        hundreds of sigma from the prior, and no detection is ever dropped because
        the drop decision was taken against the other half-cycle.

        Only tests 1 and 2 are re-run.  They are pure functions of (v_hat, C_vT,
        priors) and feed nothing back into the fit.  Test 3 and the alpha
        inflation are deliberately NOT re-run: they mutate use_for_fit and C_hst,
        which would put the masks out of step with the solve that produced v_hat.

        NOTE: the chi2/threshold logic below is duplicated from tests 1-2 inside
        _update_use_for_fit rather than shared, to keep this fix from touching
        that hot path.  If the thresholds there change, change them here too.

        Returns ok_star for the reported solution; also refreshes
        self.sigma_from_gaia_prior.
        """
        from scipy.stats import chi2 as chi2_dist

        observed = self.gaia_n_hst_used > 0
        floor_5  = float(chi2_dist.ppf(0.99, df=5))
        floor_2  = float(chi2_dist.ppf(0.99, df=2))

        def _adapt_thresh(values, k, fallback, floor=0.0):
            if len(values) < 10:
                return float(max(fallback, floor))
            p16 = float(np.percentile(values, 16))
            p50 = float(np.median(values))
            return float(max(p50 + k * max(p50 - p16, 1e-6), floor))

        # ── Test 1: combined-prior chi2 ──────────────────────────────────────
        v_prior    = getattr(self, 'v_prior', self.v_survey)
        C_prior    = getattr(self, 'C_prior', self.C_survey)
        delta_gaia = v_hat - v_prior
        chi2_gaia  = np.einsum('ni,nij,nj->n', delta_gaia,
                               np.linalg.inv(C_vT + C_prior), delta_gaia)

        obs_5p = observed & ~self.gaia_2p
        obs_2p = observed & self.gaia_2p
        th5 = _adapt_thresh(chi2_gaia[obs_5p], adaptive_k,
                            chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
        th2 = _adapt_thresh(chi2_gaia[obs_2p], adaptive_k,
                            chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
        # Same gate as in _update_use_for_fit, so the final pass can never be
        # more permissive than the in-loop tests were.
        _p50_5 = (float(np.median(chi2_gaia[obs_5p]))
                  if int(obs_5p.sum()) >= 10 else float('nan'))
        _p50_2 = (float(np.median(chi2_gaia[obs_2p]))
                  if int(obs_2p.sum()) >= 10 else float('nan'))
        th5, _gt5, _ = _gate_thresh(th5, _p50_5, 5, floor_5)
        th2, _gt2, _ = _gate_thresh(th2, _p50_2, 2, floor_2)
        ok_gaia_admit = np.where(self.gaia_2p, chi2_gaia < th2, chi2_gaia < th5)

        if ok_star_prev is not None and adaptive_delta > 0:
            th5_o = _adapt_thresh(chi2_gaia[obs_5p], adaptive_k + adaptive_delta,
                                  chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
            th2_o = _adapt_thresh(chi2_gaia[obs_2p], adaptive_k + adaptive_delta,
                                  chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
            ok_gaia_retain = np.where(self.gaia_2p,
                                      chi2_gaia < th2_o, chi2_gaia < th5_o)
            ok_gaia = np.where(ok_star_prev, ok_gaia_retain, ok_gaia_admit)
        else:
            ok_gaia = ok_gaia_admit

        # ── Test 2: diffuse prior (fixed threshold) ──────────────────────────
        chi2_diff   = np.sum((v_hat / self._sigma_diff_per_star)**2, axis=1)
        thresh_diff = float(chi2_dist.ppf(chi2_dist.cdf(4.0, df=1), df=5))
        ok_diffuse  = chi2_diff < thresh_diff

        ok_star = ok_gaia & ok_diffuse
        self.sigma_from_gaia_prior[:] = np.sqrt(np.maximum(chi2_gaia, 0.))

        if verbose:
            n_obs = int(observed.sum())
            n_fg  = int((~ok_gaia & observed).sum())
            n_fd  = int((~ok_diffuse & ok_gaia & observed).sum())
            mx    = float(np.nanmax(chi2_gaia[observed])) if n_obs else float('nan')
            print(f"  Final tests 1-2 on the REPORTED solution "
                  f"(of {n_obs} observed): {n_fg} Gaia-incompatible, "
                  f"{n_fd} diffuse  [thresh 5p={th5:.2f}, max chi2={mx:.2f}]")
            if ok_star_prev is not None:
                n_new = int((ok_star_prev & ~ok_star & observed).sum())
                if n_new:
                    print(f"    {n_new} star(s) newly rejected — the returned a_arr "
                          f"had not been tested (EM did not converge)")
        return ok_star

    def _update_use_for_fit(self, r_hat, v_hat, C_r, C_vT, clip_sigma,
                            chi2_pval=0.95, iteration=0,
                            adaptive_k=5.0, adaptive_delta=1.0,
                            sigma_pm_diffuse=100.0, sigma_plx_diffuse=20.0,
                            ok_star_prev=None, inflate_errors=False,
                            inflate_from_iter=3, inflate_alpha_max=3.0,
                            hst_fit_sigma_mult=0.5,
                            skip_star_tests=False,
                            chi2_threshold=None, alpha_scale_chi2=False,
                            thresh_gate_mult=THRESH_GATE_MULT_DEFAULT,
                            thresh_ceiling_mult=THRESH_CEILING_MULT_DEFAULT):
        """
        Update use_for_fit via two star-level chi2 tests plus per-image
        residual clipping.

        All chi2 thresholds use a data-driven adaptive form p50 + k*(p84-p50)
        computed from the empirical chi2 distribution of currently-observed stars.
        This is more robust than a fixed chi2.ppf threshold.

        Hysteresis (adaptive_delta > 0): currently-included stars require chi2 >
        p50 + (k+delta)*(p84-p50) to be expelled; currently-excluded stars must
        clear p50 + k*(p84-p50) to be re-admitted.  This dead-band prevents
        borderline stars from oscillating and stabilises EM convergence.

        Star-level tests (applied globally):

          1. Gaia prior:
               chi2_gaia = (ã_i - v_s_i)^T (C_vT_i + C_survey_i)^{-1} (ã_i - v_s_i)
             Adaptive threshold computed separately for df=5 and df=2 populations.
             Hysteresis applied when ok_star_prev is provided.

          2. Diffuse prior — catches stars with physically extreme astrometry:
               chi2_diff = sum((v_hat / sigma_diffuse)^2)
             Fixed threshold chi2.ppf(0.9999, df=5) ≈ 21.7 (data-independent).

        Per-image test (stars may be excluded from one image but kept in others):
          3. Position residuals: sigma_resid^2 < adaptive threshold from
             globally-accepted stars in this image.

        Returns (info, ok_star) where:
          info    : list of (img_name, n_used, n_total, alpha) for logging
          ok_star : (n_stars,) bool — stars passing tests 1 and 2
        """
        from scipy.stats import chi2 as chi2_dist

        observed = self.gaia_n_hst_used > 0   # stars used in previous iteration

        # The gate's premise is that alpha has ALREADY divided the error-model
        # scale out of the residuals, so any remaining excess is model error.
        # That premise does not hold everywhere:
        #   * Phase 0 (skip_star_tests) runs off ONE un-converged solve with no
        #     alpha at all, so a huge p50 is entirely legitimate.  Gating there
        #     collapsed the pre-filter to 53/15261 detections on Leo_I.
        #   * Before inflate_from_iter, or with inflate_errors=False, alpha has
        #     never been applied, so excess may well be pure error mis-scaling.
        # Outside those regimes the adaptive threshold must stay free.
        _gate_on = (not skip_star_tests) and bool(inflate_errors) \
                   and int(iteration) >= int(inflate_from_iter)
        _gm = thresh_gate_mult    if _gate_on else None
        _cm = thresh_ceiling_mult if _gate_on else None

        # Theoretical chi2 floors: adaptive thresholds may not drop below the
        # q=0.99 expected value for the relevant distribution.  This prevents
        # runaway exclusion when the empirical chi2 distribution narrows
        # (e.g. in single-epoch runs where few stars constrain the image).
        floor_5 = float(chi2_dist.ppf(0.99, df=5))  # ≈ 15.1
        floor_2 = float(chi2_dist.ppf(0.99, df=2))  # ≈ 9.2

        def _adapt_thresh(values, k, fallback, floor=0.0):
            """p50 + k*(p50-p16), floored at `floor`; fallback when few points.
            Returns (threshold, p16, p50, p84)."""
            if len(values) < 10:
                return float(max(fallback, floor)), float('nan'), float('nan'), float('nan')
            p16 = float(np.percentile(values, 16))
            p50 = float(np.median(values))
            p84 = float(np.percentile(values, 84))
            return float(max(p50 + k * max(p50 - p16, 1e-6), floor)), p16, p50, p84

        # ── 1. Combined prior chi2 test ───────────────────────────────────────
        # chi2 = (ã - v_prior)^T (C_vT + C_prior)^{-1} (ã - v_prior)
        # v_prior / C_prior are the combined Gaia+DELVE prior for DELVE-matched
        # stars, and the Gaia-only prior for unmatched stars.  For DELVE-only
        # stars (gaia_2p), v_prior uses the DELVE PM as the reference.
        v_prior    = getattr(self, 'v_prior', self.v_survey)
        C_prior    = getattr(self, 'C_prior', self.C_survey)
        delta_gaia = v_hat - v_prior                   # (n_stars, 5)
        C_comb     = C_vT + C_prior                    # (n_stars, 5, 5)
        C_comb_inv = np.linalg.inv(C_comb)
        chi2_gaia  = np.einsum('ni,nij,nj->n', delta_gaia, C_comb_inv, delta_gaia)

        # Adaptive thresholds: separate df=5 (5/6-param) and df=2 (2-param).
        # 2p Gaia solutions have near-infinite C_survey for pm/plx, so chi2_gaia
        # is effectively df=2 (only position components constrained by Gaia).
        obs_5p = observed & ~self.gaia_2p
        obs_2p = observed & self.gaia_2p
        thresh_gaia_5, p16_5, p50_5, p84_5 = _adapt_thresh(
            chi2_gaia[obs_5p], adaptive_k, chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
        thresh_gaia_2, p16_2, p50_2, p84_2 = _adapt_thresh(
            chi2_gaia[obs_2p], adaptive_k, chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
        thresh_gaia_5, _gated_5, _capped_5 = _gate_thresh(
            thresh_gaia_5, p50_5, 5, floor_5, _gm, _cm)
        thresh_gaia_2, _gated_2, _capped_2 = _gate_thresh(
            thresh_gaia_2, p50_2, 2, floor_2, _gm, _cm)
        if (_gated_5 or _gated_2) and not skip_star_tests:
            print(f"    WARNING population inconsistent with its own errors "
                  f"(p50={p50_5:.2f} > {thresh_gate_mult}x theory) — adaptive "
                  f"threshold capped at ceiling "
                  f"({thresh_ceiling_mult}x floor = {thresh_ceiling_mult*floor_5:.2f})")
        # Admission: a star must clear thresh_gaia to be (re-)included.
        ok_gaia_admit = np.where(self.gaia_2p,
                                 chi2_gaia < thresh_gaia_2,
                                 chi2_gaia < thresh_gaia_5)

        # Hysteresis: currently-included stars use a higher (looser) expulsion
        # threshold thresh_out = p50 + (k+delta)*(p84-p50).  The dead-band
        # between admission and expulsion prevents borderline stars from
        # oscillating as the adaptive threshold shifts slightly between iterations.
        if ok_star_prev is not None and adaptive_delta > 0:
            thresh_out_5, _, _, _ = _adapt_thresh(chi2_gaia[obs_5p],
                                         adaptive_k + adaptive_delta,
                                         chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
            thresh_out_2, _, _, _ = _adapt_thresh(chi2_gaia[obs_2p],
                                         adaptive_k + adaptive_delta,
                                         chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
            # Gate the expulsion thresholds with the SAME p50, so the hysteresis
            # dead-band collapses to the floor together with admission rather
            # than leaving a one-sided ratchet.
            thresh_out_5, _, _ = _gate_thresh(thresh_out_5, p50_5, 5, floor_5, _gm, _cm)
            thresh_out_2, _, _ = _gate_thresh(thresh_out_2, p50_2, 2, floor_2, _gm, _cm)
            ok_gaia_retain = np.where(self.gaia_2p,
                                      chi2_gaia < thresh_out_2,
                                      chi2_gaia < thresh_out_5)
            # Currently in: keep unless chi2 > thresh_out.
            # Currently out: admit only if chi2 < thresh_gaia.
            ok_gaia = np.where(ok_star_prev, ok_gaia_retain, ok_gaia_admit)
        else:
            thresh_out_5 = thresh_gaia_5   # symmetric (no hysteresis)
            ok_gaia = ok_gaia_admit

        # ── 2. Diffuse prior test (fixed, data-independent) ──────────────────
        # Excludes stars with physically extreme astrometry regardless of what
        # the rest of the sample is doing.  Uses per-star sigma_diff so that
        # Gaia 5p/6p stars (which have no diffuse prior) are never expelled here —
        # their Gaia chi2 test (test 1) is the sole outlier criterion.
        chi2_diff  = np.sum((v_hat / self._sigma_diff_per_star)**2, axis=1)  # (n_stars,)
        # 2-sigma equivalent: quantile of chi2(5) matching 2σ in 1D (Φ(2)≈0.9545)
        thresh_diff = float(chi2_dist.ppf(chi2_dist.cdf(4.0, df=1), df=5))  # ≈ 11.1
        ok_diffuse  = chi2_diff < thresh_diff

        ok_star = ok_gaia & ok_diffuse

        # Store chi2_gaia for diagnostics
        self.sigma_from_gaia_prior[:] = np.sqrt(chi2_gaia)

        # When called from Phase 0 pre-filter, v_hat is not yet reliable (only
        # one un-converged pass).  Skip tests 1+2 and filter on position only.
        if skip_star_tests:
            ok_star    = np.ones(self.n_stars, dtype=bool)
            ok_diffuse = np.ones(self.n_stars, dtype=bool)  # diffuse test unreliable before Phase 1
        else:
            # Logging (skipped in pre-filter: diffuse-prior chi2 is meaningless
            # before Phase 1 convergence and would just be confusing)
            n_obs          = int(observed.sum())
            n_fail_gaia    = int((~ok_gaia   & observed).sum())
            n_fail_diffuse = int((~ok_diffuse & ok_gaia & observed).sum())
            hyst_str = (f"→{thresh_out_5:.2f}" if ok_star_prev is not None
                        and adaptive_delta > 0 else "")
            n_obs_5p = int(obs_5p.sum())
            n_obs_2p = int(obs_2p.sum())
            def _pct_str(p16, p50, p84, n):
                if np.isnan(p16):
                    return f"(n={n}, <10)"
                return f"[{p16:.1f},{p50:.1f},{p84:.1f}] (n={n})"
            print(f"    thresh  5p+6p:{thresh_gaia_5:.2f}{hyst_str} {_pct_str(p16_5,p50_5,p84_5,n_obs_5p)}  "
                  f"df=2:{thresh_gaia_2:.2f} {_pct_str(p16_2,p50_2,p84_2,n_obs_2p)}  "
                  f"diffuse:{thresh_diff:.1f}")

            # ── Per-population breakdown ──────────────────────────────────────
            # test-1 (Gaia chi2) and test-2 (diffuse) failures, split by population.
            # For 5p/6p stars print individual chi2 values (few enough to be useful).
            # For 2p just show counts.
            fail_gaia_5p  = ~ok_gaia   & obs_5p
            fail_gaia_2p  = ~ok_gaia   & obs_2p
            fail_diff_5p  = ~ok_diffuse & ok_gaia & obs_5p
            fail_diff_2p  = ~ok_diffuse & ok_gaia & obs_2p

            # new rejections vs new admissions (test-1/2 only)
            if ok_star_prev is not None:
                newly_rej = ok_star_prev & ~ok_star & observed
                newly_adm = ~ok_star_prev & ok_star & observed
                chg_str = (f"  ({int(newly_rej.sum())} newly rejected, "
                           f"{int(newly_adm.sum())} newly admitted)")
            else:
                chg_str = ""

            n_fail_gaia    = int((~ok_gaia   & observed).sum())
            n_fail_diffuse = int((~ok_diffuse & ok_gaia & observed).sum())
            print(f"    chi2 outliers (of {n_obs} observed): "
                  f"{n_fail_gaia} Gaia-incompatible "
                  f"({int(fail_gaia_5p.sum())} 5p+6p, {int(fail_gaia_2p.sum())} 2p), "
                  f"{n_fail_diffuse} diffuse "
                  f"({int(fail_diff_5p.sum())} 5p+6p, {int(fail_diff_2p.sum())} 2p)"
                  f"{chg_str}")


        # ── 3. Per-image position chi2 test + alpha estimation + flag update ───
        resid_full     = self.compute_residuals(r_hat, v_hat, C_r, C_vT)
        resid_hst      = self.compute_residuals(r_hat, v_hat)
        _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)

        self.gaia_n_hst_used[:] = 0
        info = []
        n_use_changed = 0

        for img, rd in resid_full.items():
            sidx    = rd["sidx"]

            # Use HST-only chi2 (not C_total) for the per-image position test.
            # C_total inflates sigma_resid when the transformation is poorly
            # constrained (large C_r), masking genuinely bad HST positions.
            # HST-only chi2 is purely about position quality in the detector frame
            # and is stable regardless of transformation uncertainty.
            rd_hst  = resid_hst[img]
            sig_sq  = rd_hst["sigma_resid"]**2   # (n,) HST-noise-only chi2

            # Alpha-scale chi2: divide by alpha² from previous iteration so the
            # threshold is uniform across images in units of "sigma given image
            # noise".  Alpha is estimated from the previous use_for_fit to avoid
            # a chicken-and-egg dependency.  Only applied at iteration >= 3 so
            # early iterations exclude obvious outliers before alpha is reliable.
            prev_use = np.asarray(self._img_data[img]["use_for_fit"])
            # Restricted to informative-prior stars: a diffuse-prior star's
            # residual is absorbed by its own free PM, so its chi2 ~ 0 drags the
            # median (and hence alpha_prev) down.  Same restriction as the alpha
            # estimate below and as ground_to_gaia_xmatch.
            _alpha_ref = prev_use & ~self.needs_diffuse[sidx]
            if alpha_scale_chi2 and iteration >= 3 and _alpha_ref.sum() >= 4:
                chi2_prev = sig_sq[_alpha_ref]
                alpha_prev = float(max(1.0, np.sqrt(
                    np.median(chi2_prev) / _MEDIAN_CHI2_2)))
                sig_sq_eff = sig_sq / alpha_prev**2
            else:
                sig_sq_eff = sig_sq

            # Per-image threshold from globally-accepted stars.
            ok_glob_here = ok_star[sidx]

            # Threshold reference: restrict to initially-trusted stars so that
            # sources that began with use_for_alignment=False (e.g. non-stars
            # whose inflated PSF-fit covariances produce artificially small
            # sigma_resid) cannot pull the adaptive threshold down.
            init_trusted  = self._img_data[img]["use_for_align_init"]
            # Exclude diffuse-prior stars from the threshold reference: the fit
            # absorbs their residual (free PM), so their chi2 ~ 1e-7 collapses
            # p50 and p16 and the adaptive term degenerates to the floor.
            # Three-tier fallback mirrors ground_to_gaia_xmatch exactly.
            ok_thresh_ref = ok_glob_here & init_trusted & ~self.needs_diffuse[sidx]
            if ok_thresh_ref.sum() < 10:
                ok_thresh_ref = ok_glob_here & init_trusted
            if ok_thresh_ref.sum() < 10:
                ok_thresh_ref = ok_glob_here   # fall back if reference set too small

            if chi2_threshold is not None:
                # Fixed threshold; scale expulsion threshold by same ratio as
                # adaptive_k → adaptive_k+delta to preserve hysteresis width.
                thresh_admit = float(chi2_threshold)
                thresh_expel = thresh_admit * (1.0 + adaptive_delta / adaptive_k)
            else:
                thresh_admit, _, _p50_3, _ = _adapt_thresh(sig_sq_eff[ok_thresh_ref],
                                             adaptive_k,
                                             chi2_dist.ppf(chi2_pval, df=2),
                                             floor=floor_2)
                thresh_expel, _, _, _ = _adapt_thresh(sig_sq_eff[ok_thresh_ref],
                                             adaptive_k + adaptive_delta,
                                             chi2_dist.ppf(chi2_pval, df=2),
                                             floor=floor_2)
                # Alpha carve-out.  sig_sq_eff is already divided by alpha^2, so
                # after a successful inflation its centre sits near the
                # theoretical median and the gate is meaningful.  But alpha is
                # capped at inflate_alpha_max: once it saturates the required
                # inflation exceeds what was applied and the residual centre is
                # LEGITIMATELY high.  Gating there would reject exactly the
                # images that most need inflation, so for saturated images the
                # gate is skipped and only the backstop ceiling is kept.
                _aa = self._img_data[img].get("alpha_applied", 1.0)
                _alpha_sat = bool(_aa >= inflate_alpha_max - 1e-9)
                _gm3 = None if _alpha_sat else _gm
                thresh_admit, _g3, _ = _gate_thresh(
                    thresh_admit, _p50_3, 2, floor_2, _gm3, _cm)
                thresh_expel, _, _ = _gate_thresh(
                    thresh_expel, _p50_3, 2, floor_2, _gm3, _cm)
                self._img_data[img]["thresh_gated"] = bool(_g3)

            ok_resid_admit = sig_sq_eff < thresh_admit

            # Hysteresis: currently-included stars use the looser expulsion threshold.
            if adaptive_delta > 0:
                ok_resid = np.where(prev_use, sig_sq_eff < thresh_expel, ok_resid_admit)
            else:
                ok_resid = ok_resid_admit

            # Stricter residual threshold for non-initially-aligned stars (HST-only
            # admitted via callback).  These stars may have biased PM priors and
            # generally larger positional scatter; requiring a smaller sigma_resid
            # limits their influence on the transformation without excluding all of
            # them.  hst_fit_sigma_mult < 1 means they must be hst_fit_sigma_mult ×
            # tighter than Gaia-matched stars; 0.5 ≈ 0.71σ stricter.
            if hst_fit_sigma_mult < 1.0:
                _align_init_k = np.asarray(self._img_data[img]["use_for_align_init"], dtype=bool)
                _hst_in_fit   = (~_align_init_k) & np.asarray(self._img_data[img]["use_for_fit"], dtype=bool)
                if _hst_in_fit.any():
                    ok_resid = ok_resid & (
                        _align_init_k | (sig_sq_eff < thresh_admit * hst_fit_sigma_mult))

            new_use = ok_resid & ok_glob_here

            new_use = np.asarray(new_use, dtype=bool)
            # Hard ceilings: never re-admit initial-filter rejects or
            # test-4 influence-flagged detections (ratchet).
            new_use = new_use & self._img_data[img]["use_for_fit_max"]
            infl_excl = self._img_data[img].get("influence_excl")
            if infl_excl is not None:
                new_use = new_use & ~infl_excl
            # Guard against automatic re-admission of stars that were never in
            # alignment or that were removed from it.  A star can participate in
            # use_for_fit only if it started in alignment (align_init=True) OR if
            # it is CURRENTLY in use_for_fit (explicitly admitted, e.g. by
            # V2AlignmentCallback).  This prevents HST-only stars from flooding
            # the alignment tier through test-3 re-admission: once a star is
            # removed from use_for_fit it cannot re-enter via residual tests.
            align_init    = np.asarray(self._img_data[img]["use_for_align_init"], dtype=bool)
            current_fit   = np.asarray(self._img_data[img]["use_for_fit"],        dtype=bool)
            # Phase-6 outliers (real Gaia detections flagged not-trustworthy by
            # the cross-match validator) may NOT re-enter alignment: in crowded
            # fields these pairs are blends, and re-admitting them to the fit
            # widened ngc_7099 member pmdec sigma_MAD 0.165 -> 0.30 (bright) /
            # 0.61 (faint).  They do not re-enter astrometry either -- see the
            # note at the use_for_astrom update below.
            phase6_out    = np.asarray(self._img_data[img].get("phase6_outlier",
                                       np.zeros(len(current_fit), bool)), dtype=bool)
            can_enter_fit = align_init | current_fit
            new_use = new_use & can_enter_fit

            # Starving-chip demotion: while an image is demoted its ALIGNMENT
            # mask stays empty (its r is frozen), but ASTROMETRY keeps
            # following the admissions — that is the point of the demotion —
            # and the would-be count feeds the promotion check.  Zeroing here
            # (not post-hoc) keeps the churn counters honest: a demoted image
            # contributes 0 test-3 changes instead of admit/zero flapping.
            _admit = new_use
            self._img_data[img]["n_fit_wouldbe"] = int(np.sum(_admit))
            if img in getattr(self, "_align_demoted", set()):
                new_use = np.zeros_like(new_use)

            n_use_changed += int(np.sum(current_fit != new_use))

            # Astrometry mask: match the ADMISSIONS for initially-aligned stars
            # (not the post-demotion alignment mask).  HST-only stars
            # (align_init=False) keep their use_for_astrom unchanged —
            # it is managed externally by V2AlignmentCallback.
            new_use_astrom = np.asarray(self._img_data[img]["use_for_astrom"], dtype=bool).copy()
            new_use_astrom[align_init] = _admit[align_init]
            # NOTE (2026-09-02): an astrometry-tier-only re-admission of
            # Phase-6 pairs was tried here and REVERTED the same day.  In the
            # joint solve, astrometry detections feed the star's PM, so blend
            # epochs polluted stellar astrometry, inflated those stars'
            # residuals in good images, and triggered a test-3/test-4
            # expulsion cascade (ngc_7099: alignment tier 23k -> 10k dets,
            # test-4 avalanches vs 0 without it; reproduced in Draco).  If
            # Phase-6 epochs are ever reclaimed, it must happen POST-SOLVE
            # (frozen r, per-star refit) so nothing feeds back into the fit.
            self._img_data[img]["use_for_astrom"] = new_use_astrom

            # Alpha from informative-prior stars only.  Diffuse-prior stars'
            # residuals are absorbed by their own free PM, so their chi2 ~ 0
            # zeroes the median: measured on LSST det_150, all stars gave
            # alpha_raw = 0.0010 where informative-prior stars gave 0.9862.
            # bp3m's HST fields are 5p-dominated so the bias is smaller here,
            # but the estimator's intent is the same.  (Ported back from
            # ground_to_gaia_xmatch.)
            alpha_pop = new_use & ~self.needs_diffuse[sidx]
            if alpha_pop.sum() >= 4:
                chi2_hst = rd_hst["sigma_resid"][alpha_pop]**2
                alpha_raw = float(np.sqrt(np.median(chi2_hst) / _MEDIAN_CHI2_2))
                alpha_raw = min(alpha_raw, inflate_alpha_max)
            else:
                alpha_raw = 1.0
            # Persist the measured factor and its cap so save_results can report
            # them: alpha_applied alone cannot distinguish "residuals were fine"
            # from "inflation hit the ceiling and the errors are understated".
            self._img_data[img]["alpha_raw"]   = alpha_raw
            self._img_data[img]["alpha_max"]   = float(inflate_alpha_max)
            self._img_data[img]["n_alpha_ref"] = int(alpha_pop.sum())

            if inflate_errors and iteration >= inflate_from_iter:
                # alpha_raw is measured against the already-inflated C_hst, so it
                # equals alpha_true / alpha_prev.  The cumulative inflation needed
                # relative to C_hst_orig is therefore alpha_prev * alpha_raw.
                # Clamping to 1.0 prevents ever deflating below no-inflation
                # (C_hst is never made smaller than C_hst_orig).
                # alpha_prev is the starting alpha from the previous iteration
                # (or the v1 BP3M starting alpha loaded in run_alignment_v2.py),
                # so a decrease from alpha=2.0 to 2.0*alpha_raw is fully supported
                # as long as the result stays >= 1.0.
                alpha_prev = self._img_data[img].get("alpha_applied", 1.0)
                # Cap the CUMULATIVE inflation: alpha_raw is capped per step,
                # but the product compounds across iterations (observed 14.65x
                # against a 3.0 cap -- chi2 deflated ~215x, letting a
                # misfitting image float in the fit while contributing almost
                # nothing).  The saturation carve-out in the test-3 gate
                # already assumes alpha_applied <= inflate_alpha_max.
                alpha_j    = float(min(max(1.0, alpha_prev * alpha_raw),
                                       inflate_alpha_max))
                self._img_data[img]["alpha_applied"] = alpha_j
                self._img_data[img]["C_hst"] = (
                    alpha_j**2 * self._img_data[img]["C_hst_orig"])
            else:
                # Alpha not yet updated: report alpha_raw for diagnostics but keep
                # alpha_applied (and C_hst) unchanged.
                alpha_j = self._img_data[img].get("alpha_applied", 1.0)

            self._img_data[img]["use_for_fit"] = np.asarray(new_use)
            use_any = new_use | new_use_astrom
            self.gaia_n_hst_used[sidx[use_any]] += 1
            n_astrom_only = int((use_any & ~new_use).sum())
            info.append((img, int(new_use.sum()), len(new_use), alpha_j, alpha_raw, n_astrom_only))

        return info, ok_star, n_use_changed

    def _print_test_type_breakdown(self, ok_star_prev, ok_star_new,
                                   snap_use_pre, t4_flagged_info):
        """Print per-iteration breakdown of flagged detections by Gaia type and chip.

        Columns: 2p (Gaia 2-parameter), 5p/6p (full Gaia solution), total.
        Rows: tests 1/2 (global ok_star changes), test 3 (use_for_fit changes
        inside _update_use_for_fit), test 4 (influence_excl additions), and
        current in-use totals after all tests.
        """
        is_2p = self.gaia_2p  # (n_stars,) bool

        # --- Tests 1/2: stars that changed ok_star globally ---
        changed_12 = ok_star_prev != ok_star_new
        n12_2p  = int((changed_12 &  is_2p).sum())
        n12_5p  = int((changed_12 & ~is_2p).sum())

        # --- Test 3: use_for_fit changes from _update_use_for_fit ---
        # Compare snapshot (before _update_use_for_fit) vs current use_for_fit.
        # This includes both direct test-3 chi2 changes AND reflections of
        # ok_star changes, so treat it as "tests 1-3 combined per-detection".
        nt3_2p = nt3_5p = 0
        nt3_hi_2p = nt3_hi_5p = nt3_lo_2p = nt3_lo_5p = 0
        for img, d in self._img_data.items():
            if d is None:
                continue
            old = snap_use_pre.get(img)
            if old is None:
                continue
            sidx = d["sidx"]
            new  = np.asarray(d["use_for_fit"], dtype=bool)
            lost = old & ~new               # detections removed by tests 1-3
            is_2p_det = is_2p[sidx]
            n2p = int((lost &  is_2p_det).sum())
            n5p = int((lost & ~is_2p_det).sum())
            nt3_2p += n2p;  nt3_5p += n5p
            if img.endswith("_hi"):
                nt3_hi_2p += n2p;  nt3_hi_5p += n5p
            elif img.endswith("_lo"):
                nt3_lo_2p += n2p;  nt3_lo_5p += n5p

        # --- Test 4: influence_excl additions ---
        nt4_2p = nt4_5p = 0
        nt4_hi_2p = nt4_hi_5p = nt4_lo_2p = nt4_lo_5p = 0
        for img, flagged_sidx in t4_flagged_info:
            is_2p_flag = is_2p[flagged_sidx]
            n2p = int(is_2p_flag.sum())
            n5p = int((~is_2p_flag).sum())
            nt4_2p += n2p;  nt4_5p += n5p
            if img.endswith("_hi"):
                nt4_hi_2p += n2p;  nt4_hi_5p += n5p
            elif img.endswith("_lo"):
                nt4_lo_2p += n2p;  nt4_lo_5p += n5p

        # --- Current in-use totals ---
        nu_2p = nu_5p = nu_hi = nu_lo = 0
        nu_hi_2p = nu_hi_5p = nu_lo_2p = nu_lo_5p = 0
        for img, d in self._img_data.items():
            if d is None:
                continue
            use  = np.asarray(d["use_for_fit"], dtype=bool)
            sidx = d["sidx"]
            is_2p_det = is_2p[sidx]
            n2p = int((use &  is_2p_det).sum())
            n5p = int((use & ~is_2p_det).sum())
            nu_2p += n2p;  nu_5p += n5p
            if img.endswith("_hi"):
                nu_hi += n2p + n5p
                nu_hi_2p += n2p;  nu_hi_5p += n5p
            elif img.endswith("_lo"):
                nu_lo += n2p + n5p
                nu_lo_2p += n2p;  nu_lo_5p += n5p

        def _row(label, n2p, n5p):
            tot = n2p + n5p
            if tot == 0:
                return
            frac2 = n2p / tot if tot else 0
            print(f"      {label:<22}  2p={n2p:4d}  5p/6p={n5p:4d}  tot={tot:4d}"
                  f"  (2p frac={frac2:.2f})")

        print("    [type breakdown]")
        _row("tests 1-2 (stars)", n12_2p, n12_5p)
        _row("tests 1-3 (dets)", nt3_2p, nt3_5p)
        if nt3_hi_2p + nt3_hi_5p > 0 or nt3_lo_2p + nt3_lo_5p > 0:
            _row("  tests 1-3 _hi", nt3_hi_2p, nt3_hi_5p)
            _row("  tests 1-3 _lo", nt3_lo_2p, nt3_lo_5p)
        if nt4_2p + nt4_5p > 0:
            _row("test-4 (dets)", nt4_2p, nt4_5p)
            if nt4_hi_2p + nt4_hi_5p > 0 or nt4_lo_2p + nt4_lo_5p > 0:
                _row("  test-4 _hi", nt4_hi_2p, nt4_hi_5p)
                _row("  test-4 _lo", nt4_lo_2p, nt4_lo_5p)
        print(f"      {'in-use _hi':<22}  2p={nu_hi_2p:4d}  5p/6p={nu_hi_5p:4d}  tot={nu_hi:4d}"
              f"  (2p frac={nu_hi_2p/nu_hi:.2f})" if nu_hi else "")
        print(f"      {'in-use _lo':<22}  2p={nu_lo_2p:4d}  5p/6p={nu_lo_5p:4d}  tot={nu_lo:4d}"
              f"  (2p frac={nu_lo_2p/nu_lo:.2f})" if nu_lo else "")
        print(f"      {'in-use total':<22}  2p={nu_2p:4d}  5p/6p={nu_5p:4d}"
              f"  tot={nu_2p+nu_5p:4d}"
              f"  (2p frac={(nu_2p/(nu_2p+nu_5p)):.2f})" if (nu_2p+nu_5p) else "")

    def _apply_influence_clip(self, r_hat, C_r, a_arr,
                               k_sigma_resid=5.0, k_scaled_d=5.0,
                               floor_sigma_resid=None, floor_scaled_d=3.0,
                               raw_cooks_d=False, diag_rows=None, it_outer=None):
        """
        Test-4: influence-based clipping with ratchet semantics.

        Flags star-image pairs where sigma_resid > thresh_sr AND
        scaled_D > thresh_sd that are not already influence-excluded.
        Both thresholds are set adaptively from the observed distribution of
        eligible detections using the same p50 + k*(p50-p16) scheme as
        tests 1-3, clamped to theoretical null floors:

            thresh_sr = max(p50_sr + k*(p50_sr - p16_sr), floor_sigma_resid)
            thresh_sd = max(p50_sd + k*(p50_sd - p16_sd), floor_scaled_d)

        sigma_resid is the Mahalanobis position residual sqrt(r^T Cs^{-1} r).
        Under the null it follows chi(2), so the theoretical floor is
        sqrt(chi2(2).ppf(0.99)) ≈ 3.03.

        scaled_D = D * N_R / leverage normalises Cook's D so E[scaled_D] = 1
        regardless of image density.  The theoretical null distribution of
        scaled_D is not analytically clean; floor_scaled_d=3.0 is an
        empirically calibrated default.

        Thresholds are computed globally across all eligible detections (two-
        pass: compute statistics, then apply flags) so that a single outlier
        image does not bias the per-image thresholds.

        Newly flagged pairs are added to ``_img_data[img]["influence_excl"]``,
        a persistent boolean mask that _update_use_for_fit respects as a hard
        ceiling — flagged pairs are never re-admitted by tests 1-3.

        Returns
        -------
        n_new : int
            Number of detections *newly* added to influence_excl.
        flagged_info : list of (img, sidx_flagged)
            Per-image list of star indices (into self.*) that were newly flagged.
        thresh_sr, thresh_sd : float
            Thresholds actually used (for logging).
        """
        from scipy.stats import chi2 as chi2_dist

        nr = self.N_R

        # Theoretical floor for sigma_resid: sqrt(chi2(2).ppf(0.99))
        if floor_sigma_resid is None:
            floor_sigma_resid = float(np.sqrt(chi2_dist.ppf(0.99, df=2)))

        # ── Pass 1: compute statistics per image, collect eligible values ─────
        per_img = {}   # img -> (use, already_excl, sidx, sigma_resid, test_d, cooks_d, leverage)
        all_sr  = []
        all_sd  = []

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue
            use = np.asarray(d["use_for_fit"], dtype=bool)
            if use.sum() < 4:
                continue

            if "influence_excl" not in d:
                d["influence_excl"] = np.zeros(len(use), dtype=bool)
            already_excl = d["influence_excl"]
            eligible = use & ~already_excl

            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            C_r_j = C_r[cs:cs + nr, cs:cs + nr]

            sidx  = d["sidx"]
            X_mat = d["X_mat"]
            JU    = d["JU"]
            xys   = d["xys"]

            Cs     = self._compute_Cs(img, r_j)
            Cs_inv = np.linalg.inv(Cs)

            pred  = (np.einsum('nij,j->ni', X_mat, r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))
            _edd = self._ed_disp(img, r_hat)
            if not np.isscalar(_edd):
                pred = pred + _edd
            resid = xys - pred
            mah2  = np.einsum('ni,nij,nj->n', resid, Cs_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            CsR   = np.einsum('nij,nj->ni', Cs_inv, resid)
            XtCsR = np.einsum('nij,ni->nj', X_mat, CsR)
            delta_r = XtCsR @ C_r_j
            cooks_d = np.sum(XtCsR * delta_r, axis=1) / nr

            XCrX     = np.einsum('nik,kl,njl->nij', X_mat, C_r_j, X_mat)
            leverage = np.einsum('nij,nji->n', Cs_inv, XCrX)

            if raw_cooks_d:
                test_d = cooks_d
            else:
                safe_lev = np.where(leverage > 1e-12, leverage, np.inf)
                test_d   = cooks_d * nr / safe_lev

            per_img[img] = (use, already_excl, eligible, sidx, sigma_resid, test_d, cooks_d, leverage)

            if eligible.any():
                all_sr.extend(sigma_resid[eligible].tolist())
                all_sd.extend(test_d[eligible].tolist())

        # ── Adaptive thresholds from eligible population ───────────────────────
        def _adapt(values, k, floor):
            arr = np.array(values)
            if len(arr) < 10:
                return float(floor)
            p16 = float(np.percentile(arr, 16))
            p50 = float(np.median(arr))
            return float(max(p50 + k * max(p50 - p16, 1e-6), floor))

        thresh_sr = _adapt(all_sr, k_sigma_resid, floor_sigma_resid)
        thresh_sd = _adapt(all_sd, k_scaled_d,    floor_scaled_d)

        # ── Pass 2: apply flagging ─────────────────────────────────────────────
        n_new = 0
        flagged_info = []

        for img, (use, already_excl, eligible, sidx,
                  sigma_resid, test_d, cooks_d, leverage) in per_img.items():
            d = self._img_data[img]

            new_flag = (use
                        & ~already_excl
                        & (sigma_resid > thresh_sr)
                        & (test_d > thresh_sd))

            if diag_rows is not None:
                _gaia_2p_arr = np.asarray(self.gaia_2p) if hasattr(self, 'gaia_2p') else None
                is_gaia_2p = _gaia_2p_arr[sidx] if _gaia_2p_arr is not None else np.zeros(len(sidx), dtype=bool)
                for k in range(len(use)):
                    if use[k] and not already_excl[k]:
                        diag_rows.append({
                            'it_outer': it_outer,
                            'img': img,
                            'sidx': int(sidx[k]),
                            'gaia_2p': bool(is_gaia_2p[k]),
                            'cooks_d': float(cooks_d[k]),
                            'leverage': float(leverage[k]),
                            'test_d': float(test_d[k]),
                            'sigma_resid': float(sigma_resid[k]),
                            'thresh_sr': thresh_sr,
                            'thresh_sd': thresh_sd,
                            'flagged': bool(new_flag[k]),
                        })

            if new_flag.any():
                if (use & ~new_flag).sum() >= 4:
                    d["influence_excl"] = already_excl | new_flag
                    d["use_for_fit"]    = use & ~new_flag
                    n_new += int(new_flag.sum())
                    flagged_info.append((img, sidx[new_flag]))

        return n_new, flagged_info, thresh_sr, thresh_sd

    def compute_analytic_posteriors(self, r_hat, C_r, a_arr, K_img, C_vT):
        """Compute exactly marginalised per-star posteriors analytically.

        Analytic counterpart to sample_posteriors.  The conditional stellar mean
        is a linear function of r:

            v_i(r) = a_arr_i + C_vT_i  Σ_j K_{ij}  (r_j − r_hat_j)

        Marginalising over r ~ N(r_hat, C_r) gives:

            v_mean_i  = a_arr_i                    (mean unchanged)
            C_extra_i = (C_vT_i K_i) C_r (C_vT_i K_i)^T

        where K_i = Σ_{detections of star i across all images} K_{ij}
        is the (5, n_r_tot) linear sensitivity of v_i to r.

        Returns
        -------
        v_mean : (n_stars, 5)      same as a_arr (no change in mean)
        v_cov  : (n_stars, 5, 5)  C_extra = C_u − C_vT  (add C_vT to get full C_u)
        """
        nr      = self.N_R
        n_r_tot = nr * self.n_images
        n_s_tot = n_r_tot + self.n_ed
        n_stars = self.n_stars

        # Build K_all[i, :, cols(j)] = Σ_{detections of star i in img j} K_{ij}
        # cols(j) = the image's r block plus (when fit_epoch_distortion) its
        # group's epoch-D slice — K_img columns follow the same layout.
        K_all = np.zeros((n_stars, 5, n_s_tot))

        for j_idx, img in enumerate(self.image_names):
            if K_img.get(img) is None:
                continue
            d = self._img_data[img]
            use_fit    = d['use_for_fit']
            use_astrom = d.get('use_for_astrom', use_fit)
            use_any    = use_fit | use_astrom
            if not use_any.any():
                continue
            sidx = d['sidx'][use_any]
            K    = K_img[img][use_any]   # (n, 5, m)
            cs   = j_idx * nr
            ec   = self._ed_cols(img)
            if ec is None or K.shape[2] == nr:
                np.add.at(K_all[:, :, cs:cs + nr], sidx, K)
            else:
                np.add.at(K_all[:, :, cs:cs + nr], sidx, K[:, :, :nr])
                _tmp = np.zeros((n_stars, 5, self.ED_K))
                np.add.at(_tmp, sidx, K[:, :, nr:])
                K_all[:, :, ec[0]:ec[0] + self.ED_K] += _tmp

        # C_extra[i] = (C_vT[i] @ K_all[i]) @ C_r @ (C_vT[i] @ K_all[i]).T
        CvT_K   = np.einsum('nij,njk->nik', C_vT, K_all)           # (n_stars, 5, n_r_tot)
        C_extra = CvT_K @ C_r @ np.swapaxes(CvT_K, -1, -2)        # (n_stars, 5, 5)

        return a_arr.copy(), C_extra

    def sample_posteriors(self, r_hat, C_r, a_arr, K_img, C_vT,
                          n_samples=1000, seed=42):
        """
        Draw posterior samples of r, propagate to v_T,i samples.
        Returns v_mean (n_stars, 5) and v_cov (n_stars, 5, 5) marginalised over r.
        """
        rng = np.random.default_rng(seed)
        n_r = r_hat.shape[0]

        # Sample r from N(r_hat, C_r)
        try:
            L = np.linalg.cholesky(C_r + 1e-12*np.eye(n_r))
            r_samp = r_hat + (L @ rng.standard_normal((n_r, n_samples))).T
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(C_r)
            vals = np.maximum(vals, 0)
            # (n_r, n_r) @ (n_r, n_r) → scale columns → (n_r, n_samples)
            r_samp = r_hat[:, None] + (vecs * np.sqrt(vals)[None, :]) @ rng.standard_normal((n_r, n_samples))
            r_samp = r_samp.T  # (n_samples, n_r)

        # v_hat(r) = a + Σ_j (C_vT_i K_{i,j}) (r_j - r_hat_j)
        # (positive sign — same linearisation as compute_analytic_posteriors;
        # from x = X r - JU v, h_v(r) = h_v(r_hat) + K (r - r_hat).)
        v_samp = np.tile(a_arr[None, :, :], (n_samples, 1, 1))  # (n_samp, n_stars, 5)

        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            if K_img[img] is None:
                continue
            d    = self._img_data[img]
            use_align  = d["use_for_fit"]
            use_astrom = d.get("use_for_astrom", use_align)
            use_any    = use_align | use_astrom
            sidx = d["sidx"][use_any]

            K     = K_img[img][use_any]
            CvT_K = np.einsum('nij,njk->nik', C_vT[sidx], K)
            cs    = j_idx * nr
            ec    = self._ed_cols(img)
            if ec is not None and K.shape[2] > nr:
                cols = np.concatenate([np.arange(cs, cs + nr), ec])
            else:
                cols = np.arange(cs, cs + nr)
            r_j_delta = r_samp[:, cols] - r_hat[cols]
            corr      = np.einsum('sk,njk->snj', r_j_delta, CvT_K)
            v_samp[:, sidx, :] += corr

        v_mean = v_samp.mean(axis=0)                         # (n_stars, 5)
        v_cov  = np.array([np.cov(v_samp[:, i, :].T)
                           for i in range(self.n_stars)])     # (n_stars, 5, 5)

        return r_samp, v_mean, v_cov

    def compute_residuals(self, r_hat, v_hat, C_r=None, C_vT=None):
        """
        Compute per-star, per-image fit residuals and sigma-normalised residuals.

        Parameters
        ----------
        r_hat : (n_r,) MAP image transformation vector
        v_hat : (n_stars, 5) MAP stellar astrometry
        C_r   : (n_r, n_r) posterior covariance of r, optional.
        C_vT  : (n_stars, 5, 5) conditional stellar astrometry covariance, optional.
            When both C_r and C_vT are provided the total uncertainty used for
            sigma-normalisation is the sum of three independent contributions:

                C_total = C_s  +  JU C_vT JU^T  +  X C_r_j X^T

            where C_s is the HST measurement noise, JU C_vT JU^T propagates the
            conditional uncertainty in the fitted stellar astrometry, and
            X C_r_j X^T propagates the image-transformation uncertainty.
            When omitted (default), only C_s is used (HST noise only).

        Returns a dict keyed by image name, each value a dict with:
            'X_c'           : (n,) centered HST x pixel positions (X - Xo)
            'Y_c'           : (n,) centered HST y pixel positions (Y - Yo)
            'resid_x'       : (n,) residual in Gaia pseudo-image x [pixels]
            'resid_y'       : (n,) residual in Gaia pseudo-image y [pixels]
            'sigma_x'       : (n,) 1-σ total noise in pseudo-image x [pixels]
            'sigma_y'       : (n,) 1-σ total noise in pseudo-image y [pixels]
            'sigma_resid_x' : (n,) resid_x / sigma_x  [dimensionless σ]
            'sigma_resid_y' : (n,) resid_y / sigma_y  [dimensionless σ]
            'sigma_resid'   : (n,) 2D Mahalanobis distance sqrt(r^T C_total^{-1} r)
                              (chi distribution with 2 dof under the total noise model)
            'sidx'          : (n,) global star indices
            'use'           : (n,) boolean mask (True = used in fit)

        Residual defined as:
            resid = x_obs - (X r_hat - JU v_hat)
                  = xys - X_mat @ r_hat_j + JU @ v_hat_i
        """
        result = {}
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            meta = self.images[img]

            d = self._img_data.get(img)
            if d is None:
                continue
            cs   = j_idx * nr
            r_j  = r_hat[cs:cs + nr]
            sidx = d["sidx"]
            use  = d["use_for_fit"]

            X_mat = d["X_mat"]   # (n, 2, N_R)
            JU    = d["JU"]      # (n, 2, 5)
            xys   = d["xys"]     # (n, 2)

            # Model prediction: X r_j - JU v_hat_i
            pred = (np.einsum('nij,j->ni', X_mat, r_j)
                    - np.einsum('nij,nj->ni', JU, v_hat[sidx]))   # (n, 2)
            _edd = self._ed_disp(img, r_hat)
            if not np.isscalar(_edd):
                pred = pred + _edd
            resid = xys - pred  # (n, 2)

            # ── Total uncertainty in pseudo-image space ───────────────────────
            # C_s = J_poly @ C_hst @ J_poly^T  (n, 2, 2) — HST measurement noise
            C_total = self._compute_Cs(img, r_j)   # (n, 2, 2)

            if C_vT is not None:
                # JU @ C_vT_i @ JU^T  — uncertainty from fitted stellar astrometry
                C_total = C_total + np.einsum(
                    'nik,nkl,njl->nij', JU, C_vT[sidx], JU)   # (n, 2, 2)

            if C_r is not None:
                # X @ C_r_j @ X^T  — uncertainty from image transformation
                C_r_j = C_r[cs:cs + nr, cs:cs + nr]            # (N_R, N_R)
                C_total = C_total + np.einsum(
                    'nik,kl,njl->nij', X_mat, C_r_j, X_mat)    # (n, 2, 2)

            sigma_x = np.sqrt(np.maximum(C_total[:, 0, 0], 0.))   # (n,) pix
            sigma_y = np.sqrt(np.maximum(C_total[:, 1, 1], 0.))   # (n,) pix

            sigma_resid_x = np.where(sigma_x > 0, resid[:, 0] / sigma_x, np.nan)
            sigma_resid_y = np.where(sigma_y > 0, resid[:, 1] / sigma_y, np.nan)

            # 2D Mahalanobis distance: sqrt(resid^T C_total^{-1} resid) per star
            C_total_inv = np.linalg.inv(C_total)   # (n, 2, 2)
            mah2        = np.einsum('ni,nij,nj->n', resid, C_total_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            # Recover centered detector positions from cached X_mat
            # X_mat[:,0,0] = X_c (col 0) and X_mat[:,0,1] = Y_c (col 1) by build_X_matrix
            # Row 0: [x, y, 0, 0, 1, 0, ...], Row 1: [0, 0, x, y, 0, 1, ...]
            X_c = X_mat[:, 0, 0]
            Y_c = X_mat[:, 0, 1]

            result[img] = {
                "X_c":           X_c,
                "Y_c":           Y_c,
                "resid_x":       resid[:, 0],
                "resid_y":       resid[:, 1],
                "sigma_x":       sigma_x,
                "sigma_y":       sigma_y,
                "sigma_resid_x": sigma_resid_x,
                "sigma_resid_y": sigma_resid_y,
                "sigma_resid":   sigma_resid,
                "sidx":          sidx,
                "use":           use,
            }
        return result

    def compute_gdc_residuals(self, r_hat, v_hat, C_r=None, C_vT=None):
        """
        Compute per-detection residuals and full covariance in each image's
        local GDC pixel frame.

        The pseudo-image residual (xys - pred) is back-projected through J⁻¹
        to the GDC-corrected HST pixel frame.  The full covariance propagates
        three contributions back to the same frame:

            C_gdc_total = J⁻¹ @ C_total_pseudo @ J⁻¹ᵀ

        where in pseudo-image space:
            C_total_pseudo = C_s  +  JU C_vT JUᵀ  +  X C_r_j Xᵀ

        C_hst (measurement-only, already in GDC frame) is also saved separately.

        Parameters
        ----------
        r_hat  : (n_r,)          MAP image transformation vector
        v_hat  : (n_stars, 5)    MAP stellar astrometry
        C_r    : (n_r, n_r) or None   alignment parameter covariance
        C_vT   : (n_stars, 5, 5) or None   conditional stellar astrometry cov

        Returns a dict keyed by image name.  Each value is a dict with:
            'X_c'           : (n,) centered GDC pixel x  (= X - Xo)
            'Y_c'           : (n,) centered GDC pixel y  (= Y - Yo)
            'dx_gdc'        : (n,) x residual in GDC frame [pixels]
            'dy_gdc'        : (n,) y residual in GDC frame [pixels]
            'C_hst'         : (n, 2, 2) measurement-only covariance in GDC frame
            'C_gdc_total'   : (n, 2, 2) full covariance in GDC frame
                              (= C_hst when C_r and C_vT are both None)
            'sidx'          : (n,) indices into stellar_astrometry rows
            'use_for_fit'   : (n,) bool — used for transformation fitting
            'use_for_astrom': (n,) bool — used for stellar astrometry
        """
        result = {}
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue
            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            sidx  = d["sidx"]
            X_mat = d["X_mat"]   # (n, 2, N_R)
            JU    = d["JU"]      # (n, 2, 5)
            xys   = d["xys"]     # (n, 2) — Gaia pseudo-image positions

            # Pseudo-image residual: xys - (X r_j - JU v_hat_i)
            pred         = (np.einsum('nij,j->ni', X_mat, r_j)
                            - np.einsum('nij,nj->ni', JU, v_hat[sidx]))
            _edd = self._ed_disp(img, r_hat)
            if not np.isscalar(_edd):
                pred = pred + _edd
            resid_pseudo = xys - pred    # (n, 2)

            # Total covariance in pseudo-image frame
            C_total = self._compute_Cs(img, r_j)             # (n, 2, 2) = J C_hst Jᵀ
            if C_vT is not None:
                C_total = C_total + np.einsum(
                    'nik,nkl,njl->nij', JU, C_vT[sidx], JU)
            if C_r is not None:
                C_r_j   = C_r[cs:cs + nr, cs:cs + nr]
                C_total = C_total + np.einsum(
                    'nik,kl,njl->nij', X_mat, C_r_j, X_mat)

            # Jacobian and inverse; back-project residual and covariance to GDC frame
            if self.poly_order == 1:
                J     = self.R[img]                          # (2, 2) constant
                J_inv = np.linalg.inv(J)                     # (2, 2)
                dxy   = resid_pseudo @ J_inv.T               # (n, 2)
                # J⁻¹ C_total J⁻¹ᵀ  broadcast over n
                C_gdc = np.einsum('ij,njk,lk->nil',
                                  J_inv, C_total, J_inv)     # (n, 2, 2)
            else:
                J     = compute_poly_jacobian(               # (n, 2, 2)
                    r_j, d["X_c"], d["Y_c"], self.poly_order)
                J_inv = np.linalg.inv(J)                     # (n, 2, 2)
                dxy   = np.einsum('nij,nj->ni', J_inv, resid_pseudo)  # (n, 2)
                C_gdc = np.einsum('nij,njk,nlk->nil',
                                  J_inv, C_total, J_inv)     # (n, 2, 2)

            result[img] = {
                "X_c":            d["X_c"],
                "Y_c":            d["Y_c"],
                "dx_gdc":         dxy[:, 0],
                "dy_gdc":         dxy[:, 1],
                "C_hst":          d["C_hst"],
                "C_gdc_total":    C_gdc,
                "sidx":           sidx,
                "use_for_fit":    np.asarray(d["use_for_fit"],  dtype=bool),
                "use_for_astrom": np.asarray(
                    d.get("use_for_astrom", d["use_for_fit"]), dtype=bool),
            }
        return result

    def compute_star_influence(self, r_hat, C_r, a_arr):
        """
        Compute per-star, per-image leverage, influence, and Cook's distance.

        Uses the one-step Newton approximation: if star k were removed from
        image j, r_hat_j would shift by approximately δr = C_r_j @ X_k^T @ Cs_inv_k @ resid_k.

        Parameters
        ----------
        r_hat : (n_r,) converged image parameter vector
        C_r   : (n_r, n_r) posterior covariance of r
        a_arr : (n_stars, 5) converged stellar astrometry (= v_hat)

        Returns
        -------
        pd.DataFrame with one row per (star, image) detection, columns:
            Gaia_id, image_name,
            X_c, Y_c          — centred detector pixel coordinates
            mag               — HST magnitude
            resid_x, resid_y  — pixel residuals
            sigma_resid       — 2D Mahalanobis distance (resid/noise)
            leverage          — hat-matrix trace (0–2; >1 is high leverage)
            infl_a … infl_z   — influence on each image parameter (pixels)
            cooks_d           — Cook's distance analog
            use_for_fit       — was this detection included in the fit?
        """
        import pandas as pd

        nr = self.N_R
        param_names = ['a', 'b', 'c', 'd', 'Δα0', 'Δδ0'][:nr]

        rows = []
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue

            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            C_r_j = C_r[cs:cs + nr, cs:cs + nr]

            sidx  = d["sidx"]          # (n,) global star indices
            X_mat = d["X_mat"]         # (n, 2, N_R)
            JU    = d["JU"]            # (n, 2, 5)
            xys   = d["xys"]           # (n, 2)
            use        = d["use_for_fit"]                         # (n,) bool — alignment
            use_astrom = d.get("use_for_astrom", use)           # (n,) bool — astrometry

            # HST measurement noise covariance and precision
            Cs     = self._compute_Cs(img, r_j)   # (n, 2, 2)
            Cs_inv = np.linalg.inv(Cs)            # (n, 2, 2)

            # Residual using same sign convention as compute_residuals:
            #   pred = X r_j - JU v_hat
            #   resid = xys - pred = xys - X r_j + JU a_arr
            pred  = (np.einsum('nij,j->ni', X_mat, r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))   # (n, 2)
            _edd = self._ed_disp(img, r_hat)
            if not np.isscalar(_edd):
                pred = pred + _edd
            resid = xys - pred   # (n, 2)

            # Mahalanobis distance (HST noise only)
            mah2      = np.einsum('ni,nij,nj->n', resid, Cs_inv, resid)
            sigma_res = np.sqrt(np.maximum(mah2, 0.))

            # ── Influence quantities ─────────────────────────────────────────
            # XtCsR_k = X_k^T Cs_inv_k resid_k  (N_R,) per star
            CsR   = np.einsum('nij,nj->ni', Cs_inv, resid)   # (n, 2)
            XtCsR = np.einsum('nij,ni->nj', X_mat, CsR)      # (n, N_R)

            # delta_r_k = C_r_j @ XtCsR_k  (N_R,) per star
            # C_r_j is symmetric so C_r_j.T = C_r_j
            delta_r = XtCsR @ C_r_j   # (n, N_R)

            # Cook's distance: XtCsR_k . delta_r_k / N_R
            #   = delta_r^T C_r_j^{-1} delta_r / N_R  (since delta_r = C_r_j XtCsR)
            cooks_d = np.sum(XtCsR * delta_r, axis=1) / nr   # (n,)

            # Leverage: tr(Cs_inv_k @ X_k @ C_r_j @ X_k^T)
            XCrX    = np.einsum('nik,kl,njl->nij', X_mat, C_r_j, X_mat)   # (n, 2, 2)
            leverage = np.einsum('nij,nji->n', Cs_inv, XCrX)              # (n,)

            # Detector coordinates from cached X_mat
            X_c = X_mat[:, 0, 0]
            Y_c = X_mat[:, 0, 1]

            # Magnitude from stars_per_image, re-applying the SAME filter that
            # _precompute_geometry used to build _img_data.  Indexing the raw
            # frame would silently misalign magnitudes whenever the isin filter
            # dropped rows (e.g. a catalogue restricted after loading).
            spi = self.stars_per_image.get(img)
            if spi is not None and "mag" in spi.columns:
                _keep = spi["Gaia_id"].isin(self.star_id_to_idx).to_numpy()
                mag_vals = spi["mag"].to_numpy(float)[_keep]
            else:
                mag_vals = np.full(len(sidx), np.nan)

            gaia_ids = self.gaia_cat["Gaia_id"].iloc[sidx].values

            for n in range(len(sidx)):
                row = dict(
                    Gaia_id    = int(gaia_ids[n]),
                    image_name = img,
                    X_c        = float(X_c[n]),
                    Y_c        = float(Y_c[n]),
                    mag        = float(mag_vals[n]) if n < len(mag_vals) else np.nan,
                    resid_x    = float(resid[n, 0]),
                    resid_y    = float(resid[n, 1]),
                    sigma_resid= float(sigma_res[n]),
                    leverage   = float(leverage[n]),
                    cooks_d    = float(cooks_d[n]),
                    use_for_fit   = bool(use[n]),
                    use_for_astrom= bool(use_astrom[n]),
                )
                for p_idx, pname in enumerate(param_names):
                    row[f"infl_{pname}"] = float(delta_r[n, p_idx])
                rows.append(row)

        return pd.DataFrame(rows)
