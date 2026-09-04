"""
Alignment solver — instrument-independent.

Hoisted verbatim from cfht_align_detectors.py.  The physics, the outlier
rejection, the Schur solve and every figure are unchanged; the only edits were:

  * class renamed CFHTAlignmentSolver -> AlignmentSolver
  * the 5p/2p split in make_plots unified on astrometric_params_solved (it
    previously used a finite-PM test here and astrometric_params_solved in the
    Figure 2 series, so the two disagreed within one method)
  * posterior columns renamed *_cfht -> *_xmatch, since they are written by
    every instrument, not just CFHT
  * instrument-specific loading/driver code moved out to align/driver.py and
    the instrument adapters

Everything below this docstring is the original implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from astropy.time import Time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from ..geometry import DEG2MAS, gnomonic as _gnomonic_mas
from bp3m.astro_utils import (
    n_r_from_poly_order, build_X_matrix, compute_poly_jacobian,
    plane_project_tangent_derivs,
    get_tele_position, get_parallax_factors, build_U_matrix,
    michalik_sigma_plx_prior,
    GAIA_SYS_DICT, RAD2MAS, DEG2RAD,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Default alignment params per image for poly_order=1: (a, b, c, d, dRA0, dDec0).
# The authoritative count is solver.N_R = n_r_from_poly_order(poly_order); this
# module-level value is only the poly_order=1 convenience export.
N_R = 6
N_V = 5   # stellar params: (Δα*, Δδ, μα*, μδ, ϖ)

_SIGMA_POS = 1e6    # mas — flat position prior (2p / no-PM stars)
_SIGMA_PM  = 100.0  # mas/yr — flat PM prior for 2p stars

# Default prior widths for xmatch (very loose — effectively flat initially)
_DEFAULT_SIGMA_ROT_DEG  = 10.0    # rotation (bp3m default 0.10 deg)
_DEFAULT_SIGMA_SCALE    = 0.10    # pixel scale ratio
_DEFAULT_SIGMA_SKEW     = 0.01    # skew
_DEFAULT_SIGMA_POINTING = 1e6     # pointing offset (mas)

_INIT_RESID_CLIP_MAS = 5000.0  # hard initial rejection: ~1/4 MegaCam pixel = 47 mas; use 5 arcsec



# ── Helper: build prior (mirrors bp3m solver._make_image_prior) ────────────

def _make_prior(A0, B0, C0, D0, dx0, dy0,
                sigma_rot_deg=_DEFAULT_SIGMA_ROT_DEG,
                sigma_scale=_DEFAULT_SIGMA_SCALE,
                sigma_skew=_DEFAULT_SIGMA_SKEW,
                sigma_pointing=_DEFAULT_SIGMA_POINTING):
    """Return (r_prior, C_r_prior_inv) — identical Jacobian to bp3m."""
    r_prior = np.array([A0, B0, C0, D0, dx0, dy0], dtype=float)
    s   = np.sqrt(A0**2 + B0**2)
    rot = np.arctan2(B0, A0)
    cr, sr = np.cos(rot), np.sin(rot)

    J = np.array([[-s*sr, cr,  1,  0],
                  [ s*cr, sr,  0,  1],
                  [-s*cr,-sr,  0,  1],
                  [-s*sr, cr, -1,  0]], dtype=float)
    sig = np.array([sigma_rot_deg*DEG2RAD, sigma_scale, sigma_skew, sigma_skew])
    C_abcd = J @ np.diag(sig**2) @ J.T

    C_inv = np.zeros((6, 6))
    try:
        C_inv[:4, :4] = np.linalg.inv(C_abcd)
    except np.linalg.LinAlgError:
        C_inv[:4, :4] = np.diag(1.0 / (np.maximum(np.diag(C_abcd), 1e-60)))
    C_inv[4, 4] = sigma_pointing**-2
    C_inv[5, 5] = sigma_pointing**-2
    return r_prior, C_inv


# ── Main solver class ────────────────────────────────────────────────────────

class AlignmentSolver:
    """
    Per-detector alignment solver for xmatch/UNIONS, mirroring bp3m.BP3MSolver.

    Supports joint solve over multiple images (multi-visit) but defaults to
    single-image (per-detector) operation.

    Parameters
    ----------
    image_records : list of dict
        One entry per image (detector × visit).  Each dict must contain:
          'name'     : str     unique label (e.g. 'xmatch_2496941_ext01')
          'expnum'   : int
          'ext'      : int
          'mjd'      : float   MJD of observation
          'ra0'      : float   tangent-point RA  (deg)
          'dec0'     : float   tangent-point Dec (deg)
          'matched'  : DataFrame  xmatch rows for this detector
          'A0','B0','C0','D0' : float  cross-match affine params
          'dx0','dy0'         : float  cross-match offsets (mas, for propagated Gaia)
    gaia_df : DataFrame
        Full Gaia catalog for this field (merged across images).
        Required columns: source_id, ra, dec, ref_epoch,
        ra_error, dec_error, pmra, pmra_error, pmdec, pmdec_error,
        parallax, parallax_error, ra_dec_corr, ra_parallax_corr,
        ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr,
        dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr,
        pmra_pmdec_corr, gmag, ruwe, astrometric_params_solved
    star_id_to_idx : dict   source_id (int64) -> row index in gaia_df
    """

    def __init__(self, image_records, gaia_df, star_id_to_idx,
                 sigma_rot_deg=None, sigma_scale=None,
                 sigma_skew=None, sigma_pointing=None, poly_order=1,
                 exclude_2p_from_alignment=False):
        self.image_names  = [r['name'] for r in image_records]
        self.image_meta   = {r['name']: r for r in image_records}
        self.n_images     = len(image_records)
        if poly_order < 1:
            raise ValueError(f'poly_order must be >= 1, got {poly_order}')
        self.poly_order   = poly_order
        # When set, Gaia 2p stars still receive astrometric posteriors but do
        # NOT contribute to the image-transformation equations (H_rr, h_align,
        # and the Schur correction).  Their diffuse position prior lets their
        # own 5 parameters absorb their residual, so they add rows to the
        # alignment without adding constraint — inflating n_stars and shrinking
        # sigma_r by ~sqrt(N) while the true scatter does not improve.
        # Mirrors bp3m AlignmentSolver(exclude_2p_from_alignment=...).
        self.exclude_2p_from_alignment = exclude_2p_from_alignment
        self.N_R          = n_r_from_poly_order(poly_order)

        self._sigma_rot_deg  = sigma_rot_deg  if sigma_rot_deg  is not None else _DEFAULT_SIGMA_ROT_DEG
        self._sigma_scale    = sigma_scale    if sigma_scale    is not None else _DEFAULT_SIGMA_SCALE
        self._sigma_skew     = sigma_skew     if sigma_skew     is not None else _DEFAULT_SIGMA_SKEW
        self._sigma_pointing = sigma_pointing if sigma_pointing is not None else _DEFAULT_SIGMA_POINTING

        self.gaia_df       = gaia_df.reset_index(drop=True)
        self.star_id_to_idx = star_id_to_idx
        self.n_stars       = len(gaia_df)

        self._setup_gaia_priors()
        self._precompute_geometry()

    # ── Gaia prior setup (exact mirror of bp3m._cache_gaia) ─────────────────

    def _setup_gaia_priors(self):
        g = self.gaia_df

        def _get(col, default=0.0):
            if col in g.columns:
                return g[col].fillna(default).to_numpy(float)
            return np.full(self.n_stars, default)

        ruwe = _get('ruwe', np.nan)
        self.gaia_trustworthy = np.isnan(ruwe) | (ruwe <= 1.4)
        self.gaia_g   = _get('gmag', 20.0)
        self.gaia_ra  = _get('ra',  0.0)
        self.gaia_dec = _get('dec', 0.0)
        ref_epoch = _get('ref_epoch', 2016.0)
        self.gaia_time = Time(ref_epoch, format='jyear', scale='tcb')

        self.gaia_n_obs_used = np.zeros(self.n_stars, dtype=int)
        self.sigma_from_gaia_prior = np.zeros(self.n_stars, dtype=float)
        self.ok_star = np.ones(self.n_stars, dtype=bool)

        # Classify 5p / 6p / 2p (mirrors bp3m exactly)
        aps = _get('astrometric_params_solved', 3.0).astype(int)
        self.gaia_6p = aps == 95          # 6-parameter (pseudocolour)
        self.gaia_5p = (aps == 31) & ~self.gaia_6p  # 5-parameter
        self.gaia_2p = ~self.gaia_5p & ~self.gaia_6p
        self.full_gaia_astrometry = (
            np.isfinite(_get('pmra', np.nan)) &
            np.isfinite(_get('pmdec', np.nan)) &
            np.isfinite(_get('parallax', np.nan))
        )

        # v_survey: (0,0, pmra, pmdec, parallax) — corrections relative to Gaia
        self.v_survey = np.zeros((self.n_stars, N_V))
        for col, idx in [('pmra', 2), ('pmdec', 3), ('parallax', 4)]:
            self.v_survey[:, idx] = _get(col, 0.0)

        # Build C_survey (5×5 Gaia covariance per star)
        ra_e    = _get('ra_error',         1e6)
        dec_e   = _get('dec_error',        1e6)
        pmra_e  = _get('pmra_error',       1e3)
        pmdec_e = _get('pmdec_error',      1e3)
        plx_e   = _get('parallax_error',   1e3)

        corr_pairs = [
            (0,1, _get('ra_dec_corr')),
            (0,2, _get('ra_pmra_corr')),
            (0,3, _get('ra_pmdec_corr')),
            (0,4, _get('ra_parallax_corr')),
            (1,2, _get('dec_pmra_corr')),
            (1,3, _get('dec_pmdec_corr')),
            (1,4, _get('dec_parallax_corr')),
            (2,3, _get('pmra_pmdec_corr')),
            (2,4, _get('parallax_pmra_corr')),
            (3,4, _get('parallax_pmdec_corr')),
        ]
        sigmas   = np.stack([ra_e, dec_e, pmra_e, pmdec_e, plx_e], axis=1)
        corr_mat = np.zeros((self.n_stars, N_V, N_V))
        for i in range(N_V):
            corr_mat[:, i, i] = 1.0
        for i, j, arr in corr_pairs:
            corr_mat[:, i, j] = arr
            corr_mat[:, j, i] = arr

        C = sigmas[:, :, None] * corr_mat * sigmas[:, None, :]

        # ── Apply GAIA_SYS_DICT inflation (identical to bp3m lines 325-330) ──
        # mult_* are SIGMA multipliers — inflate the covariance by their square.
        C[self.gaia_6p] *= GAIA_SYS_DICT['mult_6p'] ** 2
        C[self.gaia_5p] *= GAIA_SYS_DICT['mult_5p'] ** 2
        C[self.gaia_2p] *= GAIA_SYS_DICT['mult_2p']
        sys_diag = np.array([0, 0,
                              GAIA_SYS_DICT['pm_sys_err'],
                              GAIA_SYS_DICT['pm_sys_err'],
                              GAIA_SYS_DICT['parallax_sys_err']])**2
        C += np.diag(sys_diag)[None, :, :]  # broadcast over n_stars

        self.C_survey = C
        self.v_prior  = self.v_survey
        self.C_prior  = self.C_survey

        # Invert C_survey: full 5×5 for 5p/6p, 2×2 position block for 2p
        self.C_survey_inv = np.zeros_like(C)
        fga = self.full_gaia_astrometry
        if fga.any():
            self.C_survey_inv[fga] = np.linalg.inv(C[fga])
        if (~fga).any():
            self.C_survey_inv[~fga, :2, :2] = np.linalg.inv(C[~fga, :2, :2])
        self.C_survey_inv_dot_v = np.einsum('nij,nj->ni', self.C_survey_inv, self.v_survey)

        # ── DELVE priors (information-form combination with Gaia) ─────────────
        self._add_delve_priors(g, _get)

        # ── Diffuse prior for 2p stars (Michalik et al. 2015 for parallax) ───
        # A 2p star with a surviving DELVE prior must NOT also get the flat
        # 100 mas/yr diffuse PM prior — DELVE supplies real PM information, and
        # stacking the diffuse prior on top would dilute it.
        needs_diffuse = self.gaia_2p.copy()
        # Stored so the alpha estimator (and anything else needing "does this
        # star have an informative position prior?") can use one definition.
        # Extend this, not gaia_2p, if diffuse-prior rows are ever added for
        # QSOs, HST-only detections, or survey-only sources.
        self.needs_diffuse = needs_diffuse
        sigma_plx_prior = np.full(self.n_stars, np.inf)
        if needs_diffuse.any():
            sigma_plx_prior[needs_diffuse] = michalik_sigma_plx_prior(
                self.gaia_ra[needs_diffuse],
                self.gaia_dec[needs_diffuse],
                self.gaia_g[needs_diffuse],
            )

        # _C_VG_inv: (n_stars, N_V) diagonal precision additions for diffuse prior
        self._C_VG_inv = np.zeros((self.n_stars, N_V), dtype=float)
        self._C_VG_inv[needs_diffuse, 0] = _SIGMA_POS**-2
        self._C_VG_inv[needs_diffuse, 1] = _SIGMA_POS**-2
        needs_diffuse_pm = needs_diffuse & ~self._has_delve_pm
        self._C_VG_inv[needs_diffuse_pm, 2] = _SIGMA_PM**-2
        self._C_VG_inv[needs_diffuse_pm, 3] = _SIGMA_PM**-2
        fin_plx = needs_diffuse & np.isfinite(sigma_plx_prior)
        self._C_VG_inv[fin_plx, 4] = sigma_plx_prior[fin_plx]**-2

        # _sigma_diff: (n_stars, N_V) for diffuse-prior outlier test
        self._sigma_diff_per_star = np.full((self.n_stars, N_V), 1e9)
        self._sigma_diff_per_star[needs_diffuse, 0] = 1e4
        self._sigma_diff_per_star[needs_diffuse, 1] = 1e4
        self._sigma_diff_per_star[needs_diffuse_pm, 2] = _SIGMA_PM
        self._sigma_diff_per_star[needs_diffuse_pm, 3] = _SIGMA_PM
        # A DELVE-matched 2p star's diffuse-prior chi2 test uses the DELVE PM
        # error, not the 100 mas/yr flat width, or the test could never fire.
        if self._has_delve_pm.any():
            _h2 = needs_diffuse & self._has_delve_pm
            self._sigma_diff_per_star[_h2, 2] = _get('delve_pmra_error', np.inf)[_h2]
            self._sigma_diff_per_star[_h2, 3] = _get('delve_pmdec_error', np.inf)[_h2]
        self._sigma_diff_per_star[fin_plx, 4] = sigma_plx_prior[fin_plx]

    # ── DELVE priors ─────────────────────────────────────────────────────────

    # Consistency-veto thresholds: chi2.ppf(0.9973, df) — a 3-sigma equivalent.
    _DELVE_VETO_5D = 17.7   # df=5, Gaia 5p/6p vs DELVE on all five parameters
    _DELVE_VETO_2D = 11.8   # df=2, Gaia 2p vs DELVE on position only

    def _add_delve_priors(self, g, _get):
        """
        Fold DELVE PM/parallax precision into the survey prior (bp3m parity).

        Information-form combination: C_survey_inv += C_delve_inv, and
        C_survey_inv_dot_v += C_delve_inv @ v_delve.  Adding precisions is what
        makes the two catalogues combine correctly whether Gaia contributes a
        full 5x5 (5p/6p) or only a 2x2 position block (2p) — in the latter case
        Gaia's 2x2 plus DELVE's full 5x5 is what makes the star's 5x5
        information matrix well-defined at all.

        Before combining, each DELVE prior faces a consistency veto, because a
        wrong cross-match or a bad DECam fit would otherwise inject a confident
        wrong prior:
          * Gaia 5p/6p — full 5D chi2 of (Gaia - DELVE) against C_gaia + C_delve;
          * Gaia 2p    — 2D position chi2 only, since 2p has no PM or parallax
                         to compare against.

        v_delve expresses position as an OFFSET from the Gaia catalogue position
        in mas (zero when the two agree), matching v_survey's convention where
        the position components are zero by construction.

        Caveat carried over from bp3m: DELVE used Gaia DR3 as its astrometric
        reference frame, so the two are not strictly independent.  Treating them
        as independent is a deliberate, minor approximation.

        Sets self._has_delve_pm and, for stars with a surviving prior, replaces
        v_prior / C_prior with the combined values.
        """
        self._has_delve_pm = np.zeros(self.n_stars, dtype=bool)
        if 'delve_pmra_error' not in getattr(g, 'columns', []):
            return

        d_pmra_e  = _get('delve_pmra_error',  np.inf)
        d_pmdec_e = _get('delve_pmdec_error', np.inf)
        d_pmra    = _get('delve_pmra',  0.0)
        d_pmdec   = _get('delve_pmdec', 0.0)
        d_plx     = _get('delve_parallax', 0.0)
        d_ra      = _get('delve_ra_cat',  np.nan)
        d_dec     = _get('delve_dec_cat', np.nan)

        has_d = (np.isfinite(d_pmra_e) & (d_pmra_e > 0)
                 & np.isfinite(d_pmdec_e) & (d_pmdec_e > 0))
        if not has_d.any():
            return
        idx = np.where(has_d)[0]

        # DELVE 5x5 in the solver's parameter order (dRA*, dDec, pmra, pmdec, plx).
        d_sig = np.column_stack([
            _get('delve_ra_error',       np.inf)[idx],
            _get('delve_dec_error',      np.inf)[idx],
            d_pmra_e[idx], d_pmdec_e[idx],
            _get('delve_parallax_error', np.inf)[idx],
        ])
        corr = np.zeros((len(idx), N_V, N_V))
        for k in range(N_V):
            corr[:, k, k] = 1.0
        for i, j, name in [(0, 1, 'delve_corr_ra_dec'), (0, 2, 'delve_corr_ra_pmra'),
                           (0, 3, 'delve_corr_ra_pmdec'), (0, 4, 'delve_corr_ra_plx'),
                           (1, 2, 'delve_corr_dec_pmra'), (1, 3, 'delve_corr_dec_pmdec'),
                           (1, 4, 'delve_corr_dec_plx'), (2, 3, 'delve_corr_pmra_pmdec'),
                           (2, 4, 'delve_corr_plx_pmra'), (3, 4, 'delve_corr_plx_pmdec')]:
            corr[:, i, j] = corr[:, j, i] = _get(name, 0.0)[idx]
        C_d = d_sig[:, :, None] * corr * d_sig[:, None, :]
        full5 = np.isfinite(d_sig).all(axis=1) & (d_sig > 0).all(axis=1)

        cosd = np.cos(np.radians(self.gaia_dec))
        dpos = np.column_stack([
            (self.gaia_ra[idx] - d_ra[idx]) * cosd[idx] * DEG2MAS,
            (self.gaia_dec[idx] - d_dec[idx]) * DEG2MAS,
        ])

        # ── veto: Gaia 5p/6p, full 5D ────────────────────────────────────────
        is5 = self.full_gaia_astrometry[idx]
        if is5.any():
            t5 = idx[is5]
            dv5 = np.column_stack([
                dpos[is5, 0], dpos[is5, 1],
                _get('pmra', 0.0)[t5] - d_pmra[t5],
                _get('pmdec', 0.0)[t5] - d_pmdec[t5],
                _get('parallax', 0.0)[t5] - d_plx[t5],
            ])
            chi2 = np.zeros(len(t5))
            ok = full5[is5] & np.isfinite(dv5).all(axis=1)
            if ok.any():
                Cc = self.C_survey[t5] + C_d[is5]
                x = np.linalg.solve(Cc[ok], dv5[ok, :, None]).squeeze(-1)
                chi2[ok] = np.einsum('ni,ni->n', dv5[ok], x)
            bad = chi2 > self._DELVE_VETO_5D
            if bad.any():
                has_d[t5[bad]] = False
                print(f'  DELVE prior vetoed for {int(bad.sum())} Gaia-5p star(s) '
                      f'(5D discrepant >3sigma; chi2 > {self._DELVE_VETO_5D})')

        # ── veto: Gaia 2p, position only ─────────────────────────────────────
        is2 = ~self.full_gaia_astrometry[idx] & full5 & has_d[idx]
        if is2.any():
            t2 = idx[is2]
            chi2 = np.zeros(len(t2))
            ok = np.isfinite(dpos[is2]).all(axis=1)
            if ok.any():
                Cp = self.C_survey[t2][:, :2, :2] + C_d[is2][:, :2, :2]
                x = np.linalg.solve(Cp[ok], dpos[is2][ok, :, None]).squeeze(-1)
                chi2[ok] = np.einsum('ni,ni->n', dpos[is2][ok], x)
            bad = chi2 > self._DELVE_VETO_2D
            if bad.any():
                has_d[t2[bad]] = False
                print(f'  DELVE prior vetoed for {int(bad.sum())} Gaia-2p star(s) '
                      f'(position offset >3sigma; chi2 > {self._DELVE_VETO_2D})')

        # ── combine the survivors ────────────────────────────────────────────
        surv = has_d[idx] & full5
        # A 2p star whose DELVE 5x5 is incomplete can use neither the DELVE prior
        # nor (had we left has_d set) the diffuse one.  Clear it so the diffuse
        # prior takes over rather than leaving it with no PM prior at all.
        has_d[idx[~full5]] = False
        if surv.any():
            gi = idx[surv]
            C_inv_d = np.linalg.inv(C_d[surv])
            self.C_survey_inv[gi] += C_inv_d
            v_d = np.column_stack([
                (d_ra[gi] - self.gaia_ra[gi]) * cosd[gi] * DEG2MAS,
                (d_dec[gi] - self.gaia_dec[gi]) * DEG2MAS,
                d_pmra[gi], d_pmdec[gi], d_plx[gi],
            ])
            self.C_survey_inv_dot_v[gi] += np.einsum('nij,nj->ni', C_inv_d, v_d)
            # v_prior / C_prior must describe the COMBINED prior, since every
            # downstream test (test 1 especially) measures against them.
            C_pri = np.linalg.inv(self.C_survey_inv[gi])
            self.C_prior = self.C_prior.copy()
            self.v_prior = np.asarray(self.v_prior).copy()
            self.C_prior[gi] = C_pri
            self.v_prior[gi] = np.einsum('nij,nj->ni', C_pri,
                                         self.C_survey_inv_dot_v[gi])
            n5 = int(self.full_gaia_astrometry[gi].sum())
            print(f'  DELVE priors combined for {len(gi)} star(s) '
                  f'({n5} Gaia 5p/6p, {len(gi) - n5} Gaia 2p)')
        self._has_delve_pm = has_d

    # ── Geometry precomputation ──────────────────────────────────────────────

    def _precompute_geometry(self, verbose=True):
        """
        For each image: compute JU, X_mat, xys (unpropagated Gaia at Gaia epoch),
        C_src (source position covariance), r_prior, C_r_prior_inv.

        xys is the UNPROPAGATED Gaia position in the gnomonic frame:
            xys[k] = gaia_xi_propagated[k] - JU[k] @ v_survey[k]
        so that the model   xys = X@r - JU@v + noise
        gives correct residuals at v = v_survey (catalog PM/plx values).
        """
        print("Precomputing geometry...")
        self._img_data = {}
        self.gaia_n_obs_used[:] = 0

        for img in self.image_names:
            meta    = self.image_meta[img]
            matched = meta['matched']

            # ── Build star index array for this detector ─────────────────────
            ids  = matched['gaia_source_id'].astype('int64').values
            sidx_list, row_list = [], []
            for i, gid in enumerate(ids):
                if gid in self.star_id_to_idx:
                    sidx_list.append(self.star_id_to_idx[gid])
                    row_list.append(i)
            if len(sidx_list) < 3:   # 3 stars = 6 constraints = 6 params
                self._img_data[img] = None
                continue

            sidx = np.array(sidx_list, dtype=int)
            rows = np.array(row_list, dtype=int)
            sub  = matched.iloc[rows].reset_index(drop=True)
            n    = len(sidx)

            # ── Tangent point in force for this pass ─────────────────────────
            # ra0_current accumulates the fitted dRA0/dDec0 each outer
            # iteration (rolling re-linearisation).  On the first call it is
            # the cross-match tangent point; ra0_orig is kept so the total
            # excursion can be reported as delta_ra0_mas.
            if 'ra0_current' not in meta:
                meta['ra0_orig'], meta['dec0_orig'] = meta['ra0'], meta['dec0']
                meta['ra0_current'], meta['dec0_current'] = meta['ra0'], meta['dec0']
            ra0_c, dec0_c = meta['ra0_current'], meta['dec0_current']

            # ── Source positions: FIXED detector-frame pseudopixels ──────────
            # These play the role of bp3m's (x_hst, y_hst) detector pixel
            # coordinates and must NOT be re-projected when the tangent point
            # moves.  Re-projecting them shifts the source and Gaia sides by
            # nearly the same amount, leaving the residual invariant, so the
            # tangent-point parameter loses its leverage and the fit re-requests
            # the same offset every iteration instead of converging (observed:
            # a steady 1.514 mas per pass that never decayed).  Only the Gaia
            # side is re-projected below.
            ra_src  = sub['src_ra'].values.astype(float)
            dec_src = sub['src_dec'].values.astype(float)
            xi_s  = sub['src_xi_mas'].values.astype(float)
            eta_s = sub['src_eta_mas'].values.astype(float)

            # ── Source position covariance ───────────────────────────────────
            # Use the raw xmatch measurement covariance (src_cov_*), NOT
            # sigma_xi/eta_mas.  The latter are the *total* match covariance
            # C_g + C_proj + C_model + resid_cov, so using them double-counts
            # the Gaia uncertainty that _setup_gaia_priors already applies as
            # the prior — and for 2p stars C_g ~ 1000 mas swamps everything,
            # making the xmatch measurement look ~75x worse than it is.
            C_src = np.zeros((n, 2, 2))
            if 'src_cov_xx' in sub.columns:
                C_src[:, 0, 0] = sub['src_cov_xx'].values.astype(float)
                C_src[:, 1, 1] = sub['src_cov_yy'].values.astype(float)
                C_src[:, 0, 1] = C_src[:, 1, 0] = sub['src_cov_xy'].values.astype(float)
            else:
                # Legacy xmatch output without src_cov_* — fall back, but warn:
                # results will be biased by the Gaia double-count described above.
                print("    WARNING: no src_cov_* columns; falling back to "
                      "sigma_xi/eta_mas (Gaia covariance double-counted). "
                      "Re-run the cross-match to fix.")
                C_src[:, 0, 0] = sub['sigma_xi_mas'].values.astype(float)**2
                C_src[:, 1, 1] = sub['sigma_eta_mas'].values.astype(float)**2

            # ── Time and parallax factors ────────────────────────────────────
            xm_time = Time(meta['mjd'], format='mjd')
            xm_yr   = xm_time.jyear
            gaia_yr   = self.gaia_time[sidx].jyear  # per-star Gaia epoch
            dt_yr     = xm_yr - gaia_yr            # (n,) years

            tele_xyz = get_tele_position(xm_time, curr_id='earth')
            meta['tele_xyz'] = tele_xyz

            d_plx_ra  = np.zeros(n)
            d_plx_dec = np.zeros(n)
            for k in range(n):
                d_plx_ra[k], d_plx_dec[k] = get_parallax_factors(
                    self.gaia_ra[sidx[k]], self.gaia_dec[sidx[k]], tele_xyz)

            # ── JU matrix (J=I, so JU = U) — (n, 2, 5) ─────────────────────
            JU = np.zeros((n, 2, N_V))
            for k in range(n):
                JU[k] = build_U_matrix(dt_yr[k], d_plx_ra[k], d_plx_dec[k])

            # ── X_mat — (n, 2, N_R) ──────────────────────────────────────────
            # Columns 4,5 are the TANGENT-POINT offsets (dRA0, dDec0) in mas,
            # not constant translations.  Their design columns are the per-star
            # derivatives d(xi,eta)/d(ra0,dec0), which vary ~0.3% across a
            # detector — that variation is the accuracy bp3m gains, and it is
            # what makes dRA0/dDec0 non-degenerate with a bulk pixel shift.
            #
            # bp3m's plane_project has x along -RA while our gnomonic has xi
            # along +RA, so the two x-derivatives are negated into our frame.
            # Verified against central differences of gnomonic(): agreement to
            # ~1e-11 relative.
            # Evaluated at the CURRENT tangent point: these describe how the
            # projected Gaia positions respond to moving it, so they must track
            # ra0_current even though xi_s/eta_s do not.
            dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = plane_project_tangent_derivs(
                ra_src, dec_src, ra0_c, dec0_c, 1.0)
            dxi_dra0,  dxi_ddec0  = -dxs_dra0, -dxs_ddec0
            deta_dra0, deta_ddec0 =  dys_dra0,  dys_ddec0

            # Polynomial basis scale: the detector half-width in mas, so
            # higher-order monomials stay the same order as the linear terms.
            # bp3m uses a hardcoded 2048 because its basis is in pixels.
            if not hasattr(self, '_poly_scale'):
                self._poly_scale = float(max(
                    np.nanpercentile(np.abs(xi_s), 99),
                    np.nanpercentile(np.abs(eta_s), 99), 1.0))

            X_mat = np.zeros((n, 2, self.N_R))
            X_mat[:, 0, 0] = xi_s;  X_mat[:, 0, 1] = eta_s
            X_mat[:, 0, 4] = dxi_dra0;   X_mat[:, 0, 5] = dxi_ddec0
            X_mat[:, 1, 2] = xi_s;  X_mat[:, 1, 3] = eta_s
            X_mat[:, 1, 4] = deta_dra0;  X_mat[:, 1, 5] = deta_ddec0

            # Higher-order polynomial terms, scaled so each degree-k monomial
            # stays the same order as the linear terms (bp3m divides by
            # 2048^(k-1) for its pixel basis; ours is in mas, so the scale is
            # the detector half-width recorded in _poly_scale).
            if self.poly_order > 1:
                _S = self._poly_scale
                col = 6
                for deg in range(2, self.poly_order + 1):
                    sc = _S ** (deg - 1)
                    for jj in range(deg + 1):
                        X_mat[:, 0, col] = xi_s**(deg-jj) * eta_s**jj / sc
                        col += 1
                    for jj in range(deg + 1):
                        X_mat[:, 1, col] = xi_s**(deg-jj) * eta_s**jj / sc
                        col += 1

            # ── xys: UNPROPAGATED Gaia position in tangent plane ─────────────
            # gaia_xi_mas from xmatch = propagated to xmatch epoch.
            # We need unpropagated: xys = gaia_xi_propagated - JU @ v_survey
            # Re-projected at the current tangent point, for the same reason.
            xi_g_prop, eta_g_prop = _gnomonic_mas(
                sub['gaia_ra_prop'].values.astype(float),
                sub['gaia_dec_prop'].values.astype(float), ra0_c, dec0_c)
            gaia_prop  = np.stack([xi_g_prop, eta_g_prop], axis=1)  # (n, 2)

            v_s = self.v_survey[sidx]                               # (n, 5)
            motion = np.einsum('nij,nj->ni', JU, v_s)              # (n, 2)
            xys    = gaia_prop - motion                             # (n, 2)

            # ── Initial alignment parameters ─────────────────────────────────
            A0  = float(meta.get('A0', 1.0))
            B0  = float(meta.get('B0', 0.0))
            C0  = float(meta.get('C0', 0.0))
            D0  = float(meta.get('D0', 1.0))
            # A plane translation (dx0, dy0) maps to a tangent-point offset via
            # d(xi)/d(ra0) = -cos(dec0) and d(eta)/d(dec0) = -1 (verified
            # numerically), i.e. (dRA0, dDec0) = (-dx0/cos(dec0), -dy0).  After
            # the first re-linearisation the offset is absorbed into
            # ra0_current and the prior restarts at zero.
            if meta.get('_relinearised'):
                dx0 = dy0 = 0.0
            else:
                _cd = np.cos(np.radians(dec0_c))
                dx0 = -float(meta.get('dx0', 0.0)) / (_cd if _cd != 0 else 1.0)
                dy0 = -float(meta.get('dy0', 0.0))
            r_prior, C_r_prior_inv = _make_prior(
                A0, B0, C0, D0, dx0, dy0,
                sigma_rot_deg  = self._sigma_rot_deg,
                sigma_scale    = self._sigma_scale,
                sigma_skew     = self._sigma_skew,
                sigma_pointing = self._sigma_pointing,
            )
            r_init = r_prior.copy()

            # ── Initial residual screening ────────────────────────────────────
            R0   = np.array([[A0, B0], [C0, D0]])
            pred0 = np.einsum('ij,nj->ni', R0, np.stack([xi_s, eta_s], axis=1))
            pred0[:, 0] += dx0;  pred0[:, 1] += dy0
            resid0    = gaia_prop - pred0
            med0      = np.nanmedian(resid0, axis=0)
            resid0_c  = resid0 - med0
            resid_mag = np.hypot(resid0_c[:, 0], resid0_c[:, 1])
            ok_init   = resid_mag <= _INIT_RESID_CLIP_MAS

            # Initial trust mask: trustworthy Gaia RUWE
            use_init = ok_init & self.gaia_trustworthy[sidx]
            if use_init.sum() < 4:
                use_init = ok_init.copy()

            # use_for_alignment gates whether a detection may CALIBRATE the
            # transform, separately from whether it is fitted at all.  bp3m sets
            # it False for DELVE-only detections (see
            # data_loader_flc._build_delve_only_stars_df: use_for_alignment =
            # ok_sat if delve_use_for_align else zeros, while use_for_fit stays
            # True), because DELVE astrometry is not trusted to define an image
            # frame -- but a DELVE-only star's ground positions still constrain
            # its own PM through the astrometry-only path.  Without this the
            # column would be silently ignored and DELVE-only stars would help
            # set the transform they are being measured against.
            if 'use_for_alignment' in sub.columns:
                _ufa = sub['use_for_alignment'].to_numpy()
                if _ufa.dtype == object:
                    _ufa = np.isin(np.char.lower(_ufa.astype(str)),
                                   ('true', 't', '1', 'yes'))
                else:
                    _ufa = _ufa.astype(bool)
                use_init = use_init & _ufa

            n_rej_init = int(np.sum(~use_init))
            if n_rej_init > 0:
                print(f"  {img}: {n_rej_init}/{n} rejected by initial residual screen")

            self.gaia_n_obs_used[sidx[use_init]] += 1

            self._img_data[img] = {
                'sidx'              : sidx,
                'n'                 : n,
                'xys'               : xys,           # (n, 2) unpropagated Gaia
                'JU'                : JU,             # (n, 2, 5)
                'X_mat'             : X_mat,          # (n, 2, 6)
                'C_src'             : C_src.copy(),   # (n, 2, 2) source pos cov (current, may be inflated)
                'C_src_orig'        : C_src.copy(),   # (n, 2, 2) original source pos cov (never modified)
                'alpha_applied'     : 1.0,            # cumulative alpha factor applied to C_src
                'alpha_raw'         : 1.0,            # last measured alpha (pre-cap composition)
                'alpha_max'         : np.nan,         # cap in force (inflate_alpha_max)
                'r_prior'           : r_prior,        # (6,)
                'r_init'            : r_init,         # (6,)
                'C_r_prior_inv'     : C_r_prior_inv,  # (6, 6)
                'use_for_fit'       : use_init.copy(),
                'use_for_fit_max'   : ok_init.copy(),
                'use_for_align_init': use_init.copy(),
                # Separate astrometry tier (bp3m use_for_astrom).  A detection
                # can inform a star's astrometry without being trusted to
                # constrain the transform.  For Gaia-matched rows the two tiers
                # track each other; the distinction exists for survey-only rows
                # (bp3m's HST-only stars) admitted outside the alignment fit.
                'use_for_astrom'    : use_init.copy(),
                'A0': A0, 'B0': B0, 'C0': C0, 'D0': D0,
            }

        n_total = sum(d['n'] for d in self._img_data.values() if d)
        if verbose:
            print(f"  Done: {n_total} star–image pairs across {self.n_images} image(s).")

    # ── Transformed source covariance C_s = R @ C_src @ R.T ────────────────

    def _compute_Cs(self, img, r_j):
        """C_s[k] = R_j @ C_src[k] @ R_j.T  where R_j = [[A,B],[C,D]]."""
        d    = self._img_data[img]
        R_j  = np.array([[r_j[0], r_j[1]], [r_j[2], r_j[3]]])
        return R_j @ d['C_src'] @ R_j.T   # (n, 2, 2) via broadcast

    # ── Single solve pass (Schur complement) ────────────────────────────────

    def _relinearise_tangent_point(self, r_hat, v_hat):
        """
        Accumulate the fitted dRA0/dDec0 into each image's tangent point and
        rebuild the geometry there (bp3m solver.py:900-935).

        r_j[4] is dRA0 in mas, defined so that ra0_true = ra0_current - dRA0/3.6e6
        — hence the subtraction.  After absorbing it, r_hat[4:6] is reset to zero
        so the next solve starts from zero offset at the new tangent point, and
        the geometry (xi/eta, the Jacobians, and the tangent-point derivative
        columns) is recomputed there.  Recomputing keeps the linearisation valid
        when the total excursion is large; this is why r_j[4] must never be used
        as an output — it is zero by construction after every re-linearisation.
        The reportable quantity is delta_ra0_mas = (ra0_final - ra0_orig)*3.6e6.
        """
        nr = self.N_R
        moved = 0.0
        for j_idx, img in enumerate(self.image_names):
            if self._img_data.get(img) is None:
                continue
            meta = self.image_meta[img]
            r_j = r_hat[j_idx * nr:(j_idx + 1) * nr]
            meta['ra0_current']  -= float(r_j[4]) / 3_600_000.0
            meta['dec0_current'] -= float(r_j[5]) / 3_600_000.0
            meta['ra0_final']    = meta['ra0_current']
            meta['dec0_final']   = meta['dec0_current']
            meta['_relinearised'] = True
            moved = max(moved, abs(float(r_j[4])), abs(float(r_j[5])))
            r_hat[j_idx * nr + 4] = 0.0
            r_hat[j_idx * nr + 5] = 0.0

        # Rebuild geometry at the updated tangent points, preserving the
        # detection masks that the outlier tests have established.
        # influence_excl MUST be in this list.  _precompute_geometry below
        # rebuilds _img_data from scratch, so any key not carried across is
        # silently reset -- and test 4 relies on influence_excl being a ratchet
        # (`influence_excl = already_excl | new_flag`).  Dropping it re-armed the
        # same detections every outer iteration, so n_inf never reached 0 and the
        # loop could never satisfy its convergence test: the Fornax joints sat at
        # an exactly-repeating `0 test-1/2, 4 test-3, 4 test-4` for iterations
        # 15-20 and burned all 20 iterations.  bp3m is unaffected because its
        # _update_geometry edits in place instead of rebuilding.
        keep = {img: {k: d[k] for k in
                      ('use_for_fit', 'use_for_fit_max', 'use_for_align_init',
                       'use_for_astrom', 'alpha_applied', 'alpha_raw',
                       'alpha_max', 'n_alpha_ref', 'influence_excl',
                       'thresh_gated', 'alpha_saturated_at_test3')
                      if k in d}
                for img, d in self._img_data.items() if d is not None}
        self._precompute_geometry(verbose=False)
        for img, saved in keep.items():
            if self._img_data.get(img) is not None:
                self._img_data[img].update(saved)
                self._img_data[img]['C_src'] = (
                    self._img_data[img]['alpha_applied']**2
                    * self._img_data[img]['C_src_orig'])
        return moved

    def _solve_one_pass(self, r_current, need_cov=True):
        """
        Single linear solve: marginalise over stellar astrometry analytically.

        Returns
        -------
        r_hat   : (n_r,)  absolute alignment parameter vector
        C_r     : (n_r, n_r)  posterior covariance of r
        a_arr   : (n_stars, 5)  stellar posterior mean (given r_hat)
        K_img   : dict{img → (n, 5, 6)}
        C_vT    : (n_stars, 5, 5)  conditional stellar covariance
        """
        nr = self.N_R
        n_r   = nr * self.n_images

        # ── Prior on stellar astrometry ──────────────────────────────────────
        H_vv   = self.C_survey_inv.copy()                      # (n_stars, 5, 5)
        H_vv[:, np.arange(N_V), np.arange(N_V)] += self._C_VG_inv  # add diffuse prior

        h_align = self.C_survey_inv_dot_v.copy()              # (n_stars, 5)
        h_all   = self.C_survey_inv_dot_v.copy()

        H_rr       = np.zeros((n_r, n_r))
        K_img      = {}
        XCs_xresid = {}

        for j_idx, img in enumerate(self.image_names):
            d  = self._img_data.get(img)
            cs = j_idx * nr
            # NOTE: the transformation prior is added ONCE, on the normal path
            # below (see "H_rr += d['C_r_prior_inv']" after XCsX).  bp3m adds it
            # here only inside its mutually-exclusive `if dropped:` branch; a
            # copy of that add at the top of this loop double-counted the prior
            # for every image, making it sqrt(2) too tight.  That is invisible
            # in star-rich fields (the prior term is negligible next to XCsX)
            # but dominates Gaia-sparse ones like COSMOS, where an image has
            # only 2-3 alignment stars and its transform is prior-driven: the
            # over-stiff prior forces WCS error into the stellar PMs instead.
            if d is None:
                K_img[img] = None
                continue

            use_align = np.asarray(d['use_for_fit'], dtype=bool)
            use_any   = np.asarray(d.get('use_for_astrom', use_align), dtype=bool) | use_align
            if self.exclude_2p_from_alignment:
                # 2p stars keep their astrometry (use_any) but leave the
                # transformation equations.
                use_align = use_align & ~self.gaia_2p[d['sidx']]
            sidx_any  = d['sidx'][use_any]
            sidx_aln  = d['sidx'][use_align]

            r_j    = r_current[cs:cs+nr]
            JU     = d['JU']      # (n, 2, 5)
            X      = d['X_mat']   # (n, 2, 6)
            xys    = d['xys']     # (n, 2)

            Cs     = self._compute_Cs(img, r_j)     # (n, 2, 2)
            Cs_inv = np.linalg.inv(Cs)              # (n, 2, 2)

            x_pred  = np.einsum('nij,j->ni', X, r_j)    # (n, 2)
            x_resid = xys - x_pred                       # (n, 2)

            JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)  # (n, 5, 2)

            # H_vv / h_all: stellar astrometry contributions
            np.add.at(H_vv, sidx_any,
                      np.einsum('nik,nkj->nij', JUT_Cs[use_any], JU[use_any]))
            np.subtract.at(h_all, sidx_any,
                           np.einsum('nik,nk->ni', JUT_Cs[use_any], x_resid[use_any]))
            np.subtract.at(h_align, sidx_aln,
                           np.einsum('nik,nk->ni', JUT_Cs[use_align], x_resid[use_align]))

            K = np.einsum('nik,nkl->nil', JUT_Cs, X)     # (n, 5, 6)
            K_img[img] = K

            XCsX = np.einsum('nki,nkl,nlj->ij', X[use_align], Cs_inv[use_align], X[use_align])
            H_rr[cs:cs+nr, cs:cs+nr] += XCsX
            XCs_xresid[img] = np.einsum('nki,nkl,nl->ni',
                                        X[use_align], Cs_inv[use_align], x_resid[use_align])
            H_rr[cs:cs+nr, cs:cs+nr] += d['C_r_prior_inv']

        # ── H_vv inversion → C_vT ─────────────────────────────────────────
        C_vT    = np.linalg.inv(H_vv)                              # (n_stars, 5, 5)
        a_align = np.einsum('nij,nj->ni', C_vT, h_align)
        a_all   = np.einsum('nij,nj->ni', C_vT, h_all)

        # ── Schur complement ─────────────────────────────────────────────────
        Cr_inv = H_rr.copy()
        rhs    = np.zeros(n_r)
        _schur_obs = []   # (sidx, K, cols) per image — star-major assembly

        for j_idx, img in enumerate(self.image_names):
            d  = self._img_data.get(img)
            cs = j_idx * nr
            rhs[cs:cs+nr] += (self._img_data[img]['C_r_prior_inv']
                              @ (self._img_data[img]['r_prior'] - r_current[cs:cs+nr]))
            if d is None or K_img[img] is None:
                continue
            # Must match the star set used for H_rr (XCsX), or the Schur
            # complement is inconsistent with the normal equations.
            use  = np.asarray(d['use_for_fit'], dtype=bool)
            if self.exclude_2p_from_alignment:
                use = use & ~self.gaia_2p[d['sidx']]
            sidx = d['sidx'][use]
            K    = K_img[img][use]

            rhs[cs:cs+nr] += XCs_xresid[img].sum(axis=0)
            rhs[cs:cs+nr] += np.einsum('nji,nj->i', K, a_align[sidx])

            # Star-major Schur assembly (ported from bp3m.BP3MSolver): the
            # diagonal K^T C_vT K and every cross-image block are together
            # B~^T B~ over Cholesky-whitened per-star row blocks — replaces
            # the O(n_img^2) intersect1d pair loop, mathematically identical.
            # (Also fixes the fork bug where the pair loop applied the 2p
            # exclusion to only one side of each cross block.)
            _schur_obs.append((sidx, K, np.arange(cs, cs + nr)))

        if _schur_obs:
            from scipy import sparse as _sp
            try:
                L_chol = np.linalg.cholesky(C_vT)
            except np.linalg.LinAlgError:
                w_e, Q_e = np.linalg.eigh(C_vT)
                L_chol = Q_e * np.sqrt(np.clip(w_e, 1e-30, None))[:, None, :]
            data_l, rows_l, cols_l = [], [], []
            for sidx_o, K_o, cols_o in _schur_obs:
                n_o = K_o.shape[0]
                if n_o == 0:
                    continue
                ltk = np.einsum('nji,njw->niw', L_chol[sidx_o], K_o)
                r_b = (5 * sidx_o.astype(np.int32))[:, None, None] \
                    + np.arange(5, dtype=np.int32)[None, :, None]
                data_l.append(ltk.ravel())
                rows_l.append(np.broadcast_to(r_b, (n_o, 5, nr))
                              .ravel().astype(np.int32, copy=False))
                cols_l.append(np.broadcast_to(
                    cols_o.astype(np.int32)[None, None, :],
                    (n_o, 5, nr)).ravel().astype(np.int32, copy=False))
            if data_l:
                B_til = _sp.csr_matrix(
                    (np.concatenate(data_l),
                     (np.concatenate(rows_l), np.concatenate(cols_l))),
                    shape=(5 * C_vT.shape[0], n_r))
                del data_l, rows_l, cols_l
                Cr_inv -= (B_til.T @ B_til).toarray()
                del B_til

        # ── Diagonal preconditioner + solve ─────────────────────────────────
        d_diag    = np.sqrt(np.maximum(np.abs(np.diag(Cr_inv)), 1e-30))
        d_inv     = 1.0 / d_diag
        Cr_inv_sc = d_inv[:, None] * Cr_inv * d_inv[None, :]
        if need_cov:
            try:
                C_r_sc = np.linalg.inv(Cr_inv_sc)
            except np.linalg.LinAlgError:
                C_r_sc = np.linalg.pinv(Cr_inv_sc)
            C_r     = d_inv[:, None] * C_r_sc * d_inv[None, :]
            delta_r = C_r @ rhs
        else:
            # Intermediate inner passes only need delta_r (ported from bp3m)
            C_r = None
            rhs_sc = d_inv * rhs
            try:
                from scipy import linalg as _sla
                c_fac = _sla.cho_factor(Cr_inv_sc, check_finite=False)
                delta_r = d_inv * _sla.cho_solve(c_fac, rhs_sc,
                                                 check_finite=False)
            except Exception:
                delta_r = d_inv * (np.linalg.pinv(Cr_inv_sc) @ rhs_sc)
        r_hat   = r_current + delta_r

        return r_hat, C_r, a_all, K_img, C_vT

    # ── Residual computation ─────────────────────────────────────────────────

    def compute_residuals(self, r_hat, v_hat, C_r=None, C_vT=None):
        """
        Compute per-star residuals in the tangent-plane (mas).

        Returns dict keyed by image name; each value has:
            xi_s, eta_s  : (n,) source positions (mas)
            resid_xi, resid_eta : (n,) residuals (mas)
            sigma_xi, sigma_eta : (n,) 1-sigma total uncertainty
            sigma_resid : (n,) 2D Mahalanobis distance
            sidx, use   : (n,) arrays
        """
        result = {}
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            d  = self._img_data.get(img)
            if d is None:
                continue
            cs   = j_idx * nr
            r_j  = r_hat[cs:cs+nr]
            sidx = d['sidx']
            use  = d['use_for_fit']

            X    = d['X_mat']    # (n, 2, 6)
            JU   = d['JU']       # (n, 2, 5)
            xys  = d['xys']      # (n, 2)

            pred  = (np.einsum('nij,j->ni', X, r_j)
                     - np.einsum('nij,nj->ni', JU, v_hat[sidx]))
            resid = xys - pred   # (n, 2)

            C_total = self._compute_Cs(img, r_j).copy()   # (n, 2, 2)
            if C_vT is not None:
                C_total += np.einsum('nik,nkl,njl->nij', JU, C_vT[sidx], JU)
            if C_r is not None:
                C_r_j = C_r[cs:cs+nr, cs:cs+nr]
                C_total += np.einsum('nik,kl,njl->nij', X, C_r_j, X)

            sx = np.sqrt(np.maximum(C_total[:, 0, 0], 0.))
            sy = np.sqrt(np.maximum(C_total[:, 1, 1], 0.))
            C_total_inv = np.linalg.inv(C_total)
            mah2        = np.einsum('ni,nij,nj->n', resid, C_total_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            xi_s  = d['X_mat'][:, 0, 0]
            eta_s = d['X_mat'][:, 0, 1]
            result[img] = {
                'xi_s':       xi_s,
                'eta_s':      eta_s,
                'resid_xi':   resid[:, 0],
                'resid_eta':  resid[:, 1],
                'sigma_xi':   sx,
                'sigma_eta':  sy,
                'sigma_resid': sigma_resid,
                'sidx':        sidx,
                'use':         use,
                # Full 2x2s, not just their diagonals: the off-diagonal is what
                # downstream re-weighting needs, and it is not recoverable from
                # sigma_xi/sigma_eta.  C_src is the measurement covariance in
                # force (post alpha-inflation); C_total adds the posterior and
                # alignment terms.  bp3m calls these C_hst and C_gdc_total.
                'C_src':      self._compute_Cs(img, r_j),
                'C_total':    C_total,
            }
        return result

    @staticmethod
    def _apply_prior_fallback(v_cov_full, v_mean, C_prior_arr, v_prior_arr,
                              failed_prior_test=None):
        """
        Replace the posterior with the Gaia prior where the fit made things worse.

        Ported from bp3m run_alignment._apply_prior_fallback.  A star falls back
        when any of:
          1. the full 5D posterior is less informative than the prior (logdet),
          2. the PM 2x2 block is less informative — caught separately because a
             position improvement can otherwise mask PM degradation from C_extra,
          3. the star failed the Gaia-prior chi2 test, so its posterior mean is
             inconsistent with the prior whatever the covariance size.

        NOTE this is applied on the CSV path ONLY, matching bp3m: the .npy
        arrays stay raw.  The two therefore disagree for fallback stars — that
        is deliberate, so consumers can choose.
        """
        v_cov_full = v_cov_full.copy()
        v_mean     = v_mean.copy()

        sign_post,  logdet_post  = np.linalg.slogdet(v_cov_full)
        sign_prior, logdet_prior = np.linalg.slogdet(C_prior_arr)
        use_prior = (sign_post > 0) & (sign_prior > 0) & (logdet_post > logdet_prior)

        sign_pm_post,  logdet_pm_post  = np.linalg.slogdet(v_cov_full[:, 2:4, 2:4])
        sign_pm_prior, logdet_pm_prior = np.linalg.slogdet(C_prior_arr[:, 2:4, 2:4])
        use_prior |= ((sign_pm_post > 0) & (sign_pm_prior > 0) &
                      (logdet_pm_post > logdet_pm_prior))

        if failed_prior_test is not None:
            use_prior |= failed_prior_test

        if use_prior.any():
            v_cov_full[use_prior] = C_prior_arr[use_prior]
            v_mean[use_prior]     = v_prior_arr[use_prior]
        return v_cov_full, v_mean, use_prior

    def compute_chi2_per_star(self, r_hat, v_hat, use_key='use_for_astrom'):
        """
        Per-star chi2 against the MEASUREMENT covariance alone.

        chi2_i = sum_{j: use[j]} resid_j @ C_src_j^-1 @ resid_j
        with resid_j = xys_j - (X_j @ r_k - JU_j @ v_hat_i)

        Mirrors bp3m run_alignment.compute_chi2_per_star.  Measurement-only, so
        it answers "how well does this star's astrometric solution reproduce its
        detections" without folding in alignment or posterior uncertainty.
        """
        chi2 = np.zeros(self.n_stars)
        n_det = np.zeros(self.n_stars, dtype=int)

        for j, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue
            use = np.asarray(d.get(use_key, d['use_for_fit']), dtype=bool)
            if not use.any():
                continue

            r_j   = r_hat[j * self.N_R:(j + 1) * self.N_R]
            sidx  = d['sidx'][use]
            resid = d['xys'][use] - (
                np.einsum('nkl,l->nk', d['X_mat'][use], r_j)
                - np.einsum('nij,nj->ni', d['JU'][use], v_hat[sidx]))
            C_inv = np.linalg.inv(d['C_src_orig'][use])
            np.add.at(chi2, sidx, np.einsum('ni,nij,nj->n', resid, C_inv, resid))
            np.add.at(n_det, sidx, 1)

        return chi2, n_det

    # ── Outlier rejection (_update_use_for_fit) ──────────────────────────────

    def _adapt_thresh(self, values, k, fallback, floor=0.0):
        """p50 + k*(p50-p16), floored at floor; fallback when <10 points."""
        if len(values) < 10:
            return float(max(fallback, floor)), float('nan'), float('nan'), float('nan')
        p16 = float(np.percentile(values, 16))
        p50 = float(np.median(values))
        p84 = float(np.percentile(values, 84))
        return float(max(p50 + k*max(p50-p16, 1e-6), floor)), p16, p50, p84

    # Option-5 health gate + backstop ceiling on the adaptive thresholds.
    #
    # _adapt_thresh returns p50 + k*(p50-p16), floored but previously UNCAPPED,
    # so a globally-inconsistent population defines its own "normal" and no star
    # looks like an outlier: COSMOS multi-band drove the df=5 threshold to 327
    # while stars sat 10-67 sigma from their Gaia priors.
    #
    # The gate encodes *why* adaptivity is licensed.  Adapting upward is
    # justified when the quoted uncertainties are mis-scaled -- but the alpha
    # inflation already measures and divides out exactly that scale factor.  A
    # centre still far above the theoretical median therefore means the MODEL is
    # wrong (e.g. uncorrected DCR), not the error model, and adapting to it is
    # unjustified rather than merely generous.  So when p50 exceeds
    # gate_mult * median(chi2_df), stop adapting and fall back to the floor.
    #
    # Calibration (418 bp3m v1 HST fields + LSST/Fornax, measured):
    #   healthy p50 is always BELOW theory -- HST 1.94, LSST i-band 1.87,
    #   Fornax 0.11, vs theory 4.35 (df=5).  No HST field exceeds p50=13.93.
    #   gate_mult=3.0 (13.05 at df=5) gates ~1/418 HST fields.
    #   ceiling_mult=6.0 (90.5 at df=5) is above the HST max adaptive of 83.6,
    #   so it binds in 0/418 -- a pure backstop for the in-between regime.
    GATE_MULT_DEFAULT    = 3.0
    CEILING_MULT_DEFAULT = 6.0

    @staticmethod
    def _gate_thresh(thresh, p50, df, floor,
                     gate_mult=GATE_MULT_DEFAULT,
                     ceiling_mult=CEILING_MULT_DEFAULT):
        """
        Apply the health gate and backstop ceiling to one adaptive threshold.

        Returns (thresh, gated, capped).  `gated` means the population was judged
        inconsistent and the threshold was pulled back to the floor; `capped`
        means only the backstop ceiling bound.  p50 may be NaN (fewer than 10
        reference points), in which case nothing is applied.
        """
        if not np.isfinite(p50):
            return float(thresh), False, False
        if gate_mult is not None and p50 > gate_mult * float(chi2_dist.median(df=df)):
            # Cap at the ceiling rather than collapsing to the floor — the
            # floor gutted crowded fields where inflated chi2 medians are
            # endemic (same change as bp3m/solver._gate_thresh).
            ceil = (ceiling_mult if ceiling_mult is not None else 6.0) * float(floor)
            return float(min(thresh, ceil)), True, False
        if ceiling_mult is not None:
            ceil = ceiling_mult * float(floor)
            if thresh > ceil:
                return float(ceil), False, True
        return float(thresh), False, False

    def _star_level_tests(self, v_hat, C_vT, adaptive_k=5.0, adaptive_delta=0.1,
                          ok_star_prev=None, chi2_pval=0.95, observed=None,
                          thresh_gate_mult=GATE_MULT_DEFAULT,
                          thresh_ceiling_mult=CEILING_MULT_DEFAULT,
                          gate_active=True):
        """
        Tests 1 and 2 — the star-level (global) rejection tests.

        Test 1: Gaia-prior chi2, chi2_g = dv^T (C_vT + C_prior)^-1 dv, against an
                adaptive threshold p50 + k*(p50-p16) floored at chi2(0.99).
        Test 2: diffuse-prior chi2 against a fixed threshold.

        Split out of _update_use_for_fit so the same code can be re-run on the
        FINAL a_arr after the EM loop.  These two tests are pure functions of the
        stellar solution and the priors — they touch no per-image mask, no alpha,
        and nothing the fit consumes — so re-running them cannot feed back into
        the solve.  (Test 3 and alpha are deliberately NOT here: re-running those
        on a final a_arr would move the masks out of step with the solve that
        produced it.)

        Sets self.sigma_from_gaia_prior as a side effect, since that column must
        always describe whichever solution the caller is about to report.

        Returns (ok_star, ok_gaia, ok_diffuse, chi2_g, diag).
        """
        if observed is None:
            observed = self.gaia_n_obs_used > 0
        # See the gate note in _update_use_for_fit: the gate is only meaningful
        # once alpha has removed the error-model scale.  gate_active=False makes
        # both the gate and the ceiling no-ops.
        if not gate_active:
            thresh_gate_mult = thresh_ceiling_mult = None

        floor_5 = float(chi2_dist.ppf(0.99, df=5))
        floor_2 = float(chi2_dist.ppf(0.99, df=2))

        # ── Test 1: Gaia prior chi2 ──────────────────────────────────────────
        delta_g  = v_hat - self.v_prior
        C_comb   = C_vT + self.C_prior
        C_c_inv  = np.linalg.inv(C_comb)
        chi2_g   = np.einsum('ni,nij,nj->n', delta_g, C_c_inv, delta_g)

        obs_5p = observed & ~self.gaia_2p
        obs_2p = observed & self.gaia_2p
        th5, p16_5, p50_5, p84_5 = self._adapt_thresh(
            chi2_g[obs_5p], adaptive_k, chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
        th2, p16_2, p50_2, p84_2 = self._adapt_thresh(
            chi2_g[obs_2p], adaptive_k, chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
        th5, gated_5, capped_5 = self._gate_thresh(
            th5, p50_5, 5, floor_5, thresh_gate_mult, thresh_ceiling_mult)
        th2, gated_2, capped_2 = self._gate_thresh(
            th2, p50_2, 2, floor_2, thresh_gate_mult, thresh_ceiling_mult)

        ok_gaia_admit = np.where(self.gaia_2p, chi2_g < th2, chi2_g < th5)

        if ok_star_prev is not None and adaptive_delta > 0:
            th5_out, *_ = self._adapt_thresh(
                chi2_g[obs_5p], adaptive_k+adaptive_delta,
                chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
            th2_out, *_ = self._adapt_thresh(
                chi2_g[obs_2p], adaptive_k+adaptive_delta,
                chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
            # Gate the expulsion thresholds with the SAME p50, so the hysteresis
            # dead-band collapses to the floor together with admission rather
            # than leaving a one-sided ratchet.
            th5_out, *_g = self._gate_thresh(
                th5_out, p50_5, 5, floor_5, thresh_gate_mult, thresh_ceiling_mult)
            th2_out, *_g = self._gate_thresh(
                th2_out, p50_2, 2, floor_2, thresh_gate_mult, thresh_ceiling_mult)
            ok_gaia_retain = np.where(self.gaia_2p, chi2_g < th2_out, chi2_g < th5_out)
            ok_gaia = np.where(ok_star_prev, ok_gaia_retain, ok_gaia_admit)
        else:
            ok_gaia = ok_gaia_admit
            th5_out = th5

        # ── Test 2: Diffuse prior chi2 (fixed threshold) ─────────────────────
        chi2_diff   = np.sum((v_hat / self._sigma_diff_per_star)**2, axis=1)
        thresh_diff = float(chi2_dist.ppf(chi2_dist.cdf(4.0, df=1), df=5))  # ≈11.1
        ok_diffuse  = chi2_diff < thresh_diff

        ok_star = ok_gaia & ok_diffuse
        self.sigma_from_gaia_prior[:] = np.sqrt(np.maximum(chi2_g, 0.))

        diag = dict(th5=th5, th2=th2, th5_out=th5_out, thresh_diff=thresh_diff,
                    gated_5=gated_5, gated_2=gated_2,
                    capped_5=capped_5, capped_2=capped_2,
                    p16_5=p16_5, p50_5=p50_5, p84_5=p84_5,
                    p16_2=p16_2, p50_2=p50_2, p84_2=p84_2,
                    obs_5p=obs_5p, obs_2p=obs_2p, observed=observed)
        return ok_star, ok_gaia, ok_diffuse, chi2_g, diag

    def _update_use_for_fit(self, r_hat, v_hat, C_r, C_vT, clip_sigma,
                            chi2_pval=0.95, iteration=0,
                            adaptive_k=5.0, adaptive_delta=0.1,
                            ok_star_prev=None,
                            skip_star_tests=False,
                            chi2_threshold=None,
                            inflate_errors=True,
                            inflate_from_iter=3,
                            inflate_alpha_max=3.0,
                            alpha_scale_chi2=False,
                            thresh_gate_mult=GATE_MULT_DEFAULT,
                            thresh_ceiling_mult=CEILING_MULT_DEFAULT):
        """
        Three-test outlier rejection + alpha inflation mirroring bp3m._update_use_for_fit.

        Test 1: Gaia prior chi2  (star-level, global)
        Test 2: Diffuse prior chi2 (star-level, global)
        Test 3: Per-image position residual chi2 (image-level)
        Alpha:  Inflate C_src when median chi2 > expected (starting at inflate_from_iter)
        """
        observed = self.gaia_n_obs_used > 0

        # Test 3 below still needs the df=2 chi2 floor.
        floor_2 = float(chi2_dist.ppf(0.99, df=2))

        # The gate's premise is that alpha has ALREADY divided the error-model
        # scale out of the residuals, so remaining excess is model error.  That
        # premise fails in Phase 0 (one un-converged solve, no alpha) and before
        # inflate_from_iter (alpha not yet applied) — gating there rejects almost
        # everything (Leo_I pre-filter collapsed to 53/15261).  Leave the
        # adaptive threshold free in those regimes.
        _gate_on = (not skip_star_tests) and bool(inflate_errors) \
                   and int(iteration) >= int(inflate_from_iter)
        _gm = thresh_gate_mult    if _gate_on else None
        _cm = thresh_ceiling_mult if _gate_on else None

        ok_star, ok_gaia, ok_diffuse, chi2_g, _d = self._star_level_tests(
            v_hat, C_vT, adaptive_k=adaptive_k, adaptive_delta=adaptive_delta,
            ok_star_prev=ok_star_prev, chi2_pval=chi2_pval, observed=observed,
            thresh_gate_mult=thresh_gate_mult,
            thresh_ceiling_mult=thresh_ceiling_mult,
            gate_active=_gate_on)
        th5, th2, th5_out, thresh_diff = _d['th5'], _d['th2'], _d['th5_out'], _d['thresh_diff']
        p16_5, p50_5, p84_5 = _d['p16_5'], _d['p50_5'], _d['p84_5']
        p16_2, p50_2, p84_2 = _d['p16_2'], _d['p50_2'], _d['p84_2']
        obs_5p, obs_2p = _d['obs_5p'], _d['obs_2p']

        if skip_star_tests:
            ok_star = np.ones(self.n_stars, dtype=bool)
        else:
            n_obs       = int(observed.sum())
            n_fail_gaia = int((~ok_gaia & observed).sum())
            n_fail_diff = int((~ok_diffuse & ok_gaia & observed).sum())
            hyst = f"→{th5_out:.2f}" if ok_star_prev is not None and adaptive_delta > 0 else ""
            def _pct_str(p16, p50, p84, n):
                if np.isnan(p16):
                    return f"(n={n}, <10)"
                return f"[{p16:.1f},{p50:.1f},{p84:.1f}] (n={n})"
            print(f"    thresh  5p+6p:{th5:.2f}{hyst} {_pct_str(p16_5,p50_5,p84_5,int(obs_5p.sum()))}  "
                  f"df=2:{th2:.2f} {_pct_str(p16_2,p50_2,p84_2,int(obs_2p.sum()))}  "
                  f"diffuse:{thresh_diff:.1f}")
            _gs = []
            if _d.get('gated_5'):  _gs.append('5p GATED->ceiling')
            if _d.get('gated_2'):  _gs.append('2p GATED->ceiling')
            if _d.get('capped_5'): _gs.append('5p capped')
            if _d.get('capped_2'): _gs.append('2p capped')
            print(f"    chi2 outliers (of {n_obs} observed): "
                  f"{n_fail_gaia} Gaia-incompatible, {n_fail_diff} diffuse"
                  + (f"   [{'; '.join(_gs)}]" if _gs else ""))
            if _d.get('gated_5') or _d.get('gated_2'):
                print(f"    WARNING population inconsistent with its own errors "
                      f"(p50={_d['p50_5']:.2f} > {thresh_gate_mult}x theory) — "
                      f"adaptive threshold capped at ceiling")

        # ── Test 3: Per-image position chi2 + alpha inflation ────────────────
        # Use HST-only chi2 (no C_r, no C_vT) for the threshold test —
        # identical to bp3m which uses resid_hst for this purpose.
        resid_hst = self.compute_residuals(r_hat, v_hat)

        _MED_CHI2_2 = 2.0 * np.log(2.0)
        self.gaia_n_obs_used[:] = 0
        info = []
        n_use_changed = 0
        n_thresh_gated = 0

        for img, rd in resid_hst.items():
            sidx   = rd['sidx']
            sig_sq = rd['sigma_resid']**2

            ok_glob      = ok_star[sidx]
            init_trusted = np.asarray(self._img_data[img]['use_for_align_init'])
            prev_use = np.asarray(self._img_data[img]['use_for_fit'])

            # (b) Optionally rescale chi2 by the previous alpha so the threshold
            # is uniform across images in units of "sigma given image noise"
            # (bp3m alpha_scale_chi2).  alpha_prev is taken from the previous
            # use_for_fit to avoid a chicken-and-egg dependency, and only from
            # iteration 3 so early passes remove gross outliers first.
            # Restricted to informative-prior stars for the same reason the main
            # alpha estimate is — see the alpha block below.
            _alpha_ref = prev_use & ~self.needs_diffuse[sidx]
            if alpha_scale_chi2 and iteration >= 3 and _alpha_ref.sum() >= 4:
                alpha_prev_s = float(max(1.0, np.sqrt(
                    np.median(sig_sq[_alpha_ref]) / _MED_CHI2_2)))
                sig_sq_eff = sig_sq / alpha_prev_s**2
            else:
                sig_sq_eff = sig_sq

            # (a) Threshold reference set.  Beyond bp3m's init_trusted cut (which
            # keeps non-stars, whose inflated PSF covariances give artificially
            # SMALL sigma_resid, from dragging the threshold down), exclude
            # diffuse-prior stars for the same reason: the fit absorbs their
            # residual, so their chi2 ~1e-7 collapses both p50 and p16 and the
            # adaptive term p50+k*(p50-p16) degenerates to the floor.  With 55%
            # 2p on LSST that turned the adaptive threshold into a fixed 9.21.
            ok_ref = ok_glob & init_trusted & ~self.needs_diffuse[sidx]
            if ok_ref.sum() < 10:
                ok_ref = ok_glob & init_trusted
            if ok_ref.sum() < 10:
                ok_ref = ok_glob

            if chi2_threshold is not None:
                thresh_admit = float(chi2_threshold)
                thresh_expel = thresh_admit * (1.0 + adaptive_delta / adaptive_k)
            else:
                thresh_admit, _p16_3, _p50_3, _ = self._adapt_thresh(
                    sig_sq_eff[ok_ref], adaptive_k, chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
                thresh_expel, *_ = self._adapt_thresh(
                    sig_sq_eff[ok_ref], adaptive_k+adaptive_delta,
                    chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
                # Alpha carve-out.  sig_sq_eff is already divided by alpha^2, so
                # after a successful inflation its centre should sit near the
                # theoretical median and the gate is meaningful.  But alpha is
                # capped at inflate_alpha_max: once it saturates, the true
                # required inflation is larger than what was applied and the
                # residual centre is LEGITIMATELY high by up to (needed/cap)^2.
                # Gating there would reject exactly the images that most need
                # inflation, so the gate is skipped for saturated images and only
                # the backstop ceiling is kept.
                _am = self._img_data[img].get('alpha_max', np.nan)
                _aa = self._img_data[img].get('alpha_applied', 1.0)
                _alpha_sat = bool(np.isfinite(_am) and _aa >= _am - 1e-9)
                _gm3 = None if _alpha_sat else _gm
                thresh_admit, _g3, _c3 = self._gate_thresh(
                    thresh_admit, _p50_3, 2, floor_2, _gm3, _cm)
                thresh_expel, *_ = self._gate_thresh(
                    thresh_expel, _p50_3, 2, floor_2, _gm3, _cm)
                if _g3:
                    n_thresh_gated += 1
                self._img_data[img]['thresh_gated'] = bool(_g3)
                self._img_data[img]['alpha_saturated_at_test3'] = _alpha_sat

            ok_resid_admit = sig_sq_eff < thresh_admit
            if adaptive_delta > 0:
                ok_resid = np.where(prev_use, sig_sq_eff < thresh_expel, ok_resid_admit)
            else:
                ok_resid = ok_resid_admit

            new_use = ok_resid & ok_glob
            new_use = np.asarray(new_use, dtype=bool) & np.asarray(
                self._img_data[img]['use_for_fit_max'])
            infl_excl = self._img_data[img].get('influence_excl')
            if infl_excl is not None:
                new_use &= ~infl_excl

            # Astrometry mask follows use_for_fit for initially-aligned stars;
            # any other row keeps whatever an external caller set (bp3m parity).
            _astrom = np.asarray(
                self._img_data[img].get('use_for_astrom', new_use), dtype=bool).copy()
            _astrom[init_trusted] = new_use[init_trusted]
            self._img_data[img]['use_for_astrom'] = _astrom
            align_init  = np.asarray(init_trusted, dtype=bool)
            current_fit = np.asarray(prev_use, dtype=bool)
            new_use &= (align_init | current_fit)

            n_use_changed += int(np.sum(current_fit != new_use))
            self._img_data[img]['use_for_fit'] = new_use
            self.gaia_n_obs_used[sidx[new_use]] += 1

            # Alpha: estimate noise inflation from the residual chi2 of accepted
            # stars that have an INFORMATIVE position prior.
            #
            # Diffuse-prior stars (Gaia 2p) must be excluded.  Their 5 stellar
            # parameters are unconstrained in position, so the fit absorbs their
            # residual completely — median |resid| ~0.004 mas, chi2 ~1e-7 — and
            # they carry no information about whether the measurement errors are
            # underestimated.  Where they are the majority (49% of LSST matches,
            # 11% of CFHT) they own the median and drive alpha_raw to ~0.001,
            # silently disabling inflation.  Measured on LSST det_150: all stars
            # -> alpha_raw=0.0010, informative-prior stars only -> 0.9862.
            #
            # bp3m computes this over all accepted stars, but its HST fields are
            # 5p-dominated so its median already falls on informative stars.
            # This preserves that intent rather than the literal expression.
            alpha_pop = new_use & ~self.needs_diffuse[sidx]
            if alpha_pop.sum() >= 4:
                chi2_in   = sig_sq[alpha_pop]
                alpha_raw = float(np.sqrt(np.median(chi2_in) / _MED_CHI2_2))
                alpha_raw = min(alpha_raw, inflate_alpha_max)
            else:
                alpha_raw = 1.0

            # Persist the measured factor and the cap so save_results can report
            # them; alpha_applied alone cannot distinguish "residuals were fine"
            # from "inflation hit the ceiling and the errors are understated".
            self._img_data[img]['alpha_raw'] = alpha_raw
            self._img_data[img]['n_alpha_ref'] = int(alpha_pop.sum())
            self._img_data[img]['alpha_max'] = float(inflate_alpha_max)

            # Apply alpha inflation to C_src (mirrors bp3m inflate_hst_errors)
            if inflate_errors and iteration >= inflate_from_iter:
                alpha_prev = self._img_data[img].get('alpha_applied', 1.0)
                # Cumulative cap (mirrors bp3m solver.py): per-step alpha_raw
                # is capped above, but the product compounds across iterations
                # without this clamp.
                alpha_j    = float(min(max(1.0, alpha_prev * alpha_raw),
                                       inflate_alpha_max))
                self._img_data[img]['alpha_applied'] = alpha_j
                self._img_data[img]['C_src'] = (
                    alpha_j**2 * self._img_data[img]['C_src_orig'])
            else:
                alpha_j = self._img_data[img].get('alpha_applied', 1.0)

            alpha_tag = " [α-inflated]" if inflate_errors and iteration >= inflate_from_iter else ""
            info.append((img, int(new_use.sum()), len(new_use), alpha_j, alpha_raw, 0))

        return info, ok_star, n_use_changed

    # ── Cook's D influence clipping (test 4) ─────────────────────────────────

    def _apply_influence_clip(self, r_hat, C_r, a_arr,
                               k_sigma_resid=5.0, k_scaled_d=5.0,
                               floor_sigma_resid=None, floor_scaled_d=3.0):
        """Mirrors bp3m._apply_influence_clip; ratchet semantics."""
        if floor_sigma_resid is None:
            floor_sigma_resid = float(np.sqrt(chi2_dist.ppf(0.99, df=2)))

        nr = self.N_R
        per_img = {}
        all_sr, all_sd = [], []

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None or d['use_for_fit'].sum() < 4:
                continue

            if 'influence_excl' not in d:
                d['influence_excl'] = np.zeros(d['n'], dtype=bool)
            already_excl = d['influence_excl']
            use      = np.asarray(d['use_for_fit'])
            eligible = use & ~already_excl

            cs    = j_idx * nr
            r_j   = r_hat[cs:cs+nr]
            C_r_j = C_r[cs:cs+nr, cs:cs+nr]
            sidx  = d['sidx']
            X     = d['X_mat']
            JU    = d['JU']
            xys   = d['xys']

            Cs     = self._compute_Cs(img, r_j)
            Cs_inv = np.linalg.inv(Cs)

            pred  = (np.einsum('nij,j->ni', X, r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))
            resid = xys - pred
            mah2  = np.einsum('ni,nij,nj->n', resid, Cs_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            CsR    = np.einsum('nij,nj->ni', Cs_inv, resid)
            XtCsR  = np.einsum('nij,ni->nj', X, CsR)
            delta_r = XtCsR @ C_r_j
            cooks_d = np.sum(XtCsR * delta_r, axis=1) / nr

            XCrX    = np.einsum('nik,kl,njl->nij', X, C_r_j, X)
            leverage = np.einsum('nij,nji->n', Cs_inv, XCrX)

            safe_lev = np.where(leverage > 1e-12, leverage, np.inf)
            test_d   = cooks_d * nr / safe_lev

            per_img[img] = (use, already_excl, eligible, sidx, sigma_resid, test_d)
            if eligible.any():
                all_sr.extend(sigma_resid[eligible].tolist())
                all_sd.extend(test_d[eligible].tolist())

        def _adapt(vals, k, floor):
            arr = np.array(vals)
            if len(arr) < 10:
                return float(floor)
            p16 = float(np.percentile(arr, 16))
            p50 = float(np.median(arr))
            return float(max(p50 + k*max(p50-p16, 1e-6), floor))

        thresh_sr = _adapt(all_sr, k_sigma_resid, floor_sigma_resid)
        thresh_sd = _adapt(all_sd, k_scaled_d, floor_scaled_d)

        n_new = 0
        for img, (use, already_excl, eligible, sidx, sigma_resid, test_d) in per_img.items():
            d = self._img_data[img]
            new_flag = use & ~already_excl & (sigma_resid > thresh_sr) & (test_d > thresh_sd)
            if new_flag.any() and (use & ~new_flag).sum() >= 4:
                d['influence_excl'] = already_excl | new_flag
                d['use_for_fit']    = use & ~new_flag
                n_new += int(new_flag.sum())

        return n_new, thresh_sr, thresh_sd

    # ── Public fit interface ─────────────────────────────────────────────────

    def fit(self, n_iter=20, tol=1e-4, clip_sigma=4.5,
            inflate_errors=True,
            inflate_from_iter=3, inflate_alpha_max=3.0,
            alpha_scale_chi2=False,
            min_outer_iters=None,
            prefilter=True, use_influence_clip=True,
            influence_k=5.0, floor_scaled_d=3.0,
            adaptive_delta=0.1,
            mask_tol_frac=1e-3, mask_tol_iters=3,
            verbose=True):
        """
        Phase 0/1/2 EM loop (mirrors bp3m.BP3MSolver.fit).

        Returns
        -------
        r_hat, C_r, v_hat, C_vT, a_arr, K_img
        """
        r_hat = np.concatenate([self._img_data[img]['r_init']
                                 for img in self.image_names])
        C_r   = None

        _pnames = ['A', 'B', 'C', 'D', 'dx', 'dy']
        nr = self.N_R
        def _inner_converge(r_hat, label):
            # Intermediate passes skip the explicit covariance (cho_solve,
            # need_cov=False); one covariance-bearing pass at the converged
            # point returns consistent (r, C_r, a, K, C_vT). Inner cap 100 +
            # divergence backtracking ported from bp3m.
            delta_prev = np.inf
            for it in range(100):
                r_new, _, a_i, K_i, CvT_i = self._solve_one_pass(
                    r_hat, need_cov=False)
                diff  = np.abs(r_new - r_hat)
                delta = float(np.max(diff))
                if delta > 1.5 * delta_prev and delta > tol:
                    r_new = r_hat + 0.5 * (r_new - r_hat)
                    diff  = np.abs(r_new - r_hat)
                    delta = float(np.max(diff))
                delta_prev = delta
                r_hat = r_new
                if verbose and it % 10 == 0:
                    print(f"  {label} step {it+1}: max|Δr|={delta:.3e}")
                if delta < tol:
                    if verbose:
                        print(f"  {label}: converged in {it+1} steps (max|Δr|={delta:.3e})")
                    return self._solve_one_pass(r_hat)
            if verbose:
                print(f"  {label}: WARNING max|Δr|={delta:.3e} (did not converge)")
            return self._solve_one_pass(r_hat)

        # min_outer: match bp3m — 4 when inflation is enabled, 2 otherwise
        _default_min = 4 if inflate_errors else 2
        min_outer = int(min_outer_iters) if min_outer_iters is not None else _default_min

        # ── Phase 0: pre-filter ───────────────────────────────────────────────
        if prefilter and clip_sigma is not None:
            if verbose:
                print(' Phase 0: pre-filter (one solve + position outlier rejection)')
            r_hat, C_r, a_arr, K_img, C_vT = self._solve_one_pass(r_hat)
            self._update_use_for_fit(r_hat, a_arr, C_r, C_vT, clip_sigma,
                                     ok_star_prev=None, skip_star_tests=True,
                                     inflate_errors=False)
            n_in  = sum(d['use_for_fit'].sum() for d in self._img_data.values() if d)
            n_tot = sum(d['n'] for d in self._img_data.values() if d)
            if verbose:
                print(f"  Pre-filter: {n_in}/{n_tot} stars accepted\n")

        # ── Phase 1: initial convergence ─────────────────────────────────────
        if verbose:
            print(' Phase 1: convergence with pre-filtered sample')
        r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(r_hat, 'init')

        # ── Phase 2: EM outlier rejection ────────────────────────────────────
        if clip_sigma is not None:
            if verbose:
                infl_str = (f"  inflate_errors=True (from iter {inflate_from_iter})"
                            if inflate_errors else "  inflate_errors=False")
                print(f'\n Phase 2: EM outlier rejection{infl_str}')
            ok_star_prev = np.ones(self.n_stars, dtype=bool)
            _n_consec_stable = 0
            _n_tol_stable = 0
            _n_det_tot = sum(d['n'] for d in self._img_data.values() if d)
            _mask_tol = int(max(1, round(mask_tol_frac * _n_det_tot)))

            for it_outer in range(n_iter):
                clip_info, ok_star_new, n_use_chg = self._update_use_for_fit(
                    r_hat, a_arr, C_r, C_vT, clip_sigma,
                    ok_star_prev=ok_star_prev, iteration=it_outer,
                    inflate_errors=inflate_errors,
                    inflate_from_iter=inflate_from_iter,
                    inflate_alpha_max=inflate_alpha_max,
                    alpha_scale_chi2=alpha_scale_chi2,
                    adaptive_delta=adaptive_delta)

                n_global_chg = int(np.sum(ok_star_prev != ok_star_new))
                n_total_chg  = n_global_chg + n_use_chg

                if n_global_chg == 0 and n_use_chg == 0:
                    _n_consec_stable += 1
                else:
                    _n_consec_stable = 0

                # Test 4: influence clip — only after min_outer and only when
                # tests 1-3 have been stable for < 2 consecutive iters
                n_inf = 0
                _t4_thresh_sr = _t4_thresh_sd = float('nan')
                if use_influence_clip and it_outer >= min_outer and _n_consec_stable < 2:
                    n_inf, _t4_thresh_sr, _t4_thresh_sd = self._apply_influence_clip(
                        r_hat, C_r, a_arr, k_sigma_resid=influence_k,
                        k_scaled_d=influence_k, floor_scaled_d=floor_scaled_d)
                    n_total_chg += n_inf

                if verbose:
                    t4_str = ""
                    if use_influence_clip:
                        t4_str = f", {n_inf} test-4"
                        if it_outer >= min_outer and _n_consec_stable < 2:
                            t4_str += (f"  [thresh_sr={_t4_thresh_sr:.2f}"
                                       f"  thresh_sd={_t4_thresh_sd:.2f}]")
                    print(f"\n  Outer iter {it_outer+1}: {n_global_chg} test-1/2, "
                          f"{n_use_chg} test-3{t4_str}  ({n_total_chg} total)")
                    for img, n_use, n_tot, alpha_j, alpha_raw, _ in clip_info:
                        inflated = inflate_errors and it_outer >= inflate_from_iter
                        if inflated:
                            alpha_str = f"α_applied={alpha_j:.3f}  α_raw={alpha_raw:.3f}"
                        else:
                            alpha_str = f"α={alpha_j:.3f}  α_raw={alpha_raw:.3f}"
                        print(f"    {img}: {n_use}/{n_tot}  {alpha_str}")

                ok_star_prev = ok_star_new.copy()

                # Stopping.  Tests 1-2 and 4 must be EXACTLY stable; test 3 is
                # allowed a small tolerance.  With N detections and a threshold
                # recomputed from the accepted set (and from alpha, which itself
                # depends on that set), a few borderline detections flip
                # forever.  Measured on Leo_I: tests 1-2 settle by iteration 4
                # and test 4 is always 0, but test 3 flickers 4-32 of ~11000
                # detections for all 50 iterations.  Each flip moves
                # Dalpha0/Ddelta0 by ~residual/N_stars ~ 10 uas -- the tangent
                # point is effectively the per-image residual mean, so it has
                # 1/N leverage that a/b/c/d do not (they move ~1e-8, i.e. 1e5x
                # less).  That shifts every residual and re-creates the flicker:
                # a self-sustaining limit cycle, not a transient.  The jump
                # plateaus (mean of last 10 / prev 10 = 1.02 over 50 iters) and
                # correlates with the flip count (Spearman +0.65).
                _stable_124 = (n_global_chg == 0 and n_inf == 0)
                if _stable_124 and n_use_chg == 0 and it_outer >= min_outer:
                    if verbose:
                        print("  Tests 1–4 stable — stopping.")
                    break
                if _stable_124 and it_outer >= min_outer and n_use_chg <= _mask_tol:
                    _n_tol_stable += 1
                    if _n_tol_stable >= mask_tol_iters:
                        if verbose:
                            print(f"  Tests 1-2/4 stable and test-3 flicker "
                                  f"{n_use_chg} <= {_mask_tol} for "
                                  f"{mask_tol_iters} iters — stopping.")
                        break
                else:
                    _n_tol_stable = 0

                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(r_hat, f'outer {it_outer+1}')

                # Absorb the fitted tangent-point offset and re-project there.
                _moved = self._relinearise_tangent_point(r_hat, a_arr)
                if verbose and _moved > 0:
                    print(f"    tangent point re-linearised (max |d| = {_moved:.3f} mas)")
            else:
                if verbose:
                    print(f"  Stopped after {n_iter} outer iterations "
                          f"(star set did not fully stabilise)")

            # ── Freeze the mask, then finalise ───────────────────────────────
            # Two things are wrong with whatever state the loop leaves behind:
            #   * the LAST action in the loop body is the tangent-point
            #     re-linearisation, which calls _precompute_geometry and rebuilds
            #     xi/eta -- so a_arr and C_vT are STALE with respect to the
            #     geometry they are about to be reported against;
            #   * on the exhaustion path the returned a_arr is one solve newer
            #     than anything the tests saw.
            # use_for_fit is frozen from here on (nothing below calls
            # _update_use_for_fit), so these solves are a small well-posed
            # problem: solve -> absorb the residual tangent-point offset ->
            # solve again lands on a true fixed point of the frozen mask rather
            # than an arbitrary point on the test-3 limit cycle.
            r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                r_hat, 'final (mask frozen)')
            _moved_f = self._relinearise_tangent_point(r_hat, a_arr)
            r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                r_hat, 'final (re-centred)')
            if verbose:
                print(f"  Finalised with mask frozen "
                      f"(last tangent-point absorb: {_moved_f:.5f} mas)")

            # ── Final star-level tests on the REPORTED a_arr ──────────────────
            # The loop tests a_arr at the top and re-solves at the bottom, so on
            # the exhaustion path the returned a_arr is one solve NEWER than the
            # last thing tested.  When the EM oscillates instead of converging
            # (LSST multi-band joints do: DCR makes the residuals irreconcilable),
            # the two land on opposite half-cycles, and ok_star /
            # sigma_from_gaia_prior / prior_fallback end up describing a solution
            # that is not the one written out — Gaia-incompatible stars sail
            # through with chi2_g ~ 0 while their reported PM is hundreds of
            # sigma off.  Re-run tests 1-2 on the final a_arr so those three
            # always describe the astrometry actually being reported.  Masks and
            # alpha are deliberately left alone (see _star_level_tests).
            # On the `break` path this is a no-op: a_arr was not re-solved, so it
            # recomputes the same numbers.
            ok_star_prev, _, _, _chi2_g_fin, _dfin = self._star_level_tests(
                a_arr, C_vT, ok_star_prev=ok_star_prev)
            if verbose:
                _obs = _dfin['observed']
                _nf  = int((~ok_star_prev & _obs).sum())
                print(f"  Final star tests on reported solution: {_nf}/{int(_obs.sum())} "
                      f"flagged (thresh 5p={_dfin['th5']:.2f}, "
                      f"max chi2={np.nanmax(_chi2_g_fin[_obs]) if _obs.any() else float('nan'):.2f})")

        v_hat = a_arr.copy()
        self.ok_star = ok_star_prev.copy() if clip_sigma is not None else np.ones(self.n_stars, bool)
        return r_hat, C_r, v_hat, C_vT, a_arr, K_img

    # ── Analytic marginalised posteriors ─────────────────────────────────────

    def compute_analytic_posteriors(self, a_arr, K_img, C_vT, C_r):
        """v_mean = a_arr; C_extra = C_vT_K @ C_r @ C_vT_K.T."""
        nr = self.N_R
        n_r   = nr * self.n_images
        K_all = np.zeros((self.n_stars, 5, n_r))
        for j_idx, img in enumerate(self.image_names):
            if K_img.get(img) is None:
                continue
            d    = self._img_data[img]
            use  = d['use_for_fit']
            sidx = d['sidx'][use]
            K    = K_img[img][use]
            cs   = j_idx * nr
            np.add.at(K_all[:, :, cs:cs+nr], sidx, K)
        CvT_K   = np.einsum('nij,njk->nik', C_vT, K_all)
        C_extra = CvT_K @ C_r @ np.swapaxes(CvT_K, -1, -2)
        return a_arr.copy(), C_extra

    # ── Save results ─────────────────────────────────────────────────────────

    def save_results(self, r_hat, C_r, v_hat, C_vT, v_mean, v_cov, output_dir):
        """Write bp3m-compatible output files to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        nr = self.N_R
        # 1. Transformation parameters
        rows = []
        for j_idx, img in enumerate(self.image_names):
            cs = j_idx * nr
            r_j = r_hat[cs:cs+nr]
            C_j = C_r[cs:cs+nr, cs:cs+nr]
            d   = self._img_data[img]
            A, B, C, D, dx, dy = r_j
            meta = self.image_meta[img]
            # Identity is whatever the instrument adapter put in the record
            # (expnum/ext for CFHT, visit/detector/band for LSST).  Writing it
            # generically avoids the old compat shim that synthesised a fake
            # `expnum` from `visit` and a STRING `ext` from `detector`, which
            # gave the two instruments different on-disk schemas.
            identity = {k: v for k, v in meta.items()
                        if k not in ('mjd', 'ra0', 'dec0', 'matched', 'name',
                                     'A0', 'B0', 'C0', 'D0', 'dx0', 'dy0',
                                     'ra0_orig', 'dec0_orig',
                                     'ra0_current', 'dec0_current',
                                     'ra0_final', 'dec0_final',
                                     '_relinearised', 'pixel_scale')}
            # Tangent point: always reported, as bp3m does.  r_j[4:6] are zero
            # after re-linearisation, so the total excursion lives here.
            _ra0_o  = meta.get('ra0_orig',  meta['ra0'])
            _dec0_o = meta.get('dec0_orig', meta['dec0'])
            _ra0_f  = meta.get('ra0_final',  meta.get('ra0_current',  meta['ra0']))
            _dec0_f = meta.get('dec0_final', meta.get('dec0_current', meta['dec0']))
            rows.append(dict(
                image_name   = img,
                **identity,
                mjd          = meta['mjd'],
                ra0          = _ra0_o,
                dec0         = _dec0_o,
                ra0_final    = _ra0_f,
                dec0_final   = _dec0_f,
                delta_ra0_mas  = (_ra0_f  - _ra0_o)  * 3_600_000.0,
                delta_dec0_mas = (_dec0_f - _dec0_o) * 3_600_000.0,
                n_stars_alignment = int(d['use_for_fit'].sum()),
                n_stars_astrometry_only = int(
                    (np.asarray(d.get('use_for_astrom', d['use_for_fit']), bool)
                     & ~np.asarray(d['use_for_fit'], bool)).sum()),
                # Position-uncertainty inflation.  alpha_applied is the factor
                # multiplying C_src_orig in the final fit; alpha_raw is the last
                # measured value sqrt(median(chi2)/E[chi2]).  saturated_alpha
                # means inflation hit inflate_alpha_max, i.e. the residuals are
                # worse than the solver was permitted to model and this image's
                # uncertainties are understated.
                # `alpha` is the name bp3m writes and hst_catalog_crossmatch.py
                # reads; alpha_applied is kept as the explicit synonym.
                alpha          = float(d.get('alpha_applied', 1.0)),
                alpha_applied  = float(d.get('alpha_applied', 1.0)),
                alpha_raw      = float(d.get('alpha_raw', np.nan)),
                n_alpha_ref    = int(d.get('n_alpha_ref', 0)),
                alpha_max      = float(d.get('alpha_max', np.nan)),
                saturated_alpha = bool(
                    np.isfinite(d.get('alpha_max', np.nan))
                    and d.get('alpha_applied', 1.0) >= d.get('alpha_max', np.inf) - 1e-9),
                A=A, B=B, C=C, D=D, dx=dx, dy=dy,
                scale        = float(np.sqrt(A*D - B*C)),
                pixel_scale_mas = float(np.sqrt(A*D - B*C)
                                        * meta.get('pixel_scale', np.nan)),
                rotation_deg = float(np.degrees(np.arctan2(B-C, A+D))),
                on_skew      = float((A-D)/2),
                off_skew     = float((B+C)/2),
                sigma_A  = float(np.sqrt(C_j[0,0])),
                sigma_B  = float(np.sqrt(C_j[1,1])),
                sigma_C  = float(np.sqrt(C_j[2,2])),
                sigma_D  = float(np.sqrt(C_j[3,3])),
                sigma_dx = float(np.sqrt(C_j[4,4])),
                sigma_dy = float(np.sqrt(C_j[5,5])),
                poly_order = self.poly_order,
                **{f'r_{k}': float(r_j[k]) for k in range(6, self.N_R)},
            ))
        pd.DataFrame(rows).to_csv(out / 'image_transformations.csv', index=False)

        # 2. Stellar astrometry
        g = self.gaia_df.copy()
        g['n_obs_used'] = self.gaia_n_obs_used
        # bp3m applies the prior fallback to the CSV values only; v_cov and
        # C_vT are written to .npy un-corrected.  prior_fallback flags which
        # stars were replaced so the difference is never silent.
        _failed_prior = ~np.asarray(
            getattr(self, 'ok_star', np.ones(self.n_stars, bool)), dtype=bool)
        v_cov_full, v_mean, _used_prior = self._apply_prior_fallback(
            v_cov + C_vT, v_mean, self.C_prior, self.v_prior,
            failed_prior_test=_failed_prior)
        g['prior_fallback'] = _used_prior

        # Per-star goodness of fit (bp3m chi2_hst / n_det_chi2 / chi2_hst_red;
        # the survey tag follows the *_xmatch convention used throughout).
        _chi2, _ndet = self.compute_chi2_per_star(r_hat, v_hat)
        g['chi2_xmatch']     = _chi2
        g['n_det_chi2']      = _ndet
        # np.divide(where=) rather than np.where(): the latter evaluates both
        # branches, so the n_det == 0 rows still divide by zero and emit a
        # RuntimeWarning for every detector processed.
        g['chi2_xmatch_red'] = np.divide(
            _chi2, 2 * _ndet, out=np.full(len(_chi2), np.nan, dtype=float),
            where=_ndet > 0)

        # Detections in the ALIGNMENT tier only (bp3m n_hst_alignment);
        # n_obs_used is the union tier (bp3m n_hst_used).
        _n_align = np.zeros(self.n_stars, dtype=int)
        for img in self.image_names:
            d = self._img_data.get(img)
            if d is not None:
                np.add.at(_n_align, d['sidx'][np.asarray(d['use_for_fit'], bool)], 1)
        g['n_xmatch_alignment'] = _n_align

        g['delta_racosdec']        = v_mean[:, 0]
        g['delta_dec']             = v_mean[:, 1]
        g['pmra_xmatch']             = v_mean[:, 2]
        g['pmdec_xmatch']            = v_mean[:, 3]
        g['parallax_xmatch']         = v_mean[:, 4]
        g['sigma_delta_racosdec']  = np.sqrt(np.maximum(v_cov_full[:, 0, 0], 0.))
        g['sigma_delta_dec']       = np.sqrt(np.maximum(v_cov_full[:, 1, 1], 0.))
        g['sigma_pmra_xmatch']       = np.sqrt(np.maximum(v_cov_full[:, 2, 2], 0.))
        g['sigma_pmdec_xmatch']      = np.sqrt(np.maximum(v_cov_full[:, 3, 3], 0.))
        g['sigma_parallax_xmatch']   = np.sqrt(np.maximum(v_cov_full[:, 4, 4], 0.))
        g['sigma_from_gaia_prior'] = self.sigma_from_gaia_prior
        g['ok_star']               = self.ok_star
        # Conditional (MAP alignment fixed)
        # Full off-diagonal structure of the MARGINALISED 5x5 posterior.  Names
        # and ordering follow bp3m run_alignment._save_results so downstream
        # consumers are interchangeable.  Marginalised is the unsuffixed
        # default; the conditional case carries the _cond suffix.
        _sig = np.sqrt(np.maximum(np.diagonal(v_cov_full, axis1=1, axis2=2), 0.))
        for _col, _i, _j in [
            ('corr_dra_ddec',    0, 1), ('corr_dra_pmra',    0, 2),
            ('corr_dra_pmdec',   0, 3), ('corr_dra_plx',     0, 4),
            ('corr_ddec_pmra',   1, 2), ('corr_ddec_pmdec',  1, 3),
            ('corr_ddec_plx',    1, 4), ('corr_pmra_pmdec',  2, 3),
            ('corr_pmra_plx',    2, 4), ('corr_pmdec_plx',   3, 4),
        ]:
            _den = _sig[:, _i] * _sig[:, _j]
            g[_col] = np.divide(
                v_cov_full[:, _i, _j], _den,
                out=np.full(len(_den), np.nan, dtype=float), where=_den > 0)

        # CONDITIONAL on the posterior-mean alignment (C_vT at r_hat).  bp3m
        # writes only the PM and parallax block here; the full conditional 5x5
        # is available in C_vT.npy.
        g['pmra_xmatch_cond']            = v_hat[:, 2]
        g['pmdec_xmatch_cond']           = v_hat[:, 3]
        g['parallax_xmatch_cond']        = v_hat[:, 4]
        g['sigma_pmra_xmatch_cond']      = np.sqrt(np.maximum(C_vT[:, 2, 2], 0.))
        g['sigma_pmdec_xmatch_cond']     = np.sqrt(np.maximum(C_vT[:, 3, 3], 0.))
        g['sigma_parallax_xmatch_cond']  = np.sqrt(np.maximum(C_vT[:, 4, 4], 0.))

        g.to_csv(out / 'stellar_astrometry.csv', index=False)

        # 3. Full covariance arrays
        np.save(out / 'v_cov_marginalised.npy', v_cov)
        np.save(out / 'C_vT.npy',               C_vT)
        np.save(out / 'C_r.npy',                C_r)

        # 4. Detection flags
        fit_d, idx_d = {}, {}
        for img in self.image_names:
            d = self._img_data.get(img)
            if d is None:
                continue
            fit_d[img] = d['use_for_fit']
            idx_d[img] = d['sidx']
        np.savez(out / 'use_for_fit.npz', **fit_d)
        np.savez(out / 'star_indices.npz', **idx_d)
        np.savez(out / 'use_for_astrom.npz',
                 **{img: self._img_data[img]['use_for_astrom']
                    for img in self.image_names if self._img_data.get(img)})

        # 5. Per-detection residuals
        det_data = {}
        resids = self.compute_residuals(r_hat, v_hat, C_r=C_r, C_vT=C_vT)
        for img, rd in resids.items():
            det_data[f'{img}_xi_s']        = rd['xi_s']
            det_data[f'{img}_eta_s']       = rd['eta_s']
            det_data[f'{img}_resid_xi']    = rd['resid_xi']
            det_data[f'{img}_resid_eta']   = rd['resid_eta']
            det_data[f'{img}_sigma_xi']    = rd['sigma_xi']
            det_data[f'{img}_sigma_eta']   = rd['sigma_eta']
            det_data[f'{img}_sigma_resid'] = rd['sigma_resid']
            det_data[f'{img}_sidx']        = rd['sidx']
            det_data[f'{img}_use']         = rd['use']
            det_data[f'{img}_use_for_astrom'] = np.asarray(
                self._img_data[img]['use_for_astrom'], dtype=bool)
            det_data[f'{img}_C_src']       = rd['C_src']      # (n,2,2) measurement
            det_data[f'{img}_C_total']     = rd['C_total']    # (n,2,2) full
        np.savez_compressed(out / 'detections.npz', **det_data)

        # 6. Gaia classification flags & priors (needed for plot recreation)
        gaia_flags = {
            'gaia_5p':               self.gaia_5p,
            'gaia_2p':               self.gaia_2p,
            'gaia_6p':               self.gaia_6p,
            'full_gaia_astrometry':  self.full_gaia_astrometry,
            'gaia_trustworthy':      self.gaia_trustworthy,
            'ok_star':               self.ok_star,
        }
        np.savez(out / 'gaia_flags.npz', **gaia_flags)

        # 7. Gaia survey PMs/parallax and covariance (for plots)
        np.save(out / 'v_survey.npy',  self.v_survey)
        np.save(out / 'C_survey.npy',  self.C_survey)

        # 8. Image metadata
        img_meta = {img: json.dumps(self.image_meta[img], default=str)
                    for img in self.image_names}
        with open(out / 'image_metadata.json', 'w') as f:
            json.dump(img_meta, f, indent=2)

        # 9. Run config
        cfg = {
            'poly_order':     self.poly_order,
            'n_r_per_image':  self.N_R,
            'poly_scale_mas': float(getattr(self, '_poly_scale', float('nan'))),
            'n_images':       self.n_images,
            'n_stars':        self.n_stars,
            'image_names':    self.image_names,
            'prior_hyperparams': {
                'sigma_rot_deg':   self._sigma_rot_deg,
                'sigma_scale':     self._sigma_scale,
                'sigma_skew':      self._sigma_skew,
                'sigma_pointing':  self._sigma_pointing,
            },
            'gaia_sys_dict': GAIA_SYS_DICT,
        }
        cfg['image_priors'] = {
            img: {'r_prior': self._img_data[img]['r_prior'].tolist(),
                  'C_r_prior_inv': self._img_data[img]['C_r_prior_inv'].tolist()}
            for img in self.image_names if self._img_data.get(img) is not None}
        with open(out / 'run_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

        print(f"  Saved results to {out}")
        return out

    # ── Diagnostic plots ─────────────────────────────────────────────────────

    def make_plots(self, r_hat, v_hat, v_mean, v_cov, C_vT, C_r, output_dir,
                   plot_residuals=True):
        """Generate all diagnostic plots matching bp3m output, in output_dir/plots/."""
        plot_dir = Path(output_dir) / 'plots'
        resid_plot_dir = plot_dir / 'residuals'
        plot_dir.mkdir(parents=True, exist_ok=True)
        resid_plot_dir.mkdir(parents=True, exist_ok=True)

        # ── Full marginal covariance (r-propagation + conditional) ───────────
        v_cov_full = v_cov + C_vT   # (n_stars, 5, 5)

        # ── Gaia catalog columns ──────────────────────────────────────────────
        g = self.gaia_df

        def _gc(col, default=np.nan):
            if col in g.columns:
                return pd.to_numeric(g[col], errors='coerce').to_numpy(float)
            return np.full(self.n_stars, default)

        gmag    = _gc('gmag', 20.0)
        ra      = _gc('ra',   0.0)
        dec     = _gc('dec',  0.0)
        bp_rp   = _gc('bp_rp', np.nan)

        # ── PM quantities ─────────────────────────────────────────────────────
        pmra_xmatch  = v_mean[:, 2]
        pmdec_xmatch = v_mean[:, 3]
        pmra_gaia  = self.v_survey[:, 2]
        pmdec_gaia = self.v_survey[:, 3]
        # Single definition of "has a Gaia prior" for every figure in this
        # method, on astrometric_params_solved.  Previously this was
        # full_gaia_astrometry (a finite-PM test) while the Figure 2 series used
        # astrometric_params_solved, so the two disagreed inside one method and
        # the same star could be 5p in one panel and 2p in the next.
        # full_gaia_astrometry is still used by _setup_gaia_priors for the
        # likelihood — that is a separate question and is deliberately unchanged.
        has_gaia   = ~self.gaia_2p               # bool (n_stars,)

        C_pm_gaia  = self.C_survey[:, 2:4, 2:4]
        sig_pmra_gaia  = np.sqrt(np.maximum(C_pm_gaia[:, 0, 0], 0.))
        sig_pmdec_gaia = np.sqrt(np.maximum(C_pm_gaia[:, 1, 1], 0.))
        rho_gaia       = (C_pm_gaia[:, 0, 1]
                          / np.where(sig_pmra_gaia * sig_pmdec_gaia > 0,
                                     sig_pmra_gaia * sig_pmdec_gaia, np.nan))
        sig_pm_gaia = _pm_geom_unc(sig_pmra_gaia, sig_pmdec_gaia, rho_gaia)

        C_pm_xmatch  = v_cov_full[:, 2:4, 2:4]
        det_xmatch   = np.linalg.det(C_pm_xmatch)
        sig_pm_xm = np.where(det_xmatch > 0, det_xmatch ** 0.25, np.nan)
        xm_converged = (sig_pm_xm < 90)
        sig_pmra_xmatch  = np.sqrt(np.maximum(C_pm_xmatch[:, 0, 0], 0.))
        sig_pmdec_xmatch = np.sqrt(np.maximum(C_pm_xmatch[:, 1, 1], 0.))
        rho_xmatch       = (C_pm_xmatch[:, 0, 1]
                          / np.where(sig_pmra_xmatch * sig_pmdec_xmatch > 0,
                                     sig_pmra_xmatch * sig_pmdec_xmatch, np.nan))

        # ── Figure 1: 1:1 PM comparison (top) + PM unc vs mag + improvement ──
        print("  Plotting 1:1 PM comparison + uncertainty vs magnitude...")

        fig = plt.figure(figsize=(13, 11 / 2 * 3), layout='constrained')
        gs  = fig.add_gridspec(3, 2)
        ax_pmra  = fig.add_subplot(gs[0, 0])
        ax_pmdec = fig.add_subplot(gs[0, 1])
        ax_unc   = fig.add_subplot(gs[1, :])
        ax_improve = fig.add_subplot(gs[2, :])

        _gaia_nq = has_gaia   # 5p stars (no QSO exclusion needed for xmatch)
        _gaia_only = _gaia_nq  # all 5p

        for ax, gaia_pm, xm_pm, sig_g, sig_b, comp in zip(
                [ax_pmra, ax_pmdec],
                [pmra_gaia, pmdec_gaia],
                [pmra_xmatch, pmdec_xmatch],
                [sig_pmra_gaia, sig_pmdec_gaia],
                [sig_pmra_xmatch, sig_pmdec_xmatch],
                [r'$\mu_{\alpha*}$', r'$\mu_\delta$']):
            if _gaia_only.any():
                ax.errorbar(gaia_pm[_gaia_only], xm_pm[_gaia_only],
                            xerr=sig_g[_gaia_only], yerr=sig_b[_gaia_only],
                            fmt='o', ms=3, lw=0.5, alpha=0.5, color='grey',
                            label='Gaia 5p', zorder=2)
            lim = _padded_lim(gaia_pm[_gaia_nq], xm_pm[_gaia_nq])
            ax.plot(lim, lim, 'k--', lw=1, zorder=4)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f'{comp} Gaia prior [mas/yr]')
            ax.set_ylabel(f'{comp} xmatch [mas/yr]')
            ax.set_title(f'{comp}: xmatch vs Gaia prior PM')
            ax.set_aspect('equal')
            ax.legend(fontsize=7, loc='upper left')
            _style_ax(ax)

        # Uncertainty vs magnitude panel.
        # Populations follow bp3m plot_results.make_plots, EXCEPT the z-order:
        # 5p (circles) is drawn above 2p (squares) everywhere the two share a
        # panel.  2p proper motions scatter much more widely, and on top they
        # bury the tight 5p cluster that the figure exists to show.  bp3m puts 2p
        # above; this is a deliberate, requested divergence.  Order here:
        # priors 2 < 2p 3 < 5p 4 < DELVE-only 5.
        #   Gaia prior       #444444        DELVE prior      dodgerblue  D
        #   Gaia only        mediumseagreen Gaia+DELVE       steelblue
        #   Gaia 2p          mediumpurple s DELVE only       darkorange  ^
        # DELVE-only stars are identified by a NEGATIVE source id, which is how
        # bp3m finds them (`_delve_only = (_gaia_ids < 0)`), and Gaia+DELVE by
        # solver._has_delve_pm -- the POST-VETO mask, not the raw catalogue
        # column, so a star whose DELVE prior was vetoed correctly plots as
        # Gaia-only.
        _sid = self.gaia_df['gaia_source_id'].to_numpy() \
            if 'gaia_source_id' in self.gaia_df.columns else np.ones(self.n_stars)
        _delve_only = np.asarray(_sid, dtype=np.int64) < 0
        _has_dpm = np.asarray(getattr(self, '_has_delve_pm',
                                      np.zeros(self.n_stars, bool)), dtype=bool)

        ax_unc.scatter(gmag[_gaia_nq], sig_pm_gaia[_gaia_nq],
                       s=6, alpha=0.7, color='#444444', label='Gaia prior', zorder=2)

        # The DELVE prior's own PM uncertainty, for comparison with the fit
        _d_sig = None
        if 'delve_pmra_error' in self.gaia_df.columns:
            _dra = pd.to_numeric(self.gaia_df['delve_pmra_error'],
                                 errors='coerce').to_numpy(float)
            _dde = pd.to_numeric(self.gaia_df.get('delve_pmdec_error'),
                                 errors='coerce').to_numpy(float)
            _d_sig = np.sqrt(np.maximum(_dra, 0) * np.maximum(_dde, 0))
            _m = np.isfinite(_d_sig) & (_d_sig > 0)
            if _m.any():
                ax_unc.scatter(gmag[_m], _d_sig[_m], s=6, alpha=0.6,
                               color='dodgerblue', marker='D',
                               label='DELVE prior', zorder=2)

        _xmatch_gaia_conv = xm_converged & has_gaia
        _gonly = _xmatch_gaia_conv & ~_has_dpm & ~_delve_only
        _gdlv  = _xmatch_gaia_conv & _has_dpm & ~_delve_only
        if _gonly.any():
            ax_unc.scatter(gmag[_gonly], sig_pm_xm[_gonly], s=6, alpha=0.85,
                           color='mediumseagreen', label='xmatch Gaia only',
                           zorder=4)
        if _gdlv.any():
            ax_unc.scatter(gmag[_gdlv], sig_pm_xm[_gdlv], s=6, alpha=0.7,
                           color='steelblue', label='xmatch Gaia+DELVE', zorder=4)
        _xmatch_2p_conv = xm_converged & (~has_gaia) & ~_delve_only
        if _xmatch_2p_conv.any():
            ax_unc.scatter(gmag[_xmatch_2p_conv], sig_pm_xm[_xmatch_2p_conv],
                           s=8, alpha=0.8, color='mediumpurple', marker='s',
                           label='xmatch Gaia 2p', zorder=3)
        _dlv_conv = xm_converged & _delve_only
        if _dlv_conv.any():
            ax_unc.scatter(gmag[_dlv_conv], sig_pm_xm[_dlv_conv],
                           s=10, alpha=0.8, color='darkorange', marker='^',
                           label='xmatch DELVE only', zorder=5)
        ax_unc.set_xlabel('G [mag]')
        ax_unc.set_ylabel(r'$(\det\,C_{\mu})^{1/4}$ [mas/yr]')
        ax_unc.set_title(r'Geometric-mean PM uncertainty $(\det\,C_{\mu})^{1/4}$ vs magnitude')
        ax_unc.legend()
        ax_unc.set_yscale('log')
        xlim = ax_unc.get_xlim()
        _style_ax(ax_unc)

        # Improvement factor panel
        if _gaia_only.any() and _xmatch_gaia_conv[_gaia_only].any():
            _imp_m = _gaia_only & _xmatch_gaia_conv
            ax_improve.scatter(gmag[_imp_m],
                               sig_pm_gaia[_imp_m] / sig_pm_xm[_imp_m],
                               s=6, alpha=0.6, color='grey', label='Gaia 5p', zorder=2)
        ax_improve.set_xlabel('G [mag]')
        ax_improve.set_ylabel('PM Improvement Factor')
        ax_improve.set_title('PM uncertainty improvement vs magnitude compared to Gaia prior')
        ax_improve.set_xlim(xlim)
        ax_improve.axhline(1.0, c='k', lw=2, ls='--', zorder=-1e10)
        ax_improve.legend(fontsize=7)
        _style_ax(ax_improve)

        fig.suptitle('Proper motion comparison', fontsize=13)
        _save(fig, plot_dir / 'pm_one_to_one.png')

        # ── Figure 2: PM vector diagrams coloured by geometric-mean uncertainty
        print("  Plotting PM vector diagrams...")

        # Separate 5p/6p (Gaia prior available) from 2p (survey-only PMs).
        # Classification is astrometric_params_solved throughout — the same
        # basis used by _setup_gaia_priors and by the Figure 2 series below.
        # Do NOT reintroduce a finite-PM test here: the two disagree, and
        # having both in one method is how the figures drifted apart.
        is_5p_6p = ~self.gaia_2p
        xm_5p_conv = xm_converged & is_5p_6p     # Gaia 5p/6p w/ posteriors
        xm_2p_conv = xm_converged & self.gaia_2p # Gaia 2p w/ posteriors

        gaia_pmra_h  = pmra_gaia[has_gaia]
        gaia_pmdec_h = pmdec_gaia[has_gaia]
        xm_pmra_5p = pmra_xmatch[xm_5p_conv]
        xm_pmdec_5p = pmdec_xmatch[xm_5p_conv]
        xm_pmra_2p = pmra_xmatch[xm_2p_conv]
        xm_pmdec_2p = pmdec_xmatch[xm_2p_conv]

        full_xlim = _padded_lim(gaia_pmra_h)
        full_ylim = _padded_lim(gaia_pmdec_h)
        zoom_xcen = np.nanmedian(gaia_pmra_h)
        zoom_ycen = np.nanmedian(gaia_pmdec_h)
        zoom_xhw  = max(np.abs(np.nanpercentile(gaia_pmra_h,  [16, 84]) - zoom_xcen))
        zoom_yhw  = max(np.abs(np.nanpercentile(gaia_pmdec_h, [16, 84]) - zoom_ycen))
        zoom_hw   = max(zoom_xhw, zoom_yhw) * 1.15
        zoom_xlim = (zoom_xcen - zoom_hw, zoom_xcen + zoom_hw)
        zoom_ylim = (zoom_ycen - zoom_hw, zoom_ycen + zoom_hw)

        c_gaia = sig_pm_gaia[has_gaia]
        c_xmatch_5p = sig_pm_xm[xm_5p_conv]
        c_xmatch_2p = sig_pm_xm[xm_2p_conv]
        all_unc = np.concatenate([c_gaia[np.isfinite(c_gaia)],
                                  c_xmatch_5p[np.isfinite(c_xmatch_5p)],
                                  c_xmatch_2p[np.isfinite(c_xmatch_2p)]])
        vmin_u = max(float(np.nanpercentile(all_unc, 2)), 1e-6)
        vmax_u = float(np.nanpercentile(all_unc, 98))
        norm_unc = mcolors.LogNorm(vmin=vmin_u, vmax=vmax_u)
        cmap_unc = 'plasma'

        fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout='constrained')
        sc_last = None
        for row, xlim, ylim, suffix in zip(
                [0, 1],
                [full_xlim, zoom_xlim],
                [full_ylim, zoom_ylim],
                ['full range', 'zoom (68% CI)']):
            # Column 0: Gaia
            ax = axes[row, 0]
            sc = ax.scatter(gaia_pmra_h, gaia_pmdec_h, c=c_gaia, s=6,
                            norm=norm_unc, cmap=cmap_unc, alpha=0.8, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
            ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
            ax.set_title(f'Gaia — {suffix}')
            ax.set_aspect('equal')
            _style_ax(ax)
            sc_last = sc

            # Column 1: xmatch (5p + 2p)
            ax = axes[row, 1]
            if xm_5p_conv.any():
                ax.scatter(xm_pmra_5p, xm_pmdec_5p, c=c_xmatch_5p, s=6,
                          norm=norm_unc, cmap=cmap_unc, alpha=0.8, zorder=3,
                          label='Gaia 5p')
            if xm_2p_conv.any():
                sc = ax.scatter(xm_pmra_2p, xm_pmdec_2p, c=c_xmatch_2p, s=8,
                               marker='s', norm=norm_unc, cmap=cmap_unc, alpha=0.65,
                               zorder=2, label='Gaia 2p')
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
            ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
            ax.set_title(f'xmatch — {suffix}')
            ax.set_aspect('equal')
            if row == 0:
                ax.legend(fontsize=7, loc='upper left')
            _style_ax(ax)
            sc_last = sc

        cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(r'$(\det\,C_{\mu})^{1/4}$ [mas/yr]')
        fig.suptitle('PM vector diagrams coloured by geometric-mean uncertainty', fontsize=13)
        _save(fig, plot_dir / 'pm_vector_diagram.png')

        # ── Figure 2b: PM vector diagrams with covariance error bars ─────────
        print("  Plotting PM vector diagrams with error bars...")

        C_pm_gaia_h = self.C_survey[has_gaia, 2:4, 2:4]
        C_pm_xmatch_5p_h = C_pm_xmatch[xm_5p_conv]
        C_pm_xmatch_2p_h = C_pm_xmatch[xm_2p_conv]

        fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout='constrained')
        sc_last = None
        for row, xlim, ylim, suffix in zip(
                [0, 1],
                [full_xlim, zoom_xlim],
                [full_ylim, zoom_ylim],
                ['full range', 'zoom (68% CI)']):
            # Column 0: Gaia with error bars
            ax = axes[row, 0]
            _pm_error_bars(ax, gaia_pmra_h, gaia_pmdec_h, C_pm_gaia_h)
            sc = ax.scatter(gaia_pmra_h, gaia_pmdec_h, c=c_gaia, s=6,
                            norm=norm_unc, cmap=cmap_unc, alpha=0.8, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
            ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
            ax.set_title(f'Gaia — {suffix}')
            ax.set_aspect('equal')
            _style_ax(ax)
            sc_last = sc

            # Column 1: xmatch (5p + 2p) with error bars
            ax = axes[row, 1]
            if xm_5p_conv.any():
                _pm_error_bars(ax, xm_pmra_5p, xm_pmdec_5p, C_pm_xmatch_5p_h)
                ax.scatter(xm_pmra_5p, xm_pmdec_5p, c=c_xmatch_5p, s=6,
                          norm=norm_unc, cmap=cmap_unc, alpha=0.8, zorder=3,
                          label='Gaia 5p')
            if xm_2p_conv.any():
                _pm_error_bars(ax, xm_pmra_2p, xm_pmdec_2p, C_pm_xmatch_2p_h)
                sc = ax.scatter(xm_pmra_2p, xm_pmdec_2p, c=c_xmatch_2p, s=8,
                               marker='s', norm=norm_unc, cmap=cmap_unc, alpha=0.65,
                               zorder=2, label='Gaia 2p')
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
            ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
            ax.set_title(f'xmatch — {suffix}')
            ax.set_aspect('equal')
            if row == 0:
                ax.legend(fontsize=7, loc='upper left')
            _style_ax(ax)
            sc_last = sc

        cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(r'$(\det\,C_{\mu})^{1/4}$ [mas/yr]')
        fig.suptitle(
            'PM vector diagrams with 1σ principal-axis error bars\n'
            r'(coloured by $(\det\,C_{\mu})^{1/4}$)', fontsize=13)
        _save(fig, plot_dir / 'pm_vector_diagram_errorbars.png')

        # ── Figure 2c: xmatch PM coloured by detector position ─────────────────
        print("  Plotting xmatch PM vector diagram coloured by detector position...")

        # Accumulate xi_s/eta_s per star (average across images for multi-visit)
        xi_sum  = np.zeros(self.n_stars)
        eta_sum = np.zeros(self.n_stars)
        det_cnt = np.zeros(self.n_stars, dtype=int)
        for img in self.image_names:
            rd = self._img_data.get(img)
            if rd is None:
                continue
            sidx_img = rd['sidx']
            xi_sum[sidx_img]  += rd['X_mat'][:, 0, 0]
            eta_sum[sidx_img] += rd['X_mat'][:, 0, 1]
            det_cnt[sidx_img] += 1

        obs_mask = det_cnt > 0
        xi_star_all  = np.where(obs_mask, xi_sum  / np.maximum(det_cnt, 1), np.nan)
        eta_star_all = np.where(obs_mask, eta_sum / np.maximum(det_cnt, 1), np.nan)

        # Accumulate xmatch PMs (both 5p and 2p)
        all_xmatch_pmra = np.concatenate([xm_pmra_5p, xm_pmra_2p]) if xm_5p_conv.any() or xm_2p_conv.any() else np.array([])
        all_xmatch_pmdec = np.concatenate([xm_pmdec_5p, xm_pmdec_2p]) if xm_5p_conv.any() or xm_2p_conv.any() else np.array([])

        xm_full_xlim = _padded_lim(all_xmatch_pmra) if len(all_xmatch_pmra) > 0 else (0, 1)
        xm_full_ylim = _padded_lim(all_xmatch_pmdec) if len(all_xmatch_pmdec) > 0 else (0, 1)
        _bx_cen = np.nanmedian(all_xmatch_pmra) if len(all_xmatch_pmra) > 0 else 0
        _by_cen = np.nanmedian(all_xmatch_pmdec) if len(all_xmatch_pmdec) > 0 else 0
        _bx_hw  = max(np.abs(np.nanpercentile(all_xmatch_pmra,  [16, 84]) - _bx_cen)) if len(all_xmatch_pmra) > 0 else 1
        _by_hw  = max(np.abs(np.nanpercentile(all_xmatch_pmdec, [16, 84]) - _by_cen)) if len(all_xmatch_pmdec) > 0 else 1
        _b_hw   = max(_bx_hw, _by_hw) * 1.15
        xm_zoom_xlim = (_bx_cen - _b_hw, _bx_cen + _b_hw)
        xm_zoom_ylim = (_by_cen - _b_hw, _by_cen + _b_hw)

        def _lin_norm(vals):
            fin = vals[np.isfinite(vals)]
            if len(fin) < 2:
                return mcolors.Normalize(vmin=0, vmax=1)
            vlo, vhi = np.nanpercentile(fin, [2, 98])
            return mcolors.Normalize(vmin=vlo, vmax=vhi)

        xi_star_5p  = xi_star_all[xm_5p_conv]
        eta_star_5p = eta_star_all[xm_5p_conv]
        xi_star_2p  = xi_star_all[xm_2p_conv]
        eta_star_2p = eta_star_all[xm_2p_conv]

        norm_xi  = _lin_norm(np.concatenate([xi_star_5p, xi_star_2p]) if xm_5p_conv.any() or xm_2p_conv.any() else np.array([]))
        norm_eta = _lin_norm(np.concatenate([eta_star_5p, eta_star_2p]) if xm_5p_conv.any() or xm_2p_conv.any() else np.array([]))

        fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout='constrained')
        sc_xi = sc_eta = None
        for row, xlim, ylim, row_label in zip(
                [0, 1],
                [xm_full_xlim, xm_zoom_xlim],
                [xm_full_ylim, xm_zoom_ylim],
                ['full range', 'zoom (68% CI)']):
            for col, xi_5p, eta_5p, xi_2p, eta_2p, pmra_5p, pmdec_5p, pmra_2p, pmdec_2p, norm_c, coord_label in zip(
                    [0, 1],
                    [xi_star_5p, xi_star_5p],
                    [eta_star_5p, eta_star_5p],
                    [xi_star_2p, xi_star_2p],
                    [eta_star_2p, eta_star_2p],
                    [xm_pmra_5p, xm_pmra_5p],
                    [xm_pmdec_5p, xm_pmdec_5p],
                    [xm_pmra_2p, xm_pmra_2p],
                    [xm_pmdec_2p, xm_pmdec_2p],
                    [norm_xi, norm_eta],
                    [r'$\xi_s$ [mas]', r'$\eta_s$ [mas]']):
                ax = axes[row, col]
                if xm_5p_conv.any():
                    ax.scatter(pmra_5p, pmdec_5p, c=xi_5p if col == 0 else eta_5p, s=6,
                              norm=norm_c, cmap='plasma', alpha=0.8, zorder=3, label='Gaia 5p')
                if xm_2p_conv.any():
                    sc = ax.scatter(pmra_2p, pmdec_2p, c=xi_2p if col == 0 else eta_2p, s=8,
                                   marker='s', norm=norm_c, cmap='plasma', alpha=0.65,
                                   zorder=2, label='Gaia 2p')
                ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
                ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
                ax.set_title(f'xmatch — {row_label}  (colour: {coord_label})')
                ax.set_aspect('equal')
                if row == 0:
                    ax.legend(fontsize=7, loc='upper left')
                _style_ax(ax)
                if row == 0 and col == 0:
                    sc_xi = sc if xm_2p_conv.any() else (ax.collections[-1] if xm_5p_conv.any() else None)
                if row == 0 and col == 1:
                    sc_eta = sc if xm_2p_conv.any() else (ax.collections[-1] if xm_5p_conv.any() else None)

        if sc_xi is not None:
            cbar_xi = fig.colorbar(sc_xi, ax=axes[:, 0], shrink=0.6, pad=0.02, aspect=30)
            cbar_xi.set_label(r'$\xi_s$ [mas]')
        if sc_eta is not None:
            cbar_eta = fig.colorbar(sc_eta, ax=axes[:, 1], shrink=0.6, pad=0.02, aspect=30)
            cbar_eta.set_label(r'$\eta_s$ [mas]')
        fig.suptitle('xmatch proper motions coloured by tangent-plane source position', fontsize=13)
        _save(fig, plot_dir / 'pm_vector_diagram_detector_pos.png')

        # ── Figure 3: chi2 distributions (3-panel) ────────────────────────────
        print("  Plotting chi2 distributions...")
        _plot_chi2_distributions(self, r_hat, v_hat, plot_dir)

        # ── Figure 4: sky map + CMD ───────────────────────────────────────────
        print("  Plotting sky distribution and colour-magnitude diagrams...")
        pm_size = np.sqrt(pmra_xmatch ** 2 + pmdec_xmatch ** 2)
        pm_unc  = np.sqrt(sig_pmra_xmatch ** 2 + sig_pmdec_xmatch ** 2)
        ok = xm_converged & np.isfinite(gmag) & np.isfinite(pm_size)
        is_2p = self.gaia_2p   # same basis as has_gaia above (astrometric_params_solved)
        _plot_sky_and_cmd(ra, dec, gmag, bp_rp, pm_size, pm_unc, ok, plot_dir,
                               is_member=is_2p)  # mark 2p stars

        # ── DELVE sky/CMD figures, one per available DELVE colour ─────────────
        # Separate figures rather than DELVE photometry on the Gaia axes: DES
        # griz and Gaia G/BP/RP are different systems, so a DES colour drawn on
        # an axis labelled "Gaia BP - RP" would be meaningless.  bp3m emits
        # sky_cmd_pm_delve_gr.png and sky_cmd_pm_delve_ri.png with their own
        # labels; this mirrors that, including the +/-99 sentinel masking and the
        # deliberately Gaia-free mask.
        _gc = self.gaia_df if hasattr(self, 'gaia_df') else getattr(self, 'gaia_cat', None)
        if _gc is not None:
            # NOTE the mask does NOT require finite Gaia G.  DELVE-only stars
            # (negative source_id) have no Gaia magnitude, and reusing the Gaia
            # `ok` mask would drop exactly the faint population these figures
            # exist to show.
            _ok_delve_base = xm_converged & np.isfinite(pm_size)
            _DELVE_COLORS = [
                ('delve_gmag', 'delve_rmag', 'DELVE g − r (mag)', 'DELVE r (mag)',
                 'sky_cmd_pm_delve_gr.png'),
                ('delve_rmag', 'delve_imag', 'DELVE r − i (mag)', 'DELVE i (mag)',
                 'sky_cmd_pm_delve_ri.png'),
            ]
            for _blue, _red, _clab, _mlab, _fn in _DELVE_COLORS:
                if _blue not in _gc.columns or _red not in _gc.columns:
                    continue
                _vb = pd.to_numeric(_gc[_blue], errors='coerce').to_numpy(float).copy()
                _vr = pd.to_numeric(_gc[_red],  errors='coerce').to_numpy(float).copy()
                # DELVE's missing-photometry sentinels, same bounds as bp3m
                for _v in (_vb, _vr):
                    _v[(_v < -90.0) | (_v > 50.0)] = np.nan
                _ok_d = _ok_delve_base & np.isfinite(_vb - _vr) & np.isfinite(_vr)
                if _ok_d.sum() < 5:
                    continue
                print(f"    DELVE sky/CMD: {int(_ok_d.sum())} stars -> {_fn}")
                _plot_sky_and_cmd(ra, dec, _vr, _vb - _vr, pm_size, pm_unc,
                                  _ok_d, plot_dir, is_member=is_2p, fname=_fn,
                                  color_label=_clab, mag_label=_mlab)

        # ── Figure 5: per-image residual maps (3×2) ───────────────────────────
        if not plot_residuals:
            print(f"  All plots saved to {plot_dir}/")
            return
        print("  Plotting per-image residuals...")

        resid_dict = self.compute_residuals(r_hat, v_hat, C_r=C_r, C_vT=C_vT)

        for img in sorted(resid_dict.keys()):
            rd  = resid_dict[img]
            use = rd['use']
            if use.sum() == 0:
                continue

            sidx  = rd['sidx']
            xi_s  = rd['xi_s']
            eta_s = rd['eta_s']
            rx    = rd['resid_xi']
            ry    = rd['resid_eta']
            sx    = rd['sigma_xi']
            sy    = rd['sigma_eta']

            # sigma-normalised residuals
            sr_x = np.where(sx > 0, rx / sx, np.nan)
            sr_y = np.where(sy > 0, ry / sy, np.nan)

            pmra_img  = pmra_xmatch[sidx]
            pmdec_img = pmdec_xmatch[sidx]

            use_c = use & xm_converged[sidx]

            fig, axes = plt.subplots(3, 2, figsize=(13, 15), layout='constrained')

            # Row 0: xi/eta residuals [mas]
            for ax, res_mas, comp in zip(axes[0], [rx, ry], ['ξ', 'η']):
                vmax = np.nanpercentile(np.abs(res_mas[use]), 95)
                sc = ax.scatter(xi_s[use], eta_s[use], c=res_mas[use],
                                s=10, cmap='RdYlBu_r', vmin=-vmax, vmax=vmax,
                                alpha=0.8, zorder=2)
                ax.scatter(xi_s[~use], eta_s[~use], c='0.7', s=5, alpha=0.3, zorder=1)
                fig.colorbar(sc, ax=ax, label=f'residual {comp} [mas]')
                ax.set_xlabel(r'$\xi_s$ [mas]')
                ax.set_ylabel(r'$\eta_s$ [mas]')
                ax.set_title(f'{img}  residual {comp}  (n={use.sum()})')
                ax.set_aspect('equal')
                _style_ax(ax)

            # Row 1: sigma-normalised residuals
            for ax, sr, comp in zip(axes[1], [sr_x, sr_y], ['ξ', 'η']):
                finite = np.isfinite(sr[use])
                vmax_s = np.nanpercentile(np.abs(sr[use][finite]), 95) if finite.any() else 3.
                sc = ax.scatter(xi_s[use], eta_s[use], c=sr[use],
                                s=10, cmap='RdYlBu_r', vmin=-vmax_s, vmax=vmax_s,
                                alpha=0.8, zorder=2)
                ax.scatter(xi_s[~use], eta_s[~use], c='0.7', s=5, alpha=0.3, zorder=1)
                fig.colorbar(sc, ax=ax, label=f'residual {comp} / σ [σ]')
                ax.set_xlabel(r'$\xi_s$ [mas]')
                ax.set_ylabel(r'$\eta_s$ [mas]')
                rms_sr = np.nanstd(sr[use]) if use.any() else np.nan
                ax.set_title(f'{img}  σ-residual {comp}  (RMS = {rms_sr:.2f} σ)')
                ax.set_aspect('equal')
                _style_ax(ax)

            # Row 2: xmatch PM on detector
            for ax, pm_vals, comp_tex in zip(
                    axes[2],
                    [pmra_img, pmdec_img],
                    [r'$\mu_{\alpha*}$', r'$\mu_\delta$']):
                pm_use = pm_vals[use_c]
                if len(pm_use) > 0 and np.isfinite(pm_use).any():
                    p16, p84 = np.nanpercentile(pm_use, [16, 84])
                else:
                    p16, p84 = -5, 5
                sc = ax.scatter(xi_s[use_c], eta_s[use_c], c=pm_vals[use_c],
                                s=10, cmap='viridis', vmin=p16, vmax=p84,
                                alpha=0.8, zorder=2)
                fig.colorbar(sc, ax=ax, label=f'{comp_tex} xmatch [mas/yr]')
                ax.set_xlabel(r'$\xi_s$ [mas]')
                ax.set_ylabel(r'$\eta_s$ [mas]')
                ax.set_title(f'{img}  {comp_tex} xmatch  '
                             f'(clim: [{p16:.2f}, {p84:.2f}] mas/yr)')
                ax.set_aspect('equal')
                _style_ax(ax)

            fig.suptitle(f'xmatch detector residuals & proper motions — {img}', fontsize=12)
            _save(fig, resid_plot_dir / f'residuals_{img}.png')

        print(f"  All plots saved to {plot_dir}/")


# ── Plot helper functions (mirror bp3m/pipeline/plot_results.py) ─────────────

def _pm_geom_unc(sig_pmra, sig_pmdec, rho):
    """Geometric-mean PM uncertainty: (det C_pm)^(1/4)."""
    rho = np.clip(np.nan_to_num(rho), -0.9999, 0.9999)
    det = sig_pmra ** 2 * sig_pmdec ** 2 * (1.0 - rho ** 2)
    return np.where((sig_pmra > 0) & (sig_pmdec > 0), det ** 0.25, np.nan)


def _pm_error_bars(ax, pmra, pmdec, C_pm, color='gray', alpha=0.18, lw=0.6):
    """Draw 1σ principal-axis error bars for each point in a PM vector diagram."""
    eigvals, eigvecs = np.linalg.eigh(C_pm)
    half_axes = np.sqrt(np.maximum(eigvals, 0.))
    centers = np.stack([pmra, pmdec], axis=1)
    delta   = eigvecs * half_axes[:, np.newaxis, :]
    segs = np.concatenate([
        np.stack([centers - delta[:, :, 0], centers + delta[:, :, 0]], axis=1),
        np.stack([centers - delta[:, :, 1], centers + delta[:, :, 1]], axis=1),
    ], axis=0)
    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=lw, zorder=1)
    ax.add_collection(lc)


def _padded_lim(*arrays, pad=0.04):
    """Return (lo, hi) spanning all values in *arrays with a fractional pad."""
    lo = min(np.nanmin(a) for a in arrays)
    hi = max(np.nanmax(a) for a in arrays)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _style_ax(ax):
    """Apply consistent grid + minor-tick style."""
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-',  linewidth=0.5, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.3, alpha=0.4)
    ax.tick_params(which='both', direction='in', top=True, right=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {path}')


def _plot_chi2_distributions(solver, r_hat, v_hat, plot_dir):
    """Three-panel diagnostic for the survey-only chi2 per star per image."""
    from scipy.stats import chi2 as chi2_dist_mod

    resid_xmatch = solver.compute_residuals(r_hat, v_hat)

    per_img_chi2  = {}
    per_img_all   = {}
    per_img_alpha = {}
    _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)   # median of chi2(2)

    for img, rd in resid_xmatch.items():
        use    = solver._img_data[img]['use_for_fit']
        chi2_v = rd['sigma_resid'] ** 2
        per_img_all[img]  = chi2_v
        per_img_chi2[img] = chi2_v[use]
        med = np.median(chi2_v[use]) if use.sum() >= 2 else np.nan
        per_img_alpha[img] = float(max(1.0, np.sqrt(med / _MEDIAN_CHI2_2)))

    all_accepted = np.concatenate(list(per_img_chi2.values()))
    all_vals     = np.concatenate(list(per_img_all.values()))

    # An image can end up with zero accepted detections (every one clipped, or
    # too few matches to begin with).  np.percentile on an empty array raises
    # IndexError, which previously aborted the whole run from inside a PLOT:
    # M49's per-image pass died at image 660 of 5009 this way, losing the
    # remaining 4349.  A diagnostic must never be able to kill the fit, so skip
    # the figure instead.  Nothing about the plot itself changes when there IS
    # data to draw.
    if all_accepted.size == 0 or all_vals.size == 0:
        print('    (skipping chi2 distribution plot: no accepted detections)')
        return

    thresholds = {
        '0.99':   chi2_dist_mod.ppf(0.99,   df=2),
        '0.999':  chi2_dist_mod.ppf(0.999,  df=2),
        '0.9999': chi2_dist_mod.ppf(0.9999, df=2),
    }
    thresh_colors = {'0.99': 'royalblue', '0.999': 'darkorange', '0.9999': 'crimson'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('survey-only chi2 distributions at convergence', fontsize=13)

    # Panel 1: histogram vs chi2(2) theory
    ax = axes[0]
    clip = np.percentile(all_accepted, 99.5)
    bins = np.linspace(0, max(clip, 20), 80)
    ax.hist(all_accepted, bins=bins, density=True, color='steelblue',
            alpha=0.6, label='accepted stars')
    xx = np.linspace(0.01, bins[-1], 400)
    ax.plot(xx, chi2_dist_mod.pdf(xx, df=2), 'k-', lw=1.5, label='χ²(2) theory')
    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls='--',
                   label=f'q={label}  ({thr:.1f})')
    ax.set_xlabel('σ_resid² (survey-only chi2)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution (accepted stars)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, bins[-1])

    # Panel 2: CDF with surviving fractions
    ax = axes[1]
    sorted_chi2 = np.sort(all_accepted)
    cdf = np.arange(1, len(sorted_chi2) + 1) / len(sorted_chi2)
    ax.plot(sorted_chi2, cdf, color='steelblue', lw=1.5)
    ax.plot(np.sort(all_vals), np.arange(1, len(all_vals) + 1) / len(all_vals),
            color='gray', lw=1, ls=':', alpha=0.7, label='all (incl. excluded)')
    for label, thr in thresholds.items():
        frac = float((all_accepted < thr).mean())
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls='--',
                   label=f'q={label}: {100*frac:.1f}% survive')
    ax.set_xlabel('σ_resid² threshold')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('CDF — surviving fraction vs. threshold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(thresholds['0.9999'] * 1.3, np.percentile(all_accepted, 98)))
    ax.set_ylim(0, 1)

    # Panel 3: per-image median chi2 bar chart
    ax = axes[2]
    img_names = list(per_img_chi2.keys())
    medians   = [np.median(per_img_chi2[im]) for im in img_names]
    alphas    = [per_img_alpha[im] for im in img_names]
    order   = np.argsort(medians)[::-1]
    names_s = [img_names[i] for i in order]
    meds_s  = [medians[i]   for i in order]
    alps_s  = [alphas[i]    for i in order]
    y    = np.arange(len(names_s))
    bar_h = max(0.3, min(0.8, 12.0 / max(len(names_s), 1)))
    bars = ax.barh(y, meds_s, height=bar_h, color='steelblue', alpha=0.7,
                   label='median chi2')
    for bar, alp in zip(bars, alps_s):
        bar.set_facecolor('tomato' if alp > 2 else 'steelblue')
    for i, (med, alp) in enumerate(zip(meds_s, alps_s)):
        ax.text(med + 0.05, i, f'α={alp:.2f}', va='center', fontsize=6)
    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.0, ls='--', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=max(5, min(8, 200 // max(len(names_s), 1))))
    ax.set_xlabel('Median σ_resid² (survey-only chi2, accepted stars)')
    ax.set_title('Per-image: median chi2 & alpha\n(red = α > 2)')
    finite_meds = [m for m in meds_s if np.isfinite(m)]
    xlim_right = max(max(finite_meds) * 1.25 if finite_meds else 0,
                     thresholds['0.999'] * 1.1)
    ax.set_xlim(0, xlim_right)
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, plot_dir / 'chi2_distributions.png')


def _plot_sky_and_cmd(ra, dec, gmag, bp_rp, pm_size, pm_unc, ok, plot_dir,
                            is_member=None, fname='sky_cmd_pm.png',
                            color_label='Gaia BP − RP (mag)', mag_label='Gaia G (mag)'):
    """Three panels: sky map coloured by |PM|, CMD coloured by |PM|, CMD coloured by σ_PM.
    is_member marks 2p Gaia stars (None to skip, bool array to mark with squares)."""
    pm_vals  = pm_size[ok & (pm_size > 0)]
    unc_vals = pm_unc[ok & (pm_unc > 0)]
    if len(pm_vals) < 2 or len(unc_vals) < 2:
        return
    vmin_pm  = float(np.nanpercentile(pm_vals,  1))
    vmax_pm  = float(np.nanpercentile(pm_vals, 99))
    vmin_unc = float(np.nanpercentile(unc_vals,  1))
    vmax_unc = float(np.nanpercentile(unc_vals, 99))
    norm_pm  = mcolors.LogNorm(vmin=max(vmin_pm,  1e-3), vmax=vmax_pm)
    norm_unc = mcolors.LogNorm(vmin=max(vmin_unc, 1e-3), vmax=vmax_unc)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')

    # ── Sky panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    ok_sky = ok
    sc = None
    if is_member is not None:
        is_mem_sky = is_member[ok_sky]
        if (~is_mem_sky).any():
            sc = ax.scatter(ra[ok_sky][~is_mem_sky], dec[ok_sky][~is_mem_sky],
                           c=pm_size[ok_sky][~is_mem_sky], s=6,
                           norm=norm_pm, cmap='plasma', alpha=0.8, zorder=3,
                           linewidths=0, rasterized=True, label='Gaia 5p/6p')
        if is_mem_sky.any():
            sc = ax.scatter(ra[ok_sky][is_mem_sky], dec[ok_sky][is_mem_sky],
                           c=pm_size[ok_sky][is_mem_sky], s=8, marker='s',
                           norm=norm_pm, cmap='plasma', alpha=0.65, zorder=2,
                           linewidths=0, rasterized=True, label='Gaia 2p')
    else:
        sc = ax.scatter(ra[ok_sky], dec[ok_sky], c=pm_size[ok_sky], s=6,
                       norm=norm_pm, cmap='plasma', alpha=0.8, zorder=3,
                       linewidths=0, rasterized=True)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label='|PM| (mas/yr)')
    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    ax.set_title('Sky distribution  (colour = |PM|)')
    ax.invert_xaxis()
    if is_member is not None:
        ax.legend(fontsize=7, loc='upper right')
    _style_ax(ax)

    # ── CMD panel 1: PM size ─────────────────────────────────────────────────
    ax = axes[1]
    has_cmd = ok & np.isfinite(bp_rp)
    sc = None
    if has_cmd.any():
        if is_member is not None:
            is_mem_cmd = is_member[has_cmd]
            if (~is_mem_cmd).any():
                sc = ax.scatter(bp_rp[has_cmd][~is_mem_cmd], gmag[has_cmd][~is_mem_cmd],
                               c=pm_size[has_cmd][~is_mem_cmd], s=6,
                               norm=norm_pm, cmap='plasma', alpha=0.8, zorder=3,
                               linewidths=0, rasterized=True, label='Gaia 5p/6p')
            if is_mem_cmd.any():
                sc = ax.scatter(bp_rp[has_cmd][is_mem_cmd], gmag[has_cmd][is_mem_cmd],
                               c=pm_size[has_cmd][is_mem_cmd], s=8, marker='s',
                               norm=norm_pm, cmap='plasma', alpha=0.65, zorder=2,
                               linewidths=0, rasterized=True, label='Gaia 2p')
        else:
            sc = ax.scatter(bp_rp[has_cmd], gmag[has_cmd], c=pm_size[has_cmd], s=6,
                           norm=norm_pm, cmap='plasma', alpha=0.8, zorder=3,
                           linewidths=0, rasterized=True)
        if sc is not None:
            plt.colorbar(sc, ax=ax, label='|PM| (mas/yr)')
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title('CMD  (colour = |PM|)')
    ax.invert_yaxis()
    if is_member is not None and has_cmd.any():
        ax.legend(fontsize=7, loc='upper right')
    _style_ax(ax)

    # ── CMD panel 2: PM uncertainty ──────────────────────────────────────────
    ax = axes[2]
    sc = None
    if has_cmd.any():
        if is_member is not None:
            is_mem_cmd = is_member[has_cmd]
            if (~is_mem_cmd).any():
                sc = ax.scatter(bp_rp[has_cmd][~is_mem_cmd], gmag[has_cmd][~is_mem_cmd],
                               c=pm_unc[has_cmd][~is_mem_cmd], s=6,
                               norm=norm_unc, cmap='viridis', alpha=0.8, zorder=3,
                               linewidths=0, rasterized=True, label='Gaia 5p/6p')
            if is_mem_cmd.any():
                sc = ax.scatter(bp_rp[has_cmd][is_mem_cmd], gmag[has_cmd][is_mem_cmd],
                               c=pm_unc[has_cmd][is_mem_cmd], s=8, marker='s',
                               norm=norm_unc, cmap='viridis', alpha=0.65, zorder=2,
                               linewidths=0, rasterized=True, label='Gaia 2p')
        else:
            sc = ax.scatter(bp_rp[has_cmd], gmag[has_cmd], c=pm_unc[has_cmd], s=6,
                           norm=norm_unc, cmap='viridis', alpha=0.8, zorder=3,
                           linewidths=0, rasterized=True)
        if sc is not None:
            plt.colorbar(sc, ax=ax, label='σ_PM (mas/yr)')
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title('CMD  (colour = σ_PM)')
    ax.invert_yaxis()
    if is_member is not None and has_cmd.any():
        ax.legend(fontsize=7, loc='upper right')
    _style_ax(ax)

    _save(fig, plot_dir / fname)


# ── Data loading helpers ─────────────────────────────────────────────────────

