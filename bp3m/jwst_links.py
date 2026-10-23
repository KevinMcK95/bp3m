"""Exposure / visit linking of JWST frames inside the BP3M solve (2026-10-07, jwst-integration branch).

A JWST stage-2 ``_cal`` file is one detector, so a NIRCam exposure enters the solve as 10 images
(8 SW + 2 LW) plus the NIRISS parallel, each with its own 6p transformation.  A 63" SW detector sees
0-16 Gaia stars (Draco 5-9, And XIV 0-3): on its own its transformation is either unconstrained
(frozen at the header, pointing prior 5000 mas) or a wild 1-2 star solution, and the HST+JWST stars
of the Draco joint run shifted 2-2.6 sigma against the HST-only run because of it.

All detectors of a visit share one FGS guide-star acquisition and attitude, and the SIAF ties their
frames together, so their header ERRORS are shared.  Measured on the Draco joint run (6 visits,
programs 1304 / 4513 / 9225, image_transformations.csv):
  * within a module (A1-A4 + ALONG, or B1-B4 + BLONG) the fitted pointing deviation from the header
    agrees to 1.5-4 mas across detectors and to 1-3 mas across the dithers of the visit; rotation
    deviations agree to 0.4-0.8" (posterior sigma ~6" per frame) and scale to < 1e-4;
  * module A vs module B differ by a visit-constant 0-14 mas, NIRISS vs NIRCam by 30-50 mas.
Hence two levels of coupling prior on the DEVIATIONS (r_j - r_prior_j) of the linked frames:
  'module' : all frames of one visit x module         -- tight (5 mas, 3.6", 1e-4)
  'visit'  : all JWST frames of one visit (any module / instrument) -- loose (25 mas, 7", 2e-4)
Each group adds the pairwise penalty  sum_i (D_i - D_mean)^T C (D_i - D_mean)  (all pairs with C/n),
i.e. every member's deviation from the group mean ~ N(0, Sigma_link); the solver adds the blocks to
H_rr and the pull toward the partners' current deviations to the rhs (solver._link_pairs), exactly
like the ACS hi/lo chip-pair prior but on deviations, because the detectors have different header
rotations.  Frames in a link group are never demoted to 'astrometry-only / r frozen'.

Liwen Chen's bp3m fork (lwcxx/bp3m) treated every detector independently; this module is new.
"""
from __future__ import annotations

import re

# jw<PPPPP><OOO><VVV>_<GGSAA>_<EEEEE>_<detector>
_NAME_RE = re.compile(r'^jw(?P<program>\d{5})(?P<obs>\d{3})(?P<visit>\d{3})_(?P<act>\d{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)$')

LINK_LEVELS = ('none', 'module', 'visit', 'exposure')

# prior widths of a member's deviation from its group mean
LINK_SIGMAS = {
    'exposure': dict(rot_deg=0.001, scale=5e-5, skew=5e-5, pointing_mas=5.0),   # 2026-10-23: the 10 SCAs of one exposure share the attitude exactly; widths = the field-to-field scatter of the calibrated layout (bp3m/data/nircam_sca_layout.csv: 1-8 mas, 0.3-1 mdeg, 20-50 ppm)
    'module': dict(rot_deg=0.001, scale=1e-4, skew=1e-4, pointing_mas=5.0),
    'visit':  dict(rot_deg=0.002, scale=2e-4, skew=2e-4, pointing_mas=25.0),
}


def parse_jwst_name(name: str):
    """Visit / activity / exposure / detector of a MAST JWST stem (lower-case), or None (HST, synthetic)."""
    m = _NAME_RE.match(str(name).lower())
    if not m:
        return None
    d = m.groupdict()
    d['visit_id'] = f"jw{d['program']}{d['obs']}{d['visit']}"
    d['exposure'] = f"{d['visit_id']}_{d['act']}_{d['exp']}"
    d['module'] = module_of(d['det'])
    return d


def module_of(det: str) -> str:
    det = str(det).lower()
    if det.startswith('nrca'):
        return 'NRCA'
    if det.startswith('nrcb'):
        return 'NRCB'
    if det.startswith('nis'):
        return 'NIS'
    if det.startswith('mir'):
        return 'MIRI'
    return det.upper()


def link_groups(image_names, metas=None, level: str = 'visit',
                sigma_module_mas: float | None = None, sigma_visit_mas: float | None = None) -> list:
    """[{'label', 'level', 'members': [image index, ...], 'sigmas': {...}}, ...] with >= 2 members each.

    metas (optional): name -> image meta; only images whose meta says telescope == 'JWST' (or whose
    name parses as a JWST stem when no meta is given) are linked.  Pointing widths can be overridden
    from the CLI; rotation / scale / skew widths come from LINK_SIGMAS.
    """
    level = (level or 'none').lower()
    if level not in LINK_LEVELS:
        raise ValueError(f"jwst_link must be one of {LINK_LEVELS}, got {level!r}")
    if level == 'none':
        return []
    by_module, by_visit, by_expo = {}, {}, {}
    for idx, name in enumerate(image_names):
        if metas is not None:
            meta = metas.get(name) or {}
            if str(meta.get('telescope', 'HST')).upper() != 'JWST':
                continue
        p = parse_jwst_name(name)
        if p is None:
            continue
        by_module.setdefault((p['visit_id'], p['module']), []).append(idx)
        by_visit.setdefault(p['visit_id'], []).append(idx)
        by_expo.setdefault(p['exposure'], []).append(idx)
    sig_m = dict(LINK_SIGMAS['module'])
    sig_v = dict(LINK_SIGMAS['visit'])
    if sigma_module_mas is not None:
        sig_m['pointing_mas'] = float(sigma_module_mas)
    if sigma_visit_mas is not None:
        sig_v['pointing_mas'] = float(sigma_visit_mas)
    out = []
    if level == 'exposure':
        # all NIRCam SCAs (+ parallels) of one exposure: the pooled cross-match (xmatch_exposure) already placed them on the
        # calibrated focal plane, so their r_prior differences carry the layout; this group keeps their DEVIATIONS together
        for ekey, members in sorted(by_expo.items()):
            if len(members) >= 2:
                out.append(dict(label=f"{ekey}/expo", level='exposure', members=list(members), sigmas=dict(LINK_SIGMAS['exposure'])))
    for (vid, mod), members in sorted(by_module.items()):
        if len(members) >= 2:
            out.append(dict(label=f"{vid}/{mod}", level='module', members=list(members), sigmas=sig_m))
    if level in ('visit', 'exposure'):
        for vid, members in sorted(by_visit.items()):
            # a visit with a single module is already covered by its module group
            if len(members) >= 2 and len({module_of(parse_jwst_name(image_names[i])['det']) for i in members}) >= 2:
                out.append(dict(label=f"{vid}/all", level='visit', members=list(members), sigmas=sig_v))
    return out
