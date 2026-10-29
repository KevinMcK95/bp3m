"""Gaia error / systematics model shared by every BP3M level (2026-10-28; plan hst_dist_corr/GAIA_SYSTEMATICS_BP3M_PLAN.md).

Every place that weights a Gaia measurement (solver, cross-match, catalogue builder, GDC header-frame labels, QSO vetting,
synthetic fields, ground alignment) builds the PUBLISHED covariance its own way (its own parameter order and fallbacks for
missing errors) and then calls `inflate` here for the model-dependent part, so one spec controls all of them:

    cov   = fab_scalar  Fabricius+21 scalar sigma multipliers by solution type (1.22 6p / 1.05 5p / 1.00 2p) [default]
            fab_mag     Fabricius+21 Fig. 19/20 magnitude-dependent multipliers (astro_utils.gaia_cov_scale_vector)
            as_given    no multiplier: the catalogue's *_error / *_corr already encode the model covariance
    floor = diag        Vasiliev & Baumgardt 21 floors (0.026 mas/yr PM, 0.011 mas parallax) on each star's diagonal [default]
            none        no floor
            modes       no per-star floor: the systematic is carried by shared field modes (later phase)

Spec string: "cov=<...>;floor=<...>" (order free, missing keys = defaults); "" or "default" = the defaults.
The process-wide model is read from the environment variable BP3M_GAIA_MODEL on every call, so a model set in a CLI's
main() before any worker pool exists reaches fork / forkserver / spawn workers alike (the old module-global
astro_utils.GAIA_UWU_MAG did not reach forkserver indv workers).  Defaults reproduce the pre-2026-10-28 numbers bit for bit.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
from .astro_utils import GAIA_SYS_DICT

ENV = 'BP3M_GAIA_MODEL'
_CHOICES = {'cov': ('fab_scalar', 'fab_mag', 'as_given'), 'floor': ('diag', 'none', 'modes'), 'modes': None}   # modes: a path
CANONICAL = (0, 1, 2, 3, 4)          # axis positions of (ra*, dec, pmra, pmdec, parallax) in the solver's 5x5 order


@dataclass(frozen=True)
class GaiaModel:
    cov: str = 'fab_scalar'
    floor: str = 'diag'
    modes: str = 'none'          # field-mode npz (handoff 3.2) or 'none'

    @classmethod
    def parse(cls, spec: "str | GaiaModel | None") -> "GaiaModel":
        if isinstance(spec, GaiaModel):
            return spec
        s = (spec or '').strip()
        if s in ('', 'default'):
            return cls()
        kw = {}
        for tok in s.replace(',', ';').split(';'):
            if not tok.strip():
                continue
            if '=' not in tok:
                raise ValueError(f"gaia model spec {spec!r}: token {tok!r} is not key=value")
            k, v = (t.strip() for t in tok.split('=', 1))
            if k not in _CHOICES:
                raise ValueError(f"gaia model spec {spec!r}: unknown key {k!r} (known: {sorted(_CHOICES)})")
            if _CHOICES[k] is not None and v not in _CHOICES[k]:
                raise ValueError(f"gaia model spec {spec!r}: {k}={v!r} not in {_CHOICES[k]}")
            kw[k] = v
        m = cls(**kw)
        if m.modes != 'none' and m.floor == 'diag':
            raise ValueError(f"gaia model spec {spec!r}: field modes with the per-star VB21 floor would count the same "
                             f"systematic twice; use floor=modes (floor moved into the modes) or floor=none")
        if m.modes == 'none' and m.floor == 'modes':
            raise ValueError(f"gaia model spec {spec!r}: floor=modes needs a modes=<npz> file")
        return m

    @property
    def spec(self) -> str:
        return f"cov={self.cov};floor={self.floor}" + (f";modes={self.modes}" if self.modes != 'none' else '')

    @property
    def is_default(self) -> bool:
        return self == GaiaModel()


def current() -> GaiaModel:
    """The process-wide model (environment variable BP3M_GAIA_MODEL; defaults when unset)."""
    return GaiaModel.parse(os.environ.get(ENV, ''))


def set_current(model: "str | GaiaModel | None") -> GaiaModel:
    """Set the process-wide model.  Call in a CLI's main() BEFORE any worker pool is created, so every worker inherits it."""
    m = GaiaModel.parse(model)
    os.environ[ENV] = m.spec
    return m


ENV_FILES = 'BP3M_GAIA_FILES'


def override_files() -> "list | None":
    """Gaia CSV(s) that replace the field's canonical files (environment BP3M_GAIA_FILES, os.pathsep-separated), or None."""
    v = os.environ.get(ENV_FILES, '')
    return [p for p in v.split(os.pathsep) if p] or None


def files_md5(paths) -> "str | None":
    import hashlib
    from pathlib import Path
    out = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in (paths or []) if p is not None and Path(p).exists()]
    return ','.join(out) or None


def set_override_files(paths) -> "list | None":
    """Use these Gaia CSV(s) instead of the field's canonical files at every level (set before any worker pool exists).
    Hard errors: a missing file, or a file named *_gaia.csv (overrides live in Gaia/corrected/ as *_gaiacorr_<model>.csv so
    the canonical glob fallback can never mix them with the originals)."""
    from pathlib import Path
    if not paths:
        os.environ.pop(ENV_FILES, None); return None
    paths = [str(Path(p).resolve()) for p in paths]
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"--gaia_csv_override: {p} does not exist")
        if p.endswith('_gaia.csv'):
            raise SystemExit(f"--gaia_csv_override: {p} is named like a canonical field file (*_gaia.csv); "
                             f"name overrides *_gaiacorr_<model>.csv")
    os.environ[ENV_FILES] = os.pathsep.join(paths)
    return paths


def provenance() -> dict:
    """run_config.json entries for the Gaia inputs of this process."""
    ov = override_files()
    return {'gaia_model': current().spec, 'gaia_csv_override': ov, 'gaia_csv_override_md5': files_md5(ov) if ov else None}


def resolve_inherited_files(cli_files, upstream_cfg: "dict | None", stage: str, upstream: str) -> "list | None":
    """Gaia file override for a downstream stage: the upstream alignment's recorded override (md5 must still match), or
    the CLI value, which must then equal the upstream one.  No override anywhere = the canonical files (unchanged)."""
    rec = (upstream_cfg or {}).get('gaia_csv_override') or None
    rec_md5 = (upstream_cfg or {}).get('gaia_csv_override_md5')
    if cli_files:
        from pathlib import Path
        cli = [str(Path(p).resolve()) for p in cli_files]
        if rec is not None and cli != list(rec):
            raise SystemExit(f"{stage}: --gaia_csv_override {cli} differs from the {upstream} alignment's {rec}")
        files = cli
    else:
        files = list(rec) if rec else None
    if files and rec_md5 is not None and files_md5(files) != rec_md5:
        raise SystemExit(f"{stage}: Gaia override file(s) changed since the {upstream} alignment (md5): {files}")
    out = set_override_files(files)
    if out:
        print(f"  {stage}: Gaia CSV override {out}")
    return out


def resolve_inherited(cli_spec, upstream_cfg: "dict | None", stage: str, upstream: str) -> GaiaModel:
    """The model for a downstream stage (v2, pop-fits): the CLI value if given, else the one recorded in the upstream
    run_config.json.  A CLI value that differs from a recorded upstream model is a hard error (a Gaia-model change is an
    alignment input: re-align from indv instead).  An upstream run_config without the key predates the recording
    (2026-10-28) and used the default unless run with --gaia_uwu_mag, which was not recorded either -> default + a warning.
    Sets the process-wide model (call before any worker pool exists) and returns it."""
    rec = (upstream_cfg or {}).get('gaia_model')
    up = GaiaModel.parse(rec) if rec else None
    if cli_spec is not None:
        m = GaiaModel.parse(cli_spec)
        if up is not None and m != up:
            raise SystemExit(f"{stage}: --gaia_model {m.spec!r} differs from the {upstream} alignment's {up.spec!r}; "
                             f"a Gaia-model change needs its own alignment (indv -> v1 -> v2)")
    elif up is not None:
        m = up
    else:
        m = GaiaModel()
        if upstream_cfg is not None:
            print(f"  {stage}: {upstream} run_config.json records no gaia_model (pre-2026-10-28 run) -> default "
                  f"{m.spec} (a --gaia_uwu_mag upstream run would not have been recorded)")
    set_current(m)
    if not m.is_default:
        print(f"  {stage}: Gaia model {m.spec}")
    return m


def _type_multiplier_sq(is_6p, is_5p):
    """(n,) squared scalar sigma multipliers by solution type; anything neither 6p nor 5p gets mult_2p**2 (= 1)."""
    m6, m5, m2 = (GAIA_SYS_DICT[k] ** 2 for k in ('mult_6p', 'mult_5p', 'mult_2p'))
    is_6p = np.asarray(is_6p, bool); is_5p = np.asarray(is_5p, bool)
    return np.where(is_6p, m6, np.where(is_5p, m5, m2))


def inflate(C, is_6p, is_5p, gmag=None, idx=CANONICAL, model=None):
    """Apply the model's multipliers and floor to published Gaia covariances C (n, k, k), IN PLACE; returns C.

    idx  : positions of (ra*, dec, pmra, pmdec, parallax) along C's axes; None for a parameter C does not carry
           (e.g. (None, None, 0, 1, 2) for a PM / parallax 3x3 block).
    gmag : (n,) Gaia G, needed only for cov=fab_mag.
    """
    m = current() if model is None else GaiaModel.parse(model)
    C = np.asarray(C)
    if m.cov == 'fab_scalar':
        C *= _type_multiplier_sq(is_6p, is_5p)[:, None, None]
    elif m.cov == 'fab_mag':
        if gmag is None:
            raise ValueError("gaia model cov=fab_mag needs gmag")
        from .astro_utils import gaia_cov_scale_vector
        f5 = gaia_cov_scale_vector(np.asarray(gmag, float), np.asarray(is_6p, bool), np.asarray(is_5p, bool))
        F = np.ones(C.shape[:2])
        for j, a in enumerate(idx):
            if a is not None:
                F[:, a] = f5[:, j]
        C *= F[:, :, None] * F[:, None, :]
    if m.floor == 'diag':
        pm2, plx2 = GAIA_SYS_DICT['pm_sys_err'] ** 2, GAIA_SYS_DICT['parallax_sys_err'] ** 2
        for j, add in ((2, pm2), (3, pm2), (4, plx2)):
            a = idx[j]
            if a is not None:
                C[:, a, a] += add
    return C


def inflate_pm_sigma(sig_pmra, sig_pmdec, is_6p, has_pm, gmag=None, model=None):
    """Per-star inflated PM sigmas sqrt(mult^2 sigma^2 + floor^2) for diagnostics that only use the PM diagonal
    (cross-match pulls).  Rows without a PM keep their raw sigma."""
    m = current() if model is None else GaiaModel.parse(model)
    sig_pmra = np.asarray(sig_pmra, float); sig_pmdec = np.asarray(sig_pmdec, float)
    has_pm = np.asarray(has_pm, bool); is_6p = np.asarray(is_6p, bool)
    if m.cov == 'fab_scalar':
        mult = np.where(is_6p, GAIA_SYS_DICT['mult_6p'], np.where(has_pm, GAIA_SYS_DICT['mult_5p'], GAIA_SYS_DICT['mult_2p']))
    elif m.cov == 'fab_mag':
        from .astro_utils import gaia_uwu_pm
        mult = gaia_uwu_pm(np.asarray(gmag, float), is_6p, has_pm & ~is_6p)
    else:
        mult = np.ones_like(sig_pmra)
    fl2 = GAIA_SYS_DICT['pm_sys_err'] ** 2 if m.floor == 'diag' else 0.0
    return (np.where(has_pm, np.sqrt(mult ** 2 * sig_pmra ** 2 + fl2), sig_pmra),
            np.where(has_pm, np.sqrt(mult ** 2 * sig_pmdec ** 2 + fl2), sig_pmdec))


class FieldModes:
    """Gaia field modes (handoff 3.2): per Gaia source a mean shift M (3: pmra, pmdec mas/yr, parallax mas), loadings
    W (3 x K) of K shared N(0, 1) coefficients z, and an independent remainder R (3 x 3):
        v_meas,i = v_true,i + M_i + W_i z + e_i,   e_i ~ N(0, C_star,i + R_i).
    Positions carry no field term.  `align(ids, has_pm)` returns the arrays in the order of a Gaia table; every source with
    a Gaia PM must be in the file (hard error otherwise); sources without one (2p, synthetic HST-only / DELVE rows) get 0."""

    def __init__(self, path):
        from pathlib import Path
        self.path = str(Path(path).resolve())
        z = np.load(self.path, allow_pickle=False)
        self.source_id = z['source_id'].astype(np.int64)
        self.M = np.asarray(z['M'], float); self.W = np.asarray(z['W'], float); self.R = np.asarray(z['R'], float)
        n = len(self.source_id)
        if self.M.shape != (n, 3) or self.W.ndim != 3 or self.W.shape[:2] != (n, 3) or self.R.shape != (n, 3, 3):
            raise ValueError(f"{self.path}: bad shapes M {self.M.shape} W {self.W.shape} R {self.R.shape} for {n} sources")
        if len(np.unique(self.source_id)) != n:
            raise ValueError(f"{self.path}: duplicate source_id")
        self.K = self.W.shape[2]
        self.qso_excluded = (z['qso_excluded'].astype(np.int64) if 'qso_excluded' in z.files else np.zeros(0, np.int64))
        self.meta = {k: str(z[k]) for k in ('model', 'calibration', 'sha256_calibration', 'created') if k in z.files}
        self.md5 = files_md5([self.path])

    def align(self, ids, has_pm):
        ids = np.asarray(ids, np.int64); has_pm = np.asarray(has_pm, bool); n = len(ids)
        pos = {int(s): k for k, s in enumerate(self.source_id)}
        idx = np.array([pos.get(int(i), -1) for i in ids])
        miss = has_pm & (idx < 0) & (ids > 0)
        if miss.any():
            raise ValueError(f"{self.path}: {int(miss.sum())} Gaia sources with a PM are not in the mode file "
                             f"(first {ids[miss][:5].tolist()})")
        use = has_pm & (idx >= 0) & (ids > 0)
        M = np.zeros((n, 3)); W = np.zeros((n, 3, self.K)); R = np.zeros((n, 3, 3))
        M[use] = self.M[idx[use]]; W[use] = self.W[idx[use]]; R[use] = self.R[idx[use]]
        return M, W, R

    def check_anchors(self, anchor_ids):
        """Every QSO BP3M anchors to must be absent from the mode conditioning set (no double counting)."""
        a = set(int(i) for i in anchor_ids)
        bad = a - set(int(i) for i in self.qso_excluded)
        if bad and self.meta.get('model') != 'vb21':
            raise SystemExit(f"{self.path}: {len(bad)} BP3M QSO anchors are not in qso_excluded (they would count twice), "
                             f"e.g. {sorted(bad)[:5]}")


_MODES_CACHE = {}


def field_modes(model=None) -> "FieldModes | None":
    m = current() if model is None else GaiaModel.parse(model)
    if m.modes == 'none':
        return None
    if m.modes not in _MODES_CACHE:
        _MODES_CACHE[m.modes] = FieldModes(m.modes)
    return _MODES_CACHE[m.modes]
