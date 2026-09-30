"""Deterministic basis GDC correction (hst_dist_corr/ml/basis_fit.py, chain basis3e) evaluated per detection at
catalog-load time -- the basis sibling of pos_corr_model.PosCorrModel, same hook: corrected = x_gdc - f.

A model directory holds, per instrument/detector/filter, basis_<INST>_<DET>_<FILTER>_<tag>.json (spec) and .npz
(coefficients) written by hst_dist_corr/ml/basis_train.py, plus normalization.json (inst_mag0 per key/filter).
f includes EVERYTHING the model learned (user 2026-09-30): static surfaces, magnitude surface, the mean linear
terms vs time, CTE, pixel phase, era corrections and the orbit-conditioned per-image linear terms -- nothing is
left for the per-image alignment to rediscover, so BP3M's per-image priors can be as tight as the residual widths.

Inputs, all HST-side: raw x, y (pypass catalog), chip (ext 1/4), instrumental mag, local sky, epoch (header),
and the exposure's orbit geometry from its JIT file (Latitude/Longitude track + Sun; zero if absent).
Re-implements the trainer's design matrices in numpy/scipy (no dependence on the research repo).
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

AX, AY = 2048.0, 1024.0
ORBIT_FEATURES = ['cos_th', 'sin_th', 'cos_2th', 'sin_2th', 'shadow', 'first_vf', 't_in_visit_h', 'va_tot_ppm', 'va_drift_ppm']
ORBIT_SCALE = {'t_in_visit_h': 1 / 3.0, 'va_tot_ppm': 1 / 100.0, 'va_drift_ppm': 1 / 10.0}
INSTALL = {'ACS/WFC': 2002.2, 'WFC3/UVIS': 2009.4}
HDR_KEYS0 = ('EXPSTART', 'EXPEND', 'EXPTIME', 'INSTRUME', 'DETECTOR', 'FILTER', 'FILTER1', 'FILTER2', 'ASN_ID', 'RA_TARG', 'DEC_TARG', 'ROOTNAME')
HDR_KEYS1 = ('CRVAL1', 'CRVAL2')


# ---------------------------------------------------------------- design (port of basis_fit) ----
def _bspl_1d(u, lo, hi, h, k=3):
    nint = max(1, int(round((hi - lo) / h))); t = np.linspace(lo, hi, nint + 1); t = np.r_[[lo] * k, t, [hi] * k]
    D = BSpline.design_matrix(np.clip(u, lo, hi - 1e-9), t, k).tocsr()
    return D.indices.reshape(-1, k + 1), D.data.reshape(-1, k + 1), len(t) - k - 1


class _Surface:
    def __init__(self, h):
        self.h = h; _, _, self.nx = _bspl_1d(np.array([0.0]), 0, 4096, h); _, _, self.ny = _bspl_1d(np.array([0.0]), 0, 2048, h); self.n = self.nx * self.ny
    def rows(self, x, y):
        ix, vx, _ = _bspl_1d(x, 0, 4096, self.h); iy, vy, _ = _bspl_1d(y, 0, 2048, self.h)
        return (ix[:, :, None] * self.ny + iy[:, None, :]).reshape(len(x), -1), (vx[:, :, None] * vy[:, None, :]).reshape(len(x), -1)


class _TimeLin:
    def __init__(self, spans, knot_yr, fit_ranges):
        self.bounds = [tuple(s) for s in spans]; spans = [tuple(r) for r in fit_ranges] if fit_ranges else self.bounds
        self.sets = [(lo, hi, _bspl_1d(np.array([lo]), lo, hi, knot_yr)[2]) for lo, hi in spans]
        self.knot_yr = knot_yr; self.nt = sum(nb for _, _, nb in self.sets); self.terms = [(1, 'x'), (1, 'y'), (4, 'x'), (4, 'y'), (4, '1')]; self.n = len(self.terms) * self.nt
    def rows(self, x, y, chip, yr):
        n = len(x); I, J, V = [], [], []; base = 0
        cut = [b[1] for b in self.bounds[:-1]]; span_of = np.searchsorted(np.asarray(cut), yr, side='right') if len(self.sets) > 1 else np.zeros(n, int)
        for isp, (lo, hi, nb) in enumerate(self.sets):
            m = span_of == isp
            if m.any():
                r = np.where(m)[0]; idx, val, _ = _bspl_1d(np.clip(yr[m], lo, hi), lo, hi, self.knot_yr)
                for k, (c, term) in enumerate(self.terms):
                    mc = chip[r] == c
                    if not mc.any(): continue
                    rr = r[mc]; u = {'x': (x[rr] - 2048) / 2048, 'y': (y[rr] - 1024) / 1024, '1': np.ones(len(rr))}[term]
                    I.append(np.repeat(rr, idx.shape[1])); J.append((idx[mc] + base + k * self.nt).ravel()); V.append((val[mc] * u[:, None]).ravel())
            base += nb
        if not I: return sp.csr_matrix((n, self.n))
        return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, self.n))


def _readout(x, y, chip):
    return ((np.where(chip == 1, y, 2048 - y) / 2048, np.where(chip == 1, 1.0, -1.0)), (np.minimum(x, 4096 - x) / 2048, np.where(x < 2048, 1.0, -1.0)))


class _CTEPar:
    def __init__(self, alpha, gamma, time_order, t_install, dm_ref=4.0, sky_ref=10.0):
        self.alphas = list(alpha) if np.ndim(alpha) else [alpha]; self.gamma, self.dm_ref, self.sky_ref = gamma, dm_ref, sky_ref
        self.time_order, self.t_install = time_order, t_install; self.nt = 1 if time_order < 0 else time_order + 1; self.n = 2 * self.nt * len(self.alphas)
    def rows(self, c):
        x, y, chip, t, dm = c['x'], c['y'], c['chip'], c['t'], c['dm']; n = len(x)
        sky = np.full(n, self.sky_ref) if c.get('lsky') is None else 10 ** np.nan_to_num(c['lsky'], nan=1.0)
        sk = (np.clip(sky, 0.1, None) / self.sky_ref) ** (-self.gamma); dmc = np.clip(dm, -3, 8) - self.dm_ref
        tf = [np.clip((t * 10.0 + 2016.0 - self.t_install) / 10.0, 0, None)] if self.time_order < 0 else [t ** i for i in range(self.nt)]
        return sp.csr_matrix(np.column_stack([sg * d * 10 ** (0.4 * al * dmc) * sk * f for d, sg in _readout(x, y, chip) for al in self.alphas for f in tf]))


class _Phase:
    def __init__(self, K): self.K = K; self.nf = (2 * K + 1) ** 2 - 1; self.n = 2 * self.nf
    def rows(self, c):
        n = len(c['x'])
        if c.get('phx') is None: return sp.csr_matrix((n, self.n))
        px = 2 * np.pi * c['phx']; py = 2 * np.pi * c['phy']
        fx = [np.ones(n)] + [f(k * px) for k in range(1, self.K + 1) for f in (np.cos, np.sin)]; fy = [np.ones(n)] + [f(k * py) for k in range(1, self.K + 1) for f in (np.cos, np.sin)]
        F = np.column_stack([a * b for i, a in enumerate(fx) for j, b in enumerate(fy) if i or j]); out = np.zeros((n, self.n)); c4 = c['chip'] == 4
        out[~c4, :self.nf] = F[~c4]; out[c4, self.nf:] = F[c4]; return sp.csr_matrix(out)


class _OrbitLin:
    def __init__(self, features): self.features = list(features); self.n = 3 * len(self.features)
    def rows(self, c):
        n = len(c['x'])
        if c.get('z') is None: return sp.csr_matrix((n, self.n))
        xp = (c['x'] - 2048) / 2048; yp = (c['y'] - 1024) / 1024; c4 = (c['chip'] == 4).astype(float); Z = c['z']
        return sp.csr_matrix(np.column_stack([b * Z[:, k] for k in range(Z.shape[1]) for b in (xp, yp, c4)]))


class _EraDelta:
    def __init__(self, bounds, h): self.bounds = list(bounds); self.S = _Surface(h); self.ne = len(self.bounds) + 1; self.n = self.ne * 2 * self.S.n
    def rows(self, c):
        x, y, chip, t = c['x'], c['y'], c['chip'], c['t']; n = len(x); yr = t * 10.0 + 2016.0
        era = np.searchsorted(np.asarray(self.bounds), yr, side='right'); idx, val = self.S.rows(x, y); col0 = (era * 2 + (chip == 4)) * self.S.n
        return sp.csr_matrix((val.ravel(), (np.repeat(np.arange(n), idx.shape[1]), (idx + col0[:, None]).ravel())), shape=(n, self.n))


class _Model:
    """column layout identical to basis_fit.Model: [S1|S4|M1|M4|TimeLin|extras...]"""
    def __init__(self, spec):
        self.S, self.M = _Surface(spec['h']), _Surface(spec['hm']); self.blocks = []; off = 0
        for name, s in (('S', self.S), ('M', self.M)):
            for c in (1, 4): self.blocks.append((name, c, off, s)); off += s.n
        self.timelin = _TimeLin(spec['spans'], spec['knot_yr'], spec.get('fit_ranges')); self.tl_off = off; off += self.timelin.n
        self.extras = []
        if spec.get('cte'): a_, g_, to_ = (list(spec['cte']) + [2])[:3]; self.extras.append(_CTEPar(a_, g_, to_, INSTALL.get(spec.get('key'))))
        if spec.get('phase_K'): self.extras.append(_Phase(spec['phase_K']))
        if spec.get('orbit_features'): self.extras.append(_OrbitLin(spec.get('orbit_feature_list', ORBIT_FEATURES)))
        if spec.get('era_delta'): self.extras.append(_EraDelta(spec['era_delta']['bounds'], spec['era_delta'].get('h', 512.0)))
        self.extra_off = []
        for b in self.extras: self.extra_off.append(off); off += b.n
        self.ncol = off
    def design(self, x, y, chip, t, dm, lsky=None, phx=None, phy=None, z=None, pin=True):
        n = len(x); I, J, V = [], [], []
        for name, c, off, s in self.blocks:
            m = chip == c
            if not m.any(): continue
            r = np.where(m)[0]; idx, val = s.rows(x[m], y[m]); w = 1.0 if name == 'S' else dm[m][:, None]
            I.append(np.repeat(r, idx.shape[1])); J.append((idx + off).ravel()); V.append((val * w).ravel())
        A = sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, self.ncol))
        TL = self.timelin.rows(x, y, chip, t * 10.0 + 2016.0)
        A = A + sp.hstack([sp.csr_matrix((n, self.tl_off)), TL, sp.csr_matrix((n, self.ncol - self.tl_off - TL.shape[1]))]).tocsr()
        ctx = dict(x=x, y=y, chip=chip, t=t, dm=dm, lsky=lsky, phx=phx, phy=phy, z=z)
        for b, o in zip(self.extras, self.extra_off):
            R = b.rows(ctx); A = A + sp.hstack([sp.csr_matrix((n, o)), R, sp.csr_matrix((n, self.ncol - o - b.n))]).tocsr()
        if pin: A = A - self.design(np.full(n, AX), np.full(n, AY), np.ones(n, int), t, np.zeros(n), pin=False)
        return A.tocsr()


# ---------------------------------------------------------------- orbit features from the JIT file ----
def orbit_features(hdr, jit_path, first_vf=1.0, t_in_visit_h=0.0):
    """9 scaled features for this exposure (zeros if the JIT track is unavailable). first_vf / t_in_visit_h
    need the visit context; the loader passes them when it has all exposures of the visit, else the defaults."""
    z = np.zeros(len(ORBIT_FEATURES))
    try:
        from astropy.io import fits
        import astropy.units as u
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, get_sun, get_body_barycentric_posvel
        root = str(hdr.get('ROOTNAME', ''))[:8].lower()
        with fits.open(jit_path, memmap=True) as hd:
            ext = None
            for i in range(1, len(hd)):
                if str(hd[i].header.get('EXPNAME', '')).lower()[:8] == root: ext = i; break
            if ext is None: return z, False
            d = hd[ext].data
            if d is None or len(d) < 3: return z, False
            lat = np.asarray(d['Latitude'], float); lon = np.asarray(d['Longitude'], float)
        t0 = float(hdr['EXPSTART']); t1 = float(hdr['EXPEND']); tm = 0.5 * (t0 + t1); m = len(lat) // 2
        def eci(la, lo, mjd):
            g = EarthLocation.from_geodetic(lo * u.deg, la * u.deg, 530.0 * u.km).get_gcrs(Time(mjd, format='mjd'))
            return np.array([g.cartesian.x.to_value(u.km), g.cartesian.y.to_value(u.km), g.cartesian.z.to_value(u.km)])
        r0, rm, r1 = eci(lat[0], lon[0], t0), eci(lat[m], lon[m], tm), eci(lat[-1], lon[-1], t1)
        s = get_sun(Time(tm, format='mjd')).cartesian; s = np.array([s.x.value, s.y.value, s.z.value]); s /= np.linalg.norm(s)
        psi = lambda r: np.degrees(np.arccos(np.clip(np.dot(r / np.linalg.norm(r), s), -1, 1)))
        th = np.deg2rad(psi(rm) if psi(r1) > psi(r0) else 360 - psi(rm))
        along = np.dot(rm, s); perp = np.linalg.norm(rm - along * s); shadow = float((along < 0) and (perp < 6378.137))
        ra = np.deg2rad(float(hdr.get('CRVAL1', hdr.get('RA_TARG')))); de = np.deg2rad(float(hdr.get('CRVAL2', hdr.get('DEC_TARG'))))
        p = np.array([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)])
        vr0 = (rm - r0) / max((tm - t0) * 86400, 0.5); vr1 = (r1 - rm) / max((t1 - tm) * 86400, 0.5); vh = 0.5 * (vr0 + vr1)
        ve = get_body_barycentric_posvel('earth', Time(tm, format='mjd'))[1]; ve = np.array([ve.x.to_value('km/s'), ve.y.to_value('km/s'), ve.z.to_value('km/s')])
        c = 299792.458; va_tot = np.dot(ve + vh, p) / c * 1e6; va_drift = (np.dot(vr1, p) - np.dot(vr0, p)) / 2 / c * 1e6
        if abs(va_tot) > 130 or abs(va_drift) > 30: va_tot = va_drift = 0.0
        vals = dict(cos_th=np.cos(th), sin_th=np.sin(th), cos_2th=np.cos(2 * th), sin_2th=np.sin(2 * th), shadow=shadow,
                    first_vf=float(first_vf), t_in_visit_h=float(t_in_visit_h), va_tot_ppm=va_tot, va_drift_ppm=va_drift)
        z = np.array([vals[f] * ORBIT_SCALE.get(f, 1.0) for f in ORBIT_FEATURES]); return z, True
    except Exception:
        return z, False


def find_jit(flc_path, hdr):
    """The exposure's JIT file under <field>/HST/mastDownload/HST/: <asn_id>/<asn_id>_jit.fits for associated
    exposures, <rootname>/<rootname[:8]>j_jit.fits for unassociated ones (ASN_ID = NONE), else any *_jit.fits in a
    directory starting with the visit id (the caller matches the extension by EXPNAME)."""
    root = str(hdr.get('ROOTNAME', '') or Path(flc_path).name[:9]).lower(); asn = str(hdr.get('ASN_ID', '') or '').lower()
    p = Path(flc_path).resolve(); mast = None
    for parent in p.parents:
        if (parent / 'mastDownload' / 'HST').is_dir(): mast = parent / 'mastDownload' / 'HST'; break
    if mast is None: return None
    cands = []
    if asn and asn != 'none': cands.append(mast / asn / f'{asn}_jit.fits')
    cands.append(mast / root / f'{root[:8]}j_jit.fits')
    for c in cands:
        if c.exists(): return c
    hits = sorted(glob.glob(str(mast / f'{root[:6]}*' / '*_jit.fits')))
    return Path(hits[0]) if hits else None


_CTX = {}   # mast root -> field orbit context (built once per process)


def field_orbit_context(mast):
    """Per-field context for the orbit features, mirroring hst_dist_corr/ml/orbit_features.py:
    an index EXPNAME[:8] -> (jit_path, ext) over EVERY *_jit.fits under <mast> (an exposure's ASN_ID often names a
    product association whose JIT holds only part of the visit, so the header ASN alone misses ~20% of exposures), and
    the visit context first_vf (first exposure by EXPSTART of its visit x filter, only if it has a jitter track) and
    t_in_visit_h (hours since the visit's first exposure; 0 without a track).  Cached per process."""
    mast = str(mast)
    if mast in _CTX: return _CTX[mast]
    from astropy.io import fits
    jit_idx = {}
    for jp in sorted(glob.glob(os.path.join(mast, '*', '*_jit.fits'))):
        try:
            with fits.open(jp, memmap=True) as hd:
                for i in range(1, len(hd)):
                    nm = str(hd[i].header.get('EXPNAME', '')).lower()[:8]
                    if nm and nm not in jit_idx and hd[i].data is not None and len(hd[i].data) >= 3: jit_idx[nm] = (jp, i)
        except Exception:
            continue
    rows = []
    for fp in sorted(glob.glob(os.path.join(mast, '*', '*_flc.fits'))):
        try:
            h = fits.getheader(fp, 0)
        except Exception:
            continue
        root = str(h.get('ROOTNAME', os.path.basename(fp)[:9])).lower()
        f = h.get('FILTER')
        if not f:
            f1, f2 = str(h.get('FILTER1', '')), str(h.get('FILTER2', ''))
            f = f1 if (f1.startswith('F') and 'CLEAR' not in f1) else f2
        rows.append((root, root[:6], str(f), float(h.get('EXPSTART') or np.inf), root[:8] in jit_idx))
    rows.sort(key=lambda r: r[3])
    seen_vf, vmin, ctx = set(), {}, {}
    for root, visit, filt, t0, has in rows:
        vmin.setdefault(visit, t0)
        first = (visit, filt) not in seen_vf; seen_vf.add((visit, filt))
        ctx[root] = dict(first_vf=float(first and has), t_in_visit_h=((t0 - vmin[visit]) * 24.0 if has and np.isfinite(t0) else 0.0))
    n_has = sum(r[4] for r in rows)
    print(f"  pos_corr orbit context [{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mast))))}]: {len(rows)} FLCs, "
          f"{len(jit_idx)} exposures indexed from {len(set(v[0] for v in jit_idx.values()))} JIT files, "
          f"{n_has} FLCs with a jitter track ({len(rows) - n_has} without -> orbit terms at 0)", flush=True)
    _CTX[mast] = (jit_idx, ctx); return _CTX[mast]


# ---------------------------------------------------------------- the applier ----
class PosCorrBasis:
    def __init__(self, spec: str):
        """spec = 'DIR' (all basis_*.json in DIR), 'DIR:TAG', or 'DIR:TAG:noorbit' (orbit-conditioned terms left out;
        for tests of the rest of the correction while the orbit block is being validated)."""
        parts = str(spec).split(':'); d = parts[0]; tag = parts[1] if len(parts) > 1 else ''; self.use_orbit = not (len(parts) > 2 and parts[2] == 'noorbit')
        files = sorted(glob.glob(os.path.join(d, f'basis_*_{tag or "*"}.json')))
        if not files: raise FileNotFoundError(f'pos_corr_basis: no basis_*.json in {spec}')
        self.models = {}
        for f in files:
            s = json.load(open(f)); z = np.load(f[:-5] + '.npz')
            self.models[(s['key'], s['filter'])] = (_Model(s), z['bx'], z['by'], s)
        normf = os.path.join(d, 'normalization.json')
        self.mag0 = {k: v['inst_mag0'] for k, v in json.load(open(normf))['keys'].items()} if os.path.exists(normf) else {}
        self.dir = d

    @property
    def summary(self):
        return f'{len(self.models)} basis models from {self.dir}' + ('' if self.use_orbit else ' (orbit terms OFF)') + ': ' + ', '.join(f'{k[0]}/{k[1]}' for k in sorted(self.models))

    @staticmethod
    def read_header(flc_path) -> dict:
        from astropy.io import fits
        with fits.open(flc_path, memmap=True) as hd:
            h0 = hd[0].header; h1 = hd[1].header if len(hd) > 1 else {}
        out = {k: h0.get(k) for k in HDR_KEYS0}
        for k in HDR_KEYS1:
            if h1.get(k) is not None: out[k] = h1.get(k)
        out['_flc_path'] = str(flc_path); return out

    def _filter(self, hdr):
        f = hdr.get('FILTER')
        if not f:
            f1, f2 = str(hdr.get('FILTER1', '')), str(hdr.get('FILTER2', ''))
            f = f1 if (f1.startswith('F') and 'CLEAR' not in f1) else f2
        return str(f)

    def orbit_row(self, hdr, orbit=None):
        """(9 scaled orbit features, ok) for one exposure: JIT track found through the field index (fallback: the
        header's ASN), visit context from the field unless `orbit` (dict first_vf, t_in_visit_h) overrides it."""
        flc = str(hdr.get('_flc_path', '')); root = str(hdr.get('ROOTNAME', '') or os.path.basename(flc)[:9]).lower()
        p = Path(flc).resolve(); mast = None
        for parent in p.parents:
            if (parent / 'mastDownload' / 'HST').is_dir(): mast = parent / 'mastDownload' / 'HST'; break
        jit, vctx = None, {}
        if mast is not None:
            jit_idx, ctx = field_orbit_context(mast)
            hit = jit_idx.get(root[:8]); jit = hit[0] if hit else None; vctx = ctx.get(root, {})
        if jit is None: jit = find_jit(flc, hdr)
        kw = dict(vctx); kw.update(orbit or {})
        if jit is None: return np.zeros(len(ORBIT_FEATURES)), False
        return orbit_features(hdr, jit, **{k: kw[k] for k in ('first_vf', 't_in_visit_h') if k in kw})

    def bias(self, tbl, hdr, orbit=None):
        """(bias_x, bias_y) in GDC px per catalog row; corrected = x_gdc - bias.  orbit: optional dict(first_vf, t_in_visit_h)."""
        inst, det = str(hdr.get('INSTRUME')), str(hdr.get('DETECTOR')); key = f'{inst}/{det}'; filt = self._filter(hdr)
        mdl = self.models.get((key, filt))
        if mdl is None: return None
        model, bx, by, spec = mdl; n = len(tbl)
        x = np.asarray(tbl['x'], float); y = np.asarray(tbl['y'], float)
        chip = np.asarray(tbl['chip_ext'], int) if 'chip_ext' in tbl.dtype.names else np.where(y >= 2048, 4, 1)
        y_chip = np.where(chip == 4, y - (2051.0 if inst == 'WFC3' else 2048.0), y)              # as extract_header_frame
        m0 = self.mag0.get(f'{key}/{filt}', np.nan); dm = np.clip(np.asarray(tbl['mag'], float) - m0, -3, 8)
        mjd = 0.5 * (float(hdr['EXPSTART']) + float(hdr.get('EXPEND') or hdr['EXPSTART'])); year = 2000.0 + (mjd - 51544.5) / 365.25
        t = np.full(n, (year - 2016.0) / 10.0)
        lsky = np.log10(np.clip(np.nan_to_num(np.asarray(tbl['sky'], float), nan=1.0), 0.1, None)) if 'sky' in tbl.dtype.names else None
        phx, phy = np.mod(x, 1.0), np.mod(y, 1.0)                                                   # as extract_header_frame
        z = None
        if spec.get('orbit_features') and self.use_orbit:
            zrow, ok = self.orbit_row(hdr, orbit)
            feats = spec.get('orbit_feature_list') or ORBIT_FEATURES; zrow = np.array([zrow[ORBIT_FEATURES.index(f)] for f in feats])
            z = np.tile(zrow, (n, 1))
        A = model.design(x, y_chip, chip, t, dm, lsky=lsky, phx=phx, phy=phy, z=z)
        return A @ bx, A @ by


def make_pos_corr(spec: str):
    """factory used by the loaders: a basis model dir (basis_*.json) or an MLP model dir (*_shared_*.json)."""
    d = str(spec).partition(':')[0]
    if glob.glob(os.path.join(d, 'basis_*.json')):
        return PosCorrBasis(spec)
    from bp3m.pos_corr_model import PosCorrModel
    return PosCorrModel(spec)
