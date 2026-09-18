"""Learned GDC-residual correction (hst_dist_corr/ml Stage 2) evaluated per
detection at catalog-load time — the model sibling of pos_corr.PseudoGDCSet.

A model directory holds, per instrument/detector, <INST>_<DET>_shared_<tag>.json
(feature spec, filters, robust-scaling constants, inst_mag0 per filter,
time_deg) and .npz (MLP weights) written by hst_dist_corr/ml/train_mlp.py.
Convention: f = model(features) is the Anderson-GDC bias (measured - true), so
the corrected position is x_gdc - f.  f is pinned to zero at raw (2048, 1024)
on ext 1 for a star at inst_mag0 under the same observing conditions.
This file re-implements the trainer's feature construction and forward pass
in numpy (JAX gelu = tanh approximation) so bp3m does not depend on JAX.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np

ANCHOR_X, ANCHOR_Y = 2048.0, 1024.0
OUT_SCALE = 0.05
T_COL = 4
BASE = ['xn', 'yn_chip', 'chip4', 'dmag5', 't10', 'pctefrac', 'ychip', 'phx', 'phy', 'lsky', 'lexp']
HDR_KEYS0 = ('EXPTIME', 'PCTEFRAC', 'SUN_ALT', 'PA_V3', 'CCDGAIN', 'FLASHDUR', 'EXPSTART', 'INSTRUME', 'DETECTOR', 'FILTER', 'FILTER1', 'FILTER2', 'VAFACTOR')
HDR_KEYS1 = ('VAFACTOR',)


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class PosCorrModel:
    def __init__(self, spec: str):
        """spec = 'DIR' (all *_shared_*.json in DIR), 'DIR:TAG', or comma-separated json paths."""
        spec = str(spec)
        if ',' in spec or spec.endswith('.json'):
            files = [p.strip() for p in spec.split(',') if p.strip()]
        else:
            d, _, tag = spec.partition(':')
            files = sorted(glob.glob(os.path.join(d, f'*_shared_{tag or "*"}.json')))
        if not files:
            raise FileNotFoundError(f'pos_corr_model: no model json found for {spec}')
        # several models per inst/det (e.g. seeds of an ensemble) are averaged
        self.models = {}
        for jp in files:
            s = json.load(open(jp)); z = np.load(jp[:-5] + '.npz')
            nl = len([k for k in z.files if k.startswith('w')])
            s['params'] = [(np.asarray(z[f'w{i}'], np.float64), np.asarray(z[f'b{i}'], np.float64)) for i in range(nl)]
            s['scales'] = {k: tuple(v) for k, v in s.get('scales', {}).items()}
            s['path'] = jp
            self.models.setdefault(s['key'], []).append(s)

    @property
    def summary(self):
        return ', '.join(f"{k}(x{len(v)}: {os.path.basename(v[0]['path'])}{'...' if len(v) > 1 else ''})" for k, v in self.models.items())

    def match(self, instrument: str, detector: str):
        return self.models.get(f'{instrument}/{detector}')

    @staticmethod
    def read_header(flc_path) -> dict:
        from astropy.io import fits
        with fits.open(flc_path, memmap=True) as hd:
            h0 = hd[0].header; h1 = hd[1].header if len(hd) > 1 else {}
        out = {k: h0.get(k) for k in HDR_KEYS0}
        for k in HDR_KEYS1:
            if h1.get(k) is not None: out[k] = h1.get(k)
        return out

    # ---- features (mirror of hst_dist_corr/ml/train_mlp.build_features / extra_columns) ----
    def _features(self, s, tbl, hdr):
        n = len(tbl)
        inst = str(hdr.get('INSTRUME')); det = str(hdr.get('DETECTOR'))
        filt = hdr.get('FILTER') or (hdr.get('FILTER1') if str(hdr.get('FILTER1', '')).startswith('F') else hdr.get('FILTER2'))
        key = f'{inst}/{det}'
        m0map = s.get('inst_mag0'); m0 = m0map.get(f'{key}/{filt}') if isinstance(m0map, dict) else m0map
        if m0 is None:
            vals = [v for v in m0map.values()] if isinstance(m0map, dict) else []
            m0 = float(np.median(vals)) if vals else 0.0
        x = np.asarray(tbl['x'], float); y = np.asarray(tbl['y'], float)
        chip = np.asarray(tbl['chip_ext'], int) if 'chip_ext' in tbl.dtype.names else np.where(y >= 2048, 4, 1)
        y_chip = np.where(chip == 4, y - (2051.0 if inst == 'WFC3' else 2048.0), y)
        chip4 = (chip == 4).astype(float)
        dmag = np.clip(np.asarray(tbl['mag'], float) - float(m0), -3, 8)
        mjd = float(hdr.get('EXPSTART') or 51544.5); t = (2000.0 + (mjd - 51544.5) / 365.25 - 2016.0) / 10.0
        pcte = float(hdr.get('PCTEFRAC') or 0.0) if hdr.get('PCTEFRAC') not in (None, '') else 0.0
        exptime = float(hdr.get('EXPTIME') or 100.0)
        sky = np.asarray(tbl['sky'], float)
        ph = np.column_stack([np.mod(x, 1.0) - 0.5, np.mod(y, 1.0) - 0.5])
        lsky = np.log10(np.clip(np.nan_to_num(sky, nan=1.0), 0.1, None)) / 3.0
        lexp = np.full(n, np.log10(max(exptime, 0.1)) / 3.0)
        filters = s.get('filters')
        F = None
        if filters is not None:
            fidx = filters.index(filt) if filt in filters else len(filters)
            F = np.zeros((n, len(filters) + 1)); F[:, fidx] = 1.0
        groups = s.get('feature_groups', [])
        extra_names = [c for g in groups for c in FEATURE_GROUPS.get(g, [])]
        E = None
        if extra_names:
            raw = self._extra_columns(tbl, hdr, n)
            cols = []
            for c in extra_names:
                med, mad = s['scales'].get(c, (0.0, 1.0))
                z = (raw.get(c, np.full(n, np.nan)) - med) / (mad if mad else 1.0)
                cols.append(np.clip(np.nan_to_num(z, nan=0.0, posinf=5, neginf=-5), -5, 5))
            E = np.column_stack(cols)

        ychip = y_chip / 2048.0      # CTE transfer distance of the STAR: kept in the anchor row too (as in the trainer)

        def pack(xx, yy, ch4, dm):
            cols = [(xx - 2048) / 2048, (yy - 1024) / 1024, ch4, dm / 5.0, np.full(n, t), np.full(n, pcte), ychip, ph[:, 0], ph[:, 1], lsky, lexp]
            if F is not None: cols.append(F)
            if E is not None: cols.append(E)
            return np.column_stack(cols)
        Z = pack(x, y_chip, chip4, dmag)
        Za = pack(np.full(n, ANCHOR_X), np.full(n, ANCHOR_Y), np.zeros(n), np.zeros(n))
        names = list(BASE) + ([f'filt_{f}' for f in filters] + ['filt_other'] if filters is not None else []) + extra_names
        if names != s['features']:
            raise RuntimeError(f"pos_corr_model feature mismatch for {key}: {names} vs {s['features']}")
        return Z, Za

    @staticmethod
    def _extra_columns(tbl, hdr, n):
        out = {}
        def col(name, default=np.nan):
            return np.asarray(tbl[name], float) if name in tbl.dtype.names else np.full(n, default)
        out['logflux'] = np.log10(np.clip(np.nan_to_num(col('flux'), nan=1.0), 1, None))
        for c in ('qfit', 'psf_frac', 'chi2', 'n_neighbors', 'dist_nearest', 'dist_nearest_brighter', 'sky'):
            out[c] = col(c)
        vaf = float(hdr.get('VAFACTOR') or 1.0); out['vafactor'] = np.full(n, (vaf - 1.0) * 1e4)
        out['sun_alt'] = np.full(n, float(hdr.get('SUN_ALT') if hdr.get('SUN_ALT') is not None else np.nan))
        pa = np.deg2rad(float(hdr.get('PA_V3') or 0.0)); out['pa_v3_sin'] = np.full(n, np.sin(pa)); out['pa_v3_cos'] = np.full(n, np.cos(pa))
        out['ccdgain'] = np.full(n, float(hdr.get('CCDGAIN') if hdr.get('CCDGAIN') is not None else np.nan))
        out['flashdur'] = np.full(n, float(hdr.get('FLASHDUR') if hdr.get('FLASHDUR') is not None else np.nan))
        return out

    # ---- forward pass ----
    @staticmethod
    def _mlp(s, z):
        td = int(s.get('time_deg', 0))
        if td > 0:
            t = z[:, T_COL:T_COL + 1]; zz = np.concatenate([z[:, :T_COL], z[:, T_COL + 1:]], axis=1)
        else:
            zz = z
        h = zz
        for w, b in s['params'][:-1]:
            h = _gelu(h @ w + b)
        w, b = s['params'][-1]
        out = (h @ w + b) * OUT_SCALE
        if td > 0:
            acc = out[:, 0:2].copy()
            for k in range(1, td + 1):
                acc = acc + out[:, 2 * k:2 * k + 2] * (t ** k)
            return acc
        return out

    def bias(self, tbl, hdr):
        """(bias_x, bias_y) in GDC px for every catalog row; corrected = x_gdc - bias.
        With several models for the inst/det (ensemble) the mean prediction is used."""
        ms = self.match(str(hdr.get('INSTRUME')), str(hdr.get('DETECTOR')))
        if not ms:
            return None
        acc = None
        for s in ms:
            Z, Za = self._features(s, tbl, hdr)
            f = self._mlp(s, Z) - self._mlp(s, Za)
            acc = f if acc is None else acc + f
        f = acc / len(ms)
        return f[:, 0], f[:, 1]


FEATURE_GROUPS = {
    'flux':   ['logflux', 'qfit', 'psf_frac', 'chi2'],
    'crowd':  ['n_neighbors', 'dist_nearest', 'dist_nearest_brighter'],
    'sky':    ['sky'],
    'tel':    ['vafactor', 'sun_alt', 'pa_v3_sin', 'pa_v3_cos', 'ccdgain', 'flashdur'],
    'jitter': ['jif_V2_RMS', 'jif_V3_RMS', 'jif_V23_RMS_mas', 'jif_GSSEPRMS', 'jit_SI_combined_total_RMS_arcsec', 'jit_Roll_RMS_mas', 'jit_missing'],
    'focus':  ['img_psf_frac_bright', 'img_qfit_bright', 'img_chi2_bright', 'img_psf_frac_med', 'img_qfit_med'],
    'density': ['n_labels_log', 'img_nneigh_med', 'img_dnear_med', 'img_sky_med'],
    'header': [],
}
