"""CTE v2 (pop-fit phase) on a synthetic linear field: the exact Newton solve recovers an injected CTE pattern through the
REAL production pop-fit solve (run_pop_fit._joint_solve_pop), with and without the Gaia field-mode block (plan P4), in
free-r and frozen-r mode; the legacy fixed-point loop's linearised map is measured on the same data.

Synthetic model (pseudo-image pixels; Gaia epoch 2016):
  sky(t) = xys + JU v,  JU = [[1, 0, dt, 0, 0], [0, 1, 0, dt, 0]]      (v = position offset, PM, parallax)
  X(x_c, y_c) r = sky(t)  for the true detector positions (a, b, c, d = roll, pointing shift, 8p chip offset)
  measured raw position = true + D_raw gamma_true  (the CTE v2 basis: yt*[1,xt] x [1,m,m^2] x [1,tau])
"""
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:                      # the bp3m env has no pytest: run this file directly (see __main__)
    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            assert et is not None and issubclass(et, self.exc), f'expected {self.exc.__name__}'
            return True

    class pytest:                         # noqa: N801  minimal shim
        raises = _Raises

        class mark:
            @staticmethod
            def parametrize(name, vals):
                def deco(fn):
                    fn._params = (name, vals)
                    return fn
                return deco

from bp3m.astro_utils import build_X_matrices
from bp3m.pipeline import cte_popfit_v2 as cte
from bp3m.pipeline.run_pop_fit import _joint_solve_pop

SIG_PM = 0.002            # population PM prior (px/yr)
SIG_PLX = 1e-3


class FakeSolver:
    """The attributes _joint_solve_pop and cte_popfit_v2 read, for a linear synthetic field."""

    def __init__(self, seed=1, n_star=700, n_member=350, n_gz=0, fit_chip_offset=True, sig_chip=0.01, sig_lin=5e-5,
                 noise=0.02, epochs=(2003.2, 2003.25, 2012.5, 2012.6, 2020.3, 2020.4, 2024.1, 2024.2), gamma_true=None,
                 time_order=1, group_by='chip', filters=None):
        rng = np.random.default_rng(seed)
        self.poly_order = 1; self.fit_chip_offset = fit_chip_offset
        self.N_R = 6 + (2 if fit_chip_offset else 0)
        self.n_stars = n_star
        self.member = np.zeros(n_star, bool); self.member[:n_member] = True
        self.mu_true = np.array([0.004, -0.003])
        pm = np.where(self.member[:, None], self.mu_true + 0.0005 * rng.standard_normal((n_star, 2)),
                      0.01 * rng.standard_normal((n_star, 2)))
        v_true = np.zeros((n_star, 5)); v_true[:, 2:4] = pm
        v_true[:, :2] = 0.004 * rng.standard_normal((n_star, 2))          # true position minus Gaia catalogue position
        sig_g = np.c_[np.full(n_star, 0.004), np.full(n_star, 0.004), np.full(n_star, 0.006), np.full(n_star, 0.006),
                      np.full(n_star, 0.003)] * (1 + rng.random((n_star, 1)))
        C_g = np.zeros((n_star, 5, 5)); C_g[:, range(5), range(5)] = sig_g ** 2
        v_obs = v_true + sig_g * rng.standard_normal((n_star, 5))
        self.n_gz = n_gz
        Mmean = np.zeros((n_star, 5))
        if n_gz:
            Wm = np.zeros((n_star, 5, n_gz))
            Wm[:, 2, 0] = 0.003; Wm[:, 3, 0] = -0.002                            # near-uniform PM shift mode
            if n_gz > 1:
                Wm[:, 2, 1] = 0.002 * rng.standard_normal(n_star); Wm[:, 3, 1] = 0.002 * rng.standard_normal(n_star)
            self.z_true = rng.standard_normal(n_gz)
            v_obs = v_obs + np.einsum('nik,k->ni', Wm, self.z_true)
            Ci = np.linalg.inv(C_g)
            self.gaia_gz_rows = np.arange(n_star)
            self.gaia_Kz = -np.einsum('nij,njk->nik', Ci, Wm)
            self.gaia_WCW = np.einsum('nik,nij,njl->kl', Wm, Ci, Wm)
            self.gaia_bz = np.einsum('nik,nij,nj->k', Wm, Ci, v_obs - Mmean)
            self.gaia_z = np.zeros(n_gz); self.gaia_z_cov = np.eye(n_gz)
        self.C_survey_inv = np.linalg.inv(C_g)
        self.C_survey_inv_dot_v = np.einsum('nij,nj->ni', self.C_survey_inv, v_obs - Mmean)
        self._C_VG_inv_per_star = np.zeros((n_star, 5))
        self.v_true = v_true
        sky0 = rng.uniform(-1800, 1800, (n_star, 2))                              # Gaia catalogue positions (px)
        mag = rng.uniform(-14.5, -9.0, n_star)
        filters = filters or ['F606W', 'F814W']
        self.images, self._img_data, self.image_names = {}, {}, []
        self.R = {}
        self.spi = {}
        r_true = []
        self.gamma_true = gamma_true
        for j, ep in enumerate(epochs):
            img = f'j{j:02d}xxxxq'
            th = np.deg2rad(rng.uniform(0, 360)); sc = 1.0 + 1e-4 * rng.standard_normal()
            a, b, c, d = sc * np.cos(th), sc * np.sin(th), -sc * np.sin(th), sc * np.cos(th)
            M = np.array([[a, b], [c, d]]); sh = rng.uniform(-300, 300, 2)
            chip = 0.01 * rng.standard_normal(2) if fit_chip_offset else np.zeros(0)
            dt = ep - 2016.0
            JU = np.zeros((n_star, 2, 5)); JU[:, 0, 0] = 1; JU[:, 1, 1] = 1; JU[:, 0, 2] = dt; JU[:, 1, 3] = dt
            sky = sky0 + np.einsum('nij,nj->ni', JU, v_true)
            xy = np.linalg.solve(M, (sky - sh).T).T                                 # true GDC-centred detector position
            hi = xy[:, 1] > 0
            if fit_chip_offset:
                xy = xy - np.linalg.solve(M, chip)[None, :] * hi[:, None]
            ok = (np.abs(xy[:, 0]) < 2040) & (np.abs(xy[:, 1]) < 2040)
            sidx = np.flatnonzero(ok); n = len(sidx); xy = xy[ok]; hi = hi[ok]
            meta = dict(instrument='ACS', detector='WFC', hst_time_mjd=51544.5 + (ep - 2000.0) * 365.25,
                        filter=filters[j % len(filters)], orig_rot_deg=np.rad2deg(th))
            self.images[img] = meta; self.image_names.append(img)
            X_orig = xy[:, 0] + 2048.0; Y_orig = xy[:, 1] + 2048.0
            df = pd.DataFrame(dict(Gaia_id=sidx.astype(np.int64) + 10, X_orig=X_orig, Y_orig=Y_orig, mag=mag[sidx]))
            self.spi[img] = df
            xc = xy[:, 0].copy(); yc = xy[:, 1].copy()
            if gamma_true is not None:                                        # inject CTE: measured = true + Jg D gamma
                p, grp, nf = self._cte_geom(img, meta, X_orig, Y_orig, mag[sidx], time_order, group_by)
                draw = p
                xc = xc + draw[:, 0]; yc = yc + draw[:, 1]
            xc = xc + noise * rng.standard_normal(n); yc = yc + noise * rng.standard_normal(n)
            ptg = (np.ones(n), np.zeros(n), np.zeros(n), np.ones(n))
            X = build_X_matrices(xc, yc, *ptg, poly_order=1)
            if fit_chip_offset:
                X = self._append_chip_cols(X, hi.astype(float))
            r_j = np.r_[a, b, c, d, sh, chip]
            r_true.append(r_j)
            C_hst = np.tile(noise ** 2 * np.eye(2), (n, 1, 1))
            self.R[img] = M
            pri = np.zeros((self.N_R, self.N_R)); pri[:4, :4] = np.eye(4) / sig_lin ** 2
            pri[4, 4] = pri[5, 5] = 1.0 / 100.0 ** 2
            if fit_chip_offset:
                pri[6, 6] = pri[7, 7] = 1 / sig_chip ** 2
            r_prior = r_j.copy(); r_prior[4:6] += 0.3 * rng.standard_normal(2)
            if fit_chip_offset:
                r_prior[6:] = 0.0
            self._img_data[img] = dict(n=n, sidx=sidx, use_for_fit=np.ones(n, bool), use_for_astrom=np.ones(n, bool),
                                       JU=JU[sidx], X_mat=X, xys=sky0[sidx], X_c=xc, Y_c=yc, C_hst=C_hst,
                                       C_r_prior_inv=pri, r_prior=r_prior, chip_hi=hi.astype(float))
        self.r_true = np.concatenate(r_true)
        self.star_id_to_idx = {int(i) + 10: int(i) for i in range(n_star)}

    def _cte_geom(self, img, meta, X_orig, Y_orig, mag, time_order, group_by):
        """CTE displacement of the injected gamma (raw = GDC frame here, Jg = I)."""
        cam = cte._CAMS[('ACS', 'WFC')]
        hi = Y_orig > cam[1]
        yt = np.clip(np.abs(Y_orig - np.where(hi, 4096.0, 0.0)) / 2048.0, 0, 1.1)
        xt = (X_orig - 2048.0) / 2048.0
        mp = (mag - (-11.75)) / 1.6
        tau = np.full(len(mag), (cte._year(meta['hst_time_mjd']) - cam[2]) / 10.0)
        F = cte._basis(yt, xt, mp, tau, time_order); nf = F.shape[1]
        gt = self.gamma_true                                                   # dict group -> (gx, gy)
        out = np.zeros((len(mag), 2))
        for k in range(len(mag)):
            g = f'ACS{"1" if hi[k] else "2"}'
            if group_by == 'filter':
                g = f"{g}_{meta['filter']}"
            gx, gy = gt.get(g, (np.zeros(nf), np.zeros(nf)))
            out[k] = [F[k] @ gx, F[k] @ gy]
        return out, None, nf

    def _append_chip_cols(self, X_mat, chip_hi):
        n = X_mat.shape[0]
        X_mat = np.concatenate([X_mat, np.zeros((n, 2, 2))], axis=2)
        if chip_hi is not None:
            X_mat[:, 0, self.N_R - 2] = chip_hi; X_mat[:, 1, self.N_R - 1] = chip_hi
        return X_mat

    def _compute_Cs(self, img, r_j=None):
        R = self.R[img]
        return R @ self._img_data[img]['C_hst'] @ R.T

    def _ed_disp(self, img, shared_vec):
        return 0.0


def _solve_fn(s):
    msidx = np.flatnonzero(s.member)

    def f(m, mu, r, fx, z, uz=True):
        return _joint_solve_pop(s, s.image_names, m, mu, SIG_PM, 0.0, SIG_PLX, np.eye(2) / 0.5 ** 2, np.zeros(2), r,
                                fix_r=fx, z_weights=z, update_z=uz)
    return f, msidx


def _converge(s, n=4):
    f, m = _solve_fn(s)
    r = np.concatenate([s._img_data[i]['r_prior'] for i in s.image_names]); mu = np.zeros(2)
    for _ in range(n):
        r, mu, C, CvT, a, _, _ = f(m, mu, r, False, None)
    return f, m, r, mu


def _gamma_true(nf, ycoef):
    gx = np.zeros(nf); gy = np.zeros(nf); gy[:len(ycoef)] = ycoef
    return gx, gy


TO = 1
NF = 12
# y block (readout direction): yt (= 0 at readout) and its magnitude terms, constant + growing in time
G_Y = np.zeros(NF); G_Y[0] = 0.02; G_Y[1] = 0.015; G_Y[6] = 0.03; G_Y[7] = 0.02


def _field(**kw):
    gt = {'ACS1': (np.zeros(NF), G_Y), 'ACS2': (np.zeros(NF), 0.7 * G_Y)}
    return FakeSolver(gamma_true=gt, time_order=TO, **kw)


@pytest.mark.parametrize('n_gz', [0, 2])
def test_newton_recovers_injected_cte(n_gz):
    s = _field(n_gz=n_gz, seed=3)
    f, m, r, mu = _converge(s)
    mu_nocte = mu.copy()
    r2, mu2, C, CvT, a, info = cte.run_cte_v2(s, s.image_names, s.spi, s.star_id_to_idx, f, m, mu, r, None,
                                               sigma_pm=SIG_PM, sigma_plx_tot=SIG_PLX, C_pop_prior_inv=np.eye(2) / 0.25,
                                               fix_r=False, time_order=TO, group_by='chip', sigma_gamma=0.05)
    g = np.array(info['gamma']); sg = np.array(info['sigma_gamma'])
    gt = np.r_[np.zeros(NF), G_Y, np.zeros(NF), 0.7 * G_Y]           # groups sorted: ACS1, ACS2
    pull = (g - gt) / sg
    assert info['newton_history'][-1]['max_step_over_sigma_prior'] < 1e-3
    assert np.abs(pull).max() < 4.5, pull
    assert 0.6 < np.std(pull) < 1.6, np.std(pull)
    # the well-measured injected terms are recovered at high significance
    for k in (NF + 0, NF + 6, 3 * NF + 6):
        assert abs(g[k] - gt[k]) < 3 * sg[k] and sg[k] < 0.3 * abs(gt[k]), (k, g[k], gt[k], sg[k])
    # mu_pop moves towards the truth, within its (CTE-inflated) uncertainty
    smu = np.sqrt(np.diag(C[-2:, -2:]) + np.array(info['sigma_mu_cte']) ** 2)
    assert np.all(np.abs(mu2 - s.mu_true) < 4 * smu), (mu2, s.mu_true, smu)
    assert np.linalg.norm(mu2 - s.mu_true) <= np.linalg.norm(mu_nocte - s.mu_true) + 1e-4
    if n_gz:
        assert info['n_gaia_modes'] == n_gz and np.all(np.isfinite(s.gaia_z))


def test_newton_null_and_bitwise_positions():
    """No injected CTE: gamma consistent with 0; positions with gamma = 0 rebuild the solver design exactly."""
    s = FakeSolver(seed=5, gamma_true=None, time_order=TO)
    X0 = {i: s._img_data[i]['X_mat'].copy() for i in s.image_names}
    f, m, r, mu = _converge(s)
    per, groups, nf, _, _ = cte._setup(s, s.image_names, s.spi, s.star_id_to_idx, None, TO, 'chip')
    cm = ((np.arange(len(groups) * 2 * nf) % nf) < nf // 2).astype(float)
    ctx = cte._NewtonCTE(s, s.image_names, per, groups, nf, TO, f, m, mu, r, False, None, cm)
    ctx.shift_positions(np.zeros(ctx.ngam))
    for i in s.image_names:
        assert np.array_equal(s._img_data[i]['X_mat'], X0[i])
    _, _, _, _, _, info = cte.run_cte_v2(s, s.image_names, s.spi, s.star_id_to_idx, f, m, mu, r, None,
                                         sigma_pm=SIG_PM, sigma_plx_tot=SIG_PLX, C_pop_prior_inv=np.eye(2) / 0.25,
                                         time_order=TO, group_by='chip')
    pull = np.array(info['gamma']) / np.array(info['sigma_gamma'])
    assert np.abs(pull).max() < 4.5 and np.std(pull) < 1.6


def test_newton_frozen_frame_and_filter_groups():
    """Frozen r (frame follows through the prior-weighted alignment response) + hierarchical per-filter groups."""
    gt = {'ACS1_F606W': (np.zeros(NF), G_Y), 'ACS1_F814W': (np.zeros(NF), 0.8 * G_Y),
          'ACS2_F606W': (np.zeros(NF), 0.7 * G_Y), 'ACS2_F814W': (np.zeros(NF), 0.6 * G_Y)}
    s = FakeSolver(seed=7, gamma_true=gt, time_order=TO, group_by='filter')
    f, m, r, mu = _converge(s)
    r2, mu2, C, CvT, a, info = cte.run_cte_v2(s, s.image_names, s.spi, s.star_id_to_idx, f, m, mu, r, None,
                                               sigma_pm=SIG_PM, sigma_plx_tot=SIG_PLX, C_pop_prior_inv=np.eye(2) / 0.25,
                                               fix_r=True, time_order=TO, group_by='filter', sigma_gamma=0.05,
                                               sigma_filter=0.02)
    assert info['groups'] == sorted(gt)
    assert info['newton_history'][-1]['max_step_over_sigma_prior'] < 1e-3
    g = np.array(info['gamma']); sg = np.array(info['sigma_gamma'])
    gtv = np.concatenate([np.r_[np.zeros(NF), gt[k][1]] for k in sorted(gt)])
    pull = (g - gtv) / sg
    assert np.abs(pull).max() < 5, pull
    # the frame moved with the correction (frozen-r mode)
    assert not np.allclose(r2, r)


def test_legacy_fixed_point_map_measured():
    """diag=True measures the legacy loop's linearised map; the Newton result does not depend on it."""
    s = _field(seed=11)
    f, m, r, mu = _converge(s)
    _, _, _, _, _, info = cte.run_cte_v2(s, s.image_names, s.spi, s.star_id_to_idx, f, m, mu, r, None,
                                         sigma_pm=SIG_PM, sigma_plx_tot=SIG_PLX, C_pop_prior_inv=np.eye(2) / 0.25,
                                         time_order=TO, group_by='chip', diag=True)
    ident = info['identifiability']
    assert 'fixed_point_spectral_radius' in ident and np.isfinite(ident['fixed_point_spectral_radius'])
    assert len(ident['info_ratio_free_over_fixed_frame']) == len(info['gamma'])
    print('legacy fixed-point spectral radius (synthetic):', ident['fixed_point_spectral_radius'])


@pytest.mark.parametrize('fix_r', [False, True])
def test_fast_conditioning_equals_resolve(fix_r):
    """Star posteriors moved to the solved (r, mu, z) by the solve's linear response == a fixed-r re-solve there."""
    s = _field(n_gz=2, seed=13)
    f, m, r, mu = _converge(s)
    per, groups, nf, _, _ = cte._setup(s, s.image_names, s.spi, s.star_id_to_idx, None, TO, 'chip')
    cm = ((np.arange(len(groups) * 2 * nf) % nf) < nf // 2).astype(float)
    g = 0.01 * np.random.default_rng(0).standard_normal(len(groups) * 2 * nf)
    out = []
    for fast in (True, False):
        ctx = cte._NewtonCTE(s, s.image_names, per, groups, nf, TO, f, m, mu + 1e-3, r, fix_r, None, cm, sigma_pm=SIG_PM,
                             fast_condition=fast)
        out.append(ctx.evaluate(g))
        ctx.shift_positions(np.zeros_like(g))         # back to the uncorrected design for the next context
    a, b = out
    assert np.allclose(a['a'], b['a'], rtol=1e-9, atol=1e-12 * np.abs(b['a']).max())
    assert np.allclose(a['G'], b['G'], rtol=1e-7, atol=1e-9 * np.abs(b['G']).max())


def test_fixed_point_with_modes_is_refused():
    s = _field(n_gz=1, seed=2)
    f, m, r, mu = _converge(s)
    with pytest.raises(NotImplementedError):
        cte.run_cte_v2(s, s.image_names, s.spi, s.star_id_to_idx, f, m, mu, r, None, method='fixed_point')


if __name__ == '__main__':
    import sys, time, traceback
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    n_ok = n = 0
    for name, fn in tests:
        cases = [((), {})] if not hasattr(fn, '_params') else [((), {fn._params[0]: v}) for v in fn._params[1]]
        for a, kw in cases:
            n += 1; t0 = time.time()
            try:
                fn(*a, **kw); n_ok += 1; print(f'PASS {name} {kw} ({time.time() - t0:.1f} s)')
            except Exception:
                print(f'FAIL {name} {kw}'); traceback.print_exc()
    print(f'{n_ok}/{n} passed'); sys.exit(0 if n_ok == n else 1)
