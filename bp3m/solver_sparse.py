"""
Sparse variant of BP3MSolver.

Identical to BP3MSolver in every respect except _solve_one_pass and fit():

  - The image-transformation precision matrix Cr_inv (n_r × n_r) is assembled
    in COO format and converted to CSC before solving.
  - scipy.sparse.linalg.spsolve is used to compute Δr = Cr_inv⁻¹ rhs instead
    of dense inversion followed by matrix-vector multiplication.
  - During the convergence loop, the expensive dense inversion of Cr_inv is
    skipped; only the final pass computes the full C_r (still dense) needed
    by sample_posteriors.

K_img convention (must match BP3MSolver)
-----------------------------------------
K_img[img] is stored as shape (n, N_V, N_R) — the FULL per-image K matrix for
ALL stars (not filtered to used stars).  This matches the dense solver so that
the inherited sample_posteriors can do K_img[img][use] correctly.

Speedup profile
---------------
Fornax / small overlapping fields (n_r ≈ 40):
    Dense and sparse are comparable; sparse overhead may slightly dominate.
Tiled mosaics with non-overlapping images (n_r large, H_rr block-diagonal):
    Sparse wins substantially.  A perfectly block-diagonal n_r = 800 system
    requires O(100 × 8³) = 51 k operations vs O(800³) = 512 M for dense.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .solver import BP3MSolver, N_V


class BP3MSolverSparse(BP3MSolver):
    """
    BP3MSolver with sparse assembly and direct sparse solve for Δr.

    All public methods and return values are identical to BP3MSolver.
    """

    # ── Core solver (sparse override) ─────────────────────────────────────────

    def _solve_one_pass(self, r_current, _return_C_r=True):
        """
        Single Schur-complement solve using sparse assembly for Cr_inv.

        Parameters
        ----------
        r_current : (n_r,) current image transformation vector
        _return_C_r : bool
            If False, returns C_r=None and skips the expensive dense inversion.
            Used by fit() during intermediate iterations.
        """
        nr  = self.N_R
        n_r = nr * self.n_images

        # ── Star-level precision / information (identical to dense) ───────────
        H_vv = self.C_survey_inv.copy()
        H_vv[:, np.arange(N_V), np.arange(N_V)] += self._C_VG_inv_per_star

        h = self.C_survey_inv_dot_v.copy()

        K_img      = {}
        XCs_xresid = {}
        _XCsX      = {}
        _prior_inv = {}

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data[img]
            if d is None:
                K_img[img] = None
                continue

            use  = d["use_for_fit"]
            sidx = d["sidx"][use]
            n    = d["n"]

            # Pre-filter to used stars for efficiency in the Schur assembly
            JU_use  = d["JU"][use]
            X_use   = d["X_mat"][use]
            xys_use = d["xys"][use]

            cs  = j_idx * nr
            r_j = r_current[cs:cs + nr]

            Cs_use     = self._compute_Cs(img, r_j)[use]
            Cs_inv_use = np.linalg.inv(Cs_use)

            x_pred_use  = np.einsum('nkl,l->nk', X_use, r_j)
            x_resid_use = xys_use - x_pred_use

            JUT_Cs_use = np.einsum('nki,nkl->nil', JU_use, Cs_inv_use)

            np.add.at(H_vv, sidx,
                      np.einsum('nik,nkj->nij', JUT_Cs_use, JU_use))
            np.subtract.at(h, sidx,
                           np.einsum('nik,nk->ni', JUT_Cs_use, x_resid_use))

            # K for used stars only (n_use, N_V, N_R)
            K_use = np.einsum('nik,nkl->nil', JUT_Cs_use, X_use)

            # Store full-size K (n, N_V, N_R) so inherited sample_posteriors
            # can index it with K_img[img][use] correctly.
            K_full = np.zeros((n, N_V, nr))
            K_full[use] = K_use
            K_img[img] = K_full

            _XCsX[j_idx]      = np.einsum('nki,nkl,nlj->ij',
                                           X_use, Cs_inv_use, X_use)
            _prior_inv[j_idx] = self._img_data[img]["C_r_prior_inv"]
            XCs_xresid[img]   = np.einsum('nki,nkl,nl->ni',
                                           X_use, Cs_inv_use, x_resid_use)

        # ── Invert H_vv → C_vT, compute a ────────────────────────────────────
        C_vT = np.linalg.inv(H_vv)
        a    = np.einsum('nij,nj->ni', C_vT, h)

        # ── Assemble Schur-complement precision in COO format ─────────────────
        # Cr_inv = H_rr - Σ_i K_i^T C_vT_i K_i
        # H_rr   = diag(XCsX_j + prior_j)  (block-diagonal)
        # The Schur correction adds diagonal and off-diagonal N_R×N_R blocks.

        coo_r = []
        coo_c = []
        coo_v = []

        def _coo_block(r0, c0, block):
            """Accumulate a dense block into COO lists."""
            nr_, nc_ = block.shape
            rs  = np.repeat(np.arange(r0, r0 + nr_), nc_)
            cs_ = np.tile(np.arange(c0, c0 + nc_), nr_)
            coo_r.append(rs)
            coo_c.append(cs_)
            coo_v.append(block.ravel())

        rhs = np.zeros(n_r)

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data[img]
            if d is None or K_img[img] is None:
                continue

            cs   = j_idx * nr
            use  = d["use_for_fit"]
            sidx = d["sidx"][use]
            K    = K_img[img][use]   # (n_use, N_V, N_R)

            r_prior_j = d["r_prior"]
            rhs[cs:cs + nr] += _prior_inv[j_idx] @ (r_prior_j - r_current[cs:cs + nr])
            rhs[cs:cs + nr] += XCs_xresid[img].sum(axis=0)
            rhs[cs:cs + nr] += np.einsum('nji,nj->i', K, a[sidx])

            CvT_K    = np.einsum('nij,njk->nik', C_vT[sidx], K)
            KT_CvT_K = np.einsum('nji,njk->ik',  K, CvT_K)
            diag_block = _XCsX[j_idx] + _prior_inv[j_idx] - KT_CvT_K
            _coo_block(cs, cs, diag_block)

        # Off-diagonal Schur blocks (only for image pairs sharing stars)
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data[img]
            if d is None or K_img[img] is None:
                continue
            cs   = j_idx * nr
            use  = d["use_for_fit"]
            sidx = d["sidx"][use]
            K    = K_img[img][use]   # (n_use, N_V, N_R)

            for j2_idx in range(j_idx + 1, self.n_images):
                img2 = self.image_names[j2_idx]
                d2 = self._img_data[img2]
                if d2 is None or K_img[img2] is None:
                    continue
                cs2   = j2_idx * nr
                use2  = d2["use_for_fit"]
                sidx2 = d2["sidx"][use2]
                K2    = K_img[img2][use2]   # (n_use2, N_V, N_R)

                common, idx1, idx2 = np.intersect1d(
                    sidx, sidx2, return_indices=True)
                if len(common) == 0:
                    continue

                CvT_c  = C_vT[common]
                CvT_K2 = np.einsum('nij,njk->nik', CvT_c, K2[idx2])
                block  = np.einsum('nji,njk->ik', K[idx1], CvT_K2)
                _coo_block(cs,  cs2, -block)
                _coo_block(cs2, cs,  -block.T)

        # Build sparse CSC matrix
        if coo_r:
            all_r = np.concatenate(coo_r)
            all_c = np.concatenate(coo_c)
            all_v = np.concatenate(coo_v)
        else:
            all_r = all_c = np.array([], dtype=int)
            all_v = np.array([])

        Cr_inv_sp = sp.coo_matrix(
            (all_v, (all_r, all_c)), shape=(n_r, n_r)
        ).tocsc()

        # ── Solve for Δr via sparse direct solver ─────────────────────────────
        try:
            delta_r = spla.spsolve(Cr_inv_sp, rhs)
        except Exception:
            delta_r = np.linalg.solve(Cr_inv_sp.toarray(), rhs)

        r_hat = r_current + delta_r

        # ── C_r: dense inverse with diagonal preconditioning ─────────────────
        # Skipped during convergence iterations to save time; see fit() below.
        if _return_C_r:
            Cr_arr     = Cr_inv_sp.toarray()
            d_diag     = np.sqrt(np.maximum(np.abs(np.diag(Cr_arr)), 1e-30))
            d_inv      = 1.0 / d_diag
            Cr_arr_sc  = d_inv[:, None] * Cr_arr * d_inv[None, :]
            try:
                C_r_sc = np.linalg.inv(Cr_arr_sc)
            except np.linalg.LinAlgError:
                C_r_sc = np.linalg.pinv(Cr_arr_sc)
            C_r = d_inv[:, None] * C_r_sc * d_inv[None, :]
            self._Cr_inv_sp = Cr_inv_sp
        else:
            C_r = None
            self._Cr_inv_sp = Cr_inv_sp

        return r_hat, C_r, a, K_img, C_vT

    # ── Fit with optimised iteration (skip C_r until final pass) ─────────────

    def fit(self, n_iter=10, tol=1e-6, clip_sigma=4.5,
            inflate_hst_errors=False, prefilter=True,
            chi2_threshold=None, alpha_scale_chi2=False,
            per_iter_callback=None, **_ignored):
        """
        Same as BP3MSolver.fit() but avoids the dense Cr_inv inversion during
        convergence iterations — only the final pass computes C_r.

        per_iter_callback : callable or None
            Called as ``per_iter_callback(solver, it+1, a_arr)`` after each
            iteration that produced an updated r_hat.  Same signature as the
            dense solver callback.  Extra keyword arguments from BP3MSolver.fit()
            are accepted but ignored (``**_ignored``).
        """
        r_hat = np.concatenate([self._img_data[img]["r_init"]
                                 for img in self.image_names])
        self._update_R(r_hat)

        nr      = self.N_R
        _pnames = ['a', 'b', 'c', 'd', 'w', 'z', 'Δα0', 'Δδ0']
        if nr > 8:
            _pnames += [f'poly{i}' for i in range(nr - 8)]
        _n_imgs = len(self.image_names)

        def _delta_summary(diff):
            imax      = int(np.argmax(diff))
            img_idx   = imax // nr
            param_idx = imax % nr
            max_str   = (f"{diff[imax]:.3e}"
                         f"  [{self.image_names[img_idx]} / {_pnames[param_idx]}]")
            parts = []
            for p in range(nr):
                if p in (6, 7):
                    continue
                vals = diff[p::nr]
                med  = float(np.median(vals))
                if _n_imgs > 1:
                    w68 = float(np.percentile(vals, 84) - np.percentile(vals, 16))
                    parts.append(f"{_pnames[p]}: {med:.2e} [{w68:.2e}]")
                else:
                    parts.append(f"{_pnames[p]}: {med:.2e}")
            return max_str, '  '.join(parts)

        C_r = None
        for it in range(n_iter):
            # Skip dense C_r computation during intermediate iterations
            is_last = (it == n_iter - 1)
            r_hat_new, C_r_it, a_arr, K_img, C_vT = self._solve_one_pass(
                r_hat, _return_C_r=is_last)

            diff = np.abs(r_hat_new - r_hat)
            diff[6::nr] = 0
            diff[7::nr] = 0
            delta = np.max(diff)
            r_hat = r_hat_new
            self._update_R(r_hat)
            self._update_geometry(r_hat, a_arr)

            max_str, stats_str = _delta_summary(diff)
            if clip_sigma is not None:
                clip_info, _, _ = self._update_use_for_fit(
                    r_hat, a_arr, C_r_it, C_vT, clip_sigma, iteration=it,
                    inflate_errors=inflate_hst_errors,
                    chi2_threshold=chi2_threshold,
                    alpha_scale_chi2=alpha_scale_chi2)
                print(f"  iter {it+1:2d}:  max|Δr| = {max_str}")
                print(f"    params: {stats_str}")
                for img, n_use, n_tot, alpha_applied, alpha_raw, n_astrom_only in clip_info:
                    if inflate_hst_errors and it >= 3:
                        alpha_str = (f"α_applied={alpha_applied:.3f}  "
                                     f"α_raw={alpha_raw:.3f}  [α-inflated]")
                    else:
                        alpha_str = f"α={alpha_applied:.3f}"
                    print(f"    {img}: {n_use}/{n_tot} stars,  {alpha_str}")
            else:
                print(f"  iter {it+1:2d}:  max|Δr| = {max_str}")
                print(f"    params: {stats_str}")

            if per_iter_callback is not None:
                per_iter_callback(self, it + 1)

            if delta < tol:
                # Converged: do one final pass to get C_r
                if C_r_it is None:
                    _, C_r, a_arr, K_img, C_vT = self._solve_one_pass(
                        r_hat, _return_C_r=True)
                else:
                    C_r = C_r_it
                print("  Converged.")
                break

            C_r = C_r_it   # may be None if not last iter

        else:
            # Loop exhausted without converging — ensure we have C_r
            if C_r is None:
                _, C_r, a_arr, K_img, C_vT = self._solve_one_pass(
                    r_hat, _return_C_r=True)

        print(f"  Stopped after {it+1:2d} iterations")

        v_hat = a_arr.copy()
        return r_hat, C_r, v_hat, C_vT, a_arr, K_img
