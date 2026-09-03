"""
Step 5: Bayesian astrometric alignment and proper motion measurement (BP3M).

Calls bp3m directly via its Python API using the FLC pipeline data layout.
Results are written to:
    {output_dir}/{field}/BP3M_results/
        stellar_astrometry.csv      — per-star posterior PMs + positions
        image_transformations.csv   — per-image alignment parameters
        v_cov_marginalised.npy      — (N, 5, 5) full posterior covariance
        plots/                      — diagnostic figures

The ``stellar_astrometry.csv`` produced here is the primary science output
of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

def _ensure_bp3m(bp3m_dir=None):
    pass  # bp3m is installed as a package; no sys.path manipulation needed


def run_alignment(  # noqa: C901
    output_dir: Path,
    field_name: str,
    n_iter: int = 20,
    n_samples: int = 1000,
    mcmc_posteriors: bool = False,
    clip_sigma: float = 4.5,
    poly_order: int = 1,
    split_ccd: bool = True,
    min_stars_split_ccd: int = 20,
    use_sparse: bool = False,
    inflate_hst_errors: bool = True,
    two_phase_align: bool = False,
    fit_epoch_distortion: bool = False,
    epoch_dist_order: int = 3,
    epoch_gap_days: float = 180.0,
    epoch_dist_sigma: float = 10.0,
    epoch_breaks=None,
    epoch_dist_min_images: int = 3,
    use_indv_outputs: bool = False,
    extra_run_config: dict | None = None,
    test_hysteresis_delta: float = 1.0,
    min_align_demote: int = 5,
    epoch_dist_groupby: str = 'full',
    no_prefilter: bool = False,
    no_plots: bool = False,
    images: list[str] | None = None,
    remove_images: list[str] | None = None,
    restrict_filters: list[str] | None = None,
    restrict_instdet: list[str] | None = None,
    bp3m_min_stars: int = 0,
    bp3m_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    prior_sigma_rot_deg: float | None = None,
    prior_sigma_scale: float | None = None,
    prior_sigma_skew: float | None = None,
    prior_sigma_pointing: float | None = None,
    prior_sigma_pair_rot_deg: float | None = None,
    prior_sigma_pair_scale: float | None = None,
    prior_sigma_pair_skew: float | None = None,
    prior_sigma_pair_pointing: float | None = None,
    use_pair_prior: bool = False,
    inflate_alpha_max: float = 3.0,
    use_influence_clip: bool = True,
    influence_k: float = 5.0,
    influence_floor_sr: float | None = None,
    influence_floor_sd: float = 3.0,
    influence_raw_cooks_d: bool = False,
    use_two_tier: bool = False,
    no_align_prior: bool = False,
    pos_err_floor: float = 0.05,
    plot_residuals: bool = False,
    plot_influence: bool = False,
    use_qso_anchors: bool = True,
    qso_anchors_csv: "Path | list | None" = None,
    gaia_epoch_obs: Optional[dict] = None,
    exclude_2p_from_alignment: bool = False,
    gaia_csv: "Path | list | None" = None,
    verbose_tests: bool = False,
    use_delve: bool = False,
    delve_use_for_align: bool = False,
    pos_corr_table=None,
) -> Path:
    """
    Run BP3M Bayesian alignment on a field.

    Parameters
    ----------
    output_dir       : pipeline root directory
    field_name       : field subdirectory name
    n_iter           : maximum EM outer iterations
    n_samples        : posterior samples for marginalisation
    clip_sigma       : MAD sigma for outlier rejection (0 = disabled)
    poly_order       : polynomial order for image transformation (1 = linear)
    split_ccd        : split ACS/WFC images into independent CCD halves
    min_stars_split_ccd : minimum stars per CCD half to allow splitting (default 20)
    use_sparse       : use sparse Schur-complement solver (faster for mosaics)
    inflate_hst_errors: enable per-image HST error inflation
    no_prefilter     : skip Phase-0 pre-filter pass
    no_plots         : skip diagnostic plot generation
    images           : restrict to these image names (None = all)
    remove_images    : exclude these image names
    restrict_filters : keep only images with these HST filters
    restrict_instdet : keep only images from these instrument+detector combos
    bp3m_dir         : override default bp3m location
    checkpoint_dir   : save/load fitting checkpoint here
    use_influence_clip  : enable test-4 Cook's D influence clipping
    influence_k         : adaptive multiplier k for sigma_resid and scaled_D thresholds (default 5.0)
    influence_floor_sr  : floor for sigma_resid threshold (None = theoretical chi(2) p99 ≈ 3.03)
    influence_floor_sd  : floor for scaled_D threshold (default 3.0)
    influence_raw_cooks_d : use raw Cook's D instead of null-normalised scaled_D
    no_align_prior      : zero out the alignment (a,b,c,d,delta_ra0,delta_dec0) prior

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_results/)
    """
    _ensure_bp3m(bp3m_dir)

    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader_flc import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    from bp3m.checkpointing import save_results

    import time
    import pandas as pd

    data_root   = Path(output_dir)
    output_bp3m = Path(bp3m_dir) if bp3m_dir is not None else data_root / field_name / "BP3M_results"
    output_bp3m.mkdir(parents=True, exist_ok=True)

    print("\n" + "─"*50)
    print("Step 5: Bayesian alignment (BP3M)")
    print("─"*50)
    print(f"  n_iter={n_iter}  n_samples={n_samples}  "
          f"clip_sigma={clip_sigma}  poly_order={poly_order}")
    _cmd = (
        f"run_bp3m.py {field_name} --data-root {data_root}"
        f" --flc-pipeline"
        f" --n-iter {n_iter}"
        f" --n-samples {n_samples}"
        f" --clip-sigma {clip_sigma}"
        f" --poly-order {poly_order}"
        + (" --split-ccd"          if split_ccd else "")
        + (f" --min-stars-split-ccd {min_stars_split_ccd}" if split_ccd and min_stars_split_ccd != 20 else "")
        + (" --inflate-hst-errors" if inflate_hst_errors else "")
        + (" --sparse"             if use_sparse else "")
        + (" --no-prefilter"       if no_prefilter else "")
        + (" --no-plots"           if no_plots else "")
        + (f" --images {' '.join(images)}"          if images else "")
        + (f" --remove-images {' '.join(remove_images)}" if remove_images else "")
        + (f" --restrict-to-hst-filters {' '.join(restrict_filters)}" if restrict_filters else "")
        + (f" --checkpoint {checkpoint_dir}"        if checkpoint_dir else "")
    )
    print(f"  run_bp3m command:\n    {_cmd}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading FLC pipeline data for '{field_name}'...")
    imgs, stars_per_image, gaia_catalog = load_image_data_flc(
        data_root, field_name, pos_err_floor=pos_err_floor,
        restrict_images=set(images) if images is not None else None,
        gaia_csv=gaia_csv, use_delve=use_delve,
        delve_use_for_align=delve_use_for_align,
        pos_corr_table=pos_corr_table)
    if imgs is None or len(imgs) == 0:
        raise RuntimeError(
            f"No usable images found for '{field_name}'. "
            "Check that cross-matching completed successfully."
        )

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    # ── Image filtering ───────────────────────────────────────────────────────
    if images is not None:
        requested = set(images)
        image_names = [n for n in image_names if n in requested]
    if remove_images is not None:
        drop = set(remove_images)
        image_names = [n for n in image_names if n not in drop]
    if restrict_filters is not None:
        keep_filters = {f.upper() for f in restrict_filters}
        image_names = [n for n in image_names
                       if imgs[n].get('filter', '').upper() in keep_filters]
    if restrict_instdet is not None:
        keep_id = {s.upper() for s in restrict_instdet}
        image_names = [
            n for n in image_names
            if (imgs[n].get('instrument', '') + imgs[n].get('detector', '')).upper()
               in keep_id
        ]

    if bp3m_min_stars > 0:
        before = len(image_names)
        # Count alignment-eligible stars only (use_for_alignment=True, excluding
        # DELVE-only rows which have negative Gaia_id and aren't used for alignment).
        image_names = [
            n for n in image_names
            if int(stars_per_image[n]["use_for_alignment"].sum()) >= bp3m_min_stars
        ]
        dropped = before - len(image_names)
        if dropped:
            print(f"  --bp3m_min_stars {bp3m_min_stars}: dropped {dropped} image(s) "
                  f"with fewer than {bp3m_min_stars} alignment stars")

    if not image_names:
        raise RuntimeError("No images remain after filtering.")
    print(f"  Images: {len(image_names)}")

    # Rebuild index maps after filtering
    filtered_spi = {n: stars_per_image[n] for n in image_names}
    star_id_to_idx, image_names, star_in_image = build_index_maps(
        filtered_spi, gaia_catalog)

    # Filter gaia_catalog to observed stars
    observed_ids = set()
    for spi in filtered_spi.values():
        observed_ids.update(spi['Gaia_id'].values)
    gaia_catalog = (gaia_catalog[gaia_catalog['Gaia_id'].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {gid: i for i, gid in enumerate(gaia_catalog['Gaia_id'])}

    # Keep imgs in sync with filtered_spi (e.g. after --restrict_instdet)
    imgs = {n: imgs[n] for n in image_names}

    # ── Split CCD if requested ────────────────────────────────────────────────
    if split_ccd:
        from bp3m.data_loader_flc import split_images_by_ccd
        imgs, filtered_spi = split_images_by_ccd(
            imgs, filtered_spi, min_stars_per_ccd=min_stars_split_ccd)
        image_names = sorted(filtered_spi.keys())
        star_id_to_idx, image_names, star_in_image = build_index_maps(
            filtered_spi, gaia_catalog)

    # ── Initialise solver ─────────────────────────────────────────────────────
    if use_sparse and fit_epoch_distortion:
        raise ValueError("--fit_epoch_distortion is not supported with --sparse yet")
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(imgs, filtered_spi, gaia_catalog,
                          star_id_to_idx, image_names, star_in_image,
                          poly_order=poly_order,
                          exclude_2p_from_alignment=exclude_2p_from_alignment,
                          prior_sigma_rot_deg=prior_sigma_rot_deg,
                          prior_sigma_scale=prior_sigma_scale,
                          prior_sigma_skew=prior_sigma_skew,
                          prior_sigma_pointing=prior_sigma_pointing,
                          prior_sigma_pair_rot_deg=prior_sigma_pair_rot_deg,
                          prior_sigma_pair_scale=prior_sigma_pair_scale,
                          prior_sigma_pair_skew=prior_sigma_pair_skew,
                          prior_sigma_pair_pointing=prior_sigma_pair_pointing,
                          use_pair_prior=use_pair_prior,
                          fit_epoch_distortion=fit_epoch_distortion,
                          epoch_dist_order=epoch_dist_order,
                          epoch_gap_days=epoch_gap_days,
                          epoch_dist_sigma_mas=epoch_dist_sigma,
                          epoch_breaks=epoch_breaks,
                          epoch_dist_min_images=epoch_dist_min_images,
                          epoch_dist_groupby=epoch_dist_groupby)

    print(f"  Stars: {solver.n_stars}   Images: {solver.n_images}")

    _indv_init_stats = None
    if use_indv_outputs:
        _indv_init_stats = _apply_indv_init(
            solver, image_names, data_root, field_name)

    # ── QSO anchor prior injection ────────────────────────────────────────────
    # Replaces the diffuse global prior on PM+parallax with a tight secular-
    # aberration prior for vetted QSO anchors.  Adds σ_κ^{-2} to the PM rows
    # of C_survey_inv and the matching secular-aberration RHS to
    # C_survey_inv_dot_v — the Gaia measurement contribution stays unchanged.
    print("\n  QSO anchor priors:")
    _injected_qso_ids = []
    if use_qso_anchors:
        import pandas as _qpd
        if qso_anchors_csv is not None:
            # Accept a single Path/str or a list of paths (multi-pointing)
            _anchor_paths = (
                [Path(p) for p in qso_anchors_csv]
                if isinstance(qso_anchors_csv, list)
                else [Path(qso_anchors_csv)]
            )
            _anchor_paths = [p for p in _anchor_paths if p.exists()]
        else:
            from .qso_vetting import find_qso_anchors
            _p = find_qso_anchors(Path(output_dir) / field_name / 'Gaia', field_name)
            _anchor_paths = [_p] if _p is not None and _p.exists() else []

        if _anchor_paths:
            _qdf_parts = [_qpd.read_csv(p, dtype={'source_id': 'int64'})
                          for p in _anchor_paths]
            _qdf = (_qpd.concat(_qdf_parts, ignore_index=True)
                    .drop_duplicates('source_id') if len(_qdf_parts) > 1
                    else _qdf_parts[0])
            _n_gaia_candidates = len(_qdf)
            _qdf_ok  = _qdf[_qdf['is_qso_anchor'].fillna(False)]
            _n_anchors = len(_qdf_ok)

            # Breakdown: quaia / milliquas / crf
            _n_quaia  = int(_qdf_ok.get('quaia_match',   _qpd.Series(False)).sum())
            _n_mq     = int(_qdf_ok.get('milliquas_match', _qpd.Series(False)).sum())
            _n_crf    = int(_qdf_ok.get('gaia_crf_source', _qpd.Series(False)).sum())
            _n_5p     = int(_qdf['has_5p_solution'].sum())
            _n_astrom = int(_qdf['astrometric_pass'].sum())

            print(f"    Gaia qso_candidates in field:  {_n_gaia_candidates}")
            print(f"    With 5p/6p Gaia solution:      {_n_5p}")
            print(f"    Astrometric cut survivors:     {_n_astrom}  "
                  f"(|Δv|_Mahal < 3σ vs secular aberration + zero parallax)")
            print(f"    Quaia (source_id match):       "
                  f"{int(_qdf['quaia_match'].sum())} candidates  →  {_n_quaia} anchors")
            print(f"    MILLIQUAS (RA/Dec match):      "
                  f"{int(_qdf['milliquas_match'].sum())} candidates  →  {_n_mq} anchors")
            if _n_crf:
                print(f"    Gaia CRF3 (highest purity):    {_n_crf} anchors")
            print(f"    Vetted QSO anchors total:      {_n_anchors}  "
                  f"(catalog match AND astrometric pass)")

            _sigma_qso_pm_inv_sq  = (3.5e-4) ** -2
            _sigma_qso_plx_inv_sq = (1.0e-3) ** -2
            _n_injected = 0
            _injected_qso_ids = []

            for _, _qrow in _qdf_ok.iterrows():
                _sidx = star_id_to_idx.get(int(_qrow['source_id']))
                if _sidx is None:
                    continue
                _pmra_ab  = float(_qrow['pmra_aberr_uas'])  * 1e-3
                _pmdec_ab = float(_qrow['pmdec_aberr_uas']) * 1e-3

                # NOTE (deliberate asymmetry): this tightens the SOLVE prior
                # (C_survey_inv / C_survey_inv_dot_v) but not v_prior/C_prior,
                # so test 1 still measures QSO stars against the Gaia
                # measurement alone.  A QSO pulled to the aberration PM is thus
                # judged for consistency with Gaia, not with its own anchor —
                # the test stays independent of the prior being imposed.
                solver.C_survey_inv[_sidx, 2, 2] += _sigma_qso_pm_inv_sq
                solver.C_survey_inv[_sidx, 3, 3] += _sigma_qso_pm_inv_sq
                solver.C_survey_inv[_sidx, 4, 4] += _sigma_qso_plx_inv_sq
                solver.C_survey_inv_dot_v[_sidx, 2] += _sigma_qso_pm_inv_sq * _pmra_ab
                solver.C_survey_inv_dot_v[_sidx, 3] += _sigma_qso_pm_inv_sq * _pmdec_ab
                _n_injected += 1
                _injected_qso_ids.append(int(_qrow['source_id']))

            if _n_injected > 0:
                print(f"    Injected into alignment:       {_n_injected}  "
                      f"(σ_κ = 0.35 µas/yr, σ_plx = 1 µas)")
            else:
                print(f"    Injected into alignment:       0  "
                      f"(none of the {_n_anchors} vetted anchors are in the HST field)")
        else:
            print(f"    QSO anchor file not found")
            print(f"    Re-run from Phase 1 to generate it (or pass --no_qso_anchors)")
    else:
        print("    Disabled (--no_qso_anchors)")

    if gaia_epoch_obs:
        solver._add_gaia_epoch_obs(gaia_epoch_obs)

    # ── Fit ───────────────────────────────────────────────────────────────────
    clip = clip_sigma if clip_sigma > 0 else None
    t0 = time.time()
    r_hat, C_r, v_hat, C_vT, a_arr, K_img, _ = solver.fit(
        adaptive_delta=test_hysteresis_delta,
        min_align_demote=min_align_demote,
        n_iter=n_iter,
        clip_sigma=clip,
        inflate_hst_errors=inflate_hst_errors,
        two_phase_align=two_phase_align,
        inflate_alpha_max=inflate_alpha_max,
        prefilter=not no_prefilter,
        use_influence_clip=use_influence_clip,
        influence_k=influence_k,
        influence_floor_sr=influence_floor_sr,
        influence_floor_sd=influence_floor_sd,
        influence_raw_cooks_d=influence_raw_cooks_d,
        verbose_tests=verbose_tests,
        use_two_tier=use_two_tier,
        no_align_prior=no_align_prior,
    )
    print(f"  Fit completed in {time.time()-t0:.1f}s")

    # ── Sample posteriors ─────────────────────────────────────────────────────
    if mcmc_posteriors:
        print(f"  Drawing {n_samples} posterior samples (MCMC marginalisation)...")
        _, v_mean, v_cov = solver.sample_posteriors(
            r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)
    else:
        print(f"  Computing analytic marginalised posteriors...")
        v_mean, v_cov = solver.compute_analytic_posteriors(r_hat, C_r, a_arr, K_img, C_vT)

    # ── Save results ──────────────────────────────────────────────────────────
    _save_results(
        output_bp3m, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            **(extra_run_config or {}),
            'use_indv_outputs': use_indv_outputs,
            'pos_corr_table': (str(pos_corr_table) if pos_corr_table else None),
            'pos_err_floor': pos_err_floor,
            'test_hysteresis_delta': test_hysteresis_delta,
            'min_align_demote': min_align_demote,
            'indv_init_stats': _indv_init_stats,
            'n_iter':       n_iter,
            'n_samples':    n_samples,
            'clip_sigma':   clip_sigma,
            'split_ccd':    split_ccd,
            'inflate_hst_errors': inflate_hst_errors,
            'two_phase_align': two_phase_align,
            'fit_epoch_distortion': fit_epoch_distortion,
            'epoch_dist_order': epoch_dist_order,
            'epoch_gap_days': epoch_gap_days,
            'epoch_dist_sigma_mas': epoch_dist_sigma,
            'epoch_breaks': list(epoch_breaks) if epoch_breaks else [],
            'epoch_dist_min_images': epoch_dist_min_images,
            'epoch_dist_groupby': epoch_dist_groupby,
            'n_epoch_dist_groups': len(getattr(solver, 'ed_groups', [])),
            'poly_order':   poly_order,
        },
    )

    # ── Star influence ────────────────────────────────────────────────────────
    print("  Computing star influence metrics...")
    try:
        import pandas as _pd
        influence_df = solver.compute_star_influence(r_hat, C_r, a_arr)
        influence_df.to_csv(output_bp3m / "star_influence.csv", index=False)
        print(f"  Saved: star_influence.csv  ({len(influence_df)} star-image pairs)")

        if not no_plots and plot_influence:
            from bp3m.plot_influence import plot_influence_diagnostics
            plot_dir = output_bp3m / "plots"
            plot_dir.mkdir(exist_ok=True)
            plot_influence_diagnostics(influence_df, plot_dir)
    except Exception as _exc:
        print(f"  WARNING: star influence computation failed — {_exc}")
        import traceback; traceback.print_exc()

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_bp3m,
                       plot_residuals=plot_residuals,
                       qso_anchor_ids=_injected_qso_ids)
        except Exception as exc:
            print(f"  WARNING: plots failed — {exc}")

    if checkpoint_dir is not None:
        from bp3m.checkpointing import save_inputs, save_results as _save_ckpt
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_inputs(solver, checkpoint_dir)
        _save_ckpt(r_hat, C_r, v_hat, v_mean, v_cov, C_vT,
                   K_img, solver, checkpoint_dir)

    print(f"\n  Results written to: {output_bp3m}")
    return output_bp3m


# ── Prior fallback ───────────────────────────────────────────────────────────

def _apply_prior_fallback(v_cov_full, v_mean, C_prior_arr, v_prior_arr,
                          failed_prior_test=None):
    """Return (v_cov_full, v_mean) with prior fallback applied in-place on copies.

    Stars get their posterior replaced with the prior when any of:
      1. Full 5D determinant: posterior less informative than prior.
      2. PM 2×2 block determinant: posterior PM less informative (catches cases
         where position improvement masks PM degradation from C_extra).
      3. failed_prior_test: star failed the Gaia-prior chi2 test (Gaia-
         incompatible) in the final iteration — the posterior mean is
         inconsistent with the prior regardless of covariance size.
    """
    v_cov_full = v_cov_full.copy()
    v_mean     = v_mean.copy()

    # Full 5D determinant check
    sign_post,  logdet_post  = np.linalg.slogdet(v_cov_full)
    sign_prior, logdet_prior = np.linalg.slogdet(C_prior_arr)
    use_prior = (sign_post > 0) & (sign_prior > 0) & (logdet_post > logdet_prior)

    # PM 2×2 block check (catches degradation hidden by position improvement)
    sign_pm_post,  logdet_pm_post  = np.linalg.slogdet(v_cov_full[:, 2:4, 2:4])
    sign_pm_prior, logdet_pm_prior = np.linalg.slogdet(C_prior_arr[:, 2:4, 2:4])
    use_prior_pm = ((sign_pm_post > 0) & (sign_pm_prior > 0) &
                    (logdet_pm_post > logdet_pm_prior))
    use_prior |= use_prior_pm

    # Stars that failed the Gaia-prior chi2 test — posterior inconsistent with prior
    if failed_prior_test is not None:
        use_prior |= failed_prior_test

    if use_prior.any():
        v_cov_full[use_prior] = C_prior_arr[use_prior]
        v_mean[use_prior]     = v_prior_arr[use_prior]

    # The mask is returned so callers can RECORD which stars were replaced —
    # silent replacement left downstream consumers no way to tell a fitted
    # posterior from a prior echo.
    return v_cov_full, v_mean, use_prior


# ── Internal: write result CSVs and npy ─────────────────────────────────────

def compute_chi2_per_star(solver, r_hat, v_hat, image_names, use_key='use_for_astrom'):
    """Per-star chi2 using given transformation r_hat and astrometry v_hat.

    chi2_i = sum_{j: use[j]} resid_j @ C_hst_j_inv @ resid_j
    where resid_j = x_j^obs - (X_j @ r_k - JU_j @ v_hat_i)

    Returns
    -------
    chi2 : (n_stars,) float  — summed chi2 per star
    n_det : (n_stars,) int   — number of detections included
    """
    n_stars = solver.n_stars
    chi2 = np.zeros(n_stars)
    n_det = np.zeros(n_stars, dtype=int)

    for j, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None:
            continue
        use = d.get(use_key, d['use_for_fit'])
        if not use.any():
            continue

        r_j    = r_hat[j * solver.N_R:(j + 1) * solver.N_R]
        sidx   = d['sidx'][use]
        xys    = d['xys'][use]
        X_mat  = d['X_mat'][use]
        JU     = d['JU'][use]
        # Covariance must live in the same frame as the residual.  The residual
        # below is in Gaia pseudo-image coordinates, but C_hst is the DETECTOR-
        # frame covariance; using it directly rotates the error ellipse by the
        # image rotation (up to the full orientation for HST), biasing chi2 for
        # any anisotropic C_hst.  _compute_Cs applies Cs = J C_hst J^T, exactly
        # as compute_residuals does.  (Still the inflated C_hst — bp3m's chi2
        # convention measures against the applied error model.)
        Cs     = solver._compute_Cs(img, r_j)[use]   # (n, 2, 2), pseudo frame
        C_inv  = np.linalg.inv(Cs)                   # (n, 2, 2)

        v_star = v_hat[sidx]               # (n, 5)
        motion = np.einsum('nij,nj->ni', JU, v_star)   # (n, 2)
        x_pred = np.einsum('nkl,l->nk', X_mat, r_j) - motion  # (n, 2)
        resid  = xys - x_pred              # (n, 2)

        chi2_det = np.einsum('ni,nij,nj->n', resid, C_inv, resid)
        np.add.at(chi2, sidx, chi2_det)
        np.add.at(n_det, sidx, 1)

    return chi2, n_det


def _apply_indv_init(solver, image_names, data_root, field_name):
    """--use_indv_outputs: warm-start the joint fit from per-image fits.

    Asymmetric trust (2026-09-02 design):
      - r_init            <- indv posterior r (better than transformation.csv)
      - indv-REJECTED detections (present in the indv fit but excluded from
        both alignment and astrometry there): hard-blocked from the joint
        alignment tier (they failed chi2 even under Gaia-propagation-sized
        errors -> near-certain junk).  use_for_align_init_flag is cleared so
        trust-flag re-admission cannot bring them back.
      - indv-ACCEPTED alignment detections: seed the initial use_for_fit mask
        only; the joint tests refine freely (indv vetting is only good to the
        Gaia-propagation error scale).
      - alpha is NOT imported (indv alpha=1 is a blindness artifact).

    Per-image provenance: the indv fit must postdate the current
    matched_gaia.csv (exact md5 when the indv run recorded one) and the match
    sidecar must carry the current xmatch_algo_version.  Failing images fall
    back to the standard initialisation with a reason count.
    """
    import hashlib
    import json as _json
    import pandas as _pd
    from bp3m.pipeline.cross_match import XMATCH_ALGO_VERSION

    indv_root = Path(data_root) / field_name / 'BP3M_indv_results'
    hst_root  = Path(data_root) / field_name / 'HST' / 'mastDownload' / 'HST'
    _sol_gid = np.zeros(solver.n_stars, dtype=np.int64)
    for _g, _i in solver.star_id_to_idx.items():
        _sol_gid[int(_i)] = np.int64(_g)

    stats = {'n_warm': 0, 'n_fallback': 0, 'n_rejected_dets': 0,
             'n_seed_dets': 0, 'fallback_reasons': {}}

    def _fb(reason, n=1):
        stats['n_fallback'] += n
        stats['fallback_reasons'][reason] = \
            stats['fallback_reasons'].get(reason, 0) + n

    bases: dict[str, list] = {}
    for img in image_names:
        base = img[:-3] if img.endswith(('_hi', '_lo')) else img
        bases.setdefault(base, []).append(img)

    for base, subs in sorted(bases.items()):
        d_indv = indv_root / base
        req = [d_indv / f for f in ('image_transformations.csv',
                                    'stellar_astrometry.csv',
                                    'use_for_fit.npz', 'star_indices.npz',
                                    'run_config.json')]
        if not all(f.exists() for f in req):
            _fb('no indv outputs', len(subs)); continue
        matched = hst_root / base / 'matched_gaia.csv'
        sidecar = hst_root / base / 'xmatch_status.json'
        if not matched.exists():
            _fb('no matched_gaia.csv', len(subs)); continue
        try:
            _cfg = _json.load(open(d_indv / 'run_config.json'))
        except Exception:
            _fb('unreadable indv run_config', len(subs)); continue
        _md5_rec = _cfg.get('matched_gaia_md5')
        if _md5_rec is not None:
            if hashlib.md5(matched.read_bytes()).hexdigest() != _md5_rec:
                _fb('matches changed since indv fit (md5)', len(subs)); continue
        elif matched.stat().st_mtime >= (d_indv / 'stellar_astrometry.csv').stat().st_mtime:
            _fb('matches newer than indv fit', len(subs)); continue
        try:
            _ver = _json.load(open(sidecar)).get('params', {}).get('xmatch_algo_version')
        except Exception:
            _ver = None
        if _ver != XMATCH_ALGO_VERSION:
            _fb(f'match algo v{_ver} != v{XMATCH_ALGO_VERSION}', len(subs)); continue

        it = _pd.read_csv(d_indv / 'image_transformations.csv').set_index('image_name')
        sa = _pd.read_csv(d_indv / 'stellar_astrometry.csv',
                          dtype={'Gaia_id': np.int64})
        uff = np.load(d_indv / 'use_for_fit.npz')
        si  = np.load(d_indv / 'star_indices.npz')
        _ufa_p = d_indv / 'use_for_astrom.npz'
        ufa = np.load(_ufa_p) if _ufa_p.exists() else None
        gids_indv = sa['Gaia_id'].to_numpy(np.int64)

        for sub in subs:
            d = solver._img_data.get(sub)
            if d is None or sub not in it.index or sub not in si.files:
                _fb('sub-image mismatch (chip split?)'); continue
            row = it.loc[sub]
            nr = solver.N_R
            r_vec = np.asarray(d['r_init'], float).copy()
            r_vec[0:4] = [float(row['a']), float(row['b']),
                          float(row['c']), float(row['d'])]
            if nr > 4:
                r_vec[4] = -float(row.get('delta_ra0_mas', 0.0))
            if nr > 5:
                r_vec[5] = -float(row.get('delta_dec0_mas', 0.0))
            _poly_ok = True
            for k in range(6, nr):
                if f'r_{k}' in row.index:
                    r_vec[k] = float(row[f'r_{k}'])
                else:
                    _poly_ok = False
                    break
            if not _poly_ok:
                _fb('poly_order mismatch'); continue

            sidx_i = si[sub]
            m_fit  = uff[sub].astype(bool) if sub in uff.files else \
                     np.zeros(len(sidx_i), bool)
            m_ast  = (ufa[sub].astype(bool)
                      if (ufa is not None and sub in ufa.files) else m_fit)
            g_all  = gids_indv[sidx_i]
            g_fit  = gids_indv[sidx_i[m_fit]]
            g_keep = gids_indv[sidx_i[m_fit | m_ast]]
            g_rej  = np.setdiff1d(g_all, g_keep)

            jg   = _sol_gid[d['sidx']]
            seen = np.isin(jg, g_all)
            rej  = np.isin(jg, g_rej) if len(g_rej) else np.zeros(len(jg), bool)
            acc  = np.isin(jg, g_fit) if len(g_fit) else np.zeros(len(jg), bool)

            d['r_init'] = r_vec
            _uf = np.asarray(d['use_for_fit'], bool)
            # acceptance list seeds the mask (only where indv saw the star)
            _uf = _uf & np.where(seen, acc, True)
            # rejections are hard-blocked from alignment, incl. re-admission
            _uf &= ~rej
            d['use_for_fit'] = _uf
            if 'use_for_align_init_flag' in d:
                d['use_for_align_init_flag'] = \
                    np.asarray(d['use_for_align_init_flag'], bool) & ~rej
            stats['n_rejected_dets'] += int(rej.sum())
            stats['n_seed_dets']     += int((seen & acc).sum())
            stats['n_warm'] += 1

    print(f"  use_indv_outputs: warm-started {stats['n_warm']} images "
          f"({stats['n_seed_dets']} seeded alignment dets, "
          f"{stats['n_rejected_dets']} indv-rejected dets hard-blocked); "
          f"{stats['n_fallback']} images fell back")
    for r, n in sorted(stats['fallback_reasons'].items()):
        print(f"    fallback [{r}]: {n} images")
    return stats


def _save_results(output_dir, solver, images, gaia_catalog, image_names,
                  r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
                  run_config: dict | None = None):
    import pandas as pd

    _failed_prior = ~getattr(solver, 'ok_star', np.ones(solver.n_stars, bool))
    v_cov_full, v_mean, _used_prior = _apply_prior_fallback(
        v_cov + C_vT, v_mean, solver.C_prior, solver.v_prior,
        failed_prior_test=_failed_prior)

    # 1. Image transformation parameters
    rows = []
    for j, img in enumerate(image_names):
        cs   = j * solver.N_R
        r_j  = r_hat[cs: cs + solver.N_R]
        C_j  = C_r[cs: cs + solver.N_R, cs: cs + solver.N_R]
        d_img = solver._img_data[img]
        n_align  = int(np.sum(d_img['use_for_fit']))
        use_ast  = d_img.get('use_for_astrom', d_img['use_for_fit'])
        n_astrom = int(np.sum(use_ast & ~d_img['use_for_fit']))
        a, b, c, d = r_j[:4]
        alpha_applied = float(d_img.get('alpha_applied', 1.0))
        meta = images[img]
        rows.append(dict(
            image_name=img,
            n_stars_alignment=n_align,
            n_stars_astrometry_only=n_astrom,
            a=a, b=b, c=c, d=d,
            delta_ra0_mas=(meta.get('ra0_final', meta['ra0']) - meta['ra0']) * 3_600_000.0,
            delta_dec0_mas=(meta.get('dec0_final', meta['dec0']) - meta['dec0']) * 3_600_000.0,
            ra0_final=meta.get('ra0_final', meta['ra0']),
            dec0_final=meta.get('dec0_final', meta['dec0']),
            pixel_scale_mas=np.sqrt(a*d - b*c) * images[img].get('orig_pixel_scale', 50.0),
            rotation_deg=np.degrees(np.arctan2(b - c, a + d)),
            on_skew=(a - d) / 2,
            off_skew=(b + c) / 2,
            sigma_a=np.sqrt(C_j[0,0]), sigma_b=np.sqrt(C_j[1,1]),
            sigma_c=np.sqrt(C_j[2,2]), sigma_d=np.sqrt(C_j[3,3]),
            sigma_dra0_mas=np.sqrt(C_j[4,4]),
            sigma_ddec0_mas=np.sqrt(C_j[5,5]),
            Xo_pivot=meta.get('Xo', 2048.0),
            Yo_pivot=meta.get('Yo', 2048.0),
            alpha=alpha_applied,
            # alpha_raw is the LAST measured inflation step (pre-cap);
            # saturated_alpha flags that the measurement hit inflate_alpha_max,
            # i.e. the residuals wanted more inflation than was applied.
            alpha_raw=float(d_img.get('alpha_raw', np.nan)),
            alpha_max=float(d_img.get('alpha_max', np.nan)),
            n_alpha_ref=int(d_img.get('n_alpha_ref', 0)),
            saturated_alpha=bool(
                np.isfinite(d_img.get('alpha_raw', np.nan))
                and np.isfinite(d_img.get('alpha_max', np.nan))
                and d_img.get('alpha_raw', 0.0) >= d_img.get('alpha_max', np.inf) - 1e-9),
            ed_group=int(getattr(solver, '_ed_gidx', {}).get(img, -1)),
            **{f'r_{k}': float(r_j[k]) for k in range(6, solver.N_R)},
        ))
    pd.DataFrame(rows).to_csv(output_dir / "image_transformations.csv", index=False)

    # 2. Stellar astrometry
    g = gaia_catalog.copy()
    g['n_hst_used'] = solver.gaia_n_hst_used  # detections used for alignment OR astrometry

    # Per-star alignment detection count
    n_align = np.zeros(solver.n_stars, dtype=int)
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is not None:
            np.add.at(n_align, d_img['sidx'][d_img['use_for_fit']], 1)
    g['n_hst_alignment'] = n_align

    # Per-star chi2 using best-fit (r_hat, v_hat) for use_for_astrom detections
    chi2_hst, n_chi2 = compute_chi2_per_star(
        solver, r_hat, v_hat, image_names, use_key='use_for_astrom'
    )
    g['chi2_hst']   = chi2_hst
    g['n_det_chi2'] = n_chi2
    # Reduced chi2 (chi2 per 2-dof detection): 0 when no detections
    with np.errstate(invalid='ignore', divide='ignore'):
        g['chi2_hst_red'] = np.where(n_chi2 > 0, chi2_hst / (2 * n_chi2), np.nan)

    # Star-level test diagnostics (parity with ground_to_gaia_xmatch).
    # sigma_from_gaia_prior = sqrt(chi2_gaia) from the FINAL star tests, so it
    # describes the solution actually written out; ok_star is that test's
    # verdict; prior_fallback records which rows below are the prior, not a fit.
    g['sigma_from_gaia_prior'] = solver.sigma_from_gaia_prior
    g['ok_star']               = getattr(solver, 'ok_star',
                                         np.ones(solver.n_stars, bool))
    g['prior_fallback']        = _used_prior
    g['delta_racosdec_bp3m'] = v_mean[:, 0]
    g['delta_dec_bp3m']      = v_mean[:, 1]
    g['pmra_bp3m']           = v_mean[:, 2]
    g['pmdec_bp3m']          = v_mean[:, 3]
    g['parallax_bp3m']       = v_mean[:, 4]

    g['sigma_delta_racosdec'] = np.sqrt(v_cov_full[:, 0, 0])
    g['sigma_delta_dec']      = np.sqrt(v_cov_full[:, 1, 1])
    g['sigma_pmra_bp3m']      = np.sqrt(v_cov_full[:, 2, 2])
    g['sigma_pmdec_bp3m']     = np.sqrt(v_cov_full[:, 3, 3])
    g['sigma_parallax_bp3m']  = np.sqrt(v_cov_full[:, 4, 4])

    _sig = np.sqrt(np.diagonal(v_cov_full, axis1=1, axis2=2))
    for col, i, j in [
        ('corr_dra_ddec', 0, 1), ('corr_dra_pmra', 0, 2),
        ('corr_dra_pmdec', 0, 3), ('corr_dra_plx', 0, 4),
        ('corr_ddec_pmra', 1, 2), ('corr_ddec_pmdec', 1, 3),
        ('corr_ddec_plx', 1, 4), ('corr_pmra_pmdec', 2, 3),
        ('corr_pmra_plx', 2, 4), ('corr_pmdec_plx', 3, 4),
    ]:
        denom = _sig[:, i] * _sig[:, j]
        g[col] = np.where(denom > 0, v_cov_full[:, i, j] / denom, np.nan)

    # Conditional (MAP alignment fixed)
    g['pmra_bp3m_cond']           = v_hat[:, 2]
    g['pmdec_bp3m_cond']          = v_hat[:, 3]
    g['parallax_bp3m_cond']       = v_hat[:, 4]
    g['sigma_pmra_bp3m_cond']     = np.sqrt(C_vT[:, 2, 2])
    g['sigma_pmdec_bp3m_cond']    = np.sqrt(C_vT[:, 3, 3])
    g['sigma_parallax_bp3m_cond'] = np.sqrt(C_vT[:, 4, 4])

    g.to_csv(output_dir / "stellar_astrometry.csv", index=False)

    # 3. Full covariance arrays
    np.save(output_dir / "v_cov_marginalised.npy", v_cov)
    np.save(output_dir / "C_vT.npy", C_vT)
    _n_r_only = solver.N_R * solver.n_images
    if getattr(solver, 'n_ed', 0):
        # C_r.npy keeps its legacy shape/meaning: the per-image r-block
        # MARGINAL covariance (marginalised over the epoch-D coefficients).
        np.save(output_dir / "C_r.npy", C_r[:_n_r_only, :_n_r_only])
        np.save(output_dir / "C_epoch_distortion.npy", C_r[_n_r_only:, _n_r_only:])
        np.save(output_dir / "C_r_epoch_cross.npy", C_r[:_n_r_only, _n_r_only:])
        from bp3m.astro_utils import epoch_distortion_pairs
        _prs = epoch_distortion_pairs(solver._ed_order)
        _nsh = len(_prs)
        _rows = []
        _dvec = r_hat[_n_r_only:]
        _dsig = np.sqrt(np.maximum(np.diag(C_r[_n_r_only:, _n_r_only:]), 0.0))
        for _g, _grp in enumerate(solver.ed_groups):
            for _k in range(solver.ED_K):
                _axis = 'x' if _k < _nsh else 'y'
                _i, _j = _prs[_k % _nsh]
                _meta0 = solver.images[_grp['images'][0]]
                _rows.append(dict(
                    group=_g, instrument=_grp['instrument'],
                    detector=_grp['detector'], chip=_grp['chip'],
                    filter=_grp['filter'], epoch_id=_grp['epoch_id'],
                    mean_mjd=_grp['mean_mjd'], n_images=len(_grp['images']),
                    order=solver._ed_order,
                    half_x=2048.0,
                    half_y=(507.0 if str(_grp['detector']).upper() == 'IR'
                            else 1024.0),
                    pscale_mas=float(_meta0.get('orig_pixel_scale', 50.0)),
                    axis=_axis, leg_i=_i, leg_j=_j,
                    coeff_px=float(_dvec[_g * solver.ED_K + _k]),
                    sigma_px=float(_dsig[_g * solver.ED_K + _k]),
                ))
        import pandas as _pd
        _pd.DataFrame(_rows).to_csv(output_dir / "epoch_distortion.csv", index=False)
        print(f"  Saved: epoch_distortion.csv  ({len(solver.ed_groups)} chip-groups, "
              f"order {solver._ed_order}), C_epoch_distortion.npy")
        # Linear-deviation aggregation: per-group inverse-variance-weighted
        # mean of each image's fitted-vs-header linear terms.  D itself starts
        # at degree 2 (deg 0-1 are alignment-degenerate), so the group-level
        # scale/rotation/skew deviation of the GDC lives in the per-image
        # (a,b,c,d) posteriors; this re-attributes it to the group for the
        # archive program.  Deviation matrix Delta = (F - P) P^{-1} with
        # F = fitted [[a,b],[c,d]], P = header prior; components follow bp3m
        # conventions (rot from b-c, on_skew (a-d)/2, off_skew (b+c)/2).
        _lin_rows = []
        _comp_names = ('scale', 'rot', 'on_skew', 'off_skew')
        for _g, _grp in enumerate(solver.ed_groups):
            _devs, _vars = [], []
            for _im in _grp['images']:
                _j = image_names.index(_im)
                _cs = _j * solver.N_R
                _F = r_hat[_cs:_cs + 4].reshape(2, 2)
                _P = solver._img_data[_im]['r_prior'][:4].reshape(2, 2)
                _Cj = C_r[_cs:_cs + 4, _cs:_cs + 4]
                _detP = _P[0, 0] * _P[1, 1] - _P[0, 1] * _P[1, 0]
                if abs(_detP) < 1e-12:
                    continue
                _Q = np.array([[_P[1, 1], -_P[0, 1]],
                               [-_P[1, 0], _P[0, 0]]]) / _detP
                _D = (_F - _P) @ _Q
                _sig2 = np.diag(_Cj)                       # var(a,b,c,d)
                _vD = np.empty((2, 2))                     # var of Delta elems
                _vD[0, 0] = _sig2[0]*_Q[0, 0]**2 + _sig2[1]*_Q[1, 0]**2
                _vD[0, 1] = _sig2[0]*_Q[0, 1]**2 + _sig2[1]*_Q[1, 1]**2
                _vD[1, 0] = _sig2[2]*_Q[0, 0]**2 + _sig2[3]*_Q[1, 0]**2
                _vD[1, 1] = _sig2[2]*_Q[0, 1]**2 + _sig2[3]*_Q[1, 1]**2
                _devs.append([(_D[0, 0] + _D[1, 1]) / 2,   # scale (fractional)
                              (_D[0, 1] - _D[1, 0]) / 2,   # rotation (rad)
                              (_D[0, 0] - _D[1, 1]) / 2,   # on-axis skew
                              (_D[0, 1] + _D[1, 0]) / 2])  # off-axis skew
                _vars.append([(_vD[0, 0] + _vD[1, 1]) / 4,
                              (_vD[0, 1] + _vD[1, 0]) / 4,
                              (_vD[0, 0] + _vD[1, 1]) / 4,
                              (_vD[0, 1] + _vD[1, 0]) / 4])
            if not _devs:
                continue
            _devs = np.asarray(_devs); _vars = np.asarray(_vars)
            _w = 1.0 / np.maximum(_vars, 1e-30)
            _mean = (_w * _devs).sum(0) / _w.sum(0)
            _msig = 1.0 / np.sqrt(_w.sum(0))
            _scat = _devs.std(0, ddof=1) if len(_devs) > 1 else np.zeros(4)
            _meta0 = solver.images[_grp['images'][0]]
            _hx = 2048.0
            _ps = float(_meta0.get('orig_pixel_scale', 50.0))
            _row = dict(group=_g, instrument=_grp['instrument'],
                        detector=_grp['detector'], chip=_grp['chip'],
                        filter=_grp['filter'], epoch_id=_grp['epoch_id'],
                        mean_mjd=_grp['mean_mjd'], n_images=len(_devs))
            for _k, _nm in enumerate(_comp_names):
                _row[f'{_nm}_dev'] = float(_mean[_k])
                _row[f'{_nm}_sigma'] = float(_msig[_k])
                _row[f'{_nm}_scatter'] = float(_scat[_k])
                # displacement this deviation produces at the chip x-edge
                _row[f'{_nm}_edge_mas'] = float(_mean[_k] * _hx * _ps)
            _lin_rows.append(_row)
        if _lin_rows:
            _pd.DataFrame(_lin_rows).to_csv(
                output_dir / "epoch_distortion_linear.csv", index=False)
            print("  Saved: epoch_distortion_linear.csv  (per-group IVW "
                  "linear deviations from header priors)")
            for _row in _lin_rows:
                _tag = (f"{_row['instrument']}/{_row['detector']}/"
                        f"{_row['chip']}/{_row['filter']}/e{_row['epoch_id']}")
                _parts = []
                for _nm in _comp_names:
                    _ns = (_row[f'{_nm}_dev'] / _row[f'{_nm}_sigma']
                           if _row[f'{_nm}_sigma'] > 0 else np.nan)
                    _parts.append(f"{_nm}={_row[f'{_nm}_edge_mas']:+.2f}mas"
                                  f"({_ns:+.1f}s)")
                print(f"    linear dev g{_row['group']} {_tag}: "
                      + "  ".join(_parts))
    else:
        np.save(output_dir / "C_r.npy", C_r)

    # 4. Per-detection use flags (for reproducibility and hierarchical modelling)
    # use_for_fit[img]   : (n,) bool — detection used for ALIGNMENT (constrains r_hat)
    # use_for_astrom[img]: (n,) bool — detection used for ASTROMETRY (constrains v_hat)
    # star_indices[img]  : (n,) int  — indices into stellar_astrometry.csv rows
    _fit_data    = {}
    _astrom_data = {}
    _idx_data    = {}
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is None:
            continue
        _fit_data[img]    = d_img['use_for_fit']
        _astrom_data[img] = d_img.get('use_for_astrom', d_img['use_for_fit'])
        _idx_data[img]    = d_img['sidx']
    np.savez(output_dir / "use_for_fit.npz",    **_fit_data)
    np.savez(output_dir / "use_for_astrom.npz", **_astrom_data)
    np.savez(output_dir / "star_indices.npz",   **_idx_data)

    # 5. Per-detection GDC-frame residuals
    # detections.npz: one group of arrays per image, keyed as {img}_{field}.
    #   {img}_X_c, {img}_Y_c       — centered GDC pixel positions (X - Xo, Y - Yo)
    #   {img}_dx_gdc, {img}_dy_gdc — residuals in GDC frame [pixels]  (J⁻¹ @ resid_pseudo)
    #   {img}_C_hst                — (n,2,2) measurement covariance in GDC frame
    #   {img}_sidx                 — star indices into stellar_astrometry.csv
    #   {img}_use_for_fit          — bool, used for transformation fitting
    #   {img}_use_for_astrom       — bool, used for stellar astrometry
    gdc_resid = solver.compute_gdc_residuals(r_hat, v_hat, C_r=C_r, C_vT=C_vT)
    _det_data = {}
    n_det_total = 0
    for img, rd in gdc_resid.items():
        _det_data[f"{img}_X_c"]            = rd["X_c"]
        _det_data[f"{img}_Y_c"]            = rd["Y_c"]
        _det_data[f"{img}_dx_gdc"]         = rd["dx_gdc"]
        _det_data[f"{img}_dy_gdc"]         = rd["dy_gdc"]
        _det_data[f"{img}_C_hst"]          = rd["C_hst"]
        _det_data[f"{img}_C_gdc_total"]    = rd["C_gdc_total"]
        _det_data[f"{img}_sidx"]           = rd["sidx"]
        _det_data[f"{img}_use_for_fit"]    = rd["use_for_fit"]
        _det_data[f"{img}_use_for_astrom"] = rd["use_for_astrom"]
        n_det_total += len(rd["sidx"])
    np.savez_compressed(output_dir / "detections.npz", **_det_data)

    print(f"  Saved: stellar_astrometry.csv  "
          f"({len(g)} stars, {g['n_hst_used'].sum()} HST detections)")
    print(f"  Saved: detections.npz  ({len(gdc_resid)} images, {n_det_total} detections)")

    # 6. Machine-readable run configuration for downstream tools (e.g. hst_catalog_crossmatch)
    import json as _json
    from bp3m.solver import _SIGMA_ROT_DEG, _SIGMA_SCALE, _SIGMA_SKEW, _SIGMA_POINTING
    config = {
        'poly_order':   solver.poly_order,
        'n_r_per_image': solver.N_R,
        'n_images':     len(image_names),
        'n_stars':      solver.n_stars,
        'image_names':  image_names,   # ordered to match C_r blocks
        # Prior hyperparameters (same for all images; per-image means below)
        'prior_hyperparams': {
            'sigma_rot_deg':           solver._prior_sigma_rot_deg,
            'sigma_scale':             solver._prior_sigma_scale,
            'sigma_skew':              solver._prior_sigma_skew,
            'sigma_pointing_mas':      solver._prior_sigma_pointing,
            'use_pair_prior':          solver._use_pair_prior,
            'sigma_pair_rot_deg':      solver._prior_sigma_pair_rot_deg,
            'sigma_pair_scale':        solver._prior_sigma_pair_scale,
            'sigma_pair_skew':         solver._prior_sigma_pair_skew,
            'sigma_pair_pointing_mas': solver._prior_sigma_pair_pointing,
        },
        # Per-image prior: mean vector r_prior and precision matrix C_r_prior_inv
        # (C_r_prior_inv varies per image because the Jacobian depends on rotation)
        'image_priors': {
            img: {
                'r_prior':       solver._img_data[img]['r_prior'].tolist(),
                'C_r_prior_inv': solver._img_data[img]['C_r_prior_inv'].tolist(),
            }
            for img in image_names
            if img in solver._img_data
        },
    }
    if run_config:
        config.update(run_config)
    with open(output_dir / 'run_config.json', 'w') as _f:
        _json.dump(config, _f, indent=2)
