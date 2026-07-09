"""
bp3m-synthetic — end-to-end synthetic data test for BP3M.

Workflow:
  1. Generate synthetic HST observations (generate_synthetic_data)
  2. Run BP3M alignment on the synthetic data (run_alignment)
  3. Compare posteriors to truth and plot diagnostics (compare_synthetic_results)

Usage example:
  bp3m-synthetic --name Leo_I --seed 42 --output_dir /path/to/GaiaHub_results
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog='bp3m-synthetic',
        description='End-to-end synthetic test: generate → run BP3M → compare.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True,
                        help='Field name (must match the directory under output_dir)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Root output directory (same as passed to bp3m)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducibility')
    parser.add_argument('--syn_name', type=str, default='synthetic',
                        help='Subdirectory name for this synthetic run '
                             '(use distinct names to avoid overwriting previous runs)')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='Number of BP3M EM iterations (5 is usually enough for '
                             'synthetic data which has no outliers)')
    parser.add_argument('--no_split_ccd', action='store_true',
                        help='Disable CCD-half splitting (split_ccd=True by default)')
    parser.add_argument('--poly_order', type=int, default=1,
                        help='BP3M polynomial order for image parameters')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Regenerate synthetic data even if it already exists')
    parser.add_argument('--only_5p', action='store_true',
                        help='Exclude 2-param Gaia stars from the synthetic test')
    parser.add_argument('--all_5p_gaia', action='store_true',
                        help='Promote 2-param stars to 5-param by adding synthetic '
                             'Gaia PM+parallax measurements')
    parser.add_argument('--true_pm_center', type=float, nargs=2, default=None,
                        metavar=('PMRA', 'PMDEC'),
                        help='Override true PM for all stars: draw from '
                             'N((pmra, pmdec), true_pm_width²) mas/yr')
    parser.add_argument('--true_pm_width', type=float, default=0.1,
                        help='Width of PM draw around true_pm_center (mas/yr)')
    parser.add_argument('--true_parallax_center', type=float, default=None,
                        help='Override true parallax for all stars: draw from '
                             'N(center, true_parallax_width²) mas')
    parser.add_argument('--true_parallax_width', type=float, default=0.1,
                        help='Width of parallax draw around true_parallax_center (mas)')
    parser.add_argument('--zero_parallax', action='store_true',
                        help='Set true parallax = 0 for all stars')
    # ── Alignment parameter draw widths ──────────────────────────────────────
    parser.add_argument('--rot_sigma', type=float, default=0.01,
                        help='1σ width of rotation draw per image (degrees)')
    parser.add_argument('--ratio_sigma', type=float, default=1e-4,
                        help='1σ width of pixel-scale ratio draw (fractional)')
    parser.add_argument('--skew_sigma', type=float, default=1e-5,
                        help='1σ width of on/off-axis skew draws')
    parser.add_argument('--pointing_sigma', type=float, default=500.0,
                        help='1σ width of per-image pointing offset draw (mas)')
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--images', '--image', type=str, nargs='+', default=None,
                        help='Restrict to specific image names (both --images and --image work)')
    parser.add_argument('--bp3m_min_stars', type=int, default=0,
                        help='Exclude images with fewer than this many matched Gaia stars '
                             '(applied before the solver; 0 = keep all)')
    parser.add_argument('--skip_run', action='store_true',
                        help='Generate synthetic data but do not run BP3M '
                             '(useful to re-run BP3M manually with different settings)')
    parser.add_argument('--skip_compare', action='store_true',
                        help='Skip the comparison step')
    parser.add_argument('--no_align_prior', action='store_true',
                        help='Disable alignment parameter prior during BP3M fit '
                             '(sets a,b,c,d,delta_ra0,delta_dec0 prior precision to zero). '
                             'Useful for diagnosing prior-driven biases.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        print(f"Error: output_dir '{output_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    from bp3m.pipeline.synthetic import generate_synthetic_data, compare_synthetic_results

    # ── Step 1: generate ──────────────────────────────────────────────────────
    syn_dir = generate_synthetic_data(
        output_dir=output_dir,
        field_name=args.name,
        seed=args.seed,
        syn_name=args.syn_name,
        force_regenerate=args.force_regenerate,
        only_5p=args.only_5p,
        all_5p_gaia=args.all_5p_gaia,
        rot_sigma=args.rot_sigma,
        ratio_sigma=args.ratio_sigma,
        skew_sigma=args.skew_sigma,
        pointing_sigma=args.pointing_sigma,
        true_pm_center=args.true_pm_center,
        true_pm_width=args.true_pm_width,
        true_parallax_center=args.true_parallax_center,
        true_parallax_width=args.true_parallax_width,
        zero_parallax=args.zero_parallax,
        images=args.images,
        split_ccd=not args.no_split_ccd,
    )

    if args.skip_run:
        print(f"\nSynthetic data written to {syn_dir}")
        print("Skipping BP3M run (--skip_run).  Run manually:")
        print(f"  run_alignment(output_dir='{output_dir / args.name}', "
              f"field_name='{args.syn_name}')")
        return

    # ── Step 2: run BP3M ─────────────────────────────────────────────────────
    from bp3m.pipeline.run_alignment import run_alignment
    print(f"\n{'─'*50}")
    print(f"Running BP3M on synthetic data (n_iter={args.n_iter})")
    print(f"{'─'*50}")
    run_alignment(
        output_dir=output_dir / args.name,
        field_name=args.syn_name,
        split_ccd=not args.no_split_ccd,
        n_iter=args.n_iter,
        poly_order=args.poly_order,
        bp3m_min_stars=args.bp3m_min_stars,
        images=args.images,
        no_align_prior=args.no_align_prior,
    )

    if args.skip_compare:
        return

    # ── Step 3: compare ───────────────────────────────────────────────────────
    compare_synthetic_results(
        output_dir=output_dir,
        field_name=args.name,
        syn_name=args.syn_name,
    )


if __name__ == "__main__":
    main()
