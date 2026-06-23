"""
run_joint_cte.py  —  standalone runner for the joint (r, γ_CTE, μ_pop) solver.

Usage
-----
  conda run -n bp3m-test python run_joint_cte.py \
      --field Leo_I \
      --data_root /home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results \
      --sigma_pm 0.0076 \
      --plx_pop 0.00394 \
      --sigma_plx_tot 0.0001 \
      --mu_pop_prior_sigma 0.5 \
      --n_iter 20

LVD parameters for common targets
-----------------------------------
Leo I  : d=254 kpc, σ_LOS=9.2 km/s
    sigma_pm      = 9.2 / (4.74047 * 254) = 0.0076  mas/yr
    plx_pop       = 1000 / 254             = 3.94e-3  mas
    sigma_plx_tot = 5% dist err            = 2e-4     mas

The mu_pop prior mean is warm-started automatically from the Gaia cross-match
field PM estimate (master_combined_v2.csv).
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Joint CTE + population mean PM alignment solver')
    parser.add_argument('--field', required=True,
                        help='Field name (subdirectory of data_root)')
    parser.add_argument('--data_root',
                        default='/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results',
                        help='Root data directory')
    # Population / LVD priors
    parser.add_argument('--sigma_pm', type=float, default=0.01,
                        help='LVD intrinsic PM dispersion (mas/yr)')
    parser.add_argument('--plx_pop', type=float, default=0.004,
                        help='LVD mean parallax (mas)')
    parser.add_argument('--sigma_plx_tot', type=float, default=1e-4,
                        help='LVD total parallax uncertainty (mas)')
    parser.add_argument('--mu_pop_prior_sigma', type=float, default=0.5,
                        help='Width of Gaussian prior on mu_pop (mas/yr)')
    # Iteration
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Max joint Gauss-Newton iterations')
    parser.add_argument('--member_sigma_clip', type=float, default=3.0,
                        help='Sigma for PM-distance membership selection')
    parser.add_argument('--regularize_gamma', type=float, default=1e-8,
                        help='Diagonal regularisation on H_gamma')
    parser.add_argument('--pm_sys_floor', type=float, default=0.2,
                        help='Systematic PM floor for membership sigma-clipping (mas/yr)')
    # CTE magnitude polynomial
    parser.add_argument('--cte_mag_poly_order', type=int, default=3,
                        help='Polynomial order for CTE magnitude dependence (default 3)')
    # Alignment options
    parser.add_argument('--poly_order', type=int, default=1)
    parser.add_argument('--hst_max_pm_unc', type=float, default=5.0)
    parser.add_argument('--hst_max_per_image', type=int, default=100_000)
    parser.add_argument('--hst_pm_sigma_diffuse', type=float, default=100.0)
    parser.add_argument('--pos_err_floor', type=float, default=5e-3)
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--plot_residuals', action='store_true')
    parser.add_argument('--use_sparse', action='store_true')
    parser.add_argument('--full_run', action='store_true',
                        help='Proceed to full joint solve loop after warm start '
                             '(default: stop after warm start)')
    parser.add_argument('--fit_cte_x', action='store_true',
                        help='Also fit CTE x-displacement (gamma_x). Off by default: '
                             'ACS CTE is primarily in the readout (y) direction and '
                             'fitting gamma_x tends to absorb non-CTE x-systematics.')
    args = parser.parse_args()

    from bp3m.pipeline.run_alignment_cte import run_alignment_joint_cte

    output = run_alignment_joint_cte(
        output_dir=Path(args.data_root),
        field_name=args.field,
        sigma_pm=args.sigma_pm,
        plx_pop=args.plx_pop,
        sigma_plx_tot=args.sigma_plx_tot,
        mu_pop_prior_sigma=args.mu_pop_prior_sigma,
        n_iter_joint=args.n_iter,
        member_sigma_clip=args.member_sigma_clip,
        regularize_gamma=args.regularize_gamma,
        pm_sys_floor=args.pm_sys_floor,
        mag_poly_order=args.cte_mag_poly_order,
        poly_order=args.poly_order,
        hst_max_pm_unc=args.hst_max_pm_unc,
        hst_max_per_image=args.hst_max_per_image,
        hst_pm_sigma_diffuse=args.hst_pm_sigma_diffuse,
        pos_err_floor=args.pos_err_floor,
        no_plots=args.no_plots,
        plot_residuals=args.plot_residuals,
        use_sparse=args.use_sparse,
        warmstart_only=not args.full_run,
        fit_cte_x=args.fit_cte_x,
    )
    print(f"\nDone. Results in: {output}")


if __name__ == '__main__':
    main()
