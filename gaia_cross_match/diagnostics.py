import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import KDTree

def apply_affine(x, y, A, B, C, D, xs_o, ys_o, xt_o, yt_o):
    dx, dy = x - xs_o, y - ys_o
    return xt_o + A * dx + B * dy, yt_o + C * dx + D * dy

def compute_mahalanobis(dx, dy, C):
    det = C[:,0,0]*C[:,1,1] - C[:,0,1]**2
    inv_xx = C[:,1,1] / det
    inv_yy = C[:,0,0] / det
    inv_xy = -C[:,0,1] / det
    return np.sqrt(dx**2 * inv_xx + dy**2 * inv_yy + 2 * dx * dy * inv_xy)

def generate_diagnostics(target_dir, gaia_csv_path):
    image_name = os.path.basename(target_dir)
    print(f"Generating diagnostics for {image_name}...")

    # 1. Load Data
    matched_path = os.path.join(target_dir, "matched_gaia.csv")
    trans_path = os.path.join(target_dir, "transformation.csv")
    catalog_path = os.path.join(target_dir, "catalog.fits")

    if not all(os.path.exists(p) for p in [matched_path, trans_path, catalog_path]):
        print(f"Missing files in {target_dir}"); return

    m_df = pd.read_csv(matched_path)
    t_df = pd.read_csv(trans_path) # Simple read
    # Map transformation params to a dict
    trans = dict(zip(t_df['parameter'], t_df['value']))
    
    A, B, C, D = trans['A'], trans['B'], trans['C'], trans['D']
    xs_o, ys_o, xt_o, yt_o = trans['xs_o'], trans['ys_o'], trans['xt_o'], trans['yt_o']
    M = np.array([[A, B], [C, D]])

    # Load Full Catalogs
    hst_cat = fits.getdata(catalog_path)
    # Handle byte order for NumPy
    x_h = hst_cat['x_gdc'].byteswap().astype(np.float64)
    y_h = hst_cat['y_gdc'].byteswap().astype(np.float64)
    cxx_h = hst_cat['cov_xx_gdc'].byteswap().astype(np.float64)
    cyy_h = hst_cat['cov_yy_gdc'].byteswap().astype(np.float64)
    cxy_h = hst_cat['cov_xy_gdc'].byteswap().astype(np.float64)

    gaia_full = pd.read_csv(gaia_csv_path)
    # Filter for finite just in case
    gaia_full = gaia_full[np.isfinite(gaia_full.ra) & np.isfinite(gaia_full.dec)]

    # 2. Re-project everything to find "Rejected" stars
    # We need the projected Gaia positions in the pixel frame of this image.
    # The transformation.csv has the metadata we used.
    r0, d0 = trans['ra_cen'], trans['dec_cen']
    orientat = trans['orientat']
    pixel_scale = trans['pixel_scale']
    x_cen, y_cen = trans['x_cen'], trans['y_cen']

    # Sky -> Pix projection (Simplified version used in cross_match_cli)
    def rd2pix(r, d):
        tr = np.pi/180.0
        c_ra, s_ra = np.cos((r-r0)*tr), np.sin((r-r0)*tr)
        c_de, s_de = np.cos(d*tr), np.sin(d*tr)
        c_d0, s_d0 = np.cos(d0*tr), np.sin(d0*tr)
        rrrr = s_d0*s_de + c_d0*c_de*c_ra
        dx_deg = np.degrees(c_de*s_ra/rrrr)
        dy_deg = np.degrees((c_d0*s_de - s_d0*c_de*c_ra)/rrrr)
        
        x_sky, y_sky = -dx_deg, dy_deg
        theta_init = np.radians(-orientat)
        scale_deg = pixel_scale / 3600.0
        x_p = x_cen + (x_sky * np.cos(theta_init) - y_sky * np.sin(theta_init)) / scale_deg
        y_p = y_cen + (x_sky * np.sin(theta_init) + y_sky * np.cos(theta_init)) / scale_deg
        return x_p, y_p

    xg_p, yg_p = rd2pix(gaia_full.ra, gaia_full.dec)
    
    # Filter Gaia projection for finite values
    g_finite = np.isfinite(xg_p) & np.isfinite(yg_p)
    xg_p, yg_p = xg_p[g_finite], yg_p[g_finite]
    gaia_v = gaia_full[g_finite]

    # 3. Identify Matches vs Rejections
    # Project ALL HST to Gaia pixel frame using fit
    xh_in_g, yh_in_g = apply_affine(x_h, y_h, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
    
    # Filter HST projection for finite values
    h_finite = np.isfinite(xh_in_g) & np.isfinite(yh_in_g)
    xh_in_g, yh_in_g = xh_in_g[h_finite], yh_in_g[h_finite]
    x_h_v, y_h_v = x_h[h_finite], y_h[h_finite]
    cxx_h_v, cyy_h_v, cxy_h_v = cxx_h[h_finite], cyy_h[h_finite], cxy_h[h_finite]
    
    # Simple KDTree search to find candidates
    tree_g = KDTree(np.column_stack([xg_p, yg_p]))
    ds, g_idx = tree_g.query(np.column_stack([xh_in_g, yh_in_g]), distance_upper_bound=20.0)
    
    valid = ds < 20.0
    h_idx_v = np.where(h_finite)[0][valid] # Map back to original indices
    g_idx_v = g_idx[valid]
    
    # Calculate residuals and sigmas for all valid candidates
    dx = xg_p.values[g_idx_v] - xh_in_g[valid]
    dy = yg_p.values[g_idx_v] - yh_in_g[valid]
    
    # Propagate covariance
    cxx_tot = 0.0025 + (A**2 * cxx_h_v[valid] + B**2 * cyy_h_v[valid] + 2*A*B*cxy_h_v[valid])
    cyy_tot = 0.0025 + (C**2 * cxx_h_v[valid] + D**2 * cyy_h_v[valid] + 2*C*D*cxy_h_v[valid])
    cxy_tot = (A*C*cxx_h_v[valid] + B*D*cyy_h_v[valid] + (A*D + B*C)*cxy_h_v[valid])
    
    # Mahalanobis
    det = cxx_tot * cyy_tot - cxy_tot**2
    sigs = np.sqrt((dx**2 * cyy_tot + dy**2 * cxx_tot - 2*dx*dy*cxy_tot)/det)
    
    # Build a candidate DF
    cand_df = pd.DataFrame({
        'h_idx': h_idx_v, 'g_idx': g_idx_v, 'x': x_h_v[valid], 'y': y_h_v[valid],
        'ra': gaia_v.ra.values[g_idx_v], 'dec': gaia_v.dec.values[g_idx_v],
        'mag': gaia_v.gmag.values[g_idx_v], 'dx': dx, 'dy': dy, 'sigma': sigs,
        'cxx': cxx_tot, 'cyy': cyy_tot
    })
    
    # Filter candidate DF for finiteness one last time to be safe
    cand_df = cand_df[np.isfinite(cand_df['dx']) & np.isfinite(cand_df['dy']) & np.isfinite(cand_df['sigma'])]

    # Separate Selected vs Rejected
    # Use positional index check for simplicity and robust matching
    tree_m = KDTree(np.column_stack([m_df.hst_x_gdc, m_df.hst_y_gdc]))
    dm, im = tree_m.query(np.column_stack([cand_df.x, cand_df.y]), distance_upper_bound=1e-3)
    is_match = dm < 1e-3
    
    matched = cand_df[is_match]
    rejected = cand_df[~is_match & (cand_df.sigma < 15.0)] # Only show "near" rejections

    # 4. PLOTTING
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle(f"Diagnostics: {image_name}", fontsize=20)

    # Panel 1: Pixel Map
    ax = axes[0, 0]
    if len(rejected) > 0: ax.scatter(rejected.x, rejected.y, c='red', s=5, alpha=0.3, label='Rejected')
    if len(matched) > 0: ax.scatter(matched.x, matched.y, c='blue', s=10, alpha=0.7, label='Matched')
    ax.set_title("Field Map (HST Pixels)"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend()

    # Panel 2: Sky Map
    ax = axes[0, 1]
    if len(rejected) > 0: ax.scatter(rejected.ra, rejected.dec, c='red', s=5, alpha=0.3, label='Rejected')
    if len(matched) > 0: ax.scatter(matched.ra, matched.dec, c='blue', s=10, alpha=0.7, label='Matched')
    ax.invert_xaxis()
    ax.set_title("Field Map (Sky)"); ax.set_xlabel("RA"); ax.set_ylabel("Dec"); ax.legend()

    # Panel 3: XY Residuals
    ax = axes[1, 0]
    if len(rejected) > 0: ax.scatter(rejected.dx, rejected.dy, c='red', s=10, alpha=0.3)
    if len(matched) > 0: ax.scatter(matched.dx, matched.dy, c='blue', s=15, alpha=0.7)
    ax.axhline(0, color='k', lw=1, ls='--'); ax.axvline(0, color='k', lw=1, ls='--')
    if len(matched) > 0:
        lim = max(matched.dx.abs().max(), matched.dy.abs().max()) * 2.5
        if np.isfinite(lim) and lim > 0: ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title("XY Residuals (Gaia - HST_proj)"); ax.set_xlabel("dX (pix)"); ax.set_ylabel("dY (pix)")

    # Panel 4: Normalized Residuals
    ax = axes[1, 1]
    if len(rejected) > 0: ax.scatter(rejected.dx/np.sqrt(rejected.cxx), rejected.dy/np.sqrt(rejected.cyy), c='red', s=10, alpha=0.2)
    if len(matched) > 0: ax.scatter(matched.dx/np.sqrt(matched.cxx), matched.dy/np.sqrt(matched.cyy), c='blue', s=15, alpha=0.5)
    ax.add_artist(plt.Circle((0,0), 1, color='k', fill=False, ls='--', alpha=0.5))
    ax.add_artist(plt.Circle((0,0), 5, color='r', fill=False, ls=':', alpha=0.5))
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_title("Normalized Residuals"); ax.set_xlabel("dX / sigma_x"); ax.set_ylabel("dY / sigma_y")

    # Panel 5: Combined Residual vs Magnitude (Log-Y)
    ax = axes[2, 0]
    if len(rejected) > 0: ax.scatter(rejected.mag, np.sqrt(rejected.dx**2 + rejected.dy**2), c='red', s=5, alpha=0.2, label='Rejected')
    if len(matched) > 0: ax.scatter(matched.mag, np.sqrt(matched.dx**2 + matched.dy**2), c='blue', s=10, alpha=0.6, label='Matched')
    ax.set_yscale('log')
    ax.set_title("Residual Magnitude vs Gaia Mag"); ax.set_xlabel("Gaia G"); ax.set_ylabel("Combined Residual (pixels)"); ax.legend()

    # Panel 6: Sigma Histogram
    ax = axes[2, 1]
    bins = np.linspace(0, 15, 60)
    if len(rejected) > 0: ax.hist(rejected.sigma, bins=bins, color='red', alpha=0.3, label='Rejected')
    if len(matched) > 0: ax.hist(matched.sigma, bins=bins, color='blue', alpha=0.6, label='Matched')
    ax.axvline(5, color='r', ls='--', label='5-sigma limit')
    ax.set_title("Sigma Distribution"); ax.set_xlabel("Mahalanobis Distance (sigma)"); ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(target_dir, "diagnostic_plots_standalone.png")
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--gaia-csv", required=True)
    args = parser.parse_args()
    generate_diagnostics(args.img_dir, args.gaia_csv)
