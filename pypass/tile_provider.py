"""Exact shared PSF-tile evaluation for the JAX batch path.

Historically every star spatially interpolated its own blended PSF twice
(raw + coefficients) over the FULL 101x101 model arrays — a 16-model
Catmull-Rom contraction costing ~163k flops and ~0.5 MB of temporaries per
call, 297k calls per 30k-star chip — and then carried its own float64
coefficient tile ((n_stars, ts, ts): 0.66 GB per 85k chip).

Both costs are unnecessary because everything downstream of the
interpolation is LINEAR in the PSF models:

  * ``spline_filter`` is linear, so blending prefiltered models equals
    prefiltering the blended model (already exploited by psf_coeffs_cube);
  * cropping the central (ts, ts) tile is a slice, so
    ``crop(Σ w_k · C_k) == Σ w_k · crop(C_k)``.

``ExactTileBlender`` therefore pre-crops the coefficient cube ONCE to a
(n_psf, ts, ts) tile cube (~0.7 MB) and evaluates each star's tile as the
same 16-weight contraction over the tiny cropped models — bit-equivalent to
the legacy per-star path (weights replicate ``interpolate_psf`` exactly;
summation-order float noise is ≲1e-16 relative, i.e. zero measurable
position offset against the 1e-4 px tolerance).

``LazyTiles`` exposes the per-star tiles through the same indexing contract
as the old (n_stars, ts, ts) array, materializing at most one fit_batch_jax
chunk at a time — per-star tile storage drops to (n_stars, 16) weights +
(n_stars, 16) int32 stencil indices.

Calibration record (2026-09-07, M31 j6d508ngq, 4000-star isolated A/B): the
originally proposed QUANTIZED cell tiles failed the user's 1e-4 px extreme-
max tolerance at every cell size (piecewise-constant: 2.1e-3 px max at
10 px cells; bilinear corner-blend: passes only at 10 px, max 2.1e-5 px,
with discrete-flip tails at >=25 px).  The exact blender supersedes both.
"""

import numpy as np


def _catmull_rom_weights_vec(t):
    """Catmull-Rom weights for fractional positions t. Shape (..., 4).

    Mirrors pypass.core._catmull_rom_weights exactly.
    """
    t = np.asarray(t, dtype=np.float64)
    t2, t3 = t * t, t * t * t
    return np.stack([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ], axis=-1)


class ExactTileBlender:
    """Exact per-star fit-window tiles from a pre-cropped coefficient cube."""

    def __init__(self, psf_cube, psf_coeffs_cube, xs, ys, psf_scale, hw,
                 x_offset=0.0, y_offset=0.0):
        from ._jax_kernel import tile_side, tile_radius

        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.psf_scale = int(psf_scale)
        self.hw = int(hw)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        self.ts = tile_side(hw, psf_scale)
        self.tr = tile_radius(hw, psf_scale)

        n_psf, ny_psf, nx_psf = psf_coeffs_cube.shape
        hy, hx = (ny_psf - 1) // 2, (nx_psf - 1) // 2
        tr = self.tr
        # Edge-pad handled upstream for non-standard PSFs; STDPSF arrays are
        # always larger than the tile.
        self.tile_cube = np.ascontiguousarray(
            psf_coeffs_cube[:, hy - tr:hy + tr + 1, hx - tr:hx + tr + 1])
        self.peak_cube = np.ascontiguousarray(
            psf_cube[:, hy, hx].astype(np.float64))
        self.n_psf = n_psf
        self.nx_g = len(self.xs)
        self.ny_g = len(self.ys)
        self._single = (n_psf == 1 or self.nx_g < 2 or self.ny_g < 2)

    # -------------------------------------------------------------- weights
    def weights(self, x_arr, y_arr):
        """(W, K): per-star 16 contraction weights and model indices.

        Replicates interpolate_psf's fractional-index + 4x4 Catmull-Rom
        stencil exactly (np.interp handles non-uniform grids; indices are
        clamped at grid edges just like the legacy code).
        """
        xd = np.asarray(x_arr, dtype=np.float64) + self.x_offset
        yd = np.asarray(y_arr, dtype=np.float64) + self.y_offset
        n = len(xd)
        if self._single:
            W = np.zeros((n, 16)); W[:, 0] = 1.0
            K = np.zeros((n, 16), dtype=np.int32)
            return W, K

        gx = np.arange(self.nx_g, dtype=np.float64)
        gy = np.arange(self.ny_g, dtype=np.float64)
        tx_full = np.interp(xd, self.xs, gx)
        ty_full = np.interp(yd, self.ys, gy)
        ix = np.minimum(np.floor(tx_full), self.nx_g - 2).astype(np.int64)
        iy = np.minimum(np.floor(ty_full), self.ny_g - 2).astype(np.int64)
        tx = tx_full - ix
        ty = ty_full - iy
        wx = _catmull_rom_weights_vec(tx)          # (n, 4)
        wy = _catmull_rom_weights_vec(ty)
        off = np.arange(-1, 3)
        ix_idx = np.clip(ix[:, None] + off[None, :], 0, self.nx_g - 1)
        iy_idx = np.clip(iy[:, None] + off[None, :], 0, self.ny_g - 1)
        K = (iy_idx[:, :, None] * self.nx_g
             + ix_idx[:, None, :]).reshape(n, 16).astype(np.int32)
        W = (wy[:, :, None] * wx[:, None, :]).reshape(n, 16)
        return W, K

    # ------------------------------------------------------------- evaluate
    def blend(self, W, K, out=None):
        """Contract tiles for a batch: (m, 16) weights/indices -> (m, ts, ts).

        Accumulates over the 16 stencil slots (16 gathers of (m, ts, ts))
        instead of one (m, 16, ts, ts) gather, keeping the transient at one
        chunk-sized array.
        """
        m = len(W)
        if out is None:
            out = np.zeros((m, self.ts, self.ts), dtype=np.float64)
        else:
            out[:] = 0.0
        for j in range(16):
            out += W[:, j, None, None] * self.tile_cube[K[:, j]]
        return out

    def blend_one(self, w, k):
        return np.einsum('f,fab->ab', w, self.tile_cube[k])

    def peaks(self, W, K):
        """Raw PSF value at the exact centre pixel per star."""
        return np.einsum('nf,nf->n', W, self.peak_cube[K])


class LazyTiles:
    """Per-star tiles with the (n, ts, ts) array's indexing contract.

    Stores only the contraction weights/indices; slices (fit_batch_jax
    chunks) materialize on access, single-star access blends one tile.
    """

    def __init__(self, blender, W, K):
        self.blender = blender
        self.W = W
        self.K = K

    def __len__(self):
        return len(self.W)

    @property
    def shape(self):
        return (len(self.W), self.blender.ts, self.blender.ts)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.blender.blend_one(self.W[key], self.K[key])
        return self.blender.blend(self.W[key], self.K[key])
