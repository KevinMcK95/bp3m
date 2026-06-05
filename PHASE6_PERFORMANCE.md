# Phase 6 Astrometry Performance

Phase 6 (`_measure_astrometry_proper`) is the dominant cost in `bp3m-v2`: it
fits a full 5-parameter astrometric solution (position, PM, parallax) for every
source by marginalising over the HST–Gaia alignment posterior.

## Measured timings (176k sources, 2.8M detections, 85 sub-images)

| Step | Before optimisation | After |
|------|---------------------|-------|
| `det_lookup` build | 171s | 18s |
| `tele_xyz` build (ephemeris) | 90s | 0.2s |
| `src_detections` build | 8s | 0s (reused) |
| 2nd/3rd call setup (reuse) | 261s | ~0s |
| Per-source fitting | ~82 src/s | ~100 src/s |
| **Total phase (first call)** | **~2420s** | **~1800s** |
| **Total phase (second call)** | **271s** | **~8s** |

## Optimisations applied

### Setup (one-time cost)
- `det_df.iterrows()` → `to_dict('records')` for `det_lookup` (10× faster)
- 85 serial astropy ephemeris calls → parallelised with `ThreadPoolExecutor` (500× faster)
- All three expensive lookups (`det_lookup`, `tele_xyz_cache`, `src_detections`)
  pre-built once and passed to all subsequent calls via `_p4` dict

### Per-source computation
- **C_r block assembly**: O(N²) Python inner loop → single `X_flat @ C_r_sub @ X_flat.T`
  BLAS call (vectorised)
- **`_build_system` outer loop**: all per-detection calls (`plane_project`,
  `plane_project_jacobian`, `get_parallax_factors`, `build_U_matrix`,
  `compute_poly_jacobian`) vectorised across N detections in one numpy call each
- **`build_X_matrix`**: eliminated for `poly_order=1` (common case); `y_obs`
  computed inline, `X_arr` reconstructed from `x_c`/`y_c` in `_build_system`
- **Outlier chi²**: N calls to `linalg.inv(2×2)` replaced with analytical
  batch inverse (`det = ad-bc`)
- **Load balancing**: sources sorted by detection count (heaviest first) then
  assigned round-robin to threads (LPT heuristic) — eliminates the observed
  17→100 src/s acceleration across the run

### Data integrity
- Gaia source IDs are read via `to_numpy(dtype=np.int64, na_value=0)` throughout
  to avoid float64 rounding of large 64-bit IDs (iterrows/items would silently
  truncate them)

## Current bottlenecks

1. **Per-source fitting: ~100 src/s** — still the dominant cost (1800s for 176k
   sources). The 10ms/source breaks down roughly as:
   - `_build_system` vectorised ops: ~2ms
   - `linalg.inv(Big_C)` for (2N × 2N) matrix: scales as O(N³) — **biggest
     single cost for sources with many detections**
   - Outlier rejection (up to 3 × `_build_system`): ~3× overhead for sources
     with outliers
   - Per-detection dict extraction at top of `_build_system`: ~0.5ms

2. **`det_lookup` build: 18s** — `to_dict('records')` on 2.8M rows; could be
   eliminated by indexing `det_df` directly with `set_index(['sub_name',
   'catalog_index'])` once

3. **GIL limits thread parallelism** — per-source work is ~50% Python overhead
   (dict extraction, list comprehensions), ~50% numpy. True multiprocessing
   would give better scaling but requires pickling large shared arrays.

## Next steps (approximate impact)

| Change | Expected win |
|--------|-------------|
| Columnar arrays: build parallel numpy arrays instead of list-of-dicts in the detection loop; pass directly to `_build_system` without dict extraction | ~1–2ms/source (10–20%) |
| Batch `linalg.inv(Big_C)` — or use Cholesky since Big_C is PSD | Larger win for high-N sources |
| Index `det_df` with `set_index` instead of building `det_lookup` dict | Eliminates 18s setup |
| Multiprocessing instead of threading (eliminates GIL) | Potentially 2–4× if Python overhead dominates |
| Skip outlier rejection for sources with few detections (N < 4) | Minor |
