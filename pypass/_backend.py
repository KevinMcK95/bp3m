"""Backend selection logic for py1pass fitting kernels.

Supported backends:
  'numpy' — pure NumPy/scipy implementation (always available)
  'jax'   — JAX vmap+jit implementation (requires jax package)
  'auto'  — use JAX when available and n_stars exceeds the threshold,
             otherwise fall back to NumPy

Environment variables:
  PYPASS_BACKEND         — override default backend ('numpy'|'jax'|'auto')
  PYPASS_JAX_THRESHOLD   — min stars to trigger JAX in auto mode (default 2000
                            CPU-only, 500 with GPU/TPU/MPS)
"""

import os

# ---------------------------------------------------------------------------
# JAX availability probe (import-time, cached)
# ---------------------------------------------------------------------------

JAX_AVAILABLE = False
_HAS_ACCELERATOR = False

try:
    import jax as _jax
    JAX_AVAILABLE = True
    _HAS_ACCELERATOR = any(
        d.platform in ('gpu', 'tpu')
        for d in _jax.devices()
    )
except Exception:
    pass

_THRESHOLD_CPU = 2000
_THRESHOLD_GPU = 500


def resolve_backend(backend: str, n_stars: int) -> str:
    """Return 'numpy' or 'jax' given the requested backend and star count.

    Parameters
    ----------
    backend : 'auto' | 'numpy' | 'jax'
        Requested backend.  'auto' reads PYPASS_BACKEND env var as a
        secondary default before applying the threshold heuristic.
    n_stars : int
        Number of stars to be fitted in this call.

    Returns
    -------
    str — 'numpy' or 'jax'

    Raises
    ------
    ImportError  if backend='jax' but JAX is not installed
    ValueError   if backend is not one of the three valid strings
    """
    valid = {'auto', 'numpy', 'jax'}
    if backend not in valid:
        raise ValueError(f"backend must be one of {valid!r}, got {backend!r}")

    # Environment variable as secondary default (CLI/API arg takes precedence)
    effective = backend
    if effective == 'auto':
        env = os.environ.get('PYPASS_BACKEND', 'auto').lower().strip()
        if env in valid:
            effective = env

    if effective == 'numpy':
        return 'numpy'

    if effective == 'jax':
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed.  Install with: pip install jax\n"
                "For GPU support see https://jax.readthedocs.io/en/latest/installation.html"
            )
        return 'jax'

    # auto
    if not JAX_AVAILABLE:
        return 'numpy'

    threshold = int(os.environ.get(
        'PYPASS_JAX_THRESHOLD',
        _THRESHOLD_GPU if _HAS_ACCELERATOR else _THRESHOLD_CPU,
    ))
    return 'jax' if n_stars >= threshold else 'numpy'


def backend_info() -> dict:
    """Return a summary dict of the current backend environment."""
    return {
        'jax_available':   JAX_AVAILABLE,
        'has_accelerator': _HAS_ACCELERATOR,
        'threshold_cpu':   _THRESHOLD_CPU,
        'threshold_gpu':   _THRESHOLD_GPU,
        'env_backend':     os.environ.get('PYPASS_BACKEND', 'auto'),
        'env_threshold':   os.environ.get('PYPASS_JAX_THRESHOLD', None),
    }
