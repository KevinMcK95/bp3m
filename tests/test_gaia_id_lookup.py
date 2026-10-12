"""GaiaIdFloatMap / _in_sorted must give exactly the answers of the dict / Series.isin(set) they replace."""
import numpy as np
import pandas as pd

from bp3m.data_loader_flc import GaiaIdFloatMap, _in_sorted


def _ids(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(10**18, 6 * 10**18, n, dtype=np.int64)
    # neighbours 1..8 apart collide in float64 (19-digit ids): exercises the last-id-wins rule
    return np.unique(np.r_[base, base[:500] + rng.integers(1, 9, 500)])


def test_float_map_matches_dict_including_collisions():
    ids = _ids()
    d = {float(g): np.int64(g) for g in ids}
    m = GaiaIdFloatMap(ids)
    assert len(m.keys) == len(d) < len(ids)          # collisions present
    q = np.r_[ids.astype(float), np.random.default_rng(1).uniform(1e18, 6e18, 500), [np.nan]]
    want = np.array([d.get(f, np.int64(-1)) for f in q], dtype=np.int64)
    assert np.array_equal(m.lookup(q, -1), want)
    assert all(m.get(f, -1) == d.get(f, -1) for f in q[:300])


def test_in_sorted_matches_isin():
    ids = _ids()
    rng = np.random.default_rng(2)
    v = np.r_[ids[rng.integers(0, len(ids), 3000)], rng.integers(1, 2**62, 300), [-1, 0]]
    assert np.array_equal(_in_sorted(ids, v), pd.Series(v).isin(set(ids)).to_numpy())
    assert _in_sorted(ids[:0], v).sum() == 0 and len(_in_sorted(ids, v[:0])) == 0
