"""
Unit tests for the spatial aggregation + soft-spot metric.
Verifies Separation Surplus is computed correctly on controlled synthetic data.
"""
import numpy as np
import pandas as pd
import pytest

from src import surface as S


def _synth():
    """Two coverages, same zone. Cov-A gets 5 yd separation, Cov-B gets 1 yd.
    All routes placed in one depth/lateral bin so the zone math is checkable."""
    rows = []
    for _ in range(50):
        rows.append({"depth": 8.0, "lateral": 0.0, "separation": 5.0, "pff_passCoverage": "A"})
        rows.append({"depth": 8.0, "lateral": 0.0, "separation": 1.0, "pff_passCoverage": "B"})
    return pd.DataFrame(rows)


def test_surface_mean_and_count():
    df = _synth()
    mean, cnt, xe, ye = S.surface(df)
    # the one occupied cell should hold the overall mean (3.0) and full count (100)
    occupied = np.isfinite(mean)
    assert cnt[occupied].sum() == 100
    assert mean[occupied][0] == pytest.approx(3.0)


def test_min_cell_threshold_masks_sparse():
    df = _synth().head(3)              # only 3 routes -> below MIN_CELL (8)
    mean, cnt, _, _ = S.surface(df)
    assert not np.isfinite(mean).any()   # all masked


def test_zscore_sign_and_symmetry():
    """Cov-A (above league) -> positive z; Cov-B (below) -> negative; symmetric."""
    df = _synth()
    za, _, _, _ = S.zscore_softspot_surface(df, "A")
    zb, _, _, _ = S.zscore_softspot_surface(df, "B")
    a = za[np.isfinite(za)]
    b = zb[np.isfinite(zb)]
    assert a.size and b.size
    assert (a > 0).all()
    assert (b < 0).all()
    # league mean is 3, std is 2 (values 5 and 1) -> A is +1σ, B is -1σ
    assert a[0] == pytest.approx(1.0, abs=1e-6)
    assert b[0] == pytest.approx(-1.0, abs=1e-6)


def test_zone_std_nonnegative():
    df = _synth()
    std, _, _, _ = S.zone_std_surface(df)
    finite = std[np.isfinite(std)]
    assert (finite >= 0).all()
    assert finite[0] == pytest.approx(2.0, abs=1e-6)   # std of {5,1} = 2
