"""
Unit tests for the geometric core. These verify the MATH is correct independent
of the data -- the foundation of the "positions and linear algebra, no model" claim.

Run:  .venv/bin/python -m pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from src import geometry as geo


# ----------------------------- velocity vectors -----------------------------
def test_velocity_magnitude_equals_speed():
    """||v|| must equal the scalar speed for any direction (angle convention check)."""
    speeds = np.array([0.0, 1.0, 3.7, 9.4])
    dirs = np.array([0.0, 90.0, 180.0, 275.3])
    v = geo.velocity_vectors(speeds, dirs)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), speeds, atol=1e-9)


def test_velocity_direction_convention():
    """dir=0 -> +y ; dir=90 -> +x  (NFL BDB clockwise-from-+y convention)."""
    v0 = geo.velocity_vectors(np.array([1.0]), np.array([0.0]))[0]
    v90 = geo.velocity_vectors(np.array([1.0]), np.array([90.0]))[0]
    np.testing.assert_allclose(v0, [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(v90, [1.0, 0.0], atol=1e-9)


# ----------------------------- separation -----------------------------------
def test_separation_picks_nearest():
    r = np.array([0.0, 0.0])
    defs = np.array([[3.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    dist, idx, vec = geo.separation(r, defs)
    assert idx == 1                      # nearest is the (0,1) defender
    assert dist == pytest.approx(1.0)
    np.testing.assert_allclose(vec, [0.0, -1.0])   # receiver - nearest defender


def test_separation_zero_when_coincident():
    r = np.array([2.0, 2.0])
    defs = np.array([[2.0, 2.0], [9.0, 9.0]])
    dist, idx, _ = geo.separation(r, defs)
    assert dist == pytest.approx(0.0) and idx == 0


# ----------------------------- closing speed --------------------------------
def test_closing_speed_sign():
    """Positive when the gap opens, negative when the defender closes."""
    sep_vec = np.array([2.0, 0.0])             # receiver east of defender
    opening = geo.closing_speed(sep_vec, np.array([1.0, 0.0]))   # rel vel pushes apart
    closing = geo.closing_speed(sep_vec, np.array([-1.0, 0.0]))  # rel vel closes
    assert opening > 0 and closing < 0
    assert opening == pytest.approx(1.0) and closing == pytest.approx(-1.0)


# ----------------------------- normalization --------------------------------
def test_normalize_flips_left_plays():
    df = pd.DataFrame({
        "x": [10.0, 110.0], "y": [10.0, 43.3],
        "dir": [0.0, 90.0], "o": [0.0, 90.0],
        "playDirection": ["left", "right"],
    })
    out = geo.normalize_play_direction(df)
    # left play flips: x -> 120-x, y -> 53.3-y, angle -> (a+180)%360
    assert out.loc[0, "x"] == pytest.approx(110.0)
    assert out.loc[0, "y"] == pytest.approx(43.3)
    assert out.loc[0, "dir"] == pytest.approx(180.0)
    # right play unchanged
    assert out.loc[1, "x"] == pytest.approx(110.0)
    assert out.loc[1, "dir"] == pytest.approx(90.0)


# ----------------------------- controlled area (Voronoi) --------------------
def _grid_region_area(pts, res=0.5, depth_band=15.0):
    """Replicate the function's inclusive grid to compute the true total area."""
    x_lo = max(0.0, pts[:, 0].min() - depth_band)
    x_hi = min(geo.FIELD_X, pts[:, 0].max() + depth_band)
    gx = np.arange(x_lo, x_hi + 1e-9, res)
    gy = np.arange(0.0, geo.FIELD_Y + 1e-9, res)
    return len(gx) * len(gy) * res * res


def test_controlled_area_partition_conserves():
    """Each player's cell areas must sum to the gridded field region (partition)."""
    pts = np.array([[40, 10], [45, 20], [50, 40], [43, 27], [60, 27],
                    [38, 15], [55, 35], [48, 25]], dtype=float)
    areas = [geo.controlled_area(pts[i], pts, grid_res=0.5, depth_band=15) for i in range(len(pts))]
    # CONSERVATION: every grid cell owned by exactly one player, so the per-player
    # areas sum EXACTLY to the gridded region (no double-counting). This also guards
    # against the target-duplication bug.
    assert sum(areas) == pytest.approx(_grid_region_area(pts), rel=1e-9)
    assert all(a >= 0 for a in areas)


def test_controlled_area_field_bounded():
    """A lone player owns the whole gridded region, but no more (field-bounded)."""
    pts = np.array([[60.0, 26.65]])
    area = geo.controlled_area(pts[0], pts, grid_res=0.5, depth_band=15)
    assert area == pytest.approx(_grid_region_area(pts), rel=1e-6)


def test_voronoi_ownership_sign():
    """ownership>0 where offense is nearer than defense."""
    grid = np.array([[41.0, 27.0], [59.0, 27.0]])
    off = np.array([[40.0, 27.0]])
    deff = np.array([[60.0, 27.0]])
    own = geo.voronoi_ownership(grid, off, deff)
    assert own[0] > 0      # near offense
    assert own[1] < 0      # near defense
