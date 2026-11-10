"""
Spatial aggregation: turn per-route separations into a per-coverage separation
SURFACE over the field, then into a SOFT-SPOT map (separation above/below the
coverage's own league baseline for that zone).

Coordinates are LOS-RELATIVE (play already normalized to go left->right):
  depth   = receiver_x - LOS_x   (yards downfield of the line of scrimmage)
  lateral = receiver_y - ball_y  (yards from the ball's lateral spot; + toward +y)
This makes routes from different field positions stack into a common frame, so a
12-yard dig on one play aligns with a 12-yard dig on another. We then grid this
relative frame and average separation per zone.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# LOS-relative grid. depth: behind LOS to deep; lateral: sideline to sideline.
X_BINS = np.arange(-5, 40 + 1e-9, 2.5)      # depth downfield of LOS, 2.5-yd bins
Y_BINS = np.arange(-27, 27 + 1e-9, 2.5)     # lateral from ball, 2.5-yd bins
MIN_CELL = 8                                # min routes per cell to report


def surface(df: pd.DataFrame, value: str = "separation"):
    """
    Mean of `value` per (depth,lateral) cell.
    Returns (grid_mean, grid_count, xedges, yedges); grid rows=lateral, cols=depth.
    """
    sums, xe, ye = np.histogram2d(df.depth, df.lateral, bins=[X_BINS, Y_BINS], weights=df[value])
    cnts, _, _ = np.histogram2d(df.depth, df.lateral, bins=[X_BINS, Y_BINS])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts >= MIN_CELL, sums / cnts, np.nan)
    return mean.T, cnts.T, xe, ye      # transpose so rows=y, cols=x


def coverage_surfaces(df: pd.DataFrame, value: str = "separation") -> dict:
    """One separation surface per coverage."""
    return {cov: surface(sub, value) for cov, sub in df.groupby("pff_passCoverage")}


def baseline_surface(df: pd.DataFrame, value: str = "separation"):
    """League-wide surface across all coverages (the expectation per zone)."""
    return surface(df, value)


def softspot_surface(df: pd.DataFrame, coverage: str, value: str = "separation"):
    """
    LEVEL view: (coverage surface) - (league baseline surface) per zone.
    Dominated by the man-vs-zone overall tightness difference, so man coverages
    read red everywhere and zone green everywhere. Useful for the big-picture
    'is this coverage tighter/looser than average' statement.
    """
    base_mean, _, xe, ye = baseline_surface(df, value)
    cov_mean, cov_cnt, _, _ = surface(df[df.pff_passCoverage == coverage], value)
    delta = cov_mean - base_mean
    return delta, cov_cnt, xe, ye


def zone_std_surface(df: pd.DataFrame, value: str = "separation"):
    """Per-zone standard deviation of `value` across all routes (E[x^2]-E[x]^2)."""
    m1, cnt, xe, ye = surface(df, value)
    df2 = df.copy()
    df2["_sq"] = df2[value] ** 2
    m2, _, _, _ = surface(df2, value="_sq")
    var = m2 - m1 ** 2
    return np.sqrt(np.clip(var, 0, None)), cnt, xe, ye


def zscore_softspot_surface(df: pd.DataFrame, coverage: str, value: str = "separation"):
    """
    PRIMARY soft-spot definition: z-score of a coverage's per-zone separation
    against the league distribution in that zone.

        z(zone) = (coverage_mean(zone) - league_mean(zone)) / league_std(zone)

    Scale-free: positive (green) => softer than typical for that zone by z std devs;
    negative (red) => tighter. Dividing by the per-zone std removes the man-vs-zone
    magnitude gap that made raw-yard maps saturate all-red / all-green, so BOTH the
    overall level and the within-coverage spatial shape are visible.
    """
    league_mean, _, xe, ye = baseline_surface(df, value)
    league_std, _, _, _ = zone_std_surface(df, value)
    cov_mean, cov_cnt, _, _ = surface(df[df.pff_passCoverage == coverage], value)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(league_std > 1e-6, (cov_mean - league_mean) / league_std, np.nan)
    return z, cov_cnt, xe, ye


def intra_softspot_surface(df: pd.DataFrame, coverage: str, value: str = "separation"):
    """
    SHAPE view (primary): soft spots RELATIVE TO THIS COVERAGE'S OWN AVERAGE.
    delta = coverage_surface(zone) - mean(coverage_surface over all zones).
    This removes the overall tightness level and isolates WHERE, within a given
    coverage, receivers win the most/least separation -- the actionable soft spots.
    Positive (green) => softer than this coverage's own norm; negative (red) => tighter.
    """
    cov_mean, cov_cnt, xe, ye = surface(df[df.pff_passCoverage == coverage], value)
    own_avg = np.nanmean(cov_mean)
    delta = cov_mean - own_avg
    return delta, cov_cnt, xe, ye
