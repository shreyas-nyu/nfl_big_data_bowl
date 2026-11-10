"""
Geometric primitives for the Coverage Soft-Spot pipeline.

Philosophy (the moat): we do NOT fit a parameterized model. We construct new
geometric quantities directly from the physics of player motion using linear
algebra -- velocity/acceleration vectors, separation vectors, closing speed,
and (optionally) bounded-acceleration reach sets -- all deterministic and
inspectable.

Angle convention (NFL Big Data Bowl): `dir` and `o` are in DEGREES, measured
clockwise from the positive-y axis (i.e. 0 deg points toward +y, 90 deg -> +x).
Therefore a velocity vector decomposes as:
    vx = s * sin(theta),   vy = s * cos(theta),   theta = radians(dir)
We verify downstream that ||(vx,vy)|| == s.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FIELD_X = 120.0   # yards, incl. both end zones
FIELD_Y = 53.3    # yards, field width


def velocity_vectors(speed: np.ndarray, direction_deg: np.ndarray) -> np.ndarray:
    """Return (N,2) velocity vectors from scalar speed + motion direction (deg)."""
    th = np.radians(direction_deg)
    vx = speed * np.sin(th)
    vy = speed * np.cos(th)
    return np.stack([vx, vy], axis=-1)


def numerical_acceleration(vel: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Central-difference acceleration vector (N,2) from a time-ordered velocity
    array (N,2). Endpoints use forward/backward difference. 10 Hz -> dt=0.1s.
    The dataset's scalar `a` is a smoothed magnitude with no direction; we derive
    a directional acceleration vector here instead.
    """
    a = np.gradient(vel, dt, axis=0)
    return a


def smooth(series: np.ndarray, window: int = 3) -> np.ndarray:
    """Light moving-average smoothing along axis 0 (10 Hz tracking is jittery)."""
    if window <= 1 or len(series) < window:
        return series
    kernel = np.ones(window) / window
    if series.ndim == 1:
        return np.convolve(series, kernel, mode="same")
    out = np.empty_like(series, dtype=float)
    for j in range(series.shape[1]):
        out[:, j] = np.convolve(series[:, j], kernel, mode="same")
    return out


def normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip coordinates so every play travels left->right (offense toward +x).
    Mutates a copy. Angles rotate by 180 deg under a full field flip.
    Expects columns: x, y, dir, o, playDirection.
    """
    df = df.copy()
    left = df["playDirection"] == "left"
    df.loc[left, "x"] = FIELD_X - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_Y - df.loc[left, "y"]
    for col in ("dir", "o"):
        if col in df.columns:
            df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0
    return df


def separation(receiver_xy: np.ndarray, defenders_xy: np.ndarray):
    """
    Nearest-defender separation for one receiver position.
    receiver_xy: (2,)  defenders_xy: (k,2)
    Returns (distance, index_of_nearest, separation_vector receiver-minus-defender).
    """
    d = defenders_xy - receiver_xy[None, :]        # (k,2) defender->receiver? no: def - rec
    dist = np.linalg.norm(d, axis=1)
    i = int(np.argmin(dist))
    sep_vec = receiver_xy - defenders_xy[i]        # points from nearest defender to receiver
    return float(dist[i]), i, sep_vec


def closing_speed(sep_vec: np.ndarray, rel_velocity: np.ndarray) -> float:
    """
    Component of relative velocity along the separation axis.
    Positive => gap is OPENING; negative => defender is CLOSING.
    sep_vec: receiver - defender (points toward receiver).
    rel_velocity: v_receiver - v_defender.
    """
    n = np.linalg.norm(sep_vec)
    if n == 0:
        return 0.0
    return float(np.dot(rel_velocity, sep_vec / n))


def voronoi_ownership(grid_pts: np.ndarray,
                      offense_xy: np.ndarray,
                      defense_xy: np.ndarray) -> np.ndarray:
    """
    Distance-based space ownership on a grid.
    Returns ownership = d_nearest_defender - d_nearest_offense  for each grid pt.
    > 0 => offense reaches first (space the defense concedes).
    grid_pts: (N,2). Fully vectorized matrix op (no training, no parameters).
    """
    def nearest(pts, players):
        d = np.linalg.norm(pts[:, None, :] - players[None, :, :], axis=2)
        return d.min(axis=1)
    return nearest(grid_pts, defense_xy) - nearest(grid_pts, offense_xy)


def _effective_positions(xy, vel, tau):
    """Position + velocity*tau: where each player will be in tau seconds (speed-aware).
    tau=0 reduces to plain positions (pure Voronoi). No acceleration term -> at most
    ONE stated assumption (the horizon tau); set tau=0 for the zero-parameter metric."""
    if tau == 0 or vel is None:
        return xy
    return xy + vel * tau


def controlled_area(target_xy: np.ndarray,
                    all_xy: np.ndarray,
                    grid_res: float = 0.5,
                    field_margin: float = 2.0,
                    depth_band: float = 15.0,
                    target_vel: np.ndarray | None = None,
                    all_vel: np.ndarray | None = None,
                    tau: float = 0.0) -> float:
    """
    Field-bounded Voronoi dominance AREA (sq yd) controlled by one target player.

    A grid point is 'controlled' by the target if the target is the nearest of all
    22 players. The grid is CLIPPED to the field width [0, FIELD_Y] (hard rule
    constraint) and to a local depth band around the players (nobody owns grass far
    from the play). Pure geometry: the only inputs are positions. Setting tau>0 with
    velocities makes it speed-aware (reachability flavor) -- a single stated horizon
    assumption, off by default.

    target_xy: (2,)  all_xy: (k,2) INCLUDING the target.
    Returns area in square yards.
    """
    eff_all = _effective_positions(all_xy, all_vel, tau)
    eff_tgt = _effective_positions(target_xy[None, :],
                                   None if target_vel is None else target_vel[None, :],
                                   tau)[0]
    # identify target row in all_xy by exact position match (pre-shift)
    # safer: append target explicitly and treat as index 0
    pts_players = np.vstack([eff_tgt[None, :], eff_all])

    # grid bounds: field width, local depth band around players
    x_lo = max(0.0, all_xy[:, 0].min() - depth_band)
    x_hi = min(FIELD_X, all_xy[:, 0].max() + depth_band)
    y_lo = max(0.0 + 0.0, 0.0)
    y_hi = FIELD_Y
    gx = np.arange(x_lo, x_hi + 1e-9, grid_res)
    gy = np.arange(y_lo, y_hi + 1e-9, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)           # (N,2)

    d = np.linalg.norm(P[:, None, :] - pts_players[None, :, :], axis=2)  # (N, k+1)
    owner = d.argmin(axis=1)                                  # 0 == target
    cell_area = grid_res * grid_res
    return float((owner == 0).sum() * cell_area)
