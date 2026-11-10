"""
Data loading + joining for the Coverage Soft-Spot pipeline.

Join path:
  games (gameId) -> plays (gameId, playId) -> pffScoutingData (gameId, playId, nflId)
                 -> tracking_<gameId>.csv  (gameId, playId, nflId, frameId)

Timing anchors (validated against the data):
  - 'ball_snap'    present in ~99.6% of plays  -> window start
  - 'pass_forward' present in ~90% of plays    -> throw time (primary measure point)
  - 'pass_arrived' present in <5%              -> NOT usable; do not rely on it
Fallback for the throw moment when pass_forward is missing: snap_frame + FALLBACK_FRAMES.
"""
from __future__ import annotations
import functools
import numpy as np
import pandas as pd

from . import geometry as geo

DATA = "data"
FALLBACK_SECONDS = 2.5
HZ = 10
FALLBACK_FRAMES = int(FALLBACK_SECONDS * HZ)

# Analysis population: ball actually thrown (exclude sacks/scrambles for clean route-vs-coverage).
KEEP_PASS_RESULTS = {"C", "I", "IN"}
WELL_POWERED_COVERAGES = {"Cover-1", "Cover-2", "Cover-3", "Quarters", "Cover-6", "Cover-0", "2-Man"}


@functools.lru_cache(maxsize=1)
def plays() -> pd.DataFrame:
    return pd.read_csv(f"{DATA}/plays.csv")


@functools.lru_cache(maxsize=1)
def players() -> pd.DataFrame:
    return pd.read_csv(f"{DATA}/players.csv")


@functools.lru_cache(maxsize=1)
def pff() -> pd.DataFrame:
    return pd.read_csv(f"{DATA}/pffScoutingData.csv")


@functools.lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    return pd.read_csv(f"{DATA}/games.csv")


def tracking_for_game(game_id: int) -> pd.DataFrame:
    return pd.read_csv(f"{DATA}/tracking/tracking_{game_id}.csv")


def eligible_plays(coverages: set | None = None) -> pd.DataFrame:
    """Plays where the ball was thrown, with a usable coverage label."""
    p = plays()
    mask = p.passResult.isin(KEEP_PASS_RESULTS)
    if coverages is None:
        coverages = WELL_POWERED_COVERAGES
    mask &= p.pff_passCoverage.isin(coverages)
    return p[mask].copy()


def _anchor_frames(trk_play: pd.DataFrame):
    """Return (snap_frame, throw_frame) for one play's tracking slice."""
    def first_event(names):
        for nm in names:
            ids = trk_play.loc[trk_play.event == nm, "frameId"]
            if not ids.empty:
                return int(ids.min())
        return None

    snap = first_event(["ball_snap", "autoevent_ballsnap"])
    throw = first_event(["pass_forward", "autoevent_passforward"])
    if snap is None:
        snap = int(trk_play.frameId.min())
    if throw is None:
        throw = snap + FALLBACK_FRAMES
    return snap, throw


def route_separations_for_game(game_id: int,
                               coverages: set | None = None,
                               measure_at: str = "throw") -> pd.DataFrame:
    """
    For every eligible play in a game, compute each route-runner's nearest-defender
    separation at the measurement frame. Returns one row per (play, receiver).

    Columns: gameId, playId, nflId, pff_passCoverage, offenseFormation,
             x, y (normalized, receiver position at measure frame),
             separation, nearest_def_nflId, closing_speed, frameId_measured.
    """
    elig = eligible_plays(coverages)
    elig = elig[elig.gameId == game_id]
    elig_play_ids = set(elig.playId)
    if not elig_play_ids:
        return pd.DataFrame()

    cov_map = elig.set_index("playId")[["pff_passCoverage", "offenseFormation",
                                        "possessionTeam", "defensiveTeam"]]

    trk = tracking_for_game(game_id)
    trk = trk[trk.playId.isin(elig_play_ids)].copy()
    trk = geo.normalize_play_direction(trk)

    pff_df = pff()
    routes = pff_df[(pff_df.gameId == game_id) & (pff_df.pff_role == "Pass Route")]
    route_ids = routes.groupby("playId")["nflId"].apply(set).to_dict()
    # alignment where each route-runner lined up, keyed by (playId, nflId)
    align_map = routes.set_index(["playId", "nflId"])["pff_positionLinedUp"].to_dict()
    # official position per player (QB/WR/TE/RB/...) from players.csv
    pos_map = dict(zip(players().nflId, players().officialPosition))

    out_rows = []
    for play_id, grp in trk.groupby("playId"):
        if play_id not in cov_map.index:
            continue
        meta = cov_map.loc[play_id]
        poss = meta["possessionTeam"]
        snap_f, throw_f = _anchor_frames(grp)
        measure_f = throw_f if measure_at == "throw" else snap_f

        # Line of scrimmage reference from the SNAP frame's ball position
        snap_frame_df = grp[grp.frameId == snap_f]
        ball_snap = snap_frame_df[snap_frame_df.team == "football"]
        if not ball_snap.empty:
            los_x = float(ball_snap.x.iloc[0])
            los_y = float(ball_snap.y.iloc[0])
        else:
            off_snap = snap_frame_df[snap_frame_df.team == poss]
            los_x = float(off_snap.x.max()) if not off_snap.empty else float("nan")
            los_y = geo.FIELD_Y / 2.0

        frame = grp[grp.frameId == measure_f]
        if frame.empty:
            # nearest available frame
            measure_f = int(grp.frameId.iloc[(grp.frameId - measure_f).abs().argmin()])
            frame = grp[grp.frameId == measure_f]

        players_f = frame[frame.team != "football"]
        offense = players_f[players_f.team == poss]
        defense = players_f[players_f.team != poss]
        if defense.empty or offense.empty:
            continue
        def_xy = defense[["x", "y"]].to_numpy()
        def_vel = geo.velocity_vectors(defense["s"].to_numpy(), defense["dir"].to_numpy())
        def_ids = defense["nflId"].to_numpy()

        # all 22 players' positions + velocities for the area tessellation
        all_xy = players_f[["x", "y"]].to_numpy()
        all_vel = geo.velocity_vectors(players_f["s"].to_numpy(), players_f["dir"].to_numpy())

        rset = route_ids.get(play_id, set())
        receivers = offense[offense.nflId.isin(rset)]
        for _, r in receivers.iterrows():
            rxy = np.array([r["x"], r["y"]])
            rvel = geo.velocity_vectors(np.array([r["s"]]), np.array([r["dir"]]))[0]
            dist, idx, sep_vec = geo.separation(rxy, def_xy)
            v_rel = rvel - def_vel[idx]
            cs = geo.closing_speed(sep_vec, v_rel)
            # field-bounded controlled area: positions-only (tau=0) and speed-aware (tau=1.5s)
            area_pos = geo.controlled_area(rxy, all_xy, grid_res=0.5, tau=0.0)
            area_spd = geo.controlled_area(rxy, all_xy, grid_res=0.5,
                                           target_vel=rvel, all_vel=all_vel, tau=1.5)
            out_rows.append({
                "gameId": game_id,
                "playId": int(play_id),
                "nflId": int(r["nflId"]),
                "position": pos_map.get(int(r["nflId"]), "UNK"),
                "alignment": align_map.get((play_id, int(r["nflId"])), "UNK"),
                "pff_passCoverage": meta["pff_passCoverage"],
                "offenseFormation": meta["offenseFormation"],
                "x": float(r["x"]),
                "y": float(r["y"]),
                "depth": float(r["x"]) - los_x,       # yards downfield of LOS (+ = downfield)
                "lateral": float(r["y"]) - los_y,     # yards from ball center (+ = toward +y sideline)
                "los_x": los_x,
                "separation": dist,
                "area_pos": area_pos,
                "area_spd": area_spd,
                "nearest_def_nflId": int(def_ids[idx]),
                "closing_speed": cs,
                "frameId_measured": int(measure_f),
            })
    return pd.DataFrame(out_rows)
