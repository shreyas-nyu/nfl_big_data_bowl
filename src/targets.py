"""
Parse target info from playDescription and attach the targeted receiver to the
route-separation table, for Test 1 (does the ball go where the map says space is?).

playDescription format is highly regular, e.g.:
  "(13:33) (Shotgun) T.Brady pass incomplete deep right to C.Godwin."
We extract:
  - pass_depth:     'short' | 'deep'
  - pass_direction: 'left' | 'middle' | 'right'
  - target_name:    receiver last-name token after ' to ' (e.g. 'C.Godwin')
  - completed:      bool (no 'incomplete'/'intended' and passResult == 'C')
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

from . import loader

DEPTH_DIR = re.compile(r"\b(short|deep)\s+(left|middle|right)\b", re.I)
# target name: "...to J.Smith" possibly followed by more text; grab the X.Lastname token
TARGET = re.compile(r"\bto\s+([A-Z]\.[A-Za-z'\-]+)", re.I)


def parse_description(desc: str):
    if not isinstance(desc, str):
        return (None, None, None)
    depth = direction = None
    m = DEPTH_DIR.search(desc)
    if m:
        depth, direction = m.group(1).lower(), m.group(2).lower()
    tm = TARGET.search(desc)
    target = tm.group(1) if tm else None
    return depth, direction, target


def targets_table() -> pd.DataFrame:
    """One row per thrown play with parsed depth/direction/target + outcome."""
    p = loader.plays()
    p = p[p.passResult.isin(loader.KEEP_PASS_RESULTS)].copy()
    parsed = p.playDescription.apply(parse_description)
    p["pass_depth"] = parsed.str[0]
    p["pass_direction"] = parsed.str[1]
    p["target_name"] = parsed.str[2]
    p["completed"] = p.passResult == "C"
    return p[["gameId", "playId", "pff_passCoverage", "passResult", "completed",
              "pass_depth", "pass_direction", "target_name", "playResult"]]


def _name_key(display_name: str) -> str:
    """'Tom Brady' -> 'T.Brady' to match playDescription target tokens."""
    parts = str(display_name).split()
    if len(parts) < 2:
        return str(display_name)
    return f"{parts[0][0]}.{parts[-1]}"


def attach_targets(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'is_target' to each route row by matching the parsed target name to the
    player's display name (via nflId -> players.csv).
    """
    players = loader.players()[["nflId", "displayName"]].copy()
    players["name_key"] = players.displayName.apply(_name_key)
    key = dict(zip(players.nflId, players.name_key))

    tgt = targets_table()[["gameId", "playId", "target_name", "pass_depth",
                           "pass_direction", "completed"]]
    r = routes.merge(tgt, on=["gameId", "playId"], how="left")
    r["route_name_key"] = r.nflId.map(key)
    r["is_target"] = (r.route_name_key == r.target_name) & r.target_name.notna()
    return r
