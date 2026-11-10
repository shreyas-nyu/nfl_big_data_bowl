"""
Scale the route-separation extraction across all 122 games and cache to parquet.
Run:  .venv/bin/python -m src.build_dataset
"""
from __future__ import annotations
import glob
import os
import time
import pandas as pd

from . import loader

OUT = "outputs"


def main(measure_at: str = "throw"):
    os.makedirs(OUT, exist_ok=True)
    game_files = sorted(glob.glob("data/tracking/tracking_*.csv"))
    game_ids = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in game_files]

    frames = []
    t0 = time.time()
    for i, gid in enumerate(game_ids, 1):
        df = loader.route_separations_for_game(gid, measure_at=measure_at)
        if not df.empty:
            frames.append(df)
        if i % 20 == 0 or i == len(game_ids):
            print(f"  [{i:3d}/{len(game_ids)}] game {gid}  "
                  f"rows so far={sum(len(f) for f in frames):6d}  "
                  f"elapsed={time.time()-t0:5.1f}s")

    full = pd.concat(frames, ignore_index=True)
    path = f"{OUT}/route_separations_{measure_at}.parquet"
    full.to_parquet(path, index=False)
    print(f"\nsaved {path}")
    print(f"total routes: {len(full):,} across {full.playId.nunique():,} plays "
          f"in {full.gameId.nunique()} games")
    print("\nseparation by coverage:")
    print(full.groupby('pff_passCoverage').separation
          .agg(['count', 'mean', 'median']).round(2).to_string())
    return full


if __name__ == "__main__":
    main()
