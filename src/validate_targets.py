"""
TEST 1 - Target validation: does the ball go where the soft-spot map says space is?

1a. WITHIN-PLAY (robust to field-area/exposure confounds, since all receivers on a
    play share the same situation/geometry): on each play, compare the TARGETED
    receiver's separation (and zone softness, and area) to his non-targeted teammates'.

1b. COVERAGE-LEVEL: assign each route the soft-spot z-score of the zone it occupied
    (from that coverage's map). Are targeted receivers in softer zones than untargeted
    ones? Do completions land in softer zones than incompletions?

Headline metric = separation (the discrimination test showed controlled-area does not
separate coverages); area reported as a complementary check.

Run:  .venv/bin/python -m src.validate_targets
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from . import targets, surface as S

X_BINS, Y_BINS = S.X_BINS, S.Y_BINS


def _zone_index(depth, lateral):
    xi = np.clip(np.digitize(depth, X_BINS) - 1, 0, len(X_BINS) - 2)
    yi = np.clip(np.digitize(lateral, Y_BINS) - 1, 0, len(Y_BINS) - 2)
    return yi, xi   # surfaces indexed (lateral rows, depth cols)


def run():
    routes = pd.read_parquet("outputs/route_separations_throw.parquet")
    t = targets.attach_targets(routes)

    # per-coverage z-score soft-spot surface, then attach each route's zone softness
    zsurf = {cov: S.zscore_softspot_surface(routes, cov)[0]
             for cov in t.pff_passCoverage.unique()}
    zvals = np.full(len(t), np.nan)
    arr = t[["depth", "lateral", "pff_passCoverage"]].to_numpy(dtype=object)
    for i, (d, lat, cov) in enumerate(arr):
        z = zsurf.get(cov)
        if z is None:
            continue
        yi, xi = _zone_index(float(d), float(lat))
        zvals[i] = z[yi, xi]
    t["zone_softness"] = zvals

    print("=" * 70)
    print("TEST 1a  WITHIN-PLAY: targeted receiver vs his non-targeted teammates")
    print("=" * 70)
    rows = []
    for _, grp in t.groupby(["gameId", "playId"]):
        tg = grp[grp.is_target]
        nt = grp[~grp.is_target]
        if len(tg) != 1 or len(nt) == 0:
            continue
        rows.append({
            "sep_t": tg.separation.iloc[0], "sep_o": nt.separation.mean(),
            "soft_t": tg.zone_softness.iloc[0], "soft_o": nt.zone_softness.mean(),
            "area_t": tg.area_pos.iloc[0], "area_o": nt.area_pos.mean(),
            "target_most_open": bool(tg.separation.iloc[0] >= grp.separation.max() - 1e-9),
            "n": len(grp),
        })
    w = pd.DataFrame(rows)

    def report(name, dt, do):
        d = (w[dt] - w[do]).dropna()
        tstat, p = stats.ttest_1samp(d, 0)
        print(f"  {name:16s} target={w[dt].mean():7.2f}  teammates={w[do].mean():7.2f}  "
              f"diff={d.mean():+6.2f}  t={tstat:5.1f}  p={p:.1e}")

    print(f"plays analyzed: {len(w):,}\n")
    report("separation(yd)", "sep_t", "sep_o")
    report("zone softness", "soft_t", "soft_o")
    report("ctrl area(sqyd)", "area_t", "area_o")
    chance = 100.0 / w.n.mean()
    print(f"\n  target was the MOST-OPEN receiver on the play: {100*w.target_most_open.mean():.1f}% "
          f"(random chance ~{chance:.1f}%)")

    print()
    print("=" * 70)
    print("TEST 1b  COVERAGE-LEVEL: are targets / completions in softer zones?")
    print("=" * 70)
    tgt = t[t.is_target].dropna(subset=["zone_softness"])
    non = t[~t.is_target].dropna(subset=["zone_softness"])
    tt, p = stats.ttest_ind(tgt.zone_softness, non.zone_softness, equal_var=False)
    print(f"  zone softness:  targeted={tgt.zone_softness.mean():+.3f} (n={len(tgt):,})  "
          f"untargeted={non.zone_softness.mean():+.3f} (n={len(non):,})  t={tt:.1f}  p={p:.1e}")

    comp, inc = tgt[tgt.completed], tgt[~tgt.completed]
    tc, pc = stats.ttest_ind(comp.zone_softness, inc.zone_softness, equal_var=False)
    print(f"  among targets -> zone softness:  completed={comp.zone_softness.mean():+.3f} "
          f"(n={len(comp):,})  incomplete={inc.zone_softness.mean():+.3f} (n={len(inc):,})  "
          f"t={tc:.1f}  p={pc:.1e}")
    ts, ps = stats.ttest_ind(comp.separation, inc.separation, equal_var=False)
    print(f"  among targets -> separation:     completed={comp.separation.mean():.2f}yd  "
          f"incomplete={inc.separation.mean():.2f}yd  t={ts:.1f}  p={ps:.1e}")

    return t, w


if __name__ == "__main__":
    run()
