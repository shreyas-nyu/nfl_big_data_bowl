"""
Generate the self-contained interactive web app:
  - one soft-spot field image per (coverage x skill) in the Sleeper colorway
  - a stats.json with per-view numbers (avg sep, top soft spot, who-attacks bars, the call)
  - index.html with working toggles (no server; GitHub Pages ready)

Run:  .venv/bin/python -m src.build_web
Outputs to  docs/  (GitHub Pages serves /docs on main).
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from . import surface as S
from .visualize import add_skill, COVERAGE_ORDER, SKILL_MAP

OUT = "docs"
ASSETS = f"{OUT}/assets"

# Sleeper-ish red -> (dark neutral) -> yellow -> green.
# Neutral sits DARK (near the field color) so only real signal shows color;
# mild yellow appears just off-center, bright green/red at the extremes.
RYG = LinearSegmentedColormap.from_list(
    "ryg", ["#ff4d4f", "#c23b32", "#5a2a20", "#15211b", "#4a4418",
            "#b89a1e", "#2ecc40", "#28e07b"], N=256)
BG = "#0e1a16"

SKILLS = ["All", "WR", "TE", "RB"]
POS_WINDOW = {  # human label for the top soft spot depth band
    "shallow": "0–6 yd", "intermediate": "7–15 yd", "deep": "16+ yd",
}


def _skill_mask(df, skill):
    if skill == "All":
        return df.position.map(SKILL_MAP).fillna("OTHER") != "OTHER"
    return df.position.map(SKILL_MAP).fillna("OTHER") == skill


def zscore_surface(df, coverage, skill):
    """z-score soft-spot surface for coverage x skill vs league-per-zone."""
    league_mean, _, xe, ye = S.baseline_surface(df)
    league_std, _, _, _ = S.zone_std_surface(df)
    sub = df[(df.pff_passCoverage == coverage) & _skill_mask(df, skill)]
    cov_mean, cnt, _, _ = S.surface(sub)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(league_std > 1e-6, (cov_mean - league_mean) / league_std, np.nan)
    return z, cnt, cov_mean, xe, ye


def render_field(z, xe, ye, path):
    """Dark field, continuous smoothed red-yellow-green surface, LOS marked."""
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    fig.patch.set_facecolor("#0c1120")
    ax.set_facecolor(BG)

    # smooth only where there is data; let low-sample zones fade toward the field
    has = np.isfinite(z)
    zf = gaussian_filter(np.where(has, z, 0.0), sigma=0.9)
    # per-cell opacity: full where data exists, transparent where it doesn't
    alpha = gaussian_filter(has.astype(float), sigma=1.1)
    alpha = np.clip(alpha, 0, 1) * 0.95

    lim = max(0.6, np.nanpercentile(np.abs(z[has]), 95)) if has.any() else 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    # build RGBA image so empty grass shows the field, not bright neutral
    rgba = RYG(norm(zf))
    rgba[..., 3] = alpha
    ax.imshow(rgba, origin="lower", aspect="auto",
              extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="bilinear")

    # field lines
    for d in range(0, int(xe[-1]) + 1, 5):
        ax.axvline(d, color="#1f2b26", lw=0.8)
    ax.axvline(0, color="#3a9bff", lw=2.2)                 # LOS
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel("yards downfield of LOS", color="#6f847a", fontsize=9)
    ax.set_ylabel("lateral from ball (yd)", color="#6f847a", fontsize=9)
    ax.tick_params(colors="#3f4858", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a3a33")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#0c1120")
    plt.close(fig)


def top_softspot(z, xe, ye):
    """
    Strongest soft-spot REGION (not a single noisy cell). We smooth the surface,
    restrict to the meaningful route area (>=3 yd downfield of the LOS, so screens/
    checkdowns behind the line don't dominate), and take the peak of the smoothed
    field. Returns (depth, lateral, value).
    """
    has = np.isfinite(z)
    if not has.any():
        return None
    # fill gaps with 0 (neutral) so smoothing doesn't propagate NaN, but only
    # report peaks where there was actually data nearby
    zs = gaussian_filter(np.where(has, z, 0.0), sigma=1.2)
    support = gaussian_filter(has.astype(float), sigma=1.2)
    zs = np.where(support > 0.25, zs, -np.inf)   # ignore data-poor cells
    depth_c = (xe[:-1] + xe[1:]) / 2
    lat_c = (ye[:-1] + ye[1:]) / 2
    mask = depth_c >= 3.0                          # downfield only
    region = zs[:, mask]
    if not np.isfinite(region).any():
        return None
    yi, xi_m = np.unravel_index(np.argmax(region), region.shape)
    xi = np.where(mask)[0][xi_m]
    return float(depth_c[xi]), float(lat_c[yi]), float(zs[yi, xi])


def make_call(coverage, skill, spot, avg):
    if spot is None:
        return "Not enough routes in this split to call it."
    depth, lat, val = spot
    band = "short" if depth < 7 else ("intermediate" if depth < 16 else "deep")
    side = "middle" if abs(lat) < 6 else ("left" if lat > 0 else "right")
    who = skill if skill != "All" else "a receiver"
    return (f"{coverage} is softest {band} {side} (~{depth:.0f} yd, {val:+.1f}σ). "
            f"Put {who} in that window.")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    df = pd.read_parquet("outputs/route_separations_throw.parquet")
    df = add_skill(df)

    covs = [c for c in COVERAGE_ORDER if c in df.pff_passCoverage.unique()]
    data = {"coverages": covs, "skills": SKILLS, "views": {}}

    # who-attacks bars per coverage (avg sep by skill)
    for cov in covs:
        for sk in SKILLS:
            z, cnt, cov_mean, xe, ye = zscore_surface(df, cov, sk)
            key = f"{cov}__{sk}"
            img = f"assets/{key.replace('-','').replace(' ','')}.png"
            render_field(z, xe, ye, f"{OUT}/{img}")
            spot = top_softspot(z, xe, ye)
            sub = df[(df.pff_passCoverage == cov) & _skill_mask(df, sk)]
            data["views"][key] = {
                "img": img,
                "n": int(len(sub)),
                "avg_sep": round(float(sub.separation.mean()), 2) if len(sub) else None,
                "call": make_call(cov, sk, spot, sub.separation.mean() if len(sub) else 0),
                "top_depth": round(spot[0], 1) if spot else None,
                "top_val": round(spot[2], 2) if spot else None,
            }
        # skill bars for this coverage (avg sep)
        bars = {}
        for sk in ["TE", "WR", "RB"]:
            s = df[(df.pff_passCoverage == cov) & _skill_mask(df, sk)].separation
            bars[sk] = round(float(s.mean()), 2) if len(s) else 0.0
        data["views"][f"{cov}__bars"] = bars

    # global validation numbers (from Test 1)
    data["validation"] = {
        "complete_sep": 4.33, "incomplete_sep": 2.79, "sep_p": "4e-145",
        "target_vs_team": 0.62, "target_p": "1e-88",
        "svd_var": 75, "svd_corr": -0.80,
        "n_routes": int(len(df)), "n_games": int(df.gameId.nunique()),
    }

    with open(f"{OUT}/data.json", "w") as f:
        json.dump(data, f, indent=0)
    print(f"generated {len(covs)*len(SKILLS)} field images + data.json in {OUT}/")
    return data


if __name__ == "__main__":
    main()
