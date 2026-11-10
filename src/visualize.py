"""
Dark-mode visualizations for the Coverage Soft-Spot pipeline.

Two figures:
  1. Per-coverage separation surfaces (absolute yards of separation per zone).
  2. Per-coverage SOFT-SPOT maps (green = concedes MORE separation than the
     league baseline for that zone; red = denies space), the headline deliverable.

All maps are in LOS-relative space: x = yards downfield of the line of scrimmage,
y = yards laterally from the ball. Field drawn from the offense's perspective
(attacking to the right).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import surface as S

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#9aa4af",
    "ytick.color": "#9aa4af",
    "axes.edgecolor": "#30363d",
    "font.size": 10,
})

# red -> dark -> green diverging map for soft spots
SOFTSPOT_CMAP = LinearSegmentedColormap.from_list(
    "softspot", ["#ff3b30", "#7a1f1a", "#0d1117", "#1f5a2a", "#2ecc40"], N=256)
SEP_CMAP = LinearSegmentedColormap.from_list(
    "sep", ["#0d1117", "#15502a", "#2ecc40", "#b6ff9e"], N=256)

COVERAGE_ORDER = ["Cover-0", "Cover-1", "2-Man", "Cover-2", "Cover-3", "Cover-6", "Quarters"]

SKILL_MAP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "HB": "RB"}


def add_skill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["skill"] = df.position.map(SKILL_MAP).fillna("OTHER")
    return df


def _draw_field(ax, xe, ye):
    # LOS line and yard-depth gridlines
    ax.axvline(0, color="#e6edf3", ls="--", lw=1.2, alpha=0.8)
    for d in range(5, int(xe[-1]) + 1, 5):
        ax.axvline(d, color="#30363d", lw=0.5, alpha=0.6)
    ax.axhline(0, color="#30363d", lw=0.6, alpha=0.6)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])


def plot_separation_surfaces(df: pd.DataFrame, out="outputs/separation_surfaces.png"):
    covs = [c for c in COVERAGE_ORDER if c in df.pff_passCoverage.unique()]
    n = len(covs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)

    # shared color scale
    all_means = []
    surf = {}
    for c in covs:
        m, cnt, xe, ye = S.surface(df[df.pff_passCoverage == c])
        surf[c] = (m, cnt, xe, ye)
        all_means.append(m[np.isfinite(m)])
    vmax = np.nanpercentile(np.concatenate(all_means), 95)

    for i, c in enumerate(covs):
        ax = axes[i // ncol][i % ncol]
        m, cnt, xe, ye = surf[c]
        im = ax.pcolormesh(xe, ye, m, cmap=SEP_CMAP, vmin=0, vmax=vmax, shading="flat")
        _draw_field(ax, xe, ye)
        ax.set_title(f"{c}   (n={int(np.nansum(cnt)):,} routes)", color="#e6edf3", fontsize=11)
        if i % ncol == 0:
            ax.set_ylabel("lateral from ball (yd)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("yards downfield of LOS")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("avg receiver separation at throw (yd)", color="#e6edf3")
    fig.suptitle("Separation surface by defensive coverage  (LOS-relative, offense attacks right)",
                 color="#e6edf3", fontsize=14, y=0.98)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_softspot_maps(df: pd.DataFrame, out="outputs/softspot_maps.png"):
    """
    PRIMARY deliverable: coverage soft-spot maps.

    delta(zone) = coverage_separation(zone) - LEAGUE_baseline_separation(zone).

    Subtracting the per-zone league baseline controls for the universal depth
    gradient (receivers are naturally more open near the LOS than 10 yd downfield),
    so what remains is coverage-SPECIFIC: green = this coverage concedes more
    separation here than other coverages do in the same zone; red = it denies it.
    A shared symmetric color scale lets panels be compared directly; the panel
    title reports the coverage's overall average separation (its tightness level).
    """
    covs = [c for c in COVERAGE_ORDER if c in df.pff_passCoverage.unique()]
    n = len(covs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)

    deltas = {}
    for c in covs:
        delta, cnt, xe, ye = S.zscore_softspot_surface(df, c)
        deltas[c] = (delta, cnt, xe, ye)
    allv = np.concatenate([d[0][np.isfinite(d[0])] for d in deltas.values()])
    lim = np.nanpercentile(np.abs(allv), 95)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    for i, c in enumerate(covs):
        ax = axes[i // ncol][i % ncol]
        delta, cnt, xe, ye = deltas[c]
        im = ax.pcolormesh(xe, ye, delta, cmap=SOFTSPOT_CMAP, norm=norm, shading="flat")
        _draw_field(ax, xe, ye)
        own = np.nanmean(S.surface(df[df.pff_passCoverage == c])[0])
        ax.set_title(f"{c}   (n={int(np.nansum(cnt)):,}, avg sep {own:.1f} yd)",
                     color="#e6edf3", fontsize=11)
        if i % ncol == 0:
            ax.set_ylabel("lateral from ball (yd)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("yards downfield of LOS")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("separation z-score vs. per-zone league distribution\ngreen = SOFT spot, red = denies space",
                   color="#e6edf3")
    fig.suptitle("Coverage soft-spot maps  —  where each coverage concedes space\n"
                 "(z-score vs. per-zone league; depth gradient & man/zone scale removed)",
                 color="#e6edf3", fontsize=14, y=1.0)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_skill_softspots(df: pd.DataFrame, coverage: str = "Cover-3",
                         out=None):
    """
    SKILL-POSITION cut: for one coverage, the soft-spot z-score map split by
    WR / TE / RB. Shows that different skill positions find space in different
    zones against the same coverage (the novel competition-focused insight).
    """
    out = out or f"outputs/skill_softspots_{coverage.replace('-','').lower()}.png"
    df = add_skill(df)
    skills = ["WR", "TE", "RB"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), squeeze=False)

    # z-score each skill's surface vs the league distribution per zone (all positions)
    league_mean, _, xe, ye = S.baseline_surface(df)
    league_std, _, _, _ = S.zone_std_surface(df)

    for i, sk in enumerate(skills):
        ax = axes[0][i]
        sub = df[(df.pff_passCoverage == coverage) & (df.skill == sk)]
        cov_mean, cnt, _, _ = S.surface(sub)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(league_std > 1e-6, (cov_mean - league_mean) / league_std, np.nan)
        finite = z[np.isfinite(z)]
        lim = np.nanpercentile(np.abs(finite), 92) if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-max(lim, 0.3), vcenter=0, vmax=max(lim, 0.3))
        im = ax.pcolormesh(xe, ye, z, cmap=SOFTSPOT_CMAP, norm=norm, shading="flat")
        _draw_field(ax, xe, ye)
        ax.set_title(f"{sk}  (n={int(np.nansum(cnt)):,}, avg {np.nanmean(cov_mean):.1f} yd)",
                     color="#e6edf3")
        ax.set_xlabel("yards downfield of LOS")
        if i == 0:
            ax.set_ylabel("lateral from ball (yd)")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("separation z-score vs league\ngreen = SOFT", color="#e6edf3")
    fig.suptitle(f"Where each skill position finds space vs {coverage}",
                 color="#e6edf3", fontsize=14, y=1.03)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    df = pd.read_parquet("outputs/route_separations_throw.parquet")
    print(plot_separation_surfaces(df))
    print(plot_softspot_maps(df))
    print(plot_skill_softspots(df, "Cover-3"))
    print(plot_skill_softspots(df, "Cover-2"))
