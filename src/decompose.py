"""
SVD / PCA decomposition of coverage separation surfaces  (the moat: unsupervised
linear algebra, no fitted model, structure emerges from the geometry).

We build a matrix M where each ROW is one coverage's separation surface flattened
to a vector over field zones (depth x lateral), de-meaned per zone so we study the
*shape* of how coverages deviate from the league baseline. Truncated SVD then asks:
do the leading singular vectors recover interpretable axes -- e.g. a man/zone axis
and a depth axis -- WITHOUT being told the labels?

scipy's svds is an iterative (Lanczos) solver; the matrix here is tiny
(7 coverages x ~360 zones), so it is effectively instant.

Run:  .venv/bin/python -m src.decompose
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from . import surface as S

COVERAGES = ["Cover-0", "Cover-1", "2-Man", "Cover-2", "Cover-3", "Cover-6", "Quarters"]
MAN = {"Cover-0", "Cover-1", "2-Man"}


def build_matrix(df: pd.DataFrame):
    """Rows = coverages, cols = zones. Values = per-zone separation minus league
    baseline (NaN zones -> 0 so they don't contribute). Returns M, covs, shape, edges."""
    base, _, xe, ye = S.baseline_surface(df)
    rows, covs = [], []
    for cov in COVERAGES:
        if cov not in df.pff_passCoverage.unique():
            continue
        cm, _, _, _ = S.surface(df[df.pff_passCoverage == cov])
        delta = cm - base
        rows.append(np.nan_to_num(delta, nan=0.0).ravel())
        covs.append(cov)
    M = np.vstack(rows)
    return M, covs, base.shape, (xe, ye)


def run():
    df = pd.read_parquet("outputs/route_separations_throw.parquet")
    M, covs, shape, edges = build_matrix(df)
    # center columns (per-zone mean across coverages) so SVD finds variation, not offset
    Mc = M - M.mean(axis=0, keepdims=True)

    k = min(3, min(Mc.shape) - 1)
    U, s, Vt = svds(Mc, k=k)
    order = np.argsort(s)[::-1]          # svds returns ascending
    U, s, Vt = U[:, order], s[order], Vt[order]

    total = (s ** 2).sum()
    print("=" * 64)
    print("SVD of coverage separation-surface matrix (unsupervised)")
    print("=" * 64)
    print(f"matrix: {M.shape[0]} coverages x {M.shape[1]} field zones\n")
    print("singular values & variance explained:")
    for i, sv in enumerate(s):
        print(f"  component {i+1}: sigma={sv:6.2f}   var={100*sv**2/total:5.1f}%")

    print("\ncoverage coordinates on the leading components (U * sigma):")
    coords = U * s
    hdr = "  coverage    " + "".join([f"  PC{i+1:>6}" for i in range(k)])
    print(hdr)
    for i, cov in enumerate(covs):
        tag = "MAN " if cov in MAN else "zone"
        line = f"  {cov:10s}[{tag}]" + "".join([f"  {coords[i, j]:+7.2f}" for j in range(k)])
        print(line)

    # Does PC1 separate man from zone? correlate PC1 coord with man(1)/zone(0)
    man_flag = np.array([1.0 if c in MAN else 0.0 for c in covs])
    for j in range(k):
        r = np.corrcoef(coords[:, j], man_flag)[0, 1]
        print(f"\n  corr(PC{j+1} coord, is-man) = {r:+.2f}")

    # Interpret PC loadings spatially: does a component load on depth?
    xe, ye = edges
    depth_centers = (xe[:-1] + xe[1:]) / 2
    for j in range(min(2, k)):
        load = Vt[j].reshape(shape)                   # (lateral, depth)
        depth_profile = np.nanmean(load, axis=0)      # average loading by depth
        # correlation of loading with depth = is this a 'depth' axis?
        rdepth = np.corrcoef(depth_profile, depth_centers)[0, 1]
        print(f"  PC{j+1}: spatial loading vs depth corr = {rdepth:+.2f} "
              f"({'DEPTH axis' if abs(rdepth)>0.5 else 'not depth-dominated'})")

    np.save("outputs/svd_components.npy", Vt.reshape((k,) + shape))
    print("\nsaved component maps -> outputs/svd_components.npy")
    return covs, coords, Vt.reshape((k,) + shape), edges


if __name__ == "__main__":
    run()
