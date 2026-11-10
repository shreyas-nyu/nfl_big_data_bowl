# %% [markdown]
# # Soft Spots: Where Coverages Concede Space to Skill-Position Receivers
#
# A **geometric, scheme-level** map of where each NFL defensive coverage gives up
# separation to WR / TE / RB — built purely from player positions and linear algebra,
# **not** a trained model.
#
# Data: NFL Big Data Bowl tracking (2021 season, Weeks 1-8, 122 games), joined to
# PFF coverage labels. Pipeline modules live in `src/`.

# %%
import pandas as pd
import numpy as np
from src import loader, surface as S, visualize as V, targets, decompose as D, validate_targets

pd.set_option("display.width", 120)

# %% [markdown]
# ## 1. The data: route-level separation
# One row per route-runner per pass play, measured at the throw (`pass_forward`,
# present in ~90% of plays; snap+2.5s fallback otherwise). Coordinates are
# normalized so every play goes left->right and expressed relative to the LOS.
# Build once with `python -m src.build_dataset`; here we load the cache.

# %%
df = pd.read_parquet("outputs/route_separations_throw.parquet")
print(f"{len(df):,} routes | {df.playId.nunique():,} plays | {df.gameId.nunique()} games")
df.groupby("pff_passCoverage").separation.agg(["count", "mean", "median"]).round(2)

# %% [markdown]
# **Sanity:** man coverages (Cover-0/1, 2-Man) yield the least separation (~2 yd),
# zone coverages the most (Cover-2 ~4.2 yd) — matching known football truth, with
# no fitting involved. League mean ~3 yd aligns with public NGS separation values.

# %% [markdown]
# ## 2. The method (the moat): geometry, not a model
# Each player is a kinematic vector `[x, y, vx, vy]` derived from position + speed +
# direction. From positions we compute, per route at the throw:
# - **separation** = distance to nearest defender (the headline metric)
# - **controlled area** = field-bounded Voronoi cell the receiver owns
#
# Everything is a deterministic linear-algebra operation. No parameters are fit to
# any outcome.

# %%
# separation vs controlled-area: a discrimination check
disc = {}
for col in ["separation", "area_pos"]:
    m = df.groupby("pff_passCoverage")[col].mean()
    disc[col] = round(m.std() / m.mean(), 3)
print("coverage-discrimination (CV of per-coverage means):", disc)
print("-> separation separates coverages ~9x more than controlled area; "
      "area is conserved across schemes, so SEPARATION is the headline metric.")

# %% [markdown]
# ## 3. Headline: coverage soft-spot maps
# For each coverage, the per-zone separation is standardized against the league
# distribution **in that same zone** (z-score). This removes (a) the universal
# depth gradient and (b) the man/zone scale gap, leaving a coverage-specific map:
# green = concedes more separation than typical here, red = denies it.

# %%
V.plot_softspot_maps(df)          # -> outputs/softspot_maps.png
V.plot_separation_surfaces(df)    # -> outputs/separation_surfaces.png
print("saved soft-spot + raw separation maps")

# %% [markdown]
# ## 4. Skill-position cut (the competition focus)
# The same coverage concedes space in **different zones** depending on who runs the
# route: WRs stress the intermediate/deep middle, TEs the seams, RBs the flats.

# %%
V.plot_skill_softspots(df, "Cover-3")
V.plot_skill_softspots(df, "Cover-2")
dfx = V.add_skill(df)
dfx[dfx.skill != "OTHER"].pivot_table(index="skill", columns="pff_passCoverage",
                                      values="separation", aggfunc="mean").round(2)

# %% [markdown]
# ## 5. Validation (Test 1): does the ball go where the map says space is?

# %%
validate_targets.run()

# %% [markdown]
# **Result.** Within a play, the targeted receiver has +0.62 yd more separation than
# his teammates (p~1e-88) and controls +18 sq yd more area. Among targets,
# completions land in significantly softer zones than incompletions (p~4e-30) and at
# +1.5 yd more separation (p~4e-145). The map predicts throw **success**, not intent.

# %% [markdown]
# ## 6. The moat, demonstrated: SVD recovers scheme structure unsupervised
# Stack each coverage's separation surface as a row; decompose with truncated SVD.
# We never pass the man/zone labels.

# %%
from src import plot_svd
plot_svd.plot()   # prints components + saves outputs/svd_overview.png

# %% [markdown]
# **Result.** PC1 explains 75% of variance and correlates **-0.80** with man-vs-zone
# — the single largest mode of how coverages differ in conceding space is the
# man/zone distinction, recovered without labels. PC2 (17%) separates zone coverages
# by *where* they are soft (Cover-2's deep-sideline holes vs Cover-3's underneath).
# Structure emerges from the geometry, not from a fitted model.
