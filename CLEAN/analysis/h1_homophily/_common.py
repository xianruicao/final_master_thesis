"""Shared helpers for h1_homophily analysis scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).parent
ROOT = HERE.parents[1]                                   # CLEAN
DATA_PATH = ROOT / "data" / "04_features" / "film_features_all.csv"
FIG_DIR = HERE / "figures"
TBL_DIR = HERE / "tables"
# NB: these dirs are created lazily by their writers (savefig / phase3 main()),
# not at import time — so merely importing this module leaves no empty folders.

# Warm/cool, colour-blind friendly, avoids the pink/blue gender clichés.
GENDER_PALETTE = {"F": "#D97706", "M": "#0EA5E9"}        # amber / sky
GENDER_ORDER = ["F", "M"]
RNG_SEED = 20260622

DISPLAY_TITLES = {
    "mulan_1998":                       "Mulan (1998)",
    "frozen_2013":                      "Frozen (2013)",
    "inside_out_2015":                  "Inside Out (2015)",
    "zootopia_2016":                    "Zootopia (2016)",
    "encanto_2021":                     "Encanto (2021)",
    "raya_and_the_last_dragon_2021":    "Raya (2021)",
    "toy_story_1995":                   "Toy Story (1995)",
    "monsters_inc_2001":                "Monsters Inc (2001)",
    "up":                               "Up (2009)",
    "coco_2017":                        "Coco (2017)",
    "onward_2020":                      "Onward (2020)",
    "soul_2020":                        "Soul (2020)",
    "beautyandthebeast_1991":           "Beauty & the Beast (1991)",
    "incredibles_2_2018":               "Incredibles 2 (2018)",
    "findingnemo":                      "Finding Nemo (2003)",
    "toy_story_3_2010":                 "Toy Story 3 (2010)",
    "cars2":                            "Cars 2 (2011)",
    "elemental_2023":                   "Elemental (2023)",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
    })


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["film_title"] = df["film_id"].map(DISPLAY_TITLES).fillna(df["film_id"])
    df["row_label"] = df.apply(
        lambda r: f"{r.film_title} — {r.protagonist}"
        if r.film_id == "monsters_inc_2001"
        else r.film_title,
        axis=1,
    )
    return df


def dedup_films(df: pd.DataFrame) -> pd.DataFrame:
    """Film-level dedup (Monsters Inc has 2 rows with identical global metrics)."""
    return df.drop_duplicates("film_id").reset_index(drop=True)


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


# ---------- statistics --------------------------------------------------------

def mannwhitney(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> dict:
    res = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
    return {"U": float(res.statistic), "p": float(res.pvalue)}


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation r = 1 - 2U/(n1*n2). U from MW two-sided."""
    U = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").statistic
    return float(1 - 2 * U / (len(x) * len(y)))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """δ = (#x>y − #x<y) / (n_x * n_y). Range [−1, 1]."""
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def perm_test_diff(
    x: np.ndarray, y: np.ndarray,
    stat=np.median, n_iter: int = 10_000, seed: int = RNG_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    obs = stat(x) - stat(y)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        rng.shuffle(pooled)
        diffs[i] = stat(pooled[:n_x]) - stat(pooled[n_x:])
    p_two = float((np.abs(diffs) >= abs(obs)).mean())
    p_one_less = float((diffs <= obs).mean())             # H1: x < y
    return {
        "obs_diff": float(obs),
        "p_two_sided": p_two,
        "p_one_sided_x_less_y": p_one_less,
        "null_mean": float(diffs.mean()),
        "null_std": float(diffs.std(ddof=1)),
    }


def bootstrap_diff_ci(
    x: np.ndarray, y: np.ndarray,
    stat=np.median, n_iter: int = 10_000, seed: int = RNG_SEED, alpha: float = 0.05,
) -> dict:
    rng = np.random.default_rng(seed)
    n_x, n_y = len(x), len(y)
    out = np.empty(n_iter)
    for i in range(n_iter):
        out[i] = stat(rng.choice(x, n_x, replace=True)) - stat(rng.choice(y, n_y, replace=True))
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {
        "diff_point": float(stat(x) - stat(y)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_mean": float(out.mean()),
    }


def power_note(n_f: int, n_m: int, effect: str = "medium") -> str:
    """Quick informal power note for MW with small N. Not exact."""
    return (
        f"With n_F={n_f} and n_M={n_m}, Mann-Whitney U has very limited power. "
        f"Minimum two-sided p achievable is {1/(2 ** (n_f + n_m)):.2g} only when the "
        f"groups are perfectly separated; detection of a 'medium' effect "
        f"(Cliff's δ≈0.33) is below ~30% at α=0.05."
    )
