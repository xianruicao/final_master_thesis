"""
Cross-film batch method validation for Phase 3.

Pipeline role:
  1. Notebook 06 (`CLEAN/notebooks/06_network_analysis_PAU.ipynb`) is the
     per-film production layer. It builds each film's directed addressee
     network and writes `data/processed/{film_id}/network_edges.csv`.
  2. This script is the cross-film batch layer. It consumes Notebook 06 outputs
     and rebuilds a scene co-occurrence baseline for method comparison.
  3. Notebook 09 (`CLEAN/notebooks/09_analysis.ipynb`) is the reporting layer.
     It reads cross-film outputs, draws thesis figures, and interprets results.

For each film, this script:
  1. Loads the addressee network from Notebook 06's `network_edges.csv`.
  2. Builds a scene co-occurrence network weighted by scenes where both
     characters spoke. Both networks use MIN_EDGE_COUNT=3 for parity.
  3. Computes protagonist same-gender share and a label-shuffling z-score.
  4. Compares addressee vs co-occurrence across films.

Compatibility:
  Output table/figure names still use the historic `phase3v2_*` prefix so
  downstream notebooks and generated markdown do not need data-path changes.

Note:
  The addressee z-score here is not expected to exactly equal
  `protag_samesex_z` in `film_features_all.csv`. That column uses a
  degree-preserving rewiring null on binary ties; this script uses label
  shuffling to isolate method differences under a shared null.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    FIG_DIR, GENDER_ORDER, GENDER_PALETTE, RNG_SEED, TBL_DIR,
    cliffs_delta, load_df, mannwhitney, rank_biserial, savefig, set_style,
)

CLEAN = Path(__file__).resolve().parents[2]
PROC  = CLEAN / "data" / "processed"
MIN_EDGE_COUNT = 3
N_NULL = 2000


# ---------- per-film data loading ---------------------------------------------

def film_dir_for(film_id: str) -> Path:
    """film_features uses 'up' for Up (2009) — match the folder name."""
    candidates = [film_id, film_id.replace("-", "_")]
    for c in candidates:
        p = PROC / c
        if p.exists():
            return p
    raise FileNotFoundError(f"No processed folder for {film_id}")


def load_character_meta(film_id: str) -> pd.DataFrame:
    """Return dataframe keyed on character_id with gender + flags + merge-into."""
    fdir = film_dir_for(film_id)
    cr = pd.read_csv(fdir / f"character_review_{film_id}.csv")
    cr = cr.rename(columns={"review_character_id": "character_id"})
    keep = ["character_id", "canonical_name", "gender",
            "is_protagonist", "included_in_network", "merge_into_character_id"]
    return cr[keep].copy()


def build_id_remap(meta: pd.DataFrame) -> dict[str, str]:
    """If a character has merge_into_character_id, redirect to the target."""
    out = {}
    for _, r in meta.iterrows():
        if pd.notna(r.merge_into_character_id) and r.merge_into_character_id:
            out[r.character_id] = r.merge_into_character_id
        else:
            out[r.character_id] = r.character_id
    return out


def gender_map(meta: pd.DataFrame) -> dict[str, str]:
    """character_id -> 'F'/'M'. Skip 'Both' / non-network rows."""
    out = {}
    for _, r in meta.iterrows():
        if r.included_in_network and r.gender in ("F", "M"):
            out[r.character_id] = r.gender
    return out


def protag_id(meta: pd.DataFrame, protagonist_name: str) -> str | None:
    """Find the character_id for the named protagonist."""
    hit = meta[(meta.is_protagonist) & (meta.canonical_name.str.lower() == protagonist_name.lower())]
    if len(hit) >= 1:
        return hit.iloc[0].character_id
    # fallback: substring match
    hit = meta[meta.canonical_name.str.lower() == protagonist_name.lower()]
    if len(hit) >= 1:
        return hit.iloc[0].character_id
    return None


# ---------- cross-film method-comparison network helpers -----------------------

def build_addressee_network(film_id: str, remap: dict, valid: set[str]) -> dict:
    """Load Notebook 06 addressee edges as undirected weighted comparison edges."""
    fdir = film_dir_for(film_id)
    e = pd.read_csv(fdir / "network_edges.csv")
    edges: dict[frozenset, int] = {}
    for _, r in e.iterrows():
        a = remap.get(r.speaker_clean, r.speaker_clean)
        b = remap.get(r.addressee_clean, r.addressee_clean)
        if a == b or a not in valid or b not in valid:
            continue
        key = frozenset({a, b})
        edges[key] = edges.get(key, 0) + int(r.utterance_count)
    # apply MIN_EDGE_COUNT
    edges = {k: v for k, v in edges.items() if v >= MIN_EDGE_COUNT}
    return edges


def build_cooccurrence_network(film_id: str, remap: dict, valid: set[str]) -> dict:
    """Build the scene co-occurrence baseline used only for cross-film comparison."""
    fdir = film_dir_for(film_id)
    u = pd.read_csv(fdir / "utterances_with_addressee_scene.csv",
                    usecols=["scene_id", "character_id"])
    u = u.dropna(subset=["scene_id", "character_id"])
    u["character_id"] = u["character_id"].map(lambda x: remap.get(x, x))
    u = u[u.character_id.isin(valid)]
    # per scene: set of distinct characters who spoke
    scene_chars = u.groupby("scene_id")["character_id"].apply(lambda s: sorted(set(s)))
    edges: dict[frozenset, int] = {}
    for chars in scene_chars:
        if len(chars) < 2:
            continue
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                key = frozenset({chars[i], chars[j]})
                edges[key] = edges.get(key, 0) + 1
    edges = {k: v for k, v in edges.items() if v >= MIN_EDGE_COUNT}
    return edges


# Clearer aliases for new code. The original names stay public because the
# orchestrator, wrappers, and archived notebooks may import them.
load_addressee_edges_from_notebook06 = build_addressee_network
build_scene_cooccurrence_edges_for_comparison = build_cooccurrence_network


# ---------- metrics ----------------------------------------------------------

def neighbours_of(edges: dict, node: str) -> dict[str, int]:
    """Return {neighbour: weight}."""
    out = {}
    for key, w in edges.items():
        if node in key:
            other = next(iter(key - {node}))
            out[other] = w
    return out


def protag_samesex_share(edges: dict, protag: str, gender: dict) -> float | None:
    """Weighted share of protagonist's edge weight going to same-gender alters."""
    if protag not in gender:
        return None
    nbrs = neighbours_of(edges, protag)
    if not nbrs:
        return None
    g = gender[protag]
    total = sum(nbrs.values())
    if total == 0:
        return None
    same = sum(w for n, w in nbrs.items() if gender.get(n) == g)
    return same / total


def label_shuffle_null(edges: dict, protag: str, gender: dict,
                       n_iter: int = N_NULL, seed: int = RNG_SEED) -> dict:
    """Shuffle gender labels among non-protagonist nodes; recompute share."""
    rng = np.random.default_rng(seed)
    nodes_in_edges = set().union(*edges.keys()) if edges else set()
    nodes = [n for n in nodes_in_edges if n in gender]
    if protag not in gender or protag not in nodes_in_edges:
        return {"obs": None, "null_mean": None, "null_std": None,
                "z": None, "p_one_sided": None, "p_two_sided": None,
                "n_same": None, "n_other": None}
    obs = protag_samesex_share(edges, protag, gender)
    if obs is None:
        return {"obs": None, "null_mean": None, "null_std": None,
                "z": None, "p_one_sided": None, "p_two_sided": None,
                "n_same": None, "n_other": None}
    non_protag = [n for n in nodes if n != protag]
    labels = np.array([gender[n] for n in non_protag])
    null = np.empty(n_iter)
    # precompute neighbours and weights for protagonist
    nbrs = neighbours_of(edges, protag)
    nbr_nodes = list(nbrs.keys())
    nbr_w = np.array([nbrs[n] for n in nbr_nodes], dtype=float)
    total = nbr_w.sum()
    g_lead = gender[protag]
    node_to_idx = {n: i for i, n in enumerate(non_protag)}
    nbr_idx = np.array([node_to_idx[n] for n in nbr_nodes if n in node_to_idx])
    nbr_w_in = np.array([nbrs[non_protag[i]] for i in nbr_idx], dtype=float)
    for it in range(n_iter):
        perm = rng.permutation(labels)
        same_mask = perm[nbr_idx] == g_lead
        null[it] = (nbr_w_in * same_mask).sum() / total
    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 0 else 0.0
    # one-sided "higher than null" p
    p_one = float((null >= obs).mean())
    # two-sided
    diff = abs(obs - null.mean())
    p_two = float(((null - null.mean()).__abs__() >= diff).mean())
    n_same = int(sum(1 for n in nbr_nodes if gender.get(n) == g_lead))
    n_other = len(nbr_nodes) - n_same
    return {"obs": float(obs), "null_mean": float(null.mean()), "null_std": float(sd),
            "z": float(z), "p_one_sided": p_one, "p_two_sided": p_two,
            "n_same": n_same, "n_other": n_other}


def network_summary(edges: dict, valid_nodes: set[str]) -> dict:
    if not edges:
        return {"n_nodes": 0, "n_edges": 0, "density": np.nan,
                "mean_clustering": np.nan, "total_weight": 0}
    nodes_in = set().union(*edges.keys())
    n_nodes = len(nodes_in)
    n_edges = len(edges)
    density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes >= 2 else np.nan
    # mean clustering via adjacency
    adj = {n: set() for n in nodes_in}
    for k in edges:
        a, b = list(k)
        adj[a].add(b); adj[b].add(a)
    clust = []
    for n, nbrs in adj.items():
        if len(nbrs) < 2:
            clust.append(0.0); continue
        possible = len(nbrs) * (len(nbrs) - 1) / 2
        actual = sum(1 for a in nbrs for b in nbrs if a < b and b in adj.get(a, set()))
        clust.append(actual / possible)
    return {"n_nodes": int(n_nodes), "n_edges": int(n_edges),
            "density": float(density),
            "mean_clustering": float(np.mean(clust)),
            "total_weight": int(sum(edges.values()))}


# ---------- main analysis ----------------------------------------------------

def banner(m): print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)


def per_film_row(film_id: str, protagonist: str, lead_gender: str) -> dict:
    meta = load_character_meta(film_id)
    remap = build_id_remap(meta)
    valid = set(gender_map(meta).keys())   # only F/M with included_in_network
    gender = gender_map(meta)
    protag = protag_id(meta, protagonist)

    addr_edges = build_addressee_network(film_id, remap, valid)
    cooc_edges = build_cooccurrence_network(film_id, remap, valid)

    addr_sum = network_summary(addr_edges, valid)
    cooc_sum = network_summary(cooc_edges, valid)

    addr_null = label_shuffle_null(addr_edges, protag, gender)
    cooc_null = label_shuffle_null(cooc_edges, protag, gender)

    return {
        "film_id": film_id,
        "protagonist": protagonist,
        "lead_gender": lead_gender,
        "protag_id": protag,
        # addressee
        "addr_n_nodes": addr_sum["n_nodes"], "addr_n_edges": addr_sum["n_edges"],
        "addr_density": addr_sum["density"], "addr_clustering": addr_sum["mean_clustering"],
        "addr_total_weight": addr_sum["total_weight"],
        "addr_samesex":     addr_null["obs"],
        "addr_samesex_z":   addr_null["z"],
        "addr_samesex_p":   addr_null["p_one_sided"],
        "addr_n_same":      addr_null["n_same"],
        "addr_n_other":     addr_null["n_other"],
        # co-occurrence
        "cooc_n_nodes": cooc_sum["n_nodes"], "cooc_n_edges": cooc_sum["n_edges"],
        "cooc_density": cooc_sum["density"], "cooc_clustering": cooc_sum["mean_clustering"],
        "cooc_total_weight": cooc_sum["total_weight"],
        "cooc_samesex":     cooc_null["obs"],
        "cooc_samesex_z":   cooc_null["z"],
        "cooc_samesex_p":   cooc_null["p_one_sided"],
        "cooc_n_same":      cooc_null["n_same"],
        "cooc_n_other":     cooc_null["n_other"],
    }


def main():
    set_style()
    df = load_df()

    banner("Building per-film comparison rows (addressee vs scene co-occurrence)")
    rows = []
    for _, r in df.iterrows():
        print(f"  {r.film_id:35s}  {r.protagonist:8s}  {r.lead_gender}", flush=True)
        rows.append(per_film_row(r.film_id, r.protagonist, r.lead_gender))
    cmp = pd.DataFrame(rows)
    cmp["film_title"] = cmp["film_id"].map(df.set_index("film_id")["film_title"].drop_duplicates())
    cmp["row_label"] = cmp["film_title"].fillna(cmp["film_id"])
    cmp.loc[cmp.film_id == "monsters_inc_2001", "row_label"] = \
        cmp.loc[cmp.film_id == "monsters_inc_2001"].apply(
            lambda r: f"{r.film_title} — {r.protagonist}", axis=1)

    banner("Per-film comparison table")
    show_cols = ["film_id", "protagonist", "lead_gender",
                 "addr_n_nodes", "addr_n_edges", "addr_samesex", "addr_samesex_z", "addr_samesex_p",
                 "cooc_n_nodes", "cooc_n_edges", "cooc_samesex", "cooc_samesex_z", "cooc_samesex_p"]
    print(cmp[show_cols].round(3).to_string(index=False))
    cmp.to_csv(TBL_DIR / "phase3v2_method_comparison.csv", index=False)

    # ---- 3.2 paired comparison -----------------------------------------------
    banner("3.2 — Paired comparison: addr_samesex_z vs cooc_samesex_z")
    z_addr = cmp["addr_samesex_z"].to_numpy()
    z_cooc = cmp["cooc_samesex_z"].to_numpy()
    raw_addr = cmp["addr_samesex"].to_numpy()
    raw_cooc = cmp["cooc_samesex"].to_numpy()

    paired = pd.DataFrame({
        "film": cmp.row_label, "lead_gender": cmp.lead_gender,
        "addr_z": z_addr, "cooc_z": z_cooc,
        "diff_z (addr - cooc)": z_addr - z_cooc,
        "addr_raw": raw_addr, "cooc_raw": raw_cooc,
        "diff_raw": raw_addr - raw_cooc,
    }).round(3)
    print(paired.sort_values("diff_z (addr - cooc)", key=abs, ascending=False).to_string(index=False))
    paired.to_csv(TBL_DIR / "phase3v2_paired_diffs.csv", index=False)

    print(f"\nWilcoxon signed-rank on z (addressee vs co-occurrence):")
    w_z = stats.wilcoxon(z_addr, z_cooc, alternative="two-sided", zero_method="wilcox")
    print(f"  W={w_z.statistic:.2f}  p={w_z.pvalue:.4f}  "
          f"mean diff (addr-cooc) = {(z_addr-z_cooc).mean():+.3f}  "
          f"median diff = {np.median(z_addr-z_cooc):+.3f}")

    print(f"\nWilcoxon signed-rank on raw share (addressee vs co-occurrence):")
    w_r = stats.wilcoxon(raw_addr, raw_cooc, alternative="two-sided", zero_method="wilcox")
    print(f"  W={w_r.statistic:.2f}  p={w_r.pvalue:.4f}  "
          f"mean diff = {(raw_addr-raw_cooc).mean():+.3f}  "
          f"median diff = {np.median(raw_addr-raw_cooc):+.3f}")

    spearman = stats.spearmanr(z_addr, z_cooc)
    pearson  = stats.pearsonr(z_addr, z_cooc)
    print(f"\nAgreement between methods (z-scores):")
    print(f"  Spearman ρ = {spearman.statistic:+.3f}  p = {spearman.pvalue:.4f}")
    print(f"  Pearson  r = {pearson.statistic:+.3f}  p = {pearson.pvalue:.4f}")

    # ---- 3.2 scatter ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for g in GENDER_ORDER:
        sub = cmp[cmp.lead_gender == g]
        ax.scatter(sub.addr_samesex_z, sub.cooc_samesex_z,
                   s=140, color=GENDER_PALETTE[g], edgecolor="black", linewidth=0.8,
                   label=f"Lead = {g}", zorder=3)
    for _, r in cmp.iterrows():
        ax.annotate(r.row_label, (r.addr_samesex_z, r.cooc_samesex_z),
                    xytext=(7, 5), textcoords="offset points", fontsize=8, color="#333")
    lo = min(cmp.addr_samesex_z.min(), cmp.cooc_samesex_z.min()) - 0.5
    hi = max(cmp.addr_samesex_z.max(), cmp.cooc_samesex_z.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=0.8, label="y = x")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Addressee-network same-gender homophily z (label-shuffled null)")
    ax.set_ylabel("Scene co-occurrence same-gender homophily z (label-shuffled null)")
    ax.set_title("Method comparison: addressee vs scene co-occurrence (per protagonist)")
    ax.legend(loc="lower right", title="Lead gender")
    savefig(fig, "fig15_method_comparison_scatter")

    # ---- 3.2 also a raw-share scatter ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for g in GENDER_ORDER:
        sub = cmp[cmp.lead_gender == g]
        ax.scatter(sub.addr_samesex, sub.cooc_samesex,
                   s=140, color=GENDER_PALETTE[g], edgecolor="black", linewidth=0.8,
                   label=f"Lead = {g}", zorder=3)
    for _, r in cmp.iterrows():
        ax.annotate(r.row_label, (r.addr_samesex, r.cooc_samesex),
                    xytext=(7, 5), textcoords="offset points", fontsize=8, color="#333")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Addressee-network raw same-gender share")
    ax.set_ylabel("Scene co-occurrence raw same-gender share")
    ax.set_title("Method comparison: raw same-gender share (per protagonist)")
    ax.legend(loc="lower right", title="Lead gender")
    savefig(fig, "fig16_method_comparison_raw_scatter")

    # ---- 3.3 divergence analysis --------------------------------------------
    banner("3.3 — Where do the methods diverge most?")
    paired_sorted = paired.sort_values("diff_z (addr - cooc)", key=abs, ascending=False)
    print("Top divergences (|addr_z - cooc_z|):")
    print(paired_sorted.head(6).to_string(index=False))

    print("\nDivergence by gender:")
    for g in GENDER_ORDER:
        sub = paired[paired.lead_gender == g]
        d = sub["diff_z (addr - cooc)"].to_numpy()
        print(f"  {g}: n={len(d)}  mean|diff|={np.mean(np.abs(d)):.3f}  "
              f"mean signed diff={d.mean():+.3f}  median signed diff={np.median(d):+.3f}")

    # ---- 3.4 re-run H1 with co-occurrence -----------------------------------
    banner("3.4 — Re-running H1 on co-occurrence-derived measures")

    def h1_test(x_f, x_m, label):
        mw_two = mannwhitney(x_f, x_m, "two-sided")
        mw_one = mannwhitney(x_f, x_m, "less")
        return {
            "label": label,
            "n_F": int(len(x_f)), "n_M": int(len(x_m)),
            "median_F": float(np.median(x_f)), "median_M": float(np.median(x_m)),
            "mean_F":   float(np.mean(x_f)),   "mean_M":   float(np.mean(x_m)),
            "mw_two_sided_U": mw_two["U"], "mw_two_sided_p": mw_two["p"],
            "mw_one_sided_F<M_p": mw_one["p"],
            "rank_biserial": rank_biserial(x_f, x_m),
            "cliffs_delta":  cliffs_delta(x_f, x_m),
        }

    f_zc = cmp.loc[cmp.lead_gender == "F", "cooc_samesex_z"].to_numpy()
    m_zc = cmp.loc[cmp.lead_gender == "M", "cooc_samesex_z"].to_numpy()
    f_ra = cmp.loc[cmp.lead_gender == "F", "cooc_samesex"].to_numpy()
    m_ra = cmp.loc[cmp.lead_gender == "M", "cooc_samesex"].to_numpy()

    f_za = cmp.loc[cmp.lead_gender == "F", "addr_samesex_z"].to_numpy()
    m_za = cmp.loc[cmp.lead_gender == "M", "addr_samesex_z"].to_numpy()
    f_xa = cmp.loc[cmp.lead_gender == "F", "addr_samesex"].to_numpy()
    m_xa = cmp.loc[cmp.lead_gender == "M", "addr_samesex"].to_numpy()

    panels = {
        "addressee_raw":   h1_test(f_xa, m_xa, "addressee raw samesex (recomputed)"),
        "addressee_z":     h1_test(f_za, m_za, "addressee z (label-shuffled null)"),
        "cooccurrence_raw": h1_test(f_ra, m_ra, "co-occurrence raw samesex"),
        "cooccurrence_z":  h1_test(f_zc, m_zc, "co-occurrence z (label-shuffled null)"),
    }
    for k, p in panels.items():
        print(f"\n>>> {p['label']}")
        print(f"    median F={p['median_F']:+.3f}  M={p['median_M']:+.3f}  diff={p['median_F']-p['median_M']:+.3f}")
        print(f"    mean   F={p['mean_F']:+.3f}  M={p['mean_M']:+.3f}  diff={p['mean_F']-p['mean_M']:+.3f}")
        print(f"    MW two-sided U={p['mw_two_sided_U']:.1f} p={p['mw_two_sided_p']:.4f}  |  "
              f"one-sided (F<M) p={p['mw_one_sided_F<M_p']:.4f}")
        print(f"    Cliff's δ = {p['cliffs_delta']:+.3f}    rank-biserial = {p['rank_biserial']:+.3f}")

    with open(TBL_DIR / "phase3v2_h1_under_both_methods.json", "w") as fh:
        json.dump(panels, fh, indent=2)

    # ---- 3.4 figure: side-by-side boxplots ----------------------------------
    long = pd.concat([
        cmp[["lead_gender", "row_label", "addr_samesex_z"]]
           .rename(columns={"addr_samesex_z": "z"}).assign(method="Addressee LLM"),
        cmp[["lead_gender", "row_label", "cooc_samesex_z"]]
           .rename(columns={"cooc_samesex_z": "z"}).assign(method="Scene co-occurrence"),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    sns.boxplot(data=long, x="method", y="z", hue="lead_gender",
                order=["Addressee LLM", "Scene co-occurrence"],
                hue_order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                width=0.55, showcaps=False, boxprops={"alpha": .3},
                medianprops={"color": "black", "linewidth": 2})
    sns.stripplot(data=long, x="method", y="z", hue="lead_gender",
                  order=["Addressee LLM", "Scene co-occurrence"],
                  hue_order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                  size=8, dodge=True, jitter=0.10, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="grey", lw=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Lead gender", loc="best")
    ax.set_xlabel(""); ax.set_ylabel("Same-gender homophily z (label-shuffled null)")
    ax.set_title("H1 under two network construction methods")
    savefig(fig, "fig17_h1_under_both_methods")

    print("\nPhase 3 (revised) complete.")


if __name__ == "__main__":
    main()
