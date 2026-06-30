"""
Cross-film batch tests for the added value of addressee tagging.

Pipeline role:
  1. Notebook 06 (`CLEAN/notebooks/06_network_analysis_PAU.ipynb`) is the
     per-film production layer. It creates `network_edges.csv` and per-film
     network metrics.
  2. `phase3_crossfilm_method_validation.py` provides shared cross-film helpers
     that consume Notebook 06 outputs and build the co-occurrence baseline.
  3. This script is the second cross-film batch layer. It asks where addressee
     tagging adds information that scene co-occurrence cannot encode.
  4. Notebook 09 (`CLEAN/notebooks/09_analysis.ipynb`) reports the cross-film
     results in thesis-facing figures and prose.

Four tests:
  1. Edge-set divergence: pairs that co-occur but never speak, and vice versa.
  2. Reciprocity by gender: a directed measurement co-occurrence cannot produce.
  3. Keystone identity agreement: whether each method nominates the same bridge.
  4. Divergence versus scene complexity.

Compatibility:
  Output table/figure names still use the historic `phase3v3_*` and fig18-fig20
  names so downstream notebooks and generated markdown do not need data-path
  changes.
"""
from __future__ import annotations

import json, sys, warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    FIG_DIR, GENDER_ORDER, GENDER_PALETTE, TBL_DIR,
    cliffs_delta, load_df, mannwhitney, rank_biserial, savefig, set_style,
)
# Reuse cross-film helpers that consume Notebook 06 per-film outputs.
from phase3_crossfilm_method_validation import (
    MIN_EDGE_COUNT, build_addressee_network, build_cooccurrence_network,
    build_id_remap, film_dir_for, gender_map, load_character_meta, protag_id,
)


def banner(m): print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)


# -------- Test 1: edge-set divergence ----------------------------------------

def gender_pair_label(a_gender: str, b_gender: str) -> str:
    """Same-gender label: 'FF', 'MM', 'cross' (any F-M pair)."""
    if a_gender == b_gender:
        return a_gender + a_gender
    return "cross"


def edge_divergence(film_id: str, meta: pd.DataFrame, remap: dict,
                    valid: set[str], gender: dict) -> dict:
    addr = build_addressee_network(film_id, remap, valid)
    cooc = build_cooccurrence_network(film_id, remap, valid)
    A = set(addr.keys()); C = set(cooc.keys())
    only_addr = A - C
    only_cooc = C - A
    both      = A & C

    def classify(pairs: set) -> Counter:
        out = Counter()
        for pair in pairs:
            a, b = list(pair)
            ga, gb = gender.get(a), gender.get(b)
            if ga and gb:
                out[gender_pair_label(ga, gb)] += 1
        return out

    return {
        "n_addr": len(A), "n_cooc": len(C),
        "n_both": len(both), "n_only_addr": len(only_addr), "n_only_cooc": len(only_cooc),
        "class_only_addr": dict(classify(only_addr)),
        "class_only_cooc": dict(classify(only_cooc)),
        "class_both":      dict(classify(both)),
        "class_all_addr":  dict(classify(A)),
        "class_all_cooc":  dict(classify(C)),
    }


# -------- Test 2: reciprocity from addressee network -------------------------

def addressee_reciprocity(film_id: str, remap: dict, valid: set[str]) -> dict:
    """Build directed addressee network from raw edges file; compute reciprocity."""
    fdir = film_dir_for(film_id)
    e = pd.read_csv(fdir / "network_edges.csv")
    dir_w: dict[tuple, int] = {}
    for _, r in e.iterrows():
        a = remap.get(r.speaker_clean, r.speaker_clean)
        b = remap.get(r.addressee_clean, r.addressee_clean)
        if a == b or a not in valid or b not in valid:
            continue
        dir_w[(a, b)] = dir_w.get((a, b), 0) + int(r.utterance_count)
    # apply MIN_EDGE_COUNT on directed edges
    dir_w = {k: v for k, v in dir_w.items() if v >= MIN_EDGE_COUNT}
    if not dir_w:
        return {"n_directed_edges": 0, "n_reciprocal_pairs": 0,
                "reciprocity_pair": np.nan, "reciprocity_edge": np.nan}
    pairs = {frozenset({a, b}) for (a, b) in dir_w}
    reciprocal_pairs = sum(1 for p in pairs
                           if (list(p)[0], list(p)[1]) in dir_w
                           and (list(p)[1], list(p)[0]) in dir_w)
    # edge-level reciprocity: share of directed edges whose reverse exists
    reciprocal_edges = sum(1 for (a, b) in dir_w if (b, a) in dir_w)
    return {
        "n_directed_edges": len(dir_w),
        "n_unique_pairs": len(pairs),
        "n_reciprocal_pairs": reciprocal_pairs,
        "reciprocity_pair": reciprocal_pairs / len(pairs),
        "reciprocity_edge": reciprocal_edges / len(dir_w),
    }


# -------- Test 3: keystone identity agreement --------------------------------

def keystone_in_network(edges: dict, protag: str) -> str | None:
    """Highest-betweenness non-protagonist node in a weighted undirected graph."""
    if not edges or protag is None:
        return None
    G = nx.Graph()
    for k, w in edges.items():
        a, b = list(k)
        # invert weight for betweenness "distance": higher weight = closer
        G.add_edge(a, b, weight=w, distance=1.0 / w)
    if protag not in G:
        return None
    btw = nx.betweenness_centrality(G, weight="distance", normalized=True)
    non_lead = [(n, v) for n, v in btw.items() if n != protag]
    if not non_lead:
        return None
    return max(non_lead, key=lambda x: x[1])[0]


# -------- Test 4: scene complexity per film ----------------------------------

def scene_complexity(film_id: str, remap: dict, valid: set[str]) -> dict:
    fdir = film_dir_for(film_id)
    u = pd.read_csv(fdir / "utterances_with_addressee_scene.csv",
                    usecols=["scene_id", "character_id"]).dropna()
    u["character_id"] = u.character_id.map(lambda x: remap.get(x, x))
    u = u[u.character_id.isin(valid)]
    scene_size = u.groupby("scene_id")["character_id"].apply(lambda s: len(set(s)))
    scene_size = scene_size[scene_size >= 1]
    return {
        "n_scenes_with_dialogue": int(len(scene_size)),
        "scene_mean_speakers": float(scene_size.mean()),
        "scene_max_speakers":  int(scene_size.max()),
        "share_scenes_ge2": float((scene_size >= 2).mean()),
        "share_scenes_ge3": float((scene_size >= 3).mean()),
        "share_scenes_ge4": float((scene_size >= 4).mean()),
    }


# -------- main ---------------------------------------------------------------

def main():
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load_df()

    # Also load the v2 comparison so we have addr/cooc z-scores per film
    cmp = pd.read_csv(TBL_DIR / "phase3v2_method_comparison.csv")

    rows = []
    for _, r in df.iterrows():
        film_id = r.film_id
        meta = load_character_meta(film_id)
        remap = build_id_remap(meta)
        gmap = gender_map(meta)
        valid = set(gmap.keys())
        protag = protag_id(meta, r.protagonist)

        div = edge_divergence(film_id, meta, remap, valid, gmap)
        recip = addressee_reciprocity(film_id, remap, valid)
        ks_addr = keystone_in_network(build_addressee_network(film_id, remap, valid), protag)
        ks_cooc = keystone_in_network(build_cooccurrence_network(film_id, remap, valid), protag)
        cx = scene_complexity(film_id, remap, valid)

        def name_of(cid):
            if cid is None: return None
            row = meta[meta.character_id == cid]
            return row.iloc[0].canonical_name if len(row) else cid

        def gender_of(cid):
            return gmap.get(cid)

        rows.append({
            "film_id": film_id, "protagonist": r.protagonist,
            "lead_gender": r.lead_gender,
            **{f"div_{k}": v for k, v in div.items()},
            **{f"recip_{k}": v for k, v in recip.items()},
            "keystone_addr": name_of(ks_addr),
            "keystone_addr_gender": gender_of(ks_addr),
            "keystone_cooc": name_of(ks_cooc),
            "keystone_cooc_gender": gender_of(ks_cooc),
            "keystone_agree": ks_addr == ks_cooc and ks_addr is not None,
            **cx,
        })
    F = pd.DataFrame(rows)
    F["film_title"] = F["film_id"].map(df.set_index("film_id")["film_title"].drop_duplicates())
    F.to_csv(TBL_DIR / "phase3v3_addressee_value.csv", index=False)

    # ============ TEST 1 — edge-set divergence ==============================
    banner("TEST 1 — edge-set divergence (phantom ties)")
    # Drop Monsters Inc dup row for film-level edge analysis (network is identical for both leads)
    F_film = F.drop_duplicates("film_id").reset_index(drop=True)

    print("\nPer-film edge counts and divergence:")
    show = F_film[["film_title", "lead_gender",
                   "div_n_addr", "div_n_cooc", "div_n_both",
                   "div_n_only_addr", "div_n_only_cooc"]]
    print(show.to_string(index=False))

    # aggregate "phantom" composition (edges only in cooc) — split by gender
    agg_only_cooc = Counter()
    agg_only_addr = Counter()
    agg_both = Counter()
    for _, r in F_film.iterrows():
        for k in ("FF", "MM", "cross"):
            agg_only_cooc[k] += r.div_class_only_cooc.get(k, 0) if isinstance(r.div_class_only_cooc, dict) else 0
            agg_only_addr[k] += r.div_class_only_addr.get(k, 0) if isinstance(r.div_class_only_addr, dict) else 0
            agg_both[k] += r.div_class_both.get(k, 0) if isinstance(r.div_class_both, dict) else 0

    # The rows have dicts when freshly built but become strings after CSV roundtrip — rebuild safe:
    def gen_dict(col):
        out = Counter()
        for _, r in F_film.iterrows():
            d = r[col]
            if isinstance(d, str):
                d = eval(d)
            for k, v in d.items():
                out[k] += v
        return out
    agg_only_cooc = gen_dict("div_class_only_cooc")
    agg_only_addr = gen_dict("div_class_only_addr")
    agg_both      = gen_dict("div_class_both")

    print("\nAggregate edge-set composition across 12 films:")
    print(f"  Only in CO-OCCURRENCE (phantom-talk edges): FF={agg_only_cooc.get('FF',0):4d}  "
          f"MM={agg_only_cooc.get('MM',0):4d}  cross={agg_only_cooc.get('cross',0):4d}  "
          f"total={sum(agg_only_cooc.values())}")
    print(f"  Only in ADDRESSEE   (cross-scene talk)   :  FF={agg_only_addr.get('FF',0):4d}  "
          f"MM={agg_only_addr.get('MM',0):4d}  cross={agg_only_addr.get('cross',0):4d}  "
          f"total={sum(agg_only_addr.values())}")
    print(f"  In BOTH                                   : FF={agg_both.get('FF',0):4d}  "
          f"MM={agg_both.get('MM',0):4d}  cross={agg_both.get('cross',0):4d}  "
          f"total={sum(agg_both.values())}")

    # Same-gender share within each bucket
    def same_share(c):
        same = c.get("FF", 0) + c.get("MM", 0)
        tot = same + c.get("cross", 0)
        return same / tot if tot else float("nan")
    s_phantom = same_share(agg_only_cooc)
    s_addrxsc = same_share(agg_only_addr)
    s_both    = same_share(agg_both)
    print(f"\n  Same-gender share among 'only in co-occurrence' ties: {s_phantom:.3f}")
    print(f"  Same-gender share among 'only in addressee'      ties: {s_addrxsc:.3f}")
    print(f"  Same-gender share among 'in both'                 ties: {s_both:.3f}")
    print("  (If co-occurrence-only ties skew same-gender more than the both-set,\n"
          "   it would mean co-occurrence inflates same-gender ties from crowd scenes.)")

    # Chi-squared test: same/cross composition of cooc-only vs both
    contingency = np.array([
        [agg_only_cooc.get("FF", 0) + agg_only_cooc.get("MM", 0), agg_only_cooc.get("cross", 0)],
        [agg_both.get("FF", 0)      + agg_both.get("MM", 0),      agg_both.get("cross", 0)],
    ])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    print(f"\n  Chi-squared (same vs cross, cooc-only vs both): χ²={chi2:.2f} p={p_chi:.4f} (dof={dof})")

    # Per-film: phantom rate by lead gender
    F_film["phantom_total"] = F_film.div_class_only_cooc.apply(
        lambda d: sum(eval(d).values()) if isinstance(d, str) else sum(d.values()))
    F_film["phantom_same"] = F_film.div_class_only_cooc.apply(
        lambda d: (eval(d) if isinstance(d, str) else d).get("FF",0)
                  + (eval(d) if isinstance(d, str) else d).get("MM",0))
    F_film["phantom_share_same"] = F_film.phantom_same / F_film.phantom_total.replace(0, np.nan)
    print("\nPhantom-edge same-gender share per film:")
    print(F_film[["film_title", "lead_gender",
                  "div_n_only_cooc", "phantom_same", "phantom_share_same"]]
          .round(3).to_string(index=False))

    # Figure: stacked bar of edge buckets per film
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, gender_filter, title in zip(
        axes, ["F", "M"], ["Female-led films", "Male-led films"]):
        sub = F_film[F_film.lead_gender == gender_filter].sort_values("film_title")
        x = np.arange(len(sub))
        only_cooc_same = [(eval(d) if isinstance(d, str) else d).get("FF",0)
                          + (eval(d) if isinstance(d, str) else d).get("MM",0)
                          for d in sub.div_class_only_cooc]
        only_cooc_cross = [(eval(d) if isinstance(d, str) else d).get("cross",0)
                           for d in sub.div_class_only_cooc]
        both_same = [(eval(d) if isinstance(d, str) else d).get("FF",0)
                     + (eval(d) if isinstance(d, str) else d).get("MM",0)
                     for d in sub.div_class_both]
        both_cross = [(eval(d) if isinstance(d, str) else d).get("cross",0)
                      for d in sub.div_class_both]
        ax.bar(x - 0.2, both_same,   width=0.4, color="#6B7280", label="Both — same-gender")
        ax.bar(x - 0.2, both_cross,  width=0.4, bottom=both_same, color="#D1D5DB", label="Both — cross-gender")
        ax.bar(x + 0.2, only_cooc_same,  width=0.4, color="#F59E0B", label="Co-occ only — same-gender (phantom)")
        ax.bar(x + 0.2, only_cooc_cross, width=0.4,
               bottom=only_cooc_same, color="#FEF3C7", label="Co-occ only — cross-gender (phantom)")
        ax.set_xticks(x); ax.set_xticklabels(sub.film_title, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("# edges"); ax.set_title(title, color=GENDER_PALETTE[gender_filter])
        if gender_filter == "F":
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Edge-set decomposition: both methods vs co-occurrence-only (phantom)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig18_phantom_edges_by_film")

    # ============ TEST 2 — reciprocity by gender ===========================
    banner("TEST 2 — reciprocity by lead gender (addressee, directed)")
    # Use ALL 13 protagonist rows? Reciprocity is film-level, so dedup. But we keep the
    # film-level number paired with each lead row for documentation; dedup for the stat.
    R = F.drop_duplicates("film_id").copy()
    print("\nPer-film reciprocity (addressee, MIN_EDGE_COUNT=3 directed):")
    print(R[["film_title", "lead_gender", "recip_n_directed_edges",
             "recip_n_unique_pairs", "recip_n_reciprocal_pairs",
             "recip_reciprocity_pair", "recip_reciprocity_edge"]]
          .round(3).to_string(index=False))

    for col in ["recip_reciprocity_pair", "recip_reciprocity_edge"]:
        f = R.loc[R.lead_gender == "F", col].to_numpy()
        m = R.loc[R.lead_gender == "M", col].to_numpy()
        mw = mannwhitney(f, m, "two-sided")
        print(f"\n  {col}: F median={np.median(f):.3f} M median={np.median(m):.3f}  "
              f"diff(F-M)={np.median(f)-np.median(m):+.3f}  "
              f"MW p={mw['p']:.4f}  Cliff's δ={cliffs_delta(f,m):+.2f}  r_rb={rank_biserial(f,m):+.2f}")

    # Figure: reciprocity by gender
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sns.boxplot(data=R, x="lead_gender", y="recip_reciprocity_edge", hue="lead_gender",
                order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                width=0.45, showcaps=False, boxprops={"alpha": .3},
                medianprops={"color": "black", "linewidth": 2}, legend=False)
    sns.stripplot(data=R, x="lead_gender", y="recip_reciprocity_edge", hue="lead_gender",
                  order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                  size=9, jitter=0.15, edgecolor="black", linewidth=0.8, legend=False)
    for _, r in R.iterrows():
        ax.annotate(r.film_title, (GENDER_ORDER.index(r.lead_gender), r.recip_reciprocity_edge),
                    xytext=(9, 0), textcoords="offset points", fontsize=7.5, color="#333")
    ax.set_xlabel("Lead gender")
    ax.set_ylabel("Edge-level reciprocity (directed addressee)")
    ax.set_title("Directed reciprocity by lead gender (addressee only — co-occurrence cannot measure)")
    savefig(fig, "fig19_reciprocity_by_gender")

    # ============ TEST 3 — keystone identity agreement =====================
    banner("TEST 3 — keystone identity agreement (addressee vs co-occurrence)")
    K = F.copy()
    print(K[["film_title", "protagonist", "lead_gender",
             "keystone_addr", "keystone_addr_gender",
             "keystone_cooc", "keystone_cooc_gender",
             "keystone_agree"]].to_string(index=False))
    agree_rate = K.keystone_agree.mean()
    print(f"\nAgreement rate (same character nominated as keystone): {agree_rate:.2%} "
          f"({K.keystone_agree.sum()}/{len(K)})")
    # Where do they disagree, and does gender of keystone differ?
    disagree = K[~K.keystone_agree]
    print(f"\nDisagreements ({len(disagree)} cases):")
    if len(disagree):
        print(disagree[["film_title", "protagonist", "lead_gender",
                        "keystone_addr", "keystone_addr_gender",
                        "keystone_cooc", "keystone_cooc_gender"]].to_string(index=False))
        diff_gender_cases = disagree[disagree.keystone_addr_gender != disagree.keystone_cooc_gender]
        print(f"\n  Of {len(disagree)} disagreements, "
              f"{len(diff_gender_cases)} flip the keystone's GENDER between methods.")
        if len(diff_gender_cases):
            print(diff_gender_cases[["film_title", "lead_gender",
                                     "keystone_addr", "keystone_addr_gender",
                                     "keystone_cooc", "keystone_cooc_gender"]].to_string(index=False))

    # Also: addressee vs the original `keystone` column from film_features_all.csv
    addr_vs_pipeline = K.merge(
        df[["film_id", "protagonist", "keystone", "keystone_gender"]],
        on=["film_id", "protagonist"], how="left",
        suffixes=("", "_pipeline"))
    addr_vs_pipeline["agree_with_pipeline"] = \
        addr_vs_pipeline.keystone_addr.str.lower() == addr_vs_pipeline.keystone.str.lower()
    print(f"\nFor reference — addressee keystone vs pipeline keystone (config-model based):")
    print(addr_vs_pipeline[["film_title", "protagonist", "keystone_addr",
                            "keystone", "agree_with_pipeline"]].to_string(index=False))
    print(f"Agreement with the existing pipeline keystone column: "
          f"{addr_vs_pipeline.agree_with_pipeline.mean():.2%}")

    # ============ TEST 4 — divergence vs scene complexity ==================
    banner("TEST 4 — |Δz| (addr vs cooc) vs scene complexity")
    # Merge per-protagonist |Δz| from v2 with per-film complexity
    cmp_min = cmp[["film_id", "protagonist", "lead_gender",
                   "addr_samesex_z", "cooc_samesex_z"]].copy()
    cmp_min["abs_diff_z"] = (cmp_min.addr_samesex_z - cmp_min.cooc_samesex_z).abs()
    merged = cmp_min.merge(
        F[["film_id", "protagonist", "share_scenes_ge3", "share_scenes_ge4",
           "scene_mean_speakers", "scene_max_speakers", "n_scenes_with_dialogue"]],
        on=["film_id", "protagonist"], how="left",
    )
    merged["film_title"] = merged["film_id"].map(
        df.set_index("film_id")["film_title"].drop_duplicates())

    print(merged[["film_title", "protagonist", "lead_gender", "abs_diff_z",
                  "share_scenes_ge3", "share_scenes_ge4",
                  "scene_mean_speakers"]].round(3).to_string(index=False))

    for col in ["share_scenes_ge3", "share_scenes_ge4", "scene_mean_speakers"]:
        r_s = stats.spearmanr(merged[col], merged.abs_diff_z)
        r_p = stats.pearsonr(merged[col], merged.abs_diff_z)
        print(f"\n  |Δz| vs {col}:")
        print(f"    Spearman ρ = {r_s.statistic:+.3f}, p = {r_s.pvalue:.4f}")
        print(f"    Pearson r  = {r_p.statistic:+.3f}, p = {r_p.pvalue:.4f}")

    # Figure: scatter |Δz| vs share_scenes_ge4 (the headline complexity measure)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for g in GENDER_ORDER:
        sub = merged[merged.lead_gender == g]
        ax.scatter(sub.share_scenes_ge4, sub.abs_diff_z,
                   s=130, color=GENDER_PALETTE[g], edgecolor="black",
                   linewidth=0.8, label=f"Lead = {g}", zorder=3)
    for _, r in merged.iterrows():
        lab = r.film_title if r.protagonist not in ("Mike", "Sulley") else f"{r.film_title} — {r.protagonist}"
        ax.annotate(lab, (r.share_scenes_ge4, r.abs_diff_z),
                    xytext=(7, 5), textcoords="offset points", fontsize=8, color="#333")
    # OLS line
    if len(merged) >= 3:
        b1, b0 = np.polyfit(merged.share_scenes_ge4, merged.abs_diff_z, 1)
        xs = np.linspace(merged.share_scenes_ge4.min(), merged.share_scenes_ge4.max(), 50)
        ax.plot(xs, b0 + b1*xs, color="grey", lw=1, linestyle="--",
                label=f"OLS: y = {b0:+.2f} + {b1:+.2f}·x")
    ax.set_xlabel("Share of scenes with ≥4 distinct speakers (crowd-scene density)")
    ax.set_ylabel("|Δz| between methods (addressee vs co-occurrence)")
    ax.set_title("Method divergence grows with scene complexity?")
    ax.legend(loc="best")
    savefig(fig, "fig20_divergence_vs_scene_complexity")

    print("\nPhase 3 (v3) complete — addressee value tests done.")


if __name__ == "__main__":
    main()
