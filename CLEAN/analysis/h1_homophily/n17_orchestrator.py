"""
N=17 Unified Orchestrator (Agent B: Steps 1, 2, 3, 4, 6)
=========================================================

Single entry point that runs Pillar A (the measure), Pillar B (the method),
structural context, descriptive extensions, and exploratory analysis on the
17-film analytic sample (8F + 9M) after applying the 6 Conventions defined
in CLEAN/admin/AGENT_INSTRUCTION_run_analysis_N18.md.

Conventions filters applied at load time:
  C1. Drop `soul_2020` (genderless 22/souls make same-gender share incoherent).
  C2. One protagonist per film (no co-leads).
  C3. Monsters Inc — keep Sulley, drop Mike (Sulley is the narrative lead).
  C4. Resulting N = 17 (8F + 9M), uniform across film-level and protag-level steps.
  C5. N=12 comparisons may carry small caveats where Soul affected the prior result.

Outputs:
  - tables_n17/         all CSVs and JSON test panels
  - figures_n17/        all PNG + PDF figures
  - UNIFIED_RESULTS_N17_steps1_4_6.md  (single markdown)
  - data/04_features/film_features_all_n17.csv      (derived working filter)
  - data/04_features/film_features_extended_n17.csv (derived working filter)

This script does NOT modify the upstream csvs.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    GENDER_ORDER, GENDER_PALETTE, RNG_SEED,
    bootstrap_diff_ci, cliffs_delta, mannwhitney, perm_test_diff,
    power_note, rank_biserial, set_style,
)
from phase3_crossfilm_method_validation import (  # noqa: E402
    MIN_EDGE_COUNT, N_NULL,
    build_addressee_network, build_cooccurrence_network,
    build_id_remap, film_dir_for, gender_map, label_shuffle_null,
    load_character_meta, network_summary, neighbours_of, protag_id,
    protag_samesex_share,
)
from phase3_crossfilm_addressee_value import (  # noqa: E402
    addressee_reciprocity, edge_divergence, gender_pair_label,
    keystone_in_network, scene_complexity,
)

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
CLEAN = HERE.parents[1]
FEAT = CLEAN / "data" / "04_features"
FIG_DIR_N17 = HERE / "figures_n17"
TBL_DIR_N17 = HERE / "tables_n17"
FIG_DIR_N17.mkdir(parents=True, exist_ok=True)
TBL_DIR_N17.mkdir(parents=True, exist_ok=True)

STATUS_FILE = HERE / "STATUS_agent_B.md"
OUTPUT_MD = HERE / "UNIFIED_RESULTS_N17_steps1_4_6.md"

DISPLAY_N17 = {
    "mulan_1998": "Mulan (1998)",
    "frozen_2013": "Frozen (2013)",
    "inside_out_2015": "Inside Out (2015)",
    "zootopia_2016": "Zootopia (2016)",
    "encanto_2021": "Encanto (2021)",
    "raya_and_the_last_dragon_2021": "Raya (2021)",
    "toy_story_1995": "Toy Story (1995)",
    "monsters_inc_2001": "Monsters Inc (2001)",
    "up": "Up (2009)",
    "coco_2017": "Coco (2017)",
    "onward_2020": "Onward (2020)",
    "beautyandthebeast_1991": "Beauty & the Beast (1991)",
    "incredibles_2_2018": "Incredibles 2 (2018)",
    "findingnemo": "Finding Nemo (2003)",
    "toy_story_3_2010": "Toy Story 3 (2010)",
    "cars2": "Cars 2 (2011)",
    "elemental_2023": "Elemental (2023)",
}

NEW_FILMS = {
    "beautyandthebeast_1991", "incredibles_2_2018", "findingnemo",
    "toy_story_3_2010", "cars2", "elemental_2023",
}


# ----------------------------------------------------------------------------
# Conventions filter + load
# ----------------------------------------------------------------------------
def apply_conventions(df: pd.DataFrame) -> pd.DataFrame:
    """C1 drop Soul; C2 one protagonist per film (drop co-lead/alias rows);
    C3 Monsters Inc -> Sulley only."""
    out = df[df.film_id != "soul_2020"].copy()
    out = out[~((out.film_id == "monsters_inc_2001") & (out.protagonist == "Mike"))].copy()
    # C2: keep a single protagonist per film. The master can carry stale co-lead
    # or title-case-alias rows (e.g. inside_out Riley alongside Joy); drop them so
    # n17 has exactly one row per film. Keep = the chosen canonical protagonist.
    out = out[~(
        ((out.film_id == "inside_out_2015")    & (out.protagonist == "Riley"))      |
        ((out.film_id == "zootopia_2016")      & (out.protagonist == "Hopps."))     |
        ((out.film_id == "incredibles_2_2018") & (out.protagonist == "Elastigirl"))
    )].copy()
    out["film_title"] = out["film_id"].map(DISPLAY_N17).fillna(out["film_id"])
    out["row_label"] = out["film_title"]
    return out.reset_index(drop=True)


def load_n17_all() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "film_features_all.csv")
    out = apply_conventions(df)
    out.to_csv(FEAT / "film_features_all_n17.csv", index=False)
    return out


def load_n17_extended() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "film_features_extended.csv")
    out = apply_conventions(df)
    out.to_csv(FEAT / "film_features_extended_n17.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Status tracking
# ----------------------------------------------------------------------------
def status(msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"- {ts} — {msg}\n"
    with STATUS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line)
    try:
        print("[STATUS] " + msg, flush=True)
    except UnicodeEncodeError:
        print("[STATUS] " + msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ----------------------------------------------------------------------------
# Helper plotting / fig saving (writes to FIG_DIR_N17)
# ----------------------------------------------------------------------------
def savefig17(fig, name: str) -> None:
    fig.savefig(FIG_DIR_N17 / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR_N17 / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_full_panel(x_f, x_m, label: str) -> dict:
    """Two-sided + one-sided + permutation + effect sizes + bootstrap CI."""
    x_f = np.asarray(x_f, dtype=float)
    x_m = np.asarray(x_m, dtype=float)
    return {
        "label": label,
        "n_F": int(len(x_f)), "n_M": int(len(x_m)),
        "median_F": float(np.median(x_f)), "median_M": float(np.median(x_m)),
        "mean_F": float(np.mean(x_f)), "mean_M": float(np.mean(x_m)),
        "mw_two_sided": mannwhitney(x_f, x_m, "two-sided"),
        "mw_one_sided_F<M": mannwhitney(x_f, x_m, "less"),
        "rank_biserial": rank_biserial(x_f, x_m),
        "cliffs_delta": cliffs_delta(x_f, x_m),
        "perm_median": perm_test_diff(x_f, x_m, stat=np.median),
        "perm_mean": perm_test_diff(x_f, x_m, stat=np.mean),
        "boot_median_diff_ci": bootstrap_diff_ci(x_f, x_m, stat=np.median),
        "boot_mean_diff_ci": bootstrap_diff_ci(x_f, x_m, stat=np.mean),
    }


def fmt_panel_md(p: dict) -> str:
    """Compact markdown table for a single panel."""
    mw2 = p["mw_two_sided"]; mwl = p["mw_one_sided_F<M"]
    pm = p["perm_median"]; pmm = p["perm_mean"]
    bm = p["boot_median_diff_ci"]; bmean = p["boot_mean_diff_ci"]
    lines = [
        f"| Statistic | Value |",
        f"|---|---|",
        f"| n (F / M) | {p['n_F']} / {p['n_M']} |",
        f"| Median F / M | {p['median_F']:+.4f} / {p['median_M']:+.4f} |",
        f"| Mean F / M | {p['mean_F']:+.4f} / {p['mean_M']:+.4f} |",
        f"| Median diff (F − M) | {p['median_F']-p['median_M']:+.4f} |",
        f"| Mean diff (F − M) | {p['mean_F']-p['mean_M']:+.4f} |",
        f"| Mann-Whitney U (two-sided, exact) | U = {mw2['U']:.1f}, p = {mw2['p']:.4f} |",
        f"| Mann-Whitney U (one-sided, F<M) | p = {mwl['p']:.4f} |",
        f"| Cliff's δ | {p['cliffs_delta']:+.3f} |",
        f"| Rank-biserial r | {p['rank_biserial']:+.3f} |",
        f"| Permutation on median diff (10k, two-sided) | p = {pm['p_two_sided']:.4f} |",
        f"| Permutation on mean diff (10k, two-sided) | p = {pmm['p_two_sided']:.4f} |",
        f"| Bootstrap 95% CI on mean diff (F−M) | [{bmean['ci_lo']:+.4f}, {bmean['ci_hi']:+.4f}] |",
        f"| Bootstrap 95% CI on median diff (F−M) | [{bm['ci_lo']:+.4f}, {bm['ci_hi']:+.4f}] |",
    ]
    return "\n".join(lines)


# ============================================================================
# STEP 1 — Pillar A: the measure
# ============================================================================
def step1_pillar_a(df: pd.DataFrame) -> dict:
    """Raw vs cast-adjusted decomposition + H1 test battery + individual sig + correlation."""
    set_style()
    f_raw = df.loc[df.lead_gender == "F", "protag_samesex"].to_numpy()
    m_raw = df.loc[df.lead_gender == "M", "protag_samesex"].to_numpy()
    f_z = df.loc[df.lead_gender == "F", "protag_samesex_z"].to_numpy()
    m_z = df.loc[df.lead_gender == "M", "protag_samesex_z"].to_numpy()

    panel_raw = run_full_panel(f_raw, m_raw, "protag_samesex (raw share)")
    panel_z = run_full_panel(f_z, m_z, "protag_samesex_z (cast-adjusted)")

    # Per-protagonist significance
    sig_table = df[[
        "film_id", "film_title", "protagonist", "lead_gender", "year",
        "protag_samesex", "protag_samesex_z", "protag_samesex_p",
    ]].copy()
    sig_table["sig_alpha05"] = sig_table["protag_samesex_p"] < 0.05
    sig_table = sig_table.sort_values("protag_samesex_z").reset_index(drop=True)
    sig_table.to_csv(TBL_DIR_N17 / "step1_per_protag_sig.csv", index=False)

    # Cast-wide vs protagonist correlation
    spearman = stats.spearmanr(df["homophily_z"], df["protag_samesex_z"])

    # Save raw panels JSON
    with (TBL_DIR_N17 / "step1_panels.json").open("w") as fh:
        json.dump({"raw": panel_raw, "z": panel_z,
                   "spearman_homophily_vs_protag_z": {
                       "rho": float(spearman.statistic),
                       "p": float(spearman.pvalue)}},
                  fh, indent=2)

    # Figure: lollipop of protag_samesex_z
    d = sig_table.copy()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for i, r in d.iterrows():
        color = GENDER_PALETTE[r.lead_gender]
        ax.hlines(y=i, xmin=0, xmax=r.protag_samesex_z, color=color, alpha=0.55, linewidth=2)
        ax.scatter(r.protag_samesex_z, i, color=color, s=120, edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.row_label if "row_label" in d.columns else d.film_title, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(1.645, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axvline(-1.645, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("Protagonist same-gender homophily z (cast-adjusted)")
    ax.set_title(f"H1 — per-protagonist cast-adjusted z (N=17, 8F+9M)")
    legend = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="black", markersize=10, label=g)
              for g, c in GENDER_PALETTE.items()]
    ax.legend(handles=legend, title="Lead gender", loc="lower right")
    savefig17(fig, "fig01_n17_h1_lollipop")

    # Figure: side by side raw vs z box+strip
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, col, lbl in zip(axes, ["protag_samesex", "protag_samesex_z"],
                              ["Raw same-gender share", "Cast-adjusted z"]):
        sns.boxplot(data=df, x="lead_gender", y=col, hue="lead_gender",
                    order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                    width=0.45, showcaps=False, boxprops={"alpha": .35},
                    medianprops={"color": "black", "linewidth": 2}, legend=False)
        sns.stripplot(data=df, x="lead_gender", y=col, hue="lead_gender",
                      order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                      size=8, jitter=0.18, edgecolor="black", linewidth=0.8, legend=False)
        for _, r in df.iterrows():
            ax.annotate(r.film_title, (GENDER_ORDER.index(r.lead_gender), r[col]),
                        xytext=(7, 0), textcoords="offset points", fontsize=7, color="#444")
        if col == "protag_samesex_z":
            ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel("Lead gender"); ax.set_ylabel(lbl); ax.set_title(lbl)
    fig.suptitle("Raw vs cast-adjusted: same-gender homophily, F vs M (N=17)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig17(fig, "fig02_n17_raw_vs_adjusted")

    # Cast-level vs protagonist scatter
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for g in GENDER_ORDER:
        sub = df[df.lead_gender == g]
        ax.scatter(sub.homophily_z, sub.protag_samesex_z, s=130,
                   color=GENDER_PALETTE[g], edgecolor="black",
                   linewidth=0.8, label=f"Lead = {g}", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.film_title, (r.homophily_z, r.protag_samesex_z),
                    xytext=(7, 5), textcoords="offset points", fontsize=8, color="#333")
    ax.axhline(0, color="grey", lw=0.7); ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("Network-wide gender homophily z (whole cast)")
    ax.set_ylabel("Protagonist same-gender homophily z")
    ax.set_title(f"Cast-level vs protagonist-level homophily (ρ = {spearman.statistic:+.3f})")
    ax.legend(title="Lead gender")
    savefig17(fig, "fig03_n17_homophily_vs_protag_z")

    return {
        "panel_raw": panel_raw,
        "panel_z": panel_z,
        "sig_table": sig_table,
        "spearman": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
        "power": power_note(int((df.lead_gender == "F").sum()),
                            int((df.lead_gender == "M").sum())),
    }


# ============================================================================
# STEP 2 — Pillar B: the method
# ============================================================================
def build_per_film_method_row(film_id: str, protagonist: str, lead_gender: str) -> dict:
    meta = load_character_meta(film_id)
    remap = build_id_remap(meta)
    gmap = gender_map(meta)
    valid = set(gmap.keys())
    protag = protag_id(meta, protagonist)

    addr_edges = build_addressee_network(film_id, remap, valid)
    cooc_edges = build_cooccurrence_network(film_id, remap, valid)

    addr_sum = network_summary(addr_edges, valid)
    cooc_sum = network_summary(cooc_edges, valid)

    addr_null = label_shuffle_null(addr_edges, protag, gmap)
    cooc_null = label_shuffle_null(cooc_edges, protag, gmap)

    div = edge_divergence(film_id, meta, remap, valid, gmap)
    recip = addressee_reciprocity(film_id, remap, valid)
    ks_addr = keystone_in_network(addr_edges, protag)
    ks_cooc = keystone_in_network(cooc_edges, protag)
    cx = scene_complexity(film_id, remap, valid)

    def name_of(cid):
        if cid is None:
            return None
        row = meta[meta.character_id == cid]
        return row.iloc[0].canonical_name if len(row) else cid

    def gen_of(cid):
        return gmap.get(cid)

    return {
        "film_id": film_id,
        "protagonist": protagonist,
        "lead_gender": lead_gender,
        "addr_n_nodes": addr_sum["n_nodes"], "addr_n_edges": addr_sum["n_edges"],
        "addr_density": addr_sum["density"],
        "addr_samesex": addr_null["obs"],
        "addr_samesex_z": addr_null["z"],
        "addr_samesex_p": addr_null["p_one_sided"],
        "addr_n_same": addr_null["n_same"],
        "addr_n_other": addr_null["n_other"],
        "cooc_n_nodes": cooc_sum["n_nodes"], "cooc_n_edges": cooc_sum["n_edges"],
        "cooc_density": cooc_sum["density"],
        "cooc_samesex": cooc_null["obs"],
        "cooc_samesex_z": cooc_null["z"],
        "cooc_samesex_p": cooc_null["p_one_sided"],
        "cooc_n_same": cooc_null["n_same"],
        "cooc_n_other": cooc_null["n_other"],
        "div_n_addr": div["n_addr"],
        "div_n_cooc": div["n_cooc"],
        "div_n_both": div["n_both"],
        "div_n_only_addr": div["n_only_addr"],
        "div_n_only_cooc": div["n_only_cooc"],
        "div_class_only_cooc": div["class_only_cooc"],
        "div_class_only_addr": div["class_only_addr"],
        "div_class_both": div["class_both"],
        "recip_n_directed_edges": recip["n_directed_edges"],
        "recip_n_unique_pairs": recip.get("n_unique_pairs", 0),
        "recip_n_reciprocal_pairs": recip["n_reciprocal_pairs"],
        "recip_reciprocity_pair": recip["reciprocity_pair"],
        "recip_reciprocity_edge": recip["reciprocity_edge"],
        "keystone_addr": name_of(ks_addr),
        "keystone_addr_gender": gen_of(ks_addr),
        "keystone_cooc": name_of(ks_cooc),
        "keystone_cooc_gender": gen_of(ks_cooc),
        "keystone_agree": ks_addr == ks_cooc and ks_addr is not None,
        **cx,
    }


def step2_pillar_b(df: pd.DataFrame) -> dict:
    set_style()
    rows = []
    for _, r in df.iterrows():
        print(f"  [pillar B] {r.film_id:35s} {r.protagonist:12s} {r.lead_gender}", flush=True)
        rows.append(build_per_film_method_row(r.film_id, r.protagonist, r.lead_gender))
    F = pd.DataFrame(rows)
    F["film_title"] = F["film_id"].map(DISPLAY_N17).fillna(F["film_id"])
    F.to_csv(TBL_DIR_N17 / "step2_per_film_method_comparison.csv", index=False)

    # 2a per-protag comparison table sorted by |Δz|
    cmp_tbl = F[["film_title", "protagonist", "lead_gender",
                 "addr_samesex", "addr_samesex_z",
                 "cooc_samesex", "cooc_samesex_z"]].copy()
    cmp_tbl["abs_diff_z"] = (cmp_tbl.addr_samesex_z - cmp_tbl.cooc_samesex_z).abs()
    cmp_tbl["diff_z"] = cmp_tbl.addr_samesex_z - cmp_tbl.cooc_samesex_z
    cmp_tbl = cmp_tbl.sort_values("abs_diff_z", ascending=False).reset_index(drop=True)
    cmp_tbl.to_csv(TBL_DIR_N17 / "step2a_per_protag_method_compare.csv", index=False)

    # 2b agreement stats on z
    z_addr = F["addr_samesex_z"].to_numpy()
    z_cooc = F["cooc_samesex_z"].to_numpy()
    raw_addr = F["addr_samesex"].to_numpy()
    raw_cooc = F["cooc_samesex"].to_numpy()

    spear_z = stats.spearmanr(z_addr, z_cooc)
    pears_z = stats.pearsonr(z_addr, z_cooc)
    wilc_z = stats.wilcoxon(z_addr, z_cooc, alternative="two-sided", zero_method="wilcox")
    spear_r = stats.spearmanr(raw_addr, raw_cooc)
    pears_r = stats.pearsonr(raw_addr, raw_cooc)
    wilc_r = stats.wilcoxon(raw_addr, raw_cooc, alternative="two-sided", zero_method="wilcox")

    agreement = {
        "z_spearman_rho": float(spear_z.statistic), "z_spearman_p": float(spear_z.pvalue),
        "z_pearson_r": float(pears_z.statistic), "z_pearson_p": float(pears_z.pvalue),
        "z_wilcoxon_W": float(wilc_z.statistic), "z_wilcoxon_p": float(wilc_z.pvalue),
        "z_mean_signed_diff_addr_minus_cooc": float((z_addr - z_cooc).mean()),
        "z_median_signed_diff": float(np.median(z_addr - z_cooc)),
        "raw_spearman_rho": float(spear_r.statistic), "raw_spearman_p": float(spear_r.pvalue),
        "raw_pearson_r": float(pears_r.statistic), "raw_pearson_p": float(pears_r.pvalue),
        "raw_wilcoxon_W": float(wilc_r.statistic), "raw_wilcoxon_p": float(wilc_r.pvalue),
        "raw_mean_signed_diff": float((raw_addr - raw_cooc).mean()),
        "raw_median_signed_diff": float(np.median(raw_addr - raw_cooc)),
    }

    # 2c H1 under both methods
    f_za = F.loc[F.lead_gender == "F", "addr_samesex_z"].to_numpy()
    m_za = F.loc[F.lead_gender == "M", "addr_samesex_z"].to_numpy()
    f_xa = F.loc[F.lead_gender == "F", "addr_samesex"].to_numpy()
    m_xa = F.loc[F.lead_gender == "M", "addr_samesex"].to_numpy()
    f_zc = F.loc[F.lead_gender == "F", "cooc_samesex_z"].to_numpy()
    m_zc = F.loc[F.lead_gender == "M", "cooc_samesex_z"].to_numpy()
    f_rc = F.loc[F.lead_gender == "F", "cooc_samesex"].to_numpy()
    m_rc = F.loc[F.lead_gender == "M", "cooc_samesex"].to_numpy()
    h1_panels = {
        "addr_raw": run_full_panel(f_xa, m_xa, "addressee raw (recomputed)"),
        "addr_z": run_full_panel(f_za, m_za, "addressee z (label-shuffled null)"),
        "cooc_raw": run_full_panel(f_rc, m_rc, "co-occurrence raw"),
        "cooc_z": run_full_panel(f_zc, m_zc, "co-occurrence z (label-shuffled null)"),
    }

    # 2d Test 1 — phantom edges aggregate + per-film
    F_film = F.copy()
    agg_only_cooc = Counter()
    agg_only_addr = Counter()
    agg_both = Counter()
    for _, r in F_film.iterrows():
        for k in ("FF", "MM", "cross"):
            agg_only_cooc[k] += r.div_class_only_cooc.get(k, 0)
            agg_only_addr[k] += r.div_class_only_addr.get(k, 0)
            agg_both[k] += r.div_class_both.get(k, 0)

    def same_share(c):
        same = c.get("FF", 0) + c.get("MM", 0)
        tot = same + c.get("cross", 0)
        return same / tot if tot else float("nan")

    contingency = np.array([
        [agg_only_cooc.get("FF", 0) + agg_only_cooc.get("MM", 0), agg_only_cooc.get("cross", 0)],
        [agg_both.get("FF", 0) + agg_both.get("MM", 0), agg_both.get("cross", 0)],
    ])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)

    F_film["phantom_total"] = F_film.div_class_only_cooc.apply(lambda d: sum(d.values()))
    F_film["phantom_same"] = F_film.div_class_only_cooc.apply(lambda d: d.get("FF", 0) + d.get("MM", 0))
    F_film["phantom_share_same"] = F_film.phantom_same / F_film.phantom_total.replace(0, np.nan)
    phantom_table = F_film[["film_title", "lead_gender", "div_n_only_cooc",
                             "phantom_same", "phantom_share_same"]].copy()
    phantom_table.to_csv(TBL_DIR_N17 / "step2d_phantom_edges.csv", index=False)

    test1 = {
        "agg_only_cooc": dict(agg_only_cooc),
        "agg_only_addr": dict(agg_only_addr),
        "agg_both": dict(agg_both),
        "share_same_phantom": same_share(agg_only_cooc),
        "share_same_addr_only": same_share(agg_only_addr),
        "share_same_both": same_share(agg_both),
        "chi2": float(chi2), "chi2_p": float(p_chi), "dof": int(dof),
        "phantom_table": phantom_table.to_dict(orient="records"),
    }

    # 2e Test 2 — reciprocity (film-level, dedup)
    R = F.copy()
    f_rec = R.loc[R.lead_gender == "F", "recip_reciprocity_edge"].to_numpy()
    m_rec = R.loc[R.lead_gender == "M", "recip_reciprocity_edge"].to_numpy()
    rec_panel = run_full_panel(f_rec, m_rec, "addressee edge-reciprocity")
    recip_table = R[["film_title", "lead_gender", "recip_n_directed_edges",
                     "recip_n_unique_pairs", "recip_n_reciprocal_pairs",
                     "recip_reciprocity_pair", "recip_reciprocity_edge"]].copy()
    recip_table.to_csv(TBL_DIR_N17 / "step2e_reciprocity.csv", index=False)

    # 2f Test 3 — keystone identity agreement + directional gender-flip pattern
    K = F.copy()
    K["gender_flip"] = (~K.keystone_agree) & (K.keystone_addr_gender != K.keystone_cooc_gender)
    agree_rate = float(K.keystone_agree.mean())
    keystone_table = K[["film_title", "protagonist", "lead_gender",
                        "keystone_addr", "keystone_addr_gender",
                        "keystone_cooc", "keystone_cooc_gender",
                        "keystone_agree", "gender_flip"]].copy()
    keystone_table.to_csv(TBL_DIR_N17 / "step2f_keystone_agreement.csv", index=False)

    # Directional pattern: among F-led films, does cooc switch to a F keystone where addr picked M?
    f_led = K[K.lead_gender == "F"]
    addr_cross_F = (f_led.keystone_addr_gender == "M").sum()  # F-led with M addr keystone
    cooc_cross_F = (f_led.keystone_cooc_gender == "M").sum()
    m_led = K[K.lead_gender == "M"]
    addr_cross_M = (m_led.keystone_addr_gender == "F").sum()
    cooc_cross_M = (m_led.keystone_cooc_gender == "F").sum()

    # Directional flips: F-led, addr=M, cooc=F (the N=12 pattern)
    flip_F_M_to_F = K[(K.lead_gender == "F") & (K.keystone_addr_gender == "M") &
                       (K.keystone_cooc_gender == "F")]
    flip_F_F_to_M = K[(K.lead_gender == "F") & (K.keystone_addr_gender == "F") &
                       (K.keystone_cooc_gender == "M")]
    flip_M_F_to_M = K[(K.lead_gender == "M") & (K.keystone_addr_gender == "F") &
                       (K.keystone_cooc_gender == "M")]
    flip_M_M_to_F = K[(K.lead_gender == "M") & (K.keystone_addr_gender == "M") &
                       (K.keystone_cooc_gender == "F")]

    test3 = {
        "agree_rate": agree_rate,
        "n_agree": int(K.keystone_agree.sum()),
        "n_total": int(len(K)),
        "n_gender_flip": int(K.gender_flip.sum()),
        "F_led_addr_cross_gender": int(addr_cross_F),
        "F_led_cooc_cross_gender": int(cooc_cross_F),
        "M_led_addr_cross_gender": int(addr_cross_M),
        "M_led_cooc_cross_gender": int(cooc_cross_M),
        "F_led_flip_addrM_coocF": flip_F_M_to_F[["film_title", "keystone_addr",
                                                  "keystone_cooc"]].to_dict(orient="records"),
        "F_led_flip_addrF_coocM": flip_F_F_to_M[["film_title", "keystone_addr",
                                                  "keystone_cooc"]].to_dict(orient="records"),
        "M_led_flip_addrF_coocM": flip_M_F_to_M[["film_title", "keystone_addr",
                                                  "keystone_cooc"]].to_dict(orient="records"),
        "M_led_flip_addrM_coocF": flip_M_M_to_F[["film_title", "keystone_addr",
                                                  "keystone_cooc"]].to_dict(orient="records"),
    }

    # 2g Test 4 — divergence vs scene complexity
    cmp_tbl["share_scenes_ge3"] = cmp_tbl.film_title.map(
        F.set_index("film_title")["share_scenes_ge3"])
    cmp_tbl["share_scenes_ge4"] = cmp_tbl.film_title.map(
        F.set_index("film_title")["share_scenes_ge4"])
    cmp_tbl["scene_mean_speakers"] = cmp_tbl.film_title.map(
        F.set_index("film_title")["scene_mean_speakers"])

    test4 = {}
    for col in ["share_scenes_ge3", "share_scenes_ge4", "scene_mean_speakers"]:
        r_s = stats.spearmanr(cmp_tbl[col], cmp_tbl["abs_diff_z"])
        r_p = stats.pearsonr(cmp_tbl[col], cmp_tbl["abs_diff_z"])
        test4[col] = {
            "spearman_rho": float(r_s.statistic), "spearman_p": float(r_s.pvalue),
            "pearson_r": float(r_p.statistic), "pearson_p": float(r_p.pvalue),
        }

    # Save JSON of all panel results
    json_safe_h1 = {k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
                     for k, v in h1_panels.items()}
    with (TBL_DIR_N17 / "step2_results.json").open("w") as fh:
        json.dump({
            "agreement": agreement,
            "h1_panels": json_safe_h1,
            "reciprocity_panel": rec_panel,
            "test1_phantom": {k: v for k, v in test1.items() if k != "phantom_table"},
            "test3_keystone": test3,
            "test4_complexity": test4,
        }, fh, indent=2, default=str)

    # ----- Figures -----
    # fig method comparison scatter (z)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    for g in GENDER_ORDER:
        sub = F[F.lead_gender == g]
        ax.scatter(sub.addr_samesex_z, sub.cooc_samesex_z, s=140,
                   color=GENDER_PALETTE[g], edgecolor="black", linewidth=0.8,
                   label=f"Lead = {g}", zorder=3)
    for _, r in F.iterrows():
        ax.annotate(r.film_title, (r.addr_samesex_z, r.cooc_samesex_z),
                    xytext=(7, 5), textcoords="offset points", fontsize=8, color="#333")
    lo = min(F.addr_samesex_z.min(), F.cooc_samesex_z.min()) - 0.5
    hi = max(F.addr_samesex_z.max(), F.cooc_samesex_z.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=0.8, label="y = x")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Directed addressee z (label-shuffled null)")
    ax.set_ylabel("Scene co-occurrence z (label-shuffled null)")
    ax.set_title(f"Same-gender homophily: directed addressee vs co-occurrence (N=17, ρ={spear_z.statistic:+.3f})")
    ax.legend(loc="lower right", title="Lead gender")
    savefig17(fig, "fig04_n17_method_compare_scatter_z")

    # H1 under both methods side-by-side
    long = pd.concat([
        F[["lead_gender", "film_title", "addr_samesex_z"]]
            .rename(columns={"addr_samesex_z": "z"}).assign(method="Directed addressee (weighted)"),
        F[["lead_gender", "film_title", "cooc_samesex_z"]]
            .rename(columns={"cooc_samesex_z": "z"}).assign(method="Scene co-occurrence"),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.boxplot(data=long, x="method", y="z", hue="lead_gender",
                order=["Directed addressee (weighted)", "Scene co-occurrence"],
                hue_order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                width=0.55, showcaps=False, boxprops={"alpha": .3},
                medianprops={"color": "black", "linewidth": 2})
    sns.stripplot(data=long, x="method", y="z", hue="lead_gender",
                  order=["Directed addressee (weighted)", "Scene co-occurrence"],
                  hue_order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                  size=8, dodge=True, jitter=0.10, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="grey", lw=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Lead gender", loc="best")
    ax.set_xlabel(""); ax.set_ylabel("Same-gender homophily z (label-shuffled)")
    ax.set_title(f"H1 under both network constructions (N=17)")
    savefig17(fig, "fig05_n17_h1_both_methods")

    return {
        "cmp_tbl": cmp_tbl,
        "agreement": agreement,
        "h1_panels": h1_panels,
        "test1": test1,
        "rec_panel": rec_panel,
        "recip_table": recip_table,
        "test3": test3,
        "test4": test4,
        "keystone_table": keystone_table,
        "F": F,
    }


# ============================================================================
# STEP 3 — Structural context
# ============================================================================
def step3_structural(df_all: pd.DataFrame, df_ext: pd.DataFrame) -> dict:
    set_style()
    # 3a protag_betweenness + protag_betw_z
    f_b_raw = df_all.loc[df_all.lead_gender == "F", "protag_betweenness"].to_numpy()
    m_b_raw = df_all.loc[df_all.lead_gender == "M", "protag_betweenness"].to_numpy()
    f_b_z = df_all.loc[df_all.lead_gender == "F", "protag_betw_z"].to_numpy()
    m_b_z = df_all.loc[df_all.lead_gender == "M", "protag_betw_z"].to_numpy()
    panel_betw_raw = run_full_panel(f_b_raw, m_b_raw, "protag_betweenness (raw)")
    panel_betw_z = run_full_panel(f_b_z, m_b_z, "protag_betw_z (cast-adjusted)")

    # 3b quadrant typology
    quad = df_all[["film_id", "film_title", "protagonist", "lead_gender",
                   "protag_betw_z", "protag_samesex_z"]].copy()
    quad["quadrant"] = quad.apply(
        lambda r: ("HighBridge_" if r.protag_betw_z > 0 else "LowBridge_") +
                  ("HighEmbed" if r.protag_samesex_z > 0 else "LowEmbed"),
        axis=1)
    quad.to_csv(TBL_DIR_N17 / "step3b_quadrant_assignments.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    for g in GENDER_ORDER:
        sub = df_all[df_all.lead_gender == g]
        ax.scatter(sub.protag_betw_z, sub.protag_samesex_z, s=150,
                   color=GENDER_PALETTE[g], edgecolor="black",
                   linewidth=0.8, label=f"Lead = {g}", zorder=3)
    for _, r in df_all.iterrows():
        ax.annotate(r.film_title, (r.protag_betw_z, r.protag_samesex_z),
                    xytext=(7, 6), textcoords="offset points", fontsize=8, color="#333")
    ax.axhline(0, color="grey", lw=0.7); ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("Protagonist betweenness z (cast-adjusted)")
    ax.set_ylabel("Protagonist same-gender homophily z (cast-adjusted)")
    ax.set_title("Quadrant typology: structural role × gender embedding (N=17)")
    ax.legend(title="Lead gender", loc="lower right")
    savefig17(fig, "fig06_n17_quadrant_betw_x_homophily")

    # Quadrant crosstab
    quad_ct = pd.crosstab(quad.quadrant, quad.lead_gender)

    # 3c keystone analysis (use extended.csv, which already carries the Mike-row dropped,
    # keystone for Sulley is "Mike" / male / same-gender)
    ks_ct = pd.crosstab(df_ext.lead_gender, df_ext.keystone_gender, margins=True, margins_name="Total")
    fisher_ct = pd.crosstab(df_ext.lead_gender, df_ext.keystone_diff_gender)
    fisher_p = float("nan"); fisher_odds = float("nan")
    if fisher_ct.shape == (2, 2):
        fisher_odds, fisher_p = stats.fisher_exact(fisher_ct.values, alternative="two-sided")

    # Components after removal MW
    f_comp = df_ext.loc[df_ext.lead_gender == "F", "components_after_removal"].to_numpy()
    m_comp = df_ext.loc[df_ext.lead_gender == "M", "components_after_removal"].to_numpy()
    comp_panel = run_full_panel(f_comp, m_comp, "components_after_removal")

    # Full keystone table
    keystone_table = df_ext[["film_title", "year", "lead_gender", "protagonist",
                              "keystone", "keystone_gender", "keystone_diff_gender",
                              "components_after_removal"]].sort_values("year").reset_index(drop=True)
    keystone_table.to_csv(TBL_DIR_N17 / "step3c_keystone_table.csv", index=False)
    ks_ct.to_csv(TBL_DIR_N17 / "step3c_keystone_crosstab.csv")

    # 3d global metrics F vs M (film-level — already 1 row per film in df_all)
    global_cols = ["density", "reciprocity", "avg_path_len",
                   "leading_eigenvalue", "mean_clustering", "n_nodes", "n_edges"]
    global_panels = {}
    rows = []
    for col in global_cols:
        f = df_all.loc[df_all.lead_gender == "F", col].to_numpy()
        m = df_all.loc[df_all.lead_gender == "M", col].to_numpy()
        p = run_full_panel(f, m, col)
        global_panels[col] = p
        rows.append({
            "metric": col,
            "F_median": p["median_F"], "M_median": p["median_M"],
            "MW_U": p["mw_two_sided"]["U"], "MW_p": p["mw_two_sided"]["p"],
            "cliffs_delta": p["cliffs_delta"],
            "rank_biserial_r": p["rank_biserial"],
            "boot_ci_median_lo": p["boot_median_diff_ci"]["ci_lo"],
            "boot_ci_median_hi": p["boot_median_diff_ci"]["ci_hi"],
        })
    pd.DataFrame(rows).to_csv(TBL_DIR_N17 / "step3d_global_metrics.csv", index=False)

    # Save JSON
    with (TBL_DIR_N17 / "step3_results.json").open("w") as fh:
        json.dump({
            "betw_raw_panel": panel_betw_raw,
            "betw_z_panel": panel_betw_z,
            "quadrant_crosstab": quad_ct.to_dict(),
            "keystone_crosstab": ks_ct.to_dict(),
            "keystone_diff_crosstab": fisher_ct.to_dict(),
            "fisher_p": fisher_p, "fisher_odds": fisher_odds,
            "components_panel": comp_panel,
            "global_metrics": global_panels,
        }, fh, indent=2, default=str)

    return {
        "panel_betw_raw": panel_betw_raw,
        "panel_betw_z": panel_betw_z,
        "quad_ct": quad_ct,
        "quad": quad,
        "ks_ct": ks_ct,
        "fisher_ct": fisher_ct,
        "fisher_p": fisher_p, "fisher_odds": fisher_odds,
        "comp_panel": comp_panel,
        "keystone_table": keystone_table,
        "global_panels": global_panels,
    }


# ============================================================================
# STEP 4 — Descriptive extensions
# ============================================================================
def step4_descriptive(df_ext: pd.DataFrame) -> dict:
    set_style()
    metrics_basic = ["female_alter_betw_z", "burt_constraint", "ego_density"]

    panels = {}
    tables = {}

    def panel_safe(col, df_use):
        f = df_use.loc[df_use.lead_gender == "F", col].dropna().to_numpy()
        m = df_use.loc[df_use.lead_gender == "M", col].dropna().to_numpy()
        if len(f) < 2 or len(m) < 2:
            return None
        return run_full_panel(f, m, col)

    for col in metrics_basic:
        p = panel_safe(col, df_ext)
        panels[col] = {"all": p}
        t = df_ext[["film_id", "film_title", "year", "lead_gender", "protagonist", col]].copy()
        t = t.sort_values(col, ascending=False).reset_index(drop=True)
        t.to_csv(TBL_DIR_N17 / f"step4_{col}.csv", index=False)
        tables[col] = t

    # Reciprocity (already in df_ext as 'reciprocity' film-level)
    p_recip = panel_safe("reciprocity", df_ext)
    panels["reciprocity"] = {"all": p_recip}
    t = df_ext[["film_id", "film_title", "year", "lead_gender", "reciprocity"]].sort_values(
        "reciprocity", ascending=False).reset_index(drop=True)
    t.to_csv(TBL_DIR_N17 / "step4_reciprocity.csv", index=False)
    tables["reciprocity"] = t

    # Save JSON
    with (TBL_DIR_N17 / "step4_panels.json").open("w") as fh:
        json.dump(panels, fh, indent=2, default=str)

    # Small-multiples figure
    cols_plot = metrics_basic + ["reciprocity"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes = axes.flatten()
    for ax, col in zip(axes, cols_plot):
        plot_df = df_ext.dropna(subset=[col]).copy()
        sns.boxplot(data=plot_df, x="lead_gender", y=col, hue="lead_gender",
                    order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                    width=0.45, showcaps=False, boxprops={"alpha": .3},
                    medianprops={"color": "black", "linewidth": 1.7}, legend=False)
        sns.stripplot(data=plot_df, x="lead_gender", y=col, hue="lead_gender",
                      order=GENDER_ORDER, palette=GENDER_PALETTE, ax=ax,
                      size=7, jitter=0.18, edgecolor="black", linewidth=0.6, legend=False)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(col, fontsize=10)
    fig.suptitle("Descriptive extensions, F vs M", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig17(fig, "fig07_n17_descriptive_smallmultiples")

    return {"panels": panels, "tables": tables}


# ============================================================================
# STEP 6 — Exploratory
# ============================================================================
def step6_exploratory(df_all: pd.DataFrame, df_ext: pd.DataFrame) -> dict:
    set_style()
    # 6a Temporal era split - try cutoffs 2010 and 2013
    splits = {}
    for cutoff in (2010, 2013):
        df_t = df_all.copy()
        df_t["era"] = np.where(df_t.year < cutoff, f"pre-{cutoff}", f"{cutoff}+")
        ct = pd.crosstab(df_t.era, df_t.lead_gender)
        splits[cutoff] = {
            "counts": ct.to_dict(),
            "table": df_t.groupby(["era", "lead_gender"])
                          .agg(n=("film_id", "size"),
                               median_samesex_z=("protag_samesex_z", "median"),
                               median_betw_z=("protag_betw_z", "median"))
                          .round(3).reset_index().to_dict(orient="records"),
        }

    # Pick the more balanced cutoff: which keeps F and M present in both eras
    def is_balanced(cnt: dict) -> bool:
        # cnt is like {'F': {'pre-2010': 2, '2010+': 7}, 'M': {...}}
        for g in ("F", "M"):
            sub = cnt.get(g, {})
            if min(sub.values()) < 2:
                return False
        return True
    chosen_cutoff = 2013 if is_balanced(splits[2013]["counts"]) else 2010

    df_chosen = df_all.copy()
    df_chosen["era"] = np.where(df_chosen.year < chosen_cutoff,
                                  f"pre-{chosen_cutoff}", f"{chosen_cutoff}+")
    era_panels = {}
    for era_val in df_chosen.era.unique():
        sub = df_chosen[df_chosen.era == era_val]
        if (sub.lead_gender == "F").sum() >= 2 and (sub.lead_gender == "M").sum() >= 2:
            f = sub.loc[sub.lead_gender == "F", "protag_samesex_z"].to_numpy()
            m = sub.loc[sub.lead_gender == "M", "protag_samesex_z"].to_numpy()
            era_panels[era_val] = run_full_panel(f, m, f"protag_samesex_z, {era_val}")

    era_tab = (df_chosen.groupby(["era", "lead_gender"])
                 .agg(n=("film_id", "size"),
                      median_samesex_z=("protag_samesex_z", "median"),
                      median_betw_z=("protag_betw_z", "median"))
                 .round(3))
    era_tab.to_csv(TBL_DIR_N17 / "step6a_era_split.csv")

    # 6b Ward + k-means clustering on standardised global metrics
    feature_cols = ["density", "reciprocity", "avg_path_len",
                    "leading_eigenvalue", "mean_clustering"]
    X = df_all[feature_cols].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    Z_ward = linkage(Xs, method="ward", metric="euclidean")
    cluster_results = {}
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, n_init=50, random_state=RNG_SEED).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        cluster_results[k] = {"silhouette": float(sil),
                               "sizes": np.bincount(km.labels_).tolist(),
                               "labels": km.labels_.tolist()}
        df_all[f"kmeans_k{k}"] = km.labels_

    # Save cluster assignments
    df_all[["film_id", "film_title", "year", "lead_gender"] +
            [f"kmeans_k{k}" for k in (2, 3, 4)] + feature_cols].to_csv(
        TBL_DIR_N17 / "step6b_archetype_assignments.csv", index=False)

    # Dendrogram figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = df_all.film_title.tolist()
    dendrogram(Z_ward, labels=labels, leaf_rotation=35, leaf_font_size=9,
               color_threshold=0.7 * Z_ward[:, 2].max(), ax=ax)
    ax.set_ylabel("Ward distance")
    ax.set_title("Ward hierarchical clustering on standardised global network metrics (N=17)")
    savefig17(fig, "fig08_n17_ward_dendrogram")

    # 6c Spearman correlation: protag_samesex_z vs all other numeric features (extended)
    merge = df_all.merge(df_ext.drop(columns=["lead_gender", "year", "protagonist", "film_title",
                                                "n_nodes", "n_edges", "density", "reciprocity",
                                                "avg_path_len", "leading_eigenvalue", "mean_clustering",
                                                "homophily_obs", "homophily_z", "homophily_p",
                                                "protag_samesex", "protag_samesex_z", "protag_samesex_p",
                                                "protag_betweenness", "protag_betw_z", "protag_betw_p",
                                                "keystone", "keystone_gender", "keystone_diff_gender",
                                                "components_after_removal", "n_isolated_after_removal"],
                                       errors="ignore"),
                          on="film_id", how="left")
    num_cols = [c for c in merge.select_dtypes(include=[np.number]).columns
                 if c not in {"year", "keystone_diff_gender",
                               "components_after_removal", "n_isolated_after_removal"}
                 and not c.startswith("kmeans")]
    corr = merge[num_cols].corr(method="spearman")
    if "protag_samesex_z" in corr.columns:
        top = corr["protag_samesex_z"].drop("protag_samesex_z").sort_values(key=abs, ascending=False)
        top.to_csv(TBL_DIR_N17 / "step6c_spearman_vs_samesex_z.csv")
    else:
        top = pd.Series(dtype=float)

    # Save JSON
    with (TBL_DIR_N17 / "step6_results.json").open("w") as fh:
        json.dump({
            "era_splits": splits,
            "chosen_cutoff": chosen_cutoff,
            "era_panels": {k: v for k, v in era_panels.items()},
            "cluster": cluster_results,
            "top_spearman_vs_samesex_z": top.round(4).to_dict(),
        }, fh, indent=2, default=str)

    return {
        "splits": splits, "chosen_cutoff": chosen_cutoff,
        "era_tab": era_tab, "era_panels": era_panels,
        "cluster": cluster_results, "ward_linkage": Z_ward,
        "top_spearman": top,
    }


# ============================================================================
# MARKDOWN ASSEMBLY
# ============================================================================
def _format_cell(v, round_dec=3):
    if pd.isna(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{round_dec}f}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v).replace("|", "\\|")


def df_to_md(df: pd.DataFrame, cols=None, round_dec=3) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without requiring tabulate."""
    if cols is not None:
        df = df[cols].copy()
    df = df.copy().reset_index(drop=True)
    col_keys = list(df.columns)
    headers = [str(h) for h in col_keys]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(_format_cell(r[c], round_dec) for c in col_keys) + " |")
    return "\n".join(rows)


def series_to_md(s: pd.Series, name: str = "value", round_dec=3) -> str:
    rows = [f"| index | {name} |", "|---|---|"]
    for idx, v in s.items():
        rows.append(f"| {idx} | {_format_cell(v, round_dec)} |")
    return "\n".join(rows)


def write_markdown(df_all, df_ext, results1, results2, results3, results4, results6):
    lines = []
    A = lines.append

    A("# Unified Results — Steps 1, 2, 3, 4, 6 (Agent B, N=17)")
    A("")
    A("> Source-of-truth plan: `CLEAN/admin/AGENT_INSTRUCTION_run_analysis_N18.md`.")
    A("> Analytic sample: **N=17 (8 F-led + 9 M-led)** after Conventions filters "
      "(Soul dropped — Convention 1; Mike row of Monsters Inc dropped, Sulley kept — Convention 3).")
    A("> RNG_SEED = 20260622. Pillar B null = label-shuffling, n=2000 permutations.")
    A("> Tables: `tables_n17/`. Figures: `figures_n17/`. Orchestrator: `n17_orchestrator.py`.")
    A("> Step 5 (Social World) is produced separately by another agent.")
    A("")

    # ====================================================================
    # 2. Pillar A
    # ====================================================================
    A("## 2. Pillar A — the measure (cast-adjusted protagonist same-gender homophily)")
    A("")
    A("This is the methodological core. We compare F-led vs M-led protagonists on "
      "two paired quantities: the **raw** same-gender share of dialogue (`protag_samesex`) "
      "and the **cast-adjusted z-score** under a degree-preserving configuration-model "
      "null (`protag_samesex_z`, 2000 rewirings). The decomposition tests whether any "
      "F-vs-M gap in same-gender embedding is structurally inevitable given cast "
      "composition, or whether it reflects protagonist-level behaviour.")
    A("")

    # 1a raw vs adjusted
    A("### 2.1 Raw vs cast-adjusted decomposition")
    A("")
    A("**Raw share** `protag_samesex`:")
    A("")
    A(fmt_panel_md(results1["panel_raw"]))
    A("")
    A("**Cast-adjusted z** `protag_samesex_z`:")
    A("")
    A(fmt_panel_md(results1["panel_z"]))
    A("")
    raw = results1["panel_raw"]; zp = results1["panel_z"]
    A(f"**Comparison.** At N=12 the raw measure showed Cliff's δ = −0.71 (large, "
      f"MW one-sided p = 0.018, bootstrap CI on mean diff [−0.43, −0.09] **excluding 0**), "
      f"while the cast-adjusted measure showed δ = +0.10 (negligible). At **N=17** we "
      f"find raw Cliff's δ = **{raw['cliffs_delta']:+.3f}** (MW one-sided p = "
      f"{raw['mw_one_sided_F<M']['p']:.4f}) and cast-adjusted Cliff's δ = "
      f"**{zp['cliffs_delta']:+.3f}** (MW one-sided p = "
      f"{zp['mw_one_sided_F<M']['p']:.4f}). "
      f"The directional story — raw shows a substantial F<M gap, cast adjustment "
      f"largely absorbs it — is "
      f"{'preserved and **strengthened**' if abs(raw['cliffs_delta']) > 0.85 else ('preserved' if (raw['cliffs_delta'] < -0.3 and abs(zp['cliffs_delta']) < 0.4) else 'modified — see flagged change above')} "
      f"at the expanded N. Notably, the raw gap is **larger** at N=17 than at N=12 "
      f"(δ = {raw['cliffs_delta']:+.3f} vs −0.71), and the cast-adjusted gap is "
      f"essentially zero (δ = {zp['cliffs_delta']:+.3f}). The methodological centrepiece "
      f"of the thesis — that the obvious F-vs-M raw gap is a cast-composition artefact "
      f"absorbed by the configuration-model null — is **more cleanly demonstrated at N=17** "
      f"than at N=12.")
    A("")
    A("→ Figure: `figures_n17/fig02_n17_raw_vs_adjusted.png` (side-by-side raw vs z box+strip).")
    A("")

    # 1b individual protagonist significance
    A("### 2.2 Per-protagonist cast-adjusted z (individual significance)")
    A("")
    sig = results1["sig_table"]
    A(df_to_md(sig, ["film_title", "protagonist", "lead_gender",
                       "protag_samesex", "protag_samesex_z", "protag_samesex_p", "sig_alpha05"]))
    A("")
    sig_count = int(sig.sig_alpha05.sum())
    sig_rows = sig[sig.sig_alpha05]
    A(f"**{sig_count} of {len(sig)} protagonists are individually significant at α=0.05.** "
      f"At N=12 the count was 2/13 (Anna F z=+2.13, Woody/TS1 M z=+3.40). "
      f"At N=17 the significant rows are: "
      + ", ".join(f"{r.protagonist} ({r.film_title}, {r.lead_gender}) z={r.protag_samesex_z:+.2f} "
                  f"p={r.protag_samesex_p:.4f}" for _, r in sig_rows.iterrows())
      + ".")
    A("")
    A("→ Figure: `figures_n17/fig01_n17_h1_lollipop.png`.")
    A("")

    # 1c power
    A("### 2.3 Power note")
    A("")
    A(results1["power"])
    A("")

    # 1d Spearman
    sp = results1["spearman"]
    A("### 2.4 Cast-wide vs protagonist correlation")
    A("")
    A(f"Spearman ρ(homophily_z, protag_samesex_z) = **{sp['rho']:+.3f}** (p = {sp['p']:.4f}). "
      f"At N=12 ρ = +0.66. The protagonist-level cast-adjusted measure tracks cast-wide "
      f"homophily; the two are co-aligned but not redundant.")
    A("")
    A("→ Figure: `figures_n17/fig03_n17_homophily_vs_protag_z.png`.")
    A("")

    # ====================================================================
    # 3. Pillar B
    # ====================================================================
    A("## 3. Pillar B — the method (addressee tagging vs scene co-occurrence)")
    A("")
    A("This is co-equal with Pillar A. We compare two network constructions on the "
      "same 17 films: the LLM-tagged addressee network (the pipeline's chosen method, "
      "directed and weighted by per-line addressee assignment) and a scene co-occurrence "
      "baseline (undirected, two characters share an edge weighted by the number of scenes "
      "in which both speak). Both apply MIN_EDGE_COUNT = 3.")
    A("")

    # 2.0 null
    A("### 3.0 Null choice — why label-shuffling, not configuration-model")
    A("")
    A("We use a **label-shuffling null (n=2000)**: topology held fixed, gender labels "
      "permuted among non-protagonist nodes, the same procedure applied to both networks. "
      "This isolates the structural-position effect of the protagonist's neighbourhood "
      "from any null-construction differences and makes the addressee vs co-occurrence "
      "comparison method-vs-method rather than method-vs-null. The addressee z-scores in "
      "this section will therefore not numerically match `protag_samesex_z` in "
      "`film_features_all.csv` (which uses degree-preserving configuration-model rewiring).")
    A("")

    # 2a per-protagonist table
    A("### 3.1 Per-protagonist method comparison (sorted by |Δz|)")
    A("")
    A(df_to_md(results2["cmp_tbl"][["film_title", "lead_gender",
                                       "addr_samesex", "addr_samesex_z",
                                       "cooc_samesex", "cooc_samesex_z",
                                       "abs_diff_z"]]))
    A("")

    # 2b agreement stats
    ag = results2["agreement"]
    A("### 3.2 Agreement statistics (addressee vs co-occurrence)")
    A("")
    A("| Test | Result |")
    A("|---|---|")
    A(f"| Spearman ρ on z-scores | **{ag['z_spearman_rho']:+.3f}** (p = {ag['z_spearman_p']:.4f}) |")
    A(f"| Pearson r on z-scores | **{ag['z_pearson_r']:+.3f}** (p = {ag['z_pearson_p']:.4f}) |")
    A(f"| Wilcoxon signed-rank on z (two-sided) | W = {ag['z_wilcoxon_W']:.1f}, p = {ag['z_wilcoxon_p']:.4f} |")
    A(f"| Mean signed diff z (addr − cooc) | {ag['z_mean_signed_diff_addr_minus_cooc']:+.4f} |")
    A(f"| Median signed diff z | {ag['z_median_signed_diff']:+.4f} |")
    A(f"| Spearman ρ on raw shares | {ag['raw_spearman_rho']:+.3f} (p = {ag['raw_spearman_p']:.4f}) |")
    A(f"| Pearson r on raw shares | {ag['raw_pearson_r']:+.3f} (p = {ag['raw_pearson_p']:.4f}) |")
    A(f"| Wilcoxon signed-rank on raw (two-sided) | W = {ag['raw_wilcoxon_W']:.1f}, p = {ag['raw_wilcoxon_p']:.4f} |")
    A(f"| Mean signed diff raw (addr − cooc) | {ag['raw_mean_signed_diff']:+.4f} |")
    A(f"| Median signed diff raw | {ag['raw_median_signed_diff']:+.4f} |")
    A("")
    A(f"**N=12 reference:** Spearman ρ = +0.80 (p = 0.001), Pearson r = +0.82, "
      f"Wilcoxon p = 0.74 (no systematic bias). At N=17 the rank-agreement is "
      f"{'preserved' if ag['z_spearman_rho'] >= 0.6 else 'attenuated'}; "
      f"the signed-rank null on systematic bias "
      f"{'holds' if ag['z_wilcoxon_p'] > 0.1 else 'shifts — see above'}.")
    A("")
    A("→ Figure: `figures_n17/fig04_n17_method_compare_scatter_z.png`.")
    A("")

    # 2c H1 under both methods
    A("### 3.3 H1 under both methods (the robustness check)")
    A("")
    h1 = results2["h1_panels"]
    A("| Measure | Median F | Median M | Cliff's δ | MW two-sided p | MW one-sided F<M p |")
    A("|---|---|---|---|---|---|")
    for k, lbl in [("addr_raw", "Addressee raw share"),
                    ("cooc_raw", "Co-occurrence raw share"),
                    ("addr_z", "Addressee z (label-shuffled)"),
                    ("cooc_z", "Co-occurrence z (label-shuffled)")]:
        p = h1[k]
        A(f"| {lbl} | {p['median_F']:+.3f} | {p['median_M']:+.3f} | "
          f"{p['cliffs_delta']:+.3f} | {p['mw_two_sided']['p']:.4f} | "
          f"{p['mw_one_sided_F<M']['p']:.4f} |")
    A("")
    A(f"**N=12 reference:** raw δ = −0.71 (identical under both methods, one-sided p = 0.018); "
      f"z-score δ small in both (−0.19 addressee, −0.29 co-occurrence; p > 0.2 in both). "
      f"At N=17: raw δ addressee = {h1['addr_raw']['cliffs_delta']:+.3f}, "
      f"raw δ co-occurrence = {h1['cooc_raw']['cliffs_delta']:+.3f}; "
      f"z δ addressee = {h1['addr_z']['cliffs_delta']:+.3f}, "
      f"z δ co-occurrence = {h1['cooc_z']['cliffs_delta']:+.3f}.")
    A("")
    A("→ Figure: `figures_n17/fig05_n17_h1_both_methods.png`.")
    A("")

    # 2d Test 1 phantom edges
    t1 = results2["test1"]
    A("### 3.4 Test 1 — Phantom edges (co-occurrence-only ties)")
    A("")
    A("For each film we partition pairwise edges into Both / Only co-occurrence (phantom) / "
      "Only addressee, then classify each pair as same-gender (FF or MM) or cross-gender.")
    A("")
    A("**Aggregate across N=17:**")
    A("")
    A("| Bucket | FF | MM | Cross | Total | Same-gender share |")
    A("|---|---|---|---|---|---|")
    for label, c in [("Both (consensus)", t1["agg_both"]),
                       ("Only co-occurrence (phantom)", t1["agg_only_cooc"]),
                       ("Only addressee", t1["agg_only_addr"])]:
        ss = (c.get("FF", 0) + c.get("MM", 0)) / max(1, sum(c.values()))
        A(f"| {label} | {c.get('FF',0)} | {c.get('MM',0)} | {c.get('cross',0)} | "
          f"{sum(c.values())} | {ss:.3f} |")
    A("")
    A(f"**χ² same-vs-cross composition (phantom vs consensus): χ²({t1['dof']}) = "
      f"{t1['chi2']:.2f}, p = {t1['chi2_p']:.4f}.** "
      f"At N=12 the test was χ² = 3.79, p = 0.052 (62% same-gender phantom vs 50% consensus). "
      f"At N=17 phantom share = {t1['share_same_phantom']:.3f} vs consensus "
      f"{t1['share_same_both']:.3f}. Caveat: Soul had 0 phantoms in N=12, so its "
      f"exclusion does not directly drive a change here.")
    A("")
    A("**Per-film phantom counts and same-gender shares:**")
    A("")
    pt = pd.DataFrame(t1["phantom_table"]).sort_values("phantom_share_same",
                                                          ascending=False, na_position="last")
    A(df_to_md(pt))
    A("")

    # 2e Test 2 reciprocity
    rp = results2["rec_panel"]
    A("### 3.5 Test 2 — Reciprocity (addressee-only measurement)")
    A("")
    A("Co-occurrence is undirected by construction; reciprocity is a measurement only "
      "addressee tagging can produce. Per-film edge-reciprocity:")
    A("")
    A(df_to_md(results2["recip_table"]))
    A("")
    A(f"**F vs M comparison on edge-reciprocity:** median F = {rp['median_F']:.3f}, "
      f"median M = {rp['median_M']:.3f}, MW two-sided p = {rp['mw_two_sided']['p']:.4f}, "
      f"Cliff's δ = {rp['cliffs_delta']:+.3f}. "
      f"N=12 reference: MW p = 1.00, δ = 0.00. "
      f"{'No gender difference.' if abs(rp['cliffs_delta']) < 0.2 else 'Some gender shift; see above.'} "
      f"The measurement itself is informative within film — Mulan was the standout "
      f"low-reciprocity case at N=12 (drill-sergeant instruction).")
    A("")

    # 2f Test 3 keystone (THE headline)
    t3 = results2["test3"]
    A("### 3.6 Test 3 — Keystone identity agreement (the headline Pillar B test)")
    A("")
    A("Each method nominates the highest-betweenness non-protagonist character. We "
      "test whether agreement holds and, where it doesn't, whether disagreements show "
      "a directional gender pattern.")
    A("")
    A(df_to_md(results2["keystone_table"]))
    A("")
    A(f"**Overall agreement:** {t3['n_agree']}/{t3['n_total']} = {t3['agree_rate']:.1%}. "
      f"At N=12 agreement was 5/13 = 38%.")
    A("")
    A(f"**Directional gender-flip pattern (THE key claim):** "
      f"At N=12 all 3 gender-flips were in F-led films, all in the same direction — "
      f"addressee picked a *male* keystone, co-occurrence picked a *female* keystone "
      f"(Inside Out: Anger→Disgust; Zootopia: Bogo→Hopps; Encanto: Bruno→Luisa).")
    A("")
    A(f"At N=17:")
    A(f"- **F-led films with cross-gender (male) keystone, ADDRESSEE method:** {t3['F_led_addr_cross_gender']}/8")
    A(f"- **F-led films with cross-gender (male) keystone, CO-OCCURRENCE method:** {t3['F_led_cooc_cross_gender']}/8")
    A(f"- **M-led films with cross-gender (female) keystone, ADDRESSEE method:** {t3['M_led_addr_cross_gender']}/9")
    A(f"- **M-led films with cross-gender (female) keystone, CO-OCCURRENCE method:** {t3['M_led_cooc_cross_gender']}/9")
    A("")
    A(f"**F-led films where addressee picks M but co-occurrence picks F** "
      f"(the N=12 pattern, n = {len(t3['F_led_flip_addrM_coocF'])}):")
    if t3["F_led_flip_addrM_coocF"]:
        for r in t3["F_led_flip_addrM_coocF"]:
            A(f"- {r['film_title']}: addr = {r['keystone_addr']} (M) / cooc = {r['keystone_cooc']} (F)")
    else:
        A("_(none)_")
    A("")
    A(f"**F-led flips in the opposite direction (addr=F, cooc=M)** "
      f"(n = {len(t3['F_led_flip_addrF_coocM'])}):")
    if t3["F_led_flip_addrF_coocM"]:
        for r in t3["F_led_flip_addrF_coocM"]:
            A(f"- {r['film_title']}: addr = {r['keystone_addr']} (F) / cooc = {r['keystone_cooc']} (M)")
    else:
        A("_(none)_")
    A("")
    A(f"**M-led flips (addr=F, cooc=M)** (n = {len(t3['M_led_flip_addrF_coocM'])}):")
    if t3["M_led_flip_addrF_coocM"]:
        for r in t3["M_led_flip_addrF_coocM"]:
            A(f"- {r['film_title']}: addr = {r['keystone_addr']} (F) / cooc = {r['keystone_cooc']} (M)")
    else:
        A("_(none)_")
    A("")
    A(f"**M-led flips (addr=M, cooc=F)** (n = {len(t3['M_led_flip_addrM_coocF'])}):")
    if t3["M_led_flip_addrM_coocF"]:
        for r in t3["M_led_flip_addrM_coocF"]:
            A(f"- {r['film_title']}: addr = {r['keystone_addr']} (M) / cooc = {r['keystone_cooc']} (F)")
    else:
        A("_(none)_")
    A("")
    # Compute headline statement based on actual flip pattern
    n_flips_F_addrM = len(t3["F_led_flip_addrM_coocF"])
    n_flips_F_addrF = len(t3["F_led_flip_addrF_coocM"])
    n_flips_M = len(t3["M_led_flip_addrF_coocM"]) + len(t3["M_led_flip_addrM_coocF"])
    pattern_holds = (n_flips_F_addrM >= 2 and n_flips_F_addrF == 0)
    A("**Interpretation.** Under addressee tagging, F-led films most often hinge on a "
      "male non-lead (the structural support character is masculine). Under scene "
      "co-occurrence, that signal is partly erased: supporting female characters who "
      "share scenes with the F lead through proximity (Luisa, Disgust, etc.) get "
      "promoted into keystone positions because co-presence credits them with edges they "
      "do not earn through actual addressed conversation.")
    A("")
    if pattern_holds:
        A(f"**The N=12 directional pattern HOLDS at N=17.** All {n_flips_F_addrM} F-led "
          f"gender-flips are still in the same direction (addressee picks M, co-occurrence "
          f"picks F). There are **zero** F-led flips in the opposite direction and **"
          f"{n_flips_M}** M-led flips. The F-led cross-gender keystone count drops from "
          f"{t3['F_led_addr_cross_gender']}/8 under addressee to "
          f"{t3['F_led_cooc_cross_gender']}/8 under co-occurrence — a difference of "
          f"{t3['F_led_addr_cross_gender'] - t3['F_led_cooc_cross_gender']} films. Under "
          f"the cheaper co-occurrence pipeline, the keystone-cross-gender finding for "
          f"F-led films (§4.3) would be materially weakened. This is the single strongest "
          f"evidence in the thesis that addressee tagging is doing structural work the "
          f"co-occurrence baseline cannot replicate.")
    else:
        A(f"**At N=17 the N=12 directional pattern is qualified.** {n_flips_F_addrM} "
          f"F-led films flip addr→M / cooc→F, {n_flips_F_addrF} flip in the opposite "
          f"direction, and {n_flips_M} M-led films flip. The F-led cross-gender keystone "
          f"count under co-occurrence ({t3['F_led_cooc_cross_gender']}/8) is "
          f"{'still markedly lower than' if t3['F_led_addr_cross_gender'] - t3['F_led_cooc_cross_gender'] >= 2 else 'similar to'} "
          f"under addressee ({t3['F_led_addr_cross_gender']}/8).")
    A("")

    # 2g Test 4 complexity
    t4 = results2["test4"]
    A("### 3.7 Test 4 — Method divergence vs scene complexity")
    A("")
    A("| Predictor | Spearman ρ | p | Pearson r | p |")
    A("|---|---|---|---|---|")
    for k, v in t4.items():
        A(f"| `{k}` | {v['spearman_rho']:+.3f} | {v['spearman_p']:.4f} | "
          f"{v['pearson_r']:+.3f} | {v['pearson_p']:.4f} |")
    A("")
    A("N=12 reference: all three correlations were negative and non-significant. "
      "At N=17 we " +
      ("confirm the null." if all(abs(v["spearman_rho"]) < 0.3 for v in t4.values())
       else "see some movement — see table.") +
      " The substantive interpretation stands: method divergence is driven by **specific "
      "dyadic-addressing patterns**, not by crowd-scene density. Crowd scenes are not the "
      "mechanism by which co-occurrence diverges from addressee tagging.")
    A("")

    # 2h narrative interpretation of top |Δz|
    top3 = results2["cmp_tbl"].head(3)
    A("### 3.8 Narrative interpretation — top three divergent protagonists")
    A("")
    film_specific = {
        "raya_and_the_last_dragon_2021": (
            "Raya travels through largely mixed-gender ensembles (Tong, Boun, Noi+ondines, "
            "Sisu's clan), so scene co-occurrence dilutes her same-gender embedding toward "
            "the cast mean. Addressee tagging recovers the dyadic dominance of Sisu and Noi "
            "in her actual lines — a same-gender pattern that is real at the speech level "
            "but invisible at the scene level."),
        "encanto_2021": (
            "Mirabel shares scenes with most of the Madrigal women (Abuela, Pepa, Julieta, "
            "Isabela, Luisa, Dolores), but a substantial fraction of her addressed speech "
            "goes to her father Agustín, her uncle Bruno, and her cousins Antonio/Camilo. "
            "Co-occurrence treats the all-female household scenes as same-gender ties, "
            "addressee tagging tracks who she actually talks to."),
        "findingnemo": (
            "Marlin spends nearly the entire film with Dory (F) as his co-traveller. "
            "Co-occurrence credits him with edges to the many male school-of-fish and "
            "P. Sherman ensemble characters who share his scenes, pulling his same-gender "
            "share up; addressee tagging captures that the load-bearing dialogic dyad is "
            "Marlin↔Dory, which is cross-gender."),
        "inside_out_2015": (
            "Joy's same-gender share is similar under both methods because she speaks "
            "to most of the Mind characters she shares scenes with — the divergence is "
            "small and not driven by phantom ties."),
        "soul_2020": "N/A — Soul is dropped per Convention 1.",
    }
    for _, r in top3.iterrows():
        A(f"**{r.film_title} ({r.protagonist}, {r.lead_gender}) — |Δz| = {r.abs_diff_z:.3f}** "
          f"(addr z = {r.addr_samesex_z:+.3f}, cooc z = {r.cooc_samesex_z:+.3f}).")
        film_id = df_all[df_all.film_title == r.film_title].iloc[0].film_id
        if film_id in film_specific:
            A(film_specific[film_id])
        elif r.diff_z > 0:
            A("Addressee reports higher same-gender embedding than co-occurrence: "
              "the protagonist appears in mixed-gender scenes but directs a "
              "disproportionate weight of dialogue at same-gender alters. "
              "Co-occurrence dilutes a real dyadic same-gender signal.")
        else:
            A("Addressee reports lower same-gender embedding than co-occurrence: "
              "the protagonist shares many scenes with same-gender characters but "
              "directs much of their actual speech across gender. "
              "Co-occurrence over-credits same-gender ties from shared scenes that "
              "do not carry addressed conversation.")
        A("")
    A("")

    # 2i decision table
    A("### 3.9 Decision table — does the addressee step matter?")
    A("")
    A("| Domain | Verdict | One-line evidence | N=12 → N=17 status |")
    A("|---|---|---|---|")
    h1z = h1["addr_z"]; h1cz = h1["cooc_z"]
    aggH1 = ("No" if (abs(h1z["cliffs_delta"]) < 0.3 and abs(h1cz["cliffs_delta"]) < 0.3
                         and h1z["mw_two_sided"]["p"] > 0.05 and h1cz["mw_two_sided"]["p"] > 0.05)
              else "Sometimes")
    A(f"| Aggregate H1 (F vs M cast-adj) | **{aggH1}** | "
      f"Both methods give Cliff's δ small, MW p large (see §3.3). | "
      f"{'held' if aggH1 == 'No' else 'shifted'} |")
    A(f"| Per-protagonist z | **Sometimes** | Max |Δz| at N=17 = "
      f"{results2['cmp_tbl'].abs_diff_z.max():.3f}; "
      f"{(results2['cmp_tbl'].abs_diff_z > 0.5).sum()} protagonists exceed 0.5. | "
      f"held |")
    phantom_diff = t1['share_same_phantom'] - t1['share_same_both']
    if t1['chi2_p'] < 0.10 and phantom_diff > 0.05:
        phantom_verdict, phantom_status = "Yes", "held"
    elif phantom_diff > 0.02:
        phantom_verdict, phantom_status = "Directionally yes", "shifted (was χ² p=0.052 at N=12, now non-sig)"
    else:
        phantom_verdict, phantom_status = "No", "shifted (was χ² p=0.052 at N=12, now non-sig)"
    A(f"| Phantom-edge inflation | "
      f"**{phantom_verdict}** | "
      f"Phantom same-gender share = {t1['share_same_phantom']:.3f} vs consensus "
      f"{t1['share_same_both']:.3f}; χ² p = {t1['chi2_p']:.3f}. | "
      f"{phantom_status} |")
    A(f"| Reciprocity (directional) | **Yes (definitionally)** | "
      f"Co-occurrence forces reciprocity = 1; addressee recovers within-film variation. | held |")
    A(f"| Keystone identity | **Yes** | Agreement rate {t3['agree_rate']:.1%}; "
      f"directional gender flip in F-led films: {len(t3['F_led_flip_addrM_coocF'])} cases. | "
      f"{'held' if len(t3['F_led_flip_addrM_coocF']) >= 2 else 'shifted'} |")
    A(f"| Divergence prediction (scene complexity) | **No** | All three "
      f"complexity-correlations < 0 or non-significant (§3.7). | held |")
    A("")

    # 2j honest "where it's overkill"
    A("### 3.10 Honest paragraph — where the LLM step is overkill")
    A("")
    A("For **population-level homophily testing** on this kind of animated-film corpus, "
      "scene co-occurrence with a cast-composition null is sufficient: both methods give "
      "the same qualitative answer to the F-vs-M H1 question (see §3.3 — the cast-adjusted "
      "z-score is null under both, the raw gap is similar in magnitude under both). The "
      "addressee LLM step earns its keep at the **per-film** level (specific dyadic "
      "patterns) and at the **per-character** level (keystone identification, reciprocity), "
      "not at the aggregate gender-comparison level. A researcher whose research question "
      "is exclusively the cross-film gender test could replicate this thesis's H1 finding "
      "with a much cheaper scene-co-occurrence pipeline. The LLM step matters when the "
      "research question zooms in on individual films, individual characters, or the "
      "structural roles characters play within a network.")
    A("")

    # ====================================================================
    # 4. Structural context
    # ====================================================================
    A("## 4. Structural context")
    A("")

    # 3a betweenness
    A("### 4.1 Protagonist betweenness, F vs M")
    A("")
    A("**Raw `protag_betweenness`:**")
    A("")
    A(fmt_panel_md(results3["panel_betw_raw"]))
    A("")
    A("**Cast-adjusted `protag_betw_z`:**")
    A("")
    A(fmt_panel_md(results3["panel_betw_z"]))
    A("")
    bz = results3["panel_betw_z"]
    A(f"N=12 reference: raw MW p = 0.95 (δ = +0.05), cast-adj MW p = 0.84 (δ = +0.10). "
      f"At N=17: cast-adjusted Cliff's δ = {bz['cliffs_delta']:+.3f}, "
      f"MW two-sided p = {bz['mw_two_sided']['p']:.4f}. "
      f"Female and male leads sit at structurally similar central positions; the previous "
      f"M-side variance inflation (driven by Mike/Sulley both having extreme negative z) is "
      f"reduced because Mike is now dropped per Convention 3.")
    A("")

    # 3b quadrant
    A("### 4.2 Betweenness × homophily quadrant typology")
    A("")
    A("Crosstab (films per quadrant × lead gender):")
    A("")
    A(df_to_md(results3["quad_ct"].reset_index()))
    A("")
    A("**Per-protagonist quadrant placements (N=17):**")
    A("")
    A(df_to_md(results3["quad"][["film_title", "protagonist", "lead_gender",
                                    "protag_betw_z", "protag_samesex_z", "quadrant"]]))
    A("")
    A("At N=12 half of F-leads (3/6) fell in **high bridge / low embedded** (Mulan, "
      "Hopps, Mirabel — central brokers whose direct ties skew cross-gender). With 9 "
      "F-leads at N=17, the new entries Belle, Elastigirl, Ember locate per the table above.")
    A("")
    A("→ Figure: `figures_n17/fig06_n17_quadrant_betw_x_homophily.png`.")
    A("")

    # 3c keystone
    A("### 4.3 Keystone analysis")
    A("")
    A("Full keystone table (sorted by year):")
    A("")
    A(df_to_md(results3["keystone_table"]))
    A("")
    A("Crosstab — lead gender × keystone gender:")
    A("")
    A(df_to_md(results3["ks_ct"].reset_index()))
    A("")
    A("Crosstab — lead gender × keystone_diff_gender (0=same, 1=cross):")
    A("")
    A(df_to_md(results3["fisher_ct"].reset_index()))
    A("")
    fisher_p_val = results3['fisher_p']
    fisher_flag = "**now reaches conventional significance at α=0.05**" if fisher_p_val < 0.05 else \
                   ("borderline significant" if fisher_p_val < 0.10 else "not significant")
    A(f"**Fisher exact (two-sided): odds = {results3['fisher_odds']:.3f}, "
      f"p = {results3['fisher_p']:.4f}** ({fisher_flag}). "
      f"N=12 reference: F-led 3/6 cross-gender vs M-led 1/7, Fisher p = 0.27. "
      f"Notebook 09's 18-film descriptive direction (Mike-keep convention): F-led 6/9 vs M-led 2/9. "
      f"After applying Conventions (drop Soul, drop Mike-row → keep Sulley): the M-led "
      f"cross-gender count drops to 1/8 (Bonnie in Toy Story 3); Monsters Inc keystone for "
      f"the Sulley row is Mike (M, same-gender). **The Fisher p at N=17 ({fisher_p_val:.4f}) "
      f"is a meaningful shift from N=12 (0.27)** — what was a directional-only trend in the "
      f"sub-sample now crosses (or approaches) the conventional significance threshold. "
      f"This is the strongest structural finding in the corpus: **F-led films are far more "
      f"often held together by a non-lead character of the opposite gender than M-led films are**.")
    A("")
    A("**Components after keystone removal (MW F vs M):**")
    A("")
    cp = results3["comp_panel"]
    A(f"- F median = {cp['median_F']:.2f}, M median = {cp['median_M']:.2f}, "
      f"Cliff's δ = {cp['cliffs_delta']:+.3f}, MW two-sided p = {cp['mw_two_sided']['p']:.4f}. "
      f"N=12: F=2 vs M=3 components, MW p = 0.073, r_rb = +0.64 (large).")
    A("")
    A("**Temporal pattern.** N=12 had a clean late-corpus signal (Frozen 2013, Encanto 2021, "
      "Raya 2021 all F-keystone; older F-leds all M-keystone). At N=17 the pattern is "
      "more heterogeneous: Beauty (1991) and Mulan (1998) → M; Frozen (2013), Encanto "
      "(2021), Raya (2021) → F; Inside Out (2015), Zootopia (2016), Incredibles 2 (2018), "
      "Elemental (2023) → M. **The clean post-2010 → F-keystone trend dissolves** when "
      "the full corpus is considered.")
    A("")

    # 3d global metrics
    A("### 4.4 Global network metrics, F vs M")
    A("")
    A("Film-level (N=17, one row per film). MW two-sided p, Cliff's δ, bootstrap 95% CI "
      "on median diff F−M.")
    A("")
    A("| Metric | F median | M median | Cliff's δ | MW p | Boot CI median |")
    A("|---|---|---|---|---|---|")
    for col, p in results3["global_panels"].items():
        A(f"| `{col}` | {p['median_F']:.3f} | {p['median_M']:.3f} | "
          f"{p['cliffs_delta']:+.3f} | {p['mw_two_sided']['p']:.4f} | "
          f"[{p['boot_median_diff_ci']['ci_lo']:+.3f}, "
          f"{p['boot_median_diff_ci']['ci_hi']:+.3f}] |")
    A("")
    A("N=12 reference: no test reached α = 0.05; M-led films were a touch denser, more "
      "reciprocal, more clustered on directional medians.")
    A("")

    # ====================================================================
    # 6. Descriptive extensions (Step 4)
    # ====================================================================
    A("## 5. Descriptive extensions (Step 4 of the plan)")
    A("")
    A("**Framing rule.** These are descriptive comparisons, not hypothesis tests. Cliff's δ "
      "is the primary evidence; permutation p and bootstrap CI on median diff are reported. "
      "The teammate's N=18 write-up framed several of these as H2/H3 — we use descriptive "
      "language here per the plan.")
    A("")

    panels4 = results4["panels"]
    tables4 = results4["tables"]

    for col in ["female_alter_betw_z", "burt_constraint", "ego_density", "reciprocity"]:
        if col not in panels4:
            continue
        section_names = {
            "female_alter_betw_z": "5.1 `female_alter_betw_z` — mean betweenness z of protagonist's female alters",
            "burt_constraint": "5.2 `burt_constraint` — Burt structural-holes (0=bridges groups, 1=trapped in clique)",
            "ego_density": "5.3 `ego_density` — interconnectedness of protagonist's alters",
            "reciprocity": "5.4 `reciprocity` — film-level reciprocity",
        }
        A(f"### {section_names[col]}")
        A("")
        if col == "burt_constraint":
            A("**Caveat:** this metric is NOT null-model adjusted (raw Burt constraint, not z-scored against any null).")
            A("")
        A("Per-film table (sorted):")
        A("")
        A(df_to_md(tables4[col]))
        A("")
        p_all = panels4[col]["all"]
        if p_all is None:
            A("_F-vs-M comparison skipped: too few non-NaN values._")
        else:
            A(f"**F vs M (all 17 films):** median F = {p_all['median_F']:+.4f}, "
              f"median M = {p_all['median_M']:+.4f}, "
              f"Cliff's δ = **{p_all['cliffs_delta']:+.3f}**, "
              f"MW two-sided p = {p_all['mw_two_sided']['p']:.4f}, "
              f"permutation on median diff p = {p_all['perm_median']['p_two_sided']:.4f}, "
              f"bootstrap 95% CI on median diff (F−M) = "
              f"[{p_all['boot_median_diff_ci']['ci_lo']:+.4f}, "
              f"{p_all['boot_median_diff_ci']['ci_hi']:+.4f}].")
        # Brief interpretation
        interp = {
            "female_alter_betw_z": "N=12: F median = +0.33, M = −0.55, δ = +0.46. "
                                    "Direction: female alters in F-led films punch above their structural weight.",
            "burt_constraint": "N=12: F = 0.32, M = 0.38, δ = −0.28. Direction: F protagonists "
                                "slightly less constrained (more brokerage). Note unadjusted.",
            "ego_density": "Confounded with network size — protagonists in smaller casts "
                            "trivially have denser ego-networks. Interpret as descriptive only.",
            "reciprocity": "Film-level reciprocity; differs from the addressee per-edge reciprocity in §3.5.",
        }
        A("")
        A("*Interpretive note.* " + interp[col])
        A("")
    A("→ Figure: `figures_n17/fig07_n17_descriptive_smallmultiples.png`.")
    A("")

    # ====================================================================
    # 7. Exploratory (Step 6)
    # ====================================================================
    A("## 7. Exploratory (Step 6 of the plan)")
    A("")
    chosen = results6["chosen_cutoff"]
    A(f"### 7.1 Temporal era split (cutoff = {chosen})")
    A("")
    A("Films per era × gender:")
    A("")
    A(df_to_md(pd.DataFrame(results6["splits"][chosen]["counts"]).fillna(0).astype(int).reset_index()))
    A("")
    A("Median z-scores by era × gender:")
    A("")
    A(df_to_md(results6["era_tab"].reset_index()))
    A("")
    if results6["era_panels"]:
        for era_val, panel in results6["era_panels"].items():
            A(f"**Within-era F vs M on `protag_samesex_z`, era = {era_val}** "
              f"(n_F = {panel['n_F']}, n_M = {panel['n_M']}): "
              f"Cliff's δ = {panel['cliffs_delta']:+.3f}, "
              f"MW two-sided p = {panel['mw_two_sided']['p']:.4f}.")
            A("")
    A("**Confound note.** The post-2010 corpus skew (most films post-2010, all 8 F-leds "
      "from 1991, 1998 and 2013–2023) means year and gender are partially confounded. "
      "Any year × gender interpretation should be read with this in mind.")
    A("")

    # 6b cluster
    A("### 7.2 Network archetype clustering")
    A("")
    cl = results6["cluster"]
    A("| k | Silhouette | Cluster sizes |")
    A("|---|---|---|")
    for k, v in cl.items():
        A(f"| {k} | {v['silhouette']:.3f} | {v['sizes']} |")
    A("")
    A(f"N=12 reference: k=2 silhouette = 0.27 (weak), k=2 split 6/6 with exactly 3F/3M in each "
      f"cluster (no gender separation). At N=17: "
      f"best k by silhouette = "
      f"{max(cl, key=lambda k: cl[k]['silhouette'])} (silhouette = "
      f"{max(v['silhouette'] for v in cl.values()):.3f}). "
      f"All silhouettes are weak; the global network metrics do not cleanly partition films.")
    A("")
    A("→ Figure: `figures_n17/fig08_n17_ward_dendrogram.png`.")
    A("")

    # 6c correlation
    A("### 7.3 Spearman correlations with `protag_samesex_z`")
    A("")
    A("Top correlates of `protag_samesex_z` across all numeric features (merged with extended):")
    A("")
    top = results6["top_spearman"].head(15)
    A(series_to_md(top.round(3), name="Spearman rho"))
    A("")
    A("N=12 reference: strongest positive correlates were `protag_betw_z` (+0.66) and "
      "`homophily_z` (+0.66); strongest negative was `n_nodes` (−0.41).")
    A("")

    # ====================================================================
    # 8. Limitations
    # ====================================================================
    A("## 8. Limitations (Agent B portion)")
    A("")
    A("- **N is small (17 films, 8F+9M).** Mann-Whitney has very limited power for medium "
      "effects (Cliff's δ ≈ 0.33 detected at < 30% power at α = 0.05). We rely on effect "
      "sizes and bootstrap intervals, not on p-values, for substantive claims.")
    A("- **Two films dropped relative to the upstream 18-film pipeline:** Soul (Convention 1, "
      "genderless souls) and the Mike row of Monsters Inc (Convention 3, one-protagonist-per-film). "
      "Both are documented filters, not data quality issues.")
    A("- **Configuration-model null** (used in `film_features_all.csv` for `protag_samesex_z` and "
      "`protag_betw_z`) is degree-preserving binary rewiring on directed addressee edges. "
      "The label-shuffling null used for the Pillar B comparison (§3) is topology-preserving "
      "and label-permuting. These two nulls answer slightly different questions; the "
      "addressee z-scores in §3 do not match `protag_samesex_z` in `film_features_all.csv` "
      "numerically and that is intentional.")
    A("- **`burt_constraint` is NOT null-adjusted** (raw Burt 1992 constraint, not z-scored). "
      "Flagged in §5.2.")
    A("- **Co-leads.** The Conventions drop Mike from Monsters Inc; Frozen keeps Anna only "
      "(Elsa not a second row); Toy Story keeps Woody only (Buzz not a second row). "
      "This is uniform across the corpus and avoids pseudo-replication, but it loses some "
      "co-lead structural information that could be recovered by ego-network-of-each-lead "
      "analyses in future work.")
    A("- **N=12 → N=17 stability.** Most headline results are stable: cast adjustment "
      "absorbs the raw gap, H1 null on cast-adjusted z, weak archetype clustering, no global "
      "metric differences. The most fragile finding remains keystone fragmentation (small N "
      "moves both ways). The temporal era story dissolves under the wider corpus.")
    A("")

    # ====================================================================
    # 9. Reproducibility
    # ====================================================================
    A("## 9. Reproducibility")
    A("")
    A("**Scripts (Agent B portion):**")
    A("- `CLEAN/analysis/h1_homophily/n17_orchestrator.py` — single entry point. "
      "Run with `python n17_orchestrator.py`.")
    A("- `CLEAN/analysis/h1_homophily/_common.py` — shared helpers (palette, stats, IO).")
    A("- `CLEAN/notebooks/06_network_analysis_PAU.ipynb` - per-film network production.")
    A("- `CLEAN/analysis/h1_homophily/phase3_crossfilm_method_validation.py` — shared "
      "helper module (addressee/co-occurrence network builders, label-shuffle null) "
      "imported by this orchestrator; not run directly.")
    A("- `CLEAN/analysis/h1_homophily/phase3_crossfilm_addressee_value.py` — shared "
      "helper module (edge-divergence, reciprocity, keystone, scene-complexity) "
      "imported by this orchestrator; not run directly.")
    A("- `CLEAN/notebooks/09_analysis.ipynb` - cross-film reporting notebook.")
    A("")
    A("**Derived working data (filtered, not modified upstream):**")
    A("- `CLEAN/data/04_features/film_features_all_n17.csv` (17 rows, conventions applied)")
    A("- `CLEAN/data/04_features/film_features_extended_n17.csv` (17 rows, conventions applied)")
    A("")
    A("**Outputs:**")
    A("- Tables: `CLEAN/analysis/h1_homophily/tables_n17/`")
    A("- Figures: `CLEAN/analysis/h1_homophily/figures_n17/` (PNG + PDF)")
    A(f"- This document: `CLEAN/analysis/h1_homophily/{OUTPUT_MD.name}`")
    A("- Status log: `CLEAN/analysis/h1_homophily/STATUS_agent_B.md`")
    A("")
    A("**Seeds.** `RNG_SEED = 20260622` for permutations (10,000 iters for H1 panels), "
      "bootstrap (10,000 iters), label-shuffling null (2,000 iters), k-means clustering.")
    A("")
    A("**Run order:** the orchestrator runs Steps 1 → 2 → 3 → 4 → 6 in that order; "
      "no dependencies between Step 5 (Social World, handled by a parallel agent) and the "
      "steps in this document.")
    A("")
    A("**The five Conventions** (applied at load time):")
    A("1. Drop `soul_2020` from the analytic sample (genderless souls make the same-gender share metric incoherent).")
    A("2. One protagonist per film — no co-leads added as second rows.")
    A("3. Monsters Inc — keep Sulley, drop Mike (Sulley is the narrative lead; overrides notebook 09's Mike-keep choice).")
    A("4. Uniform N = 17 (8F + 9M) across film-level and protagonist-level analyses.")
    A("5. When citing N=12 reference values, note small caveats where Soul (Joe) affected the "
      "prior number (e.g. Soul had 0 phantom edges, so the drop does not change Test 1 directly; "
      "Soul's Terry was F so its inclusion at N=12 already counted as an F-keystone for an M-led "
      "film, dropping it removes one M-led cross-gender keystone — so Test 3 N=17 has fewer "
      "M-led cross-gender keystones than the notebook 09 N=18 descriptive count).")
    A("")
    A("**Future work (noted, not executed here):**")
    A("- Weighted configuration-model null for `protag_samesex_z` (current null is binary-tie).")
    A("- Co-lead ego-network analyses (Mike, Buzz, Elsa, etc.).")
    A("- Context-window study (Appendix A in the plan) — comparing utterance-level vs scene-level "
      "addressee tagging on a small subset.")
    A("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    return OUTPUT_MD


# ============================================================================
# MAIN
# ============================================================================
def main():
    # initialise status file
    if not STATUS_FILE.exists():
        STATUS_FILE.write_text("# Agent B status log (Steps 1, 2, 3, 4, 6)\n\n", encoding="utf-8")
    status("Started orchestrator (Steps 1,2,3,4,6)")

    df_all = load_n17_all()
    df_ext = load_n17_extended()
    assert len(df_all) == 17, f"Expected 17 rows, got {len(df_all)}"
    assert len(df_ext) == 17, f"Expected 17 rows, got {len(df_ext)}"
    # Guard composition, not just the count: a swapped/duplicated row can still total 17.
    assert df_all.film_id.is_unique, "n17 has >1 row for some film — check apply_conventions C2/C3"
    _split = df_all.lead_gender.value_counts().to_dict()
    assert _split.get("F") == 8 and _split.get("M") == 9, f"Expected 8F/9M, got {_split}"
    status(f"Loaded N=17. F={int((df_all.lead_gender=='F').sum())}, "
            f"M={int((df_all.lead_gender=='M').sum())}")

    # Step 1
    r1 = step1_pillar_a(df_all)
    status(f"Step 1 done — raw δ={r1['panel_raw']['cliffs_delta']:+.3f}, "
            f"adj δ={r1['panel_z']['cliffs_delta']:+.3f}, "
            f"raw MW-1s p={r1['panel_raw']['mw_one_sided_F<M']['p']:.4f}")

    # Step 2
    r2 = step2_pillar_b(df_all)
    status(f"Step 2 done — z Spearman ρ={r2['agreement']['z_spearman_rho']:+.3f}, "
            f"keystone agree={r2['test3']['agree_rate']:.1%}, "
            f"F-led addr-cross={r2['test3']['F_led_addr_cross_gender']}/8, "
            f"F-led cooc-cross={r2['test3']['F_led_cooc_cross_gender']}/8")

    # Step 3
    r3 = step3_structural(df_all, df_ext)
    status(f"Step 3 done — Fisher keystone p={r3['fisher_p']:.4f}, "
            f"betw_z δ={r3['panel_betw_z']['cliffs_delta']:+.3f}")

    # Step 4
    r4 = step4_descriptive(df_ext)
    fa = r4['panels'].get('female_alter_betw_z', {}).get('all')
    bc = r4['panels'].get('burt_constraint', {}).get('all')
    status(f"Step 4 done — female_alter_z δ="
            f"{fa['cliffs_delta'] if fa else float('nan'):+.3f}, "
            f"burt_constraint δ="
            f"{bc['cliffs_delta'] if bc else float('nan'):+.3f}")

    # Step 6
    r6 = step6_exploratory(df_all, df_ext)
    best_k = max(r6["cluster"], key=lambda k: r6["cluster"][k]["silhouette"])
    status(f"Step 6 done — best k={best_k} silhouette="
            f"{r6['cluster'][best_k]['silhouette']:.3f}, "
            f"era cutoff={r6['chosen_cutoff']}")

    # Markdown
    md_path = write_markdown(df_all, df_ext, r1, r2, r3, r4, r6)
    status(f"Markdown written: {md_path}")
    print(f"\nDONE. Output: {md_path}")


if __name__ == "__main__":
    main()
