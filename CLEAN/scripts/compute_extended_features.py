"""
Extended network features for cross-film gender analysis.

Metrics (all weighted, using utterance_count as edge weight):

1. burt_constraint  [Burt 1992, 2004 - Structural Holes]
   How constrained is the protagonist by a single dominant relationship?
   High (→1) = embedded in one tight clique, no brokerage.
   Low (→0) = bridges different groups, high narrative autonomy.
   Computed via nx.constraint() on the weighted undirected graph.

2. female_alter_betw_z
   Mean betweenness z-score of the protagonist's FEMALE alters only.
   Positive = female friends are structurally load-bearing.
   Negative = female friends are peripheral to the plot structure.

3. ego_density
   Fraction of possible directed ties among the protagonist's alters
   that actually exist. 0 = star topology; 1 = all alters mutually connected.
   Pulled from protagonist_metrics.csv (computed in notebook 06).

4. alter_importance_ratio
   Fraction of the protagonist's alters with above-average betweenness.
   Measures whether the protagonist's circle is structurally load-bearing.

References:
- Burt, R.S. (1992). Structural Holes. Harvard University Press.
- Burt, R.S. (2004). "Structural Holes and Good Ideas." AJS 110(2).
- Moretti, F. (2011). "Network Theory, Plot Analysis." New Left Review 68.
"""
import pandas as pd
import networkx as nx
from pathlib import Path

CLEAN_ROOT = Path(__file__).resolve().parent.parent
PROCESSED  = CLEAN_ROOT / "data" / "02_processed"
FEAT       = CLEAN_ROOT / "data" / "04_features"

feat_all = pd.read_csv(FEAT / "film_features_all.csv")

rows = []

for _, film_row in feat_all.iterrows():
    film_id     = film_row["film_id"]
    protagonist = film_row["protagonist"]

    edges_path = PROCESSED / film_id / "network_edges.csv"
    if not edges_path.exists():
        print(f"SKIP {film_id} — no network_edges.csv")
        continue

    edges = pd.read_csv(edges_path, dtype=str)
    edges["utterance_count"] = pd.to_numeric(edges["utterance_count"], errors="coerce").fillna(0)
    if edges.empty:
        continue

    # build weighted undirected graph
    G_und = nx.Graph()
    for _, e in edges.iterrows():
        u = e["speaker_clean"]; v = e["addressee_clean"]; w = float(e["utterance_count"])
        if pd.notna(u) and pd.notna(v) and u != v and w > 0:
            if G_und.has_edge(u, v):
                G_und[u][v]["weight"] += w
            else:
                G_und.add_edge(u, v, weight=w)

    prot_node = film_id + "_" + protagonist.lower().replace(" ", "_").replace("-", "_")

    if prot_node not in G_und.nodes():
        print(f"SKIP {film_id} — protagonist node not found: {prot_node}")
        continue

    alters = list(G_und.neighbors(prot_node))
    betw_path = PROCESSED / film_id / "betweenness_null.csv"

    # ── 1. burt_constraint ──────────────────────────────────────────────────
    burt_constraint = None
    try:
        burt_constraint = nx.constraint(G_und, weight="weight").get(prot_node)
    except Exception as ex:
        print(f"  constraint error {film_id}: {ex}")

    # ── 2. female_alter_betw_z ───────────────────────────────────────────────
    female_alter_betw_z = None
    if betw_path.exists() and alters:
        betw_df = pd.read_csv(betw_path)
        alter_betw = betw_df[betw_df["character"].isin(alters)]
        female_alters = alter_betw[alter_betw["gender"] == "F"]
        if len(female_alters) > 0:
            female_alter_betw_z = female_alters["z_score"].mean()

    # ── 3. ego_density (from protagonist_metrics.csv) ────────────────────────
    ego_density = None
    n_samegender_alters = None
    prot_met_path = PROCESSED / film_id / "protagonist_metrics.csv"
    if prot_met_path.exists():
        pm = pd.read_csv(prot_met_path)
        match = pm[pm["protagonist"].str.lower() == protagonist.lower()]
        if not match.empty:
            ego_density = match["ego_density"].values[0]
            if "n_samegender_alters" in match.columns:
                n_samegender_alters = match["n_samegender_alters"].values[0]

    # ── 4. alter_importance_ratio ────────────────────────────────────────────
    alter_importance_ratio = None
    if betw_path.exists() and alters:
        betw_df = pd.read_csv(betw_path)
        mean_betw = betw_df["betw_obs"].mean()
        alter_rows = betw_df[betw_df["character"].isin(alters)]
        if len(alter_rows) > 0:
            alter_importance_ratio = (alter_rows["betw_obs"] > mean_betw).mean()

    fmt = lambda x, d=3: f"{x:.{d}f}" if x is not None else "N/A"
    print(f"{film_id} ({protagonist}): "
          f"burt={fmt(burt_constraint)}, "
          f"female_alter_z={fmt(female_alter_betw_z,2)}, "
          f"ego_density={fmt(ego_density,3)}, "
          f"alter_importance={fmt(alter_importance_ratio,3)}")

    rows.append({
        "film_id":               film_id,
        "protagonist":           protagonist,
        "burt_constraint":       round(burt_constraint, 4) if burt_constraint is not None else None,
        "female_alter_betw_z":   round(female_alter_betw_z, 3) if female_alter_betw_z is not None else None,
        "ego_density":           ego_density,
        "alter_importance_ratio": round(alter_importance_ratio, 4) if alter_importance_ratio is not None else None,
        "n_samegender_alters":   n_samegender_alters,
    })

out = pd.DataFrame(rows)

print("\n=== SUMMARY TABLE ===")
print(out[["film_id", "burt_constraint", "female_alter_betw_z", "ego_density"]].to_string(index=False))

# merge with main film_features
merged = feat_all.merge(out, on=["film_id", "protagonist"], how="left")
out_path = FEAT / "film_features_extended.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Shape: {merged.shape}")
