"""
Generate three methodology/data figures for the dissertation.
  fig_pipeline_diagram      — §1/§4: raw screenplay → network → features → claims
  fig_method_schematic      — §4: one Frozen scene as addressee vs co-presence
  fig_frozen_network        — §3: full Frozen addressee network
Output → CLEAN/analysis/h1_homophily/figures_n17/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "figures_n17"
OUT.mkdir(exist_ok=True)

DATA = Path(__file__).parent.parent.parent / "data/processed/frozen_2013"

COLOR_F = "#D66B5A"   # warm red for female
COLOR_M = "#4A7FB5"   # blue for male
GRAY    = "#888888"
LIGHT   = "#F5F5F5"
BG      = "white"

# ──────────────────────────────────────────────────────────────
# FIGURE 1 — Pipeline diagram
# ──────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.5)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    stages = [
        ("Raw\nScreenplays", "60+ .txt/.pdf\nfiles", "#E8D5C4"),
        ("Structured\nDialogue", "speaker · addressee\n· scene · utterance", "#C8DDE8"),
        ("Character\nNetworks", "addressee (directed)\nco-presence (undirected)", "#C8E8D0"),
        ("Protagonist\nFeatures", "homophily z, betweenness\nkeystone, reciprocity", "#DDD0E8"),
        ("Defensible\nClaims", "H1 test · method\ncomparison · keystone", "#F0E8C0"),
    ]

    labels_a = ["Parsing &\nclean", "LLM addressee\ntagging", "Cast-adj null\n(Pillar A)", "Method comparison\n(Pillar B)"]

    box_w, box_h = 1.9, 1.5
    gap = 0.55
    y0 = 1.0
    xs = [0.3 + i * (box_w + gap) for i in range(5)]

    for i, (title, sub, color) in enumerate(stages):
        x = xs[i]
        rect = FancyBboxPatch((x, y0), box_w, box_h,
                              boxstyle="round,pad=0.06", linewidth=1.2,
                              edgecolor="#666", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y0 + box_h*0.68, title,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
        ax.text(x + box_w/2, y0 + box_h*0.28, sub,
                ha="center", va="center", fontsize=7.5, color="#444")

        if i < 4:
            ax1 = x + box_w + 0.04
            ax2 = xs[i+1] - 0.04
            mid_x = (ax1 + ax2) / 2
            mid_y = y0 + box_h/2
            ax.annotate("", xy=(ax2, mid_y), xytext=(ax1, mid_y),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
            ax.text(mid_x, mid_y + 0.55, labels_a[i],
                    ha="center", va="bottom", fontsize=7, color="#555", style="italic")

    # Pillar labels beneath arrows 3 and 4
    ax.text(xs[2] + box_w + gap/2, y0 - 0.28, "Pillar A",
            ha="center", fontsize=7.5, color="#7B5EA7", fontweight="bold")
    ax.text(xs[3] + box_w + gap/2, y0 - 0.28, "Pillar B",
            ha="center", fontsize=7.5, color="#5EA77B", fontweight="bold")

    ax.set_title("From raw screenplay to defensible claim",
                 fontsize=11, fontweight="bold", pad=10, color="#222")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pipeline_diagram.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("✓ fig_pipeline_diagram")


# ──────────────────────────────────────────────────────────────
# FIGURE 7 — Method schematic (one scene, two networks)
# ──────────────────────────────────────────────────────────────
def fig_method_schematic():
    """
    Scene: Frozen s019 (Anna, Elsa, Hans)
    Simplified to 5 utterances for clarity.
    Addressee: directed weighted edges (who addresses whom)
    Co-presence: undirected, all three share the scene
    """
    # Composite example (Frozen, coronation scene — Anna/Elsa dialogue arc).
    # Weights are deliberately unequal to make the "weighted" aspect visible:
    # Anna drives the conversation (5 lines to Elsa), Elsa responds (3 lines to Anna),
    # Hans contributes minimally (1 line to Elsa).
    lines = [
        ("Anna", "Elsa", "1"),
        ("Anna", "Elsa", "2"),
        ("Anna", "Elsa", "3"),
        ("Anna", "Elsa", "4"),
        ("Anna", "Elsa", "5"),
        ("Elsa", "Anna", "1"),
        ("Elsa", "Anna", "2"),
        ("Elsa", "Anna", "3"),
        ("Hans", "Elsa", "1"),
    ]

    chars = ["Anna", "Elsa", "Hans"]
    genders = {"Anna": "F", "Elsa": "F", "Hans": "M"}
    colors_node = {c: COLOR_F if genders[c]=="F" else COLOR_M for c in chars}

    # Addressee edge weights
    addr_weights = {}
    for spk, adr, _ in lines:
        if spk in chars and adr in chars:
            addr_weights[(spk, adr)] = addr_weights.get((spk, adr), 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)

    pos = {"Anna": (0, 0), "Elsa": (1, 1), "Hans": (2, 0)}

    def draw_nodes(ax):
        for char, (x, y) in pos.items():
            ax.scatter(x, y, s=900, color=colors_node[char], zorder=4,
                       edgecolors="#333", linewidths=1.5)
            gender_str = "(F)" if genders[char]=="F" else "(M)"
            ax.text(x, y - 0.22, f"{char}\n{gender_str}",
                    ha="center", va="top", fontsize=9.5, fontweight="bold", color="#222")

    # ── Left: Addressee network ──
    ax = axes[0]
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title("(a) Addressee network\ndirected, weighted", fontsize=10.5,
                 fontweight="bold", pad=8, color="#222")

    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(chars)
    for (spk, adr), w in addr_weights.items():
        G_dir.add_edge(spk, adr, weight=w)

    max_w = max(addr_weights.values())
    for (spk, adr), w in addr_weights.items():
        x1, y1 = pos[spk]; x2, y2 = pos[adr]
        lw = 1.5 + 3.5 * (w / max_w)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444",
                                   lw=lw, connectionstyle="arc3,rad=0.18",
                                   shrinkA=18, shrinkB=18))
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.14 if (spk,adr)==("Elsa","Anna") else -0.14
        ax.text(mx + 0.05, my + offset, str(w),
                ha="center", fontsize=8.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    draw_nodes(ax)

    # legend for edge weight
    ax.text(0.02, 0.02, "edge weight = # lines addressed",
            transform=ax.transAxes, fontsize=8, color=GRAY, style="italic")

    # ── Right: Co-presence network ──
    ax = axes[1]
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title("(b) Co-presence network\nundirected, unweighted", fontsize=10.5,
                 fontweight="bold", pad=8, color="#222")

    for c1 in chars:
        for c2 in chars:
            if c1 < c2:
                x1, y1 = pos[c1]; x2, y2 = pos[c2]
                ax.plot([x1, x2], [y1, y2], color="#555", lw=2.5, zorder=2)

    draw_nodes(ax)
    ax.text(0.02, 0.02, "any two characters sharing a scene are linked",
            transform=ax.transAxes, fontsize=8, color=GRAY, style="italic")

    # scene label
    fig.text(0.5, 0.01,
             "Scene: Frozen (2013) — Coronation hall, Anna proposes marriage to Hans",
             ha="center", fontsize=8.5, color=GRAY, style="italic")

    # gender legend
    handles = [
        mpatches.Patch(facecolor=COLOR_F, edgecolor="#333", label="Female character"),
        mpatches.Patch(facecolor=COLOR_M, edgecolor="#333", label="Male character"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_method_schematic.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("✓ fig_method_schematic")


# ──────────────────────────────────────────────────────────────
# FIGURE 4 — Frozen addressee network (full)
# ──────────────────────────────────────────────────────────────
def fig_frozen_network():
    edges_df = pd.read_csv(DATA / "network_edges.csv")
    nodes_df = pd.read_csv(DATA / "network_nodes.csv")

    # strip prefix
    edges_df["src"] = edges_df["speaker_clean"].str.replace("frozen_2013_", "")
    edges_df["tgt"] = edges_df["addressee_clean"].str.replace("frozen_2013_", "")
    nodes_df["char"] = nodes_df["character_id"].str.replace("frozen_2013_", "")
    nodes_df["char"] = nodes_df["char"].str.capitalize()
    edges_df["src"] = edges_df["src"].str.capitalize()
    edges_df["tgt"] = edges_df["tgt"].str.capitalize()

    G = nx.DiGraph()
    char_gender = dict(zip(nodes_df["char"], nodes_df["gender"]))
    for _, row in nodes_df.iterrows():
        G.add_node(row["char"], gender=row["gender"])
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["tgt"], weight=row["utterance_count"])

    # layout
    np.random.seed(42)
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.2)

    # node sizes by total degree (utterance count)
    deg = dict(G.degree(weight="weight"))
    max_d = max(deg.values())
    node_sizes = {n: 400 + 2200 * (deg.get(n, 0) / max_d) for n in G.nodes()}

    # edge widths
    max_w = max(d["weight"] for _, _, d in G.edges(data=True))
    edge_widths = [0.5 + 3.5 * (d["weight"] / max_w) for _, _, d in G.edges(data=True)]
    edge_alphas = [0.3 + 0.5 * (d["weight"] / max_w) for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    node_colors = [COLOR_F if char_gender.get(n) == "F" else COLOR_M for n in G.nodes()]
    sizes = [node_sizes[n] for n in G.nodes()]

    # draw edges individually for alpha
    for (u, v, d), lw, alpha in zip(G.edges(data=True), edge_widths, edge_alphas):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#888",
                                   lw=lw, alpha=alpha,
                                   connectionstyle="arc3,rad=0.12",
                                   shrinkA=12, shrinkB=12))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=sizes,
                           edgecolors="#333", linewidths=1.2)

    # protagonist (Anna) ring
    if "Anna" in pos:
        ax.scatter(*pos["Anna"], s=node_sizes["Anna"]*1.5,
                   facecolors="none", edgecolors="#222", linewidths=2.5, zorder=5)

    # labels
    label_offsets = {}
    for n in G.nodes():
        x, y = pos[n]
        label_offsets[n] = (x, y + 0.09)

    for n, (x, y) in label_offsets.items():
        ax.text(x, y, n.replace("_", " ").title(),
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if n == "Anna" else "normal",
                color="#111",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # legend
    handles = [
        mpatches.Patch(facecolor=COLOR_F, edgecolor="#333", label="Female character"),
        mpatches.Patch(facecolor=COLOR_M, edgecolor="#333", label="Male character"),
        plt.scatter([], [], s=150, facecolors=COLOR_F, edgecolors="#222",
                    linewidths=2.5, label="Protagonist (Anna)"),
    ]
    handles[-1] = mpatches.Patch(facecolor="none", edgecolor="#222",
                                  label="Protagonist (Anna) — double ring",
                                  linewidth=2)
    ax.legend(handles=handles[:2] + [mpatches.Patch(facecolor="none", edgecolor="#222",
              label="Protagonist = double ring", linewidth=2)],
              loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)

    ax.set_title("Frozen (2013) — addressee dialogue network\n"
                 "Node size ∝ total dialogue volume · Edge width ∝ directed utterance count",
                 fontsize=10.5, fontweight="bold", color="#222", pad=10)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_frozen_network.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("✓ fig_frozen_network")


if __name__ == "__main__":
    fig_pipeline()
    fig_method_schematic()
    fig_frozen_network()
    print("\nAll done → figures_n17/")
