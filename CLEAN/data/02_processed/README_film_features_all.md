# `film_features_all.csv` — Data Dictionary

One row **per protagonist per film**. Co-lead films (e.g. *Monsters, Inc.*) have two rows
(Mike, Sulley). Produced by `CLEAN/notebooks/08_network_analysis_PAU.ipynb`: each time you run
notebook 08 for a film, its row is **upserted** (replaced in place) into this table, keyed by
`(film_id, protagonist)`.

**Pipeline:** screenplay → notebook 07 tags *who each line is spoken to*
(`utterances_with_addressee_scene.csv`) → notebook 08 builds a directed dialogue network and
computes the features below.

**How the network is built:** every line whose `addressee_type == "individual"` becomes a
directed edge **speaker → addressee**, weighted by the number of such lines. Nodes are
**character_id** (unique); display names come from `canonical_name`. Edges with fewer than
`MIN_EDGE_COUNT` (default **3**) lines are dropped. Group / monologue / non-human / unclear
lines are **not** turned into edges.

**Gender** is read from each film's `character_review_<film>.csv` (manually reviewed).

> ⚠️ All 12 films in the current table are tagged with the **same** model (Claude, via notebook 07),
> so the rows are mutually comparable.

---

## 1. Identification

| column | meaning |
|---|---|
| `film_id` | Film identifier (e.g. `frozen_2013`). |
| `year` | Release year. |
| `protagonist` | Lead character's display name (the focal character of this row). |
| `lead_gender` | Gender of the protagonist — `F` or `M`. **This is the grouping variable for the thesis** (female-led vs male-led). |

## 2. Global network shape (whole cast, not protagonist-specific)

| column | meaning | how to read |
|---|---|---|
| `n_nodes` | Number of characters in the network. | Bigger = larger speaking cast. |
| `n_edges` | Number of directed speaker→addressee ties (≥3 lines). | — |
| `density` | Share of all possible ties that actually exist (0–1). | Higher = more tightly interconnected cast. |
| `reciprocity` | Fraction of ties that go **both ways** (A→B and B→A). | High = conversations are mutual, not one-directional. |
| `avg_path_len` | Average shortest path between characters (on the largest component). | Low = everyone is socially "close"; high = a stretched-out network. |
| `leading_eigenvalue` | Spectral radius of the weighted adjacency matrix. | Rough measure of how concentrated/intense the dialogue volume is. Scales with how much a few pairs dominate the talking. |
| `mean_clustering` | Average clustering coefficient (do a character's contacts also talk to each other). | High = lots of triangles / cliques; low = star-like. |

## 3. Network-wide gender homophily (whole cast)

Measures whether **same-gender pairs talk to each other more than expected**, across the entire
cast (independent of the protagonist).

| column | meaning | how to read |
|---|---|---|
| `homophily_obs` | Observed share of dialogue (edge weight) that is same-gender. | — |
| `homophily_z` | How many standard deviations the observed value sits above the configuration-model null. | >0 = more same-gender clustering than chance. |
| `homophily_p` | One-sided p-value: probability of seeing this much same-gender clustering by chance. | **< 0.05 = significantly homophilous cast.** |

## 4. Protagonist same-gender homophily ⭐ (the core thesis measure)

Does the **lead** talk to her/his **own gender** more than a randomized network would predict?

| column | meaning | how to read |
|---|---|---|
| `protag_samesex` | Share of the protagonist's ties that go to the **same gender** as the lead. | 0.80 = 80% of the lead's conversations are with same-gender characters. Read together with cast composition (a male-heavy cast inflates a male lead's value). |
| `protag_samesex_z` | Same value expressed as standard deviations above the configuration-model null. | — |
| `protag_samesex_p` | One-sided p: probability the lead is *this* same-gender-skewed by chance. | **< 0.05 = the lead is genuinely embedded in same-gender ties beyond what the cast structure forces.** A high p means "not different from random" — the apparent skew is just cast composition. |

## 5. Protagonist centrality / brokerage

Is the lead the structural **bridge** that holds the network together?

| column | meaning | how to read |
|---|---|---|
| `protag_betweenness` | Betweenness centrality of the lead (0–1): how often the lead lies on shortest paths between other characters. | High = the lead is a broker connecting otherwise-separate groups. |
| `protag_betw_z` | Betweenness vs the configuration-model null, in standard deviations. | — |
| `protag_betw_p` | One-sided p: probability the lead is *this* central by chance. | **< 0.05 = the lead is a significant structural bridge, not just incidentally central.** |

## 6. Removal analysis (Moretti 2011) — narrative prominence vs structural necessity

Following Moretti's *Hamlet* experiment: remove the most important **supporting** character
(the "keystone") and see whether the rest of the cast falls apart. (Removing the lead itself is
trivial, so we test the top **non-lead** bridge.)

| column | meaning | how to read |
|---|---|---|
| `keystone` | The removed character: the highest-betweenness character **other than** the lead. | — |
| `keystone_gender` | Gender of the keystone (`F`/`M`). | — |
| `keystone_diff_gender` | `1` if the keystone is a **different** gender from the lead, else `0`. | Lets you ask: does the network hinge on a cross-gender supporting character? |
| `components_after_removal` | Number of disconnected pieces the network breaks into once the keystone is removed. | `1` = stays whole; `>1` = fragments (the keystone was load-bearing). |
| `n_isolated_after_removal` | How many characters become completely isolated after removal. | Higher = more dependence on that single keystone. |

---

## Reading the p-values (the null model)

Columns ending in `_z` / `_p` come from a **configuration-model null**: the observed network is
randomly rewired 2000 times while preserving each character's number of ties, and the real value
is compared against that random distribution.

- **p < 0.05** → the feature is significantly *higher* than chance (one-sided tests).
- **p ≈ 0.5–1.0** → indistinguishable from chance (often the case when a result is just an artifact
  of cast composition).

## Known caveats (discuss before final write-up)

1. **N is small.** With ~6 female and ~6 male leads, the female-vs-male group test is
   under-powered — these per-film numbers describe each film; the cross-film hypothesis test
   needs the full set and should be read cautiously.
2. **The null is built on the *binary* tie structure**, while `protag_samesex` is *weight-based*
   (how much the lead talks, not just to how many). A weighted null is on the methods to-do list;
   until then read `protag_samesex_p` as approximate. (Open question for Pau.)
3. **`MIN_EDGE_COUNT = 3`** drops brief ties, so very short interactions don't appear in the
   network — this can make some ego-networks look sparser than the story feels.
4. **Co-leads** (Monsters, Inc.) appear as two rows; decide per analysis whether to keep both,
   average them, or pick one.
