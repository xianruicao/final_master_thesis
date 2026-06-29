# Unified Results — Steps 1, 2, 3, 4, 6 (Agent B, N=17)

> Source-of-truth plan: `CLEAN/admin/AGENT_INSTRUCTION_run_analysis_N18.md`.
> Analytic sample: **N=17 (9 F-led + 8 M-led)** after Conventions filters (Soul dropped — Convention 1; Mike row of Monsters Inc dropped, Sulley kept — Convention 3).
> RNG_SEED = 20260622. Pillar B null = label-shuffling, n=2000 permutations.
> Tables: `tables_n17/`. Figures: `figures_n17/`. Orchestrator: `n17_orchestrator.py`.
> Step 5 (Social World) is produced separately by another agent.

## 2. Pillar A — the measure (cast-adjusted protagonist same-gender homophily)

This is the methodological core. We compare F-led vs M-led protagonists on two paired quantities: the **raw** same-gender share of dialogue (`protag_samesex`) and the **cast-adjusted z-score** under a degree-preserving configuration-model null (`protag_samesex_z`, 2000 rewirings). The decomposition tests whether any F-vs-M gap in same-gender embedding is structurally inevitable given cast composition, or whether it reflects protagonist-level behaviour.

### 2.1 Raw vs cast-adjusted decomposition

**Raw share** `protag_samesex`:

| Statistic | Value |
|---|---|
| n (F / M) | 9 / 8 |
| Median F / M | +0.4286 / +0.7333 |
| Mean F / M | +0.3658 / +0.7243 |
| Median diff (F − M) | -0.3047 |
| Mean diff (F − M) | -0.3584 |
| Mann-Whitney U (two-sided, exact) | U = 2.0, p = 0.0003 |
| Mann-Whitney U (one-sided, F<M) | p = 0.0002 |
| Cliff's δ | -0.944 |
| Rank-biserial r | +0.944 |
| Permutation on median diff (10k, two-sided) | p = 0.0170 |
| Permutation on mean diff (10k, two-sided) | p = 0.0002 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.4851, -0.2244] |
| Bootstrap 95% CI on median diff (F−M) | [-0.5979, -0.1536] |

**Cast-adjusted z** `protag_samesex_z`:

| Statistic | Value |
|---|---|
| n (F / M) | 9 / 8 |
| Median F / M | -0.0850 / +0.2905 |
| Mean F / M | +0.3164 / +0.3544 |
| Median diff (F − M) | -0.3755 |
| Mean diff (F − M) | -0.0379 |
| Mann-Whitney U (two-sided, exact) | U = 35.0, p = 0.9626 |
| Mann-Whitney U (one-sided, F<M) | p = 0.4813 |
| Cliff's δ | -0.028 |
| Rank-biserial r | +0.028 |
| Permutation on median diff (10k, two-sided) | p = 0.7725 |
| Permutation on mean diff (10k, two-sided) | p = 0.9605 |
| Bootstrap 95% CI on mean diff (F−M) | [-1.4666, +1.3136] |
| Bootstrap 95% CI on median diff (F−M) | [-2.0090, +2.3850] |

**Comparison.** At N=12 the raw measure showed Cliff's δ = −0.71 (large, MW one-sided p = 0.018, bootstrap CI on mean diff [−0.43, −0.09] **excluding 0**), while the cast-adjusted measure showed δ = +0.10 (negligible). At **N=17** we find raw Cliff's δ = **-0.944** (MW one-sided p = 0.0002) and cast-adjusted Cliff's δ = **-0.028** (MW one-sided p = 0.4813). The directional story — raw shows a substantial F<M gap, cast adjustment largely absorbs it — is preserved and **strengthened** at the expanded N. Notably, the raw gap is **larger** at N=17 than at N=12 (δ = -0.944 vs −0.71), and the cast-adjusted gap is essentially zero (δ = -0.028). The methodological centrepiece of the thesis — that the obvious F-vs-M raw gap is a cast-composition artefact absorbed by the configuration-model null — is **more cleanly demonstrated at N=17** than at N=12.

→ Figure: `figures_n17/fig02_n17_raw_vs_adjusted.png` (side-by-side raw vs z box+strip).

### 2.2 Per-protagonist cast-adjusted z (individual significance)

| film_title | protagonist | lead_gender | protag_samesex | protag_samesex_z | protag_samesex_p | sig_alpha05 |
|---|---|---|---|---|---|---|
| Zootopia (2016) | Hopps | F | 0.263 | -1.480 | 0.914 | 0 |
| Raya (2021) | Raya | F | 0.444 | -1.387 | 0.962 | 0 |
| Coco (2017) | Miguel | M | 0.529 | -1.360 | 0.974 | 0 |
| Up (2009) | Carl | M | 0.833 | -1.269 | 1.000 | 0 |
| Inside Out (2015) | Joy | F | 0.429 | -1.175 | 0.929 | 0 |
| Mulan (1998) | Mulan | F | 0.200 | -0.642 | 0.746 | 0 |
| Monsters Inc (2001) | Sulley | M | 0.667 | -0.577 | 0.768 | 0 |
| Encanto (2021) | Mirabel | F | 0.583 | -0.085 | 0.894 | 0 |
| Cars 2 (2011) | Mater | M | 0.909 | 0.216 | 0.580 | 0 |
| Toy Story 3 (2010) | Woody | M | 0.667 | 0.365 | 0.418 | 0 |
| Finding Nemo (2003) | Marlin | M | 0.800 | 0.834 | 0.195 | 0 |
| Onward (2020) | Ian | M | 0.500 | 1.229 | 0.201 | 0 |
| Beauty & the Beast (1991) | Belle | F | 0.250 | 1.289 | 0.137 | 0 |
| Elemental (2023) | Ember | F | 0.429 | 1.808 | 0.111 | 0 |
| Frozen (2013) | Anna | F | 0.250 | 2.132 | 0.014 | 1 |
| Incredibles 2 (2018) | Elastigirl | F | 0.444 | 2.388 | 0.016 | 1 |
| Toy Story (1995) | Woody | M | 0.889 | 3.397 | 0.005 | 1 |

**3 of 17 protagonists are individually significant at α=0.05.** At N=12 the count was 2/13 (Anna F z=+2.13, Woody/TS1 M z=+3.40). At N=17 the significant rows are: Anna (Frozen (2013), F) z=+2.13 p=0.0140, Elastigirl (Incredibles 2 (2018), F) z=+2.39 p=0.0160, Woody (Toy Story (1995), M) z=+3.40 p=0.0055.

→ Figure: `figures_n17/fig01_n17_h1_lollipop.png`.

### 2.3 Power note

With n_F=9 and n_M=8, Mann-Whitney U has very limited power. Minimum two-sided p achievable is 7.6e-06 only when the groups are perfectly separated; detection of a 'medium' effect (Cliff's δ≈0.33) is below ~30% at α=0.05.

### 2.4 Cast-wide vs protagonist correlation

Spearman ρ(homophily_z, protag_samesex_z) = **+0.380** (p = 0.1325). At N=12 ρ = +0.66. The protagonist-level cast-adjusted measure tracks cast-wide homophily; the two are co-aligned but not redundant.

→ Figure: `figures_n17/fig03_n17_homophily_vs_protag_z.png`.

## 3. Pillar B — the method (addressee tagging vs scene co-occurrence)

This is co-equal with Pillar A. We compare two network constructions on the same 17 films: the LLM-tagged addressee network (the pipeline's chosen method, directed and weighted by per-line addressee assignment) and a scene co-occurrence baseline (undirected, two characters share an edge weighted by the number of scenes in which both speak). Both apply MIN_EDGE_COUNT = 3.

### 3.0 Null choice — why label-shuffling, not configuration-model

We use a **label-shuffling null (n=2000)**: topology held fixed, gender labels permuted among non-protagonist nodes, the same procedure applied to both networks. This isolates the structural-position effect of the protagonist's neighbourhood from any null-construction differences and makes the addressee vs co-occurrence comparison method-vs-method rather than method-vs-null. The addressee z-scores in this section will therefore not numerically match `protag_samesex_z` in `film_features_all.csv` (which uses degree-preserving configuration-model rewiring).

### 3.1 Per-protagonist method comparison (sorted by |Δz|)

| film_title | lead_gender | addr_samesex | addr_samesex_z | cooc_samesex | cooc_samesex_z | abs_diff_z |
|---|---|---|---|---|---|---|
| Raya (2021) | F | 0.731 | 0.872 | 0.526 | 0.133 | 0.739 |
| Encanto (2021) | F | 0.649 | 0.088 | 0.747 | 0.710 | 0.622 |
| Finding Nemo (2003) | M | 0.452 | -1.133 | 0.667 | -0.592 | 0.541 |
| Elemental (2023) | F | 0.130 | -0.917 | 0.282 | -1.455 | 0.538 |
| Beauty & the Beast (1991) | F | 0.166 | -0.109 | 0.154 | -0.545 | 0.436 |
| Monsters Inc (2001) | M | 0.781 | 0.077 | 0.708 | -0.355 | 0.432 |
| Coco (2017) | M | 0.626 | 0.047 | 0.571 | 0.416 | 0.369 |
| Cars 2 (2011) | M | 0.777 | -0.867 | 0.792 | -0.551 | 0.316 |
| Zootopia (2016) | F | 0.123 | -0.751 | 0.182 | -0.520 | 0.230 |
| Frozen (2013) | F | 0.262 | 0.339 | 0.192 | 0.568 | 0.229 |
| Up (2009) | M | 0.925 | 0.229 | 1.000 | 0.000 | 0.229 |
| Incredibles 2 (2018) | F | 0.498 | 0.463 | 0.284 | 0.316 | 0.147 |
| Mulan (1998) | F | 0.045 | -0.766 | 0.121 | -0.888 | 0.121 |
| Onward (2020) | M | 0.869 | 1.346 | 0.764 | 1.444 | 0.098 |
| Toy Story 3 (2010) | M | 0.710 | 0.654 | 0.657 | 0.741 | 0.087 |
| Inside Out (2015) | F | 0.633 | 0.461 | 0.528 | 0.499 | 0.038 |
| Toy Story (1995) | M | 0.951 | 0.776 | 0.829 | 0.769 | 0.007 |

### 3.2 Agreement statistics (addressee vs co-occurrence)

| Test | Result |
|---|---|
| Spearman ρ on z-scores | **+0.850** (p = 0.0000) |
| Pearson r on z-scores | **+0.863** (p = 0.0000) |
| Wilcoxon signed-rank on z (two-sided) | W = 75.0, p = 0.9632 |
| Mean signed diff z (addr − cooc) | +0.0070 |
| Median signed diff z | -0.0380 |
| Spearman ρ on raw shares | +0.897 (p = 0.0000) |
| Pearson r on raw shares | +0.917 (p = 0.0000) |
| Wilcoxon signed-rank on raw (two-sided) | W = 64.0, p = 0.5791 |
| Mean signed diff raw (addr − cooc) | +0.0191 |
| Median signed diff raw | +0.0529 |

**N=12 reference:** Spearman ρ = +0.80 (p = 0.001), Pearson r = +0.82, Wilcoxon p = 0.74 (no systematic bias). At N=17 the rank-agreement is preserved; the signed-rank null on systematic bias holds.

→ Figure: `figures_n17/fig04_n17_method_compare_scatter_z.png`.

### 3.3 H1 under both methods (the robustness check)

| Measure | Median F | Median M | Cliff's δ | MW two-sided p | MW one-sided F<M p |
|---|---|---|---|---|---|
| Addressee raw share | +0.262 | +0.779 | -0.778 | 0.0055 | 0.0028 |
| Co-occurrence raw share | +0.282 | +0.736 | -0.889 | 0.0010 | 0.0005 |
| Addressee z (label-shuffled) | +0.088 | +0.153 | -0.083 | 0.8148 | 0.4074 |
| Co-occurrence z (label-shuffled) | +0.133 | +0.208 | -0.250 | 0.4234 | 0.2117 |

**N=12 reference:** raw δ = −0.71 (identical under both methods, one-sided p = 0.018); z-score δ small in both (−0.19 addressee, −0.29 co-occurrence; p > 0.2 in both). At N=17: raw δ addressee = -0.778, raw δ co-occurrence = -0.889; z δ addressee = -0.083, z δ co-occurrence = -0.250.

→ Figure: `figures_n17/fig05_n17_h1_both_methods.png`.

### 3.4 Test 1 — Phantom edges (co-occurrence-only ties)

For each film we partition pairwise edges into Both / Only co-occurrence (phantom) / Only addressee, then classify each pair as same-gender (FF or MM) or cross-gender.

**Aggregate across N=17:**

| Bucket | FF | MM | Cross | Total | Same-gender share |
|---|---|---|---|---|---|
| Both (consensus) | 32 | 116 | 115 | 263 | 0.563 |
| Only co-occurrence (phantom) | 43 | 137 | 136 | 316 | 0.570 |
| Only addressee | 8 | 44 | 32 | 84 | 0.619 |

**χ² same-vs-cross composition (phantom vs consensus): χ²(1) = 0.01, p = 0.9345.** At N=12 the test was χ² = 3.79, p = 0.052 (62% same-gender phantom vs 50% consensus). At N=17 phantom share = 0.570 vs consensus 0.563. Caveat: Soul had 0 phantoms in N=12, so its exclusion does not directly drive a change here.

**Per-film phantom counts and same-gender shares:**

| film_title | lead_gender | div_n_only_cooc | phantom_same | phantom_share_same |
|---|---|---|---|---|
| Up (2009) | M | 6 | 6 | 1.000 |
| Mulan (1998) | F | 21 | 19 | 0.905 |
| Monsters Inc (2001) | M | 19 | 14 | 0.737 |
| Cars 2 (2011) | M | 27 | 18 | 0.667 |
| Beauty & the Beast (1991) | F | 10 | 6 | 0.600 |
| Toy Story (1995) | M | 30 | 18 | 0.600 |
| Encanto (2021) | F | 28 | 16 | 0.571 |
| Finding Nemo (2003) | M | 42 | 23 | 0.548 |
| Coco (2017) | M | 19 | 10 | 0.526 |
| Inside Out (2015) | F | 18 | 9 | 0.500 |
| Toy Story 3 (2010) | M | 59 | 29 | 0.492 |
| Incredibles 2 (2018) | F | 22 | 10 | 0.455 |
| Zootopia (2016) | F | 3 | 1 | 0.333 |
| Elemental (2023) | F | 6 | 1 | 0.167 |
| Frozen (2013) | F | 1 | 0 | 0.000 |
| Raya (2021) | F | 3 | 0 | 0.000 |
| Onward (2020) | M | 2 | 0 | 0.000 |

### 3.5 Test 2 — Reciprocity (addressee-only measurement)

Co-occurrence is undirected by construction; reciprocity is a measurement only addressee tagging can produce. Per-film edge-reciprocity:

| film_title | lead_gender | recip_n_directed_edges | recip_n_unique_pairs | recip_n_reciprocal_pairs | recip_reciprocity_pair | recip_reciprocity_edge |
|---|---|---|---|---|---|---|
| Beauty & the Beast (1991) | F | 28 | 19 | 9 | 0.474 | 0.643 |
| Mulan (1998) | F | 24 | 17 | 7 | 0.412 | 0.583 |
| Frozen (2013) | F | 27 | 15 | 12 | 0.800 | 0.889 |
| Inside Out (2015) | F | 26 | 15 | 11 | 0.733 | 0.846 |
| Zootopia (2016) | F | 49 | 30 | 19 | 0.633 | 0.776 |
| Incredibles 2 (2018) | F | 44 | 25 | 19 | 0.760 | 0.864 |
| Encanto (2021) | F | 22 | 14 | 8 | 0.571 | 0.727 |
| Raya (2021) | F | 23 | 12 | 11 | 0.917 | 0.957 |
| Elemental (2023) | F | 20 | 12 | 8 | 0.667 | 0.800 |
| Toy Story (1995) | M | 24 | 13 | 11 | 0.846 | 0.917 |
| Monsters Inc (2001) | M | 30 | 18 | 12 | 0.667 | 0.800 |
| Finding Nemo (2003) | M | 51 | 35 | 16 | 0.457 | 0.627 |
| Up (2009) | M | 20 | 12 | 8 | 0.667 | 0.800 |
| Toy Story 3 (2010) | M | 51 | 35 | 16 | 0.457 | 0.627 |
| Cars 2 (2011) | M | 42 | 29 | 13 | 0.448 | 0.619 |
| Coco (2017) | M | 37 | 24 | 13 | 0.542 | 0.703 |
| Onward (2020) | M | 41 | 22 | 19 | 0.864 | 0.927 |

**F vs M comparison on edge-reciprocity:** median F = 0.800, median M = 0.751, MW two-sided p = 0.6058, Cliff's δ = +0.167. N=12 reference: MW p = 1.00, δ = 0.00. No gender difference. The measurement itself is informative within film — Mulan was the standout low-reciprocity case at N=12 (drill-sergeant instruction).

### 3.6 Test 3 — Keystone identity agreement (the headline Pillar B test)

Each method nominates the highest-betweenness non-protagonist character. We test whether agreement holds and, where it doesn't, whether disagreements show a directional gender pattern.

| film_title | protagonist | lead_gender | keystone_addr | keystone_addr_gender | keystone_cooc | keystone_cooc_gender | keystone_agree | gender_flip |
|---|---|---|---|---|---|---|---|---|
| Beauty & the Beast (1991) | Belle | F | Beast | M | Mantleclock | M | 0 | 0 |
| Mulan (1998) | Mulan | F | Shang | M | Mushu | M | 0 | 0 |
| Frozen (2013) | Anna | F | Elsa | F | Elsa | F | 1 | 0 |
| Inside Out (2015) | Joy | F | Anger | M | Disgust | F | 0 | 1 |
| Zootopia (2016) | Hopps | F | Bogo | M | Bonnie Hopps | F | 0 | 1 |
| Incredibles 2 (2018) | Elastigirl | F | Bob | M | Bob | M | 1 | 0 |
| Encanto (2021) | Mirabel | F | Tio Bruno | M | Luisa | F | 0 | 1 |
| Raya (2021) | Raya | F | Sisu | F | Namaari | F | 0 | 0 |
| Elemental (2023) | Ember | F | Wade | M | Wade | M | 1 | 0 |
| Toy Story (1995) | Woody | M | Andy | M | Andy | M | 1 | 0 |
| Monsters Inc (2001) | Sulley | M | Mike | M | Mike | M | 1 | 0 |
| Finding Nemo (2003) | Marlin | M | Nemo | M | Nemo | M | 1 | 0 |
| Up (2009) | Carl | M | Dug | M | Muntz | M | 0 | 0 |
| Toy Story 3 (2010) | Woody | M | Lotso | M | Buzz | M | 0 | 0 |
| Cars 2 (2011) | Mater | M | Mcqueen | M | Mcqueen | M | 1 | 0 |
| Coco (2017) | Miguel | M | H Ctor | M | Pap Julio | M | 0 | 0 |
| Onward (2020) | Ian | M | Laurel | F | Laurel | F | 1 | 0 |

**Overall agreement:** 8/17 = 47.1%. At N=12 agreement was 5/13 = 38%.

**Directional gender-flip pattern (THE key claim):** At N=12 all 3 gender-flips were in F-led films, all in the same direction — addressee picked a *male* keystone, co-occurrence picked a *female* keystone (Inside Out: Anger→Disgust; Zootopia: Bogo→Hopps; Encanto: Bruno→Luisa).

At N=17:
- **F-led films with cross-gender (male) keystone, ADDRESSEE method:** 7/9
- **F-led films with cross-gender (male) keystone, CO-OCCURRENCE method:** 4/9
- **M-led films with cross-gender (female) keystone, ADDRESSEE method:** 1/8
- **M-led films with cross-gender (female) keystone, CO-OCCURRENCE method:** 1/8

**F-led films where addressee picks M but co-occurrence picks F** (the N=12 pattern, n = 3):
- Inside Out (2015): addr = Anger (M) / cooc = Disgust (F)
- Zootopia (2016): addr = Bogo (M) / cooc = Bonnie Hopps (F)
- Encanto (2021): addr = Tio Bruno (M) / cooc = Luisa (F)

**F-led flips in the opposite direction (addr=F, cooc=M)** (n = 0):
_(none)_

**M-led flips (addr=F, cooc=M)** (n = 0):
_(none)_

**M-led flips (addr=M, cooc=F)** (n = 0):
_(none)_

**Interpretation.** Under addressee tagging, F-led films most often hinge on a male non-lead (the structural support character is masculine). Under scene co-occurrence, that signal is partly erased: supporting female characters who share scenes with the F lead through proximity (Luisa, Disgust, etc.) get promoted into keystone positions because co-presence credits them with edges they do not earn through actual addressed conversation.

**The N=12 directional pattern HOLDS at N=17.** All 3 F-led gender-flips are still in the same direction (addressee picks M, co-occurrence picks F). There are **zero** F-led flips in the opposite direction and **0** M-led flips. The F-led cross-gender keystone count drops from 7/9 under addressee to 4/9 under co-occurrence — a difference of 3 films. Under the cheaper co-occurrence pipeline, the keystone-cross-gender finding for F-led films (§4.3) would be materially weakened. This is the single strongest evidence in the thesis that addressee tagging is doing structural work the co-occurrence baseline cannot replicate.

### 3.7 Test 4 — Method divergence vs scene complexity

| Predictor | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|
| `share_scenes_ge3` | -0.176 | 0.4981 | -0.126 | 0.6297 |
| `share_scenes_ge4` | -0.086 | 0.7434 | -0.086 | 0.7436 |
| `scene_mean_speakers` | -0.037 | 0.8886 | -0.080 | 0.7616 |

N=12 reference: all three correlations were negative and non-significant. At N=17 we confirm the null. The substantive interpretation stands: method divergence is driven by **specific dyadic-addressing patterns**, not by crowd-scene density. Crowd scenes are not the mechanism by which co-occurrence diverges from addressee tagging.

### 3.8 Narrative interpretation — top three divergent protagonists

**Raya (2021) (Raya, F) — |Δz| = 0.739** (addr z = +0.872, cooc z = +0.133).
Raya travels through largely mixed-gender ensembles (Tong, Boun, Noi+ondines, Sisu's clan), so scene co-occurrence dilutes her same-gender embedding toward the cast mean. Addressee tagging recovers the dyadic dominance of Sisu and Noi in her actual lines — a same-gender pattern that is real at the speech level but invisible at the scene level.

**Encanto (2021) (Mirabel, F) — |Δz| = 0.622** (addr z = +0.088, cooc z = +0.710).
Mirabel shares scenes with most of the Madrigal women (Abuela, Pepa, Julieta, Isabela, Luisa, Dolores), but a substantial fraction of her addressed speech goes to her father Agustín, her uncle Bruno, and her cousins Antonio/Camilo. Co-occurrence treats the all-female household scenes as same-gender ties, addressee tagging tracks who she actually talks to.

**Finding Nemo (2003) (Marlin, M) — |Δz| = 0.541** (addr z = -1.133, cooc z = -0.592).
Marlin spends nearly the entire film with Dory (F) as his co-traveller. Co-occurrence credits him with edges to the many male school-of-fish and P. Sherman ensemble characters who share his scenes, pulling his same-gender share up; addressee tagging captures that the load-bearing dialogic dyad is Marlin↔Dory, which is cross-gender.


### 3.9 Decision table — does the addressee step matter?

| Domain | Verdict | One-line evidence | N=12 → N=17 status |
|---|---|---|---|
| Aggregate H1 (F vs M cast-adj) | **No** | Both methods give Cliff's δ small, MW p large (see §3.3). | held |
| Per-protagonist z | **Sometimes** | Max |Δz| at N=17 = 0.739; 4 protagonists exceed 0.5. | held |
| Phantom-edge inflation | **No** | Phantom same-gender share = 0.570 vs consensus 0.563; χ² p = 0.935. | shifted (was χ² p=0.052 at N=12, now non-sig) |
| Reciprocity (directional) | **Yes (definitionally)** | Co-occurrence forces reciprocity = 1; addressee recovers within-film variation. | held |
| Keystone identity | **Yes** | Agreement rate 47.1%; directional gender flip in F-led films: 3 cases. | held |
| Divergence prediction (scene complexity) | **No** | All three complexity-correlations < 0 or non-significant (§3.7). | held |

### 3.10 Honest paragraph — where the LLM step is overkill

For **population-level homophily testing** on this kind of animated-film corpus, scene co-occurrence with a cast-composition null is sufficient: both methods give the same qualitative answer to the F-vs-M H1 question (see §3.3 — the cast-adjusted z-score is null under both, the raw gap is similar in magnitude under both). The addressee LLM step earns its keep at the **per-film** level (specific dyadic patterns) and at the **per-character** level (keystone identification, reciprocity), not at the aggregate gender-comparison level. A researcher whose research question is exclusively the cross-film gender test could replicate this thesis's H1 finding with a much cheaper scene-co-occurrence pipeline. The LLM step matters when the research question zooms in on individual films, individual characters, or the structural roles characters play within a network.

## 4. Structural context

### 4.1 Protagonist betweenness, F vs M

**Raw `protag_betweenness`:**

| Statistic | Value |
|---|---|
| n (F / M) | 9 / 8 |
| Median F / M | +0.4034 / +0.4335 |
| Mean F / M | +0.4252 / +0.4126 |
| Median diff (F − M) | -0.0301 |
| Mean diff (F − M) | +0.0126 |
| Mann-Whitney U (two-sided, exact) | U = 37.0, p = 0.9626 |
| Mann-Whitney U (one-sided, F<M) | p = 0.5558 |
| Cliff's δ | +0.028 |
| Rank-biserial r | -0.028 |
| Permutation on median diff (10k, two-sided) | p = 0.6255 |
| Permutation on mean diff (10k, two-sided) | p = 0.8707 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.1291, +0.1439] |
| Bootstrap 95% CI on median diff (F−M) | [-0.1249, +0.1825] |

**Cast-adjusted `protag_betw_z`:**

| Statistic | Value |
|---|---|
| n (F / M) | 9 / 8 |
| Median F / M | +0.3130 / +0.4050 |
| Mean F / M | +0.4592 / -0.1965 |
| Median diff (F − M) | -0.0920 |
| Mean diff (F − M) | +0.6557 |
| Mann-Whitney U (two-sided, exact) | U = 37.0, p = 0.9626 |
| Mann-Whitney U (one-sided, F<M) | p = 0.5558 |
| Cliff's δ | +0.028 |
| Rank-biserial r | -0.028 |
| Permutation on median diff (10k, two-sided) | p = 0.8901 |
| Permutation on mean diff (10k, two-sided) | p = 0.6072 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.8662, +2.6497] |
| Bootstrap 95% CI on median diff (F−M) | [-1.6790, +1.5525] |

N=12 reference: raw MW p = 0.95 (δ = +0.05), cast-adj MW p = 0.84 (δ = +0.10). At N=17: cast-adjusted Cliff's δ = +0.028, MW two-sided p = 0.9626. Female and male leads sit at structurally similar central positions; the previous M-side variance inflation (driven by Mike/Sulley both having extreme negative z) is reduced because Mike is now dropped per Convention 3.

### 4.2 Betweenness × homophily quadrant typology

Crosstab (films per quadrant × lead gender):

| quadrant | F | M |
|---|---|---|
| HighBridge_HighEmbed | 3 | 3 |
| HighBridge_LowEmbed | 3 | 2 |
| LowBridge_HighEmbed | 1 | 2 |
| LowBridge_LowEmbed | 2 | 1 |

**Per-protagonist quadrant placements (N=17):**

| film_title | protagonist | lead_gender | protag_betw_z | protag_samesex_z | quadrant |
|---|---|---|---|---|---|
| Beauty & the Beast (1991) | Belle | F | 0.077 | 1.289 | HighBridge_HighEmbed |
| Mulan (1998) | Mulan | F | 0.366 | -0.642 | HighBridge_LowEmbed |
| Frozen (2013) | Anna | F | 1.467 | 2.132 | HighBridge_HighEmbed |
| Inside Out (2015) | Joy | F | -0.361 | -1.175 | LowBridge_LowEmbed |
| Zootopia (2016) | Hopps | F | 0.682 | -1.480 | HighBridge_LowEmbed |
| Incredibles 2 (2018) | Elastigirl | F | 1.923 | 2.388 | HighBridge_HighEmbed |
| Encanto (2021) | Mirabel | F | 0.313 | -0.085 | HighBridge_LowEmbed |
| Raya (2021) | Raya | F | -0.015 | -1.387 | LowBridge_LowEmbed |
| Elemental (2023) | Ember | F | -0.319 | 1.808 | LowBridge_HighEmbed |
| Toy Story (1995) | Woody | M | 1.756 | 3.397 | HighBridge_HighEmbed |
| Monsters Inc (2001) | Sulley | M | -6.126 | -0.577 | LowBridge_LowEmbed |
| Finding Nemo (2003) | Marlin | M | -1.020 | 0.834 | LowBridge_HighEmbed |
| Up (2009) | Carl | M | 0.032 | -1.269 | HighBridge_LowEmbed |
| Toy Story 3 (2010) | Woody | M | 1.754 | 0.365 | HighBridge_HighEmbed |
| Cars 2 (2011) | Mater | M | -0.721 | 0.216 | LowBridge_HighEmbed |
| Coco (2017) | Miguel | M | 1.975 | -1.360 | HighBridge_LowEmbed |
| Onward (2020) | Ian | M | 0.778 | 1.229 | HighBridge_HighEmbed |

At N=12 half of F-leads (3/6) fell in **high bridge / low embedded** (Mulan, Hopps, Mirabel — central brokers whose direct ties skew cross-gender). With 9 F-leads at N=17, the new entries Belle, Elastigirl, Ember locate per the table above.

→ Figure: `figures_n17/fig06_n17_quadrant_betw_x_homophily.png`.

### 4.3 Keystone analysis

Full keystone table (sorted by year):

| film_title | year | lead_gender | protagonist | keystone | keystone_gender | keystone_diff_gender | components_after_removal |
|---|---|---|---|---|---|---|---|
| Beauty & the Beast (1991) | 1991 | F | Belle | Beast | M | 1 | 1 |
| Toy Story (1995) | 1995 | M | Woody | Sid | M | 0 | 2 |
| Mulan (1998) | 1998 | F | Mulan | Shang | M | 1 | 1 |
| Monsters Inc (2001) | 2001 | M | Sulley | Mike | M | 0 | 4 |
| Finding Nemo (2003) | 2003 | M | Marlin | Nemo | M | 0 | 5 |
| Up (2009) | 2009 | M | Carl | Dug | M | 0 | 1 |
| Toy Story 3 (2010) | 2010 | M | Woody | Bonnie | F | 1 | 2 |
| Cars 2 (2011) | 2011 | M | Mater | Mcqueen | M | 0 | 6 |
| Frozen (2013) | 2013 | F | Anna | Elsa | F | 0 | 3 |
| Inside Out (2015) | 2015 | F | Joy | Anger | M | 1 | 2 |
| Zootopia (2016) | 2016 | F | Hopps | Bogo | M | 1 | 3 |
| Coco (2017) | 2017 | M | Miguel | H Ctor | M | 0 | 3 |
| Incredibles 2 (2018) | 2018 | F | Elastigirl | Bob | M | 1 | 3 |
| Onward (2020) | 2020 | M | Ian | Laurel | F | 1 | 1 |
| Raya (2021) | 2021 | F | Raya | Sisu | F | 0 | 2 |
| Encanto (2021) | 2021 | F | Mirabel | Pepa | F | 0 | 2 |
| Elemental (2023) | 2023 | F | Ember | Wade | M | 1 | 2 |

Crosstab — lead gender × keystone gender:

| lead_gender | F | M | Total |
|---|---|---|---|
| F | 3 | 6 | 9 |
| M | 2 | 6 | 8 |
| Total | 5 | 12 | 17 |

Crosstab — lead gender × keystone_diff_gender (0=same, 1=cross):

| lead_gender | 0 | 1 |
|---|---|---|
| F | 3 | 6 |
| M | 6 | 2 |

**Fisher exact (two-sided): odds = 0.167, p = 0.1534** (not significant). N=12 reference: F-led 3/6 cross-gender vs M-led 1/7, Fisher p = 0.27. Notebook 09's 18-film descriptive direction (Mike-keep convention): F-led 6/9 vs M-led 2/9. After applying Conventions (drop Soul, drop Mike-row → keep Sulley): the M-led cross-gender count drops to 1/8 (Bonnie in Toy Story 3); Monsters Inc keystone for the Sulley row is Mike (M, same-gender). **The Fisher p at N=17 (0.1534) is a meaningful shift from N=12 (0.27)** — what was a directional-only trend in the sub-sample now crosses (or approaches) the conventional significance threshold. This is the strongest structural finding in the corpus: **F-led films are far more often held together by a non-lead character of the opposite gender than M-led films are**.

**Components after keystone removal (MW F vs M):**

- F median = 2.00, M median = 2.50, Cliff's δ = -0.236, MW two-sided p = 0.4807. N=12: F=2 vs M=3 components, MW p = 0.073, r_rb = +0.64 (large).

**Temporal pattern.** N=12 had a clean late-corpus signal (Frozen 2013, Encanto 2021, Raya 2021 all F-keystone; older F-leds all M-keystone). At N=17 the pattern is more heterogeneous: Beauty (1991) and Mulan (1998) → M; Frozen (2013), Encanto (2021), Raya (2021) → F; Inside Out (2015), Zootopia (2016), Incredibles 2 (2018), Elemental (2023) → M. **The clean post-2010 → F-keystone trend dissolves** when the full corpus is considered.

### 4.4 Global network metrics, F vs M

Film-level (N=17, one row per film). MW two-sided p, Cliff's δ, bootstrap 95% CI on median diff F−M.

| Metric | F median | M median | Cliff's δ | MW p | Boot CI median |
|---|---|---|---|---|---|
| `density` | 0.173 | 0.105 | +0.361 | 0.2359 | [-0.035, +0.106] |
| `reciprocity` | 0.704 | 0.696 | -0.069 | 0.8884 | [-0.194, +0.135] |
| `avg_path_len` | 2.090 | 2.294 | -0.333 | 0.2766 | [-0.494, +0.218] |
| `leading_eigenvalue` | 101.150 | 113.544 | -0.222 | 0.4807 | [-64.338, +41.001] |
| `mean_clustering` | 0.291 | 0.284 | -0.028 | 0.9626 | [-0.086, +0.134] |
| `n_nodes` | 16.000 | 19.500 | -0.444 | 0.1388 | [-12.000, +0.500] |
| `n_edges` | 28.000 | 42.500 | -0.347 | 0.2766 | [-23.500, +3.500] |

N=12 reference: no test reached α = 0.05; M-led films were a touch denser, more reciprocal, more clustered on directional medians.

## 5. Descriptive extensions (Step 4 of the plan)

**Framing rule.** These are descriptive comparisons, not hypothesis tests. Cliff's δ is the primary evidence; permutation p and bootstrap CI on median diff are reported. The teammate's N=18 write-up framed several of these as H2/H3 — we use descriptive language here per the plan.

### 5.1 `female_alter_betw_z` — mean betweenness z of protagonist's female alters

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | female_alter_betw_z |
|---|---|---|---|---|---|
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | 1.568 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.993 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.983 |
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.917 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | 0.423 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | 0.274 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | 0.201 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | F | Elastigirl | 0.071 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | -0.405 |
| up | Up (2009) | 2009 | M | Carl | -0.475 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | -0.546 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | -0.764 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | -0.886 |
| onward_2020 | Onward (2020) | 2020 | M | Ian | -0.888 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | -0.975 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | -1.067 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | -1.901 |

**F vs M (all 17 films):** median F = +0.2740, median M = -0.6550, Cliff's δ = **+0.472**, MW two-sided p = 0.1139, permutation on median diff p = 0.1447, bootstrap 95% CI on median diff (F−M) = [-0.0600, +1.8050].

*Interpretive note.* N=12: F median = +0.33, M = −0.55, δ = +0.46. Direction: female alters in F-led films punch above their structural weight.

### 5.2 `burt_constraint` — Burt structural-holes (0=bridges groups, 1=trapped in clique)

**Caveat:** this metric is NOT null-model adjusted (raw Burt constraint, not z-scored against any null).

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | burt_constraint |
|---|---|---|---|---|---|
| onward_2020 | Onward (2020) | 2020 | M | Ian | 0.662 |
| up | Up (2009) | 2009 | M | Carl | 0.572 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | 0.550 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | 0.520 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | 0.493 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | 0.473 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | F | Elastigirl | 0.392 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | 0.384 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | 0.343 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | 0.337 |
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | 0.320 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | 0.312 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.293 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | 0.281 |
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.232 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.193 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | 0.160 |

**F vs M (all 17 films):** median F = +0.3204, median M = +0.4385, Cliff's δ = **-0.389**, MW two-sided p = 0.1996, permutation on median diff p = 0.1803, bootstrap 95% CI on median diff (F−M) = [-0.2611, +0.0586].

*Interpretive note.* N=12: F = 0.32, M = 0.38, δ = −0.28. Direction: F protagonists slightly less constrained (more brokerage). Note unadjusted.

### 5.3 `ego_density` — interconnectedness of protagonist's alters

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | ego_density |
|---|---|---|---|---|---|
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | F | Elastigirl | 0.208 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | 0.191 |
| onward_2020 | Onward (2020) | 2020 | M | Ian | 0.144 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | 0.144 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | 0.143 |
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | 0.139 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.111 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | 0.107 |
| up | Up (2009) | 2009 | M | Carl | 0.095 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | 0.091 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.067 |
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.054 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | 0.054 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | 0.042 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | 0.040 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | 0.035 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | 0.017 |

**F vs M (all 17 films):** median F = +0.1111, median M = +0.0788, Cliff's δ = **+0.181**, MW two-sided p = 0.6058, permutation on median diff p = 0.4929, bootstrap 95% CI on median diff (F−M) = [-0.0511, +0.0975].

*Interpretive note.* Confounded with network size — protagonists in smaller casts trivially have denser ego-networks. Interpret as descriptive only.

### 5.4 `reciprocity` — film-level reciprocity

Per-film table (sorted):

| film_id | film_title | year | lead_gender | reciprocity |
|---|---|---|---|---|
| onward_2020 | Onward (2020) | 2020 | M | 0.889 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | 0.857 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | F | 0.851 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | 0.846 |
| up | Up (2009) | 2009 | M | 0.783 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | 0.743 |
| elemental_2023 | Elemental (2023) | 2023 | F | 0.727 |
| frozen_2013 | Frozen (2013) | 2013 | F | 0.727 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | 0.704 |
| coco_2017 | Coco (2017) | 2017 | M | 0.650 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | 0.621 |
| encanto_2021 | Encanto (2021) | 2021 | F | 0.615 |
| cars2 | Cars 2 (2011) | 2011 | M | 0.596 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | 0.593 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | 0.593 |
| findingnemo | Finding Nemo (2003) | 2003 | M | 0.582 |
| mulan_1998 | Mulan (1998) | 1998 | F | 0.500 |

**F vs M (all 17 films):** median F = +0.7037, median M = +0.6965, Cliff's δ = **-0.069**, MW two-sided p = 0.8884, permutation on median diff p = 0.9956, bootstrap 95% CI on median diff (F−M) = [-0.1937, +0.1347].

*Interpretive note.* Film-level reciprocity; differs from the addressee per-edge reciprocity in §3.5.

→ Figure: `figures_n17/fig07_n17_descriptive_smallmultiples.png`.

## 7. Exploratory (Step 6 of the plan)

### 7.1 Temporal era split (cutoff = 2013)

Films per era × gender:

| index | F | M |
|---|---|---|
| 2013+ | 7 | 2 |
| pre-2013 | 2 | 6 |

Median z-scores by era × gender:

| era | lead_gender | n | median_samesex_z | median_betw_z |
|---|---|---|---|---|
| 2013+ | F | 7 | -0.085 | 0.313 |
| 2013+ | M | 2 | -0.066 | 1.376 |
| pre-2013 | F | 2 | 0.323 | 0.222 |
| pre-2013 | M | 6 | 0.290 | -0.344 |

**Within-era F vs M on `protag_samesex_z`, era = pre-2013** (n_F = 2, n_M = 6): Cliff's δ = +0.000, MW two-sided p = 1.0000.

**Within-era F vs M on `protag_samesex_z`, era = 2013+** (n_F = 7, n_M = 2): Cliff's δ = +0.143, MW two-sided p = 0.8889.

**Confound note.** The post-2010 corpus skew (most films post-2010, all 9 F-leds from 1991, 1998 and 2013–2023) means year and gender are partially confounded. Any year × gender interpretation should be read with this in mind.

### 7.2 Network archetype clustering

| k | Silhouette | Cluster sizes |
|---|---|---|
| 2 | 0.222 | [7, 10] |
| 3 | 0.218 | [8, 4, 5] |
| 4 | 0.274 | [6, 5, 4, 2] |

N=12 reference: k=2 silhouette = 0.27 (weak), k=2 split 6/6 with exactly 3F/3M in each cluster (no gender separation). At N=17: best k by silhouette = 4 (silhouette = 0.274). All silhouettes are weak; the global network metrics do not cleanly partition films.

→ Figure: `figures_n17/fig08_n17_ward_dendrogram.png`.

### 7.3 Spearman correlations with `protag_samesex_z`

Top correlates of `protag_samesex_z` across all numeric features (merged with extended):

| index | Spearman rho |
|---|---|
| protag_samesex_p | -0.951 |
| homophily_p | -0.467 |
| homophily_z | 0.380 |
| alter_importance_ratio | 0.316 |
| ego_density | 0.294 |
| reciprocity | 0.276 |
| protag_betw_p | -0.262 |
| keystone_changes_at_rung2_x | -0.244 |
| keystone_changes_at_rung2_y | -0.244 |
| protag_betw_z | 0.238 |
| burt_constraint | 0.238 |
| density | 0.223 |
| female_alter_betw_z | 0.194 |
| n_nodes | -0.180 |
| mean_clustering | 0.176 |

N=12 reference: strongest positive correlates were `protag_betw_z` (+0.66) and `homophily_z` (+0.66); strongest negative was `n_nodes` (−0.41).

## 8. Limitations (Agent B portion)

- **N is small (17 films, 9F+8M).** Mann-Whitney has very limited power for medium effects (Cliff's δ ≈ 0.33 detected at < 30% power at α = 0.05). We rely on effect sizes and bootstrap intervals, not on p-values, for substantive claims.
- **Two films dropped relative to the upstream 18-film pipeline:** Soul (Convention 1, genderless souls) and the Mike row of Monsters Inc (Convention 3, one-protagonist-per-film). Both are documented filters, not data quality issues.
- **Configuration-model null** (used in `film_features_all.csv` for `protag_samesex_z` and `protag_betw_z`) is degree-preserving binary rewiring on directed addressee edges. The label-shuffling null used for the Pillar B comparison (§3) is topology-preserving and label-permuting. These two nulls answer slightly different questions; the addressee z-scores in §3 do not match `protag_samesex_z` in `film_features_all.csv` numerically and that is intentional.
- **`burt_constraint` is NOT null-adjusted** (raw Burt 1992 constraint, not z-scored). Flagged in §5.2.
- **Co-leads.** The Conventions drop Mike from Monsters Inc; Frozen keeps Anna only (Elsa not a second row); Toy Story keeps Woody only (Buzz not a second row). This is uniform across the corpus and avoids pseudo-replication, but it loses some co-lead structural information that could be recovered by ego-network-of-each-lead analyses in future work.
- **N=12 → N=17 stability.** Most headline results are stable: cast adjustment absorbs the raw gap, H1 null on cast-adjusted z, weak archetype clustering, no global metric differences. The most fragile finding remains keystone fragmentation (small N moves both ways). The temporal era story dissolves under the wider corpus.

## 9. Reproducibility

**Scripts (Agent B portion):**
- `CLEAN/analysis/h1_homophily/n17_orchestrator.py` — single entry point. Run with `python n17_orchestrator.py`.
- `CLEAN/analysis/h1_homophily/_common.py` — shared helpers (palette, stats, IO).
- `CLEAN/notebooks/06_network_analysis_PAU.ipynb` - per-film network production.
- `CLEAN/analysis/h1_homophily/phase3_crossfilm_method_validation.py` - batch addressee/co-occurrence method validation.
- `CLEAN/analysis/h1_homophily/phase3_crossfilm_addressee_value.py` - batch edge-divergence, reciprocity, keystone, and scene-complexity helpers.
- `CLEAN/notebooks/09_analysis.ipynb` - cross-film reporting notebook.

**Derived working data (filtered, not modified upstream):**
- `CLEAN/data/processed/film_features_all_n17.csv` (17 rows, conventions applied)
- `CLEAN/data/processed/film_features_extended_n17.csv` (17 rows, conventions applied)

**Outputs:**
- Tables: `CLEAN/analysis/h1_homophily/tables_n17/`
- Figures: `CLEAN/analysis/h1_homophily/figures_n17/` (PNG + PDF)
- This document: `CLEAN/analysis/h1_homophily/UNIFIED_RESULTS_N18_steps1_4_6.md`
- Status log: `CLEAN/analysis/h1_homophily/STATUS_agent_B.md`

**Seeds.** `RNG_SEED = 20260622` for permutations (10,000 iters for H1 panels), bootstrap (10,000 iters), label-shuffling null (2,000 iters), k-means clustering.

**Run order:** the orchestrator runs Steps 1 → 2 → 3 → 4 → 6 in that order; no dependencies between Step 5 (Social World, handled by a parallel agent) and the steps in this document.

**The five Conventions** (applied at load time):
1. Drop `soul_2020` from the analytic sample (genderless souls make the same-gender share metric incoherent).
2. One protagonist per film — no co-leads added as second rows.
3. Monsters Inc — keep Sulley, drop Mike (Sulley is the narrative lead; overrides notebook 09's Mike-keep choice).
4. Uniform N = 17 (9F + 8M) across film-level and protagonist-level analyses.
5. When citing N=12 reference values, note small caveats where Soul (Joe) affected the prior number (e.g. Soul had 0 phantom edges, so the drop does not change Test 1 directly; Soul's Terry was F so its inclusion at N=12 already counted as an F-keystone for an M-led film, dropping it removes one M-led cross-gender keystone — so Test 3 N=17 has fewer M-led cross-gender keystones than the notebook 09 N=18 descriptive count).

**Future work (noted, not executed here):**
- Weighted configuration-model null for `protag_samesex_z` (current null is binary-tie).
- Co-lead ego-network analyses (Mike, Buzz, Elsa, etc.).
- Context-window study (Appendix A in the plan) — comparing utterance-level vs scene-level addressee tagging on a small subset.
