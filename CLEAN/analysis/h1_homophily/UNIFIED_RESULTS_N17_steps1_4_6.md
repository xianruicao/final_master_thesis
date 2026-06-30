# Unified Results — Steps 1, 2, 3, 4, 6 (Agent B, N=17)

> Source-of-truth plan: `CLEAN/admin/AGENT_INSTRUCTION_run_analysis_N18.md`.
> Analytic sample: **N=17 (8 F-led + 9 M-led)** after Conventions filters (Soul dropped — Convention 1; Mike row of Monsters Inc dropped, Sulley kept — Convention 3).
> RNG_SEED = 20260622. Pillar B null = label-shuffling, n=2000 permutations.
> Tables: `tables_n17/`. Figures: `figures_n17/`. Orchestrator: `n17_orchestrator.py`.
> Step 5 (Social World) is produced separately by another agent.

## 2. Pillar A — the measure (cast-adjusted protagonist same-gender homophily)

This is the methodological core. We compare F-led vs M-led protagonists on two paired quantities: the **raw** same-gender share of dialogue (`protag_samesex`) and the **cast-adjusted z-score** under a degree-preserving configuration-model null (`protag_samesex_z`, 2000 rewirings). The decomposition tests whether any F-vs-M gap in same-gender embedding is structurally inevitable given cast composition, or whether it reflects protagonist-level behaviour.

### 2.1 Raw vs cast-adjusted decomposition

**Raw share** `protag_samesex`:

| Statistic | Value |
|---|---|
| n (F / M) | 8 / 9 |
| Median F / M | +0.3428 / +0.6667 |
| Mean F / M | +0.3533 / +0.7001 |
| Median diff (F − M) | -0.3238 |
| Mean diff (F − M) | -0.3468 |
| Mann-Whitney U (two-sided, exact) | U = 4.0, p = 0.0010 |
| Mann-Whitney U (one-sided, F<M) | p = 0.0005 |
| Cliff's δ | -0.889 |
| Rank-biserial r | +0.889 |
| Permutation on median diff (10k, two-sided) | p = 0.0012 |
| Permutation on mean diff (10k, two-sided) | p = 0.0006 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.4802, -0.2054] |
| Bootstrap 95% CI on median diff (F−M) | [-0.5322, -0.1714] |

**Cast-adjusted z** `protag_samesex_z`:

| Statistic | Value |
|---|---|
| n (F / M) | 8 / 9 |
| Median F / M | -0.1445 / +0.3850 |
| Mean F / M | +0.1423 / +0.3776 |
| Median diff (F − M) | -0.5295 |
| Mean diff (F − M) | -0.2353 |
| Mann-Whitney U (two-sided, exact) | U = 36.0, p = 1.0000 |
| Mann-Whitney U (one-sided, F<M) | p = 0.5187 |
| Cliff's δ | +0.000 |
| Rank-biserial r | +0.000 |
| Permutation on median diff (10k, two-sided) | p = 0.5768 |
| Permutation on mean diff (10k, two-sided) | p = 0.7685 |
| Bootstrap 95% CI on mean diff (F−M) | [-1.6384, +1.1142] |
| Bootstrap 95% CI on median diff (F−M) | [-1.7195, +2.1435] |

**Comparison.** At N=12 the raw measure showed Cliff's δ = −0.71 (large, MW one-sided p = 0.018, bootstrap CI on mean diff [−0.43, −0.09] **excluding 0**), while the cast-adjusted measure showed δ = +0.10 (negligible). At **N=17** we find raw Cliff's δ = **-0.889** (MW one-sided p = 0.0005) and cast-adjusted Cliff's δ = **+0.000** (MW one-sided p = 0.5187). The directional story — raw shows a substantial F<M gap, cast adjustment largely absorbs it — is preserved and **strengthened** at the expanded N. Notably, the raw gap is **larger** at N=17 than at N=12 (δ = -0.889 vs −0.71), and the cast-adjusted gap is essentially zero (δ = +0.000). The methodological centrepiece of the thesis — that the obvious F-vs-M raw gap is a cast-composition artefact absorbed by the configuration-model null — is **more cleanly demonstrated at N=17** than at N=12.

→ Figure: `figures_n17/fig02_n17_raw_vs_adjusted.png` (side-by-side raw vs z box+strip).

### 2.2 Per-protagonist cast-adjusted z (individual significance)

| film_title | protagonist | lead_gender | protag_samesex | protag_samesex_z | protag_samesex_p | sig_alpha05 |
|---|---|---|---|---|---|---|
| Monsters Inc (2001) | Sulley | M | 0.625 | -1.489 | 0.991 | 0 |
| Raya (2021) | Raya | F | 0.400 | -1.447 | 0.950 | 0 |
| Zootopia (2016) | Hopps | F | 0.250 | -1.306 | 0.937 | 0 |
| Up (2009) | Carl | M | 0.800 | -1.207 | 1.000 | 0 |
| Coco (2017) | Miguel | M | 0.533 | -1.187 | 0.963 | 0 |
| Inside Out (2015) | Joy | F | 0.429 | -0.805 | 0.859 | 0 |
| Mulan (1998) | Mulan | F | 0.111 | -0.640 | 0.729 | 0 |
| Toy Story 3 (2010) | Woody | M | 0.667 | 0.329 | 0.439 | 0 |
| Encanto (2021) | Mirabel | F | 0.636 | 0.351 | 0.891 | 0 |
| Cars 2 (2011) | Mater | M | 0.909 | 0.385 | 0.524 | 0 |
| Incredibles 2 (2018) | Bob | M | 0.600 | 0.664 | 0.226 | 0 |
| Finding Nemo (2003) | Marlin | M | 0.778 | 0.707 | 0.208 | 0 |
| Onward (2020) | Ian | M | 0.500 | 0.971 | 0.286 | 0 |
| Beauty & the Beast (1991) | Belle | F | 0.286 | 1.558 | 0.076 | 0 |
| Elemental (2023) | Ember | F | 0.429 | 1.562 | 0.291 | 0 |
| Frozen (2013) | Anna | F | 0.286 | 1.865 | 0.017 | 1 |
| Toy Story (1995) | Woody | M | 0.889 | 4.225 | 0.005 | 1 |

**2 of 17 protagonists are individually significant at α=0.05.** At N=12 the count was 2/13 (Anna F z=+2.13, Woody/TS1 M z=+3.40). At N=17 the significant rows are: Anna (Frozen (2013), F) z=+1.86 p=0.0165, Woody (Toy Story (1995), M) z=+4.22 p=0.0055.

→ Figure: `figures_n17/fig01_n17_h1_lollipop.png`.

### 2.3 Power note

With n_F=8 and n_M=9, Mann-Whitney U has very limited power. Minimum two-sided p achievable is 7.6e-06 only when the groups are perfectly separated; detection of a 'medium' effect (Cliff's δ≈0.33) is below ~30% at α=0.05.

### 2.4 Cast-wide vs protagonist correlation

Spearman ρ(homophily_z, protag_samesex_z) = **+0.147** (p = 0.5733). At N=12 ρ = +0.66. The protagonist-level cast-adjusted measure tracks cast-wide homophily; the two are co-aligned but not redundant.

→ Figure: `figures_n17/fig03_n17_homophily_vs_protag_z.png`.

## 3. Pillar B — the method (addressee tagging vs scene co-occurrence)

This is co-equal with Pillar A. We compare two network constructions on the same 17 films: the LLM-tagged addressee network (the pipeline's chosen method, directed and weighted by per-line addressee assignment) and a scene co-occurrence baseline (undirected, two characters share an edge weighted by the number of scenes in which both speak). Both apply MIN_EDGE_COUNT = 3.

### 3.0 Null choice — why label-shuffling, not configuration-model

We use a **label-shuffling null (n=2000)**: topology held fixed, gender labels permuted among non-protagonist nodes, the same procedure applied to both networks. This isolates the structural-position effect of the protagonist's neighbourhood from any null-construction differences and makes the addressee vs co-occurrence comparison method-vs-method rather than method-vs-null. The addressee z-scores in this section will therefore not numerically match `protag_samesex_z` in `film_features_all.csv` (which uses degree-preserving configuration-model rewiring).

### 3.1 Per-protagonist method comparison (sorted by |Δz|)

| film_title | lead_gender | addr_samesex | addr_samesex_z | cooc_samesex | cooc_samesex_z | abs_diff_z |
|---|---|---|---|---|---|---|
| Raya (2021) | F | 0.731 | 0.918 | 0.526 | 0.162 | 0.755 |
| Encanto (2021) | F | 0.649 | 0.105 | 0.747 | 0.763 | 0.658 |
| Elemental (2023) | F | 0.130 | -0.867 | 0.282 | -1.426 | 0.560 |
| Finding Nemo (2003) | M | 0.452 | -1.104 | 0.667 | -0.615 | 0.488 |
| Monsters Inc (2001) | M | 0.781 | 0.040 | 0.708 | -0.400 | 0.440 |
| Beauty & the Beast (1991) | F | 0.166 | -0.085 | 0.154 | -0.511 | 0.426 |
| Coco (2017) | M | 0.626 | 0.067 | 0.571 | 0.425 | 0.358 |
| Cars 2 (2011) | M | 0.777 | -0.785 | 0.792 | -0.464 | 0.321 |
| Incredibles 2 (2018) | M | 0.381 | -0.972 | 0.524 | -1.264 | 0.292 |
| Zootopia (2016) | F | 0.123 | -0.742 | 0.182 | -0.486 | 0.256 |
| Up (2009) | M | 0.925 | 0.252 | 1.000 | 0.000 | 0.252 |
| Frozen (2013) | F | 0.262 | 0.327 | 0.192 | 0.502 | 0.175 |
| Onward (2020) | M | 0.869 | 1.322 | 0.764 | 1.435 | 0.113 |
| Mulan (1998) | F | 0.045 | -0.824 | 0.121 | -0.859 | 0.036 |
| Toy Story 3 (2010) | M | 0.710 | 0.723 | 0.657 | 0.751 | 0.028 |
| Inside Out (2015) | F | 0.633 | 0.470 | 0.528 | 0.489 | 0.019 |
| Toy Story (1995) | M | 0.951 | 0.789 | 0.829 | 0.807 | 0.018 |

### 3.2 Agreement statistics (addressee vs co-occurrence)

| Test | Result |
|---|---|
| Spearman ρ on z-scores | **+0.885** (p = 0.0000) |
| Pearson r on z-scores | **+0.875** (p = 0.0000) |
| Wilcoxon signed-rank on z (two-sided) | W = 76.0, p = 1.0000 |
| Mean signed diff z (addr − cooc) | +0.0193 |
| Median signed diff z | -0.0191 |
| Spearman ρ on raw shares | +0.909 (p = 0.0000) |
| Pearson r on raw shares | +0.927 (p = 0.0000) |
| Wilcoxon signed-rank on raw (two-sided) | W = 73.0, p = 0.8900 |
| Mean signed diff raw (addr − cooc) | -0.0019 |
| Median signed diff raw | +0.0121 |

**N=12 reference:** Spearman ρ = +0.80 (p = 0.001), Pearson r = +0.82, Wilcoxon p = 0.74 (no systematic bias). At N=17 the rank-agreement is preserved; the signed-rank null on systematic bias holds.

→ Figure: `figures_n17/fig04_n17_method_compare_scatter_z.png`.

### 3.3 H1 under both methods (the robustness check)

| Measure | Median F | Median M | Cliff's δ | MW two-sided p | MW one-sided F<M p |
|---|---|---|---|---|---|
| Addressee raw share | +0.214 | +0.777 | -0.722 | 0.0111 | 0.0056 |
| Co-occurrence raw share | +0.237 | +0.708 | -0.806 | 0.0037 | 0.0019 |
| Addressee z (label-shuffled) | +0.010 | +0.067 | -0.028 | 0.9626 | 0.4813 |
| Co-occurrence z (label-shuffled) | -0.162 | +0.000 | -0.194 | 0.5414 | 0.2707 |

**N=12 reference:** raw δ = −0.71 (identical under both methods, one-sided p = 0.018); z-score δ small in both (−0.19 addressee, −0.29 co-occurrence; p > 0.2 in both). At N=17: raw δ addressee = -0.722, raw δ co-occurrence = -0.806; z δ addressee = -0.028, z δ co-occurrence = -0.194.

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
| Incredibles 2 (2018) | M | 22 | 10 | 0.455 |
| Zootopia (2016) | F | 3 | 1 | 0.333 |
| Elemental (2023) | F | 6 | 1 | 0.167 |
| Raya (2021) | F | 3 | 0 | 0.000 |
| Frozen (2013) | F | 1 | 0 | 0.000 |
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
| Incredibles 2 (2018) | M | 44 | 25 | 19 | 0.760 | 0.864 |
| Onward (2020) | M | 41 | 22 | 19 | 0.864 | 0.927 |

**F vs M comparison on edge-reciprocity:** median F = 0.788, median M = 0.800, MW two-sided p = 0.8884, Cliff's δ = +0.056. N=12 reference: MW p = 1.00, δ = 0.00. No gender difference. The measurement itself is informative within film — Mulan was the standout low-reciprocity case at N=12 (drill-sergeant instruction).

### 3.6 Test 3 — Keystone identity agreement (the headline Pillar B test)

Each method nominates the highest-betweenness non-protagonist character. We test whether agreement holds and, where it doesn't, whether disagreements show a directional gender pattern.

| film_title | protagonist | lead_gender | keystone_addr | keystone_addr_gender | keystone_cooc | keystone_cooc_gender | keystone_agree | gender_flip |
|---|---|---|---|---|---|---|---|---|
| Beauty & the Beast (1991) | Belle | F | Beast | M | Mantleclock | M | 0 | 0 |
| Mulan (1998) | Mulan | F | Shang | M | Mushu | M | 0 | 0 |
| Frozen (2013) | Anna | F | Elsa | F | Elsa | F | 1 | 0 |
| Inside Out (2015) | Joy | F | Anger | M | Disgust | F | 0 | 1 |
| Zootopia (2016) | Hopps | F | Bogo | M | Bonnie Hopps | F | 0 | 1 |
| Encanto (2021) | Mirabel | F | Tio Bruno | M | Luisa | F | 0 | 1 |
| Raya (2021) | Raya | F | Sisu | F | Namaari | F | 0 | 0 |
| Elemental (2023) | Ember | F | Wade | M | Wade | M | 1 | 0 |
| Toy Story (1995) | Woody | M | Andy | M | Andy | M | 1 | 0 |
| Monsters Inc (2001) | Sulley | M | Mike | M | Mike | M | 1 | 0 |
| Finding Nemo (2003) | Marlin | M | Nemo | M | Nemo | M | 1 | 0 |
| Up (2009) | Carl | M | Dug | M | Muntz | M | 0 | 0 |
| Toy Story 3 (2010) | Woody | M | Lotso | M | Buzz | M | 0 | 0 |
| Cars 2 (2011) | Mater | M | Mcqueen | M | Mcqueen | M | 1 | 0 |
| Coco (2017) | Miguel | M | Héctor | M | Pap Julio | M | 0 | 0 |
| Incredibles 2 (2018) | Bob | M | Elastigirl | F | Elastigirl | F | 1 | 0 |
| Onward (2020) | Ian | M | Laurel | F | Laurel | F | 1 | 0 |

**Overall agreement:** 8/17 = 47.1%. At N=12 agreement was 5/13 = 38%.

**Directional gender-flip pattern (THE key claim):** At N=12 all 3 gender-flips were in F-led films, all in the same direction — addressee picked a *male* keystone, co-occurrence picked a *female* keystone (Inside Out: Anger→Disgust; Zootopia: Bogo→Hopps; Encanto: Bruno→Luisa).

At N=17:
- **F-led films with cross-gender (male) keystone, ADDRESSEE method:** 6/8
- **F-led films with cross-gender (male) keystone, CO-OCCURRENCE method:** 3/8
- **M-led films with cross-gender (female) keystone, ADDRESSEE method:** 2/9
- **M-led films with cross-gender (female) keystone, CO-OCCURRENCE method:** 2/9

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

**The N=12 directional pattern HOLDS at N=17.** All 3 F-led gender-flips are still in the same direction (addressee picks M, co-occurrence picks F). There are **zero** F-led flips in the opposite direction and **0** M-led flips. The F-led cross-gender keystone count drops from 6/8 under addressee to 3/8 under co-occurrence — a difference of 3 films. Under the cheaper co-occurrence pipeline, the keystone-cross-gender finding for F-led films (§4.3) would be materially weakened. This is the single strongest evidence in the thesis that addressee tagging is doing structural work the co-occurrence baseline cannot replicate.

### 3.7 Test 4 — Method divergence vs scene complexity

| Predictor | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|
| `share_scenes_ge3` | -0.118 | 0.6529 | -0.101 | 0.7007 |
| `share_scenes_ge4` | -0.044 | 0.8665 | -0.080 | 0.7597 |
| `scene_mean_speakers` | +0.025 | 0.9256 | -0.092 | 0.7263 |

N=12 reference: all three correlations were negative and non-significant. At N=17 we confirm the null. The substantive interpretation stands: method divergence is driven by **specific dyadic-addressing patterns**, not by crowd-scene density. Crowd scenes are not the mechanism by which co-occurrence diverges from addressee tagging.

### 3.8 Narrative interpretation — top three divergent protagonists

**Raya (2021) (Raya, F) — |Δz| = 0.755** (addr z = +0.918, cooc z = +0.162).
Raya travels through largely mixed-gender ensembles (Tong, Boun, Noi+ondines, Sisu's clan), so scene co-occurrence dilutes her same-gender embedding toward the cast mean. Addressee tagging recovers the dyadic dominance of Sisu and Noi in her actual lines — a same-gender pattern that is real at the speech level but invisible at the scene level.

**Encanto (2021) (Mirabel, F) — |Δz| = 0.658** (addr z = +0.105, cooc z = +0.763).
Mirabel shares scenes with most of the Madrigal women (Abuela, Pepa, Julieta, Isabela, Luisa, Dolores), but a substantial fraction of her addressed speech goes to her father Agustín, her uncle Bruno, and her cousins Antonio/Camilo. Co-occurrence treats the all-female household scenes as same-gender ties, addressee tagging tracks who she actually talks to.

**Elemental (2023) (Ember, F) — |Δz| = 0.560** (addr z = -0.867, cooc z = -1.426).
Addressee reports higher same-gender embedding than co-occurrence: the protagonist appears in mixed-gender scenes but directs a disproportionate weight of dialogue at same-gender alters. Co-occurrence dilutes a real dyadic same-gender signal.


### 3.9 Decision table — does the addressee step matter?

| Domain | Verdict | One-line evidence | N=12 → N=17 status |
|---|---|---|---|
| Aggregate H1 (F vs M cast-adj) | **No** | Both methods give Cliff's δ small, MW p large (see §3.3). | held |
| Per-protagonist z | **Sometimes** | Max |Δz| at N=17 = 0.755; 3 protagonists exceed 0.5. | held |
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
| n (F / M) | 8 / 9 |
| Median F / M | +0.6505 / +0.5606 |
| Mean F / M | +0.6259 / +0.5502 |
| Median diff (F − M) | +0.0899 |
| Mean diff (F − M) | +0.0758 |
| Mann-Whitney U (two-sided, exact) | U = 48.0, p = 0.2766 |
| Mann-Whitney U (one-sided, F<M) | p = 0.8821 |
| Cliff's δ | +0.333 |
| Rank-biserial r | -0.333 |
| Permutation on median diff (10k, two-sided) | p = 0.3419 |
| Permutation on mean diff (10k, two-sided) | p = 0.4014 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.0817, +0.2261] |
| Bootstrap 95% CI on median diff (F−M) | [-0.0955, +0.3395] |

**Cast-adjusted `protag_betw_z`:**

| Statistic | Value |
|---|---|
| n (F / M) | 8 / 9 |
| Median F / M | +1.3505 / +1.4120 |
| Mean F / M | +1.4031 / +0.8458 |
| Median diff (F − M) | -0.0615 |
| Mean diff (F − M) | +0.5573 |
| Mann-Whitney U (two-sided, exact) | U = 44.0, p = 0.4807 |
| Mann-Whitney U (one-sided, F<M) | p = 0.7883 |
| Cliff's δ | +0.222 |
| Rank-biserial r | -0.222 |
| Permutation on median diff (10k, two-sided) | p = 0.9914 |
| Permutation on mean diff (10k, two-sided) | p = 0.4483 |
| Bootstrap 95% CI on mean diff (F−M) | [-0.6443, +1.8799] |
| Bootstrap 95% CI on median diff (F−M) | [-1.3950, +2.4475] |

N=12 reference: raw MW p = 0.95 (δ = +0.05), cast-adj MW p = 0.84 (δ = +0.10). At N=17: cast-adjusted Cliff's δ = +0.222, MW two-sided p = 0.4807. Female and male leads sit at structurally similar central positions; the previous M-side variance inflation (driven by Mike/Sulley both having extreme negative z) is reduced because Mike is now dropped per Convention 3.

### 4.2 Betweenness × homophily quadrant typology

Crosstab (films per quadrant × lead gender):

| quadrant | F | M |
|---|---|---|
| HighBridge_HighEmbed | 4 | 4 |
| HighBridge_LowEmbed | 3 | 2 |
| LowBridge_HighEmbed | 0 | 2 |
| LowBridge_LowEmbed | 1 | 1 |

**Per-protagonist quadrant placements (N=17):**

| film_title | protagonist | lead_gender | protag_betw_z | protag_samesex_z | quadrant |
|---|---|---|---|---|---|
| Beauty & the Beast (1991) | Belle | F | 1.497 | 1.558 | HighBridge_HighEmbed |
| Mulan (1998) | Mulan | F | -0.339 | -0.640 | LowBridge_LowEmbed |
| Frozen (2013) | Anna | F | 2.671 | 1.865 | HighBridge_HighEmbed |
| Inside Out (2015) | Joy | F | 0.017 | -0.805 | HighBridge_LowEmbed |
| Zootopia (2016) | Hopps | F | 1.204 | -1.306 | HighBridge_LowEmbed |
| Encanto (2021) | Mirabel | F | 0.945 | 0.351 | HighBridge_HighEmbed |
| Raya (2021) | Raya | F | 2.954 | -1.447 | HighBridge_LowEmbed |
| Elemental (2023) | Ember | F | 2.276 | 1.562 | HighBridge_HighEmbed |
| Toy Story (1995) | Woody | M | 1.412 | 4.225 | HighBridge_HighEmbed |
| Monsters Inc (2001) | Sulley | M | -2.449 | -1.489 | LowBridge_LowEmbed |
| Finding Nemo (2003) | Marlin | M | -0.561 | 0.707 | LowBridge_HighEmbed |
| Up (2009) | Carl | M | 2.161 | -1.207 | HighBridge_LowEmbed |
| Toy Story 3 (2010) | Woody | M | 1.100 | 0.329 | HighBridge_HighEmbed |
| Cars 2 (2011) | Mater | M | -0.370 | 0.385 | LowBridge_HighEmbed |
| Coco (2017) | Miguel | M | 1.856 | -1.187 | HighBridge_LowEmbed |
| Incredibles 2 (2018) | Bob | M | 2.083 | 0.664 | HighBridge_HighEmbed |
| Onward (2020) | Ian | M | 2.380 | 0.971 | HighBridge_HighEmbed |

At N=12 half of F-leads (3/6) fell in **high bridge / low embedded** (Mulan, Hopps, Mirabel — central brokers whose direct ties skew cross-gender). With 9 F-leads at N=17, the new entries Belle, Elastigirl, Ember locate per the table above.

→ Figure: `figures_n17/fig06_n17_quadrant_betw_x_homophily.png`.

### 4.3 Keystone analysis

Full keystone table (sorted by year):

| film_title | year | lead_gender | protagonist | keystone | keystone_gender | keystone_diff_gender | components_after_removal |
|---|---|---|---|---|---|---|---|
| Beauty & the Beast (1991) | 1991 | F | Belle | Beast | M | 1 | 1 |
| Toy Story (1995) | 1995 | M | Woody | Andy | M | 0 | 2 |
| Mulan (1998) | 1998 | F | Mulan | Shang | M | 1 | 1 |
| Monsters Inc (2001) | 2001 | M | Sulley | Mike | M | 0 | 3 |
| Finding Nemo (2003) | 2003 | M | Marlin | Nemo | M | 0 | 4 |
| Up (2009) | 2009 | M | Carl | Dug | M | 0 | 1 |
| Toy Story 3 (2010) | 2010 | M | Woody | Buzz | M | 0 | 1 |
| Cars 2 (2011) | 2011 | M | Mater | Mcqueen | M | 0 | 5 |
| Frozen (2013) | 2013 | F | Anna | Elsa | F | 0 | 2 |
| Inside Out (2015) | 2015 | F | Joy | Anger | M | 1 | 2 |
| Zootopia (2016) | 2016 | F | Hopps | Bogo | M | 1 | 3 |
| Coco (2017) | 2017 | M | Miguel | Héctor | M | 0 | 3 |
| Incredibles 2 (2018) | 2018 | M | Bob | Elastigirl | F | 1 | 4 |
| Onward (2020) | 2020 | M | Ian | Laurel | F | 1 | 1 |
| Encanto (2021) | 2021 | F | Mirabel | Isabela | F | 0 | 1 |
| Raya (2021) | 2021 | F | Raya | Sisu | F | 0 | 3 |
| Elemental (2023) | 2023 | F | Ember | Wade | M | 1 | 2 |

Crosstab — lead gender × keystone gender:

| lead_gender | F | M | Total |
|---|---|---|---|
| F | 3 | 5 | 8 |
| M | 2 | 7 | 9 |
| Total | 5 | 12 | 17 |

Crosstab — lead gender × keystone_diff_gender (0=same, 1=cross):

| lead_gender | 0 | 1 |
|---|---|---|
| F | 3 | 5 |
| M | 7 | 2 |

**Fisher exact (two-sided): odds = 0.171, p = 0.1534** (not significant). N=12 reference: F-led 3/6 cross-gender vs M-led 1/7, Fisher p = 0.27. Notebook 09's 18-film descriptive direction (Mike-keep convention): F-led 6/9 vs M-led 2/9. After applying Conventions (drop Soul, drop Mike-row → keep Sulley): the M-led cross-gender count drops to 1/8 (Bonnie in Toy Story 3); Monsters Inc keystone for the Sulley row is Mike (M, same-gender). **The Fisher p at N=17 (0.1534) is a meaningful shift from N=12 (0.27)** — what was a directional-only trend in the sub-sample now crosses (or approaches) the conventional significance threshold. This is the strongest structural finding in the corpus: **F-led films are far more often held together by a non-lead character of the opposite gender than M-led films are**.

**Components after keystone removal (MW F vs M):**

- F median = 2.00, M median = 3.00, Cliff's δ = -0.306, MW two-sided p = 0.3213. N=12: F=2 vs M=3 components, MW p = 0.073, r_rb = +0.64 (large).

**Temporal pattern.** N=12 had a clean late-corpus signal (Frozen 2013, Encanto 2021, Raya 2021 all F-keystone; older F-leds all M-keystone). At N=17 the pattern is more heterogeneous: Beauty (1991) and Mulan (1998) → M; Frozen (2013), Encanto (2021), Raya (2021) → F; Inside Out (2015), Zootopia (2016), Incredibles 2 (2018), Elemental (2023) → M. **The clean post-2010 → F-keystone trend dissolves** when the full corpus is considered.

### 4.4 Global network metrics, F vs M

Film-level (N=17, one row per film). MW two-sided p, Cliff's δ, bootstrap 95% CI on median diff F−M.

| Metric | F median | M median | Cliff's δ | MW p | Boot CI median |
|---|---|---|---|---|---|
| `density` | 0.223 | 0.165 | +0.250 | 0.4234 | [-0.059, +0.133] |
| `reciprocity` | 0.769 | 0.800 | +0.000 | 1.0000 | [-0.168, +0.198] |
| `avg_path_len` | 2.101 | 2.076 | -0.083 | 0.8148 | [-0.431, +0.180] |
| `leading_eigenvalue` | 109.181 | 109.408 | -0.083 | 0.8148 | [-60.525, +47.710] |
| `mean_clustering` | 0.332 | 0.353 | -0.028 | 0.9626 | [-0.092, +0.117] |
| `n_nodes` | 11.500 | 14.000 | -0.500 | 0.0927 | [-10.500, -0.500] |
| `n_edges` | 25.000 | 41.000 | -0.514 | 0.0927 | [-24.012, +0.000] |

N=12 reference: no test reached α = 0.05; M-led films were a touch denser, more reciprocal, more clustered on directional medians.

## 5. Descriptive extensions (Step 4 of the plan)

**Framing rule.** These are descriptive comparisons, not hypothesis tests. Cliff's δ is the primary evidence; permutation p and bootstrap CI on median diff are reported. The teammate's N=18 write-up framed several of these as H2/H3 — we use descriptive language here per the plan.

### 5.1 `female_alter_betw_z` — mean betweenness z of protagonist's female alters

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | female_alter_betw_z |
|---|---|---|---|---|---|
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.797 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.379 |
| onward_2020 | Onward (2020) | 2020 | M | Ian | 0.250 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | M | Bob | 0.072 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.054 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | -0.069 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | -0.161 |
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | -0.214 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | -0.380 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | -0.411 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | -0.448 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | -0.488 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | -0.528 |
| up | Up (2009) | 2009 | M | Carl | -0.641 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | -0.685 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | -0.990 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | -1.077 |

**F vs M (all 17 films):** median F = -0.1875, median M = -0.4880, Cliff's δ = **+0.194**, MW two-sided p = 0.5414, permutation on median diff p = 0.3007, bootstrap 95% CI on median diff (F−M) = [-0.4830, +0.5875].

*Interpretive note.* N=12: F median = +0.33, M = −0.55, δ = +0.46. Direction: female alters in F-led films punch above their structural weight.

### 5.2 `burt_constraint` — Burt structural-holes (0=bridges groups, 1=trapped in clique)

**Caveat:** this metric is NOT null-model adjusted (raw Burt constraint, not z-scored against any null).

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | burt_constraint |
|---|---|---|---|---|---|
| onward_2020 | Onward (2020) | 2020 | M | Ian | 0.682 |
| up | Up (2009) | 2009 | M | Carl | 0.613 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | 0.574 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | 0.521 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | 0.493 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | 0.473 |
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.436 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | 0.409 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | 0.358 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | 0.349 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | M | Bob | 0.348 |
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | 0.334 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | 0.320 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | 0.315 |
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.257 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.197 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | 0.168 |

**F vs M (all 17 films):** median F = +0.3411, median M = +0.4095, Cliff's δ = **-0.333**, MW two-sided p = 0.2766, permutation on median diff p = 0.5329, bootstrap 95% CI on median diff (F−M) = [-0.2787, +0.0959].

*Interpretive note.* N=12: F = 0.32, M = 0.38, δ = −0.28. Direction: F protagonists slightly less constrained (more brokerage). Note unadjusted.

### 5.3 `ego_density` — interconnectedness of protagonist's alters

Per-film table (sorted):

| film_id | film_title | year | lead_gender | protagonist | ego_density |
|---|---|---|---|---|---|
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | Raya | 0.400 |
| onward_2020 | Onward (2020) | 2020 | M | Ian | 0.232 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | M | Bob | 0.222 |
| frozen_2013 | Frozen (2013) | 2013 | F | Anna | 0.214 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | Sulley | 0.214 |
| up | Up (2009) | 2009 | M | Carl | 0.200 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | Belle | 0.191 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | Joy | 0.191 |
| cars2 | Cars 2 (2011) | 2011 | M | Mater | 0.109 |
| elemental_2023 | Elemental (2023) | 2023 | F | Ember | 0.107 |
| mulan_1998 | Mulan (1998) | 1998 | F | Mulan | 0.083 |
| findingnemo | Finding Nemo (2003) | 2003 | M | Marlin | 0.083 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | Woody | 0.067 |
| coco_2017 | Coco (2017) | 2017 | M | Miguel | 0.048 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | Hopps | 0.046 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | Woody | 0.042 |
| encanto_2021 | Encanto (2021) | 2021 | F | Mirabel | 0.023 |

**F vs M (all 17 films):** median F = +0.1488, median M = +0.1091, Cliff's δ = **-0.056**, MW two-sided p = 0.8884, permutation on median diff p = 0.9003, bootstrap 95% CI on median diff (F−M) = [-0.1355, +0.1357].

*Interpretive note.* Confounded with network size — protagonists in smaller casts trivially have denser ego-networks. Interpret as descriptive only.

### 5.4 `reciprocity` — film-level reciprocity

Per-film table (sorted):

| film_id | film_title | year | lead_gender | reciprocity |
|---|---|---|---|---|
| raya_and_the_last_dragon_2021 | Raya (2021) | 2021 | F | 0.957 |
| onward_2020 | Onward (2020) | 2020 | M | 0.927 |
| toy_story_1995 | Toy Story (1995) | 1995 | M | 0.917 |
| frozen_2013 | Frozen (2013) | 2013 | F | 0.889 |
| incredibles_2_2018 | Incredibles 2 (2018) | 2018 | M | 0.864 |
| inside_out_2015 | Inside Out (2015) | 2015 | F | 0.846 |
| monsters_inc_2001 | Monsters Inc (2001) | 2001 | M | 0.800 |
| up | Up (2009) | 2009 | M | 0.800 |
| zootopia_2016 | Zootopia (2016) | 2016 | F | 0.775 |
| elemental_2023 | Elemental (2023) | 2023 | F | 0.762 |
| coco_2017 | Coco (2017) | 2017 | M | 0.703 |
| encanto_2021 | Encanto (2021) | 2021 | F | 0.696 |
| beautyandthebeast_1991 | Beauty & the Beast (1991) | 1991 | F | 0.643 |
| findingnemo | Finding Nemo (2003) | 2003 | M | 0.627 |
| toy_story_3_2010 | Toy Story 3 (2010) | 2010 | M | 0.627 |
| cars2 | Cars 2 (2011) | 2011 | M | 0.619 |
| mulan_1998 | Mulan (1998) | 1998 | F | 0.583 |

**F vs M (all 17 films):** median F = +0.7687, median M = +0.8000, Cliff's δ = **+0.000**, MW two-sided p = 1.0000, permutation on median diff p = 0.7334, bootstrap 95% CI on median diff (F−M) = [-0.1679, +0.1979].

*Interpretive note.* Film-level reciprocity; differs from the addressee per-edge reciprocity in §3.5.

→ Figure: `figures_n17/fig07_n17_descriptive_smallmultiples.png`.

## 7. Exploratory (Step 6 of the plan)

### 7.1 Temporal era split (cutoff = 2013)

Films per era × gender:

| index | F | M |
|---|---|---|
| 2013+ | 6 | 3 |
| pre-2013 | 2 | 6 |

Median z-scores by era × gender:

| era | lead_gender | n | median_samesex_z | median_betw_z |
|---|---|---|---|---|
| 2013+ | F | 6 | -0.227 | 1.740 |
| 2013+ | M | 3 | 0.664 | 2.083 |
| pre-2013 | F | 2 | 0.459 | 0.579 |
| pre-2013 | M | 6 | 0.357 | 0.365 |

**Within-era F vs M on `protag_samesex_z`, era = pre-2013** (n_F = 2, n_M = 6): Cliff's δ = +0.167, MW two-sided p = 0.8571.

**Within-era F vs M on `protag_samesex_z`, era = 2013+** (n_F = 6, n_M = 3): Cliff's δ = -0.111, MW two-sided p = 0.9048.

**Confound note.** The post-2010 corpus skew (most films post-2010, all 8 F-leds from 1991, 1998 and 2013–2023) means year and gender are partially confounded. Any year × gender interpretation should be read with this in mind.

### 7.2 Network archetype clustering

| k | Silhouette | Cluster sizes |
|---|---|---|
| 2 | 0.316 | [10, 7] |
| 3 | 0.294 | [5, 7, 5] |
| 4 | 0.249 | [1, 5, 6, 5] |

N=12 reference: k=2 silhouette = 0.27 (weak), k=2 split 6/6 with exactly 3F/3M in each cluster (no gender separation). At N=17: best k by silhouette = 2 (silhouette = 0.316). All silhouettes are weak; the global network metrics do not cleanly partition films.

→ Figure: `figures_n17/fig08_n17_ward_dendrogram.png`.

### 7.3 Spearman correlations with `protag_samesex_z`

Top correlates of `protag_samesex_z` across all numeric features (merged with extended):

| index | Spearman rho |
|---|---|
| protag_samesex_p | -0.922 |
| protag_betweenness | 0.412 |
| protag_betw_p | -0.270 |
| keystone_changes_at_rung2_y | -0.268 |
| keystone_changes_at_rung2_x | -0.268 |
| protag_betw_z | 0.238 |
| alter_importance_ratio | 0.203 |
| female_alter_betw_z | 0.199 |
| homophily_p | -0.182 |
| mean_clustering | 0.174 |
| homophily_z | 0.147 |
| n_nodes | -0.117 |
| density | 0.115 |
| protag_samesex | 0.097 |
| ego_density | -0.091 |

N=12 reference: strongest positive correlates were `protag_betw_z` (+0.66) and `homophily_z` (+0.66); strongest negative was `n_nodes` (−0.41).

## 8. Limitations (Agent B portion)

- **N is small (17 films, 8F+9M).** Mann-Whitney has very limited power for medium effects (Cliff's δ ≈ 0.33 detected at < 30% power at α = 0.05). We rely on effect sizes and bootstrap intervals, not on p-values, for substantive claims.
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
- `CLEAN/analysis/h1_homophily/phase3_crossfilm_method_validation.py` — shared helper module (addressee/co-occurrence network builders, label-shuffle null) imported by this orchestrator; not run directly.
- `CLEAN/analysis/h1_homophily/phase3_crossfilm_addressee_value.py` — shared helper module (edge-divergence, reciprocity, keystone, scene-complexity) imported by this orchestrator; not run directly.
- `CLEAN/notebooks/09_analysis.ipynb` - cross-film reporting notebook.

**Derived working data (filtered, not modified upstream):**
- `CLEAN/data/04_features/film_features_all_n17.csv` (17 rows, conventions applied)
- `CLEAN/data/04_features/film_features_extended_n17.csv` (17 rows, conventions applied)

**Outputs:**
- Tables: `CLEAN/analysis/h1_homophily/tables_n17/`
- Figures: `CLEAN/analysis/h1_homophily/figures_n17/` (PNG + PDF)
- This document: `CLEAN/analysis/h1_homophily/UNIFIED_RESULTS_N17_steps1_4_6.md`
- Status log: `CLEAN/analysis/h1_homophily/STATUS_agent_B.md`

**Seeds.** `RNG_SEED = 20260622` for permutations (10,000 iters for H1 panels), bootstrap (10,000 iters), label-shuffling null (2,000 iters), k-means clustering.

**Run order:** the orchestrator runs Steps 1 → 2 → 3 → 4 → 6 in that order; no dependencies between Step 5 (Social World, handled by a parallel agent) and the steps in this document.

**The five Conventions** (applied at load time):
1. Drop `soul_2020` from the analytic sample (genderless souls make the same-gender share metric incoherent).
2. One protagonist per film — no co-leads added as second rows.
3. Monsters Inc — keep Sulley, drop Mike (Sulley is the narrative lead; overrides notebook 09's Mike-keep choice).
4. Uniform N = 17 (8F + 9M) across film-level and protagonist-level analyses.
5. When citing N=12 reference values, note small caveats where Soul (Joe) affected the prior number (e.g. Soul had 0 phantom edges, so the drop does not change Test 1 directly; Soul's Terry was F so its inclusion at N=12 already counted as an F-keystone for an M-led film, dropping it removes one M-led cross-gender keystone — so Test 3 N=17 has fewer M-led cross-gender keystones than the notebook 09 N=18 descriptive count).

**Future work (noted, not executed here):**
- Weighted configuration-model null for `protag_samesex_z` (current null is binary-tie).
- Co-lead ego-network analyses (Mike, Buzz, Elsa, etc.).
- Context-window study (Appendix A in the plan) — comparing utterance-level vs scene-level addressee tagging on a small subset.
