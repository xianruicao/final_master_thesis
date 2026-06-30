# H1 Homophily Pipeline Map

This folder contains the cross-film H1 homophily analysis. The important
distinction is per-film production versus batch analysis versus reporting.

## Layer 1: Per-Film Network Production

`CLEAN/notebooks/06_network_analysis_PAU.ipynb`

Runs one film at a time. It builds the directed weighted addressee network and
writes per-film files under `CLEAN/data/02_processed/{film_id}/`, including:

- `network_edges.csv`
- `network_node_summary.csv`
- `protagonist_metrics.csv`
- `film_features.csv`

Use `CLEAN/scripts/run_06_all_films.py` to execute Notebook 06 across the N=17
film set.

## Layer 2: Batch Cross-Film Analysis

`CLEAN/analysis/h1_homophily/n17_orchestrator.py`

The single entry point for the cross-film batch. Runs the N=17 pipeline and writes
`tables_n17/`, `figures_n17/`, and the unified results markdown. It also derives the
analysis-ready working tables `film_features_all_n17.csv` /
`film_features_extended_n17.csv` into `CLEAN/data/04_features/` (Soul dropped, one
lead per film). Run this — not the modules below — to reproduce the thesis results.

### Shared helper modules (imported, not run directly)

`CLEAN/analysis/h1_homophily/phase3_crossfilm_method_validation.py`
`CLEAN/analysis/h1_homophily/phase3_crossfilm_addressee_value.py`

These two files are **helper libraries**, not standalone analyses. They provide the
per-film network functions that consume Notebook 06 outputs — building the addressee
network and the scene co-occurrence baseline, the label-shuffle null, and the
addressee-value tests (edge divergence, reciprocity, keystone agreement, scene
complexity). `n17_orchestrator.py` (and `notebooks/09_analysis_appendix.ipynb`)
`import` these functions and apply them to the N=17 sample.

> Do **not** run these files directly. Each still carries a legacy `main()` that
> executes on the full 18-film master (Soul included, N=18) and writes `phase3v2_*` /
> `phase3v3_*` tables and `fig15`–`fig20` to `figures/`. That standalone output is
> superseded by the orchestrator's N=17 steps and is **not** the thesis result. Only
> the imported helper functions are current.

## Layer 3: Reporting

`CLEAN/notebooks/09_analysis.ipynb`

Reads cross-film batch outputs and produces thesis-facing figures,
interpretation, and prose. It should not be treated as the source of per-film
network construction logic.
