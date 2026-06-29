# H1 Homophily Pipeline Map

This folder contains the cross-film H1 homophily analysis. The important
distinction is per-film production versus batch analysis versus reporting.

## Layer 1: Per-Film Network Production

`CLEAN/notebooks/06_network_analysis_PAU.ipynb`

Runs one film at a time. It builds the directed weighted addressee network and
writes per-film files under `CLEAN/data/processed/{film_id}/`, including:

- `network_edges.csv`
- `network_node_summary.csv`
- `protagonist_metrics.csv`
- `film_features.csv`

Use `CLEAN/scripts/run_06_all_films.py` to execute Notebook 06 across the N=17
film set.

## Layer 2: Batch Cross-Film Analysis

`CLEAN/analysis/h1_homophily/phase3_crossfilm_method_validation.py`

Consumes Notebook 06 per-film outputs and compares addressee networks against a
scene co-occurrence baseline for protagonist same-gender homophily.

`CLEAN/analysis/h1_homophily/phase3_crossfilm_addressee_value.py`

Consumes the same per-film outputs and tests where addressee tagging adds
structural information beyond scene co-occurrence: edge divergence, reciprocity,
keystone disagreement, and scene complexity.

`CLEAN/analysis/h1_homophily/n17_orchestrator.py`

Runs the N=17 cross-film batch pipeline and writes `tables_n17/`, `figures_n17/`,
and the unified results markdown.

## Layer 3: Reporting

`CLEAN/notebooks/09_analysis.ipynb`

Reads cross-film batch outputs and produces thesis-facing figures,
interpretation, and prose. It should not be treated as the source of per-film
network construction logic.
