# Notebooks Pipeline Map

Interactive, per-film layer of the project. Numbered roughly in run order. Batch
equivalents live in `../scripts/`; cross-film H1 analysis lives in
`../analysis/h1_homophily/` (see its `PIPELINE_README.md`).

Paths inside the notebooks resolve `CLEAN = Path('..')`, so **run them with the
working directory set to `notebooks/`**.

Data locations:
- `../data/00_corpus/` — corpus inputs
- `../data/02_processed/{film_id}/` — per-film artifacts
- `../data/04_features/` — cross-film feature tables (`film_features_*.csv`)

---

## ⚠️ Gotchas before you run

- **Re-running `09_analysis.ipynb`?** Its first cell only runs
  `n17_orchestrator.py` **if `../analysis/h1_homophily/tables_n17/step2f_keystone_agreement.csv` is missing.**
  If the file exists it **skips regeneration and uses the (possibly stale) tables**.
  → To force fresh results, **delete that file** (or the whole `tables_n17/`) first:
  ```bash
  rm CLEAN/analysis/h1_homophily/tables_n17/step2f_keystone_agreement.csv
  ```
- **Do NOT re-run `02_Validation_Gold_set.ipynb`.** The addressee gold set
  (nb 02–04) is frozen at 270 rows / 18 films. Re-running it would change the
  frozen validation set.

---

## A. Method validation track (run once, then frozen)

Validates the addressee-tagging method on a hand-labelled gold set before it is
applied to all films.

| nb | purpose |
|---|---|
| `02_Validation_Gold_set.ipynb` | Build the addressee gold set. **Frozen — do not re-run.** |
| `03_validation_addressee_prediction.ipynb` | Run the addressee predictor on the validation set. |
| `04_inter_annotator_agreement.ipynb` | Inter-annotator agreement (addressee & addressee type). |

## B. Production track (per film → features)

| nb | purpose | writes |
|---|---|---|
| `01_parse_scenes_and_characters.ipynb` | Parse screenplay → scenes + characters. (`character_review_*.csv` are hand-annotated.) | `data/02_processed/{film}/scenes.csv`, `utterances.csv` |
| `05_addressee_tagging_scene_level.ipynb` | Tag who each line is spoken to, all films. | `data/02_processed/{film}/utterances_with_addressee_scene.csv` |
| `06_network_analysis_PAU.ipynb` | Build the directed dialogue network; compute homophily / centrality / removal. One film at a time (batch via `scripts/run_06_all_films.py`). | per-film network files + upserts `data/04_features/film_features_all.csv` |
| `07_compute_extended_features.ipynb` | Extended ego-network metrics (wraps `scripts/compute_extended_features.py`). | `data/04_features/film_features_extended.csv` |
| `08_compute_social_features.ipynb` | Social-world metrics (wraps `scripts/build_film_features_social.py`). | `data/04_features/film_features_social.csv` |

## C. Analysis & reporting (N=17)

All read the `_n17` working tables from `data/04_features/` (derived by
`n17_orchestrator.py`: Soul dropped, one lead per film).

| nb | purpose |
|---|---|
| `09_analysis.ipynb` | Main H1 reporting: cast-adjusted homophily + method validation. Auto-runs the orchestrator if `tables_n17/` is missing (see gotcha above). |
| `09_analysis_appendix.ipynb` | Supporting analyses for 09. |
| `10_descriptive_only.ipynb` | Descriptive context beyond H1 (cast composition, gender dialogue flows, temporal F→F, most-addressed alter). |
| `11_min_edge_sensitivity.ipynb` | Sensitivity check on `MIN_EDGE_COUNT = 3`. |

---

## Flow

```
screenplay
  → 01 parse ─┐
              │   (gold set: 02 → 03 → 04, frozen)
  → 05 addressee tagging
  → 06 network analysis ──────────────→ film_features_all.csv
        ├─ 07 extended ───────────────→ film_features_extended.csv
        └─ 08 social ─────────────────→ film_features_social.csv
  → analysis/h1_homophily/n17_orchestrator.py → *_n17.csv (in data/04_features/)
  → 09_analysis / 09_analysis_appendix / 10_descriptive_only / 11   (reporting)
```
