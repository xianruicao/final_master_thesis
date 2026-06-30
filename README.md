# Gender Homophily in Animated-Film Dialogue Networks

Master's thesis codebase. We turn animated-film screenplays into directed
**dialogue networks** (who speaks to whom) and test **H1: do protagonists talk to their own gender more than the cast composition alone would predict?**, across a corpus of **N = 17** Disney / Pixar films.

The pipeline: screenplay text → scene & character parsing → LLM addressee tagging → per-film networks → cross-film feature tables → cast-adjusted homophily analysis.

---

## Repository layout

Everything lives under `CLEAN/`.

```
CLEAN/
├── data/
│   ├── 00_corpus/         corpus definition: final_films.csv (the 17 films +
│   │                      parse config) and the rebuilt characters.csv master
│   ├── 01_raw_scripts/    raw screenplay .txt files (one per film)
│   ├── 02_processed/      per-film artifacts — one folder per film
│   │                      (scenes, utterances, addressee-tagged lines,
│   │                       network_edges, null models, figures)
│   ├── 03_validation_dataset/  addressee gold-set inputs + results (frozen)
│   └── 04_features/        cross-film feature tables (film_features_*.csv)
│                           + their README (data dictionary & lineage)
├── notebooks/             interactive per-film pipeline + reporting
│                          → see notebooks/PIPELINE_README.md
├── scripts/               batch / helper scripts (run the steps over all films)
│                          → see scripts/PIPELINE_README.md
└── analysis/
    └── h1_homophily/      cross-film H1 analysis, the n17 orchestrator, and
                           generated tables_n17/ + figures_n17/
                           → see analysis/h1_homophily/PIPELINE_README.md
```

Each layer has its own README; this file is the entry point and reproduction guide.

---

## Environment

- **Python 3.11** (see `.python-version`)
- Dependencies are managed with [**uv**](https://docs.astral.sh/uv/) (`pyproject.toml`).

```bash
uv sync
```

This installs everything the notebooks, scripts, and analysis layer need.

### API keys (required for addressee tagging)

Addressee tagging (notebooks 03 & 05) calls LLM APIs. Create a `.env` file in the
repo root — it is git-ignored:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

You only need keys if you re-run the tagging steps; the tagged outputs are already
checked in under `data/02_processed/`.

---

## Reproducing the work, end to end

Run notebooks with the working directory set to `CLEAN/notebooks/`. Batch
equivalents in `CLEAN/scripts/` are noted in brackets.

| # | step | run | output |
|---|---|---|---|
| 1 | Parse scenes & characters | `01_parse_scenes_and_characters.ipynb`  [`scripts/regen_all_films.py`] | `data/02_processed/{film}/scenes.csv`, `utterances.csv`; `data/00_corpus/characters.csv` |
| 2 | Addressee tagging (LLM) | `05_addressee_tagging_scene_level.ipynb` | `data/02_processed/{film}/utterances_with_addressee_scene.csv` |
| 3 | Build per-film networks | `06_network_analysis_PAU.ipynb`  [`scripts/run_06_all_films.py`] | per-film network files + `data/04_features/film_features_all.csv` |
| 4 | Extended + social features | `07_compute_extended_features.ipynb`, `08_compute_social_features.ipynb` | `data/04_features/film_features_extended.csv`, `film_features_social.csv` |
| 5 | Cross-film H1 analysis | `python analysis/h1_homophily/n17_orchestrator.py` | `analysis/h1_homophily/tables_n17/`, `figures_n17/`, and `data/04_features/film_features_*_n17.csv` |
| 6 | Reporting / figures | `09_analysis.ipynb`, `09_analysis_appendix.ipynb`, `10_descriptive_only.ipynb`, `11_min_edge_sensitivity.ipynb` | thesis figures & tables |

**Method validation (optional, already frozen):** notebooks
`02_Validation_Gold_set` → `03_validation_addressee_prediction` →
`04_inter_annotator_agreement` validate the addressee tagger against a hand-labelled
gold set (270 rows / 18 films).

### ⚠️ Two things that will bite you

1. **Re-running `09_analysis.ipynb`** only regenerates results if
   `analysis/h1_homophily/tables_n17/step2f_keystone_agreement.csv` is **missing**.
   If it exists, the notebook reuses the (possibly stale) tables. To force a fresh
   run, delete it first:
   ```bash
   rm CLEAN/analysis/h1_homophily/tables_n17/step2f_keystone_agreement.csv
   ```
2. **Do not re-run `02_Validation_Gold_set.ipynb`** — the gold set is frozen.

---

## The 17-film convention

The master tables in `data/04_features/` carry 18 films (the 17 study films +
`soul_2020`) and a dual-protagonist row for Monsters Inc. The analysis layer
applies "conventions" (drop Soul, one lead per film) to produce the
analysis-ready `*_n17.csv` working tables. See `data/04_features/FEATURES_README.md` for
the full data dictionary and lineage.
