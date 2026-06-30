# `scripts/` — batch / helper scripts

Command-line scripts that run the pipeline across **all films** (the notebooks in
`../notebooks/` are the interactive, per-film layer). Run them from the repo root
(the folder that contains `CLEAN/`):

```bash
python CLEAN/scripts/<script>.py
```

Paths are resolved relative to the file, so the working directory only matters for
`scan_character_reviews.py` (see below).

## Data locations they touch

- `data/00_corpus/`   — corpus inputs (`final_films.csv`, rebuilt `characters.csv`)
- `data/02_processed/{film_id}/` — per-film artifacts (scenes, utterances, network edges, nulls)
- `data/04_features/` — cross-film feature tables (`film_features_*.csv`)

## The scripts

| script | role | reads | writes |
|---|---|---|---|
| `screenplay_parser_utils.py` | **helper module, not run directly.** Screenplay parsing helpers shared by `regen_all_films.py` and notebook 01. | — | — |
| `regen_all_films.py` | Replay notebook 01 (§1–4, §6–8) for every film in `final_films.csv`; rebuild a clean master characters table. **Does not** touch the hand-reviewed `character_review_*.csv` (read-only inputs). | `data/00_corpus/final_films.csv`, screenplays | `data/02_processed/{film}/scenes.csv`, `.../utterances.csv`, `data/00_corpus/characters.csv` |
| `scan_character_reviews.py` | QA tool. Scans every `character_review_*.csv` for suspicious "names" (scene cues, sentences, etc.) that may be leaking into the networks. Prints a report; writes nothing. **Run from repo root** (uses a `CLEAN/data/...` glob). | `data/02_processed/**/character_review_*.csv` | — (stdout report) |
| `run_06_all_films.py` | Run notebook 06 (network analysis) end-to-end for all 17 films. Uses `film_features_all_n17.csv` only to decide the film/protagonist list. | `data/04_features/film_features_all_n17.csv`, notebook 06 template, per-film inputs | per-film network outputs + upserts `data/04_features/film_features_all.csv` |
| `compute_extended_features.py` | Compute extended ego-network metrics (Burt constraint, ego density, female-alter betweenness, …) per protagonist. | `data/04_features/film_features_all.csv`, `data/02_processed/{film}/network_edges.csv` · `betweenness_null.csv` · `protagonist_metrics.csv` | `data/04_features/film_features_extended.csv` |
| `build_film_features_social.py` | Rebuild the "social" metrics table (monologue %, isolation %, best friend, who-addresses-whom) from per-film artifacts. Formulas verified against the original csv on Mulan + Frozen. | `data/04_features/film_features_all.csv`, per-film artifacts | `data/04_features/film_features_social.csv` |

## Rough order

```
regen_all_films.py            (rebuild scenes/utterances + characters master)
  → [notebooks 05/06 produce per-film networks; run_06_all_films.py batches 06]
  → compute_extended_features.py    → film_features_extended.csv
  → build_film_features_social.py   → film_features_social.csv

scan_character_reviews.py    — QA, run any time after reviews exist
```

See `data/04_features/FEATURES_README.md` for what the `film_features_*` tables contain and
how the `_n17` working set is derived.
