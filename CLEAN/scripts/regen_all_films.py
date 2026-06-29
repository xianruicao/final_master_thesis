"""Regenerate scenes.csv + utterances.csv for every film and rebuild a single
clean master characters.csv.

This replays notebook 01 (`01_parse_scenes_and_characters.ipynb`) cells
§1–§4 and §6–§8 for every film in `data/00_corpus/final_films.csv`, reusing the
exact helpers in `screenplay_parser_utils.py`.

What it does NOT do
-------------------
- It never writes `character_candidates.csv`.
- It never writes `character_review_{film_id}.csv`.
  The reviewed templates are hand-annotated inputs; they are read only.

Outputs (per film)        : data/02_processed/{film_id}/scenes.csv
                            data/02_processed/{film_id}/utterances.csv
Output (single, rebuilt)  : data/00_corpus/characters.csv   (built fresh, all films)

The run is two-phase: every film is parsed and validated in memory first; only
if all films pass does anything get written. This keeps the outputs consistent
(no half-written master, no stale film rows).
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CLEAN_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CLEAN_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from screenplay_parser_utils import (  # noqa: E402
    apply_reviewed_characters_to_utterances,
    build_scenes_dataframe,
    build_utterances_dataframe,
    clean_script_text,
    load_film_metadata,
    parse_dialogue_within_scene,
    segment_scenes,
    trim_repeated_screenplay,
)

FILMS_CSV = CLEAN_ROOT / "data" / "00_corpus" / "final_films.csv"
PROCESSED_DIR = CLEAN_ROOT / "data" / "02_processed"
METADATA_DIR = CLEAN_ROOT / "data" / "00_corpus"
MASTER_CHARS = METADATA_DIR / "characters.csv"

# Only films whose raw file contains the full screenplay twice (notebook §2).
TRIM_DUPLICATE_FILMS = {"beautyandthebeast_1991"}

VALID_GENDERS = {"F", "M", "Unknown", "Both"}
VALID_BOOLS = {"True", "False", "true", "false", "TRUE", "FALSE", "1", "0"}


# ─────────────────────────────────────────────────────────────────────────────
# §6 — validation (replicates notebook cell §23)
# ─────────────────────────────────────────────────────────────────────────────
def validate_reviewed(reviewed_df: pd.DataFrame, film_id: str) -> list[str]:
    errors: list[str] = []
    for idx, row in reviewed_df.iterrows():
        name = str(row.get("canonical_name", f"row {idx}")).strip() or f"row {idx}"
        # Alias rows collapse into their target — skip required-field checks.
        if str(row.get("merge_into_character_id", "")).strip() != "":
            continue
        if str(row.get("gender", "")).strip() not in VALID_GENDERS:
            errors.append(f"[{film_id}] row {idx} ({name}): bad gender '{row.get('gender')}'")
        if str(row.get("is_named", "")).strip() not in VALID_BOOLS:
            errors.append(f"[{film_id}] row {idx} ({name}): bad is_named '{row.get('is_named')}'")
        if str(row.get("is_protagonist", "")).strip() not in VALID_BOOLS:
            errors.append(f"[{film_id}] row {idx} ({name}): bad is_protagonist '{row.get('is_protagonist')}'")
        if str(row.get("included_in_network", "")).strip() not in VALID_BOOLS:
            errors.append(f"[{film_id}] row {idx} ({name}): bad included_in_network '{row.get('included_in_network')}'")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# §8 — Block 2 characters schema (replicates notebook cell §25)
# ─────────────────────────────────────────────────────────────────────────────
def _coerce_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_characters_csv_rows(reviewed_df: pd.DataFrame, film_id: str) -> pd.DataFrame:
    """Convert the filled-in review template into the Block 2 characters schema."""
    df = reviewed_df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    count_cols = [
        "utterance_count", "spoken_utterance_count", "song_utterance_count",
        "word_count_total", "word_count_spoken_only", "word_count_song_only",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    is_alias = df["merge_into_character_id"].str.strip().ne("")
    aliases = df[is_alias].copy()
    canonical = df[~is_alias].copy().reset_index(drop=True)

    if not aliases.empty:
        active_count_cols = [c for c in count_cols if c in canonical.columns]
        for _, alias_row in aliases.iterrows():
            target_id = alias_row["merge_into_character_id"]
            mask = canonical["review_character_id"] == target_id
            if not mask.any():
                print(f"  Warning: merge target '{target_id}' not found — skipping alias "
                      f"'{alias_row['canonical_name']}'")
                continue
            for col in active_count_cols:
                canonical.loc[mask, col] += alias_row.get(col, 0)
            existing = canonical.loc[mask, "raw_speaker_labels"].iloc[0]
            combined = " | ".join(sorted({
                lbl.strip()
                for lbl in (existing + " | " + alias_row.get("raw_speaker_labels", "")).split("|")
                if lbl.strip()
            }))
            canonical.loc[mask, "raw_speaker_labels"] = combined

    out_rows = []
    for _, row in canonical.iterrows():
        out_rows.append({
            "film_id":                film_id,
            "character_id":           row["review_character_id"],
            "canonical_name":         row["canonical_name"],
            "raw_speaker_labels":     row["raw_speaker_labels"],
            "gender":                 row["gender"],
            "gender_rationale":       row.get("gender_rationale", ""),
            "is_protagonist":         _coerce_bool(row["is_protagonist"]),
            "is_named":               _coerce_bool(row["is_named"]),
            "included_in_network":    _coerce_bool(row["included_in_network"]),
            "utterance_count":        int(row.get("utterance_count", 0)),
            "word_count_spoken_only": int(row.get("word_count_spoken_only", 0)),
            "word_count_song_only":   int(row.get("word_count_song_only", 0)),
            "notes":                  row.get("review_notes", ""),
        })
    return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-film pipeline (§1–§4, §6–§8 without writing)
# ─────────────────────────────────────────────────────────────────────────────
def process_film(film_id: str) -> dict:
    film = load_film_metadata(FILMS_CSV, film_id)
    output_dir = PROCESSED_DIR / film_id

    review_path = output_dir / f"character_review_{film_id}.csv"
    if not review_path.exists():
        raise FileNotFoundError(f"[{film_id}] review template not found: {review_path}")

    # §2 — read + normalize
    script_path = CLEAN_ROOT / str(film["script_path"])
    raw_text = script_path.read_text(encoding="utf-8", errors="ignore")
    title_header = str(film.get("movie_name", "")).upper().strip() or None
    cleaned_text = clean_script_text(raw_text, title_header=title_header)
    if film_id in TRIM_DUPLICATE_FILMS:
        trimmed_text, _audit = trim_repeated_screenplay(cleaned_text)
    else:
        trimmed_text = cleaned_text

    # §3 — segment
    scene_segments = segment_scenes(
        text=trimmed_text,
        film_id=film_id,
        marker_type=str(film["marker_type"]),
        scene_marker_regex=str(film["scene_marker_regex"]),
    )

    # §4 — parse dialogue
    parsed_utterances = []
    for scene in scene_segments:
        parsed_utterances.extend(
            parse_dialogue_within_scene(
                scene=scene,
                script_template=str(film["script_template"]),
                min_speaker_indent=int(film["min_speaker_indent"] or 0),
                min_dialogue_indent=int(film["min_dialogue_indent"] or 0),
            )
        )
    utterances_df = build_utterances_dataframe(parsed_utterances, film_id=film_id)
    scenes_df = build_scenes_dataframe(scene_segments, utterances_df)

    # §6 — load + validate reviewed template
    reviewed_df = pd.read_csv(review_path, dtype=str).fillna("")
    errors = validate_reviewed(reviewed_df, film_id)
    if errors:
        return {"film_id": film_id, "errors": errors}

    # §7 — apply reviewed characters (Pass 1 scope + merges), no save yet
    reviewed_ids = set(reviewed_df["parser_character_id"].astype(str).str.strip())
    utterances_in_scope = utterances_df[
        utterances_df["character_id"].astype(str).isin(reviewed_ids)
    ].copy()
    reviewed_for_utterances = reviewed_df.rename(columns={"review_character_id": "character_id"})
    utterances_final = apply_reviewed_characters_to_utterances(
        utterances_df=utterances_in_scope,
        reviewed_characters_df=reviewed_for_utterances,
    ).reset_index(drop=True)

    # §8 — build character rows
    film_characters_df = build_characters_csv_rows(reviewed_df, film_id)

    return {
        "film_id": film_id,
        "errors": [],
        "output_dir": output_dir,
        "scenes_df": scenes_df,
        "utterances_df": utterances_final,
        "characters_df": film_characters_df,
        "n_scenes": len(scenes_df),
        "n_utterances": len(utterances_final),
        "n_characters": len(film_characters_df),
    }


def main() -> None:
    films = pd.read_csv(FILMS_CSV, dtype=str)
    film_ids = films["movie_id"].astype(str).str.strip().tolist()
    print(f"Films in final_films.csv: {len(film_ids)}\n")

    # Phase 1 — parse + validate everything in memory.
    results = []
    all_errors = []
    for film_id in film_ids:
        try:
            res = process_film(film_id)
        except Exception as exc:  # noqa: BLE001
            all_errors.append(f"[{film_id}] {type(exc).__name__}: {exc}")
            print(f"  ✗ {film_id}: {type(exc).__name__}: {exc}")
            continue
        if res["errors"]:
            all_errors.extend(res["errors"])
            print(f"  ✗ {film_id}: {len(res['errors'])} validation issue(s)")
            continue
        results.append(res)
        print(f"  ✓ {film_id}: {res['n_scenes']} scenes, "
              f"{res['n_utterances']} utterances, {res['n_characters']} characters")

    if all_errors:
        print(f"\n❌ Aborting — {len(all_errors)} issue(s); nothing written:\n")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)

    # Phase 2 — write everything.
    if MASTER_CHARS.exists():
        backup = MASTER_CHARS.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.copy2(MASTER_CHARS, backup)
        print(f"\nBacked up existing master → {backup.name}")

    master_frames = []
    for res in results:
        res["scenes_df"].to_csv(res["output_dir"] / "scenes.csv", index=False)
        res["utterances_df"].to_csv(res["output_dir"] / "utterances.csv", index=False)
        master_frames.append(res["characters_df"])

    master_df = pd.concat(master_frames, ignore_index=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(MASTER_CHARS, index=False)

    print(f"\n✅ Done.")
    print(f"  Films processed      : {len(results)}")
    print(f"  Master characters.csv: {MASTER_CHARS}")
    print(f"  Total characters     : {len(master_df)} across "
          f"{master_df['film_id'].nunique()} films")


if __name__ == "__main__":
    main()
