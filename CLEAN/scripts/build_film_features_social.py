"""Regenerate film_features_social.csv from per-film artifacts.

Formulas were reverse-engineered from the original 12-row csv and verified on
Mulan + Frozen (all 9 metrics matched within +/- 0.1pp). See verification table
in UNIFIED_RESULTS_N18_step5_social.md.

Output: CLEAN/data/04_features/film_features_social.csv
  - one row per (film_id, protagonist) pair where protagonist is the film's lead
  - covers all 18 films; Monsters Inc has both Mike and Sulley rows so Conventions
    can pick either at load time

Run: python CLEAN/scripts/build_film_features_social.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "02_processed"
FEAT = ROOT / "data" / "04_features"
ALL_CSV = FEAT / "film_features_all.csv"
OUT_CSV = FEAT / "film_features_social.csv"


def find_protag_id(film_id: str, protag_name: str, chars: pd.DataFrame) -> str | None:
    """Look up a character's review_character_id by canonical_name."""
    hit = chars[chars.canonical_name.str.lower() == protag_name.lower()]
    if not hit.empty:
        return hit.iloc[0].review_character_id
    # fallback: case-insensitive contains
    hit = chars[chars.canonical_name.str.lower().str.contains(protag_name.lower(), na=False)]
    if not hit.empty:
        return hit.iloc[0].review_character_id
    return None


def compute_row(film_id: str, protag_name: str, lead_gender: str) -> dict:
    film_dir = PROC / film_id
    ua_path = film_dir / "utterances_with_addressee_scene.csv"
    chars_path = film_dir / f"character_review_{film_id}.csv"
    edges_path = film_dir / "network_edges.csv"

    ua = pd.read_csv(ua_path)
    chars = pd.read_csv(chars_path)
    edges = pd.read_csv(edges_path)

    protag_id = find_protag_id(film_id, protag_name, chars)
    if protag_id is None:
        raise ValueError(f"protagonist {protag_name!r} not found in {chars_path}")

    char_gender = dict(zip(chars.review_character_id, chars.gender))
    char_canon = dict(zip(chars.review_character_id, chars.canonical_name))

    # Protagonist outgoing utterances
    mu_out = ua[ua.character_id == protag_id]
    N = len(mu_out)
    if N == 0:
        raise ValueError(f"no utterances for protagonist {protag_id} in {ua_path}")

    monologue_pct = (mu_out.addressee_type == "monologue").sum() / N * 100
    non_human_pct = (mu_out.addressee_type == "non_human").sum() / N * 100
    isolation_pct = monologue_pct + non_human_pct

    # Best friend = top outgoing individual addressee
    ind = mu_out[mu_out.addressee_type == "individual"]
    if ind.empty:
        bf_id, bf_count = None, 0
    else:
        top = ind["addressee"].value_counts()
        bf_id = top.index[0]
        bf_count = int(top.iloc[0])

    best_friend = char_canon.get(bf_id, bf_id) if bf_id else None
    bf_gender = char_gender.get(bf_id) if bf_id else None
    bf_ratio_pct = bf_count / N * 100 if bf_id else 0.0

    # bf_reciprocity = bf->protag / protag->bf using network_edges
    if bf_id:
        b2p = edges[(edges.speaker_clean == bf_id) & (edges.addressee_clean == protag_id)].utterance_count.sum()
        p2b = edges[(edges.speaker_clean == protag_id) & (edges.addressee_clean == bf_id)].utterance_count.sum()
        bf_reciprocity = b2p / p2b if p2b > 0 else float("nan")
    else:
        bf_reciprocity = float("nan")

    # out_in_ratio = protag total outgoing / total incoming, from edges
    out_total = edges[edges.speaker_clean == protag_id].utterance_count.sum()
    in_total = edges[edges.addressee_clean == protag_id].utterance_count.sum()
    out_in_ratio = out_total / in_total if in_total > 0 else float("nan")

    # pct_addressed_by_male = % of incoming utterances spoken by M-gender characters
    inc = ua[ua["addressee"].astype(str).str.contains(protag_id, na=False) & (ua.character_id != protag_id)]
    if len(inc) > 0:
        pct_male = (inc.gender == "M").sum() / len(inc) * 100
    else:
        pct_male = float("nan")

    return {
        "film_id": film_id,
        "protagonist": protag_name,
        "lead_gender": lead_gender,
        "monologue_pct": round(monologue_pct, 1),
        "non_human_pct": round(non_human_pct, 1),
        "isolation_pct": round(isolation_pct, 1),
        "best_friend": best_friend,
        "bf_gender": bf_gender,
        "bf_ratio_pct": round(bf_ratio_pct, 1),
        "bf_reciprocity": round(bf_reciprocity, 2) if pd.notna(bf_reciprocity) else None,
        "out_in_ratio": round(out_in_ratio, 2) if pd.notna(out_in_ratio) else None,
        "pct_addressed_by_male": round(pct_male, 1) if pd.notna(pct_male) else None,
    }


def main():
    base = pd.read_csv(ALL_CSV)
    rows = []
    for _, r in base.iterrows():
        try:
            row = compute_row(r.film_id, r.protagonist, r.lead_gender)
            rows.append(row)
            print(f"  OK  {r.film_id:32s} {r.protagonist}")
        except Exception as exc:
            print(f"  FAIL {r.film_id:32s} {r.protagonist}: {exc}")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")
    return out


if __name__ == "__main__":
    main()
