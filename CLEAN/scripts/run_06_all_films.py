"""Run notebook 06 (network_analysis_PAU) end-to-end for every film in the
N=17 corpus.

What this does
--------------
Prerequisite:
  `data/04_features/film_features_all_n17.csv` must already exist as the current
  N=17 film-list snapshot. This script uses that file only to decide which
  films/protagonists to run through Notebook 06.

For each (film_id, protagonist, year, lead_gender) in that snapshot:
  1. Load the notebook 06 template.
  2. Patch §0 in memory (FILM_ID / PROTAGONIST / YEAR), keep ALIAS = {} (merges
     are already applied upstream in utterances_with_addressee_scene.csv).
  3. Execute the patched notebook via nbclient, working dir = CLEAN/notebooks.
  4. Discard the executed notebook; we only want the side-effect outputs:
        data/02_processed/{film_id}/protagonist_metrics.csv  (+ new rung columns)
        data/02_processed/{film_id}/film_features.csv         (+ new rung columns)
        data/02_processed/{film_id}/figures/*.png             (refreshed)
     plus the per-film upsert into data/04_features/film_features_all.csv.

What this does NOT do
---------------------
- It does not run `n17_orchestrator.py`. That is a separate step you run after
  this script finishes, to refresh the N=17 derived tables and figures.
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

CLEAN_ROOT  = Path(__file__).resolve().parents[1]
NOTEBOOKS   = CLEAN_ROOT / "notebooks"
NB_PATH     = NOTEBOOKS / "06_network_analysis_PAU.ipynb"
PROC        = CLEAN_ROOT / "data" / "02_processed"
FEAT        = CLEAN_ROOT / "data" / "04_features"
N17_FILE    = FEAT / "film_features_all_n17.csv"

# Monsters Inc has co-leads (Sulley & Mike); convention is Sulley only.
DROP_PROTAG = {("monsters_inc_2001", "Mike")}
DROP_FILMS  = {"soul_2020"}

PARAM_RE = re.compile(
    r'FILM_ID\s*=\s*".*?"|'
    r'PROTAGONIST\s*=\s*".*?"|'
    r'YEAR\s*=\s*\d+',
)


def patch_params_source(src: str, film_id: str, protagonist: str, year: int) -> str:
    """Replace the three top-of-cell assignments in §0 with the chosen film."""
    new_src = src
    new_src = re.sub(r'FILM_ID\s*=\s*"[^"]*"',     f'FILM_ID        = "{film_id}"',     new_src, count=1)
    new_src = re.sub(r'PROTAGONIST\s*=\s*"[^"]*"', f'PROTAGONIST    = "{protagonist}"', new_src, count=1)
    new_src = re.sub(r'YEAR\s*=\s*\d+',            f'YEAR           = {year}',          new_src, count=1)
    return new_src


def run_one(film_id: str, protagonist: str, year: int, *, timeout_s: int = 900) -> dict:
    nb = nbformat.read(NB_PATH, as_version=4)

    # locate §0 by its hallmark line and patch in place
    patched = False
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if "FILM_ID" in src and "PROTAGONIST" in src and "YEAR" in src and 'DATA_ROOT' in src:
            c.source = patch_params_source(src, film_id, protagonist, year)
            patched = True
            break
    if not patched:
        raise RuntimeError("Could not find §0 params cell in notebook 06.")

    # Run notebook with cwd = notebooks/ so its relative paths resolve.
    client = NotebookClient(
        nb,
        timeout=timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    )
    t0 = time.time()
    client.execute()
    return {"film_id": film_id, "elapsed_s": round(time.time() - t0, 1)}


def verify_outputs(film_id: str) -> dict:
    """Smoke-check that the expected output files exist and have the new columns."""
    fdir = PROC / film_id
    fails = []
    expected = [
        "protagonist_metrics.csv",
        "film_features.csv",
        "network_edges.csv",
        "network_node_summary.csv",
        "homophily_null.csv",
        "betweenness_null.csv",
    ]
    for fn in expected:
        if not (fdir / fn).exists():
            fails.append(f"missing {fn}")
    # check the new rung columns are present
    if (fdir / "protagonist_metrics.csv").exists():
        pm = pd.read_csv(fdir / "protagonist_metrics.csv")
        for col in ("keystone_rung1", "keystone_rung2", "keystone_rung3",
                    "keystone_changes_at_rung2", "keystone_changes_at_rung3"):
            if col not in pm.columns:
                fails.append(f"protagonist_metrics.csv missing column {col}")
    if (fdir / "film_features.csv").exists():
        ff = pd.read_csv(fdir / "film_features.csv")
        for col in ("keystone_rung1", "keystone_rung2", "keystone_rung3"):
            if col not in ff.columns:
                fails.append(f"film_features.csv missing column {col}")
    return {"film_id": film_id, "ok": not fails, "fails": fails}


def load_film_list() -> list[dict]:
    df = pd.read_csv(N17_FILE)
    df = df[~df.film_id.isin(DROP_FILMS)]
    df = df[~df.apply(lambda r: (r.film_id, r.protagonist) in DROP_PROTAG, axis=1)]
    cols = ["film_id", "protagonist", "year", "lead_gender"]
    return df[cols].to_dict(orient="records")


def main():
    films = load_film_list()
    print(f"Running notebook 06 across N = {len(films)} films")
    print("=" * 78)

    results = []
    for i, f in enumerate(films, 1):
        film_id    = f["film_id"]
        protag     = f["protagonist"]
        year       = int(f["year"])
        lead       = f["lead_gender"]
        print(f"[{i:2d}/{len(films)}]  {film_id:35s}  protag={protag:12s}  year={year}  lead={lead}",
              flush=True)
        try:
            r = run_one(film_id, protag, year)
            v = verify_outputs(film_id)
            r.update(v)
            results.append(r)
            status = "OK " if r["ok"] else "FAIL"
            extra  = "" if r["ok"] else f"  -> {r['fails']}"
            print(f"           {status}  ({r['elapsed_s']}s){extra}", flush=True)
        except Exception as e:
            print(f"           ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({"film_id": film_id, "ok": False, "fails": [f"exception: {e}"]})

    # Final report
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    ok   = [r for r in results if r.get("ok")]
    bad  = [r for r in results if not r.get("ok")]
    print(f"  succeeded : {len(ok)}/{len(results)}")
    print(f"  failed    : {len(bad)}/{len(results)}")
    for r in bad:
        print(f"    - {r['film_id']}: {r.get('fails')}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
