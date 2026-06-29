import csv, glob, os, re

files = glob.glob("CLEAN/data/02_processed/**/character_review_*.csv", recursive=True)

# scene / stage-direction keyword cues
SCENE_KW = re.compile(r"\b(INT|EXT|CLOSE ON|CLOSE-ON|INSERT|CUT TO|CUT BACK|ANGLE|REVEAL|PRE-?LAP|"
                      r"CONTINUOUS|FADE|DISSOLVE|INTERCUT|MONTAGE|SUPER|TITLE|OVER BLACK|WIDE|"
                      r"PAN|ZOOM|FLASHBACK|SMASH|MATCH CUT|BEAT|LATER|MOMENTS LATER|V\.?O|O\.?S|O\.?C)\b",
                      re.IGNORECASE)

# words that signal a sentence rather than a name (common verbs / function words mid-phrase)
SENTENCE_WORDS = re.compile(r"\b(the|a|an|in|on|of|to|her|his|their|its|some|pushes|pulls|puts|wraps|"
                            r"swivels|wipes|reads|sit|sits|then|reserve|right|refuse|request|only|have|"
                            r"i|we|she|he|they|off|back|open|front|graphic|just|number)\b", re.IGNORECASE)

def is_suspicious(name):
    reasons = []
    n = name.strip()
    if not n:
        return ["empty"]
    words = n.split()
    # scene/stage keyword
    if SCENE_KW.search(n):
        reasons.append("scene/stage keyword")
    # ends with sentence punctuation
    if n.endswith("...") or n.endswith(".") and len(words) > 1:
        reasons.append("ends with punctuation")
    # very long -> sentence
    if len(words) >= 5:
        reasons.append(f"long ({len(words)} words)")
    # contains lowercase function/verb words (mid-sentence) AND more than 2 words
    if len(words) >= 3 and len(SENTENCE_WORDS.findall(n)) >= 2:
        reasons.append("reads like a sentence")
    # starts lowercase
    if n[0].islower():
        reasons.append("starts lowercase")
    return reasons

def norm_inc(v):
    v = (v or "").strip().lower()
    if v in ("1", "true", "yes", "y"):   return True
    if v in ("0", "false", "no", "n"):   return False
    return None  # blank / unknown

results = {}
for f in sorted(files):
    film = os.path.basename(os.path.dirname(f))
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    susp = []
    notes_count = 0
    for r in rows:
        if (r.get("review_notes", "") or "").strip():
            notes_count += 1
        name = r.get("canonical_name", "") or r.get("parser_canonical_name", "")
        reasons = is_suspicious(name)
        if reasons:
            inc = norm_inc(r.get("included_in_network", ""))
            notes = (r.get("review_notes", "") or "").strip()
            susp.append((name, inc, notes, reasons))
    # categorize by normalized inclusion
    leaking = [s for s in susp if s[1] is True]    # suspicious AND still in network
    flagged = [s for s in susp if s[1] is False]   # suspicious but excluded
    unhandled = [s for s in susp if s[1] is None]
    results[film] = dict(total=len(rows), susp=len(susp), leaking=leaking,
                         flagged=flagged, unhandled=unhandled, notes=notes_count)

print(f"{'FILM':40} {'rows':>5} {'notes':>5} {'susp':>5} {'LEAK':>5} {'excl':>5} {'blank':>5}")
print("-"*75)
for film, d in results.items():
    print(f"{film:40} {d['total']:>5} {d['notes']:>5} {d['susp']:>5} {len(d['leaking']):>5} "
          f"{len(d['flagged']):>5} {len(d['unhandled']):>5}")

print("\n\n===== LEAKING (suspicious name still IN network) =====")
any_leak = False
for film, d in results.items():
    if d['leaking']:
        any_leak = True
        print(f"\n## {film}  ({len(d['leaking'])} leaking)")
        for name, inc, notes, reasons in d['leaking']:
            print(f"   - {name!r:55} | {', '.join(reasons)} | notes={notes!r}")
if not any_leak:
    print("  (none)")

print("\n\n===== BLANK inclusion flag on suspicious rows =====")
for film, d in results.items():
    if d['unhandled']:
        print(f"\n## {film}  ({len(d['unhandled'])} blank)")
        for name, inc, notes, reasons in d['unhandled']:
            print(f"   - {name!r:55} | {', '.join(reasons)}")
