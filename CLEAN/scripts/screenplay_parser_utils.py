from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_TEMPLATES = {
    "indented_screenplay",
    "inline_colon_upper",
    "inline_colon_mixed",
}

MARKER_TYPES = {
    "slug_line",
    "scene_header",
    "separator_line",
    "numbered_section",
}

DEFAULT_EXCLUDED_SPEAKERS = {
    "NARRATOR",
    "ALL",
}

DEFAULT_NON_CHARACTER_MARKERS = {
    "CUT TO",
    "FADE IN",
    "FADE OUT",
    "DISSOLVE",
    "SMASH CUT",
    "JUMP CUT",
    "MONTAGE",
    "OMITTED",
    "CONTINUED",
}

PAREN_ONLY_RE = re.compile(r"^\([^)]*\)$")
SPEAKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9' &./\-]*(?:\s*\([^)]*\))?$")
INLINE_UPPER_SPEAKER_RE = re.compile(r"^([A-Z0-9][A-Z0-9' &./\-]{0,80}):\s*(.*)$")
INLINE_MIXED_SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z0-9'&./\-]*(?:\s+[A-Za-z#][A-Za-z0-9'&./\-]*){0,7}):\s*(.*)$")
BRACKETED_SPEAKER_RE = re.compile(r"^\[([^\]]+)\]$")
SONG_MARKERS = ("♪", "♫")

# PDF page footers stamped on every page, e.g. "7/17/2020 Soul Academy Draft 12."
# (date + script title + "Draft" + revision number). These leak into dialogue
# because they sit on their own line between action/dialogue blocks.
DRAFT_FOOTER_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}\s+.+\bDraft\b\s+\d+\.?\s*$", re.IGNORECASE)

# A standalone numeric line ("42", "79.") is almost always a PDF page number and
# is stripped. The exception is a numeric *character name* (e.g. "22" in Soul):
# page numbers are near-unique and increase through the file, while a numeric
# character recurs many times at the same value. A bare-digit value appearing at
# least this many times is treated as a character name and preserved.
NUMERIC_LINE_RE = re.compile(r"\d+\.?")
NUMERIC_SPEAKER_RE = re.compile(r"\d{1,3}")
MIN_NUMERIC_SPEAKER_OCCURRENCES = 10


@dataclass
class SceneSegment:
    scene_number: int
    scene_id: str
    scene_header: str
    start_line: int
    end_line: int
    text: str
    marker_type: str


@dataclass
class ParsedUtterance:
    scene_number: int
    scene_id: str
    line_number: int
    speaker: str
    raw_speaker: str
    text: str
    utterance_type: str
    parenthetical: str
    cue_type: str


def find_clean_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if candidate.is_dir() and (candidate / "admin").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate CLEAN root from current script location.")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("â€™", "'").replace("â€˜", "'")
    text = text.replace("â€œ", '"').replace("â€", '"')
    text = text.replace("â€“", "-").replace("â€”", "-")
    return text


def clean_script_text(text: str, title_header: str | None = None) -> str:
    normalized = normalize_unicode(text)
    lines = normalized.split("\n")
    cleaned: list[str] = []
    title_header_upper = title_header.strip().upper() if title_header else ""

    # Distinguish page numbers from numeric character names (e.g. "22" in Soul):
    # a bare-digit value that recurs many times is a character cue, not a page
    # number, and must survive cleaning so its dialogue can be attributed.
    numeric_line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d+", stripped):
            numeric_line_counts[stripped] = numeric_line_counts.get(stripped, 0) + 1
    numeric_character_names = {
        value
        for value, count in numeric_line_counts.items()
        if count >= MIN_NUMERIC_SPEAKER_OCCURRENCES
    }

    for line in lines:
        line = line.rstrip()
        stripped = line.strip()

        if title_header_upper and stripped.upper().startswith(title_header_upper):
            if len(stripped.split()) <= 6 or "final draft" in stripped.lower():
                continue
            # Running header/footer carrying a writer credit and a trailing page
            # number, e.g. "RAYA AND THE LAST DRAGON - Q. Nguyen and A. Lim 13."
            # PDF conversions stamp these on every page; they otherwise leak into
            # adjacent dialogue blocks. Length test above misses them (too many
            # tokens), so strip any title-prefixed line ending in a page number.
            if re.search(r"\s\d{1,4}\.?\s*$", stripped):
                continue

        # Standalone page numbers, with or without a trailing period
        # (e.g. "42" or "79."). PDF conversions stamp these on every page and
        # they otherwise leak into adjacent dialogue blocks. Numeric character
        # names (recurring values) are kept.
        if re.fullmatch(r"\d+\.?", stripped) and stripped.rstrip(".") not in numeric_character_names:
            continue

        if DRAFT_FOOTER_RE.match(stripped):
            continue

        cleaned.append(line)

    output: list[str] = []
    blank_count = 0
    for line in cleaned:
        if line.strip():
            blank_count = 0
            output.append(line)
            continue
        blank_count += 1
        if blank_count <= 2:
            output.append("")

    return "\n".join(output).strip("\n") + "\n"


def trim_repeated_screenplay(
    text: str,
    repeated_start_pattern: str = r"^Scene 1\s*$",
    min_repeat_distance_lines: int = 200,
) -> tuple[str, dict]:
    lines = text.splitlines()
    pattern = re.compile(repeated_start_pattern)
    match_indices = [index for index, line in enumerate(lines) if pattern.match(line.strip())]

    if len(match_indices) < 2:
        return text, {
            "duplicate_trimmed": False,
            "repeat_start_line": None,
            "original_line_count": len(lines),
            "trimmed_line_count": len(lines),
        }

    first_index = match_indices[0]
    for candidate_index in match_indices[1:]:
        if candidate_index - first_index < min_repeat_distance_lines:
            continue
        trimmed_lines = lines[:candidate_index]
        return "\n".join(trimmed_lines).strip("\n") + "\n", {
            "duplicate_trimmed": True,
            "repeat_start_line": candidate_index + 1,
            "original_line_count": len(lines),
            "trimmed_line_count": len(trimmed_lines),
        }

    return text, {
        "duplicate_trimmed": False,
        "repeat_start_line": None,
        "original_line_count": len(lines),
        "trimmed_line_count": len(lines),
    }


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def normalize_speaker(raw_speaker: str) -> str:
    speaker = re.sub(r"\([^)]*\)", "", raw_speaker.strip())
    speaker = re.sub(r"\s+", " ", speaker).strip()
    return speaker


def smart_title(value: str) -> str:
    tokens = value.split()
    normalized_tokens: list[str] = []
    for token in tokens:
        if token.isupper() or token.islower():
            normalized_tokens.append(token.title())
        else:
            normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def infer_canonical_name(raw_speaker: str) -> str:
    speaker = normalize_speaker(raw_speaker)
    speaker = re.sub(r"^(YOUNG|TEEN|LITTLE|OLDER|OLD|ADULT)\s+", "", speaker)
    # Collapse numbered crowd cues ("CDA agent #1", "Soldier #2") into the base
    # name so the parser groups them under a single character. The base name's
    # canonical row is added in characters_review with the consolidated gender.
    speaker = re.sub(r"\s*#\s*\d+$", "", speaker)
    speaker = re.sub(r"\s+\d+$", "", speaker)
    return smart_title(speaker.strip())


def infer_character_id(film_id: str, canonical_name: str) -> str:
    return f"{film_id}_{slugify(canonical_name)}"


def _match_marker(line: str, marker_pattern: re.Pattern[str], marker_type: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if not marker_pattern.match(stripped):
        return False

    if marker_type == "separator_line":
        return bool(re.fullmatch(r"={5,}", stripped))
    if marker_type == "numbered_section":
        return bool(re.fullmatch(r"\d{2}\s+[A-Z][A-Za-z0-9\s\-\.']*$", stripped))
    if marker_type == "scene_header":
        return stripped.startswith("Scene ")
    return True


def segment_scenes(
    text: str,
    film_id: str,
    marker_type: str,
    scene_marker_regex: str,
) -> list[SceneSegment]:
    if marker_type not in MARKER_TYPES:
        raise ValueError(f"Unsupported marker_type: {marker_type}")

    lines = text.splitlines()
    pattern = re.compile(scene_marker_regex)
    marker_indices: list[int] = []

    for index, line in enumerate(lines):
        if not _match_marker(line, pattern, marker_type):
            continue
        if marker_type == "separator_line":
            prev_blank = index == 0 or not lines[index - 1].strip()
            next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
            if not (prev_blank and next_blank):
                continue
        marker_indices.append(index)

    if not marker_indices:
        raise ValueError(f"{film_id}: no scene markers matched regex for marker_type={marker_type}.")

    marker_indices = sorted(set(marker_indices))
    scenes: list[SceneSegment] = []

    for scene_number, start_index in enumerate(marker_indices, start=1):
        end_index = marker_indices[scene_number] - 1 if scene_number < len(marker_indices) else len(lines) - 1
        block_lines = lines[start_index : end_index + 1]
        scene_header = lines[start_index].strip() if lines[start_index].strip() else f"scene_{scene_number}"
        scenes.append(
            SceneSegment(
                scene_number=scene_number,
                scene_id=f"{film_id}_s{scene_number:03d}",
                scene_header=scene_header,
                start_line=start_index + 1,
                end_line=end_index + 1,
                text="\n".join(block_lines).strip(),
                marker_type=marker_type,
            )
        )

    return scenes


def looks_like_scene_heading(line: str) -> bool:
    stripped = line.strip().upper()
    if not stripped:
        return False
    return stripped.startswith(("INT.", "EXT.", "INT/EXT.", "EXT/INT.", "I/E.", "EST.")) or " -- " in stripped


def looks_like_transition(line: str) -> bool:
    stripped = line.strip().upper()
    if not stripped:
        return False
    return stripped.endswith(" TO:") or stripped in {"FADE IN:", "FADE OUT:", "CUT TO:"}


def is_valid_speaker_name(
    speaker: str,
    excluded_speakers: Iterable[str] | None = None,
    max_words: int = 8,
) -> bool:
    excluded = set(excluded_speakers or DEFAULT_EXCLUDED_SPEAKERS)
    normalized = normalize_speaker(speaker).upper()
    if not normalized or normalized in excluded:
        return False
    if any(marker in normalized for marker in DEFAULT_NON_CHARACTER_MARKERS):
        return False
    # Speaker names normally contain a letter; the exception is a numeric
    # character name (e.g. "22" in Soul). Page numbers are stripped upstream in
    # clean_script_text, so a short all-digit cue reaching here is a real
    # character.
    if not re.search(r"[A-Z]", normalized) and not NUMERIC_SPEAKER_RE.fullmatch(normalized):
        return False
    if len(normalized.split()) > max_words:
        return False
    if normalized.endswith("-"):
        return False
    return True


def looks_like_indented_speaker(line: str, min_indent: int) -> bool:
    stripped = line.strip()
    if not stripped or looks_like_scene_heading(line) or looks_like_transition(line):
        return False
    if stripped.startswith(("(", "[", '"', "-")):
        return False
    if stripped != stripped.upper():
        return False
    if leading_spaces(line) < min_indent:
        return False
    if not SPEAKER_RE.match(stripped):
        return False
    return is_valid_speaker_name(stripped)


def parse_inline_speaker_line(line: str, script_template: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or looks_like_scene_heading(line) or looks_like_transition(line):
        return None
    if stripped.startswith(("(", "[", '"', "-")):
        return None

    # Handle "Speaker [action]: dialogue" or "Speaker (action): dialogue" format
    # (used in the Mulan script). The action is preserved by prepending it as
    # "[action]" to the dialogue — the inline-bracket extraction downstream
    # then moves it into the parenthetical column.
    if script_template == "inline_colon_mixed":
        action_match = re.match(
            r"^([A-Z][A-Za-z0-9'&./\-]*(?:\s+[A-Za-z#][A-Za-z0-9'&./\-]*){0,7})"
            r"\s*[\[(]([^\])]*)[\])]\s*:\s*(.*)$",
            stripped,
        )
        if action_match:
            speaker = action_match.group(1).strip()
            action = action_match.group(2).strip()
            dialogue = action_match.group(3).strip()
            if is_valid_speaker_name(speaker):
                preserved = f"[{action}] {dialogue}".strip() if action else dialogue
                return speaker, preserved

    pattern = None
    if script_template == "inline_colon_upper":
        pattern = INLINE_UPPER_SPEAKER_RE
    elif script_template == "inline_colon_mixed":
        pattern = INLINE_MIXED_SPEAKER_RE

    if pattern is None:
        return None

    match = pattern.match(stripped)
    if not match:
        return None

    raw_speaker = match.group(1).strip()
    if not is_valid_speaker_name(raw_speaker):
        return None
    return raw_speaker, match.group(2).strip()


def parse_bracketed_speaker_line(line: str) -> str | None:
    match = BRACKETED_SPEAKER_RE.match(line.strip())
    if not match:
        return None
    raw_speaker = match.group(1).strip()
    if not is_valid_speaker_name(raw_speaker):
        return None
    return raw_speaker


def parse_standalone_speaker_line(line: str, min_indent: int) -> str | None:
    stripped = line.strip()
    if not stripped or ":" in stripped:
        return None
    if looks_like_scene_heading(line) or looks_like_transition(line):
        return None
    if stripped.startswith(("(", "[", '"', "-")):
        return None
    if stripped != stripped.upper():
        return None
    if leading_spaces(line) < min_indent:
        return None
    if not SPEAKER_RE.match(stripped):
        return None
    if not is_valid_speaker_name(stripped):
        return None
    return stripped


def detect_speaker_cue(
    line: str,
    script_template: str,
    min_speaker_indent: int,
) -> tuple[str, str, str] | None:
    if script_template == "indented_screenplay":
        if looks_like_indented_speaker(line, min_speaker_indent):
            return line.strip(), "", "indented"
        return None

    inline_match = parse_inline_speaker_line(line, script_template)
    if inline_match:
        return inline_match[0], inline_match[1], "inline_colon"

    bracketed = parse_bracketed_speaker_line(line)
    if bracketed:
        return bracketed, "", "bracketed"

    standalone = parse_standalone_speaker_line(line, min_indent=min_speaker_indent)
    if standalone:
        return standalone, "", "standalone"

    return None


def infer_utterance_type(
    dialogue_parts: list[str],
    parentheticals: list[str],
    cue_type: str,
) -> str:
    merged_parentheticals = " ".join(parentheticals).lower()
    if any(marker in " ".join(dialogue_parts) for marker in SONG_MARKERS):
        return "song"
    if "sing" in merged_parentheticals or "sung" in merged_parentheticals:
        return "song"
    if cue_type == "bracketed":
        return "song"
    non_empty = [part for part in dialogue_parts if part.strip()]
    if len(non_empty) >= 2:
        uppercase_like = sum(1 for part in non_empty if part == part.upper() and re.search(r"[A-Z]", part))
        if uppercase_like / len(non_empty) >= 0.8:
            return "song"
    return "spoken"


def parse_dialogue_within_scene(
    scene: SceneSegment,
    script_template: str,
    min_speaker_indent: int,
    min_dialogue_indent: int,
) -> list[ParsedUtterance]:
    if script_template not in SCRIPT_TEMPLATES:
        raise ValueError(f"Unsupported script_template: {script_template}")

    lines = scene.text.splitlines()
    utterances: list[ParsedUtterance] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        cue = detect_speaker_cue(
            line=line,
            script_template=script_template,
            min_speaker_indent=min_speaker_indent,
        )
        if not cue:
            index += 1
            continue

        raw_speaker, first_dialogue, cue_type = cue
        absolute_line_number = scene.start_line + index
        index += 1

        parentheticals: list[str] = []
        while index < len(lines) and PAREN_ONLY_RE.match(lines[index].strip()):
            parentheticals.append(lines[index].strip())
            index += 1

        dialogue_parts: list[str] = [first_dialogue] if first_dialogue else []
        while index < len(lines):
            current = lines[index]
            stripped = current.strip()

            if not stripped:
                index += 1
                if dialogue_parts:
                    dialogue_parts.append("")
                continue
            if detect_speaker_cue(current, script_template, min_speaker_indent):
                break
            if looks_like_scene_heading(current) or looks_like_transition(current):
                break
            if PAREN_ONLY_RE.match(stripped):
                parentheticals.append(stripped)
                index += 1
                continue
            if script_template == "indented_screenplay" and leading_spaces(current) < min_dialogue_indent and dialogue_parts:
                break
            if script_template == "indented_screenplay" and leading_spaces(current) < min_dialogue_indent:
                index += 1
                continue
            dialogue_parts.append(stripped)
            index += 1

        dialogue = " ".join(part for part in dialogue_parts if part)
        dialogue = re.sub(r"\([^)]*\)", "", dialogue)
        dialogue = re.sub(r"\s+", " ", dialogue).strip()
        if not dialogue:
            continue

        utterance_type = infer_utterance_type(
            dialogue_parts=dialogue_parts,
            parentheticals=parentheticals,
            cue_type=cue_type,
        )
        cleaned_dialogue = dialogue
        for marker in SONG_MARKERS:
            cleaned_dialogue = cleaned_dialogue.replace(marker, "")
        cleaned_dialogue = re.sub(r"\s+", " ", cleaned_dialogue).strip()

        # extract inline [stage directions] from text → move to parenthetical
        inline_brackets = re.findall(r"\[([^\]]+)\]", cleaned_dialogue)
        if inline_brackets:
            cleaned_dialogue = re.sub(r"\s*\[[^\]]+\]", "", cleaned_dialogue).strip()
            cleaned_dialogue = re.sub(r"\s+", " ", cleaned_dialogue).strip()
            parentheticals = parentheticals + [f"[{b}]" for b in inline_brackets]

        # strip page numbers inserted by PDF conversion (e.g. "Some text. 42." or "10. Some text")
        cleaned_dialogue = re.sub(r"\s+\d{1,3}\.\s*$", "", cleaned_dialogue).strip()
        cleaned_dialogue = re.sub(r"^\d{1,3}\.\s+", "", cleaned_dialogue).strip()

        utterances.append(
            ParsedUtterance(
                scene_number=scene.scene_number,
                scene_id=scene.scene_id,
                line_number=absolute_line_number,
                speaker=normalize_speaker(raw_speaker),
                raw_speaker=raw_speaker,
                text=cleaned_dialogue,
                utterance_type=utterance_type,
                parenthetical=" | ".join(parentheticals),
                cue_type=cue_type,
            )
        )

    return utterances


def build_utterances_dataframe(
    utterances: list[ParsedUtterance],
    film_id: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    total_scenes = max((item.scene_number for item in utterances), default=1)

    for event_id, utterance in enumerate(utterances, start=1):
        canonical_name = infer_canonical_name(utterance.speaker)
        character_id = infer_character_id(film_id, canonical_name)
        scene_position_pct = 0.0 if total_scenes == 1 else round((utterance.scene_number - 1) / (total_scenes - 1), 4)
        rows.append(
            {
                "film_id": film_id,
                "event_id": event_id,
                "scene_id": utterance.scene_id,
                "utterance_id": f"{utterance.scene_id}_u{event_id:04d}",
                "scene_position_pct": scene_position_pct,
                "speaker": utterance.speaker,
                "raw_speaker": utterance.raw_speaker,
                "character_id": character_id,
                "canonical_name": canonical_name,
                "text": utterance.text,
                "word_count": len(utterance.text.split()),
                "utterance_type": utterance.utterance_type,
                "addressee": pd.NA,
                "addressee_type": pd.NA,
                "line_number": utterance.line_number,
                "parenthetical": utterance.parenthetical,
                "cue_type": utterance.cue_type,
            }
        )

    return pd.DataFrame(rows)


def build_scenes_dataframe(
    scenes: list[SceneSegment],
    utterances_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    total_scenes = max(len(scenes), 1)

    for scene in scenes:
        scene_utts = utterances_df[utterances_df["scene_id"] == scene.scene_id] if not utterances_df.empty else pd.DataFrame()
        rows.append(
            {
                "film_id": scene.scene_id.rsplit("_s", 1)[0],
                "scene_id": scene.scene_id,
                "scene_number": scene.scene_number,
                "scene_header": scene.scene_header,
                "marker_type": scene.marker_type,
                "start_line": scene.start_line,
                "end_line": scene.end_line,
                "scene_position_pct": 0.0 if total_scenes == 1 else round((scene.scene_number - 1) / (total_scenes - 1), 4),
                "utterance_count": int(len(scene_utts)),
                "spoken_utterance_count": int((scene_utts.get("utterance_type", pd.Series(dtype=str)) == "spoken").sum()),
                "song_utterance_count": int((scene_utts.get("utterance_type", pd.Series(dtype=str)) == "song").sum()),
            }
        )

    return pd.DataFrame(rows)


def build_character_candidates_dataframe(utterances_df: pd.DataFrame) -> pd.DataFrame:
    if utterances_df.empty:
        return pd.DataFrame(
            columns=[
                "film_id",
                "character_id",
                "canonical_name",
                "raw_speaker_labels",
                "raw_speaker_label_variant_count",
                "utterance_count",
                "spoken_utterance_count",
                "song_utterance_count",
                "word_count_total",
                "word_count_spoken_only",
                "word_count_song_only",
                "first_scene_id",
                "first_event_id",
                "notes",
            ]
        )

    def join_unique(values: pd.Series) -> str:
        return " | ".join(sorted({str(value).strip() for value in values if str(value).strip()}))

    grouped = (
        utterances_df.groupby(["film_id", "character_id", "canonical_name"], as_index=False)
        .agg(
            raw_speaker_labels=("raw_speaker", join_unique),
            raw_speaker_label_variant_count=("raw_speaker", lambda values: len({str(v).strip() for v in values if str(v).strip()})),
            utterance_count=("utterance_id", "count"),
            spoken_utterance_count=("utterance_type", lambda values: int((values == "spoken").sum())),
            song_utterance_count=("utterance_type", lambda values: int((values == "song").sum())),
            word_count_total=("word_count", "sum"),
            word_count_spoken_only=("word_count", lambda values: int(utterances_df.loc[values.index, "word_count"][utterances_df.loc[values.index, "utterance_type"] == "spoken"].sum())),
            word_count_song_only=("word_count", lambda values: int(utterances_df.loc[values.index, "word_count"][utterances_df.loc[values.index, "utterance_type"] == "song"].sum())),
            first_scene_id=("scene_id", "first"),
            first_event_id=("event_id", "min"),
        )
        .sort_values(["utterance_count", "canonical_name"], ascending=[False, True])
    )
    grouped["notes"] = ""
    return grouped


def build_character_review_template(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame(
            columns=[
                "film_id",
                "parser_character_id",
                "review_character_id",
                "parser_canonical_name",
                "canonical_name",
                "raw_speaker_labels",
                "raw_speaker_label_variant_count",
                "gender",
                "gender_rationale",
                "is_named",
                "is_protagonist",
                "included_in_network",
                "merge_into_character_id",
                "utterance_count",
                "spoken_utterance_count",
                "song_utterance_count",
                "word_count_total",
                "word_count_spoken_only",
                "word_count_song_only",
                "first_scene_id",
                "first_event_id",
                "review_notes",
            ]
        )

    review_df = candidates_df.rename(
        columns={
            "character_id": "parser_character_id",
            "canonical_name": "parser_canonical_name",
            "notes": "parser_notes",
        }
    ).copy()
    review_df["review_character_id"] = review_df["parser_character_id"]
    review_df["canonical_name"] = review_df["parser_canonical_name"]
    review_df["gender"] = ""
    review_df["gender_rationale"] = ""
    review_df["is_named"] = ""
    review_df["is_protagonist"] = ""
    review_df["included_in_network"] = ""
    review_df["merge_into_character_id"] = ""
    review_df["review_notes"] = review_df["parser_notes"]
    review_df = review_df.drop(columns=["parser_notes"])

    ordered_columns = [
        "film_id",
        "parser_character_id",
        "review_character_id",
        "parser_canonical_name",
        "canonical_name",
        "raw_speaker_labels",
        "raw_speaker_label_variant_count",
        "gender",
        "gender_rationale",
        "is_named",
        "is_protagonist",
        "included_in_network",
        "merge_into_character_id",
        "utterance_count",
        "spoken_utterance_count",
        "song_utterance_count",
        "word_count_total",
        "word_count_spoken_only",
        "word_count_song_only",
        "first_scene_id",
        "first_event_id",
        "review_notes",
    ]
    return review_df[ordered_columns]


def build_character_summary_dataframe(utterances_df: pd.DataFrame) -> pd.DataFrame:
    if utterances_df.empty:
        return pd.DataFrame(
            columns=[
                "film_id",
                "character_id",
                "canonical_name",
                "utterance_count",
                "spoken_utterance_count",
                "song_utterance_count",
                "word_count_total",
                "word_count_spoken_only",
                "word_count_song_only",
                "scene_count",
                "first_scene_id",
                "last_scene_id",
                "first_event_id",
                "last_event_id",
            ]
        )

    summary_df = (
        utterances_df.groupby(["film_id", "character_id", "canonical_name"], as_index=False)
        .agg(
            utterance_count=("utterance_id", "count"),
            spoken_utterance_count=("utterance_type", lambda values: int((values == "spoken").sum())),
            song_utterance_count=("utterance_type", lambda values: int((values == "song").sum())),
            word_count_total=("word_count", "sum"),
            word_count_spoken_only=("word_count", lambda values: int(utterances_df.loc[values.index, "word_count"][utterances_df.loc[values.index, "utterance_type"] == "spoken"].sum())),
            word_count_song_only=("word_count", lambda values: int(utterances_df.loc[values.index, "word_count"][utterances_df.loc[values.index, "utterance_type"] == "song"].sum())),
            scene_count=("scene_id", "nunique"),
            first_scene_id=("scene_id", "first"),
            last_scene_id=("scene_id", "last"),
            first_event_id=("event_id", "min"),
            last_event_id=("event_id", "max"),
        )
        .sort_values(["utterance_count", "canonical_name"], ascending=[False, True])
    )
    return summary_df


def apply_reviewed_characters_to_utterances(
    utterances_df: pd.DataFrame,
    reviewed_characters_df: pd.DataFrame,
) -> pd.DataFrame:
    if utterances_df.empty:
        return utterances_df.copy()

    required_columns = {
        "film_id",
        "character_id",
        "canonical_name",
    }
    missing_columns = required_columns - set(reviewed_characters_df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Reviewed characters.csv is missing required columns: {missing_display}")

    parser_key_column = None
    for candidate in ["parser_character_id", "source_character_id", "raw_character_id", "character_id"]:
        if candidate in reviewed_characters_df.columns:
            parser_key_column = candidate
            break
    if parser_key_column is None:
        raise ValueError(
            "Reviewed characters.csv must include one parser-link column: "
            "parser_character_id, source_character_id, raw_character_id, or character_id."
        )

    review_df = reviewed_characters_df.copy()
    review_df["film_id"] = review_df["film_id"].astype(str)
    review_df[parser_key_column] = review_df[parser_key_column].astype(str)
    review_df["character_id"] = review_df["character_id"].astype(str)
    review_df["canonical_name"] = review_df["canonical_name"].astype(str)

    optional_columns = [
        "gender",
        "gender_rationale",
        "is_named",
        "is_protagonist",
        "included_in_network",
        "notes",
    ]

    # Resolve merges: rows flagged with a non-empty merge_into_character_id are
    # aliases of another character (e.g. "Big Kid Ember" -> "Ember"). Their
    # utterances must be attributed to the *target* character, not the alias.
    # The review template never rewrites review_character_id for aliases, so we
    # collapse them here by copying the target row's character_id, canonical_name,
    # and metadata onto each alias row before the parser-id merge below.
    if "merge_into_character_id" in review_df.columns:
        review_df["merge_into_character_id"] = (
            review_df["merge_into_character_id"].fillna("").astype(str).str.strip()
        )
        # Columns copied from the target row onto each alias row (besides the id itself).
        propagated_columns = ["canonical_name"] + [
            column for column in optional_columns if column in review_df.columns
        ]
        # Immutable snapshots captured before any mutation, keyed by each row's
        # original effective id (review_character_id).
        target_map = dict(
            zip(review_df["character_id"], review_df["merge_into_character_id"])
        )
        canonical_lookup = (
            review_df[review_df["merge_into_character_id"] == ""]
            .drop_duplicates(subset="character_id", keep="first")
            .set_index("character_id")
        )

        def _resolve_target(character_id: str) -> str:
            """Follow merge_into links to the final canonical id (chain-safe)."""
            seen = set()
            while target_map.get(character_id, "") and character_id not in seen:
                seen.add(character_id)
                character_id = target_map[character_id]
            return character_id

        for idx in review_df.index[review_df["merge_into_character_id"] != ""]:
            final_id = _resolve_target(review_df.at[idx, "character_id"])
            if final_id not in canonical_lookup.index:
                continue
            review_df.at[idx, "character_id"] = final_id
            for column in propagated_columns:
                review_df.at[idx, column] = canonical_lookup.at[final_id, column]
            review_df.at[idx, "merge_into_character_id"] = ""

    merge_columns = ["film_id", parser_key_column, "character_id", "canonical_name"]
    merge_columns.extend([column for column in optional_columns if column in review_df.columns])

    merged = utterances_df.merge(
        review_df[merge_columns],
        left_on=["film_id", "character_id"],
        right_on=["film_id", parser_key_column],
        how="left",
        suffixes=("", "_reviewed"),
    )

    unmatched = merged["canonical_name_reviewed"].isna()
    if unmatched.any():
        missing_ids = sorted(merged.loc[unmatched, "character_id"].dropna().astype(str).unique())
        raise ValueError(
            "Reviewed characters.csv does not cover all parser character_ids. Missing: "
            + ", ".join(missing_ids[:20])
        )

    merged["parser_character_id"] = merged["character_id"]
    merged["parser_canonical_name"] = merged["canonical_name"]
    merged["character_id"] = merged["character_id_reviewed"]
    merged["canonical_name"] = merged["canonical_name_reviewed"]

    columns_to_drop = [parser_key_column, "character_id_reviewed", "canonical_name_reviewed"]
    merged = merged.drop(columns=[column for column in columns_to_drop if column in merged.columns])
    return merged


def validate_parse(
    film_id: str,
    scenes_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    strict: bool,
    min_scenes: int = 20,
    min_utterances: int = 100,
) -> None:
    if not strict:
        return
    if scenes_df.empty:
        raise ValueError(f"{film_id}: no scenes parsed.")
    if utterances_df.empty:
        raise ValueError(f"{film_id}: no utterances parsed.")
    if len(scenes_df) < min_scenes:
        raise ValueError(f"{film_id}: parsed only {len(scenes_df)} scenes; expected at least {min_scenes}.")
    if len(utterances_df) < min_utterances:
        raise ValueError(f"{film_id}: parsed only {len(utterances_df)} utterances; expected at least {min_utterances}.")


def parse_screenplay(
    text: str,
    film_id: str,
    marker_type: str,
    scene_marker_regex: str,
    script_template: str,
    min_speaker_indent: int = 18,
    min_dialogue_indent: int = 12,
    title_header: str | None = None,
    strict: bool = False,
    trim_duplicate_screenplay: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned = clean_script_text(text, title_header=title_header)
    if trim_duplicate_screenplay:
        cleaned, _ = trim_repeated_screenplay(cleaned)
    scenes = segment_scenes(
        text=cleaned,
        film_id=film_id,
        marker_type=marker_type,
        scene_marker_regex=scene_marker_regex,
    )

    parsed_utterances: list[ParsedUtterance] = []
    retained_scenes: list[SceneSegment] = []

    for scene in scenes:
        scene_utterances = parse_dialogue_within_scene(
            scene=scene,
            script_template=script_template,
            min_speaker_indent=min_speaker_indent,
            min_dialogue_indent=min_dialogue_indent,
        )
        if marker_type == "scene_header" and len(scene_utterances) < 3:
            continue
        retained_scenes.append(scene)
        parsed_utterances.extend(scene_utterances)

    utterances_df = build_utterances_dataframe(parsed_utterances, film_id=film_id)
    scenes_df = build_scenes_dataframe(retained_scenes, utterances_df=utterances_df)
    candidates_df = build_character_candidates_dataframe(utterances_df=utterances_df)

    validate_parse(
        film_id=film_id,
        scenes_df=scenes_df,
        utterances_df=utterances_df,
        strict=strict,
    )

    return utterances_df, scenes_df, candidates_df


def load_film_metadata(films_csv: Path, movie_id: str) -> dict:
    films_df = pd.read_csv(films_csv).fillna("")
    matches = films_df[films_df["movie_id"].astype(str) == str(movie_id)]
    if matches.empty:
        raise ValueError(f"{movie_id}: movie_id not found in {films_csv}.")
    return matches.iloc[0].to_dict()
