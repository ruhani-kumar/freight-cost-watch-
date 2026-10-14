import json
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from src import config, llm

EFFECTS = {"raises_cost", "no_effect", "lowers_cost", "back_to_normal", "unrelated"}

SYSTEM_PROMPT = (
    "You read short freight-industry notes and fill in a small JSON form. "
    "Answer with JSON only and use only what the note says."
)

PROMPT_TEMPLATE = """Note date: {date}
Note applies to: {applies_to}
Note text: {note}

Fill in this JSON form:
{{
  "effect": one of "raises_cost", "no_effect", "lowers_cost", "back_to_normal", "unrelated",
  "end_date": "YYYY-MM-DD" or null,
  "magnitude_min_pct": number or null,
  "magnitude_max_pct": number or null
}}

Rules:
- "raises_cost": the note describes something that pushes freight costs UP on the routes it applies to.
- "no_effect": the note says there was no disruption, no change, or that costs were not significantly affected.
- "lowers_cost": the note describes something that makes costs lower or conditions better.
- "back_to_normal": the note says a previous disruption has ended and conditions are normal again.
- "unrelated": the note does not concern the routes in this dataset.
- "end_date": the last day the cost effect lasts, only if the note states it, or says it lasted a week (then note date + 6 days). Use null if the effect has no stated end, or if effect is not "raises_cost".
- "magnitude_min_pct" and "magnitude_max_pct": the cost increase range in percent if the note gives one, otherwise null.
"""


@dataclass
class ParsedNote:
    note_id: str
    note_date: str
    applies_to: str
    text: str
    effect: str
    end_date: Optional[str]
    magnitude_min_pct: Optional[float]
    magnitude_max_pct: Optional[float]


def load_notes(path=config.NOTES_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _clean_date(value) -> Optional[str]:
    if not value:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _clean_number(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def parse_note(row: pd.Series) -> ParsedNote:
    prompt = PROMPT_TEMPLATE.format(date=row["date"], applies_to=row["applies_to"], note=row["note"])
    form = llm.ask(SYSTEM_PROMPT, prompt, as_json=True)
    effect = form.get("effect")
    if effect not in EFFECTS:
        raise ValueError(f"{row['note_id']}: unexpected effect {effect!r}")
    return ParsedNote(
        note_id=row["note_id"],
        note_date=_clean_date(row["date"]),
        applies_to=row["applies_to"],
        text=row["note"],
        effect=effect,
        end_date=_clean_date(form.get("end_date")),
        magnitude_min_pct=_clean_number(form.get("magnitude_min_pct")),
        magnitude_max_pct=_clean_number(form.get("magnitude_max_pct")),
    )


def parse_all_notes(path=config.NOTES_CSV) -> list:
    return [parse_note(row) for _, row in load_notes(path).iterrows()]


def save_parsed(notes: list, path=config.PARSED_NOTES_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(n) for n in notes], indent=2), encoding="utf-8")


def load_parsed(path=config.PARSED_NOTES_JSON) -> list:
    return [ParsedNote(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


if __name__ == "__main__":
    parsed = parse_all_notes()
    save_parsed(parsed)
    llm.save_usage()
    for n in parsed:
        print(n.note_id, n.applies_to, n.note_date, n.effect, n.end_date, n.magnitude_min_pct, n.magnitude_max_pct)
