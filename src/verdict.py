from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src import config
from src.notes import ParsedNote, load_parsed
from src.retrieval import NoteIndex, build_query


@dataclass
class Verdict:
    route: str
    week_of: pd.Timestamp
    justified: bool
    matched_note_id: Optional[str]
    rejections: dict


def applies_to_route(note: ParsedNote, route: str) -> bool:
    return note.applies_to == config.ALL_ROUTES or note.applies_to == route


def note_window(note: ParsedNote) -> tuple:
    start = pd.Timestamp(note.note_date)
    if note.end_date:
        return start, pd.Timestamp(note.end_date)
    return start, start + pd.Timedelta(weeks=config.OPEN_ENDED_NOTE_WEEKS) - pd.Timedelta(days=1)


def overlaps_week(note: ParsedNote, week_of: pd.Timestamp) -> bool:
    start, end = note_window(note)
    week_end = week_of + pd.Timedelta(days=6)
    return start <= week_end and end >= week_of


def magnitude_is_plausible(note: ParsedNote, rise_pct: float) -> bool:
    if note.magnitude_max_pct is None:
        return True
    return rise_pct <= note.magnitude_max_pct + config.MAGNITUDE_TOLERANCE_PCT


def rejection_reason(note: ParsedNote, route: str, week_of: pd.Timestamp, rise_pct: float) -> Optional[str]:
    if note.effect != "raises_cost":
        return f"note effect is '{note.effect}', not a cost rise"
    if not applies_to_route(note, route):
        return f"note applies to '{note.applies_to}', not this route"
    if not overlaps_week(note, week_of):
        return "note dates do not cover this week"
    if not magnitude_is_plausible(note, rise_pct):
        return "rise is much larger than the size the note describes"
    return None


def decide(route: str, week_of: pd.Timestamp, rise_pct: float, index: NoteIndex, notes_by_id: dict) -> Verdict:
    rejections = {}
    passing = []
    for note_id, distance in index.search(build_query(route, week_of)):
        note = notes_by_id[note_id]
        reason = rejection_reason(note, route, week_of, rise_pct)
        if reason:
            rejections[note_id] = reason
        else:
            passing.append((note.applies_to == config.ALL_ROUTES, distance, note_id))
    if passing:
        return Verdict(route, week_of, True, min(passing)[2], rejections)
    return Verdict(route, week_of, False, None, rejections)


def judge_candidates(candidates: pd.DataFrame, notes: list) -> list:
    index = NoteIndex(notes)
    notes_by_id = {n.note_id: n for n in notes}
    return [
        decide(row.route, row.week_of, row.vs_own_pct, index, notes_by_id)
        for row in candidates.itertuples()
    ]


def verdicts_table(candidates: pd.DataFrame, verdicts: list) -> pd.DataFrame:
    out = candidates.copy()
    out["justified"] = [v.justified for v in verdicts]
    out["matched_note_id"] = [v.matched_note_id for v in verdicts]
    out["rejections"] = [v.rejections for v in verdicts]
    return out


if __name__ == "__main__":
    from src.flagging import find_candidates

    found = find_candidates()
    table = verdicts_table(found, judge_candidates(found, load_parsed()))
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(config.VERDICTS_CSV, index=False)
    shown = ["route", "week_of", "vs_own_pct", "justified", "matched_note_id"]
    print(table[shown].round(1).to_string())
