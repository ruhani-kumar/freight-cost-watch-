import re

from src import config, llm
from src.notes import ParsedNote

SYSTEM_PROMPT = (
    "You rewrite one freight-industry note as a short plain-English explanation of why costs rose. "
    "Use only facts that appear in the note. Do not add numbers, dates, causes or note ids that are not in the note. "
    "Do not use the words 'this week' or 'last week'; describe only the cause. "
    "Write one or two sentences and nothing else."
)

PROMPT_TEMPLATE = """Route: {route}
Week starting: {week_of}
Note: {note}

Explain in one or two plain sentences why freight costs on this route rose, using only the note."""

UNEXPLAINED_BASE = (
    "No supporting note found for this route and week, so the cost rise is unexplained "
    "and flagged for human review."
)

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
NOTE_ID_PATTERN = re.compile(r"\bN\d{3}\b")
TIME_WORDS_PATTERN = re.compile(r"\b(this|last|next) week\b", re.IGNORECASE)


def is_grounded(candidate: str, note: ParsedNote, route: str, week_of: str) -> bool:
    if not candidate or len(candidate) > config.MAX_EXPLANATION_CHARS or "\n" in candidate:
        return False
    if NOTE_ID_PATTERN.search(candidate) or TIME_WORDS_PATTERN.search(candidate):
        return False
    allowed = set(NUMBER_PATTERN.findall(f"{note.text} {route} {week_of}"))
    return all(number in allowed for number in NUMBER_PATTERN.findall(candidate))


def explain_justified(route: str, week_of: str, note: ParsedNote) -> str:
    prompt = PROMPT_TEMPLATE.format(route=route, week_of=week_of, note=note.text)
    candidate = " ".join(llm.ask(SYSTEM_PROMPT, prompt).split())
    body = candidate if is_grounded(candidate, note, route, week_of) else note.text
    return f"Matches note {note.note_id} dated {note.note_date}: {body}"


def explain_unexplained(rejections: dict, notes_by_id: dict) -> str:
    for note_id, reason in rejections.items():
        if reason.startswith("note applies to"):
            continue
        note = notes_by_id[note_id]
        return f"{UNEXPLAINED_BASE} The closest note ({note_id}, {note.note_date}) was not used: {reason}."
    return UNEXPLAINED_BASE
