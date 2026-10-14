import re
import sys
from dataclasses import dataclass, field

import pandas as pd

from src import config, llm
from src.explain import NOTE_ID_PATTERN, NUMBER_PATTERN
from src.notes import load_parsed

MONTH_NUMBERS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

SYSTEM_PROMPT = (
    "You answer questions about freight cost findings. Use only the FACTS given. "
    "Copy numbers, dates and note ids exactly as written in the facts. "
    "Do not add numbers, dates, causes or note ids that are not in the facts. "
    "Never infer or speculate beyond the facts, for example about how long an effect lasted. "
    "Report each week's verdict separately: say which weeks are explained and by which note, "
    "and say which weeks are unexplained. "
    "If the facts do not answer the question, say so. Answer in at most four sentences."
)

UNKNOWN_QUESTION = (
    "I can answer questions about the flagged route-weeks and the notes. "
    "Try naming a route or city, a month or year, a verdict (justified or unexplained), or a note id."
)


@dataclass
class Filters:
    routes: list = field(default_factory=list)
    months: set = field(default_factory=set)
    years: set = field(default_factory=set)
    verdict: str = ""
    note_ids: list = field(default_factory=list)

    def has_row_filters(self) -> bool:
        return bool(self.routes or self.months or self.years or self.verdict)

    def is_empty(self) -> bool:
        return not (self.has_row_filters() or self.note_ids)


@dataclass
class Answer:
    text: str
    sources: list
    used_llm: bool


def known_routes(path=config.SHIPMENTS_CSV) -> list:
    shipments = pd.read_csv(path, usecols=["origin", "destination"])
    return sorted((shipments["origin"] + "-" + shipments["destination"]).unique())


def find_routes(question: str, routes: list) -> list:
    text = question.lower()
    exact = [r for r in routes if all(city.lower() in text for city in r.split("-"))]
    if exact:
        return exact
    return [r for r in routes if any(city.lower() in text for city in r.split("-"))]


def find_months(question: str) -> set:
    text = question.lower()
    found = set()
    for name, number in MONTH_NUMBERS.items():
        if name == "may":
            if re.search(r"\b(in|during|of|for|since|from|to|until) may\b|\bmay 20\d\d\b", text):
                found.add(number)
        elif re.search(rf"\b{name}\b", text):
            found.add(number)
    return found


def find_years(question: str) -> set:
    return {int(y) for y in re.findall(r"\b(20\d\d)\b", question)}


def find_verdict(question: str) -> str:
    text = question.lower()
    if re.search(r"unexplained|not justified|unjustified|no reason|human review|suspicious", text):
        return "Yes"
    if re.search(r"\bjustified\b|\bexplained\b", text):
        return "No (justified)"
    return ""


def find_note_ids(question: str) -> list:
    return sorted({m.upper() for m in re.findall(r"\bn\d{3}\b", question, flags=re.IGNORECASE)})


def parse_question(question: str, routes: list) -> Filters:
    return Filters(
        routes=find_routes(question, routes),
        months=find_months(question),
        years=find_years(question),
        verdict=find_verdict(question),
        note_ids=find_note_ids(question),
    )


def select_rows(flags: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    if not filters.has_row_filters():
        return flags.iloc[0:0]
    keep = pd.Series(True, index=flags.index)
    week_end = flags["week_of"] + pd.Timedelta(days=6)
    if filters.routes:
        keep &= flags["route"].isin(filters.routes)
    if filters.months:
        keep &= flags["week_of"].dt.month.isin(filters.months) | week_end.dt.month.isin(filters.months)
    if filters.years:
        keep &= flags["week_of"].dt.year.isin(filters.years) | week_end.dt.year.isin(filters.years)
    if filters.verdict:
        keep &= flags["flagged"] == filters.verdict
    return flags[keep].sort_values(["route", "week_of"]).head(config.MAX_CHAT_ROWS)


def row_fact(row, notes_by_id: dict) -> str:
    if row.flagged == "No (justified)":
        note = notes_by_id[row.matched_note_id]
        verdict = f"explained by note {note.note_id} dated {note.note_date}, which says: {note.text}"
    else:
        verdict = "unexplained, no supporting note found"
    return (
        f"- {row.route}, week of {row.week_of:%Y-%m-%d}: cost {row.cost_per_tonne_km} per tonne-km, "
        f"{row.vs_own_history}, {row.vs_similar_routes}; verdict: {verdict}. System reason: {row.reason}"
    )


def build_facts(rows: pd.DataFrame, notes_by_id: dict, note_ids: list) -> str:
    lines = [row_fact(row, notes_by_id) for row in rows.itertuples()]
    for note_id in note_ids:
        note = notes_by_id.get(note_id)
        if note:
            lines.append(
                f"- Note {note.note_id} dated {note.note_date}, applies to {note.applies_to}, "
                f"read as effect '{note.effect}': {note.text}"
            )
    return "\n".join(lines)


def no_match_message(filters: Filters) -> str:
    parts = []
    if filters.routes:
        parts.append("route " + ", ".join(filters.routes))
    if filters.months:
        parts.append("month " + ", ".join(str(m) for m in sorted(filters.months)))
    if filters.years:
        parts.append("year " + ", ".join(str(y) for y in sorted(filters.years)))
    if filters.verdict:
        parts.append("verdict " + filters.verdict)
    scope = "; ".join(parts) if parts else "that question"
    return f"The system flagged no week matching ({scope}), so it has no cost rise to explain there."


def is_grounded_answer(answer: str, facts: str, question: str) -> bool:
    if not answer or len(answer) > config.MAX_CHAT_ANSWER_CHARS:
        return False
    allowed_numbers = set(NUMBER_PATTERN.findall(f"{facts} {question}"))
    allowed_ids = set(NOTE_ID_PATTERN.findall(f"{facts} {question}".upper()))
    if any(n not in allowed_numbers for n in NUMBER_PATTERN.findall(answer)):
        return False
    return all(i in allowed_ids for i in NOTE_ID_PATTERN.findall(answer))


def covers_verdicts(answer: str, rows: pd.DataFrame) -> bool:
    if (rows["flagged"] == "Yes").any() and "unexplained" not in answer.lower():
        return False
    justified = rows[rows["flagged"] == "No (justified)"]
    return all(
        f"{row.week_of:%Y-%m-%d}" in answer and row.matched_note_id in answer.upper()
        for row in justified.itertuples()
    )


def row_summary(row, notes_by_id: dict) -> str:
    base = (
        f"- {row.route}, week of {row.week_of:%Y-%m-%d}: cost {row.cost_per_tonne_km} per tonne-km "
        f"({row.vs_own_history}; {row.vs_similar_routes})"
    )
    if row.flagged == "No (justified)":
        note = notes_by_id[row.matched_note_id]
        return f"{base} - explained by note {note.note_id}: {note.text}"
    return f"{base} - unexplained, no supporting note found"


def build_summary(rows: pd.DataFrame, notes_by_id: dict, note_ids: list) -> str:
    lines = [row_summary(row, notes_by_id) for row in rows.itertuples()]
    for note_id in note_ids:
        note = notes_by_id[note_id]
        lines.append(f"- Note {note.note_id} dated {note.note_date}, applies to {note.applies_to}: {note.text}")
    return "\n".join(lines)


def answer_question(question: str, flags: pd.DataFrame, notes_by_id: dict, routes: list) -> Answer:
    filters = parse_question(question, routes)
    if filters.is_empty():
        return Answer(UNKNOWN_QUESTION, [], False)
    rows = select_rows(flags, filters)
    known_note_ids = [n for n in filters.note_ids if n in notes_by_id]
    if rows.empty and not known_note_ids:
        return Answer(no_match_message(filters), [], False)
    facts = build_facts(rows, notes_by_id, known_note_ids)
    sources = [f"{r.route} {r.week_of:%Y-%m-%d}" for r in rows.itertuples()] + known_note_ids
    draft = " ".join(llm.ask(SYSTEM_PROMPT, f"FACTS:\n{facts}\n\nQUESTION: {question}").split())
    if is_grounded_answer(draft, facts, question) and covers_verdicts(draft, rows):
        return Answer(draft, sources, True)
    return Answer(build_summary(rows, notes_by_id, known_note_ids), sources, True)


def load_context(flags_path=config.FLAGS_CSV):
    flags = pd.read_csv(flags_path, keep_default_na=False, parse_dates=["week_of"])
    notes_by_id = {n.note_id: n for n in load_parsed()}
    return flags, notes_by_id, known_routes()


def show(result: Answer) -> None:
    print(result.text)
    if result.sources:
        print("Based on: " + "; ".join(result.sources))


def main(argv: list) -> int:
    flags, notes_by_id, routes = load_context()
    if argv:
        show(answer_question(" ".join(argv), flags, notes_by_id, routes))
        return 0
    print("Ask about the findings (blank line to quit).")
    while True:
        question = input("> ").strip()
        if not question:
            return 0
        show(answer_question(question, flags, notes_by_id, routes))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
