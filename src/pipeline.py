import argparse
import shutil

import pandas as pd

from src import config, llm
from src.explain import explain_justified, explain_unexplained
from src.flagging import find_candidates
from src.notes import parse_all_notes, save_parsed
from src.verdict import Verdict, judge_candidates

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def format_change(value: float, label: str) -> str:
    return f"{value:+.1f}% vs {label}"


def build_row(candidate, verdict: Verdict, notes_by_id: dict) -> dict:
    week = f"{candidate.week_of:%Y-%m-%d}"
    if verdict.justified:
        reason = explain_justified(candidate.route, week, notes_by_id[verdict.matched_note_id])
    else:
        reason = explain_unexplained(verdict.rejections, notes_by_id)
    return {
        "route": candidate.route,
        "week_of": week,
        "cost_per_tonne_km": round(candidate.cost_per_tonne_km, 2),
        "vs_own_history": format_change(candidate.vs_own_pct, "this route's past average"),
        "vs_similar_routes": format_change(candidate.vs_peers_pct, "similar-length routes this week"),
        "flagged": "No (justified)" if verdict.justified else "Yes",
        "matched_note_id": verdict.matched_note_id or "",
        "reason": reason,
    }


def run(fresh: bool = False) -> pd.DataFrame:
    llm.reset_usage()
    if fresh and config.CACHE_DIR.exists():
        shutil.rmtree(config.CACHE_DIR)
    notes = parse_all_notes()
    save_parsed(notes)
    notes_by_id = {n.note_id: n for n in notes}
    candidates = find_candidates()
    verdicts = judge_candidates(candidates, notes)
    rows = [build_row(c, v, notes_by_id) for c, v in zip(candidates.itertuples(), verdicts)]
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(config.FLAGS_CSV, index=False)
    llm.save_usage()
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run(fresh=args.fresh)
    print(f"Wrote {len(result)} rows to {config.FLAGS_CSV}")
    print(llm.usage_summary())
