from types import SimpleNamespace

import pandas as pd

from src import config, llm, pipeline
from src.notes import ParsedNote
from src.verdict import Verdict

NOTE = ParsedNote("N002", "2025-01-20", "Ahmedabad-Mumbai", "A festival week surcharge applied.", "raises_cost", "2025-01-26", None, None)
CANDIDATE = SimpleNamespace(
    route="Ahmedabad-Mumbai",
    week_of=pd.Timestamp("2025-01-20"),
    cost_per_tonne_km=3.2904,
    vs_own_pct=29.47,
    vs_peers_pct=22.49,
)


def test_percent_format_matches_the_sample_style():
    assert pipeline.format_change(35.52, "this route's past average") == "+35.5% vs this route's past average"
    assert pipeline.format_change(-3.24, "x") == "-3.2% vs x"


def test_justified_row_has_note_id_and_justified_flag(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda system, prompt, as_json=False: "A festival week surcharge applied.")
    verdict = Verdict("Ahmedabad-Mumbai", CANDIDATE.week_of, True, "N002", {})
    row = pipeline.build_row(CANDIDATE, verdict, {"N002": NOTE})
    assert row["flagged"] == "No (justified)"
    assert row["matched_note_id"] == "N002"
    assert row["reason"].startswith("Matches note N002 dated 2025-01-20:")
    assert row["cost_per_tonne_km"] == 3.29
    assert row["week_of"] == "2025-01-20"


def test_unexplained_row_has_blank_note_id_and_yes_flag():
    verdict = Verdict("Ahmedabad-Mumbai", CANDIDATE.week_of, False, None, {})
    row = pipeline.build_row(CANDIDATE, verdict, {})
    assert row["flagged"] == "Yes"
    assert row["matched_note_id"] == ""
    assert "unexplained" in row["reason"]


def test_columns_match_the_sample_file_exactly():
    header = config.SAMPLE_OUTPUT_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == pipeline.OUTPUT_COLUMNS
    row = pipeline.build_row(CANDIDATE, Verdict("r", CANDIDATE.week_of, False, None, {}), {})
    assert list(row.keys()) == header
