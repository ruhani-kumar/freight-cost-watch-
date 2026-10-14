import pandas as pd
import pytest

from src import chat, llm
from src.notes import ParsedNote

ROUTES = ["Ahmedabad-Mumbai", "Chennai-Bangalore", "Mumbai-Delhi", "Mumbai-Pune"]

FLOOD = ParsedNote(
    "N001", "2025-02-24", "Chennai-Bangalore",
    "Heavy flooding disrupted truck movement from Feb 24 to Mar 8.",
    "raises_cost", "2025-03-08", None, None,
)
NOTES = {"N001": FLOOD}


def make_flags():
    rows = [
        ("Chennai-Bangalore", "2025-03-03", 3.48, "+22.8% vs this route's past average",
         "+32.0% vs similar-length routes this week", "No (justified)", "N001",
         "Matches note N001 dated 2025-02-24: Flooding forced detours."),
        ("Chennai-Bangalore", "2025-03-10", 3.57, "+22.1% vs this route's past average",
         "+33.8% vs similar-length routes this week", "Yes", "",
         "No supporting note found for this route and week, so the cost rise is unexplained and flagged for human review."),
        ("Mumbai-Pune", "2025-12-08", 4.43, "+7.2% vs this route's past average",
         "+38.8% vs similar-length routes this week", "Yes", "",
         "No supporting note found for this route and week, so the cost rise is unexplained and flagged for human review."),
    ]
    flags = pd.DataFrame(
        rows,
        columns=["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                 "flagged", "matched_note_id", "reason"],
    )
    flags["week_of"] = pd.to_datetime(flags["week_of"])
    return flags


def fake_llm(monkeypatch, text):
    calls = []

    def ask(system, prompt, as_json=False):
        calls.append(prompt)
        return text

    monkeypatch.setattr(llm, "ask", ask)
    return calls


def test_route_is_found_by_full_name_or_single_city():
    assert chat.find_routes("what happened on Chennai-Bangalore", ROUTES) == ["Chennai-Bangalore"]
    assert chat.find_routes("anything on Mumbai?", ROUTES) == ["Ahmedabad-Mumbai", "Mumbai-Delhi", "Mumbai-Pune"]
    assert chat.find_routes("route Mumbai-Pune please", ROUTES) == ["Mumbai-Pune"]


def test_months_and_years_are_found():
    assert chat.find_months("why was it pricier in March 2025?") == {3}
    assert chat.find_months("between Feb and mar") == {2, 3}
    assert chat.find_years("in March 2025 or 2024") == {2024, 2025}


def test_may_is_only_a_month_when_used_like_one():
    assert chat.find_months("what happened in May?") == {5}
    assert chat.find_months("this may be wrong") == set()


def test_unexplained_is_not_mistaken_for_explained():
    assert chat.find_verdict("show unexplained rows") == "Yes"
    assert chat.find_verdict("which rows are not justified") == "Yes"
    assert chat.find_verdict("which rows are justified") == "No (justified)"
    assert chat.find_verdict("tell me about routes") == ""


def test_note_ids_are_found_case_insensitively():
    assert chat.find_note_ids("what does n001 say, and N001 again") == ["N001"]


def test_select_rows_filters_by_route_and_month():
    rows = chat.select_rows(make_flags(), chat.Filters(routes=["Chennai-Bangalore"], months={3}))
    assert list(rows["week_of"].dt.strftime("%Y-%m-%d")) == ["2025-03-03", "2025-03-10"]


def test_select_rows_filters_by_verdict():
    rows = chat.select_rows(make_flags(), chat.Filters(verdict="Yes"))
    assert set(rows["route"]) == {"Chennai-Bangalore", "Mumbai-Pune"}
    assert len(rows) == 2


def test_week_that_straddles_a_month_boundary_counts_for_both_months():
    flags = make_flags()
    flags.loc[0, "week_of"] = pd.Timestamp("2025-02-24")
    assert len(chat.select_rows(flags, chat.Filters(months={3}))) == 2
    assert len(chat.select_rows(flags, chat.Filters(months={2}))) == 1


def test_facts_include_the_cited_note_text_and_the_numbers():
    flags = make_flags()
    rows = chat.select_rows(flags, chat.Filters(routes=["Chennai-Bangalore"], months={3}))
    facts = chat.build_facts(rows, NOTES, [])
    assert "explained by note N001" in facts
    assert "Heavy flooding disrupted truck movement" in facts
    assert "+22.8%" in facts and "unexplained, no supporting note found" in facts


def test_grounded_answer_is_returned_with_sources(monkeypatch):
    fake_llm(monkeypatch, "Week of 2025-03-03 is explained by note N001 (flooding); the week of 2025-03-10 is unexplained.")
    result = chat.answer_question(
        "why did Chennai-Bangalore get pricier in March 2025?", make_flags(), NOTES, ROUTES
    )
    assert result.used_llm
    assert "N001" in result.text
    assert "Chennai-Bangalore 2025-03-03" in result.sources


def test_answer_with_an_invented_number_falls_back_to_the_facts(monkeypatch):
    fake_llm(monkeypatch, "Costs rose 77.7% because of flooding.")
    result = chat.answer_question("why was Chennai-Bangalore pricier in March?", make_flags(), NOTES, ROUTES)
    assert result.text.startswith("- Chennai-Bangalore, week of 2025-03-03")
    assert "77.7" not in result.text


def test_answer_citing_a_note_not_in_the_facts_falls_back_to_the_facts(monkeypatch):
    fake_llm(monkeypatch, "Note N003 explains this rise.")
    result = chat.answer_question("why was Chennai-Bangalore pricier in March?", make_flags(), NOTES, ROUTES)
    assert "N003" not in result.text


def test_no_matching_week_makes_no_llm_call(monkeypatch):
    calls = fake_llm(monkeypatch, "should not be used")
    result = chat.answer_question("anything on Ahmedabad-Mumbai in 2024?", make_flags(), NOTES, ROUTES)
    assert calls == []
    assert not result.used_llm
    assert "flagged no week" in result.text


def test_unrecognised_question_makes_no_llm_call(monkeypatch):
    calls = fake_llm(monkeypatch, "should not be used")
    result = chat.answer_question("what is the meaning of life", make_flags(), NOTES, ROUTES)
    assert calls == []
    assert result.text == chat.UNKNOWN_QUESTION


def test_note_question_returns_the_note_text(monkeypatch):
    fake_llm(monkeypatch, "Note N001 says flooding disrupted truck movement.")
    result = chat.answer_question("what does N001 say?", make_flags(), NOTES, ROUTES)
    assert result.used_llm and "N001" in result.text
    assert result.sources == ["N001"]


def test_unknown_note_id_is_not_invented(monkeypatch):
    calls = fake_llm(monkeypatch, "should not be used")
    result = chat.answer_question("what does N999 say?", make_flags(), NOTES, ROUTES)
    assert calls == []
    assert "flagged no week" in result.text or result.text == chat.UNKNOWN_QUESTION


@pytest.mark.parametrize("text,expected", [("", False), ("x" * 2000, False), ("fine", True)])
def test_answer_length_and_emptiness_guard(text, expected):
    assert chat.is_grounded_answer(text, "facts", "question") is expected


def test_answer_that_hides_unexplained_weeks_falls_back_to_the_facts(monkeypatch):
    fake_llm(monkeypatch, "Heavy flooding disrupted truck movement as note N001 says.")
    result = chat.answer_question("why was Chennai-Bangalore pricier in March?", make_flags(), NOTES, ROUTES)
    assert result.text.startswith("- Chennai-Bangalore, week of 2025-03-03")


def test_answer_that_mentions_the_note_id_without_the_week_falls_back(monkeypatch):
    fake_llm(monkeypatch, "Note N001 is the closest note; the week of 2025-03-10 is unexplained.")
    result = chat.answer_question("why was Chennai-Bangalore pricier in March?", make_flags(), NOTES, ROUTES)
    assert result.text.startswith("- Chennai-Bangalore, week of 2025-03-03")


def test_fallback_summary_is_one_short_line_per_week():
    flags = make_flags()
    rows = chat.select_rows(flags, chat.Filters(routes=["Chennai-Bangalore"]))
    summary = chat.build_summary(rows, NOTES, [])
    assert len(summary.splitlines()) == 2
    assert "explained by note N001" in summary and "unexplained, no supporting note found" in summary


def test_answer_that_omits_the_cited_note_falls_back_to_the_facts(monkeypatch):
    fake_llm(monkeypatch, "The week of 2025-03-03 had flooding and 2025-03-10 is unexplained.")
    result = chat.answer_question("why was Chennai-Bangalore pricier in March?", make_flags(), NOTES, ROUTES)
    assert result.text.startswith("- Chennai-Bangalore, week of 2025-03-03")
