from src import explain, llm
from src.notes import ParsedNote

FLOOD = ParsedNote(
    "N001",
    "2025-02-24",
    "Chennai-Bangalore",
    "Heavy flooding on the Chennai-Bangalore highway disrupted truck movement from Feb 24 to Mar 8, forcing longer detours.",
    "raises_cost",
    "2025-03-08",
    None,
    None,
)


def fake_llm(monkeypatch, answer):
    monkeypatch.setattr(llm, "ask", lambda system, prompt, as_json=False: answer)


def test_grounded_sentence_is_kept_and_prefixed_with_the_citation(monkeypatch):
    fake_llm(monkeypatch, "Flooding on the highway forced longer detours.")
    reason = explain.explain_justified("Chennai-Bangalore", "2025-03-03", FLOOD)
    assert reason == "Matches note N001 dated 2025-02-24: Flooding on the highway forced longer detours."


def test_sentence_with_invented_number_falls_back_to_the_note_text(monkeypatch):
    fake_llm(monkeypatch, "Flooding pushed costs up by 40 percent.")
    reason = explain.explain_justified("Chennai-Bangalore", "2025-03-03", FLOOD)
    assert reason.endswith(FLOOD.text)


def test_sentence_citing_another_note_falls_back_to_the_note_text(monkeypatch):
    fake_llm(monkeypatch, "As note N002 says, flooding hurt.")
    assert explain.explain_justified("Chennai-Bangalore", "2025-03-03", FLOOD).endswith(FLOOD.text)


def test_sentence_claiming_this_week_falls_back_to_the_note_text(monkeypatch):
    fake_llm(monkeypatch, "Costs rose this week because of flooding.")
    assert explain.explain_justified("Chennai-Bangalore", "2025-03-03", FLOOD).endswith(FLOOD.text)


def test_empty_or_multiline_answer_falls_back_to_the_note_text(monkeypatch):
    fake_llm(monkeypatch, "")
    assert explain.explain_justified("Chennai-Bangalore", "2025-03-03", FLOOD).endswith(FLOOD.text)
    assert not explain.is_grounded("line one\nline two", FLOOD, "Chennai-Bangalore", "2025-03-03")


def test_overlong_answer_is_rejected():
    assert not explain.is_grounded("word " * 200, FLOOD, "Chennai-Bangalore", "2025-03-03")


def test_numbers_from_the_note_are_allowed():
    assert explain.is_grounded("Flooding lasted from Feb 24 to Mar 8.", FLOOD, "Chennai-Bangalore", "2025-03-03")


def test_unexplained_without_any_note_uses_the_base_message():
    assert explain.explain_unexplained({}, {}) == explain.UNEXPLAINED_BASE


def test_unexplained_names_the_closest_relevant_rejected_note():
    notes = {"N009": ParsedNote("N009", "2025-03-17", "Chennai-Bangalore", "text", "back_to_normal", None, None, None)}
    rejections = {"N009": "note effect is 'back_to_normal', not a cost rise"}
    reason = explain.explain_unexplained(rejections, notes)
    assert "N009" in reason and "2025-03-17" in reason and "not a cost rise" in reason


def test_unexplained_skips_notes_about_other_routes():
    rejections = {"N007": "note applies to 'Delhi-Jaipur', not this route"}
    assert explain.explain_unexplained(rejections, {}) == explain.UNEXPLAINED_BASE
