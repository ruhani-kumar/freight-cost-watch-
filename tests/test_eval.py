import pandas as pd

from eval import run_eval

LABELS = pd.DataFrame(
    [
        ("A", "2025-01-06", "No (justified)", "N001", "justified"),
        ("B", "2025-01-06", "Yes", "", "unexplained"),
        ("C", "2025-01-06", "not flagged", "", "normal"),
        ("D", "2025-01-06", "not flagged", "", "normal"),
    ],
    columns=["route", "week_of", "expected", "expected_note_id", "kind"],
)


def flags_frame(rows):
    return pd.DataFrame(rows, columns=["route", "week_of", "flagged", "matched_note_id", "reason"])


def test_perfect_predictions_score_one():
    flags = flags_frame([
        ("A", "2025-01-06", "No (justified)", "N001", "Matches note N001 dated x: y"),
        ("B", "2025-01-06", "Yes", "", "r"),
    ])
    report = run_eval.score(run_eval.predictions_for(LABELS, flags))
    assert report["flag_precision"] == 1.0 and report["flag_recall"] == 1.0
    assert report["verdict_accuracy_on_caught"] == 1.0
    assert report["wrongly_justified_rows"] == []


def test_missing_flag_and_extra_flag_are_counted():
    flags = flags_frame([
        ("A", "2025-01-06", "No (justified)", "N001", "Matches note N001 dated x: y"),
        ("C", "2025-01-06", "Yes", "", "r"),
    ])
    report = run_eval.score(run_eval.predictions_for(LABELS, flags))
    assert report["false_negatives"] == 1
    assert report["false_positives"] == 1
    assert report["flag_recall"] == 0.5


def test_justified_where_unexplained_was_expected_is_reported_as_dangerous():
    flags = flags_frame([
        ("A", "2025-01-06", "No (justified)", "N001", "Matches note N001 dated x: y"),
        ("B", "2025-01-06", "No (justified)", "N003", "Matches note N003 dated x: y"),
    ])
    report = run_eval.score(run_eval.predictions_for(LABELS, flags))
    assert report["wrongly_justified_rows"] == [{"route": "B", "week_of": "2025-01-06", "predicted_note_id": "N003"}]


def test_wrong_note_id_lowers_note_accuracy():
    flags = flags_frame([("A", "2025-01-06", "No (justified)", "N002", "Matches note N002 dated x: y")])
    report = run_eval.score(run_eval.predictions_for(LABELS, flags))
    assert report["correct_note_id_on_justified"] == 0.0


def test_audit_passes_a_correct_justified_row():
    flags = flags_frame([("Chennai-Bangalore", "2025-03-03", "No (justified)", "N001", "Matches note N001 dated x: y")])
    assert run_eval.audit_justified_rows(flags) == []


def test_audit_catches_wrong_route_wrong_window_and_unknown_note():
    flags = flags_frame([
        ("Delhi-Jaipur", "2025-03-03", "No (justified)", "N001", "Matches note N001 dated x: y"),
        ("Chennai-Bangalore", "2025-03-10", "No (justified)", "N001", "Matches note N001 dated x: y"),
        ("Chennai-Bangalore", "2025-03-03", "No (justified)", "N005", "Matches note N005 dated x: y"),
    ])
    failures = run_eval.audit_justified_rows(flags)
    assert len(failures) == 3


def test_audit_catches_reason_that_does_not_cite_the_note():
    flags = flags_frame([("Chennai-Bangalore", "2025-03-03", "No (justified)", "N001", "Flooding happened")])
    assert len(run_eval.audit_justified_rows(flags)) == 1


def test_labeled_set_is_well_formed():
    labels = run_eval.load_labels()
    assert set(labels["expected"]) <= {"No (justified)", "Yes", "not flagged"}
    justified = labels[labels["expected"] == "No (justified)"]
    assert (justified["expected_note_id"] != "").all()
    assert labels.duplicated(["route", "week_of"]).sum() == 0
    assert pd.to_datetime(labels["week_of"]).dt.weekday.eq(0).all()


def run_frame(core_value, reason):
    return pd.DataFrame(
        [("A", "2025-01-06", core_value, reason)],
        columns=["route", "week_of", "cost_per_tonne_km", "reason"],
    )


def test_runs_with_only_different_wording_still_count_as_identical():
    result = run_eval.compare_runs([run_frame("3.29", "wording one"), run_frame("3.29", "wording two")])
    assert result[0]["core_identical"] is True
    assert result[0]["reasons_different"] == 1


def test_runs_with_a_different_number_are_not_identical():
    result = run_eval.compare_runs([run_frame("3.29", "same"), run_frame("3.30", "same")])
    assert result[0]["core_identical"] is False
    assert result[0]["reasons_different"] == 0


def test_every_later_run_is_compared_with_the_first():
    tables = [run_frame("1", "a"), run_frame("1", "a"), run_frame("2", "a")]
    result = run_eval.compare_runs(tables)
    assert [item["run"] for item in result] == [2, 3]
    assert [item["core_identical"] for item in result] == [True, False]


def test_report_says_fail_when_any_run_differs():
    usage = {"model": "m", "temperature": 0, "seed": 0, "llm_calls": 1, "input_tokens": 2, "output_tokens": 3}
    runs = [{"number": n, "hash": "h", "usage": usage} for n in (1, 2)]
    text = run_eval.format_repro_report(runs, [{"run": 2, "core_identical": False, "reasons_different": 0}])
    assert "RESULT: FAIL" in text
