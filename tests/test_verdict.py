import pandas as pd
import pytest

from src import verdict
from src.notes import ParsedNote
from src.retrieval import NoteIndex, build_query


def note(note_id, applies_to, note_date, effect="raises_cost", end_date=None, mag_max=None, text="text"):
    return ParsedNote(note_id, note_date, applies_to, text, effect, end_date, None, mag_max)


FLOOD = note("N001", "Chennai-Bangalore", "2025-02-24", end_date="2025-03-08")
FESTIVAL = note("N002", "Ahmedabad-Mumbai", "2025-01-20", end_date="2025-01-26")
DIESEL = note("N003", "All Routes", "2025-05-05", mag_max=7.0)
NO_EFFECT = note("N005", "Mumbai-Delhi", "2024-07-29", effect="no_effect")
RECOVERED = note("N009", "Chennai-Bangalore", "2025-03-17", effect="back_to_normal")
ALL_NOTES = [FLOOD, FESTIVAL, DIESEL, NO_EFFECT, RECOVERED]


class StubIndex:
    def search(self, query, k=5):
        return [(n.note_id, 0.1 * i) for i, n in enumerate(ALL_NOTES)]


def decide(route, week, rise=20.0):
    return verdict.decide(
        route, pd.Timestamp(week), rise, StubIndex(), {n.note_id: n for n in ALL_NOTES}
    )


def test_note_matching_route_and_week_is_justified():
    result = decide("Chennai-Bangalore", "2025-03-03")
    assert result.justified and result.matched_note_id == "N001"


def test_week_after_the_note_ends_is_not_justified():
    result = decide("Chennai-Bangalore", "2025-03-10")
    assert not result.justified and result.matched_note_id is None
    assert result.rejections["N001"] == "note dates do not cover this week"


def test_week_before_the_note_starts_is_not_justified():
    assert not decide("Chennai-Bangalore", "2025-02-17").justified


def test_note_for_a_different_route_is_not_used():
    result = decide("Delhi-Jaipur", "2025-03-03")
    assert not result.justified
    assert "not this route" in result.rejections["N001"]


def test_no_effect_note_never_justifies():
    result = decide("Mumbai-Delhi", "2024-07-29")
    assert not result.justified
    assert "not a cost rise" in result.rejections["N005"]


def test_back_to_normal_note_never_justifies():
    assert "not a cost rise" in decide("Chennai-Bangalore", "2025-03-17").rejections["N009"]


def test_week_overlapping_the_note_start_counts():
    assert decide("Ahmedabad-Mumbai", "2025-01-20").matched_note_id == "N002"
    assert decide("Ahmedabad-Mumbai", "2025-01-13").matched_note_id is None


def test_all_routes_note_applies_to_any_route_inside_its_window():
    assert decide("Mumbai-Pune", "2025-05-12", rise=6.0).matched_note_id == "N003"


def test_open_ended_note_stops_after_the_configured_weeks():
    assert decide("Mumbai-Pune", "2025-05-26", rise=6.0).matched_note_id == "N003"
    assert decide("Mumbai-Pune", "2025-06-02", rise=6.0).matched_note_id is None


def test_rise_far_bigger_than_the_note_describes_is_not_justified():
    result = decide("Mumbai-Delhi", "2025-05-12", rise=25.0)
    assert not result.justified
    assert "larger than" in result.rejections["N003"]


def test_route_specific_note_beats_all_routes_note():
    both = [DIESEL, note("N900", "Mumbai-Pune", "2025-05-05", end_date="2025-05-18")]

    class Both:
        def search(self, query, k=5):
            return [(n.note_id, 0.1) for n in both]

    result = verdict.decide(
        "Mumbai-Pune", pd.Timestamp("2025-05-12"), 6.0, Both(), {n.note_id: n for n in both}
    )
    assert result.matched_note_id == "N900"


@pytest.fixture(scope="module")
def real_index():
    from src.notes import load_notes

    rows = load_notes()
    notes = [
        ParsedNote(r.note_id, r.date, r.applies_to, r.note, "raises_cost", None, None, None)
        for r in rows.itertuples()
    ]
    return NoteIndex(notes)


def test_retrieval_finds_the_flood_note_for_the_flood_week(real_index):
    ids = [i for i, _ in real_index.search(build_query("Chennai-Bangalore", pd.Timestamp("2025-03-03")))]
    assert "N001" in ids


def test_retrieval_finds_the_festival_note_for_the_festival_week(real_index):
    ids = [i for i, _ in real_index.search(build_query("Ahmedabad-Mumbai", pd.Timestamp("2025-01-20")))]
    assert "N002" in ids


def test_retrieval_finds_the_diesel_note(real_index):
    ids = [i for i, _ in real_index.search(build_query("Mumbai-Pune", pd.Timestamp("2025-05-12")))]
    assert "N003" in ids
