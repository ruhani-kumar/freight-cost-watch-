import numpy as np
import pandas as pd

from src import flagging

def make_row(route, week, cost, own, peer, z):
    return {
        "route": route,
        "week_of": pd.Timestamp(week),
        "cost_per_tonne_km": cost,
        "own_history_avg": own,
        "vs_own_pct": (cost / own - 1) * 100,
        "vs_peers_pct": (cost / peer - 1) * 100,
        "z_score": z,
    }

def candidate_flag(cost, own, peer, z):
    weekly = pd.DataFrame([make_row("R", "2024-01-01", cost, own, peer, z)])
    return bool(flagging.candidate_mask(weekly).iloc[0])


def test_z_score_uses_only_earlier_weeks():
    values = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 5.0]
    weekly = pd.DataFrame(
        {
            "route": "R",
            "week_of": pd.date_range("2024-01-01", periods=len(values), freq="7D"),
            "cost_per_tonne_km": values,
            "own_history_avg": [np.nan] + [np.mean(values[max(0, i - 8):i]) for i in range(1, len(values))],
        }
    )
    out = flagging.add_z_scores(weekly)
    assert out["z_score"].iloc[:8].isna().all()
    assert out["z_score"].iloc[8] > 10

def test_big_jump_on_both_comparisons_is_flagged():
    assert candidate_flag(cost=1.3, own=1.0, peer=1.0, z=5.0)

def test_small_wobble_is_not_flagged():
    assert not candidate_flag(cost=1.03, own=1.0, peer=1.0, z=3.0)

def test_rise_vs_own_only_is_not_flagged():
    assert not candidate_flag(cost=1.2, own=1.0, peer=1.3, z=5.0)

def test_rise_inside_normal_noise_is_not_flagged():
    assert not candidate_flag(cost=1.08, own=1.0, peer=1.0, z=1.0)

def test_slow_creep_far_above_peers_is_flagged_even_with_low_z():
    assert candidate_flag(cost=1.07, own=1.0, peer=0.8, z=0.8)

def test_find_candidates_on_real_data_is_stable_and_nonempty():
    first = flagging.find_candidates()
    second = flagging.find_candidates()
    assert len(first) > 0
    pd.testing.assert_frame_equal(first, second)
