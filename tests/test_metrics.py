import pandas as pd
import pytest

from src import metrics


def make_shipments(rows):
    df = pd.DataFrame(rows, columns=["route", "route_type", "shipment_date", "cost", "tonnes", "km"])
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["freight_cost_inr"] = df["cost"]
    df["tonne_km"] = df["tonnes"] * df["km"]
    df["shipment_id"] = range(len(df))
    return df


def test_week_is_monday_to_sunday(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text(
        "shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,"
        "freight_cost_inr,shipment_date,transporter\n"
        "1,A,B,Short,x,10,100,1000,2024-01-07,t\n"
        "2,A,B,Short,x,10,100,1000,2024-01-08,t\n"
    )
    df = metrics.load_shipments(csv)
    assert list(df["week_of"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-08"]


def test_weekly_cost_is_total_cost_over_total_tonne_km():
    ships = make_shipments([
        ("A-B", "Short", "2024-01-01", 1000, 10, 100),
        ("A-B", "Short", "2024-01-02", 3000, 20, 100),
    ])
    w = metrics.weekly_cost(ships)
    assert w.loc[0, "cost_per_tonne_km"] == pytest.approx(4000 / 3000)


def _weekly(values_by_route, route_type="Short"):
    rows = []
    for route, vals in values_by_route.items():
        for i, v in enumerate(vals):
            rows.append((route, route_type, pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i), v))
    return pd.DataFrame(rows, columns=["route", "route_type", "week_of", "cost_per_tonne_km"])


def test_own_history_excludes_current_week_and_uses_available_weeks():
    w = metrics.add_own_history(_weekly({"R": [1.0, 2.0, 3.0, 10.0]}))
    assert pd.isna(w.loc[0, "own_history_avg"])
    assert w.loc[1, "own_history_avg"] == pytest.approx(1.0)
    assert w.loc[3, "own_history_avg"] == pytest.approx(2.0)
    assert w.loc[3, "history_weeks_used"] == 3


def test_own_history_caps_at_8_weeks():
    w = metrics.add_own_history(_weekly({"R": [100.0] + [1.0] * 8 + [5.0]}))
    assert w.loc[9, "own_history_avg"] == pytest.approx(1.0)
    assert w.loc[9, "history_weeks_used"] == 8


def test_peer_average_excludes_self_and_other_types():
    w = pd.concat([
        _weekly({"A": [2.0], "B": [4.0], "C": [6.0]}, "Medium"),
        _weekly({"D": [100.0]}, "Long"),
    ], ignore_index=True)
    w = metrics.add_peer_average(w).set_index("route")
    assert w.loc["A", "peer_avg"] == pytest.approx(5.0)
    assert w.loc["C", "peer_avg"] == pytest.approx(3.0)
    assert pd.isna(w.loc["D", "peer_avg"])


def test_pct_changes():
    w = pd.DataFrame({"cost_per_tonne_km": [1.2], "own_history_avg": [1.0], "peer_avg": [1.5]})
    w = metrics.add_pct_changes(w)
    assert w.loc[0, "vs_own_pct"] == pytest.approx(20.0)
    assert w.loc[0, "vs_peers_pct"] == pytest.approx(-20.0)
