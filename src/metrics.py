import pandas as pd

from src import config


def load_shipments(path=config.SHIPMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_cost(shipments: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        shipments.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            n_shipments=("shipment_id", "count"),
        )
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_own_history(weekly: pd.DataFrame, window: int = config.HISTORY_WINDOW_WEEKS) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["_prior"] = weekly.groupby("route")["cost_per_tonne_km"].shift(1)
    rolling = weekly.groupby("route")["_prior"].rolling(window, min_periods=1)
    weekly["own_history_avg"] = rolling.mean().reset_index(level=0, drop=True)
    weekly["history_weeks_used"] = rolling.count().reset_index(level=0, drop=True)
    return weekly.drop(columns="_prior")


def add_peer_average(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    grp = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total = grp.transform("sum")
    count = grp.transform("count")
    weekly["n_peers"] = count - 1
    weekly["peer_avg"] = ((total - weekly["cost_per_tonne_km"]) / weekly["n_peers"]).where(weekly["n_peers"] > 0)
    return weekly


def add_pct_changes(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["vs_own_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_history_avg"] - 1) * 100
    weekly["vs_peers_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_avg"] - 1) * 100
    return weekly


def build_weekly_metrics(path=config.SHIPMENTS_CSV) -> pd.DataFrame:
    weekly = weekly_cost(load_shipments(path))
    weekly = add_own_history(weekly)
    weekly = add_peer_average(weekly)
    return add_pct_changes(weekly)


if __name__ == "__main__":
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = build_weekly_metrics()
    out.to_csv(config.OUTPUT_DIR / "weekly_metrics.csv", index=False)
    print(f"Wrote {len(out)} route-weeks to {config.OUTPUT_DIR / 'weekly_metrics.csv'}")
