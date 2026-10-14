import pandas as pd

from src import config
from src.metrics import build_weekly_metrics


def add_z_scores(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["route", "week_of"]).copy()
    noise = weekly.groupby("route")["cost_per_tonne_km"].transform(
        lambda s: s.shift(1).expanding(min_periods=config.MIN_WEEKS_FOR_NOISE).std()
    )
    weekly["z_score"] = (weekly["cost_per_tonne_km"] - weekly["own_history_avg"]) / noise
    return weekly


def candidate_mask(weekly: pd.DataFrame) -> pd.Series:
    rising = (weekly["vs_own_pct"] >= config.MIN_RISE_VS_OWN_PCT) & (
        weekly["vs_peers_pct"] >= config.MIN_RISE_VS_PEERS_PCT
    )
    unusual = (weekly["z_score"] >= config.MIN_Z_SCORE) | (
        weekly["vs_peers_pct"] >= config.STRONG_PEER_GAP_PCT
    )
    return rising & unusual


def mark_candidates(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = add_z_scores(weekly)
    weekly["is_candidate"] = candidate_mask(weekly)
    return weekly


def find_candidates(path=config.SHIPMENTS_CSV) -> pd.DataFrame:
    marked = mark_candidates(build_weekly_metrics(path))
    return marked[marked["is_candidate"]].reset_index(drop=True)


if __name__ == "__main__":
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    candidates = find_candidates()
    candidates.to_csv(config.OUTPUT_DIR / "candidates.csv", index=False)
    print(f"Wrote {len(candidates)} candidate weeks to {config.OUTPUT_DIR / 'candidates.csv'}")
    shown = ["route", "week_of", "cost_per_tonne_km", "vs_own_pct", "vs_peers_pct", "z_score"]
    print(candidates[shown].round(2).to_string())
