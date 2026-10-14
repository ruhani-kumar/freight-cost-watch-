import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import config

LABELS_CSV = Path(__file__).resolve().parent / "labeled_set.csv"
REPORT_JSON = config.OUTPUT_DIR / "eval_report.json"
REPRO_DIR = config.OUTPUT_DIR / "repro"
REPRO_REPORT = config.OUTPUT_DIR / "repro_diff.txt"
REASON_COLUMN = "reason"

NOT_FLAGGED = "not flagged"
JUSTIFIED = "No (justified)"

GOLD_NOTE_WINDOWS = {
    "N001": ("Chennai-Bangalore", "2025-02-24", "2025-03-08"),
    "N002": ("Ahmedabad-Mumbai", "2025-01-20", "2025-01-26"),
    "N003": ("All Routes", "2025-05-05", "2025-06-01"),
}


def load_labels(path=LABELS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def load_flags(path=config.FLAGS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def predictions_for(labels: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    flagged = flags.set_index(["route", "week_of"])
    rows = []
    for label in labels.itertuples():
        key = (label.route, label.week_of)
        if key in flagged.index:
            hit = flagged.loc[key]
            rows.append((hit["flagged"], hit["matched_note_id"]))
        else:
            rows.append((NOT_FLAGGED, ""))
    out = labels.copy()
    out["predicted"] = [r[0] for r in rows]
    out["predicted_note_id"] = [r[1] for r in rows]
    return out


def safe_ratio(numerator: int, denominator: int):
    return round(numerator / denominator, 3) if denominator else None


def score(scored: pd.DataFrame) -> dict:
    actual_positive = scored["expected"] != NOT_FLAGGED
    predicted_positive = scored["predicted"] != NOT_FLAGGED
    true_pos = int((actual_positive & predicted_positive).sum())
    false_pos = int((~actual_positive & predicted_positive).sum())
    false_neg = int((actual_positive & ~predicted_positive).sum())
    caught = scored[actual_positive & predicted_positive]
    expected_justified = scored[scored["expected"] == JUSTIFIED]
    wrongly_justified = scored[(scored["predicted"] == JUSTIFIED) & (scored["expected"] != JUSTIFIED)]
    wrong_note = expected_justified[expected_justified["predicted_note_id"] != expected_justified["expected_note_id"]]
    return {
        "labeled_rows": len(scored),
        "flag_precision": safe_ratio(true_pos, true_pos + false_pos),
        "flag_recall": safe_ratio(true_pos, true_pos + false_neg),
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "verdict_accuracy_on_caught": safe_ratio(int((caught["predicted"] == caught["expected"]).sum()), len(caught)),
        "correct_note_id_on_justified": safe_ratio(len(expected_justified) - len(wrong_note), len(expected_justified)),
        "wrongly_justified_rows": wrongly_justified[["route", "week_of", "predicted_note_id"]].to_dict("records"),
        "mismatches": scored[scored["expected"] != scored["predicted"]][
            ["route", "week_of", "expected", "predicted", "kind"]
        ].to_dict("records"),
    }


def audit_justified_rows(flags: pd.DataFrame) -> list:
    failures = []
    for row in flags[flags["flagged"] == JUSTIFIED].itertuples():
        note_id = row.matched_note_id
        if note_id not in GOLD_NOTE_WINDOWS:
            failures.append((row.route, row.week_of, f"cites {note_id!r}, which is not a cost-raising note"))
            continue
        gold_route, start, end = GOLD_NOTE_WINDOWS[note_id]
        week_start = pd.Timestamp(row.week_of)
        week_end = week_start + pd.Timedelta(days=6)
        if gold_route not in ("All Routes", row.route):
            failures.append((row.route, row.week_of, f"{note_id} applies to {gold_route}"))
        if not (pd.Timestamp(start) <= week_end and pd.Timestamp(end) >= week_start):
            failures.append((row.route, row.week_of, f"{note_id} dates {start} to {end} do not cover this week"))
        if f"note {note_id} " not in row.reason:
            failures.append((row.route, row.week_of, "reason text does not cite the matched note id"))
    return failures


def read_run(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compare_runs(tables: list) -> list:
    base = tables[0]
    core_columns = [c for c in base.columns if c != REASON_COLUMN]
    comparisons = []
    for number, table in enumerate(tables[1:], start=2):
        comparisons.append(
            {
                "run": number,
                "core_identical": bool(base[core_columns].equals(table[core_columns])),
                "reasons_different": int((base[REASON_COLUMN] != table[REASON_COLUMN]).sum()),
            }
        )
    return comparisons


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_pipeline_once(number: int) -> dict:
    subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--fresh"],
        cwd=config.ROOT,
        check=True,
        capture_output=True,
    )
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    saved = REPRO_DIR / f"run_{number}.csv"
    shutil.copy(config.FLAGS_CSV, saved)
    usage = json.loads(config.TOKEN_LOG_JSON.read_text(encoding="utf-8"))
    shutil.copy(config.TOKEN_LOG_JSON, REPRO_DIR / f"token_log_run_{number}.json")
    return {"number": number, "path": saved, "hash": file_hash(saved), "usage": usage}


def format_repro_report(runs: list, comparisons: list) -> str:
    first_usage = runs[0]["usage"]
    lines = [
        f"Reproducibility check: {len(runs)} fresh runs of python -m src.pipeline --fresh",
        f"LLM: {first_usage['model']}, temperature {first_usage['temperature']}, seed {first_usage['seed']}",
        "Each run clears the saved LLM answers first, so the LLM is genuinely called again every time.",
        "",
    ]
    for run in runs:
        usage = run["usage"]
        lines.append(
            f"run {run['number']}: file hash {run['hash']}, llm_calls {usage['llm_calls']}, "
            f"input_tokens {usage['input_tokens']}, output_tokens {usage['output_tokens']}"
        )
    lines.append("")
    for item in comparisons:
        lines.append(
            f"run 1 vs run {item['run']}: numbers, flags, verdicts and note ids identical = {item['core_identical']}; "
            f"reason texts that differ = {item['reasons_different']}"
        )
    all_core = all(item["core_identical"] for item in comparisons)
    lines.append("")
    lines.append(f"RESULT: {'PASS' if all_core else 'FAIL'} - all columns except reason wording identical across runs = {all_core}")
    return "\n".join(lines)


def run_reproducibility(count: int = 3) -> int:
    runs = [run_pipeline_once(number) for number in range(1, count + 1)]
    comparisons = compare_runs([read_run(r["path"]) for r in runs])
    report = format_repro_report(runs, comparisons)
    REPRO_REPORT.write_text(report, encoding="utf-8")
    print(report)
    return 0 if all(item["core_identical"] for item in comparisons) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", action="store_true")
    if parser.parse_args().repro:
        return run_reproducibility()
    flags = load_flags()
    report = score(predictions_for(load_labels(), flags))
    failures = audit_justified_rows(flags)
    report["justified_rows_audited"] = int((flags["flagged"] == JUSTIFIED).sum())
    report["audit_failures"] = failures
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 1 if failures or report["wrongly_justified_rows"] else 0


if __name__ == "__main__":
    sys.exit(main())
