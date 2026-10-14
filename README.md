# Freight Cost Watch

An assistant that watches weekly freight cost per tonne-km on each route, flags rises that look out of the ordinary, and checks a small set of notes to see whether a real reason explains each rise. Every "justified" verdict cites one note, and a note is only cited if it applies to that route, that week, and says costs went up. If no note qualifies, the row is "Yes" (unexplained) with a blank `matched_note_id`.

Final output: `outputs/flags.csv` (columns match `data/sample_output_format_v2.csv`).

## Result in one line

26 flagged route-weeks: 6 justified (3 by the flood and festival notes, 3 by the diesel note) and 20 unexplained. 3 fresh runs gave identical numbers, flags, verdicts and note ids. Cost of one full run: 16 LLM calls, 4,386 input tokens, 495 output tokens, $0 (local model).

## How to run

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
python -m src.pipeline --fresh
python -m eval.run_eval
python -m eval.run_eval --repro
python -m src.chat "why did Chennai-Bangalore get pricier in March?"
python -m pytest
```

- `python -m src.pipeline --fresh` clears the saved LLM answers and runs everything. It writes `outputs/flags.csv` and `outputs/token_log.json`. Without `--fresh`, saved LLM answers are reused.
- `python -m eval.run_eval` scores the output against the labeled set and audits every justified row.
- `python -m eval.run_eval --repro` runs the pipeline 3 times from scratch and writes `outputs/repro_diff.txt`.
- `python -m src.chat` answers plain-English questions about the findings (see "Asking questions"). Run it with no argument for an interactive prompt.
- Requires Python 3.11, and Ollama running locally with `qwen2.5:7b`.

## How it works

The design principle: code does everything that must be exact and repeatable (the numbers, the date and route checks, the verdict). The LLM does two small jobs only: reading each note into structured fields, and rewording a matched note into a plain sentence. The LLM never decides a verdict and never supplies a note id.

| Step | File | What it does |
|---|---|---|
| 1. Weekly cost | `src/metrics.py` | Group by route, work out each Monday, compute cost per tonne-km, add the two baselines |
| 2. Flag | `src/flagging.py` | Pick rows that are rising and out of the ordinary |
| 3. Read notes | `src/notes.py`, `src/llm.py` | LLM fills a small form per note: effect, end date, size range |
| 4. Retrieve | `src/retrieval.py` | Embed notes, store in ChromaDB, fetch the 5 closest per flagged week |
| 5. Decide | `src/verdict.py` | Strict code checks decide justified or unexplained |
| 6. Explain | `src/explain.py` | Write the `reason`, with checks against invented content |
| 7. Output | `src/pipeline.py` | Run all steps and write the CSV and token log |

### 1. Grouping and weekly cost

- Route is `origin-destination`. `route_type` (Short, Medium, Long) is taken from the file, not derived.
- `week_of` is the Monday of the shipment's Monday to Sunday week.
- Weekly cost per tonne-km = sum of freight cost / sum of (tonnes x km) over that route's shipments in the week. This is the spec formula applied to the group; it is not an average of per-shipment ratios.
- Data: 2,940 shipments, 7 routes, 104 weeks, so 728 route-weeks. No nulls, duplicates or bad values. Every route has every week.

### Baseline definitions (exactly as the spec says)

- **vs. own history:** the mean of the route's weekly cost per tonne-km over the trailing 8 weeks, using only weeks strictly before the current week (no look-ahead). The percentage is `cost / that mean - 1`.
  - **Fewer than 8 prior weeks:** all prior weeks available are used, with no padding or extrapolation. A route's 2nd week uses 1 prior week, its 3rd week uses 2, and so on until 8. The number used is stored per row as `history_weeks_used` in `outputs/weekly_metrics.csv`.
  - A route's very first week has no history, so it has no percentage and cannot be flagged.
- **vs. similar routes:** the mean cost per tonne-km in the same week across all other routes with the same `route_type`, with the route itself excluded. If a route had no peers in a week the value would be blank. Short has 2 routes and Long has 2 routes, so for those a route's peer average is a single other route. Medium has 3 routes.
- Percentages are computed on unrounded values and rounded to 1 decimal only when written. The cost column is rounded to 2 decimals. The first three rows of the sample output file reproduce exactly (4.17 / +35.5% / +21.0%, 3.29 / +29.5% / +22.5%, 3.98 / +9.2% / +23.6%).

### 2. Flagging rule

A week is flagged only if cost is rising and unusual:

1. At least 5% above the route's own history, and
2. at least 5% above similar routes, and
3. either a z-score of at least 2 against that route's own week-to-week noise (computed from earlier weeks only, needing at least 8 of them), or at least 20% above similar routes.

Condition 3 exists because a route that creeps up slowly (Mumbai-Pune in late 2025) inflates its own noise estimate, so its z-score stays low even though it sits 20 to 39% above its peers. The thresholds are in `src/config.py` and were chosen by looking at the distribution of the two percentages: a simple "5% and 5%" rule produced extra flags on Delhi-Chennai noise, and the z-score removed them.

### 3. Notes and retrieval (the AI part)

- **Note reading (10 LLM calls):** the LLM reads each note and returns `effect` (raises_cost, no_effect, lowers_cost, back_to_normal, unrelated), an `end_date` if the note gives one, and a size range if given. The route and start date come straight from the CSV columns. Output is validated, and an unexpected `effect` stops the run. I checked all 10 results by hand against the notes.
- **Retrieval:** notes are embedded with `BAAI/bge-small-en-v1.5` and stored in an in-memory ChromaDB collection. For each flagged week the query is "Freight cost increase on the {route} route during the week of {date}" and the 5 closest notes come back.
- **Verdict (plain code):** for each retrieved note, in order, it must (a) have effect `raises_cost`, (b) apply to this route or to "All Routes", (c) have dates that overlap the Monday to Sunday week, (d) if it states a size, be plausible: the week's rise vs own history must be at most the note's maximum plus 4 percentage points. The first passing note is cited, and a route-specific note beats an "All Routes" note. If none pass, the row is "Yes". Every rejected note is stored with its reason.

Notes with a trap, all handled by the checks above: N001 ended Mar 8 and N009 says roads were normal from Mar 9 (so Chennai-Bangalore Mar 10 and Mar 17 are unexplained), N004 is about routes not in the dataset, N005/N006/N008/N010 say nothing changed or costs were not affected, N007 describes an improvement.

### 4. Explanations

- **Justified rows:** the LLM rewrites the matched note in one or two sentences. The code adds the citation itself ("Matches note N002 dated 2025-01-20: ..."). The sentence is rejected, and the note's own text is used instead, if it is empty, over 400 characters, multi-line, mentions any note id, says "this week" or "last week", or contains any number not present in the note or the route and week.
- **Unexplained rows:** no LLM. A fixed template says no supporting note was found and the rise is flagged for human review, and names the closest note that applied to this route and why it was not used (for example "note effect is 'no_effect', not a cost rise").

## Asking questions (stretch goal)

`src/chat.py` answers questions such as "why did Chennai-Bangalore get pricier in March?", "which rows are unexplained for Mumbai-Pune?" or "what does N003 say?" using only `outputs/flags.csv` and the parsed notes.

- Plain code, not the LLM, works out what is being asked: it looks for route or city names, month names, years, "justified" or "unexplained", and note ids, then selects the matching rows.
- The LLM (same local model, temperature 0, seed 0) only phrases an answer from a block of FACTS: each row's numbers, verdict, and the full text of any cited note.
- The answer is thrown away, and a deterministic one-line-per-week summary is shown instead, if it is empty or over 900 characters, contains a number or note id not in the facts, leaves out the word "unexplained" when any unexplained week is in scope, or fails to name both the week and the note id of every justified week in scope.
- If no flagged week matches, or the question names nothing the system knows, no LLM call is made and the reply says so.
- Chat never writes `outputs/token_log.json`, so the official cost log stays the cost of the main run.

Limits: the checks cover numbers, ids, and which weeks are explained, not every phrase. In testing the model once described the 2025-02-24 week as "the first two weeks of March", which is loose wording, though the dates and note ids it gave were correct. Questions about a whole route over a long period usually fall back to the plain week-by-week list.

## How I checked it was right

`eval/labeled_set.csv` holds 46 route-weeks labeled by hand from the notes and the weekly numbers, each with a one-line reason: 6 justified, 13 unexplained, 9 trap weeks (a note exists but says costs were unaffected, improved or recovered, or cost fell after the flood), 10 plain normal weeks and 8 noisy-but-ordinary weeks.

Results on the final output (`outputs/eval_report.json`):

| Measure | Value |
|---|---|
| Flag precision / recall | 1.0 / 1.0 (19 of 19 real problems caught, 0 false alarms) |
| Verdict correct on caught weeks | 100% |
| Correct note id on justified weeks | 100% |
| Wrongly justified rows | 0 |
| Audit of all 6 justified rows in the output | pass |

The audit checks every justified row, not just labeled ones, against hand-written route and date windows for the cost-raising notes N001 to N003, and checks that the reason text cites the matched note id. This is the spot-check the brief says will be applied. The script exits with an error if anything fails.

**Why I trusted it, and its limits.** A score of 100% alone proves little here, because I wrote the labels after seeing the pipeline's output, and the 4-week diesel window and several diesel labels are my own judgment. So I tested whether the eval can fail by breaking the system on purpose:

| Deliberate breakage | What the eval showed |
|---|---|
| Diesel window set to 8 weeks | 3 wrongly justified rows, 3 audit failures |
| Magnitude check switched off | 1 wrongly justified row |
| Flagging thresholds too loose | Precision fell to 0.70 |
| Flagging thresholds too strict | Recall fell to 0.32 |

My first answer key missed the "too loose" case because all its normal weeks were nearly flat. I added the 8 noisy-but-ordinary weeks and it then caught it. In short: this is a small hand-built check that catches regressions and the specific traps. It does not prove correctness on weeks I did not label, and the labels carry my judgment calls.

There are also 76 unit tests (`python -m pytest`) covering the maths, the flagging rule, the wrong-route, wrong-window, "no effect" and magnitude traps, the explanation guards, the output format, and the eval itself.

## Reproducibility

`python -m eval.run_eval --repro` ran the full pipeline 3 times as separate processes, each starting with `--fresh` (saved LLM answers cleared, so the LLM was genuinely called again each time). Saved in `outputs/repro_diff.txt` and `outputs/repro/`.

- LLM settings: `qwen2.5:7b`, **temperature 0, seed 0**. Temperature 0 makes the model choose its most likely word each time, and the seed pins any remaining sampling.
- Result: **PASS**. All columns except `reason` (numbers, flags, verdicts, note ids) were identical across all 3 runs.
- Wording: runs 2 and 3 were byte-identical. Run 1 differed from them in 1 of 6 justified `reason` texts (Chennai-Bangalore 2025-02-24, same note N001, same facts, different phrasing), even at temperature 0 and a fixed seed. I did not determine the cause. It does not move any verdict because no verdict depends on the LLM's output: the cited note comes from the code, and the sentence is only text after it.
- The 20 unexplained reasons are fixed templates and never vary.

## Cost of one full run

From `outputs/token_log.json`, a cold run (cache cleared, `--fresh`):

| | |
|---|---|
| LLM calls | 16 (10 note reads + 6 explanations of justified rows) |
| Input tokens | 4,386 |
| Output tokens | 495 |
| Model | qwen2.5:7b via Ollama, running locally |
| Cost | $0 (self-hosted, no per-token fee) |

For reference, at an illustrative hosted rate of about $0.30 per million tokens for both input and output (a third-party aggregator figure for Qwen2.5-7B-Instruct, not an official price list, and rates vary by provider), the run would cost about $0.0015 (4,881 tokens). Set the rates in `src/config.py` (`INPUT_PRICE_PER_MILLION_USD`, `OUTPUT_PRICE_PER_MILLION_USD`) to include a price in the log.

LLM volume is deliberately small: unexplained rows (20 of 26) use a template and cost nothing, note reads are only 10 calls, and answers are cached on disk by prompt so a rerun makes no LLM calls.

## Known limits and assumptions

- **Diesel note (N003) has no end date.** I assume it explains cost rises for 4 weeks from its start (2025-05-05 to 2025-06-01), and I also reject rises much bigger than the 5 to 7% it describes. This is my judgment and it decides several rows: 3 weeks are justified by N003, and later diesel-period weeks (for example Mumbai-Pune 2025-06-16 and 06-23) are left unexplained. I chose the cautious side because the brief says a wrong "justified" is worse than "couldn't find one". The window is `OPEN_ENDED_NOTE_WEEKS` in `src/config.py`.
- **Nationwide rises do not stand out.** A rise that hits every route (diesel) also lifts the similar-route average, so most routes' diesel weeks are not flagged at all. Only routes that stand out against their peers are flagged. This follows from the "vs. similar routes" definition.
- **Thin peer groups.** For Short and Long, "similar routes" is a single other route.
- **Borderline weeks are not labeled.** A few weeks near the thresholds (for example Mumbai-Pune early 2024 and Mumbai-Delhi October 2024) could reasonably go either way and are not in the eval set.
- **The provided `sample_output_format_v2.csv` is malformed.** Its third data row has an unquoted comma in "(N006, 2025-09-22)", so a standard CSV reader cannot parse it. I matched its header exactly and my output quotes such fields properly.
- **Eval is small and hand-made**, as described above.
- **Wording drift** at temperature 0 was observed once, as described under Reproducibility.

## Project layout

```
README.md  requirements.txt
data/        the 3 input CSVs
src/
  config.py      thresholds, paths, model name, prices
  metrics.py     weekly grouping and the two baselines
  flagging.py    candidate rule
  notes.py       LLM reads notes into structured fields
  retrieval.py   embeddings and ChromaDB
  verdict.py     justified or unexplained decision
  explain.py     reason text and guards
  llm.py         the one LLM wrapper: temperature 0, seed, cache, token log
  pipeline.py    end to end, writes the CSV and token log
  chat.py        plain-English questions about the findings
eval/
  labeled_set.csv   46 hand-labeled route-weeks
  run_eval.py       scoring, audit, and the 3-run reproducibility check
tests/               unit tests
outputs/
  flags.csv         final result
  token_log.json    cost of one full run
  repro_diff.txt    3-run comparison (copies of each run in repro/)
  eval_report.json  latest eval scores
  weekly_metrics.csv, candidates.csv, verdicts.csv, parsed_notes.json   intermediate results
  llm_cache/        saved LLM answers (safe to delete; use --fresh)
```
