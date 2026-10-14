import hashlib
import json

import requests

from src import config

_usage = {"llm_calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}


def _cache_key(system: str, prompt: str, as_json: bool) -> str:
    payload = json.dumps(
        [config.LLM_MODEL, config.LLM_TEMPERATURE, config.LLM_SEED, system, prompt, as_json]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call_ollama(system: str, prompt: str, as_json: bool) -> dict:
    body = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED},
    }
    if as_json:
        body["format"] = "json"
    response = requests.post(config.OLLAMA_URL, json=body, timeout=config.LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def ask(system: str, prompt: str, as_json: bool = False):
    path = config.CACHE_DIR / f"{_cache_key(system, prompt, as_json)}.json"
    if path.exists():
        _usage["cache_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["answer"]
    result = _call_ollama(system, prompt, as_json)
    text = result["message"]["content"]
    answer = json.loads(text) if as_json else text.strip()
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)
    _usage["llm_calls"] += 1
    _usage["input_tokens"] += input_tokens
    _usage["output_tokens"] += output_tokens
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"answer": answer, "input_tokens": input_tokens, "output_tokens": output_tokens}),
        encoding="utf-8",
    )
    return answer


def reset_usage() -> None:
    for key in _usage:
        _usage[key] = 0


def estimated_cost_usd() -> float:
    input_cost = _usage["input_tokens"] / 1_000_000 * config.INPUT_PRICE_PER_MILLION_USD
    output_cost = _usage["output_tokens"] / 1_000_000 * config.OUTPUT_PRICE_PER_MILLION_USD
    return round(input_cost + output_cost, 6)


def usage_summary() -> dict:
    return dict(
        _usage,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        seed=config.LLM_SEED,
        input_price_per_million_usd=config.INPUT_PRICE_PER_MILLION_USD,
        output_price_per_million_usd=config.OUTPUT_PRICE_PER_MILLION_USD,
        estimated_cost_usd=estimated_cost_usd(),
    )


def save_usage(path=config.TOKEN_LOG_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(usage_summary(), indent=2), encoding="utf-8")
