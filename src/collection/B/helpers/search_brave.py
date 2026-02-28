import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

BASE_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class Config:
    input_csv: str
    output_dir: str
    exceptions_csv: str
    max_results: int
    overwrite: bool
    sleep_s: float
    max_retries: int
    country: str
    search_lang: str
    ui_lang: str
    safesearch: str
    quote: bool


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_exception(path: str, row: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_query(search_term: str, quote: bool) -> str:
    q = str(search_term).strip()
    q = " ".join(q.split())
    if quote and q:
        q = f"\"{q}\""
    return q


def extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_retryable(status: Optional[int]) -> bool:
    return status in (408, 429, 500, 502, 503, 504)


def call_brave_with_retry(
    session: requests.Session,
    *,
    q: str,
    count: int,
    offset: int,
    country: str,
    search_lang: str,
    ui_lang: str,
    safesearch: str,
    max_retries: int,
    timeout_s: int = 30,
) -> Dict[str, object]:
    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json",
    }
    params: Dict[str, str] = {
        "q": q,
        "count": str(count),
        "offset": str(offset),
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
        "safesearch": safesearch,
    }

    attempt = 0
    while True:
        try:
            resp = session.get(BASE_URL, headers=headers, params=params, timeout=timeout_s)
            status_code = int(resp.status_code)

            if status_code >= 400:
                if attempt < max_retries and is_retryable(status_code):
                    attempt += 1
                    backoff = min(60.0, (2.0 ** (attempt - 1)) + random.random())
                    time.sleep(backoff)
                    continue

                try:
                    body = resp.text[:500]
                except Exception:
                    body = ""
                raise RuntimeError(f"HTTP {status_code}: {body}")

            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Unexpected response (not a dict).")
            return data

        except requests.RequestException as e:
            if attempt >= max_retries:
                raise RuntimeError(str(e)) from e
            attempt += 1
            backoff = min(30.0, (2.0 ** (attempt - 1)) + random.random())
            time.sleep(backoff)
        except Exception:
            if attempt >= max_retries:
                raise
            attempt += 1
            backoff = min(30.0, (2.0 ** (attempt - 1)) + random.random())
            time.sleep(backoff)


def extract_items(res: Dict[str, object], base_rank: int) -> List[Dict[str, object]]:
    web_obj = res.get("web", {})
    if not isinstance(web_obj, dict):
        return []

    results = web_obj.get("results", [])
    if not isinstance(results, list):
        return []

    out: List[Dict[str, object]] = []
    for i, it in enumerate(results):
        if not isinstance(it, dict):
            continue
        rank = base_rank + i
        url = str(it.get("url", ""))
        out.append(
            {
                "rank": int(rank),
                "link": url,
                "title": str(it.get("title", "")),
                "snippet": str(it.get("description", "")),
                "displayLink": extract_domain(url),
            }
        )
    return out


def output_path(output_dir: str, section: str, record_id: str) -> str:
    sec = str(section).strip().lower() if section else "unknown"
    dir_path = os.path.join(output_dir, sec)
    ensure_dir(dir_path)
    return os.path.join(dir_path, f"{record_id}.json")


def write_json(path: str, payload: Dict[str, object], overwrite: bool) -> bool:
    if (not overwrite) and os.path.exists(path):
        return False
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True


def run(cfg: Config) -> None:
    if not BRAVE_API_KEY:
        raise SystemExit("Missing BRAVE_API_KEY env var.")

    ensure_dir(cfg.output_dir)

    df = pd.read_csv(cfg.input_csv, low_memory=False)
    for col in ["id", "section", "search_term"]:
        if col not in df.columns:
            raise SystemExit(f"Input CSV must contain columns: id,section,search_term. Missing: {col}")

    df = df.dropna(subset=["id", "search_term"]).copy()
    df["id"] = df["id"].astype(str)

    total = int(df.shape[0])
    session = requests.Session()

    max_results = max(1, min(int(cfg.max_results), 200))
    count = min(20, max_results)
    pages = (max_results + count - 1) // count

    for idx, row in df.iterrows():
        record_id = str(row["id"]).strip()
        section = str(row["section"]).strip().lower() if not pd.isna(row["section"]) else "unknown"
        q = build_query(str(row["search_term"]), cfg.quote)

        progress = f"[{int(idx) + 1}/{total}]"
        pct = ((int(idx) + 1) / max(1, total)) * 100.0
        print(f"{progress} {pct:6.2f}%  id={record_id}  section={section}", flush=True)

        if not record_id or not q:
            log_exception(
                cfg.exceptions_csv,
                {
                    "row_index": int(idx),
                    "id": record_id,
                    "section": section,
                    "search_term": str(row.get("search_term", "")),
                    "error": "missing_id_or_query",
                },
            )
            continue

        out_path = output_path(cfg.output_dir, section, record_id)
        if (not cfg.overwrite) and os.path.exists(out_path):
            continue

        all_items: List[Dict[str, object]] = []
        ok = True
        error_msg = ""

        for page in range(pages):
            offset = page
            if offset > 9:
                break

            base_rank = offset * count + 1
            try:
                res = call_brave_with_retry(
                    session,
                    q=q,
                    count=count,
                    offset=offset,
                    country=cfg.country,
                    search_lang=cfg.search_lang,
                    ui_lang=cfg.ui_lang,
                    safesearch=cfg.safesearch,
                    max_retries=cfg.max_retries,
                )
                all_items.extend(extract_items(res, base_rank=base_rank))
            except Exception as e:
                ok = False
                error_msg = str(e)
                log_exception(
                    cfg.exceptions_csv,
                    {
                        "row_index": int(idx),
                        "id": record_id,
                        "section": section,
                        "query": q,
                        "offset": int(offset),
                        "count": int(count),
                        "error": error_msg[:500],
                    },
                )
                break

            time.sleep(cfg.sleep_s)

        payload: Dict[str, object] = {
            "id": record_id,
            "section": section,
            "query": q,
            "engine": "brave_web",
            "max_results_requested": int(max_results),
            "items": all_items[:max_results],
            "ok": bool(ok),
            "error": error_msg,
            "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "params": {
                "country": cfg.country,
                "search_lang": cfg.search_lang,
                "ui_lang": cfg.ui_lang,
                "safesearch": cfg.safesearch,
                "count": int(count),
            },
        }

        write_json(out_path, payload, overwrite=cfg.overwrite)



