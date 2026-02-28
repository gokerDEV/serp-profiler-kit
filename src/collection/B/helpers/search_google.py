import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")
CX = os.getenv("GOOGLE_CX", "")


@dataclass(frozen=True)
class Config:
    input_csv: str
    output_dir: str
    exceptions_csv: str
    max_results: int
    overwrite: bool
    sleep_s: float
    max_retries: int
    hl: str
    gl: str
    safe: str
    quote: bool


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_exception(path: str, row: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def build_query(search_term: str, quote: bool) -> str:
    q = str(search_term).strip()
    q = " ".join(q.split())
    if quote and q:
        q = f"\"{q}\""
    return q


def is_retryable(status: Optional[int]) -> bool:
    return status in (429, 500, 502, 503, 504, 403)


def call_cse_with_retry(
    service,
    *,
    q: str,
    cx: str,
    start: int,
    num: int,
    hl: str,
    gl: str,
    safe: str,
    max_retries: int,
) -> Dict[str, object]:
    attempt = 0
    while True:
        try:
            res = (
                service.cse()
                .list(
                    q=q,
                    cx=cx,
                    start=start,
                    num=num,
                    hl=hl,
                    gl=gl,
                    safe=safe,
                )
                .execute()
            )
            if not isinstance(res, dict):
                raise ValueError("Unexpected response (not a dict).")
            return res
        except HttpError as e:
            status: Optional[int] = None
            try:
                status = int(getattr(e.resp, "status", None))
            except Exception:
                status = None

            attempt += 1
            if attempt > max_retries or not is_retryable(status):
                raise

            backoff = min(60.0, (2.0 ** (attempt - 1)) + random.random())
            time.sleep(backoff)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = min(30.0, (2.0 ** (attempt - 1)) + random.random())
            time.sleep(backoff)


def extract_items(res: Dict[str, object], base_rank: int) -> List[Dict[str, object]]:
    items = res.get("items", [])
    if not isinstance(items, list):
        return []

    out: List[Dict[str, object]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        rank = base_rank + i
        out.append(
            {
                "rank": int(rank),
                "link": str(it.get("link", "")),
                "title": str(it.get("title", "")),
                "snippet": str(it.get("snippet", "")),
                "displayLink": str(it.get("displayLink", "")),
                "formattedUrl": str(it.get("formattedUrl", "")),
            }
        )
    return out


def output_path(output_dir: str, section: str, record_id: str) -> str:
    # Store by section for easier browsing
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
    if not API_KEY:
        raise SystemExit("Missing GOOGLE_API_KEY env var.")
    if not CX:
        raise SystemExit("Missing CX env var.")

    ensure_dir(cfg.output_dir)

    df = pd.read_csv(cfg.input_csv, low_memory=False)
    for col in ["id", "section", "search_term"]:
        if col not in df.columns:
            raise SystemExit(f"Input CSV must contain columns: id,section,search_term. Missing: {col}")

    df = df.dropna(subset=["id", "search_term"]).copy()
    df["id"] = df["id"].astype(str)

    service = build("customsearch", "v1", developerKey=API_KEY)

    # Google CSE: num max is 10. 
    num = 10
    max_results = max(1, min(int(cfg.max_results), 100))

    pages = (max_results + num - 1) // num

    for idx, row in df.iterrows():
        record_id = str(row["id"]).strip()
        section = str(row["section"]).strip().lower() if not pd.isna(row["section"]) else "unknown"
        q = build_query(str(row["search_term"]), cfg.quote)

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

        for page in range(1, pages + 1):
            start = (page - 1) * num + 1
            # Hard cap: cannot go beyond 100.
            if start + num - 1 > 100:
                break

            try:
                res = call_cse_with_retry(
                    service,
                    q=q,
                    cx=CX,
                    start=start,
                    num=num,
                    hl=cfg.hl,
                    gl=cfg.gl,
                    safe=cfg.safe,
                    max_retries=cfg.max_retries,
                )
                all_items.extend(extract_items(res, base_rank=start))
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
                        "page": int(page),
                        "start": int(start),
                        "error": error_msg[:500],
                    },
                )
                break

            time.sleep(cfg.sleep_s)

        payload: Dict[str, object] = {
            "id": record_id,
            "section": section,
            "query": q,
            "engine": "google_cse",
            "max_results_requested": int(max_results),
            "items": all_items[:max_results],
            "ok": bool(ok),
            "error": error_msg,
            "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "params": {"hl": cfg.hl, "gl": cfg.gl, "safe": cfg.safe, "cx": CX},
        }

        write_json(out_path, payload, overwrite=cfg.overwrite)



