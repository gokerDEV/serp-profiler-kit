import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import glob
import pandas as pd


# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="Z Step - Build/Update core index.parquet")
parser.add_argument("--serp", required=True, help="Path to serp.csv")
parser.add_argument("--scraping-root", required=True, help="Root dir: scraping/")
parser.add_argument("--out", required=True, help="Output parquet path (e.g., data/A/index.parquet)")
parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
parser.add_argument("--html-captcha-threshold-kb", type=int, default=50, help="Captcha threshold in KB")
args = parser.parse_args()


# -----------------------------
# Config
# -----------------------------
SERP_FILE = args.serp
SCRAPING_ROOT = Path(args.scraping_root).resolve()
OUTPUT_FILE = args.out
MAX_WORKERS = int(args.workers)
CAPTCHA_BYTES = int(args.html_captcha_threshold_kb) * 1024
SOURCE_DOMAIN = "theguardian.com"


# -----------------------------
# Status policy
# -----------------------------
# stronger is better
STATUS_PRIORITY = {
    "ok": 5,
    "soft_fail": 4,
    "captcha": 3,
    "fail": 2,
    "missing_artifact": 1,
}

# -----------------------------------
# Helpers
# -----------------------------------
def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def normalize_url(url: str) -> str:
    u = normalize_text(url)
    if u.endswith("/"):
        u = u[:-1]
    return u


def deterministic_record_id(row: pd.Series) -> str:
    """
    Stable key for SERP row identity.
    """
    search_term = normalize_text(row.get("search_term", "")).lower()
    search_engine = normalize_text(row.get("search_engine", "")).lower()
    rank_raw = row.get("rank", "")
    url = normalize_url(normalize_text(row.get("url", "")))

    try:
        rank = int(rank_raw)
    except Exception:
        rank = 0

    raw = f"{search_term}|{search_engine}|{rank}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_iter_source_from_path(json_path: Path, scraping_root: Path) -> Tuple[int, str, str]:
    """
    Expected pattern:
      scraping/01-bot/000/<file>.json
      scraping/02-bot/000/<file>.json
      scraping/03-extension/042/<file>.json

    Returns:
      iteration, source, folder_rel
    """
    try:
        rel = json_path.relative_to(scraping_root)
        parts = rel.parts
        # parts[0] = "1-node" etc.
        if len(parts) < 3:
            return (0, "unknown", str(rel.parent).replace("\\", "/"))

        iter_source = parts[0]  # e.g., "1-node"
        folder_rel = str(rel.parent).replace("\\", "/")

        if "-" in iter_source:
            left, right = iter_source.split("-", 1)
            try:
                iteration = int(left)
            except Exception:
                iteration = 0
            source = right.strip().lower() if right.strip() else "unknown"
        else:
            # fallback
            iteration = 0
            source = iter_source.strip().lower() or "unknown"

        return (iteration, source, folder_rel)
    except Exception:
        return (0, "unknown", str(json_path.parent))


def canonical_status(
    raw_ok: Optional[bool],
    raw_error: str,
    html_size_bytes: Optional[int],
    json_parse_ok: bool,
) -> Tuple[str, str]:
    """
    Normalized status rules (your latest spec):
      - invalid json -> fail
      - missing html handled outside
      - raw_ok=false & raw_error non-empty -> soft_fail
      - raw_ok=false & raw_error empty -> fail
      - html_size < 50KB -> captcha
      - else ok
    """
    if not json_parse_ok:
        return ("fail", "invalid_json")

    if raw_ok is False:
        if normalize_text(raw_error):
            return ("soft_fail", "json_ok_false_with_error")
        return ("fail", "json_ok_false_no_error")

    if html_size_bytes is not None and html_size_bytes < CAPTCHA_BYTES:
        return ("captcha", "html_lt_50kb")

    return ("ok", "ok_from_json")


def parse_json_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    return None


def process_file_group(file_name_key: str, json_paths: List[str], scraping_root_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    For one file_name, evaluate all iterations and choose best candidate:
    - Higher status priority wins
    - If tie, higher iteration wins
    """
    scraping_root = Path(scraping_root_str)

    best: Optional[Dict[str, Any]] = None

    # parse + sort by iteration desc first (fast short-circuit for good candidates)
    metas: List[Tuple[int, str, str, Path]] = []
    for p in json_paths:
        jp = Path(p)
        itr, src, folder_rel = parse_iter_source_from_path(jp, scraping_root)
        metas.append((itr, src, folder_rel, jp))
    metas.sort(key=lambda x: x[0], reverse=True)

    for itr, src, folder_rel, json_path in metas:
        rec: Dict[str, Any] = {
            "folder": f"scraping/{folder_rel}".replace("\\", "/"),
            "source": src,
            "iteration": itr,
            "raw_ok": None,
            "raw_error": None,
            "http_status": None,
            "collected_at": None,
            "html_size_bytes": None,
            "status": "missing_artifact",
            "status_reason": "artifact_missing",
        }

        # artifacts
        base = str(json_path)[:-5]  # strip ".json"
        html_path = Path(base + ".html")
        png_path = Path(base + ".png")

        has_json = json_path.exists()
        has_html = html_path.exists()
        has_png = png_path.exists()

        if not (has_json and has_html and has_png):
            rec["status"] = "missing_artifact"
            rec["status_reason"] = "missing_html_or_json_or_png"
        else:
            # json parse
            json_ok = True
            payload: Dict[str, Any] = {}
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                json_ok = False

            if json_ok:
                rec["raw_ok"] = parse_json_bool(payload.get("ok"))
                rec["raw_error"] = normalize_text(payload.get("error"))
                rec["http_status"] = payload.get("http_status")
                rec["collected_at"] = payload.get("collected_at")

                try:
                    rec["html_size_bytes"] = int(html_path.stat().st_size)
                except Exception:
                    rec["html_size_bytes"] = None

                st, reason = canonical_status(
                    raw_ok=rec["raw_ok"],
                    raw_error=rec["raw_error"] or "",
                    html_size_bytes=rec["html_size_bytes"],
                    json_parse_ok=True,
                )
                rec["status"] = st
                rec["status_reason"] = reason
            else:
                rec["status"] = "fail"
                rec["status_reason"] = "invalid_json"

        if best is None:
            best = rec
        else:
            bpr = STATUS_PRIORITY.get(best["status"], 0)
            cpr = STATUS_PRIORITY.get(rec["status"], 0)
            if (cpr > bpr) or (cpr == bpr and rec["iteration"] > best["iteration"]):
                best = rec

        # strong short-circuit:
        # first "ok" in iteration-desc order is good enough
        if best and best["status"] == "ok":
            break

    if best is None:
        best = {
            "folder": None,
            "source": None,
            "iteration": None,
            "raw_ok": None,
            "raw_error": None,
            "http_status": None,
            "collected_at": None,
            "html_size_bytes": None,
            "status": "missing_artifact",
            "status_reason": "no_candidate",
        }

    return file_name_key, best


def main() -> None:
    print(f"[1/5] Loading SERP: {SERP_FILE}")
    df = pd.read_csv(SERP_FILE, low_memory=False)

    # Rename link to url if present
    if "link" in df.columns and "url" not in df.columns:
        print("    Renaming 'link' column to 'url'...")
        df.rename(columns={"link": "url"}, inplace=True)

    required_cols = ["file_name", "url", "search_term", "search_engine", "rank"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in SERP: {missing}")

    # normalize
    df["file_name"] = df["file_name"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()

    # drop invalid
    df = df[df["file_name"] != ""].copy()

    # record id
    if "record_id" not in df.columns:
        df["record_id"] = df.apply(deterministic_record_id, axis=1)

    # Extract Domain from Url (Step Z requirement)
    from urllib.parse import urlparse
    def extract_domain(url):
        try:
            return urlparse(str(url)).netloc
        except Exception:
            return None
            
    df['domain'] = df['url'].apply(extract_domain)

    # dedup exact row identity
    before = len(df)
    df = df.drop_duplicates(subset=["record_id"], keep="first").copy()
    print(f"    SERP rows: {before} -> {len(df)} after record_id dedup")

    valid_filenames = set(df["file_name"].unique())

    # Source Domain Fingerprinting
    print(f"    Flagging source domain: {SOURCE_DOMAIN}")
    df["is_source_domain"] = df["url"].astype(str).str.contains(SOURCE_DOMAIN, case=False, na=False)

    print(f"[2/5] Scanning scraping artifacts in: {SCRAPING_ROOT}")
    # only .json files under iteration folders
    all_json = glob.glob(str(SCRAPING_ROOT / "**" / "*.json"), recursive=True)
    print(f"    Found json files: {len(all_json)}")

    # group by file_name key
    groups: Dict[str, List[str]] = {}
    for jp in all_json:
        stem = Path(jp).stem  # exact key expected to match serp.file_name
        if stem in valid_filenames:
            groups.setdefault(stem, []).append(jp)
        elif f"{stem}.html" in valid_filenames:
            # Fallback for extension artifacts named without .html suffix (e.g. foo.json vs foo.html.json)
            groups.setdefault(f"{stem}.html", []).append(jp)

    print(f"    Matched file_name groups: {len(groups)} / {len(valid_filenames)}")

    print(f"[3/5] Resolving best candidate per file_name (workers={MAX_WORKERS})")
    resolved: Dict[str, Dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_file_group, fname, paths, str(SCRAPING_ROOT))
            for fname, paths in groups.items()
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            fname, rec = fut.result()
            resolved[fname] = rec
            if i % 5000 == 0:
                print(f"    progress: {i}/{len(futures)}")

    print("[4/5] Merging with SERP")
    if resolved:
        res_df = pd.DataFrame.from_dict(resolved, orient="index")
        merged = df.merge(res_df, left_on="file_name", right_index=True, how="left")
    else:
        merged = df.copy()

    # fill missing
    merged["status"] = merged.get("status", pd.Series(index=merged.index)).fillna("missing_artifact")
    merged["status_reason"] = merged.get("status_reason", pd.Series(index=merged.index)).fillna("no_file_found")

    # optional typed cleanup
    for col in ["iteration", "http_status", "html_size_bytes"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    print("[5/5] Saving parquet")
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    # summary
    status_counts = merged["status"].value_counts(dropna=False).to_dict()
    print(f"Done. Rows={len(merged)}")
    print("Status distribution:", status_counts)


if __name__ == "__main__":
    main()
