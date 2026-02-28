import argparse
import csv
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import shutil


@dataclass(frozen=True)
class Config:
    serp_csv: str
    out_root: str
    scraped_csv: str
    bot_cmd: str
    chunk_size: int
    batch_size: int
    sleep_s: float
    timeout_ms: int
    preset: str
    retry_failed: bool
    max_sites: int
    min_free_gb: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_url(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return ""

    try:
        parts = urlsplit(s)
    except Exception:
        return ""

    scheme = (parts.scheme or "http").lower()
    netloc = parts.netloc.lower()

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)

    # Normalize tracking params away to stabilize "unique pages"
    drop_prefixes = ("utm_",)
    drop_keys = {
        "gclid",
        "fbclid",
        "yclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }

    filtered: list[tuple[str, str]] = []
    for k, v in query_pairs:
        k_low = k.lower()
        if k_low in drop_keys:
            continue
        if any(k_low.startswith(p) for p in drop_prefixes):
            continue
        filtered.append((k, v))

    filtered.sort(key=lambda x: (x[0], x[1]))
    query = urlencode(filtered, doseq=True)

    frag = ""
    return urlunsplit((scheme, netloc, path, query, frag))


def sanitize_file_base(file_base: str) -> str:
    """
    file_name CSV’den “safe filename” olarak geliyor varsayıyoruz.
    Yine de güvenlik için çok sınırlı bir sanitize uyguluyoruz.
    """
    s = str(file_base).strip()
    if not s:
        return ""
    # Olası uzantıları temizle (yanlışlıkla gelirse)
    for ext in (".html", ".htm", ".png", ".jpg", ".jpeg", ".json"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    # izin verilen karakterler
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in allowed else "_" for ch in s)
    # çok uzunsa kırp
    return cleaned[:180]


def get_free_gb(path_for_fs: str) -> float:
    usage = shutil.disk_usage(path_for_fs)
    return usage.free / (1024 ** 3)


def stop_if_low_disk(cfg: Config) -> None:
    free_gb = get_free_gb(cfg.out_root)
    if free_gb < float(cfg.min_free_gb):
        raise SystemExit(
            f"Stopping: free disk space is {free_gb:.2f} GB < {cfg.min_free_gb} GB threshold."
        )


def read_latest_state(scraped_csv: str) -> Tuple[Dict[str, Dict[str, str]], int]:
    if not os.path.exists(scraped_csv):
        return {}, 0

    df = pd.read_csv(scraped_csv, low_memory=False)
    required = {"normalized_url", "status", "seq"}
    if not required.issubset(set(df.columns)):
        return {}, 0

    df = df.dropna(subset=["normalized_url", "status", "seq"]).copy()
    df["normalized_url"] = df["normalized_url"].astype(str)
    df["status"] = df["status"].astype(str)
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
    df = df.dropna(subset=["seq"])
    df["seq"] = df["seq"].astype(int)

    # last status for each normalized_url
    df = df.sort_values(["normalized_url", "seq", "started_at"], kind="mergesort")
    last = df.groupby("normalized_url", as_index=False).tail(1)

    state: Dict[str, Dict[str, str]] = {}
    for _, r in last.iterrows():
        nurl = str(r["normalized_url"])
        state[nurl] = {
            "status": str(r.get("status", "")),
            "folder": str(r.get("folder", "")),
            "file_base": str(r.get("file_base", "")),
            "seq": str(int(r.get("seq", 0))),
        }

    max_seq = int(df["seq"].max()) if not df.empty else 0
    return state, max_seq + 1


def append_scraped_row(scraped_csv: str, row: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(scraped_csv) or ".")
    file_exists = os.path.exists(scraped_csv)
    with open(scraped_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def iter_serp_rows(serp_csv: str, chunk_size: int) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(serp_csv, low_memory=False, chunksize=chunk_size):
        yield chunk


def compute_paths(
    out_root: str,
    seq: int,
    batch_size: int,
    file_base: str,
) -> Tuple[str, str, str, str, str]:
    batch_idx = seq // batch_size
    folder = f"{batch_idx:03d}"
    folder_path = os.path.join(out_root, folder)
    ensure_dir(folder_path)

    html_path = os.path.join(folder_path, f"{file_base}.html")
    ss_path = os.path.join(folder_path, f"{file_base}.png")
    metrics_path = os.path.join(folder_path, f"{file_base}.json")

    return folder, file_base, html_path, ss_path, metrics_path


def resolve_collision(folder_path: str, file_base: str, normalized_url: str) -> str:
    """
    Aynı file_base klasörde varsa (çok düşük ihtimal), sonuna kısa hash ekler.
    """
    html_path = os.path.join(folder_path, f"{file_base}.html")
    png_path = os.path.join(folder_path, f"{file_base}.png")
    json_path = os.path.join(folder_path, f"{file_base}.json")

    if not (os.path.exists(html_path) or os.path.exists(png_path) or os.path.exists(json_path)):
        return file_base

    short = hashlib_sha1_8(normalized_url)
    return f"{file_base}_{short}"


def hashlib_sha1_8(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def run_bot(cfg: Config, url: str, out_html: str, out_ss: str, out_metrics: str) -> Tuple[bool, str]:
    cmd = shlex.split(cfg.bot_cmd)
    cmd += [
        "--url",
        url,
        "--out-html",
        out_html,
        "--out-ss",
        out_ss,
        "--out-metrics",
        out_metrics,
        "--timeout-ms",
        str(cfg.timeout_ms),
        "--preset",
        cfg.preset,
    ]

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return True, (p.stdout or "").strip()[:500]
        err = (p.stderr or p.stdout or "").strip()
        return False, err[:1000]
    except Exception as e:
        return False, str(e)[:1000]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Iterate serp_results.csv, call bot, persist html/png/metrics with resume."
    )
    ap.add_argument("--serp", type=str, required=True, help="serp_results.csv (must include link,file_name columns)")
    ap.add_argument("--out-root", type=str, default="data/scraping", help="Root folder for batches 000..")
    ap.add_argument("--scraped-csv", type=str, default="data/scraping/scraped.csv", help="Append-only log for resume")
    ap.add_argument("--bot-cmd", type=str, default="python src/collection/C/helpers/bot.py", help="Command to run bot (e.g. 'node bot.js')")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument("--preset", type=str, default="lighthouse-desktop")
    ap.add_argument("--retry-failed", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--max-sites", type=int, default=0, help="0 means no limit")
    ap.add_argument("--min-free-gb", type=int, default=10, help="Stop if free disk < this value")
    
    # Audit/Summary controls from scraping_control helper
    ap.add_argument("--check", action="store_true", help="Run full audit and regenerate scraped/missed logs")
    ap.add_argument("--summary", action="store_true", help="Read and summarize existing scraped log")
    args = ap.parse_args()

    if args.check or args.summary:
        from src.collection.C.helpers import scraping_control
        scraping_control.SCRAPED_CSV = str(args.scraped_csv)
        if args.check:
            if not args.serp:
                raise SystemExit("Error: --serp is required with --check")
            scraping_control.audit_and_build(args.serp, batch_size=int(args.batch_size))
        elif args.summary:
            scraping_control.summarize_scraped_csv(str(args.scraped_csv))
        return
        
    if not args.serp:
        raise SystemExit("Error: --serp argument is required")


    cfg = Config(
        serp_csv=str(args.serp),
        out_root=str(args.out_root),
        scraped_csv=str(args.scraped_csv),
        bot_cmd=str(args.bot_cmd),
        chunk_size=int(args.chunk_size),
        batch_size=int(args.batch_size),
        sleep_s=float(args.sleep),
        timeout_ms=int(args.timeout_ms),
        preset=str(args.preset),
        retry_failed=(str(args.retry_failed).lower() == "true"),
        max_sites=int(args.max_sites),
        min_free_gb=int(args.min_free_gb),
    )

    ensure_dir(cfg.out_root)
    ensure_dir(os.path.dirname(cfg.scraped_csv) or ".")

    state, next_seq = read_latest_state(cfg.scraped_csv)
    processed_ok: Set[str] = set()
    processed_failed: Set[str] = set()

    for nurl, info in state.items():
        st = info.get("status", "")
        if st == "ok":
            processed_ok.add(nurl)
        elif st == "failed":
            processed_failed.add(nurl)

    total_started = 0
    total_ok = 0
    total_failed = 0

    # Dedupe on normalized_url across whole serp file
    seen_urls: Set[str] = set()

    for chunk in iter_serp_rows(cfg.serp_csv, cfg.chunk_size):
        stop_if_low_disk(cfg)

        for col in ["link", "file_name"]:
            if col not in chunk.columns:
                raise SystemExit(f"serp CSV must have '{col}' column")

        # iterate row-wise for stable mapping: link + file_name
        for _, r in chunk.iterrows():
            stop_if_low_disk(cfg)

            raw_url = str(r["link"]).strip()
            nurl = normalize_url(raw_url)
            if not nurl:
                continue

            if nurl in seen_urls:
                continue
            seen_urls.add(nurl)

            if nurl in processed_ok:
                continue
            if (not cfg.retry_failed) and (nurl in processed_failed):
                continue

            if cfg.max_sites > 0 and total_ok >= cfg.max_sites:
                print(f"Reached --max-sites={cfg.max_sites}. Stopping.")
                print(f"ok={total_ok} failed={total_failed} started={total_started}")
                return

            file_base_raw = r["file_name"]
            file_base = sanitize_file_base(file_base_raw)
            if not file_base:
                # fallback: stable from normalized URL
                file_base = f"u_{hashlib_sha1_8(nurl)}"

            seq = next_seq
            next_seq += 1

            folder_idx = seq // cfg.batch_size
            folder = f"{folder_idx:03d}"
            folder_path = os.path.join(cfg.out_root, folder)
            ensure_dir(folder_path)

            # collision safety (should be rare)
            file_base = resolve_collision(folder_path, file_base, nurl)

            folder, file_base, out_html, out_ss, out_metrics = compute_paths(
                out_root=cfg.out_root,
                seq=seq,
                batch_size=cfg.batch_size,
                file_base=file_base,
            )

            started_at = utc_now_iso()
            append_scraped_row(
                cfg.scraped_csv,
                {
                    "seq": seq,
                    "normalized_url": nurl,
                    "link": raw_url,
                    "folder": folder,
                    "file_base": file_base,
                    "status": "started",
                    "started_at": started_at,
                    "finished_at": "",
                    "error": "",
                },
            )
            total_started += 1

            ok, msg = run_bot(cfg, url=nurl, out_html=out_html, out_ss=out_ss, out_metrics=out_metrics)
            finished_at = utc_now_iso()

            if ok:
                total_ok += 1
                processed_ok.add(nurl)
                append_scraped_row(
                    cfg.scraped_csv,
                    {
                        "seq": seq,
                        "normalized_url": nurl,
                        "link": raw_url,
                        "folder": folder,
                        "file_base": file_base,
                        "status": "ok",
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "error": "",
                    },
                )
            else:
                total_failed += 1
                processed_failed.add(nurl)
                append_scraped_row(
                    cfg.scraped_csv,
                    {
                        "seq": seq,
                        "normalized_url": nurl,
                        "link": raw_url,
                        "folder": folder,
                        "file_base": file_base,
                        "status": "failed",
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "error": msg,
                    },
                )

            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)

    print("Done.")
    print(f"ok={total_ok} failed={total_failed} started={total_started}")
    print(f"scraped log: {cfg.scraped_csv}")


if __name__ == "__main__":
    main()
