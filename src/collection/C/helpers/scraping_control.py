import os
import json
import csv
import argparse
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

TASKS_ROOT = "data/tasks"
SCRAPING_ROOT = "/Volumes/KIOXIA/scraping/"      # source=node_scraper
SCRAPING_ROOT_WEB = "data/web_scraping/"          # source=web_scraper
SCRAPED_CSV = "data/scraped.csv"
MISSED_CSV = "data/missed.csv"
SOURCE_COVERAGE_CSV = "data/source_coverage.csv"

STATUS_PRIORITY = {
    "ok": 4,
    "captcha": 3,
    "soft_fail": 2,
    "fail": 1,
}


@dataclass(frozen=True)
class ProbeResult:
    status: str               # ok | captcha | soft_fail | fail
    source: str               # node_scraper | web_scraper | none
    cause: str                # "" | captcha | soft_fail | fail
    error: str
    html_exists: bool
    json_exists: bool
    ss_exists: bool
    collected_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def detect_captcha(html_text: str, json_error: str) -> bool:
    t = (html_text or "").lower()
    e = (json_error or "").lower()

    captcha_markers = [
        "captcha",
        "cf-chl-",          # cloudflare challenge
        "attention required",
        "verify you are human",
        "are you human",
        "recaptcha",
        "hcaptcha",
        "press and hold",
        "cloudflare",
        "/captcha/",
    ]
    return any(m in t for m in captcha_markers) or any(m in e for m in captcha_markers)


def classify_from_files(html_path: str, json_path: str, ss_path: str) -> ProbeResult:
    has_html = os.path.exists(html_path)
    has_json = os.path.exists(json_path)
    has_ss = os.path.exists(ss_path)

    # default
    status = "fail"
    cause = "fail"
    error = ""
    collected_at = utc_now_iso()

    if not (has_html and has_json and has_ss):
        return ProbeResult(
            status=status,
            source="none",
            cause=cause,
            error="missing_artifact",
            html_exists=has_html,
            json_exists=has_json,
            ss_exists=has_ss,
            collected_at=collected_at,
        )

    # parse json
    j = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as ex:
        return ProbeResult(
            status="fail",
            source="none",
            cause="fail",
            error=f"json_parse_error:{str(ex)[:200]}",
            html_exists=has_html,
            json_exists=has_json,
            ss_exists=has_ss,
            collected_at=collected_at,
        )

    ok = bool(j.get("ok", False))
    error = str(j.get("error", "") or "").strip()
    if j.get("collected_at"):
        collected_at = str(j.get("collected_at"))

    html_text = ""
    try:
        # küçük bir bölüm okumak yeterli
        with open(html_path, "r", encoding="utf-8", errors="ignore") as hf:
            html_text = hf.read(200000)
    except Exception:
        html_text = ""

    # 1) ok:true ise önce captcha kontrolü
    if ok:
        is_captcha = detect_captcha(html_text, error)
        if is_captcha:
            return ProbeResult(
                status="captcha",
                source="none",
                cause="captcha",
                error=error,
                html_exists=has_html,
                json_exists=has_json,
                ss_exists=has_ss,
                collected_at=collected_at,
            )
        return ProbeResult(
            status="ok",
            source="none",
            cause="",
            error="",
            html_exists=has_html,
            json_exists=has_json,
            ss_exists=has_ss,
            collected_at=collected_at,
        )

    # 2) ok:false => soft_fail / captcha / fail
    # redirected_to_home gibi kontrollü "soft failure" sebeplerini ayırıyoruz.
    lower_err = error.lower()
    if detect_captcha(html_text, error):
        return ProbeResult(
            status="captcha",
            source="none",
            cause="captcha",
            error=error or "captcha_detected",
            html_exists=has_html,
            json_exists=has_json,
            ss_exists=has_ss,
            collected_at=collected_at,
        )

    if lower_err:
        return ProbeResult(
            status="soft_fail",
            source="none",
            cause="soft_fail",
            error=error,
            html_exists=has_html,
            json_exists=has_json,
            ss_exists=has_ss,
            collected_at=collected_at,
        )

    return ProbeResult(
        status="fail",
        source="none",
        cause="fail",
        error="ok_false_without_error",
        html_exists=has_html,
        json_exists=has_json,
        ss_exists=has_ss,
        collected_at=collected_at,
    )


def probe_source(root: str, batch_id: str, file_name: str, source_tag: str) -> ProbeResult:
    folder_path = os.path.join(root, batch_id)
    html_path = os.path.join(folder_path, f"{file_name}.html")
    json_path = os.path.join(folder_path, f"{file_name}.json")
    ss_path = os.path.join(folder_path, f"{file_name}.png")

    r = classify_from_files(html_path, json_path, ss_path)
    return ProbeResult(
        status=r.status,
        source=source_tag if r.status != "fail" or r.error != "missing_artifact" else "none",
        cause=r.cause,
        error=r.error,
        html_exists=r.html_exists,
        json_exists=r.json_exists,
        ss_exists=r.ss_exists,
        collected_at=r.collected_at,
    )


def choose_best(node_r: ProbeResult, web_r: ProbeResult) -> ProbeResult:
    # Öncelik: ok > captcha > soft_fail > fail
    pn = STATUS_PRIORITY.get(node_r.status, 0)
    pw = STATUS_PRIORITY.get(web_r.status, 0)

    if pn > pw:
        return node_r
    if pw > pn:
        return web_r

    # Eşitlikte: web_scraper sonradan toplandığı için yeni veri olarak tercih edelim.
    # (isterseniz node öncelikli yapabiliriz)
    if web_r.source == "web_scraper":
        return web_r
    return node_r


def print_batch_summary(batch_stats: Dict[str, Dict[str, int]]) -> None:
    print("-" * 90)
    print(f"{'Batch':<8} | {'Total':<7} | {'OK':<7} | {'CAPTCHA':<8} | {'SOFT_FAIL':<10} | {'FAIL':<7} | {'Status'}")
    print("-" * 90)

    g = {"total": 0, "ok": 0, "captcha": 0, "soft_fail": 0, "fail": 0}

    for bid in sorted(batch_stats.keys()):
        s = batch_stats[bid]
        for k in g:
            g[k] += s.get(k, 0)

        processed = s["ok"] + s["captcha"] + s["soft_fail"] + s["fail"]
        if processed == 0:
            st = "Not Started"
        elif processed == s["total"]:
            st = "Complete"
        else:
            st = "In Progress"

        print(
            f"{bid:<8} | {s['total']:<7} | {s['ok']:<7} | {s['captcha']:<8} | "
            f"{s['soft_fail']:<10} | {s['fail']:<7} | {st}"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<8} | {g['total']:<7} | {g['ok']:<7} | {g['captcha']:<8} | "
        f"{g['soft_fail']:<10} | {g['fail']:<7} |"
    )
    if g["total"] > 0:
        print(
            f"{'%':<8} | {'100%':<7} | {g['ok']/g['total']*100:>6.1f}% | "
            f"{g['captcha']/g['total']*100:>7.1f}% | {g['soft_fail']/g['total']*100:>9.1f}% | "
            f"{g['fail']/g['total']*100:>6.1f}% |"
        )
    print("-" * 90)


def print_domain_stats(fail_domains: Counter, captcha_domains: Counter, soft_fail_domains: Counter) -> None:
    print("\nTop 10 Domains by Status")
    print("=" * 90)

    print(f"{'FAIL Domains':<60} | {'Count':<8}")
    print("-" * 90)
    for d, c in fail_domains.most_common(10):
        print(f"{d:<60} | {c:<8}")

    print("-" * 90)
    print(f"{'SOFT_FAIL Domains':<60} | {'Count':<8}")
    print("-" * 90)
    for d, c in soft_fail_domains.most_common(10):
        print(f"{d:<60} | {c:<8}")

    print("-" * 90)
    print(f"{'CAPTCHA Domains':<60} | {'Count':<8}")
    print("-" * 90)
    for d, c in captcha_domains.most_common(10):
        print(f"{d:<60} | {c:<8}")
    print("-" * 90)


def audit_and_build(serp_csv_path: str, batch_size: int = 1000) -> None:
    if not os.path.exists(serp_csv_path):
        raise SystemExit(f"SERP CSV not found: {serp_csv_path}")

    print(f"Auditing with: {serp_csv_path}")
    print(f"Node root: {SCRAPING_ROOT}")
    print(f"Web root : {SCRAPING_ROOT_WEB}")

    df = pd.read_csv(serp_csv_path, low_memory=False)
    for col in ["file_name", "link", "section", "search_term"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    df["file_name"] = df["file_name"].astype(str).fillna("").str.strip()
    df["link"] = df["link"].astype(str).fillna("").str.strip()
    df["section"] = df["section"].astype(str).fillna("").str.strip()
    df["search_term"] = df["search_term"].astype(str).fillna("").str.strip()

    before = len(df)
    df = df[df["link"].str.startswith("http", na=False)].copy()
    df = df.drop_duplicates(subset=["file_name"], keep="first")
    print(f"Unique pages (by filename): {len(df)} (was {before})")

    rows_scraped: List[Dict[str, str]] = []
    rows_missed: List[Dict[str, str]] = []
    rows_cov: List[Dict[str, str]] = []

    batch_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "captcha": 0, "soft_fail": 0, "fail": 0}
    )
    fail_domains = Counter()
    captcha_domains = Counter()
    soft_fail_domains = Counter()

    for idx, row in enumerate(df.itertuples(index=False), start=0):
        batch_idx = idx // batch_size
        batch_id = f"{batch_idx:03d}"

        file_name = str(getattr(row, "file_name")).strip()
        link = str(getattr(row, "link")).strip()
        section = str(getattr(row, "section", "")).strip()
        search_term = str(getattr(row, "search_term", "")).strip()

        batch_stats[batch_id]["total"] += 1

        node_r = probe_source(SCRAPING_ROOT, batch_id, file_name, "node_scraper")
        web_r = probe_source(SCRAPING_ROOT_WEB, batch_id, file_name, "web_scraper")
        final_r = choose_best(node_r, web_r)

        # source coverage log
        rows_cov.append(
            {
                "link": link,
                "folder": batch_id,
                "file_name": file_name,
                "node_status": node_r.status,
                "web_status": web_r.status,
                "chosen_status": final_r.status,
                "chosen_source": final_r.source,
            }
        )

        batch_stats[batch_id][final_r.status] += 1

        domain = ""
        try:
            domain = urlparse(link).netloc.lower()
        except Exception:
            domain = ""

        if final_r.status == "captcha":
            if domain:
                captcha_domains[domain] += 1
        elif final_r.status == "soft_fail":
            if domain:
                soft_fail_domains[domain] += 1
        elif final_r.status == "fail":
            if domain:
                fail_domains[domain] += 1

        scraped_at = final_r.collected_at if final_r.collected_at else utc_now_iso()

        rows_scraped.append(
            {
                "link": link,
                "folder": batch_id,
                "file_name": file_name,
                "status": final_r.status,
                "source": final_r.source,  # node_scraper | web_scraper | none
                "cause": final_r.cause,    # "" | captcha | soft_fail | fail
                "error": final_r.error,
                "scraped_at": scraped_at,
            }
        )

        if final_r.status in ("captcha", "soft_fail", "fail"):
            rows_missed.append(
                {
                    "link": link,
                    "folder": batch_id,
                    "file_name": file_name,
                    "section": section,
                    "search_term": search_term,
                    "source": final_r.source,
                    "cause": final_r.cause,      # requested field
                    "error": final_r.error,
                }
            )

    # backups
    for path in [SCRAPED_CSV, MISSED_CSV, SOURCE_COVERAGE_CSV]:
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")

    # write scraped.csv
    os.makedirs(os.path.dirname(SCRAPED_CSV) or ".", exist_ok=True)
    with open(SCRAPED_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["link", "folder", "file_name", "status", "source", "cause", "error", "scraped_at"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_scraped)

    # write missed.csv
    os.makedirs(os.path.dirname(MISSED_CSV) or ".", exist_ok=True)
    with open(MISSED_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["link", "folder", "file_name", "section", "search_term", "source", "cause", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_missed)

    # write source coverage
    os.makedirs(os.path.dirname(SOURCE_COVERAGE_CSV) or ".", exist_ok=True)
    with open(SOURCE_COVERAGE_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["link", "folder", "file_name", "node_status", "web_status", "chosen_status", "chosen_source"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_cov)

    print("\nAudit complete.")
    print(f"- {SCRAPED_CSV} rows: {len(rows_scraped)}")
    print(f"- {MISSED_CSV} rows: {len(rows_missed)}")
    print(f"- {SOURCE_COVERAGE_CSV} rows: {len(rows_cov)}")

    print_batch_summary(batch_stats)
    print_domain_stats(fail_domains, captcha_domains, soft_fail_domains)

    # quick decision hints
    total = len(rows_scraped)
    missed = len(rows_missed)
    miss_rate = (missed / total * 100.0) if total else 0.0

    c_counts = Counter([r["cause"] for r in rows_missed])
    print("\nDecision Hints")
    print("=" * 40)
    print(f"Total pages      : {total}")
    print(f"Missed pages     : {missed} ({miss_rate:.2f}%)")
    print(f"captcha          : {c_counts.get('captcha', 0)}")
    print(f"soft_fail        : {c_counts.get('soft_fail', 0)}")
    print(f"fail             : {c_counts.get('fail', 0)}")

    # Simple recommendation
    # - captcha yüksekse retry verimi düşük olabilir (anti-bot)
    # - soft_fail yüksekse kurala dayalı fix/retry faydalı olabilir
    # - fail yüksekse artifact eksikliği/IO sorunu iyileştirmesi gerekir
    if miss_rate < 5:
        print("Recommendation   : Missing rate düşük. Hedef analiz için direkt devam edebilirsiniz.")
    elif c_counts.get("soft_fail", 0) >= c_counts.get("captcha", 0):
        print("Recommendation   : Önce soft_fail subset üzerinde sınırlı retry stratejisi deneyin.")
    else:
        print("Recommendation   : CAPTCHA ağırlıklı. Ek toplama maliyetini dikkatli değerlendirin.")


def summarize_scraped_csv(path: str = SCRAPED_CSV) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    batch_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "captcha": 0, "soft_fail": 0, "fail": 0}
    )
    fail_domains = Counter()
    captcha_domains = Counter()
    soft_fail_domains = Counter()

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            batch_id = str(row.get("folder", "")).strip()
            status = str(row.get("status", "")).strip().lower()
            link = str(row.get("link", "")).strip()

            if not batch_id:
                continue

            batch_stats[batch_id]["total"] += 1
            if status not in ("ok", "captcha", "soft_fail", "fail"):
                status = "fail"
            batch_stats[batch_id][status] += 1

            try:
                domain = urlparse(link).netloc.lower()
            except Exception:
                domain = ""

            if domain:
                if status == "captcha":
                    captcha_domains[domain] += 1
                elif status == "soft_fail":
                    soft_fail_domains[domain] += 1
                elif status == "fail":
                    fail_domains[domain] += 1

    print_batch_summary(batch_stats)
    print_domain_stats(fail_domains, captcha_domains, soft_fail_domains)



