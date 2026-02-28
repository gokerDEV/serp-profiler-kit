import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import requests


BASE_URL = "https://content.guardianapis.com/search"


@dataclass(frozen=True)
class MonthWindow:
    start: date
    end: date  # inclusive


@dataclass(frozen=True)
class GuardianArticle:
    article_id: str
    web_title: str
    web_url: str
    publication_date: str
    section_id: str
    pillar_name: Optional[str]
    wordcount: Optional[int]
    body_text: str


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = last_day_of_month(date(y, m, 1))
    return date(y, m, min(d.day, last_day.day))


def last_day_of_month(d: date) -> date:
    if d.month == 12:
        next_month = date(d.year + 1, 1, 1)
    else:
        next_month = date(d.year, d.month + 1, 1)
    return next_month - timedelta(days=1)


def last_full_n_months_windows(end_day: date, n_months: int) -> List[MonthWindow]:
    windows: List[MonthWindow] = []
    end_month_start = month_start(end_day)
    for i in range(n_months):
        start = add_months(end_month_start, -i)
        end = last_day_of_month(start)
        if start.year == end_day.year and start.month == end_day.month:
            end = end_day
        windows.append(MonthWindow(start=start, end=end))
    windows.reverse()
    return windows


def build_params(
    api_key: str,
    section: str,
    window: MonthWindow,
    page: int,
    page_size: int,
) -> Dict[str, str]:
    return {
        "api-key": api_key,
        "format": "json",
        "section": section,
        "type": "article",
        "from-date": window.start.isoformat(),
        "to-date": window.end.isoformat(),
        "order-by": "newest",
        "page": str(page),
        "page-size": str(page_size),
        "show-fields": "body,bodyText,headline,trailText,wordcount",
        "show-tags": "none",
    }


def parse_article(obj: Dict[str, object]) -> Optional[GuardianArticle]:
    try:
        article_id = str(obj.get("id", ""))
        web_title = str(obj.get("webTitle", ""))
        web_url = str(obj.get("webUrl", ""))
        publication_date = str(obj.get("webPublicationDate", ""))
        section_id = str(obj.get("sectionId", ""))
        pillar_name_raw = obj.get("pillarName", None)
        pillar_name = str(pillar_name_raw) if pillar_name_raw is not None else None

        fields = obj.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}

        body_text = ""
        if "bodyText" in fields and isinstance(fields["bodyText"], str):
            body_text = fields["bodyText"]
        elif "body" in fields and isinstance(fields["body"], str):
            body_text = fields["body"]
        else:
            body_text = ""

        wordcount_val: Optional[int] = None
        wc = fields.get("wordcount", None)
        if isinstance(wc, int):
            wordcount_val = wc
        elif isinstance(wc, str) and wc.isdigit():
            wordcount_val = int(wc)

        if not article_id or not web_title or not publication_date:
            return None

        return GuardianArticle(
            article_id=article_id,
            web_title=web_title,
            web_url=web_url,
            publication_date=publication_date,
            section_id=section_id,
            pillar_name=pillar_name,
            wordcount=wordcount_val,
            body_text=body_text,
        )
    except Exception:
        return None


def fetch_search_page(
    session: requests.Session,
    params: Dict[str, str],
    timeout_s: int,
) -> Dict[str, object]:
    r = session.get(BASE_URL, params=params, timeout=timeout_s)
    if r.status_code == 400:
        raise requests.HTTPError(r.text, response=r)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected JSON shape (root not dict).")
    return data


def iter_guardian_articles_for_window(
    api_key: str,
    section: str,
    window: MonthWindow,
    limit: int,
    page_size: int,
    min_wordcount: int,
    sleep_s: float,
    timeout_s: int,
) -> Iterable[GuardianArticle]:
    session = requests.Session()
    collected = 0
    page = 1
    total_pages: Optional[int] = None

    while collected < limit:
        if total_pages is not None and page > total_pages:
            break

        params = build_params(api_key, section, window, page=page, page_size=page_size)

        try:
            data = fetch_search_page(session, params=params, timeout_s=timeout_s)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 400:
                body = ""
                try:
                    body = e.response.text if e.response is not None else ""
                except Exception:
                    body = ""
                print(
                    f"Guardian API 400 (likely page out of range). "
                    f"section={section} window={window.start}..{window.end} page={page} "
                    f"collected={collected}/{limit}. Response={body[:300]}"
                )
                break
            raise

        response = data.get("response", {})
        if not isinstance(response, dict):
            break

        pages_val = response.get("pages", None)
        if isinstance(pages_val, int):
            total_pages = pages_val
        elif isinstance(pages_val, str) and pages_val.isdigit():
            total_pages = int(pages_val)

        results = response.get("results", [])
        if not isinstance(results, list) or len(results) == 0:
            break

        for item in results:
            if not isinstance(item, dict):
                continue

            art = parse_article(item)
            if art is None:
                continue

            wc_ok = (art.wordcount is not None and art.wordcount >= min_wordcount)
            text_ok = (art.wordcount is None and len(art.body_text.strip()) > 2000)
            if not (wc_ok or text_ok):
                continue

            yield art
            collected += 1
            if collected >= limit:
                break

        page += 1
        time.sleep(sleep_s)

    if collected < limit:
        print(
            f"Warning: collected {collected}/{limit} items for "
            f"section={section} window={window.start}..{window.end}. "
            f"Try lowering --per-month or lowering --min-wordcount."
        )


def sanitize_filename(value: str, max_len: int = 120) -> str:
    # Keep ASCII-ish safe set; replace others with underscore.
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9\-_.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "untitled"
    return s[:max_len]


def article_storage_path(
    base_dir: str,
    section: str,
    window: MonthWindow,
    article_id: str,
    title: str,
    ext: str,
) -> str:
    # Partition by section and month for easier debugging.
    month_key = f"{window.start.year:04d}-{window.start.month:02d}"
    safe_section = sanitize_filename(section)
    safe_title = sanitize_filename(title)
    safe_id = sanitize_filename(article_id.replace("/", "__"), max_len=80)
    dir_path = os.path.join(base_dir, safe_section, month_key)
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{safe_id}__{safe_title}.{ext}"
    return os.path.join(dir_path, filename)


def save_article_text(
    *,
    text_dir: str,
    text_format: str,
    overwrite: bool,
    section: str,
    window: MonthWindow,
    art: GuardianArticle,
) -> Optional[str]:
    """
    Saves article text for debugging. Returns filepath if written, else None.
    """
    if not text_dir:
        return None

    ext = "txt" if text_format == "txt" else "json"
    path = article_storage_path(
        base_dir=text_dir,
        section=section,
        window=window,
        article_id=art.article_id,
        title=art.web_title,
        ext=ext,
    )

    if (not overwrite) and os.path.exists(path):
        return None

    if text_format == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(art.web_title.strip() + "\n")
            f.write(art.web_url.strip() + "\n")
            f.write(art.publication_date.strip() + "\n\n")
            f.write(art.body_text)
        return path

    payload: Dict[str, object] = {
        "guardian_id": art.article_id,
        "title": art.web_title,
        "url": art.web_url,
        "publication_date": art.publication_date,
        "section": art.section_id,
        "pillar": art.pillar_name,
        "wordcount": art.wordcount,
        "body_text": art.body_text,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def process_article_on_the_fly(
    art: GuardianArticle,
    section: str,
    window: MonthWindow,
    save_text: bool,
    text_dir: str,
    text_format: str,
    overwrite_text: bool,
) -> Dict[str, object]:
    saved_path: str = ""
    if save_text:
        written = save_article_text(
            text_dir=text_dir,
            text_format=text_format,
            overwrite=overwrite_text,
            section=section,
            window=window,
            art=art,
        )
        if written is not None:
            saved_path = written

    return {
        "guardian_id": art.article_id,
        "section": art.section_id,
        "pillar": art.pillar_name if art.pillar_name is not None else "",
        "pub_date": art.publication_date,
        "title": art.web_title,
        "url": art.web_url,
        "wordcount": art.wordcount if art.wordcount is not None else "",
        "text_saved_path": saved_path,
    }


def write_rows_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", type=str, default=os.environ.get("GUARDIAN_API_KEY", ""))
    p.add_argument("--out", type=str, default="guardian_articles_index.csv")
    p.add_argument("--months", type=int, default=4)
    p.add_argument("--per-month", type=int, default=100)
    p.add_argument("--min-wordcount", type=int, default=300)
    p.add_argument("--page-size", type=int, default=200)
    p.add_argument("--sleep", type=float, default=1.05)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--end-date", type=str, default="")  # YYYY-MM-DD optional

    # New: optional text saving for debugging
    p.add_argument("--save-text", type=str, default="false", choices=["true", "false"])
    p.add_argument("--text-dir", type=str, default="guardian_texts")
    p.add_argument("--text-format", type=str, default="txt", choices=["txt", "json"])
    p.add_argument("--overwrite-text", type=str, default="false", choices=["true", "false"])

    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Provide --api-key or set GUARDIAN_API_KEY env var.")

    sections = ["world", "business", "technology", "environment", "science", "politics"]

    if args.end_date:
        end_day = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_day = date.today()

    windows = last_full_n_months_windows(end_day=end_day, n_months=int(args.months))

    save_text_flag = args.save_text == "true"
    overwrite_text_flag = args.overwrite_text == "true"

    for section in sections:
        for w in windows:
            batch: List[Dict[str, object]] = []

            for art in iter_guardian_articles_for_window(
                api_key=args.api_key,
                section=section,
                window=w,
                limit=int(args.per_month),
                page_size=int(args.page_size),
                min_wordcount=int(args.min_wordcount),
                sleep_s=float(args.sleep),
                timeout_s=int(args.timeout),
            ):
                row = process_article_on_the_fly(
                    art=art,
                    section=section,
                    window=w,
                    save_text=save_text_flag,
                    text_dir=args.text_dir,
                    text_format=args.text_format,
                    overwrite_text=overwrite_text_flag,
                )
                batch.append(row)

                # Reduce lifetime of large strings
                _ = art.body_text

                if len(batch) >= 50:
                    write_rows_csv(args.out, batch)
                    batch = []

            if batch:
                write_rows_csv(args.out, batch)

            print(
                f"Done section={section} window={w.start.isoformat()}..{w.end.isoformat()} "
                f"limit={args.per_month} save_text={save_text_flag}"
            )


if __name__ == "__main__":
    main()
