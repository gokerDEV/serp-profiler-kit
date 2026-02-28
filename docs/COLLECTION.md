# Collection Steps

This document details the data collection agents and scripts responsible for querying APIs, parsing search engine result pages, downloading web content, and indexing the final curated raw dataset.

---

## Directory Structure

```text
src/
  collection/
    A/
      collect_news.py
    B/
      collect_serp.py
      helpers/
    C/
      collect_sources.py
      helpers/
    Z/
      build.py
```

## Steps

### A Step: Search News (Guardian API)

- **Script:** `src/collection/A/collect_news.py`
- **Output:** `data/keywords.csv`

This step retrieves news seeds directly from the Guardian API and prepares the keyword-level input used downstream. 
It accepts configurable time windows and section IDs to query the API.
To keep the dataset shareable, we store query-facing metadata (for example title-level fields and URLs) rather than full copyrighted article text.

### B Step: Search SERP (Brave, Mojeek, Google)

- **Script:** `src/collection/B/collect_serp.py`
- **Input:** `data/keywords.csv`
- **Output:** Raw JSONs per search term, and ultimately a merged `data/serp.csv`

Runs search queries collected in Step A across supported search engines. 
You can run a specific engine via the `--engine` flag (e.g. `google`, `brave`, `mojeek`).
The script outputs JSON files which are subsequently combined into a single, unified `serp.csv` file using the `--merge` flag.

Recommended core fields in `serp.csv`:
- `search_engine`
- `query_id`
- `section`
- `search_term`
- `rank`
- `link`
- `file_name`
- `title`
- `snippet`

### C Step: Search Sources (Scraping)

- **Script:** `src/collection/C/collect_sources.py`
- **Input:** `data/serp.csv`
- **Output root:** `data/raw/scraping/<iteration-source>/<batch>/`

This step visits the URLs obtained in Step B in an automated manner to download target pages. It can parallelize operations and skips already downloaded records. Use the `--retry-failed` flag to only retry records that failed previously.

Iteration/source folder examples:
- `scraping/01-bot/000/...`
- `scraping/02-bot/000/...`
- `scraping/03-extension/042/...`

Each record is expected to produce three artifacts with the exact same base filename (aligned with `serp.csv`'s `file_name`):
- `<file_name>.html`
- `<file_name>.json` (containing metadata like http status, raw errors, viewport)
- `<file_name>.png` (optional visual screenshot)

### Z Step: Build Core Index

- **Script:** `src/collection/Z/build.py`
- **Inputs:** `data/serp.csv` and `data/raw/scraping/*`
- **Output:** `data/index.parquet`

Z reconciles the row definitions from `serp.csv` with the available physical scraping artifacts inside the scraping directory, creating and updating the canonical core index. This index forms the bedrock for all downstream Extraction and Analysis steps.

#### Status taxonomy (normalized)

The script infers the quality of each collected page via the following taxonomy:
- `ok`
- `captcha`
- `soft_fail`
- `fail`
- `missing_artifact`

#### Status rules

- **Missing Artifact:** If the artifact set is fundamentally incomplete (html/json missing), sets `status = missing_artifact`.
- **Invalid JSON:** If JSON cannot be parsed mechanically, sets `status = fail` and `status_reason = invalid_json`.
- **CAPTCHA Heuristic:** Size-based heuristic. If `html_size_bytes < threshold` (configurable, default 50KB), then `status = captcha`, `status_reason = html_lt_50kb`. Marker-based CAPTCHA detection is intentionally disabled to avoid false positives.
- **Failures:** If the raw file indicates `raw_ok == false`:
  - With a descriptive `raw_error` string: `status = soft_fail`
  - Without a descriptive `raw_error` string: `status = fail`
- **Success:** Otherwise sets `status = ok`.

#### Iteration selection/update policy

- Processes folders sequentially handling paths like `01-bot`, `02-bot`.
- Existing `status=ok` records must not be overwritten by weaker, subsequent outcomes.
- A new candidate replaces the current record only if the qualitative status improves.
- Practical precedence: `ok > soft_fail > captcha > fail > missing_artifact`
- This seamlessly supports incremental backfill runs across multiple scraping passes and platforms (bots, extensions).

#### Core Index Columns

**Identity / SERP Core**
- `record_id` (deterministic unique hash key)
- `query_id`
- `search_engine`
- `section`
- `search_term`
- `rank`
- `link`
- `url_normalized`
- `domain`
- `file_name`
- `title`
- `snippet`

**Scraping Location Metadata**
- `folder` (for example `scraping/03-extension/042`)
- `source` (e.g., `bot` or `extension`)
- `iteration` (integer level parsed from path)

**Raw Scrape Signal Data**
- `raw_ok` (boolean, usually parsed from metadata JSON)
- `raw_error` (string, error details if any)
- `http_status` (integer, e.g. 200, 403, 404)
- `collected_at` (ISO-8601 timestamp string)
- `html_size_bytes` (integer)

**Normalized Analytics Outcome**
- `status`
- `status_reason`
