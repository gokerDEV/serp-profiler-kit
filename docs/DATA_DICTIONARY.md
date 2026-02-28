# DATA_DICTIONARY.md

## Overview

This document describes the public dataset schema and feature definitions, directly corresponding to `src/schema_v1.yml`.

**Latest Dataset Schema Version:** v1

---

## 1. Core Index (`data/index.parquet`)

The base dataset containing scraping metadata and SERP details.

| Column | Type | Description |
|:-------|:-----|:------------|
| `record_id` | str | Unique identifier for the record (hash of key fields). Join key for all features. |
| `search_engine` | str | Search engine name (google, brave, mojeek). |
| `search_term` | str | Query used for search. |
| `rank` | int | Position in SERP (1-20). |
| `url` | str | URL of the result. |
| `file_name` | str | Name of the scraped HTML artifact. |
| `folder` | str | Path to the artifact folder. |
| `source` | str | Source of scraping (bot, extension, web). |
| `iteration` | int | Iteration number of the scrape. |
| `collected_at` | str | Timestamp of collection (ISO 8601). |
| `html_size_bytes` | float | Size of the HTML file in bytes. |
| `status` | str | Scraping status (ok, captcha, fail, etc.). |
| `title` | str | Title from SERP or HTML. |
| `snippet` | str | Snippet from SERP. |
| `section` | str | News section/topic. |
| `http_status` | float | HTTP Status code (200, 403, 404). |

---

## 2. Extraction A: Runtime Metrics (`data/features/A/runtime_metrics.parquet`)

Performance metrics from the scraping runtime.

| Column | Type | Description |
|:-------|:-----|:------------|
| `ttfb_ms` | float | Time to First Byte (ms). |
| `dom_content_loaded_ms` | float | DOM Content Loaded time (ms). |
| `load_time_ms` | float | Full Page Load time (ms). |
| `lcp_ms` | float | Largest Contentful Paint (ms). |
| `cls` | float | Cumulative Layout Shift score. |
| `status_A` | str | Step status. |
| `reason_A` | str | Error details if failed. |

---

## 3. Extraction B: Accessibility (`data/features/B/accessibility_metrics.parquet`)

Accessibility auditing metrics (Axe-core and Contrast).

| Column | Type | Description |
|:-------|:-----|:------------|
| `contrast_violation_count` | float | Number of color contrast violations. |
| `contrast_pass_count` | float | Number of color contrast passes. |
| `min_contrast_ratio` | float | Minimum observed contrast ratio. |
| `axe_violations_total` | float | Total accessibility violations. |
| `axe_passes_total` | float | Total passed checks. |
| `axe_incomplete_total` | float | Total incomplete checks. |
| `axe_inapplicable_total` | float | Total inapplicable checks. |
| `axe_impact_critical` | float | Count of critical impact issues. |
| `axe_impact_serious` | float | Count of serious impact issues. |
| `axe_impact_moderate` | float | Count of moderate impact issues. |
| `axe_impact_minor` | float | Count of minor impact issues. |
| `status_B` | str | Step status. |
| `reason_B` | str | Error details if failed. |

---

## 4. Extraction C: HTML Structure (`data/features/C/html_structure.parquet`)

Structural analysis of the HTML document.

| Column | Type | Description |
|:-------|:-----|:------------|
| `tag_count` | float | Total number of HTML tags. |
| `link_count` | float | Number of `<a>` tags. |
| `image_count` | float | Number of `<img>` tags. |
| `script_count` | float | Number of `<script>` tags. |
| `h1_count` | float | Number of `<h1>` tags. |
| `h2_count` | float | Number of `<h2>` tags. |
| `h3_count` | float | Number of `<h3>` tags. |
| `has_schema` | bool | True if Schema.org JSON-LD detected. |
| `has_published_time` | bool | True if publication date detected in meta. |
| `has_author` | bool | True if author metadata detected. |
| `canonical` | str | Canonical URL from meta tags. |
| `domain` | str | Domain extracted from URL. |
| `extractor_used` | str | Method used for extraction. |
| `text_quality_flag` | str | Quality flag (e.g., ok, noisy, empty). |
| `status_C` | str | Step status. |
| `reason_C` | str | Error details if failed. |

---

## 5. Extraction D: Readability (`data/features/D/text_readability.parquet`)

Content readability metrics calculated on main text.

| Column | Type | Description |
|:-------|:-----|:------------|
| `char_count` | float | Total characters in main content. |
| `word_count` | float | Total words. |
| `sentence_count` | float | Total sentences. |
| `flesch_reading_ease` | float | Flesch Reading Ease score. |
| `flesch_kincaid_grade` | float | US Grade Level. |
| `gunning_fog` | float | Gunning Fog Index. |
| `smog_index` | float | SMOG Index. |
| `automated_readability_index` | float | Automated Readability Index (ARI). |
| `status_D` | str | Step status. |
| `reason_D` | str | Error details if failed. |

---

## 6. Extraction E: Semantic Similarity (`data/features/E/semantic_similarity.parquet`)

Cosine similarity scores between Search Term (Query) and Document Fields.

| Column | Type | Description |
|:-------|:-----|:------------|
| `sim_title` | float | Similarity: Query vs Title. |
| `sim_description` | float | Similarity: Query vs Meta Description. |
| `sim_h1` | float | Similarity: Query vs H1. |
| `sim_content` | float | Similarity: Query vs Main Content. |
| `model_name` | str | Name of the embedding model. |
| `status_E` | str | Step status. |
| `reason_E` | str | Error details if failed. |

---

## 7. Analysis: Outliers (`data/outliers.parquet`)

Flags indicating if a record is an statistical outlier based on defined analysis methodologies.

| Column | Type | Description |
|:-------|:-----|:------------|
| `is_outlier_technical` | str | `outlier` / `ok`. |
| `is_outlier_readability` | str | `outlier` / `ok`. |
| `is_outlier_similarity` | str | `outlier` / `ok`. |
| `is_outlier_accessibility`| str | `outlier` / `ok`. |

---

## 8. General Features

Additional features generated and used across the dataset ecosystem.

| Column | Type | Description |
|:-------|:-----|:------------|
| `is_source_domain` | bool | Flag indicating if the link belongs to the source domain (theguardian.com) |

---

## 9. Final Dataset (`data/dataset.parquet`)

The merged dataset containing all Index columns + All Feature columns + General fields + Outlier flags.
Duplicated URLs or queries depend on specific analysis scopes, but the primary view merges via the unique `record_id`.

## Variable Aliasing Mapping 

For backwards backward compatibility, `schema_v1.yml` permits aliases during mapping:
- `similarity_guardian_title_vs_result_title` -> `sim_title`
- `similarity_guardian_description_vs_result_description` -> `sim_description`
- `similarity_guardian_content_vs_result_content` -> `sim_content`
- `main_text_char_count` -> `char_count`
- `main_text_word_count` -> `word_count`
- `img_count` -> `image_count`
- `p_count` -> `tag_count`
- `has_jsonld` -> `has_schema`
- `link` -> `url`

## Missing Values
- Numeric indicators may be `null` or missing if the associated step failed (`status_X != ok`) or if the targeted artifact format was corrupt/missing.
- Outlier flags evaluate absent states predominantly as `ok`.
