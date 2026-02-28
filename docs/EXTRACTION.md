# Extraction Steps

This document details the feature extraction pipeline. All steps read `data/index.parquet` and typically process only `status == "ok"` rows.

---

## A: Runtime metrics

- **Script:** `src/extraction/A/runtime_metrics.py`
- **Output:** `data/features/A/runtime_metrics.parquet`

Features:
- `record_id`
- `ttfb_ms`
- `dom_content_loaded_ms`
- `load_time_ms`
- `lcp_ms`
- `cls`
- `step_status`, `step_reason`

## B: Accessibility metrics

- **Script:** `src/extraction/B/accessibility_metrics.py`
- **Output:** `data/features/B/accessibility_metrics.parquet`

Feature Engineering:
- `axe_score`: Calculated as `axe_passes_total / (axe_passes_total + axe_violations_total)`.
- `contrast_score`: Calculated as `contrast_pass_count / (contrast_pass_count + contrast_violation_count)`.
- Scores are assigned as NaN if the denominator is zero to avoid division errors.

Features:
- `record_id`
- contrast metrics
- axe summary metrics
- `axe_score`, `contrast_score`
- `step_status`, `step_reason`

## C: HTML structure & Content Extraction

- **Script:** `src/extraction/C/html_structure.py`
- **Output:** `data/features/C/html_structure.parquet`
- **Artifacts:** `extraction-root/[iteration]-[source]/[file_name].json` (Normalized content)

Content Extraction (Phase 1):
- Parses HTML using `BeautifulSoup`.
- Extracts: `title`, `description`, `canonical`, `h_tags` (h1-h6), `links`, `images`, `scripts`.
- Saves normalized JSON for downstream tasks (D, E).

Features (Phase 2 - Structure Metrics):
- `record_id`
- `h1_count`, `h2_count`, ...
- `internal_link_count`, `external_link_count`
- `image_count`, `missing_alt_count`
- `schema_org_types`
- `step_status`, `step_reason`

## D: Text & Readability

- **Script:** `src/extraction/D/text_readability.py`
- **Input:** Extracted `.json` files from Step C.
- **Output:** `data/features/D/text_readability.parquet`

Logic:
1. Reads content from JSON. Performance Optimization: Content is truncated to the first 100,000 characters for metric calculation.
2. **Quality Check:** Marks status=`soft_fail` and reason=`technical_noise_detected` if words > 100 chars (e.g. base64).
3. **Outlier Check:** Marks status=`soft_fail` and reason=`implausible_score` if scores are physically impossible (e.g. Flesch < -500 & Flesch > 200).
4. Computes metrics using `textstat`.

Features:
- `record_id`
- `char_count`, `word_count`, `sentence_count`
- `flesch_reading_ease`
- `flesch_kincaid_grade`
- `gunning_fog`
- `smog_index`
- `step_status`, `step_reason`

## E: Semantic Similarity

- **Script:** `src/extraction/E/semantic_similarity.py`
- **Input:** Extracted `.json` files from Step C.
- **Output:** `data/features/E/semantic_similarity.parquet`
- **Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`)

Process:
- Encodes `search_term` (Query).
- Encodes `title`, `description`, `h1`, `content` (Document fields). Truncated to 1,000 characters.
- Computes Cosine Similarity.
- **Note:** Content is truncated to 5000 chars before encoding to fit model limits (approx 256 tokens).

Features:
- `record_id`
- `sim_title`, `sim_description`, `sim_h1`, `sim_content` (Score 0.0 - 1.0)
- `step_status`, `step_reason`

## F: Vision features (Optional)

- **Script:** `src/extraction/F/vision_features.py`
- **Output:** `data/features/F/vision_features.parquet`
- Takes screenshots and computes visual metrics (e.g. clutter score).

---

## Z: Final Merge & Acceptance

- **Script:** `src/extraction/Z/merge.py`
- **Output:** `data/dataset-[TIMESTAMP].parquet`

### Merge strategy
- Sequential Join: Features from steps A, B, C, D, and E are joined via record_id.
- Status Renaming: Individual step_status columns are renamed to status_A, status_B, etc., for traceability.
- Global Status: If any individual step status is not "ok," the global record status is set to "fail".
- Outliers: Outlier flags from data/outliers.parquet are integrated, and flagged records are excluded by default.

### Acceptance Criteria

Filters defined in acceptance.json are applied to ensure data quality. Key criteria include:
- http_status must be 200.
- word_count must be at least 20.
- Critical metrics (e.g., sim_title, flesch_reading_ease, lcp_ms, axe_score) must not be null.
- The global status must be "ok".