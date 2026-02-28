# SERP Profiler Kit

This repository hosts the codebase for our SERP profiling and dataset-generation pipelines.

## Context and versioning

This project evolved into two parallel tracks:

### v1 — Published pipeline (frozen)
v1 corresponds to the code used for the published study on IEEE Xplore:
- Paper: [Disentangling Technical and Content Attributes in Search Engine Ranking: A Comparative Study of Google and Bing](https://ieeexplore.ieee.org/document/11363468)  
- Code: **Branch/Tag `v1`** (kept stable for reproducibility)

v1 remains available for:
- reproducing the published experiments exactly,
- validating the original dataset generation steps,
- comparing the new pipeline against the published baseline.

### v2 — New pipeline (active)
v2 is a re-designed pipeline written from scratch for the second paper.
It aims to:
- improve modularity and rerun ergonomics,
- strengthen artifact reconciliation,
- formalize feature extraction and dataset assembly,
- prepare for a future unified analysis layer (v1 + v2 reconciliation).

---

## SERP News Propagation Dataset Pipeline (v2)

A modular pipeline to collect SERP outputs, reconcile scraping artifacts, extract features, and generate a reproducible research dataset.

### Project structure

```text
src/
  collection/
    A/ .. Z/
  extraction/
    A/ B/ C/ D/ E/ F/ .. Z/
  generators/
    A/ .. Z/
  analysis/
    A/ .. Z/
data/
  raw/
  features/
    A/ B/ C/ D/ E/ F/ G/ Z/
  reports/
```

### Data flow

1. `data/keywords.csv` and `data/serp.csv` define the seed and SERP universe.
2. `src/collection/Z/build.py` reconciles scraping artifacts and creates `data/index.parquet`.
3. `src/extraction/*` scripts compute feature sets from status=ok rows.
4. `src/extraction/Z/merge.py` produces final `data/dataset.parquet`.
5. `src/analysis/*` uses `keywords.csv + dataset.parquet` for statistical analysis.

### Why this design

- Fast iteration under time constraints
- Clear modular boundaries
- Incremental reruns with iteration-source scraping folders
- Public-release friendly (derived features, no copyrighted full text by default)

### Quick start

See `docs/REPRODUCIBILITY.md` for full run commands.

#### Detailed Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Collection](docs/COLLECTION.md)
- [Extraction](docs/EXTRACTION.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [Analysis](docs/ANALYSIS.md)
- [Generators](docs/GENERATOR.md)

### Public release policy

Recommended to share:
- `keywords.csv`
- `dataset.parquet`
- documentation files

Recommended NOT to share by default:
- full copyrighted article texts
- raw HTML dumps
- screenshots (unless legal review approves)

### License

This project is licensed under the MIT License - see the LICENSE file for details.
