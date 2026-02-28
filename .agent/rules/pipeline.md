---
trigger: always_on
---

# Pipeline Rules

1. **Single Entry Point**: Each module (e.g., `src/*/[A..Z]`) must have exactly one main script in its root directory (e.g., `src/analysis/B/outlier_analysis.py`).
2. **Helpers Directory**: All helper modules, utilities, and secondary scripts must be placed in a `helpers/` subdirectory (not `helper/`).
3. **Helper Scripts**: Additional investigation or diagnostic scripts (e.g., `investigate_readability.py`) must be located in `helpers/` and should be callable from the main analysis script (e.g., via command-line arguments or function calls), rather than being standalone root scripts.