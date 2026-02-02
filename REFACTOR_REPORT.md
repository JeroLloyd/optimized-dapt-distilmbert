# Refactor Report — Thesis NLP Codebase

## Overview
This report documents the refactor performed to make the codebase suitable for thesis artifact evaluation: improved readability, modularity, reproducibility, and research-safety without changing research logic or results.

---

## Step 1: Code Structure Audit ✅
- File organization and module boundaries
  - Main pipeline scripts retained as top-level scripts:
    - `01_prepare_dapt.py` (data preparation / cleaning)
    - `02_run_dapt.py` (domain-adaptive pre-training)
    - `04_finetune_models.py` (fine-tuning phases)
    - `05_optimize_model.py` (quantization / optimization)
    - `06_final_benchmark.py` (benchmarking & plotting)
    - `07_generate_masking_example.py`, `07_visualize_pipeline.py` (visual artifacts)
  - Added `nlp_thesis/utils.py` with small helper functions (seed setting, logging, IO helpers). This centralizes reproducibility and logging.

- Function responsibility and SRP
  - Each script now has a `main()` and small clearly-scoped helper functions.
  - `prepare_corpus`, `generate_masking_figure`, `run_training_phase`, `run_optimization`, `benchmark` each perform a single, testable task.

- Naming consistency
  - Constants use `DEFAULT_` prefix where appropriate and are in uppercase.
  - Function and variable names are descriptive and consistent across modules.

- Dead code / unused imports
  - Removed duplicated helper definitions and unused prints.
  - Replaced specific tokenizer classes with `AutoTokenizer` where appropriate for robustness.

- Hard-coded values
  - Replaced many hard-coded values with CLI arguments and `DEFAULT_` constants (seed, batch size, epochs, checkpoint paths, etc.).

---

## Step 2: NLP & ML Best-Practice Check ✅
- Separation of concerns
  - Preprocessing (`01_prepare_dapt.py`), model pretraining (`02_run_dapt.py`), fine-tuning (`04_finetune_models.py`), optimization (`05_optimize_model.py`), evaluation (`06_final_benchmark.py`) are separated.

- Deterministic behavior
  - Centralized `set_seed(seed, deterministic=False)` in `nlp_thesis/utils.py` and each script accepts `--seed` CLI argument.

- Tokenizer handling & padding/truncation
  - Tokenization is explicit about `padding='max_length'`, `truncation=True` and `max_length` (where applicable).

- Train/validation/test splitting
  - The dataset is concatenated across available splits then re-split with fixed seed and explicit sizes (train/test 80/20, then half-split test into val/test). This reduces leakage risk from existing split idiosyncrasies.

- Data leakage checks
  - DAPT corpus uses only unlabeled text; fine-tuning uses labeled dataset with separate splits. Both processes are file-separated.
  - Datasets & Methodology: DAPT uses the LazadaQA corpus (unlabeled) to adapt the DistilBERT encoder to Taglish/Lazada text, while supervised fine-tuning and evaluation are performed on the labeled **FiReCS** dataset (default `ccosme/FiReCS`). No synthetic data generation or augmentation is performed.

---

## Step 3: Reproducibility & Experiment Control ✅
- Random seed control added and used at script entry-points.
- Config management
  - CLI options for critical hyperparameters: batch size, epochs, learning rate, mlm probability, file paths, etc.
- Logging
  - Replaced ad-hoc print statements with structured logging via `nlp_thesis.utils.get_logger`.
- Explicit experiment params
  - Training arguments now accept and expose the main hyperparameters (epochs, lr, batch size)

---

## Step 4: Code Cleaning & Refactoring ✅
- Modularity
  - Small `nlp_thesis/utils.py` added for common functionality (seeding, logging, safe IO).
- Readability
  - Added concise docstrings and comments for non-obvious behaviors (masking figure, grouping logic).
- Removed duplication
  - Fixed duplicate `clean_memory` and ambiguous variable usage across scripts.

---

## Step 5: Scientific Safety Check ✅
- Silent bugs found and fixed
  - Indentation error in `02_run_dapt.py` where training block was accidentally unindented.
  - Token reading for masking example used `TRAIN_FILE` constant even with CLI options; replaced to use CLI variable.
  - Duplicate function definitions in `04_finetune_models.py` removed.

- Metric & evaluation checks
  - Metrics computed with explicit `macro` averaging for multi-class F1 (consistent with manuscript claims).
  - Warm-up inference done before timing in benchmarking.

- Flagged assumptions
  - The evaluation uses concatenated splits and random splitting; documented in report. Examiner should verify that the provided seeds and dataset splits match those described in the manuscript.

---

## Files Changed (major)
- `nlp_thesis/utils.py` (new)
- `01_prepare_dapt.py` (added CLI, logging, seed control)
- `02_run_dapt.py` (added CLI, logging, seed control, fixed GPU warnings, deterministic flags)
- `04_finetune_models.py` (added CLI, logging, fixed tokenization and trainer tokenizer argument; improved memory cleanup)
- `05_optimize_model.py` (added CLI and logging)
- `06_final_benchmark.py` (added CLI, logging, seed control)
- `07_generate_masking_example.py` (small improvements: logging, CLI)
- `07_visualize_pipeline.py` (replaced prints with logging where safe)

---

## Summary of Changes (table)

Readability
- Centralized small helpers in `nlp_thesis/utils.py` ✅
- Replaced prints with structured logging ✅
- Standardized variable naming & function signatures ✅

Reproducibility
- Added `--seed` CLI flags across entry scripts; centralized `set_seed` ✅
- Configurable hyperparameters via CLI and `DEFAULT_` constants ✅
- Saved trainer logs, metrics and history to reproducible files ✅

NLP correctness
- Tokenization explicit (padding/truncation/max_length) ✅
- MLM masking example uses DataCollator for consistent masking ✅
- Training & validation splits computed deterministically ✅

Research risk reduction
- Avoided silent behaviour (warnings and errors now logged) ✅
- Documented assumptions (split procedure, DAPT content) ✅
- Prevented accidental GPU-only requirement (scripts can run on CPU with warning) ✅

---

## Defense Notes (short)
- Why acceptable for a thesis
  - The structure mirrors the manuscript: data preparation → DAPT → fine-tuning → optimization → evaluation. Each step is reproducible with explicit seeds and saved artifacts (model files, logs, plots).

- What an examiner might question
  - Splitting strategy: we concatenate available splits then re-split to ensure consistent train/validation/test sizes; be prepared to justify why re-splitting was used instead of relying on original splits.
  - DAPT corpus provenance: `USE_LOCAL_ONLY` default uses local LazadaQA; if external HF data were used, provenance should be documented (dataset versions) in the thesis artifacts.
  - Quantization specifics: quantization config chosen (avx512_vnni / dynamic) is hardware-dependent; report performance targets and baseline numbers clearly.

---

## Quick Repro Instructions (for examiners)
- Create environment:
  - Windows: run `setup_environment.bat`
- Prepare DAPT corpus (default):
  - python 01_prepare_dapt.py --input LazadaQA-Taglish-7k.csv --output dapt_corpus_clean.txt
- Run DAPT (GPU recommended):
  - python 02_run_dapt.py --train-file dapt_corpus_clean.txt --checkpoint distilbert-base-multilingual-cased
- Fine-tune models:
  - python 04_finetune_models.py --dataset ccosme/FiReCS --seed 42 --batch-size 16 --epochs 4
- Optimize Model B:
  - python 05_optimize_model.py --source ./models/model_B_finetuned --out ./models/model_D_optimized
- Benchmark:
  - python 06_final_benchmark.py --dataset ccosme/FiReCS

---

Added artifacts
- `run_all.bat` — a **debug-mode** smoke-run that performs a short end-to-end check: samples 200 lines from `dapt_corpus_clean.txt`, runs DAPT for 1 epoch (small batch), fine-tunes for 1 epoch, skips optimization if `optimum` is not installed, benchmarks any available models, and generates pipeline visuals. Use `run_all.bat full` to run the full (non-debug) pipeline.

If you'd like, I can now:
1) Run a quick static check or tests (flake/pyright) in the workspace, or
2) Add smoke tests (unit tests) that assert tokenization and single-batch forward passes for models used in the thesis.

