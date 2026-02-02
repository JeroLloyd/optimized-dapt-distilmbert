# optimized-dapt-distilmbert

This repository contains code used for a thesis experiment: domain-adaptive pretraining (DAPT) using DistilBERT and downstream fine-tuning and optimization.

Contents:
- Scripts for data prep, DAPT, fine-tuning, optimization, and benchmarking
- `nlp_thesis/utils.py`: lightweight reproducibility & logging helpers
- `run_all.bat`: quick smoke-run (debug mode by default)
- `REFACTOR_REPORT.md`: audit, refactor details, and defense notes for examiners

Quick start (Windows):
1. Create a venv and activate it: `py -3.10 -m venv venv` & `.\venv\Scripts\Activate.ps1`
2. Install requirements: `pip install -r requirements.txt`
3. Prepare DAPT corpus (UNLABELED Lazada data): `python 01_prepare_dapt.py --input LazadaQA-Taglish-7k.csv --output dapt_corpus_clean.txt`
4. Fine-tune (SUPERVISED) models on a labeled dataset (default: `ccosme/FiReCS`): `python 04_finetune_models.py --dataset ccosme/FiReCS`
4. Run the smoke-run (debug): `.\run_all.bat`

Datasets & Methodology:
- DAPT (domain-adaptive pretraining) uses the Lazada chat/QA corpus (`LazadaQA-Taglish-7k.csv`) as an UNLABELED source to adapt DistilBERT to Taglish/Lazadian text.
- Supervised fine-tuning and evaluation use the labeled dataset **FiReCS** (default `ccosme/FiReCS`).
- No synthetic data generation or augmentation is performed in the pipeline.

Notes on artifacts:
- Large model files, dataset CSVs, and virtual environment files are excluded using `.gitignore`.
- For artifact evaluation, provide model weights and dataset provenance separately (e.g., a release with download links or using Git LFS).

License: MIT (see `LICENSE`).
