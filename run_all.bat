@echo off
REM run_all.bat — quick reproducibility smoke-run for thesis pipeline (Windows)
REM Usage: run_all.bat [full]
REM If run without args, runs in "debug" mode (small sample, 1 epoch). Pass 'full' to run full (long) mode.

setlocal ENABLEDELAYEDEXPANSION
rem Initialize run log (absolute path next to script)
set "RUNALL_LOG=%~dp0run_all.log"
rem start fresh log
echo START RUN_ALL %DATE% %TIME% > "%RUNALL_LOG%"
echo =============================================
echo Thesis pipeline — smoke-run (debug)
echo =============================================

:: Check Python availability
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Python not found on PATH. Please install Python and ensure it's on PATH.
  exit /b 1
)

:: Option: full run if first argument is 'full'
set MODE=debug
if "%1"=="full" set MODE=full
echo Mode: %MODE%

:: Call environment setup if available (optional)
if exist setup_environment.bat (
  echo Running setup_environment.bat to ensure environment packages...
  call setup_environment.bat
) else (
  echo No setup_environment.bat found; make sure required packages in requirements.txt are installed.
)

:: STEP 1: Prepare DAPT corpus (local-only) — always generate canonical file first
echo [1/8] Preparing DAPT corpus (using local QA)...
python 01_prepare_dapt.py --input LazadaQA-Taglish-7k.csv --output dapt_corpus_clean.txt --use-local-only --seed 42
echo ... produced dapt_corpus_clean.txt (if file exists)
if not exist dapt_corpus_clean.txt (
  echo ERROR: dapt_corpus_clean.txt not found after data prep. Exiting.
  echo [FAIL] STEP1 Prepare DAPT >> "%RUNALL_LOG%"
  exit /b 1
)

echo [OK] STEP1 Prepare DAPT >> "%RUNALL_LOG%"

:: STEP 2: Create small debug sample unless running full mode
if "%MODE%"=="debug" (
  echo [2/8] Creating debug sample (first 200 lines) -> dapt_corpus_debug.txt
  powershell -Command "Get-Content -Path 'dapt_corpus_clean.txt' -TotalCount 200 | Set-Content -Path 'dapt_corpus_debug.txt' -Encoding UTF8"
  set TRAINFILE=dapt_corpus_debug.txt
) else (
  set TRAINFILE=dapt_corpus_clean.txt
)
echo Training file: %TRAINFILE%

:: STEP 3: Run DAPT (1 epoch in debug mode)
echo [3/8] Running DAPT (masked LM)
if "%MODE%"=="debug" (
  echo Running DAPT in debug mode: 1 epoch, small batch
  python 02_run_dapt.py --train-file %TRAINFILE% --epochs 1 --batch-size 4 --seed 42
) else (
  python 02_run_dapt.py --train-file %TRAINFILE% --seed 42
)
if %ERRORLEVEL% NEQ 0 (
  echo [FAIL] STEP3 DAPT (python) exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
  echo ERROR at DAPT: exit %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo [OK] STEP3 DAPT >> "%RUNALL_LOG%"
)

:: STEP 4: Generate masking example
echo [4/8] Generating masking example image
python 07_generate_masking_example.py --train-file %TRAINFILE% --checkpoint distilbert-base-multilingual-cased
if %ERRORLEVEL% NEQ 0 (
  echo [FAIL] STEP4 Masking example exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
  echo ERROR at masking example: exit %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo [OK] STEP4 Masking example >> "%RUNALL_LOG%"
)

:: STEP 5: Fine-tune models (1 epoch in debug mode)
echo [5/8] Fine-tuning models (this may download models from HF)
if "%MODE%"=="debug" (
  python 04_finetune_models.py --dataset ccosme/FiReCS --epochs 1 --batch-size 4 --seed 42
) else (
  python 04_finetune_models.py --dataset ccosme/FiReCS --seed 42
)
if %ERRORLEVEL% NEQ 0 (
  echo [FAIL] STEP5 Fine-tune exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
  echo ERROR at fine-tune: exit %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo [OK] STEP5 Fine-tune >> "%RUNALL_LOG%"
)

:: STEP 6: Conditionally run optimization if 'optimum' is available
python -c "import pkgutil,sys; sys.exit(0 if pkgutil.find_loader('optimum') else 1)"
if %ERRORLEVEL% EQU 0 (
  echo [6/8] 'optimum' detected — running model optimization (may be slow)
  python 05_optimize_model.py --source ./models/model_B_finetuned --out ./models/model_D_optimized
  if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] STEP6 Optimization exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
    echo ERROR at optimization: exit %ERRORLEVEL%
    exit /b %ERRORLEVEL%
  ) else (
    echo [OK] STEP6 Optimization >> "%RUNALL_LOG%"
  )
) else (
  echo [6/8] 'optimum' not found — skipping optimization. To enable, install: pip install optimum onnxruntime onnxruntime-tools
  echo [SKIP] STEP6 Optimization >> "%RUNALL_LOG%"
)

:: STEP 7: Benchmark available models
echo [7/8] Running benchmark
python 06_final_benchmark.py --dataset ccosme/FiReCS --seed 42
if %ERRORLEVEL% NEQ 0 (
  echo [FAIL] STEP7 Benchmark exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
  echo ERROR at benchmark: exit %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo [OK] STEP7 Benchmark >> "%RUNALL_LOG%"
)

:: STEP 8: Visualize pipeline / collect artifacts
echo [8/8] Creating pipeline visualization and summary images
python 07_visualize_pipeline.py
if %ERRORLEVEL% NEQ 0 (
  echo [FAIL] STEP8 Visualization exit=%ERRORLEVEL% >> "%RUNALL_LOG%"
  echo ERROR at visualization: exit %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo [OK] STEP8 Visualization >> "%RUNALL_LOG%"
)

echo =============================================
echo Smoke-run complete. Check console output and artifacts in 'models/' and generated images (pipeline_summary.png, thesis_benchmarks.png)
echo =============================================

echo END RUN_ALL %DATE% %TIME% >> "%RUNALL_LOG%"
endlocal
exit /b 0


















































