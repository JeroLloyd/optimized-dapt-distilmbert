@echo off
:: Thesis Environment Setup Script
:: Targeted for RTX 4060 & Python 3.10

echo ==================================================
echo  NLP THESIS ENVIRONMENT SETUP
echo ==================================================

:: 1. Check Python Version
python --version
echo [INFO] Ensure the above says Python 3.10.x. 
echo If it says 3.13 or 3.14, STOP and install Python 3.10.
pause

:: 2. Upgrade PIP
echo.
echo [STEP 1/4] Upgrading PIP...
python -m pip install --upgrade pip

:: 3. Install PyTorch with CUDA 12.4 (Stable for RTX 4060)
echo.
echo [STEP 2/4] Installing PyTorch (CUDA 12.4)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: 4. Install NLP Dependencies
echo.
echo [STEP 3/4] Installing Transformers, Datasets, Scikit-learn...
python -m pip install transformers datasets accelerate scikit-learn pandas optimum onnx onnxruntime-gpu

:: 5. Verification
echo.
echo [STEP 4/4] Verifying GPU Visibility...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

echo.
echo ==================================================
echo SETUP COMPLETE.
echo If "CUDA Available" is True, you are ready to run 02_run_dapt.py.
echo ==================================================
pause 