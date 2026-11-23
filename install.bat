@echo off
echo ============================================================
echo Violence Detection Project - Installation Script
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
echo This will download ~570 MB. Please be patient.
echo.

pip install tensorflow opencv-python tqdm jupyter ipykernel scikit-learn matplotlib seaborn fastapi uvicorn python-multipart pillow

echo.
echo [2/3] Creating required directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
echo Created: models/, logs/, temp/

echo.
echo [3/3] Verifying installation...
python check_setup.py

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Run: jupyter notebook
echo   2. Open: 01_data_preprocessing_optical_flow.ipynb
echo   3. Click: Cell -^> Run All
echo.
echo For detailed instructions, see START_TRAINING.md
echo ============================================================
pause
