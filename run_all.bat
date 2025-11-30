@echo off
REM End-to-end pipeline for Prompted Segmentation for Drywall QA
REM Windows batch script

echo ============================================================
echo Prompted Segmentation for Drywall QA - Complete Pipeline
echo ============================================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo ============================================================
echo Step 1: Installing dependencies
echo ============================================================
pip install -r requirements.txt

REM Download datasets
echo.
echo ============================================================
echo Step 2: Downloading datasets from Roboflow
echo ============================================================
echo Please ensure ROBOFLOW_API_KEY is set in .env file
python data\download_datasets.py
if %errorlevel% neq 0 (
    echo ERROR: Dataset download failed!
    echo Please check your ROBOFLOW_API_KEY in .env file
    pause
    exit /b 1
)

REM Prepare data
echo.
echo ============================================================
echo Step 3: Preparing data (converting annotations, creating splits)
echo ============================================================
python data\prepare_data.py
if %errorlevel% neq 0 (
    echo ERROR: Data preparation failed!
    pause
    exit /b 1
)

REM Train model
echo.
echo ============================================================
echo Step 4: Training model
echo ============================================================
echo This may take 2-4 hours depending on your GPU...
python train.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

REM Evaluate model
echo.
echo ============================================================
echo Step 5: Evaluating model on test set
echo ============================================================
python evaluate.py
if %errorlevel% neq 0 (
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

REM Generate report
echo.
echo ============================================================
echo Step 6: Generating project report
echo ============================================================
python generate_report.py
if %errorlevel% neq 0 (
    echo ERROR: Report generation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Pipeline completed successfully!
echo ============================================================
echo.
echo Results:
echo   - Model checkpoint: checkpoints\best_model.pth
echo   - Predictions: predictions\
echo   - Report: REPORT.md
echo   - Training logs: logs\
echo.
echo To view training curves:
echo   tensorboard --logdir logs\tensorboard
echo.
pause
