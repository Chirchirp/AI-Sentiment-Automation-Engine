@echo off
echo ==================================================
echo AI Sentiment Automation Engine - Installation
echo ==================================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

python --version
echo OK: Python found
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo OK: Dependencies installed
echo.

echo ==================================================
echo Installation Complete!
echo ==================================================
echo.
echo Quick Start:
echo   1. Run: streamlit run streamlit_app.py
echo   2. Upload: sample_training_data.csv
echo   3. Train your first model
echo   4. Try auto-labelling with sample_unlabelled_data.csv
echo.
echo Documentation:
echo   - QUICKSTART.md - Get started in 5 minutes
echo   - README.md - Complete guide
echo   - USAGE_GUIDE.md - Detailed examples
echo.
echo Optional CLI usage:
echo   python sentiment_cli.py train sample_training_data.csv ^
echo     --text-col text --label-col sentiment
echo.
echo ==================================================
pause
