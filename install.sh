#!/bin/bash

echo "=================================================="
echo "AI Sentiment Automation Engine - Installation"
echo "=================================================="
echo ""

# Check Python
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.8+"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD --version
echo "✅ Python found"
echo ""

# Install dependencies
echo "Installing dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Installation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  1. Run: streamlit run streamlit_app.py"
echo "  2. Upload: sample_training_data.csv"
echo "  3. Train your first model"
echo "  4. Try auto-labelling with sample_unlabelled_data.csv"
echo ""
echo "Documentation:"
echo "  - QUICKSTART.md - Get started in 5 minutes"
echo "  - README.md - Complete guide"
echo "  - USAGE_GUIDE.md - Detailed examples"
echo ""
echo "Optional CLI usage:"
echo "  python sentiment_cli.py train sample_training_data.csv \\"
echo "    --text-col text --label-col sentiment"
echo ""
echo "=================================================="
