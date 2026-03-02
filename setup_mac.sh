#!/bin/bash

echo "========================================="
echo "HandGUI - macOS Setup Script"
echo "========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed."
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

echo "✓ pip3 found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. ⚠️  IMPORTANT - Grant Camera Permission:"
echo "   Go to: System Settings → Privacy & Security → Camera"
echo "   Enable camera access for Terminal (or your Python app)"
echo ""
echo "2. (Optional) For Virtual Camera support:"
echo "   • Download OBS Studio: https://obsproject.com/download"
echo "   • Install and open OBS Studio"
echo "   • Go to Tools → Start Virtual Camera"
echo "   • Keep OBS running in the background"
echo ""
echo "3. Run the application:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "Note: The app will work in preview mode without OBS."
echo "You only need OBS if you want to use it with Discord/Zoom."
echo ""
