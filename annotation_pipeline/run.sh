#!/bin/bash

# Image Annotation Pipeline - Quick Start Script

echo "========================================="
echo "  Image Annotation Pipeline"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✓ Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Load environment variables from .env if present
if [ -f .env ]; then
  echo "Loading environment variables from .env"
  # Export only non-comment lines with KEY=VALUE
  set -a
  # shellcheck disable=SC2046
  source <(grep -v '^#' .env | sed -e 's/\r$//')
  set +a
fi

echo ""
echo "========================================="
echo "  Starting Flask Server"
echo "========================================="
echo ""
echo "📍 Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python app.py

