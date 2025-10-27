#!/bin/bash
# Setup script for OmniParser v2 local inference
# Reference: https://huggingface.co/microsoft/OmniParser-v2.0

set -e

echo "=========================================="
echo "OmniParser v2 Setup"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating venv first: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "1. Installing PyTorch..."
echo "   Detecting system..."

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ NVIDIA GPU detected"
    echo "   Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "   ℹ️  No NVIDIA GPU detected"
    echo "   Installing CPU-only PyTorch..."
    pip install torch torchvision
fi

echo ""
echo "2. Installing Transformers and Ultralytics..."
pip install transformers ultralytics huggingface-hub pillow numpy

echo ""
echo "3. Downloading OmniParser v2 models..."
python3 << 'PYEOF'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    
    print("   Downloading microsoft/OmniParser-v2.0...")
    cache_dir = Path.home() / '.cache' / 'omniparser'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id="microsoft/OmniParser-v2.0",
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir / "OmniParser-v2.0"),
        local_dir_use_symlinks=False
    )
    
    print(f"   ✓ Models downloaded to: {cache_dir / 'OmniParser-v2.0'}")
    
except Exception as e:
    print(f"   ✗ Download failed: {e}")
    print("   Models will be downloaded on first use.")
    sys.exit(0)
PYEOF

echo ""
echo "4. Testing OmniParser installation..."
python3 -c "
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from ultralytics import YOLO
    print('   ✓ All dependencies imported successfully')
    print(f'   PyTorch version: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'   ✗ Import failed: {e}')
    exit(1)
"

echo ""
echo "=========================================="
echo "✓ OmniParser v2 setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  1. Test standalone:"
echo "     python omniparser_local.py --image /path/to/screenshot.png"
echo ""
echo "  2. Use in annotation pipeline:"
echo "     Set in .env:"
echo "       ANNOTATOR_PREPROCESS_BACKEND=omni"
echo "       OMNIPARSER_URL=local"
echo ""
echo "  3. In web UI:"
echo "     Click 'Preprocess (Boxes)' to use OmniParser"
echo ""
echo "Note: First run will be slower as models initialize."
echo ""


