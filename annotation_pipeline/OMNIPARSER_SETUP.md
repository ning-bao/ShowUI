# OmniParser v2 Setup Guide

This guide will help you set up [Microsoft's OmniParser v2](https://huggingface.co/microsoft/OmniParser-v2.0) for local UI element detection.

## What is OmniParser v2?

OmniParser v2 is a screen parsing tool from Microsoft that detects interactive UI elements in screenshots. It consists of:
- **Icon Detection**: YOLOv8 model (AGPL license)
- **Icon Caption**: Florence-2 model (MIT license)

Key improvements in v2:
- 60% faster than v1 (0.6s/frame on A100, 0.8s on single RTX 4090)
- 39.6 average accuracy on ScreenSpot Pro benchmark
- Larger and cleaner training dataset

Reference: https://huggingface.co/microsoft/OmniParser-v2.0

## Installation

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
cd /mnt/f/USYD/Research/ShowUI/annotation_pipeline
bash setup_omniparser.sh
```

This will:
1. Install PyTorch (with CUDA if available)
2. Install Transformers, Ultralytics, and other dependencies
3. Download OmniParser v2 models from Hugging Face
4. Verify the installation

### Option 2: Manual Installation

1. **Install PyTorch**

   With CUDA (if you have NVIDIA GPU):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   CPU-only:
   ```bash
   pip install torch torchvision
   ```

2. **Install other dependencies**

   ```bash
   pip install transformers ultralytics huggingface-hub pillow numpy
   ```

3. **Models will auto-download on first use** (or pre-download with the script)

## Usage

### 1. Standalone Testing

Test OmniParser directly:

```bash
python omniparser_local.py --image /path/to/screenshot.png
```

Options:
- `--conf 0.25`: Detection confidence threshold (0-1)
- `--captions`: Generate captions (slower, uses Florence-2)
- `--device cuda`: Force CPU or CUDA
- `--output result.json`: Save to file

Example with captions:
```bash
python omniparser_local.py --image test.png --captions --output elements.json
```

### 2. Integration with Annotation Pipeline

Add to your `.env` file:

```env
# Use OmniParser v2 for preprocessing
ANNOTATOR_PREPROCESS_BACKEND=omni
OMNIPARSER_URL=local

# Optional: set to 'all' or 0 for unlimited elements
ANNOTATOR_PREPROCESS_MAX_ELEMENTS=all
```

Then use the web UI or API:
- **Web UI**: Click "🧩 Preprocess (Boxes)" button
- **API**: POST to `/api/preprocess/<filename>`

### 3. Using HTTP Service (Alternative)

If you deploy OmniParser as an HTTP service:

```env
ANNOTATOR_PREPROCESS_BACKEND=omni
OMNIPARSER_URL=http://your-server:8000/parse
OMNIPARSER_API_KEY=your-key-if-needed
OMNIPARSER_TIMEOUT=30
```

## Performance Notes

- **First run**: Slower as models load into memory
- **GPU recommended**: 10-20x faster than CPU
- **Memory**: ~4GB VRAM for detection, +2GB if using captions
- **Latency**: 
  - A100: ~0.6s per image
  - RTX 4090: ~0.8s per image
  - CPU: ~5-10s per image

## Fallback Behavior

The pipeline automatically falls back to OpenCV-based detection if:
- OmniParser dependencies are not installed
- Model loading fails
- Inference fails

To force OpenCV backend:
```env
ANNOTATOR_PREPROCESS_BACKEND=cv
```

## Troubleshooting

### "No module named 'omniparser_local'"

The `omniparser_local.py` file must be in the same directory as `annotator.py`, or in Python path.

Solution:
```bash
cd /mnt/f/USYD/Research/ShowUI/annotation_pipeline
# Ensure omniparser_local.py exists
ls -la omniparser_local.py
```

### "CUDA out of memory"

Reduce batch size or use CPU:
```bash
python omniparser_local.py --image test.png --device cpu
```

### Models not downloading

Manually download:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/OmniParser-v2.0",
    local_dir="~/.cache/omniparser/OmniParser-v2.0"
)
```

### ImportError: transformers/ultralytics

Reinstall dependencies:
```bash
pip install --upgrade transformers ultralytics torch
```

## License Notes

- **Icon Detection (YOLOv8)**: AGPL-3.0
- **Icon Caption (Florence-2)**: MIT
- Ensure compliance with AGPL if distributing the detection model

## References

- [Hugging Face Model Card](https://huggingface.co/microsoft/OmniParser-v2.0)
- [OmniParser GitHub](https://github.com/microsoft/OmniParser) (check for v2 updates)
- [ArXiv Paper](https://arxiv.org/abs/2408.00203)


