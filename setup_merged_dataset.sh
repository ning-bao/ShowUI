#!/bin/bash
# Setup script for preparing merged ShowUI-desktop + ScreenSpot dataset

set -e

echo "🚀 Setting up merged ShowUI-desktop + ScreenSpot dataset..."

# Check if we're in the right directory
if [ ! -f "prepare_merged_dataset.py" ]; then
    echo "❌ Error: prepare_merged_dataset.py not found. Please run this from the ShowUI root directory."
    exit 1
fi

# Default paths
DATA_DIR="${1:-~/showui_data}"
OUTPUT_DIR="${2:-~/showui_data/ShowUI-desktop-merged}"

echo "📁 Data directory: $DATA_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

# Check if datasets exist
if [ ! -d "$DATA_DIR/ScreenSpot" ]; then
    echo "❌ Error: ScreenSpot dataset not found at $DATA_DIR/ScreenSpot"
    echo "Please ensure you have downloaded the ScreenSpot dataset."
    exit 1
fi

if [ ! -d "$DATA_DIR/ShowUI-desktop" ]; then
    echo "❌ Error: ShowUI-desktop dataset not found at $DATA_DIR/ShowUI-desktop"
    echo "Please ensure you have downloaded the ShowUI-desktop dataset."
    exit 1
fi

# Run the preparation script
echo "🔄 Running dataset preparation..."
python3 prepare_merged_dataset.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --screenspot_splits screenspot_desktop

echo "✅ Dataset preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update dataset mapping in data/dset_shared_grounding.py:"
echo '   "showui-desktop-merged": "ShowUI-desktop-merged"'
echo ""
echo "2. Run training with:"
echo "   --dataset_dir $DATA_DIR"
echo "   --train_dataset showui-desktop-merged"
