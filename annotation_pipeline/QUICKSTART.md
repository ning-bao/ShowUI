# Quick Start Guide

Get up and running with the Image Annotation Pipeline in minutes!

## Installation (One-Time Setup)

```bash
# Navigate to the project
cd /mnt/f/USYD/Research/ShowUI/annotation_pipeline

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional but recommended)
cp .env.example .env
nano .env  # Add your OpenAI API key
```

**⚠️ Note:** An OpenAI API key is **required**. The system will not start without a valid API key.

## Starting the Application

### Option 1: Using the Quick Start Script (Recommended)
```bash
./run.sh
```

### Option 2: Manual Start
```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## First Steps

### 1. Upload Images
- Click "📤 Upload" button
- Drag & drop or select images
- Supported formats: JPG, PNG, GIF, BMP

### 2. Generate Annotations
**For a single image:**
- Select image from sidebar
- Click "🤖 Generate Annotation"

**For all images:**
- Click "🤖 Annotate All" in sidebar
- Check "Force re-annotate" if needed
- Click "Start Batch Annotation"

### 3. View & Manage Annotations
- Click any image to view its annotations
- Click "👁️ Toggle Visualization" to see bounding boxes
- Click any element card to highlight it
- Click "🗑️ Delete" to remove elements

## Project Structure

```
annotation_pipeline/
├── app.py                    # Flask server
├── requirements.txt          # Dependencies
├── run.sh                    # Quick start script
├── demo.py                   # Demo/test script
│
├── data/
│   ├── images/              # Your images go here
│   └── annotations/         # Generated JSON files
│
├── static/
│   ├── css/style.css        # UI styling
│   └── js/main.js           # Frontend logic
│
├── templates/
│   └── index.html           # Web interface
│
└── utils/
    ├── annotator.py         # GPT-5 annotation
    └── visualizer.py        # Visualization
```

## Testing the Setup

Run the demo script to test the pipeline:

```bash
python demo.py
```

This will:
- Check for images in `data/images/`
- Generate sample annotations
- Save them to `data/annotations/`
- Display annotation details

## Common Tasks

### Adding Images Manually
```bash
# Copy images to the data folder
cp /path/to/your/images/*.jpg data/images/
```

### Viewing Annotation JSON
```bash
# View annotation for an image
cat data/annotations/your_image.json
```

### Clearing All Annotations
```bash
# Remove all annotation files
rm data/annotations/*.json
```

## OpenAI API Integration

The system is **fully integrated** with OpenAI API and works in two modes:

### Mode 1: OpenAI API (Intelligent Annotation)

1. **Get API Key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key

2. **Set in .env file:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Restart server:**
   ```bash
   python app.py
   ```
   
   You'll see: `✓ OpenAI API initialized`

### Requirements

The system **requires** a valid OpenAI API key:
- No dummy/test mode available
- Real AI annotations only
- Set OPENAI_API_KEY in .env before starting

### Customizing the AI Prompt

Edit `utils/annotator.py` → `_get_annotation_prompt()` method to customize how the AI analyzes images.

The default prompt is optimized for UI/UX screenshots and identifies:
- Buttons and clickable elements
- Input fields and forms
- Dropdowns and selects
- Icons and navigation
- Interactive controls

### Using Paste JSON Feature

Don't want to use AI? Paste your own JSON:

1. Select an image
2. Click "📋 Paste JSON"
3. Paste your annotation JSON
4. Click "Validate JSON" to check
5. Click "Apply Annotation" to save

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/images` | GET | List all images |
| `/api/image/<filename>` | GET | Get image file |
| `/api/annotation/<filename>` | GET | Get annotation JSON |
| `/api/annotate/<filename>` | POST | Generate annotation |
| `/api/annotation/<filename>` | PUT | Update annotation |
| `/api/annotation/<filename>/element/<index>` | DELETE | Delete element |
| `/api/visualize/<filename>` | GET | Get visualized image |
| `/api/upload` | POST | Upload new image |
| `/api/batch-annotate` | POST | Annotate all images |

## Annotation Format

```json
{
  "img_size": [1920, 1080],
  "element": [
    {
      "instruction": "Click the submit button to complete the form",
      "bbox": [0.1, 0.5, 0.3, 0.6],
      "point": [0.2, 0.55]
    }
  ]
}
```

- **img_size**: Image dimensions `[width, height]` in pixels
- **bbox**: Normalized coordinates `[x1, y1, x2, y2]` (0.0 to 1.0)
- **point**: Normalized center `[x, y]` (0.0 to 1.0)

## Troubleshooting

### Port 5000 already in use
```bash
# Use a different port
flask run --port 5001
```

### Images not appearing
- Check images are in `data/images/`
- Verify file extensions (.jpg, .png, etc.)
- Refresh the page (F5)

### Annotations not saving
- Check folder permissions
- Look for errors in terminal
- Check browser console (F12)

### Module not found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## Tips & Best Practices

1. **Image Size**: Keep images under 5MB for best performance
2. **File Names**: Use simple names without special characters
3. **Batch Processing**: Process 10-20 images at a time for stability
4. **Backups**: Regularly backup your `data/annotations/` folder
5. **Browser**: Use Chrome or Firefox for best compatibility

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review the code comments in `utils/annotator.py` for GPT-5 integration
- Test with `demo.py` to verify setup

## What's Next?

Once you have the pipeline running:

1. ✅ Upload your UI screenshots
2. ✅ Generate annotations (dummy or GPT-5)
3. ✅ Review and refine annotations
4. ✅ Export JSON for your ML pipeline
5. ✅ Integrate with your training code

---

**Happy Annotating! 🎯**

