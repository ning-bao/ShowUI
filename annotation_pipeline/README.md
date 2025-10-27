# Image Annotation Pipeline

A web-based data annotation pipeline for UI/UX element detection and annotation. This tool allows you to annotate images with bounding boxes and interaction instructions, with support for GPT-5 API integration for automated annotation.

## Features

- 🖼️ **Web Interface**: Modern, responsive web interface to view and manage annotations
- 🤖 **OpenAI API Integration**: Automatic annotation generation using GPT-4o with vision capabilities
- 📋 **Paste JSON**: Manually paste your own annotation JSON instead of using AI generation
- 📊 **Visualization**: View annotations overlaid on images with bounding boxes and center points
- ✏️ **Element Management**: Select and delete individual annotation elements
- 📤 **Upload**: Drag-and-drop or click to upload images
- 🔄 **Batch Processing**: Annotate multiple images at once
- 💾 **JSON Export**: Annotations saved in structured JSON format
- 🔐 **Environment Config**: Secure API key management using .env files
- 🎭 **Dual Mode**: Works with OpenAI API or dummy mode for testing

## Project Structure

```
annotation_pipeline/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/
│   ├── css/
│   │   └── style.css     # Application styles
│   └── js/
│       └── main.js       # Frontend JavaScript
├── templates/
│   └── index.html        # Main web interface
├── data/
│   ├── images/           # Uploaded images
│   └── annotations/      # JSON annotation files
└── utils/
    ├── __init__.py
    ├── annotator.py      # GPT-5 annotation logic
    └── visualizer.py     # Annotation visualization
```

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd /mnt/f/USYD/Research/ShowUI/annotation_pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   nano .env  # or use your favorite editor
   ```
   
   Set your OpenAI API key in `.env`:
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
   
   See [ENV_SETUP.md](ENV_SETUP.md) for detailed configuration options.

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Using the Web Interface

1. **Upload Images**:
   - Click the "📤 Upload" button in the sidebar
   - Drag and drop an image or click "Choose File"
   - Supported formats: JPG, JPEG, PNG, GIF, BMP

2. **View Annotations**:
   - Click on any image in the sidebar
   - View annotation details in the right panel
   - Toggle between original and visualized view

3. **Generate Annotations**:
   
   **Option A: AI Generation**
   - Select an image
   - Click "🤖 Generate Annotation" to create annotations using OpenAI
   - Use "🤖 Annotate All" to process all images in batch
   
   **Option B: Paste JSON**
   - Select an image
   - Click "📋 Paste JSON"
   - Paste your annotation JSON
   - Click "Validate JSON" to check format
   - Click "Apply Annotation" to save

4. **Manage Elements**:
   - Click on any element card to highlight it
   - Click "🗑️ Delete" to remove an element
   - Changes are saved automatically

### Annotation Format

Annotations are saved as JSON files with the following structure:

```json
{
  "img_size": [1920, 1080],
  "element": [
    {
      "instruction": "Click the submit button to save changes",
      "bbox": [100, 200, 300, 250],
      "point": [200, 225]
    }
  ]
}
```

Where:
- `img_size`: Image dimensions in pixels `[width, height]`
- `bbox`: **ABSOLUTE PIXEL** coordinates `[x1, y1, x2, y2]` (integers)
- `point`: **ABSOLUTE PIXEL** center point `[x, y]` (integers)
- `instruction`: Natural language instruction for the UI element (≤120 chars)

**⚠️ Important:** As of v2.1, coordinates use **absolute pixels**, not normalized (0-1) values. This change improves accuracy and aligns with ShowUI RL training requirements.

## API Endpoints

### GET `/api/images`
Get list of all images with annotation status

### GET `/api/image/<filename>`
Retrieve a specific image file

### GET `/api/annotation/<filename>`
Get annotation JSON for an image

### POST `/api/annotate/<filename>`
Generate annotation for an image using GPT-5

### PUT `/api/annotation/<filename>`
Update annotation for an image

### DELETE `/api/annotation/<filename>/element/<index>`
Delete a specific element from annotation

### GET `/api/visualize/<filename>`
Get image with annotations visualized

### POST `/api/upload`
Upload a new image

### POST `/api/batch-annotate`
Annotate all images in the folder
- Body: `{ "force": boolean }` - force re-annotation of existing

## OpenAI API Integration

### Setup

1. **Get an OpenAI API key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key

2. **Configure the `.env` file:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   OPENAI_MODEL=gpt-4o
   OPENAI_MAX_TOKENS=4096
   ```

3. **Start the server:**
   ```bash
   python app.py
   ```
   
   You'll see:
   ```
   ✓ OpenAI API initialized
     Model: gpt-4o
   ```

### OpenAI API Required

The system requires a valid OpenAI API key to function:

- Uses GPT-4o vision for intelligent annotation
- Detects UI elements automatically
- Generates natural language instructions with absolute pixel coordinates
- Costs apply per API call (~$0.01-0.05 per image)
- **No dummy mode** - real AI annotation only

### The Annotation Prompt

The system uses a carefully crafted prompt (in `utils/annotator.py`) that instructs the AI to:

- Identify all interactive UI elements
- Generate clear, actionable instructions
- Return normalized bounding boxes and center points
- Focus on buttons, inputs, dropdowns, etc.
- Exclude decorative elements

**Customize the prompt** by editing the `_get_annotation_prompt()` method in `utils/annotator.py`.

### Cost Management

- **Estimate:** $0.01-0.05 per image
- **For 1000 images:** ~$20-50
- **Set limits:** Configure spending limits in OpenAI dashboard
- **Monitor usage:** Check usage at https://platform.openai.com/usage

See [ENV_SETUP.md](ENV_SETUP.md) for detailed cost estimation.

## Development

### Adding New Features

The codebase is modular and easy to extend:

- **Backend**: Add new routes in `app.py`
- **Frontend**: Add new UI components in `templates/index.html` and `static/js/main.js`
- **Styling**: Modify `static/css/style.css`
- **Annotation Logic**: Extend `utils/annotator.py`
- **Visualization**: Modify `utils/visualizer.py`

### Customizing Annotation Format

To change the annotation format:

1. Update the return format in `utils/annotator.py`
2. Modify the display logic in `static/js/main.js`
3. Update the visualization in `utils/visualizer.py`

## Troubleshooting

### Images not loading
- Check that images are in the `data/images/` folder
- Verify file permissions
- Check Flask logs for errors

### Annotations not saving
- Ensure `data/annotations/` folder has write permissions
- Check browser console for JavaScript errors

### GPT-5 API errors
- Verify API key is set correctly
- Check API quota and rate limits
- Review error messages in Flask logs

## Future Enhancements

- [ ] Support for multiple annotation types (classification, segmentation)
- [ ] Export annotations to different formats (COCO, YOLO, etc.)
- [ ] Annotation history and versioning
- [ ] Multi-user collaboration
- [ ] Keyboard shortcuts for faster annotation
- [ ] Import existing annotations
- [ ] Advanced filtering and search

## License

This project is part of the ShowUI research project at USYD.

## Contributing

For questions or contributions, please contact the research team.

