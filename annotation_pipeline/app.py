from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from utils.annotator import GPTAnnotator
from utils.visualizer import visualize_annotations

# Load environment variables robustly (works regardless of CWD)
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=False)
else:
    # Fallback: load .env from the script directory if present
    script_env = (Path(__file__).resolve().parent / '.env')
    if script_env.exists():
        load_dotenv(dotenv_path=script_env, override=False)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/images'
app.config['ANNOTATION_FOLDER'] = 'data/annotations'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Get max file size from env (in MB)
max_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '16'))
app.config['MAX_CONTENT_LENGTH'] = max_size_mb * 1024 * 1024

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATION_FOLDER'], exist_ok=True)

# Initialize annotator
try:
    annotator = GPTAnnotator()
except ValueError as e:
    print(f"❌ Error: {e}")
    print("The application requires a valid OpenAI API key to function.")
    print("Please set OPENAI_API_KEY in your .env file and restart the server.")
    exit(1)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/images')
def get_images():
    """Get list of all images with their annotation status"""
    images = []
    image_folder = Path(app.config['UPLOAD_FOLDER'])
    annotation_folder = Path(app.config['ANNOTATION_FOLDER'])
    
    for img_path in image_folder.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            annotation_path = annotation_folder / f"{img_path.stem}.json"
            images.append({
                'filename': img_path.name,
                'has_annotation': annotation_path.exists(),
                'annotation_path': str(annotation_path) if annotation_path.exists() else None
            })
    
    return jsonify({'images': images})


@app.route('/api/image/<filename>')
def get_image(filename):
    """Serve image file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/annotation/<filename>')
def get_annotation(filename):
    """Get annotation for a specific image"""
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation not found'}), 404
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    return jsonify(annotation)


@app.route('/api/annotate/<filename>', methods=['POST'])
def annotate_image(filename):
    """Generate annotation for an image using OpenAI API"""
    image_path = Path(app.config['UPLOAD_FOLDER']) / filename
    
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    # Generate annotation using OpenAI API
    try:
        annotation = annotator.annotate(str(image_path))
    except Exception as e:
        # The annotator prints detailed diagnostics to stdout/stderr.
        # Return the error string so the frontend can surface it to the user.
        return jsonify({'error': f'Annotation failed: {str(e)}'}), 500
    
    # Save annotation
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{image_path.stem}.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    return jsonify(annotation)


@app.route('/api/preprocess/<filename>', methods=['POST'])
def preprocess_image(filename):
    """Run preprocessing only and return proposed boxes/points (no LLM)."""
    image_path = Path(app.config['UPLOAD_FOLDER']) / filename
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    try:
        body = request.get_json(silent=True) or {}
        max_elems = body.get('max_elements')
        result = annotator.preprocess_only(str(image_path), max_elements=max_elems)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Preprocess failed: {str(e)}'}), 500


@app.route('/api/image/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete an image and its annotation"""
    image_path = Path(app.config['UPLOAD_FOLDER']) / filename
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # Delete image file
        image_path.unlink()
        
        # Delete annotation if exists
        if annotation_path.exists():
            annotation_path.unlink()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Failed to delete image: {str(e)}'}), 500


@app.route('/api/annotation/<filename>', methods=['PUT'])
def update_annotation(filename):
    """Update annotation for an image"""
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    annotation_data = request.json
    
    with open(annotation_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return jsonify({'success': True})


@app.route('/api/annotation/<filename>/paste', methods=['POST'])
def paste_annotation(filename):
    """
    Create annotation from pasted JSON
    Allows users to paste their own annotation JSON instead of using AI generation
    """
    try:
        # Get the pasted JSON data
        pasted_data = request.json.get('annotation')
        
        if not pasted_data:
            return jsonify({'error': 'No annotation data provided'}), 400
        
        # Validate the annotation structure
        if 'img_size' not in pasted_data or 'element' not in pasted_data:
            return jsonify({'error': 'Invalid annotation format. Must contain "img_size" and "element" fields'}), 400
        
        # Validate img_size
        if not isinstance(pasted_data['img_size'], list) or len(pasted_data['img_size']) != 2:
            return jsonify({'error': 'img_size must be a list of [width, height]'}), 400
        
        # Validate elements
        if not isinstance(pasted_data['element'], list):
            return jsonify({'error': 'element must be a list'}), 400
        
        for idx, elem in enumerate(pasted_data['element']):
            if not isinstance(elem, dict):
                return jsonify({'error': f'Element {idx} must be an object'}), 400
            
            if 'instruction' not in elem or 'bbox' not in elem or 'point' not in elem:
                return jsonify({'error': f'Element {idx} must have instruction, bbox, and point fields'}), 400
            
            if not isinstance(elem['bbox'], list) or len(elem['bbox']) != 4:
                return jsonify({'error': f'Element {idx} bbox must be [x1, y1, x2, y2]'}), 400
            
            if not isinstance(elem['point'], list) or len(elem['point']) != 2:
                return jsonify({'error': f'Element {idx} point must be [x, y]'}), 400
        
        # Save the annotation
        annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
        
        with open(annotation_path, 'w') as f:
            json.dump(pasted_data, f, indent=2)
        
        return jsonify({'success': True, 'annotation': pasted_data})
        
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing annotation: {str(e)}'}), 500


@app.route('/api/annotation/<filename>/element/<int:element_index>', methods=['DELETE'])
def delete_element(filename, element_index):
    """Delete a specific element from annotation"""
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation not found'}), 404
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    if 0 <= element_index < len(annotation.get('element', [])):
        annotation['element'].pop(element_index)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return jsonify({'success': True, 'annotation': annotation})
    
    return jsonify({'error': 'Invalid element index'}), 400


@app.route('/api/visualize/<filename>')
def visualize_image(filename):
    """Get visualized image with annotations"""
    image_path = Path(app.config['UPLOAD_FOLDER']) / filename
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation not found'}), 404
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    # Generate visualization
    vis_image_base64 = visualize_annotations(str(image_path), annotation)
    
    return jsonify({'image': vis_image_base64})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload new image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': True, 'filename': filename})


@app.route('/api/batch-annotate', methods=['POST'])
def batch_annotate():
    """Annotate all images in the folder"""
    image_folder = Path(app.config['UPLOAD_FOLDER'])
    results = []
    
    for img_path in image_folder.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{img_path.stem}.json"
            
            # Skip if already annotated (unless force flag is set)
            force = request.json.get('force', False) if request.json else False
            if annotation_path.exists() and not force:
                results.append({
                    'filename': img_path.name,
                    'status': 'skipped',
                    'reason': 'already_annotated'
                })
                continue
            
            try:
                annotation = annotator.annotate(str(img_path))
                with open(annotation_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                
                results.append({
                    'filename': img_path.name,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'filename': img_path.name,
                    'status': 'error',
                    'error': str(e)
                })
    
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

