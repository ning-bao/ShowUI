from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from utils.annotator import GPTAnnotator
from utils.visualizer import visualize_annotations
try:
    from . import db as dbm
except Exception:
    try:
        from ShowUI.annotation_pipeline import db as dbm
    except Exception:
        import db as dbm
import threading
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

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
dbm.init_db()

# Initialize annotator
try:
    annotator = GPTAnnotator()
except ValueError as e:
    print(f"❌ Error: {e}")
    print("The application requires a valid OpenAI API key to function.")
    print("Please set OPENAI_API_KEY in your .env file and restart the server.")
    exit(1)


########################
# Batch Job Infrastructure
########################

class BatchJobManager:
    def __init__(self, max_workers: int = None):
        try:
            max_workers_env = int(os.getenv('BATCH_MAX_WORKERS', '3'))
        except Exception:
            max_workers_env = 3
        self.max_workers = max_workers if max_workers is not None else max_workers_env
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.jobs = {}
        self.job_events = {}
        self.lock = threading.Lock()

    def create_job(self, images: list, force: bool) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                'status': 'running',
                'total': len(images),
                'completed': 0,
                'success': 0,
                'skipped': 0,
                'errors': 0,
                'results': [],
                'force': force,
                'created_at': time.time(),
            }
            self.job_events[job_id] = queue.Queue()

        # submit tasks
        for img in images:
            self.executor.submit(self._process_image_task, job_id, img, force)

        # also start a monitor thread to finalize when complete
        threading.Thread(target=self._monitor_job_done, args=(job_id,), daemon=True).start()
        return job_id

    def _monitor_job_done(self, job_id: str):
        # Wait until completed == total then send complete event
        while True:
            with self.lock:
                job = self.jobs.get(job_id)
                if not job:
                    return
                if job['completed'] >= job['total']:
                    job['status'] = 'complete'
                    self._emit(job_id, {
                        'type': 'complete',
                        'summary': {
                            'total': job['total'],
                            'success': job['success'],
                            'skipped': job['skipped'],
                            'errors': job['errors'],
                        }
                    })
                    # small delay then close stream by placing a sentinel
                    self._emit(job_id, {'type': 'end'})
                    return
            time.sleep(0.2)

    def _emit(self, job_id: str, payload: dict):
        q = self.job_events.get(job_id)
        if q:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    def _process_image_task(self, job_id: str, img_entry: dict, force: bool):
        image_folder = Path(app.config['UPLOAD_FOLDER'])
        annotation_folder = Path(app.config['ANNOTATION_FOLDER'])
        img_path: Path = img_entry['path']
        relative_path = str(img_path.relative_to(image_folder)).replace('\\', '/')
        annotation_path = annotation_folder / f"{img_path.stem}.json"

        # Skip logic
        if annotation_path.exists() and not force:
            with self.lock:
                job = self.jobs.get(job_id)
                if job:
                    job['completed'] += 1
                    job['skipped'] += 1
                    job['results'].append({'filename': relative_path, 'status': 'skipped'})
            self._emit(job_id, {
                'type': 'image_done',
                'filename': relative_path,
                'status': 'skipped',
                'completed': self.jobs.get(job_id, {}).get('completed', 0),
                'total': self.jobs.get(job_id, {}).get('total', 0),
            })
            return

        # Preprocess
        try:
            self._emit(job_id, {'type': 'preprocess_start', 'filename': relative_path})
            hints = annotator._compute_preprocess_hints(str(img_path), max_elements=annotator.preprocess_max_elements)
            self._emit(job_id, {'type': 'preprocessed', 'filename': relative_path, 'hints': len(hints)})
        except Exception as e:
            with self.lock:
                job = self.jobs.get(job_id)
                if job:
                    job['completed'] += 1
                    job['errors'] += 1
                    job['results'].append({'filename': relative_path, 'status': 'error', 'error': f'Preprocess failed: {e}'})
            self._emit(job_id, {
                'type': 'image_done',
                'filename': relative_path,
                'status': 'error',
                'error': f'Preprocess failed: {e}',
                'completed': self.jobs.get(job_id, {}).get('completed', 0),
                'total': self.jobs.get(job_id, {}).get('total', 0),
            })
            return

        # Send OpenAI request with hints
        try:
            self._emit(job_id, {'type': 'request_sent', 'filename': relative_path})
            annotation = annotator.annotate_with_hints(str(img_path), hints)
            annotation_folder.mkdir(parents=True, exist_ok=True)
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            try:
                # mark annotated in DB
                dbm.set_has_annotation(relative_path, True)
            except Exception:
                pass
            with self.lock:
                job = self.jobs.get(job_id)
                if job:
                    job['completed'] += 1
                    job['success'] += 1
                    job['results'].append({'filename': relative_path, 'status': 'success'})
            self._emit(job_id, {
                'type': 'image_done',
                'filename': relative_path,
                'status': 'success',
                'completed': self.jobs.get(job_id, {}).get('completed', 0),
                'total': self.jobs.get(job_id, {}).get('total', 0),
            })
        except Exception as e:
            with self.lock:
                job = self.jobs.get(job_id)
                if job:
                    job['completed'] += 1
                    job['errors'] += 1
                    job['results'].append({'filename': relative_path, 'status': 'error', 'error': str(e)})
            self._emit(job_id, {
                'type': 'image_done',
                'filename': relative_path,
                'status': 'error',
                'error': str(e),
                'completed': self.jobs.get(job_id, {}).get('completed', 0),
                'total': self.jobs.get(job_id, {}).get('total', 0),
            })

    def get_job(self, job_id: str):
        with self.lock:
            return dict(self.jobs.get(job_id, {}))

    def sse_stream(self, job_id: str):
        q = self.job_events.get(job_id)
        if not q:
            # empty stream end
            def _gen_empty():
                yield "event: end\n\n"
            return _gen_empty()

        def _gen():
            # initial snapshot
            with self.lock:
                job = self.jobs.get(job_id)
                if job:
                    init_payload = {
                        'type': 'init',
                        'total': job['total'],
                        'completed': job['completed'],
                        'success': job['success'],
                        'skipped': job['skipped'],
                        'errors': job['errors'],
                    }
                    yield f"data: {json.dumps(init_payload)}\n\n"
            # stream events
            while True:
                try:
                    payload = q.get(timeout=15)
                except Exception:
                    # heartbeat to keep connection alive
                    yield ": keep-alive\n\n"
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get('type') == 'end':
                    yield f"event: end\n\n"
                    return
                yield f"data: {json.dumps(payload)}\n\n"

        return _gen()


job_manager = BatchJobManager()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/images')
def get_images():
    """Get list of images with pagination, backed by SQLite. Falls back to FS if DB empty."""
    # pagination params
    try:
        page = int(request.args.get('page', '1'))
        page_size = int(request.args.get('page_size', '500'))
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 500
        if page_size > 5000:
            page_size = 5000
    except Exception:
        page, page_size = 1, 500

    total = dbm.count_images()
    if total == 0:
        # First time: index from filesystem quickly and return
        images = []
        image_folder = Path(app.config['UPLOAD_FOLDER'])
        annotation_folder = Path(app.config['ANNOTATION_FOLDER'])
        for img_path in image_folder.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                rel = str(img_path.relative_to(image_folder)).replace('\\', '/')
                ann = annotation_folder / f"{img_path.stem}.json"
                has_ann = ann.exists()
                try:
                    size_b = img_path.stat().st_size
                except Exception:
                    size_b = None
                # upsert to DB
                try:
                    dbm.upsert_image(rel, has_annotation=has_ann, size_bytes=size_b)
                except Exception:
                    pass
                images.append({'filename': rel, 'has_annotation': has_ann, 'annotation_path': str(ann) if has_ann else None})
        total = len(images)
        return jsonify({'images': images, 'total': total, 'page': 1, 'page_size': total})

    offset = (page - 1) * page_size
    rows = dbm.list_images(limit=page_size, offset=offset)
    # Keep response shape backward compatible
    images = []
    for r in rows:
        images.append({
            'filename': r['filename'],
            'has_annotation': bool(r['has_annotation']),
            'annotation_path': None,
        })
    return jsonify({'images': images, 'total': total, 'page': page, 'page_size': page_size})


@app.route('/api/folders')
def get_folders():
    """Return a list of folder paths from the DB (including empty folders)."""
    try:
        folders = dbm.list_all_folders()
    except Exception:
        folders = []
    return jsonify({'folders': folders})


@app.route('/api/image/<path:filename>')
def get_image(filename):
    """Serve image file with caching headers"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    # Add cache headers to prevent image re-fetching
    response.cache_control.max_age = 3600  # Cache for 1 hour
    response.cache_control.public = True
    return response


@app.route('/api/annotation/<path:filename>')
def get_annotation(filename):
    """Get annotation for a specific image"""
    annotation_path = Path(app.config['ANNOTATION_FOLDER']) / f"{Path(filename).stem}.json"
    
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation not found'}), 404
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    return jsonify(annotation)


@app.route('/api/annotate/<path:filename>', methods=['POST'])
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
    
    try:
        dbm.set_has_annotation(str(filename), True)
    except Exception:
        pass
    return jsonify(annotation)


@app.route('/api/preprocess/<path:filename>', methods=['POST'])
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


@app.route('/api/image/<path:filename>', methods=['DELETE'])
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
        try:
            dbm.delete_image(str(filename))
        except Exception:
            pass
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
    try:
        dbm.set_has_annotation(str(filename), True)
    except Exception:
        pass
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
        try:
            dbm.set_has_annotation(str(filename), True)
        except Exception:
            pass
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
        relative_path = request.form.get('relative_path', '')
        
        if relative_path:
            parts = Path(relative_path).parts
            if len(parts) > 1:
                safe_parts = [secure_filename(p) for p in parts]
                filename = '/'.join(safe_parts[1:])
            else:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(file.filename)
        
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(file_path))
        try:
            size_b = None
            try:
                size_b = file_path.stat().st_size
            except Exception:
                pass
            ann_path = Path(app.config['ANNOTATION_FOLDER']) / f"{file_path.stem}.json"
            dbm.upsert_image(filename, has_annotation=ann_path.exists(), size_bytes=size_b)
        except Exception:
            pass
        return jsonify({'success': True, 'filename': filename})


@app.route('/api/folder', methods=['POST'])
def create_folder():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Missing folder name'}), 400
    safe = secure_filename(name)
    folder_path = Path(app.config['UPLOAD_FOLDER']) / safe
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        return jsonify({'success': True, 'folder': str(safe)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-annotate', methods=['POST'])
def batch_annotate():
    """Start an asynchronous batch annotation job over all images. Returns a job_id."""
    image_folder = Path(app.config['UPLOAD_FOLDER'])
    body = request.get_json(silent=True) or {}
    force = body.get('force', False)

    images = []
    filenames = body.get('filenames')
    if isinstance(filenames, list) and filenames:
        # Limit to provided filenames
        allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        for name in filenames:
            try:
                # normalize path
                p = image_folder / str(name)
                if p.is_file() and p.suffix.lower() in allowed_ext:
                    images.append({'path': p})
            except Exception:
                continue
    else:
        # All images
        for img_path in image_folder.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                images.append({'path': img_path})

    job_id = job_manager.create_job(images, force)
    job = job_manager.get_job(job_id)
    return jsonify({'job_id': job_id, 'total': job.get('total', 0)})


@app.route('/api/batch-annotate/status/<job_id>')
def batch_status(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'job_not_found'}), 404
    return jsonify(job)


@app.route('/api/batch-annotate/stream/<job_id>')
def batch_stream(job_id):
    stream = job_manager.sse_stream(job_id)
    return Response(stream, mimetype='text/event-stream')


@app.route('/api/move-images', methods=['POST'])
def move_images():
    """Move images to a target folder."""
    body = request.get_json(silent=True) or {}
    filenames = body.get('filenames', [])
    target_folder = body.get('target_folder', '').strip()
    
    if not filenames:
        return jsonify({'error': 'No images selected'}), 400
    
    if not target_folder:
        return jsonify({'error': 'No target folder specified'}), 400
    
    image_folder = Path(app.config['UPLOAD_FOLDER'])
    annotation_folder = Path(app.config['ANNOTATION_FOLDER'])
    target_path = image_folder / target_folder
    
    # Ensure target folder exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    errors = []
    
    for filename in filenames:
        try:
            source_path = image_folder / filename
            if not source_path.exists():
                errors.append(f'{filename}: not found')
                continue
            
            # Get just the filename without any path
            base_name = Path(filename).name
            dest_path = target_path / base_name
            
            # Move the image file
            shutil.move(str(source_path), str(dest_path))
            
            # Move annotation if it exists
            ann_source = annotation_folder / f"{Path(filename).stem}.json"
            if ann_source.exists():
                ann_dest = annotation_folder / target_folder
                ann_dest.mkdir(parents=True, exist_ok=True)
                ann_dest_file = ann_dest / f"{Path(filename).stem}.json"
                shutil.move(str(ann_source), str(ann_dest_file))
            
            # Update database
            new_path = f"{target_folder}/{base_name}"
            try:
                dbm.delete_image(str(filename))
                has_ann = (annotation_folder / target_folder / f"{Path(filename).stem}.json").exists()
                dbm.upsert_image(new_path, has_annotation=has_ann)
            except Exception as e:
                print(f"DB update error for {filename}: {e}")
            
            moved_count += 1
            
        except Exception as e:
            errors.append(f'{filename}: {str(e)}')
            continue
    
    return jsonify({
        'success': True,
        'moved': moved_count,
        'errors': errors
    })


@app.route('/api/export', methods=['POST'])
def export_dataset():
    """Export selected images to ShowUI-desktop format."""
    import subprocess
    import tempfile
    
    body = request.get_json(silent=True) or {}
    filenames = body.get('filenames', [])
    split_name = body.get('split', 'train')
    
    if not filenames:
        return jsonify({'error': 'No images selected'}), 400
    
    # Create temporary output directory
    output_dir = Path(tempfile.gettempdir()) / f'showui_export_{int(time.time())}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run export script
        script_path = Path(__file__).parent / 'scripts' / 'export_showui_desktop.py'
        images_path = Path(app.config['UPLOAD_FOLDER'])
        annotations_path = Path(app.config['ANNOTATION_FOLDER'])
        
        result = subprocess.run(
            [
                'python3', str(script_path),
                '--images', str(images_path),
                '--annotations', str(annotations_path),
                '--output', str(output_dir),
                '--split', split_name
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Export failed',
                'details': result.stderr
            }), 500
        
        # Count exported files
        exported_images = len(list((output_dir / 'images').rglob('*'))) if (output_dir / 'images').exists() else 0
        
        return jsonify({
            'success': True,
            'output_path': str(output_dir),
            'exported_images': exported_images,
            'message': f'Exported {exported_images} images to {output_dir}'
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Export timeout'}), 500
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

