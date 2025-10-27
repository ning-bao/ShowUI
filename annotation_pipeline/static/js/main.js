// Global state
let currentImage = null;
let currentAnnotation = null;
let currentPreprocess = null;
let selectedElementIndex = null;
let showingVisualization = true;
let showingPreprocessOverlay = false;
let hasUnsavedChanges = false;
let allImages = [];
let currentFilter = 'all';
let selectedImages = new Set();
let sortOrder = 'name-asc';

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadImages();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Upload modals
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadModal = document.getElementById('uploadModal');
    const uploadFolderBtn = document.getElementById('uploadFolderBtn');
    const folderModal = document.getElementById('folderModal');
    
    uploadBtn.addEventListener('click', () => uploadModal.classList.add('active'));
    uploadFolderBtn.addEventListener('click', () => folderModal.classList.add('active'));
    
    // Close modals
    document.querySelectorAll('.modal .close').forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            this.closest('.modal').classList.remove('active');
        });
    });
    
    // Bulk actions
    document.getElementById('bulkAnnotateBtn').addEventListener('click', bulkAnnotate);
    document.getElementById('bulkDeleteBtn').addEventListener('click', bulkDelete);
    
    // Sort button
    document.getElementById('sortBtn').addEventListener('click', cycleSortOrder);
    
    // File inputs
    document.getElementById('fileInput').addEventListener('change', handleFileSelect);
    document.getElementById('folderInput').addEventListener('change', handleFolderSelect);
    
    // Drag and drop for single files
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFiles(Array.from(files));
    });
    
    // Action buttons
    document.getElementById('annotateBtn').addEventListener('click', generateAnnotation);
    document.getElementById('preprocessBtn').addEventListener('click', runPreprocess);
    document.getElementById('toggleViewBtn').addEventListener('click', toggleVisualization);
    document.getElementById('togglePreprocessOverlayBtn').addEventListener('click', togglePreprocessOverlay);
    document.getElementById('saveBtn').addEventListener('click', saveAnnotation);
    document.getElementById('deleteImageBtn').addEventListener('click', deleteCurrentImage);
    
    // Paste JSON
    const pasteJsonBtn = document.getElementById('pasteJsonBtn');
    const pasteJsonModal = document.getElementById('pasteJsonModal');
    pasteJsonBtn.addEventListener('click', () => {
        if (!currentImage) {
            showToast('Please select an image first', 'error');
            return;
        }
        document.getElementById('jsonInput').value = '';
        document.getElementById('jsonError').style.display = 'none';
        pasteJsonModal.classList.add('active');
    });
    document.getElementById('validateJsonBtn').addEventListener('click', validatePastedJson);
    document.getElementById('applyJsonBtn').addEventListener('click', applyPastedJson);
    
    // Search and filter
    document.getElementById('searchInput').addEventListener('input', filterImages);
    document.getElementById('statusFilter').addEventListener('change', filterImages);
    
    // Close modals on outside click
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.classList.remove('active');
        }
    });
    
    // Resize handler
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if ((currentAnnotation && showingVisualization) || (currentPreprocess && showingPreprocessOverlay)) {
                drawAnnotations();
            }
        }, 100);
    });
}

// Load images from server
async function loadImages() {
    try {
        const response = await fetch('/api/images');
        const data = await response.json();
        allImages = data.images;
        displayImageList();
    } catch (error) {
        console.error('Error loading images:', error);
        showToast('Error loading images', 'error');
    }
}

// Display image list with filters
function displayImageList() {
    const imageList = document.getElementById('imageList');
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const statusFilter = document.getElementById('statusFilter').value;
    
    let filtered = allImages.filter(img => {
        if (searchTerm && !img.filename.toLowerCase().includes(searchTerm)) {
            return false;
        }
        
        if (statusFilter === 'annotated' && !img.has_annotation) return false;
        if (statusFilter === 'not-annotated' && img.has_annotation) return false;
        
        return true;
    });
    
    // Sort
    filtered.sort((a, b) => {
        if (sortOrder === 'name-asc') {
            return a.filename.localeCompare(b.filename);
        } else if (sortOrder === 'name-desc') {
            return b.filename.localeCompare(a.filename);
        } else if (sortOrder === 'status-asc') {
            return (a.has_annotation ? 1 : 0) - (b.has_annotation ? 1 : 0);
        } else if (sortOrder === 'status-desc') {
            return (b.has_annotation ? 1 : 0) - (a.has_annotation ? 1 : 0);
        }
        return 0;
    });
    
    if (filtered.length === 0) {
        imageList.innerHTML = '<div class="no-data">No images found</div>';
        return;
    }
    
    imageList.innerHTML = filtered.map(img => `
        <div class="image-item" data-filename="${img.filename}">
            <div class="image-item-checkbox">
                <input type="checkbox" 
                       data-filename="${img.filename}" 
                       ${selectedImages.has(img.filename) ? 'checked' : ''}
                       onclick="toggleImageSelection('${img.filename}', event)">
            </div>
            <div class="image-item-icon" onclick="selectImage('${img.filename}')">🖼️</div>
            <div class="image-item-info" onclick="selectImage('${img.filename}')">
                <div class="image-item-name" title="${img.filename}">${img.filename}</div>
                <div class="image-item-status">
                    <span class="status-badge ${img.has_annotation ? 'annotated' : 'not-annotated'}">
                        ${img.has_annotation ? '✓ Annotated' : '✗ Not annotated'}
                    </span>
                </div>
            </div>
        </div>
    `).join('');
    
    updateBulkActionsVisibility();
}

// Filter images
function filterImages() {
    displayImageList();
}

// Toggle image selection
function toggleImageSelection(filename, event) {
    event.stopPropagation();
    
    if (selectedImages.has(filename)) {
        selectedImages.delete(filename);
    } else {
        selectedImages.add(filename);
    }
    
    updateBulkActionsVisibility();
}

// Update bulk actions visibility
function updateBulkActionsVisibility() {
    const count = selectedImages.size;
    const selectedCountEl = document.getElementById('selectedCount');
    const bulkAnnotateBtn = document.getElementById('bulkAnnotateBtn');
    const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
    
    if (count > 0) {
        selectedCountEl.textContent = `${count} selected`;
        selectedCountEl.style.display = 'inline';
        bulkAnnotateBtn.style.display = 'inline-block';
        bulkDeleteBtn.style.display = 'inline-block';
    } else {
        selectedCountEl.style.display = 'none';
        bulkAnnotateBtn.style.display = 'none';
        bulkDeleteBtn.style.display = 'none';
    }
}

// Cycle sort order
function cycleSortOrder() {
    const orders = ['name-asc', 'name-desc', 'status-asc', 'status-desc'];
    const currentIndex = orders.indexOf(sortOrder);
    sortOrder = orders[(currentIndex + 1) % orders.length];
    
    const btn = document.getElementById('sortBtn');
    const labels = {
        'name-asc': 'Name ↑',
        'name-desc': 'Name ↓',
        'status-asc': 'Status ↑',
        'status-desc': 'Status ↓'
    };
    btn.title = labels[sortOrder];
    
    displayImageList();
}

// Bulk annotate
async function bulkAnnotate() {
    if (selectedImages.size === 0) return;
    
    if (!confirm(`Annotate ${selectedImages.size} selected images?`)) {
        return;
    }
    
    const progressMsg = document.createElement('div');
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Bulk Annotating...</strong>
                <span>0 / ${selectedImages.size}</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    let completed = 0;
    const imagesToProcess = Array.from(selectedImages);
    
    for (const filename of imagesToProcess) {
        try {
            const response = await fetch(`/api/annotate/${filename}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                completed++;
            }
        } catch (error) {
            console.error(`Error annotating ${filename}:`, error);
        }
        
        progressMsg.querySelector('.floating-progress-text span').textContent = `${completed} / ${imagesToProcess.length}`;
    }
    
    progressMsg.querySelector('.floating-progress-text').innerHTML = `
        <strong>✓ Complete!</strong>
        <span>${completed} images annotated</span>
    `;
    progressMsg.classList.add('success');
    
    setTimeout(() => {
        progressMsg.remove();
        selectedImages.clear();
        loadImages();
        showToast(`Bulk annotated ${completed} images`, 'success');
    }, 1500);
}

// Bulk delete
async function bulkDelete() {
    if (selectedImages.size === 0) return;
    
    if (!confirm(`Delete ${selectedImages.size} selected images? This cannot be undone.`)) {
        return;
    }
    
    const progressMsg = document.createElement('div');
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Deleting...</strong>
                <span>0 / ${selectedImages.size}</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    let completed = 0;
    const imagesToDelete = Array.from(selectedImages);
    
    for (const filename of imagesToDelete) {
        try {
            const response = await fetch(`/api/image/${filename}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                completed++;
            }
        } catch (error) {
            console.error(`Error deleting ${filename}:`, error);
        }
        
        progressMsg.querySelector('.floating-progress-text span').textContent = `${completed} / ${imagesToDelete.length}`;
    }
    
    progressMsg.querySelector('.floating-progress-text').innerHTML = `
        <strong>✓ Complete!</strong>
        <span>${completed} images deleted</span>
    `;
    progressMsg.classList.add('success');
    
    setTimeout(() => {
        progressMsg.remove();
        selectedImages.clear();
        
        if (imagesToDelete.includes(currentImage)) {
            currentImage = null;
            document.getElementById('emptyState').style.display = 'flex';
            document.getElementById('imageViewer').style.display = 'none';
        }
        
        loadImages();
        showToast(`Deleted ${completed} images`, 'success');
    }, 1500);
}

// Select an image
async function selectImage(filename) {
    if (hasUnsavedChanges) {
        if (!confirm('You have unsaved changes. Continue without saving?')) {
            return;
        }
    }
    
    currentImage = filename;
    hasUnsavedChanges = false;
    showingVisualization = false;
    showingPreprocessOverlay = false;
    selectedElementIndex = null;
    currentPreprocess = null;
    currentAnnotation = null;
    
    // Update active state
    document.querySelectorAll('.image-item').forEach(item => {
        item.classList.remove('active');
    });
    const activeItem = document.querySelector(`.image-item[data-filename="${filename}"]`);
    if (activeItem) activeItem.classList.add('active');
    
    // Show image viewer
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('imageViewer').style.display = 'flex';
    document.getElementById('currentImageName').textContent = filename;
    
    // Load image
    const img = document.getElementById('displayImage');
    img.src = `/api/image/${filename}`;
    
    // Update toggle buttons
    document.getElementById('toggleViewBtn').textContent = '👁️ Annotations';
    document.getElementById('togglePreprocessOverlayBtn').textContent = '👁️ Detections';
    
    // Clear detection list
    document.getElementById('detectionList').innerHTML = '<div class="no-data">No detections</div>';
    document.getElementById('detectionCount').textContent = '0';
    
    // Clear canvas
    const canvas = document.getElementById('imageCanvas');
    canvas.style.display = 'none';
    
    // Wait for image to load
    img.onload = async () => {
        // Load annotation if exists
        try {
            const response = await fetch(`/api/annotation/${filename}`);
            if (response.ok) {
                currentAnnotation = await response.json();
                displayAnnotation();
            } else {
                currentAnnotation = null;
                displayNoAnnotation();
            }
        } catch (error) {
            currentAnnotation = null;
            displayNoAnnotation();
        }
        
        updateSaveButton();
    };
}

// Display annotation
function displayAnnotation() {
    const elementList = document.getElementById('elementList');
    const elementCount = document.getElementById('elementCount');
    
    if (!currentAnnotation || !currentAnnotation.element || currentAnnotation.element.length === 0) {
        displayNoAnnotation();
        return;
    }
    
    elementCount.textContent = currentAnnotation.element.length;
    
    elementList.innerHTML = currentAnnotation.element.map((elem, index) => `
        <div class="element-card" data-index="${index}" onclick="selectElement(${index})">
            <div class="element-header">
                <span class="element-number">#${index + 1}</span>
                <button class="btn btn-small btn-danger" onclick="deleteElement(${index}, event)">
                    🗑️
                </button>
            </div>
            <div class="element-instruction">${elem.instruction}</div>
            <div class="element-details">
                <div class="detail-item">
                    <span class="detail-label">BBox:</span>
                    <span class="detail-value">[${elem.bbox.map(v => Math.round(v)).join(', ')}]</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Point:</span>
                    <span class="detail-value">[${elem.point.map(v => Math.round(v)).join(', ')}]</span>
                </div>
            </div>
        </div>
    `).join('');
    
    drawAnnotations();
}

// Display no annotation
function displayNoAnnotation() {
    const elementList = document.getElementById('elementList');
    const elementCount = document.getElementById('elementCount');
    
    elementCount.textContent = '0';
    elementList.innerHTML = '<div class="no-data">No annotations. Click "Generate".</div>';
}

// Display detections
function displayDetections() {
    const detectionList = document.getElementById('detectionList');
    const detectionCount = document.getElementById('detectionCount');
    
    if (!currentPreprocess || !currentPreprocess.element || currentPreprocess.element.length === 0) {
        detectionList.innerHTML = '<div class="no-data">No detections</div>';
        detectionCount.textContent = '0';
        return;
    }
    
    detectionCount.textContent = currentPreprocess.element.length;
    
    detectionList.innerHTML = currentPreprocess.element.map((elem, index) => `
        <div class="detection-item" data-index="${index}">
            <div class="detection-info">
                <div class="detection-bbox">[${elem.bbox.map(v => Math.round(v)).join(', ')}]</div>
            </div>
            <div class="detection-actions">
                <button class="btn-icon btn-danger" onclick="removeDetection(${index}, event)" title="Remove">
                    ✕
                </button>
            </div>
        </div>
    `).join('');
}

// Remove detection
function removeDetection(index, event) {
    event.stopPropagation();
    if (!currentPreprocess || !currentPreprocess.element) return;
    
    currentPreprocess.element.splice(index, 1);
    displayDetections();
    drawAnnotations();
    showToast('Detection removed', 'success');
}

// Draw annotations on canvas
function drawAnnotations() {
    const img = document.getElementById('displayImage');
    const canvas = document.getElementById('imageCanvas');
    
    if ((!currentAnnotation || !showingVisualization) && (!currentPreprocess || !showingPreprocessOverlay)) {
        canvas.style.display = 'none';
        return;
    }
    
    if (!img.complete || img.naturalWidth === 0) {
        setTimeout(drawAnnotations, 100);
        return;
    }
    
    canvas.style.display = 'block';
    const displayWidth = img.width;
    const displayHeight = img.height;
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#8000FF'];
    
    const elements = [];
    const baseSize = currentAnnotation ? currentAnnotation.img_size : (currentPreprocess ? currentPreprocess.img_size : [displayWidth, displayHeight]);
    const scaleX = displayWidth / baseSize[0];
    const scaleY = displayHeight / baseSize[1];
    
    // Draw preprocess overlay
    if (currentPreprocess && showingPreprocessOverlay) {
        currentPreprocess.element.forEach((elem, index) => {
            const color = '#FF6B00'; // Orange for detections
            const bbox = elem.bbox;
            const point = elem.point;
            
            const x1 = bbox[0] * scaleX;
            const y1 = bbox[1] * scaleY;
            const x2 = bbox[2] * scaleX;
            const y2 = bbox[3] * scaleY;
            const px = point[0] * scaleX;
            const py = point[1] * scaleY;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);
            
            ctx.fillStyle = color + '20';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
    
    // Draw annotations overlay
    if (currentAnnotation && showingVisualization) {
        currentAnnotation.element.forEach((elem, index) => {
            const color = colors[index % colors.length];
            const isSelected = index === selectedElementIndex;
            const bbox = elem.bbox;
            const point = elem.point;
            
            const x1 = bbox[0] * scaleX;
            const y1 = bbox[1] * scaleY;
            const x2 = bbox[2] * scaleX;
            const y2 = bbox[3] * scaleY;
            const px = point[0] * scaleX;
            const py = point[1] * scaleY;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            ctx.fillStyle = color + '20';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py, isSelected ? 5 : 4, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - 22, 30, 22);
            ctx.fillStyle = 'white';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`#${index + 1}`, x1 + 5, y1 - 7);
        });
    }
}

// Select an element
function selectElement(index) {
    selectedElementIndex = index;
    
    document.querySelectorAll('.element-card').forEach(card => {
        card.classList.remove('selected');
    });
    const card = document.querySelector(`.element-card[data-index="${index}"]`);
    if (card) card.classList.add('selected');
    
    drawAnnotations();
}

// Delete an element
async function deleteElement(index, event) {
    event.stopPropagation();
    
    if (!confirm('Delete this element?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/annotation/${currentImage}/element/${index}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const data = await response.json();
            currentAnnotation = data.annotation;
            selectedElementIndex = null;
            displayAnnotation();
            showToast('Element deleted', 'success');
            loadImages();
        } else {
            throw new Error('Failed to delete element');
        }
    } catch (error) {
        console.error('Error deleting element:', error);
        showToast('Error deleting element', 'error');
    }
}

// Delete current image
async function deleteCurrentImage() {
    if (!currentImage) return;
    
    if (!confirm(`Delete "${currentImage}"? This cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/image/${currentImage}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('Image deleted', 'success');
            currentImage = null;
            document.getElementById('emptyState').style.display = 'flex';
            document.getElementById('imageViewer').style.display = 'none';
            loadImages();
        } else {
            throw new Error('Failed to delete image');
        }
    } catch (error) {
        console.error('Error deleting image:', error);
        showToast('Error deleting image', 'error');
    }
}

// Generate annotation
async function generateAnnotation() {
    if (!currentImage) return;
    
    const btn = document.getElementById('annotateBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';
    
    // Show floating progress
    const progressMsg = document.createElement('div');
    progressMsg.id = 'annotateProgress';
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Generating Annotations...</strong>
                <span>OmniParser + GPT-5 working</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    try {
        const response = await fetch(`/api/annotate/${currentImage}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            currentAnnotation = await response.json();
            
            progressMsg.querySelector('.floating-progress-text').innerHTML = `
                <strong>✓ Complete!</strong>
                <span>Generated ${currentAnnotation.element.length} elements</span>
            `;
            progressMsg.classList.add('success');
            
            setTimeout(() => {
                progressMsg.remove();
                displayAnnotation();
                showToast('Annotation generated', 'success');
                loadImages();
            }, 800);
        } else {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to generate annotation');
        }
    } catch (error) {
        console.error('Error generating annotation:', error);
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✗ Failed</strong>
            <span>Error generating annotation</span>
        `;
        progressMsg.classList.add('error');
        setTimeout(() => progressMsg.remove(), 2000);
        showToast('Error generating annotation', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🤖 Generate';
    }
}

// Run preprocessing
async function runPreprocess() {
    if (!currentImage) return;
    
    const progressMsg = document.createElement('div');
    progressMsg.id = 'preprocessProgress';
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>OmniParser Processing...</strong>
                <span>Detecting UI elements</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    try {
        const resp = await fetch(`/api/preprocess/${currentImage}`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({}) 
        });
        if (!resp.ok) throw new Error('Preprocess failed');
        currentPreprocess = await resp.json();
        
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✓ Complete!</strong>
            <span>Found ${currentPreprocess.element.length} elements</span>
        `;
        progressMsg.classList.add('success');
        
        setTimeout(() => {
            progressMsg.remove();
            showToast(`Preprocess found ${currentPreprocess.element.length} elements`, 'success');
        }, 800);
        
        showingPreprocessOverlay = true;
        const btn2 = document.getElementById('togglePreprocessOverlayBtn');
        if (btn2) btn2.textContent = '👁️ Hide Detections';
        
        displayDetections();
        drawAnnotations();
    } catch (e) {
        console.error(e);
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✗ Failed</strong>
            <span>Error running preprocess</span>
        `;
        progressMsg.classList.add('error');
        setTimeout(() => progressMsg.remove(), 2000);
        showToast('Error running preprocess', 'error');
    }
}

// Toggle visualization
function toggleVisualization() {
    if (!currentImage || !currentAnnotation) return;
    
    showingVisualization = !showingVisualization;
    const btn = document.getElementById('toggleViewBtn');
    btn.textContent = showingVisualization ? '👁️ Hide Annotations' : '👁️ Show Annotations';
    drawAnnotations();
}

// Toggle preprocess overlay
function togglePreprocessOverlay() {
    if (!currentImage) return;
    showingPreprocessOverlay = !showingPreprocessOverlay;
    const btn2 = document.getElementById('togglePreprocessOverlayBtn');
    btn2.textContent = showingPreprocessOverlay ? '👁️ Hide Detections' : '👁️ Show Detections';
    drawAnnotations();
}

// Save annotation
async function saveAnnotation() {
    if (!currentImage || !currentAnnotation) return;
    
    const btn = document.getElementById('saveBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Saving...';
    
    try {
        const response = await fetch(`/api/annotation/${currentImage}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentAnnotation)
        });
        
        if (response.ok) {
            hasUnsavedChanges = false;
            updateSaveButton();
            showToast('Annotation saved', 'success');
        } else {
            throw new Error('Failed to save annotation');
        }
    } catch (error) {
        console.error('Error saving annotation:', error);
        showToast('Error saving annotation', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 Save';
    }
}

// Update save button visibility
function updateSaveButton() {
    const saveBtn = document.getElementById('saveBtn');
    saveBtn.style.display = hasUnsavedChanges ? 'inline-block' : 'none';
}

// Handle file selection
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
        uploadFiles(files);
    }
}

// Handle folder selection
function handleFolderSelect(event) {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
        uploadFiles(files, true);
    }
}

// Upload files
async function uploadFiles(files, isFolder = false) {
    const modal = isFolder ? document.getElementById('folderModal') : document.getElementById('uploadModal');
    const uploadArea = isFolder ? document.getElementById('folderArea') : document.getElementById('uploadArea');
    const uploadProgress = isFolder ? document.getElementById('folderProgress') : document.getElementById('uploadProgress');
    const uploadStatus = isFolder ? document.getElementById('folderStatus') : document.getElementById('uploadStatus');
    const progressFill = isFolder ? document.getElementById('folderProgressFill') : document.getElementById('progressFill');
    
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';
    
    let completed = 0;
    const total = files.length;
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                console.error(`Failed to upload ${file.name}`);
            }
        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
        }
        
        completed++;
        const percent = (completed / total) * 100;
        progressFill.style.width = `${percent}%`;
        uploadStatus.textContent = `Uploading... ${completed}/${total}`;
    }
    
    uploadStatus.textContent = 'Upload complete!';
    
    setTimeout(() => {
        modal.classList.remove('active');
        uploadArea.style.display = 'block';
        uploadProgress.style.display = 'none';
        progressFill.style.width = '0%';
        
        // Clear file inputs
        if (isFolder) {
            document.getElementById('folderInput').value = '';
        } else {
            document.getElementById('fileInput').value = '';
        }
        
        loadImages();
        showToast(`Uploaded ${completed} image(s)`, 'success');
    }, 1000);
}

// Batch annotation
async function startBatchAnnotation() {
    const btn = document.getElementById('startBatchBtn');
    const batchProgress = document.getElementById('batchProgress');
    const batchStatus = document.getElementById('batchStatus');
    const progressFill = document.getElementById('batchProgressFill');
    const force = document.getElementById('forceAnnotate').checked;
    
    btn.disabled = true;
    batchProgress.style.display = 'block';
    
    try {
        batchStatus.textContent = 'Processing images...';
        progressFill.style.width = '50%';
        
        const response = await fetch('/api/batch-annotate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ force: force })
        });
        
        if (response.ok) {
            const data = await response.json();
            progressFill.style.width = '100%';
            
            const success = data.results.filter(r => r.status === 'success').length;
            const skipped = data.results.filter(r => r.status === 'skipped').length;
            const errors = data.results.filter(r => r.status === 'error').length;
            
            batchStatus.textContent = `Complete! ${success} annotated, ${skipped} skipped, ${errors} errors`;
            
            setTimeout(() => {
                document.getElementById('batchModal').classList.remove('active');
                batchProgress.style.display = 'none';
                progressFill.style.width = '0%';
                btn.disabled = false;
                
                loadImages();
                showToast(`Batch complete: ${success} images annotated`, 'success');
            }, 2000);
        } else {
            throw new Error('Batch annotation failed');
        }
    } catch (error) {
        console.error('Error in batch annotation:', error);
        batchStatus.textContent = 'Batch annotation failed!';
        showToast('Error in batch annotation', 'error');
        btn.disabled = false;
        
        setTimeout(() => {
            batchProgress.style.display = 'none';
            progressFill.style.width = '0%';
        }, 2000);
    }
}

// Validate pasted JSON
function validatePastedJson() {
    const jsonInput = document.getElementById('jsonInput');
    const jsonError = document.getElementById('jsonError');
    
    try {
        const jsonText = jsonInput.value.trim();
        
        if (!jsonText) {
            jsonError.textContent = 'Please paste some JSON data';
            jsonError.style.display = 'block';
            return false;
        }
        
        const data = JSON.parse(jsonText);
        
        if (!data.img_size || !Array.isArray(data.img_size) || data.img_size.length !== 2) {
            jsonError.textContent = 'Invalid format: img_size must be [width, height]';
            jsonError.style.display = 'block';
            return false;
        }
        
        if (!data.element || !Array.isArray(data.element)) {
            jsonError.textContent = 'Invalid format: element must be an array';
            jsonError.style.display = 'block';
            return false;
        }
        
        for (let i = 0; i < data.element.length; i++) {
            const elem = data.element[i];
            
            if (!elem.instruction || typeof elem.instruction !== 'string') {
                jsonError.textContent = `Element ${i}: instruction must be a string`;
                jsonError.style.display = 'block';
                return false;
            }
            
            if (!elem.bbox || !Array.isArray(elem.bbox) || elem.bbox.length !== 4) {
                jsonError.textContent = `Element ${i}: bbox must be [x1, y1, x2, y2]`;
                jsonError.style.display = 'block';
                return false;
            }
            
            if (!elem.bbox.every(v => typeof v === 'number' && !isNaN(v))) {
                jsonError.textContent = `Element ${i}: bbox values must be numbers`;
                jsonError.style.display = 'block';
                return false;
            }
            
            if (!elem.point || !Array.isArray(elem.point) || elem.point.length !== 2) {
                jsonError.textContent = `Element ${i}: point must be [x, y]`;
                jsonError.style.display = 'block';
                return false;
            }
            
            if (!elem.point.every(v => typeof v === 'number' && !isNaN(v))) {
                jsonError.textContent = `Element ${i}: point values must be numbers`;
                jsonError.style.display = 'block';
                return false;
            }
        }
        
        jsonError.style.display = 'none';
        showToast('JSON is valid!', 'success');
        return true;
        
    } catch (error) {
        jsonError.textContent = `JSON Parse Error: ${error.message}`;
        jsonError.style.display = 'block';
        return false;
    }
}

// Apply pasted JSON
async function applyPastedJson() {
    if (!currentImage) {
        showToast('Please select an image first', 'error');
        return;
    }
    
    if (!validatePastedJson()) {
        return;
    }
    
    const jsonInput = document.getElementById('jsonInput');
    const btn = document.getElementById('applyJsonBtn');
    
    btn.disabled = true;
    btn.textContent = '⏳ Applying...';
    
    try {
        const annotation = JSON.parse(jsonInput.value.trim());
        
        const response = await fetch(`/api/annotation/${currentImage}/paste`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ annotation: annotation })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentAnnotation = data.annotation;
            displayAnnotation();
            
            document.getElementById('pasteJsonModal').classList.remove('active');
            
            showToast('Annotation applied', 'success');
            loadImages();
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Failed to apply annotation');
        }
    } catch (error) {
        console.error('Error applying annotation:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '📋 Apply';
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Handle image load for canvas sizing
document.getElementById('displayImage').addEventListener('load', function() {
    if ((currentAnnotation && showingVisualization) || (currentPreprocess && showingPreprocessOverlay)) {
        drawAnnotations();
    }
});
