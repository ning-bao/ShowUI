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
let lastSelectedIndex = -1;
let filteredImages = [];
let viewMode = 'list'; // 'list' or 'folder'
let expandedFolders = new Set();
let batchEventSource = null;
let allFolders = [];
let selectMode = false; // Select mode for easier image selection

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    // Restore state from localStorage
    const savedVisualization = localStorage.getItem('showingVisualization');
    const savedPreprocess = localStorage.getItem('showingPreprocessOverlay');
    
    if (savedVisualization !== null) {
        showingVisualization = savedVisualization === 'true';
    }
    if (savedPreprocess !== null) {
        showingPreprocessOverlay = savedPreprocess === 'true';
    }
    
    loadImages();
    setupEventListeners();
    setupSidebarResizer();
});

// Setup event listeners
function setupEventListeners() {
    // Upload modals
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadModal = document.getElementById('uploadModal');
    const uploadFolderBtn = document.getElementById('uploadFolderBtn');
    const folderModal = document.getElementById('folderModal');
    const createFolderBtn = document.getElementById('createFolderBtn');
    
    if (uploadBtn && uploadModal) uploadBtn.addEventListener('click', () => uploadModal.classList.add('active'));
    if (uploadFolderBtn && folderModal) uploadFolderBtn.addEventListener('click', () => folderModal.classList.add('active'));
    if (createFolderBtn) createFolderBtn.addEventListener('click', createFolderPrompt);
    
    // Close modals
    document.querySelectorAll('.modal .close').forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            const modal = this.closest('.modal');
            if (!modal) return;
            modal.classList.remove('active');
            if (modal.id === 'batchModal' && batchEventSource) {
                try { batchEventSource.close(); } catch (_) {}
                batchEventSource = null;
            }
        });
    });
    
    // Bulk actions
    const bulkAnnotateBtn = document.getElementById('bulkAnnotateBtn');
    const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
    const exportBtn = document.getElementById('exportBtn');
    
    if (bulkAnnotateBtn) bulkAnnotateBtn.addEventListener('click', bulkAnnotate);
    if (bulkDeleteBtn) bulkDeleteBtn.addEventListener('click', bulkDelete);
    if (exportBtn) exportBtn.addEventListener('click', exportDataset);

    // Batch annotate (all images) modal start
    const startBatchBtn = document.getElementById('startBatchBtn');
    if (startBatchBtn) startBatchBtn.addEventListener('click', startBatchAnnotation);
    
    // Sort select
    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) sortSelect.addEventListener('change', handleSortChange);
    
    // File inputs
    const fileInput = document.getElementById('fileInput');
    const folderInput = document.getElementById('folderInput');
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    if (folderInput) folderInput.addEventListener('change', handleFolderSelect);
    
    // Drag and drop for single files
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
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
    }
    
    // Action buttons
    const annotateBtn = document.getElementById('annotateBtn');
    const preprocessBtn = document.getElementById('preprocessBtn');
    const toggleViewBtn = document.getElementById('toggleViewBtn');
    const togglePreprocessOverlayBtn = document.getElementById('togglePreprocessOverlayBtn');
    const saveBtn = document.getElementById('saveBtn');
    const deleteImageBtn = document.getElementById('deleteImageBtn');
    
    if (annotateBtn) annotateBtn.addEventListener('click', generateAnnotation);
    if (preprocessBtn) preprocessBtn.addEventListener('click', runPreprocess);
    if (toggleViewBtn) toggleViewBtn.addEventListener('click', toggleVisualization);
    if (togglePreprocessOverlayBtn) togglePreprocessOverlayBtn.addEventListener('click', togglePreprocessOverlay);
    if (saveBtn) saveBtn.addEventListener('click', saveAnnotation);
    if (deleteImageBtn) deleteImageBtn.addEventListener('click', deleteCurrentImage);
    
    // Paste JSON
    const pasteJsonBtn = document.getElementById('pasteJsonBtn');
    const pasteJsonModal = document.getElementById('pasteJsonModal');
    const validateJsonBtn = document.getElementById('validateJsonBtn');
    const applyJsonBtn = document.getElementById('applyJsonBtn');
    
    if (pasteJsonBtn && pasteJsonModal) {
        pasteJsonBtn.addEventListener('click', () => {
            if (!currentImage) {
                showToast('Please select an image first', 'error');
                return;
            }
            const jsonInput = document.getElementById('jsonInput');
            const jsonError = document.getElementById('jsonError');
            if (jsonInput) jsonInput.value = '';
            if (jsonError) jsonError.style.display = 'none';
            pasteJsonModal.classList.add('active');
        });
    }
    if (validateJsonBtn) validateJsonBtn.addEventListener('click', validatePastedJson);
    if (applyJsonBtn) applyJsonBtn.addEventListener('click', applyPastedJson);
    
    // Search and filter
    const searchInput = document.getElementById('searchInput');
    const statusFilter = document.getElementById('statusFilter');
    if (searchInput) searchInput.addEventListener('input', filterImages);
    if (statusFilter) statusFilter.addEventListener('change', filterImages);
    
    // Select all checkbox
    const selectAllCheckbox = document.getElementById('selectAllCheckbox');
    if (selectAllCheckbox) selectAllCheckbox.addEventListener('change', handleSelectAll);
    
    // Toggle view mode
    const toggleViewModeBtn = document.getElementById('toggleViewModeBtn');
    if (toggleViewModeBtn) toggleViewModeBtn.addEventListener('click', toggleViewMode);
    
    // Toggle select mode
    const toggleSelectModeBtn = document.getElementById('toggleSelectModeBtn');
    if (toggleSelectModeBtn) toggleSelectModeBtn.addEventListener('click', toggleSelectMode);
    
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
async function createFolderPrompt() {
    const name = prompt('Enter new folder name:');
    if (!name) return;
    try {
        const res = await fetch('/api/folder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (!res.ok) throw new Error('Failed to create folder');
        await loadImages();
        showToast('Folder created', 'success');
    } catch (e) {
        showToast('Failed to create folder', 'error');
    }
}

// Sidebar resizer
function setupSidebarResizer() {
    const sidebar = document.querySelector('.sidebar');
    const resizer = document.getElementById('sidebarResizer');
    if (!sidebar || !resizer) return;
    let isDragging = false;

    const minWidth = 200;
    const maxWidth = 600;

    resizer.addEventListener('mousedown', (e) => {
        isDragging = true;
        document.body.style.cursor = 'col-resize';
        e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const newWidth = Math.min(Math.max(e.clientX, minWidth), maxWidth);
        sidebar.style.width = newWidth + 'px';
    });

    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        document.body.style.cursor = '';
    });
}

// Load images from server
async function loadImages() {
    try {
        // request larger page size to support bigger datasets without pagination UI changes
        const response = await fetch('/api/images?page=1&page_size=5000');
        const data = await response.json();
        allImages = data.images || [];

        // Try to load folder list (to render empty folders in folder view)
        try {
            const foldersResp = await fetch('/api/folders');
            if (foldersResp.ok) {
                const foldersData = await foldersResp.json();
                allFolders = Array.isArray(foldersData.folders) ? foldersData.folders : [];
            } else {
                allFolders = [];
            }
        } catch (_) {
            allFolders = [];
        }
        displayImageList();
    } catch (error) {
        console.error('Error loading images:', error);
        showToast('Error loading images', 'error');
    }
}

// Display image list with filters
function displayImageList() {
    const imageList = document.getElementById('imageList');
    if (!imageList) return;
    
    const searchInput = document.getElementById('searchInput');
    const statusFilterEl = document.getElementById('statusFilter');
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const statusFilter = statusFilterEl ? statusFilterEl.value : 'all';
    
    filteredImages = allImages.filter(img => {
        if (searchTerm && !img.filename.toLowerCase().includes(searchTerm)) {
            return false;
        }
        
        if (statusFilter === 'annotated' && !img.has_annotation) return false;
        if (statusFilter === 'not-annotated' && img.has_annotation) return false;
        
        return true;
    });
    
    // Sort
    filteredImages.sort((a, b) => {
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
    
    if (filteredImages.length === 0) {
        imageList.innerHTML = '<div class="no-data">No images found</div>';
        // Ensure bulk UI updates still occur even when nothing is listed
        updateBulkActionsVisibility();
        updateSelectAllCheckbox();
        return;
    }
    
    if (viewMode === 'folder') {
        displayFolderView();
    } else {
        displayListView();
    }
    
    updateBulkActionsVisibility();
    updateSelectAllCheckbox();
}

// Display list view
function displayListView() {
    const imageList = document.getElementById('imageList');
    
    imageList.innerHTML = filteredImages.map((img, index) => {
        const escapedFilename = img.filename.replace(/'/g, "\\'");
        const isSelected = selectedImages.has(img.filename);
        return `
            <div class="image-item ${isSelected ? 'selected' : ''}" 
                 data-filename="${img.filename}" 
                 data-index="${index}"
                 draggable="${isSelected ? 'true' : 'false'}">
                <div class="image-item-checkbox">
                    <input type="checkbox" 
                           data-filename="${img.filename}"
                           data-index="${index}"
                           ${isSelected ? 'checked' : ''}
                           onclick="toggleImageSelection('${escapedFilename}', ${index}, event)">
                </div>
                <div class="image-item-thumb" onclick="handleImageClick('${escapedFilename}', ${index}, event)">
                    <img src="/api/image/${img.filename}" alt="${img.filename}" />
                </div>
                <div class="image-item-info" onclick="handleImageClick('${escapedFilename}', ${index}, event)">
                    <div class="image-item-name" title="${img.filename}">${img.filename}</div>
                    <div class="image-item-status">
                        <span class="status-badge ${img.has_annotation ? 'annotated' : 'not-annotated'}">
                            ${img.has_annotation ? '✓ Annotated' : '✗ Not annotated'}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    // Setup drag events for selected images
    setupImageDragEvents();
}

// Build folder tree structure
function buildFolderTree() {
    const tree = {};
    
    filteredImages.forEach((img, index) => {
        const parts = img.filename.split('/');
        
        if (parts.length === 1) {
            // Root level file
            if (!tree['__root__']) {
                tree['__root__'] = { folders: {}, files: [] };
            }
            tree['__root__'].files.push({ ...img, index });
        } else {
            // Nested file
            let current = tree;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const folder = parts[i];
                if (!current[folder]) {
                    current[folder] = { folders: {}, files: [] };
                }
                current = current[folder].folders;
            }
            
            const parentFolder = parts[parts.length - 2];
            const parentObj = getParentFolder(tree, parts.slice(0, -1));
            parentObj.files.push({ ...img, index, displayName: parts[parts.length - 1] });
        }
    });
    
    return tree;
}

// Ensure that known folders (possibly empty) exist in the tree so users can expand them
function ensureFoldersInTree(tree, folders) {
    if (!Array.isArray(folders) || folders.length === 0) return;
    for (const folderPath of folders) {
        if (typeof folderPath !== 'string' || folderPath.trim() === '') continue;
        const parts = folderPath.split('/');
        let current = tree;
        for (let i = 0; i < parts.length; i++) {
            const folder = parts[i];
            if (!current[folder]) {
                current[folder] = { folders: {}, files: [] };
            }
            if (i === parts.length - 1) {
                // last segment ensures existence; next loop handles children via .folders
                break;
            }
            current = current[folder].folders;
        }
    }
}

// Get parent folder object
function getParentFolder(tree, path) {
    let current = tree;
    
    for (const folder of path) {
        if (!current[folder]) {
            current[folder] = { folders: {}, files: [] };
        }
        if (folder === path[path.length - 1]) {
            return current[folder];
        }
        current = current[folder].folders;
    }
    
    return current;
}

// Display folder view
function displayFolderView() {
    const imageList = document.getElementById('imageList');
    const tree = buildFolderTree();
    ensureFoldersInTree(tree, allFolders);
    
    const html = renderFolderTree(tree, '');
    if (!html && Object.keys(tree).length === 0) {
        imageList.innerHTML = '<div class="no-data">No folders or images</div>';
    } else if (!html && tree['__root__']) {
        // Only root files, render them
        imageList.innerHTML = renderFiles(tree['__root__'].files);
    } else {
        imageList.innerHTML = html;
    }
    
    // Setup drag and drop for folders after rendering
    setupFolderDragAndDrop();
}

// Render folder tree recursively
function renderFolderTree(tree, path) {
    let html = '';
    
    // Sort folders and files
    const folders = Object.keys(tree).filter(k => k !== '__root__' && tree[k] && typeof tree[k] === 'object' && 'folders' in tree[k]).sort();
    
    for (const folderName of folders) {
        const folderData = tree[folderName];
        const folderPath = path ? `${path}/${folderName}` : folderName;
        const isExpanded = expandedFolders.has(folderPath);
        const fileCount = countFilesInFolder(folderData);
        const escapedPath = folderPath.replace(/'/g, "\\'");
        
            html += `
                <div class="folder-item">
                    <div class="folder-header" data-folder-path="${folderPath}" onclick="toggleFolder('${escapedPath}')">
                        <span class="folder-toggle ${isExpanded ? 'expanded' : ''}">▶</span>
                        <span class="folder-icon">📁</span>
                        <span class="folder-name">${folderName}</span>
                        <span class="folder-count">${fileCount}</span>
                    </div>
                    <div class="folder-children ${isExpanded ? 'expanded' : ''}">
                        ${renderFolderTree(folderData.folders, folderPath)}
                        ${renderFiles(folderData.files)}
                    </div>
                </div>
            `;
    }
    
    // Render root files at the end
    if (tree['__root__'] && tree['__root__'].files && tree['__root__'].files.length > 0) {
        html += renderFiles(tree['__root__'].files);
    }
    
    return html;
}

// Render files
function renderFiles(files) {
    if (!files || files.length === 0) return '';
    
    return files.map(img => {
        const displayName = img.displayName || img.filename;
        const escapedFilename = img.filename.replace(/'/g, "\\'");
        const isSelected = selectedImages.has(img.filename);
        return `
            <div class="image-item ${isSelected ? 'selected' : ''}" 
                 data-filename="${img.filename}" 
                 data-index="${img.index}"
                 draggable="${isSelected ? 'true' : 'false'}">
                <div class="image-item-checkbox">
                    <input type="checkbox" 
                           data-filename="${img.filename}"
                           data-index="${img.index}"
                           ${isSelected ? 'checked' : ''}
                           onclick="toggleImageSelection('${escapedFilename}', ${img.index}, event)">
                </div>
                <div class="image-item-thumb" onclick="handleImageClick('${escapedFilename}', ${img.index}, event)">
                    <img src="/api/image/${img.filename}" alt="${img.filename}" />
                </div>
                <div class="image-item-info" onclick="handleImageClick('${escapedFilename}', ${img.index}, event)">
                    <div class="image-item-name" title="${img.filename}">${displayName}</div>
                    <div class="image-item-status">
                        <span class="status-badge ${img.has_annotation ? 'annotated' : 'not-annotated'}">
                            ${img.has_annotation ? '✓ Annotated' : '✗ Not annotated'}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Count files in folder recursively
function countFilesInFolder(folderData) {
    let count = folderData.files.length;
    
    for (const subfolder in folderData.folders) {
        count += countFilesInFolder(folderData.folders[subfolder]);
    }
    
    return count;
}

// Toggle folder expansion
function toggleFolder(path) {
    if (expandedFolders.has(path)) {
        expandedFolders.delete(path);
    } else {
        expandedFolders.add(path);
    }
    displayImageList();
}

// Toggle view mode
function toggleViewMode() {
    viewMode = viewMode === 'list' ? 'folder' : 'list';
    const btn = document.getElementById('toggleViewModeBtn');
    if (btn) {
        if (viewMode === 'folder') {
            btn.title = 'List View';
            btn.textContent = '📋';
        } else {
            btn.title = 'Folder View';
            btn.textContent = '🗂️';
        }
    }
    displayImageList();
}

// Toggle select mode
function toggleSelectMode() {
    selectMode = !selectMode;
    const btn = document.getElementById('toggleSelectModeBtn');
    if (btn) {
        if (selectMode) {
            btn.classList.add('active');
            btn.title = 'Exit Select Mode';
            btn.style.backgroundColor = 'var(--primary-color)';
            btn.style.color = 'white';
            showToast('Select Mode: Click images to select, Shift+Click for range', 'info');
        } else {
            btn.classList.remove('active');
            btn.title = 'Toggle Select Mode';
            btn.style.backgroundColor = '';
            btn.style.color = '';
        }
    }
    // Update image items to make them draggable in select mode
    displayImageList();
}

// Filter images
function filterImages() {
    displayImageList();
}

// Toggle image selection
function toggleImageSelection(filename, index, event) {
    event.stopPropagation();
    
    // Check for Shift key (range selection)
    if (event.shiftKey && lastSelectedIndex !== -1) {
        const start = Math.min(lastSelectedIndex, index);
        const end = Math.max(lastSelectedIndex, index);
        
        // Select all images in range
        for (let i = start; i <= end; i++) {
            if (i < filteredImages.length) {
                selectedImages.add(filteredImages[i].filename);
            }
        }
        updateImageSelectionUI();
    }
    // Check for Ctrl/Cmd key (multi-select)
    else if (event.ctrlKey || event.metaKey) {
        if (selectedImages.has(filename)) {
            selectedImages.delete(filename);
        } else {
            selectedImages.add(filename);
        }
        lastSelectedIndex = index;
        updateSingleImageUI(filename);
    }
    // Normal single selection toggle
    else {
        if (selectedImages.has(filename)) {
            selectedImages.delete(filename);
        } else {
            selectedImages.add(filename);
        }
        lastSelectedIndex = index;
        updateSingleImageUI(filename);
    }
    
    updateBulkActionsVisibility();
    updateSelectAllCheckbox();
}

// Handle select all checkbox
function handleSelectAll(event) {
    if (event.target.checked) {
        // Select all filtered images
        filteredImages.forEach(img => {
            selectedImages.add(img.filename);
        });
    } else {
        // Deselect all filtered images
        filteredImages.forEach(img => {
            selectedImages.delete(img.filename);
        });
    }
    updateImageSelectionUI();
    updateBulkActionsVisibility();
    updateSelectAllCheckbox();
}

// Update select all checkbox state
function updateSelectAllCheckbox() {
    const selectAllCheckbox = document.getElementById('selectAllCheckbox');
    if (!selectAllCheckbox) return; // Element not found, skip update
    
    if (filteredImages.length === 0) {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = false;
    } else {
        const selectedCount = filteredImages.filter(img => selectedImages.has(img.filename)).length;
        if (selectedCount === 0) {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = false;
        } else if (selectedCount === filteredImages.length) {
            selectAllCheckbox.checked = true;
            selectAllCheckbox.indeterminate = false;
        } else {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = true;
        }
    }
}

// Update bulk actions visibility
function updateBulkActionsVisibility() {
    const count = selectedImages.size;
    const selectedCountEl = document.getElementById('selectedCount');
    const bulkActionButtons = document.getElementById('bulkActionButtons');
    
    if (!selectedCountEl || !bulkActionButtons) return; // Elements not found
    
    if (count > 0) {
        selectedCountEl.textContent = `${count} selected`;
        bulkActionButtons.style.display = 'flex';
    } else {
        bulkActionButtons.style.display = 'none';
    }
}

// Cycle sort order
function handleSortChange(event) {
    sortOrder = event.target.value;
    displayImageList();
}

// Bulk annotate
async function bulkAnnotate() {
    if (selectedImages.size === 0) return;
    
    if (!confirm(`Annotate ${selectedImages.size} selected images?`)) {
        return;
    }

    // Use async batch API with SSE for the selected filenames
    const progressMsg = document.createElement('div');
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Bulk Annotating...</strong>
                <span>Queued...</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);

    try {
        const filenames = Array.from(selectedImages);
        const startResp = await fetch('/api/batch-annotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames })
        });
        if (!startResp.ok) throw new Error('Failed to start batch');
        const { job_id, total } = await startResp.json();
        let completed = 0, success = 0, skipped = 0, errors = 0;

        if (batchEventSource) { try { batchEventSource.close(); } catch(_){} batchEventSource = null; }
        batchEventSource = new EventSource(`/api/batch-annotate/stream/${job_id}`);

        batchEventSource.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                if (data.type === 'image_done') {
                    completed = data.completed;
                    if (data.status === 'success') success++; else if (data.status === 'skipped') skipped++; else errors++;
                    progressMsg.querySelector('.floating-progress-text span').textContent = `${completed}/${total} — ✓ ${success}, ○ ${skipped}, ✗ ${errors}`;
                    if (data.filename && data.status === 'success') {
                        const idx = allImages.findIndex(i => i.filename === data.filename);
                        if (idx !== -1) {
                            allImages[idx].has_annotation = true;
                            displayImageList();
                        }
                    }
                } else if (data.type === 'complete') {
                    progressMsg.querySelector('.floating-progress-text').innerHTML = `
                        <strong>✓ Complete!</strong>
                        <span>✓ ${data.summary.success}, ○ ${data.summary.skipped}, ✗ ${data.summary.errors}</span>
                    `;
                    progressMsg.classList.add('success');
                }
            } catch (_) {}
        };

        batchEventSource.addEventListener('end', () => {
            try { batchEventSource.close(); } catch(_){}
            batchEventSource = null;
            setTimeout(() => {
                progressMsg.remove();
                selectedImages.clear();
                loadImages();
                showToast('Bulk annotation complete', 'success');
            }, 1200);
        });

        batchEventSource.onerror = () => {
            try { batchEventSource.close(); } catch(_){}
            batchEventSource = null;
            progressMsg.querySelector('.floating-progress-text').innerHTML = `
                <strong>✗ Failed</strong>
                <span>Connection lost</span>
            `;
            progressMsg.classList.add('error');
            setTimeout(() => progressMsg.remove(), 1500);
        };
    } catch (e) {
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✗ Failed</strong>
            <span>${e.message}</span>
        `;
        progressMsg.classList.add('error');
        setTimeout(() => progressMsg.remove(), 1500);
    }
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
    
    // Update toggle buttons based on saved state
    const btn = document.getElementById('toggleViewBtn');
    const btn2 = document.getElementById('togglePreprocessOverlayBtn');
    btn.textContent = showingVisualization ? '👁️ Hide Annotations' : '👁️ Show Annotations';
    btn2.textContent = showingPreprocessOverlay ? '👁️ Hide Detections' : '👁️ Show Detections';
    
    // Clear detection list
    document.getElementById('detectionList').innerHTML = '<div class="no-data">No detections</div>';
    document.getElementById('detectionCount').textContent = '0';
    
    // Clear canvas
    const canvas = document.getElementById('imageCanvas');
    canvas.style.display = 'none';
    
    // Wait for image to load
    img.onload = async () => {
        // Load annotation if exists (avoid 404 spam if we know it's not annotated yet)
        const meta = allImages.find(i => i.filename === filename);
        if (meta && !meta.has_annotation) {
            currentAnnotation = null;
            displayNoAnnotation();
        } else {
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
        }
        
        updateSaveButton();
    };

    img.onerror = () => {
        // If image failed to load, still try to render empty panels safely
        currentAnnotation = null;
        displayNoAnnotation();
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
    localStorage.setItem('showingVisualization', showingVisualization);
    const btn = document.getElementById('toggleViewBtn');
    btn.textContent = showingVisualization ? '👁️ Hide Annotations' : '👁️ Show Annotations';
    drawAnnotations();
}

// Toggle preprocess overlay
function togglePreprocessOverlay() {
    if (!currentImage) return;
    showingPreprocessOverlay = !showingPreprocessOverlay;
    localStorage.setItem('showingPreprocessOverlay', showingPreprocessOverlay);
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
    if (!saveBtn) return;
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
        
        if (isFolder && file.webkitRelativePath) {
            formData.append('relative_path', file.webkitRelativePath);
        }
        
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
        batchStatus.textContent = 'Starting batch job...';
        progressFill.style.width = '0%';
        
        const startResp = await fetch('/api/batch-annotate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ force: force })
        });
        
        if (!startResp.ok) {
            throw new Error('Failed to start batch');
        }
        const { job_id, total } = await startResp.json();
        let completed = 0;
        let success = 0;
        let skipped = 0;
        let errors = 0;
        let batchCompleted = false;
        
        if (batchEventSource) {
            try { batchEventSource.close(); } catch (_) {}
            batchEventSource = null;
        }
        
        batchStatus.textContent = `Queued ${total} images...`;
        
        batchEventSource = new EventSource(`/api/batch-annotate/stream/${job_id}`);
        
        batchEventSource.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                if (data.type === 'init') {
                    // reset based on server totals
                } else if (data.type === 'preprocess_start') {
                    batchStatus.textContent = `Preprocessing ${data.filename}...`;
                } else if (data.type === 'preprocessed') {
                    batchStatus.textContent = `Sending ${data.filename} to OpenAI...`;
                } else if (data.type === 'request_sent') {
                    // no-op, keep status
                } else if (data.type === 'image_done') {
                    completed = data.completed;
                    const pct = total > 0 ? Math.round((completed / total) * 100) : 100;
                    progressFill.style.width = pct + '%';
                    if (data.status === 'success') success++;
                    else if (data.status === 'skipped') skipped++;
                    else if (data.status === 'error') errors++;
                    batchStatus.textContent = `${completed}/${total} done — ✓ ${success}, ○ ${skipped}, ✗ ${errors}`;
                    // Update image list status optimistically
                    if (data.filename) {
                        const idx = allImages.findIndex(i => i.filename === data.filename);
                        if (idx !== -1 && data.status === 'success') {
                            allImages[idx].has_annotation = true;
                            displayImageList();
                        }
                    }
                } else if (data.type == 'complete') {
                    progressFill.style.width = '100%';
                    batchCompleted = true;
                    batchStatus.textContent = `Complete! ✓ ${data.summary.success}, ○ ${data.summary.skipped}, ✗ ${data.summary.errors}`;
                }
            } catch (e) {
                // ignore parse issues
            }
        };
        
        batchEventSource.addEventListener('end', () => {
            try { batchEventSource.close(); } catch (_) {}
            batchEventSource = null;
            setTimeout(() => {
                document.getElementById('batchModal').classList.remove('active');
                batchProgress.style.display = 'none';
                progressFill.style.width = '0%';
                btn.disabled = false;
                loadImages();
                showToast('Batch annotation complete', 'success');
            }, 1200);
        });
        
        batchEventSource.onerror = () => {
            // If we've already received completion, don't mark as failed
            if (batchCompleted) {
                try { batchEventSource.close(); } catch (_) {}
                batchEventSource = null;
                return;
            }
            // Connection error - show neutral status
            try { batchEventSource.close(); } catch (_) {}
            batchEventSource = null;
            batchStatus.textContent = 'Connection interrupted. Processing continues in background.';
            btn.disabled = false;
        };
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

// Update single image UI without re-rendering
function updateSingleImageUI(filename) {
    const imageItem = document.querySelector(`.image-item[data-filename="${filename}"]`);
    if (!imageItem) return;
    
    const isSelected = selectedImages.has(filename);
    const checkbox = imageItem.querySelector('input[type="checkbox"]');
    
    // Update checkbox
    if (checkbox) {
        checkbox.checked = isSelected;
    }
    
    // Update classes and draggable
    if (isSelected) {
        imageItem.classList.add('selected');
        imageItem.draggable = true;
    } else {
        imageItem.classList.remove('selected');
        imageItem.draggable = false;
    }
}

// Update all image items UI without re-rendering
function updateImageSelectionUI() {
    const imageItems = document.querySelectorAll('.image-item');
    imageItems.forEach(item => {
        const filename = item.dataset.filename;
        if (!filename) return;
        
        const isSelected = selectedImages.has(filename);
        const checkbox = item.querySelector('input[type="checkbox"]');
        
        // Update checkbox
        if (checkbox) {
            checkbox.checked = isSelected;
        }
        
        // Update classes and draggable
        if (isSelected) {
            item.classList.add('selected');
            item.draggable = true;
        } else {
            item.classList.remove('selected');
            item.draggable = false;
        }
    });
    
    // Re-setup drag events for newly draggable items
    setupImageDragEvents();
}

// Handle image click based on mode
function handleImageClick(filename, index, event) {
    if (selectMode) {
        // In select mode, click toggles selection
        if (event && event.shiftKey && lastSelectedIndex !== -1) {
            // Shift+click for range selection
            const start = Math.min(lastSelectedIndex, index);
            const end = Math.max(lastSelectedIndex, index);
            for (let i = start; i <= end; i++) {
                if (i < filteredImages.length) {
                    if (selectedImages.has(filteredImages[i].filename)) {
                        selectedImages.delete(filteredImages[i].filename);
                    } else {
                        selectedImages.add(filteredImages[i].filename);
                    }
                }
            }
            updateImageSelectionUI();
        } else {
            // Normal click toggles
            if (selectedImages.has(filename)) {
                selectedImages.delete(filename);
            } else {
                selectedImages.add(filename);
            }
            updateSingleImageUI(filename);
        }
        lastSelectedIndex = index;
        updateBulkActionsVisibility();
        updateSelectAllCheckbox();
    } else {
        // In normal mode, click selects the image for viewing
        selectImage(filename);
    }
}

// Setup drag events for images
function setupImageDragEvents() {
    const imageItems = document.querySelectorAll('.image-item[draggable="true"]');
    imageItems.forEach(item => {
        // Skip if already has drag listeners
        if (item.dataset.dragListeners === 'true') return;
        
        item.addEventListener('dragstart', (e) => {
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', 'moving-images');
            item.classList.add('dragging');
        });
        
        item.addEventListener('dragend', (e) => {
            item.classList.remove('dragging');
        });
        
        item.dataset.dragListeners = 'true';
    });
}

// Setup drag and drop for folders
function setupFolderDragAndDrop() {
    const folderHeaders = document.querySelectorAll('.folder-header');
    folderHeaders.forEach(header => {
        header.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            header.classList.add('drag-over');
        });
        
        header.addEventListener('dragleave', (e) => {
            header.classList.remove('drag-over');
        });
        
        header.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            header.classList.remove('drag-over');
            
            // Get folder path from the data attribute
            const targetFolder = header.dataset.folderPath;
            
            if (!targetFolder) {
                showToast('Invalid folder', 'error');
                return;
            }
            
            if (selectedImages.size === 0) {
                showToast('No images selected to move', 'error');
                return;
            }
            
            await moveImagesToFolder(Array.from(selectedImages), targetFolder);
        });
    });
}

// Move images to a folder
async function moveImagesToFolder(filenames, targetFolder) {
    if (!filenames || filenames.length === 0) return;
    
    const progressMsg = document.createElement('div');
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Moving images...</strong>
                <span>0 / ${filenames.length}</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    try {
        const response = await fetch('/api/move-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames, target_folder: targetFolder })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Move failed');
        }
        
        const data = await response.json();
        
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✓ Complete!</strong>
            <span>Moved ${data.moved} images</span>
        `;
        progressMsg.classList.add('success');
        
        setTimeout(() => {
            progressMsg.remove();
            selectedImages.clear();
            loadImages();
            showToast(`Moved ${data.moved} images to ${targetFolder}`, 'success');
        }, 1500);
        
    } catch (error) {
        console.error('Move error:', error);
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✗ Move Failed</strong>
            <span>${error.message}</span>
        `;
        progressMsg.classList.add('error');
        setTimeout(() => progressMsg.remove(), 3000);
    }
}

// Export dataset
async function exportDataset() {
    if (selectedImages.size === 0) {
        showToast('No images selected for export', 'error');
        return;
    }
    
    const splitName = prompt('Enter dataset split name (e.g., train, val, test):', 'train');
    if (!splitName) return;
    
    if (!confirm(`Export ${selectedImages.size} selected images as "${splitName}" split?`)) {
        return;
    }
    
    const progressMsg = document.createElement('div');
    progressMsg.className = 'floating-progress';
    progressMsg.innerHTML = `
        <div class="floating-progress-content">
            <div class="spinner"></div>
            <div class="floating-progress-text">
                <strong>Exporting Dataset...</strong>
                <span>Processing ${selectedImages.size} images...</span>
            </div>
        </div>
    `;
    document.body.appendChild(progressMsg);
    
    try {
        const filenames = Array.from(selectedImages);
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames, split: splitName })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Export failed');
        }
        
        const data = await response.json();
        
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✓ Export Complete!</strong>
            <span>Exported ${data.exported_images} images</span>
        `;
        progressMsg.classList.add('success');
        
        setTimeout(() => {
            progressMsg.remove();
            showToast(`Exported to: ${data.output_path}`, 'success');
            
            // Show detailed info
            alert(`Export Complete!\n\nLocation: ${data.output_path}\nImages: ${data.exported_images}\nFormat: ShowUI-desktop`);
        }, 2000);
        
    } catch (error) {
        console.error('Export error:', error);
        progressMsg.querySelector('.floating-progress-text').innerHTML = `
            <strong>✗ Export Failed</strong>
            <span>${error.message}</span>
        `;
        progressMsg.classList.add('error');
        setTimeout(() => progressMsg.remove(), 3000);
    }
}

// Handle image load for canvas sizing
const displayImage = document.getElementById('displayImage');
if (displayImage) {
    displayImage.addEventListener('load', function() {
        if ((currentAnnotation && showingVisualization) || (currentPreprocess && showingPreprocessOverlay)) {
            drawAnnotations();
        }
    });
}
