/**
 * Asakta Vision AI - Main Application Logic
 * Consolidating detection, editing, and submission workflows.
 */

let detections = [];
let uploadedFile = null; // Legacy single file
let batchImages = [];    // New Batch Array: [{file, src, detections, master, dims, processed: bool}]
let activeBatchIndex = -1;
let masterResult = null; 
let imageDimensions = { width: 0, height: 0 };
let CLASS_OPTIONS = [
    "POLE_9M", "POLE_11M", "POLE_8.1M",
    "INS_PIN", "INS_DISC",
    "T_RISING", "TAPPING_CHANNEL", "SIDE_ARM_CHANNEL", "V_CROSS",
    "CONDUCTOR", "STREET_LIGHT", "DTR", 
    "WIRE_BROKEN", "VEGETATION", "OBJECT"
];

// Distinct Premium Color Palette
const CLASS_COLORS = {
    "POLE_9M": "#0ea5e9",        // Cyan-Blue
    "POLE_11M": "#38bdf8",       // Lighter Cyan
    "POLE_8.1M": "#0284c7",      // Darker Cyan
    "INS_PIN": "#00ff00",        // Bright Green
    "INS_DISC": "#22c55e",       // Emerald Green
    "T_RISING": "#ff00ff",       // Magenta
    "TAPPING_CHANNEL": "#d946ef", // Fuchsia
    "SIDE_ARM_CHANNEL": "#a855f7", // Purple
    "V_CROSS": "#f43f5e",        // Rose
    "CONDUCTOR": "#00ffff",      // Cyan
    "STREET_LIGHT": "#f59e0b",   // Amber
    "DTR": "#8b5cf6",            // Violet
    "WIRE_BROKEN": "#ef4444",    // Bright Red (Fault)
    "VEGETATION": "#fbbf24",     // Amber (Encroachment)
    "OBJECT": "#a8a29e"          // Stone
};

// UI Persistence State
let expandedGroups = new Set();

function getFakeConfidenceValue(rawConf) {
    return Math.round(rawConf * 100) + '%';
}

// Drawing State
let isDrawMode = false;
let isDrawing = false;
let drawStart = null; 
let pendingBbox = null; 

// History & Shortcuts
let historyStack = [];
let redoStack = [];
const MAX_HISTORY = 50;
let lastSaveTime = Date.now();

function saveToHistory() {
    if (historyStack.length >= MAX_HISTORY) historyStack.shift();
    historyStack.push(JSON.stringify(detections));
    redoStack = []; // Clear redo on new action
}

function undo() {
    if (historyStack.length > 1) {
        redoStack.push(historyStack.pop());
        detections = JSON.parse(historyStack[historyStack.length - 1]);
        renderResults();
        renderBoxes();
        showToast("Undo", "primary");
    }
}

function redo() {
    if (redoStack.length > 0) {
        const state = redoStack.pop();
        historyStack.push(state);
        detections = JSON.parse(state);
        renderResults();
        renderBoxes();
        showToast("Redo", "primary");
    }
}

function saveDraft() {
    // In-memory save for active batch image
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
    }
    // Persist to localStorage immediately
    persistDraftToStorage();
    lastSaveTime = Date.now();
}

// ─── localStorage Draft Persistence ──────────────────────────────────────────
const DRAFT_KEY = 'asakta_worker_draft';

function persistDraftToStorage() {
    if (!batchImages.length) return;
    try {
        const payload = {
            savedAt: Date.now(),
            batchImages: batchImages.map(b => ({
                src:        b.src,
                name:       b.file?.name || b.name || 'image.jpg',
                detections: b.detections || [],
                master:     b.master || null,
                dims:       b.dims || {},
                processed:  b.processed || false
            })),
            activeBatchIndex,
            detections,
            masterResult,
            imageDimensions
        };
        localStorage.setItem(DRAFT_KEY, JSON.stringify(payload));
    } catch(e) {
        console.warn('[Draft] localStorage write failed', e);
    }
}

function clearDraftStorage() {
    localStorage.removeItem(DRAFT_KEY);
}

function checkAndRestoreDraft() {
    try {
        const raw = localStorage.getItem(DRAFT_KEY);
        if (!raw) return;
        const draft = JSON.parse(raw);
        if (!draft || !draft.batchImages || draft.batchImages.length === 0) return;

        const ageMinutes = Math.round((Date.now() - draft.savedAt) / 60000);
        const imageCount = draft.batchImages.length;
        const processedCount = draft.batchImages.filter(b => b.processed).length;

        // Show restore banner
        const banner = document.getElementById('draftRestoreBanner');
        if (!banner) return;

        document.getElementById('draftInfo').textContent =
            `${imageCount} image${imageCount > 1 ? 's' : ''} · ${processedCount} analysed · saved ${ageMinutes < 1 ? 'just now' : ageMinutes + 'm ago'}`;

        banner.classList.remove('hidden');
        banner.onclick = () => {};

        document.getElementById('btnRestoreDraft').onclick = () => {
            batchImages = draft.batchImages.map(b => ({
                file:       { name: b.name },
                name:       b.name,
                src:        b.src,
                detections: b.detections || [],
                master:     b.master || null,
                dims:       b.dims || {},
                processed:  b.processed || false
            }));
            activeBatchIndex     = draft.activeBatchIndex >= 0 ? draft.activeBatchIndex : 0;
            detections           = draft.detections || [];
            masterResult         = draft.masterResult || null;
            imageDimensions      = draft.imageDimensions || {};

            // Restore preview image
            if (batchImages[activeBatchIndex]?.src) {
                const preview = document.getElementById('preview');
                if (preview) preview.src = batchImages[activeBatchIndex].src;
                const imgContainer = document.getElementById('imageContainer');
                const submitSection = document.getElementById('submitSection');
                if (imgContainer) imgContainer.classList.remove('hidden');
                if (submitSection) submitSection.classList.remove('hidden');
            }

            renderBatchStrip();
            renderResults();
            renderBoxes();

            banner.classList.add('hidden');
            showToast('Draft restored!', 'success');
        };

        document.getElementById('btnDiscardDraft').onclick = () => {
            clearDraftStorage();
            banner.classList.add('hidden');
        };

    } catch(e) {
        console.warn('[Draft] Restore check failed', e);
    }
}

// Warn user before closing tab if work is in progress
window.addEventListener('beforeunload', (e) => {
    const hasWork = batchImages.some(b => b.processed) || detections.length > 0;
    if (hasWork) {
        e.preventDefault();
        e.returnValue = 'You have unsaved annotations. Are you sure you want to leave?';
        return e.returnValue;
    }
});

// After final submission, clear the draft
function clearDraftAfterSubmit() {
    clearDraftStorage();
    batchImages = [];
    activeBatchIndex = -1;
    detections = [];
    masterResult = null;
}

document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('upload');
    const previewImg = document.getElementById('preview');
    const dropZone = document.getElementById('dropZone');

    // Check for saved draft on page load
    checkAndRestoreDraft();

    // Auto-save to localStorage every 10 seconds if work is in progress
    setInterval(() => {
        if (batchImages.some(b => b.processed) || detections.length > 0) {
            persistDraftToStorage();
        }
    }, 10000);

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
            if (e.key === 's') { e.preventDefault(); saveDraft(); }
        }
        if (e.key === 'Delete' || e.key === 'Backspace') {
            // Logic to delete selected box if applicable
        }
    });

    if (uploadInput) {
        uploadInput.addEventListener('change', handleUpload);
    }

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            // Always allow adding more images via the dropzone click
            uploadInput.click();
        });
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-500', 'bg-blue-500/5');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
            if (e.dataTransfer.files.length) {
                uploadInput.files = e.dataTransfer.files;
                handleUpload({ target: uploadInput });
            }
        });
    }

    const overlay = document.getElementById('detectionOverlay');
    if (overlay) {
        // Unified pointer handling for both mouse and touch
        overlay.addEventListener('mousedown', handleDragStart);
        overlay.addEventListener('touchstart', handleDragStart, { passive: false });
    }
});

function handleUpload(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    
    // Add to batch
    files.forEach(file => {
        batchImages.push({
            file: file,
            src: URL.createObjectURL(file), // Local preview URL
            detections: [],
            master: null,
            dims: { width: 0, height: 0 },
            processed: false
        });
    });

    // Reset UI
    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('batchStripWrapper').classList.remove('hidden');
    document.getElementById('imageContainer').classList.remove('hidden');
    document.getElementById('dropZone').classList.add('py-4');
    document.getElementById('dropZone').classList.remove('p-10');

    renderBatchStrip();
    
    // Select the first new image if none active
    if (activeBatchIndex === -1) {
        selectBatchImage(batchImages.length - files.length);
    }

    // Clear the input so the same files can be selected again if needed
    e.target.value = '';
}

function renderBatchStrip() {
    const strip = document.getElementById('batchStrip');
    strip.innerHTML = '';
    
    document.getElementById('batchCount').textContent = `${batchImages.length} Files`;

    batchImages.forEach((item, index) => {
        const thumb = document.createElement('div');
        thumb.className = `batch-thumb ${index === activeBatchIndex ? 'active' : ''} ${item.processed ? 'processed' : ''}`;
        thumb.onclick = () => selectBatchImage(index);
        
        const img = document.createElement('img');
        img.src = item.src;
        thumb.appendChild(img);

        // Hover Remove Button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'batch-thumb-remove';
        removeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeBatchImage(index);
        };
        thumb.appendChild(removeBtn);
        
        strip.appendChild(thumb);
    });

    // Add "Add More" Button at the end of the strip
    const addMore = document.createElement('div');
    addMore.className = 'batch-thumb-add';
    addMore.innerHTML = `
        <i class="fa-solid fa-plus text-blue-400"></i>
        <span class="text-[8px] font-bold text-gray-400 mt-1">ADD MORE</span>
    `;
    addMore.onclick = () => document.getElementById('upload').click();
    strip.appendChild(addMore);
}

function removeBatchImage(index) {
    // Revoke URL to free memory
    URL.revokeObjectURL(batchImages[index].src);
    
    batchImages.splice(index, 1);
    
    if (batchImages.length === 0) {
        resetSession();
    } else {
        // Handle index shifting
        if (activeBatchIndex === index) {
            const next = Math.min(index, batchImages.length - 1);
            activeBatchIndex = -1; // Force clean select
            selectBatchImage(next);
        } else if (activeBatchIndex > index) {
            activeBatchIndex--;
        }
        renderBatchStrip();
    }
}

function selectBatchImage(index) {
    if (index < 0 || index >= batchImages.length) return;
    
    // Save current active state before switching (if needed)
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
        batchImages[activeBatchIndex].master = masterResult;
        batchImages[activeBatchIndex].dims = {...imageDimensions};
    }

    activeBatchIndex = index;
    const item = batchImages[index];
    
    // Load new image state
    uploadedFile = item.file; 
    detections = [...item.detections];
    masterResult = item.master;
    imageDimensions = {...item.dims};

    // Update UI
    document.getElementById('preview').src = item.src;
    renderBatchStrip();
    renderResults();
    renderBoxes();
}

function resetSession() {
    if (batchImages.length > 0) {
        if (!confirm("Clear current batch and all detection results?")) return;
    }

    batchImages = [];
    activeBatchIndex = -1;
    detections = [];
    uploadedFile = null;
    masterResult = null;
    
    // UI Reset
    document.getElementById('uploadPrompt').classList.remove('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('resultBox').innerHTML = '';
    document.getElementById('masterIdentityCard').classList.add('hidden');
    document.getElementById('dropZone').classList.remove('py-4');
    document.getElementById('dropZone').classList.add('p-10');
    document.getElementById('upload').value = ''; // Clear file input
    
    showToast("Session reset", "info");
}

async function resizeImage(file, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > maxWidth) {
                        height *= maxWidth / width;
                        width = maxWidth;
                    }
                } else {
                    if (height > maxHeight) {
                        width *= maxHeight / height;
                        height = maxHeight;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, 'image/jpeg', 0.85);
            };
        };
    });
}

async function processImage() {
    if (!uploadedFile) {
        showToast("Please upload an image first", "warning");
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    // UI state: Loading
    btn.disabled = true;
    btnText.textContent = "Processing...";
    loader.classList.remove('hidden');

    let imageToUpload = uploadedFile;
    if (uploadedFile.size > 1024 * 1024) {
        imageToUpload = await resizeImage(uploadedFile, 1280, 1280);
    }

    const formData = new FormData();
    formData.append("image", imageToUpload, "image.jpg");

    try {
        // Prevent indefinite hanging with a 60s timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const response = await fetch("/predict", { 
            method: "POST",
            headers: { "ngrok-skip-browser-warning": "69420" },
            body: formData,
            signal: controller.signal
        }).finally(() => clearTimeout(timeoutId));
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `System Error (${response.status})`);
        }
        
        const data = await response.json();
        
        // Use normalized labels
        detections = data.detections.map(d => ({ 
            ...d, 
            label: d.label.toUpperCase(), 
            confirmed: false 
        }));
        
        masterResult = data.master; // Store Master Identity
        imageDimensions = { width: data.width, height: data.height };

        // Save to Batch State
        if (activeBatchIndex !== -1) {
            batchImages[activeBatchIndex].detections = [...detections];
            batchImages[activeBatchIndex].master = masterResult;
            batchImages[activeBatchIndex].dims = {...imageDimensions};
            batchImages[activeBatchIndex].processed = true;
            renderBatchStrip();
        }

        if (data.annotated_image) {
            // Check if it's already a full data URI or just b64
            const imgSrc = data.annotated_image.startsWith('data:') ? data.annotated_image : `data:image/jpeg;base64,${data.annotated_image}`;
            document.getElementById("preview").src = imgSrc;
            document.getElementById("imageContainer").classList.remove("hidden");
            document.getElementById("submitSection").classList.remove("hidden");
        }

        renderResults();
        renderBoxes();
        saveToHistory(); // Initial state after prediction
        saveDraft(); // Auto-save draft immediately after detection
        showToast("Analysis complete", "success");
    } catch (err) {
        showToast(err.message || "Cloud connection error", "danger");
    } finally {
        btn.disabled = false;
        btnText.textContent = "Run Detection";
        loader.classList.add('hidden');
    }
}

async function saveDraftToServer() {
    if (!batchImages.length) return;
    
    try {
        const payload = {
            id: (batchImages[0]?.name || 'session') + '_' + (sessionStorage.getItem('username') || 'worker'),
            type: 'worker',
            data: JSON.stringify({
                detections: detections,
                master: masterResult,
                dimensions: imageDimensions,
                batchCount: batchImages.length,
                processedCount: batchImages.filter(b => b.processed).length
            })
        };
        await fetch('/api/save_draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        lastSaveTime = Date.now();
    } catch (e) {
        console.warn('[Draft] Server save failed', e);
    }
}

// Auto-save to server every 30 seconds
setInterval(() => {
    if (Date.now() - lastSaveTime > 30000) {
        saveDraft();
    }
}, 30000);

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undo();
    }
    if (e.ctrlKey && e.key === 'y') {
        e.preventDefault();
        redo();
    }
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveDraft();
        showToast("Draft Saved", "success");
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
        // Only delete if a box is highlighted/selected
        const highlighted = document.querySelector('.detection-box.highlighted');
        if (highlighted) {
            const idx = parseInt(highlighted.id.split('-')[1]);
            removeDetection(idx);
            showToast("Object Removed", "warning");
        }
    }
});

function renderResults() {
    const container = document.getElementById("resultBox");
    const masterCard = document.getElementById("masterIdentityCard");
    container.innerHTML = "";

    // 1. Populate Master Asset Identity Card
    if (masterResult) {
        masterCard.classList.remove('hidden');
        document.getElementById("masterClass").textContent = masterResult.final_class.replace(/_/g, ' ');
        document.getElementById("masterVoltage").textContent = masterResult.voltage;
        document.getElementById("masterReason").textContent = masterResult.reason;
        
        const confEl = document.getElementById("masterConfidence");
        const confVal = masterResult.confidence ? masterResult.confidence.toLowerCase() : 'low';
        if (confVal === 'high') confEl.className = "text-[8px] font-bold text-emerald-400/80 uppercase tracking-widest";
        else if (confVal === 'medium') confEl.className = "text-[8px] font-bold text-amber-400/80 uppercase tracking-widest";
        else confEl.className = "text-[8px] font-bold text-rose-400/80 uppercase tracking-widest";

        // 1.1 Update Pole Stability Row
        const stabilityRow = document.getElementById("poleStabilityRow");
        const angleEl = document.getElementById("poleLeanAngle");
        const statusBadge = document.getElementById("poleStatusBadge");

        if (masterResult.pole_lean_angle !== undefined) {
            stabilityRow.classList.remove("hidden");
            const angle = parseFloat(masterResult.pole_lean_angle);
            angleEl.textContent = `${angle.toFixed(1)}°`;

            // Categorization logic based on config thresholds
            let status = "Vertical";
            let badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
            
            if (masterResult.pole_type === 'strut_pole') {
                status = "Strut Pole";
                badgeClass = "bg-blue-500/10 text-blue-400 border border-blue-500/20";
            } else if (masterResult.pole_status === 'fault') {
                status = "Critical lean";
                badgeClass = "bg-rose-500/10 text-rose-400 border border-rose-500/20";
            } else if (masterResult.pole_status === 'warning') {
                status = "Leaning";
                badgeClass = "bg-amber-500/10 text-amber-400 border border-amber-500/20";
            }

            statusBadge.textContent = status;
            statusBadge.className = `px-2 py-0.5 rounded text-[8px] font-bold uppercase ${badgeClass}`;
        } else {
            stabilityRow.classList.add("hidden");
        }
    } else {
        masterCard.classList.add('hidden');
    }

    if (detections.length === 0) {
        container.innerHTML = `
            <div class="text-center py-20 bg-black/30 rounded-2xl border border-dashed border-gray-800">
                <i class="fa-solid fa-wand-magic-sparkles text-4xl text-gray-700 mb-4 block"></i>
                <p class="text-gray-600 text-sm italic">Waiting for analysis results...</p>
            </div>
        `;
        return;
    }

    // 2. Grouping detections by label
    const groups = {};
    detections.forEach((obj, i) => {
        const lbl = obj.label;
        if (!groups[lbl]) groups[lbl] = [];
        groups[lbl].push({ ...obj, originalIndex: i });
    });

    // 3. Render Each Component Group
    for (const label in groups) {
        const groupItems = groups[label];
        const groupCount = groupItems.length;
        const groupDiv = document.createElement("div");
        groupDiv.className = "mb-4 border border-gray-800 rounded-xl overflow-hidden bg-gray-900/40 transition-all shadow-sm";

        const baseColor = CLASS_COLORS[label.toUpperCase()] || "#a8a29e";

        // Group Header
        const header = document.createElement("div");
        header.className = "flex items-center justify-between p-4 cursor-pointer hover:bg-gray-800/50 transition-colors group";
        header.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl flex items-center justify-center" style="background: ${baseColor}15; border: 1px solid ${baseColor}30;">
                    <i class="fa-solid ${label === 'POLE' ? 'fa-tower-broadcast' : (label === 'INSULATOR' ? 'fa-bolt' : 'fa-layer-group')}" style="color: ${baseColor};"></i>
                </div>
                <div>
                    <h3 class="text-xs font-bold uppercase tracking-wider" style="color: ${baseColor};">${label}S</h3>
                    <p class="text-[9px] text-gray-500 font-medium uppercase tracking-tighter">${groupCount} identified</p>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <span class="px-2 py-0.5 text-[10px] font-mono rounded-lg" style="background: ${baseColor}15; color: ${baseColor}; border: 1px solid ${baseColor}30;">${groupCount}</span>
                <i class="fa-solid ${expandedGroups.has(label) ? 'fa-chevron-up' : 'fa-chevron-down'} text-[10px] text-gray-600" id="icon-${label}"></i>
            </div>
        `;
        
        const itemsContainer = document.createElement("div");
        const isOpen = expandedGroups.has(label);
        itemsContainer.className = `${isOpen ? '' : 'hidden'} p-2 space-y-2 bg-black/40 border-t border-gray-800/50`;
        itemsContainer.id = `container-${label}`;
        
        header.onclick = () => {
            const isHidden = itemsContainer.classList.contains('hidden');
            const icon = document.getElementById(`icon-${label}`);
            if (isHidden) {
                itemsContainer.classList.remove('hidden');
                expandedGroups.add(label);
                icon.className = 'fa-solid fa-chevron-up text-[10px] text-gray-600';
            } else {
                itemsContainer.classList.add('hidden');
                expandedGroups.delete(label);
                icon.className = 'fa-solid fa-chevron-down text-[10px] text-gray-600';
            }
        };

        groupItems.forEach((obj, idx) => {
            const i = obj.originalIndex;
            const itemDiv = document.createElement("div");
            itemDiv.className = `flex items-center justify-between p-3 rounded-lg border transition-all result-card-hover ${obj.confirmed ? 'border-green-500/40 bg-green-500/5' : 'bg-white/5 border-white/5'} hover:border-white/20`;

            // Interactive Sync: Glow the box on the image when hovering the result item
            itemDiv.onmouseenter = () => {
                const box = document.getElementById(`box-${i}`);
                if (box) box.classList.add('highlighted');
                const label = document.getElementById(`label-${i}`);
                if (label) {
                    label.classList.remove('opacity-0');
                    label.classList.add('opacity-100');
                    label.style.transform = 'scale(1.1)';
                }
            };
            itemDiv.onmouseleave = () => {
                const box = document.getElementById(`box-${i}`);
                if (box) box.classList.remove('highlighted');
                const label = document.getElementById(`label-${i}`);
                if (label) {
                    label.classList.remove('opacity-100');
                    label.classList.add('opacity-0');
                    label.style.transform = '';
                }
            };

            // Dynamic details based on component type
            let detailStr = "";
            let metaIcon = "fa-tag";
            const fakeConfStr = getFakeConfidenceValue(obj.confidence);

            if (obj.label.includes('INS') && obj.details) {
                detailStr = `${obj.details.type} (${obj.details.voltage}) | Sheds: ${obj.details.sheds} | Conf: ${fakeConfStr}`;
                metaIcon = "fa-bolt";
            } else if (obj.label.includes('CROSSARM') && obj.details) {
                detailStr = `Geometry: ${obj.details.shape} | Conf: ${fakeConfStr}`;
                metaIcon = "fa-compass-drafting";
            } else if (obj.label.includes('POLE') && obj.details) {
                const lean = obj.details.lean || 0;
                const isExtreme = lean > 10;
                const leanColor = isExtreme ? 'text-rose-400 font-black' : (lean > 5 ? 'text-amber-400' : 'text-emerald-400');
                const abnormalityTag = isExtreme ? `<span class="bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded-full text-[7px] border border-rose-500/30 ml-2 animate-pulse">ABNORMALITY</span>` : "";
                detailStr = `<span class="${leanColor}">LEAN: ${lean}°</span>${abnormalityTag} | ${obj.details.type} | <span class="text-white/60">Conf: ${fakeConfStr}</span>`;
                metaIcon = "fa-triangle-exclamation";
            } else if (obj.label === 'WIRE_BROKEN') {
                detailStr = `<span class="text-rose-500 font-bold underline">CRITICAL: SNAPPED CONDUCTOR</span> | Conf: ${fakeConfStr}`;
                metaIcon = "fa-scissors";
            } else if (obj.label === 'VEGETATION') {
                detailStr = `<span class="text-amber-500 font-bold">ENCROACHMENT DETECTED</span> | Conf: ${fakeConfStr}`;
                metaIcon = "fa-leaf";
            } else {
                detailStr = `Confidence: ${fakeConfStr}`;
            }

            itemDiv.innerHTML = `
                <div class="flex items-center gap-3">
                    <div class="w-1.5 h-1.5 rounded-full" style="background: ${baseColor}"></div>
                    <div>
                        <p class="text-[10px] font-bold text-gray-200 uppercase tracking-tight">${label} ID-${i + 1}</p>
                        <div class="flex items-center gap-1.5 mt-0.5">
                            <i class="fa-solid ${metaIcon} text-[8px] text-gray-600"></i>
                            <p class="text-[9px] text-gray-500 font-medium uppercase tracking-widest">${detailStr}</p>
                        </div>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="toggleConfirm(${i})" class="btn ${obj.confirmed ? 'bg-green-600 text-white' : 'btn-outline border-white/5 bg-white/5 hover:border-white/20'} !p-2 !h-8 !w-8 !rounded-lg text-[10px]">
                        <i class="fa-solid ${obj.confirmed ? 'fa-check-double' : 'fa-check'}"></i>
                    </button>
                    <button onclick="removeDetection(${i})" class="p-2 text-gray-600 hover:text-rose-400 transition-colors">
                        <i class="fa-solid fa-trash-can text-[10px]"></i>
                    </button>
                </div>
            `;
            itemsContainer.appendChild(itemDiv);
        });

        groupDiv.appendChild(header);
        groupDiv.appendChild(itemsContainer);
        container.appendChild(groupDiv);
    }
    
    // Update final submit button state
    const allConfirmed = detections.every(d => d.confirmed);
    const submitBtn = document.getElementById('finalSubmitBtn');
    if (submitBtn) {
        if (allConfirmed && detections.length > 0) {
            submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = false;
        } else {
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = true;
        }
    }
}

function updateLabel(index, val) {
    saveToHistory();
    if (val === "Custom") {
        const customVal = prompt("Enter custom class name:");
        if (customVal) {
            detections[index].label = customVal.toUpperCase();
        }
    } else {
        detections[index].label = val;
    }
    renderResults();
    renderBoxes();
}

function toggleConfirm(index) {
    saveToHistory();
    detections[index].confirmed = !detections[index].confirmed;
    renderResults();
    renderBoxes();
}

function removeDetection(index) {
    saveToHistory();
    detections.splice(index, 1);
    renderResults();
    renderBoxes();
}

function removeGroup(label) {
    if (confirm(`Are you sure you want to remove all "${label}" detections?`)) {
        detections = detections.filter(d => d.label !== label);
        renderResults();
        renderBoxes();
        showToast(`Removed all ${label} items`, "primary");
    }
}

function renderBoxes() {
    const overlay = document.getElementById('detectionOverlay');
    const img = document.getElementById('preview');
    if (!overlay || !img || detections.length === 0) return;

    overlay.innerHTML = "";
    
    // 0. Add SVG Filters for Glow Effect
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    `;
    overlay.appendChild(defs);

    // Calculate scaling
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    const scaleX = displayWidth / imageDimensions.width;
    const scaleY = displayHeight / imageDimensions.height;

    const labelCounts = {};

    detections.forEach((obj, i) => {
        if (!obj.bbox) return;

        const baseLabel = obj.label.toUpperCase();
        labelCounts[baseLabel] = (labelCounts[baseLabel] || 0) + 1;
        const currentCount = labelCounts[baseLabel];
        const confText = getFakeConfidenceValue(obj.confidence);
        
        // Add lean angle to label text if it's a pole
        let labelText = `${baseLabel} ${confText}`;
        if (baseLabel.includes('POLE') && obj.details && obj.details.lean !== undefined) {
            labelText += ` | LEAN: ${obj.details.lean}°`;
        }

        const [x1, y1, x2, y2] = obj.bbox;
        const w = (x2 - x1) * scaleX;
        const h = (y2 - y1) * scaleY;
        const x = x1 * scaleX;
        const y = y1 * scaleY;

        const baseColor = CLASS_COLORS[baseLabel] || "#a8a29e";
        const color = obj.manual ? "#f43f5e" : baseColor; 
        
        // --- 1. Draw Shape (Polygon or Rect) ---
        let shape;
        if (obj.polygon && obj.polygon.length > 2) {
            shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            const pointsStr = obj.polygon.map(pt => `${pt[0] * scaleX},${pt[1] * scaleY}`).join(" ");
            shape.setAttribute("points", pointsStr);
        } else {
            shape = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            shape.setAttribute("x", x);
            shape.setAttribute("y", y);
            shape.setAttribute("width", w);
            shape.setAttribute("height", h);
        }

        shape.setAttribute("stroke", color);
        shape.setAttribute("stroke-width", obj.manual ? "2.5" : "1.5");
        
        // DIAGNOSTIC HUD: Solid-feeling translucent fill for structural objects
        if (baseLabel !== "CONDUCTOR") {
            shape.setAttribute("fill", color);
            shape.setAttribute("fill-opacity", "0.15");
        } else {
            shape.setAttribute("fill", "transparent");
            shape.setAttribute("class", "conductor-trace");
        }
        
        shape.setAttribute("id", `box-${i}`);
        shape.classList.add("detection-box");
        shape.style.pointerEvents = "auto";
        if (obj.manual) shape.classList.add("manual-box");
        overlay.appendChild(shape);

        // --- 2. Calculate Label Position ---
        let labelX = x;
        let labelY = y;
        if (obj.polygon && obj.polygon.length > 0) {
            const topPoint = obj.polygon.reduce((min, p) => p[1] < min[1] ? p : min, obj.polygon[0]);
            const avgX = obj.polygon.reduce((sum, p) => sum + p[0], 0) / obj.polygon.length;
            labelX = avgX * scaleX;
            labelY = topPoint[1] * scaleY;
        } else {
            labelX = x + (w / 2);
            labelY = y;
        }

        // --- 3. Draw Pill Label (Background + Text) ---
        const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        labelGroup.setAttribute("class", "label-pill opacity-0 transition-opacity duration-200 pointer-events-none");
        labelGroup.setAttribute("id", `label-${i}`);
        
        const labelTextEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        labelTextEl.textContent = labelText;
        labelTextEl.setAttribute("font-size", "11px");
        labelTextEl.setAttribute("font-family", "Outfit, Inter, sans-serif");
        labelTextEl.setAttribute("font-weight", "700");
        labelTextEl.setAttribute("fill", "#ffffff");
        labelTextEl.setAttribute("text-anchor", "middle");
        labelTextEl.setAttribute("dominant-baseline", "middle");
        
        // Hide temporarily to measure
        labelTextEl.style.visibility = "hidden";
        overlay.appendChild(labelTextEl);
        const bbox = labelTextEl.getBBox();
        overlay.removeChild(labelTextEl);
        labelTextEl.style.visibility = "visible";

        const paddingH = 8;
        const paddingV = 4;
        const rectW = bbox.width + paddingH * 2;
        const rectH = bbox.height + paddingV * 2;
        
        const labelRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        labelRect.setAttribute("x", labelX - rectW / 2);
        labelRect.setAttribute("y", (labelY - rectH - 5 < 0) ? labelY + 5 : labelY - rectH - 5);
        labelRect.setAttribute("width", rectW);
        labelRect.setAttribute("height", rectH);
        labelRect.setAttribute("rx", "6");
        labelRect.setAttribute("fill", color);
        labelRect.setAttribute("class", "label-bg");
        
        labelTextEl.setAttribute("x", labelX);
        labelTextEl.setAttribute("y", (labelY - rectH - 5 < 0) ? labelY + 5 + rectH / 2 : labelY - rectH - 5 + rectH / 2);
        
        labelGroup.appendChild(labelRect);
        labelGroup.appendChild(labelTextEl);
        overlay.appendChild(labelGroup);

        // Hover events on the shape to toggle label visibility
        shape.addEventListener('mouseenter', () => {
            labelGroup.classList.remove('opacity-0');
            labelGroup.classList.add('opacity-100');
        });
        shape.addEventListener('mouseleave', () => {
            labelGroup.classList.remove('opacity-100');
            labelGroup.classList.add('opacity-0');
        });
    });
}

window.addEventListener('resize', renderBoxes);

/**
 * Compresses an image using Canvas before submission.
 * Reduces resolution to max 1280px and quality to 0.7 for optimal server speed.
 */
async function compressImage(src, maxWidth = 1200, quality = 0.75) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            let width = img.width;
            let height = img.height;

            if (width > maxWidth) {
                height *= maxWidth / width;
                width = maxWidth;
            }

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert to JPEG with specified quality
            const b64 = canvas.toDataURL('image/jpeg', quality);
            resolve(b64.split(',')[1]); // Return raw b64 data
        };
        img.onerror = reject;
    });
}

async function submitAsset() {
    if (batchImages.length === 0) return;

    const btn = document.getElementById('finalSubmitBtn');
    btn.disabled = true;
    const originalInner = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Submitting Batch...`;

    try {
        const payload = {
            master: masterResult, // Overall asset classification
            images: []
        };

        // Process each image in the batch with compression
        for (const item of batchImages) {
            try {
                const b64 = await compressImage(item.src);
                payload.images.push({
                    image_b64: b64,
                    detections: item.detections,
                    pole_angle: item.pole_angle || (item.detections.find(d => d.label === 'POLE')?.lean || 0.0)
                });
            } catch (pErr) {
                console.warn("Compression failed for an image, using original", pErr);
                // Fallback to original fetching if canvas fails
                const blob = await fetch(item.src).then(r => r.blob());
                const b64 = await new Promise(resolve => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                });
                payload.images.push({
                    image_b64: b64,
                    detections: item.detections,
                    pole_angle: item.pole_angle || (item.detections.find(d => d.label === 'POLE')?.lean || 0.0)
                });
            }
        }

        const res = await fetch('/api/save_asset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            showToast("Success! Asset Uploaded to Admin", "success");
            setTimeout(() => {
                resetSession();
                // Optionally redirect or clear
            }, 1500);
        } else {
            throw new Error(result.message);
        }
    } catch (err) {
        console.error(err);
        showToast("Global submission failed", "danger");
        btn.disabled = false;
        btn.innerHTML = originalInner;
    }
}

// =========================
// MANUAL ANNOTATION LOGIC
// =========================

function toggleDrawMode() {
    isDrawMode = !isDrawMode;
    const btn = document.getElementById('btnDrawMode');
    const overlay = document.getElementById('detectionOverlay');
    const status = document.getElementById('drawStatus');
    const imageContainer = document.getElementById('imageContainer');

    // Reset state when toggling
    cancelManualDraw();

    if (isDrawMode) {
        btn.classList.add('active');
        overlay.classList.add('draw-mode');
        overlay.classList.remove('pointer-events-none');
        overlay.style.pointerEvents = "auto";
        imageContainer.classList.add('draw-active');
        status.classList.remove('hidden');
        status.classList.add('flex');
        updateDrawStatus("PRESS & DRAG TO DRAW BOX");
        showToast("Draw Mode: Enabled", "primary");
    } else {
        btn.classList.remove('active');
        overlay.classList.remove('draw-mode');
        overlay.classList.add('pointer-events-none');
        overlay.style.pointerEvents = "none";
        imageContainer.classList.remove('draw-active');
        status.classList.add('hidden');
        status.classList.remove('flex');
        showToast("Draw Mode: Disabled", "primary");
    }
}

function updateDrawStatus(text) {
    const statusEl = document.getElementById('drawStatus');
    const label = statusEl.querySelector('span:last-child');
    if (label) {
        label.textContent = text;
        label.style.letterSpacing = "0.05em";
    }
}

function addNewClass() {
    const className = prompt("Enter name for the new object category:");
    if (className && className.trim()) {
        const upperName = className.trim().toUpperCase();
        if (!CLASS_OPTIONS.includes(upperName)) {
            CLASS_OPTIONS.push(upperName);
            showToast(`Added '${upperName}' to category list`, "success");
            renderResults(); // Refresh list to show new option in dropdowns
        } else {
            showToast("Category already exists", "warning");
        }
    }
}

function manageClasses() {
    if (CLASS_OPTIONS.length === 0) {
        showToast("No custom classes to manage", "warning");
        return;
    }

    let listStr = CLASS_OPTIONS.map((c, i) => `${i + 1}. ${c}`).join('\n');
    let selection = prompt(`Select class number to manage:\n\n${listStr}\n\n(Enter number)`);
    
    if (!selection) return;
    let idx = parseInt(selection) - 1;

    if (idx >= 0 && idx < CLASS_OPTIONS.length) {
        let oldName = CLASS_OPTIONS[idx];
        let action = prompt(`Managing "${oldName}"\nType 'R' to Rename or 'D' to Delete:`).toUpperCase();
        
        if (action === 'R') {
            let newName = prompt(`Enter new name for ${oldName}:`);
            if (newName && newName.trim()) {
                renameClass(oldName, newName.trim().toUpperCase());
            }
        } else if (action === 'D') {
            deleteClass(oldName);
        }
    }
}

function renameClass(oldName, newName) {
    // 1. Update the options list
    const optIdx = CLASS_OPTIONS.indexOf(oldName);
    if (optIdx !== -1) {
        CLASS_OPTIONS[optIdx] = newName;
    }

    // 2. Update all existing detections using this label
    let updateCount = 0;
    detections.forEach(det => {
        if (det.label === oldName) {
            det.label = newName;
            updateCount++;
        }
    });

    renderResults();
    renderBoxes();
    showToast(`Renamed ${oldName} to ${newName} (${updateCount} items updated)`, "success");
}

function deleteClass(name) {
    if (confirm(`Delete category "${name}"? Existing detections will remain but their category label will be static.`)) {
        CLASS_OPTIONS = CLASS_OPTIONS.filter(c => c !== name);
        renderResults();
        showToast(`Deleted category ${name}`, "warning");
    }
}

// =========================
// POINTER NORMALIZATION
// =========================

function getPointerPos(e) {
    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();
    
    // Support both mouse and touch events
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x: clientX - rect.left,
        y: clientY - rect.top,
        rawX: clientX,
        rawY: clientY
    };
}

function handleDragStart(e) {
    if (!isDrawMode || !uploadedFile) return;
    
    // Prevent accidental triggers and scrolling while drawing
    e.stopPropagation();
    if (e.type === 'touchstart') e.preventDefault(); 
    if (e.type === 'mousedown' && e.button !== 0) return; // Only left click

    const pos = getPointerPos(e);
    drawStart = { x: pos.x, y: pos.y };
    isDrawing = true;

    // Add global listeners to track movement outside the SVG
    window.addEventListener('mousemove', handleDragMove);
    window.addEventListener('touchmove', handleDragMove, { passive: false });
    window.addEventListener('mouseup', handleDragEnd);
    window.addEventListener('touchend', handleDragEnd);

    // Initial Marker
    const hud = document.getElementById('manualPointHud');
    hud.innerHTML = ""; // Clear old markers
    const marker = document.createElement('div');
    marker.className = 'point-marker';
    const overlay = document.getElementById('detectionOverlay');
    marker.style.left = `${(pos.x / overlay.clientWidth) * 100}%`;
    marker.style.top = `${(pos.y / overlay.clientHeight) * 100}%`;
    marker.id = 'startMarker';
    hud.appendChild(marker);

    updateDrawStatus("RELEASE TO FINISH BOX");
}

function handleDragMove(e) {
    if (!isDrawing || !drawStart) return;
    if (e.type === 'touchmove') e.preventDefault();

    const pos = getPointerPos(e);
    const overlay = document.getElementById('detectionOverlay');
    
    let ghost = document.getElementById('ghostBox');
    if (!ghost) {
        ghost = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        ghost.setAttribute("id", "ghostBox");
        ghost.setAttribute("class", "ghost-box active");
        overlay.appendChild(ghost);
    }

    const x = Math.min(drawStart.x, pos.x);
    const y = Math.min(drawStart.y, pos.y);
    const w = Math.abs(drawStart.x - pos.x);
    const h = Math.abs(drawStart.y - pos.y);

    ghost.setAttribute("x", x);
    ghost.setAttribute("y", y);
    ghost.setAttribute("width", w);
    ghost.setAttribute("height", h);
}

function handleDragEnd(e) {
    if (!isDrawing) return;
    
    // Remove global listeners
    window.removeEventListener('mousemove', handleDragMove);
    window.removeEventListener('touchmove', handleDragMove);
    window.removeEventListener('mouseup', handleDragEnd);
    window.removeEventListener('touchend', handleDragEnd);

    const pos = getPointerPos(e.type === 'touchend' ? { touches: e.changedTouches } : e);
    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();

    const x1 = Math.min(drawStart.x, pos.x);
    const y1 = Math.min(drawStart.y, pos.y);
    const x2 = Math.max(drawStart.x, pos.x);
    const y2 = Math.max(drawStart.y, pos.y);

    // Minimum size threshold to prevent accidental clicks
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    
    if (width < 15 || height < 15) {
        cancelManualDraw();
        return;
    }

    isDrawing = false;
    pendingBbox = [x1, y1, x2, y2];
    
    // Add visual "selection" appearance
    const ghost = document.getElementById('ghostBox');
    if (ghost) ghost.classList.add('final-preview');

    showLabelPicker(pos.rawX, pos.rawY);
    updateDrawStatus("SELECT CATEGORY BELOW");
}

function showLabelPicker(clientX, clientY) {
    const modal = document.getElementById('labelPickerModal');
    const backdrop = document.getElementById('labelPickerBackdrop');
    const options = document.getElementById('pickerOptions');
    options.innerHTML = '';

    CLASS_OPTIONS.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'label-picker-btn';
        btn.textContent = opt;
        btn.onclick = (e) => {
            e.stopPropagation();
            saveManualDraw(opt);
        };
        options.appendChild(btn);
    });

    backdrop.classList.remove('hidden');
    modal.classList.remove('hidden');
    
    // RESPONSIVE POSITIONING
    if (window.innerWidth < 640) {
        // Mobile: Center on screen
        modal.style.left = '50%';
        modal.style.top = '50%';
        modal.style.transform = 'translate(-50%, -50%)';
    } else {
        // Desktop: Float near click but stay within viewport
        let left = clientX + 30;
        let top = clientY - 100;

        const modalWidth = 340;
        const modalHeight = 400;

        // Viewport clamping
        if (left + modalWidth > window.innerWidth) left = window.innerWidth - modalWidth - 20;
        if (left < 20) left = 20;
        
        if (top + modalHeight > window.innerHeight) top = window.innerHeight - modalHeight - 20;
        if (top < 20) top = 20;

        modal.style.left = `${left}px`;
        modal.style.top = `${top}px`;
        modal.style.transform = 'none';
    }
}

function saveManualDraw(labelOverride = null) {
    // Prevent event from bubbling to dropZone if this was called from a button click
    if (window.event) window.event.stopPropagation();
    
    const customInput = document.getElementById('customLabelInput');
    const custom = customInput.value.trim().toUpperCase();
    const label = labelOverride || custom || "OBJECT";

    if (!pendingBbox) return;

    const overlay = document.getElementById('detectionOverlay');
    const rect = overlay.getBoundingClientRect();

    // Scale back to original image coordinates
    const scaleX = imageDimensions.width / rect.width;
    const scaleY = imageDimensions.height / rect.height;

    const newDet = {
        label: label,
        confidence: 1.0,
        bbox: [
            pendingBbox[0] * scaleX,
            pendingBbox[1] * scaleY,
            pendingBbox[2] * scaleX,
            pendingBbox[3] * scaleY
        ],
        confirmed: true,
        manual: true
    };

    detections.push(newDet);
    saveToHistory();
    
    // Add to CLASS_OPTIONS if new
    if (custom && !CLASS_OPTIONS.includes(custom)) {
        CLASS_OPTIONS.push(custom);
    }

    cancelManualDraw();
    renderResults();
    renderBoxes();
    showToast(`Added manual ${label}`, "success");
}

function cancelManualDraw(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    
    drawStart = null;
    isDrawing = false;
    pendingBbox = null;
    
    const ghost = document.getElementById('ghostBox');
    if (ghost) ghost.remove();
    
    const marker = document.getElementById('startMarker');
    if (marker) marker.remove();

    const modal = document.getElementById('labelPickerModal');
    const backdrop = document.getElementById('labelPickerBackdrop');
    if (modal) modal.classList.add('hidden');
    if (backdrop) backdrop.classList.add('hidden');
    
    const customInput = document.getElementById('customLabelInput');
    if (customInput) customInput.value = '';
    
    updateDrawStatus("STEP 1: CLICK TO START");
}

function showToast(msg, type = "primary") {
    const toast = document.createElement("div");
    const colors = {
        success: "bg-emerald-600",
        danger: "bg-rose-600",
        warning: "bg-amber-600",
        primary: "bg-blue-600"
    };
    
    toast.className = `fixed bottom-8 left-1/2 -translate-x-1/2 ${colors[type]} text-white px-6 py-3 rounded-xl shadow-2xl z-[100] animate-fade-in font-bold flex items-center gap-3`;
    
    const icons = {
        success: "fa-circle-check",
        danger: "fa-circle-xmark",
        warning: "fa-triangle-exclamation",
        primary: "fa-info-circle"
    };
    
    toast.innerHTML = `<i class="fa-solid ${icons[type]}"></i> ${msg}`;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translate(-50%, 20px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// AR STREAMING LOGIC
// ==========================================
let arStream = null;
let arInterval = null;

async function toggleARMode() {
    if (arStream) {
        stopAR();
    } else {
        await startAR();
    }
}

async function startAR() {
    const arVideo = document.getElementById('arVideo');
    const arContainer = document.getElementById('arContainer');
    const arCanvas = document.getElementById('arCanvas');
    
    try {
        arStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        if (arVideo) {
            arVideo.srcObject = arStream;
            arContainer.classList.remove('hidden');
            
            // Hide normal upload UI
            document.getElementById('dropZone').classList.add('hidden');
            document.getElementById('imageContainer').classList.add('hidden');
            document.getElementById('submitSection').classList.add('hidden');
            
            arVideo.onloadedmetadata = () => {
                arCanvas.width = arVideo.videoWidth;
                arCanvas.height = arVideo.videoHeight;
                arInterval = setInterval(processARFrame, 800); // sample every 800ms
            };
        }
        showToast("AR Mode Active", "success");
    } catch (err) {
        showToast("Camera access denied or unavailable", "danger");
        console.error("AR Start Error:", err);
    }
}

function stopAR() {
    const arContainer = document.getElementById('arContainer');
    const arCanvas = document.getElementById('arCanvas');

    if (arStream) {
        arStream.getTracks().forEach(t => t.stop());
        arStream = null;
    }
    if (arInterval) clearInterval(arInterval);
    arInterval = null;
    
    if (arContainer && document.getElementById('dropZone')) {
        arContainer.classList.add('hidden');
        document.getElementById('dropZone').classList.remove('hidden');
    }
    if (arCanvas) {
        const ctx = arCanvas.getContext('2d');
        ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    }
    showToast("AR Mode Stopped", "info");
}

async function processARFrame() {
    const arVideo = document.getElementById('arVideo');
    const arCanvas = document.getElementById('arCanvas');
    
    if (!arVideo || !arCanvas || !arStream) return;
    
    const ctx = arCanvas.getContext('2d');
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = arVideo.videoWidth;
    tempCanvas.height = arVideo.videoHeight;
    tempCanvas.getContext('2d').drawImage(arVideo, 0, 0);
    const frameBase64 = tempCanvas.toDataURL('image/jpeg', 0.6);
    
    document.getElementById('arOverlayLoading').classList.remove('hidden');

    try {
        const response = await fetch('/predict_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frameBase64 })
        });
        
        if (response.ok) {
            const data = await response.json();
            drawARBoxes(ctx, data.detections, data.width, data.height);
            
            // Sync side-panel
            detections = data.detections.map(d => ({ ...d, label: d.label.toUpperCase(), confirmed: false }));
            masterResult = data.master;
            renderResults();
        }
    } catch (e) {
        console.warn("AR Frame drop", e);
    } finally {
        document.getElementById('arOverlayLoading').classList.add('hidden');
    }
}

function drawARBoxes(ctx, detectionsData, origW, origH) {
    if (!ctx) return;
    const arCanvas = document.getElementById('arCanvas');
    ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    
    const scaleX = arCanvas.width / origW;
    const scaleY = arCanvas.height / origH;
    
    detectionsData.forEach(d => {
        const color = CLASS_COLORS[d.label.toUpperCase()] || "#00ff00";
        if (d.polygon && d.polygon.length > 0) {
            ctx.beginPath();
            ctx.moveTo(d.polygon[0][0] * scaleX, d.polygon[0][1] * scaleY);
            for(let i=1; i<d.polygon.length; i++) {
                ctx.lineTo(d.polygon[i][0] * scaleX, d.polygon[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = color;
            ctx.stroke();
            ctx.fillStyle = color + "33";
            ctx.fill();
        } else if (d.bbox && d.bbox.length === 4) {
            const [x1, y1, x2, y2] = d.bbox;
            ctx.lineWidth = 3;
            ctx.strokeStyle = color;
            ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2-x1) * scaleX, (y2-y1) * scaleY);
        }
        
        // draw label
        const cx = (d.bbox && !!d.bbox.length ? d.bbox[0] * scaleX : d.polygon[0][0] * scaleX);
        const cy = (d.bbox && !!d.bbox.length ? d.bbox[1] * scaleY : d.polygon[0][1] * scaleY);
        
        ctx.fillStyle = color;
        ctx.fillRect(cx, cy - 20, ctx.measureText(d.label).width + 10, 20);
        ctx.fillStyle = "#000000";
        ctx.font = "bold 12px Inter";
        ctx.fillText(d.label.toUpperCase(), cx + 5, cy - 5);
    });
}
async function downloadCurrentResult() {
    if (!detections || detections.length === 0) {
        showToast("No active results to download", "warning");
        return;
    }

    try {
        showToast("Preparing Assets...", "primary");
        
        // 1. Prepare asset payload
        const assetData = {
            id: `TEMP_${Date.now()}`,
            worker_name: "Local Session",
            status: "draft",
            timestamp: new Date().toLocaleString(),
            asset_class: masterResult ? masterResult.final_class : "Unclassified",
            voltage: masterResult ? masterResult.voltage : "Unknown",
            reason: masterResult ? masterResult.reason : "Manual Review",
            images: [{
                image_b64: uploadedFile, // original from memory
                detections: detections
            }]
        };

        // 2. Add Annotated Image for direct save if requested
        // Instead of a separate API call, we'll use a hidden link to trigger the download
        // of the annotated image from the server once an asset is SAVED, 
        // or just download the canvas/B64 directly.
        
        // We'll use the existing PDF generation for the main download
        const response = await fetch('/api/save_draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: assetData.id, type: 'export', data: JSON.stringify(assetData) })
        });

        if (response.ok) {
             // For the PDF report, we trigger a print or server-side gen
             // In this system, we'll offer the PDF summary.
             showToast("Report Ready", "success");
             // Note: In a real system we'd redirect to a PDF route.
             // For now, we'll let the user know it's saved in their History.
             window.location.href = `/admin/asset/${assetData.id}`;
        }
    } catch (err) {
        console.error("Export Error:", err);
        showToast("Export failed", "danger");
    }
}
