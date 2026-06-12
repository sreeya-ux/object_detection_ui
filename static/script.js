/**
 * Asakta Vision AI - Main Application Logic
 * Consolidating detection, editing, and submission workflows.
 */

let detections = [];
let uploadedFile = null; // Legacy single file
let batchImages = [];    // New Batch Array: [{file, src, detections, master, dims, processed: bool}]
let activeBatchIndex = -1;
let masterResult = null;
let surveyResult = {};
let imageDimensions = { width: 0, height: 0 };
let currentInputMode = 'image';
let uploadedVideoFile = null;
let videoObjectUrl = null;
let videoDuration = 0;
let videoTrimStart = 0;
let videoTrimDuration = 30;
let activeVideoJobId = null;
let processedVideoDownloadUrl = null;

function normalizePoleIdText(text) {
    if (!text) return "Not Found";
    const compact = String(text)
        .toUpperCase()
        .replace(/[|!]/g, "1")
        .replace(/\}/g, "7")
        .replace(/\]/g, "7")
        .replace(/\)/g, "7")
        .replace(/\{/g, "6")
        .replace(/\[/g, "6")
        .replace(/\(/g, "6")
        .replace(/[₹¥`'"]/g, "")
        .replace(/[^A-Z0-9]/g, "");

    if (compact.length < 3) return "Not Found";

    const letterView = compact
        .replace(/0/g, "O")
        .replace(/5/g, "S")
        .replace(/2/g, "Z")
        .replace(/6/g, "G")
        .replace(/8/g, "B");

    const rdss = letterView.match(/R[CDOG]SS?/);
    if (rdss) {
        const after = compact.slice(rdss.index + rdss[0].length)
            .replace(/[OQ]/g, "0")
            .replace(/(?<=\d)[IL]|[IL](?=\d)/g, "1")
            .replace(/S/g, "5")
            .replace(/B/g, "8")
            .replace(/Z/g, "2");
        const digits = after.match(/\d{1,4}/);
        return digits ? `RDSS ${digits[0]}` : "RDSS";
    }

    return compact.length <= 12 ? compact : "Not Found";
}

function sanitizeMasterResult(master) {
    if (!master) return master;
    const normalizedPoleId = normalizePoleIdText(master.pole_id);
    const textLines = Array.isArray(master.ocr_text_lines) ? master.ocr_text_lines.filter(Boolean) : [];
    const normalizedLines = textLines.length
        ? textLines.map((line, idx) => idx === 0 ? normalizedPoleId : String(line).toUpperCase().replace(/\s+/g, ' ').trim())
        : (normalizedPoleId !== "Not Found" ? [normalizedPoleId] : []);
    return { ...master, pole_id: normalizedPoleId, ocr_text_lines: normalizedLines };
}
let CLASS_OPTIONS = [
    "STRUT_POLE",
    "INS_PIN", "INS_DISC",
    "T_RISING", "TAPPING_CHANNEL", "SIDE_ARM_CHANNEL", "V_CROSS",
    "CONDUCTOR", "STREET_LIGHT", "SPECIAL_CLAMP", "STAY_SET", "DTR",
    "WIRE_BROKEN", "VEGETATION", "OBJECT"
];

// Distinct Premium Color Palette
const CLASS_COLORS = {
    "POLE": "#f97316",           // Orange
    "MAIN_POLE": "#f97316",      // Orange
    "STRUT_POLE": "#ff7f50",     // Coral/Salmon (Distinct for Strut)
    "INS_PIN": "#00ff00",        // Bright Green
    "INS_DISC": "#22c55e",       // Emerald Green
    "INSULATOR": "#00ff00",      // Default Green
    "CROSSARM": "#ff00ff",       // Magenta
    "T_RISING": "#ff00ff",       // Magenta
    "TAPPING_CHANNEL": "#d946ef", // Fuchsia
    "SIDE_ARM_CHANNEL": "#a855f7", // Purple
    "V_CROSS": "#f43f5e",        // Rose
    "CONDUCTOR": "#00ffff",      // Cyan
    "STREET_LIGHT": "#f59e0b",   // Amber
    "SPECIAL_CLAMP": "#64748b",   // Slate Gray
    "STAY_SET": "#475569",        // Dark Slate Blue
    "DTR": "#8b5cf6",            // Violet
    "WIRE_BROKEN": "#ef4444",    // Bright Red (Fault)
    "VEGETATION": "#fbbf24",     // Amber (Encroachment)
    "OBJECT": "#a8a29e"          // Stone
};

const TRAINING_CLASS_COLORS = {
    ...CLASS_COLORS,
    "POLE": "#8b5cf6",
    "INSULATOR": "#22c55e",
    "CROSSARM": "#d946ef"
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
    if (historyStack.length > 0) {
        redoStack.push(JSON.stringify(detections));
        detections = JSON.parse(historyStack.pop());
        if (activeBatchIndex !== -1) {
            batchImages[activeBatchIndex].detections = [...detections];
        }
        renderResults();
        renderBoxes();
        showToast("Undo", "primary");
    }
}

function redo() {
    if (redoStack.length > 0) {
        historyStack.push(JSON.stringify(detections));
        detections = JSON.parse(redoStack.pop());
        if (activeBatchIndex !== -1) {
            batchImages[activeBatchIndex].detections = [...detections];
        }
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
    if (batchImages[0]?.mediaType === 'video') return;
    try {
        const payload = {
            savedAt: Date.now(),
            batchImages: batchImages.map(b => ({
                src: b.src,
                name: b.file?.name || b.name || 'image.jpg',
                detections: b.detections || [],
                master: b.master || null,
                dims: b.dims || {},
                processed: b.processed || false
            })),
            activeBatchIndex,
            detections,
            masterResult,
            imageDimensions
        };
        localStorage.setItem(DRAFT_KEY, JSON.stringify(payload));
    } catch (e) {
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
        banner.onclick = () => { };

        document.getElementById('btnRestoreDraft').onclick = () => {
            const restoredMaster = sanitizeMasterResult(draft.masterResult || null);
                batchImages = draft.batchImages.map(b => ({
                file: { name: b.name },
                name: b.name,
                src: b.src,
                detections: b.detections || [],
                    master: sanitizeMasterResult(b.master || restoredMaster),
                    specific_pole_id: normalizePoleIdText(b.specific_pole_id || b.master?.pole_id || restoredMaster?.pole_id),
                    specific_ocr_text_lines: Array.isArray(b.specific_ocr_text_lines) ? b.specific_ocr_text_lines : [],
                    dims: b.dims || {},
                    processed: b.processed || false
                }));
            activeBatchIndex = draft.activeBatchIndex >= 0 ? draft.activeBatchIndex : 0;
            detections = draft.detections || [];
            masterResult = restoredMaster;
            imageDimensions = draft.imageDimensions || {};

            // Initialize history stack for restored draft state
            historyStack = [];
            redoStack = [];

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

    } catch (e) {
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

function getTrainClassCategory(cls) {
    const name = cls.toUpperCase();
    if (name.includes('INSULATOR')) return 'INSULATOR';
    if (name.includes('POLE')) return 'POLE';
    if (name.includes('CONDUCTOR') || name.includes('WIRE') || name.includes('LINE') || name.includes('CLEAT')) return 'CONDUCTOR';
    if (name.includes('ARM') || name.includes('CROSS') || name.includes('CHANNEL') || name.includes('RISING')) return 'CROSSARM';
    if (name.includes('LIGHT') || name.includes('LAMP')) return 'STREET_LIGHT';
    if (name.includes('DTR') || name.includes('TRANSFORMER') || name.includes('SWITCH')) return 'DTR';
    return 'OBJECT';
}

function openStatsModal() {
    const modal = document.getElementById('statsModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        loadUserTrainingStats();
    }
}

function closeStatsModal() {
    const modal = document.getElementById('statsModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

function toggleActiveDatasetCollapse() {
    const list = document.getElementById('userActiveDatasetList');
    const icon = document.getElementById('datasetCollapseIcon');
    if (list && icon) {
        if (list.classList.contains('hidden')) {
            list.classList.remove('hidden');
            icon.className = 'fa-solid fa-chevron-up text-[10px] text-gray-500';
        } else {
            list.classList.add('hidden');
            icon.className = 'fa-solid fa-chevron-down text-[10px] text-gray-500';
        }
    }
}

async function loadUserTrainingStats() {
    const panel = document.getElementById('statClassGrid');
    if (!panel) return;

    try {
        const res = await fetch('/api/training_stats');
        if (!res.ok) return;

        const s = await res.json();
        const threshold = s.threshold || 50;
        const totalSamples = s.total_samples || 0;
        const byClass = s.by_class || {};
        const imagesPerClass = s.images_per_class || {};
        const classConfidence = s.class_confidence || {};
        const datasetList = document.getElementById('userActiveDatasetList');

        // Overview cards
        const el = (id) => document.getElementById(id);
        if (el('statTotalImages'))  el('statTotalImages').textContent  = (s.trained_images || 0).toLocaleString();
        if (el('statTotalClasses')) el('statTotalClasses').textContent = s.total_classes || 0;
        if (el('statTotalObjects')) el('statTotalObjects').textContent = (s.total_annotations || 0).toLocaleString();
        if (el('statAvgConf'))      el('statAvgConf').textContent      = typeof s.avg_confidence === 'number' ? `${Math.round(s.avg_confidence * 100)}%` : '—';

        // Progress bar
        if (el('statProgressLabel')) el('statProgressLabel').textContent = `${totalSamples} / ${threshold}`;
        if (el('statProgressBar'))   el('statProgressBar').style.width = `${Math.min(100, (totalSamples / threshold) * 100)}%`;
        if (el('statLastApproved'))  el('statLastApproved').textContent  = s.last_approved ? `Last: ${s.last_approved}` : 'No approvals yet';

        // Active datasets
        if (datasetList) {
            datasetList.innerHTML = (s.datasets || []).map(ds => `
                <div class="p-3 rounded-xl border border-white/5 bg-gray-900/50 flex flex-col gap-1">
                    <div class="text-[9px] text-white font-bold truncate">${ds.name}</div>
                    <div class="flex items-center gap-3 mt-1">
                        <span class="text-[8px] text-blue-400 font-mono font-bold">${(ds.images || 0).toLocaleString()} images</span>
                        <span class="text-[8px] text-emerald-400 font-mono font-bold">${(ds.annotations || 0).toLocaleString()} objects</span>
                    </div>
                    <div class="text-[7px] text-gray-600 uppercase font-bold tracking-widest truncate">${ds.path || ''}</div>
                </div>
            `).join('') || '<div class="text-[9px] text-gray-600 italic">No active datasets found.</div>';
        }

        // Class breakdown — sorted by count descending
        if (Object.keys(byClass).length === 0) {
            panel.innerHTML = '<div class="text-[9px] text-gray-600 italic col-span-full">No training samples yet.</div>';
            return;
        }

        const maxCount = Math.max(...Object.values(byClass), 1);
        const CLASS_COLORS = {
            main_pole: '#60a5fa', strut_pole: '#a78bfa', pole: '#60a5fa',
            insulator: '#34d399', crossarm: '#fb923c', conductor: '#fbbf24',
            street_light: '#f472b6', v_cross_arm: '#fb923c', tapping_arm: '#f97316',
            side_arm: '#ef4444', t_rising: '#e879f9', insulators: '#34d399',
            top_cleat: '#94a3b8', special_clamp: '#64748b', stay_set: '#475569',
            box_arm: '#c084fc', ab_switch: '#38bdf8', dtr: '#2dd4bf',
        };

        const sorted = Object.entries(byClass).sort(([, a], [, b]) => b - a);

        panel.innerHTML = sorted.map(([cls, count]) => {
            const color = CLASS_COLORS[cls] || '#94a3b8';
            const conf = typeof classConfidence[cls] === 'number'
                ? `${Math.round(classConfidence[cls] * 100)}%`
                : '—';
            const imgCount = imagesPerClass[cls] || 0;
            const barW = Math.round((count / maxCount) * 100);
            const displayName = cls.replace(/_/g, ' ');

            return `<div class="p-3 bg-black/25 rounded-xl border border-white/5 flex flex-col gap-2">
                <div class="flex items-center justify-between gap-1">
                    <span class="text-[9px] font-bold capitalize truncate leading-tight" style="color:${color}">${displayName}</span>
                    <span class="text-[9px] font-mono font-bold text-white shrink-0">${count.toLocaleString()}</span>
                </div>
                <div class="w-full bg-gray-800/60 rounded-full h-[3px]">
                    <div class="h-[3px] rounded-full transition-all duration-500" style="width:${barW}%;background:${color}"></div>
                </div>
                <div class="flex flex-col gap-0.5">
                    <div class="text-[7.5px] text-gray-400 font-semibold tracking-wide">identified: <span class="text-white font-mono">${count.toLocaleString()}</span></div>
                    <div class="text-[7.5px] text-gray-400 font-semibold tracking-wide">images: <span class="text-blue-300 font-mono">${imgCount.toLocaleString()}</span></div>
                    <div class="text-[7.5px] text-gray-400 font-semibold tracking-wide">confidence: <span class="text-amber-400 font-mono">${conf}</span></div>
                </div>
            </div>`;
        }).join('');

    } catch (e) {
        panel.innerHTML = '<div class="text-[9px] text-gray-600 italic col-span-full">Training stats unavailable.</div>';
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('upload');
    const videoInput = document.getElementById('videoUpload');
    const previewImg = document.getElementById('preview');
    const dropZone = document.getElementById('dropZone');
    const trimSlider = document.getElementById('videoTrimSlider');
    const trimTrack = document.getElementById('videoTrimTrack');
    const profileMenuButton = document.getElementById('profileMenuButton');
    const profileMenu = document.getElementById('profileMenu');
    const gpuToggle = document.getElementById('gpuToggle');

    // Check for saved draft on page load
    checkAndRestoreDraft();
    loadUserTrainingStats();
    if (gpuToggle) loadRuntimeDeviceStatus();

    profileMenuButton?.addEventListener('click', (event) => {
        event.stopPropagation();
        const opening = profileMenu?.classList.contains('hidden');
        profileMenu?.classList.toggle('hidden', !opening);
        profileMenuButton.setAttribute('aria-expanded', String(Boolean(opening)));
    });
    profileMenu?.addEventListener('click', event => event.stopPropagation());
    document.addEventListener('click', () => {
        profileMenu?.classList.add('hidden');
        profileMenuButton?.setAttribute('aria-expanded', 'false');
    });
    gpuToggle?.addEventListener('click', toggleGpuAcceleration);

    // Auto-save to localStorage every 10 seconds if work is in progress
    setInterval(() => {
        if (batchImages.some(b => b.processed) || detections.length > 0) {
            persistDraftToStorage();
        }
    }, 10000);

    setInterval(loadUserTrainingStats, 30000);

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

    if (videoInput) {
        videoInput.addEventListener('change', handleVideoUpload);
    }

    if (trimSlider) {
        trimSlider.addEventListener('input', (e) => {
            videoTrimStart = parseFloat(e.target.value || '0');
            videoTrimDuration = Math.min(videoTrimDuration, Math.max(0, videoDuration - videoTrimStart));
            updateVideoTrimUI();
            appendVideoLog(`Trim window set to ${formatSeconds(videoTrimStart)} - ${formatSeconds(videoTrimStart + videoTrimDuration)} (${videoTrimDuration.toFixed(1)} sec)`, 'info');
        });
    }

    if (trimTrack) {
        const MIN_TRIM_SECONDS = 3;
        const MAX_TRIM_SECONDS = 30;
        const pointerToSeconds = (event) => {
            const rect = trimTrack.getBoundingClientRect();
            const clientX = event.touches ? event.touches[0].clientX : event.clientX;
            const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
            return ratio * videoDuration;
        };
        const setTrimFromPointer = (event, mode, startSnapshot, durationSnapshot, pointerStartSeconds) => {
            if (!videoDuration) return;
            const pointerSeconds = pointerToSeconds(event);
            const delta = pointerSeconds - pointerStartSeconds;
            const maxDuration = Math.min(MAX_TRIM_SECONDS, videoDuration);

            if (mode === 'resize-left') {
                const originalEnd = startSnapshot + durationSnapshot;
                const nextStart = Math.max(0, Math.min(originalEnd - MIN_TRIM_SECONDS, startSnapshot + delta));
                videoTrimStart = nextStart;
                videoTrimDuration = Math.max(MIN_TRIM_SECONDS, Math.min(maxDuration, originalEnd - nextStart));
            } else if (mode === 'resize-right') {
                const nextDuration = Math.max(MIN_TRIM_SECONDS, Math.min(maxDuration, durationSnapshot + delta));
                videoTrimDuration = Math.min(nextDuration, videoDuration - startSnapshot);
                videoTrimStart = startSnapshot;
            } else {
                const maxStart = Math.max(0, videoDuration - durationSnapshot);
                videoTrimStart = Math.max(0, Math.min(maxStart, startSnapshot + delta));
                videoTrimDuration = Math.min(durationSnapshot, videoDuration - videoTrimStart);
            }
            updateVideoTrimUI();
        };

        trimTrack.addEventListener('pointerdown', (event) => {
            event.preventDefault();
            const target = event.target;
            const mode = target.classList.contains('left')
                ? 'resize-left'
                : target.classList.contains('right')
                    ? 'resize-right'
                    : 'move';
            const startSnapshot = videoTrimStart;
            const durationSnapshot = videoTrimDuration;
            const pointerStartSeconds = pointerToSeconds(event);
            setTrimFromPointer(event, mode, startSnapshot, durationSnapshot, pointerStartSeconds);
            const onMove = (moveEvent) => setTrimFromPointer(moveEvent, mode, startSnapshot, durationSnapshot, pointerStartSeconds);
            const onUp = () => {
                document.removeEventListener('pointermove', onMove);
                document.removeEventListener('pointerup', onUp);
                appendVideoLog(`Trim window set to ${formatSeconds(videoTrimStart)} - ${formatSeconds(videoTrimStart + videoTrimDuration)} (${videoTrimDuration.toFixed(1)} sec)`, 'info');
            };
            document.addEventListener('pointermove', onMove);
            document.addEventListener('pointerup', onUp);
        });
    }

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (e.target.closest('#videoTrimPanel') || e.target.closest('#videoPreview')) return;
            if (currentInputMode === 'video') {
                videoInput.click();
            } else {
                // Allow adding more images even if batch is active
                uploadInput.click();
            }
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
                if (currentInputMode === 'video') {
                    videoInput.files = e.dataTransfer.files;
                    handleVideoUpload({ target: videoInput });
                } else {
                    uploadInput.files = e.dataTransfer.files;
                    handleUpload({ target: uploadInput });
                }
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

function renderRuntimeDeviceStatus(status) {
    const toggle = document.getElementById('gpuToggle');
    const statusText = document.getElementById('gpuStatusText');
    if (!toggle || !statusText) return;
    const available = Boolean(status?.available);
    const enabled = Boolean(status?.enabled && available);
    toggle.disabled = !available;
    toggle.classList.toggle('is-active', enabled);
    toggle.setAttribute('aria-checked', String(enabled));
    statusText.textContent = !available ? 'GPU unavailable - CPU mode' : (enabled ? 'GPU active' : 'CPU mode');
    statusText.className = `text-[8px] uppercase tracking-widest mt-1 ${enabled ? 'text-emerald-400' : 'text-gray-500'}`;
}

async function loadRuntimeDeviceStatus() {
    try {
        const response = await fetch('/api/runtime_device', { cache: 'no-store' });
        if (!response.ok) throw new Error('Unable to read GPU status');
        renderRuntimeDeviceStatus(await response.json());
    } catch (error) {
        const statusText = document.getElementById('gpuStatusText');
        if (statusText) statusText.textContent = 'GPU status unavailable';
    }
}

async function toggleGpuAcceleration() {
    const toggle = document.getElementById('gpuToggle');
    if (!toggle || toggle.disabled) return;
    const nextEnabled = toggle.getAttribute('aria-checked') !== 'true';
    toggle.disabled = true;
    try {
        const response = await fetch('/api/runtime_device', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: nextEnabled })
        });
        const status = await response.json();
        if (!response.ok) throw new Error(status.error || 'Unable to change GPU mode');
        renderRuntimeDeviceStatus(status);
        showToast(status.enabled ? 'GPU acceleration enabled' : 'CPU mode enabled', 'success');
    } catch (error) {
        showToast(error.message || 'Unable to change GPU mode', 'danger');
        await loadRuntimeDeviceStatus();
    }
}

function switchInputMode(mode) {
    if (!['image', 'video'].includes(mode) || currentInputMode === mode) return;
    resetSession(true);
    currentInputMode = mode;

    const isVideo = mode === 'video';
    document.getElementById('btnImageMode')?.classList.toggle('active', !isVideo);
    document.getElementById('btnVideoMode')?.classList.toggle('active', isVideo);
    document.getElementById('uploadPromptTitle').textContent = isVideo ? 'Select inspection video' : 'Select inspection image(s)';
    document.getElementById('uploadPromptSubtitle').textContent = isVideo ? 'DRAG AND DROP OR CLICK - ONLY 30 SEC WILL BE PROCESSED' : 'DRAG AND DROP OR CLICK';
    document.querySelector('#btnRun .btn-text').textContent = isVideo ? 'Run Video Interface' : 'Run Photo Interface';

    const drawBtn = document.getElementById('btnDrawMode');
    if (drawBtn) {
        drawBtn.disabled = isVideo;
        drawBtn.classList.toggle('opacity-40', isVideo);
        drawBtn.classList.toggle('cursor-not-allowed', isVideo);
    }
}

function formatSeconds(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const mins = Math.floor(safe / 60);
    const secs = (safe % 60).toFixed(1).padStart(4, '0');
    return mins > 0 ? `${mins}:${secs}` : `${safe.toFixed(1)}s`;
}

function resetVideoLogs(status = 'Idle') {
    setVideoProgress(0, 'Waiting');
    setVideoButtonProgressVisible(false);
}

function setVideoProgress(percent, message = '') {
    const bar = document.getElementById('videoProgressBar');
    const pct = document.getElementById('videoProgressPercent');
    const safePercent = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
    if (bar) bar.style.width = `${safePercent}%`;
    if (pct) pct.textContent = `${safePercent}%`;
}

function setVideoButtonProgressVisible(isVisible) {
    const wrap = document.getElementById('videoProgressWrap');
    const btn = document.getElementById('btnRun');
    if (wrap) wrap.classList.toggle('hidden', !isVisible);
    if (btn) btn.classList.toggle('is-video-processing', isVisible);
}

function appendVideoLog(message, level = 'info') {
    console.debug('[Video]', level, message);
}

function updateVideoTrimUI() {
    const slider = document.getElementById('videoTrimSlider');
    const label = document.getElementById('videoTrimLabel');
    const durationLabel = document.getElementById('videoDurationLabel');
    const trimBar = document.getElementById('videoTrimBar');
    const trimWindow = document.getElementById('videoTrimWindow');
    const trimHandleLabel = document.getElementById('videoTrimHandleLabel');
    const trimLength = Math.min(videoTrimDuration, Math.max(0, videoDuration - videoTrimStart));

    if (label) {
        label.textContent = `${formatSeconds(videoTrimStart)} - ${formatSeconds(videoTrimStart + trimLength)} (${trimLength.toFixed(1)} sec selected)`;
    }
    if (durationLabel) {
        durationLabel.textContent = `${formatSeconds(videoDuration)} total`;
    }
    if (trimBar) {
        const pct = videoDuration > 0 ? Math.min(100, (trimLength / videoDuration) * 100) : 100;
        trimBar.style.width = `${pct}%`;
        trimBar.style.marginLeft = videoDuration > 0 ? `${Math.min(100, (videoTrimStart / videoDuration) * 100)}%` : '0%';
    }
    if (trimWindow) {
        const widthPct = videoDuration > 0 ? Math.min(100, (trimLength / videoDuration) * 100) : 100;
        const leftPct = videoDuration > 0 ? Math.min(100 - widthPct, (videoTrimStart / videoDuration) * 100) : 0;
        trimWindow.style.width = `${widthPct}%`;
        trimWindow.style.left = `${leftPct}%`;
    }
    if (trimHandleLabel) {
        trimHandleLabel.textContent = `${trimLength.toFixed(1)} sec`;
    }
    if (slider) {
        slider.max = Math.max(0, videoDuration - trimLength).toFixed(1);
    }
    if (slider && Number(slider.value) !== videoTrimStart) {
        slider.value = String(videoTrimStart);
    }
    syncVideoPreviewToTrim();
    updateVideoPreviewControls();
}

function getVideoTrimEnd() {
    return Math.min(videoDuration || 0, videoTrimStart + videoTrimDuration);
}

function isShowingProcessedVideo() {
    return Boolean(batchImages[0]?.mediaType === 'video' && batchImages[0]?.processed);
}

function syncVideoPreviewToTrim(force = false) {
    const video = document.getElementById('videoPreview');
    if (!video || !uploadedVideoFile || isShowingProcessedVideo() || !videoDuration) return;
    const trimEnd = getVideoTrimEnd();
    if (force || video.currentTime < videoTrimStart || video.currentTime > trimEnd) {
        video.currentTime = videoTrimStart;
    }
    updateVideoPreviewControls();
}

function getVideoPreviewClipDuration() {
    const video = document.getElementById('videoPreview');
    if (isShowingProcessedVideo()) {
        return Number.isFinite(video?.duration) ? video.duration : 0;
    }
    return Math.min(videoTrimDuration, Math.max(0, videoDuration - videoTrimStart));
}

function getVideoPreviewClipElapsed() {
    const video = document.getElementById('videoPreview');
    if (!video) return 0;
    if (isShowingProcessedVideo()) {
        return Number.isFinite(video.currentTime) ? video.currentTime : 0;
    }
    return Math.max(0, Math.min(getVideoPreviewClipDuration(), video.currentTime - videoTrimStart));
}

function updateVideoPreviewControls() {
    const video = document.getElementById('videoPreview');
    const scrubber = document.getElementById('videoPreviewScrubber');
    const timeLabel = document.getElementById('videoPreviewTimeLabel');
    const playBtn = document.getElementById('videoPreviewPlayBtn');
    const downloadBtn = document.getElementById('videoDownloadBtn');
    const duration = getVideoPreviewClipDuration();
    const elapsed = getVideoPreviewClipElapsed();

    if (scrubber) {
        scrubber.max = duration ? String(duration) : '0';
        scrubber.value = String(Math.min(elapsed, duration || 0));
    }
    if (timeLabel) {
        timeLabel.textContent = `${formatSeconds(elapsed)} / ${formatSeconds(duration)}`;
    }
    if (playBtn) {
        playBtn.innerHTML = `<i class="fa-solid ${video && !video.paused ? 'fa-pause' : 'fa-play'}"></i>`;
    }
    updateVideoPreviewMuteControl();
    if (downloadBtn) {
        downloadBtn.classList.toggle('hidden', !processedVideoDownloadUrl || !isShowingProcessedVideo());
    }
}

function updateVideoPreviewMuteControl() {
    const video = document.getElementById('videoPreview');
    const muteBtn = document.getElementById('videoPreviewMuteBtn');
    if (!muteBtn) return;
    const isMuted = !video || video.muted || video.volume === 0;
    muteBtn.innerHTML = `<i class="fa-solid ${isMuted ? 'fa-volume-xmark' : 'fa-volume-high'}"></i>`;
    muteBtn.title = isMuted ? 'Unmute video' : 'Mute video';
    muteBtn.setAttribute('aria-label', muteBtn.title);
    muteBtn.classList.toggle('is-muted', isMuted);
}

function toggleVideoPreviewMute(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    const video = document.getElementById('videoPreview');
    if (!video) return;
    const isEffectivelyMuted = video.muted || video.volume === 0;
    if (isEffectivelyMuted) {
        if (video.volume === 0) video.volume = 1;
        video.muted = false;
    } else {
        video.muted = true;
    }
    updateVideoPreviewMuteControl();
}

function downloadProcessedVideo(event) {
    if (event) event.stopPropagation();
    if (!processedVideoDownloadUrl) {
        showToast("No processed video available to download", "warning");
        return;
    }
    const link = document.createElement('a');
    link.href = processedVideoDownloadUrl;
    link.download = `processed-video-${Date.now()}.webm`;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function toggleVideoPreviewPlayback(event) {
    if (event) event.stopPropagation();
    const video = document.getElementById('videoPreview');
    if (!video) return;
    if (!isShowingProcessedVideo() && (video.currentTime < videoTrimStart || video.currentTime >= getVideoTrimEnd())) {
        video.currentTime = videoTrimStart;
    }
    if (video.paused) {
        video.play();
    } else {
        video.pause();
    }
    updateVideoPreviewControls();
}

function seekVideoPreviewClip(value) {
    const video = document.getElementById('videoPreview');
    if (!video) return;
    const elapsed = Math.max(0, Number(value) || 0);
    video.currentTime = isShowingProcessedVideo() ? elapsed : videoTrimStart + elapsed;
    updateVideoPreviewControls();
}

function seekToVideoDetection(frameTime) {
    const video = document.getElementById('videoPreview');
    if (!video || !Number.isFinite(Number(frameTime))) return;

    const trimStart = Number(batchImages[0]?.trimStart || 0);
    const targetTime = isShowingProcessedVideo()
        ? Math.max(0, Number(frameTime) - trimStart)
        : Math.max(0, Number(frameTime));
    const duration = Number.isFinite(video.duration) ? video.duration : targetTime;

    video.currentTime = Math.min(targetTime, Math.max(0, duration - 0.01));
    video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    video.pause();
    updateVideoPreviewControls();
}

function handleVideoUpload(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    uploadedVideoFile = file;
    videoObjectUrl = URL.createObjectURL(file);
    videoDuration = 0;
    videoTrimStart = 0;
    videoTrimDuration = 30;
    processedVideoDownloadUrl = null;
    resetVideoLogs('Ready');
    appendVideoLog(`Selected video: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`, 'info');

    batchImages = [{
        file,
        name: file.name,
        src: videoObjectUrl,
        detections: [],
        master: null,
        dims: { width: 0, height: 0 },
        processed: false,
        mediaType: 'video'
    }];
    activeBatchIndex = 0;
    detections = [];
    masterResult = null;
    imageDimensions = { width: 0, height: 0 };

    const video = document.getElementById('videoPreview');
    video.muted = false;
    if (video.volume === 0) video.volume = 1;
    video.src = videoObjectUrl;
    video.load();
    video.ontimeupdate = () => {
        if (!uploadedVideoFile || !videoDuration) {
            updateVideoPreviewControls();
            return;
        }
        if (!isShowingProcessedVideo() && video.currentTime >= getVideoTrimEnd()) {
            video.pause();
            video.currentTime = videoTrimStart;
        }
        updateVideoPreviewControls();
    };
    video.onplay = updateVideoPreviewControls;
    video.onpause = updateVideoPreviewControls;
    video.onvolumechange = updateVideoPreviewMuteControl;
    video.onloadeddata = updateVideoPreviewControls;
    video.onloadedmetadata = () => {
        videoDuration = Number.isFinite(video.duration) ? video.duration : 0;
        videoTrimDuration = Math.min(30, videoDuration || 30);
        const slider = document.getElementById('videoTrimSlider');
        if (slider) {
            slider.max = Math.max(0, videoDuration - videoTrimDuration).toFixed(1);
            slider.value = '0';
        }
        document.getElementById('videoTrimPanel')?.classList.remove('hidden');
        updateVideoTrimUI();
        syncVideoPreviewToTrim(true);
        appendVideoLog(`Loaded metadata: ${formatSeconds(videoDuration)} total duration`, 'success');
        appendVideoLog('Drag the trim window to move it, or drag either edge to squeeze the duration.', 'info');
    };

    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('videoContainer').classList.remove('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');

    const dz = document.getElementById('dropZone');
    dz.classList.add('py-4', 'border-transparent');
    dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30');

    renderResults();
    e.target.value = '';
}

function processMedia() {
    if (currentInputMode === 'video') {
        return processVideo();
    }
    return processImage();
}

function handleUpload(e) {
    if (currentInputMode !== 'image') return;
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
    
    const dz = document.getElementById('dropZone');
    dz.classList.add('py-4', 'border-transparent');
    dz.classList.remove('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');

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

    // Add "Plus" button for adding more views
    if (batchImages.length < 3) {
        const addBtn = document.createElement('div');
        addBtn.className = 'batch-thumb-add group h-[40px] w-[40px] shrink-0 cursor-pointer';
        addBtn.title = "Add another view (Max 3)";
        addBtn.onclick = () => document.getElementById('upload').click();
        addBtn.innerHTML = `
            <div class="w-full h-full rounded-lg border-2 border-dashed border-white/10 flex items-center justify-center group-hover:border-blue-500/50 group-hover:bg-blue-500/5 transition-all">
                <i class="fa-solid fa-plus text-[10px] text-gray-500 group-hover:text-blue-400"></i>
            </div>
        `;
        strip.appendChild(addBtn);
    }

    strip.appendChild(document.createElement('div')); // Spacer
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

    // Save current active state before switching (only if index is different)
    if (activeBatchIndex !== -1 && activeBatchIndex !== index) {
        batchImages[activeBatchIndex].detections = [...detections];
        batchImages[activeBatchIndex].master = masterResult;
        batchImages[activeBatchIndex].dims = { ...imageDimensions };
    }

    activeBatchIndex = index;
    const item = batchImages[index];

    // Load new image state
    uploadedFile = item.file;
    detections = [...item.detections];
    if (item.master) {
        masterResult = item.master;
    } else if (masterResult) {
        item.master = masterResult;
    }
    imageDimensions = { ...item.dims };

    // Initialize history stack for the new image state to enable Undo/Redo
    historyStack = [];
    redoStack = [];

    // Update UI
    document.getElementById('preview').src = item.src;
    renderBatchStrip();
    renderResults();
    renderBoxes();
}

function resetSession(force = false) {
    if (!force && batchImages.length > 0) {
        if (!confirm("Clear current batch and all detection results?")) return;
    }

    batchImages = [];
    activeBatchIndex = -1;
    detections = [];
    uploadedFile = null;
    uploadedVideoFile = null;
    masterResult = null;
    videoDuration = 0;
    videoTrimStart = 0;
    videoTrimDuration = 30;
    processedVideoDownloadUrl = null;
    if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
    }

    // UI Reset
    document.getElementById('uploadPrompt').classList.remove('hidden');
    document.getElementById('batchStripWrapper').classList.add('hidden');
    document.getElementById('imageContainer').classList.add('hidden');
    document.getElementById('videoContainer')?.classList.add('hidden');
    document.getElementById('videoTrimPanel')?.classList.add('hidden');
    const video = document.getElementById('videoPreview');
    if (video) {
        video.pause();
        video.ontimeupdate = null;
        video.onplay = null;
        video.onpause = null;
        video.onvolumechange = null;
        video.onloadeddata = null;
        video.onloadedmetadata = null;
        video.removeAttribute('src');
        video.load();
    }
    updateVideoPreviewControls();
    document.getElementById('resultBox').innerHTML = '';
    document.getElementById('masterIdentityCard').classList.add('hidden');
    const dz = document.getElementById('dropZone');
    dz.classList.remove('py-4', 'border-transparent');
    dz.classList.add('p-10', 'border-dashed', 'border-white/5', 'hover:border-blue-500/30', 'cursor-pointer');
    document.getElementById('upload').value = ''; // Clear file input
    document.getElementById('videoUpload').value = '';
    setVideoButtonProgressVisible(false);

    // Hide and reset Submit Section
    const submitSection = document.getElementById('submitSection');
    if (submitSection) submitSection.classList.add('hidden');
    
    const finalSubmitBtn = document.getElementById('finalSubmitBtn');
    if (finalSubmitBtn) {
        finalSubmitBtn.disabled = true;
        finalSubmitBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> SUBMIT RESULTS';
    }

    document.querySelector('#btnRun .btn-text').textContent = currentInputMode === 'video' ? 'Run Video Interface' : 'Run Photo Interface';
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
    if (batchImages.length === 0) {
        showToast("Please upload at least one image", "warning");
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    btn.disabled = true;
    loader.classList.remove('hidden');

    try {
        const formData = new FormData();
        
        // If there are multiple images, we send them as image1, image2, image3
        // If only one, we send as 'image' for backward compatibility
        if (batchImages.length === 1) {
            btnText.textContent = "Analyzing Image...";
            let file = batchImages[0].file;
            let imageToUpload = file;
            if (file.size > 1024 * 1024) {
                imageToUpload = await resizeImage(file, 1280, 1280);
            }
            formData.append("image", imageToUpload, "image.jpg");
        } else {
            btnText.textContent = `Merging ${batchImages.length} Views...`;
            for (let i = 0; i < Math.min(batchImages.length, 3); i++) {
                let file = batchImages[i].file;
                let imageToUpload = file;
                // Resizing slightly more for multi-image to ensure high speed
                if (file.size > 0.5 * 1024 * 1024) {
                    imageToUpload = await resizeImage(file, 1024, 1024);
                }
                formData.append(`image${i+1}`, imageToUpload, `image${i+1}.jpg`);
            }
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000); // Higher timeout for multi-image

        const response = await fetch("/predict", {
            method: "POST",
            headers: { "ngrok-skip-browser-warning": "69420" },
            body: formData,
            signal: controller.signal
        }).finally(() => clearTimeout(timeoutId));

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            showToast(`Analysis Failed: ${errorData.error || response.status}`, "danger");
            return;
        }

        const data = await response.json();

        // Handle Merged vs Single Result
        if (batchImages.length > 1) {
            masterResult = sanitizeMasterResult(data.master);
            // Update all images in batch with their specific local detections and master result
            batchImages.forEach((img, idx) => {
                img.processed = true;
                img.master = { ...masterResult }; // Keep the consolidated master result
                
                if (data.all_pole_ids && data.all_pole_ids[idx]) {
                    img.specific_pole_id = normalizePoleIdText(data.all_pole_ids[idx]);
                } else {
                    img.specific_pole_id = "Not Found";
                }
                img.specific_ocr_text_lines = Array.isArray(data.all_ocr_text_lines?.[idx]) ? data.all_ocr_text_lines[idx] : [];

                if (data.all_images && data.all_images[idx]) {
                    img.src = 'data:image/jpeg;base64,' + data.all_images[idx];
                }
                if (data.all_detections && data.all_detections[idx]) {
                    img.detections = data.all_detections[idx].map(d => ({
                        ...d,
                        label: d.label.toUpperCase(),
                        confirmed: false
                    }));
                }
                if (data.all_dims && data.all_dims[idx]) {
                    img.dims = data.all_dims[idx];
                }
            });
            
            // Map merged detections to the active view correctly
            detections = [...batchImages[activeBatchIndex].detections];
            imageDimensions = { ...batchImages[activeBatchIndex].dims };
            batchImages[activeBatchIndex].processed = true;

        } else {
            // Single image path
            batchImages[0].detections = data.detections.map(d => ({
                ...d,
                label: d.label.toUpperCase(),
                confirmed: false
            }));
            batchImages[0].master = sanitizeMasterResult(data.master);
            batchImages[0].specific_pole_id = batchImages[0].master.pole_id || "Not Found";
            batchImages[0].specific_ocr_text_lines = batchImages[0].master.ocr_text_lines || [];
            batchImages[0].dims = { width: data.width, height: data.height };
            batchImages[0].processed = true;
            
            if (data.annotated_image) {
                batchImages[0].src = 'data:image/jpeg;base64,' + data.annotated_image;
            }
            
            detections = [...batchImages[0].detections];
            masterResult = batchImages[0].master;
            imageDimensions = { ...batchImages[0].dims };
        }

        // Refresh UI
        selectBatchImage(activeBatchIndex !== -1 ? activeBatchIndex : 0);
        
        // FORCED REFRESH: Ensure side panel and boxes appear
        renderResults();
        renderBoxes();
        
        document.getElementById("imageContainer").classList.remove("hidden");
        document.getElementById("submitSection").classList.remove("hidden");
        
        renderBatchStrip();
        saveDraft();
        showToast(batchImages.length > 1 ? "Multi-view merge complete" : "Analysis complete", "success");

    } catch (err) {
        showToast(err.message || "Processing error", "danger");
    } finally {
        btn.disabled = false;
        btnText.textContent = "Run Interface";
        loader.classList.add('hidden');
    }
}

async function processVideo() {
    if (!uploadedVideoFile) {
        showToast("Please upload a video first", "warning");
        return;
    }

    const btn = document.getElementById('btnRun');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    const clipDuration = Math.min(videoTrimDuration, Math.max(0, videoDuration - videoTrimStart));

    btn.disabled = true;
    loader.classList.remove('hidden');
    btnText.textContent = `Processing ${clipDuration.toFixed(1)} sec clip...`;
    resetVideoLogs('Processing');
    setVideoButtonProgressVisible(true);
    setVideoProgress(0, 'Starting video job');
    appendVideoLog(`Preparing upload for ${formatSeconds(videoTrimStart)} - ${formatSeconds(videoTrimStart + clipDuration)} (${clipDuration.toFixed(1)} sec)`, 'active');
    appendVideoLog('Sending video to backend endpoint /predict_video...', 'active');

    let progressTimer = null;
    try {
        activeVideoJobId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `video-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const formData = new FormData();
        formData.append("video", uploadedVideoFile, uploadedVideoFile.name || "inspection-video.mp4");
        formData.append("trim_start", String(videoTrimStart || 0));
        formData.append("trim_duration", String(videoTrimDuration || 30));
        formData.append("job_id", activeVideoJobId);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 240000);

        const response = await fetch("/predict_video", {
            method: "POST",
            headers: { "ngrok-skip-browser-warning": "69420" },
            body: formData,
            signal: controller.signal
        }).finally(() => {
            clearTimeout(timeoutId);
            if (progressTimer) clearInterval(progressTimer);
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            appendVideoLog(`Backend returned error: ${errorData.error || response.status}`, 'error');
            showToast(`Video analysis failed: ${errorData.error || response.status}`, "danger");
            return;
        }

        const queuedJob = await response.json();
        activeVideoJobId = queuedJob.job_id || activeVideoJobId;
        appendVideoLog(`Video job queued: ${activeVideoJobId}`, 'active');

        const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
        const deadline = Date.now() + 900000;
        let data = null;
        let lastProgressMessage = '';
        while (Date.now() < deadline) {
            const progressRes = await fetch(`/api/video_progress/${activeVideoJobId}`, {
                headers: { "ngrok-skip-browser-warning": "69420" }
            });
            if (!progressRes.ok) {
                await sleep(1000);
                continue;
            }
            const progress = await progressRes.json();
            setVideoProgress(progress.progress ?? progress.percent ?? 0, progress.message || 'Processing');
            if (progress.message && progress.message !== lastProgressMessage) {
                appendVideoLog(progress.message, progress.status === 'failed' ? 'error' : (progress.status === 'complete' ? 'success' : 'active'));
                lastProgressMessage = progress.message;
            }
            if (progress.status === 'complete') {
                data = progress.result || {};
                break;
            }
            if (progress.status === 'failed') {
                throw new Error(progress.error || 'Video processing failed');
            }
            await sleep(1000);
        }
        if (!data) {
            throw new Error('Video processing timed out');
        }

        setVideoProgress(100, 'Video analysis complete');
        appendVideoLog(`Backend response received: ${data.processed_frames || 0} frames processed`, 'success');
        const processedBaseUrl = data.video_url || data.processed_video_url;
        const processedUrl = `${processedBaseUrl}?t=${Date.now()}`;
        processedVideoDownloadUrl = processedBaseUrl;
        const video = document.getElementById('videoPreview');
        video.src = processedUrl;
        video.load();
        video.onloadedmetadata = updateVideoPreviewControls;
        video.ontimeupdate = updateVideoPreviewControls;
        video.onplay = updateVideoPreviewControls;
        video.onpause = updateVideoPreviewControls;
        video.onvolumechange = updateVideoPreviewMuteControl;
        appendVideoLog('Processed video loaded into preview player.', 'success');

        const videoDetections = (data.detections || []).map(d => ({
            ...d,
            label: String(d.label || '').toUpperCase(),
            confirmed: false
        }));

        batchImages = [{
            file: uploadedVideoFile,
            name: uploadedVideoFile.name,
            src: processedUrl,
            detections: videoDetections,
            master: sanitizeMasterResult(data.master),
            dims: { width: data.width, height: data.height },
            processed: true,
            mediaType: 'video',
            classCounts: data.class_counts || {},
            detectedClasses: data.detected_classes || Object.keys(data.frame_detection_counts || {}),
            frameDetectionCounts: data.frame_detection_counts || {},
            trimStart: data.trim_start,
            trimDuration: data.trim_duration || videoTrimDuration
        }];
        activeBatchIndex = 0;
        detections = [...videoDetections];
        masterResult = batchImages[0].master;
        surveyResult = data.survey_questionnaire || {};
        imageDimensions = { ...batchImages[0].dims };

        document.getElementById("videoContainer").classList.remove("hidden");
        document.getElementById("videoTrimPanel")?.classList.add("hidden");
        document.getElementById("imageContainer").classList.add("hidden");
        document.getElementById("submitSection").classList.remove("hidden");
        renderResults();
        appendVideoLog(`Detected ${videoDetections.length} pole track result${videoDetections.length === 1 ? '' : 's'} across the selected clip.`, 'success');
        showToast("Video analysis complete", "success");
    } catch (err) {
        if (progressTimer) clearInterval(progressTimer);
        const message = err.name === 'AbortError' ? "Video processing timed out" : (err.message || "Video processing error");
        setVideoProgress(100, 'Video analysis failed');
        appendVideoLog(message, 'error');
        showToast(message, "danger");
    } finally {
        if (progressTimer) clearInterval(progressTimer);
        btn.disabled = false;
        btnText.textContent = "Run Video Interface";
        loader.classList.add('hidden');
        setVideoButtonProgressVisible(false);
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
        masterResult = sanitizeMasterResult(masterResult);
        document.getElementById("masterClass").textContent = masterResult.final_class.replace(/_/g, ' ');
        document.getElementById("masterVoltage").textContent = masterResult.voltage;
        
        const poleIdEl = document.getElementById("masterPoleId");
        if (poleIdEl) {
            poleIdEl.textContent = normalizePoleIdText(masterResult.pole_id);
        }
        const poleTextLinesRow = document.getElementById("poleTextLinesRow");
        const poleExtraLinesEl = document.getElementById("masterPoleExtraLines");
        const visibleLines = Array.isArray(masterResult.ocr_text_lines)
            ? masterResult.ocr_text_lines.filter(Boolean)
            : [];
        const extraLines = visibleLines.slice(1);
        if (poleTextLinesRow && poleExtraLinesEl) {
            if (extraLines.length) {
                poleExtraLinesEl.textContent = extraLines.join('\n');
                poleTextLinesRow.classList.remove('hidden');
            } else {
                poleExtraLinesEl.textContent = '---';
                poleTextLinesRow.classList.add('hidden');
            }
        }

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

            if (masterResult.pole_type === 'strut_pole' || masterResult.pole_status === 'strut_pole') {
                status = "Strut Pole";
                badgeClass = "bg-blue-500/10 text-blue-400 border border-blue-500/20";
            } else {
                status = "Main Pole";
                badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
                if (masterResult.pole_status === 'fault') {
                    badgeClass = "bg-rose-500/10 text-rose-400 border border-rose-500/20";
                    status = "Main Pole (Critical)";
                } else if (masterResult.pole_status === 'warning') {
                    badgeClass = "bg-amber-500/10 text-amber-400 border border-amber-500/20";
                    status = "Main Pole (Leaning)";
                }
            }

            statusBadge.textContent = status;
            statusBadge.className = `px-2 py-0.5 rounded text-[8px] font-bold uppercase ${badgeClass}`;
        } else {
            stabilityRow.classList.add("hidden");
        }

        // 1.1.5 Compute Max Hardware Counts Across Batch
        const maxCounts = {};
        let processedImagesCount = 0;
        batchImages.forEach(imgData => {
            if (!imgData.processed || !imgData.detections) return;
            processedImagesCount++;
            const currentCounts = {};
            imgData.detections.forEach(d => {
                // Group by main category (e.g. PIN_INSULATOR -> INSULATOR)
                let type = d.label;
                if (type.includes('INSULATOR')) type = 'INSULATOR';
                else if (type.includes('POLE')) type = 'POLE';
                else if (type.includes('CROSSARM')) type = 'CROSSARM';
                else if (type.includes('CONDUCTOR')) type = 'CONDUCTOR';
                else type = type.split('_')[0];
                
                currentCounts[type] = (currentCounts[type] || 0) + 1;
            });
            for (const type in currentCounts) {
                if (!maxCounts[type] || currentCounts[type] > maxCounts[type]) {
                    maxCounts[type] = currentCounts[type];
                }
            }
        });

        const batchCountsRow = document.getElementById("batchCountsRow");
        if (batchCountsRow) {
            if (Object.keys(maxCounts).length > 0 && processedImagesCount > 1) {
                batchCountsRow.classList.remove("hidden");
                batchCountsRow.innerHTML = `<span class="w-full text-[8px] text-gray-500 font-bold uppercase mb-1">Max Hardware Detected Across ${processedImagesCount} Views</span>`;
                
                for (const type in maxCounts) {
                    let icon = 'fa-tag';
                    if (type === 'POLE') icon = 'fa-tower-broadcast';
                    else if (type === 'INSULATOR') icon = 'fa-bolt';
                    else if (type === 'CROSSARM') icon = 'fa-compass-drafting';
                    else if (type === 'CONDUCTOR') icon = 'fa-layer-group';
                    
                    batchCountsRow.innerHTML += `
                        <div class="px-2 py-1 bg-white/5 border border-white/10 rounded flex items-center gap-1.5 shadow-sm">
                            <i class="fa-solid ${icon} text-[9px] text-blue-400"></i>
                            <span class="text-[9px] font-bold text-white">${maxCounts[type]} <span class="text-gray-500">${type}S</span></span>
                        </div>
                    `;
                }
            } else {
                batchCountsRow.classList.add("hidden");
            }
        }
    } else {
        masterCard.classList.add('hidden');
    }
    // Check if the batch has been processed
    const processed = batchImages.some(img => img.processed);
    if (!processed) {
        container.innerHTML = `
            <div class="text-center py-20 bg-black/30 rounded-2xl border border-dashed border-gray-800">
                <i class="fa-solid fa-wand-magic-sparkles text-4xl text-gray-700 mb-4 block"></i>
                <p class="text-gray-600 text-sm italic font-bold">${currentInputMode === 'video' ? 'Waiting for video analysis results...' : 'Waiting for analysis results...'}</p>
            </div>
        `;
        return;
    }

    const activeMedia = batchImages[0]?.mediaType || 'image';
    if (activeMedia === 'video') {
        const counts = batchImages[0].classCounts || {};
        const countEntries = Object.entries(counts).sort(([, a], [, b]) => b - a);
        const detectedClasses = (batchImages[0].detectedClasses || [])
            .map(label => String(label || '').toUpperCase())
            .filter(Boolean);
        const classEntries = countEntries.length
            ? countEntries.map(([label, count]) => ({ label, count, persistent: true }))
            : detectedClasses.map(label => ({ label, count: null, persistent: false }));
        const trimStart = Number(batchImages[0].trimStart || 0);
        const trimDuration = Number(batchImages[0].trimDuration || 0);
        const summary = document.createElement("div");
        summary.className = "mb-3 p-4 rounded-2xl border border-blue-500/20 bg-blue-500/5";
        summary.innerHTML = `
            <div class="flex items-center justify-between gap-3 mb-3">
                <div class="flex items-center gap-2">
                    <i class="fa-solid fa-film text-blue-400"></i>
                    <span class="text-[10px] font-bold uppercase tracking-widest text-blue-200">Whole Video Classes</span>
                </div>
                <span class="text-[8px] font-mono text-gray-500">${formatSeconds(trimStart)} - ${formatSeconds(trimStart + trimDuration)}</span>
            </div>
            <div class="grid grid-cols-2 gap-2">
                ${(classEntries.length ? classEntries : [
                    { label: 'MAIN_POLE', count: 0, persistent: true },
                    { label: 'STRUT_POLE', count: 0, persistent: true }
                ]).map(({ label, count, persistent }) => `
                    <div class="p-3 rounded-xl bg-black/25 border border-white/5">
                        <div class="text-[8px] text-gray-500 uppercase tracking-widest font-bold">${label.replace(/_/g, ' ')}</div>
                        <div class="text-2xl font-black mt-1" style="color:${CLASS_COLORS[label] || '#94a3b8'}">${persistent ? (count || 0) : 'DETECTED'}</div>
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(summary);
    }

    // 2. Aggregate all detections across all batch images
    let allDetectionsList = [];
    batchImages.forEach((img, imgIdx) => {
        if (img.detections) {
            img.detections.forEach((d, detIdx) => {
                allDetectionsList.push({ ...d, imgIdx, detIdx });
            });
        }
    });

    if (allDetectionsList.length === 0) {
        const detectedVideoClasses = activeMedia === 'video'
            ? (batchImages[0]?.detectedClasses || [])
            : [];
        const emptyState = document.createElement("div");
        emptyState.innerHTML = `
            <div class="text-center py-20 bg-black/30 rounded-2xl border border-dashed border-gray-800">
                <i class="fa-solid fa-circle-info text-4xl text-gray-700 mb-4 block"></i>
                <p class="text-gray-600 text-sm italic font-bold uppercase tracking-wider">
                    ${detectedVideoClasses.length
                        ? 'Classes detected, but no persistent physical pole track was confirmed.'
                        : 'No objects detected.'}
                </p>
            </div>
        `;
        container.appendChild(emptyState);
        const submitBtn = document.getElementById('finalSubmitBtn');
        if (submitBtn) {
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
            submitBtn.disabled = true;
        }
        return;
    }

    allDetectionsList.sort((a, b) => a.label.localeCompare(b.label));

    const itemsContainer = document.createElement("div");
    itemsContainer.className = "space-y-2 pb-4";
    
    const allConfirmedList = allDetectionsList.length > 0 && allDetectionsList.every(d => d.confirmed);
    const masterHeader = document.createElement("div");
    masterHeader.className = "flex items-center justify-between p-3 mb-3 border border-gray-800 rounded-xl bg-gray-900/60 sticky top-0 z-10 backdrop-blur-md";
    masterHeader.innerHTML = `
        <div class="flex items-center gap-3">
            <i class="fa-solid fa-list-ul text-blue-400"></i>
            <span class="text-[10px] font-bold text-gray-300 uppercase tracking-widest">ALL DETECTIONS (${allDetectionsList.length})</span>
        </div>
        <button onclick="bulkConfirmAll()" 
                class="px-3 py-1.5 rounded-lg text-[8px] font-bold uppercase tracking-widest transition-all flex items-center gap-1.5 ${allConfirmedList ? 'bg-green-600/20 text-green-400 border border-green-500/30' : 'bg-white/5 text-gray-500 border border-white/10 hover:bg-white/10 hover:text-white'}">
            <i class="fa-solid ${allConfirmedList ? 'fa-check-double' : 'fa-check'}"></i>
            ${allConfirmedList ? 'ALL CHECKED' : 'BULK CHECK ALL'}
        </button>
    `;
    container.appendChild(masterHeader);

    allDetectionsList.forEach((obj) => {
        const label = obj.label;
        const baseColor = CLASS_COLORS[label.toUpperCase()] || "#a8a29e";
        
        const itemDiv = document.createElement("div");
        itemDiv.className = `flex items-center justify-between p-3 rounded-lg border transition-all result-card-hover ${obj.confirmed ? 'border-green-500/40 bg-green-500/5' : 'bg-white/5 border-white/5'} hover:border-white/20`;
        if (activeMedia === 'video' && obj.details?.frame_time !== undefined) {
            itemDiv.classList.add('cursor-pointer');
            itemDiv.title = `Go to ${formatSeconds(obj.details.frame_time)} in the video`;
            itemDiv.onclick = (event) => {
                if (event.target.closest('button')) return;
                seekToVideoDetection(obj.details.frame_time);
            };
        }

        // Interactive Sync: Glow the box on the image when hovering the result item
        itemDiv.onmouseenter = () => {
            // Auto switch image view if needed
            if (activeBatchIndex !== obj.imgIdx) {
                selectBatchImage(obj.imgIdx);
            }
            const box = document.getElementById(`box-${obj.detIdx}`);
            if (box) box.classList.add('highlighted');
            const lbl = document.getElementById(`label-${obj.detIdx}`);
            if (lbl) {
                lbl.classList.remove('opacity-0');
                lbl.classList.add('opacity-100');
                lbl.style.transform = 'scale(1.1)';
            }
        };
        itemDiv.onmouseleave = () => {
            const box = document.getElementById(`box-${obj.detIdx}`);
            if (box) box.classList.remove('highlighted');
            const lbl = document.getElementById(`label-${obj.detIdx}`);
            if (lbl) {
                lbl.classList.remove('opacity-100');
                lbl.classList.add('opacity-0');
                lbl.style.transform = '';
            }
        };

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
                if (activeMedia === 'video') {
                    const appearances = obj.details.appearances || 1;
                    const fragments = obj.details.track_fragments || 1;
                    const track = obj.details.track_ids?.length ? `MERGED TRACKS ${obj.details.track_ids.join(',')}` : 'MERGED POLE';
                    detailStr = `${track} | ${fragments} fragment${fragments === 1 ? '' : 's'} | ${appearances} frame${appearances === 1 ? '' : 's'} | Conf: ${fakeConfStr}`;
                    metaIcon = obj.label === 'STRUT_POLE' ? "fa-ruler-combined" : "fa-tower-broadcast";
                } else {
                const lean = obj.details.lean || 0;
                const isStrut = obj.details.type === 'strut_pole';
                const isExtreme = !isStrut && lean > 10;
                
                if (isStrut) {
                    detailStr = `<span class="text-blue-400">ANGLE: ${lean}°</span> | <span class="text-blue-300">strut pole</span> | <span class="text-white/60">Conf: ${fakeConfStr}</span>`;
                    metaIcon = "fa-ruler-combined";
                } else {
                    const leanColor = isExtreme ? 'text-rose-400 font-black' : (lean > 5 ? 'text-amber-400' : 'text-emerald-400');
                    const abnormalityTag = isExtreme ? `<span class="bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded-full text-[7px] border border-rose-500/30 ml-2 animate-pulse">ABNORMALITY</span>` : "";
                    detailStr = `<span class="${leanColor}">LEAN: ${lean}°</span>${abnormalityTag} | MAIN POLE | <span class="text-white/60">Conf: ${fakeConfStr}</span>`;
                    metaIcon = "fa-triangle-exclamation";
                }
                }
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
                    <div class="flex items-center gap-2">
                        <p class="text-[10px] font-bold text-gray-200 uppercase tracking-tight">${label} ID-${obj.detIdx + 1}</p>
                        ${batchImages.length > 1 ? `<span class="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded border border-blue-500/30 text-[7px] font-bold">IMAGE ${obj.imgIdx + 1}</span>` : ''}
                        ${activeMedia === 'video' && obj.details?.frame_time !== undefined ? `<span class="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-300 rounded border border-emerald-500/20 text-[7px] font-bold">${formatSeconds(obj.details.frame_time)}</span>` : ''}
                    </div>
                    <div class="flex items-center gap-1.5 mt-0.5">
                        <i class="fa-solid ${metaIcon} text-[8px] text-gray-600"></i>
                        <p class="text-[9px] text-gray-500 font-medium uppercase tracking-widest">${detailStr}</p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-2">
                <button onclick="toggleConfirmGlobal(${obj.imgIdx}, ${obj.detIdx})" class="btn ${obj.confirmed ? 'bg-green-600 text-white' : 'btn-outline border-white/5 bg-white/5 hover:border-white/20'} !p-2 !h-8 !w-8 !rounded-lg text-[10px]">
                    <i class="fa-solid ${obj.confirmed ? 'fa-check-double' : 'fa-check'}"></i>
                </button>
                <button onclick="removeDetectionGlobal(${obj.imgIdx}, ${obj.detIdx})" class="p-2 text-gray-600 hover:text-rose-400 transition-colors">
                    <i class="fa-solid fa-trash-can text-[10px]"></i>
                </button>
            </div>
        `;
        itemsContainer.appendChild(itemDiv);
    });

    container.appendChild(itemsContainer);

    // Update final submit button state
    const allConfirmed = allDetectionsList.every(d => d.confirmed);
    const submitBtn = document.getElementById('finalSubmitBtn');
    if (submitBtn) {
        if (allConfirmed && allDetectionsList.length > 0) {
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

function toggleConfirmGlobal(imgIdx, detIdx) {
    saveToHistory();
    batchImages[imgIdx].detections[detIdx].confirmed = !batchImages[imgIdx].detections[detIdx].confirmed;
    if (activeBatchIndex === imgIdx) {
        detections[detIdx].confirmed = batchImages[imgIdx].detections[detIdx].confirmed;
    }
    renderResults();
    renderBoxes();
}

function removeDetectionGlobal(imgIdx, detIdx) {
    saveToHistory();
    batchImages[imgIdx].detections.splice(detIdx, 1);
    if (activeBatchIndex === imgIdx) {
        detections.splice(detIdx, 1);
    }
    renderResults();
    renderBoxes();
}

function toggleConfirm(index) {
    toggleConfirmGlobal(activeBatchIndex, index);
}

function removeDetection(index) {
    removeDetectionGlobal(activeBatchIndex, index);
}

function bulkConfirmAll() {
    saveToHistory();
    // Check if everything is confirmed across ALL images
    const allConfirmed = !batchImages.some(img => img.detections && img.detections.some(d => !d.confirmed));
    const targetState = !allConfirmed;
    
    batchImages.forEach(img => {
        if (img.detections) {
            img.detections.forEach(d => d.confirmed = targetState);
        }
    });
    
    if (batchImages[activeBatchIndex] && batchImages[activeBatchIndex].detections) {
        detections = [...batchImages[activeBatchIndex].detections];
    }
    
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
    if (!overlay || !img) return;

    overlay.innerHTML = "";

    if (detections.length === 0) return;

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
        
        // Smart contrast: if the background color is white or cyan/yellow, use dark text
        const isLightBg = color === "#ffffff" || color === "#00ffff" || color === "#fbbf24";
        labelTextEl.setAttribute("fill", isLightBg ? "#0f172a" : "#ffffff");
        
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

    // Save current active state before submission
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
        batchImages[activeBatchIndex].master = masterResult;
        batchImages[activeBatchIndex].dims = { ...imageDimensions };
    }

    const allDetectionsList = [];
    batchImages.forEach(img => {
        if (img.detections) {
            img.detections.forEach(d => allDetectionsList.push(d));
        }
    });

    if (allDetectionsList.length === 0) {
        showToast("No active results to submit", "warning");
        return;
    }

    const btn = document.getElementById('finalSubmitBtn');
    btn.disabled = true;
    const originalInner = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Submitting Batch...`;

    try {
        const payload = {
            master: masterResult || (batchImages[0] ? batchImages[0].master : {}),
            survey_data: surveyResult || {},
            images: []
        };

        // Process each image in the batch with compression
        for (const item of batchImages) {
            try {
                let b64 = "";
                if (item.src.startsWith('blob:')) {
                    b64 = await compressImage(item.src);
                } else if (item.src.startsWith('data:image')) {
                    b64 = item.src.split(',')[1];
                } else {
                    const blob = await fetch(item.src).then(r => r.blob());
                    b64 = await new Promise(resolve => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(blob);
                    });
                }
                payload.images.push({
                    image_b64: b64,
                    detections: item.detections || [],
                    pole_angle: item.master ? item.master.pole_lean_angle : (masterResult ? masterResult.pole_lean_angle : 0.0)
                });
            } catch (pErr) {
                console.warn("Compression failed for an image, using fallback", pErr);
                const blob = await fetch(item.src).then(r => r.blob());
                const b64 = await new Promise(resolve => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                });
                payload.images.push({
                    image_b64: b64,
                    detections: item.detections || [],
                    pole_angle: item.master ? item.master.pole_lean_angle : (masterResult ? masterResult.pole_lean_angle : 0.0)
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
            showToast("Success! Submitted for Review", "success");
            setTimeout(() => {
                resetSession(true);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }, 2000);
        } else {
            throw new Error(result.message);
        }
    } catch (err) {
        console.error(err);
        showToast(`Submission failed: ${err.message}`, "danger");
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

    saveToHistory();
    detections.push(newDet);
    if (activeBatchIndex !== -1) {
        batchImages[activeBatchIndex].detections = [...detections];
    }

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
            for (let i = 1; i < d.polygon.length; i++) {
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
            ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
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
// Duplicate submitAsset removed

async function downloadCurrentResult() {
    if (!detections || detections.length === 0) {
        showToast("No active results to download", "warning");
        return;
    }

    try {
        showToast("Preparing Admin Studio...", "primary");

        const activeImg = activeBatchIndex !== -1 ? batchImages[activeBatchIndex] : null;

        const assetData = {
            id: `TEMP_${Date.now()}`,
            worker_name: "Local Session",
            status: "draft",
            timestamp: new Date().toLocaleString(),
            asset_class: masterResult ? masterResult.final_class : "Unclassified",
            voltage: masterResult ? masterResult.voltage : "Unknown",
            reason: masterResult ? masterResult.reason : "Manual Review",
            images: [{
                image_b64: activeImg ? activeImg.src : null,
                detections: detections
            }]
        };

        // Save as draft first so the admin studio can read it
        const response = await fetch('/api/save_draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: assetData.id, type: 'export', data: JSON.stringify(assetData) })
        });

        if (response.ok) {
            showToast("Opening Admin Studio", "success");
            window.location.href = `/admin/asset/${assetData.id}`;
        }
    } catch (err) {
        console.error("Export Error:", err);
        showToast("Export failed", "danger");
    }
}

function scrollResultBox(direction) {
    const box = document.getElementById("resultBox");
    if (box) {
        const scrollAmount = 150;
        const targetScroll = direction === 'up' ? -scrollAmount : scrollAmount;
        try {
            box.scrollBy({
                top: targetScroll,
                behavior: 'smooth'
            });
        } catch (e) {
            box.scrollTop += targetScroll;
        }
    }
}
