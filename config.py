"""
config.py
─────────
Centralized settings for the infrastructure detection pipeline.
"""

# ── Database ──────────────────────────────────────────────────
DB_TYPE = "postgres" # "sqlite" or "postgres"
DB_NAME = "object_detection_ui_main"

# PostgreSQL credentials
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASS = "asakta"
PG_DB   = "object_detection_ui_main"

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR     = "models"
BACKUP_DIR     = "dry_backup"
UPLOADS_DIR    = "uploads"
OUTPUTS_DIR    = "static/results"

# ── Image processing ──────────────────────────────────────────
# We upscale slightly to ensure small insulators are visible
# but not too much to avoid excessive memory usage.
INFERENCE_SIZE = 1024
CROP_SIZE      = 640

# ── Rule Engine Parameters ────────────────────────────────────
POLE_IDEAL_ANGLE_DEG    = 90.0
POLE_TOLERANCE_DEG      = 5.0
POLE_FAULT_DEG          = 15.0
POLE_STRUT_THRESHOLD_DEG = 25.0
POLE_LEANING_THRESHOLD_DEG = 5.0

CROSSARM_IDEAL_ANGLE_DEG = 0.0
CROSSARM_TOLERANCE_DEG   = 8.0
CROSSARM_FAULT_DEG       = 15.0

# ── Crossarm shape thresholds ────────────────────────────────
CROSSARM_MIN_AR_STRAIGHT    = 4.0
CROSSARM_CONDUCTOR_V_SPREAD = 0.8
CROSSARM_T_VERTICAL_RATIO   = 0.05

# ── HT+LT separation ─────────────────────────────────────────
HT_LT_HEIGHT_THRESHOLD = 0.40  # was 0.25

# ── Detection thresholds ─────────────────────────────────────
DETECTION_CONF  = 0.01  # Base sensitivity
DEFAULT_IMGSZ   = 1024  # Restored to 1024 for maximum accuracy
DETECTION_IOU   = 0.45

# Class-specific confidence overrides to reduce false positives
THRESHOLD_INSULATOR = 0.10  # Lowered to ensure faint insulators are detected
THRESHOLD_CROSSARM  = 0.35  # Lowered from 0.70 to prevent missing clear crossarms
THRESHOLD_POLE      = 0.35  # Lowered from 0.50 for better pole detection
THRESHOLD_CONDUCTOR = 0.40  # Keep strict to avoid noisy wire boxes

# Global tilt compensation limit
GLOBAL_TILT_MAX_DEG = 20.0

# ── Insulator secondary processing ────────────────────────────
# Only insulators above this confidence will be cropped and run
# through the shed/disc counter (best_disc.pt).
INSULATOR_MIN_CONF      = 0.80  # Only classify insulators above 80% confidence
INSULATOR_CROP_PADDING  = 25    # slightly more padding for context
SHED_MODEL_CONF         = 0.25

# ── Augmentation ─────────────────────────────────────────────
AUG_TARGET_COUNT    = 800   # boost any class below this count
AUG_SILHOUETTE_PROB = 0.25  # 25% chance of silhouette-style augmentation

# ── OBB component class keywords ─────────────────────────────
OBB_CLASS_KEYWORDS = {
    "insulator":    ["insulator"],
    "pole":         ["pole"],
    "crossarm":     ["crossarm", "cross_arm", "cross arm", "arm", "v-arm", "v_arm", "t-arm", "t_arm"],
    "conductor":    ["conductor", "wire", "cable", "power_line"],
    "lamp_head":    ["lamp", "lamp_head"]
}

COMPONENT_CLASSES = [
    "insulator",      # 0
    "pole",           # 1
    "strut_pole",     # 2
    "crossarm",       # 3
    "conductor",      # 4
    "street_light",   # 5
]


# ── Final asset classes (Rule Engine) ───────────────────────
POLE_CLASSES = {
    "HT_pole":      "High Tension Main Pole",
    "LT_pole":      "Low Tension Main Pole",
    "strut_pole":   "Structural Strut Pole",
    "transformer":  "DTR / Transformer Structure"
}

# ── OCR Configuration ─────────────────────────────────────────
USE_LLM_OCR = False

# ── Insulator Classifier Variables (failing imports fix) ──────
INSULATOR_PIN_RATIO_MIN = 1.1
INSULATOR_DISC_RATIO_MAX = 0.9
SHED_VOLTAGE_MAP = {3: "11kV", 4: "33kV"}
SHED_VOLTAGE_GT3 = "33kV"
PIN_INSULATOR_IDEAL_ANGLE = 90.0
PIN_INSULATOR_TOLERANCE_DEG = 10.0
PIN_INSULATOR_FAULT_DEG = 20.0
PIN_TILT_AR_THRESHOLD = 1.0

# ── API Keys (Automatically populated from secure store) ─────
GEMINI_API_KEY = "AIzaSyAGJVjCry9BYdb9oMcvR5ySLLAkhnlZR34"

