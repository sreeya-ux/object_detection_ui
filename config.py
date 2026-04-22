"""
config.py
─────────
Single source of truth for all thresholds, class names,
and dataset references.

CHANGE LOG:
  - Removed insulator fault detection datasets (broken disc, cracked cap)
  - Added adjustment fault thresholds (tilt, skew, misalignment)
  - Fault detection = geometric misalignment only, not physical damage
"""

# ── Insulator aspect-ratio thresholds ───────────────────────
INSULATOR_PIN_RATIO_MIN   = 1.1   # was 1.5 - now more inclusive of angled shots
INSULATOR_DISC_RATIO_MAX  = 0.7   # height/width < this  → likely disc

# ── Adjustment fault thresholds ──────────────────────────────
# "Adjustment fault" = component is physically present but
#  misaligned, tilted, or improperly seated — not physically broken.
#
# Detected purely from bounding box geometry and OBB angle.
# No separate fault detection model needed.

# Pin insulator tilt
# Correctly mounted pin insulator points straight up (OBB angle ≈ 90°)
PIN_INSULATOR_IDEAL_ANGLE   = 90.0   # degrees
PIN_INSULATOR_TOLERANCE_DEG = 15.0   # ±15° → ok
PIN_INSULATOR_FAULT_DEG     = 25.0   # beyond ±25° → adjustment fault

# Pin insulator aspect ratio when tilted
# A tilted pin insulator has a wider bbox → lower height/width ratio
PIN_TILT_AR_THRESHOLD       = 1.2    # if ratio < 1.2 for a "pin" → may be tilted

# Crossarm level check
# A correctly mounted crossarm is horizontal (OBB angle ≈ 0°)
CROSSARM_IDEAL_ANGLE_DEG    = 0.0
CROSSARM_TOLERANCE_DEG      = 8.0    # ±8° → ok
CROSSARM_FAULT_DEG          = 15.0   # beyond ±15° → adjustment fault

# Pole lean
# Vertical pole has OBB angle ≈ 90°
POLE_IDEAL_ANGLE_DEG        = 90.0
POLE_TOLERANCE_DEG          = 5.0    # ±5° → ok
POLE_FAULT_DEG              = 10.0   # beyond ±10° → adjustment fault
# Note: strut poles lean intentionally — skip fault check if pole_type=strut
POLE_STRUT_THRESHOLD_DEG    = 30.0   # lean > 30° → definitely strut, not fault

# ── Shed count → voltage (Indian standard) ───────────────────
SHED_VOLTAGE_MAP = {
    0: "LT",
    1: "6.3kV",
    2: "6.3kV",
    3: "11kV",
    4: "33kV",
    5: "33kV"
}
SHED_VOLTAGE_GT3 = "33kV"

# ── Crossarm shape thresholds ────────────────────────────────
CROSSARM_MIN_AR_STRAIGHT    = 4.0
CROSSARM_CONDUCTOR_V_SPREAD = 0.8
CROSSARM_T_VERTICAL_RATIO   = 0.05

# ── HT+LT separation ─────────────────────────────────────────
HT_LT_HEIGHT_THRESHOLD = 0.40  # was 0.25

# ── Detection thresholds ─────────────────────────────────────
DETECTION_CONF  = 0.01  # Base sensitivity
DETECTION_IOU   = 0.45

# Class-specific confidence overrides to reduce false positives
THRESHOLD_INSULATOR = 0.07
THRESHOLD_CROSSARM  = 0.35
THRESHOLD_POLE      = 0.10
THRESHOLD_CONDUCTOR = 0.01

# ── Insulator secondary processing ────────────────────────────
# Only insulators above this confidence will be cropped and run
# through the shed/disc counter (best_disc.pt).
INSULATOR_MIN_CONF      = 0.15  # was 0.40
INSULATOR_CROP_PADDING  = 25    # slightly more padding for context
SHED_MODEL_CONF         = 0.25

# ── Augmentation ─────────────────────────────────────────────
AUG_TARGET_COUNT    = 800   # boost any class below this count
AUG_SILHOUETTE_PROB = 0.0   # Disabled as per user request to remove silhouettes

# ── OBB component class keywords ─────────────────────────────
OBB_CLASS_KEYWORDS = {
    "insulator":    ["insulator"],
    "pole":         ["pole"],
    "crossarm":     ["crossarm", "cross_arm", "cross arm", "arm", "v-arm", "v_arm", "t-arm", "t_arm"],
    "conductor":    ["conductor", "wire", "cable", "power_line"],
    "lamp_head":    ["lamp", "lamp_head"]
}

# ── Component classes (YOLO Model) ───────────────────────────
# Expanded classes for granular infrastructure detection
COMPONENT_CLASSES = [
    "POLE_9M",           # 0
    "POLE_11M",          # 1
    "POLE_8.1M",         # 2
    "INS_PIN",           # 3
    "INS_DISC",          # 4
    "T_RISING",          # 5
    "TAPPING_CHANNEL",   # 6
    "SIDE_ARM_CHANNEL",  # 7
    "V_CROSS",           # 8
    "CONDUCTOR",         # 9
    "STREET_LIGHT",      # 10
    "DTR",               # 11
    "WIRE_BROKEN",       # 12
    "VEGETATION",        # 13
    "OBJECT",            # 14
]


# ── Final asset classes (Rule Engine) ───────────────────────
POLE_CLASSES = [
    "33kV_pole",         # 0
    "33kV_HT_pole",      # 1
    "11kV_HT_pole",      # 2
    "6.3kV_pole",        # 3
    "6.3kV_HT_pole",     # 4
    "HT_pole",           # 5
    "LT_pole",           # 6
    "HT_LT_pole",        # 7
    "DTR_HT_pole",       # 8
    "DTR_and_pole",      # 9
    "LT_AB_cable_pole",  # 10
    "LT_pole_plain",     # 11
    "street_light_pole", # 12
    "HT_tapping_point",  # 13
    "M9_tower",          # 14
    "POLE_9M_11kV",      # 15
    "POLE_11M_33kV",     # 16
    "POLE_8.1M_LT",      # 17
    "T_RISING_POLE",     # 18
    "V_CROSS_POLE",      # 19
]

# ── Datasets ──────────────────────────────────────────────────
DATASETS = [
    ("akshay-anand-bfabb", "pole-data",          1, "pole_data"),
    ("zac-ogogt",          "utility-poles-kcumt", 1, "utility_poles"),
    ("hanshin-university", "all_image",           1, "all_image"),
    ("indian-institute-of-technology-jodhpour",
     "cable-detection-liolt", 1,                     "cable_detection"),
]

TTPLA_GITHUB = "https://github.com/R3ab/ttpla_dataset"

# Mappings: original dataset class name → COMPONENT_CLASSES index
#
# COMPONENT_CLASSES indices:
#   0 = insulator
#   1 = pole
#   2 = crossarm
#   3 = conductor
#   4 = street_light
#
# None = skip this label entirely from training

DS_CLASS_MAPS = {
    # Pole-Data-1: has pole, crossarm, streetlight, transformer, telcobox
    "pole_data": {
        "pole":        1,
        "crossarm":    2,
        "streetlight": 4,
        "transformer": None,
        "telcobox":    None,
        "wire":        3,
    },
    # Utility Poles: has concrete/steel/wood pole, crossarms, insulator, street light
    "utility_poles": {
        "concrete pole":  1,
        "concrete_pole":  1,
        "steel pole":     1,
        "wood pole":      1,
        "single crossarm":2,
        "double crossarm":2,
        "insulator":      0,
        "street light":   4,
        "streetlight":    4,
        "transformer":    None,
        "lightning arrester": None,
        "cutout":         None,
        "splice case":    None,
    },
    # ALL_image (Hanshin Univ): 12 classes incl. 4 insulator types + power lines
    # Classes: Tree, Transformer, Arrester, CommunicationLine, COS,
    #          ElectricLine, ElectricPole, GILBS,
    #          InsulatorA, InsulatorB, InsulatorC, InsulatorD
    "all_image": {
        "ElectricPole":      1,
        "electricpole":      1,
        "electric pole":     1,
        "InsulatorA":        0,   # all 4 insulator types → class 0
        "InsulatorB":        0,
        "InsulatorC":        0,
        "InsulatorD":        0,
        "insulatora":        0,
        "insulatorb":        0,
        "insulatorc":        0,
        "insulatord":        0,
        "ElectricLine":      3,   # power conductors
        "electricline":      3,
        "CommunicationLine": 3,   # communication wire treated as conductor
        "communicationline": 3,
        "Transformer":       None,
        "transformer":       None,
        "Arrester":          None,
        "arrester":          None,
        "COS":               None,
        "cos":               None,
        "GILBS":             None,
        "gilbs":             None,
        "Tree":              None,
        "tree":              None,
    },
    # Cable detection: has cable, tower types (we only want conductor)
    "cable_detection": {
        "cable":         3,
        "wire":          3,
        "power_line":    3,
        "tower_lattice": None,
        "tower_wooden":  None,
        "tower_tucohy":  None,
    },
}

# ── Component class keyword mapping ──────────────────────────
# Used to route model class names to correct pipeline streams.
OBB_CLASS_KEYWORDS = {
    "insulator": ["insulator", "hardware", "ins_pin", "ins_disc"],
    "pole":      ["pole", "support", "backbone", "pole_9m", "pole_11m", "pole_8.1m"],
    "crossarm":  ["crossarm", "arm", "bracket", "t_rising", "v_cross", "channel"],
    "conductor": ["conductor", "wire", "line", "cable"],
    "dtr_tank":  ["transformer", "dtr", "tank"],
    "lamp_head": ["lamp", "street_light", "st_light"],
    "ab_cable":  ["ab_cable", "bundle", "hanging_cable"],
    "lattice":   ["lattice", "tower_frame"],
    "jumper":    ["jumper", "loop", "tap_wire"],
    "broken_wire": ["wire_broken", "broken_wire", "snapped"],
    "vegetation": ["vegetation", "tree", "plant", "bush"],
}
