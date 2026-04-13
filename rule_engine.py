"""
rule_engine.py
───────────────
Maps detected component signals → final pole class.

Priority order:
  1. Hard structural overrides  (lattice, DTR, lamp, jumper, AB cable)
  2. Combined HT+LT pole check
  3. Voltage from insulator     (shed count model → Indian standard)
  4. Sub-type from crossarm     (straight/V/T → intermediate vs tension)
  5. Pole orientation           (vertical vs strut)
  6. Generic fallback

Each decision includes a reason string for traceability.
"""

from dataclasses import dataclass, field
from typing import Optional

from config import POLE_CLASSES


# ── Input structure for rule engine ──────────────────────────

@dataclass
class ComponentSignals:
    """All signals gathered from component detection."""

    # Insulator signals
    insulator_type:   str   = "unknown"   # "pin" | "disc" | "unknown"
    insulator_voltage: str  = "unknown"   # "LT" | "6.3kV" | "11kV" | "33kV"
    shed_count:       int   = 0
    insulator_conf:   str   = "low"       # "high" | "medium" | "low"

    # Structural signals
    has_dtr:          bool  = False
    has_ab_cable:     bool  = False
    has_lattice:      bool  = False
    has_jumper:       bool  = False
    has_ht_and_lt:    bool  = False

    # Pole geometry
    pole_type:        str   = "vertical_pole"   # "vertical_pole" | "strut_pole"
    lean_angle_deg:   float = 0.0

    # Crossarm signals
    crossarm_count:   int   = 0
    crossarm_shape:   str   = "none"   # "straight" | "v_arm" | "t_raising" | "none"

    # Conductor signals
    conductor_count:  int   = 0


# ── Output structure ──────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Final classification result with full explanation."""
    final_class:    str
    class_id:       int
    reason:         str
    voltage:        str
    confidence:     str   # "high" | "medium" | "low"
    signals_used:   list  = field(default_factory=list)
    faults:         list  = field(default_factory=list) # Critical anomalies


# ── Rule engine ───────────────────────────────────────────────

def classify_pole(signals: ComponentSignals) -> ClassificationResult:
    """
    Main rule engine. Takes all component signals and returns
    the final pole classification.

    Args:
        signals : ComponentSignals populated from pipeline

    Returns:
        ClassificationResult with class name, ID, reason, confidence.
    """

    def make(cls_name, reason, voltage=None, conf="medium", signals_used=None):
        cls_id = POLE_CLASSES.index(cls_name) if cls_name in POLE_CLASSES else -1
        
        # --- NEW: ANOMALY DETECTION (Requirement 2.4) ---
        found_faults = []
        if signals.lean_angle_deg > 5.0 and signals.pole_type == "vertical_pole":
            found_faults.append(f"CRITICAL LEAN ({signals.lean_angle_deg}°)")
            
        if signals.insulator_type == "unknown" and signals.conductor_count > 0:
            found_faults.append("MISSING INSULATOR / BROKEN MOUNT")

        if found_faults:
            reason = " { FAULT DETECTED } " + reason + " | " + ", ".join(found_faults)

        return ClassificationResult(
            final_class   = cls_name,
            class_id      = cls_id,
            reason        = reason,
            voltage       = voltage or signals.insulator_voltage,
            confidence    = conf,
            signals_used  = signals_used or [],
            faults        = found_faults
        )

    # ════════════════════════════════════════════════════════
    # PRIORITY 1 — Hard structural overrides
    # These are unambiguous — no other signals needed.
    # ════════════════════════════════════════════════════════

    if signals.has_lattice:
        return make(
            "M9_tower",
            "Lattice steel frame detected → M+9 tower",
            voltage="33kV+", conf="high",
            signals_used=["has_lattice"]
        )

    if signals.has_dtr:
        return make(
            "DTR_HT_pole",
            "Distribution transformer tank detected on pole",
            conf="high",
            signals_used=["has_dtr"]
        )



    if signals.has_jumper:
        return make(
            "HT_tapping_point",
            "Jumper wire loop detected → HT tapping point",
            conf="high",
            signals_used=["has_jumper"]
        )

    if signals.has_ab_cable:
        return make(
            "LT_AB_cable_pole",
            "Aerial bundled cable detected → LT AB cable pole",
            voltage="LT", conf="high",
            signals_used=["has_ab_cable"]
        )

    # ════════════════════════════════════════════════════════
    # PRIORITY 2 — Combined HT + LT pole
    # ════════════════════════════════════════════════════════

    if signals.has_ht_and_lt:
        return make(
            "HT_LT_pole",
            "HT wires at top + LT wires lower → combined HT/LT pole",
            conf="medium",
            signals_used=["has_ht_and_lt"]
        )

    # ════════════════════════════════════════════════════════
    # PRIORITY 3 — Voltage-based classification from insulator
    # ════════════════════════════════════════════════════════

    voltage = signals.insulator_voltage

    # ── 33kV ──────────────────────────────────────────────
    if voltage == "33kV":
        signals_used = ["insulator_voltage=33kV", f"shed_count={signals.shed_count}"]

        # Tension pole: V arm OR T arm OR double crossarm OR strut
        is_tension = (
            signals.crossarm_shape in ("v_arm", "t_raising")
            or signals.crossarm_count >= 2
            or signals.pole_type == "strut_pole"
        )

        if is_tension:
            signals_used.append(f"crossarm_shape={signals.crossarm_shape}")
            signals_used.append(f"crossarm_count={signals.crossarm_count}")
            return make(
                "33kV_HT_pole",
                f"33kV + tension indicators ({signals.crossarm_shape}) → HT pole",
                voltage="33kV", conf="medium",
                signals_used=signals_used
            )

        return make(
            "33kV_pole",
            "33kV + single straight crossarm → intermediate pole",
            voltage="33kV", conf="medium",
            signals_used=signals_used
        )

    # ── 11kV ──────────────────────────────────────────────
    if voltage == "11kV":
        signals_used = ["insulator_voltage=11kV", "shed_count=3"]

        if signals.pole_type == "strut_pole":
            signals_used.append("pole_type=strut_pole")
            return make(
                "11kV_HT_pole",  # strut = tension position
                f"11kV + strut pole ({signals.lean_angle_deg}° lean) → tension",
                voltage="11kV", conf="medium",
                signals_used=signals_used
            )

        return make(
            "11kV_HT_pole",
            "3 sheds detected → 11kV HT pole",
            voltage="11kV",
            conf="high" if signals.insulator_conf == "high" else "medium",
            signals_used=signals_used
        )

    # ── 6.3kV ─────────────────────────────────────────────
    if voltage == "6.3kV":
        signals_used = [
            "insulator_voltage=6.3kV",
            f"shed_count={signals.shed_count}"
        ]

        if signals.pole_type == "strut_pole":
            signals_used.append("pole_type=strut_pole")
            return make(
                "6.3kV_HT_pole",
                f"6.3kV + strut pole → tension/dead-end pole",
                voltage="6.3kV", conf="medium",
                signals_used=signals_used
            )

        return make(
            "6.3kV_pole",
            f"Small pin insulator ({signals.shed_count} sheds) → 6.3kV pole",
            voltage="6.3kV", conf="medium",
            signals_used=signals_used
        )

    # ── LT zone ───────────────────────────────────────────
    if voltage == "LT":
        signals_used = ["insulator_voltage=LT", "no_HV_insulator"]

        # STRUCTURAL OVERRIDE: V-Arm or T-Raising Arm are HT components!
        # If we see these shapes, it's NOT an LT pole.
        if signals.crossarm_shape in ("v_arm", "t_raising"):
            signals_used.append(f"structural_shape={signals.crossarm_shape}")
            return make(
                "11kV_HT_pole",
                f"HT crossarm geometry ({signals.crossarm_shape}) overrides LT insulator",
                voltage="11kV", conf="medium",
                signals_used=signals_used
            )

        if signals.conductor_count >= 5:
            signals_used.append(f"conductor_count={signals.conductor_count}")
            return make(
                "LT_pole",
                "LT zone + 5 conductors → 3-phase LT pole",
                voltage="LT", conf="medium",
                signals_used=signals_used
            )

        return make(
            "LT_pole",
            "LT zone insulator → LT distribution pole",
            voltage="LT", conf="medium",
            signals_used=signals_used
        )

    # ════════════════════════════════════════════════════════
    # PRIORITY 4 — Conductor-only fallback (no insulator detected)
    # ════════════════════════════════════════════════════════

    if signals.conductor_count > 0 and signals.crossarm_count == 0:
        return make(
            "LT_pole",
            "Conductors present, no crossarm, no insulator → likely LT",
            voltage="LT", conf="low",
            signals_used=["conductor_count>0", "crossarm_count=0"]
        )

    # ════════════════════════════════════════════════════════
    # PRIORITY 5 — Generic fallback
    # ════════════════════════════════════════════════════════

    return make(
        "HT_pole",
        "No strong signals → generic HT pole (needs more data)",
        conf="low",
        signals_used=["fallback"]
    )
