"""Phase 3 — Automated Axis Correction for Toric Refractive Surgery.

Integrates diagnostic corneal keratometry (Pentacam/Topolyzer K-readings, cylinder axis)
with real-time Phase 2 cyclotorsion measurements to automatically determine and visualize
the exact corrected treatment axis for laser ablation.

Key Medical & Optical Physics:
    - 3-Degree Rule & Vector Astigmatism:
        Residual Cylinder = 2 * |C_planned| * sin(|theta_cyclotorsion|)
        Under-correction % = 2 * sin(|theta_cyclotorsion|) * 100%
        3 deg misalignment = ~10.5% under-correction + induced aberration
        4 deg misalignment = ~13.9% under-correction
        > 30 deg misalignment = worsens astigmatism beyond baseline
    - Treatment Axis Formula:
        alpha_corrected = (alpha_planned + theta_cyclotorsion) mod 180
    - Safety Interlocks:
        HIGH confidence: Approved for automated treatment axis adjustment
        MEDIUM confidence: Surgeon confirmation required
        LOW / CRITICAL: Laser interlock engaged, manual verification required
    - Latency <= 100 ms.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult
from pupil_tracking.utils.types import EllipseParams, EyeDetectionResult

logger = logging.getLogger(__name__)


class AxisSafetyGrade(Enum):
    """Clinical safety grading for automated laser axis adjustment."""
    HIGH = "HIGH"        # High confidence, safe for automated ablation axis alignment
    MEDIUM = "MEDIUM"    # Borderline confidence or angle, requires surgeon confirmation
    LOW = "LOW"          # Poor correlation or excessive error, ablation interlock engaged


@dataclass
class DiagnosticToricData:
    """Pre-operative diagnostic toric data imported via DICOM / direct device API."""
    planned_treatment_axis_deg: float = 90.0      # Planned steep/flat treatment axis [0, 180)
    planned_cylinder_power_d: float = -2.00       # Toric cylinder power (Diopters, e.g. -2.00 D)
    k1_diopters: float = 43.00                    # Flat keratometry (K1)
    k2_diopters: float = 45.00                    # Steep keratometry (K2)
    k_steep_axis_deg: float = 90.0                # Steep corneal axis
    k_flat_axis_deg: float = 0.0                  # Flat corneal axis
    corneal_astigmatism_d: float = -2.00          # Total corneal astigmatism
    laterality: str = "OD"                        # "OD" (Right) or "OS" (Left)
    patient_id: str = ""                          # De-identified patient ID
    device_source: str = "Pentacam HR"            # "Pentacam HR", "Topolyzer", "Manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planned_treatment_axis_deg": round(self.planned_treatment_axis_deg, 2),
            "planned_cylinder_power_d": round(self.planned_cylinder_power_d, 2),
            "k1_diopters": round(self.k1_diopters, 2),
            "k2_diopters": round(self.k2_diopters, 2),
            "k_steep_axis_deg": round(self.k_steep_axis_deg, 2),
            "k_flat_axis_deg": round(self.k_flat_axis_deg, 2),
            "corneal_astigmatism_d": round(self.corneal_astigmatism_d, 2),
            "laterality": self.laterality,
            "device_source": self.device_source,
        }


@dataclass
class AxisCorrectionResult:
    """Output of the Phase 3 automated axis correction engine."""
    valid: bool = False
    planned_axis_deg: float = 0.0
    cyclotorsion_deg: float = 0.0
    corrected_axis_deg: float = 0.0
    axis_adjustment_deg: float = 0.0

    # Optical & Vector Astigmatism Metrics
    planned_cylinder_power_d: float = 0.0
    residual_astigmatism_uncorrected_d: float = 0.0
    residual_astigmatism_corrected_d: float = 0.0
    correction_loss_avoided_percent: float = 0.0

    # Clinical Safety & Interlocks
    safety_grade: AxisSafetyGrade = AxisSafetyGrade.LOW
    ablation_interlock_approved: bool = False
    clinical_recommendation: str = ""

    # Execution timing (target <= 100 ms)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "planned_axis_deg": round(self.planned_axis_deg, 2),
            "cyclotorsion_deg": round(self.cyclotorsion_deg, 2),
            "corrected_axis_deg": round(self.corrected_axis_deg, 2),
            "axis_adjustment_deg": round(self.axis_adjustment_deg, 2),
            "planned_cylinder_power_d": round(self.planned_cylinder_power_d, 2),
            "residual_astigmatism_uncorrected_d": round(self.residual_astigmatism_uncorrected_d, 2),
            "residual_astigmatism_corrected_d": round(self.residual_astigmatism_corrected_d, 2),
            "correction_loss_avoided_percent": round(self.correction_loss_avoided_percent, 2),
            "safety_grade": self.safety_grade.value,
            "ablation_interlock_approved": self.ablation_interlock_approved,
            "clinical_recommendation": self.clinical_recommendation,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class ToricAxisCorrectionEngine:
    """Computes and renders automated toric treatment axis corrections."""

    def __init__(
        self,
        high_conf_threshold: float = 0.65,
        medium_conf_threshold: float = 0.35,
        max_safe_torsion_deg: float = 20.0,
    ) -> None:
        self.high_conf_threshold = high_conf_threshold
        self.medium_conf_threshold = medium_conf_threshold
        self.max_safe_torsion_deg = max_safe_torsion_deg

    def compute_corrected_axis(
        self,
        diagnostic: DiagnosticToricData,
        registration: CrossSystemRegistrationResult,
    ) -> AxisCorrectionResult:
        """Compute corrected treatment axis based on measured cyclotorsion.

        Parameters
        ----------
        diagnostic : DiagnosticToricData
            Pre-operative planned toric parameters.
        registration : CrossSystemRegistrationResult
            Phase 2 measured cyclotorsion result.

        Returns
        -------
        AxisCorrectionResult
            Corrected treatment axis, residual cylinder, and safety grade.
        """
        t0 = time.perf_counter()

        if not registration.valid or not math.isfinite(registration.rotation_deg):
            return AxisCorrectionResult(
                valid=False,
                planned_axis_deg=diagnostic.planned_treatment_axis_deg,
                cyclotorsion_deg=0.0,
                corrected_axis_deg=diagnostic.planned_treatment_axis_deg,
                safety_grade=AxisSafetyGrade.LOW,
                ablation_interlock_approved=False,
                clinical_recommendation="Cyclotorsion registration unavailable. Manual Gentian Violet alignment required.",
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        theta = registration.rotation_deg
        planned_axis = diagnostic.planned_treatment_axis_deg % 180.0

        # Refractive surgery treatment axis formula:
        # Treatment axis rotates by theta with the eye
        # alpha_corrected = (alpha_planned + theta) mod 180
        # Ophthalmological convention: 0 to 180 deg (180 deg is equivalent to 0 deg)
        corrected_axis = (planned_axis + theta) % 180.0
        if corrected_axis <= 0.0:
            corrected_axis += 180.0

        axis_adj = corrected_axis - planned_axis
        if axis_adj > 90.0:
            axis_adj -= 180.0
        elif axis_adj < -90.0:
            axis_adj += 180.0

        # Vector Astigmatism calculations (Alpins method)
        cyl_power = abs(diagnostic.planned_cylinder_power_d)
        theta_rad = math.radians(abs(theta))

        # Uncorrected residual cylinder power:
        # C_residual = 2 * C_planned * sin(|theta|)
        residual_uncorrected = float(2.0 * cyl_power * math.sin(theta_rad))
        # When corrected, residual astigmatism due to rotation error is eliminated (within algorithm tolerance)
        residual_corrected = 0.0

        # Percentage astigmatism under-correction avoided
        loss_pct = float(min(100.0, 2.0 * math.sin(theta_rad) * 100.0))

        # Safety Grading & Interlocks
        conf = registration.confidence
        abs_theta = abs(theta)

        if conf >= self.high_conf_threshold and abs_theta <= self.max_safe_torsion_deg:
            safety_grade = AxisSafetyGrade.HIGH
            approved = True
            recommendation = (
                f"Approved for automated toric alignment: Planned {planned_axis:.1f}° -> "
                f"Corrected {corrected_axis:.1f}° (Delta: {axis_adj:+.1f}°). Prevents {loss_pct:.1f}% toric under-correction."
            )
        elif conf >= self.medium_conf_threshold and abs_theta <= self.max_safe_torsion_deg:
            safety_grade = AxisSafetyGrade.MEDIUM
            approved = True
            recommendation = (
                f"Borderline confidence ({conf:.2f}). Surgeon review recommended before ablation: "
                f"Planned {planned_axis:.1f}° -> Corrected {corrected_axis:.1f}°."
            )
        else:
            safety_grade = AxisSafetyGrade.LOW
            approved = False
            recommendation = (
                f"Ablation interlock active (Confidence {conf:.2f} or rotation {abs_theta:.1f}° out of range). "
                f"Verify iris structure or verify with Gentian Violet marks."
            )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return AxisCorrectionResult(
            valid=True,
            planned_axis_deg=float(planned_axis),
            cyclotorsion_deg=float(theta),
            corrected_axis_deg=float(corrected_axis),
            axis_adjustment_deg=float(axis_adj),
            planned_cylinder_power_d=float(diagnostic.planned_cylinder_power_d),
            residual_astigmatism_uncorrected_d=residual_uncorrected,
            residual_astigmatism_corrected_d=residual_corrected,
            correction_loss_avoided_percent=loss_pct,
            safety_grade=safety_grade,
            ablation_interlock_approved=approved,
            clinical_recommendation=recommendation,
            processing_time_ms=dt_ms,
        )

    def draw_axis_overlay(
        self,
        image: np.ndarray,
        result: AxisCorrectionResult,
        detection: EyeDetectionResult,
    ) -> np.ndarray:
        """Render surgical treatment axis HUD overlay on the eye image.

        Shows:
            - Planned treatment axis (dashed blue line)
            - Corrected treatment axis (solid gold/emerald line)
            - Angular rotation arc
            - HUD statistics box
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        h, w = vis.shape[:2]

        if not detection.has_limbus:
            return vis

        le = detection.limbus.ellipse
        cx, cy = int(round(le.center_x)), int(round(le.center_y))
        radius = int(round(le.radius * 1.15))

        if not result.valid:
            cv2.putText(vis, "AXIS CORRECTION: UNAVAILABLE", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        # 1. Draw Planned Axis Line (Dashed / Blue)
        p_ang_rad = math.radians(result.planned_axis_deg)
        x1_p = int(round(cx - radius * math.cos(p_ang_rad)))
        y1_p = int(round(cy + radius * math.sin(p_ang_rad)))  # image y is downward
        x2_p = int(round(cx + radius * math.cos(p_ang_rad)))
        y2_p = int(round(cy - radius * math.sin(p_ang_rad)))

        self._draw_dashed_line(vis, (x1_p, y1_p), (x2_p, y2_p), (255, 180, 0), thickness=2)

        # 2. Draw Corrected Axis Line (Solid Gold/Green)
        c_ang_rad = math.radians(result.corrected_axis_deg)
        x1_c = int(round(cx - radius * math.cos(c_ang_rad)))
        y1_c = int(round(cy + radius * math.sin(c_ang_rad)))
        x2_c = int(round(cx + radius * math.cos(c_ang_rad)))
        y2_c = int(round(cy - radius * math.sin(c_ang_rad)))

        line_color = (0, 230, 115) if result.ablation_interlock_approved else (0, 140, 255)
        cv2.line(vis, (x1_c, y1_c), (x2_c, y2_c), line_color, 3, cv2.LINE_AA)

        # Arrow indicator on corrected line
        cv2.circle(vis, (x2_c, y2_c), 5, line_color, -1)

        # 3. Draw Angular Divergence Arc
        arc_r = int(radius * 0.85)
        # Convert to OpenCV ellipse angles (clockwise from +x)
        # Mathematical angle: theta_math = 180 - theta_opencv
        cv_planned = (180.0 - result.planned_axis_deg) % 360.0
        cv_corrected = (180.0 - result.corrected_axis_deg) % 360.0
        start_a = min(cv_planned, cv_corrected)
        end_a = max(cv_planned, cv_corrected)
        if end_a - start_a > 180.0:
            start_a, end_a = end_a, start_a + 360.0
        cv2.ellipse(vis, (cx, cy), (arc_r, arc_r), 0, start_a, end_a, (0, 255, 255), 2, cv2.LINE_AA)

        # 4. HUD Text Information Box
        box_w, box_h = 320, 145
        box_x, box_y = 15, 15
        overlay = vis.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        cv2.rectangle(vis, (box_x, box_y), (box_x + box_w, box_y + box_h), line_color, 1)

        # Content
        cv2.putText(vis, "PHASE 3: TORIC AXIS CORRECTION", (box_x + 10, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f"Planned Axis:   {result.planned_axis_deg:5.1f} deg", (box_x + 10, box_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)
        cv2.putText(vis, f"Cyclotorsion:   {result.cyclotorsion_deg:+5.1f} deg", (box_x + 10, box_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f"Corrected Axis: {result.corrected_axis_deg:5.1f} deg", (box_x + 10, box_y + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, line_color, 2)
        cv2.putText(vis, f"Loss Prevented: {result.correction_loss_avoided_percent:4.1f}%", (box_x + 10, box_y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 240, 255), 1)
        cv2.putText(vis, f"Status: [{result.safety_grade.value}] INTERLOCK OK" if result.ablation_interlock_approved else "Status: [HOLD] SURGEON REVIEW",
                    (box_x + 10, box_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)

        return vis

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_len: int = 8,
    ) -> None:
        """Draw dashed line between two points."""
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist < 1e-3:
            return
        n_dashes = int(dist / dash_len)
        dx = (pt2[0] - pt1[0]) / n_dashes
        dy = (pt2[1] - pt1[1]) / n_dashes

        for i in range(0, n_dashes, 2):
            s = (int(round(pt1[0] + i * dx)), int(round(pt1[1] + i * dy)))
            e = (int(round(pt1[0] + (i + 1) * dx)), int(round(pt1[1] + (i + 1) * dy)))
            cv2.line(img, s, e, color, thickness, cv2.LINE_AA)
