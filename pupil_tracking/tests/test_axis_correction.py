"""Unit tests for Phase 3 ToricAxisCorrectionEngine.

Verifies:
    - 3-degree rule & vector astigmatism calculation:
        Residual Cyl = 2 * |C| * sin(|theta|)
        3 deg -> ~10.5% under-correction
        4 deg -> ~13.9% under-correction
    - Axis wrapping [0, 180) degrees
    - Safety interlocks (HIGH / MEDIUM / LOW)
    - Surgical HUD overlay rendering
    - Execution latency <= 100 ms (target < 10 ms)
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult, RegistrationFailureKind
from pupil_tracking.registration.axis_correction import (
    AxisCorrectionResult,
    AxisSafetyGrade,
    DiagnosticToricData,
    ToricAxisCorrectionEngine,
)
from pupil_tracking.tests.test_cross_modality_registration import make_elita_detection


class TestToricAxisCorrectionEngine:
    def test_alpins_3_degree_rule_calculation(self):
        """Verify the 3-degree rule: 3-4 deg misalignment causes 10%-14% under-correction."""
        engine = ToricAxisCorrectionEngine()
        diag = DiagnosticToricData(
            planned_treatment_axis_deg=90.0,
            planned_cylinder_power_d=-2.50,
        )

        # 1. Test 3.0 deg rotation
        reg_3 = CrossSystemRegistrationResult(
            valid=True,
            failure=RegistrationFailureKind.OK,
            rotation_deg=3.0,
            confidence=0.85,
        )
        res_3 = engine.compute_corrected_axis(diag, reg_3)
        assert res_3.valid is True
        assert res_3.corrected_axis_deg == pytest.approx(93.0, abs=0.01)
        # 3.0 deg under-correction is approximately 10.47%
        assert 10.0 <= res_3.correction_loss_avoided_percent <= 11.0
        # Residual cylinder power = 2 * 2.50 * sin(3 deg) = 5.0 * 0.05234 = 0.26 D
        assert res_3.residual_astigmatism_uncorrected_d == pytest.approx(0.26, abs=0.02)

        # 2. Test 4.0 deg rotation
        reg_4 = CrossSystemRegistrationResult(
            valid=True,
            failure=RegistrationFailureKind.OK,
            rotation_deg=4.0,
            confidence=0.85,
        )
        res_4 = engine.compute_corrected_axis(diag, reg_4)
        # 4.0 deg under-correction is approximately 13.92%
        assert 13.5 <= res_4.correction_loss_avoided_percent <= 14.5
        # Residual cylinder power = 2 * 2.50 * sin(4 deg) = 5.0 * 0.06976 = 0.35 D
        assert res_4.residual_astigmatism_uncorrected_d == pytest.approx(0.35, abs=0.02)

    def test_axis_wrapping(self):
        """Verify proper angular wrapping in [0, 180) degrees."""
        engine = ToricAxisCorrectionEngine()

        # Wrap past 180 deg
        diag_1 = DiagnosticToricData(planned_treatment_axis_deg=177.0)
        reg_1 = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=5.0, confidence=0.8)
        res_1 = engine.compute_corrected_axis(diag_1, reg_1)
        assert res_1.corrected_axis_deg == pytest.approx(2.0, abs=0.01)

        # Wrap below 0 deg
        diag_2 = DiagnosticToricData(planned_treatment_axis_deg=3.0)
        reg_2 = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=-6.0, confidence=0.8)
        res_2 = engine.compute_corrected_axis(diag_2, reg_2)
        assert res_2.corrected_axis_deg == pytest.approx(177.0, abs=0.01)

    def test_safety_interlocks(self):
        """Verify safety grading and ablation interlock."""
        engine = ToricAxisCorrectionEngine()
        diag = DiagnosticToricData(planned_treatment_axis_deg=90.0)

        # 1. High confidence, normal rotation -> Approved
        reg_high = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=3.5, confidence=0.85)
        res_high = engine.compute_corrected_axis(diag, reg_high)
        assert res_high.safety_grade == AxisSafetyGrade.HIGH
        assert res_high.ablation_interlock_approved is True

        # 2. Medium confidence -> Medium grade, approved with review
        reg_med = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=3.5, confidence=0.50)
        res_med = engine.compute_corrected_axis(diag, reg_med)
        assert res_med.safety_grade == AxisSafetyGrade.MEDIUM
        assert res_med.ablation_interlock_approved is True

        # 3. Low confidence -> Interlock engaged
        reg_low = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=3.5, confidence=0.20)
        res_low = engine.compute_corrected_axis(diag, reg_low)
        assert res_low.safety_grade == AxisSafetyGrade.LOW
        assert res_low.ablation_interlock_approved is False

        # 4. Excessive rotation (> 20 deg) -> Interlock engaged
        reg_excess = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=25.0, confidence=0.90)
        res_excess = engine.compute_corrected_axis(diag, reg_excess)
        assert res_excess.safety_grade == AxisSafetyGrade.LOW
        assert res_excess.ablation_interlock_approved is False

    def test_overlay_rendering(self):
        """Verify surgical axis overlay rendering and HUD box."""
        engine = ToricAxisCorrectionEngine()
        diag = DiagnosticToricData(planned_treatment_axis_deg=90.0, planned_cylinder_power_d=-2.00)
        reg = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=4.5, confidence=0.85)
        res = engine.compute_corrected_axis(diag, reg)

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        det = make_elita_detection((256.0, 256.0))

        vis = engine.draw_axis_overlay(img, res, det)
        assert vis.shape == (512, 512, 3)
        # Check that pixels are non-zero (drawn HUD and axes)
        assert np.count_nonzero(vis) > 500

    def test_latency_under_100ms(self):
        """Phase 3 requirement: latency <= 100 ms."""
        engine = ToricAxisCorrectionEngine()
        diag = DiagnosticToricData(planned_treatment_axis_deg=90.0)
        reg = CrossSystemRegistrationResult(valid=True, failure=RegistrationFailureKind.OK, rotation_deg=3.0, confidence=0.85)

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        det = make_elita_detection((256.0, 256.0))

        t0 = time.perf_counter()
        res = engine.compute_corrected_axis(diag, reg)
        _ = engine.draw_axis_overlay(img, res, det)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms <= 100.0, f"Latency {elapsed_ms:.2f}ms exceeded 100ms"
        assert res.processing_time_ms <= 10.0
