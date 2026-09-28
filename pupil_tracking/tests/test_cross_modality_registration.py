"""Synthetic rigid-rotation tests for CrossModalityRegistrationEngine.

These tests do not establish accuracy on clinical images or nonrigid eye motion.
They verify rotation recovery on generated images and implementation behavior.
    - Rotation recovery on synthetic rigid rotations
    - Model-based toric metric arithmetic (not clinical outcomes)
    - Intorsion vs Excyclotorsion classification for OD and OS laterality
    - RGB and Grayscale ELITA image compatibility
    - Warm-reference CPU latency <= 150 ms on the test environment
"""

from __future__ import annotations

import math
import time
import cv2
import numpy as np
import pytest

from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.tests.test_pentacam_detector import create_synthetic_pentacam_image
from pupil_tracking.utils.types import (
    CalibrationInfo,
    CornealCenterResult,
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)


def rotate_image(img: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    """Rotate image by angle_deg (positive = counter-clockwise) around center."""
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def make_elita_detection(center: tuple[float, float], pr: float = 60.0, lr: float = 160.0) -> EyeDetectionResult:
    """Create a mock EyeDetectionResult matching the synthetic eye."""
    cx, cy = center
    pe = EllipseParams(center_x=cx, center_y=cy, semi_major=pr, semi_minor=pr, angle_deg=0.0, fit_quality=0.95)
    le = EllipseParams(center_x=cx, center_y=cy, semi_major=lr, semi_minor=lr, angle_deg=0.0, fit_quality=0.95)
    cc = CornealCenterResult(center_px=(cx, cy), offset_px=(0.0, 0.0), offset_magnitude_px=0.0, offset_angle_deg=0.0)

    return EyeDetectionResult(
        pupil=PupilDetection(ellipse=pe, detected=True, confidence=0.95),
        limbus=LimbusDetection(ellipse=le, detected=True, confidence=0.95),
        corneal_center=cc,
        overall_quality=DetectionQuality.CLINICAL,
        overall_confidence=0.95,
    )


class TestCrossModalityRegistrationEngine:
    def test_init(self):
        engine = CrossModalityRegistrationEngine()
        assert engine.num_angles == 360
        assert engine.num_radial == 64

    @pytest.mark.parametrize("target_angle", [0.0, 3.0, -3.0, 5.0, -8.0])
    def test_rotation_recovery_accuracy(self, target_angle: float):
        """Verify cyclotorsion recovery within <= 1.5 deg error margin across multiple angles."""
        img_ref = create_synthetic_pentacam_image(with_ui=True)
        h, w = img_ref.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Create rotated ELITA image (converted to 3-channel RGB as in surgery)
        img_rot_gray = rotate_image(img_ref, target_angle, center)
        img_elita_rgb = cv2.cvtColor(img_rot_gray, cv2.COLOR_GRAY2BGR)

        det_elita = make_elita_detection(center)
        engine = CrossModalityRegistrationEngine()
        # Warmup for JIT/import overhead on cold start
        _ = engine.register(img_ref, img_elita_rgb, elita_detection=det_elita, laterality="OD", mode="static")

        res = engine.register(
            pentacam_image=img_ref,
            elita_image=img_elita_rgb,
            elita_detection=det_elita,
            laterality="OD",
            mode="static",
        )

        assert res.valid is True
        error_deg = abs(res.rotation_deg - target_angle)
        # Clinical requirement: error <= 1.5 deg (tolerance target +- 1.0 deg)
        assert error_deg <= 1.5, f"Expected {target_angle} deg, got {res.rotation_deg:.2f} deg (error {error_deg:.2f} deg)"
        assert res.confidence >= 0.30
        assert res.processing_time_ms < 150.0

    def test_clinical_cutoff_classification(self):
        """Verify clinical cutoff classification: ACCEPTABLE (<=1.5), BORDERLINE (1.5-3.0), CRITICAL (>3.0)."""
        engine = CrossModalityRegistrationEngine()
        img_ref = create_synthetic_pentacam_image(with_ui=False)
        center = (256.0, 256.0)
        det_elita = make_elita_detection(center)

        # 1. Target 1.0 deg -> ACCEPTABLE
        img_1 = rotate_image(img_ref, 1.0, center)
        res_1 = engine.register(img_ref, img_1, elita_detection=det_elita)
        assert res_1.clinical_impact == "ACCEPTABLE"
        assert res_1.astigmatism_loss_percent < 5.0

        # 2. Target 2.0 deg -> BORDERLINE
        img_2 = rotate_image(img_ref, 2.0, center)
        res_2 = engine.register(img_ref, img_2, elita_detection=det_elita)
        assert res_2.clinical_impact == "BORDERLINE"

        # 3. Target 5.0 deg -> CRITICAL
        img_3 = rotate_image(img_ref, 5.0, center)
        res_3 = engine.register(img_ref, img_3, elita_detection=det_elita)
        assert res_3.clinical_impact == "CRITICAL"
        # Residual cylinder is linear near zero; effectiveness loss is quadratic.
        assert res_3.astigmatism_loss_percent > 10.0
        measured_theta = math.radians(abs(res_3.rotation_deg))
        assert res_3.astigmatism_loss_percent == pytest.approx(
            2.0 * math.sin(measured_theta) * 100.0, abs=0.1
        )
        assert res_3.toric_effectiveness_loss_percent == pytest.approx(
            (1.0 - math.cos(2.0 * measured_theta)) * 100.0, abs=0.1
        )

    def test_reference_detection_is_cached_by_image_content(self, monkeypatch):
        engine = CrossModalityRegistrationEngine()
        img_ref = create_synthetic_pentacam_image(with_ui=False)
        center = (256.0, 256.0)
        img_elita = rotate_image(img_ref, 2.0, center)
        detection = make_elita_detection(center)
        original_detect = engine.pentacam_detector.detect
        calls = 0

        def counted_detect(image, *args, **kwargs):
            nonlocal calls
            calls += 1
            return original_detect(image, *args, **kwargs)

        monkeypatch.setattr(engine.pentacam_detector, "detect", counted_detect)
        engine.register(img_ref, img_elita, elita_detection=detection)
        engine.register(img_ref, img_elita, elita_detection=detection)
        assert calls == 1

        img_ref[0, 0] = (int(img_ref[0, 0]) + 1) % 256
        engine.register(img_ref, img_elita, elita_detection=detection)
        assert calls == 2

    def test_laterality_direction(self):
        """Verify OD vs OS intorsion and excyclotorsion assignment."""
        engine = CrossModalityRegistrationEngine()
        img_ref = create_synthetic_pentacam_image(with_ui=False)
        center = (256.0, 256.0)
        det_elita = make_elita_detection(center)

        # Rotate +3.0 deg (counter-clockwise)
        img_ccw = rotate_image(img_ref, 3.0, center)

        # For OD (Right Eye): Counter-Clockwise is EXCYCLOTORSION
        res_od = engine.register(img_ref, img_ccw, elita_detection=det_elita, laterality="OD")
        assert res_od.torsion_direction == "EXCYCLOTORSION"

        # For OS (Left Eye): Counter-Clockwise is INTORSION
        res_os = engine.register(img_ref, img_ccw, elita_detection=det_elita, laterality="OS")
        assert res_os.torsion_direction == "INTORSION"

    def test_grayscale_and_rgb_modes(self):
        """Verify both grayscale (e.g. IR camera) and RGB (e.g. current ELITA camera) work identically."""
        engine = CrossModalityRegistrationEngine()
        img_ref = create_synthetic_pentacam_image(with_ui=False)
        center = (256.0, 256.0)
        det_elita = make_elita_detection(center)

        img_rot = rotate_image(img_ref, 3.0, center)

        # 1. Grayscale mode
        res_gray = engine.register(img_ref, img_rot, elita_detection=det_elita)
        # 2. RGB mode
        img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2BGR)
        res_rgb = engine.register(img_ref, img_rgb, elita_detection=det_elita)

        assert abs(res_gray.rotation_deg - res_rgb.rotation_deg) < 0.2
        assert res_gray.valid is True and res_rgb.valid is True

    def test_latency_under_150ms(self):
        """Ensure end-to-end execution satisfies the strict <= 150 ms requirement."""
        engine = CrossModalityRegistrationEngine()
        img_ref = create_synthetic_pentacam_image(with_ui=False)
        center = (256.0, 256.0)
        det_elita = make_elita_detection(center)
        img_rot = rotate_image(img_ref, 2.0, center)

        # Warm-up
        _ = engine.register(img_ref, img_rot, elita_detection=det_elita)

        # Benchmark
        t0 = time.perf_counter()
        res = engine.register(img_ref, img_rot, elita_detection=det_elita)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms <= 150.0, f"Latency {elapsed_ms:.1f}ms exceeded 150ms limit"
        assert res.processing_time_ms <= 150.0
