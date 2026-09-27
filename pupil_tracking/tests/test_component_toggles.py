"""Tests for independent component toggling in Settings.

Verifies:
    1. Centration pipeline runs independently when iris registration is OFF.
    2. Iris registration component can be toggled ON/OFF independently.
    3. Gentian violet / purple limbal ink tracker can be toggled ON/OFF independently.
    4. Iris feature detection function can be toggled ON/OFF independently.
    5. Each component can be tested independently without mutual dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.streams.ink_tracker import (
    InkTrackerStream,
    detect_limbal_purple_markers,
)
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.utils.config import PupilTrackingConfig, get_config
from pupil_tracking.utils.types import (
    CalibrationInfo,
    CornealCenterResult,
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)


def _make_synthetic_eye_with_ink(width: int = 400, height: int = 400) -> tuple[np.ndarray, EyeDetectionResult]:
    """Create a synthetic eye image with pupil, limbus, and purple ink markings."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cx, cy = width // 2, height // 2

    # Draw iris (dark gray)
    cv2.circle(img, (cx, cy), 90, (110, 110, 110), -1)
    # Draw pupil (near black)
    cv2.circle(img, (cx, cy), 35, (25, 25, 25), -1)

    # Place two vivid purple ink markings near the limbus (Gentian violet: BGR around 200, 0, 220)
    # Marker 1: near 3 o'clock (r = 90 px, dx = 85, dy = 0)
    m1_x, m1_y = cx + 85, cy
    cv2.circle(img, (m1_x, m1_y), 8, (210, 20, 200), -1)

    # Marker 2: near 9 o'clock (r = 90 px, dx = -85, dy = 0)
    m2_x, m2_y = cx - 85, cy
    cv2.circle(img, (m2_x, m2_y), 8, (210, 20, 200), -1)

    detection = EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(center_x=float(cx), center_y=float(cy), semi_major=35.0, semi_minor=35.0),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(center_x=float(cx), center_y=float(cy), semi_major=90.0, semi_minor=90.0),
            confidence=0.92,
        ),
        corneal_center=CornealCenterResult(
            center_px=(float(cx), float(cy)),
            valid=True,
            confidence=0.95,
        ),
    )
    return img, detection


class TestIndependentComponentToggles:
    """Test independent enable/disable and execution of modular components."""

    def test_centration_runs_with_registration_disabled(self):
        """When iris registration is OFF, the system runs purely in centration mode."""
        cfg = get_config()
        orig = cfg.registration.enabled
        try:
            cfg.registration.enabled = False

            engine = RegistrationEngine()
            assert engine.enabled is False

            img, det = _make_synthetic_eye_with_ink()

            # Master register() should immediately exit with centration_only metadata
            res = engine.register(img, img, det, det)
            assert res.metadata.get("disabled") is True
            assert res.metadata.get("mode") == "centration_only"
            assert res.total_processing_time_ms < 10.0  # Ultra-low latency bypass

            # Cross-modality registration should also bypass
            p_res = engine.register_pentacam(img, img, elita_detection=det)
            assert p_res.valid is False
            assert p_res.clinical_impact == "DISABLED"
            assert "Centration Only" in p_res.quality_assessment
        finally:
            cfg.registration.enabled = orig

    def test_registration_toggle_on_and_off(self):
        """Toggle registration engine ON and OFF dynamically."""
        engine = RegistrationEngine()
        engine.set_master_enabled(True)
        assert engine.enabled is True

        engine.set_master_enabled(False)
        assert engine.enabled is False

        img, det = _make_synthetic_eye_with_ink()
        res_off = engine.register(img, img, det, det)
        assert res_off.metadata.get("disabled") is True

        engine.set_master_enabled(True)
        assert engine.enabled is True

    def test_purple_ink_tracker_standalone_detection(self):
        """detect_limbal_purple_markers detects purple markings near limbus."""
        img, det = _make_synthetic_eye_with_ink()
        markers = detect_limbal_purple_markers(img, det)
        # Should detect the 2 placed Gentian violet ink marks
        assert len(markers) == 2
        for mx, my, mr in markers:
            assert mr >= 4.0
            # Distance to limbus center should be close to 85 px
            dist = np.hypot(mx - 200, my - 200)
            assert 70.0 <= dist <= 100.0

    def test_purple_ink_tracker_stream_toggle(self):
        """InkTrackerStream can be toggled ON and OFF independently."""
        stream = InkTrackerStream()
        assert stream.enabled is True

        img, det = _make_synthetic_eye_with_ink()

        # When enabled, it computes rotation
        res_on = stream.run(img, img, det, det)
        assert res_on.stream.value == "ink_markers"
        assert "disabled" not in res_on.metadata

        # Turn OFF ink tracker
        stream.enabled = False
        assert stream.enabled is False

        res_off = stream.run(img, img, det, det)
        assert res_off.metadata.get("disabled") is True
        assert res_off.torsion_deg is None

    def test_iris_feature_detector_standalone(self):
        """IrisFeatureDetector can run and extract features independently."""
        img, det = _make_synthetic_eye_with_ink()
        detector = IrisFeatureDetector()

        res = detector.detect(img, det.pupil.ellipse, det.limbus.ellipse)
        assert res is not None
        assert hasattr(res, "status")
        # Should successfully extract or compute without error
        assert hasattr(res, "feature_set") and hasattr(res.feature_set, "features")

    def test_registration_engine_stream_toggles(self):
        """RegistrationEngine enables/disables specific streams independently."""
        engine = RegistrationEngine()

        # Toggle ink tracker stream
        engine.set_ink_tracker_enabled(False)
        if "ink_markers" in engine.streams:
            assert engine.streams["ink_markers"].enabled is False

        engine.set_ink_tracker_enabled(True)
        if "ink_markers" in engine.streams:
            assert engine.streams["ink_markers"].enabled is True

        # Toggle iris feature stream
        engine.set_iris_features_enabled(False)
        if "custom_feature" in engine.streams:
            assert engine.streams["custom_feature"].enabled is False
        if "deep_matcher" in engine.streams:
            assert engine.streams["deep_matcher"].enabled is False

        engine.set_iris_features_enabled(True)
        if "custom_feature" in engine.streams:
            assert engine.streams["custom_feature"].enabled is True
        if "deep_matcher" in engine.streams:
            assert engine.streams["deep_matcher"].enabled is True

        # Toggle phase correlation stream
        engine.set_phase_correlation_enabled(False)
        if "phase_correlation" in engine.streams:
            assert engine.streams["phase_correlation"].enabled is False

        engine.set_phase_correlation_enabled(True)
        if "phase_correlation" in engine.streams:
            assert engine.streams["phase_correlation"].enabled is True

    def test_config_defaults_and_fields(self):
        """PupilTrackingConfig contains all required toggle attributes."""
        cfg = PupilTrackingConfig()
        assert hasattr(cfg.registration, "enabled")
        assert hasattr(cfg.registration, "enable_iris_features")
        assert hasattr(cfg.registration, "enable_ink_tracker")
        assert hasattr(cfg.registration, "enable_phase_correlation")

        assert cfg.registration.enabled is True
        assert cfg.registration.enable_iris_features is True
        assert cfg.registration.enable_ink_tracker is True
        assert cfg.registration.enable_phase_correlation is True
