"""
Master registration engine for cyclotorsion detection.

Orchestrates the full pipeline:
    1. Validate inputs (two images + two detections)
    2. Run all enabled streams in parallel
    3. Fuse stream results
    4. Return unified RegistrationResult
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from pupil_tracking.registration.fusion import FusionEngine
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.registration.streams.phase_correlation import PhaseCorrelationStream
from pupil_tracking.registration.streams.deep_matcher import DeepMatcherStream
from pupil_tracking.registration.streams.ink_tracker import InkTrackerStream
from pupil_tracking.registration.streams.vessel_tracker import VesselTrackerStream
from pupil_tracking.registration.streams.custom_feature import CustomFeatureStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    RegistrationQuality,
    RegistrationResult,
    StreamResult,
)

logger = logging.getLogger(__name__)


class RegistrationEngine:
    """Master engine for iris registration and cyclotorsion detection.

    Manages all detection streams and the fusion engine.  Provides
    the main ``register()`` method that accepts two eye images with
    their detections and returns a fused cyclotorsion result.

    Usage
    -----
    >>> engine = RegistrationEngine()
    >>> result = engine.register(img_ref, img_curr, det_ref, det_curr)
    >>> print(f"Torsion: {result.torsion_deg:.3f}° ({result.quality.value})")
    """

    def __init__(self):
        cfg = get_config().registration
        self.enabled = getattr(cfg, "enabled", True)
        self.streams: Dict[str, BaseStream] = {}
        self.fusion = FusionEngine()

        # Initialise enabled streams
        if cfg.enable_phase_correlation:
            self.streams["phase_correlation"] = PhaseCorrelationStream()

        if cfg.enable_deep_matcher:
            self.streams["deep_matcher"] = DeepMatcherStream()

        if cfg.enable_ink_tracker:
            self.streams["ink_markers"] = InkTrackerStream()

        if cfg.enable_vessel_tracker:
            self.streams["limbal_vessels"] = VesselTrackerStream()

        if cfg.enable_custom_feature:
            stream = CustomFeatureStream()
            if stream.available:
                self.streams["custom_feature"] = stream
            else:
                logger.info("Custom feature stream skipped — no model available")

        logger.info(
            "RegistrationEngine initialised with %d streams (enabled=%s): %s",
            len(self.streams), self.enabled, list(self.streams.keys()),
        )

    def register(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> RegistrationResult:
        """Run all streams and fuse results."""
        start = time.perf_counter()

        # Check if master registration is disabled (centration-only mode)
        if not self.enabled:
            logger.debug("Registration component disabled — running centration only")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
                quality=RegistrationQuality.NO_RESULT,
                metadata={"disabled": True, "mode": "centration_only"},
            )

        # Validate inputs
        if not detection_ref.has_both:
            logger.warning("Reference image lacks pupil+limbus detection")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
            )

        if not detection_curr.has_both:
            logger.warning("Current image lacks pupil+limbus detection")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
            )

        # Run all enabled streams
        stream_results: Dict[str, StreamResult] = {}
        for name, stream in self.streams.items():
            logger.debug("Running stream: %s", name)
            sr = stream.run(img_ref, img_curr, detection_ref, detection_curr)
            stream_results[name] = sr
            logger.debug(
                "Stream %s: torsion=%.3f° conf=%.3f valid=%s",
                name,
                sr.torsion_deg if sr.torsion_deg is not None else float('nan'),
                sr.confidence,
                sr.valid,
            )

        # Fuse
        result = self.fusion.fuse(stream_results)
        result.total_processing_time_ms = (
            (time.perf_counter() - start) * 1000.0
        )

        logger.info(
            "Registration: torsion=%.3f° conf=%.3f quality=%s streams=%d/%d time=%.0fms",
            result.torsion_deg,
            result.confidence,
            result.quality.value,
            result.agreeing_streams,
            result.active_streams,
            result.total_processing_time_ms,
        )

        return result

    def get_stream_names(self) -> List[str]:
        """Return names of all initialised streams."""
        return list(self.streams.keys())

    def enable_stream(self, name: str) -> bool:
        """Enable a stream by name. Returns True if found."""
        if name in self.streams:
            self.streams[name].enabled = True
            return True
        return False

    def disable_stream(self, name: str) -> bool:
        """Disable a stream by name. Returns True if found."""
        if name in self.streams:
            self.streams[name].enabled = False
            return True
        return False

    def set_master_enabled(self, enabled: bool) -> None:
        """Enable or disable the entire registration component.

        When disabled, the system runs purely in centration mode without
        computing cyclotorsion or running registration streams.
        """
        self.enabled = bool(enabled)
        logger.info("Registration master enabled set to: %s", self.enabled)

    def set_ink_tracker_enabled(self, enabled: bool) -> None:
        """Enable or disable Stream C (purple limbal ink tracker)."""
        if enabled:
            self.enable_stream("ink_markers")
        else:
            self.disable_stream("ink_markers")
        logger.info("Ink marker stream enabled set to: %s", enabled)

    def set_iris_features_enabled(self, enabled: bool) -> None:
        """Enable or disable iris landmark / feature extraction streams."""
        if enabled:
            self.enable_stream("custom_feature")
            self.enable_stream("deep_matcher")
        else:
            self.disable_stream("custom_feature")
            self.disable_stream("deep_matcher")
        logger.info("Iris feature streams enabled set to: %s", enabled)

    def set_phase_correlation_enabled(self, enabled: bool) -> None:
        """Enable or disable polar FFT phase correlation stream."""
        if enabled:
            self.enable_stream("phase_correlation")
        else:
            self.disable_stream("phase_correlation")
        logger.info("Phase correlation stream enabled set to: %s", enabled)

    def register_pentacam(
        self,
        img_pentacam: np.ndarray,
        img_elita: np.ndarray,
        pentacam_result: Optional[Any] = None,
        elita_detection: Optional[EyeDetectionResult] = None,
        laterality: str = "OD",
        mode: str = "static",
    ):
        """Cross-modality registration between seated Pentacam IR and supine ELITA image."""
        if not self.enabled:
            logger.info("Registration disabled — skipping cross-modality registration (centration only mode)")
            from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult, RegistrationFailureKind
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.UNKNOWN_ERROR,
                failure_reason="Registration component is turned OFF in Settings (Centration Only Mode).",
                clinical_impact="DISABLED",
                quality_assessment="Bypassed (Centration Only Mode)",
                torsion_direction="NEUTRAL",
                laterality=laterality,
            )

        from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
        engine = CrossModalityRegistrationEngine()
        return engine.register(
            pentacam_image=img_pentacam,
            elita_image=img_elita,
            pentacam_result=pentacam_result,
            elita_detection=elita_detection,
            laterality=laterality,
            mode=mode,
        )

