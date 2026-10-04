"""Pentacam reference-iris extraction.

Turns a Pentacam input -- a cropped eye image, a raw grayscale cross-section,
or a full device screenshot with UI chrome -- into a *detection-ready*
reference representation:

    input -> eye region -> pupil/limbus geometry -> iris ROI + masks
          -> mask-aware features -> polar ribbon -> quality metadata

This is deliberately the last stage of the phase.  It does **not** match the
result against ELITA, estimate rotation, or compute torsion.

Everything is deterministic: classical CV only, reinforcement learning off,
no CNN, no learned model, and no reliance on call history (the detector's RL
branch stays disabled so repeated calls return identical output).

Reuse, not duplication: pupil/limbus localization comes from the existing
:class:`~pupil_tracking.pentacam.detector.PentacamIrisDetector`, and the iris
ROI, masking, features and polar ribbon come from the mature
:mod:`pupil_tracking.iris` and :mod:`pupil_tracking.registration.polar`
components that already back the ELITA path.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.iris.types import IrisStatus
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.preprocess import (
    build_ui_mask,
    content_bbox,
    crop,
    normalize_gray,
    to_bgr,
    to_gray,
)
from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamEyeROI,
    PentacamQuality,
    PentacamReferenceResult,
)
from pupil_tracking.registration.polar import PolarUnwrapper

logger = logging.getLogger(__name__)


class PentacamReferenceExtractor:
    """Produces a detection-ready Pentacam reference iris representation.

    Parameters
    ----------
    polar_num_angles, polar_num_radial
        Polar ribbon resolution.  ``None`` falls back to the shared
        registration config so both modalities unwrap at the same density.
    inner_inset_frac, outer_inset_frac
        Iris annulus insets, as a fraction of the pupil and limbus radius.
    min_retained_fraction
        Minimum fraction of the frame that must survive UI suppression before
        a region can be called an eye region.
    min_usable_fraction
        Minimum fraction of the iris annulus that must be non-occluded.
    min_polar_coverage
        Minimum fraction of polar ribbon cells that must resample real pixels.
    min_angular_coverage
        Minimum 1 - largest_gap/360 angular coverage of accepted features.
    min_features
        Minimum number of accepted features for a usable reference.
    good_usable_fraction, good_polar_coverage, good_angular_coverage
        Thresholds above which a valid reference is graded ``GOOD`` rather than
        ``ACCEPTABLE``.
    """

    def __init__(
        self,
        *,
        polar_num_angles: Optional[int] = None,
        polar_num_radial: Optional[int] = None,
        inner_inset_frac: float = 0.12,
        outer_inset_frac: float = 0.12,
        min_retained_fraction: float = 0.02,
        min_usable_fraction: float = 0.30,
        min_polar_coverage: float = 0.50,
        min_angular_coverage: float = 0.50,
        min_features: int = 10,
        good_usable_fraction: float = 0.60,
        good_polar_coverage: float = 0.75,
        good_angular_coverage: float = 0.75,
    ) -> None:
        self.min_retained_fraction = float(min_retained_fraction)
        self.min_usable_fraction = float(min_usable_fraction)
        self.min_polar_coverage = float(min_polar_coverage)
        self.min_angular_coverage = float(min_angular_coverage)
        self.min_features = int(min_features)
        self.good_usable_fraction = float(good_usable_fraction)
        self.good_polar_coverage = float(good_polar_coverage)
        self.good_angular_coverage = float(good_angular_coverage)

        self.detector = PentacamIrisDetector()

        # Force the classical, deterministic path: no RL tuning (which would
        # make output depend on the previous call) and no CNN encoder.
        iris_config = IrisConfig(
            inner_inset_frac=float(inner_inset_frac),
            outer_inset_frac=float(outer_inset_frac),
            rl_enabled=False,
            use_cnn_segmentation=False,
            use_cnn_encoding=False,
        )
        self.iris_detector = IrisFeatureDetector(config=iris_config)
        self.polar_unwrapper = PolarUnwrapper(
            num_angles=polar_num_angles,
            num_radial=polar_num_radial,
        )

    # ---- public API ------------------------------------------------------ #

    def extract(self, image: Optional[np.ndarray]) -> PentacamReferenceResult:
        """Run the full reference pipeline.  Never raises for bad input."""
        started = time.perf_counter()
        result = PentacamReferenceResult()

        if not _is_supported(image):
            result.status = PentacamDetectionStatus.NO_IMAGE
            result.failure_reason = _describe_bad_input(image)
            result.timings_ms["total"] = _elapsed_ms(started)
            return result

        gray = to_gray(image)
        # Geometry and polar stages want two channels; colour-aware iris
        # masking needs three.  Normalise once here rather than letting each
        # stage guess, which also accepts single-channel (H, W, 1) arrays.
        bgr = to_bgr(image)
        result.image_width = int(gray.shape[1])
        result.image_height = int(gray.shape[0])

        # --- eye region (content-derived, no layout assumptions) ---------- #
        t = time.perf_counter()
        ui_mask = build_ui_mask(gray)
        result.eye_roi = self._build_eye_roi(gray, ui_mask)
        result.timings_ms["eye_roi"] = _elapsed_ms(t)

        if not result.eye_roi.valid:
            result.status = PentacamDetectionStatus.DEGENERATE
            result.failure_reason = f"no usable eye region: {result.eye_roi.reason}"
            result.timings_ms["total"] = _elapsed_ms(started)
            return result

        # --- pupil / limbus geometry -------------------------------------- #
        t = time.perf_counter()
        detection = self.detector.detect(gray)
        result.geometry = detection.geometry
        result.image_type = detection.image_type
        result.timings_ms["detection"] = _elapsed_ms(t)

        rejection = self._reject_geometry(detection)
        if rejection is not None:
            result.status, result.failure_reason = rejection
            result.timings_ms["total"] = _elapsed_ms(started)
            return result

        # UI chrome counts as occlusion so features never land on text or
        # reticles, even when they fall inside the iris annulus.
        external_occlusion = ui_mask == 0

        # --- iris ROI, masks and features --------------------------------- #
        t = time.perf_counter()
        iris_result = self.iris_detector.detect(
            bgr,
            pupil=detection.geometry.pupil,
            limbus=detection.geometry.limbus,
            external_occlusion=external_occlusion,
        )
        result.iris_roi = iris_result.feature_set.roi
        result.feature_set = iris_result.feature_set
        result.mask_stats = dict(iris_result.mask_stats)
        result.timings_ms["features"] = _elapsed_ms(t)

        # --- polar ribbon -------------------------------------------------- #
        t = time.perf_counter()
        result.polar = self._unwrap(gray, detection)
        polar_coverage = _polar_coverage(result.polar)
        result.mask_stats["polar_coverage"] = polar_coverage
        angular_coverage = float(detection.feature_set.angular_coverage_ratio)
        result.mask_stats["angular_coverage_ratio"] = angular_coverage
        result.timings_ms["polar"] = _elapsed_ms(t)

        # --- quality gates ------------------------------------------------- #
        rejection = self._reject_reference(
            result, iris_result, polar_coverage, angular_coverage
        )
        if rejection is not None:
            result.status, result.failure_reason = rejection
            result.timings_ms["total"] = _elapsed_ms(started)
            return result

        usable = float(result.feature_set.usable_fraction)

        result.valid = True
        result.status = PentacamDetectionStatus.OK
        result.quality = (
            PentacamQuality.GOOD
            if (
                usable >= self.good_usable_fraction
                and polar_coverage >= self.good_polar_coverage
                and angular_coverage >= self.good_angular_coverage
            )
            else PentacamQuality.ACCEPTABLE
        )
        result.confidence = self._confidence(result)
        result.timings_ms["total"] = _elapsed_ms(started)
        return result

    # ---- stages ---------------------------------------------------------- #

    def _build_eye_roi(
        self, gray: np.ndarray, ui_mask: np.ndarray
    ) -> PentacamEyeROI:
        """Locate the eye region and build its normalized grayscale view.

        ROI confidence is the fraction of the declared region that is genuine
        ocular content rather than suppressed UI chrome, so a screenshot with
        text and reticles scores lower than a clean crop of the same eye.
        """
        roi = PentacamEyeROI()
        total = float(ui_mask.size)
        retained = float(np.count_nonzero(ui_mask)) / total if total else 0.0
        roi.retained_fraction = retained

        bbox = content_bbox(ui_mask, min_fraction=self.min_retained_fraction)
        if bbox is None:
            roi.reason = (
                f"only {retained:.1%} of the frame survived UI suppression "
                f"(minimum {self.min_retained_fraction:.1%})"
            )
            return roi

        x, y, w, h = bbox
        roi.x, roi.y, roi.width, roi.height = x, y, w, h
        roi.valid = True
        roi.reason = "ok"

        window = ui_mask[y : y + h, x : x + w]
        roi.confidence = float(np.count_nonzero(window)) / float(window.size)

        roi_gray = crop(gray, bbox)
        roi.grayscale = normalize_gray(roi_gray)
        return roi

    @staticmethod
    def _reject_geometry(
        detection: PentacamDetectionResult,
    ) -> Optional[tuple]:
        """First failing geometry gate, or ``None`` when geometry is usable.

        Gate 4 is the ``limbus_localized`` check: a limbus radius pinned to a
        search-window bound is not a measured limbus, so it must not yield a
        reference iris.  This mirrors the detector's own validity rule rather
        than second-guessing it.
        """
        geometry = detection.geometry

        if not geometry.pupil_detected or geometry.pupil is None:
            return (
                PentacamDetectionStatus.NO_PUPIL,
                detection.failure_reason or "pupil not detected",
            )

        if not geometry.limbus_detected or geometry.limbus is None:
            return (
                PentacamDetectionStatus.NO_LIMBUS,
                detection.failure_reason or "limbus not detected",
            )

        if not geometry.limbus_localized:
            return (
                PentacamDetectionStatus.NO_LIMBUS,
                detection.failure_reason
                or "limbus not localized: radius pinned to the search window bound",
            )

        if not detection.valid:
            return (
                PentacamDetectionStatus.NO_IMAGE,
                detection.failure_reason or "detector rejected the image",
            )

        return None

    def _unwrap(self, gray: np.ndarray, detection: PentacamDetectionResult) -> Any:
        """Unwrap the iris annulus into a polar ribbon."""
        geometry = detection.geometry
        return self.polar_unwrapper.unwrap(
            gray,
            pupil_center=(geometry.pupil.center_x, geometry.pupil.center_y),
            pupil_axes=(geometry.pupil.semi_major, geometry.pupil.semi_minor),
            pupil_angle_deg=geometry.pupil.angle_deg,
            limbus_center=(geometry.limbus.center_x, geometry.limbus.center_y),
            limbus_axes=(geometry.limbus.semi_major, geometry.limbus.semi_minor),
            limbus_angle_deg=geometry.limbus.angle_deg,
        )

    def _reject_reference(
        self,
        result: PentacamReferenceResult,
        iris_result: Any,
        polar_coverage: float,
        angular_coverage: float,
    ) -> Optional[tuple]:
        """First failing reference gate, or ``None`` when the reference is usable."""
        if iris_result.status == IrisStatus.NO_ROI or not result.iris_roi.valid:
            return (
                PentacamDetectionStatus.DEGENERATE,
                f"implausible iris geometry: "
                f"{result.iris_roi.reason or 'invalid ROI'}",
            )

        if not result.polar.valid:
            return (
                PentacamDetectionStatus.DEGENERATE,
                "polar unwrapping produced no valid angular range",
            )

        usable = float(result.feature_set.usable_fraction)
        if usable < self.min_usable_fraction:
            return (
                PentacamDetectionStatus.INSUFFICIENT_FEATURES,
                f"usable iris fraction {usable:.3f} below minimum "
                f"{self.min_usable_fraction:.3f}",
            )

        if polar_coverage < self.min_polar_coverage:
            return (
                PentacamDetectionStatus.INSUFFICIENT_FEATURES,
                f"polar ribbon coverage {polar_coverage:.3f} below minimum "
                f"{self.min_polar_coverage:.3f}",
            )

        if angular_coverage < self.min_angular_coverage:
            return (
                PentacamDetectionStatus.INSUFFICIENT_FEATURES,
                f"angular coverage {angular_coverage:.3f} below minimum "
                f"{self.min_angular_coverage:.3f}",
            )

        accepted = int(result.feature_set.num_accepted)
        if accepted < self.min_features:
            return (
                PentacamDetectionStatus.INSUFFICIENT_FEATURES,
                f"only {accepted} accepted features, minimum {self.min_features}",
            )

        return None

    @staticmethod
    def _confidence(result: PentacamReferenceResult) -> float:
        """Blend mean feature confidence with ribbon coverage, clamped to [0, 1]."""
        features = result.feature_set.features
        if not features:
            return 0.0
        mean_conf = float(
            np.mean([f.confidence for f in features if f.valid])
        ) if any(f.valid for f in features) else 0.0
        polar = _polar_coverage(result.polar)
        return float(np.clip(mean_conf * (0.5 + 0.5 * polar), 0.0, 1.0))


# ---- helpers ------------------------------------------------------------- #


def _is_supported(image: Optional[np.ndarray]) -> bool:
    if image is None or not isinstance(image, np.ndarray):
        return False
    if image.size == 0 or image.ndim not in (2, 3):
        return False
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return False
    return True


def _describe_bad_input(image: Optional[np.ndarray]) -> str:
    if image is None:
        return "no image supplied"
    if not isinstance(image, np.ndarray):
        return f"expected numpy.ndarray, got {type(image).__name__}"
    if image.size == 0:
        return f"empty image array with shape {image.shape}"
    if image.ndim not in (2, 3):
        return f"unsupported image dimensions: {image.shape}"
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return f"unsupported channel count: {image.shape[2]}"
    return "unsupported image"


def _polar_coverage(polar: Any) -> float:
    """Fraction of polar ribbon cells that sampled a real in-bounds pixel."""
    if polar is None or polar.mask is None or polar.mask.size == 0:
        return 0.0
    return float(np.count_nonzero(polar.mask)) / float(polar.mask.size)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
