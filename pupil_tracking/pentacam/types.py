"""Pentacam detection result types.

This module defines the data contract for Pentacam image detection.
It is intentionally minimal and isolated from the ELITA iris pipeline.

All types are additive and do not modify existing detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.iris.types import IrisFeatureSet, IrisROI
from pupil_tracking.utils.types import EllipseParams

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from pupil_tracking.registration.polar import PolarImage


class PentacamImageType(Enum):
    """Classification of Pentacam image content."""
    SCHEIMPFLUG_CROSS_SECTION = "SCHEIMPFLUG_CROSS_SECTION"
    ANTERIOR_SEGMENT = "ANTERIOR_SEGMENT"
    CORNEAL_MAP = "CORNEAL_MAP"
    UNKNOWN = "UNKNOWN"


class PentacamDetectionStatus(Enum):
    """Overall status of Pentacam detection."""
    OK = "OK"
    NO_IMAGE = "NO_IMAGE"
    NO_PUPIL = "NO_PUPIL"
    NO_LIMBUS = "NO_LIMBUS"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    DEGENERATE = "DEGENERATE"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class PentacamQuality(Enum):
    """Quality grade for Pentacam detection."""
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    MARGINAL = "MARGINAL"
    POOR = "POOR"
    NO_DETECTION = "NO_DETECTION"


@dataclass
class PentacamGeometry:
    """Detected anatomical geometry from a Pentacam image.

    Coordinates are in Pentacam image pixel space.
    The coordinate system is device-specific and may differ from ELITA.
    """
    pupil: Optional[EllipseParams] = None
    limbus: Optional[EllipseParams] = None

    pupil_detected: bool = False
    limbus_detected: bool = False

    # False when the radial limbus search could not bracket an iris-to-sclera
    # transition, meaning limbus_radius_px is pinned to a search-window bound
    # instead of a measured limbus.  Defaults True so externally supplied
    # geometry keeps its previous behaviour.
    limbus_localized: bool = True

    pupil_radius_px: float = 0.0
    limbus_radius_px: float = 0.0
    pupil_limbus_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "pupil_detected": self.pupil_detected,
            "limbus_detected": self.limbus_detected,
            "limbus_localized": self.limbus_localized,
            "pupil_radius_px": round(self.pupil_radius_px, 2),
            "limbus_radius_px": round(self.limbus_radius_px, 2),
            "pupil_limbus_ratio": round(self.pupil_limbus_ratio, 4),
        }
        if self.pupil is not None:
            d["pupil"] = self.pupil.to_dict()
        if self.limbus is not None:
            d["limbus"] = self.limbus.to_dict()
        return d


@dataclass
class PentacamFeature:
    """A single detected feature from a Pentacam image.

    Features are in Pentacam image pixel coordinates.
    The coordinate system is device-specific.
    """
    id: int = -1
    x: float = 0.0
    y: float = 0.0
    angle_deg: float = 0.0
    radial_norm: float = 0.5
    response: float = 0.0
    confidence: float = 0.0
    valid: bool = True
    descriptor: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "angle_deg": float(self.angle_deg),
            "radial_norm": float(self.radial_norm),
            "response": float(self.response),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "descriptor_len": (
                len(self.descriptor) if self.descriptor is not None else 0
            ),
        }


@dataclass
class PentacamFeatureSet:
    """Container for features extracted from a Pentacam image."""
    features: List[PentacamFeature] = field(default_factory=list)
    num_candidates: int = 0
    num_accepted: int = 0
    angular_coverage_ratio: float = 0.0
    largest_angular_gap_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": [f.to_dict() for f in self.features],
            "num_candidates": self.num_candidates,
            "num_accepted": self.num_accepted,
            "angular_coverage_ratio": round(self.angular_coverage_ratio, 4),
            "largest_angular_gap_deg": round(self.largest_angular_gap_deg, 2),
        }


@dataclass
class PentacamDetectionResult:
    """Top-level result of Pentacam image detection.

    This is the primary object a future cross-system matcher will consume.
    It carries Pentacam-side geometry, features, and quality assessment.
    """
    valid: bool = False
    status: PentacamDetectionStatus = PentacamDetectionStatus.NO_IMAGE
    image_type: PentacamImageType = PentacamImageType.UNKNOWN

    geometry: PentacamGeometry = field(default_factory=PentacamGeometry)
    feature_set: PentacamFeatureSet = field(default_factory=PentacamFeatureSet)

    image_width: int = 0
    image_height: int = 0
    coordinate_system: str = "pentacam_pixel"

    quality: PentacamQuality = PentacamQuality.NO_DETECTION
    confidence: float = 0.0
    failure_reason: str = ""

    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "image_type": self.image_type.value,
            "geometry": self.geometry.to_dict(),
            "feature_set": self.feature_set.to_dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_system": self.coordinate_system,
            "quality": self.quality.value,
            "confidence": round(self.confidence, 4),
            "failure_reason": self.failure_reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class PentacamEyeROI:
    """The eye-containing region of a Pentacam input, in full-image pixels.

    The region is derived from image content (UI chrome suppression), not from
    a fixed layout, so a cropped eye image and a device screenshot both resolve
    correctly without hardcoded coordinates.

    ``(x, y)`` is the top-left corner of the region in the *original* image, so
    a point ``(px, py)`` in :attr:`grayscale` corresponds to
    ``(px + x, py + y)`` in the source frame.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    valid: bool = False
    confidence: float = 0.0
    # Fraction of the source frame retained as usable ocular content.
    retained_fraction: float = 0.0
    reason: str = ""

    # Contrast-normalized grayscale crop of the region (float32, 0-255).
    grayscale: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)

    def contains(self, x: float, y: float) -> bool:
        """True when full-image pixel ``(x, y)`` lies inside this region."""
        return (
            self.x <= x < self.x + self.width
            and self.y <= y < self.y + self.height
        )

    def to_full_image(self, px: float, py: float) -> Tuple[float, float]:
        """Map a point in :attr:`grayscale` back to full-image coordinates."""
        return (float(px) + self.x, float(py) + self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "area": self.area,
            "valid": bool(self.valid),
            "confidence": round(float(self.confidence), 4),
            "retained_fraction": round(float(self.retained_fraction), 4),
            "reason": self.reason,
            "grayscale_shape": (
                list(self.grayscale.shape) if self.grayscale is not None else None
            ),
        }


@dataclass
class PentacamReferenceResult:
    """Detection-ready Pentacam reference iris representation.

    Produced by :class:`pupil_tracking.pentacam.reference.PentacamReferenceExtractor`.
    This is the last stage of this phase: it stops short of cross-system
    matching, registration and torsion estimation.

    Quality gates applied by the producer, in order:

    1. input present and single/three/four channel,
    2. eye region located,
    3. pupil detected,
    4. limbus detected *and* ``geometry.limbus_localized`` true,
    5. geometrically plausible iris ROI,
    6. usable (non-occluded) iris area,
    7. sufficient angular coverage,
    8. at least one accepted feature.

    The first failing gate sets :attr:`status` and :attr:`failure_reason`.
    """

    valid: bool = False
    status: PentacamDetectionStatus = PentacamDetectionStatus.NO_IMAGE
    quality: PentacamQuality = PentacamQuality.NO_DETECTION
    confidence: float = 0.0
    failure_reason: str = ""

    image_type: PentacamImageType = PentacamImageType.UNKNOWN
    image_width: int = 0
    image_height: int = 0
    coordinate_system: str = "pentacam_pixel"

    eye_roi: PentacamEyeROI = field(default_factory=PentacamEyeROI)
    geometry: PentacamGeometry = field(default_factory=PentacamGeometry)

    # Iris annulus geometry and the mask-aware feature representation.
    iris_roi: IrisROI = field(default_factory=IrisROI)
    feature_set: IrisFeatureSet = field(default_factory=IrisFeatureSet)

    # Polar ribbon of the iris annulus (mask marks resampled usable pixels).
    polar: Optional["PolarImage"] = None

    mask_stats: Dict[str, float] = field(default_factory=dict)

    # Per-stage wall-clock milliseconds plus the total.
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        polar_meta: Optional[Dict[str, Any]] = None
        if self.polar is not None:
            polar_meta = {
                "valid": bool(self.polar.valid),
                "num_angles": int(self.polar.num_angles),
                "num_radial": int(self.polar.num_radial),
                "inner_radius": round(float(self.polar.inner_radius), 2),
                "outer_radius": round(float(self.polar.outer_radius), 2),
                "center": [float(self.polar.center[0]), float(self.polar.center[1])],
                "shape": (
                    list(self.polar.image.shape)
                    if self.polar.image is not None
                    else None
                ),
                "valid_fraction": _mask_fraction(self.polar.mask),
            }
        return {
            "valid": bool(self.valid),
            "status": self.status.value,
            "quality": self.quality.value,
            "confidence": round(float(self.confidence), 4),
            "failure_reason": self.failure_reason,
            "image_type": self.image_type.value,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "coordinate_system": self.coordinate_system,
            "eye_roi": self.eye_roi.to_dict(),
            "geometry": self.geometry.to_dict(),
            "iris_roi": self.iris_roi.to_dict(),
            "feature_set": self.feature_set.to_dict(),
            "polar": polar_meta,
            "mask_stats": {k: _round(v) for k, v in self.mask_stats.items()},
            "timings_ms": {k: _round(v) for k, v in self.timings_ms.items()},
        }


def _mask_fraction(mask: Optional[np.ndarray]) -> Optional[float]:
    """Fraction of set pixels in a uint8/bool mask, or None when absent."""
    if mask is None or mask.size == 0:
        return None
    nonzero = int(np.count_nonzero(mask))
    return round(nonzero / float(mask.size), 4)


def _round(value: Any) -> Any:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value
