"""Pentacam detection and cross-modality registration module.

This module is ADDITIVE and ISOLATED from the existing pupil/limbus
detection system. It provides types, detectors, and cross-modality registration
engines between seated Pentacam IR imagery and intra-operative ELITA imagery.

No existing production code is modified by this module.
"""

from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamEyeROI,
    PentacamFeature,
    PentacamFeatureSet,
    PentacamGeometry,
    PentacamImageType,
    PentacamQuality,
    PentacamReferenceResult,
)
from pupil_tracking.pentacam.cross_system import (
    CrossSystemRegistrationInput,
    CrossSystemRegistrationResult,
    RegistrationFailureKind,
    TransformationModel,
)
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.reference import PentacamReferenceExtractor
from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine

__all__ = [
    "PentacamDetectionResult",
    "PentacamDetectionStatus",
    "PentacamEyeROI",
    "PentacamFeature",
    "PentacamFeatureSet",
    "PentacamGeometry",
    "PentacamImageType",
    "PentacamQuality",
    "PentacamReferenceResult",
    "CrossSystemRegistrationInput",
    "CrossSystemRegistrationResult",
    "RegistrationFailureKind",
    "TransformationModel",
    "PentacamIrisDetector",
    "PentacamReferenceExtractor",
    "CrossModalityRegistrationEngine",
]
