"""
Registration pipeline for cyclotorsion detection and iris registration.

Provides multi-stream torsion estimation from paired (reference + current)
eye images using phase correlation, deep matching, ink markers, limbal
vessels, and custom-trained iris feature models, plus Phase 3 automated
toric axis correction.
"""

from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.axis_correction import (
    ToricAxisCorrectionEngine,
    AxisCorrectionResult,
    DiagnosticToricData,
    AxisSafetyGrade,
)
from pupil_tracking.registration.streams.ink_tracker import detect_limbal_purple_markers

__all__ = [
    "RegistrationEngine",
    "PolarUnwrapper",
    "IrisEnhancer",
    "ToricAxisCorrectionEngine",
    "AxisCorrectionResult",
    "DiagnosticToricData",
    "AxisSafetyGrade",
    "detect_limbal_purple_markers",
]
