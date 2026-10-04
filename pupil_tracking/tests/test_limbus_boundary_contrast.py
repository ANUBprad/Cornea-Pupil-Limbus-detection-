"""
Classical limbus boundary-contrast regression test.

``UnifiedDetector._classical_limbus`` ranks concentric candidates with
circularity, fit quality, centrality and coverage. All four saturate for
candidates that share a centre, so a weak internal edge (iris collarette,
pupil margin) could outscore the true limbus purely by fitting a tighter
circle. The ranking therefore also uses the strength of the grayscale
transition across the fitted boundary.

Measured on the clinical set: true limbi reach 0.21-0.52 of the frame's
grayscale spread, competing spurious circles 0.002-0.095.

These tests use synthetic frames only, so no clinical fixture is committed.

Run:
    python -m pytest pupil_tracking/tests/test_limbus_boundary_contrast.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pupil_tracking.core.detector import (  # noqa: E402
    LIMBUS_CONTRAST_SCALE,
    LIMBUS_CONTRAST_WEIGHT,
    UnifiedDetector,
)
from pupil_tracking.core.smart_fitter import FitResult, FitType  # noqa: E402

_SIZE = 256
_CENTER = (128, 128)


def _ramp(low: float = 20.0, high: float = 200.0) -> np.ndarray:
    """Frame whose grayscale spread comes from something unrelated to any circle."""
    row = np.linspace(low, high, _SIZE, dtype=np.float32)
    return np.tile(row, (_SIZE, 1))


def _frame(inside_delta: float, radius: float = 90.0,
           center: tuple[int, int] = _CENTER) -> np.ndarray:
    """Blurred ramp whose interior is offset by ``inside_delta`` gray levels."""
    img = _ramp().copy()
    mask = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    cv2.circle(mask, center, int(round(radius)), 1, -1)
    img += np.float32(inside_delta) * mask
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def _fit(radius: float, center: tuple[float, float] = (128.0, 128.0)) -> FitResult:
    return FitResult(
        fit_type=FitType.CIRCLE,
        valid=True,
        center_x=center[0],
        center_y=center[1],
        semi_major=radius,
        semi_minor=radius,
        radius=radius,
    )


def _contrast(img: np.ndarray, radius: float,
              center: tuple[float, float] = (128.0, 128.0)) -> float:
    return UnifiedDetector._boundary_contrast(img, _fit(radius, center))


def test_strong_boundary_saturates_the_feature():
    strong = _contrast(_frame(60.0), 90.0)
    assert strong == 1.0


def test_weak_boundary_scores_low():
    weak = _contrast(_frame(6.0), 90.0)
    assert weak < 0.25


def test_strong_boundary_beats_weak_by_a_wide_margin():
    """The discriminating condition the ranking fix depends on."""
    strong = _contrast(_frame(60.0), 90.0)
    weak = _contrast(_frame(6.0), 90.0)
    assert strong - weak > 0.5


def test_feature_is_invariant_to_brightness_scaling():
    """Normalising by the frame's own spread cancels exposure differences."""
    base = _frame(30.0)
    assert abs(_contrast(base, 90.0) - _contrast(base * 2.0, 90.0)) < 0.05


def test_feature_ignores_polarity():
    """Limbus polarity differs between docked and pre-docked frames."""
    brighter_inside = _frame(30.0)
    darker_inside = _frame(-30.0)
    assert abs(_contrast(brighter_inside, 90.0)
               - _contrast(darker_inside, 90.0)) < 0.05


def test_offset_sampling_ignores_a_local_blemish():
    """A single dark spot must not decide the score; the median absorbs it."""
    clean = _contrast(_frame(30.0), 90.0)
    blotched = _frame(30.0)
    cv2.circle(blotched, (128, 40), 9, 0.0, -1)
    blotched = cv2.GaussianBlur(blotched, (0, 0), 1.5)
    assert abs(_contrast(blotched, 90.0) - clean) < 0.1


def test_flat_frame_yields_no_evidence():
    flat = np.full((_SIZE, _SIZE), 128.0, dtype=np.float32)
    assert _contrast(flat, 90.0) == 0.0


def test_degenerate_radius_is_handled():
    assert _contrast(_frame(60.0), 0.0) == 0.0


def test_boundary_near_the_frame_edge_does_not_raise():
    """Sample offsets are clipped, so an off-centre limbus stays in bounds."""
    value = _contrast(_frame(60.0), 120.0, center=(14.0, 128.0))
    assert 0.0 <= value <= 1.0


def test_contrast_term_is_bounded_by_its_weight():
    """Ranking influence can never exceed the configured weight."""
    saturated = _contrast(_frame(60.0), 90.0)
    assert LIMBUS_CONTRAST_WEIGHT * saturated <= LIMBUS_CONTRAST_WEIGHT
    assert 0.0 < LIMBUS_CONTRAST_WEIGHT < 1.0
    assert LIMBUS_CONTRAST_SCALE > 0.0
