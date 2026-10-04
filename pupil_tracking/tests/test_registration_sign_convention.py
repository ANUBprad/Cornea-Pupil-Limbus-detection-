"""
Cyclotorsion sign-convention regression test.

A known rotation applied with ``make_synthetic_pair`` must be reported with
the same sign by ``RegistrationEngine.register()``.

Sign basis (verified empirically, do not re-derive):
  * ``cv2.getRotationMatrix2D`` with a positive angle rotates image content
    COUNTER-CLOCKWISE on screen (image y axis points down).
  * ``arctan2(dy, dx)`` / polar angle therefore DECREASES by that angle.
  * So the true displacement is ``angles_ref - angles_curr``, not
    ``angles_curr - angles_ref``.

Run:
    python -m pytest pupil_tracking/tests/test_registration_sign_convention.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pupil_tracking.core.detector import UnifiedDetector  # noqa: E402
from pupil_tracking.iris.paired import PairConfig, make_synthetic_pair  # noqa: E402
from pupil_tracking.registration.engine import RegistrationEngine  # noqa: E402
from pupil_tracking.registration.streams.phase_correlation import (  # noqa: E402
    PhaseCorrelationStream,
)

# Clinical images that yield a usable pupil *and* limbus ellipse under the
# classical (ML-free) fallback path.
_IMAGES = ["eye_02", "eye_06", "eye_13"]
_CLEAN_DIR = _PROJECT_ROOT / "clinical_data" / "clean"
_ANGLES = (3.0, -3.0)


def _ellipse(detection, which):
    part = getattr(detection, which, None)
    return None if part is None else getattr(part, "ellipse", None)


@pytest.fixture(scope="module")
def detector():
    return UnifiedDetector()


@pytest.fixture(scope="module")
def engine():
    return RegistrationEngine()


@pytest.mark.parametrize("image_name", _IMAGES)
@pytest.mark.parametrize("angle_deg", _ANGLES)
def test_known_rotation_reported_with_same_sign(
    image_name, angle_deg, detector, engine
):
    """Known +rotation -> positive torsion_deg; known -rotation -> negative."""
    img = cv2.imread(str(_CLEAN_DIR / f"{image_name}.jpeg"))
    assert img is not None, f"missing clinical image {image_name}"

    ref = detector.detect(img)
    limbus = _ellipse(ref, "limbus")
    assert limbus is not None, f"{image_name}: no limbus ellipse"

    pair = make_synthetic_pair(
        img,
        PairConfig(rotation_deg=angle_deg, center=(limbus.center_x, limbus.center_y)),
    )
    cur = detector.detect(pair.image_b)

    result = engine.register(img, pair.image_b, ref, cur)

    assert result.torsion_deg is not None, f"{image_name} {angle_deg}: no torsion"
    assert result.valid, f"{image_name} {angle_deg}: registration rejected"

    if angle_deg > 0:
        assert result.torsion_deg > 0, (
            f"{image_name}: applied {angle_deg:+.1f} deg but reported "
            f"{result.torsion_deg:+.3f} deg (sign inverted)"
        )
    else:
        assert result.torsion_deg < 0, (
            f"{image_name}: applied {angle_deg:+.1f} deg but reported "
            f"{result.torsion_deg:+.3f} deg (sign inverted)"
        )

    # Magnitude, not just direction: a right-signed but wildly wrong angle is
    # still a failure. Measured worst case across these images is 0.76 deg.
    assert abs(result.torsion_deg - angle_deg) < 1.5, (
        f"{image_name}: applied {angle_deg:+.1f} deg but reported "
        f"{result.torsion_deg:+.3f} deg (magnitude off by "
        f"{abs(result.torsion_deg - angle_deg):.2f} deg)"
    )


def test_eye_01_rejects_unreliable_phase_correlation_peak(detector):
    """Phase correlation must abstain instead of emitting a bogus shift.

    Under the classical fallback detector the rotated eye_01 limbus collapses
    (207.5 -> 165.5 px), so the two polar images cover different physical iris
    radii. The correlation surface then has no dominant peak and argmax lands
    on an arbitrary angle (observed: +110 deg for a true -3 deg rotation).

    The stream must reject that peak rather than report it.
    """
    img = cv2.imread(str(_CLEAN_DIR / "eye_01.jpeg"))
    assert img is not None, "missing clinical image eye_01"

    ref = detector.detect(img)
    limbus = _ellipse(ref, "limbus")
    assert limbus is not None, "eye_01: no limbus ellipse"

    pair = make_synthetic_pair(
        img,
        PairConfig(rotation_deg=-3.0, center=(limbus.center_x, limbus.center_y)),
    )
    cur = detector.detect(pair.image_b)

    result = PhaseCorrelationStream().compute(img, pair.image_b, ref, cur)
    metadata = result.metadata or {}

    assert not result.valid, (
        "phase correlation accepted an unreliable peak and reported "
        f"{result.torsion_deg} deg for a true -3.0 deg rotation"
    )
    assert result.torsion_deg is None, (
        f"rejected result must not carry a shift, got {result.torsion_deg}"
    )
    assert metadata.get("error") == "unreliable_peak"
    # The diagnostics must survive so the rejection is explainable.
    assert "psr" in metadata and "peak_separation" in metadata


def test_peak_separation_detects_tied_peaks():
    """Separation is ~1 for a lone peak and ~0 when two peaks are tied."""
    lone = np.zeros(360)
    lone[100] = 1.0
    assert PhaseCorrelationStream._peak_separation(lone, 100) > 0.9

    tied = np.zeros(360)
    tied[100] = 1.0
    tied[220] = 1.0
    assert PhaseCorrelationStream._peak_separation(tied, 100) < 0.1
