"""Unit tests for cyclotorsion detection streams."""

import numpy as np
import pytest
from pupil_tracking.registration.streams.phase_correlation import PhaseCorrelationStream
from pupil_tracking.registration.streams.ink_tracker import InkTrackerStream
from pupil_tracking.registration.streams.vessel_tracker import VesselTrackerStream
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    EllipseParams,
    DetectionQuality,
    StreamName,
)


def _make_dummy_detection(center=(100.0, 100.0), pupil_r=25.0, limbus_r=60.0):
    det = EyeDetectionResult()
    det.pupil.detected = True
    det.pupil.ellipse = EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=pupil_r,
        semi_minor=pupil_r,
        angle_deg=0.0,
    )
    det.pupil.confidence = 0.9
    det.pupil.quality = DetectionQuality.SURGICAL

    det.limbus.detected = True
    det.limbus.ellipse = EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=limbus_r,
        semi_minor=limbus_r,
        angle_deg=0.0,
    )
    det.limbus.confidence = 0.9
    det.limbus.quality = DetectionQuality.SURGICAL
    return det


def test_phase_correlation_stream_zero_rotation():
    stream = PhaseCorrelationStream()
    assert stream.name == StreamName.PHASE_CORRELATION

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    y, x = np.ogrid[:200, :200]
    dist = np.sqrt((x - 100)**2 + (y - 100)**2)
    angle = np.arctan2(y - 100, x - 100)
    iris = (dist >= 25) & (dist <= 60)
    texture = (128 + 100 * np.sin(6 * angle)).astype(np.uint8)
    for c in range(3):
        img[iris, c] = texture[iris]

    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = stream.run(img, img, det1, det2)
    assert res.stream == StreamName.PHASE_CORRELATION
    assert res.valid is True
    assert abs(res.torsion_deg) < 1.0  # Zero rotation check


def test_ink_tracker_stream_empty():
    stream = InkTrackerStream()
    assert stream.name == StreamName.INK_MARKERS

    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = stream.run(img1, img2, det1, det2)
    assert res.stream == StreamName.INK_MARKERS
    # No ink marks found in black images
    assert res.inlier_count == 0


def test_ink_tracker_match_markers_by_angle():
    """_match_markers pairs each ref marker with its nearest curr marker.

    Regression guard: this method was stranded inside a module-level function
    after a dedent, so it did not exist on the class and every non-empty ink
    detection raised AttributeError. The empty-image test above returns before
    reaching it, so this direct call is what keeps the method reachable.
    """
    stream = InkTrackerStream()
    angles_ref = np.array([0.0, np.pi / 2])
    angles_curr = np.array([np.radians(5.0), np.pi / 2 + np.radians(5.0)])

    matches = stream._match_markers(angles_ref, angles_curr)
    assert matches == [(0, 0), (1, 1)]

    # A curr marker with no ref partner inside the 30-degree window is dropped.
    unmatched = stream._match_markers(np.array([0.0]), np.array([np.pi]))
    assert unmatched == []


def test_vessel_tracker_stream():
    stream = VesselTrackerStream()
    assert stream.name == StreamName.LIMBAL_VESSELS

    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = stream.run(img1, img2, det1, det2)
    assert res.stream == StreamName.LIMBAL_VESSELS
