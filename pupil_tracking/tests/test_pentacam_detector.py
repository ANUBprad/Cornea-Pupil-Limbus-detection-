"""Unit tests for PentacamIrisDetector.

Verifies:
    - Pupil and limbus localization on IR / Scheimpflug images
    - Pentacam UI chrome and text overlay rejection (feathers/reticles excluded)
    - Iris landmark extraction inside the valid annulus
    - Quality grading and confidence calculation
    - Non-crashing behavior on empty or invalid inputs
    - Execution latency <= 150 ms
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamGeometry,
    PentacamQuality,
)
from pupil_tracking.utils.types import EllipseParams


def create_synthetic_pentacam_image(
    size: tuple[int, int] = (512, 512),
    pupil_r: float = 60.0,
    limbus_r: float = 160.0,
    with_ui: bool = True,
) -> np.ndarray:
    """Create a realistic synthetic Pentacam IR image with texture and UI overlays."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    cx, cy = w / 2.0, h / 2.0

    # Background sclera: medium intensity
    cv2_circle(img, (int(cx), int(cy)), int(limbus_r * 1.3), 160)

    # Iris stroma: darker with radial & concentric variation
    y, x = np.ogrid[:h, :w]
    dist = np.hypot(x - cx, y - cy)
    angle = np.arctan2(y - cy, x - cx)

    iris_mask = (dist >= pupil_r) & (dist <= limbus_r)

    # Broad multi-scale natural iris stroma (radial collarette + stromal fibers)
    fibers = (
        35.0 * np.sin(1.0 * angle + 0.3) +
        25.0 * np.cos(2.0 * angle - 0.7) +
        20.0 * np.sin(5.0 * angle + 1.2) +
        15.0 * np.cos(11.0 * angle) +
        10.0 * np.sin(dist / 5.0)
    )
    # Distinct asymmetric crypts and pigment spots (non-periodic)
    crypts = np.zeros((h, w), dtype=float)
    crypt_angles = [0.35, 0.95, 1.60, 2.25, 2.80, 3.50, 4.15, 4.80, 5.45, 6.00]
    for i, ca in enumerate(crypt_angles):
        cr = pupil_r + (limbus_r - pupil_r) * (0.35 + 0.3 * (i % 3) / 2.0)
        px = cx + cr * math.cos(ca)
        py = cy + cr * math.sin(ca)
        crypts += 45.0 * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2.0 * 5.0 ** 2))

    total_iris = np.clip(100.0 + fibers - crypts, 20, 220).astype(np.uint8)
    img[iris_mask] = total_iris[iris_mask]

    # Pupil aperture: very dark
    pupil_mask = dist < pupil_r
    img[pupil_mask] = 15

    # Corneal reflection glint
    glint_mask = np.hypot(x - (cx + 10), y - (cy - 10)) < 4
    img[glint_mask] = 240

    if with_ui:
        # Realistic Pentacam UI:
        # 1. Central corneal vertex fixation crosshair (restricted to inside pupil center)
        img[int(cy) - 8:int(cy) + 9, int(cx) - 1:int(cx) + 2] = 255
        img[int(cy) - 1:int(cy) + 2, int(cx) - 8:int(cx) + 9] = 255
        # 2. Corner banners & annotations (outside limbus, e.g. 'OD Pentacam HR', K-readings)
        img[15:40, 15:150] = 254
        img[h - 40:h - 15, 15:120] = 254
        # 3. Outer chrome margin
        img[:8, :] = 0
        img[-8:, :] = 0
        img[:, :8] = 0
        img[:, -8:] = 0

    return img


def cv2_circle(img: np.ndarray, center: tuple[int, int], radius: int, val: int) -> None:
    import cv2
    cv2.circle(img, center, radius, val, -1)


class TestPentacamIrisDetector:
    def test_init(self):
        det = PentacamIrisDetector()
        assert det.num_angles == 72
        assert det.min_features == 6

    def test_detect_synthetic_image(self):
        img = create_synthetic_pentacam_image(with_ui=True)
        det = PentacamIrisDetector()

        res = det.detect(img)
        assert res.valid is True
        assert res.status == PentacamDetectionStatus.OK
        assert res.geometry.pupil_detected is True
        assert res.geometry.limbus_detected is True
        assert len(res.feature_set.features) >= 10
        assert res.feature_set.angular_coverage_ratio > 0.50
        assert res.confidence > 0.30
        assert res.processing_time_ms < 150.0

    def test_ui_overlay_rejection(self):
        """Verify that UI crosshair lines and annotations are NOT accepted as iris features."""
        img = create_synthetic_pentacam_image(with_ui=True)
        det = PentacamIrisDetector()

        res = det.detect(img)
        assert res.valid is True

        # Check that no feature lies on the crosshair line (x == 256 or y == 256)
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        for f in res.feature_set.features:
            # No features on the central fixation target
            assert not (abs(f.x - cx) <= 8.0 and abs(f.y - cy) <= 8.0)
            # No features inside the corner text banners
            assert not (15 <= f.y <= 40 and 15 <= f.x <= 150)
            assert not (h - 40 <= f.y <= h - 15 and 15 <= f.x <= 120)
            # No features in the letterbox black margins
            assert f.x >= 8 and f.x < w - 8 and f.y >= 8 and f.y < h - 8

    def test_in_annulus_confinement(self):
        """100% of accepted features must lie strictly within the iris annulus."""
        img = create_synthetic_pentacam_image(with_ui=False)
        det = PentacamIrisDetector()

        res = det.detect(img)
        assert res.valid is True

        geom = res.geometry
        pr = geom.pupil_radius_px
        lr = geom.limbus_radius_px
        pcx, pcy = geom.pupil.center_x, geom.pupil.center_y

        for f in res.feature_set.features:
            d = math.hypot(f.x - pcx, f.y - pcy)
            assert d >= pr * 0.95
            assert d <= lr * 1.05

    def test_empty_image_fails_gracefully(self):
        det = PentacamIrisDetector()
        empty = np.zeros((0, 0), dtype=np.uint8)
        res = det.detect(empty)
        assert res.valid is False
        assert res.status == PentacamDetectionStatus.NO_IMAGE


def _flat_iris_scene(
    size: int = 512,
    pupil_r: float = 60.0,
    iris_r: float = 300.0,
    iris_val: int = 105,
    sclera_val: int = 200,
) -> np.ndarray:
    """Noise-free concentric scene: dark pupil, flat iris, bright sclera.

    Growing ``iris_r`` past the detector's limbus search window removes every
    iris-to-sclera transition from that window, leaving a flat radial profile.
    That is the unbracketed-search case the fallback must report honestly.
    """
    img = np.full((size, size), sclera_val, dtype=np.uint8)
    c = size // 2
    cv2_circle(img, (c, c), int(iris_r), iris_val)
    cv2_circle(img, (c, c), int(pupil_r), 15)
    return img


class TestPentacamLimbusFallback:
    """A limbus radius pinned to a search-window bound is not a localization."""

    def test_unbracketed_search_is_not_valid(self):
        det = PentacamIrisDetector()
        res = det.detect(_flat_iris_scene(pupil_r=60.0, iris_r=300.0))

        assert res.geometry.limbus_localized is False
        assert res.valid is False
        assert res.status == PentacamDetectionStatus.NO_LIMBUS
        assert res.quality != PentacamQuality.GOOD
        assert res.confidence == 0.0
        assert res.feature_set.features == []
        assert "window bound" in res.failure_reason

    def test_unbracketed_search_radius_is_pinned_to_lower_bound(self):
        det = PentacamIrisDetector()
        geom = det.detect(_flat_iris_scene(pupil_r=60.0, iris_r=300.0)).geometry

        assert geom.pupil_detected is True
        assert geom.limbus_detected is True
        assert geom.limbus_radius_px == pytest.approx(geom.pupil_radius_px * 1.7)

    def test_degenerate_search_window_is_not_valid(self):
        """min_limbus_r >= max_limbus_r means no search runs at all."""
        det = PentacamIrisDetector()
        res = det.detect(_flat_iris_scene(size=300, pupil_r=95.0, iris_r=140.0))

        assert res.geometry.pupil_detected is True
        assert res.geometry.limbus_localized is False
        assert res.valid is False
        assert res.status == PentacamDetectionStatus.NO_LIMBUS
        assert res.quality != PentacamQuality.GOOD
        assert res.confidence == 0.0

    def test_genuine_transition_still_accepted(self):
        det = PentacamIrisDetector()
        res = det.detect(create_synthetic_pentacam_image(with_ui=True))

        assert res.geometry.limbus_localized is True
        assert res.valid is True
        assert res.status == PentacamDetectionStatus.OK
        assert res.geometry.limbus_radius_px == pytest.approx(160.0, abs=15.0)

    def test_external_geometry_defaults_to_localized(self):
        det = PentacamIrisDetector()
        img = create_synthetic_pentacam_image(with_ui=True)
        geom = PentacamGeometry(
            pupil=EllipseParams(center_x=256.0, center_y=256.0,
                                semi_major=60.0, semi_minor=60.0),
            limbus=EllipseParams(center_x=256.0, center_y=256.0,
                                 semi_major=160.0, semi_minor=160.0),
            pupil_detected=True,
            limbus_detected=True,
            pupil_radius_px=60.0,
            limbus_radius_px=160.0,
        )

        assert geom.limbus_localized is True
        assert det.detect(img, geometry=geom).valid is True

    def test_unlocalized_external_geometry_rejected(self):
        det = PentacamIrisDetector()
        img = create_synthetic_pentacam_image(with_ui=True)
        geom = PentacamGeometry(
            pupil=EllipseParams(center_x=256.0, center_y=256.0,
                                semi_major=60.0, semi_minor=60.0),
            limbus=EllipseParams(center_x=256.0, center_y=256.0,
                                 semi_major=120.0, semi_minor=120.0),
            pupil_detected=True,
            limbus_detected=True,
            limbus_localized=False,
            pupil_radius_px=60.0,
            limbus_radius_px=120.0,
        )

        res = det.detect(img, geometry=geom)
        assert res.valid is False
        assert res.status == PentacamDetectionStatus.NO_LIMBUS

    def test_to_dict_exposes_limbus_localized(self):
        det = PentacamIrisDetector()
        d = det.detect(_flat_iris_scene(pupil_r=60.0, iris_r=300.0)).to_dict()

        assert d["geometry"]["limbus_localized"] is False
        assert d["status"] == PentacamDetectionStatus.NO_LIMBUS.value
        assert d["valid"] is False


class TestPentacamLimbusSearchFloor:
    """The lower search bound must clear the pupil edge without excluding the limbus.

    The floor used to be 2.0x pupil radius. A real Pentacam capture measured its
    iris->sclera transition at 1.95x, so the window opened past the transition,
    argmax stayed on index 0, and b428614 correctly reported NO_LIMBUS.
    """

    def test_sub_two_ratio_limbus_is_localized(self):
        img = create_synthetic_pentacam_image(pupil_r=60.0, limbus_r=117.0)
        det = PentacamIrisDetector()
        res = det.detect(img)
        geom = res.geometry

        assert geom.limbus_localized is True
        assert geom.limbus_radius_px == pytest.approx(117.0, abs=5.0)

        # Interior maximum, not pinned to either window bound.
        h, w = img.shape
        pr = geom.pupil_radius_px
        floor = pr * 1.7
        ceiling = min(pr * 3.8,
                      min(geom.pupil.center_x, w - geom.pupil.center_x,
                          geom.pupil.center_y, h - geom.pupil.center_y) * 0.95)
        assert floor < geom.limbus_radius_px < ceiling

        assert res.valid is True
        assert res.status == PentacamDetectionStatus.OK

    @pytest.mark.parametrize("limbus_r", [117.0, 130.0, 150.0, 160.0])
    def test_known_limbus_ratios_stay_accurate(self, limbus_r):
        img = create_synthetic_pentacam_image(pupil_r=60.0, limbus_r=limbus_r)
        det = PentacamIrisDetector()
        geom = det.detect(img).geometry

        assert geom.limbus_localized is True
        assert geom.limbus_radius_px == pytest.approx(limbus_r, abs=15.0)

    def test_search_floor_excludes_the_pupil_edge(self):
        """The pupil edge at ~1.0x pr, and the 1.5x boundary, stay excluded."""
        det = PentacamIrisDetector()
        geom = det.detect(_flat_iris_scene(pupil_r=60.0, iris_r=300.0)).geometry
        pr = geom.pupil_radius_px

        assert geom.limbus_localized is False
        assert geom.limbus_radius_px == pytest.approx(pr * 1.7)
        assert geom.limbus_radius_px >= pr * 1.5

    def test_limbus_genuinely_below_the_floor_is_refused(self):
        """A limbus under the floor must be refused, never approximated.

        With the old 2.0 floor this scene reported a confident, interior
        maximum of 133px against a 95px ground truth.
        """
        img = create_synthetic_pentacam_image(pupil_r=60.0, limbus_r=95.0)
        det = PentacamIrisDetector()
        res = det.detect(img)

        assert res.geometry.limbus_localized is False
        assert res.valid is False
        assert res.status == PentacamDetectionStatus.NO_LIMBUS
        assert res.confidence == 0.0

    def test_degenerate_search_window_returns_the_floor(self):
        """No search runs, so the reported radius is the floor, not a measurement."""
        det = PentacamIrisDetector()
        geom = det.detect(_flat_iris_scene(size=300, pupil_r=95.0, iris_r=140.0)).geometry

        assert geom.limbus_localized is False
        assert geom.limbus_radius_px == pytest.approx(geom.pupil_radius_px * 1.7)
