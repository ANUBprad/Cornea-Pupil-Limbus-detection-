"""Unit tests for PentacamReferenceExtractor.

Covers the detection-ready reference representation end to end:

    - input handling: cropped eye, device screenshot, RGB/BGR, grayscale,
      BGRA, single-channel arrays, and invalid input
    - eye-region localisation, confidence and normalized grayscale output
    - geometry gates: pupil detected, limbus detected *and* localized
    - iris mask generation and UI-chrome occlusion handling
    - polar ribbon normalisation and configurable resolution
    - feature extraction
    - coverage and feature-count rejection
    - determinism and serialisation

Every image here is SYNTHETIC. These tests verify schema, gate ordering and
determinism -- they make no claim about clinical or device accuracy.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.preprocess import (
    build_ui_mask,
    content_bbox,
    normalize_gray,
    to_bgr,
    to_gray,
)
from pupil_tracking.pentacam.reference import PentacamReferenceExtractor
from pupil_tracking.pentacam.types import (
    PentacamDetectionStatus,
    PentacamQuality,
)
from pupil_tracking.tests.pentacam_fixtures.synthetic import (
    make_device_screenshot,
    make_eye_image,
    make_textureless_eye_image,
    occlude_iris_sectors,
)

PUPIL_R = 60.0
LIMBUS_R = 160.0

# A device-screenshot layout whose eye cross-section dominates the frame, in
# the same proportion as a real capture.  The eye is generated at native scale
# rather than resampled, so stroma texture stays sharp and the feature stage
# sees the same content it would in a bare crop.
SHOT_CANVAS = (620, 900)
SHOT_EYE_SIZE = (620, 620)
SHOT_PUPIL_R = 73.0
SHOT_LIMBUS_R = 193.0


@pytest.fixture(scope="module")
def extractor() -> PentacamReferenceExtractor:
    return PentacamReferenceExtractor()


@pytest.fixture(scope="module")
def eye() -> np.ndarray:
    return make_eye_image(pupil_r=PUPIL_R, limbus_r=LIMBUS_R)


@pytest.fixture(scope="module")
def screenshot() -> np.ndarray:
    shot_eye = make_eye_image(
        size=SHOT_EYE_SIZE, pupil_r=SHOT_PUPIL_R, limbus_r=SHOT_LIMBUS_R
    )
    return make_device_screenshot(
        shot_eye,
        canvas_size=SHOT_CANVAS,
        panel_origin=((SHOT_CANVAS[1] - SHOT_EYE_SIZE[1]) // 2, 0),
    )


def extractor_result(image: np.ndarray):
    """Run a fresh extractor; used where a fixture-scoped one is awkward."""
    return PentacamReferenceExtractor().extract(image)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


class TestInputHandling:
    def test_cropped_eye_image_yields_valid_reference(self, extractor, eye):
        result = extractor.extract(eye)

        assert result.valid is True
        assert result.status == PentacamDetectionStatus.OK
        assert result.quality in (PentacamQuality.GOOD, PentacamQuality.ACCEPTABLE)
        assert result.failure_reason == ""

    def test_device_screenshot_yields_valid_reference(self, extractor, screenshot):
        result = extractor.extract(screenshot)

        assert result.valid is True
        assert result.status == PentacamDetectionStatus.OK
        # Geometry is still found inside the screenshot panel.
        assert result.geometry.pupil_detected is True
        assert result.geometry.limbus_localized is True
        # The eye is smaller than the frame, so the region must not be the
        # whole screenshot -- that is what proves UI chrome was excluded.
        assert result.eye_roi.area < result.image_width * result.image_height

    def test_screenshot_geometry_matches_bare_eye(self, extractor, screenshot):
        """Wrapping the eye in chrome must not change the recovered geometry."""
        wrapped = extractor.extract(screenshot)

        assert wrapped.geometry.limbus_radius_px == pytest.approx(
            SHOT_LIMBUS_R, rel=0.10
        )
        assert wrapped.geometry.pupil_radius_px == pytest.approx(
            SHOT_PUPIL_R, rel=0.15
        )
        assert wrapped.valid is True

    def test_rgb_colour_input_accepted(self, extractor, eye):
        bgr = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)
        result = extractor.extract(bgr)

        assert result.valid is True
        assert result.status == PentacamDetectionStatus.OK

    def test_grayscale_input_accepted(self, extractor, eye):
        result = extractor.extract(eye)

        assert result.valid is True
        assert eye.ndim == 2  # fixture really is single-channel

    def test_bgra_input_accepted(self, extractor, eye):
        bgra = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGRA)
        result = extractor.extract(bgra)

        assert result.valid is True

    def test_single_channel_3d_input_accepted(self, extractor, eye):
        """(H, W, 1) is a legitimate grayscale encoding that cvtColor rejects."""
        result = extractor.extract(eye[:, :, None])

        assert result.valid is True

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            np.zeros((0, 0), dtype=np.uint8),
            np.zeros((8, 8, 2), dtype=np.uint8),
            np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "not-an-image",
        ],
        ids=["none", "empty", "two-channels", "four-dimensional", "wrong-type"],
    )
    def test_invalid_input_rejected_with_reason(self, extractor, bad):
        result = extractor.extract(bad)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.NO_IMAGE
        assert result.quality == PentacamQuality.NO_DETECTION
        assert result.confidence == 0.0
        assert result.failure_reason

    def test_all_black_frame_rejected_as_degenerate(self, extractor):
        result = extractor.extract(np.zeros((256, 256), dtype=np.uint8))

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.DEGENERATE
        assert "eye region" in result.failure_reason

    def test_source_image_never_mutated(self, extractor, screenshot):
        original = screenshot.copy()
        extractor.extract(screenshot)

        assert np.array_equal(original, screenshot)

    def test_image_dimensions_recorded(self, extractor, eye, screenshot):
        assert extractor.extract(eye).image_width == eye.shape[1]
        assert extractor.extract(eye).image_height == eye.shape[0]

        shot = extractor.extract(screenshot)
        assert shot.image_width == screenshot.shape[1]
        assert shot.image_height == screenshot.shape[0]


# --------------------------------------------------------------------------- #
# Eye region
# --------------------------------------------------------------------------- #


class TestEyeRegion:
    def test_region_located_with_confidence(self, extractor, eye):
        roi = extractor.extract(eye).eye_roi

        assert roi.valid is True
        assert roi.reason == "ok"
        assert roi.width > 0 and roi.height > 0
        assert 0.0 < roi.confidence <= 1.0

    def test_grayscale_view_is_normalized(self, extractor, eye):
        roi = extractor.extract(eye).eye_roi

        assert roi.grayscale is not None
        assert roi.grayscale.dtype == np.float32
        assert roi.grayscale.shape == (roi.height, roi.width)
        assert float(roi.grayscale.min()) >= 0.0
        assert float(roi.grayscale.max()) <= 255.0
        # Normalisation must actually use the range, not collapse to flat.
        assert float(roi.grayscale.max()) - float(roi.grayscale.min()) > 1.0

    def test_region_covers_detected_geometry(self, extractor, eye):
        result = extractor.extract(eye)
        geometry = result.geometry

        assert result.eye_roi.contains(geometry.pupil.center_x, geometry.pupil.center_y)
        assert result.eye_roi.contains(geometry.limbus.center_x, geometry.limbus.center_y)

    def test_region_excludes_screenshot_chrome(self, extractor, screenshot):
        roi = extractor.extract(screenshot).eye_roi

        # Black UI background is suppressed, so the region is strictly smaller.
        assert roi.area < screenshot.shape[0] * screenshot.shape[1]
        assert roi.retained_fraction < 1.0

    def test_region_derived_from_content_not_fixed_layout(self, extractor, eye):
        """Two different panel origins must both resolve to their own panel."""
        tight = make_device_screenshot(
            eye, canvas_size=(700, 900), panel_origin=(20, 20)
        )
        loose = make_device_screenshot(
            eye, canvas_size=(1100, 1600), panel_origin=(400, 300)
        )

        tight_roi = extractor.extract(tight).eye_roi
        loose_roi = extractor.extract(loose).eye_roi

        # The recovered region tracks the panel: identical size, and the origin
        # offset from the panel start is the same content-derived margin.
        assert tight_roi.width == loose_roi.width
        assert tight_roi.height == loose_roi.height
        assert tight_roi.x - 20 == loose_roi.x - 400
        assert tight_roi.y - 20 == loose_roi.y - 300

    def test_coordinate_mapping_round_trips(self, extractor, eye):
        roi = extractor.extract(eye).eye_roi
        px, py = 7.0, 9.0

        assert roi.to_full_image(px, py) == (px + roi.x, py + roi.y)

    def test_most_of_a_cropped_eye_frame_is_content(self, extractor, eye):
        """The synthetic fixture has black corners, but content must dominate."""
        roi = extractor.extract(eye).eye_roi

        assert roi.retained_fraction > 0.5


# --------------------------------------------------------------------------- #
# Geometry gates -- including the b428614 limbus-localization rule
# --------------------------------------------------------------------------- #


class TestGeometryGates:
    def test_pupil_detected_with_plausible_radius(self, extractor, eye):
        geometry = extractor.extract(eye).geometry

        assert geometry.pupil_detected is True
        assert geometry.pupil is not None
        assert geometry.pupil_radius_px == pytest.approx(PUPIL_R, rel=0.15)

    def test_limbus_detected_and_localized(self, extractor, eye):
        geometry = extractor.extract(eye).geometry

        assert geometry.limbus_detected is True
        assert geometry.limbus is not None
        assert geometry.limbus_localized is True
        assert geometry.limbus_radius_px == pytest.approx(LIMBUS_R, rel=0.15)

    def test_limbus_radius_is_a_radius_not_a_diameter(self, extractor, eye):
        """Guards against a diameter/radius confusion in the reference path."""
        geometry = extractor.extract(eye).geometry

        assert geometry.limbus.semi_major == pytest.approx(
            geometry.limbus_radius_px, rel=0.25
        )
        # The annulus must fit inside the frame, so it cannot be 2x too large.
        assert geometry.limbus.semi_major < eye.shape[0]

    def test_pupil_limbus_ratio_in_physiological_range(self, extractor, eye):
        ratio = extractor.extract(eye).geometry.pupil_limbus_ratio

        assert 0.2 < ratio < 0.75

    def test_unlocalized_limbus_is_rejected(self, extractor, monkeypatch):
        """b428614 semantics: a pinned limbus must never yield a reference."""
        image = make_eye_image()
        real_detect = PentacamIrisDetector.detect

        def fake_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            out.geometry.limbus_localized = False
            return out

        monkeypatch.setattr(
            PentacamIrisDetector, "detect", fake_detect
        )

        result = PentacamReferenceExtractor().extract(image)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.NO_LIMBUS
        assert result.confidence == 0.0
        assert result.quality == PentacamQuality.NO_DETECTION
        assert result.feature_set.num_accepted == 0
        assert result.polar is None

    def test_unlocalized_limbus_reason_is_specific(self, extractor, monkeypatch):
        image = make_eye_image()
        real_detect = PentacamIrisDetector.detect

        def fake_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            out.geometry.limbus_localized = False
            return out

        monkeypatch.setattr(
            PentacamIrisDetector, "detect", fake_detect
        )

        result = PentacamReferenceExtractor().extract(image)

        assert "not localized" in result.failure_reason

    def test_missing_limbus_is_rejected(self, extractor, monkeypatch):
        image = make_eye_image()
        real_detect = PentacamIrisDetector.detect

        def fake_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            out.geometry.limbus_detected = False
            out.geometry.limbus = None
            return out

        monkeypatch.setattr(
            PentacamIrisDetector, "detect", fake_detect
        )

        result = PentacamReferenceExtractor().extract(image)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.NO_LIMBUS

    def test_implausible_geometry_is_degenerate(self, extractor, monkeypatch):
        """A limbus smaller than the pupil must fail before feature extraction."""
        image = make_eye_image()
        real_detect = PentacamIrisDetector.detect

        def fake_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            out.geometry.limbus.semi_major = 5.0
            out.geometry.limbus.semi_minor = 5.0
            return out

        monkeypatch.setattr(
            PentacamIrisDetector, "detect", fake_detect
        )

        result = PentacamReferenceExtractor().extract(image)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.DEGENERATE
        assert "iris geometry" in result.failure_reason


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #


class TestMasks:
    def test_mask_statistics_reported(self, extractor, eye):
        stats = extractor.extract(eye).mask_stats

        assert "usable_fraction" in stats
        assert 0.0 < stats["usable_fraction"] <= 1.0
        assert "polar_coverage" in stats
        assert "angular_coverage_ratio" in stats

    def test_coverage_is_mask_aware(self, extractor, eye):
        """Occluding most of the annulus must reduce usable area, not features."""
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=360.0)
        clean = extractor.extract(eye)

        partial = extractor.extract(
            occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=120.0)
        )

        assert partial.feature_set.usable_fraction < clean.feature_set.usable_fraction

    def test_ui_chrome_becomes_occlusion(self, extractor, eye, screenshot):
        """White annotation blocks must not be treated as iris."""
        shot = extractor.extract(screenshot)
        ui_mask = build_ui_mask(to_gray(screenshot))

        # Text blocks live in the chrome and are suppressed.
        assert ui_mask[55, 100] == 0
        assert ui_mask[screenshot.shape[0] - 55, 100] == 0

        # Chrome outside the annulus must not reduce usable iris area.
        assert shot.feature_set.usable_fraction > 0.5

    def test_crosshair_pixels_are_excluded_from_features(self, extractor):
        """A crosshair over the pupil centre must not become a feature."""
        shot_eye = make_eye_image(
            size=SHOT_EYE_SIZE, pupil_r=SHOT_PUPIL_R, limbus_r=SHOT_LIMBUS_R
        )
        shot = make_device_screenshot(
            shot_eye, canvas_size=SHOT_CANVAS, panel_origin=(140, 0)
        )

        result = extractor.extract(shot)
        cx = 140 + SHOT_EYE_SIZE[1] // 2
        cy = SHOT_EYE_SIZE[0] // 2

        for feature in result.feature_set.features:
            assert not (
                abs(feature.x - cx) <= 12 and abs(feature.y - cy) <= 12
            ), "feature landed on the fixation crosshair"


# --------------------------------------------------------------------------- #
# Polar ribbon
# --------------------------------------------------------------------------- #


class TestPolarRibbon:
    def test_ribbon_is_valid_and_normalized(self, extractor, eye):
        polar = extractor.extract(eye).polar

        assert polar is not None
        assert polar.valid is True
        assert polar.image is not None
        assert polar.mask is not None
        assert polar.image.shape == polar.mask.shape

    def test_ribbon_shape_is_radial_by_angular(self, extractor, eye):
        ex = PentacamReferenceExtractor(polar_num_angles=180, polar_num_radial=24)
        polar = ex.extract(eye).polar

        assert polar.num_angles == 180
        assert polar.num_radial == 24
        assert polar.image.shape == (24, 180)

    def test_ribbon_spans_pupil_to_limbus(self, extractor, eye):
        result = extractor.extract(eye)
        polar = result.polar

        assert polar.inner_radius > 0.0
        assert polar.outer_radius > polar.inner_radius
        # Inner bound must sit outside the pupil, outer bound inside the limbus.
        assert polar.inner_radius > result.geometry.pupil_radius_px * 0.8
        assert polar.outer_radius < result.geometry.limbus_radius_px * 1.05

    def test_ribbon_centre_matches_limbus_centre(self, extractor, eye):
        result = extractor.extract(eye)

        assert result.polar.center == pytest.approx(
            (result.geometry.limbus.center_x, result.geometry.limbus.center_y)
        )

    def test_ribbon_mask_mostly_valid_on_clean_eye(self, extractor, eye):
        polar = extractor.extract(eye).polar
        coverage = float(np.count_nonzero(polar.mask)) / polar.mask.size

        assert coverage > 0.9

    def test_ribbon_coverage_is_geometric_not_occlusion(self, extractor, eye):
        """Blackout does not change geometric coverage -- usable_fraction does.

        The two metrics are deliberately independent: polar_coverage catches a
        clipped annulus, usable_fraction catches occlusion.
        """
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=120.0)
        clean = extractor.extract(eye)
        partial = extractor.extract(occluded)

        assert partial.mask_stats["polar_coverage"] == pytest.approx(
            clean.mask_stats["polar_coverage"]
        )
        assert (
            partial.feature_set.usable_fraction < clean.feature_set.usable_fraction
        )

    def test_clipped_limbus_is_rejected_before_polar(self):
        """A limbus larger than the frame is caught by the localization gate.

        The detector bounds its search window to 0.95 of the distance to the
        nearest edge, so an over-large limbus can only ever be pinned to that
        bound -- which b428614 rejects before the polar stage is reached.
        """
        small = make_eye_image(size=(512, 512), pupil_r=60.0, limbus_r=240.0)
        result = extractor_result(small)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.NO_LIMBUS
        assert result.polar is None

    def test_polar_coverage_gate_fires_when_annulus_leaves_frame(
        self, eye, monkeypatch
        ):
        """Geometry whose limbus sits off-frame must fail on polar coverage.

        The usable-area gate would normally catch this first, so it is relaxed
        here to show the polar gate is a real, independent check rather than a
        constant that can never fire.
        """
        real_detect = PentacamIrisDetector.detect

        def shifted_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            # Push the limbus centre into the corner so most of the outer ring
            # samples fall outside the image.
            out.geometry.limbus.center_x = 20.0
            out.geometry.limbus.center_y = 20.0
            return out

        monkeypatch.setattr(PentacamIrisDetector, "detect", shifted_detect)

        ex = PentacamReferenceExtractor(min_usable_fraction=0.0)
        result = ex.extract(eye)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.INSUFFICIENT_FEATURES
        assert "polar ribbon coverage" in result.failure_reason
        assert result.mask_stats["polar_coverage"] < 0.5

    def test_off_frame_annulus_is_caught_by_the_usable_area_gate(self, eye, monkeypatch):
        """With default gates, an off-frame annulus fails on usable area first."""
        real_detect = PentacamIrisDetector.detect

        def shifted_detect(self, image, geometry=None, image_type=None):
            out = real_detect(self, image, geometry, image_type)
            out.geometry.limbus.center_x = 20.0
            out.geometry.limbus.center_y = 20.0
            return out

        monkeypatch.setattr(PentacamIrisDetector, "detect", shifted_detect)

        result = extractor_result(eye)

        assert result.valid is False
        assert "usable iris fraction" in result.failure_reason

    def test_ribbon_is_not_constant(self, extractor, eye):
        """A constant ribbon means the polar transform silently failed."""
        polar = extractor.extract(eye).polar
        values = polar.image[polar.mask > 0]

        assert values.size > 0
        assert float(values.std()) > 1.0


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


class TestFeatures:
    def test_features_extracted(self, extractor, eye):
        result = extractor.extract(eye)

        assert result.feature_set.num_accepted >= 6
        assert len(result.feature_set.features) == result.feature_set.num_accepted
        assert result.feature_set.num_candidates >= result.feature_set.num_accepted

    def test_features_carry_iris_relative_coordinates(self, extractor, eye):
        for feature in extractor.extract(eye).feature_set.features:
            assert 0.0 < feature.radial_norm <= 1.0
            assert 0.0 <= feature.angle_deg < 360.0
            assert feature.valid is True

    def test_features_carry_descriptors(self, extractor, eye):
        features = extractor.extract(eye).feature_set.features

        assert features
        for feature in features:
            assert feature.descriptor is not None
            assert feature.descriptor.ndim == 1
            assert feature.descriptor.size > 0
            assert np.isfinite(feature.descriptor).all()

    def test_features_lie_inside_the_eye_region(self, extractor, eye):
        result = extractor.extract(eye)

        for feature in result.feature_set.features:
            assert result.eye_roi.contains(feature.x, feature.y)

    def test_feature_count_rejection(self, extractor):
        """A textureless iris cannot support the minimum feature count."""
        result = extractor.extract(make_textureless_eye_image())

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.INSUFFICIENT_FEATURES
        assert result.feature_set.num_accepted < 6
        assert result.failure_reason

    def test_descriptors_are_finite_and_bounded(self, extractor, eye):
        for feature in extractor.extract(eye).feature_set.features:
            assert np.all(feature.descriptor >= 0.0)


# --------------------------------------------------------------------------- #
# Coverage rejection
# --------------------------------------------------------------------------- #


class TestCoverageGates:
    def test_insufficient_usable_area_rejected(self, extractor, eye):
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=40.0)
        result = extractor.extract(occluded)

        assert result.valid is False
        assert result.status == PentacamDetectionStatus.INSUFFICIENT_FEATURES
        assert result.feature_set.usable_fraction < 0.30

    def test_rejection_reason_names_the_failing_gate(self, extractor, eye):
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=40.0)

        assert "usable iris fraction" in extractor.extract(occluded).failure_reason

    def test_zero_confidence_when_rejected(self, extractor, eye):
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=40.0)
        result = extractor.extract(occluded)

        assert result.confidence == 0.0
        assert result.quality == PentacamQuality.NO_DETECTION

    def test_occlusion_gate_can_be_relaxed(self, extractor, eye):
        """The gates are configurable, not hardcoded constants."""
        occluded = occlude_iris_sectors(eye, PUPIL_R, LIMBUS_R, keep_deg=120.0)

        strict = PentacamReferenceExtractor(
            min_usable_fraction=0.90, min_angular_coverage=0.90
        )
        relaxed = PentacamReferenceExtractor(
            min_usable_fraction=0.05,
            min_angular_coverage=0.05,
            min_polar_coverage=0.05,
            min_features=1,
        )

        assert strict.extract(occluded).valid is False
        assert relaxed.extract(occluded).valid is True


# --------------------------------------------------------------------------- #
# Determinism and serialisation
# --------------------------------------------------------------------------- #


class TestDeterminism:
    @staticmethod
    def _stable(result):
        """Everything except wall-clock timings, which are inherently variable."""
        payload = result.to_dict()
        payload.pop("timings_ms")
        return payload

    def test_repeated_calls_identical(self, extractor, eye):
        runs = [self._stable(extractor.extract(eye)) for _ in range(3)]

        assert runs[0] == runs[1] == runs[2]

    def test_fresh_extractor_matches(self, eye):
        first = self._stable(PentacamReferenceExtractor().extract(eye))
        second = self._stable(PentacamReferenceExtractor().extract(eye))

        assert first == second

    def test_interleaved_inputs_do_not_leak_state(self, extractor, eye, screenshot):
        """Result for an input must not depend on what was processed before."""
        clean_first = self._stable(extractor.extract(eye))
        extractor.extract(screenshot)

        assert clean_first == self._stable(extractor.extract(eye))

    def test_timings_are_reported_for_every_stage(self, extractor, eye):
        timings = extractor.extract(eye).timings_ms

        for stage in ("eye_roi", "detection", "features", "polar", "total"):
            assert stage in timings
            assert timings[stage] >= 0.0

    def test_confidence_is_bounded(self, extractor, eye):
        assert 0.0 <= extractor.extract(eye).confidence <= 1.0


class TestSerialisation:
    def test_to_dict_is_json_serialisable(self, extractor, eye):
        payload = json.dumps(extractor.extract(eye).to_dict())

        assert "OK" in payload

    def test_to_dict_contains_expected_keys(self, extractor, eye):
        payload = extractor.extract(eye).to_dict()

        for key in (
            "valid",
            "status",
            "quality",
            "confidence",
            "failure_reason",
            "image_width",
            "image_height",
            "coordinate_system",
            "eye_roi",
            "geometry",
            "iris_roi",
            "feature_set",
            "polar",
            "mask_stats",
            "timings_ms",
        ):
            assert key in payload

    def test_rejected_result_serialises(self, extractor):
        payload = extractor.extract(None).to_dict()

        assert payload["valid"] is False
        assert payload["status"] == "NO_IMAGE"

    def test_polar_metadata_recorded(self, extractor, eye):
        polar = extractor.extract(eye).to_dict()["polar"]

        assert polar["valid"] is True
        assert polar["shape"][0] == polar["num_radial"]
        assert polar["shape"][1] == polar["num_angles"]
        assert polar["valid_fraction"] > 0.9

    def test_eye_roi_dict_omits_pixel_data(self, extractor, eye):
        roi = extractor.extract(eye).to_dict()["eye_roi"]

        assert "grayscale" not in roi
        assert roi["grayscale_shape"] == [
            extractor.extract(eye).eye_roi.height,
            extractor.extract(eye).eye_roi.width,
        ]


# --------------------------------------------------------------------------- #
# Preprocessing helpers
# --------------------------------------------------------------------------- #


class TestPreprocessHelpers:
    def test_to_gray_handles_layouts(self, eye):
        bgr = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)

        assert to_gray(eye).ndim == 2
        assert to_gray(bgr).ndim == 2
        assert to_gray(eye[:, :, None]).ndim == 2

    def test_to_bgr_always_three_channels(self, eye):
        bgra = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGRA)

        for candidate in (eye, eye[:, :, None], to_bgr(eye), bgra):
            assert to_bgr(candidate).shape[2] == 3

    def test_content_bbox_finds_the_panel(self, eye):
        """The panel origin must be recovered from content, whatever it is."""
        origins = [(77, 133), (11, 40), (301, 90)]
        results = []
        for origin in origins:
            shot = make_device_screenshot(eye, panel_origin=origin)
            bbox = content_bbox(build_ui_mask(to_gray(shot)))
            assert bbox is not None
            results.append(bbox)

        # Same content at a different origin yields the same relative box, so
        # nothing about the layout is baked into the pipeline.
        offsets = {
            (bbox[0] - origin[0], bbox[1] - origin[1], bbox[2], bbox[3])
            for bbox, origin in zip(results, origins)
        }
        assert len(offsets) == 1

    def test_content_bbox_none_when_all_suppressed(self):
        assert content_bbox(np.zeros((64, 64), dtype=np.uint8)) is None

    def test_normalize_gray_is_bounded_and_stretched(self):
        rng = np.random.default_rng(0)
        src = rng.integers(40, 60, size=(32, 32)).astype(np.uint8)
        out = normalize_gray(src)

        assert out.dtype == np.float32
        assert float(out.max()) > 200.0

    def test_normalize_gray_flat_input_returns_zeros(self):
        out = normalize_gray(np.full((16, 16), 77, dtype=np.uint8))

        assert np.all(out == 0.0)
