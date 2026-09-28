"""Unit tests for polar unwrapping module."""

import numpy as np
import pytest
from pupil_tracking.registration.polar import PolarUnwrapper, PolarImage


def test_polar_unwrapper_initialization():
    unwrapper = PolarUnwrapper(num_angles=360, num_radial=64)
    assert unwrapper.num_angles == 360
    assert unwrapper.num_radial == 64


def test_polar_unwrapping_synthetic():
    # Synthetic eye image: 200x200
    img = np.zeros((200, 200), dtype=np.uint8)
    # Pupil at center (100, 100), radius 20
    # Limbus at center (100, 100), radius 60
    # Add a radial wedge texture
    y, x = np.ogrid[:200, :200]
    dist_from_center = np.sqrt((x - 100)**2 + (y - 100)**2)
    angle_from_center = np.arctan2(y - 100, x - 100)

    iris_mask = (dist_from_center >= 20) & (dist_from_center <= 60)
    texture = (128 + 100 * np.sin(4 * angle_from_center)).astype(np.uint8)
    img[iris_mask] = texture[iris_mask]

    unwrapper = PolarUnwrapper(num_angles=180, num_radial=32)
    polar = unwrapper.unwrap(
        img,
        pupil_center=(100.0, 100.0),
        pupil_axes=(20.0, 20.0),
        pupil_angle_deg=0.0,
        limbus_center=(100.0, 100.0),
        limbus_axes=(60.0, 60.0),
        limbus_angle_deg=0.0,
    )

    assert isinstance(polar, PolarImage)
    assert polar.valid is True
    assert polar.image is not None
    assert polar.image.shape == (32, 180)


def test_polar_unwrapping_uses_nearest_source_pixels():
    image = np.arange(200 * 200, dtype=np.uint8).reshape(200, 200)
    unwrapper = PolarUnwrapper(
        num_angles=8,
        num_radial=5,
        inner_margin=0.0,
        outer_margin=0.0,
    )

    polar = unwrapper.unwrap(
        image,
        pupil_center=(100.0, 100.0),
        pupil_axes=(20.0, 20.0),
        pupil_angle_deg=0.0,
        limbus_center=(100.0, 100.0),
        limbus_axes=(60.0, 60.0),
        limbus_angle_deg=0.0,
    )

    expected_x = np.rint(np.linspace(20.0, 60.0, 5) + 100.0).astype(int)
    np.testing.assert_array_equal(polar.image[:, 0], image[100, expected_x])
    np.testing.assert_array_equal(polar.mask[:, 0], np.full(5, 255, dtype=np.uint8))
