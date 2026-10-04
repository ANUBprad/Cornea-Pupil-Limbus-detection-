"""
Stream A: Phase-Only Correlation (POC) for cyclotorsion detection.

Computes the rotational shift between two polar-unwrapped iris images
using FFT-based phase correlation along the angular axis.

This is the most reliable baseline method, working well on images
with sufficient iris texture regardless of illumination changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class PhaseCorrelationStream(BaseStream):
    """FFT-based phase-only correlation for angular shift detection.

    Algorithm:
        1. Polar-unwrap both iris images
        2. Enhance with CLAHE + column normalisation
        3. Compute cross-power spectrum along angular axis
        4. Find peak in inverse FFT → angular shift = torsion
        5. Sub-pixel refinement via parabolic interpolation

    The confidence is derived from the peak-to-sidelobe ratio (PSR)
    and the percentage of valid pixels in both polar images.

    A peak that is weak (low PSR) or tied with a rival peak (low peak
    separation) is rejected rather than returned, because argmax over a
    near-flat correlation surface yields an arbitrary angle.
    """

    def __init__(self):
        super().__init__(StreamName.PHASE_CORRELATION)
        cfg = get_config().registration
        self.unwrapper = PolarUnwrapper()
        self.enhancer = IrisEnhancer()
        self.upsample_factor = cfg.poc_upsample_factor
        self.min_psr = cfg.poc_min_psr
        self.min_peak_separation = cfg.poc_min_peak_separation

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Compute torsion via phase correlation of polar iris images."""

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        # Step 1: Polar unwrap
        polar_ref = self.unwrapper.unwrap_from_detection(img_ref, detection_ref)
        polar_curr = self.unwrapper.unwrap_from_detection(img_curr, detection_curr)

        if not polar_ref.valid or not polar_curr.valid:
            return self._make_result(
                metadata={"error": "polar_unwrap_failed"}
            )

        # Step 2: Enhance
        enh_ref = self.enhancer.enhance_polar(polar_ref.image, polar_ref.mask)
        enh_curr = self.enhancer.enhance_polar(polar_curr.image, polar_curr.mask)

        # Step 3: Phase correlation along angular axis
        # Average along radial axis to get 1-D angular profiles
        profile_ref = np.mean(enh_ref, axis=0)
        profile_curr = np.mean(enh_curr, axis=0)

        # Combined mask: only use angles where both images are valid
        if polar_ref.mask is not None and polar_curr.mask is not None:
            valid_mask = (np.mean(polar_ref.mask, axis=0) > 127) & \
                         (np.mean(polar_curr.mask, axis=0) > 127)
            valid_frac = np.mean(valid_mask)
        else:
            valid_mask = np.ones(len(profile_ref), dtype=bool)
            valid_frac = 1.0

        if valid_frac < 0.3:
            return self._make_result(
                metadata={"error": "insufficient_valid_pixels",
                           "valid_fraction": float(valid_frac)}
            )

        # Zero out invalid regions
        profile_ref = profile_ref * valid_mask
        profile_curr = profile_curr * valid_mask

        # 1-D phase correlation
        shift_deg, psr, peak_val, separation = self._phase_correlate_1d(
            profile_ref, profile_curr, polar_ref.num_angles
        )

        # Reject unreliable peaks instead of returning an arbitrary angle.
        # A weak PSR means no dominant peak; a small separation means the
        # winner is tied with a rival, so argmax is a coin flip. Both happen
        # when the two polar images cover different physical iris radii, i.e.
        # when the pupil/limbus geometry disagrees between ref and curr.
        if psr < self.min_psr or separation < self.min_peak_separation:
            return self._make_result(
                metadata={
                    "error": "unreliable_peak",
                    "psr": float(psr),
                    "peak_value": float(peak_val),
                    "peak_separation": float(separation),
                    "valid_fraction": float(valid_frac),
                    "rejected_shift_deg": float(shift_deg),
                }
            )

        # Confidence from PSR and valid fraction
        confidence = self._compute_confidence(psr, valid_frac, peak_val)

        return self._make_result(
            torsion_deg=shift_deg,
            confidence=confidence,
            metadata={
                "psr": float(psr),
                "peak_value": float(peak_val),
                "peak_separation": float(separation),
                "valid_fraction": float(valid_frac),
                "num_angles": polar_ref.num_angles,
            },
        )

    def _phase_correlate_1d(
        self,
        signal_ref: np.ndarray,
        signal_curr: np.ndarray,
        num_angles: int,
    ) -> tuple:
        """1-D phase-only correlation with sub-sample refinement.

        Returns
        -------
        shift_deg : float
            Angular shift in degrees.
        psr : float
            Peak-to-sidelobe ratio.
        peak_val : float
            Peak correlation value.
        separation : float
            Relative gap between the winning peak and the best peak outside
            its lobe. Near zero means the winner is tied with a rival.
        """
        n = len(signal_ref)

        # FFT
        fft_ref = np.fft.fft(signal_ref)
        fft_curr = np.fft.fft(signal_curr)

        # Cross-power spectrum
        cross = fft_ref * np.conj(fft_curr)
        magnitude = np.abs(cross)
        magnitude = np.maximum(magnitude, 1e-10)
        phase_only = cross / magnitude

        # Inverse FFT → correlation
        correlation = np.real(np.fft.ifft(phase_only))

        # Find peak
        peak_idx = np.argmax(correlation)
        peak_val = correlation[peak_idx]

        # Sub-pixel refinement via parabolic interpolation
        if 0 < peak_idx < n - 1:
            y_left = correlation[peak_idx - 1]
            y_center = correlation[peak_idx]
            y_right = correlation[peak_idx + 1]
            denom = 2.0 * (2.0 * y_center - y_left - y_right)
            if abs(denom) > 1e-10:
                sub_pixel = (y_left - y_right) / denom
            else:
                sub_pixel = 0.0
            refined_idx = peak_idx + sub_pixel
        else:
            refined_idx = float(peak_idx)

        # Handle wrap-around: if shift > half range, it's negative
        if refined_idx > n / 2:
            refined_idx -= n

        # Convert sample shift to degrees
        deg_per_sample = 360.0 / num_angles
        shift_deg = refined_idx * deg_per_sample

        # Peak-to-sidelobe ratio
        sidelobes = np.delete(correlation, peak_idx)
        sidelobe_mean = np.mean(sidelobes)
        sidelobe_std = np.std(sidelobes) + 1e-10
        psr = (peak_val - sidelobe_mean) / sidelobe_std

        # Peak separation: the winning peak must stand clear of the best rival
        # peak, ignoring the winning lobe itself. A tiny gap means two peaks are
        # effectively tied and the argmax index is not meaningful.
        separation = self._peak_separation(correlation, int(peak_idx))

        return float(shift_deg), float(psr), float(peak_val), float(separation)

    @staticmethod
    def _peak_separation(correlation: np.ndarray, peak_idx: int, exclude: int = 5) -> float:
        """Relative gap between the winning correlation peak and the best rival.

        Parameters
        ----------
        correlation : np.ndarray
            1-D correlation surface from the inverse FFT.
        peak_idx : int
            Index of the winning peak.
        exclude : int
            Samples either side of the peak treated as part of its own lobe.
        """
        n = len(correlation)
        if n == 0 or abs(float(correlation[peak_idx])) < 1e-12:
            return 0.0

        circular_distance = np.abs((np.arange(n) - peak_idx + n // 2) % n - n // 2)
        rivals = correlation[circular_distance > exclude]
        if rivals.size == 0:
            return 0.0

        top1 = float(correlation[peak_idx])
        return float((top1 - float(np.max(rivals))) / abs(top1))

    @staticmethod
    def _compute_confidence(psr: float, valid_frac: float, peak_val: float) -> float:
        """Compute confidence from PSR, valid fraction, and peak value.

        Higher PSR → sharper peak → more reliable estimate.
        Higher valid fraction → more data → more reliable.
        """
        # PSR confidence: sigmoid mapping, PSR=5 → ~0.5, PSR=15 → ~0.95
        psr_conf = 1.0 / (1.0 + np.exp(-0.3 * (psr - 8.0)))

        # Valid fraction penalty
        validity_conf = min(valid_frac / 0.6, 1.0)

        # Peak value check (should be positive and significant)
        peak_conf = min(max(peak_val, 0.0), 1.0)

        return float(np.clip(psr_conf * validity_conf * peak_conf, 0.0, 1.0))
