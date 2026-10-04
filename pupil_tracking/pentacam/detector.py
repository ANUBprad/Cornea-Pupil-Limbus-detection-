"""Pentacam iris and anterior segment detector.

Provides robust, additive anatomical feature detection on Pentacam (seated IR)
anterior segment and Scheimpflug eye imagery:
    1. Rejects Pentacam UI chrome, overlays, reticles, and annotation text.
    2. Localizes anatomical pupil and limbus boundaries.
    3. Normalizes iris stroma between pupil and limbus.
    4. Detects distinctive iris landmarks (crypts, collarette, furrows, pigment spots).
    5. Extracts contrast-invariant multi-scale descriptors for cross-modality matching.

Implementation goals (not clinical validation):
    - Purely additive, non-mutating.
    - Zero interference with existing ELITA centration pipeline.
    - Rejects UI feathers and overlays.
    - Runtime depends on image size, hardware, and OpenCV build; benchmark on target data.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.pentacam.preprocess import build_ui_mask
from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamFeature,
    PentacamFeatureSet,
    PentacamGeometry,
    PentacamImageType,
    PentacamQuality,
)
from pupil_tracking.utils.types import EllipseParams

logger = logging.getLogger(__name__)


class PentacamIrisDetector:
    """Detects pupil, limbus, and distinctive iris landmarks on Pentacam imagery.

    Designed for cross-modality iris registration against ELITA RGB/IR images.
    """

    def __init__(
        self,
        num_angles: int = 72,
        num_radii: int = 8,
        min_contrast: float = 3.5,
        min_features: int = 6,
        inner_inset_frac: float = 0.10,
        outer_inset_frac: float = 0.08,
    ) -> None:
        self.num_angles = num_angles
        self.num_radii = num_radii
        self.min_contrast = min_contrast
        self.min_features = min_features
        self.inner_inset_frac = inner_inset_frac
        self.outer_inset_frac = outer_inset_frac

    def detect(
        self,
        image: np.ndarray,
        geometry: Optional[PentacamGeometry] = None,
        image_type: PentacamImageType = PentacamImageType.ANTERIOR_SEGMENT,
    ) -> PentacamDetectionResult:
        """Detect iris structure and landmarks from a Pentacam image.

        Parameters
        ----------
        image : np.ndarray
            Pentacam image (BGR or grayscale uint8).
        geometry : Optional[PentacamGeometry]
            Pre-computed pupil/limbus geometry, if available.
        image_type : PentacamImageType
            Type of Pentacam image.

        Returns
        -------
        PentacamDetectionResult
            Validated detection result with geometry and accepted iris features.
        """
        t0 = time.perf_counter()

        if image is None or image.size == 0:
            return PentacamDetectionResult(
                valid=False,
                status=PentacamDetectionStatus.NO_IMAGE,
                failure_reason="Empty or null image input",
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        # 1. Build UI / Overlay exclusion mask
        ui_mask = self._build_ui_mask(gray)

        # 2. Localize or validate pupil and limbus
        if geometry is None or not (geometry.pupil_detected and geometry.limbus_detected):
            detected_geom, geom_err = self._localize_geometry(gray, ui_mask)
            if detected_geom is None:
                return PentacamDetectionResult(
                    valid=False,
                    status=PentacamDetectionStatus.NO_PUPIL if "pupil" in geom_err else PentacamDetectionStatus.NO_LIMBUS,
                    failure_reason=geom_err,
                    image_width=w,
                    image_height=h,
                    processing_time_ms=(time.perf_counter() - t0) * 1000.0,
                )
            geom = detected_geom
        else:
            geom = geometry

        if not geom.limbus_localized:
            return PentacamDetectionResult(
                valid=False,
                status=PentacamDetectionStatus.NO_LIMBUS,
                geometry=geom,
                image_width=w,
                image_height=h,
                failure_reason=(
                    f"Limbus search pinned to a window bound "
                    f"(r={geom.limbus_radius_px:.1f}px); no iris-to-sclera "
                    "transition was bracketed"
                ),
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. Extract iris annulus mask
        annulus_mask = self._build_annulus_mask(gray.shape, geom, ui_mask)

        # 4. Detect distinctive iris landmarks inside the annulus
        features, feature_metrics = self._extract_features(gray, geom, annulus_mask)

        # 5. Determine quality grade and status
        n_accepted = len(features)
        if n_accepted < self.min_features:
            status = PentacamDetectionStatus.INSUFFICIENT_FEATURES
            quality = PentacamQuality.POOR
            valid = False
            fail_reason = f"Only {n_accepted} features accepted (minimum {self.min_features})"
        else:
            status = PentacamDetectionStatus.OK
            valid = True
            fail_reason = ""
            if n_accepted >= 40 and feature_metrics["angular_coverage"] >= 0.70:
                quality = PentacamQuality.GOOD
            elif n_accepted >= 20:
                quality = PentacamQuality.ACCEPTABLE
            else:
                quality = PentacamQuality.MARGINAL

        confidence = float(np.clip(
            (n_accepted / 50.0) * 0.6 + feature_metrics["angular_coverage"] * 0.4,
            0.0, 1.0,
        )) if valid else 0.0

        feature_set = PentacamFeatureSet(
            features=features,
            num_candidates=feature_metrics["num_candidates"],
            num_accepted=n_accepted,
            angular_coverage_ratio=feature_metrics["angular_coverage"],
            largest_angular_gap_deg=feature_metrics["largest_gap_deg"],
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return PentacamDetectionResult(
            valid=valid,
            status=status,
            image_type=image_type,
            geometry=geom,
            feature_set=feature_set,
            image_width=w,
            image_height=h,
            coordinate_system="pentacam_pixel",
            quality=quality,
            confidence=confidence,
            failure_reason=fail_reason,
            processing_time_ms=dt_ms,
        )

    def _build_ui_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create a mask where valid ocular image is 255 and UI chrome/text is 0.

        Thin wrapper over the shared :func:`pupil_tracking.pentacam.preprocess.build_ui_mask`
        so the detector and the reference pipeline use one implementation.
        """
        return build_ui_mask(gray)

    def _localize_geometry(
        self,
        gray: np.ndarray,
        ui_mask: np.ndarray,
    ) -> Tuple[Optional[PentacamGeometry], str]:
        """Automatically find pupil and limbus on Pentacam IR imagery.

        Uses Otsu thresholding + contour fitting + concentric circular search.
        """
        h, w = gray.shape[:2]

        # In IR Scheimpflug images, the pupil is the darkest central circular cavity
        blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

        # 1. Pupil localization
        # Pupil is dark: search lowest 15th percentile of intensity inside valid mask
        valid_pixels = blurred[ui_mask > 0]
        if len(valid_pixels) == 0:
            return None, "No valid ocular pixels outside UI"

        p15 = float(np.percentile(valid_pixels, 15))
        pupil_bin = ((blurred < max(p15, 25.0)) & (ui_mask > 0)).astype(np.uint8) * 255

        # Morphological close to bridge corneal glints
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        pupil_bin = cv2.morphologyEx(pupil_bin, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(pupil_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pupil = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_pupil_area = math.pi * (min(h, w) * 0.05) ** 2
            max_pupil_area = math.pi * (min(h, w) * 0.35) ** 2
            if not (min_pupil_area <= area <= max_pupil_area):
                continue
            if len(cnt) < 5:
                continue

            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
            smaj = max(d1, d2) / 2.0
            smin = min(d1, d2) / 2.0
            eccentricity = math.sqrt(max(0.0, 1.0 - (smin / (smaj + 1e-6)) ** 2))
            if eccentricity > 0.65:
                continue  # Pupil should not be extremely oblong

            # Distance to image center
            center_dist = math.hypot(cx - w / 2.0, cy - h / 2.0)
            center_penalty = center_dist / (max(h, w) * 0.5)

            score = area / (1.0 + center_penalty * 3.0)
            if score > best_score:
                best_score = score
                best_pupil = EllipseParams(
                    center_x=float(cx),
                    center_y=float(cy),
                    semi_major=float(smaj),
                    semi_minor=float(smin),
                    angle_deg=float(angle),
                    fit_quality=0.90,
                    eccentricity=float(eccentricity),
                    circularity=float(smin / smaj),
                )

        if best_pupil is None:
            # Fallback to central region circular estimation
            cx, cy = w / 2.0, h / 2.0
            pr = min(h, w) * 0.12
            best_pupil = EllipseParams(
                center_x=float(cx),
                center_y=float(cy),
                semi_major=float(pr),
                semi_minor=float(pr),
                angle_deg=0.0,
                fit_quality=0.50,
            )

        # 2. Limbus localization
        # Limbus is approximately concentric with the pupil. The lower bound only
        # needs to clear the pupil edge (~1.0x pr): at 2.0 it sat above a measured
        # iris->sclera transition (1.95x pr), pinning argmax to index 0.
        pcx, pcy = best_pupil.center_x, best_pupil.center_y
        pr = best_pupil.radius

        min_limbus_r = pr * 1.7
        max_limbus_r = min(pr * 3.8, min(pcx, w - pcx, pcy, h - pcy) * 0.95)

        # The limbus is only localized when the steepest iris-to-sclera brightening
        # lands strictly inside the search window.  On either edge the reported
        # radius is an artifact of the window bounds, not a measured transition.
        limbus_localized = False

        if min_limbus_r >= max_limbus_r:
            limbus_r = min_limbus_r
        else:
            # Radial derivative search around pupil center
            n_angles = 36
            test_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            r_samples = np.linspace(min_limbus_r, max_limbus_r, 40)

            radial_grads = []
            for r in r_samples:
                xs = np.clip(np.round(pcx + r * np.cos(test_angles)).astype(int), 0, w - 1)
                ys = np.clip(np.round(pcy + r * np.sin(test_angles)).astype(int), 0, h - 1)
                vals = blurred[ys, xs]
                radial_grads.append(np.mean(vals))

            radial_grads = np.array(radial_grads)
            # Limbus transition from iris to sclera is an increase in brightness
            diffs = np.gradient(radial_grads)
            best_r_idx = int(np.argmax(diffs))
            limbus_r = float(r_samples[best_r_idx])
            limbus_localized = 0 < best_r_idx < len(r_samples) - 1

        best_limbus = EllipseParams(
            center_x=float(pcx),
            center_y=float(pcy),
            semi_major=float(limbus_r),
            semi_minor=float(limbus_r),
            angle_deg=0.0,
            fit_quality=0.85,
        )

        geom = PentacamGeometry(
            pupil=best_pupil,
            limbus=best_limbus,
            pupil_detected=True,
            limbus_detected=True,
            limbus_localized=limbus_localized,
            pupil_radius_px=float(best_pupil.radius),
            limbus_radius_px=float(best_limbus.radius),
            pupil_limbus_ratio=float(best_pupil.radius / max(best_limbus.radius, 1e-6)),
        )

        return geom, ""

    def _build_annulus_mask(
        self,
        shape: Tuple[int, int],
        geom: PentacamGeometry,
        ui_mask: np.ndarray,
    ) -> np.ndarray:
        """Create a boolean mask for the annular iris region, excluding UI overlays."""
        h, w = shape[:2]
        annulus = np.zeros((h, w), dtype=np.uint8)

        le = geom.limbus
        pe = geom.pupil

        if le is None or pe is None:
            return annulus

        # Outer limbus ellipse (inset)
        outer_smaj = int(round(le.semi_major * (1.0 - self.outer_inset_frac)))
        outer_smin = int(round(le.semi_minor * (1.0 - self.outer_inset_frac)))
        cv2.ellipse(
            annulus,
            (int(round(le.center_x)), int(round(le.center_y))),
            (max(1, outer_smaj), max(1, outer_smin)),
            le.angle_deg,
            0, 360,
            255, -1,
        )

        # Inner pupil ellipse (inset outwards)
        inner_smaj = int(round(pe.semi_major * (1.0 + self.inner_inset_frac)))
        inner_smin = int(round(pe.semi_minor * (1.0 + self.inner_inset_frac)))
        cv2.ellipse(
            annulus,
            (int(round(pe.center_x)), int(round(pe.center_y))),
            (max(1, inner_smaj), max(1, inner_smin)),
            pe.angle_deg,
            0, 360,
            0, -1,
        )

        # Combine with UI exclusion mask
        annulus = cv2.bitwise_and(annulus, ui_mask)
        return annulus

    def _extract_features(
        self,
        gray: np.ndarray,
        geom: PentacamGeometry,
        annulus_mask: np.ndarray,
    ) -> Tuple[List[PentacamFeature], Dict]:
        """Detect distinctive iris landmarks on a polar lattice and classify them.

        The whole ``(num_angles x num_radii)`` lattice is evaluated with array
        operations rather than a per-candidate Python loop: patches, texture
        statistics and the 16-bin orientation descriptors are all gathered and
        reduced in a handful of vectorized passes. Semantics are identical to
        the former scalar loop (including candidate ordering and the
        ``num_candidates`` count, which includes every lattice point).
        """
        h, w = gray.shape[:2]
        pe = geom.pupil
        le = geom.limbus

        if pe is None or le is None:
            return [], {"num_candidates": 0, "angular_coverage": 0.0, "largest_gap_deg": 360.0}

        pcx, pcy = pe.center_x, pe.center_y
        lcx, lcy = le.center_x, le.center_y

        # Enhance local contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Texture response: Laplacian of Gaussian
        laplacian = cv2.Laplacian(enhanced, cv2.CV_32F, ksize=3)
        abs_lap = np.abs(laplacian)

        # Precompute gradient magnitude and orientation once for the entire image
        gx_full = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy_full = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        mag_full, ori_full = cv2.cartToPolar(gx_full, gy_full, angleInDegrees=True)

        # Dynamic local contrast threshold derived from iris stroma
        iris_pixels = enhanced[annulus_mask > 0]
        if len(iris_pixels) < 50:
            return [], {"num_candidates": 0, "angular_coverage": 0.0, "largest_gap_deg": 360.0}

        p10 = float(np.percentile(iris_pixels, 10))
        p90 = float(np.percentile(iris_pixels, 90))
        iris_contrast_span = max(p90 - p10, 10.0)

        angles_deg = np.linspace(0.0, 360.0, self.num_angles, endpoint=False)
        rad_norms = np.linspace(0.15, 0.85, self.num_radii)
        num_candidates = int(angles_deg.size) * int(rad_norms.size)

        # Limbus and pupil radius (constant across the lattice)
        r_pupil = pe.radius
        r_limbus = le.radius

        # Lattice coordinates, angle-major then radial (matches original order).
        aa = angles_deg[:, None]
        rr = rad_norms[None, :]
        r_px = r_pupil + rr * (r_limbus - r_pupil)
        cx = pcx + rr * (lcx - pcx)
        cy = pcy + rr * (lcy - pcy)
        a_rad = np.deg2rad(aa)
        fx = cx + r_px * np.cos(a_rad)
        fy = cy + r_px * np.sin(a_rad)

        ix = np.rint(fx).astype(np.int32)
        iy = np.rint(fy).astype(np.int32)

        # Valid candidates: inside image, inside annulus, patch fully in bounds.
        half_patch = 3
        inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ok = inb & (annulus_mask[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)] > 0)
        ok &= (
            (ix >= half_patch) & (ix < w - half_patch)
            & (iy >= half_patch) & (iy < h - half_patch)
        )

        if not np.any(ok):
            metrics = self._compute_coverage_metrics([])
            metrics["num_candidates"] = num_candidates
            return [], metrics

        sel_a, sel_r = np.nonzero(ok)
        sfx = fx[sel_a, sel_r].ravel()
        sfy = fy[sel_a, sel_r].ravel()
        six = ix[sel_a, sel_r].ravel()
        siy = iy[sel_a, sel_r].ravel()
        n = int(six.size)

        # Gather every 7x7 patch in a single fancy-index operation -> (n, 7, 7)
        offs = np.arange(-half_patch, half_patch + 1)
        py = siy[:, None] + offs[None, :]
        px = six[:, None] + offs[None, :]
        patches = enhanced[py[:, :, None], px[:, None, :]]
        lap_patches = abs_lap[py[:, :, None], px[:, None, :]]
        mag_patches = mag_full[py[:, :, None], px[:, None, :]]
        ori_patches = ori_full[py[:, :, None], px[:, None, :]]

        mean_lap = lap_patches.mean(axis=(1, 2))
        local_std = patches.std(axis=(1, 2))

        keep = (mean_lap >= self.min_contrast) & (local_std >= (self.min_contrast * 0.7))

        # 16-bin normalized gradient orientation descriptor (contrast-invariant).
        # Bin width is fixed at 360/16 = 22.5 deg, so indices are computed
        # arithmetically and accumulated with one bincount instead of 576
        # np.histogram calls (each of which rebuilt identical bin edges).
        bins = np.floor(ori_patches / 22.5).astype(np.int32)
        np.clip(bins, 0, 15, out=bins)
        rows = np.arange(n, dtype=np.int64)[:, None, None]
        hist = np.bincount(
            (rows * 16 + bins).ravel(),
            weights=mag_patches.ravel(),
            minlength=n * 16,
        ).reshape(n, 16)
        hist_norm = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-7)

        confidence = np.clip(
            (mean_lap / 20.0) * 0.5 + (local_std / iris_contrast_span) * 0.5,
            0.1, 1.0,
        )

        accepted_features: List[PentacamFeature] = []
        for j in np.nonzero(keep)[0]:
            feat = PentacamFeature(
                id=len(accepted_features),
                x=float(sfx[j]),
                y=float(sfy[j]),
                angle_deg=float(angles_deg[sel_a[j]]),
                radial_norm=float(rad_norms[sel_r[j]]),
                response=float(mean_lap[j]),
                confidence=float(confidence[j]),
                valid=True,
                descriptor=hist_norm[j].astype(np.float32),
            )
            accepted_features.append(feat)

        # Compute coverage metrics
        metrics = self._compute_coverage_metrics(accepted_features)
        metrics["num_candidates"] = num_candidates

        return accepted_features, metrics

    def _compute_coverage_metrics(self, features: List[PentacamFeature]) -> Dict:
        """Compute angular coverage and largest angular gap."""
        if not features:
            return {"angular_coverage": 0.0, "largest_gap_deg": 360.0}

        angles = sorted([f.angle_deg for f in features])
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append((angles[0] + 360.0) - angles[-1])
        largest_gap = float(max(gaps))

        # 30-degree bin occupancy (12 bins)
        bins = set(int(a // 30.0) for a in angles)
        coverage_ratio = float(len(bins) / 12.0)

        return {
            "angular_coverage": coverage_ratio,
            "largest_gap_deg": largest_gap,
        }
