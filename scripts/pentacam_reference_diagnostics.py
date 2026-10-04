"""Pentacam reference-extraction diagnostics.

Runs the pipeline over the synthetic fixtures and the real supplied Pentacam
screenshot, recording per-stage and end-to-end latency plus every quality gate.

Writes JSON to _phase_artifacts/pentacam/ (gitignored).  The clinical screenshot
is read from the desktop and is never copied into the repository.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())

from pupil_tracking.pentacam.reference import PentacamReferenceExtractor
from pupil_tracking.tests.pentacam_fixtures.synthetic import (
    make_device_screenshot,
    make_eye_image,
    make_textureless_eye_image,
    occlude_iris_sectors,
)

SCREENSHOT = r"C:\Users\ANUBHAV\Desktop\WhatsApp Image 2026-10-04 at 15.26.49.jpeg"
OUT_DIR = os.path.join("_phase_artifacts", "pentacam")
REPEATS = 5


def describe(label, image, extractor):
    result = extractor.extract(image)

    # Warm latency, averaged over repeats.
    totals, stages = [], {k: [] for k in result.timings_ms}
    for _ in range(REPEATS):
        run = extractor.extract(image)
        totals.append(run.timings_ms["total"])
        for key, value in run.timings_ms.items():
            stages[key].append(value)

    payload = {
        "label": label,
        "input_shape": list(image.shape),
        "valid": result.valid,
        "status": result.status.value,
        "quality": result.quality.value,
        "confidence": round(result.confidence, 4),
        "failure_reason": result.failure_reason,
        "eye_roi": result.eye_roi.to_dict(),
        "geometry": {
            "pupil_detected": result.geometry.pupil_detected,
            "limbus_detected": result.geometry.limbus_detected,
            "limbus_localized": result.geometry.limbus_localized,
            "pupil_radius_px": round(result.geometry.pupil_radius_px, 2),
            "limbus_radius_px": round(result.geometry.limbus_radius_px, 2),
            "pupil_limbus_ratio": round(result.geometry.pupil_limbus_ratio, 4),
        },
        "gates": {
            "usable_fraction": round(result.feature_set.usable_fraction, 4),
            "polar_coverage": result.mask_stats.get("polar_coverage"),
            "angular_coverage_ratio": result.mask_stats.get("angular_coverage_ratio"),
            "num_accepted_features": result.feature_set.num_accepted,
            "num_candidates": result.feature_set.num_candidates,
            "rejection_reasons": result.feature_set.rejection_reasons,
        },
        "polar": (
            None
            if result.polar is None
            else {
                "shape": list(result.polar.image.shape),
                "inner_radius": round(result.polar.inner_radius, 2),
                "outer_radius": round(result.polar.outer_radius, 2),
            }
        ),
        "latency_ms": {
            "median_total": round(statistics.median(totals), 1),
            "min_total": round(min(totals), 1),
            "max_total": round(max(totals), 1),
            "median_by_stage": {
                k: round(statistics.median(v), 1) for k, v in stages.items() if v
            },
        },
    }
    print(
        f"{label:34} valid={str(result.valid):5} {result.status.value:22} "
        f"limbus_r={result.geometry.limbus_radius_px:7.1f} "
        f"feat={result.feature_set.num_accepted:3} "
        f"total={payload['latency_ms']['median_total']:7.1f}ms"
    )
    return payload


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor = PentacamReferenceExtractor()

    eye = make_eye_image()
    shot_eye = make_eye_image(size=(620, 620), pupil_r=73.0, limbus_r=193.0)
    screenshot_fixture = make_device_screenshot(
        shot_eye, canvas_size=(620, 900), panel_origin=(140, 0)
    )

    cases = [
        ("cropped eye (512x512, BGR-native)", eye),
        ("cropped eye as BGR", cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)),
        ("cropped eye as BGRA", cv2.cvtColor(eye, cv2.COLOR_GRAY2BGRA)),
        ("cropped eye as (H,W,1)", eye[:, :, None]),
        ("device screenshot fixture", screenshot_fixture),
        ("occluded iris (40 deg kept)", occlude_iris_sectors(eye, 60.0, 160.0, 40.0)),
        ("occluded iris (120 deg kept)", occlude_iris_sectors(eye, 60.0, 160.0, 120.0)),
        ("textureless iris", make_textureless_eye_image()),
        ("all-black frame", np.zeros((256, 256), dtype=np.uint8)),
        ("invalid input (None)", None),
    ]

    if os.path.exists(SCREENSHOT):
        real = cv2.imread(SCREENSHOT)
        cases.insert(5, ("REAL Pentacam screenshot", real))

    records = []
    for label, image in cases:
        if image is None:
            result = extractor.extract(None)
            records.append({
                "label": label,
                "valid": result.valid,
                "status": result.status.value,
                "failure_reason": result.failure_reason,
            })
            print(f"{label:34} valid={str(result.valid):5} {result.status.value}")
            continue
        records.append(describe(label, image, extractor))

    report = {
        "generated_by": "scripts/pentacam_reference_diagnostics.py",
        "note": (
            "All images except the real screenshot are SYNTHETIC. Gate values "
            "here are the evidence for the thresholds chosen in "
            "PentacamReferenceExtractor. Timings are CPU and machine-specific."
        ),
        "gate_thresholds": {
            "min_retained_fraction": extractor.min_retained_fraction,
            "min_usable_fraction": extractor.min_usable_fraction,
            "min_polar_coverage": extractor.min_polar_coverage,
            "min_angular_coverage": extractor.min_angular_coverage,
            "min_features": extractor.min_features,
            "good_usable_fraction": extractor.good_usable_fraction,
            "good_polar_coverage": extractor.good_polar_coverage,
            "good_angular_coverage": extractor.good_angular_coverage,
        },
        "repeats": REPEATS,
        "cases": records,
    }

    path = os.path.join(OUT_DIR, "reference_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
