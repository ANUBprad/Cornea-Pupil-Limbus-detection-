# Docker

A reproducible container for the **non-interactive** parts of the detector:
`launch_gui.py video` batch processing, and the pytest suite.

The application is a desktop application first (Tkinter GUI plus live-camera
capture), so the container is deliberately scoped to what genuinely runs
headless. See [What the image does not support](#what-the-image-does-not-support).

Verified with Docker 29.8.0 (Docker Desktop, Windows host, linux/amd64).

---

## Build

```bash
docker build -t pupil-limbus-detector .
```

Produces the runtime image (~686 MB). A second target adds CPU PyTorch for the
full test suite:

```bash
docker build --target test -t pupil-limbus-detector-test .
```

Produces the test image (~1.75 GB).

Both targets share one base and install pinned versions from
[`docker/requirements-runtime.txt`](../docker/requirements-runtime.txt):

| Package | Version | Why |
|---|---|---|
| `numpy` | 2.2.6 | array ops |
| `opencv-python-headless` | 4.11.0.86 | image/video IO, drawing, classical CV |
| `onnxruntime` | 1.24.3 | production ML backend |
| `pytest` | 9.1.1 | test-only, so the suite can run in the image |
| `torch` (test target only) | 2.5.1 (CPU wheel) | required to *collect* 11 test modules, see below |

This is intentionally smaller than the top-level `requirements.txt`. PyTorch is
optional in production by the project's own design
(`pupil_tracking/ml/inference.py`: "PyTorch is optional in production — we
prefer ONNX Runtime"), so the runtime image omits torch and the training stack
(`torchvision`, `segmentation-models-pytorch`, `albumentations`, `scipy`).
Training stays a native-only workflow.

`opencv-python-headless` is used instead of `opencv-python` because the GUI
wheel needs `libGL`, which pulls in roughly 200 MB of Mesa/LLVM drivers that
this image cannot use. Measured effect: 1.03 GB → 686 MB.

---

## Run

The entry point is `python launch_gui.py`, so container arguments are exactly
the CLI arguments documented in the README.

```bash
# Process a video and write the annotated output plus a results CSV
docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/out:/out" \
  pupil-limbus-detector video -i /data/clip.mp4 -o /out/clip_tracked.mp4
```

PowerShell equivalent:

```powershell
docker run --rm `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/data:/data:ro" `
  -v "${PWD}/out:/out" `
  pupil-limbus-detector video -i /data/clip.mp4 -o /out/clip_tracked.mp4
```

Notes:

- The input directory must be mounted; nothing is baked into the image.
- The `-o` path must be writable. `/app` itself is owned by the unprivileged
  runtime user, so writing to `/app/...` also works if you mount a volume there.
- A results CSV is written next to the `-o` path automatically.
- `--help` is the default command, so a bare `docker run --rm pupil-limbus-detector`
  is a quick smoke test.
- The audit log is written to `/app/logs` inside the container and is discarded
  on exit unless you mount that directory.

### Models

Weights are never baked into the image. Mount them read-only at `/app/models`,
using the layout from [`models/README.md`](../models/README.md):

| Path inside container | File |
|---|---|
| `/app/models/onnx/segmentation_quantized.onnx` | INT8 quantized (preferred) |
| `/app/models/onnx/segmentation.onnx` | full-precision ONNX |
| `/app/models/best_model.pth` | PyTorch checkpoint (dev only; needs the test image) |

`UnifiedDetector` picks the first available backend and degrades gracefully:

1. `segmentation_quantized.onnx`
2. `segmentation.onnx`
3. `best_model.pth` (requires torch)
4. no ML backend → classical CV fallback

**With no model mounted the container still works.** Detection falls back to the
classical pipeline and logs:

```
[WARN] Optimized pipeline unavailable - classic pipeline will be used.
[WARNING] PyTorch not installed — ML segmentation inference disabled.
[INFO] UnifiedDetector initialised (ML=unavailable, SmartFitter=enabled, ...)
```

Model weights trained on clinical data have separate distribution rights — see
`models/README.md` before shipping them anywhere.

---

## What the image does not support

These are **not** oversights; they are verified limitations.

| Feature | Status | Reason |
|---|---|---|
| `video` mode | Supported | no display calls; `VideoCapture`/`VideoWriter` work |
| pytest suite | Supported (test target) | see [Tests](#tests) |
| `gui` mode | Not supported | needs a Tkinter display; also uses Windows-only `ctypes.windll` for DPI |
| `camera` mode | Not supported | needs a V4L2 device **and** a display for the preview window |
| `image` mode | Not supported | prints the full report, then blocks on `cv2.imshow` + `cv2.waitKey(0)` |
| Training / export scripts | Not supported | need torch, torchvision, `segmentation-models-pytorch`, albumentations |
| GPU inference | Not supported | CPU-only wheels; use `onnxruntime-gpu` or a native install |

The image uses `opencv-python-headless`, so the display helpers
(`cv2.imshow`, `cv2.waitKey`, `cv2.namedWindow`) are not compiled in at all —
`gui`, `camera` and `image` cannot show a window here regardless of mounts.

Run those modes natively. This is the same split the project's own
`build_app.bat` makes for the Windows distribution.

---

## Tests

```bash
docker build --target test -t pupil-limbus-detector-test .

# Full suite. clinical_data is mounted because the clinical-accuracy tests read
# it; it is never baked into the image.
docker run --rm \
  -v "$PWD/clinical_data:/app/clinical_data:ro" \
  pupil-limbus-detector-test pupil_tracking/tests -q --tb=no -p no:randomly
```

The `test` target exists because `pupil_tracking/iris/__init__.py` imports
`rl_agent`, which subclasses `torch.nn.Module` at module scope behind an
`except ImportError: nn = None` guard. Without torch, pytest cannot collect 11
test modules at all — including `test_component_toggles.py` and
`test_registration_sign_convention.py` — and the run aborts during collection.
That is a pre-existing issue in the application code, not a Docker one; the
lean runtime image simply cannot collect the suite.

`scripts/` is copied only into the test target, because
`pupil_tracking/tests/test_corrected_output.py` imports `annotate_live_video`
from it.

### Verified results

| Run | Result |
|---|---|
| Container (test target, clinical data mounted) | **524 passed, 9 failed, 15 skipped** in 92 s |
| Native host, same session | **522 passed, 11 failed, 15 skipped** in 250 s |

The 9 container failures are exactly the known native baseline:

- `test_clinical_accuracy.py` — 5 limbus cases (`eye_08`, `eye_09`, `eye_11`,
  `eye_12`, `eye_14`) and `eye_10` confidence
- `test_iris_cnn.py::test_segmentation_torch_model_forward`
- `test_refactored_modules.py::test_eye_01_unchanged_after_ring_constraint`
- `test_vectorized_subpixel.py::test_large_contour_performance` (timing flake)

The 2 extra native failures are host-specific and do not reproduce in the
container:

- `test_iris_cnn.py::test_encoder_torch_model_forward_l2_unit` — the native
  torch build is `2.5.1+cu121`; the model lands on CUDA while the input stays on
  CPU (`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`
). The container's CPU-only torch does not hit this.
- `test_pentacam_detector.py::test_detect_synthetic_image` — asserts
  `processing_time_ms < 150.0` and measured 231 ms under host load.

The container therefore reproduces the native baseline and adds no new
failures.

### Verified end-to-end run

A synthetic 20-frame clip was processed in the runtime image (no model mounted,
classical fallback):

```
Processed:    20 frames
Elapsed:      1.3 s
Average FPS:  15.7
Results CSV:  /data/out_tracked.csv
Output video: /data/out_tracked.mp4
```

with `quality=CLINICAL`, pupil and limbus both detected, and exit code 0.

---

## Reproducibility and security

- Base image is pinned to a major/minor tag (`python:3.12-slim-bookworm`, Python
  3.12.15). For byte-identical rebuilds, pin the base by digest.
- Python dependencies are pinned to exact versions, matching the native
  Windows/Python 3.12 baseline. To refresh them, update
  `docker/requirements-runtime.txt` and re-run the suite in the test image.
- Dependencies are installed before the source is copied, so source edits reuse
  the cached pip layer. There are no compilers in the image.
- The container runs as an unprivileged user (`appuser`, uid 10001).
- `.dockerignore` keeps `clinical_data/` (tracked in git, but patient imagery)
  and every model-weight extension out of the build context, along with logs,
  phase artifacts, caches and local environments.
- No secrets, credentials or `.env` files are read by the image or copied into
  it.
- Nothing listens on a network port; this is a batch container, not a service.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker: 'python' is not a docker command` (Windows) | `python launch_gui.py` is the entrypoint; use `docker run … pupil-limbus-detector video -i …` |
| `Cannot open video` | the input path is outside the container — mount its directory |
| `Permission denied` writing `-o` | the target directory is not mounted writable |
| `ONNX model not found: None` | expected when no weights are mounted; detection falls back to classical CV |
| `cv2.error: ... not implemented` | you called a GUI mode; use `video` mode or run natively |
| `Interrupted: N errors during collection` | you used the runtime image for tests — build `--target test` |
