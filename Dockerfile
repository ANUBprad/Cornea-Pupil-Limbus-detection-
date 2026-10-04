# Two targets share one base:
#
#   docker build .                      -> runtime image (default, this file's last stage)
#   docker build --target test .         -> runtime + CPU torch, for the full pytest suite
#
# Runtime scope: the non-interactive entry point (`launch_gui.py video`) and the
# pytest suite. The Tkinter GUI, live-camera mode, the interactive `image`
# overlay window, and model training are native-only -- see docs/DOCKER.md.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser

# No apt packages are required: every runtime dependency ships a cp312
# manylinux wheel, and opencv-python-headless does not link against libGL.

WORKDIR /app

# Dependencies are installed before the application is copied so that source
# edits do not invalidate the pip layer.
COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && rm /tmp/requirements-runtime.txt

# The audit logger and the model loader resolve `logs/` and `models/` from the
# working directory, so both must exist and stay writable by the runtime user.
# /app/models is the mount point for weights; weights are never baked in.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs /app/models \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser launch_gui.py ./launch_gui.py
COPY --chown=appuser:appuser pupil_tracking ./pupil_tracking

USER appuser

# Test-only target. `pupil_tracking.iris.__init__` imports rl_agent, which
# subclasses torch.nn.Module at module scope behind an `except ImportError:
# nn = None` guard, so without torch pytest cannot even collect the 11 test
# modules that reach that package. This stage restores native collection parity
# and is not part of the shipped runtime.
FROM base AS test

# test_corrected_output.py imports annotate_live_video from scripts/, so the
# test target needs it. The shipped runtime image does not.
COPY --chown=appuser:appuser scripts ./scripts

USER root
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1
USER appuser

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["pupil_tracking/tests", "-q"]

# Shipped runtime. Declared last so a plain `docker build .` produces this one.
FROM base AS runtime

ENTRYPOINT ["python", "launch_gui.py"]
CMD ["--help"]
