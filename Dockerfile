# TODO(production): pin this by sha256 digest rather than by tag. A tag can be
# repointed upstream, which means the image that passed review is not
# guaranteed to be the image that ships.
FROM python:3.13-slim-bookworm

# opencv-python-headless still needs libGL's loader and glib at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Upgrade the base pip toolchain before installing anything else. The
# python:3.13-slim-bookworm base ships with a setuptools that Trivy flags as
# HIGH (CVE-2025-47273, path traversal in PackageIndex). Bumping it here keeps
# the base image current without swapping to a heavier one.
RUN pip install --no-cache-dir --upgrade "pip>=25.0" "setuptools>=78.1.1" "wheel"

# Dependencies first so a code edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
# The four challenge documents ship with the image so the UI has something to
# demonstrate on without the reviewer hunting for a file.
COPY samples/ ./samples/

# Run unprivileged. A dedicated uid inside the container is one of the
# defence-in-depth layers documented in WRITEUP.md: even if an attacker got
# code execution inside the image, they land as appuser on a read-only
# filesystem, with no home to write to and no capabilities to escalate.
RUN groupadd --gid 10001 appgrp \
 && useradd --uid 10001 --gid 10001 --home-dir /nonexistent \
    --no-create-home --shell /usr/sbin/nologin appuser
USER appuser:appgrp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
