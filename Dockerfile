# TODO(production): pin this by sha256 digest rather than by tag. A tag can be
# repointed upstream, which means the image that passed review is not
# guaranteed to be the image that ships.
FROM python:3.13-slim-bookworm

# opencv-python-headless still needs libGL's loader and glib at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first so a code edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
# The four challenge documents ship with the image so the UI has something to
# demonstrate on without the reviewer hunting for a file.
COPY samples/ ./samples/

# Run unprivileged. The service writes nothing to disk, so it needs no volumes
# and no write access anywhere.
RUN useradd --system --uid 10001 --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
