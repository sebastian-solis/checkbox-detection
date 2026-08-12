"""HTTP service for checkbox detection on document images.

POST /detect is the contract required by the challenge. Everything else exists
to make the solution inspectable: a browser UI to see detections on the page, a
debug endpoint exposing the numbers behind each decision, and a health check.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import DetectionSettings, UploadSettings
from app.detection import classical, render
from app.schemas import (
    Box,
    DebugBox,
    DebugResponse,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
)
from app.security import UploadRejected, validate_and_decode

VERSION = "1.0.0"

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger("checkbox-detection")

detection_settings = DetectionSettings.from_env()
upload_settings = UploadSettings.from_env()

app = FastAPI(
    title="Checkbox Detection",
    version=VERSION,
    description=(
        "Detects and classifies checkboxes in scanned document images. "
        "Uploads are processed in memory and never written to disk."
    ),
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tag every request so logs correlate without recording any content."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(UploadRejected)
async def handle_upload_rejected(request: Request, exc: UploadRejected):
    """Return validation failures as 400 without echoing the payload."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Upload rejected: %s (request_id=%s)", exc, request_id)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc), request_id=request_id).model_dump(),
    )


@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect and classify checkboxes in a document image",
)
async def detect(request: Request, file: UploadFile = File(...)) -> DetectResponse:
    """Return every checkbox found in the uploaded image.

    This is the endpoint specified by the challenge and its response shape is
    fixed: a list of boxes, each with pixel coordinates and whether it is
    marked.
    """
    checkboxes, _, elapsed_ms, _ = await _run_detection(request, file)
    return DetectResponse(
        boxes=[
            Box(bbox=checkbox.bbox.as_list(), is_checked=checkbox.is_checked)
            for checkbox in checkboxes
        ]
    )


@app.post(
    "/detect/debug",
    response_model=DebugResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect checkboxes and return the measurements behind each decision",
)
async def detect_debug(
    request: Request, file: UploadFile = File(...)
) -> DebugResponse:
    """Same detection, plus the confidence and ink ratio driving each verdict."""
    checkboxes, image, elapsed_ms, _ = await _run_detection(request, file)
    height, width = image.shape[:2]
    checked_count = sum(1 for checkbox in checkboxes if checkbox.is_checked)

    return DebugResponse(
        boxes=[
            DebugBox(
                bbox=checkbox.bbox.as_list(),
                is_checked=checkbox.is_checked,
                confidence=checkbox.confidence,
                ink_ratio=round(checkbox.ink_ratio, 4),
            )
            for checkbox in checkboxes
        ],
        image_width=width,
        image_height=height,
        elapsed_ms=elapsed_ms,
        checked_count=checked_count,
        unchecked_count=len(checkboxes) - checked_count,
    )


@app.post(
    "/detect/visualize",
    responses={200: {"content": {"image/png": {}}}, 400: {"model": ErrorResponse}},
    summary="Return the document with every detection outlined",
)
async def detect_visualize(request: Request, file: UploadFile = File(...)) -> Response:
    """Render detections onto the page: green for marked, red for unmarked."""
    checkboxes, image, _, request_id = await _run_detection(request, file)
    annotated = render.annotate(image, checkboxes)
    return Response(
        content=render.encode_png(annotated),
        media_type="image/png",
        headers={"X-Request-ID": request_id, "X-Detection-Count": str(len(checkboxes))},
    )


@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def _run_detection(request: Request, file: UploadFile):
    """Shared path: validate, decode, detect, and log without leaking content."""
    request_id = getattr(request.state, "request_id", "unknown")
    payload = await file.read()

    image = validate_and_decode(payload, file.content_type, upload_settings)

    started = time.perf_counter()
    checkboxes = classical.detect(image, detection_settings)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    height, width = image.shape[:2]
    # Measurements only. The document body never reaches the log.
    logger.info(
        "detection complete request_id=%s dimensions=%dx%d detections=%d elapsed_ms=%.1f",
        request_id,
        width,
        height,
        len(checkboxes),
        elapsed_ms,
    )

    return checkboxes, image, elapsed_ms, request_id


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# The four challenge documents are shipped as demo assets so the UI can be tried
# without hunting for a file. This serves a fixed directory of files that came
# with the build, not anything a caller ever uploaded.
if _SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=_SAMPLES_DIR), name="samples")
