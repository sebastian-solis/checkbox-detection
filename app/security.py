"""Validation for untrusted uploads.

The documents this service is built for are mortgage appraisals: they carry
borrower names, property addresses and lender details. That shapes two rules
followed throughout the service.

First, nothing is written to disk. Uploads are decoded in memory and dropped
when the request ends, so there is no spool directory to leak, back up, or
forget to purge.

Second, nothing derived from the document body is ever logged. Logs carry a
request id and measurements, never content.

The limits below exist because decoding an attacker-supplied image is the one
place this service does unbounded work on unbounded input.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import UploadSettings


class UploadRejected(Exception):
    """Raised when an upload fails validation. Message is safe to return."""


# Magic byte prefixes for the formats we accept. The declared Content-Type and
# the filename are both attacker-controlled, so the bytes are what decides.
_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"II*\x00",  # TIFF little endian
    b"MM\x00*",  # TIFF big endian
    b"BM",  # BMP
    b"RIFF",  # WEBP (container; the WEBP tag sits at offset 8)
)


def validate_and_decode(
    payload: bytes, content_type: str | None, settings: UploadSettings
) -> np.ndarray:
    """Turn raw upload bytes into an image, refusing anything suspicious.

    Order matters: cheap checks run before expensive ones so a malicious upload
    is rejected before any decoding work happens.
    """
    if not payload:
        raise UploadRejected("Empty upload.")

    if len(payload) > settings.max_file_bytes:
        raise UploadRejected(
            f"File exceeds the {_human_bytes(settings.max_file_bytes)} limit."
        )

    if content_type and content_type not in settings.allowed_content_types:
        raise UploadRejected(f"Unsupported content type: {content_type}.")

    if not _has_known_magic(payload):
        raise UploadRejected(
            "File content is not a supported image format "
            "(PNG, JPEG, TIFF, BMP or WEBP)."
        )

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UploadRejected("File could not be decoded as an image.")

    height, width = image.shape[:2]
    if height * width > settings.max_pixels:
        raise UploadRejected(
            f"Image resolution {width}x{height} exceeds the "
            f"{settings.max_pixels:,} pixel limit."
        )

    return image


def _has_known_magic(payload: bytes) -> bool:
    """Check the leading bytes against known image signatures."""
    return any(payload.startswith(prefix) for prefix in _MAGIC_PREFIXES)


def _human_bytes(size: int) -> str:
    """Format a byte count so small limits do not render as '0 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"
