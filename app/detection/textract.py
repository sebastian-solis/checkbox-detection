"""Amazon Textract as an alternative detection backend.

Textract already solves this problem: `AnalyzeDocument` with the `FORMS` feature
returns `SELECTION_ELEMENT` blocks carrying a bounding box and a
`SelectionStatus` of `SELECTED` or `NOT_SELECTED`. The challenge permits any
tool, HomeVision already runs on AWS, and pretending the managed service does
not exist would be the wrong kind of purity.

It is not the default, for reasons argued in WRITEUP.md: the reviewer would need
their own AWS account to run the submission at all, a network round trip costs
one to three seconds per page against tens of milliseconds locally, there is a
per-page charge, and every page leaves the process. Those are deployment
trade-offs rather than accuracy ones, so the sensible shape is a swappable
backend with the local pipeline in front.

Selecting `engine=textract` without credentials configured returns a clear error
rather than a stack trace.
"""

from __future__ import annotations

import numpy as np

from app.detection.types import BoundingBox, Checkbox


class TextractUnavailable(Exception):
    """Raised when the Textract backend cannot be used. Message is user-safe."""


def detect(image_bytes: bytes, image_shape: tuple[int, int], region: str | None = None):
    """Return checkboxes for a page using Textract's selection elements.

    Textract reports geometry normalised to the page, so coordinates are scaled
    back into the pixel space of the submitted image to keep the API contract
    identical whichever backend produced the answer.
    """
    client = _client(region)

    try:
        response = client.analyze_document(
            Document={"Bytes": image_bytes}, FeatureTypes=["FORMS"]
        )
    except Exception as exc:
        raise TextractUnavailable(f"Textract request failed: {type(exc).__name__}") from exc

    return parse_blocks(response.get("Blocks", []), image_shape)


def parse_blocks(blocks: list[dict], image_shape: tuple[int, int]) -> list[Checkbox]:
    """Convert Textract blocks into the same Checkbox objects the API returns.

    Split out from the network call so it can be tested against recorded
    responses without credentials or a billed request.
    """
    height, width = image_shape[:2]
    checkboxes: list[Checkbox] = []

    for block in blocks:
        if block.get("BlockType") != "SELECTION_ELEMENT":
            continue

        box = block.get("Geometry", {}).get("BoundingBox")
        if not box:
            continue

        x1 = round(box["Left"] * width)
        y1 = round(box["Top"] * height)
        x2 = round((box["Left"] + box["Width"]) * width)
        y2 = round((box["Top"] + box["Height"]) * height)

        # Guard against a degenerate box: the contract promises x2 > x1.
        if x2 <= x1 or y2 <= y1:
            continue

        is_checked = block.get("SelectionStatus") == "SELECTED"
        confidence = float(block.get("Confidence", 0.0)) / 100.0

        checkboxes.append(
            Checkbox(
                bbox=BoundingBox(x1, y1, x2, y2),
                is_checked=is_checked,
                confidence=round(confidence, 3),
                # Textract reports a decision, not an ink measurement. None says
                # "this backend does not produce that number"; zero would read as
                # a measurement, and NaN is not valid JSON.
                ink_ratio=None,
            )
        )

    return checkboxes


def _client(region: str | None):
    """Build a Textract client, failing with a readable message if it cannot."""
    try:
        import boto3
    except ImportError as exc:
        raise TextractUnavailable(
            "The textract engine needs boto3. Install it with "
            "'pip install boto3', or use the default classical engine."
        ) from exc

    from botocore.exceptions import BotoCoreError, NoCredentialsError

    try:
        return boto3.client("textract", region_name=region)
    except (BotoCoreError, NoCredentialsError) as exc:
        raise TextractUnavailable(
            "No AWS credentials are configured for the textract engine. "
            "Use the default classical engine, which needs none."
        ) from exc


def encode_for_textract(image: np.ndarray) -> bytes:
    """Encode a decoded image back to PNG bytes for the Textract payload."""
    import cv2

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise TextractUnavailable("Could not encode the page for Textract.")
    return buffer.tobytes()
