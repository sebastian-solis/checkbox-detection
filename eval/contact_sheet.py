"""Render detections as a grid of labelled crops for review.

Correcting ground truth by squinting at a full appraisal page is slow and
error prone: a checkbox is thirty pixels on a page four thousand tall. This
lays every detection out as an enlarged tile with its index printed beside it,
so a reviewer can confirm or reject a whole page in a couple of screens.

Usage:
    python eval/contact_sheet.py neighborhood_site_section
    python eval/contact_sheet.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
ANNOTATIONS_DIR = Path(__file__).resolve().parent / "annotations"
SHEET_DIR = Path(__file__).resolve().parent / "sheets"

TILE = 96
PADDING = 8
COLUMNS = 10
LABEL_HEIGHT = 22


def build_sheet(stem: str) -> list[Path]:
    """Write one or more contact sheets for a sample, return their paths."""
    image = cv2.imread(str(SAMPLES_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"No image for {stem}")

    payload = json.loads((ANNOTATIONS_DIR / f"{stem}.json").read_text())
    boxes = payload["boxes"]

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    per_sheet = COLUMNS * 8

    for sheet_index, start in enumerate(range(0, len(boxes), per_sheet)):
        chunk = boxes[start : start + per_sheet]
        rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        cell = TILE + PADDING * 2
        canvas = np.full(
            (rows * (cell + LABEL_HEIGHT), COLUMNS * cell, 3), 245, dtype=np.uint8
        )

        for position, entry in enumerate(chunk):
            row, column = divmod(position, COLUMNS)
            x1, y1, x2, y2 = entry["bbox"]

            # Pad the crop so the box edges and a little context are visible.
            margin = max(6, (x2 - x1) // 3)
            crop = image[
                max(0, y1 - margin) : min(image.shape[0], y2 + margin),
                max(0, x1 - margin) : min(image.shape[1], x2 + margin),
            ]
            if crop.size == 0:
                continue

            scale = min(TILE / crop.shape[1], TILE / crop.shape[0])
            resized = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )

            top = row * (cell + LABEL_HEIGHT) + LABEL_HEIGHT + PADDING
            left = column * cell + PADDING
            canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized

            label = f"{entry['index']}:{'X' if entry['is_checked'] else '.'}"
            cv2.putText(
                canvas,
                label,
                (left, row * (cell + LABEL_HEIGHT) + LABEL_HEIGHT - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 40, 40) if entry["is_checked"] else (90, 90, 90),
                1,
                cv2.LINE_AA,
            )

        destination = SHEET_DIR / f"{stem}_sheet{sheet_index}.png"
        cv2.imwrite(str(destination), canvas)
        written.append(destination)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", nargs="?", help="Sample name without extension.")
    parser.add_argument("--all", action="store_true", help="Every annotated sample.")
    arguments = parser.parse_args()

    if arguments.all:
        stems = [path.stem for path in sorted(ANNOTATIONS_DIR.glob("*.json"))]
    elif arguments.stem:
        stems = [arguments.stem]
    else:
        parser.error("give a sample name or --all")

    for stem in stems:
        for path in build_sheet(stem):
            print(path)


if __name__ == "__main__":
    main()
