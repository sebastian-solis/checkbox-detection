"""Bootstrap ground-truth files, then get out of the way.

Typing a hundred bounding boxes per page by hand is not a good use of anyone's
time, so this seeds each annotation file with the detector's current output and
renders a review sheet beside it. What it produces is a draft, not ground
truth: every file is written with ``"reviewed": false`` and the scoring harness
refuses to read it until a human flips that flag.

Review workflow:
  1. python eval/bootstrap_annotations.py
  2. Open eval/review/<name>_review.png. Every box carries its index.
  3. In eval/annotations/<name>.json: fix wrong is_checked values, delete boxes
     that are not checkboxes, add the ones that were missed.
  4. Set "reviewed": true.
  5. python eval/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DetectionSettings
from app.detection import classical

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
ANNOTATIONS_DIR = Path(__file__).resolve().parent / "annotations"
REVIEW_DIR = Path(__file__).resolve().parent / "review"

_CHECKED_COLOUR = (60, 160, 60)
_UNCHECKED_COLOUR = (60, 60, 210)


def main() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    settings = DetectionSettings.from_env()

    images = sorted(SAMPLES_DIR.glob("*.png"))
    if not images:
        print(f"No sample images in {SAMPLES_DIR}", file=sys.stderr)
        raise SystemExit(1)

    for image_path in images:
        annotation_path = ANNOTATIONS_DIR / f"{image_path.stem}.json"
        if annotation_path.exists():
            existing = json.loads(annotation_path.read_text())
            if existing.get("reviewed"):
                print(f"{image_path.stem}: already reviewed, leaving untouched")
                continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        detections = classical.detect(image, settings)

        annotation_path.write_text(
            json.dumps(
                {
                    "source_image": image_path.name,
                    "reviewed": False,
                    "note": (
                        "Draft seeded from detector output. Correct it by hand "
                        "against review/, then set reviewed to true."
                    ),
                    "boxes": [
                        {
                            "index": index,
                            "bbox": detection.bbox.as_list(),
                            "is_checked": detection.is_checked,
                        }
                        for index, detection in enumerate(detections)
                    ],
                },
                indent=2,
            )
            + "\n"
        )

        _write_review_sheet(image, detections, REVIEW_DIR / f"{image_path.stem}_review.png")
        print(f"{image_path.stem}: seeded {len(detections)} boxes (reviewed=false)")

    print(
        "\nDrafts written. They are not ground truth until you correct them and "
        'set "reviewed": true.'
    )


def _write_review_sheet(image, detections, destination: Path) -> None:
    """Render every detection with its index so corrections are easy to key in."""
    canvas = image.copy()
    thickness = max(2, int(canvas.shape[1] / 800))
    font_scale = max(0.4, canvas.shape[1] / 3000)

    for index, detection in enumerate(detections):
        colour = _CHECKED_COLOUR if detection.is_checked else _UNCHECKED_COLOUR
        box = detection.bbox
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), colour, thickness)
        cv2.putText(
            canvas,
            str(index),
            (box.x1, max(12, box.y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    cv2.imwrite(str(destination), canvas)


if __name__ == "__main__":
    main()
