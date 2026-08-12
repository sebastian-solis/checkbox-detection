"""Dev helper: run the detector over the sample images and dump annotations."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DetectionSettings
from app.detection import classical

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
OUT = Path(__file__).resolve().parents[1] / "build" / "preview"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = DetectionSettings.from_env()
    for path in sorted(SAMPLES.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        boxes = classical.detect(image, settings)
        checked = sum(1 for b in boxes if b.is_checked)
        print(f"{path.name}: {len(boxes)} boxes, {checked} checked, {len(boxes)-checked} unchecked")
        canvas = image.copy()
        for b in boxes:
            colour = (0, 170, 0) if b.is_checked else (0, 0, 220)
            cv2.rectangle(canvas, (b.bbox.x1, b.bbox.y1), (b.bbox.x2, b.bbox.y2), colour, 3)
        cv2.imwrite(str(OUT / f"{path.stem}_annotated.png"), canvas)


if __name__ == "__main__":
    main()
