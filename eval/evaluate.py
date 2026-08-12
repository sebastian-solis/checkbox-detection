"""Measure detection quality against hand-checked annotations.

The point of this harness is that "it looks right" is not a result. Tuning a
heuristic by eye on four images is how you overfit to four images, so every
threshold change in app/config.py gets scored here before it is kept.

A detection counts as a true positive when it overlaps a ground-truth box by at
least ``iou_threshold``. Classification accuracy is measured only over matched
boxes, because scoring is_checked on a box that was never found conflates two
different failures.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --iou 0.4 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DetectionSettings
from app.detection import classical
from app.detection.types import BoundingBox, Checkbox

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
ANNOTATIONS_DIR = Path(__file__).resolve().parent / "annotations"


@dataclass
class DocumentScore:
    """Per-document counts, kept raw so totals can be pooled correctly."""

    name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    correct_classifications: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def classification_accuracy(self) -> float:
        return (
            self.correct_classifications / self.true_positives
            if self.true_positives
            else 0.0
        )


class UnreviewedAnnotations(Exception):
    """Raised when ground truth is still the detector's own unchecked output."""


def load_annotations(path: Path) -> list[tuple[BoundingBox, bool]]:
    """Read a ground-truth file: a list of {bbox, is_checked} objects.

    Annotation files are bootstrapped from the detector's own output to save
    typing coordinates by hand, which makes them worthless as ground truth
    until a human has corrected them. Scoring a detector against its own
    predictions returns a perfect score and tells you nothing, so files still
    carrying ``"reviewed": false`` are refused outright rather than quietly
    inflating the report.
    """
    payload = json.loads(path.read_text())

    if not payload.get("reviewed", False):
        raise UnreviewedAnnotations(
            f"{path.name} is still marked reviewed=false. Correct the boxes by "
            'hand against the overlay, then set "reviewed": true.'
        )

    return [
        (BoundingBox(*entry["bbox"]), bool(entry["is_checked"]))
        for entry in payload["boxes"]
    ]


def score_document(
    truth: list[tuple[BoundingBox, bool]],
    predictions: list[Checkbox],
    iou_threshold: float,
) -> DocumentScore:
    """Greedily match predictions to ground truth by best overlap.

    Greedy matching is enough here because checkboxes never overlap each other
    on a form, so there is no ambiguous assignment to resolve.
    """
    unmatched = list(predictions)
    true_positives = 0
    correct_classifications = 0
    false_negatives = 0

    for truth_box, truth_checked in truth:
        best_index = -1
        best_iou = iou_threshold

        for index, prediction in enumerate(unmatched):
            overlap = truth_box.intersection_over_union(prediction.bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index

        if best_index >= 0:
            matched = unmatched.pop(best_index)
            true_positives += 1
            if matched.is_checked == truth_checked:
                correct_classifications += 1
        else:
            false_negatives += 1

    return DocumentScore(
        name="",
        true_positives=true_positives,
        false_positives=len(unmatched),
        false_negatives=false_negatives,
        correct_classifications=correct_classifications,
    )


def run(iou_threshold: float, verbose: bool) -> int:
    settings = DetectionSettings.from_env()
    annotation_files = sorted(ANNOTATIONS_DIR.glob("*.json"))

    if not annotation_files:
        print(
            f"No annotations found in {ANNOTATIONS_DIR}.\n"
            "Bootstrap them with: python eval/bootstrap_annotations.py",
            file=sys.stderr,
        )
        return 1

    scores: list[DocumentScore] = []
    skipped: list[str] = []
    for annotation_path in annotation_files:
        image_path = SAMPLES_DIR / f"{annotation_path.stem}.png"
        if not image_path.exists():
            print(f"skipping {annotation_path.name}: no matching image", file=sys.stderr)
            continue

        try:
            truth = load_annotations(annotation_path)
        except UnreviewedAnnotations as exc:
            # Skipping loudly rather than failing the whole run: a partially
            # reviewed corpus still produces an honest number over the part that
            # was reviewed, as long as the report says which documents it covers.
            skipped.append(annotation_path.stem)
            print(f"skipped, not reviewed: {exc}", file=sys.stderr)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        predictions = classical.detect(image, settings)

        score = score_document(truth, predictions, iou_threshold)
        score.name = annotation_path.stem
        scores.append(score)

        if verbose:
            print(
                f"  {score.name}: truth={len(truth)} predicted={len(predictions)} "
                f"tp={score.true_positives} fp={score.false_positives} "
                f"fn={score.false_negatives}"
            )

    if not scores:
        print(
            "\nNothing scored: no annotation file has been reviewed yet.\n"
            'Correct the drafts by hand and set "reviewed": true.',
            file=sys.stderr,
        )
        return 1

    _report(scores, iou_threshold)

    if skipped:
        print(
            f"\nCovers {len(scores)} of {len(scores) + len(skipped)} documents. "
            f"Not yet reviewed, so excluded: {', '.join(skipped)}."
        )
    return 0


def _report(scores: list[DocumentScore], iou_threshold: float) -> None:
    print(f"\nDetection quality at IoU >= {iou_threshold}\n")
    header = f"{'document':<38} {'prec':>6} {'recall':>7} {'f1':>6} {'class acc':>10}"
    print(header)
    print("-" * len(header))

    for score in scores:
        print(
            f"{score.name:<38} {score.precision:>6.3f} {score.recall:>7.3f} "
            f"{score.f1:>6.3f} {score.classification_accuracy:>10.3f}"
        )

    # Pooled over all boxes rather than averaged over documents: a page with 130
    # checkboxes should not carry the same weight as one with 40.
    pooled = DocumentScore(
        name="TOTAL",
        true_positives=sum(s.true_positives for s in scores),
        false_positives=sum(s.false_positives for s in scores),
        false_negatives=sum(s.false_negatives for s in scores),
        correct_classifications=sum(s.correct_classifications for s in scores),
    )
    print("-" * len(header))
    print(
        f"{pooled.name:<38} {pooled.precision:>6.3f} {pooled.recall:>7.3f} "
        f"{pooled.f1:>6.3f} {pooled.classification_accuracy:>10.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Minimum overlap for a detection to count as a match (default 0.5).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-document counts.")
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.iou, arguments.verbose))


if __name__ == "__main__":
    main()
