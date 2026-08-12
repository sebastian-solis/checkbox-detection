# Writeup: approach, trade-offs, limitations

## The problem, restated

Find every checkbox on a scanned document image and say whether it is marked.
The documents are US residential appraisal forms: dense ruled tables, printed
body text at several sizes, scan skew, JPEG artifacts, and in one case a
diagonal watermark across the page. A single page carries somewhere between
forty and a hundred and thirty checkboxes.

## Approach: classical computer vision

The pipeline is deterministic OpenCV:

1. **Grayscale and deskew.** Line extraction assumes rules are axis-aligned, so
   a two degree scan skew quietly destroys recall. The rotation estimate is
   discarded if it exceeds five degrees, on the grounds that a larger estimate
   is more likely to be wrong than the page is to be that crooked.
2. **Adaptive binarization.** Ink becomes white, paper black.
3. **Morphological line extraction.** Opening with a long thin horizontal
   kernel keeps horizontal rules and deletes text; the same in the vertical
   direction. A checkbox border is structurally a short horizontal run meeting
   a short vertical one, so combining the two masks leaves the borders standing
   on a page that is otherwise mostly glyphs.
4. **Contour filtering.** Candidates are kept on size, aspect ratio, extent and
   vertex count. Sizes are expressed as fractions of page width rather than in
   pixels, so the same thresholds hold at 200 or 600 DPI.
5. **Overlap suppression.** Thick borders and the two line passes can each
   produce a contour for the same box; the largest of an overlapping cluster
   wins.
6. **Classification.** The border is inset by 22% and the ink fraction of the
   remaining interior is measured. Above `CHECKED_INK_THRESHOLD` the box counts
   as marked.

### Why not a learned model

The challenge ships four images. That is enough to sanity-check a heuristic and
nowhere near enough to train a detector, and, more importantly, nowhere near
enough to validate one honestly. Fine-tuning something like YOLO on four pages
produces a model that has memorised four pages.

The classical pipeline also buys three properties that matter in this domain:
it runs in tens of milliseconds on CPU with no GPU and no model artifact; every
decision reduces to a measurable quantity, which is the difference between "the
model said so" and "the interior was 31% ink against a 14% threshold"; and it
has no training data to govern, which in a regulated setting is one fewer thing
to audit.

If HomeVision already has labelled appraisal pages at volume, that calculus
changes and a learned detector is the better answer. This is a decision made
under the constraints of the exercise, not a claim that heuristics beat models.

### The most instructive bug

The first working revision reported over a thousand boxes per page. It was
matching the closed bowls of the letters `o`, `e`, `a` and `d`.

The cause was an inverted filter. I had reasoned that a hollow square traced by
its border would enclose little area and score a low extent, so I rejected high
extent shapes. In fact `cv2.contourArea` on an external contour returns the area
of the polygon it encloses, so a square border scores near 1.0 and was being
rejected, while ragged letter loops scored lower and sailed through. Every real
checkbox was being discarded and every letter kept.

Flipping it to a *minimum* extent turned the single loosest filter into the
strongest one. Detections on the densest page went from 1563 to 129, and text
false positives went to zero. `tests/test_detection.py::test_ignores_body_text`
is the regression test for it.

## Measurement, and what is not yet measured

`eval/evaluate.py` scores precision, recall and F1 at a configurable IoU
threshold, plus classification accuracy over matched boxes, per document and
pooled.

Annotation drafts are seeded from the detector's own output because typing a
hundred bounding boxes per page by hand is a poor use of time. That makes them
worthless as ground truth until corrected, so every file is written with
`"reviewed": false` and **the harness never scores a file until a human flips
that flag**. Scoring a detector against its own predictions returns a perfect
result and tells you nothing; the guard exists so that number can never
accidentally be reported. Unreviewed documents are skipped and named in the
output, so a partial corpus still yields an honest figure over the part that was
actually checked.

### Results

Three of the four documents have been reviewed to completion. Correcting each
meant inspecting every detection as an enlarged crop (`eval/contact_sheet.py`
renders them as a labelled grid), removing what was not a checkbox, fixing wrong
classifications, and hunting for boxes that were missed.

At IoU >= 0.5, over 240 hand-checked checkboxes:

| document | precision | recall | F1 | classification |
|---|---|---|---|---|
| manufactured_home_appraisal | 0.987 | 0.987 | 0.987 | 1.000 |
| uniform_residential_appraisal | 0.944 | 1.000 | 0.971 | 1.000 |
| neighborhood_site_section | 0.955 | 0.977 | 0.966 | 0.976 |
| **pooled** | **0.960** | **0.992** | **0.975** | **0.996** |

The fourth document is excluded for the reason set out below, and the harness
names it in its output rather than quietly averaging over what it has.

### Finding the misses without scanning by eye

Recall is the hard thing to measure honestly, because a missed checkbox leaves
nothing on screen to notice. Reading a four-thousand-pixel page looking for
absences is slow and unreliable.

Instead: run the detector a second time with deliberately loose thresholds, then
subtract everything the strict run already found. Whatever is left is the set of
shapes the strict pass rejected, and that is a short list worth looking at. On
`uniform_residential_appraisal` the loose pass produced 209 candidates against
125, leaving 84 to inspect, and every one turned out to be a text fragment or a
table-cell corner. That is what a recall of 1.000 on that page rests on, rather
than a claim that I looked carefully.

The same technique found nothing on the other pages either, which is why the one
false negative on record, on `neighborhood_site_section`, was caught the slow way
before this existed.

### What measuring bought, twice

The first score put precision at 0.875 against a recall of 0.977, which located
the problem precisely: the detector was not missing checkboxes, it was inventing
them. Four of the six false positives were letters of the vertical
`NEIGHBORHOOD` label, set in white type on a solid black sidebar. The counters of
`O`, `D` and `B` are small light squares fully enclosed by dark, geometrically
indistinguishable from an empty checkbox.

Polarity separates them: real ink sits on bright paper, a letter counter is
surrounded by the dark body of its glyph. That check took precision from 0.875
to 0.955 with **no loss of recall**, and removed four more false positives on
`uniform_residential_appraisal`, which carries the same style of sidebar.

Reviewing the second document exposed a second systematic source: a red diagonal
watermark whose ring encloses a pale square that passes every geometric test.
Print ink on these forms is black and effectively unsaturated, so mean
saturation over the strokes separates them. Precision on that page went from
0.963 to 0.987, again with recall untouched.

Both fixes came from a number telling me where to look. Two earlier attempts to
chase a single missed checkbox by eye either did nothing or made things actively
worse, and I only knew because I was counting.

### An open question about ground truth

Reviewing the second document surfaced something worth raising rather than
guessing at. On the Market Conditions Addendum the trend selectors are not
drawn as checkboxes at all: they are table cells you mark with an X. An
unmarked trend cell and an ordinary data cell are the same rectangle with the
same border, and nothing in the pixels separates them. Detections there cluster
into three exact sizes, 53x45, 37x50 and 25x27, which is the grid repeating
rather than a detector wobbling.

So "how many checkboxes are on this page" has no answer from the image alone. It
depends on which columns are meant to be marked, which is form semantics.

Three ways to resolve it, and I would want HomeVision's view before picking one:
detect the ruled grid and treat only cells in known selector columns as
checkboxes, which needs a per-form template; return every candidate cell and let
the consumer filter by position; or accept the ambiguity and report unmarked
cells with low confidence, which is closest to what the service does now.

This is exactly the kind of question I would rather ask than answer unilaterally,
because the right choice depends on how MIRA consumes the output downstream.

## Known limitations

- **Some checkboxes are missed.** On the manufactured home appraisal, the
  marked box beside "Other (describe)" on the Assignment Type row is not
  detected. Its border appears to merge with the surrounding table rule, so no
  closed contour survives filtering.
- **Wide near-square table cells can pass.** The aspect ratio window runs to
  1.55, so a squat cell is admissible on geometry alone. Tightening it trades
  precision against recall on genuinely rectangular checkboxes, and without
  reviewed ground truth I have no basis to pick a better point.
- **One ink threshold for all mark styles.** A single `X` and a fully shaded box
  produce very different ink fractions. The threshold is set low enough to catch
  light marks, which makes a heavily smudged empty box a plausible false
  positive.
- **The confidence score is a heuristic, not a probability.** It reports
  distance from the decision boundary, rescaled to 0.5 to 1.0. It is useful for
  ranking borderline cases and should not be read as a calibrated likelihood.
- **Single page, single image.** Multi-page PDFs are out of scope; the caller
  rasterises and submits pages individually.
- **Deskew is global.** A page with different skew top and bottom, as happens
  with a curled scan, is not corrected per region.

## What production would need

Marked in the code as `TODO(production)` where relevant.

- **Pin the base image by digest.** A tag can be repointed upstream, so the
  image that passed review is not guaranteed to be the image that ships.
- **Authentication and rate limiting.** The service is currently unauthenticated.
  Detection is CPU-bound and an unauthenticated caller can trivially saturate it.
- **Async processing for large batches.** A synchronous request per page is fine
  interactively and wrong for a thousand-page loan file. A queue with a webhook
  on completion fits both the workload and the existing SQS and Temporal stack.
- **Metrics and tracing.** Detection counts, latency percentiles and rejection
  reasons to Datadog, which is already in use here.
- **A labelled corpus and a regression gate.** The single highest-value
  investment: a few hundred annotated pages spanning form types and scan
  qualities, with the evaluation harness wired into CI so a threshold change
  that improves one document and quietly ruins another cannot merge.
- **Formal data handling controls.** In-memory processing and content-free
  logging are implemented. Production would add TLS termination, encryption of
  any derived artifact at rest, a documented retention position, and an access
  audit trail over who invoked detection on which document.

## A note on tools

I used AI assistance throughout, which I understand is how the team works day
to day. It was most useful for scaffolding the FastAPI surface and the test
matrix, and least useful for the detection tuning, where the answer came from
rendering overlays and looking at what was actually being matched. The inverted
extent filter described above is a good illustration: no amount of plausible
reasoning found it, and one annotated crop made it obvious in seconds.
