# Checkbox Detection

Detects and classifies checkboxes in scanned document images. Built for the
HomeVision backend take-home challenge.

Given a document image, the service returns the pixel coordinates of every
checkbox it finds and whether each one is marked.

---

## Quick start

### With Docker (recommended)

```bash
docker compose up --build
```

Then open <http://localhost:8000>.

### Without Docker

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000>.

Verify it is up:

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","version":"1.0.0"}
```

---

## The API

Interactive documentation is served at <http://localhost:8000/docs>.

### `POST /detect`

The endpoint specified by the challenge.

**Request:** an image file, sent as `multipart/form-data` under the field `file`.
Accepted formats: PNG, JPEG, TIFF, BMP, WEBP.

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@samples/uniform_residential_appraisal.png"
```

**Response:**

```json
{
  "boxes": [
    { "bbox": [312, 2757, 379, 2807], "is_checked": true },
    { "bbox": [1124, 2206, 1192, 2256], "is_checked": false }
  ]
}
```

`bbox` holds `[x1, y1, x2, y2]`: the top-left and bottom-right corners in pixels.

### `POST /detect/debug`

The same detection plus the numbers behind each verdict: `confidence`,
`ink_ratio`, image dimensions and elapsed time. Useful for tuning thresholds and
for understanding a borderline call.

```bash
curl -X POST http://localhost:8000/detect/debug \
  -F "file=@samples/neighborhood_site_section.png"
```

### `POST /detect/visualize`

Returns the document as a PNG with every detection outlined: green for marked,
red for unmarked.

```bash
curl -X POST http://localhost:8000/detect/visualize \
  -F "file=@samples/uniform_residential_appraisal.png" \
  -o annotated.png
```

### `GET /health`

Liveness probe. Returns status and version.

### `GET /`

A browser UI: drop in a document, see the overlay, read the per-box
measurements.

---

## Errors

Validation failures return `400` with a reason and a request id. Error bodies
never echo the uploaded content.

```json
{
  "detail": "File content is not a supported image format (PNG, JPEG, TIFF, BMP or WEBP).",
  "request_id": "bcd398f4-8582-4fb7-8837-8c0d09020a46"
}
```

| Condition | Status | Message |
|---|---|---|
| No `file` field | 422 | FastAPI validation error |
| Empty file | 400 | `Empty upload.` |
| Content type not allowed | 400 | `Unsupported content type: application/pdf.` |
| Bytes are not an image | 400 | `File content is not a supported image format ...` |
| Over the size limit | 400 | `File exceeds the 20.0 MB limit.` |
| Over the pixel limit | 400 | `Image resolution ... exceeds the ... pixel limit.` |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

36 tests covering the geometry primitives, the detector against synthetic pages
with known answers, the HTTP contract, and every upload rejection path.

```
36 passed in 0.76s
```

---

## Measuring detection quality

"It looks right" is not a result, so quality is scored rather than eyeballed.

```bash
# 1. Seed annotation drafts from the current detector output
python eval/bootstrap_annotations.py

# 2. Correct them by hand against eval/review/<name>_review.png,
#    then set "reviewed": true in each eval/annotations/<name>.json

# 3. Score
python eval/evaluate.py --verbose
```

Reported per document and pooled across all of them: precision, recall, F1 at a
configurable IoU threshold, plus classification accuracy over matched boxes.
Documents that have not been reviewed are named in the output and excluded from
the score rather than silently averaged in.

Current result over the three documents reviewed, 240 hand-checked checkboxes at
IoU >= 0.5:

```
document                                 prec  recall     f1  class acc
manufactured_home_appraisal             0.987   0.987  0.987      1.000
neighborhood_site_section               0.955   0.977  0.966      0.976
uniform_residential_appraisal           0.944   1.000  0.971      1.000
TOTAL                                   0.960   0.992  0.975      0.996
```

The fourth document, `market_conditions_addendum`, is deliberately left
unreviewed: its trend selectors are table cells rather than drawn checkboxes, so
an unmarked selector and a data cell are the same rectangle. `WRITEUP.md`
explains why that needs a product decision rather than an annotation pass.

The harness **never scores annotations still marked `reviewed: false`**. Drafts
are seeded from the detector's own output, so scoring against them unreviewed
would return a perfect result and mean nothing. Unreviewed documents are skipped
and named; if none have been reviewed, the run exits non-zero rather than
printing a number.

---

## Configuration

Every threshold is an environment variable; defaults live in `app/config.py`.
Size bounds are fractions of the image width, so they hold across scan
resolutions rather than assuming 300 DPI.

| Variable | Default | Purpose |
|---|---|---|
| `MIN_BOX_WIDTH_RATIO` | `0.009` | Smallest checkbox, as a fraction of page width |
| `MAX_BOX_WIDTH_RATIO` | `0.028` | Largest checkbox, same units |
| `MIN_ASPECT_RATIO` | `0.65` | Reject shapes too tall to be a checkbox |
| `MAX_ASPECT_RATIO` | `1.55` | Reject shapes too wide to be a checkbox |
| `MIN_EXTENT` | `0.72` | Reject ragged shapes such as letter loops |
| `MIN_SURROUND_LUMA` | `120` | Reject light shapes cut out of a dark background |
| `MAX_STROKE_SATURATION` | `90` | Reject shapes traced in colour, such as a watermark |
| `STROKE_LUMA_CEILING` | `200` | Below this counts as a stroke when judging colour |
| `SURROUND_MARGIN_RATIO` | `0.35` | Width of the ring sampled around a candidate |
| `CHECKED_INK_THRESHOLD` | `0.14` | Ink fraction above which a box counts as marked |
| `INTERIOR_INSET_RATIO` | `0.22` | Border margin excluded before measuring ink |
| `DESKEW_ENABLED` | `true` | Straighten the page before extracting lines |
| `MAX_FILE_BYTES` | `20971520` | Upload size ceiling |
| `MAX_PIXELS` | `50000000` | Decoded resolution ceiling |

---

## Handling of document contents

The sample documents are real mortgage appraisals: they carry borrower names,
property addresses and lender details. That shaped three decisions.

- **Nothing is written to disk.** Uploads are decoded in memory and released
  when the request ends. There is no spool directory to leak, back up, or
  forget to purge. The container runs read-only with a tmpfs `/tmp`.
- **Nothing derived from the document body is logged.** Logs carry a request
  id, image dimensions, detection count and timing. Never content, never
  filenames.
- **Uploads are validated on their bytes, not their labels.** Content type and
  filename are attacker-controlled, so format is decided by magic bytes, and
  both file size and decoded resolution are capped before any work happens.

`WRITEUP.md` covers what production would additionally require.

---

## Project layout

```
app/
  main.py              FastAPI application and endpoints
  config.py            All tunable thresholds, environment-backed
  schemas.py           Response models; /detect's shape is the fixed contract
  security.py          Upload validation and decoding
  detection/
    types.py           BoundingBox and Checkbox value objects
    preprocess.py      Grayscale, binarization, deskew, line extraction
    classical.py       Detection and filled/unfilled classification
    render.py          Overlay drawing
static/index.html      Browser UI
eval/
  bootstrap_annotations.py  Seed annotation drafts from detector output
  contact_sheet.py          Render detections as labelled crops for review
  evaluate.py               Precision, recall, F1 and classification accuracy
tests/                 Unit and integration tests
samples/               The four challenge documents
tools/preview.py       Dev helper: run over all samples, dump overlays
```
