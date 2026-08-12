"""Rasterise PDF pages so the detector can read them.

The challenge asks for images, and `POST /detect` takes images and nothing else.
But a mortgage loan file is a PDF, so a service that only accepts PNGs pushes
the rasterisation problem onto every caller. This module exists so the API can
take the format the documents actually arrive in.

On the choice of library: PyMuPDF is the usual reach and it is AGPL, which is a
licence a commercial product cannot absorb without consequences. pypdfium2
wraps Google's PDFium under BSD-3-Clause and Apache-2.0, needs no system
package, and renders these forms just as well.
"""

from __future__ import annotations

import cv2
import numpy as np
import pypdfium2 as pdfium

from app.config import PdfSettings


class PdfRejected(Exception):
    """Raised when a PDF cannot be used. The message is safe to return."""


def rasterise(payload: bytes, settings: PdfSettings) -> list[np.ndarray]:
    """Render every page of a PDF to a BGR image array.

    Pages are rendered at a fixed DPI rather than at their native size because
    the detector's thresholds are ratios of page width, so what matters is that
    a checkbox occupies a sensible number of pixels, not that the raster matches
    any particular scan resolution.
    """
    try:
        document = pdfium.PdfDocument(payload)
    except Exception as exc:  # pdfium raises its own error types
        raise PdfRejected("File could not be opened as a PDF.") from exc

    try:
        page_count = len(document)
        if page_count == 0:
            raise PdfRejected("PDF contains no pages.")
        if page_count > settings.max_pages:
            raise PdfRejected(
                f"PDF has {page_count} pages, over the {settings.max_pages} page limit."
            )

        scale = settings.render_dpi / 72.0  # PDF user space is 72 units per inch
        pages: list[np.ndarray] = []

        for index in range(page_count):
            page = document[index]
            try:
                bitmap = page.render(scale=scale)
                rgb = bitmap.to_numpy()
            finally:
                page.close()

            # to_numpy gives RGB or RGBA; OpenCV works in BGR.
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
            elif rgb.ndim == 3:
                pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            else:
                pages.append(cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR))

        return pages
    finally:
        # Release the native handles rather than waiting for the garbage
        # collector: this runs per request on documents holding personal data.
        document.close()


def looks_like_pdf(payload: bytes) -> bool:
    """PDFs start with %PDF- followed by a version."""
    return payload[:5] == b"%PDF-"
