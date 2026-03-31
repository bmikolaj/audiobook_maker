"""
strip_header_footer.py



Tested only with The Atlantic articles so far



Removes from each page of a browser-printed Atlantic article PDF:
  - Repeating header  ("URL", top ~15pt)
  - Repeating footer  ("X of 17 / timestamp", bottom ~15pt)
  - UI overlay images (bookmark / share icons — tiny 1-bit DeviceGray images)
  - UI overlay vector shapes (the rounded-rectangle popup widget)
  - "View this story as PDF" box thumbnail (small RGB image in lower page zone)
  - Article illustrations (large RGB images in the page body) are KEPT

Usage:
    python strip_header_footer.py input.pdf output.pdf

Tune the constants below if adapting to a different PDF.
"""

import sys
from pathlib import Path

import fitz   # PyMuPDF


# ── Crop settings ──────────────────────────────────────────────────────────────
HEADER_CROP = 15   # points removed from top  (header sits in top ~12pt)
FOOTER_CROP = 15   # points removed from bottom (footer sits in bottom ~12pt)

# ── UI overlay detection ───────────────────────────────────────────────────────
# Any image whose rendered rect starts below this y-value is treated as a UI
# overlay unless it is a large content image (see MAX_UI_IMAGE_WIDTH below).
UI_OVERLAY_Y_THRESHOLD = 650

# RGB images wider than this (in points, on the page) are considered article
# illustrations and are left untouched even if they dip into the overlay zone.
MAX_UI_IMAGE_WIDTH = 300


def is_ui_image(img_info: tuple, page: fitz.Page) -> bool:
    """Return True if this image should be treated as a UI widget to remove."""
    xref   = img_info[0]
    px_w   = img_info[2]   # pixel width
    px_h   = img_info[3]   # pixel height
    bpc    = img_info[4]   # bits per component
    cs     = img_info[5]   # colour space name

    # 1-bit DeviceGray images are always UI icons (bookmark, share glyphs).
    if cs == "DeviceGray" and bpc == 1:
        return True

    # Small RGB images that live entirely in the lower overlay zone are also UI
    # (e.g. the "View this story as PDF" magazine-cover thumbnail).
    rects = page.get_image_rects(xref)
    if rects and cs == "DeviceRGB":
        rendered_w = rects[0].width
        y0         = rects[0].y0
        if rendered_w <= MAX_UI_IMAGE_WIDTH and y0 >= UI_OVERLAY_Y_THRESHOLD:
            return True

    return False


def remove_ui_overlays(page: fitz.Page) -> None:
    """
    Redact UI images and the vector-shape popup widget, preserving all text.

    Strategy
    --------
    • For each UI image: add a redact annotation covering every rect where it
      is rendered, then apply with IMAGE_REMOVE and TEXT_NONE.
    • For the vector popup shapes (rounded rectangles in the overlay zone):
      add a single broad redact annotation spanning the zone and apply with
      LINE_ART_REMOVE_IF_COVERED and TEXT_NONE.
    """
    redact_rects: list[fitz.Rect] = []

    # ── Collect UI image rects ────────────────────────────────────────────────
    for img_info in page.get_images(full=True):
        if is_ui_image(img_info, page):
            xref = img_info[0]
            for r in page.get_image_rects(xref):
                redact_rects.append(r)

    # ── Collect vector-shape overlay rect ─────────────────────────────────────
    # The popup widget is made of filled rectangles in the lower page zone.
    # One broad annotation covering the zone is enough to wipe them out.
    drawings = page.get_drawings()
    for d in drawings:
        dr = d.get("rect")
        if dr and dr.y0 >= UI_OVERLAY_Y_THRESHOLD:
            redact_rects.append(dr)

    if not redact_rects:
        return

    # ── Apply redactions ──────────────────────────────────────────────────────
    # Use white fill so that vector shapes are "covered" by the annotation.
    # TEXT_NONE preserves any text that happens to sit under the overlay.
    # REMOVE_IF_TOUCHED removes any line art that the redact rect touches,
    # which is necessary for shapes that slightly extend outside our rects.
    for r in redact_rects:
        page.add_redact_annot(r, fill=None)   # transparent — no white box over text

    page.apply_redactions(
        images   = fitz.PDF_REDACT_IMAGE_REMOVE,
        graphics = fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
        text     = fitz.PDF_REDACT_TEXT_NONE,   # ← leaves text untouched
    )


def strip_pdf(input_path: str | Path, output_path: str | Path) -> None:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    doc = fitz.open(str(input_path))

    for page in doc:
        # 1. Remove UI overlay graphics (images + vector shapes).
        remove_ui_overlays(page)

        # 2. Crop out the header and footer bands.
        rect = page.rect
        crop = fitz.Rect(
            rect.x0,
            rect.y0 + HEADER_CROP,
            rect.x1,
            rect.y1 - FOOTER_CROP,
        )
        page.set_cropbox(crop)

    # fitz cannot save over a file it still has open, so always write to a
    # temporary file first, then move it into place.  This also makes same-path
    # calls (input_path == output_path) safe.
    tmp = output_path.with_suffix(".tmp.pdf")
    try:
        doc.save(str(tmp), garbage=4, deflate=True)
        doc.close()
        tmp.replace(output_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python strip_header_footer.py input.pdf output.pdf")
        sys.exit(1)
    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    strip_pdf(input_path, output_path)
    print(f"Saved → {output_path}")
    print(f"  Header/footer cropped: {HEADER_CROP}pt top, {FOOTER_CROP}pt bottom")
    print(f"  UI overlays removed (images + vector shapes below y={UI_OVERLAY_Y_THRESHOLD})")