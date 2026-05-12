"""Split a PDF into one file per chapter using its Table of Contents."""

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:150] or "chapter"


def split_pdf_by_chapters(
                            input_path: Path,
                            output_dir: Path | None = None,
                            toc_level: int = 1,
                        ) -> list[Path]:

    doc = fitz.open(input_path)
    toc = doc.get_toc()
    if not toc:
        raise ValueError(f"No table of contents found in {input_path}")

    chapters = [(title, page - 1) for level, title, page in toc if level == toc_level]
    if not chapters:
        raise ValueError(
            f"No TOC entries at level {toc_level}; available levels: "
            f"{sorted({lvl for lvl, _, _ in toc})}"
        )

    output_dir = output_dir or input_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    page_count = doc.page_count
    written: list[Path] = []
    width = len(str(len(chapters)))

    for idx, (title, start) in enumerate(chapters):
        end = chapters[idx + 1][1] - 1 if idx + 1 < len(chapters) else page_count - 1
        if start > end:
            print(f"Skipping '{title}': start page {start + 1} > end page {end + 1}")
            continue

        chapter_doc = fitz.open()
        chapter_doc.insert_pdf(doc, from_page=start, to_page=end)

        filename = f"{str(idx + 1).zfill(width)} - {sanitize_filename(title)}.pdf"
        out_path = output_dir / filename
        chapter_doc.save(out_path)
        chapter_doc.close()
        written.append(out_path)
        print(f"Wrote {out_path.name} (pages {start + 1}-{end + 1})")

    doc.close()
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Directory for the split PDFs (defaults to a folder next to the input)",
    )
    parser.add_argument(
        "-l", "--level", type=int, default=1,
        help="TOC level to split on (1 = top-level chapters)",
    )
    args = parser.parse_args()
    split_pdf_by_chapters(args.input, args.output_dir, args.level)

