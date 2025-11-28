#!/usr/bin/env python
"""
Simple PDF text extractor CLI.

Usage:
  python tools/extract_pdf_text.py <input.pdf> [-o OUTPUT.txt] [--start N] [--end M]

Tries multiple backends in order:
  1) pypdf (preferred)
  2) PyPDF2
  3) pdfminer.six

Notes:
  - If the PDF is scanned (image-based), these libraries will return little or no text.
    In that case, OCR (e.g., Tesseract) is required, which is not included here.
  - Page numbers are 1-based and inclusive.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List


def _extract_with_pypdf(path: str, start: Optional[int], end: Optional[int]) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pypdf not available") from e

    reader = PdfReader(path)
    n = len(reader.pages)
    s = 1 if start is None else max(1, min(start, n))
    eidx = n if end is None else max(1, min(end, n))

    out_lines: List[str] = []
    for i in range(s - 1, eidx):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out_lines.append(f"\n=== Page {i+1} ===\n")
        out_lines.append(text)
    return "\n".join(out_lines)


def _extract_with_pypdf2(path: str, start: Optional[int], end: Optional[int]) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyPDF2 not available") from e

    reader = PdfReader(path)
    n = len(reader.pages)
    s = 1 if start is None else max(1, min(start, n))
    eidx = n if end is None else max(1, min(end, n))

    out_lines: List[str] = []
    for i in range(s - 1, eidx):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out_lines.append(f"\n=== Page {i+1} ===\n")
        out_lines.append(text)
    return "\n".join(out_lines)


def _extract_with_pdfminer(path: str, start: Optional[int], end: Optional[int]) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pdfminer.six not available") from e

    # pdfminer uses 1-based page numbers via page_numbers=[...]-1, so build zero-based list
    # We stream whole range to keep memory modest
    try:
        from pdfminer.pdfpage import PDFPage  # type: ignore
    except Exception:
        # Fallback: let extract_text handle the full doc (no page selection)
        return extract_text(path)

    with open(path, "rb") as f:
        pages = list(PDFPage.get_pages(f))
    n = len(pages)
    s = 1 if start is None else max(1, min(start, n))
    eidx = n if end is None else max(1, min(end, n))

    # pdfminer’s extract_text supports page_numbers as zero-based indices
    page_numbers = list(range(s - 1, eidx))
    return extract_text(path, page_numbers=page_numbers)


def extract_pdf_text(path: str, start: Optional[int], end: Optional[int]) -> str:
    # Try pypdf
    try:
        return _extract_with_pypdf(path, start, end)
    except ImportError:
        pass
    except Exception:
        # fall through to other backends
        pass

    # Try PyPDF2
    try:
        return _extract_with_pypdf2(path, start, end)
    except ImportError:
        pass
    except Exception:
        pass

    # Try pdfminer.six
    try:
        return _extract_with_pdfminer(path, start, end)
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF backend available. Install one of: pypdf, PyPDF2, or pdfminer.six"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract text from a PDF")
    ap.add_argument("input", help="Path to input PDF")
    ap.add_argument("-o", "--output", help="Path to output text file")
    ap.add_argument("--start", type=int, default=None, help="Start page (1-based, inclusive)")
    ap.add_argument("--end", type=int, default=None, help="End page (1-based, inclusive)")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.isfile(in_path):
        print(f"Input PDF not found: {in_path}", file=sys.stderr)
        return 2

    try:
        text = extract_pdf_text(in_path, args.start, args.end)
    except Exception as e:
        print(f"Failed to extract text: {e}", file=sys.stderr)
        return 1

    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + ".txt"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

