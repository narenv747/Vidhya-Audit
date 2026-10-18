"""Turn a scanned answer-script PDF into data/{SCRIPT_ID}/page_NN.jpg.
Usage (from backend/):  python ingest_pdf.py ../raw/Math_Basic.pdf MATH_001 [--cover-first]
pip install pymupdf.  --cover-first treats PDF page 1 as the cover sheet (page_00); otherwise pages start at page_01."""
import sys
from pathlib import Path
import fitz  # PyMuPDF

pdf, sid = sys.argv[1], sys.argv[2]
start = 0 if "--cover-first" in sys.argv else 1
out = Path(__file__).resolve().parent.parent / "data" / sid
out.mkdir(parents=True, exist_ok=True)
doc = fitz.open(pdf)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=110).save(out / f"page_{start + i:02d}.jpg")
print(f"{len(doc)} pages -> {out}")
