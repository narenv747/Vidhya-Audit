"""The audit pipeline: cover-sheet maths check + skipped-answer detection."""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vision import call_vision_json, MOCK
from redink import load_page, red_pixels

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE = ROOT / "data", ROOT / "audit_cache"
RED_MIN = int(os.getenv("RED_MIN_PIXELS", "30"))
IMG_EXT = (".jpg", ".jpeg", ".png")

COVER_PROMPT = """[COVER] This is the front cover of a handwritten Indian board-exam answer script. It has a marks grid.
Read every row of the grid. Return ONLY this JSON:
{"rows":[{"q":"1a","marks":4,"bbox":[x,y,w,h]}],"recorded_total":62,"total_bbox":[x,y,w,h]}
Rules: q is the question number as printed. marks is the number written by the examiner, or null if the cell is blank.
recorded_total is the grand total the examiner wrote, or null. Do NOT add up the marks yourself.
Every bbox is the cell's position as numbers between 0 and 1: x,y = top-left corner, w,h = size, as fractions of the page width and height."""

PAGE_PROMPT = """[PAGE] This is one page of a handwritten Indian board-exam answer script, checked by an examiner who uses red ink for ticks and marks.
Find every separate answer the student wrote on this page. Return ONLY this JSON:
{"regions":[{"q":"14b","bbox":[x,y,w,h],"has_student_writing":true,"has_evaluator_mark":false}]}
Rules: q is the question number the student wrote (or null if you cannot tell). has_evaluator_mark is true only if you see a tick,
cross, score or comment from the examiner on or beside that answer. Boxes are numbers between 0 and 1: x,y = top-left, w,h = size,
as fractions of page width and height. Skip blank areas and printed headers."""


# ---------- helpers ----------
def valid_id(script_id):
    """Script IDs become folder names, so only allow safe characters (blocks '..' and slashes)."""
    return isinstance(script_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", script_id) is not None


def page_files(script_id):
    """Returns {page_number: path} for data/{script_id}/page_NN.*"""
    folder = DATA / script_id
    if not folder.is_dir():
        raise FileNotFoundError(f"No folder for script {script_id}")
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXT and p.stem.startswith("page_"):
            try:
                out[int(p.stem.split("_")[1])] = p
            except ValueError:
                pass
    return out


def _valid(b):
    return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
            and 0 <= b[0] <= 1 and 0 <= b[1] <= 1 and b[2] > 0.02 and b[3] > 0.01
            and b[0] + b[2] <= 1.02 and b[1] + b[3] <= 1.02 and b[2] * b[3] < 0.95)


def _clean_bbox(b, i=0, n=1):
    """If the AI's box is sloppy, use a coarse horizontal band instead."""
    if _valid(b):
        return [round(float(v), 4) for v in b]
    h = 1.0 / max(n, 1)
    return [0.05, round(i * h, 4), 0.9, round(h, 4)]


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------- the two checks ----------
def audit_cover(cover_path):
    d = call_vision_json(cover_path, COVER_PROMPT)
    rows, n = [], len(d.get("rows", []))
    for i, r in enumerate(d.get("rows", [])):
        m = _num(r.get("marks"))
        rows.append({"q": str(r.get("q", "")), "marks": None if m is None else (int(m) if m == int(m) else m),
                     "bbox": _clean_bbox(r.get("bbox"), i, n)})
    marks = [r["marks"] for r in rows if r["marks"] is not None]
    computed = sum(marks)                       # the maths is done HERE, never by the AI
    rec = _num(d.get("recorded_total"))
    cover = {"image": cover_path.name, "rows": rows,
             "recorded_total": None if rec is None else (int(rec) if rec == int(rec) else rec),
             "computed_total": computed,
             "total_bbox": _clean_bbox(d.get("total_bbox"), 0, 1) if _valid(d.get("total_bbox")) else [0.7, 0.9, 0.15, 0.05]}
    all_read = len(marks) == len(rows) and len(rows) > 0
    return cover, all_read


def audit_page(page_no, path):
    """Returns a list of omission findings (without ids) for one page."""
    d = call_vision_json(path, PAGE_PROMPT)
    regions = [r for r in d.get("regions", []) if r.get("has_student_writing")]
    img = load_page(path)
    found = []
    for i, r in enumerate(regions):
        bbox = _clean_bbox(r.get("bbox"), i, len(regions))
        px = red_pixels(img, bbox)
        vlm_marked = bool(r.get("has_evaluator_mark"))
        # Vision model is primary (phone photos make raw red-pixel counts noisy); red-ink pixels corroborate it.
        if not vlm_marked:
            conf = 0.6 + (0.3 if px < RED_MIN else 0.1 if px < 4 * RED_MIN else 0.0)
        elif px < RED_MIN:                       # AI sees a mark but no red ink: maybe pencil or blue pen, examiner should look
            conf = 0.45
        else:
            continue
        found.append({"page": page_no, "q": r.get("q"), "bbox": bbox, "confidence": round(conf, 2),
                      "red_pixels": px})
    return found


def run_audit(script_id, force=False):
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{script_id}.json"
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text())

    pages = page_files(script_id)
    if not pages:
        raise FileNotFoundError(f"Script {script_id} has no page_NN images")
    if 0 in pages:
        cover, all_read = audit_cover(pages[0])
    else:   # no cover sheet in this script: skip the totals check, still audit the answer pages
        cover, all_read = {"image": None, "rows": [], "recorded_total": None, "computed_total": 0,
                           "total_bbox": [0.0, 0.0, 0.0, 0.0], "missing": True}, False

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda kv: audit_page(*kv), [(n, p) for n, p in pages.items() if n != 0]))
    omissions = [f for page in results for f in page]
    cover_marks = {r["q"]: r["marks"] for r in cover["rows"]}

    findings = []
    if cover["recorded_total"] is not None and cover["recorded_total"] != cover["computed_total"]:
        findings.append({"type": "total_mismatch", "severity": "medium", "page": 0, "q": None,
                         "bbox": cover["total_bbox"], "confidence": 0.95 if all_read else 0.8,
                         "reason": f"Recorded {cover['recorded_total']}, rows sum to {cover['computed_total']}"})
    for o in sorted(omissions, key=lambda f: (f["page"], f["bbox"][1])):
        q = o["q"]
        blank_on_cover = q is not None and cover_marks.get(str(q), "missing") in (None, 0)
        sev = "high" if blank_on_cover else "medium"
        reason = ("Handwritten answer present, no marks or ticks" +
                  (f"; cover sheet shows no marks for Q{q}" if blank_on_cover else ""))
        findings.append({"type": "omission", "severity": sev, "page": o["page"], "q": q,
                         "bbox": o["bbox"], "confidence": o["confidence"], "reason": reason})
    for i, f in enumerate(findings, 1):
        f["id"] = f"f{i}"
        f.setdefault("q", None)
    out = {"script_id": script_id, "mock": MOCK, "cover": cover,
           "findings": [{k: f[k] for k in ("id", "type", "severity", "page", "q", "bbox", "confidence", "reason")}
                        for f in findings]}
    cache_file.write_text(json.dumps(out, indent=2))
    return out


def list_scripts():
    rows = []
    if not DATA.exists():
        return rows
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith(".") and valid_id(p.name)):
        nums = sorted(page_files(d.name))
        risk, n, mock = "unknown", 0, False
        c = CACHE / f"{d.name}.json"
        if c.exists():
            try:
                cached = json.loads(c.read_text())
                fs = cached["findings"]
                n, mock = len(fs), bool(cached.get("mock"))
                risk = "red" if any(f["type"] == "omission" for f in fs) else ("yellow" if fs else "green")
            except (json.JSONDecodeError, KeyError):
                pass      # unreadable cache: show as not audited, it will be rebuilt on open
        rows.append({"script_id": d.name, "pages": len(nums), "page_numbers": nums,
                     "risk": risk, "n_findings": n, "mock": mock})
    return rows
