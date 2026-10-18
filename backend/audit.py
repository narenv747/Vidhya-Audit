"""The audit pipeline.
Per page, the vision AI only READS the red ink (question numbers, margin marks, circled page total).
Python then does all the checking:
  Rule 1  the uncircled red marks on a page must add up to the red-circled total at the bottom left.
  Rule 2  every question number written on a page needs a mark (0, 1, 2 ...) in the left margin.
An optional cover sheet (page_00) is still checked: its recorded total against the sum of its rows."""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vision import call_vision_json, MOCK

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE = ROOT / "data", ROOT / "audit_cache"
IMG_EXT = (".jpg", ".jpeg", ".png")
DEFAULT_TOTAL_BOX = [0.02, 0.86, 0.22, 0.10]          # bottom left, where the circled page total sits
RED_TYPES = ("omission", "page_total_mismatch")        # these make a script "red" in the list

COVER_PROMPT = """[COVER] This is the front cover of a handwritten Indian board-exam answer script. It has a marks grid.
Read every row of the grid. Return ONLY this JSON:
{"rows":[{"q":"1a","marks":4,"bbox":[x,y,w,h]}],"recorded_total":62,"total_bbox":[x,y,w,h]}
Rules: q is the question number as printed. marks is the number written by the examiner, or null if the cell is blank.
recorded_total is the grand total the examiner wrote, or null. Do NOT add up the marks yourself.
Every bbox is the cell's position as numbers between 0 and 1: x,y = top-left corner, w,h = size, as fractions of the page width and height."""

PAGE_PROMPT = """[PAGE] This is one page of a handwritten Indian board-exam answer script. The examiner marks it in RED ink.
Return ONLY this JSON:
{"has_student_writing":true,
 "questions":[{"q":"14","bbox":[x,y,w,h],"marks":[3]}],
 "carry_marks":[2],
 "page_total":{"value":11,"bbox":[x,y,w,h]}}
Rules:
1. questions: every question the student STARTS on this page, in top-to-bottom order. q is the question number exactly as the student wrote it
   (for example "1", "14", "21"). One entry per question, even when it has parts (a), (b), (c). Do not list headings such as "Section A"
   or the "Teacher's Signature" line. bbox covers that question's answer on this page.
2. marks: the red numbers the examiner wrote in the LEFT margin (or just left of the answer) for that question, NOT circled.
   Include 0 if the examiner wrote 0. If the question has several red numbers, list all of them. If it has no red number, use [].
   Ticks, slashes, underlines, crosses and comments are NOT marks. Never read the student's own writing or printed numbers as marks.
3. carry_marks: red uncircled numbers near the top of the page that sit ABOVE the first question number on this page.
   They belong to a question that started on the previous page. Use [] if there are none.
4. page_total: the red number inside a red CIRCLE near the BOTTOM-LEFT of the page. Use null if there is no circled number.
   A circled number is only ever the page total, never a question mark.
5. has_student_writing is false only for a blank or ruled-only page.
6. Read digits carefully and do not guess. If you cannot read a number, leave it out. Do NOT add anything up yourself.
All boxes are numbers between 0 and 1: x,y = top-left corner, w,h = size, as fractions of the page width and height."""


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
    return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)
            and 0 <= b[0] <= 1 and 0 <= b[1] <= 1 and b[2] > 0.02 and b[3] > 0.01
            and b[0] + b[2] <= 1.02 and b[1] + b[3] <= 1.02 and b[2] * b[3] < 0.95)


def _clean_bbox(b, i=0, n=1):
    """If the AI's box is sloppy, use a coarse horizontal band instead."""
    if _valid(b):
        return [round(float(v), 4) for v in b]
    h = 1.0 / max(n, 1)
    return [0.05, round(i * h, 4), 0.9, round(h, 4)]


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _tidy(x):
    return int(x) if x == int(x) else x


def _marks(v):
    """A list of mark numbers (0 counts), ignoring anything that is not a believable mark."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for m in v:
        n = _num(m)
        if n is not None and 0 <= n <= 100:
            out.append(_tidy(n))
    return out


def _fmt(nums):
    return " + ".join(str(n) for n in nums) if nums else "no marks"


# ---------- reading one page ----------
def audit_page(path):
    """Ask the AI to read one page, then clean its answer into a fixed shape."""
    d = call_vision_json(path, PAGE_PROMPT)
    raw = [r for r in d.get("questions", []) if isinstance(r, dict)]
    qs = []
    for i, r in enumerate(raw):
        q = r.get("q")
        qs.append({"q": None if q in (None, "") else str(q),
                   "bbox": _clean_bbox(r.get("bbox"), i, len(raw)), "marks": _marks(r.get("marks"))})
    qs.sort(key=lambda r: r["bbox"][1])
    pt, total, tbox = d.get("page_total"), None, DEFAULT_TOTAL_BOX
    if isinstance(pt, dict):
        v = _num(pt.get("value"))
        if v is not None:
            total = _tidy(v)
            tbox = [round(float(x), 4) for x in pt["bbox"]] if _valid(pt.get("bbox")) else DEFAULT_TOTAL_BOX
    elif _num(pt) is not None:
        total = _tidy(_num(pt))
    return {"has_writing": bool(d.get("has_student_writing", True)), "questions": qs,
            "carry_marks": _marks(d.get("carry_marks")), "page_total": total, "total_bbox": tbox}


# ---------- the two rules (pure Python, no AI) ----------
def page_summary(a):
    marks = [m for q in a["questions"] for m in q["marks"]] + a["carry_marks"]
    return {"marks": marks, "sum": _tidy(sum(marks)), "page_total": a["page_total"],
            "has_writing": a["has_writing"], "questions": [{"q": q["q"], "marks": q["marks"]} for q in a["questions"]]}


def rule_findings(pages):
    """pages = {page_no: cleaned analysis}. Returns findings (without ids), checked across pages."""
    out = []
    for n in sorted(pages):
        a = pages[n]
        if not a["has_writing"]:
            continue
        s = page_summary(a)
        marks, total = s["marks"], a["page_total"]
        # Rule 1: uncircled red marks must add up to the circled page total
        if total is None:
            if marks:                       # a continuation page with no marks at all needs no total
                out.append({"type": "page_total_missing", "severity": "medium", "page": n, "q": None,
                            "bbox": DEFAULT_TOTAL_BOX, "confidence": 0.8, "expected": s["sum"],
                            "reason": f"Marks on this page add up to {s['sum']} ({_fmt(marks)}), but there is no circled page total"})
        elif abs(total - s["sum"]) > 1e-6:
            out.append({"type": "page_total_mismatch", "severity": "high", "page": n, "q": None,
                        "bbox": a["total_bbox"], "confidence": 0.9, "expected": s["sum"],
                        "reason": f"Marks on this page add up to {s['sum']} ({_fmt(marks)}), but the circled page total is {total}"})
        # Rule 2: every question number needs a mark. A mark at the top of the next page counts for the last question here.
        qs, nxt = a["questions"], pages.get(n + 1)
        for i, q in enumerate(qs):
            if q["marks"]:
                continue
            if i == len(qs) - 1 and nxt and nxt["carry_marks"]:
                continue
            label = f"Question {q['q']}" if q["q"] else "A question"
            out.append({"type": "omission", "severity": "high", "page": n, "q": q["q"], "bbox": q["bbox"],
                        "confidence": 0.85 if q["q"] else 0.6,
                        "reason": f"{label} is written here but has no mark in the left margin"})
    return out


# ---------- cover sheet (optional) ----------
def audit_cover(cover_path):
    d = call_vision_json(cover_path, COVER_PROMPT)
    rows, n = [], len(d.get("rows", []))
    for i, r in enumerate(d.get("rows", [])):
        m = _num(r.get("marks"))
        rows.append({"q": str(r.get("q", "")), "marks": None if m is None else _tidy(m),
                     "bbox": _clean_bbox(r.get("bbox"), i, n)})
    marks = [r["marks"] for r in rows if r["marks"] is not None]
    computed = sum(marks)                       # the maths is done HERE, never by the AI
    rec = _num(d.get("recorded_total"))
    cover = {"image": cover_path.name, "rows": rows,
             "recorded_total": None if rec is None else _tidy(rec),
             "computed_total": computed,
             "total_bbox": _clean_bbox(d.get("total_bbox"), 0, 1) if _valid(d.get("total_bbox")) else [0.7, 0.9, 0.15, 0.05]}
    all_read = len(marks) == len(rows) and len(rows) > 0
    return cover, all_read


def run_audit(script_id, force=False):
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{script_id}.json"
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text())

    files = page_files(script_id)
    if not files:
        raise FileNotFoundError(f"Script {script_id} has no page_NN images")
    if 0 in files:
        cover, all_read = audit_cover(files[0])
    else:   # no cover sheet: the per-page totals are the check
        cover, all_read = {"image": None, "rows": [], "recorded_total": None, "computed_total": 0,
                           "total_bbox": [0.0, 0.0, 0.0, 0.0], "missing": True}, False

    nums = [n for n in files if n != 0]
    with ThreadPoolExecutor(max_workers=4) as ex:
        analyses = dict(zip(nums, ex.map(lambda n: audit_page(files[n]), nums)))

    findings = []
    if cover["recorded_total"] is not None and cover["recorded_total"] != cover["computed_total"]:
        findings.append({"type": "total_mismatch", "severity": "medium", "page": 0, "q": None,
                         "bbox": cover["total_bbox"], "confidence": 0.95 if all_read else 0.8,
                         "expected": cover["computed_total"],
                         "reason": f"Recorded {cover['recorded_total']}, rows sum to {cover['computed_total']}"})
    order = {"page_total_mismatch": 0, "page_total_missing": 1, "omission": 2}
    findings += sorted(rule_findings(analyses), key=lambda f: (f["page"], order[f["type"]], f["bbox"][1]))
    for i, f in enumerate(findings, 1):
        f["id"] = f"f{i}"
    keys = ("id", "type", "severity", "page", "q", "bbox", "confidence", "reason", "expected")
    out = {"script_id": script_id, "mock": MOCK, "cover": cover,
           "pages": {str(n): page_summary(a) for n, a in analyses.items()},
           "findings": [{k: f[k] for k in keys if k in f} for f in findings]}
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
                risk = "red" if any(f["type"] in RED_TYPES for f in fs) else ("yellow" if fs else "green")
            except (json.JSONDecodeError, KeyError):
                pass      # unreadable cache: show as not audited, it will be rebuilt on open
        rows.append({"script_id": d.name, "pages": len(nums), "page_numbers": nums,
                     "risk": risk, "n_findings": n, "mock": mock})
    return rows
