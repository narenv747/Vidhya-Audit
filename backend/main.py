import json, os, re, shutil, time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audit import DATA, ROOT, page_files, run_audit, list_scripts, valid_id

DECISIONS = ROOT / "decisions"
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "40"))
MAX_PAGES = 80
OK_DECISIONS = {"accepted", "dismissed", "corrected"}

app = FastAPI(title="Vidya-Audit")
# Only local pages may call the API. Add more with ALLOWED_ORIGINS="https://a.com,https://b.com"
extra = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=extra,
                   allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["GET", "POST", "PUT"], allow_headers=["*"])


def check_id(script_id: str) -> str:
    if not valid_id(script_id):
        raise HTTPException(400, "Script ID may only use letters, numbers, - and _ (max 64 characters).")
    return script_id


@app.get("/scripts")
def scripts():
    return list_scripts()


@app.get("/image/{script_id}/{page}")
def image(script_id: str, page: int):
    check_id(script_id)
    if not (DATA / script_id).is_dir():
        raise HTTPException(404, "Unknown script")
    p = page_files(script_id).get(page)
    if p is None:
        raise HTTPException(404, "Unknown page")
    return FileResponse(p)


@app.post("/audit/{script_id}")
def audit_post(script_id: str, force: bool = False):
    check_id(script_id)
    if force:   # finding ids change on a re-run, so old decisions would point at the wrong findings
        (DECISIONS / f"{script_id}.json").unlink(missing_ok=True)
    try:
        return run_audit(script_id, force=force)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(502, str(e))


@app.get("/audit/{script_id}")
def audit_get(script_id: str):
    return audit_post(script_id)


# ---------- examiner decisions (so they survive a page refresh) ----------
class DecisionBody(BaseModel):
    decisions: dict[str, str] = {}
    log: list[dict] = []


@app.get("/decisions/{script_id}")
def decisions_get(script_id: str):
    check_id(script_id)
    f = DECISIONS / f"{script_id}.json"
    if not f.exists():
        return {"decisions": {}, "log": []}
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return {"decisions": {}, "log": []}


@app.put("/decisions/{script_id}")
def decisions_put(script_id: str, body: DecisionBody):
    check_id(script_id)
    if not (DATA / script_id).is_dir():
        raise HTTPException(404, "Unknown script")
    if any(v not in OK_DECISIONS for v in body.decisions.values()):
        raise HTTPException(400, "Unknown decision value.")
    DECISIONS.mkdir(exist_ok=True)
    log = [{"t": str(l.get("t", ""))[:20], "s": script_id, "m": str(l.get("m", ""))[:300]} for l in body.log[:200]]
    (DECISIONS / f"{script_id}.json").write_text(json.dumps({"decisions": body.decisions, "log": log}, indent=2))
    return {"ok": True}


# ---------- upload ----------
def _natural(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), script_id: str = Form(""), has_cover: bool = Form(True)):
    """Accepts one PDF, or several JPG/PNG page images (sorted by file name). Saves them as data/<id>/page_NN.jpg."""
    sid = script_id.strip() or time.strftime("UPLOAD_%Y%m%d_%H%M%S")
    check_id(sid)
    target = DATA / sid
    if target.exists():
        raise HTTPException(409, f"A script called {sid} already exists. Choose a different ID.")

    pages: list[bytes] = []        # JPEG bytes, in page order
    for f in sorted(files, key=lambda u: _natural(u.filename or "")):
        data = await f.read()
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} is larger than {MAX_MB} MB.")
        name = (f.filename or "").lower()
        if name.endswith(".pdf"):
            try:
                import fitz
            except ImportError:
                raise HTTPException(503, "PDF upload needs PyMuPDF. Run: pip install pymupdf")
            try:
                doc = fitz.open(stream=data, filetype="pdf")
                pages += [pg.get_pixmap(dpi=110).tobytes("jpg") for pg in doc]
            except Exception:
                raise HTTPException(400, f"{f.filename} is not a readable PDF.")
        elif name.endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, f"{f.filename} is not a readable image.")
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            pages.append(buf.tobytes())
        else:
            raise HTTPException(400, f"{f.filename}: only PDF, JPG and PNG files are supported.")
        if len(pages) > MAX_PAGES:
            raise HTTPException(413, f"A script can have at most {MAX_PAGES} pages.")
    if not pages:
        raise HTTPException(400, "No pages found in the upload.")

    tmp = DATA / f".tmp_{sid}"      # write everything first, then rename, so a failure never leaves half a script
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    start = 0 if has_cover else 1
    for i, b in enumerate(pages):
        (tmp / f"page_{start + i:02d}.jpg").write_bytes(b)
    tmp.rename(target)
    return {"script_id": sid, "pages": len(pages), "has_cover": has_cover}


# Serve the UI from the same server: open http://localhost:8000/app/
if (ROOT / "frontend").is_dir():
    app.mount("/app", StaticFiles(directory=ROOT / "frontend", html=True), name="app")
