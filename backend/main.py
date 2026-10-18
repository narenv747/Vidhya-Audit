from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from audit import DATA, page_files, run_audit, list_scripts

app = FastAPI(title="Vidya-Audit")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/scripts")
def scripts():
    return list_scripts()


@app.get("/image/{script_id}/{page}")
def image(script_id: str, page: int):
    if not (DATA / script_id).is_dir():
        raise HTTPException(404, "Unknown script")
    p = page_files(script_id).get(page)
    if p is None:
        raise HTTPException(404, "Unknown page")
    return FileResponse(p)


@app.post("/audit/{script_id}")
def audit_post(script_id: str, force: bool = False):
    try:
        return run_audit(script_id, force=force)   # uses the cache unless force=true
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(502, str(e))


@app.get("/audit/{script_id}")
def audit_get(script_id: str):
    return audit_post(script_id)
