# Vidya-Audit (backend)

AI quality-check for handwritten board-exam scripts. It FLAGS (a) wrong totals on the cover marks grid and (b) answers the examiner skipped. A human examiner always decides; the tool never changes marks.

## Run
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # add ONE API key (or set MOCK_VISION=1 to test with no key)
python make_demo_data.py    # optional: 2 fake scripts in ../data
python run_all.py           # audits every script -> ../audit_cache/*.json
python eval.py              # accuracy vs truth.json -> ../eval/results.md
uvicorn main:app --port 8000
```
Data layout: `data/{script_id}/page_00.jpg` (cover), `page_01.jpg`, ... and optional `truth.json`.

## API
`GET /scripts` · `GET /image/{id}/{page}` · `POST /audit/{id}` (cached; `?force=true` to redo) · `GET /audit/{id}`

## How it works
Vision AI reads the marks grid and finds answer regions; Python adds up the marks; OpenCV counts red-ink pixels beside each answer. Writing + no red ink + no examiner mark = flagged. Swap AI models in `.env` only.
