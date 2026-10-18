# Vidya-Audit

AI quality-check for handwritten board-exam answer scripts. It **flags** two kinds of mistakes for a human examiner to review. It never changes marks.

## Problem and user

**User:** evaluation-camp supervisors, deputy chief examiners and moderators who cross-check marked answer scripts (also teachers checking mock papers).

**Problem:** Marks are decided by hand, so clerical errors slip in: totals added wrongly, whole answers or pages left unmarked. News reports from Karnataka describe all of these:
- Wrong totalling by examiners, found at a Karnataka evaluation review ([Deccan Herald](https://www.deccanherald.com/india/karnataka/sslc-exams-over-350-evaluators-2546259))
- Two unevaluated pages that cost a student 20 marks in Physics ([Deccan Herald](https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fincomplete-evaluation-costs-student-dear-2545959))
- A mark of 98 entered as 8 ([Deccan Herald](https://www.deccanherald.com/city/8-to-98-its-a-snakes-and-ladders-game-for-tumakuru-girl-458913.html))

Today these are usually caught only if the student pays to get a scanned copy and applies for re-totalling or revaluation.

**What Vidya-Audit flags** (the examiner writes marks in the left margin in red, and circles the page total at the bottom left):
1. **Page total is wrong or missing:** the uncircled red marks on a page must add up to the red-circled total. If they don't, or there is marks but no circled total, it is flagged.
2. **Question has no mark:** every question number the student wrote needs a mark (0, 1, 2 ...) in the left margin. A mark at the top of the next page counts for a question that runs over the page.
3. **Cover sheet total** (optional, when a cover marks grid exists): the written grand total must equal the sum of its rows.

## How it works

1. A vision AI **reads** each page: the question numbers, the red margin marks beside them, and the circled page total. It does not judge answers or add anything up.
2. **Python does all the checking:** it adds the marks on each page, compares the sum with the circled total, and looks for questions with no mark, including marks that continue onto the next page.
3. Each flag gets a confidence score and a plain-language reason.
4. We tried OpenCV red-ink pixel counting as a second check, but on phone photos the printed margin line and edge glare made it too noisy, so we left it out of the decision.
5. The web page shows the scan with highlighted boxes. The reviewer chooses Confirm issue, Correct total or Dismiss. Decisions are saved and can be downloaded as JSON.

Why AI here: reading handwriting and ticks on photos needs a vision model. Counting and comparison are plain code so they cannot be wrong in the way a model can.

## Data and AI used

- **Vision AI:** Claude (Anthropic) by default; Google Gemini is supported. Switch with `VISION_PROVIDER` and `VISION_MODEL` in `backend/.env`. Only `backend/vision.py` talks to the AI.
- **Computer vision:** OpenCV, NumPy.
- **Data:** two synthetic demo scripts made by `backend/make_demo_data.py`. The Karnataka SSLC model question papers and model answers were used as reference material. No student scripts are stored in this repository.
- **Sample marked scripts used for tuning:** TODO: say where they came from and that you had permission to use them.

## How to run it

Needs Python 3.10+.

```bash
cd backend
python -m venv ../venv
# Windows PowerShell: ..\venv\Scripts\Activate.ps1   (Mac/Linux: source ../venv/bin/activate)
pip install -r ../requirements.txt
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

Open `.env` and add ONE API key. To try it with no key, set `MOCK_VISION=1`; the findings are then fake and the page shows a warning.

```bash
python make_demo_data.py    # optional: 2 demo scripts
python run_all.py           # audit every script in ../data
python eval.py              # compare with truth.json -> ../eval/results.md
uvicorn main:app --port 8000
```

Open **http://localhost:8000/app/**. Use "Add a script" on the left to upload a PDF or page images (first page = cover sheet).

Folder layout: `data/<script_id>/page_00.jpg` (cover), `page_01.jpg`, ... and an optional `truth.json`.

## AI tools and reused code

- Claude (Anthropic) was used as a coding assistant to write and review parts of the backend and the frontend. We tested the result ourselves.
- Libraries: FastAPI, Uvicorn, OpenCV, NumPy, PyMuPDF, python-dotenv, Anthropic and Google GenAI SDKs.
- Fonts: Fraunces and Hanken Grotesk from Google Fonts.
- Model question papers: Karnataka School Examination and Assessment Board, used as reference only.

## Limits

- **Accuracy on real scans is not yet measured.** `eval/results.md` covers only the demo scripts, and if it was run with `MOCK_VISION=1` it measures nothing real. TODO: add results from a real run here.
- The checks depend on the AI reading handwritten digits correctly (for example a 1 or a 7). A misread digit can cause a false flag, which is why a human reviews every flag.
- A continuation page with no marks and no circled total is not flagged. The question it belongs to is checked on the page where it started.
- Phone photos with heavy shadows, or marks in pencil or blue pen instead of red, can cause missed or false flags.
- It does not read or grade the content of answers.
- Handwriting is sent to the chosen AI provider, so use real student scripts only with permission and prefer a provider or setup approved for that data.
- Student scans and results are git-ignored (`data/`, `audit_cache/`, `decisions/`) and must never be committed.
- No sign-in: it is a single-user prototype.
