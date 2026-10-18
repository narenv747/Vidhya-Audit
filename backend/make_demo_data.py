"""Makes 4 fake answer scripts (with truth.json) so you can test before real data arrives.  Run from backend/:  python make_demo_data.py"""
import json
from pathlib import Path
import cv2, numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
RED, INK = (40, 40, 220), (90, 60, 30)   # BGR


def blank(h=1400, w=1000):
    return np.full((h, w, 3), 245, np.uint8)


def scribble(img, y0, y1):
    rng = np.random.default_rng(y0)
    for y in range(y0, y1, 28):
        cv2.line(img, (90, y), (90 + int(rng.integers(500, 780)), y + int(rng.integers(-3, 4))), INK, 2)


def tick(img, x, y):
    cv2.polylines(img, [np.array([[x, y], [x + 18, y + 24], [x + 55, y - 30]])], False, RED, 5)


def make(sid, grid, recorded, pages, truth_omissions):
    d = DATA / sid
    d.mkdir(parents=True, exist_ok=True)
    c = blank()
    cv2.putText(c, "MARKS GRID", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    for i, (q, m) in enumerate(grid):
        y = 400 + i * 110
        cv2.putText(c, f"Q{q}", (110, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        if m is not None:
            cv2.putText(c, str(m), (400, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, RED, 3)
    cv2.putText(c, f"Total {recorded}", (600, 1260), cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 3)
    cv2.imwrite(str(d / "page_00.jpg"), c)
    for n, marked in pages.items():
        p = blank()
        scribble(p, 300, 700)
        if marked:
            tick(p, 880, 450)
        cv2.imwrite(str(d / f"page_{n:02d}.jpg"), p)
    true_total = sum(m for _, m in grid if m)
    (d / "truth.json").write_text(json.dumps({"script_id": sid, "true_total": true_total,
        "recorded_total": recorded, "omissions": truth_omissions}))


# Mock AI reads the cover as rows 1a=4, 1b=5, 14b=3 and recorded_total=10.
make("S001", [("1a", 4), ("1b", 5), ("14b", 3)], 10, {1: True, 2: False}, [{"page": 2, "q": "14b"}])
make("S002", [("1a", 4), ("1b", 5), ("14b", 3)], 12, {1: True, 2: True}, [])
print("Made demo scripts in", DATA)
