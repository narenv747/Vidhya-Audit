"""Compares findings with data/{id}/truth.json.  Run:  python eval.py   -> eval/results.md
truth.json: {"script_id":"S001","true_total":65,"recorded_total":62,"omissions":[{"page":9,"q":"14b"}]}"""
import json
from audit import DATA, CACHE, ROOT, run_audit

tot = {"tp": 0, "fp": 0, "fn": 0}
om = {"tp": 0, "fp": 0, "fn": 0}
total_ok = n = 0
rows = []
for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
    t = d / "truth.json"
    if not t.exists():
        continue
    truth, pred = json.loads(t.read_text()), run_audit(d.name)
    n += 1
    # 1) total mismatch
    exp = truth.get("true_total") != truth.get("recorded_total")
    got = any(f["type"] == "total_mismatch" for f in pred["findings"])
    if exp and got: tot["tp"] += 1
    elif got: tot["fp"] += 1
    elif exp: tot["fn"] += 1
    total_ok += pred["cover"]["computed_total"] == truth.get("true_total")
    # 2) omissions: match on page, and on question if both are known
    exp_o = [(o["page"], str(o["q"]) if o.get("q") else None) for o in truth.get("omissions", [])]
    got_o = [(f["page"], str(f["q"]) if f["q"] else None) for f in pred["findings"] if f["type"] == "omission"]
    used = set()
    for g in got_o:
        hit = next((i for i, e in enumerate(exp_o) if i not in used and e[0] == g[0] and (e[1] == g[1] or e[1] is None or g[1] is None)), None)
        if hit is None:
            om["fp"] += 1
        else:
            used.add(hit); om["tp"] += 1
    om["fn"] += len(exp_o) - len(used)
    rows.append(f"| {d.name} | {len(exp_o)} | {len(got_o)} | {'yes' if exp else 'no'} | {'yes' if got else 'no'} |")

def pr(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
    return p, r

mock_used = any(run_audit(d.name).get("mock") for d in DATA.iterdir() if d.is_dir() and (d / "truth.json").exists())
po, ro = pr(om)
pt, rt = pr(tot)
lines = ["# Vidya-Audit evaluation", "", f"Scripts evaluated: {n}", "",
         f"- Skipped-answer detection: precision {po:.0%}, recall {ro:.0%} (TP {om['tp']}, FP {om['fp']}, FN {om['fn']})",
         f"- Total-mismatch detection: precision {pt:.0%}, recall {rt:.0%} (TP {tot['tp']}, FP {tot['fp']}, FN {tot['fn']})",
         f"- Cover total re-computed correctly: {total_ok}/{n}", "",
         *(["", "WARNING: these scores used mock AI answers and do not measure real accuracy."] if mock_used else []),   *(["", "WARNING: these scores used mock AI answers and do not measure real accuracy."] if mock_used else []),
         "| Script | Skipped (truth) | Skipped (found) | Total wrong (truth) | Total wrong (found) |", "|---|---|---|---|---|", *rows]
(ROOT / "eval").mkdir(exist_ok=True)
(ROOT / "eval" / "results.md").write_text("\n".join(lines))
print("\n".join(lines))
