"""Compares findings with data/{id}/truth.json.  Run:  python eval.py   -> eval/results.md
truth.json: {"script_id":"S001","true_total":65,"recorded_total":62,
             "omissions":[{"page":9,"q":"14b"}],          # questions with no mark in the margin
             "page_total_issues":[{"page":3}]}            # pages whose circled total is missing or wrong (optional)"""
import json
from audit import DATA, CACHE, ROOT, run_audit

tot = {"tp": 0, "fp": 0, "fn": 0}
om = {"tp": 0, "fp": 0, "fn": 0}
pt = {"tp": 0, "fp": 0, "fn": 0}
total_ok = n = 0
mock_used = False
rows = []
for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
    t = d / "truth.json"
    if not t.exists():
        continue
    truth, pred = json.loads(t.read_text()), run_audit(d.name)
    n += 1
    mock_used = mock_used or bool(pred.get("mock"))
    # 1) cover total mismatch
    exp = truth.get("true_total") != truth.get("recorded_total")
    got = any(f["type"] == "total_mismatch" for f in pred["findings"])
    if exp and got: tot["tp"] += 1
    elif got: tot["fp"] += 1
    elif exp: tot["fn"] += 1
    total_ok += pred["cover"]["computed_total"] == truth.get("true_total")
    # 2) unmarked questions: match on page, and on question if both are known
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
    # 3) page totals missing or wrong, matched on page
    exp_p = {p["page"] for p in truth.get("page_total_issues", [])}
    got_p = {f["page"] for f in pred["findings"] if f["type"] in ("page_total_mismatch", "page_total_missing")}
    pt["tp"] += len(exp_p & got_p); pt["fp"] += len(got_p - exp_p); pt["fn"] += len(exp_p - got_p)
    rows.append(f"| {d.name} | {len(exp_o)} | {len(got_o)} | {len(exp_p)} | {len(got_p)} | {'yes' if exp else 'no'} | {'yes' if got else 'no'} |")

def pr(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
    return p, r

po, ro = pr(om)
pp, rp = pr(pt)
ptt, rt = pr(tot)
lines = ["# Vidya-Audit evaluation", "", f"Scripts evaluated: {n}", "",
         *(["**WARNING: these scores used mock AI answers and do not measure real accuracy.**", ""] if mock_used else []),
         f"- Unmarked-question detection: precision {po:.0%}, recall {ro:.0%} (TP {om['tp']}, FP {om['fp']}, FN {om['fn']})",
         f"- Page-total check (missing or wrong): precision {pp:.0%}, recall {rp:.0%} (TP {pt['tp']}, FP {pt['fp']}, FN {pt['fn']})",
         f"- Cover-total mismatch: precision {ptt:.0%}, recall {rt:.0%} (TP {tot['tp']}, FP {tot['fp']}, FN {tot['fn']})",
         f"- Cover total re-computed correctly: {total_ok}/{n}", "",
         "| Script | Unmarked (truth) | Unmarked (found) | Page-total issues (truth) | Page-total issues (found) | Cover total wrong (truth) | Cover total wrong (found) |",
         "|---|---|---|---|---|---|---|", *rows]
(ROOT / "eval").mkdir(exist_ok=True)
(ROOT / "eval" / "results.md").write_text("\n".join(lines))
print("\n".join(lines))
