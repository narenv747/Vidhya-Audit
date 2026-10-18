"""Audit every script in data/ and save audit_cache/*.json.  Run:  python run_all.py [--force]"""
import sys
from audit import DATA, run_audit

force = "--force" in sys.argv
ids = sorted(p.name for p in DATA.iterdir() if p.is_dir())
for i, sid in enumerate(ids, 1):
    try:
        r = run_audit(sid, force=force)
        print(f"[{i}/{len(ids)}] {sid}: {len(r['findings'])} findings")
    except Exception as e:
        print(f"[{i}/{len(ids)}] {sid}: FAILED - {e}")
