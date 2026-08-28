"""Parse the demo/sweep-*.txt logs into a single JSON of curves and results."""
import json
import re
import sys
from pathlib import Path

DEMO = Path("/Users/jimharris/Documents/kdmd-SNN/demo")
OUT = Path(__file__).parent / "sweep_data.json"

tag_re = re.compile(r"^### \[(\w+)\] (.+?) — ")
step_re = re.compile(r"^\s*step\s+(\d+): mean loss ([\d.]+)")
result_re = re.compile(
    r"RESULT \[(\w+)\]: test accuracy ([\d.]+) \((\d+)/(\d+)\), final loss ([\d.]+), train ([\d.]+)s"
)

curves: dict[str, list[list[float]]] = {}
results: dict[str, dict] = {}

for log in sorted(DEMO.glob("sweep-*.txt")):
    tag = None
    for line in log.read_text().splitlines():
        m = tag_re.match(line)
        if m:
            tag = m.group(1)
            # ensemble logs repeat the tag per member; keep first member's curve
            if tag in curves and log.name != "sweep-AF-log.txt":
                tag = None
            elif tag not in curves:
                curves[tag] = []
            continue
        m = step_re.match(line)
        if m and tag:
            curves[tag].append([int(m.group(1)), float(m.group(2))])
            continue
        m = result_re.search(line)
        if m:
            results[m.group(1)] = {
                "acc": float(m.group(2)),
                "correct": int(m.group(3)),
                "total": int(m.group(4)),
                "final_loss": float(m.group(5)),
                "train_s": float(m.group(6)),
                "log": log.name,
            }

OUT.write_text(json.dumps({"curves": curves, "results": results}, indent=1))
print(f"tags with curves: {sorted(curves)}")
print(f"results: {sorted(results)}")
for t in ["I", "L", "O", "R", "X", "Z1", "Z2", "AF"]:
    r = results.get(t)
    n = len(curves.get(t, []))
    print(f"  {t}: acc={r['acc'] if r else '?'} curve_pts={n}")
