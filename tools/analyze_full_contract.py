"""Summarize bench_gdn_full_contract.py logs: core vs full-contract speedups."""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def main() -> None:
    for path in sys.argv[1:]:
        rows = [
            json.loads(line)
            for line in open(path, encoding="utf-8")
            if line.startswith("{")
        ]
        records = [row for row in rows if row.get("type") == "full_contract"]
        print(f"=== {path} ===")
        for row in rows:
            if row.get("type") == "validation":
                print(
                    "validation (window {}): production-vs-fp32 {:.4f}, "
                    "replay-full-vs-fp32 {:.4f}, replay-full-vs-production {:.4f}".format(
                        row["window"],
                        row["production_vs_fp32"],
                        row["replay_full_vs_fp32"],
                        row["replay_full_vs_production"],
                    )
                )
        for source in sorted({row["source"] for row in records}):
            subset = [row for row in records if row["source"] == source]
            print(f"\n-- source: {source} --")
            for window in sorted({row["window"] for row in subset}):
                print(f"\n  window L={window}")
                print(
                    f"  {'batch':>5} {'production':>11} {'prep':>8} {'core':>9} "
                    f"{'full':>9} {'core x':>7} {'full x':>7} {'best variant':>13}"
                )
                grouped: dict[int, list[dict]] = defaultdict(list)
                for row in subset:
                    if row["window"] == window:
                        grouped[row["batch"]].append(row)
                for batch in sorted(grouped):
                    best = max(grouped[batch], key=lambda item: item["full_speedup"])
                    print(
                        f"  {batch:>5} {best['production_ms'] * 1e3:10.2f}us "
                        f"{best['prep_ms'] * 1e3:7.2f}us {best['core_ms'] * 1e3:8.2f}us "
                        f"{best['full_ms'] * 1e3:8.2f}us {best['core_speedup']:7.3f} "
                        f"{best['full_speedup']:7.3f} {best['variant']:>13}"
                    )


if __name__ == "__main__":
    main()
