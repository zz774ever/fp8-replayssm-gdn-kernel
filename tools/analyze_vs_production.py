"""Summarize bench_gdn_vs_production.py logs: replay vs the production operator."""

from __future__ import annotations

import json
import sys


def main() -> None:
    for path in sys.argv[1:]:
        rows = [
            json.loads(line)
            for line in open(path, encoding="utf-8")
            if line.startswith("{")
        ]
        compared = [row for row in rows if row.get("type") == "compare"]
        print(f"=== {path} ===")
        for row in rows:
            if row.get("type") == "validation":
                print(
                    "validation: production vs fp32 = {:.4f}, replay vs fp32 = "
                    "{:.4f}, production vs replay = {:.4f}".format(
                        row["production_vs_fp32_output_rel_l2"],
                        row["replay_vs_fp32_output_rel_l2"],
                        row["production_vs_replay_output_rel_l2"],
                    )
                )
        for window in sorted({row["window"] for row in compared}):
            print(f"\n-- window L={window} (single layer, one decode step) --")
            print(
                f"{'batch':>5} {'production':>11} {'our-bf16':>10} "
                f"{'replay best':>12} {'variant':>12} {'speedup':>8}"
            )
            for row in sorted(
                (item for item in compared if item["window"] == window),
                key=lambda item: item["batch"],
            ):
                print(
                    f"{row['batch']:>5} {row['production_ms'] * 1e3:10.2f}us "
                    f"{row['our_bf16_step_ms'] * 1e3:9.2f}us "
                    f"{row[row['best_variant'] + '_cycle_ms'] * 1e3:11.2f}us "
                    f"{row['best_variant']:>12} "
                    f"{row['best_speedup_vs_production']:8.3f}"
                )
        if compared:
            print(
                "\nnote: 'our-bf16' is the same-style reference kernel and is "
                "slower than the production operator, so the production column "
                "is the baseline that matters."
            )


if __name__ == "__main__":
    main()
