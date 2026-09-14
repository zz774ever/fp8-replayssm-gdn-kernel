"""Summarize long-horizon drift logs from prototype/qwen35_drift_study.py."""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def main() -> None:
    for path in sys.argv[1:]:
        rows = load(path)
        capture = [row for row in rows if row.get("type") == "capture"]
        summaries = [row for row in rows if row.get("type") == "drift_summary"]
        curves = [row for row in rows if row.get("type") == "drift_curve"]
        print(f"=== {path} ===")
        for entry in capture:
            print(
                f"captured layers {entry['layers']} steps "
                f"{entry['steps_per_layer']}"
            )
        header = (
            f"{'layer':>5} {'arm':>14} {'steps':>6} {'d@128':>8} {'d@256':>8} "
            f"{'d@512':>8} {'d@1024':>9} {'d@2048':>9} {'mean':>8} {'max':>8} "
            f"{'mean out':>9}"
        )
        print(header)
        for row in sorted(summaries, key=lambda item: (item["layer"], item["arm"])):
            def value(key):
                return row.get(key, float("nan"))

            print(
                f"{row['layer']:>5} {row['arm']:>14} {row['steps']:>6} "
                f"{value('state_drift_at_128'):8.4f} {value('state_drift_at_256'):8.4f} "
                f"{value('state_drift_at_512'):8.4f} {value('state_drift_at_1024'):9.4f} "
                f"{value('state_drift_at_2048'):9.4f} {row['state_drift_mean']:8.4f} "
                f"{row['state_drift_max']:8.4f} {row['output_drift_mean']:9.5f}"
            )
        shape: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in curves:
            for step, state_drift, _output in row["points"]:
                shape[(row["layer"], row["arm"])].append((step, state_drift))
        print("\nstate drift curve at layer 0 (step: drift):")
        for (layer, arm), points in sorted(shape.items()):
            if layer != 0:
                continue
            sampled = [
                f"{step}:{value:.3f}"
                for step, value in points
                if step in (0, 64, 128, 256, 512, 1024, 2048)
            ]
            print(f"  {arm:>14} " + "  ".join(sampled))


if __name__ == "__main__":
    main()
