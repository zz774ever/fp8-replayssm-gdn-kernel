"""Summarize the GDN ReplaySSM kernel sweep (prototype/bench_gdn_replayssm.py)."""

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
        config = next((row for row in rows if row.get("type") == "config"), {})
        validation = [row for row in rows if row.get("type") == "validation"]
        benches = [row for row in rows if row.get("type") == "benchmark"]
        errors = [row for row in rows if row.get("type") == "benchmark_error"]
        print(f"=== {path} ===")
        print(
            f"device={config.get('device')} heads={config.get('heads')} "
            f"value_heads={config.get('value_heads')} K={config.get('key_dim')} "
            f"V={config.get('value_dim')} state={config.get('state_bytes_per_layer_per_seq')}B"
        )
        for row in validation:
            outputs = {
                key: value
                for key, value in row.items()
                if key.startswith("output_relative_l2")
            }
            rendered = ", ".join(
                f"{key.replace('output_relative_l2_', '')}={value:.5f}"
                for key, value in sorted(outputs.items())
            )
            print(
                f"  validation window={row['window']}: {rendered}, "
                f"flush state rel L2 {row['flush_state_relative_l2']:.5f}"
            )
        if errors:
            print(f"  {len(errors)} configs failed to build/run")
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in benches:
            grouped[(row["window"], row["batch"])].append(row)
        windows = sorted({row["window"] for row in benches})
        batches = sorted({row["batch"] for row in benches})
        for window in windows:
            print(f"\n-- window L={window} --")
            print(
                f"{'batch':>5} {'best speedup':>12} {'best cfg':>14} "
                f"{'amort ms':>9} {'bf16 ms':>8} {'traffic x':>9} "
                f"{'replay GB/s':>11} {'bf16 GB/s':>10}"
            )
            for batch in batches:
                group = grouped.get((window, batch))
                if not group:
                    continue
                winner = max(group, key=lambda item: item["speedup_vs_bf16"])
                print(
                    f"{batch:>5} {winner['speedup_vs_bf16']:12.3f} "
                    f"{('v%d/w%d' % (winner['block_v'], winner['num_warps'])):>14} "
                    f"{winner['amortized_ms'] * 1e3:9.2f} "
                    f"{winner['bf16_step_ms'] * 1e3:8.2f} "
                    f"{winner['traffic_ratio_vs_bf16']:9.2f} "
                    f"{winner['replay_gbps']:11.1f} {winner['bf16_gbps']:10.1f}"
                )
        if any("split_ms" in row for row in benches):
            print("\n-- tiled vs split (precompute + apply) --")
            print(
                f"{'window':>7} {'batch':>6} {'tiled best':>11} {'cfg':>10} "
                f"{'split best':>11} {'cfg':>10} {'gain':>7}"
            )
            for window in windows:
                for batch in batches:
                    group = grouped.get((window, batch))
                    if not group:
                        continue
                    tiled = max(group, key=lambda item: item["speedup_vs_bf16"])
                    split_group = [row for row in group if "split_speedup_vs_bf16" in row]
                    if not split_group:
                        continue
                    split = max(split_group, key=lambda item: item["split_speedup_vs_bf16"])
                    print(
                        f"{window:>7} {batch:>6} {tiled['speedup_vs_bf16']:11.3f} "
                        f"{('v%d' % tiled['block_v']):>10} "
                        f"{split['split_speedup_vs_bf16']:11.3f} "
                        f"{('v%d' % split['block_v']):>10} "
                        f"{split['split_speedup_vs_bf16'] / tiled['speedup_vs_bf16']:7.3f}"
                    )
        print("\n-- block size effect at the largest window, largest batch --")
        if any("cycle_tiled_speedup" in row for row in benches):
            print("\n-- full flush cycle (ring grows 1..L) --")
            header = (
                "window batch worst-case-tiled worst-case-split "
                "cycle-tiled cycle-split"
            )
            print(header)
            for row in sorted(benches, key=lambda item: (item["window"], item["batch"])):
                if "cycle_tiled_speedup" not in row:
                    continue
                print(
                    "{:>6} {:>5} {:>16.3f} {:>16.3f} {:>11.3f} {:>11.3f}".format(
                        row["window"],
                        row["batch"],
                        row["speedup_vs_bf16"],
                        row.get("split_speedup_vs_bf16", float("nan")),
                        row["cycle_tiled_speedup"],
                        row["cycle_split_speedup"],
                    )
                )
        for row in sorted(
            (r for r in benches if r["window"] == 16 and r["batch"] == 64),
            key=lambda r: (r["block_v"], r["num_warps"]),
        ):
            print(
                f"  block_v={row['block_v']:>3} warps={row['num_warps']} "
                f"replay={row['replay_ms'] * 1e3:7.2f}us "
                f"flush={row['flush_ms'] * 1e3:7.2f}us "
                f"speedup={row['speedup_vs_bf16']:5.3f} "
                f"traffic={row['traffic_ratio_vs_bf16']:5.2f}x "
                f"replay_bw={row['replay_gbps']:6.1f}GB/s"
            )


if __name__ == "__main__":
    main()
