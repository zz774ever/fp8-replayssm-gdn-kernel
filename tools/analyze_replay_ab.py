"""Summarize JSON-line logs produced by prototype/qwen35_replay_ab.py.

The important comparison is arm-to-arm: the quantization-free arm ("none") runs
through exactly the same replay plumbing as the FP8 arm, so comparing them
isolates quantization from the fp32-reference-vs-production-kernel arithmetic
floor that is present in *every* replay arm.
"""

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


def first_divergence(left: list, right: list) -> int | None:
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    return None


def match_fraction(left: list, right: list) -> float:
    steps = min(len(left), len(right))
    if steps == 0:
        return float("nan")
    return sum(
        1 for index in range(steps) if left[index] == right[index]
    ) / steps


def mean_abs_delta(left: list, right: list, upto: int | None = None) -> float | None:
    steps = min(len(left), len(right))
    if upto is not None:
        steps = min(steps, upto)
    if steps == 0:
        return None
    return sum(abs(left[i] - right[i]) for i in range(steps)) / steps


def main() -> None:
    for path in sys.argv[1:]:
        rows = load(path)
        runs = {
            row["label"]: row
            for row in rows
            if row.get("type") == "run"
        }
        summaries = [
            row for row in rows if row.get("type") == "summary"
        ]
        noise = [row for row in rows if row.get("type") == "noise_floor"]
        print(f"=== {path} ===")
        for row in noise:
            print(
                f"noise floor {row['label']}: identical={row['tokens_identical_to_baseline']}"
                f" max|dlogp|={row['max_abs_logprob_delta']:.2e}"
            )
        by_prompt: dict[int, list[dict]] = defaultdict(list)
        for summary in summaries:
            by_prompt[summary["prompt_index"]].append(summary)
        for prompt_index, group in sorted(by_prompt.items()):
            baseline_label = f"baseline_p{prompt_index}"
            baseline = runs[baseline_label]
            print(f"\n-- prompt {prompt_index} (baseline steps={len(baseline['tokens'])}) --")
            header = (
                f"{'arm':28} {'vs base':>8} {'1st div':>8} {'vs none':>8} "
                f"{'1st div':>8} {'prefix|dlogp|':>14} {'state mean':>11} "
                f"{'drift pre-div':>14} {'margin@div':>11}"
            )
            print(header)
            for summary in sorted(group, key=lambda item: (item["window"], item["label"])):
                label = summary["label"]
                arm = runs.get(label)
                if arm is None:
                    continue
                none_label = f"replay_none_L{summary['window']}_p{prompt_index}"
                none_arm = runs.get(none_label)
                vs_base_div = first_divergence(baseline["tokens"], arm["tokens"])
                vs_base_match = match_fraction(baseline["tokens"], arm["tokens"])
                if none_arm is not None and label != none_label:
                    vs_none_div = first_divergence(none_arm["tokens"], arm["tokens"])
                    vs_none_match = match_fraction(none_arm["tokens"], arm["tokens"])
                    arm_delta = mean_abs_delta(
                        none_arm["logprobs"], arm["logprobs"], vs_none_div
                    )
                    pair_text = f"{vs_none_match:8.3f} {str(vs_none_div):>8}"
                else:
                    pair_text = f"{'-':>8} {'-':>8}"
                    arm_delta = None
                prefix_delta = summary["prefix_mean_abs_logprob_delta"]
                if prefix_delta is None:
                    prefix_delta = mean_abs_delta(
                        baseline["logprobs"], arm["logprobs"]
                    )
                pre_div = summary.get("drift_before_divergence") or {}
                margin = summary.get("baseline_margin") or {}
                print(
                    f"{label:28} {vs_base_match:8.3f} {str(vs_base_div):>8} "
                    f"{pair_text} "
                    f"{(arm_delta if arm_delta is not None else prefix_delta):14.6f} "
                    f"{summary['state_relative_l2']['mean']:11.5f} "
                    f"{pre_div.get('value', float('nan')):14.4f} "
                    f"{str(margin.get('at_first_divergence')):>11}"
                )
            margins = [
                (summary.get("baseline_margin") or {})
                for summary in group
            ]
            margins = [entry for entry in margins if entry]
            if margins:
                first = margins[0]
                print(
                    f"   baseline margin: min={first['min']:.4g} median={first['median']:.4g} "
                    f"frac<0.05={first['fraction_below_0.05']:.3f} "
                    f"frac<0.2={first['fraction_below_0.2']:.3f}"
                )
            print_flip_attribution(runs, prompt_index)


def print_flip_attribution(runs: dict[str, dict], prompt_index: int) -> None:
    """Attribute the *first* divergence of each arm to the context's top-2 margin.

    Counting every downstream mismatch is misleading: once two sequences
    diverge, every later step differs by construction. The informative quantity
    is the margin of the context in which the flip happened, because that says
    how large a numerical perturbation is able to change the greedy choice.
    """
    baseline = runs.get(f"baseline_p{prompt_index}")
    if baseline is None or not baseline.get("margins"):
        return
    base_margins = baseline["margins"]
    ties = [index for index, value in enumerate(base_margins) if value == 0.0]
    print(
        f"   baseline zero-margin steps: {len(ties)} of {len(base_margins)} at {ties}"
    )
    print(
        f"   {'arm':26} {'1st flip':>9} {'margin@flip':>12} {'mean|dlogp|':>12} "
        f"{'max|dlogp|':>11}"
    )
    for label, arm in sorted(runs.items()):
        if label.startswith("baseline") or not label.endswith(f"p{prompt_index}"):
            continue
        flip = first_divergence(baseline["tokens"], arm["tokens"])
        prefix = flip if flip is not None else len(arm["tokens"])
        deltas = [
            abs(baseline["logprobs"][index] - arm["logprobs"][index])
            for index in range(prefix)
        ]
        margin = base_margins[flip] if flip is not None and flip < len(base_margins) else None
        print(
            f"   {label:26} {str(flip):>9} {str(margin):>12} "
            f"{(sum(deltas) / len(deltas)) if deltas else float('nan'):12.5f} "
            f"{(max(deltas) if deltas else float('nan')):11.4f}"
        )
    exact = {
        window: runs.get(f"replay_none_L{window}_p{prompt_index}") for window in (4, 8)
    }
    for window, arm in sorted(exact.items()):
        fp8 = runs.get(f"replay_vblock_L{window}_p{prompt_index}")
        if arm is None or fp8 is None:
            continue
        flip = first_divergence(arm["tokens"], fp8["tokens"])
        if flip is None:
            print(f"   FP8 L{window} tracked the exact arm for all {len(arm['tokens'])} steps")
            continue
        print(
            f"   FP8 L{window} vs exact: 1st flip at {flip}, "
            f"exact margin {arm['margins'][flip]:.3f}, fp8 margin {fp8['margins'][flip]:.3f}, "
            f"|dlogp| {abs(arm['logprobs'][flip] - fp8['logprobs'][flip]):.4f}"
        )


if __name__ == "__main__":
    main()
