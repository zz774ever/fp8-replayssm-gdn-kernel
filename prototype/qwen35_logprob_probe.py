"""Probe unmodified Qwen3.5 greedy decoding for near-ties in the top-2 logprobs.

The replay A/B consistently diverged at steps where the baseline's top-1 minus
top-2 logprob was exactly zero. This script checks that directly by dumping the
ranked alternatives for the steps with the smallest margins.
"""

from __future__ import annotations

import argparse
import glob
import json

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--report-lowest", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in two sentences why recurrent state caching matters for "
            "autoregressive decoding."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    matches = glob.glob(args.model)
    model = matches[0] if len(matches) == 1 else args.model
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=args.logprobs),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    tokenizer = llm.get_tokenizer()
    rows = []
    for index, token in enumerate(completion.token_ids):
        entry = completion.logprobs[index] if completion.logprobs else {}
        ranked = sorted(
            ((item, value.logprob) for item, value in entry.items()),
            key=lambda pair: -pair[1],
        )
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        rows.append(
            {
                "step": index,
                "token": token,
                "ranked": ranked,
                "margin": margin,
                "chosen_in_top": any(item == token for item, _ in ranked),
            }
        )
    finite = [row["margin"] for row in rows if row["margin"] != float("inf")]
    print(
        json.dumps(
            {
                "type": "margin_summary",
                "steps": len(rows),
                "entries_per_step": sorted({len(row["ranked"]) for row in rows}),
                "exact_ties": sum(1 for value in finite if value == 0.0),
                "below_1e-6": sum(1 for value in finite if value < 1e-6),
                "below_1e-3": sum(1 for value in finite if value < 1e-3),
                "below_0.05": sum(1 for value in finite if value < 0.05),
                "median": sorted(finite)[len(finite) // 2] if finite else None,
                "chosen_always_in_top": all(row["chosen_in_top"] for row in rows),
            }
        ),
        flush=True,
    )
    for row in sorted(rows, key=lambda item: item["margin"])[: args.report_lowest]:
        print(
            json.dumps(
                {
                    "type": "low_margin_step",
                    "step": row["step"],
                    "margin": row["margin"],
                    "alternatives": [
                        {
                            "token": item,
                            "logprob": value,
                            "text": tokenizer.decode([item]),
                        }
                        for item, value in row["ranked"][:5]
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    del torch


if __name__ == "__main__":
    main()
