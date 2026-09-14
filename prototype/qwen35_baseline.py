"""Small reproducible Qwen3.5-4B eager vLLM baseline on one GPU."""

from __future__ import annotations

import argparse
import glob
import json
import time

import torch

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/root/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/*"
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    prompts = [
        "Explain in one sentence why recurrent state caching helps decoding."
    ] * args.batch
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        json.dumps(
            {
                "model": model,
                "batch": args.batch,
                "generated_tokens": generated,
                "elapsed_seconds": elapsed,
                "tokens_per_second": generated / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "sample": outputs[0].outputs[0].text,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
