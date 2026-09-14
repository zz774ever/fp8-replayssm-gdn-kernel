"""Find which sampler entry point the engine actually calls.

Needed because patching ``Sampler.sample`` had no effect on the generated
tokens, so the forced-token hook has to be placed on whichever function really
runs. Counts calls on every candidate and reports the winner.
"""

from __future__ import annotations

import glob
import json

import torch

COUNTS: dict[str, int] = {}


def _wrap(cls, name):
    original = getattr(cls, name)

    def wrapped(*args, **kwargs):
        COUNTS[name] = COUNTS.get(name, 0) + 1
        return original(*args, **kwargs)

    wrapped.__name__ = name
    setattr(cls, name, wrapped)


def main() -> None:
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu import model_runner as mr

    for name in ("forward", "sample", "__call__"):
        if hasattr(Sampler, name):
            _wrap(Sampler, name)
    for name in ("sample", "sample_tokens"):
        if hasattr(mr.GPUModelRunner, name):
            _wrap(mr.GPUModelRunner, name)
    from vllm import LLM, SamplingParams

    matches = glob.glob(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"
    )
    llm = LLM(
        model=matches[0],
        dtype="bfloat16",
        max_model_len=256,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    out = llm.generate(
        ["Explain in one sentence why state caching matters."],
        SamplingParams(temperature=0.0, max_tokens=5),
        use_tqdm=False,
    )
    print(
        json.dumps(
            {
                "type": "probe",
                "counts": COUNTS,
                "sampler_class": type(getattr(llm.llm_engine, "model_executor", None)).__name__,
                "tokens": list(out[0].outputs[0].token_ids),
                "cuda": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
