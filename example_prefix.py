#!/usr/bin/env python3
"""Correctness verification: legacy vs continuous-batching scheduling.

Runs the same prompts under both scheduling modes with near-deterministic
sampling (very low temperature) and verifies that generated token_ids match.
"""

import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(
        "/home/huggingface/models--Qwen--Qwen3-0.6B/snapshots/"
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    # A mix of short prompts (fit in one prefill step) and long prompts
    # (require chunking in CB mode; first-seq-only chunking in legacy).
    shared_prefix = """
You are a helpful AI assistant.
Your task is to answer questions accurately.
The following is a long system instruction:

"""*100

    prompts = [
        shared_prefix + "Question: What is Python?",
        shared_prefix + "Question: What is C++?",
        shared_prefix + "Question: Explain transformer architecture.",
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # Use low temperature for near-deterministic sampling so both modes
    # produce comparable outputs.
    sampling_params = SamplingParams(temperature=1e-6, max_tokens=128)

    engine_kwargs = dict(
        enforce_eager=True,
        max_num_seqs=2,
        max_num_batched_tokens=16384,
    )

    # ── Continuous Batching ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Mode: Continuous Batching (enable_continuous_batching=True)")
    print("=" * 60)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    llm_cb = LLM(path, **engine_kwargs,
                 enable_continuous_batching=True,
                 enable_chunked_prefill=True)
    
    llm_cb.add_request(prompts[0], sampling_params)
    # 跑几步，让A进入decode
    for _ in range(20):
        llm_cb.step()


    # 此时A:
    # running
    # block ref_count=1

    # request B到来
    llm_cb.add_request(prompts[1], sampling_params)

    # 再跑
    for _ in range(10):
        llm_cb.step()

    llm_cb.exit()
    del llm_cb


if __name__ == "__main__":
    main()
