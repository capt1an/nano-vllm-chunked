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
    prompts = [
        "Repeat exactly: APPLE",
        "Please list all prime numbers smaller than 100.",
        # longer prompts to exercise chunked-prefill paths
        (
            "Write a comprehensive summary of the key differences between "
            "Python and C++, covering memory management, typing discipline, "
            "compilation model, performance characteristics, and typical use "
            "cases. Be thorough and specific."
        ),
        (
            "Explain how a transformer neural network works, starting from "
            "the attention mechanism, multi-head attention, positional encoding, "
            "feed-forward layers, layer normalization, residual connections, "
            "and how these components combine to process sequential data."
        ),
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
        max_num_seqs=4,
        max_num_batched_tokens=512,
    )

    # ── Legacy ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Mode: Legacy (enable_continuous_batching=False)")
    print("=" * 60)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    llm_legacy = LLM(path, **engine_kwargs,
                     enable_continuous_batching=False,
                     enable_chunked_prefill=True)
    outputs_legacy = llm_legacy.generate(prompts, sampling_params)
    for output in outputs_legacy:
        print()
        print("=" * 50)
        print(tokenizer.decode(output["token_ids"]))
    llm_legacy.exit()
    del llm_legacy
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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
    outputs_cb = llm_cb.generate(prompts, sampling_params)
    for output in outputs_cb:
        print()
        print("=" * 50)
        print(tokenizer.decode(output["token_ids"]))
    llm_cb.exit()
    del llm_cb


if __name__ == "__main__":
    main()
