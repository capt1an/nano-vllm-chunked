#!/usr/bin/env python3
"""
Correctness verification: LRU KV cache eviction.

Workflow:

1. Request A finishes -> A prefix blocks become cached.
2. Request B finishes -> B prefix blocks become cached.
3. Request A again -> A prefix cache is accessed and timestamp updated.
4. Request C arrives and requires new KV blocks.
5. LRU should evict B blocks instead of recently accessed A blocks.
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


    # ==============================
    # Shared prefix
    # ==============================
    shared_prefixA = """
You are a helpful AI assistant.
Your task is to answer questions accurately.
The following is a long system instruction:

""" * 100
    shared_prefixB = """
Correctness verification: legacy vs continuous-batching scheduling.

Runs the same prompts under both scheduling modes with near-deterministic
sampling (very low temperature) and verifies that generated token_ids match.

""" * 200

    prompts = [

        # Request A
        shared_prefixA +
        """
Question:
Explain Python programming language in detail.
""",

        # Request B
        shared_prefixB +"""
You are a helpful AI assistant.
Your task is to answer questions accurately.

Question:
Explain C++ programming language in detail.
""",

        # Request A again
        shared_prefixA +
        """
Question:
What are the advantages of Python?
""",

        # Request C
        """
You are a helpful AI assistant.
Your task is to answer questions accurately.

Question:
Explain operating systems in detail.
"""*50
    ]


    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


    sampling_params = SamplingParams(
        temperature=1e-6,
        max_tokens=32,
    )


    engine_kwargs = dict(
        enforce_eager=True,

        # 需要允许多个request存在
        max_num_seqs=2,

        # 小一点，让调度频繁发生
        max_num_batched_tokens=16384,
    )


    print("=" * 60)
    print("LRU KV Cache Eviction Test")
    print("=" * 60)


    torch.manual_seed(42)


    llm = LLM(
        path,
        **engine_kwargs,
        enable_continuous_batching=True,
        enable_chunked_prefill=True,
    )


    # ==================================================
    # Step 1:
    # Request A
    #
    # Finish -> cached
    # ==================================================

    print("\n[1] Send Request A")

    llm.add_request(
        prompts[0],
        sampling_params
    )


    while not llm.is_finished():
        llm.step()


    print("\nAfter Request A finished:")
    llm.scheduler.block_manager.print_status()



    # ==================================================
    # Step 2:
    # Request B
    #
    # Finish -> cached
    # ==================================================

    print("\n[2] Send Request B")

    llm.add_request(
        prompts[1],
        sampling_params
    )


    while not llm.is_finished():
        llm.step()


    print("\nAfter Request B finished:")
    llm.scheduler.block_manager.print_status()



    # ==================================================
    # Step 3:
    # Access A again
    #
    # Update LRU timestamp
    # ==================================================

    print("\n[3] Access Request A prefix again")


    llm.add_request(
        prompts[2],
        sampling_params
    )


    while not llm.is_finished():
        llm.step()


    print("\nAfter accessing A:")
    llm.scheduler.block_manager.print_status()



    # ==================================================
    # Step 4:
    # Request C
    #
    # Force eviction
    # ==================================================

    print("\n[4] Send Request C")
    

    llm.add_request(
        prompts[3],
        sampling_params
    )


    while not llm.is_finished():
        llm.step()


    print("\nAfter Request C:")
    llm.scheduler.block_manager.print_status()



    llm.exit()



if __name__ == "__main__":
    main()