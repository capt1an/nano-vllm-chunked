"""Qwen3-MoE 双卡推理示例。请从仓库根目录用 python -m examples.qwen3_moe 运行。"""

import argparse
from pathlib import Path
from time import perf_counter


DEFAULT_MODEL = (
    "/home/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
    "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="请用三句话解释什么是混合专家（MoE）模型。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须大于 0")
    if not Path(args.model, "config.json").is_file():
        parser.error("--model 必须指向包含 config.json 的本地模型目录")

    # 延迟导入：--help 不需要初始化推理依赖。
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    if torch.cuda.device_count() < 2:
        parser.error("需要两张可见 GPU；请设置 CUDA_VISIBLE_DEVICES=0,1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for device in range(2):
        free, total = torch.cuda.mem_get_info(device)
        print(
            f"GPU {device}: {torch.cuda.get_device_name(device)}, "
            f"空闲 {free / 2**30:.1f} / {total / 2**30:.1f} GiB",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # 直接传 token ids，避免 chat template 的特殊 token 被重复添加。
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    max_model_len = 1024
    if len(prompt_ids) + args.max_tokens > max_model_len:
        parser.error("prompt token 数 + --max-tokens 不能超过本示例的 1024 token 上限")

    print(f"输入 {len(prompt_ids)} tokens；TP=2，EP 未启用，CUDA Graph 关闭", flush=True)
    print("开始加载权重和 warmup（首次 torch.compile 可能较慢）……", flush=True)
    start = perf_counter()
    llm = LLM(
        args.model,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enable_continuous_batching=False,
        enable_chunked_prefill=False,
    )
    try:
        print(f"初始化完成：{perf_counter() - start:.1f}s", flush=True)
        # 当前 Sampler 不允许 temperature=0；小正数只是近似 greedy。
        params = SamplingParams(temperature=1e-6, max_tokens=args.max_tokens)
        start = perf_counter()
        output = llm.generate([prompt_ids], params, use_tqdm=False)[0]
        elapsed = perf_counter() - start
        print(f"\n问题：{args.prompt}\n回答：{output['text']}")
        print(f"输出 token ids：{output['token_ids']}")
        print(f"生成 {len(output['token_ids'])} tokens，用时 {elapsed:.2f}s")
        # 这只是执行链路 smoke test；合理文本不等于数值正确。
        # 下一步应对齐参考实现的 router、单层输出和 logits。
    finally:
        llm.exit()


if __name__ == "__main__":
    # TP worker 使用 multiprocessing spawn，需要 main guard。
    main()
