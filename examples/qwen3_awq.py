"""Run a local Qwen3 AWQ checkpoint: python -m examples.qwen3_awq MODEL_PATH."""
import argparse
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) for prompt in ['What is 2 + 3? Answer briefly.', '用一句话介绍北京。']]
    llm = LLM(args.model, enforce_eager=args.eager, tensor_parallel_size=1,
              max_model_len=512, max_num_batched_tokens=128,
              max_num_seqs=8, gpu_memory_utilization=0.3)
    try:
        for output in llm.generate(prompts, SamplingParams(temperature=1e-6, max_tokens=32)):
            print(output['text'])
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
