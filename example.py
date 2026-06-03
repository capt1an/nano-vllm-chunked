import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    qwenpath = "/home/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    llamapath = "/home/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    mistralpath = "/home/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
    optpath = "/home/huggingface/models--peft-internal-testing--opt-125m/snapshots/81556e26994ef650e173b86bdadc11b6b8170fd4"

    path = os.path.expanduser(optpath)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # opt模型是causal language model（自回归语言模型）没有 chat
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for prompt in prompts
    # ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
