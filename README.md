<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
### Qwen3 AWQ inference

Dense Qwen3 AWQ checkpoints are detected from `quantization_config`. The initial
implementation supports AWQ GEMM INT4, group size 128, asymmetric zero points,
FP16 execution and a single GPU (`tensor_parallel_size=1`). TP/EP, quantized
attention bias and custom `modules_to_not_convert` lists are not supported.
Embedding and normalization checkpoint weights are converted to the execution
dtype; the tied LM head continues to share the embedding storage.

```bash
python -m examples.qwen3_awq /path/to/Qwen3-4B-AWQ --eager
# Enable CUDA Graph for decode:
python -m examples.qwen3_awq /path/to/Qwen3-4B-AWQ
python -m unittest discover -s tests -p test_awq.py -v
```

Weights remain packed in GPU memory. The Triton kernel unpacks and dequantizes
weight tiles during GEMM with FP32 accumulation; it does not retain a full FP16
copy of each linear weight. This first implementation prioritizes correctness,
with no tuned prefill/decode kernel selection. GPU tests compare against a
PyTorch dequantization reference and exercise CUDA Graph replay. The example
runs two short prompts; it is not a model-quality or performance benchmark.
