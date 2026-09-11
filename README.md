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

### Experimental scheduled prefill/decode disaggregation

Two independent `LLMEngine` instances each own a Scheduler, BlockManager and
ModelRunner. P schedules chunked prefill; D reserves local blocks and waits in
`WAITING_FOR_REMOTE_KVS` while other ready requests continue executing. Each
`SchedulerOutput` carries `ConnectorMetadata`; worker completion events return
to the scheduler even on steps with no model batch.

The worker connector uses **NIXL/UCX READ into registered GPU memory**. The HTTP
control channel carries only registrations, block mappings and acquire/release
messages. The example's process pipes carry requests, handoffs and outputs;
no KV tensors are copied through Python CPU buffers or the router.

Install the optional transport dependency in the existing inference environment:

```bash
uv sync --extra pd --inexact
python -m examples.qwen3_pd /path/to/Qwen3 --eager --check
# CUDA Graph decode:
python -m examples.qwen3_pd /path/to/Qwen3 --check
```

The existing CUDA/attention dependencies must also be available. This version
requires two local CUDA GPUs, TP=1, no EP, and the same model path, dtype, block
size and attention layout on both ends. Device arguments are physical indices
or GPU UUIDs; the launcher sets each process's mask before importing CUDA code.
`--port` defaults to 5600: P/D control ports are 5600/5601 and their independent
model rendezvous ports are 5700/5701. Ports must be unused. The control service
is intended for trusted local peers; it is not an authenticated public API.

A short arithmetic prompt and 256/257-token boundary prompts are submitted
concurrently. `--check` compares output token IDs against ordinary inference.
Different batch shapes can change reduced-precision logits, and near-zero
sampling temperature still samples tied logits: arbitrary prompts are not
promised bitwise-identical generation across schedules. The tensor-level
integration test separately checks exact KV values and untouched padding:

```bash
PYTHONPATH=. python tests/pd_tensor_smoke.py
python -m unittest discover -s tests -v
```

#### Engine API and ownership

Configure each engine using `PDConfig("prefill"/"decode", port=...)` in
`Config.pd_config`; `None` preserves ordinary inference. Engine IDs are generated
per config and must be distinct across engines/restarts. Configure separate
`distributed_init_method` and `worker_shm_name` values for colocated instances.

Use `add_request(..., request_id=..., kv_transfer_params=...)` and keep calling
`step()` independently on both engines. Read P's `get_kv_handoffs()` and pass the
original prompt, sampling parameters and handoff to D. Read D's normal step
outputs and both engines' `get_transfer_failures()`. `generate()` is disabled
for PD engines because a blocking single-engine call cannot route handoffs.
`examples/qwen3_pd.py` provides a minimal concurrent router, not an HTTP serving
API. `cancel_request(request_id)` supports queued, running and transferring work.

P discards its sampled token and retains the computed prompt blocks. D imports
prompt KV, recomputes the last prompt token to obtain logits, and owns all user
sampling/EOS/max-token decisions. Each PD admission reserves its full KV budget:
P reserves the prompt, D reserves prompt plus maximum completion length. Future
D blocks are kept separately from its current block table. Oversized requests
fail; temporarily unavailable capacity queues requests. PD has no preemption
or prefix sharing; ordinary inference retains its existing prefix cache behavior.

A transfer has a unique `transfer_id`, separate from request and engine IDs.
`KVTransferParams` describes the handoff; `KVLoadTask` adds destination blocks.
`ConnectorMetadata` contains only newly submitted commands. Worker handles and
in-flight records survive per-step metadata clearing. `ConnectorOutput` carries
terminal events. Sequence pickle remains a compact TP execution snapshot, not
a complete scheduler checkpoint or the cross-engine request protocol.

#### Failure and shutdown behavior

P publishes KV only after producer writes complete. A consumer acquires one
read lease, imports the remote registration, reads the mapped valid token
regions, and confirms release after its memory accesses stop. P frees retained
blocks only after that release. Unclaimed handoffs expire after
`PDConfig.transfer_timeout` (default 60 seconds). **Claimed blocks never expire
merely because time passed**: a dead peer or lost release confirmation retains
P blocks and requires coordinated peer shutdown/restart. This initial protocol
chooses retention over recycling memory that may still be read.

D reports cancellation/failure only after NIXL confirms its handle is released.
Active transfers which cannot be cancelled retain their blocks. Stop D before P
so acknowledgements can finish; worker shutdown raises if remote accesses remain
rather than unregistering active memory. This is not a crash-recovery protocol.
No layerwise pipeline, remote prefix reuse, heterogeneous parallelism or
throughput improvement is claimed.

On the tested environment (NIXL 0.10.1 / nixl-cu12 1.4.1), UCX emits a
`cuDevicePrimaryCtxGetState ... error code 4` message during process teardown.
Exact tensor checks and generation comparisons pass, but this
teardown diagnostic has not been eliminated. Reduced-precision inference can
also produce the existing TorchDynamo recompilation-limit diagnostic.
