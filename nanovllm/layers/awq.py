"""Single-GPU AWQ GEMM (INT4 weights, FP16 activations, group size 128)."""
import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.distributed import get_tensor_parallel_world_size


def validate_awq(config):
    expected = dict(quant_method="awq", bits=4, group_size=128, zero_point=True)
    if any(config.get(k) != v for k, v in expected.items()) or str(config.get("version", "")).lower() != "gemm":
        raise ValueError("Supported quantization: AWQ GEMM, bits=4, group_size=128, zero_point=True")
    if config.get("modules_to_not_convert"):
        raise ValueError("AWQ modules_to_not_convert is not supported yet")


def dequantize_awq(qweight, qzeros, scales):
    """Independent PyTorch reference, returning an unpacked [K, N] weight."""
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=qweight.device)
    weight = ((qweight.unsqueeze(-1) >> shifts) & 15).flatten(-2)
    zeros = ((qzeros.unsqueeze(-1) >> shifts) & 15).flatten(-2)
    return ((weight - zeros.repeat_interleave(128, 0)).float()
            * scales.float().repeat_interleave(128, 0)).half()


@triton.jit
def _gemm(X, W, Z, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr = 16, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    # AWQ nibble ordering: logical 0..7 -> packed slots 0,4,1,5,2,6,3,7.
    shift = ((n % 8) // 2 + (n % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + k[:, None] * (N // 8) + n[None, :] // 8,
                    (k[:, None] < K) & (n[None, :] < N), 0)
        z = tl.load(Z + (k[:, None] // 128) * (N // 8) + n[None, :] // 8,
                    (k[:, None] < K) & (n[None, :] < N), 0)
        s = tl.load(S + (k[:, None] // 128) * N + n[None, :],
                    (k[:, None] < K) & (n[None, :] < N), 0)
        w = ((((w >> shift[None, :]) & 15) - ((z >> shift[None, :]) & 15)).to(tl.float32) * s.to(tl.float32)).to(tl.float16)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


class AWQLinear(nn.Module):
    def __init__(self, input_size, output_sizes, shard_ids=None):
        super().__init__()
        if get_tensor_parallel_world_size() != 1:
            raise ValueError("AWQ currently requires tensor_parallel_size=1")
        if input_size % 128 or any(n % 8 for n in output_sizes):
            raise ValueError("AWQ requires K divisible by 128 and each output partition divisible by 8")
        self.input_size = input_size
        self.output_size = sum(output_sizes)
        self.partitions = dict(zip(shard_ids, output_sizes)) if shard_ids is not None else {None: self.output_size}
        self.loaded = set()
        for name, shape, dtype in (
            ("qweight", (input_size, self.output_size // 8), torch.int32),
            ("qzeros", (input_size // 128, self.output_size // 8), torch.int32),
            ("scales", (input_size // 128, self.output_size), torch.float16),
        ):
            param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            param.weight_loader = self._loader(name)
            self.register_parameter(name, param)

    def _loader(self, name):
        def load(param, tensor, shard_id=None):
            if shard_id not in self.partitions:
                raise ValueError(f"Invalid AWQ shard: {shard_id}")
            divisor = 1 if name == "scales" else 8
            offset = 0
            for key, size in self.partitions.items():
                if key == shard_id:
                    break
                offset += size
            target = param.data.narrow(1, offset // divisor, self.partitions[shard_id] // divisor)
            if target.shape != tensor.shape or target.dtype != tensor.dtype:
                raise ValueError(f"AWQ {name}/{shard_id}: expected {target.shape}/{target.dtype}, got {tensor.shape}/{tensor.dtype}")
            target.copy_(tensor)
            self.loaded.add((name, shard_id))
        return load

    def validate_loaded(self):
        missing = {(name, shard) for name in ("qweight", "qzeros", "scales") for shard in self.partitions} - self.loaded
        if missing:
            raise ValueError(f"Missing AWQ checkpoint tensors: {missing}")

    def forward(self, x):
        if x.dtype != torch.float16 or not x.is_cuda:
            raise ValueError("AWQ requires CUDA FP16 activations")
        x = x.contiguous()
        if x.shape[-1] != self.input_size:
            raise ValueError("AWQ input size mismatch")
        m = x.numel() // self.input_size
        y = torch.empty((*x.shape[:-1], self.output_size), device=x.device, dtype=x.dtype)
        if m:
            _gemm[(triton.cdiv(m, 16), triton.cdiv(self.output_size, 64))](
                x, self.qweight, self.qzeros, self.scales, y, m, self.output_size, self.input_size)
        return y
