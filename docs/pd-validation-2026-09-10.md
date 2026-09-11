# PD 验证记录（2026-09-10）

环境：双 RTX A6000 48 GiB，Qwen3-0.6B 本地 snapshot c1899de289a04d12100db370d81485cdf75e47ca，NIXL / UCX。测试针对当前工作区，未提交 git commit。

## 方法

新增 `tests/pd_stress.py`：两个独立 engine，同一对 engine 连续多轮请求，逐个检查 handoff 和完成事件唯一性。每轮完成后等待两端 drain，断言 requests、remote_waiting、预留 blocks 和在途传输均为空，所有物理 block 的引用计数为零。主动检查正常退出，超时/异常则失败。

输入长度覆盖 1、31、127、128、129、255、256、257、383、511、512、513 token。重复同一组 12 个 prompt，测试共享输入下独立 block 的隔离和复用。生成开启 ignore_eos，强制达到指定输出长度；temperature=1e-6，使用项目现有 sampler。每种输入单独执行普通 eager baseline，保留所有不一致的请求及首个分歧位置。

“running 峰值”是在调度轮之间观测的队列峰值，并非 GPU kernel 同时执行的请求数。P/D 峰值不保证发生在同一时刻。耗时含调度、传输、生成和 drain，不含模型启动及 baseline；这些是正确性压力测试，不是严格性能 benchmark。

## 已完成结果

| 模式 | 每轮请求 × 轮数 | 配置并发上限 | 输出长度 | P/D running 峰值 | 完全匹配 baseline |
|---|---:|---:|---:|---:|---:|
| eager | 128 × 3 | 32 | 32 | 32 / 32 | 361 / 384 |
| CUDA Graph | 256 × 3 | 64 | 32 | 64 / 22 | 709 / 768 |
| CUDA Graph 长输出 | 128 × 1 | 64 | 128 | 64 / 42 | 119 / 128 |

上述七轮共 1,280 个请求均完成，未发现丢失/重复完成、卡死或 block 未回收；每轮两端各 464 个 block 全部空闲。eager 各轮约 12.41 / 13.62 / 13.47 秒，Graph 各轮约 25.80 / 23.99 / 20.68 秒。部分诊断任务有运行时间重叠，不能据此比较 eager 和 Graph 性能。

长输出组约 15.03 秒完成生成和 drain，9 个不一致也全部来自 1-token prompt，首个分歧索引为 28 或 30。三组合计 1,189 / 1,280 个完整输出匹配。

前两组共 82 个不一致全部来自 1-token prompt，首次分歧出现在生成 token 的零基索引 28 或 30。其他 11 种长度全部匹配。

## 差异诊断

`tests/pd_sampling_repeat.py` 在普通单引擎上重复相同 1-token prompt：16 次串行得到 3 种输出，32 请求批量得到 2 种输出。记录到 token 13482 和 5435 的 logits 同为 17.25。现有 sampler 通过 exponential 噪声采样，极小温度也不会消除最高分并列时的随机性。

82 个差异结果中，80 个完整序列在这次 baseline 重复实验中复现。证据证明单次 baseline 不是确定性 oracle；但不能据此宣布所有差异均已逐个证明无误，也没有修改 sampler 或放宽匹配标准。

## 传输与状态压力验证

- `tests/pd_tensor_stress.py`：32 路并发、10 轮复用，共 320 次真实 GPU KV READ。每轮改变源内容；P/D pool 容量不同、block 重映射、257-token 尾块。比较整个目标 cache，rtol=0、atol=0，包括所有未写区域，全部通过。
- `tests/test_pd_stress.py`：1,000 请求，10 轮，48 个小 block、最多 32 请求、13-token 调度预算。确定性随机注入延迟、乱序、取消、失败和重复事件；每步检查 block 不重叠、free/owned 完整分区和引用计数；每轮完整回收，通过。这是 scheduler 事件模拟，不是实际网络故障注入。
- PD 单元测试 24 项通过；完整回归 50 项通过（12.93 秒），包含已有 GPU 测试。

## 复现

```bash
export PYTHONPATH=.
MODEL=/path/to/Qwen3-0.6B
.venv/bin/python tests/pd_stress.py "$MODEL" --eager --report /tmp/pd-stress-eager.json
.venv/bin/python tests/pd_stress.py "$MODEL" --concurrency 64 --requests 256 --waves 3 --report /tmp/pd-stress-graph.json
.venv/bin/python tests/pd_stress.py "$MODEL" --concurrency 64 --requests 128 --waves 1 --max-tokens 128 --port 5800 --report /tmp/pd-stress-long.json
.venv/bin/python tests/pd_tensor_stress.py
.venv/bin/python tests/pd_sampling_repeat.py "$MODEL"
.venv/bin/python -m unittest discover -s tests -p 'test_pd*.py' -v
```

真实模型任务应串行执行，保持 GPU 空闲。曾因诊断并行占用显存导致长输出测试在 cache 分配阶段启动失败，清空后重跑；另一次诊断默认 TCP 8767 端口冲突，改用独立 5899 后成功。这两次启动失败不计为完成测试。

原始 JSON 和日志位于 `/tmp/pd-stress-{eager,graph,long}.{json,log}`、`/tmp/pd-tensor-stress.log`、`/tmp/pd-baseline-repeat.{json,log}`。

## 验证边界

本次没有覆盖数小时 soak、真实进程崩溃/断网、跨机器 RDMA、所有模型和量化组合。仍可观察到 UCX CUDA teardown error 日志；完成和资源回收断言与它分开记录。调度器 `_seen_transfer_ids` 会保留历史去重 ID，随累计请求增长；block 回收通过不等于所有 Python 元数据内存有界。P 已被 D 认领的 lease 在 D 崩溃后无法自动安全回收，当前仍是已知可用性限制。
