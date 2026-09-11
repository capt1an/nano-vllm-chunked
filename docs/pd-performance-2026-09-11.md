# Qwen3-0.6B：两卡 TP 与 PD 性能对照

## 测试配置

2026-09-11，双 RTX A6000 48 GiB。模型为本地 `Qwen/Qwen3-0.6B`，snapshot `c1899de289a04d12100db370d81485cdf75e47ca`。

主基线按用户要求使用 nano-vLLM 普通两卡 **TP=2**，不是两个独立副本。PD 为同样两张 GPU 上的 **1P+1D，每端 TP=1**。二者保持 nano-vLLM 默认调度参数：max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、gpu_memory_utilization=0.9，启用 CUDA Graph、continuous batching、chunked prefill。

普通 TP 引擎保留 prefix cache；PD 保留独占 block 与全量空间预留。TP 的 KV heads 被分片，PD 每端保存完整模型和本地 KV，不能简单按 block 数量比较两者的有效容量。

基准不修改生产 scheduler 或 connector。普通引擎每步不额外 sleep；PD 保留 `qwen3_pd` 示例每轮 1ms sleep，让后台控制通信线程取得执行机会。router 轮询间隔为 0.1ms。比较的是这两个实际运行策略，并非仅比较 GPU kernel 时间。

## 负载与计时

本地 ShareGPT 数据集：`learnanything/sharegpt_v3_unfiltered_cleaned_split`，snapshot `ddc6e27875b7198a46702daec3e61a55e222fc14`。抽取第一条 human 消息，应用 Qwen3 chat template，关闭 thinking，固定随机种子 20260911，去除完全重复的输入。没有重复或拼接文本凑长度。

每轮同时提交 64 个请求，每个固定生成 64 token，ignore_eos=True、temperature=1e-6。

| 场景 | 输入长度筛选范围 | 正式轮实际平均输入长度 |
|---|---:|---:|
| short | 64–256 | 118.8 |
| long | 1024–2048 | 1455.9 |
| mixed | 长短各半 | 771.1 |

每个场景预热一轮，再测试三轮。预热和正式轮使用不同输入，两种部署复用完全相同的 token 化输入和轮次划分。每轮之间等待请求及传输全部 drain，并检查 block 回收和 worker 正常退出。每种部署正式计量 576 个请求，生成 36,864 个 token。

- **输出吞吐**：三轮输出 token 总数 / 三轮 makespan 之和，不把输入 token 算入输出吞吐。
- **TTFT**：router 开始提交整批请求到收到 first-token 事件，包含排队和 PD handoff。
- **E2E**：同一起点到 router 收到完成事件。
- **ITL**：worker 在相邻生成 token 的 `engine.step()` 结束时记录时间戳之差，包含调度等待；不是 HTTP 客户端流式 ITL。
- 延迟分位数合并三轮样本后计算，不把三个 p95 简单平均。模型加载、图捕获、预热、tokenizer 和最后 drain 不计入 makespan。

这是 **burst/offline** 基准，不是固定 QPS 的在线服务压测，不能据此推导 SLO 最大吞吐。三轮输入不同，因此波动同时反映输入变化；没有随机化部署测试顺序。固定输出长度用于对齐工作量，不要求采样文本逐字一致。

## 测量结果与结论

**在本次负载和默认预算下，当前 PD 没有超过两卡 TP=2。** 两者都使用两张 GPU，因此这不是 GPU 数量不同造成的比较偏差。所有 1,152 个正式计量请求（另有 384 个预热请求）均完成，所有 worker 正常退出；测试结束后两张 GPU 均回到 1 MiB。UCX teardown error 仍有出现，单独保留于日志。

| 输入 | 部署 | output tok/s | tok/s/GPU | TTFT p95 ms | ITL p95 ms | E2E p95 ms |
|---|---|---:|---:|---:|---:|---:|
| short | tp | 7787.0 | 3893.5 | 91.8 | 7.18 | 528.8 |
| short | pd | 3726.7 | 1863.4 | 673.6 | 7.56 | 1115.5 |
| long | tp | 2119.0 | 1059.5 | 1050.6 | 17.05 | 1941.4 |
| long | pd | 1295.9 | 647.9 | 2270.4 | 22.16 | 3596.6 |
| mixed | tp | 3030.1 | 1515.1 | 559.6 | 14.23 | 1324.5 |
| mixed | pd | 1953.0 | 976.5 | 1322.8 | 15.40 | 2193.9 |

相对 TP，PD 的输出吞吐在 short / long / mixed 上分别降低 **52.1% / 38.8% / 35.5%**；本次 TTFT、ITL 和 E2E 的 p95 也都没有改善。不能据此在简历中宣称 PD 提升了吞吐或降低了尾延迟。

以下属于实现层面的潜在解释，尚未通过分段 profiling 量化：0.6B 在 TP=2 下计算与通信的实际成本较低；PD 额外经历 handoff、HTTP acquire/ACK、KV READ 和末 token 重算，P/D 还各使用完整模型而非两卡分片。当前实现的全局 CUDA 同步、Python 控制线程池及每轮 1ms sleep 也可能影响性能。现有结果不能把差距全部归因于某一个环节，也不代表 PD 在其他模型、长输出或持续到达负载下必然无收益。

三轮耗时和吞吐保存在同目录 `pd-performance-2026-09-11-summary.json`，完整时间戳在原始 JSON。这里只测固定 64-token 输出及 64 请求 burst；若继续优化，应该先分段计时定位排队、prefill、acquire、READ、ACK、decode 的成本，再用相同基线复测。

## 复现

```bash
PYTHONPATH=. .venv/bin/python -m examples.bench_pd /path/to/Qwen3-0.6B \
  --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --modes tp pd --report /tmp/pd-benchmark-tp-pd.json
PYTHONPATH=. .venv/bin/python -m examples.summarize_pd_bench /tmp/pd-benchmark-tp-pd.json
```

`--workloads-from` 可以读取旧报告中的 token 化输入，避免重复加载数据集并保证输入完全一致。本次输入 SHA256：`387180c637a212af3a96beb33313fe3c9895960b181a670af13cbf9898699956`。

原始结果 `/tmp/pd-benchmark-tp-pd.json` 包含配置、输入 token、逐请求生成 token 与时间戳；日志 `/tmp/pd-benchmark-tp-pd.log`。旧的单卡/双副本探索性结果 `/tmp/pd-benchmark-06b.json` 使用不同预算和无 sleep 循环，不混入主对照。此前一次重测因其他用户占用 GPU 在显存分配阶段失败，未计入测试；确认 GPU 空闲后重新运行本次 TP/PD 对照。
