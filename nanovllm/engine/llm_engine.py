import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SchedulerOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.world_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner") or self.model_runner is None:
            return
        self.model_runner.call("exit")
        self.scheduler.shutdown()
        del self.model_runner
        self.model_runner = None
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, *,
                    request_id=None, kv_transfer_params=None):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, request_id=request_id,
                       kv_transfer_params=kv_transfer_params)
        self.scheduler.add(seq)
        return seq.request_id

    def step(self):
        output: SchedulerOutput = self.scheduler.schedule()
        # print("scheduler output: ", output.seqs)
        token_ids, connector_output = self.model_runner.call("execute_step", output)
        self.scheduler.postprocess(output, token_ids)
        self.scheduler.update_connector_output(connector_output)
        finished = [(seq.seq_id, seq.completion_token_ids) for seq in output.seqs if seq.is_finished]
        if self.config.pd_config and self.config.pd_config.role == "prefill":
            finished = []
        return finished, output.num_prefill_tokens, len(output.decode_seqs)

    def get_kv_handoffs(self):
        handoffs = list(self.scheduler.handoffs)
        self.scheduler.handoffs.clear()
        return handoffs

    def get_transfer_failures(self):
        failures = list(self.scheduler.failures)
        self.scheduler.failures.clear()
        return failures

    def cancel_request(self, request_id):
        self.scheduler.cancel_request(request_id)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if self.config.pd_config:
            raise ValueError("PD engines require add_request/step and handoff routing; see examples/qwen3_pd.py")
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            finished, num_prefill_tokens, num_decode_tokens = self.step()
            elapsed = perf_counter() - t
            # Mixed batch: both counters update in the same step.
            if num_prefill_tokens > 0:
                prefill_throughput = num_prefill_tokens / elapsed
            if num_decode_tokens > 0:
                decode_throughput = num_decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
