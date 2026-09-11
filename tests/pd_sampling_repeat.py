"""Diagnose baseline sampling repeatability; run with PYTHONPATH=. python tests/pd_sampling_repeat.py MODEL."""
import json
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.layers.sampler import Sampler
from transformers import AutoTokenizer
original = Sampler.forward
traces = []
def traced(self, logits, temperatures):
    values, ids = logits.float().topk(3, dim=-1)
    result = original(self, logits, temperatures)
    traces.append(dict(values=values.tolist(), ids=ids.tolist(), chosen=result.tolist()))
    return result
def main():
    Sampler.forward = traced
    import sys
    model = sys.argv[1]
    prompt=AutoTokenizer.from_pretrained(model).encode('The capital of France is Paris. Water freezes at zero degrees Celsius. ')[:1]
    llm=LLM(model,distributed_init_method='tcp://127.0.0.1:5899',enforce_eager=True,max_model_len=1024,max_num_batched_tokens=128,max_num_seqs=32,gpu_memory_utilization=.3)
    p=SamplingParams(temperature=1e-6,max_tokens=32,ignore_eos=True)
    try:
        serial=[llm.generate([prompt],p,use_tqdm=False)[0]['token_ids'] for _ in range(16)]
        concurrent=[x['token_ids'] for x in llm.generate([prompt]*32,p,use_tqdm=False)]
    finally:
        llm.exit()
    Path('/tmp/pd-baseline-repeat.json').write_text(json.dumps(dict(serial=serial,concurrent=concurrent,traces=traces)))
    print('Unique serial outputs:',len({tuple(x) for x in serial}), 'unique concurrent outputs:',len({tuple(x) for x in concurrent}))


if __name__ == "__main__":
    main()
