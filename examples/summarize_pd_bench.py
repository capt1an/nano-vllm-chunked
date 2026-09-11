"""Summarize bench_pd JSON files with pooled latency quantiles."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    data=json.loads(Path(path).read_text())
    print(f"\nModel: {data['options']['model']}\n")
    print('| 输入 | 部署 | output tok/s | tok/s/GPU | TTFT p95 ms | ITL p95 ms | E2E p95 ms |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for case in ('short','long','mixed'):
        for mode in ('single','replicas','tp','pd'):
            rows=[r for r in data['records'] if r['case']==case and r['mode']==mode and not r['warmup']]
            if not rows:
                continue
            assert len(rows)==data['options']['rounds'], 'incomplete benchmark'
            tok_s=sum(r['output_tokens'] for r in rows)/sum(r['seconds'] for r in rows)
            ttft=[v for r in rows for v in r['first'].values()]
            e2e=[v for r in rows for v in r['finished'].values()]
            itl=[v for r in rows for stamps in r['timestamps'].values() for v in np.diff(stamps)]
            print(f"| {case} | {mode} | {tok_s:.1f} | {tok_s/rows[0]['gpu_count']:.1f} | "
                  f"{np.percentile(ttft,95)*1000:.1f} | {np.percentile(itl,95)*1000:.2f} | "
                  f"{np.percentile(e2e,95)*1000:.1f} |")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+')
    for report in parser.parse_args().reports:
        summarize(report)
