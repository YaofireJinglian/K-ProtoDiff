"""Paired sampler ablation on existing journal EMA checkpoints (no training)."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from Models.KPD.k_diffusion import Model

MODES = ['legacy_full', 'ddim50', 'reflection_fixed50', 'adaptive50']


def summarize(records, output):
    records = [r for r in records if r['seed'] == 2026 and not r.get('smoke', False)]
    lines = ['# Sampling Efficiency Ablation', '',
             '> 仅使用 seed 2026 的期刊版 EMA 权重、FP32、batch=32、长度=24；每组预热一次、重复三次。',
             '> 报告三次计时重复的平均值，三位小数，不报告跨种子标准差。显存列为三次运行峰值的平均值；仅含本进程 PyTorch 分配，不含其他进程、CUDA 上下文和驱动内存。',
             '> legacy_full 是期刊版权重搭配旧采样器，并非重新训练的会议版。只比较效率，不据此声称质量相同或训练加速。', '',
             '| Dataset | Sampler | Seeds | Seconds | Samples/s | Peak allocated MiB | Peak reserved MiB | Denoiser calls | Speedup vs legacy | Peak memory reduction % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return f'{statistics.mean(v):.3f}'
    for dataset in sorted({r['dataset'] for r in records}):
        subset = [r for r in records if r['dataset']==dataset]
        base = {r['seed']: statistics.mean(x['seconds'] for x in r['measurements'])
                for r in subset if r['mode']=='legacy_full'}
        base_memory = {r['seed']: statistics.mean(x['peak_allocated_mib'] for x in r['measurements'])
                       for r in subset if r['mode']=='legacy_full'}
        for mode in MODES:
            group = [r for r in subset if r['mode']==mode]
            if not group: continue
            values = {key: [statistics.mean(x[key] for x in r['measurements']) for r in group]
                      for key in ['seconds','samples_per_second','peak_allocated_mib','peak_reserved_mib','denoiser_calls']}
            ratios = [base[r['seed']]/statistics.mean(x['seconds'] for x in r['measurements'])
                      for r in group if r['seed'] in base]
            memory_reductions = [100*(1-statistics.mean(x['peak_allocated_mib'] for x in r['measurements'])/base_memory[r['seed']])
                                 for r in group if r['seed'] in base_memory]
            lines.append('| ' + ' | '.join([dataset,mode,str(len(group)),
                         *(fmt(v) for v in values.values()),fmt(ratios) if ratios else 'pending',
                         fmt(memory_reductions) if memory_reductions else 'pending']) + ' |')
    if any(r.get('shared_gpu') for r in records):
        lines.insert(2, '> 初测：GPU 存在其他项目进程，非独占计时，不能据此作正式加速结论；后续需独占复测。')
    output.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['stocks','etth','eeg','energy','electricity','weather','illness','exchange','traffic','fmri'])
    p.add_argument('--seeds',nargs='+',type=int,choices=[2026],default=[2026])
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=ROOT/'OUTPUT/efficiency_ablation')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--shared-gpu',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=True)
    # Suppress progress rendering consistently for every sampler.
    import Models.KPD.k_diffusion as module
    from functools import partial
    module.tqdm=partial(module.tqdm,disable=True)
    for dataset in a.datasets:
        cfg=yaml.safe_load((ROOT/f'Config/journal/{dataset}.yaml').read_text())
        for seed in a.seeds:
            path=ROOT/f'checkpoints/journal/Checkpoints_journal_{dataset}_seed{seed}_24/checkpoint-10.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            weights={key.removeprefix('ema_model.'):value for key,value in saved['ema'].items() if key.startswith('ema_model.')}
            model=Model(**cfg['model']['params'])
            model.load_state_dict(weights,strict=True)
            del saved, weights
            model=model.cuda().eval()
            original_stride=model.adaptive_stride_gain
            shape=(a.batch_size,model.seq_length,model.feature_size)
            order=list(a.modes)
            # Rotate ordering across seeds to reduce systematic order effects.
            offset=seed%len(order)
            order=order[offset:]+order[:offset]
            for mode in order:
                record_path=a.output/f'{dataset}__seed{seed}__{mode}.json'
                if record_path.exists(): continue
                model.adaptive_stride_gain=0.0 if mode=='reflection_fixed50' else original_stride
                model.sampling_timesteps=50
                calls=[0]
                handle=model.model.register_forward_hook(lambda *args: calls.__setitem__(0,calls[0]+1))
                def sample():
                    if mode=='legacy_full': return model.sample(shape)
                    if mode=='ddim50': return model.fast_sample(shape)
                    return model.adaptive_reflection_sample(shape,max_nfe=50)
                measurements=[]
                try:
                    for repeat in range(-1,a.repeats):
                        torch.manual_seed(seed+100000+repeat)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        calls[0]=0
                        start=time.perf_counter()
                        with torch.no_grad(): generated=sample()
                        torch.cuda.synchronize()
                        seconds=time.perf_counter()-start
                        allocated=torch.cuda.max_memory_allocated()/2**20
                        reserved=torch.cuda.max_memory_reserved()/2**20
                        if not torch.isfinite(generated).all(): raise ValueError('nonfinite generated samples')
                        del generated
                        if repeat>=0:
                            measurements.append(dict(seconds=seconds,samples_per_second=a.batch_size/seconds,
                                peak_allocated_mib=allocated,peak_reserved_mib=reserved,denoiser_calls=calls[0]))
                finally:
                    handle.remove()
                record=dict(dataset=dataset,seed=seed,mode=mode,measurements=measurements,
                    checkpoint=str(path.relative_to(ROOT)),batch_size=a.batch_size,shape=shape,
                    parameters=sum(x.numel() for x in model.parameters()),
                    parameter_mib=sum(x.numel()*x.element_size() for x in model.parameters())/2**20,
                    torch_version=torch.__version__,cuda_version=torch.version.cuda,
                    device=torch.cuda.get_device_name(),smoke=a.smoke,shared_gpu=a.shared_gpu,scope='sampling_only',
                    timestamp=time.time(),adaptive_stride_gain=model.adaptive_stride_gain)
                record_path.write_text(json.dumps(record,indent=2)+'\n')
                records=[json.loads(f.read_text()) for f in a.output.glob('*.json')]
                summarize(records,a.output/'EFFICIENCY_RESULTS.md')
                if not a.smoke:
                    summarize(records,ROOT/'EFFICIENCY_RESULTS.md')
                print(dataset,seed,mode,measurements,flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__=='__main__': main()
