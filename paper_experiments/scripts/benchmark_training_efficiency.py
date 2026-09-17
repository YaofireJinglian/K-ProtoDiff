"""Controlled training-update microbenchmark; not a full convergence experiment."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from Models.KPD.k_diffusion import Model

DATASETS = ['stocks','etth','eeg','energy','electricity','weather','illness','exchange','traffic','fmri']


def report(directory, destination):
    records = [json.loads(f.read_text()) for f in directory.glob('*.json')]
    records = [r for r in records if r['seed'] == 2026 and not r.get('smoke', False)]
    lines = ['# Training Efficiency Ablation', '',
        '> 单尺度 point 与 multiscale（含原型辅助损失）对照；其余使用同一数据集期刊配置。不是原版会议训练的逐项复现。',
        '> FP32，Adam，包含梯度累积、梯度裁剪和参数更新；不含数据读取/传输、EMA、调度器、评测及保存。输入批次预驻 GPU。',
        '> 正式测量：预热 10 次更新，再实测 100 次更新。该总耗时不是完整训练耗时，不能作为收敛速度结论。',
        '> 仅使用 seed 2026，报告单次测量值，三位小数，不报告跨种子标准差。显存包含输入缓存、模型、优化器、梯度及激活，不含 CUDA 驱动开销。', '',
        '| Dataset | Prototype | Seeds | Effective batch | ms/update | 100 updates (s) | Samples/s | Peak allocated MiB | Peak reserved MiB | Parameters (M) | Time overhead % | Memory overhead % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(values):
        return f'{statistics.mean(values):.3f}'
    for dataset in DATASETS:
        base = {r['seed']: r for r in records if r['dataset']==dataset and r['mode']=='point'}
        for mode in ['point','multiscale']:
            group = sorted([r for r in records if r['dataset']==dataset and r['mode']==mode],key=lambda r:r['seed'])
            if not group: continue
            cells=[dataset,mode,str(len(group)),str(group[0]['effective_batch'])]
            for key in ['ms_per_update','seconds','samples_per_second','peak_allocated_mib','peak_reserved_mib','parameters_m']:
                cells.append(fmt([r[key] for r in group]))
            for key in ['ms_per_update','peak_allocated_mib']:
                overhead=[100*(r[key]/base[r['seed']][key]-1) for r in group if r['seed'] in base]
                cells.append(fmt(overhead) if overhead else 'pending')
            lines.append('| '+' | '.join(cells)+' |')
    if any(r.get('shared_gpu') for r in records):
        lines.insert(2, '> 初测：GPU 存在其他项目进程，非独占计时，不能据此作正式加速结论；后续需独占复测。')
    destination.write_text('\n'.join(lines)+'\n')


def measure(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cfg=yaml.safe_load((ROOT/f'Config/journal/{args.dataset}.yaml').read_text())
    params=dict(cfg['model']['params'])
    params['prototype_mode']=args.mode
    params['sampling_mode']='legacy'  # Never invoked during training.
    batch=cfg['dataloader']['batch_size']
    accumulation=cfg['solver']['gradient_accumulate_every']
    if args.smoke: batch=min(batch,2)
    data_cfg=yaml.safe_load((ROOT/f'Config/baselines/{args.dataset}.yaml').read_text())
    data=np.load(ROOT/data_cfg['real_path'],mmap_mode='r')
    # Formal shared truth arrays use [0,1]; training uses [-1,1].
    rng=np.random.default_rng(args.seed)
    indices=rng.integers(0,len(data),size=(8,batch))
    pool=torch.as_tensor(np.array(data[indices],dtype=np.float32,copy=True),device='cuda')*2-1
    model=Model(**params).cuda().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['solver']['base_lr'],betas=(0.9,0.96))
    losses=[]
    def update(step):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            x=pool[(step*accumulation+micro)%len(pool)]
            loss=model(x,target=x)/accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        return loss.detach()
    for step in range(args.warmup): update(step)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for step in range(args.updates): losses.append(update(step))
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated()/2**20
    reserved=torch.cuda.max_memory_reserved()/2**20
    if not torch.isfinite(torch.stack(losses)).all(): raise ValueError('Nonfinite training loss')
    record=dict(dataset=args.dataset,mode=args.mode,seed=args.seed,updates=args.updates,warmup=args.warmup,
        micro_batch=batch,gradient_accumulation=accumulation,effective_batch=batch*accumulation,
        seconds=seconds,ms_per_update=seconds*1000/args.updates,
        samples_per_second=args.updates*batch*accumulation/seconds,
        peak_allocated_mib=peak,peak_reserved_mib=reserved,
        parameters_m=sum(p.numel() for p in model.parameters())/1e6,
        parameter_mib=sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        timestamp=time.time(),smoke=args.smoke,shared_gpu=args.shared_gpu,model_params=params,scope='100_update_training_microbenchmark',
        data_path=data_cfg['real_path'],input_pool_batches=8)
    dest=args.output/f'{args.dataset}__{args.mode}__seed{args.seed}.json'
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--mode',choices=['point','multiscale'])
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--output',type=Path,default=ROOT/'OUTPUT/training_efficiency')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--shared-gpu',action='store_true')
    a=p.parse_args()
    if a.updates<1 or a.warmup<1: p.error('updates and warmup must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.dataset:
        if not a.mode: p.error('--mode required with --dataset')
        measure(a)
        return
    for dataset in DATASETS:
        for seed in [2026]:
            modes=['point','multiscale'] if seed%2==0 else ['multiscale','point']
            for mode in modes:
                dest=a.output/f'{dataset}__{mode}__seed{seed}.json'
                if not dest.exists():
                    subprocess.run([sys.executable,'-u',__file__,'--dataset',dataset,'--mode',mode,
                        '--seed',str(seed),'--output',str(a.output)]+(['--shared-gpu'] if a.shared_gpu else []),check=True,cwd=ROOT)
                report(a.output,ROOT/'TRAINING_EFFICIENCY_RESULTS.md')


if __name__=='__main__': main()
