"""Predefined held-out journal ablations, isolated from main publication records."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATASETS=['stocks','eeg','traffic','fmri']
ART=ROOT/'checkpoints/journal_ablation'
VARIANTS={
    'full':{},
    'point':{'prototype_mode':'point'},
    'point24':{'prototype_mode':'point','num_prototypes':24},
    'scale4':{'prototype_scales':[4],'num_prototypes':24},
    'scale8':{'prototype_scales':[8],'num_prototypes':24},
    'scale16':{'prototype_scales':[16],'num_prototypes':24},
    'scale4_8':{'prototype_scales':[4,8],'num_prototypes':12},
    'no_aux':{'prototype_loss_weight':0.0},
    'no_consistency':{'prototype_consistency_weight':0.0},
    'no_balance':{'prototype_balance_weight':0.0},
    'no_diversity':{'prototype_diversity_weight':0.0},
}
BUDGETS=[10,20,50,100,200]


def samplers(variant):
    if variant=='full':
        return ['legacy_full','fixed_reflection50','no_correction50'] + [f'{mode}{n}' for mode in ['ddim','adaptive'] for n in BUDGETS]
    return ['legacy_full'] if variant in ['point','point24'] else ['adaptive50']


def groups():
    return {
        '核心模块': [('point','legacy_full'),('full','legacy_full'),('full','adaptive50')],
        '原型尺度（总数24；参数量不完全相等）':[(v,'adaptive50') for v in ['scale4','scale8','scale16','scale4_8','full']],
        '单尺度容量对照': [('point','legacy_full'),('point24','legacy_full'),('full','legacy_full')],
        '原型辅助损失':[(v,'adaptive50') for v in ['full','no_aux','no_consistency','no_balance','no_diversity']],
        '快速采样机制': [('full',s) for s in ['ddim50','fixed_reflection50','no_correction50','adaptive50']],
        '预算与质量权衡': [('full',f'{m}{n}') for n in BUDGETS for m in ['ddim','adaptive']],
    }


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def build_report():
    out=ROOT/'OUTPUT/journal_ablation'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        completed=list(ART.glob('*/*/*/record.json'))
        lines=['# Journal Ablation Results','',
          f'更新时间 UTC：{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}；质量评测完成 {len(completed)}/92。','',
          '> Stocks、EEG、Traffic、fMRI；seed 2026；各训练变体从头训练，步数等同该数据集期刊配置。',
          '> 固定时间划分：70% 训练、15% 验证、15% 测试；本轮只用固定验证样本（最多1024），不使用测试集，不据此宣称最终泛化结论。',
          '> 同一训练变体的采样对照共用 EMA 权重；固定生成 batch 32、FP32。单种子值保留三位小数，无跨种子标准差。',
          '> 秒数和显存为生成全部验证样本的在线观测，含逐批 CPU 输出复制，不是严格独占性能测试；正式效率见 EFFICIENCY_RESULTS.md 和 TRAINING_EFFICIENCY_RESULTS.md。',
          '> NFE 为每批去噪网络调用数的平均值。原型总数匹配不等于参数量完全匹配，因此另外报告参数量。',
          '> no_correction50 仅关闭反射修正强度，保留用于选择步长的原型门控计算。','',
          '检查点：`checkpoints/journal_ablation/`；配置：`Config/journal_ablation/`；失败不自动当作完成。','']
        keys=['Context-FID','KL','DS_mean','PS_mean','Segment-DTW-L6','Segment-DTW-L8','Segment-DTW-L10']
        for group,pairs in groups().items():
            lines += [f'## {group}','',
                '| Dataset | Variant | Sampler | Status | C-FID | KL | DS | PS | DTW6 | DTW8 | DTW10 | Seconds | Peak MiB | NFE/batch | Params M |',
                '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for dataset in DATASETS:
                for variant,sampler in pairs:
                    path=ART/dataset/variant/sampler/'record.json'
                    row=[dataset,variant,sampler]
                    if path.exists():
                        r=json.loads(path.read_text())
                        row+=['完成']+[f'{r["metrics"][k]:.3f}' for k in keys]
                        row += [f'{r[k]:.3f}' for k in ['seconds','peak_mib','nfe_per_batch','parameters_m']]
                    else:
                        state_path=ART/dataset/variant/'status.json'
                        state=json.loads(state_path.read_text()).get('status','待运行') if state_path.exists() else '待运行'
                        row+=[state]+['—']*11
                    lines.append('| '+' | '.join(row)+' |')
            lines.append('')
        destination=ROOT/'ABLATION_RESULTS.md'
        temporary=destination.with_suffix('.md.tmp')
        temporary.write_text('\n'.join(lines))
        temporary.replace(destination)


def initialize():
    for dataset in DATASETS:
        base=yaml.safe_load((ROOT/f'Config/journal/{dataset}.yaml').read_text())
        spec=dict(dataset=dataset,seed=2026,base_config=f'Config/journal/{dataset}.yaml',
            training_steps=base['solver']['max_epochs'],validation_samples=1024,sampling_batch=32,
            evaluation_split='validation',training_variants=VARIANTS,
            sampler_grid={v:samplers(v) for v in VARIANTS})
        path=ROOT/f'Config/journal_ablation/{dataset}.yaml'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists(): path.write_text(yaml.safe_dump(spec,sort_keys=False))
    build_report()


def run_job(dataset,variant,smoke=False):
    import numpy as np
    import torch
    from journal_search import prepare_data,train_model
    torch.set_num_threads(4)
    spec=yaml.safe_load((ROOT/f'Config/journal_ablation/{dataset}.yaml').read_text())
    base=yaml.safe_load((ROOT/spec['base_config']).read_text())
    config=copy.deepcopy(base)
    config['model']['params'].update(spec['training_variants'][variant])
    out=(ROOT/'OUTPUT/journal_ablation_smoke' if smoke else ART)/dataset/variant
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists(): return
        try:
            save(out/'status.json',{'status':'训练中','updated':time.time()})
            if not smoke: build_report()
            (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            data,val=prepare_data(base,out,32 if smoke else spec['validation_samples'])
            model=train_model(config,data,out,2 if smoke else spec['training_steps'],spec['seed'])
            selected=spec['sampler_grid'][variant]
            if smoke: selected=['adaptive2'] if variant!='point' else ['ddim2']
            for sampler in selected:
                dst=out/sampler
                dst.mkdir(exist_ok=True)
                if (dst/'record.json').exists(): continue
                save(out/'status.json',{'status':f'采样/评测 {sampler}','updated':time.time()})
                if not smoke: build_report()
                p=config['model']['params']
                model.reflection_strength=p.get('reflection_strength',0.1)
                model.adaptive_stride_gain=p.get('adaptive_stride_gain',1.0)
                if sampler=='fixed_reflection50': model.adaptive_stride_gain=0
                if sampler=='no_correction50': model.reflection_strength=0
                import re
                budget=int(re.search(r'\d+$',sampler)[0]) if sampler!='legacy_full' else model.num_timesteps
                model.sampling_timesteps=budget
                info_path=dst/'sampling.json'
                if not (dst/'samples.npy').exists() or not info_path.exists():
                    calls=[0]
                    hook=model.model.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
                    pieces=[]
                    torch.manual_seed(spec['seed']+10000)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    try:
                        with torch.no_grad():
                            for offset in range(0,len(val),spec['sampling_batch']):
                                shape=(min(spec['sampling_batch'],len(val)-offset),model.seq_length,model.feature_size)
                                if sampler=='legacy_full': fake=model.sample(shape)
                                elif sampler.startswith('ddim'): fake=model.fast_sample(shape)
                                else: fake=model.adaptive_reflection_sample(shape,max_nfe=budget)
                                pieces.append(fake.cpu().numpy())
                                del fake
                        torch.cuda.synchronize()
                    finally: hook.remove()
                    info=dict(seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,
                        nfe_per_batch=calls[0]/len(pieces),parameters_m=sum(p.numel() for p in model.parameters())/1e6)
                    fake=(np.concatenate(pieces)+1)/2
                    if fake.shape!=val.shape or not np.isfinite(fake).all(): raise ValueError('Invalid samples')
                    np.save(dst/'samples.npy',fake)
                    save(info_path,info)
                if smoke: continue
                info=json.loads(info_path.read_text())
                env=os.environ.copy()
                for key in ['TF_USE_LEGACY_KERAS','LD_LIBRARY_PATH','REQUIRE_TF_GPU','XLA_FLAGS','TF_XLA_FLAGS']: env.pop(key,None)
                with (dst/'evaluation.log').open('a') as log:
                    subprocess.run([str(ROOT/'.venv/bin/python'),'-u','scripts/eval_seeded.py','--root',str(dst),
                        '--ori_path',str(out/'validation.npy'),'--fake_path',str(dst/'samples.npy'),
                        '--n_iter','1','--seed',str(spec['seed'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                metrics={k:float(v) for k,v in (line.split(':',1) for line in (dst/'metrics.txt').read_text().splitlines())}
                if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite metrics')
                save(dst/'record.json',dict(dataset=dataset,variant=variant,sampler=sampler,seed=spec['seed'],
                    metrics=metrics,split='validation',train_steps=spec['training_steps'],**info))
                build_report()
            save(out/'complete.json',{'smoke':smoke,'completed_at':time.time()})
            save(out/'status.json',{'status':'完成','updated':time.time()})
        except Exception as exc:
            save(out/'status.json',{'status':'失败待检查','error':str(exc),'updated':time.time()})
            raise
        finally:
            if not smoke: build_report()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--init',action='store_true')
    p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--variant',choices=list(VARIANTS))
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.init: initialize(); return
    if not a.dataset or not a.variant: p.error('--dataset and --variant required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    run_job(a.dataset,a.variant,a.smoke)


if __name__=='__main__': main()
