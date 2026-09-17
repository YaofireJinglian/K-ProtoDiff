"""Priority queue for missing independent-seed uncertainty, owned by watchdog."""
import json
import time
import yaml

DATASETS = ['stocks','illness','exchange','etth','eeg','energy','electricity','weather','traffic']
METHODS = ['timevae','diffusion_ts','k_protodiff','timegan','pad_ts','timevqvae','gtgan']
METRICS = {'C-FID','KL','DS','PS','Seg-DTW-L6','Seg-DTW-L8','Seg-DTW-L10'}


def entries():
    yield from [('fmri','gtgan',seed) for seed in [2026,2027]]
    for method in METHODS:
        for dataset in DATASETS:
            for seed in [2026,2027,2028]:
                yield dataset, method, seed


def tick(root, state, active, gpus, occupied, launch, no_launch, python_path=None):
    jobs = state.setdefault('uncertainty_jobs', {})
    policy_path=root/'Config/journal_runtime.yaml'
    policy=yaml.safe_load(policy_path.read_text()) if policy_path.exists() else {}
    resume=policy.get('baseline_resume_request')
    if resume and state.get('baseline_resume_consumed') != resume:
        for key, job in jobs.items():
            if job['status'] == 'paused_by_user':
                job.setdefault('resume_history', []).append(
                    {'request':resume, 'attempts':job.get('attempts', 0)})
                job.update(status='pending', attempts=0)
        state['baseline_resume_consumed']=resume
    cancel=policy.get('baseline_cancel_request')
    if cancel and state.get('baseline_cancel_consumed') != cancel:
        methods=set(policy.get('baseline_cancel_methods',[]))
        for dataset,method,seed in entries():
            if method not in methods:
                continue
            key=f'{dataset}__{method}__seed{seed}'
            job=jobs.setdefault(key,{'status':'pending','attempts':0})
            record=root/'OUTPUT/main_results_records'/f'{key}.json'
            if record.exists() and METRICS.issubset(json.loads(record.read_text()).get('metrics',{})):
                continue
            job.setdefault('cancel_history',[]).append(
                {'request':cancel,'previous_status':job.get('status'),'recorded_at':time.time()})
            job.update(status='skipped_by_user',cancel_request=cancel)
        state['baseline_cancel_consumed']=cancel
    recovery=policy.get('baseline_recovery_request')
    if recovery and state.get('baseline_recovery_consumed') != recovery:
        for key,job in jobs.items():
            if job['status']!='failed_needs_review':
                continue
            targets=policy.get('baseline_recovery_targets')
            if targets is not None and key not in targets:
                continue
            log=root/'OUTPUT/baseline_reproduction_logs'/f'{key}.log'
            if not log.exists():
                continue
            tail=log.read_text(errors='replace')[-15000:]
            errors=['CUDA error','CUDA unknown','double free','FillPhiloxRandom']
            if targets is not None:
                errors.append('CUDA out of memory')
            if any(error in tail for error in errors):
                job.setdefault('recovery_history',[]).append(dict(request=recovery,attempts=job['attempts']))
                job.update(status='pending',attempts=0)
        state['baseline_recovery_consumed']=recovery
    ready = []
    for dataset, method, seed in entries():
        key = f'{dataset}__{method}__seed{seed}'
        job = jobs.setdefault(key, {'status':'pending','attempts':0})
        record = root/'OUTPUT/main_results_records'/f'{key}.json'
        if record.exists() and METRICS.issubset(json.loads(record.read_text()).get('metrics',{})):
            job['status'] = 'complete'
            continue
        if job['status'] == 'skipped_by_user':
            continue
        name = 'kpd_uncertainty_'+key
        if name in active:
            job['status'] = 'running'
            occupied.add(job['gpu'])
            continue
        if job['status'] == 'running':
            job['status'] = 'failed_needs_review'
        if job['attempts'] or job['status'] == 'failed_needs_review':
            continue
        ready.append((dataset,method,seed,key,name,job))
    if policy.get('baseline_interleave',False):
        # Start slow families early; cap GT-GAN to leave capacity for other work.
        rank={'gtgan':0,'timegan':1,'diffusion_ts':2,'pad_ts':3,'k_protodiff':4,'timevqvae':5,'timevae':6}
        running={m:sum(j['status']=='running' and f'__{m}__' in k for k,j in jobs.items()) for m in METHODS}
        ready.sort(key=lambda e:(running[e[1]],rank[e[1]],DATASETS.index(e[0]) if e[0] in DATASETS else -1,e[2]))
        balanced=[]
        while ready:
            ready.sort(key=lambda e:(running[e[1]],rank[e[1]]))
            entry=ready.pop(0)
            if entry[1]=='gtgan' and running['gtgan']>=2:
                continue
            balanced.append(entry)
            running[entry[1]]+=1
        ready=balanced
    if not no_launch:
        for gpu, entry in zip([r['gpu'] for r in gpus if r['gpu'] not in occupied], ready):
            dataset,method,seed,key,name,job = entry
            command = (['bash','scripts/run_tf_reproduction.sh'] if method in ['timevae','timegan'] else
                       [python_path or root/'.venv/bin/python','-u','scripts/run_baseline_reproduction.py'])
            command += ['--dataset',dataset,'--method',method,'--seed',seed,'--gpu',gpu]
            log = root/'OUTPUT/baseline_reproduction_logs'/f'{key}.log'
            launch(name,command,log)
            job.update(status='running',gpu=gpu,attempts=1,launched_at=time.time(),log=str(log))
            occupied.add(gpu)
            active.add(name)
    lines = ['# 基线三种子不确定性补跑','',
             '固定种子 2026/2027/2028；每格报告真实复现均值 ± 样本标准差，三位小数。',
             '使用统一数据/评测及固定迁移超参，不宣称逐值恢复论文实验。',
             'fMRI GT-GAN 两个失败种子从头重跑，旧权重保留。失败任务不盲目自动重启。','',
             f'完成 {sum(j["status"]=="complete" for j in jobs.values())}/{len(jobs)}；'
             f'运行 {sum(j["status"]=="running" for j in jobs.values())}；'
             f'失败待检查 {sum(j["status"]=="failed_needs_review" for j in jobs.values())}；'
             f'用户取消 {sum(j["status"]=="skipped_by_user" for j in jobs.values())}。','',
             '| 任务 | 状态 | GPU |','|---|---|---:|']
    for key,job in jobs.items():
        lines.append(f'| {key} | {job["status"]} | {job.get("gpu","—")} |')
    (root/'BASELINE_UNCERTAINTY_STATUS.md').write_text('\n'.join(lines)+'\n')
