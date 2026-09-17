"""One reserved GPU, resume-safe predefined ablations, no main-table writes."""
import fcntl
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml
from journal_ablation import ROOT,ART,DATASETS,VARIANTS,save,build_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,choices=range(8),required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    args=parser.parse_args()
    gpu=args.gpu
    directory=ROOT/'OUTPUT/journal_ablation'
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/f'queue_gpu{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy_path=ROOT/'Config/journal_runtime.yaml'
    policy=yaml.safe_load(policy_path.read_text())
    if gpu not in set(policy.get('ablation_reserved_gpus',[])) or gpu in policy['allowed_gpus']:
        raise RuntimeError(f'GPU {gpu} is not reserved')
    child=None
    def stop(sig,frame): raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]: signal.signal(sig,stop)
    try:
        print(f'Waiting for existing GPU {gpu} processes to finish; no existing jobs interrupted.',flush=True)
        while True:
            uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            if uuid not in apps: break
            time.sleep(30)
        failures=[]
        # Prioritize core and sampler mechanisms across all four datasets.
        for variant in VARIANTS:
            for dataset in args.datasets:
                out=ART/dataset/variant
                if (out/'complete.json').exists(): continue
                state=out/'status.json'
                if state.exists():
                    import json
                    if json.loads(state.read_text()).get('status')=='失败待检查':
                        failures.append([dataset,variant,'previous failure'])
                        continue
                command=[str(ROOT/'.venv-cu130/bin/python'),'-u','scripts/journal_ablation.py',
                         '--dataset',dataset,'--variant',variant,'--gpu',str(gpu)]
                print('START',dataset,variant,flush=True)
                with (directory/f'{dataset}_{variant}.log').open('a') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait()
                if code: failures.append([dataset,variant,code])
                save(directory/'queue_status.json',{'last_job':[dataset,variant],'failures':failures,'updated':time.time()})
                build_report()
        print('Queue exhausted; failures:',failures,flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            child.wait()
        with (directory/'policy.lock').open('a') as policy_lock:
            fcntl.flock(policy_lock,fcntl.LOCK_EX)
            policy=yaml.safe_load(policy_path.read_text())
            reserved=set(policy.get('ablation_reserved_gpus',[]))
            if gpu in reserved:
                reserved.remove(gpu)
                policy['ablation_reserved_gpus']=sorted(reserved)
                policy['allowed_gpus']=sorted(set(policy['allowed_gpus'])|{gpu})
                policy['baseline_allowed_gpus']=sorted(set(policy.get('baseline_allowed_gpus',[]))|{gpu})
                tmp=policy_path.with_suffix(f'.ablation_gpu{gpu}.tmp')
                tmp.write_text(yaml.safe_dump(policy,sort_keys=False))
                tmp.replace(policy_path)
        print(f'GPU {gpu} returned to main scheduler.',flush=True)


if __name__=='__main__': main()
