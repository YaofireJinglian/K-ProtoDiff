"""Fixed-three-seed baseline reruns; never attach new SDs to published means."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def diffusion(x, cfg, out, method, smoke):
    import torch
    import yaml
    from types import SimpleNamespace
    if method == 'diffusion_ts':
        sys.path.insert(0, str(ROOT / 'baselines/Diffusion-TS'))
        from engine.solver import Trainer
        from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS as Model
    else:
        sys.path.insert(0, str(ROOT))
        from Layer.engine.solver import Trainer
        from Models.KPD.k_diffusion import Model
    config = copy.deepcopy(cfg['config'])
    config['solver']['results_folder'] = str(out / 'model')
    if smoke:
        config['solver'].update(max_epochs=1, save_cycle=1)
        config['model']['params'].update(timesteps=10, sampling_timesteps=10)
    model = Model(**config['model']['params']).cuda()
    free, _ = torch.cuda.mem_get_info()
    resident = x.nbytes < min(free // 8, 2 * 1024**3)
    data = torch.tensor(x * 2 - 1, device='cuda' if resident else 'cpu')
    loader = torch.utils.data.DataLoader(data,
        batch_size=min(len(x), config['dataloader']['batch_size']), shuffle=True,
        pin_memory=not resident)
    trainer = Trainer(config, SimpleNamespace(name=method), model, {'dataloader':loader})
    final = config['solver']['max_epochs'] // config['solver']['save_cycle']
    checkpoint = trainer.results_folder / f'checkpoint-{final}.pt'
    if checkpoint.exists():
        trainer.load(final)
    else:
        from fast_baseline_diffusion import train
        train(trainer,x,out)
    from fast_baseline_diffusion import sample
    return sample(trainer,len(x),min(len(x),cfg['sample_size']),list(x.shape[1:]),out)


def pad(x, cfg, out, smoke):
    import torch
    sys.path.insert(0, str(ROOT / 'baselines/PaD-TS'))
    from Model import PaD_TS
    from training import Trainer
    from diffmodel_init import create_gaussian_diffusion
    from resample import Batch_Same_Sampler
    from data_preprocessing.sampling import sampling
    model = PaD_TS(input_shape=tuple(x.shape[1:]), **cfg['model'])
    options = dict(cfg['diffusion'])
    if smoke:
        options['diffusion_steps'] = 10
    diffusion = create_gaussian_diffusion(**options)
    batch = min(len(x), cfg['batch_size'])
    loader = torch.utils.data.DataLoader(torch.tensor(x * 2 - 1), batch_size=batch,
                                        shuffle=True, drop_last=True)
    train = dict(cfg['training'])
    if smoke:
        train.update(lr_anneal_steps=1, save_interval=1)
    trainer = Trainer(model=model, diffusion=diffusion, data=loader, batch_size=batch,
                      schedule_sampler=Batch_Same_Sampler(diffusion), save_dir=str(out)+'/', **train)
    checkpoints = sorted(out.glob('model_*.pt'))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location='cuda', weights_only=False)
        trainer.model.load_state_dict(state['model_state_dict'])
        trainer.opt.load_state_dict(state['opt_state_dict'])
        trainer.step = int(state['step'])
    trainer.train()
    return (sampling(model, diffusion, len(x), x.shape[1], x.shape[2], batch)[:len(x)].cpu().numpy()+1)/2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', required=True, choices=['timevae','timegan','timevqvae','gtgan','diffusion_ts','pad_ts','k_protodiff'])
    p.add_argument('--seed', type=int, choices=[2026,2027,2028], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='2')
    if a.method == 'timegan':
        # Same XLA option already benchmarked by the fMRI TimeGAN adapter.
        os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2'
    import numpy as np
    import yaml
    from run_remaining_fmri import timevae, timegan, timevqvae, gtgan, evaluate
    spec = yaml.safe_load((ROOT/'Config/baselines'/f'{a.dataset}.yaml').read_text())
    if a.seed not in spec['seeds']:
        raise ValueError('Unregistered seed')
    root = ROOT/('OUTPUT/baseline_reproduction_smoke' if a.smoke else 'checkpoints/baseline_reproduction')
    out = root/a.dataset/a.method/f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = ROOT/'OUTPUT/main_results_records'/f'{a.dataset}__{a.method}__seed{a.seed}.json'
        if not a.smoke and record.exists():
            print('Already complete:', record, flush=True)
            return
        cfg = dict(spec[a.method], seed=a.seed)
        random.seed(a.seed)
        np.random.seed(a.seed)
        if a.method not in ['timevae','timegan']:
            import torch
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
        real = ROOT/spec['real_path']
        x = np.load(real).astype(np.float32)
        if a.smoke:
            x = x[:32]
        fake = out/'samples.npy'
        metadata = {'dataset':spec['dataset'], 'method':a.method, 'seed':a.seed,
                    'parameters':cfg, 'smoke':a.smoke, 'real_path':str(real),
                    'data_shape':list(x.shape), 'protocol':'local full-data baseline reproduction; not paper-exact metrics',
                    'launched_at':datetime.now(timezone.utc).isoformat(),
                    'execution_version':'baseline-throughput-v1',
                    'tf_xla':a.method=='timegan',
                    'sampling_protocol':'fixed chunk seeds for diffusion_ts/k_protodiff'}
        previous = out/'run_config.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            for key in ['parameters','smoke','real_path','data_shape']:
                if old[key] != metadata[key]:
                    raise ValueError(f'Immutable run configuration changed: {key}')
        else:
            previous.write_text(json.dumps(metadata, indent=2)+'\n')
        if not fake.exists():
            print('TRAIN', a.dataset, a.method, a.seed, x.shape, flush=True)
            if a.method in ['diffusion_ts','k_protodiff']:
                result = diffusion(x, cfg, out, a.method, a.smoke)
            elif a.method == 'pad_ts':
                result = pad(x, cfg, out, a.smoke)
            else:
                result = {'timevae':timevae,'timegan':timegan,'timevqvae':timevqvae,'gtgan':gtgan}[a.method](x,cfg,out,a.smoke)
            result = np.asarray(result)
            if result.shape != x.shape or not np.isfinite(result).all():
                raise ValueError(f'Invalid samples: {result.shape} vs {x.shape}')
            np.save(fake, result)
        if not a.smoke:
            evaluate(a.method, a.seed, fake, real, ROOT/'OUTPUT/baseline_reproduction_metrics'/a.dataset/a.method/f'seed{a.seed}',
                     dataset=spec['dataset'], dataset_key=a.dataset)
        print('COMPLETED', a.dataset, a.method, a.seed, 'smoke', a.smoke, flush=True)


if __name__ == '__main__':
    main()
