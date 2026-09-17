"""Generate a compact, reproducible K-ProtoDiff-J sample set for analysis."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ema_pytorch import EMA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.io_utils import instantiate_from_config, seed_everything


def checkpoint_path(dataset, seed, window, manifest):
    info = manifest["datasets"][dataset]
    if info["reuse_original_checkpoint"]:
        base = ROOT / "checkpoints/journal" / f"Checkpoints_journal_{dataset}_seed{seed}_{window}"
    else:
        base = ROOT / "checkpoints/journal_tuned" / f"Checkpoints_journal_tuned_{dataset}_seed{seed}_{window}"
    return base / "checkpoint-10.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stocks", "exchange", "fmri"], required=True)
    parser.add_argument("--seed", type=int, choices=[2026, 2027, 2028], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    config_path = ROOT / "Config/journal_tuned" / f"{args.dataset}.yaml"
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads((ROOT / "Config/journal_tuned/manifest.json").read_text())
    params = config["dataloader"]["train_dataset"]["params"]
    window = int(params["window"])
    checkpoint = checkpoint_path(args.dataset, args.seed, window, manifest)
    destination = (ROOT / "OUTPUT/qualitative_downstream/generated" / args.dataset /
                   "k_protodiff_j" / f"seed{args.seed}")
    destination.mkdir(parents=True, exist_ok=True)
    sample_path = destination / "samples.npy"
    metadata_path = destination / "metadata.json"
    if sample_path.exists() and metadata_path.exists():
        print(f"EXISTS {sample_path}", flush=True)
        return
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    seed_everything(args.seed + 10000, cudnn_deterministic=True)
    model = instantiate_from_config(config["model"]).to(device)
    ema_cfg = config["solver"]["ema"]
    ema = EMA(model, beta=ema_cfg["decay"], update_every=ema_cfg["update_interval"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    generator = ema.ema_model.eval()

    batch_size = int(config["dataloader"]["sample_size"])
    chunks = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, args.num_samples, batch_size):
            size = min(batch_size, args.num_samples - offset)
            chunks.append(generator.generate_mts(batch_size=size).cpu().numpy())
            print(f"{args.dataset} seed{args.seed}: {offset + size}/{args.num_samples}", flush=True)
    elapsed = time.perf_counter() - started
    samples = np.concatenate(chunks, axis=0).astype(np.float32)
    samples = np.clip((samples + 1.0) / 2.0, 0.0, 1.0)
    if samples.shape != (args.num_samples, config["model"]["params"]["seq_length"],
                         config["model"]["params"]["feature_size"]):
        raise ValueError(f"Unexpected output shape: {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("Generated samples contain non-finite values")
    temporary = sample_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, samples)
    os.replace(temporary, sample_path)
    metadata = {
        "dataset": args.dataset,
        "method": "K-ProtoDiff-J",
        "seed": args.seed,
        "num_samples": args.num_samples,
        "shape": list(samples.shape),
        "scale": "[0, 1]",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "sampling_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"COMPLETE {sample_path} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
