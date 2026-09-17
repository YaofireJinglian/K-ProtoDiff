"""Seed the unchanged shared evaluator, including its TS2Vec initialization."""
import random
import runpy
import sys
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
seed = int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
runpy.run_path(str(root/'metric_pytorch.py'),run_name='__main__')
