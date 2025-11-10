import os
import torch
import numpy as np

from Layer.engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
root = ''
ori_path = root + 'etth_norm_truth_24_train.npy'
fake_path = root + '/ddpm_fake_ETTh.npy'
ori_data = np.load(ori_path)
fake_data = np.load(fake_path)
visualization(ori_data=ori_data, generated_data=fake_data, analysis='pca', compare=ori_data.shape[0])

visualization(ori_data=ori_data, generated_data=fake_data, analysis='tsne', compare=ori_data.shape[0])

visualization(ori_data=ori_data, generated_data=fake_data, analysis='kernel', compare=ori_data.shape[0])