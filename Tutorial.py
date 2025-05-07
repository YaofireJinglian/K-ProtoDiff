import os
import torch
import numpy as np

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

# class Args_Example:
#     def __init__(self) -> None:
#         self.config_path = './Config/sines.yaml'
#         self.save_dir = './toy_exp'
#         self.gpu = 0
#         os.makedirs(self.save_dir, exist_ok=True)

# args =  Args_Example()
# configs = load_yaml_config(args.config_path)
# device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# dl_info = build_dataloader(configs, args)
# model = instantiate_from_config(configs['model']).to(device)
# trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
# trainer.train()
# dataset = dl_info['dataset']
# seq_length, feature_dim = dataset.window, dataset.var_num

root = '/home/user/dyh/ts/Diffusion-TS/OUTPUT/ETTh_x_x_mark_best_ts_500'
ori_path = root + '/samples/etth_norm_truth_24_train.npy'
fake_path = root + '/ddpm_fake_ETTh_x_x_mark_best_ts_500.npy'
ori_data = np.load(ori_path)
fake_data = np.load(fake_path)
# if dataset.auto_norm:
#     fake_data = unnormalize_to_zero_to_one(fake_data)
#     np.save(os.path.join(args.save_dir, f'ddpm_fake_sines.npy'), fake_data)

visualization(ori_data=ori_data, generated_data=fake_data, analysis='pca', compare=ori_data.shape[0])

visualization(ori_data=ori_data, generated_data=fake_data, analysis='tsne', compare=ori_data.shape[0])

visualization(ori_data=ori_data, generated_data=fake_data, analysis='kernel', compare=ori_data.shape[0])