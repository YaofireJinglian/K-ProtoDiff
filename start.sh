#!/bin/bash

# 定义变量
# root="/home/user/dyh/ts/Diffusion-TS/OUTPUT/"
# name="ETTh_x_x_mark"
# cfg="etth"

# # 运行训练命令
# python main.py --name $name --config_file ./Config/$cfg.yaml --gpu 0 --train

# # 运行样本生成命令
# python main.py --name $name --config_file ./Config/$cfg.yaml --gpu 0 --sample 0 --milestone 10

# # 运行评估命令，修正字符串拼接
# python ./Experiments/metric_pytorch.py --root "$root$cfg" \
# --ori_path "$root/samples/${cfg}_norm_truth_24_train.npy" \
# --fake_path "$root$name/ddpm_fake_${name}.npy"


root="/home/user/dyh/ts/Diffusion-TS/OUTPUT/"
name="ETTh"
cfg="etth"

# 运行训练命令
python main.py --name $name --config_file ./Config/$cfg.yaml --gpu 0 --train

# 运行样本生成命令
python main.py --name $name --config_file ./Config/$cfg.yaml --gpu 0 --sample 0 --milestone 0

# 运行评估命令，修正字符串拼接
# python ./Experiments/metric_pytorch.py --root "$root$cfg" \
# --ori_path "$root/samples/${cfg}_norm_truth_24_train.npy" \
# --fake_path "$root$name/ddpm_fake_${name}.npy"
