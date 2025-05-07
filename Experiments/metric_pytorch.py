import os
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss
from Utils.dwt import dtw_metric
from Utils.k_l import kl_divergence_hist
data_list = ['Electricity','Energy','ETTh','Exchange','Illness','Stocks','Weather',"EEG",'Traffic',]

# data_list = ['Electricity','Energy','ETTh','Exchange','Illness','Stocks','Weather',"EEG"]
for data_name in data_list:
    # root = '/home/user/dyh/ts/PaD-TS/OUTPUT/' + data_name + '_24'
    # ori_path = '/home/user/dyh/ts/PaD-TS/OUTPUT/samples/'+ data_name + '_norm_truth_24_train.npy'
    # fake_path = root + '/ddpm_fake_' + data_name + '_24.npy'

    # ========================================================================
    # root = '/home/user/dyh/ts/Diffusion-TS/OUTPUT/' + data_name
    # ori_path = root + '/samples/' + data_name + '_norm_truth_24_train.npy'
    # fake_path = root + '/ddpm_fake_' + data_name + '.npy'
    # ========================================================================
    # root = '/home/user/dyh/ts/TimeGAN/OUTPUT/' + data_name
    # ori_path = root + '/timegen_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/timegen_fake_'+ data_name + '_24.npy'
    #=========================================================================
    # root = '/home/user/dyh/ts/timeVAE-pytorch-main/OUTPUT/' + data_name
    # ori_path = root + '/timeVAE_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/timeVAE_fake_'+ data_name + '_24.npy'
    #=========================================================================
    # root = '/home/user/dyh/ts/GT-GAN/OUTPUT/' + data_name
    # ori_path = root + '/GT-GAN_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/GT-GAN_fake_'+ data_name + '_24.npy'
    #=========================================================================
    root = '/home/user/dyh/ts/TimeVQVAE/OUTPUT/' + data_name
    ori_path = root + '/TimeVQVAE_truth_'+ data_name + '_24.npy'
    fake_path = root + '/TimeVQVAE_fake_'+ data_name + '_24.npy'

    ori_data = np.load(ori_path)
    fake_data = np.load(fake_path)
    print("===================================="+root+"====================================")
    print(ori_data.shape,fake_data.shape)
    iterations = 5
    context_fid_score = []
    dwt = dtw_metric(ori_data, fake_data)
    k_l = kl_divergence_hist(ori_data, fake_data)
    print("dwt:",dwt)
    print("kl:",k_l)
    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        # context_fid = Context_FID(ori_data[:fake_data.shape[0]], fake_data[:])
        context_fid_score.append(context_fid)
        print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')

    def random_choice(size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    if data_name != 'Traffic':
        size = int(x_real.shape[0] / iterations)

        for i in range(iterations):
            real_idx = random_choice(x_real.shape[0], size)
            fake_idx = random_choice(x_fake.shape[0], size)
            corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
            loss = corr.compute(x_fake[fake_idx, :, :])
            correlational_score.append(loss.item())
            print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')
        cs = display_scores(correlational_score)
    else:
        cs = 'null'
    cfs = display_scores(context_fid_score)
    
        



    # 汇总结果
    result_text = (
        f"Original Data Path: {ori_path}\n"
        f"Fake Data Path: {fake_path}\n\n"
        f"context_fid_score: {cfs}\n"
        f"correlational_score: {cs}\n"
        f"dynamic_time_warping: {dwt}\n"
        f"kullback_leibler_divergence: {k_l}\n"
    )

    # 打开文件并保存结果
    with open(os.path.join(root, 'result_dtw_norm.txt'), 'w') as file:
        file.write(result_text)

    # 打印输出到控制台
    print(result_text)
