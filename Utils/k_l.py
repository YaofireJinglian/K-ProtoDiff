import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def kl_divergence_hist(ori_data, fake_data, bins=100):
    # Flatten sequences: (N, T, D) → (N, T*D)
    ori_flat = ori_data.reshape(len(ori_data), -1)
    fake_flat = fake_data.reshape(len(fake_data), -1)

    # Concatenate all samples into one 1D array
    ori_all = ori_flat.flatten()
    fake_all = fake_flat.flatten()

    # Estimate histogram for each
    p_hist, bin_edges = np.histogram(ori_all, bins=bins, density=True)
    q_hist, _ = np.histogram(fake_all, bins=bin_edges, density=True)

    # Avoid division by zero
    p_hist += 1e-8
    q_hist += 1e-8

    kl_div = entropy(p_hist, q_hist)  # KL(P || Q)
    return kl_div

# 使用方式
# ori_data = np.load('../OUTPUT/ETTh/samples/etth_norm_truth_24_train.npy')
# fake_data = np.load('../OUTPUT/ETTh/ddpm_fake_ETTh.npy')

# kl_score = kl_divergence_hist(ori_data, fake_data)
# print("KL Divergence Score:", kl_score)
