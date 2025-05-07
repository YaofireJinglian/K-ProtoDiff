import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed

def multi_dim_dtw(seq1, seq2):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    if len(seq1) == 0 or len(seq2) == 0:
        raise ValueError("Input sequences cannot be empty.")
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

def dtw_metric(ori_data, fake_data, n_jobs=-1):
    # 显式检查数据形状
    # print("Original data shape:", ori_data.shape)
    # print("Fake data shape:", fake_data.shape)
    
    if ori_data.shape[0] != fake_data.shape[0]:
        # 自动对齐到最小样本量
        min_sequences = min(ori_data.shape[0], fake_data.shape[0])
        # print(f"Warning: Data size mismatch. Using first {min_sequences} sequences.")
        ori_data = ori_data[:min_sequences]
        fake_data = fake_data[:min_sequences]
    
    dtw_distances = Parallel(n_jobs=n_jobs)(
        delayed(multi_dim_dtw)(ori_data[i], fake_data[i]) 
        for i in range(len(ori_data))
    )
    return np.mean(dtw_distances)

# 加载数据
# ori_data = np.load('../OUTPUT/ETTh/samples/etth_norm_truth_24_train.npy')
# fake_data = np.load('../OUTPUT/ETTh/ddpm_fake_ETTh.npy')

# # 计算DTW
# avg_dtw = dtw_metric(ori_data, fake_data)
# print("Average DTW Distance:", avg_dtw)

# import matplotlib.pyplot as plt

# # 绘制图像
# plt.plot(ori_data[0, :, 0], label="Original")
# plt.plot(fake_data[0, :, 0], label="Fake")
# plt.legend()

# # 保存到当前文件夹（默认PNG格式）
# plt.savefig("original_vs_fake.png")  # 文件名可自定义

# # 清除当前图形，避免后续绘图重叠
# plt.close()