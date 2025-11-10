import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error
from metric_utils import extract_time
from tqdm.auto import tqdm
import os
class PostHocDataset(Dataset):
    def __init__(self, data, time_lens):
        """
        构建预测器的数据集，每个样本是 (x, t, y)
        x: [seq_len-1, dim-1], y: [seq_len-1, 1]
        """
        self.data = []
        for i in range(len(data)):
            x = data[i][:-1, :-1]
            y = data[i][1:, -1:]
            t = time_lens[i] - 1
            self.data.append((x, t, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    将一个 batch 的数据 padding 到统一长度
    """
    xs, ts, ys = zip(*batch)
    lengths = torch.tensor(ts, dtype=torch.long)
    max_len = max(lengths)

    x_padded = torch.zeros(len(xs), max_len, xs[0].shape[1])
    y_padded = torch.zeros(len(ys), max_len, 1)

    for i in range(len(xs)):
        x_len = xs[i].shape[0]
        x_padded[i, :x_len] = torch.tensor(xs[i], dtype=torch.float32)
        y_padded[i, :x_len] = torch.tensor(ys[i], dtype=torch.float32)

    return x_padded, lengths, y_padded

class PostHocGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PostHocGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat = self.sigmoid(self.fc(output))
        return y_hat


def predictive_score_metrics(ori_data, generated_data, iterations=5000, batch_size=128, device='cuda:2' if torch.cuda.is_available() else 'cpu'):
    no, seq_len, dim = ori_data.shape
    ori_time, _ = extract_time(ori_data)
    generated_time, _ = extract_time(generated_data)
    input_dim = dim - 1
    hidden_dim = input_dim // 2

    # 构建训练集和测试集
    train_dataset = PostHocDataset(generated_data, generated_time)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = PostHocDataset(ori_data, ori_time)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = PostHocGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for _ in tqdm(range(iterations), desc="Training Post-hoc GRU"):
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            preds = model(x, lengths)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 测试
    model.eval()
    all_mae = []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            preds = model(x, lengths).cpu().numpy()
            y = y.numpy()

            for i in range(len(lengths)):
                true_len = lengths[i].item()
                mae = mean_absolute_error(y[i][:true_len], preds[i][:true_len])
                all_mae.append(mae)

    predictive_score = np.mean(all_mae)
    return predictive_score

ori_data = np.load('/home/user/dyh/ts/K-BasisDiff/OUTPUT/Stock_2/samples/stock_2_norm_truth_24_train.npy')

fake_data = np.load('/home/user/dyh/ts/K-BasisDiff/OUTPUT/Stock_2/ddpm_fake_Stock_2.npy')
print('/home/user/dyh/ts/K-BasisDiff/OUTPUT/Stock_2/samples/stock_2_norm_truth_24_train.npy')
pred_score = predictive_score_metrics(ori_data, fake_data)


# print("ETTm1:")
# ori_data = np.load('../OUTPUT/ETTm1/samples/ettm1_norm_truth_24_train.npy')
# fake_data = np.load('../OUTPUT/ETTm1/ddpm_fake_ETTm1.npy')
# pred_score = predictive_score_metrics(ori_data, fake_data)
# print("Predictive Score (MAE):", pred_score)