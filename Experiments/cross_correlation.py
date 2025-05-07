import torch
from torch import nn

# @torch.no_grad()
# def cacf_torch(x, max_lag, dim=(0, 1), chunk_size=100):  # traffic 
#     def get_lower_triangular_indices(n):
#         return [list(x) for x in torch.tril_indices(n, n)]

#     ind = get_lower_triangular_indices(x.shape[2])
#     x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    
#     x_l_all = x[..., ind[0]]
#     x_r_all = x[..., ind[1]]
    
#     B, L, N_pairs = x_l_all.shape
#     cacf_list = []

#     for start in range(0, N_pairs, chunk_size):
#         end = min(start + chunk_size, N_pairs)
#         x_l = x_l_all[..., start:end]  # [B, L, chunk]
#         x_r = x_r_all[..., start:end]

#         sub_list = []
#         for i in range(max_lag):
#             y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
#             cacf_i = torch.mean(y, dim=1)
#             sub_list.append(cacf_i)

#         chunk_cacf = torch.cat(sub_list, dim=1)  # [B, max_lag * chunk]
#         cacf_list.append(chunk_cacf)

#     cacf = torch.cat(cacf_list, dim=1)  # [B, max_lag * N_pairs]
#     return cacf.reshape(B, -1, len(ind[0]))  # [B, max_lag, N_pairs]
# other
def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real.to(x_fake.device))

        
        # loss = loss / x_fake.shape[-1]  # traffic 归一化尺度,other注释掉这行
       
        return loss / 10.