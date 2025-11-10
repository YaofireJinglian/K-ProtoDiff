import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .kan_list.ChebyKAN import ChebyKANLinear
from .kan_list.TaylorKAN import TaylorKANLinear
from .kan_list.FourierKAN import FourierKANLinear


def wn(layer):

    if hasattr(layer, 'weight'):
        return nn.utils.weight_norm(layer)
    else:
        return layer

class KBasisNet(nn.Module):
    def __init__(self, input_len, output_len, bottleneck=10, degree=2, scale_2=0.3, base_activation = torch.nn.SiLU,bias=True):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(TaylorKANLinear(
        in_features=input_len, out_features=bottleneck,
        order=3,
        scale_base=1.0,
        scale_taylor=scale_2,
        base_activation=base_activation,
        use_bias=True,
    )),
            nn.ReLU(),
            wn(TaylorKANLinear(
        in_features=bottleneck,out_features=bottleneck,
        order=3,
        scale_base=1.0,
        scale_taylor=scale_2,
        base_activation=base_activation,
        use_bias=True,
    ))
        )

        self.linear2 = nn.Sequential(
            wn(TaylorKANLinear(
        in_features=bottleneck,out_features=bottleneck,
        order=3,
        scale_base=1.0,
        scale_taylor=scale_2,
        base_activation=base_activation,
        use_bias=True,
    )),
            nn.ReLU(),
            wn(TaylorKANLinear(
        in_features=bottleneck,out_features=output_len,
        order=3,
        scale_base=1.0,
        scale_taylor=scale_2,
        base_activation=base_activation,
        use_bias=True,
    ))
        )

        self.skip = wn(TaylorKANLinear(
        in_features=input_len, out_features=bottleneck,
        order=3,
        scale_base=1.0,
        scale_taylor=scale_2,
        base_activation=base_activation,
        use_bias=True,
    ))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x) + self.skip(x))
        x = self.linear2(x)
        return x


class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self,d_model,n_heads, mask=False,d_keys=None,
                 d_values=None,dropout=0):
        super().__init__()
        
        self.mask = mask

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model,d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.value_projection = wn(nn.Linear(d_model, d_values * n_heads))
        self.out_projection = wn(nn.Linear(d_values * n_heads, d_model))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        num = len(queries.shape)
        if num == 2:
            L, _ = queries.shape
            S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(L, H, -1).permute(1,0,2)
            keys = self.key_projection(keys).view(S, H, -1).permute(1,0,2)
            values = self.value_projection(values).view(S, H, -1).permute(1,0,2)
            # queries = queries.view(L, H, -1).permute(1,0,2)
            # keys = keys.view(S, H, -1).permute(1,0,2)
            # values = values.view(S, H, -1).permute(1,0,2)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale  # qk 交叉自注意力

            attn = self.attend(dots) # softmax
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D) 交叉第二次乘

            out = out.permute(1,0,2).reshape(L,-1)
        else:
            B,L, _ = queries.shape
            B,S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B,L, H, -1).permute(0,2,1,3)
            keys = self.key_projection(keys).view(B,S, H, -1).permute(0,2,1,3)
            values = self.value_projection(values).view(B,S, H, -1).permute(0,2,1,3)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D)

            out = out.permute(0,2,1,3).reshape(B,L,-1)
            
        return self.out_projection(out),attn


# basis_add, basis_attn = self.cross_attention_basis(
#     basis_raw, series_raw, series_raw
# )
# basis_out = basis_raw + self.dropout_basis(basis_add) 
# basis_out = self.layer_norm11(basis_out)



# class MLP_bottle_with_highfreq(nn.Module):
#     def __init__(self, input_len, output_len, bottleneck, bias=True):
#         super().__init__()

#         # 主干路径（低频）
#         self.linear1 = nn.Sequential(
#             wn(FourierKANLinear(input_len, bottleneck, degree, 1.0, p2, base_activation, True)),
#             nn.ReLU(),
#             wn(FourierKANLinear(bottleneck, bottleneck, degree, 1.0, p2, base_activation, True))
#         )

#         self.linear2 = nn.Sequential(
#             wn(FourierKANLinear(bottleneck, bottleneck, degree, 1.0, p2, base_activation, True)),
#             nn.ReLU(),
#             wn(FourierKANLinear(bottleneck, output_len, degree, 1.0, p2, base_activation, True))
#         )

#         self.skip = wn(FourierKANLinear(input_len, bottleneck, degree, 1.0, p2, base_activation, True))
#         self.act = nn.ReLU()

#         # 高频提取模块：1D卷积 + ReLU
#         self.highfreq_conv = nn.Sequential(
#             nn.Conv1d(1, 4, kernel_size=3, padding=1),  # (B, 1, T) -> (B, 4, T)
#             nn.ReLU(),
#             nn.Conv1d(4, 1, kernel_size=3, padding=1),  # (B, 4, T) -> (B, 1, T)
#         )

#         self.fusion = nn.Linear(bottleneck + input_len, bottleneck)

#     def forward(self, x):  # x: [B, T]
#         x_low = self.act(self.linear1(x) + self.skip(x))  # [B, bottleneck]

#         # 提取高频：x → [B, T] → [B, 1, T] → Conv → [B, 1, T] → [B, T]
#         x_high = self.highfreq_conv(x.unsqueeze(1)).squeeze(1)  # [B, T]

#         # 融合：将高频特征与 bottleneck 拼接并融合
#         x_fused = torch.cat([x_low, x_high], dim=-1)  # [B, bottleneck + T]
#         x_fused = self.fusion(x_fused)  # [B, bottleneck]

#         out = self.linear2(x_fused)  # [B, output_len]
#         return out