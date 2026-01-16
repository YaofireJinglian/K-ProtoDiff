import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def wn(layer):
    # 检查 layer 是否有 'weight' 属性
    if hasattr(layer, 'weight'):
        return nn.utils.weight_norm(layer)
    else:

        return layer
class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self,d_model=64,n_heads=4, mask=False,d_keys=None,
                 d_values=None,dropout=0.1):
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
        out = self.out_projection(out)
        # print("out:",out.shape)
        return out,attn