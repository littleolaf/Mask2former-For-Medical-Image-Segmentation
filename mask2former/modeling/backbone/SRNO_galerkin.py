import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class simple_attn_old(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 16, kernel_size=1, bias=False),
            nn.GELU(), # nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // 16, out_channels=in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out

    
class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))
        
        # self.o_proj1 = nn.Conv2d(midc, midc, 1)
        # self.o_proj2 = nn.Conv2d(midc, midc, 1)
        # self.act = nn.GELU()

        # 添加通道注意力
        # 通道注意力，即两个全连接层连接
        self.channel_atten = ChannelAttention(midc)

        self.FFN = nn.Sequential(
            nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, groups=midc),
            nn.Conv2d(midc, midc, kernel_size=1,),
            nn.GELU()
        )
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x
        #计算通道权重
        channel_weight = self.channel_atten(x)
        # x = channel_weight*x
        # bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        out = v.permute(0, 3, 1, 2) + bias
        # 用通道权重加权
        out = out*channel_weight + bias
        # 前向传播
        # out = self.FFN(out) + out
        # bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return out
