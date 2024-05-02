import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model)).to(device)
        self.bias = nn.Parameter(torch.zeros(d_model)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))
        
        # self.o_proj1 = nn.Conv2d(midc, midc, 1)
        # self.o_proj2 = nn.Conv2d(midc, midc, 1)
        # self.act = nn.GELU()

    
    def forward(self, x,*args):
        B, N, C = x.shape
        
        qkv = x.reshape(B, N, self.heads, self.headc).permute(0, 2, 1, 3)
        if args:
            v = args[0].reshape(B, N, self.heads, self.headc).permute(0, 2, 1, 3)
            q = k = qkv
            bias = args[0]
        else:
            q = k = v = qkv
            bias = x

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / N
        v = torch.matmul(q, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = v + bias
        
        return out
