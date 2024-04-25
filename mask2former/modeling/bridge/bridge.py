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

    
    def forward(self, x, name='0'):
        B, N, C = x.shape
        bias = x

        qkv = x.reshape(B, N, self.heads, self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q = k = v = qkv

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / N
        v = torch.matmul(q, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = v + bias
        
        return out

def FFN(in_channels, x):
    fc = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels).to(device),
        nn.Conv2d(in_channels, in_channels, kernel_size=1,).to(device),
        nn.GELU().to(device)
    )
    return fc(x)

def bridge(x:dict):
    inputs = [value for value in x.values()]
    x1,x2,x3,x4 = inputs
    B,C,H,W = x1.shape
    simple_atten = simple_attn(midc=C,heads=64)
    N1 = H*W
    N2 = int(N1 + (H*W/2))
    N3 = int(N2 + (H*W/4))
    H2 = int(H/2)
    H3 = int(H/4)
    H4 = int(H/8)
    W2 = int(W/2)
    W3 = int(W/4)
    W4 = int(W/8)


    y1 = x1.permute(0, 2, 3, 1).reshape(B, -1, C)
    y2 = x2.permute(0, 2, 3, 1).reshape(B, -1, C)
    y3 = x3.permute(0, 2, 3, 1).reshape(B, -1, C)
    y4 = x4.permute(0, 2, 3, 1).reshape(B, -1, C)
    total = torch.cat([y1,y2,y3,y4],-2)

    out1 = simple_atten(total)
    # out1 = total + out1

    z1 = out1[:,:N1,:].reshape(B, H, W, C).permute(0, 3, 1, 2)
    z2 = out1[:,N1:N2,:].reshape(B, H2, W2, C*2).permute(0, 3, 1, 2)
    z3 = out1[:,N2:N3,:].reshape(B, H3, W3, C*4).permute(0, 3, 1, 2)
    z4 = out1[:,N3:,:].reshape(B, H4, W4, C*8).permute(0, 3, 1, 2)
    
    z1 = FFN(C,z1) + z1
    z2 = FFN(C*2,z2) + z2
    z3 = FFN(C*4,z3) + z3
    z4 = FFN(C*8,z4) + z4
    features = {
        'res2':z1,
        'res3':z2,
        'res4':z3,
        'res5':z4,
    }
    return features


if __name__ == "__main__":
    x1 = torch.randn(2,256,128,128)   
    x2 = torch.randn(2,512,64,64)
    x3 = torch.randn(2,1024,32,32)   
    x4 = torch.randn(2,2048,16,16)

    input = {
        'res2':x1,
        'res3':x2,
        'res4':x3,
        'res5':x4,
    }
    print(bridge(input))