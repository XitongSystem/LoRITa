import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, factor = 1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        if factor == 1:
            # self.net = nn.Sequential(
            #     nn.LayerNorm(dim),
            #     nn.Linear(dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, dim),
            # )
            self.net1 = nn.Linear(dim, hidden_dim)
            self.net2 = nn.Linear(hidden_dim, dim)

        else:
            self.net1 = OrderedDict()
            for i in range(factor - 1):
                self.net1['compress_'+str(i)] = nn.Linear(dim, dim, bias=False)
            self.net1['compress_'+str(factor - 1)] = nn.Linear(dim, hidden_dim)
            
            self.net2 = OrderedDict()
            for i in range(factor - 1):
                self.net2['compress_'+str(i)] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.net2['compress_'+str(factor - 1)] = nn.Linear(hidden_dim, dim)

            self.net1 = nn.Sequential(self.net1)
            self.net2 = nn.Sequential(self.net2)

    def forward(self, x):
        x = self.norm(x)
        x = self.net1(x)
        x = self.act(x)
        x = self.net2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, factor = 1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        if factor == 1:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
            self.to_out = nn.Linear(inner_dim, dim, bias = False)
        else:
            self.to_qkv = OrderedDict()
            self.to_out = OrderedDict()

            for i in range(factor - 1):
                self.to_qkv['compress_'+str(i)] = nn.Linear(dim, dim, bias=False)
                self.to_out['compress_'+str(i)] = nn.Linear(inner_dim, inner_dim, bias=False)

            self.to_qkv['compress_'+str(factor - 1)] = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out['compress_'+str(factor - 1)] = nn.Linear(inner_dim, dim, bias=False)

            self.to_qkv = nn.Sequential(self.to_qkv)
            self.to_out = nn.Sequential(self.to_out)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, factor):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, factor = factor),
                FeedForward(dim, mlp_dim, factor = factor)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        channels = 3, dim_head = 64, factor=1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, factor)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
