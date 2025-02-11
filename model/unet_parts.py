import torch
import torch.nn as nn

from functools import partial

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import math
# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class Block2D(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        if dim_out % groups == 0:
            self.norm = nn.GroupNorm(groups, dim_out)
        else:
            self.norm = LayerNorm2D(dim_out)

        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock2D(nn.Module):
    def __init__(self, dim, dim_out, *, groups = 8):
        super().__init__()
        self.block1 = Block2D(dim, dim_out, groups = groups)
        self.block2 = Block2D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlockWithTimestep(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, groups = 8):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm2D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2D(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d', p1 = 2, p2 = 2, p3=2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1)
    )

def Upsample2D(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample2D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z=d)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_kv = nn.Conv3d(dim, hidden_dim*2, 1, bias = False)
        self.to_q = nn.Conv3d(dim, hidden_dim, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context):
        b, c, h, w, d = x.shape
        kv = self.to_kv(context).chunk(2, dim = 1)
        k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), kv)

        q = self.to_q(x)
        q = rearrange(q, 'b (h c) x y z -> b h c (x y z)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z=d)
        return self.to_out(out)


class LinearAttention2D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm2D(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


class PlaneAwareLinearAttention2D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm2D(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, '(b num_planes) (h c) x y -> b h c (x y num_planes)', num_planes=3, h = self.heads), qkv)

        q = q.softmax(dim = -2) 
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y num_planes) -> (b num_planes) (h c) x y', h = self.heads, x = h, y = w, num_planes=3)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z=d)
        return self.to_out(out)

class DecoderCrossAttention(nn.Module):
    def __init__(self, channels, channels_decoder_features, 
                 n_heads, include_query_as_key=False):
        """
        Module for aggregating information from multiple frames
        channels: dimension of skip connection features
        channels_decoder_features: dimension of decoder features
        """
        super(DecoderCrossAttention, self).__init__()
        self.xattention = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)
        if channels_decoder_features != channels:
            self.channel_equaliser = nn.Conv3d(channels_decoder_features, channels, 1)
        else:
            self.channel_equaliser = nn.Identity()
        self.include_query_as_key = include_query_as_key

    def forward(self, skip_connection_features, decoder_features):

        B, Cond = skip_connection_features.shape[0:2]
        if len(decoder_features.shape) == 4:
            C, H_vol, W_vol, D_vol = decoder_features.shape
            decoder_features = decoder_features.unsqueeze(0).expand(B, C, H_vol, W_vol, D_vol)
        else:
            assert len(decoder_features.shape) == 5
            B, C, H_vol, W_vol, D_vol = decoder_features.shape
        decoder_features = self.channel_equaliser(decoder_features)
        B, C, H_vol, W_vol, D_vol = decoder_features.shape

        assert C == skip_connection_features.shape[2]
        assert H_vol == skip_connection_features.shape[3]
        assert W_vol == skip_connection_features.shape[4]
        assert D_vol == skip_connection_features.shape[5]

        # put the frames as the sequence dimension
        x = decoder_features.permute(0, 2, 3, 4, 1)
        x = x.reshape([-1, C]).unsqueeze(1)
        
        y = skip_connection_features.permute(0, 3, 4, 5, 1, 2)
        y = y.reshape([-1, Cond, C])
        if self.include_query_as_key:
            y = torch.cat([y, x], dim=1)

        x = self.xattention(x, y, y)[0]

        # reshape back to volumes
        x = x.reshape([B, H_vol, W_vol, D_vol, 1, C])
        # permute back
        x = x.permute(0, 4, 5, 1, 2, 3)
        x = x.reshape([-1, *x.shape[2:]])

        x = x + decoder_features
        x = self.norm(x)

        return x


class DecoderCrossAttention2D(nn.Module):
    def __init__(self, channels, channels_decoder_features, 
                 n_heads, include_query_as_key=False):
        """
        Module for aggregating information from multiple frames
        channels: dimension of skip connection features
        channels_decoder_features: dimension of decoder features
        """
        super(DecoderCrossAttention2D, self).__init__()
        self.xattention = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)
        if channels_decoder_features != channels:
            self.channel_equaliser = nn.Conv2d(channels_decoder_features, channels, 1)
        else:
            self.channel_equaliser = nn.Identity()
        self.include_query_as_key = include_query_as_key

    def forward(self, skip_connection_features, decoder_features):

        B, Cond = skip_connection_features.shape[0:2]
        if len(decoder_features.shape) == 3:
            C, H, W = decoder_features.shape
            decoder_features = decoder_features.unsqueeze(0).expand(B, C, H, W)
        else:
            assert len(decoder_features.shape) == 4
            B, C, H, W = decoder_features.shape

        decoder_features = self.channel_equaliser(decoder_features)
        B, C, H, W = decoder_features.shape

        assert C == skip_connection_features.shape[2]
        assert H == skip_connection_features.shape[3]
        assert W == skip_connection_features.shape[4]

        # put the frames as the sequence dimension
        x = decoder_features.permute(0, 2, 3, 1)
        x = x.reshape([-1, C]).unsqueeze(1)
        
        y = skip_connection_features.permute(0, 3, 4, 1, 2)
        y = y.reshape([-1, Cond, C])

        if self.include_query_as_key:
            y = torch.cat([y, x], dim=1)

        x = self.xattention(x, y, y)[0]

        # reshape back to volumes
        x = x.reshape([B, H, W, 1, C])
        # permute back
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape([-1, *x.shape[2:]])

        x = x + decoder_features
        x = self.norm(x)

        return x


class ResnetBlock2DWithTimestep(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block2D(dim, dim_out, groups = groups)
        self.block2 = Block2D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)