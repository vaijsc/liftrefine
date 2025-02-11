import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from .unet_parts import (
    ResnetBlock, 
    ResnetBlock2DWithTimestep,
    ResnetBlockWithTimestep,
    SinusoidalPosEmb, 
    PreNorm,
    PreNorm2D,
    LayerNorm,
    LinearAttention,
    LinearAttention2D,
    PlaneAwareLinearAttention2D,
    DecoderCrossAttention,
    Residual,
    Upsample,
    Upsample2D,
    Downsample,
    Downsample2D,
    LayerNorm2D,
    CrossAttention,
)

from model.sync_dreamer_attention import DepthTransformer
# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97

class TanhCode(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super(TanhCode, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, code_):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.clamp(min=-1 + self.eps, max=1 - self.eps).atanh() if self.scale == 1 \
            else (code / self.scale).clamp(min=-1 + self.eps, max=1 - self.eps).atanh()


class TriplaneDiffusion(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(TriplaneDiffusion, self).__init__()
        
        self.cfg = cfg
        # self.code_act = TanhCode(self.cfg.unet_diffusion.act_scale)
        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.plane_channels * 3
        dim = self.cfg.unet_diffusion.base_channels
        self.init_conv = nn.Conv2d(in_channels * 2, dim, 7, padding = 3)
        self.channel = in_channels
        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== unet channels ==========
        channel_mult = self.cfg.unet_diffusion.channels_cfg
        self.attn_resolutions = self.cfg.unet_diffusion.attention_res
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.data.input_size[0]
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlock2DWithTimestep(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm2D(dim_in, LinearAttention2D(dim_in))))
            # downsampling
            layers.append(Downsample2D(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2DWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm2D(mid_dim, LinearAttention2D(mid_dim)))
        self.mid_block2 = ResnetBlock2DWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlock2DWithTimestep(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm2D(dim_out, LinearAttention2D(dim_out))))
            # layers.append(DepthTransformer(dim_out, 4, cfg.model.unet.model_channels // 2, context_dim=cfg.model.unet.model_channels))

            layers.append(Upsample2D(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))


        self.final_res_block = ResnetBlock2DWithTimestep(dim * 2, in_channels, time_emb_dim = time_dim)

        out_layers = [LayerNorm2D(in_channels), torch.nn.Tanh()]
        self.out_layers = nn.Sequential(*out_layers)


    def forward(self, x, time, cond, cond_drop_prob=0):
        if cond_drop_prob > 0:
            drop_idx = torch.bernoulli(1 - cond_drop_prob * torch.ones(cond.shape[0], device=cond.device)).type(cond.dtype)
            drop_idx = drop_idx.reshape(-1, 1, 1, 1)
            cond     = drop_idx * cond

        return self.forward_func(x, time, cond)
    

    def forward_with_cond_scale(self, *args, cond_scale = 1., rescaled_phi = 0., **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)


    def forward_func(self, x, t, cond):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, C, H, W = x.shape
        emb = self.time_mlp(t)
        
        ft_map_idx = 0

        x = torch.cat([x, cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.unet_diffusion.resblocks_per_downsample]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:]
            else:
                downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, emb)
                if r_idx == self.cfg.unet_diffusion.resblocks_per_downsample - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x)
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for up in self.ups:
            res_blocks = up[:self.cfg.unet_diffusion.resblocks_per_downsample]
            attn, upsample = up[self.cfg.unet_diffusion.resblocks_per_downsample:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, emb)

            x = attn(x)
            # cond = torch.nn.functional.interpolate(cond, [x.size(-1), x.size(-1), x.size(-1)])
            # x = depth_transformer(x, cond)
            x = upsample(x)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, emb)
        return self.out_layers(x)



class PlaneAwareDiffusion(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(PlaneAwareDiffusion, self).__init__()
        
        self.cfg = cfg
        # self.code_act = TanhCode(self.cfg.unet_diffusion.act_scale)
        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.plane_channels
        dim = self.cfg.unet_diffusion.base_channels
        self.init_conv = nn.Conv2d(in_channels * 2, dim, 7, padding = 3)
        self.channel = in_channels
        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== unet channels ==========
        channel_mult = self.cfg.unet_diffusion.channels_cfg
        self.attn_resolutions = self.cfg.unet_diffusion.attention_res
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.data.input_size[0]
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlock2DWithTimestep(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm2D(dim_in, PlaneAwareLinearAttention2D(dim_in))))
            # downsampling
            layers.append(Downsample2D(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2DWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm2D(mid_dim, PlaneAwareLinearAttention2D(mid_dim)))
        self.mid_block2 = ResnetBlock2DWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlock2DWithTimestep(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm2D(dim_out, PlaneAwareLinearAttention2D(dim_out))))
            # layers.append(DepthTransformer(dim_out, 4, cfg.model.unet.model_channels // 2, context_dim=cfg.model.unet.model_channels))

            layers.append(Upsample2D(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))


        self.final_res_block = ResnetBlock2DWithTimestep(dim * 2, in_channels, time_emb_dim = time_dim)

        out_layers = [LayerNorm2D(in_channels), torch.nn.Tanh()]
        self.out_layers = nn.Sequential(*out_layers)


    def forward(self, x, time, cond, cond_drop_prob=0):
        if cond_drop_prob > 0:
            drop_idx = torch.bernoulli(1 - cond_drop_prob * torch.ones(cond.shape[0], device=cond.device)).type(cond.dtype)
            drop_idx = drop_idx.reshape(-1, 1, 1, 1)
            cond     = drop_idx * cond

        return self.forward_func(x, time, cond)
    

    def forward_with_cond_scale(self, *args, cond_scale = 1., rescaled_phi = 0., **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)


    def forward_func(self, x, t, cond):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, C, H, W = x.shape
        emb = self.time_mlp(t)
        emb = emb.repeat(3, 1)

        ft_map_idx = 0
        x = rearrange(x, "bs (num_planes c) h w -> (bs num_planes) c h w", num_planes=3)
        cond = rearrange(cond, "bs (num_planes c) h w -> (bs num_planes) c h w", num_planes=3)

        x = torch.cat([x, cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.unet_diffusion.resblocks_per_downsample]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:]
            else:
                downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, emb)
                if r_idx == self.cfg.unet_diffusion.resblocks_per_downsample - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x)
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for up in self.ups:
            res_blocks = up[:self.cfg.unet_diffusion.resblocks_per_downsample]
            attn, upsample = up[self.cfg.unet_diffusion.resblocks_per_downsample:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, emb)

            x = attn(x)
            # cond = torch.nn.functional.interpolate(cond, [x.size(-1), x.size(-1), x.size(-1)])
            # x = depth_transformer(x, cond)
            x = upsample(x)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, emb)
        x = self.out_layers(x)
        x = rearrange(x, "(bs num_planes) c h w -> bs (num_planes c) h w", num_planes=3)

        return x


class VolumeDiffusion(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(VolumeDiffusion, self).__init__()
        self.cfg = cfg  
        # self.code_act = TanhCode(self.cfg.unet_diffusion.act_scale)
        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.model_channels
        dim = self.cfg.unet_diffusion.base_channels
        self.init_conv = nn.Conv3d(in_channels * 2, dim, 7, padding = 3)
        self.channel = in_channels
        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== unet channels ==========
        channel_mult = self.cfg.unet_diffusion.channels_cfg
        self.attn_resolutions = self.cfg.unet_diffusion.attention_res
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.model.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlockWithTimestep(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            # downsampling
            layers.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlockWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlockWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlockWithTimestep(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            # layers.append(DepthTransformer(dim_out, 4, cfg.model.unet.model_channels // 2, context_dim=cfg.model.unet.model_channels))

            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))


        self.final_res_block = ResnetBlockWithTimestep(dim * 2, in_channels, time_emb_dim = time_dim)

        out_layers = [LayerNorm(in_channels), torch.nn.Tanh()]
        self.out_layers = nn.Sequential(*out_layers)


    def forward(self, x, time, cond, cond_drop_prob=0):
        if cond_drop_prob > 0:
            drop_idx = torch.bernoulli(1 - cond_drop_prob * torch.ones(cond.shape[0], device=cond.device)).type(cond.dtype)
            drop_idx = drop_idx.reshape(-1, 1, 1, 1, 1)
            cond     = drop_idx * cond

        return self.forward_func(x, time, cond)
    

    def forward_with_cond_scale(self, *args, cond_scale = 1., rescaled_phi = 0., **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)


    def forward_func(self, x, t, cond):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        emb = self.time_mlp(t)
        
        ft_map_idx = 0

        x = torch.cat([x, cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.unet_diffusion.resblocks_per_downsample]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:]
            else:
                downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, emb)
                if r_idx == self.cfg.unet_diffusion.resblocks_per_downsample - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x)
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for up in self.ups:
            res_blocks = up[:self.cfg.unet_diffusion.resblocks_per_downsample]
            attn, upsample = up[self.cfg.unet_diffusion.resblocks_per_downsample:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, emb)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, emb)
        return self.out_layers(x)


class VolumeDiffusionCrossAttn(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(VolumeDiffusion, self).__init__()
        self.cfg = cfg  
        # self.code_act = TanhCode(self.cfg.unet_diffusion.act_scale)
        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.model_channels
        dim = self.cfg.unet_diffusion.base_channels
        self.init_conv = nn.Conv3d(in_channels, dim, 7, padding = 3)
        self.channel = in_channels
        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_emb   =  nn.Parameter(data = torch.rand((1, time_dim)))

        # ========== unet channels ==========
        channel_mult = self.cfg.unet_diffusion.channels_cfg
        self.attn_resolutions = self.cfg.unet_diffusion.attention_res
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.model.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlockWithTimestep(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
                layers.append(Residual(PreNorm(dim_in, CrossAttention(dim_in))))
            # downsampling
            layers.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlockWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_crossattn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim)))
        self.mid_block2 = ResnetBlockWithTimestep(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.unet_diffusion.resblocks_per_downsample):
                layers.append(ResnetBlockWithTimestep(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            layers.append(Residual(PreNorm(dim_out, CrossAttention(dim_out))))
            # layers.append(DepthTransformer(dim_out, 4, cfg.model.unet.model_channels // 2, context_dim=cfg.model.unet.model_channels))

            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))


        self.final_res_block = ResnetBlockWithTimestep(dim * 2, in_channels, time_emb_dim = time_dim)

        out_layers = [LayerNorm(in_channels), torch.nn.Tanh()]
        self.out_layers = nn.Sequential(*out_layers)


    def forward(self, x, time, cond, cond_drop_prob=0):
        if cond_drop_prob > 0:
            drop_idx = torch.bernoulli(1 - cond_drop_prob * torch.ones(cond.shape[0], device=cond.device)).type(cond.dtype)
            drop_idx = drop_idx.reshape(-1, 1, 1, 1, 1)
            cond     = drop_idx * cond

        return self.forward_func(x, time, cond)
    

    def forward_with_cond_scale(self, *args, cond_scale = 1., rescaled_phi = 0., **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)


    def forward_func(self, x, t, cond):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, C, D, H, W = x.shape
        emb = self.time_mlp(t)
        cond_emb = self.cond_emb.repeat(B, 1)

        ft_map_idx = 0

        x = self.init_conv(x)
        r = x.clone()

        x_cond = self.init_conv(cond)

        h = []
        h_cond = []
        for down in self.downs:
            res_blocks = down[:self.cfg.unet_diffusion.resblocks_per_downsample]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, crossattn, downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:]
            else:
                downsample = down[self.cfg.unet_diffusion.resblocks_per_downsample:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, emb)
                x_cond = res_block(x_cond, cond_emb)
                if r_idx == self.cfg.unet_diffusion.resblocks_per_downsample - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                    x_cond = attn(x_cond)
                    x = crossattn(x, x_cond)
                    
                h.append(x)
                h_cond.append(x_cond)
                ft_map_idx += 1
            x = downsample(x)
            x_cond = downsample(x_cond)

        x = self.mid_block1(x, emb)
        x_cond = self.mid_block1(x_cond, cond_emb)

        x = self.mid_attn(x)
        x_cond = self.mid_attn(x_cond)
        x = self.mid_crossattn(x, x_cond)
        
        x = self.mid_block2(x, emb)
        x_cond = self.mid_block2(x_cond, cond_emb)

        for up in self.ups:
            res_blocks = up[:self.cfg.unet_diffusion.resblocks_per_downsample]
            attn, crossattn, upsample = up[self.cfg.unet_diffusion.resblocks_per_downsample:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, emb)
                
                scf_cond = h_cond.pop()
                x_cond = torch.cat((x_cond, scf_cond), dim = 1)
                x_cond = res_block(x_cond, cond_emb)

            x = attn(x)
            x_cond = attn(x_cond)
            x = crossattn(x, x_cond)
            
            x = upsample(x)
            x_cond = upsample(x_cond)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, emb)
        return self.out_layers(x)