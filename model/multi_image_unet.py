import torch
import torch.nn as nn
from einops import rearrange


from .unet_parts import (
    ResnetBlock, 
    ResnetBlock2D,
    SinusoidalPosEmb, 
    PreNorm,
    PreNorm2D,
    LayerNorm2D,
    LayerNorm,
    LinearAttention,
    LinearAttention2D,
    DecoderCrossAttention,
    DecoderCrossAttention2D,
    Residual,
    Upsample,
    Upsample2D,
    Downsample,
    Downsample2D
)
# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97

class DDPMUNet(nn.Module):
    """
    3D DDPM U-Net which accepts input shaped as
    B x Cond x Channels x Height x Width x Depth
    """
    def __init__(self, cfg):
        super(DDPMUNet, self).__init__()
        
        self.cfg = cfg
        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.input_dim
        if self.cfg.model.use_depth:
            in_channels += 1
            
        dim = self.cfg.model.unet.model_channels
        self.init_conv = nn.Conv3d(in_channels, dim, 7, padding = 3)

        # ========== unet channels ==========
        channel_mult = self.cfg.model.unet.channel_mult
        self.attn_resolutions = self.cfg.model.unet.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.model.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock(dim_in, dim_in))
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
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(mid_dim, mid_dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))
            # the first latent gonna be null latent
            self.query_volume =  nn.Parameter(data = torch.rand((mid_dim,
                                                                current_side,
                                                                current_side,
                                                                current_side)),
                                                                requires_grad=True)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                if self.cfg.model.unet.attention_aggregation:
                    self.volume_aggregators.append(DecoderCrossAttention(dim_in, dim_out, cfg.model.n_heads,
                                                                         include_query_as_key = False))

                layers.append(ResnetBlock(dim_out + dim_in, dim_out))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(dim, dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))
            self.volume_aggregators = nn.ModuleList(self.volume_aggregators[::-1])

        self.final_res_block = ResnetBlock(dim * 2, dim)

        # ========== 3D conv upsampler =========
        self.conv_upsampler = nn.Identity()

        out_res_block = [ResnetBlock(dim, cfg.model.unet.volume_out_channels), LayerNorm(cfg.model.unet.volume_out_channels)]
        out_res_block.append(torch.nn.Tanh())
        self.out_res_block = nn.Sequential(*out_res_block)


    def forward(self, x):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, D, H, W = x.shape

        ft_map_idx = 0

        x = self.init_conv(x.reshape(-1, C, H, W, D))
        r = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx]).clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.model.unet.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.model.unet.blocks_per_res:]
            else:
                downsample = down[self.cfg.model.unet.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x)
                if r_idx == self.cfg.model.unet.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx]))
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx])

        ft_map_idx += 1
        if self.cfg.model.unet.attention_aggregation:
            x = self.volume_aggregators[ft_map_idx](x, self.query_volume)
        else:
            x = torch.mean(x, dim=1, keepdim=False)
        ft_map_idx -= 1

        for up in self.ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                if self.cfg.model.unet.attention_aggregation:
                    scf = self.volume_aggregators[ft_map_idx](scf, x)
                else:
                    scf = torch.mean(scf, dim=1, keepdim=False)
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        if self.cfg.model.unet.attention_aggregation:
            x = torch.cat((x, self.volume_aggregators[ft_map_idx](r, x)), dim = 1)
        else:
            x = torch.cat((x, torch.mean(r, dim=1, keepdim=False)), dim = 1)

        x = self.final_res_block(x)

        x = self.conv_upsampler(x)
        x = self.out_res_block(x)
        return x

class TriplaneUNet(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(TriplaneUNet, self).__init__()
        
        self.cfg = cfg

        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.input_dim
        dim = self.cfg.model.unet.model_channels
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding = 3)

        # ========== unet channels ==========
        channel_mult = self.cfg.model.unet.channel_mult
        self.attn_resolutions = self.cfg.model.unet.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.data.input_size[0]
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock2D(dim_in, dim_in))
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
        self.mid_block1 = ResnetBlock2D(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm2D(mid_dim, LinearAttention2D(mid_dim)))
        self.mid_block2 = ResnetBlock2D(mid_dim, mid_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock2D(dim_out + dim_in, dim_out))
            layers.append(Residual(PreNorm2D(dim_out, LinearAttention2D(dim_out))))
            layers.append(Upsample2D(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        self.final_res_block = ResnetBlock2D(dim * 2, cfg.model.unet.plane_channels * 3)

    def forward(self, x):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, H, W = x.shape

        ft_map_idx = 0

        assert x.shape[1] == 1, "Accepting only single-image viewset"
        x = self.init_conv(x.reshape(-1, C, H, W))
        r = x.clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.model.unet.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.model.unet.blocks_per_res:]
            else:
                downsample = down[self.cfg.model.unet.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x)
                if r_idx == self.cfg.model.unet.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x)
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for up in self.ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)

        # reshape into triplanes
        return x.reshape(x.shape[0], 3,
                         self.cfg.model.unet.plane_channels, 
                         *x.shape[2:])


class VoTriUNet(nn.Module):
    """
    3D DDPM U-Net which accepts input shaped as
    B x Cond x Channels x Height x Width x Depth
    """
    def __init__(self, cfg):
        super(VoTriUNet, self).__init__()
        
        self.cfg = cfg

        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.input_dim 

        dim = self.cfg.model.unet.model_channels
        self.init_conv = nn.Conv3d(in_channels, dim, 7, padding = 3)

        # ========== unet channels ==========
        channel_mult = self.cfg.model.unet.channel_mult
        self.attn_resolutions = self.cfg.model.unet.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.model.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock(dim_in, dim_in))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            print(current_side)
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            # downsampling
            layers.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(mid_dim, mid_dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))

            # the first latent gonna be null latent
            self.query_volume =  nn.Parameter(data = torch.rand((mid_dim,
                                                                current_side,
                                                                current_side,
                                                                current_side)),
                                                                requires_grad=True)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                if self.cfg.model.unet.attention_aggregation:
                    self.volume_aggregators.append(DecoderCrossAttention(dim_in, dim_out, cfg.model.n_heads,
                                                                         include_query_as_key = False))

                layers.append(ResnetBlock(dim_out + dim_in, dim_out))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(dim, dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))
            self.volume_aggregators = nn.ModuleList(self.volume_aggregators[::-1])

        volume_out_channels = self.cfg.model.unet.volume_out_channels
        self.final_res_block = nn.Sequential(*[ResnetBlock(dim * 2, volume_out_channels), \
                                                LayerNorm(volume_out_channels), \
                                                torch.nn.Tanh()])
        
        ############################################ plane decoder
        out_dim = cfg.model.volume_size * volume_out_channels
        channel_mult = cfg.model.unet.votri_channel_mult

        dims = [out_dim, *map(lambda m: out_dim // m, channel_mult)]
        print(f"Plane decoder dim: {dims}")
        
        in_out = list(zip(dims[:-1], dims[1:]))
        self.plane_ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock2D(dim_in, dim_in))
            layers.append(Residual(PreNorm2D(dim_in, LinearAttention2D(dim_in))))
            layers.append(Upsample2D(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1))
            self.plane_ups.append(nn.ModuleList([*layers]))


        final_plane_res_block = [ResnetBlock2D(dims[-1], cfg.model.unet.plane_channels),
                                    LayerNorm2D(cfg.model.unet.plane_channels)]
        self.final_plane_res_block = nn.Sequential(*final_plane_res_block)


    def forward(self, x):
        volume_features = self.extract_volumes(x)
        return self.split_planes(volume_features)


    def extract_volumes(self, x):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, D, H, W = x.shape

        ft_map_idx = 0

        x = self.init_conv(x.reshape(-1, C, H, W, D))
        r = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx]).clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.model.unet.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.model.unet.blocks_per_res:]
            else:
                downsample = down[self.cfg.model.unet.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x)
                if r_idx == self.cfg.model.unet.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx]))
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx])

        ft_map_idx += 1
        if self.cfg.model.unet.attention_aggregation:
            x = self.volume_aggregators[ft_map_idx](x, self.query_volume)
        else:
            x = torch.mean(x, dim=1, keepdim=False)
        ft_map_idx -= 1

        for up in self.ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                if self.cfg.model.unet.attention_aggregation:
                    scf = self.volume_aggregators[ft_map_idx](scf, x)
                else:
                    scf = torch.mean(scf, dim=1, keepdim=False)
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        if self.cfg.model.unet.attention_aggregation:
            x = torch.cat((x, self.volume_aggregators[ft_map_idx](r, x)), dim = 1)
        else:
            x = torch.cat((x, torch.mean(r, dim=1, keepdim=False)), dim = 1)

        volume_features = self.final_res_block(x)
        return volume_features


    def split_planes(self, volume_features):
        hw_features = rearrange(volume_features, "bs c d h w -> bs (c d) h w")
        dw_features = rearrange(volume_features, "bs c d h w -> bs (c h) d w")
        dh_features = rearrange(volume_features, "bs c d h w -> bs (c w) d h")

        hw_planes = self.extract_planes(hw_features)
        dw_planes = self.extract_planes(dw_features)
        dh_planes = self.extract_planes(dh_features)

        return torch.stack([hw_planes, dw_planes, dh_planes], dim=1), volume_features


    def extract_planes(self, x):
        for up in self.plane_ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x)
            x = attn(x)
            x = upsample(x)
        
        return self.final_plane_res_block(x)