import torch
import torch.nn as nn
import torch.nn.functional as F

from . import get_cameras_from_data_dict
from .multi_image_unet import DDPMUNet, TriplaneUNet, VoTriUNet
from .renderer import PostActivatedVolumeRenderer, PostActivatedFeatureVolumeRenderer
from .triplane_renderer import TriplaneRenderer
from .unet_parts import ResnetBlock2D, SinusoidalPosEmb
from .triplanes import Triplanes
from .resnet import ResnetFeatureExtractor2D
from pytorch3d.structures import Volumes
from einops.layers.torch import Rearrange
from einops import rearrange
# from utility.load_model import load_models

class Reconstructor(nn.Module):
    def __new__(cls, cfg):
        if cfg.model.unet.volume_repr == "voxel":
            return VoxelReconstructor(cfg)
        elif cfg.model.unet.volume_repr == "triplanes":
            return TriplaneReconstructor(cfg)
        elif cfg.model.unet.volume_repr == "votri":
            return VoTriReconstructor(cfg)
        elif cfg.model.unet.volume_repr == "det_votri":
            return DeterministicVoTriReconstructor(cfg)
        elif cfg.model.unet.volume_repr == "lrm":
            return LRMReconstructor(cfg) 

class VoxelReconstructor(nn.Module):
    def __init__(self, cfg):
        super(VoxelReconstructor, self).__init__()
        print('Instantiated')
        self.cfg = cfg
        self.renderer = PostActivatedVolumeRenderer(cfg)

        if self.cfg.optimization.use_resnet:
            self.feature_extractor = ResnetFeatureExtractor2D(cfg)
        else:
            self.feature_extractor = FeatureExtractor2D(cfg)

        self.unprojector = ImageUnprojector(cfg)
        self.reconstructor = DDPMUNet(cfg)

        self._voxel_size = tuple(cfg.render.volume_extent_world * mult/ cfg.model.volume_size \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)

    def encode(self, input_images, input_cameras):
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Image feature extraction ============
        input_features = self.feature_extractor(input_images)
        # ============ Image unprojection ============
        volumes = self.unprojector(input_features, input_cameras)

        # ============ Volume reconstruction ============
        volume_features = self.reconstructor(volumes)

        return volume_features, None

    def decode(self, volume_features, target_cameras):
        num_render          = target_cameras["R"].size(1)
        BS                  = volume_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, volume_features.device)
        
        latent = volume_features.unsqueeze(1)
        latent = latent.expand(BS, num_render, *latent.shape[2:])
        latent = latent.reshape(BS*num_render, *latent.shape[2:])
        # Instantiate the Volumes object (densities and colors are already 5D)
        volumes = Volumes(
            densities = latent,
            features = None,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     volumes=volumes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, volume_features

    def forward(self, input_images, input_cameras, target_cameras):
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Image feature extraction ============
        input_features = self.feature_extractor(input_images)
        # ============ Image unprojection ============
        volumes = self.unprojector(input_features, input_cameras)

        # ============ Volume reconstruction ============
        volume_features = self.reconstructor(volumes)

        num_render          = target_cameras["R"].size(1)
        BS                  = volume_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, volume_features.device)
        
        latent = volume_features.unsqueeze(1)
        latent = latent.expand(BS, num_render, *latent.shape[2:])
        latent = latent.reshape(BS*num_render, *latent.shape[2:])
        # Instantiate the Volumes object (densities and colors are already 5D)
        volumes = Volumes(
            densities = latent,
            features = None,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     volumes=volumes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, volume_features


class TriplaneReconstructor(nn.Module):
    def __init__(self, cfg):
        super(TriplaneReconstructor, self).__init__()
        self.cfg = cfg
        self.renderer = TriplaneRenderer(cfg)

        self.reconstructor = TriplaneUNet(cfg)

        self._voxel_size = tuple(cfg.render.volume_extent_world * mult / cfg.data.input_size[0] \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)


    def forward(self, input_images, input_cameras, target_cameras):
    
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        num_render          = target_cameras["R"].size(1)
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, input_images.device)

        # ============ Volume reconstruction ============
        input_images        = rearrange(input_images, "bs cond c h w -> (bs cond) 1 c h w")
        triplane_features   = self.reconstructor(input_images)

        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features
    
    
    def decode(self, triplane_features, target_cameras):
        num_render          = target_cameras["R"].size(1)
        BS                  = triplane_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, triplane_features.device)
        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])
        
        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features
    
    
    def encode(self, input_images, input_cameras):
    
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Volume reconstruction ============
        input_images        = rearrange(input_images, "bs cond c h w -> (bs cond) 1 c h w")
        triplane_features   = self.reconstructor(input_images)
        return triplane_features, None 
    

class VoTriReconstructor(nn.Module):
    def __init__(self, cfg):
        super(VoTriReconstructor, self).__init__()
        print('Instantiated')
        self.cfg = cfg
        self.renderer = TriplaneRenderer(cfg)
        self.volume_renderer = PostActivatedFeatureVolumeRenderer(cfg)
        self.volume_out_channels = self.cfg.model.unet.volume_out_channels

        if self.cfg.optimization.use_resnet:
            self.feature_extractor = ResnetFeatureExtractor2D(cfg)
        else:
            self.feature_extractor = FeatureExtractor2D(cfg)

        self.unprojector = ImageUnprojector(cfg)
        self.reconstructor = VoTriUNet(cfg)

        self._voxel_size_volume = tuple(cfg.render.volume_extent_world * mult/ cfg.model.volume_size \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation_volume = tuple(tr for tr in cfg.render.volume_offsets)

        size = cfg.model.volume_size * 2**(len(cfg.model.unet.votri_channel_mult) -1)
        self._voxel_size = tuple(cfg.render.volume_extent_world * mult / size \
                                for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)

        # dummy attributes expected by GaussianDiffusion
        self.image_size = cfg.model.volume_size
        self.channels = cfg.model.unet.volume_out_channels

    def forward(self, input_images, input_cameras, target_cameras):
        latent, volume_features = self.encode(input_images, input_cameras)
        r_img, r_depth, triplane_features = self.decode(latent, target_cameras)
        return r_img, r_depth, triplane_features
    

    def encode(self, input_images, input_cameras):
        volume_features = self.encode_volume(input_images, input_cameras)
        triplane_features = self.reconstructor.split_planes(volume_features)
        latent = rearrange(triplane_features, "b num_planes c h w -> b (num_planes c) h w")
        return latent, volume_features
    
    
    def encode_volume(self, input_images, input_cameras):
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Image feature extraction ============
        input_features = self.feature_extractor(input_images)
        # ============ Image unprojection ============
        volumes = self.unprojector(input_features, input_cameras) # H W D volume

        # ============ Volume reconstruction ============
        return self.reconstructor.extract_volumes(volumes)
    
    
    def split_planes(self, volume_features):
        triplane_features = self.reconstructor.split_planes(volume_features)
        latent = rearrange(triplane_features, "b num_planes c h w -> b (num_planes c) h w")
        return latent


    def decode_volume(self, volume_features, target_cameras):
        latent = self.split_planes(volume_features)
        return self.decode(latent, target_cameras)
    

    def render_volumes(self, volume_features, target_cameras):
        num_render = target_cameras["R"].size(1)
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, volume_features.device)

        BS              = volume_features.size(0)
        volume_features = volume_features.unsqueeze(1)
        volume_features = volume_features.expand(BS, num_render, *volume_features.shape[2:])
        volume_features = volume_features.reshape(BS*num_render, *volume_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        volumes = Volumes(
            densities = volume_features,
            features = None,
            voxel_size=self._voxel_size_volume,
            volume_translation=self._volume_translation_volume
        )

        # ============ Rendering ============
        r_features, r_depth = self.volume_renderer(cameras=target_cameras,
                                     volumes=volumes
                                     )[0].split([self.volume_out_channels + 3, 1], dim=-1)
        
        r_features  = rearrange(r_features, "(bs nv) h w c -> bs nv c h w", nv=num_render)
        r_depth     = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        features  = torch.cat([r_features[:, :, :-3], r_depth], dim=2)
        rgbs      = r_features[:, :, -3:] * 2 - 1

        return rgbs, features, r_depth


    def decode(self, latent, target_cameras):
        num_render          = target_cameras["R"].size(1)
        triplane_features   = rearrange(latent, "b (num_planes c) h w -> b num_planes c h w", num_planes=3)
        BS                  = triplane_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, triplane_features.device)
        
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features


class ImageUnprojector(nn.Module):

    def __init__(self, cfg):
        # only conditioning images should be passed to this network
        super().__init__()
        self.cfg = cfg

        sample_volume = Volumes(
            densities = torch.zeros([1, 1, self.cfg.model.volume_size,
                                           self.cfg.model.volume_size,
                                           self.cfg.model.volume_size]),
            features = torch.zeros([1, 3, self.cfg.model.volume_size,
                                          self.cfg.model.volume_size,
                                          self.cfg.model.volume_size]),
            voxel_size= tuple(cfg.render.volume_extent_world * mult/ cfg.model.volume_size \
                         for mult in cfg.render.volume_size_mults),
            volume_translation=tuple(tr for tr in cfg.render.volume_offsets)
        )
        
        grid_coordinates = sample_volume.get_coord_grid() # [B*Renders, D, H, W, 3]
        # all grids are the same shape so we can just take the first one
        grid_coordinates = grid_coordinates[0]
        self.register_buffer('grid_coordinates', grid_coordinates)

    def forward(self, images_kept, cameras):
        # takes in volume densities and colors as predicted by the
        # convolutional network. Aggregates unprojected features from
        # the conditioning images and outputs new densities and colors
        B, Cond, C, H, W = images_kept.shape
        Renders = cameras.T.shape[0] // B
        N_volumes = B * Cond
        H_vol = self.cfg.model.volume_size
        W_vol = self.cfg.model.volume_size
        D_vol = self.cfg.model.volume_size

        # project the locations on the voxel grid onto the conditioning images
        image_coordinates = cameras.transform_points_ndc(self.grid_coordinates.reshape(-1, 3)) # [B*renders, H*W*D, 2]
        # only keep the training images
        image_coordinates = image_coordinates.reshape(B, Renders, H_vol*W_vol*D_vol, 3)
        image_coordinates = image_coordinates[:, :Cond, ...]
        image_coordinates = image_coordinates.reshape(N_volumes, H_vol*W_vol*D_vol, 3)
        # image_coordinates have dim 3 but that does not mean that they are in
        # homogeneous coordinates, because they are in NDC coordinates. The last dimension
        # is the depth in the NDC volume but the first two are already the coordinates
        # in the image. So we only need to take the first two dimensions.

        image_coordinates = image_coordinates[..., :2] 
        # flip x and y coordinates because x and y in NDC are +ve for top left corner of the
        # image, while grid_sample expects (-1, -1) to correspond to the top left pixel
        image_coordinates *= -1 
        image_coordinates = image_coordinates.reshape(N_volumes, H_vol, W_vol, D_vol, 2)
        # depth_volume      = rearrange(depth_volume, "" .reshape(N_volumes, H_vol, W_vol, D_vol, 1)

        # gridsample image values
        gridsample_input = images_kept.reshape(B*Cond, C, H, W)
        # reshape the grid for gridsample to 4D because when input is 4D, 
        # output is expected to be 4D as well
        image_coordinates = image_coordinates.reshape(B*Cond,
                                                    H_vol,
                                                    W_vol,
                                                    D_vol,
                                                    2).reshape(B*Cond,
                                                                H_vol, 
                                                                W_vol*D_vol, 2)
        # use default align_corners=False because camera projection outputs 0, 0 for the top left corner
        # of the top left pixel 
        unprojected_colors = F.grid_sample(gridsample_input, image_coordinates, padding_mode="border", align_corners=False)
        # unprojected_colors is a slightly misleading name, it is actually the unprojected
        # features in the conditioning images, including the pose embedding.
        unprojected_colors = unprojected_colors.reshape(B*Cond, C, H_vol, W_vol, D_vol)
        unprojected_colors = unprojected_colors.reshape(B, Cond, *unprojected_colors.shape[1:])

        return unprojected_colors


class FeatureExtractor2D(nn.Module):
    """
    2D feature extract that also downsamples by a factor of 4
    after processing with 3 convolutional layers. 
    """
    def __init__(self, cfg, groups=8):
        super().__init__()
        self.cfg = cfg

        if self.cfg.model.feature_extractor_2d.pass_features != 'both':
            dim = cfg.model.unet.input_dim
        else:
            dim = cfg.model.unet.input_dim // 2
    
        input_dim = cfg.model.input_dim 

        if self.cfg.optimization.use_rays:
            input_dim += 6

        # ========== 2D feature extraction - 1 + 2 x N + 1 conv layers ==========
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
        )

        if self.cfg.model.feature_extractor_2d.pass_features != 'high_res':
            self.feature_extractor = nn.ModuleList(
                [ResnetBlock2D(dim, dim) for i in 
                range(self.cfg.model.feature_extractor_2d.res_blocks)]
            )
            self.downsample = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
                nn.Conv2d(dim * 4, dim, 1)
            )

    def forward(self, viewset2d):
        # viewset: [B, Cond, C, H, W]
        # t: [B, Cond]
        B, Cond, C, H, W = viewset2d.shape

        viewset2d = viewset2d.reshape(B*Cond, C, H, W)
        viewset2d = self.init_conv(viewset2d)

        if self.cfg.model.feature_extractor_2d.pass_features != 'high_res':
            latents = [viewset2d]
            latents_sz = viewset2d.shape[2:]

            for layer in self.feature_extractor:
                viewset2d = layer(viewset2d)
            viewset2d = self.downsample(viewset2d)

            if self.cfg.model.feature_extractor_2d.pass_features == 'both':
                latents.append(F.interpolate(
                    viewset2d,
                    latents_sz,
                    mode='nearest'
                ))
                viewset2d = torch.cat(latents, dim=1)

        viewset2d = viewset2d.reshape(B, Cond, *viewset2d.shape[1:])

        return viewset2d
    
class IdentityWithTimestep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return x


class DeterministicVoTriReconstructor(nn.Module):
    def __init__(self, cfg):
        super(DeterministicVoTriReconstructor, self).__init__()
        print('Instantiated')
        self.cfg = cfg
        self.renderer = TriplaneRenderer(cfg)
        self.volume_renderer = PostActivatedVolumeRenderer(cfg)
        self.volume_out_channels = self.cfg.model.unet.volume_out_channels

        if self.cfg.optimization.use_resnet:
            self.feature_extractor = ResnetFeatureExtractor2D(cfg)
        else:
            self.feature_extractor = FeatureExtractor2D(cfg)

        self.unprojector = ImageUnprojector(cfg)
        self.reconstructor = VoTriUNet(cfg)

        self._voxel_size_volume = tuple(cfg.render.volume_extent_world * mult/ cfg.model.volume_size \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation_volume = tuple(tr for tr in cfg.render.volume_offsets)

        size = cfg.model.volume_size * 2**(len(cfg.model.unet.votri_channel_mult) -1)
        self._voxel_size = tuple(cfg.render.volume_extent_world * mult / size \
                                for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)

        # dummy attributes expected by GaussianDiffusion
        self.image_size = cfg.model.volume_size
        self.channels = cfg.model.unet.volume_out_channels

    def forward(self, input_images, input_cameras, target_cameras):
        latent, volume_features = self.encode(input_images, input_cameras)
        r_img, r_depth, triplane_features = self.decode(latent, target_cameras)
        return r_img, r_depth, triplane_features
    

    def encode(self, input_images, input_cameras):
        volume_features = self.encode_volume(input_images, input_cameras)
        triplane_features = self.reconstructor.split_planes(volume_features)
        latent = rearrange(triplane_features, "b num_planes c h w -> b (num_planes c) h w")
        return latent, volume_features
    
    
    def encode_volume(self, input_images, input_cameras):
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Image feature extraction ============
        input_features = self.feature_extractor(input_images)
        # ============ Image unprojection ============
        volumes = self.unprojector(input_features, input_cameras) # H W D volume

        # ============ Volume reconstruction ============
        return self.reconstructor.extract_volumes(volumes)
    
    
    def split_planes(self, volume_features):
        triplane_features = self.reconstructor.split_planes(volume_features)
        latent = rearrange(triplane_features, "b num_planes c h w -> b (num_planes c) h w")
        return latent


    def decode_volume(self, volume_features, target_cameras):
        latent = self.split_planes(volume_features)
        return self.decode(latent, target_cameras)
    

    def render_volumes(self, volume_features, target_cameras):
        num_render = target_cameras["R"].size(1)
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, volume_features.device)

        BS              = volume_features.size(0)
        volume_features = volume_features.unsqueeze(1)
        volume_features = volume_features.expand(BS, num_render, *volume_features.shape[2:])
        volume_features = volume_features.reshape(BS*num_render, *volume_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        volumes = Volumes(
            densities = volume_features,
            features = None,
            voxel_size=self._voxel_size_volume,
            volume_translation=self._volume_translation_volume
        )

        # ============ Rendering ============
        r_img, r_depth = self.volume_renderer(cameras=target_cameras,
                                     volumes=volumes
                                     )[0].split([3, 1], dim=-1)
        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth


    def decode(self, latent, target_cameras):
        num_render          = target_cameras["R"].size(1)
        triplane_features   = rearrange(latent, "b (num_planes c) h w -> b num_planes c h w", num_planes=3)
        BS                  = triplane_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, triplane_features.device)
        
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features


class LRMReconstructor(nn.Module):
    def __init__(self, cfg):
        super(LRMReconstructor, self).__init__()
        self.cfg = cfg
        self.renderer = TriplaneRenderer(cfg)

        from model.lrm.lrm import LRMGenerator
        lrm_small_model_kwagrs ={'camera_embed_dim': 768, \
                                    'transformer_dim': 768, \
                                    'transformer_layers': 12, \
                                    'transformer_heads': 16, \
                                    'triplane_low_res': 32, \
                                    'triplane_high_res': 64, \
                                    'triplane_dim': cfg.model.unet.plane_channels, 
                                    'encoder_freeze': False}
        self.reconstructor = LRMGenerator(**lrm_small_model_kwagrs)
        self._voxel_size = tuple(cfg.render.volume_extent_world * mult / (64) \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)


    def forward(self, input_images, input_cameras, target_cameras):
    
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        num_render          = target_cameras["R"].size(1)
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, input_images.device)

        # ============ Volume reconstruction ============
        input_images        = rearrange(input_images, "bs cond c h w -> (bs cond) 1 c h w")
        triplane_features   = self.reconstructor(input_images)

        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features
    
    
    def decode(self, triplane_features, target_cameras):
        num_render          = target_cameras["R"].size(1)
        BS                  = triplane_features.size(0)
        # ============ Preparing outputs in the target shape ============
        target_cameras = get_cameras_from_data_dict(self.cfg, target_cameras, triplane_features.device)
        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(BS, num_render, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(BS*num_render, 
                                                      *triplane_features.shape[2:])
        
        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_depth = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)

        r_img   = rearrange(r_img, "(bs nv) h w c -> bs nv c h w", nv=num_render) * 2 - 1
        r_depth = rearrange(r_depth, "(bs nv) h w c -> bs nv c h w", nv=num_render)

        return r_img, r_depth, triplane_features
    
    
    def encode(self, input_images, input_cameras):
    
        # ============ Building input images ============
        BS, Cond, C, H, W   = input_images.shape
        input_cameras       = get_cameras_from_data_dict(self.cfg, input_cameras, input_images.device)

        # ============ Volume reconstruction ============
        input_images        = rearrange(input_images, "bs cond c h w -> (bs cond) 1 c h w")
        triplane_features   = self.reconstructor(input_images, input_cameras)
        return triplane_features, None 