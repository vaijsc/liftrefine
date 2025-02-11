import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Tuple, Union

from .triplanes import Triplanes

from pytorch3d.ops.utils import eyes
from pytorch3d.transforms import Transform3d

from pytorch3d.renderer import (
    VolumeRenderer,
    CamerasBase,
    VolumeSampler,
    NDCMultinomialRaysampler
)

from pytorch3d.renderer.implicit.utils import (
    _validate_ray_bundle_variables, 
    ray_bundle_variables_to_ray_points
)

from pytorch3d.renderer.implicit.raysampling import (
    HeterogeneousRayBundle, 
    RayBundle
)
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod
)
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf
from pytorch3d.renderer.implicit.utils import RayBundle

from pytorch3d.structures import Volumes
from pytorch3d.renderer import HarmonicEmbedding
from einops import rearrange, repeat


class TriplaneRenderer(VolumeRenderer):
    def __init__(self, cfg, sample_mode: str = "bilinear") -> None:
        """
        Overrides the renderer in Pytorch3D. Takes in non-activated volumes and
        non-activated features. Applies the activation before alpha compositing, 
        after the raymarcher has returned the interpolated volume values. 
        For reference refer to https://arxiv.org/abs/2111.11215
        """
        raysampler = NDCMultinomialRaysampler(
            image_width=cfg.data.input_size[0],
            image_height=cfg.data.input_size[1],
            n_pts_per_ray=cfg.render.n_pts_per_ray,
            min_depth=cfg.render.min_depth,
            max_depth=cfg.render.max_depth,
            stratified_sampling=cfg.render.stratified_sampling
        )

        # instantiate the standard ray marcher
        raymarcher = NeRFEmissionAbsorptionRaymarcher(cfg.data.white_background)

        super().__init__(raysampler=raysampler, raymarcher=raymarcher, sample_mode=sample_mode)
        self.cfg = cfg
        assert self.cfg.render.post_activation
        self.gain = cfg.render.n_pts_per_ray / (cfg.render.max_depth-cfg.render.min_depth)

        self.hidden_dim = self.cfg.model.unet.plane_hidden_channels
        self.out_feature_dim = 3

        self.renderer_3d = torch.nn.Sequential(
            nn.Linear(self.cfg.model.unet.plane_channels, self.hidden_dim),
            nn.Softplus(),
        )
        self.density_head = nn.Linear(self.hidden_dim, 1)
        self.dir_encoder = HarmonicEmbedding(4) 

        self.color_head = torch.nn.Sequential(
            nn.Linear(self.hidden_dim + self.dir_encoder.get_output_dim(), self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.out_feature_dim)
        )

        nn.init.xavier_uniform_(self.density_head.weight, 1 / self.gain) 
        nn.init.constant_(self.density_head.bias, 4 / self.gain)

        nn.init.xavier_uniform_(self.color_head[2].weight, 1 / self.gain) 
        nn.init.constant_(self.color_head[2].bias, 4 / self.gain)

    def density_activation_fn(self, sigma, lengths):
        dists = lengths[...,1:] - lengths[...,:-1]
        # In NeRF last appended length is 1e10 but in DVGO the last appended length is stepsize
        # https://github.com/sunset1995/DirectVoxGO/blob/341e1fc4e96efff146d42cd6f31b8199a3e536f7/lib/dvgo.py#LL309C1-L309C1
        dists = torch.cat([dists, torch.tensor([1/self.gain], device=dists.device).expand(dists[...,:1].shape)], -1)
        dists = dists[..., None] # last dimension of sigma is 1 - make it 1 for dists too

        noise = 0.
        if self.cfg.render.raw_noise_std > 0. and self.training:
            noise = torch.randn(sigma.shape, device=sigma.device) * self.cfg.render.raw_noise_std

        alpha = 1 - torch.exp(-F.softplus((sigma + noise) * self.gain - 6.0) * dists)

        return alpha

    def color_activation_fn(self, colors):
        return torch.sigmoid(colors) * 1.002 - 0.001

    def decode_sigma_color(self, rays_features, ray_bundle, sigma_only=False):
        rays_features   = self.renderer_3d(rays_features.reshape(-1, rays_features.shape[-1])).reshape(*rays_features.shape[:-1], -1)
        raw_sigma       = self.density_head(rays_features.reshape(-1, rays_features.shape[-1])).reshape(*rays_features.shape[:-1], -1)
        
        if sigma_only:
            return raw_sigma
        else:
            direction = ray_bundle.directions # bs h w c
            dir_feat  = self.dir_encoder(direction)
            dir_feat  = repeat(dir_feat, "bs h w c -> bs h w length c", length=rays_features.size(3))

            rays_features   = torch.cat([rays_features, dir_feat], dim=-1)
            rays_features   = self.color_head(rays_features.reshape(-1, rays_features.shape[-1])).reshape(*rays_features.shape[:-1], -1)
            
            return rays_features, raw_sigma


    def forward(self,  cameras: CamerasBase, triplanes: Triplanes):
        volumetric_function = MaskedTriplaneSampler(triplanes, sample_mode=self._sample_mode)

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        # use stratified sampling if specified in config and if
        # the model is in training mode        
        if self.training and self.cfg.render.stratified_sampling:
            stratified_sampling = True
        else:
            stratified_sampling = False 

        ray_bundle = self.renderer.raysampler(
            cameras=cameras, volumetric_function=volumetric_function,
            stratified_sampling=stratified_sampling
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        # pyre-fixme[23]: Unable to unpack `object` into 2 values.
        rays_features, ray_mask_out = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras
        )
        raw_sigma_coarse = self.decode_sigma_color(rays_features, ray_bundle, sigma_only=True)

        rays_densities = self.density_activation_fn(raw_sigma_coarse, 
                                                    ray_bundle.lengths)
        rays_densities *= ray_mask_out

        if self.cfg.render.n_pts_per_ray_fine != 0:
            # sample new locations and override existing rays_densities and rays_features
            with torch.no_grad():
                rays_densities = rays_densities[..., 0] # get rid of the channel dimension
                absorption = _shifted_cumprod(
                    (1.0 + 1e-5) - rays_densities, shift=1
                )
                weights = rays_densities * absorption

                # compute bin edges - they are midpoints between samples apart
                # from the first and last sample which are the edges of the ray
                # weights then were sampled from midpoints between bin edges,
                # apart from the first and last sample which are the edges of the ray
                # this is technically incorrrect but given that the samples at the
                # edges should be transparent anyway, it should not matter
                bins = (ray_bundle.lengths[..., 1:] + ray_bundle.lengths[..., :-1]) / 2
                bins = torch.cat([ray_bundle.lengths[..., :1],
                                  bins, 
                                  ray_bundle.lengths[..., -1:]], -1)

                z_fine_samples = sample_pdf(
                    bins.cpu(), weights.cpu(), self.cfg.render.n_pts_per_ray_fine
                ).to(bins.device).detach()

                # concatenate with the existing samples and sort
                new_lengths, _ = torch.sort(torch.cat(
                    [ray_bundle.lengths, z_fine_samples], -1), -1)
                ray_bundle = RayBundle(ray_bundle.origins,
                                       ray_bundle.directions,
                                       new_lengths,
                                       ray_bundle.xys)

            rays_features, ray_mask_out = volumetric_function(
                ray_bundle=ray_bundle, cameras=cameras
                )
            rays_features, raw_sigma_fine = self.decode_sigma_color(rays_features, ray_bundle, sigma_only=False)
            rays_densities = self.density_activation_fn(raw_sigma_fine, 
                                                        ray_bundle.lengths)

            rays_densities *= ray_mask_out

        if self.cfg.model.explicit_rendering:
            rays_features = self.color_activation_fn(rays_features)

        # finally, march along the sampled rays to obtain the renders
        images = self.renderer.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
        )
        # images - minibatch x ... x (feature_dim + opacity_dim)

        if not self.cfg.model.explicit_rendering:
            rgb = self.renderer_2d(images[..., :-1].permute(0, 3, 1, 2))
            rgb = self.color_activation_fn(rgb).permute(0, 2, 3, 1)
            images = torch.cat([rgb, images[..., -1:]], 3)

        return images, ray_bundle

class MaskedTriplaneSampler(nn.Module):
    def __init__(self, triplanes: Triplanes, sample_mode: str = "bilinear") -> None:
        """
        Args:
            volumes: An instance of the `Volumes` class representing a
                batch of volumes that are being rendered.
            sample_mode: Defines the algorithm used to sample the volumetric
                voxel grid. Can be either "bilinear" or "nearest".
        """
        super().__init__()
        if not isinstance(triplanes, Triplanes):
            raise ValueError("'volumes' have to be an instance of the 'Volumes' class.")
        self._triplanes = triplanes
        self._sample_mode = sample_mode
    
    def _get_ray_directions_transform(self):
        """
        Compose the ray-directions transform by removing the translation component
        from the volume global-to-local coords transform.
        """
        world2local = self._triplanes.get_world_to_local_coords_transform().get_matrix()
        directions_transform_matrix = eyes(
            4,
            N=world2local.shape[0],
            device=world2local.device,
            dtype=world2local.dtype,
        )
        directions_transform_matrix[:, :3, :3] = world2local[:, :3, :3]
        directions_transform = Transform3d(matrix=directions_transform_matrix)
        return directions_transform

    def forward(
        self, ray_bundle: Union[RayBundle, HeterogeneousRayBundle], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input ray parametrization, the forward function samples
        `self._volumes` at the respective 3D ray-points.
        Can also accept ImplicitronRayBundle as argument for ray_bundle.

        Args:
            ray_bundle: A RayBundle or HeterogeneousRayBundle object with the following fields:
                rays_origins_world: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                rays_directions_world: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                rays_lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape
                `(minibatch, ..., num_points_per_ray, opacity_dim)` containing the
                density vectors sampled from the volume at the locations of
                the ray points.
            rays_features: A tensor of shape
                `(minibatch, ..., num_points_per_ray, feature_dim)` containing the
                feature vectors sampled from the volume at the locations of
                the ray points.
        """

        # take out the interesting parts of ray_bundle
        rays_origins_world = ray_bundle.origins
        rays_directions_world = ray_bundle.directions
        rays_lengths = ray_bundle.lengths

        # validate the inputs
        _validate_ray_bundle_variables(
            rays_origins_world, rays_directions_world, rays_lengths
        )
        if self._triplanes.features().shape[0] != rays_origins_world.shape[0]:
            raise ValueError("Input volumes have to have the same batch size as rays.")

        #########################################################
        # 1) convert the origins/directions to the local coords #
        #########################################################

        # origins are mapped with the world_to_local transform of the volumes
        rays_origins_local = self._triplanes.world_to_local_coords(rays_origins_world)

        # obtain the Transform3d object that transforms ray directions to local coords
        directions_transform = self._get_ray_directions_transform()

        # transform the directions to the local coords
        rays_directions_local = directions_transform.transform_points(
            rays_directions_world.view(rays_lengths.shape[0], -1, 3)
        ).view(rays_directions_world.shape)

        ############################
        # 2) obtain the ray points #
        ############################

        # this op produces a fairly big tensor (minibatch, ..., n_samples_per_ray, 3)
        rays_points_local = ray_bundle_variables_to_ray_points(
            rays_origins_local, rays_directions_local, rays_lengths
        )

        ########################
        # 3) sample the volume #
        ########################

        rays_features_out = torch.zeros((*rays_points_local.shape[:-1],
                                        self._triplanes.features().shape[2]),
                                       device=self._triplanes.device)

        rays_mask_out = torch.ones((*rays_points_local.shape[:-1], 1),
                                    device=rays_points_local.device, requires_grad=False,
                                    dtype=torch.bool)

        for point_dimensions, feature_plane in zip([[0, 1], [0, 2], [1, 2]],
                                                   self._triplanes.features().split(1, dim=1)):
            rays_points_plane = rays_points_local[..., point_dimensions]
            # reshape to a size which grid_sample likes
            rays_points_plane_flat = rays_points_plane.view(
                rays_points_local.shape[0], -1, 1, 2
            )

            # run the grid sampler
            rays_features_plane = torch.nn.functional.grid_sample(
                feature_plane.squeeze(1), # B x C x H x W
                rays_points_plane_flat,
                align_corners=False,
                mode=self._sample_mode
            ) # batch x C x -1 x 1

            rays_features_plane = rays_features_plane.permute(0, 2, 3, 1).view(
                *rays_points_local.shape[:-1], self._triplanes.features().shape[2]
            )

            # add the features from the interpolated plane
            rays_features_out += rays_features_plane
        
        for dim_idx in [0, 1, 2]:
            outside_volume = torch.logical_or(rays_points_local[..., dim_idx:dim_idx+1] > 1,
                                              rays_points_local[..., dim_idx:dim_idx+1] < -1)
            rays_mask_out[outside_volume] = 0.0

        return rays_features_out, rays_mask_out

class NeRFEmissionAbsorptionRaymarcher(torch.nn.Module):
    """
    A modified version of the pytorch3d EmissionAbsorptionRaymarcher 
    which returns a silhouette which is calculated from the accumulated
    weights and not the product of absorptions, as in Pytorch3D implementation.

    Only the returned value is different, the rest is the same.
    """

    def __init__(self, white_bg, surface_thickness: int = 1) -> None:
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness
        self.white_bg = white_bg

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.
        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        # this line is different from pytorch3d implementation
        opacities = (weights[..., None] * 1).sum(dim=-2)
        # opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        depths = (weights[..., None] * kwargs["ray_bundle"].lengths.unsqueeze(-1)).sum(dim=-2)

        pix_alpha = weights.sum(dim=-1) 
        if self.white_bg:
            features  = features + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

        depths    = depths + (1 - pix_alpha.unsqueeze(-1)) * kwargs["ray_bundle"].lengths[..., -1:]  # (B, 3)


        return torch.cat((features, depths), dim=-1)