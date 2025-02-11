'''
Diffusion distillation loop
'''
import numpy as np
import imageio
import time
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from einops import rearrange, reduce, repeat

from external.external_utils import PerceptualLoss
from utils.camera_utils import RelativeCameraLoader, get_interpolated_path
from utils.common_utils import get_lpips_fn, get_metrics, split_list, normalize, unnormalize, huber
from utils.co3d_dataloader import CO3Dv2Wrapper
from utils.co3d_dataloader import CO3D_ALL_CATEGORIES, CO3D_ALL_TEN
from utils.render_utils import init_ray_sampler, init_light_field_renderer

def distillation_loop(
        gpu,
        args,
        model_tuple,
        save_dir,
        seq_name,
        scene_cameras,
        scene_rgb,
        scene_mask,
        scene_valid_region,
        input_idx,
        use_diffusion=True,
        max_itr=3000,
        loss_fn_vgg=None,
    ):
    '''
    Loop for diffusion distillation
    Saves optimized torch-ngp

    Args:
        gpu (int): gpu id
        args (Namespace): SparseFusion options
        opt (Namesapce): torch-ngp options
        model_tuple (EFT, VAE, VLDM): a tuple of three models
        save_dir (str): save directory
        seq_name (str): save sequence name
        scene_cameras (PyTorch3D Camera): cameras
        scene_rgb (Tensor): gt rgb
        scene_mask (Tensor): foreground mask
        scene_valid_region (Tensor): valid image region 

    '''

    os.makedirs(f'{args.exp_dir}/render_imgs/{seq_name}/', exist_ok=True)
    os.makedirs(f'{args.exp_dir}/render_gifs/', exist_ok=True)
    eft, vae, vldm = model_tuple

    #@ GET RELATIVE CAMERA LOADER
    relative_cam = RelativeCameraLoader(relative=True, center_at_origin=True)
    relative_cam_no_origin = RelativeCameraLoader(relative=True, center_at_origin=False)

    #@ GET RELATIVE CAMERAS     
    scene_cameras_rel = relative_cam.get_relative_camera(scene_cameras, query_idx=[0], center_at_origin=True)
    scene_cameras_vox = relative_cam_no_origin.get_relative_camera(scene_cameras, query_idx=[0], center_at_origin=False)

    #@ GET ADDITIONAL CAMERAS
    scene_cameras_aug = get_interpolated_path(scene_cameras, n=50, method='circle', theta_offset_max=0.17)
    scene_cameras_aug = relative_cam.concat_cameras([scene_cameras, scene_cameras_aug])
    scene_cameras_aug_rel = relative_cam.get_relative_camera(scene_cameras_aug, query_idx=[0], center_at_origin=True)
    scene_cameras_aug_vox = relative_cam_no_origin.get_relative_camera(scene_cameras_aug, query_idx=[0], center_at_origin=False)
    blank_rgb = torch.zeros_like(scene_rgb[:1])
    blank_rgb = blank_rgb.repeat(len(scene_cameras_aug), 1, 1, 1)
    scene_rgb_aug = torch.cat((scene_rgb, blank_rgb))

    #@ ADJUST RENDERERS
    cam_dist_mean = torch.mean(torch.linalg.norm(scene_cameras.get_camera_center(), axis=1))
    min_depth = cam_dist_mean - 5.0
    volume_extent_world = cam_dist_mean + 5.0
    sampler_grid, _, sampler_feat = init_ray_sampler(gpu, 256, 256, min=min_depth, max=volume_extent_world, scale_factor=2)
    _, _, renderer_feat = init_light_field_renderer(gpu, 256, 256, min=min_depth, max=volume_extent_world, scale_factor=8.0)
    
    #! ###############################
    #! ####### PREPROCESSING #########
    #! ###############################
    #@ CACHE VIEW CONDITIONED FEATURES
    if use_diffusion:
        eft_feature_cache = {}
        timer = time.time()
        for ci in tqdm(range(len(scene_cameras_aug_rel))):

            #@ GET EFT REL CAMERAS
            gpnr_render_camera, render_rgb, batch_mask, input_cameras, input_rgb, input_masks = relative_cam(scene_cameras_aug_rel, scene_rgb_aug, query_idx=[ci], context_idx=input_idx)
            eft.encode(input_cameras, input_rgb)

            #@ GET EFT FEATURES ANDS IMAGE
            with torch.no_grad():
                epipolar_features , _, _ = renderer_feat(
                    cameras=gpnr_render_camera, 
                    volumetric_function=eft.batched_forward,
                    n_batches=16,
                    input_cameras=input_cameras,
                    input_rgb=input_rgb
                )
                lr_render_, epipolar_latents = (
                    epipolar_features.split([3, 256], dim=-1)
                )

            query_camera = relative_cam.get_camera_slice(scene_cameras_aug_rel, [ci])
            query_camera_vox = relative_cam.get_camera_slice(scene_cameras_aug_vox, [ci])
            epipolar_latents = rearrange(epipolar_latents, 'b h w f -> b f h w')
            print(epipolar_latents.shape)
            diffusion_z = vldm.sample(cond_images=epipolar_latents, batch_size=1)
            diffusion_z = 1.0 / args.z_scale_factor * diffusion_z
            diffusion_image = unnormalize(vae.decode(diffusion_z)).clip(0.0, 1.0).cpu()
            # diffusion_image = rearrange(diffusion_image, 'b c h w -> b h w c')
            print(diffusion_image.shape)
                        
            lr_image = rearrange(lr_render_, 'b h w f -> b f h w')
            lr_image = F.interpolate(lr_image, scale_factor=8.0, mode='bilinear')
            print(lr_image.shape)
            #@ OPTIONAL DIFFUSION IMAGE
            # diffusion_image = None

            eft_feature_cache[ci] = {'camera':query_camera, 'camera_vox':query_camera_vox, 'features':epipolar_latents, 
                                        'diffusion_image':diffusion_image, 'eft_image':diffusion_image}
            
        print(f'cached {len(eft_feature_cache)} features in {(time.time() - timer):02f} seconds')
        
        if len(eft_feature_cache) >= len(scene_cameras_rel):
            n_per_row = 8
            vis_rows = []
            for i in range(0, len(scene_cameras_rel), n_per_row):
                temp_row = []
                for j in range(n_per_row):
                    img = eft_feature_cache[i + j]['eft_image']
                    vis_img = rearrange(img, 'b c h w -> b h w c')[0].detach().cpu().numpy()
                    temp_row.append(vis_img)
                temp_row = np.hstack((temp_row))
                vis_rows.append(temp_row)
            vis_grid = np.vstack(vis_rows)
            imageio.imwrite(f'{args.exp_dir}/log/{seq_name}_eft_grid.jpg', (vis_grid*255).astype(np.uint8))

    