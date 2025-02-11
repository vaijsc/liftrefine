# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

from pathlib import Path
from collections import namedtuple
from collections import OrderedDict

import sys
import os
import imageio

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, rearrange

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from model.metric import compute_fid
from utils import exists, to_gpu, jet_depth, concat_camera, cycle, interpolate_camera
from ema_pytorch import EMA

# trainer class
class DiffusionEvaluator(object):
    def __init__(
        self,
        reconstructor,
        diffusion_model,
        accelerator,
        dataloader=None,
        test_batch_size=16,
        checkpoint_path=None,
        num_test_views=3,
        test_ema=False,
        optimization_cfg=None,
        amp=False,
        fp16=False,
        split_batches=True,
        evaluation_dir=None,
        run_name="pixelnerf",
    ):
        super().__init__()

        self.accelerator = accelerator
        self.inception_path = "../../pretrained/metric/inception-2015-12-05.pt"
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        self.num_gpus = accelerator.state.num_processes
        accelerator.print(f"Number of GPUs available: {self.num_gpus}")
        self.num_samples = 30000 // self.num_gpus        # number of sample image per gpu

        self.test_ema       = test_ema
        self.num_test_views = num_test_views

        if self.test_ema:
            self.reconstructor      = EMA(reconstructor, include_online_model=False)
            self.diffusion_model    = EMA(diffusion_model, include_online_model=False)
        else:
            self.reconstructor      = reconstructor
            self.diffusion_model    = diffusion_model

        self.batch_size = test_batch_size

        # dataset and dataloader
        self.dataloader = self.accelerator.prepare(dataloader)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self.eval_dir = evaluation_dir
        if self.accelerator.is_main_process:
            os.makedirs(self.eval_dir, exist_ok=True)


    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(path),
            map_location=device,
        )

        if self.test_ema:
            print("Loading ema model")
            self.reconstructor.load_state_dict(data["reconstructor_ema"], strict=True)
            self.diffusion_model.load_state_dict(data["diffusion_ema"], strict=True)
            self.reconstructor = self.reconstructor.ema_model
            self.diffusion_model = self.diffusion_model.ema_model
        else:
            print("Loading model")
            self.reconstructor.load_state_dict(data["reconstructor"], strict=True)
            self.diffusion_model.load_state_dict(data["diffusion_model"], strict=True)


    @torch.inference_mode()
    def evaluate_co3d(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                nviews = input_images.size(1)
                if self.num_test_views < nviews:
                    num_views = self.num_test_views
                    input_images = input_images[:, :num_views]
                    input_cameras["R"] = input_cameras["R"][:, :num_views]
                    input_cameras["T"] = input_cameras["T"][:, :num_views]
                    input_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                    input_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]
                elif self.num_test_views > nviews:
                    raise Exception("Sth wrong here")

            
                sample_data = self.model.inference(input_images, input_cameras, target_cameras, \
                                                                cond_scale = 2., rescaled_phi = 0.7)
                all_images      = torch.clip(sample_data["rgb"].cpu() * 0.5 + 0.5, 0, 1)

                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "{:06}.png".format(1))
                        img.save(path)
                    
        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")
    
    
    @torch.inference_mode()
    def evaluate_co3d_qualitative(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                    "focal_lengths": data["render_focal_lengths"], \
                                                    "principal_points": data["render_principal_points"]}

                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                cond_cameras = {}
                nviews = input_images.size(1)
                if self.num_test_views <= nviews:
                    num_views = self.num_test_views
                    cond_images = input_images[:, :num_views]
                    cond_cameras["R"] = input_cameras["R"][:, :num_views]
                    cond_cameras["T"] = input_cameras["T"][:, :num_views]
                    cond_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                    cond_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]
                elif self.num_test_views > nviews:
                    raise Exception("Sth wrong here")

                ###################### saving rendering video ######################
                cond_latent, volume_features = self.reconstructor.encode(cond_images, cond_cameras)

                # all_cond_images     = self.reconstructor.decode(cond_latent, render_cameras)
                # saved_videos        = all_cond_images # torch.cat([all_cond_images, all_sample_images], dim=-1)
                # saved_videos        = torch.clip(saved_videos.cpu() * 0.5 + 0.5, 0, 1)

                # for batch_idx, (name, saved_video) in enumerate(zip(object_names, saved_videos)):
                #     object_path = os.path.join(self.eval_dir, str(name.item()))
                #     os.makedirs(object_path, exist_ok=True)

                #     videos = []
                #     for frame in saved_video:
                #         frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
                #         frame = Image.fromarray(frame.astype(np.uint8))
                #         videos.append(frame)

                #     videos[0].save(fp=os.path.join(object_path, "{:06}.gif".format(1)),
                #                     format='png',
                #                     append_images=videos[1:],
                #                     save_all=True,
                #                     duration=100,
                #                     loop=0)
                #################################################################
                ###################### saving target image ######################
                rendered_features = self.reconstructor.model.render_volumes(volume_features, target_cameras)[1]
                nviews = rendered_features.size(1)

                image_clip = data["clip_images"][:, 0]
                batch = {"image_clip" : image_clip,\
                         "image_cond" : rendered_features.flatten(0, 1)}
                cond = self.diffusion_model.get_input_test(batch)
                uncond = self.diffusion_model.get_unconditional_conditioning(cond)
                N = image_clip.size(0)

                samples, z_denoise_row = self.diffusion_model.sample_log(cond=cond,batch_size=N,ddim=True,
                                                        ddim_steps=200, eta=1., device=image_clip.device, \
                                                        unconditional_guidance_scale=2.0,
                                                        unconditional_conditioning=uncond)
                
                all_sample_images = self.diffusion_model.decode_first_stage(samples)
                all_sample_images = torch.nn.functional.interpolate(all_sample_images, scale_factor=0.5)

                all_sample_images = rearrange(all_sample_images, " (bs nview) c h w -> bs nview c h w", nview=nviews)
                all_images      = torch.clip(all_sample_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "sample_{:06}.png".format(1))
                        img.save(path)

                ###############################################
                all_cond_images = self.reconstructor.decode(cond_latent, target_cameras)
                all_images      = torch.clip(all_cond_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "cond_{:06}.png".format(1))
                        img.save(path)
                ###############################################

        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")


    @torch.inference_mode()
    def evaluate_co3d_qualitative_autoregressive(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                    "focal_lengths": data["render_focal_lengths"], \
                                                    "principal_points": data["render_principal_points"]}

                target_cameras  = {"R" : data["target_camera_Rs"], \
                                        "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                destination_cameras  = {"R" : data["target_camera_Rs"].flatten(0, 1), \
                                        "T" : data["target_camera_Ts"].flatten(0, 1), \
                                            "focal_lengths": data["target_focal_lengths"].flatten(0, 1), \
                                            "principal_points": data["target_principal_points"].flatten(0, 1)}
                
                cond_cameras = {}
                nviews = input_images.size(1)
                if self.num_test_views <= nviews:
                    num_views = self.num_test_views
                    cond_images = input_images[:, :num_views]
                    cond_cameras["R"] = input_cameras["R"][:, :num_views]
                    cond_cameras["T"] = input_cameras["T"][:, :num_views]
                    cond_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                    cond_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]
                elif self.num_test_views > nviews:
                    raise Exception("Sth wrong here")
               
                init_cameras = {}
                init_cameras["R"] = input_cameras["R"][:, 0]
                init_cameras["T"] = input_cameras["T"][:, 0]
                init_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, 0]
                init_cameras["principal_points"] = input_cameras["principal_points"][:, 0]

                cond_latent, _ = self.reconstructor.encode(cond_images, cond_cameras)

                all_sample_images = []
                T = [1.0]
                for t in T:
                    interpolated_cameras = interpolate_camera(init_cameras, destination_cameras, t)
                    
                    ###################### saving rendering video ######################
                    _, volume_features = self.reconstructor.encode(cond_images, cond_cameras)

                    ###################### saving target image ######################
                    rendered_features = self.reconstructor.model.render_volumes(volume_features, interpolated_cameras)[1]
                    nviews = rendered_features.size(1)

                    image_clip = data["clip_images"].repeat(1, nviews, 1, 1, 1)
                    batch = {"image_clip" : image_clip.flatten(0, 1),\
                            "image_cond" : rendered_features.flatten(0, 1)}


                    cond = self.diffusion_model.get_input_test(batch)
                    uncond = self.diffusion_model.get_unconditional_conditioning(cond)
                    N = image_clip.size(0) * nviews

                    samples, z_denoise_row = self.diffusion_model.sample_log(cond=cond,batch_size=N,ddim=True,
                                                            ddim_steps=200, eta=1., device=image_clip.device, \
                                                            unconditional_guidance_scale=2.0,
                                                            unconditional_conditioning=uncond)
                    
                    sample_images = self.diffusion_model.decode_first_stage(samples)
                    sample_images = torch.nn.functional.interpolate(sample_images, scale_factor=0.5)
                    sample_images = rearrange(sample_images, " (bs nview) c h w -> bs nview c h w", nview=nviews)


                    all_sample_images.append(sample_images)
                    cond_images = torch.cat([cond_images, sample_images], dim=1)
                    cond_cameras = concat_camera([cond_cameras, interpolated_cameras])

                intermediate_images = torch.cat(all_sample_images, dim=-1).squeeze(1)
                intermediate_images = torch.clip(intermediate_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, intermediate_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    saved_image = (images.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img         = Image.fromarray(saved_image)
                    path        = os.path.join(object_path, "intermediate_{:06}.png".format(1))
                    img.save(path)
                ###############################################
         
         
                all_sample_images = torch.clip(all_sample_images[-1].cpu() * 0.5 + 0.5, 0, 1).squeeze(1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_sample_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    saved_image = (images.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img         = Image.fromarray(saved_image)
                    path        = os.path.join(object_path, "diffusion_{:06}.png".format(1))
                    img.save(path)
                ###############################################


                ###################### saving rendering video ######################
                new_latent, _ = self.reconstructor.encode(cond_images, cond_cameras)
                all_sample_images = self.reconstructor.decode(new_latent, target_cameras)
                all_images      = torch.clip(all_sample_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "sample_{:06}.png".format(1))
                        img.save(path)

                all_cond_images    = self.reconstructor.decode(cond_latent, render_cameras)
                all_new_images     = self.reconstructor.decode(new_latent, render_cameras)
                saved_videos       =  torch.cat([all_cond_images, all_new_images], dim=-1)
                saved_videos       = torch.clip(saved_videos.cpu() * 0.5 + 0.5, 0, 1)

                for batch_idx, (name, saved_video) in enumerate(zip(object_names, saved_videos)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    videos = []
                    for frame in saved_video:
                        frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
                        frame = Image.fromarray(frame.astype(np.uint8))
                        videos.append(frame)

                    videos[0].save(fp=os.path.join(object_path, "{:06}.gif".format(1)),
                                    format='png',
                                    append_images=videos[1:],
                                    save_all=True,
                                    duration=100,
                                    loop=0)

                ###############################################
                all_cond_images = self.reconstructor.decode(cond_latent, target_cameras)
                all_images      = torch.clip(all_cond_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "cond_{:06}.png".format(1))
                        img.save(path)
                ###############################################

        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")

    
    @torch.inference_mode()
    def evaluate_shapenet_qualitative_autoregressive(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                target_cameras  = {"R" : data["target_camera_Rs"], \
                                        "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}
                
                interpolated_cameras  = []
                for i in range(data["interpolated_camera_Rs"].size(1)):
                    interpolated_camera  = {"R" : data["interpolated_camera_Rs"][:, i:i+1], \
                                        "T" : data["interpolated_camera_Ts"][:, i:i+1], \
                                            "focal_lengths": data["interpolated_focal_lengths"][:, i:i+1], \
                                            "principal_points": data["interpolated_principal_points"][:, i:i+1]}
                    interpolated_cameras.append(interpolated_camera)

                cond_cameras = {}
                nviews = input_images.size(1)
                if self.num_test_views <= nviews:
                    num_views = self.num_test_views
                    cond_images = input_images[:, :num_views]
                    cond_cameras["R"] = input_cameras["R"][:, :num_views]
                    cond_cameras["T"] = input_cameras["T"][:, :num_views]
                    cond_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                    cond_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]
                elif self.num_test_views > nviews:
                    raise Exception("Sth wrong here")

                cond_latent, _ = self.reconstructor.encode(cond_images, cond_cameras)

                all_sample_images = []
                for interpolated_camera in interpolated_cameras:
                    ###################### saving rendering video ######################
                    _, volume_features = self.reconstructor.encode(cond_images, cond_cameras)

                    ###################### saving target image ######################
                    rendered_features = self.reconstructor.model.render_volumes(volume_features, interpolated_camera)[1]
                    nviews = rendered_features.size(1)

                    image_clip = data["clip_images"].repeat(1, nviews, 1, 1, 1)
                    batch = {"image_clip" : image_clip.flatten(0, 1),\
                            "image_cond" : rendered_features.flatten(0, 1)}


                    cond = self.diffusion_model.get_input_test(batch)
                    uncond = self.diffusion_model.get_unconditional_conditioning(cond)
                    N = image_clip.size(0) * nviews

                    samples, z_denoise_row = self.diffusion_model.sample_log(cond=cond,batch_size=N,ddim=True,
                                                            ddim_steps=50, eta=1., device=image_clip.device, \
                                                            unconditional_guidance_scale=2.0,
                                                            unconditional_conditioning=uncond)
                    
                    sample_images = self.diffusion_model.decode_first_stage(samples)
                    sample_images = torch.nn.functional.interpolate(sample_images, scale_factor=0.5)
                    sample_images = rearrange(sample_images, " (bs nview) c h w -> bs nview c h w", nview=nviews)


                    all_sample_images.append(sample_images)
                    cond_images = torch.cat([cond_images, sample_images], dim=1)
                    cond_cameras = concat_camera([cond_cameras, interpolated_camera])

                intermediate_images = torch.cat(all_sample_images, dim=-1).squeeze(1)
                intermediate_images = torch.clip(intermediate_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, intermediate_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    saved_image = (images.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img         = Image.fromarray(saved_image)
                    path        = os.path.join(object_path, "intermediate_{:06}.png".format(1))
                    img.save(path)
                ###############################################
            

                ###################### saving rendering video ######################
                new_latent, _ = self.reconstructor.encode(cond_images, cond_cameras)
                all_sample_images = self.reconstructor.decode(new_latent, target_cameras)
                all_images      = torch.clip(all_sample_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "sample_{:06}.png".format(1))
                        img.save(path)

                all_cond_images    = self.reconstructor.decode(cond_latent, render_cameras)
                all_new_images     = self.reconstructor.decode(new_latent, render_cameras)
                saved_videos       =  torch.cat([all_cond_images, all_new_images], dim=-1)
                saved_videos       = torch.clip(saved_videos.cpu() * 0.5 + 0.5, 0, 1)

                for batch_idx, (name, saved_video) in enumerate(zip(object_names, saved_videos)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    videos = []
                    for frame in saved_video:
                        frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
                        frame = Image.fromarray(frame.astype(np.uint8))
                        videos.append(frame)

                    videos[0].save(fp=os.path.join(object_path, "{:06}.gif".format(1)),
                                    format='png',
                                    append_images=videos[1:],
                                    save_all=True,
                                    duration=100,
                                    loop=0)

                ###############################################
                all_cond_images = self.reconstructor.decode(cond_latent, target_cameras)
                all_images      = torch.clip(all_cond_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name.item()))
                    os.makedirs(object_path, exist_ok=True)

                    for image in images:
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "cond_{:06}.png".format(1))
                        img.save(path)
                ###############################################

        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")


    @torch.inference_mode()
    def evaluate_shapenet_qualitative(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):            
            with self.accelerator.autocast():
                target_idx      = data["target_idx"]
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                target_cameras  = {"R" : data["target_camera_Rs"], \
                                        "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}
                cond_cameras = {}
                nviews = input_images.size(1)
                if self.num_test_views <= nviews:
                    num_views = self.num_test_views
                    cond_images = input_images[:, :num_views]
                    cond_cameras["R"] = input_cameras["R"][:, :num_views]
                    cond_cameras["T"] = input_cameras["T"][:, :num_views]
                    cond_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                    cond_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]
                elif self.num_test_views > nviews:
                    raise Exception("Sth wrong here")

                cond_latent, _ = self.reconstructor.encode(cond_images, cond_cameras)
                all_cond_images = self.reconstructor.decode(cond_latent, target_cameras)
                all_images      = torch.clip(all_cond_images.cpu() * 0.5 + 0.5, 0, 1)
                for batch_idx, (idx, name, images) in enumerate(zip(target_idx, object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, str(name))
                    os.makedirs(object_path, exist_ok=True)

                    for i, image in zip(idx, images):
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "{:06}.png".format(i.item()))
                        img.save(path)
                ###############################################

        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")


    
        

  
