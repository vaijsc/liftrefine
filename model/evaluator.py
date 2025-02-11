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

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, rearrange

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from utils import exists, to_gpu, jet_depth
from ema_pytorch import EMA

# trainer class
class Evaluator(object):
    def __init__(
        self,
        reconstruction_model,
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
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp
        self.use_rays       = optimization_cfg.use_rays
        self.test_ema       = test_ema
        self.num_test_views = num_test_views

        if self.test_ema:
            self.model      = EMA(reconstruction_model, include_online_model=False)
        else:
            self.model      = reconstruction_model

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
            self.model.load_state_dict(data["ema"], strict=True)
            self.model = self.model.ema_model
        else:
            print("Loading model")
            self.model.load_state_dict(data["model"], strict=True)

    @torch.inference_mode()
    def evaluate_shapenet(self):
        accelerator = self.accelerator
        device = accelerator.device

        for data in tqdm(self.dataloader):
            target_idx  = data["target_idx"]
            data        = to_gpu(data, device)
            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                if self.use_rays:
                    input_rays      = data["input_rays"]
                    input_images    = torch.cat([input_images, input_rays], dim=2)

                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                
                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                all_images      = self.model.inference(input_images, input_cameras, target_cameras)
                all_images      = torch.clip(all_images*0.5 + 0.5, 0, 1)

                for batch_idx, (name, images) in enumerate(zip(object_names, all_images)):
                    object_path = os.path.join(self.eval_dir, name)
                    os.makedirs(object_path, exist_ok=True)
                    for view_idx, image in zip(target_idx[batch_idx], images):
                        saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img         = Image.fromarray(saved_image)
                        path        = os.path.join(object_path, "{:06}.png".format(view_idx))
                        img.save(path)
                        
        accelerator.wait_for_everyone()
        accelerator.print("evaluation complete")


    @torch.inference_mode()
    def evaluate_co3d(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"Starting to evalaute CO3D with {self.num_test_views} views")
        for data in tqdm(self.dataloader):
            data        = to_gpu(data, device)
            
            with self.accelerator.autocast():
                object_names    = data["object_names"]
                input_images    = data["input_images"]
                if self.use_rays:
                    input_rays      = data["input_rays"]
                    input_images    = torch.cat([input_images, input_rays], dim=2)

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

                all_images      = self.model.inference(input_images, input_cameras, target_cameras)
                all_images      = torch.clip(all_images*0.5 + 0.5, 0, 1)

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


    
        

  
