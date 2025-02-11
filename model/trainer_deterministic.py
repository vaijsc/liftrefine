# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

from pathlib import Path
from collections import namedtuple
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam

from einops import reduce, rearrange

from tqdm.auto import tqdm

from accelerate import Accelerator
from torchvision.utils import make_grid

from logger.logger import Logger

import sys
import os
import imageio
from accelerate import DistributedDataParallelKwargs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lpips
import pickle

from model.metric import compute_fid, Metricator
from utils import exists, cycle, to_gpu, jet_depth, feature_map_pca, count_parameters, concat_camera,\
                    get_cosine_schedule_with_warmup, get_constant_hyperparameter_schedule_with_warmup, \
                    make_grid_4d, drop_view
from ema_pytorch import EMA

# trainer class
class TrainerDeterministic(object):
    def __init__(
        self,
        reconstructor,
        accelerator,
        train_dataloader=None,
        val_dataloader=None,
        train_batch_size=16,
        optimization_cfg=None,
        train_lr=1e-4,
        train_num_steps=100000,
        test_num_steps=100,
        gradient_accumulate_every=1,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        eval_every=1000,
        logging_every=100,
        summary_every=1000,
        save_every=1000,
        warmup_period=0,
        checkpoint_path=None,
        amp=False,
        fp16=False,
        split_batches=True,
        is_resume=False,
        logdir=None,
        run_name="pixelnerf",
    ):
        super().__init__()

        self.accelerator = accelerator
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        self.reconstructor = reconstructor

        self.lpips_weight   = optimization_cfg.lpips_weight
        self.rendered_weight = optimization_cfg.rendered_weight

        self.eval_every = eval_every
        self.save_every = save_every
        self.logging_every = logging_every
        self.summary_every = summary_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.test_num_steps  = test_num_steps

        # dataset and dataloader

        train_dataloader = self.accelerator.prepare(train_dataloader)
        self.train_dataloader = cycle(train_dataloader)
        val_dataloader = self.accelerator.prepare(val_dataloader)
        self.val_dataloader = cycle(val_dataloader)

        # optimizer
        reconstructor_params = [p for n, p in self.reconstructor.named_parameters() if p.requires_grad]
        reconstructor_names = [n for n, p in self.reconstructor.named_parameters() if p.requires_grad]

        self.reconstructor_opt = Adam(reconstructor_params, lr=train_lr, betas=adam_betas)
        reconstructor_scheduler = get_cosine_schedule_with_warmup(self.reconstructor_opt, warmup_period, \
                                                    train_num_steps, constant=optimization_cfg.constant_lr)


        if self.accelerator.is_main_process:
            num_params = count_parameters(self.reconstructor)
            print(f"Total number of trainable parameters - Reconstructor: {num_params // 1e6}M")

        self.reconstructor_ema = EMA(self.reconstructor, beta=ema_decay, update_every=ema_update_every, include_online_model=False)

        # step counter state
        self.step = 0
        # prepare model, dataloader, optimizer with accelerator
        self.reconstructor, self.reconstructor_opt, self.reconstructor_scheduler = self.accelerator.prepare(
                                                                    self.reconstructor, self.reconstructor_opt,\
                                                                    reconstructor_scheduler)
        
        
        if checkpoint_path is not None:
            self.load(checkpoint_path, is_resume)

        if self.accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = Logger(logdir)

            print(f"run dir: {logdir}")
            self.run_dir = logdir
            self.ckpt_dir = os.path.join(self.run_dir, "checkpoint")
            self.image_dir = os.path.join(self.run_dir, "images")
            self.results_folder = Path(self.run_dir)
            self.results_folder.mkdir(exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)


    def save(self, name):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "reconstructor": self.accelerator.get_state_dict(self.reconstructor),
            "reconstructor_opt": self.reconstructor_opt.state_dict(),
            "reconstructor_ema": self.reconstructor_ema.state_dict(),
            "reconstructor_scheduler": self.reconstructor_scheduler.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        torch.save(data, os.path.join(self.ckpt_dir, f"{name}.pt"))


    def load(self, path, is_resume=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(path),
            map_location=device,
        )

        reconstructor = self.accelerator.unwrap_model(self.reconstructor)
        reconstructor.load_state_dict(data["reconstructor"], strict=is_resume)

        if is_resume:
            self.step = data["step"]
            if "reconstructor_scheduler" in data:
                self.reconstructor_scheduler.load_state_dict(data["reconstructor_scheduler"])
          
            self.reconstructor_ema.load_state_dict(data["reconstructor_ema"], strict=False)

            if "version" in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data["scaler"]):
                self.accelerator.scaler.load_state_dict(data["scaler"])
            

    def get_losses(self, losses, prefix='train'):
        loss_dict = {}

        novel_rgb_loss    = losses["novel_rgb_loss"]/self.gradient_accumulate_every
        novel_lpips_loss  = losses["novel_lpips_loss"]/self.gradient_accumulate_every
        novel_loss        = novel_rgb_loss + novel_lpips_loss * self.lpips_weight
        loss_dict[f"{prefix}/reconstructor_loss"] = novel_loss.item()
        loss = novel_loss

        if self.rendered_weight > 0:
            rendered_rgb_loss    = losses["rendered_rgb_loss"]/self.gradient_accumulate_every
            rendered_lpips_loss  = losses["rendered_lpips_loss"]/self.gradient_accumulate_every
            rendered_loss = rendered_rgb_loss + rendered_lpips_loss * self.lpips_weight
            loss_dict[f"{prefix}/rendered_loss"] = rendered_loss.item()
            loss  += rendered_loss * self.rendered_weight

        return loss, loss_dict


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        best_psnr = 0.0
        best_lpips = float("inf")
        self.reconstructor_opt.zero_grad()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss_dict                             = {}
                total_loss_dict["train/reconstructor_loss"] = 0.0
                total_loss_dict["train/rendered_loss"]      = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dataloader)  # .to(device)
                    data = to_gpu(data, device)

                    with self.accelerator.autocast():
                        input_images    = data["input_images"]     
                        input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                    
                        target_images   = data["target_images"]
                        target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                        if self.rendered_weight > 0:
                            rendered_images = data["rendered_images"]
                            rendered_cameras  = {"R" : data["rendered_camera_Rs"], "T" : data["rendered_camera_Ts"], \
                                                "focal_lengths": data["rendered_focal_lengths"], \
                                                "principal_points": data["rendered_principal_points"]}
                        else:
                            rendered_images = None
                            rendered_cameras = None

                        reconstructor_loss_dict, misc = self.reconstructor(input_images, input_cameras, \
                                                                target_images, target_cameras, \
                                                                rendered_images, rendered_cameras)

                        reconstructor_loss, loss_dict = self.get_losses(reconstructor_loss_dict, "train")
                        for key in loss_dict:
                            total_loss_dict[key] += loss_dict[key]

                        loss = reconstructor_loss

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.reconstructor.parameters(), 1.0)
                total_loss = total_loss_dict["train/reconstructor_loss"] + total_loss_dict["train/rendered_loss"]
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.reconstructor_opt.step()
                self.reconstructor_opt.zero_grad()

                self.reconstructor_scheduler.step()
                accelerator.wait_for_everyone()

                if accelerator.is_main_process and self.step % self.logging_every == 0:
                    total_loss_dict["lr/reconstructor"] = self.reconstructor_scheduler.get_last_lr()[0]
                    self.logger.add_scalars(total_loss_dict, global_step=self.step)

                self.reconstructor_ema.update()
                if accelerator.is_main_process:
                    if self.step != 0 and self.step % self.save_every == 0:
                        self.save("last")

                    if self.step % self.summary_every == 0:  
                        with torch.inference_mode():
                            with self.accelerator.autocast():
                                _, misc = self.reconstructor_ema(input_images, input_cameras, \
                                                                target_images, target_cameras, \
                                                                rendered_images, rendered_cameras)
                                misc["target_cameras"] = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                            "focal_lengths": data["render_focal_lengths"], \
                                                            "principal_points": data["render_principal_points"]}
                                self.logger_summary(misc, "train")
          
                            data = next(self.val_dataloader)  # .to(device)
                            with self.accelerator.autocast():
                                input_images    = data["input_images"]      
                                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                    "focal_lengths": data["input_focal_lengths"], \
                                                    "principal_points": data["input_principal_points"]}
                                                    
                                target_images   = data["target_images"]
                                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                        "focal_lengths": data["target_focal_lengths"], \
                                                        "principal_points": data["target_principal_points"]}
                                
                                if self.rendered_weight > 0:
                                    rendered_images = data["rendered_images"]
                                    rendered_cameras  = {"R" : data["rendered_camera_Rs"], "T" : data["rendered_camera_Ts"], \
                                                        "focal_lengths": data["rendered_focal_lengths"], \
                                                        "principal_points": data["rendered_principal_points"]}
                                else:
                                    rendered_images = None
                                    rendered_cameras = None
                                
                                _, misc = self.reconstructor_ema(input_images, input_cameras, \
                                                                target_images, target_cameras, \
                                                                rendered_images, rendered_cameras)
                                misc["target_cameras"] = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                            "focal_lengths": data["render_focal_lengths"], \
                                                            "principal_points": data["render_principal_points"]}
                                self.logger_summary(misc, "test")


                if self.step % self.eval_every == 0 and self.step >= self.eval_every * 1:
                    reconstructor = self.reconstructor_ema.ema_model
                    metricator = Metricator(accelerator.device)
                    with torch.inference_mode():
                        all_psnr = []
                        all_ssim = []
                        all_lpips = []
                   
                        for i in tqdm(range(self.test_num_steps), disable=not accelerator.is_main_process):
                            data = next(self.val_dataloader) 
                            with self.accelerator.autocast():
                                input_images    = data["input_images"]
                                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                    "focal_lengths": data["input_focal_lengths"], \
                                                    "principal_points": data["input_principal_points"]}

                                target_images   = data["target_images"]
                                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                    "focal_lengths": data["target_focal_lengths"], \
                                                    "principal_points": data["target_principal_points"]}

                                cond_latent, _          = reconstructor.encode(input_images, input_cameras)
                                pred_images             = reconstructor.decode(cond_latent, target_cameras)
                                pred_images             = pred_images.flatten(0, 1)       
                                target_images           = target_images.flatten(0, 1)       
                                psnr, ssim, lpips       = metricator.compute_metrics(pred_images, target_images)
                                
                                psnr                = accelerator.gather(psnr).cpu()
                                ssim                = accelerator.gather(ssim).cpu()
                                lpips               = accelerator.gather(lpips).cpu()
                       
                                if accelerator.is_main_process:
                                    all_psnr.append(psnr)
                                    all_ssim.append(ssim)
                                    all_lpips.append(lpips)
                                  

                        if accelerator.is_main_process:
                            all_psnr = torch.cat(all_psnr, dim=0).mean().item()
                            all_ssim = torch.cat(all_ssim, dim=0).mean().item()
                            all_lpips = torch.cat(all_lpips, dim=0).mean().item()
                        
                            self.logger.add_scalars({"PSNR/det" : all_psnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM/det" : all_ssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS/det" : all_lpips}, global_step=self.step)
                
                            if best_psnr < all_psnr:
                                print(f"Saving best at step {self.step} - best_psnr: {best_psnr} < psnr: {all_psnr}\n")
                                best_psnr = all_psnr
                                self.save("best_psnr")
                            else:
                                print(f"Skipping at {self.step} - best_psnr: {best_psnr} > psnr: {all_psnr}\n")
                        
                            if best_lpips > all_lpips:
                                best_lpips = all_lpips
                                self.save("best_lpips")
                                       
                self.step += 1
                pbar.update(1)

        accelerator.print("training complete")


    @torch.inference_mode
    def logger_summary(self, misc, prefix="train"):
        print("logger summary")
        self.logger_reconstructor(misc, prefix)

        if self.rendered_weight > 0:
            rgbs = misc["rendered_rgbs"].flatten(0, 1).cpu().detach() * 0.5 + 0.5
            gt_rgbs = misc["gt_rendered_rgbs"].flatten(0, 1).cpu().detach() * 0.5 + 0.5
            depths = misc["rendered_depths"].flatten(0, 1).squeeze(1).cpu().detach()

            depths   = torch.from_numpy(jet_depth(depths)).permute(0, 3, 1, 2)
            
            saved_images = torch.cat([gt_rgbs, rgbs, depths], dim=-1)
            saved_images = torch.nn.functional.interpolate(saved_images, scale_factor=8.0, mode="bilinear")
            saved_images = make_grid(saved_images, nrow=1, padding=0)

            image_dict = {
                f"visualization/{prefix}_rendered": saved_images,
            }
            saved_images = np.clip(saved_images.permute(1, 2, 0).numpy(), 0, 1) * 255.0
            img = Image.fromarray(saved_images.astype(np.uint8))
            img.save(os.path.join(self.image_dir, f"{prefix}_rendered_step-{self.step}.png"))
            self.logger.add_images(image_dict, self.step)


    def logger_reconstructor(self, misc, preprefix="train"):
        with torch.inference_mode():
            render_cameras  = misc["target_cameras"]
            render_images   = self.reconstructor_ema.ema_model.decode(misc["latent"], render_cameras).cpu().detach() * 0.5 + 0.5

        input_images = misc["input"].cpu().detach()
        target_images = misc["target"].cpu().detach()
        
        novel_view_images = misc["novel_rgbs"].cpu().detach()
        novel_view_depths = misc["novel_depths"].squeeze(2).cpu().detach()
    
        bs              = input_images.size(0)
        num_cond        = input_images.size(1)
        num_render      = target_images.size(1)

        input_images        = rearrange(input_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        target_images       = rearrange(target_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        novel_view_images   = rearrange(novel_view_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5 
        novel_view_depths   = rearrange(novel_view_depths, "b t h w -> (b t) h w") 

        novel_view_depths   = torch.from_numpy(jet_depth(novel_view_depths)).permute(0, 3, 1, 2)

        target_images   = torch.cat([target_images, novel_view_depths, \
                                                    novel_view_images], dim=-1)

        render_videos   = make_grid_4d(render_images, nrow=num_render)
        input_images    = make_grid(input_images, nrow=num_cond, padding=0)
        target_images   = make_grid(target_images, nrow=num_render, padding=0)

        input_images    = torch.nn.functional.pad(input_images, (4, 4, 4, 4), mode='constant', value=0.3)
        target_images   = torch.nn.functional.pad(target_images, (4, 4, 4, 4), mode='constant', value=0.8)
        
        image_dict = {
            f"visualization/{preprefix}_input": input_images,
            f"visualization/{preprefix}_target": target_images,
        }
        saved_images = torch.cat([input_images, target_images], dim=-1)
        saved_images = np.clip(saved_images.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{preprefix}_image_step-{self.step}.png"))

        videos = []
        for frame in render_videos:
            frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
            frame = Image.fromarray(frame.astype(np.uint8))
            videos.append(frame)

        videos[0].save(fp=os.path.join(self.image_dir, f"{preprefix}_video_step-{self.step}.gif"),
                        format='png',
                        append_images=videos[1:],
                        save_all=True,
                        duration=100,
                        loop=0)
                            
        self.logger.add_images(image_dict, self.step)
        self.logger.add_videos(f"visualization/{preprefix}_render_videos", render_videos.unsqueeze(0), self.step)
        self.visualize_features(misc["novel_features"], preprefix)
       
            
    def visualize(self, rgbs, depths, features, prefix="clean"):
        video_rgbs  = rgbs.cpu() * 0.5 + 0.5
        depths      = depths.squeeze(2).cpu()
        features    = features.cpu()
        self.visualize_features(features, prefix)

        num_render      = rgbs.size(1)
        depths          = rearrange(depths, "b t h w -> (b t) h w")
        colorized_depth = torch.from_numpy(jet_depth(depths)).permute(0, 3, 1, 2)
        video_depths    = rearrange(colorized_depth, "(b t) c h w -> b t c h w", t=num_render)
        torch.cuda.empty_cache()

        return video_rgbs, video_depths



    def visualize_features(self, features, prefix="clean"):
        features    = features.cpu().detach()
        if features.size(1) == 3:
            # should be triplane
            hw_features      = features[:, 0]
            dw_features      = features[:, 1]
            dh_features      = features[:, 2]

            hw_features = torch.nn.functional.interpolate(hw_features, (256, 256), mode='bilinear')
            dw_features = torch.nn.functional.interpolate(dw_features, (256, 256), mode='bilinear')
            dh_features = torch.nn.functional.interpolate(dh_features, (256, 256), mode='bilinear')
        else:
            hw_features = rearrange(features, "bs c d h w -> bs (c d) h w")
            dw_features = rearrange(features, "bs c d h w -> bs (c h) d w")
            dh_features = rearrange(features, "bs c d h w -> bs (c w) d h")

            hw_features = torch.nn.functional.interpolate(hw_features, scale_factor=16.0, mode='bilinear')
            dw_features = torch.nn.functional.interpolate(dw_features, scale_factor=16.0, mode='bilinear')
            dh_features = torch.nn.functional.interpolate(dh_features, scale_factor=16.0, mode='bilinear')

        hw_images = feature_map_pca(hw_features).unsqueeze(1)
        dw_images = feature_map_pca(dw_features).unsqueeze(1)
        dh_images = feature_map_pca(dh_features).unsqueeze(1)

        features_image = torch.cat([hw_images, dw_images, dh_images], dim=1)
        features_image = rearrange(features_image, "bs n c h w -> (bs n) c h w")
        features_image   = make_grid(features_image, nrow=3)
        pca_image_dict = {
            f"visualization/{prefix}_feature": features_image,
        }

        saved_images = np.clip(features_image.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{prefix}_feature_step-{self.step}.png"))

        self.logger.add_images(pca_image_dict, self.step)