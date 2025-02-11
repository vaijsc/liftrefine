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
class TrainerSingleStep(object):
    def __init__(
        self,
        unet_model,
        reconstructor,
        diffusion_model,
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
        pretrained_reconstructor=None,
        checkpoint_path=None,
        amp=False,
        fp16=False,
        split_batches=True,
        is_resume=False,
        logdir=None,
        run_name="pixelnerf",
        stat_path=""
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

        # self.accelerator.native_amp = amp

        self.reconstructor = reconstructor
        self.diffusion_model = diffusion_model
        self.unet_model = unet_model

        self.classifier_free_guidance = optimization_cfg.classifier_free_guidance
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
        unet_params = [p for n, p in self.unet_model.named_parameters() if p.requires_grad]
        unet_names = [n for n, p in self.unet_model.named_parameters() if p.requires_grad]

        self.unet_opt = Adam(unet_params, lr=train_lr, betas=adam_betas)
        unet_scheduler = get_cosine_schedule_with_warmup(self.unet_opt, warmup_period, \
                                                    train_num_steps, constant=optimization_cfg.constant_lr)

        if self.accelerator.is_main_process:
            num_params = count_parameters(self.unet_model)
            print(f"Total number of trainable parameters - Unet: {num_params // 1e6}M")


        self.unet_ema = EMA(self.unet_model, beta=ema_decay, update_every=ema_update_every, include_online_model=False)

        # step counter state
        self.step = 0
        # prepare model, dataloader, optimizer with accelerator
        self.unet_model, self.unet_opt, self.unet_scheduler = self.accelerator.prepare(
                                                                    self.unet_model, self.unet_opt,\
                                                                    unet_scheduler)
        
        assert pretrained_reconstructor is not None
        if pretrained_reconstructor is not None:
            self.load_reconstructor(pretrained_reconstructor)

        if checkpoint_path is not None:
            self.load(checkpoint_path, is_resume)
        
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
     
        for param in self.reconstructor.parameters():
            param.requires_grad = False

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
            "unet_model": self.accelerator.get_state_dict(self.unet_model),
            "unet_opt": self.unet_opt.state_dict(),
            "unet_ema": self.unet_ema.state_dict(),
            "unet_scheduler": self.unet_scheduler.state_dict(),
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

        unet_model = self.accelerator.unwrap_model(self.unet_model)
        unet_model.load_state_dict(data["unet_model"], strict=is_resume)

        if is_resume:
            self.step = data["step"]
            if "unet_scheduler" in data:
                self.unet_scheduler.load_state_dict(data["unet_scheduler"])

            if self.accelerator.is_main_process:
                self.unet_ema.load_state_dict(data["unet_ema"], strict=False)

            if "version" in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data["scaler"]):
                self.accelerator.scaler.load_state_dict(data["scaler"])


    def load_reconstructor(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(path),
            map_location=device,
        )

        self.reconstructor      = EMA(self.reconstructor, include_online_model=False)
        self.diffusion_model    = EMA(self.diffusion_model, include_online_model=False)
        
        self.reconstructor.load_state_dict(data["reconstructor_ema"], strict=True)
        self.diffusion_model.load_state_dict(data["diffusion_ema"], strict=True)

        self.reconstructor = self.reconstructor.ema_model
        self.diffusion_model = self.diffusion_model.ema_model
            

    def get_losses(self, losses, prefix='train'):
        loss_dict = {}

        novel_rgb_loss    = losses["novel_rgb_loss"]/self.gradient_accumulate_every
        novel_lpips_loss  = losses["novel_lpips_loss"]/self.gradient_accumulate_every
       
        loss = novel_rgb_loss + novel_lpips_loss * self.lpips_weight
        loss_dict[f"{prefix}/reconstructor_loss"] = loss.item()

        return loss, loss_dict


    def predict_original(self, unet, cond, num_train_timesteps=1000):
        input_noise = torch.randn_like(cond)
        max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
        max_timesteps = max_timesteps * (num_train_timesteps - 1)

        alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047)**0.5
        model_pred = unet(input_noise, max_timesteps, cond)
        
        latents = (input_noise - sigma_T * model_pred) / alpha_T
        return latents


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        best_psnr = 0.0
        best_psnr_gap = float("inf")
        self.unet_opt.zero_grad()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss_dict                   = {}
                total_loss_dict["train/sds_loss"] = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dataloader)  # .to(device)

                    with self.accelerator.autocast():
                        input_images    = data["input_images"]     
                        input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                    
                        target_images   = data["target_images"]
                        target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                        cond_images, cond_cameras = drop_view(input_images, input_cameras)
                        with torch.no_grad():
                            _, volume_features = self.reconstructor.encode(cond_images, cond_cameras)
                            target_features = self.reconstructor.model.render_volumes(volume_features, target_cameras)[1]
                        
                        pred_original_samples = self.predict_original(self.unet_model, volume_features, num_train_timesteps=1000)
                        pred_target_images = self.reconstructor.decode_volumes(pred_original_samples, target_cameras)

                        clip_images = data["clip_images"][:, 0]
                        rendered_images = torch.nn.functional.interpolate(pred_target_images.flatten(0, 1),  mode='bilinear', scale_factor=2)

                        batch = {"image_clip" : clip_images,\
                                 "image_target": rendered_images, \
                                 "image_cond" : target_features.flatten(0, 1)}

                        sds_loss = self.diffusion_model.foward_sds_loss(batch, cfg=self.classifier_free_guidance)
                        sds_loss /= self.gradient_accumulate_every
                        total_loss_dict["train/sds_loss"] += sds_loss.item()

                    self.accelerator.backward(sds_loss)

                accelerator.clip_grad_norm_(self.unet_model.parameters(), 1.0)
                total_loss = total_loss_dict["train/sds_loss"]
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.unet_opt.step()
                self.unet_opt.zero_grad()

                self.unet_scheduler.step()
                accelerator.wait_for_everyone()

                if accelerator.is_main_process and self.step % self.logging_every == 0:
                    total_loss_dict["lr"] = self.unet_scheduler.get_last_lr()[0]
                    self.logger.add_scalars(total_loss_dict, global_step=self.step)

                self.unet_ema.update()

                
                if accelerator.is_main_process:
                    if self.step != 0 and self.step % self.save_every == 0:
                        self.save("last")

                    if self.step % self.summary_every == 0:  
                        self.logger_summary(data, prefix="train")

                        data_test = next(self.val_dataloader) 
                        self.logger_summary(data_test, prefix="test")
                        
                if self.step % self.eval_every == 0 and self.step >= self.eval_every * 1:
                    metricator = Metricator(accelerator.device)
                    with torch.inference_mode():
                        all_psnr = []
                        all_ssim = []
                        all_lpips = []
                   
                        all_spsnr = []
                        all_sssim = []
                        all_slpips = []

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

                                cond_latent, volume_features = self.reconstructor.encode(input_images, input_cameras)
                                pred_images = self.reconstructor.decode(cond_latent, target_cameras)

                                pred_original_samples = self.predict_original(self.unet_ema.ema_model, volume_features, num_train_timesteps=1000)
                                sample_images = self.reconstructor.decode_volumes(pred_original_samples, target_cameras)
                                
                                pred_images             = pred_images.flatten(0, 1)       
                                sample_images           = sample_images.flatten(0, 1)       
                                target_images           = target_images.flatten(0, 1)       
                                psnr, ssim, lpips       = metricator.compute_metrics(pred_images, target_images)
                                spsnr, sssim, slpips    = metricator.compute_metrics(sample_images, target_images) # sample psnr

                                psnr                = accelerator.gather(psnr).cpu()
                                ssim                = accelerator.gather(ssim).cpu()
                                lpips               = accelerator.gather(lpips).cpu()
                       
                                spsnr                = accelerator.gather(spsnr).cpu()
                                sssim                = accelerator.gather(sssim).cpu()
                                slpips               = accelerator.gather(slpips).cpu()

                                if accelerator.is_main_process:
                                    all_psnr.append(psnr)
                                    all_ssim.append(ssim)
                                    all_lpips.append(lpips)
                                  
                                    all_spsnr.append(spsnr)
                                    all_sssim.append(sssim)
                                    all_slpips.append(slpips)

                        if accelerator.is_main_process:
                            all_psnr = torch.cat(all_psnr, dim=0).mean().item()
                            all_ssim = torch.cat(all_ssim, dim=0).mean().item()
                            all_lpips = torch.cat(all_lpips, dim=0).mean().item()
                        
                            all_spsnr = torch.cat(all_spsnr, dim=0).mean().item()
                            all_sssim = torch.cat(all_sssim, dim=0).mean().item()
                            all_slpips = torch.cat(all_slpips, dim=0).mean().item()
                        

                            self.logger.add_scalars({"PSNR/det" : all_psnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM/det" : all_ssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS/det" : all_lpips}, global_step=self.step)
                        
                            self.logger.add_scalars({"PSNR/sample" : all_spsnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM/sample" : all_sssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS/sample" : all_slpips}, global_step=self.step)

                            self.logger.add_scalars({"PSNR/gap" : all_psnr - all_spsnr}, global_step=self.step)

                            if best_psnr < all_psnr:
                                print(f"Saving best at step {self.step} - best_psnr: {best_psnr} < psnr: {all_psnr}\n")
                                best_psnr = all_psnr
                                self.save("best_psnr")
                            else:
                                print(f"Skipping at {self.step} - best_psnr: {best_psnr} > psnr: {all_psnr}\n")
                            
                            psnr_gap = all_psnr - all_spsnr
                            if best_psnr_gap > psnr_gap:
                                print(f"Saving best gap at step {self.step} - psnr_gap: {psnr_gap} < best_psnr_gap: {best_psnr_gap}\n")
                                best_psnr_gap = psnr_gap
                                self.save("best_gap")
                            else:
                                print(f"Skipping at {self.step} - best_psnr_gap: {best_psnr_gap} < psnr_gap: {psnr_gap}\n")
                
                self.step += 1
                pbar.update(1)

        accelerator.print("training complete")

    
    @torch.inference_mode
    def logger_summary(self, data, prefix="train"):
        with self.accelerator.autocast():
            input_images    = data["input_images"]     
            input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                "focal_lengths": data["input_focal_lengths"], \
                                "principal_points": data["input_principal_points"]}
        
            target_images   = data["target_images"]
            target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                "focal_lengths": data["target_focal_lengths"], \
                                "principal_points": data["target_principal_points"]}

            cond_images, cond_cameras = drop_view(input_images, input_cameras)
                
            _, volume_features = self.reconstructor.encode(cond_images, cond_cameras)
            pred_target_images = self.reconstructor.decode_volumes(volume_features, target_cameras)

            pred_original_samples = self.predict_original(self.unet_model, volume_features, num_train_timesteps=1000)
            sample_target_images = self.reconstructor.decode_volumes(pred_original_samples, target_cameras)

            render_cameras = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                        "focal_lengths": data["render_focal_lengths"], \
                                        "principal_points": data["render_principal_points"]}

            pred_render_images   = self.reconstructor.decode_volumes(volume_features, render_cameras).cpu().detach() 
            sample_render_images   = self.reconstructor.decode_volumes(pred_original_samples, render_cameras).cpu().detach() 

        num_cond = cond_images.size(1)
        num_render = target_images.size(1)
        cond_images   = rearrange(cond_images.cpu(), "b t c h w -> (b t) c h w") * 0.5 + 0.5
        target_images = rearrange(target_images.cpu(), "b t c h w -> (b t) c h w") * 0.5 + 0.5
        pred_images   = rearrange(pred_target_images.cpu(), "b t c h w -> (b t) c h w") * 0.5 + 0.5 
        sample_images = rearrange(sample_target_images.cpu(), "b t c h w -> (b t) c h w") * 0.5 + 0.5 

        target_images   = torch.cat([target_images, pred_images, sample_images], dim=-1)

        cond_images     = make_grid(cond_images, nrow=num_cond, padding=0)
        target_images   = make_grid(target_images, nrow=num_render, padding=0)

        cond_images    = torch.nn.functional.pad(cond_images, (4, 4, 4, 4), mode='constant', value=0.3)
        target_images   = torch.nn.functional.pad(target_images, (4, 4, 4, 4), mode='constant', value=0.8)

        saved_images = torch.cat([cond_images, target_images], dim=-1)

        image_dict = {
            f"visualization/{prefix}_image": saved_images,
        }
        self.logger.add_images(image_dict, self.step)

        saved_images = np.clip(saved_images.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{prefix}_image_step-{self.step}.png"))


        #############################################

        render_videos = torch.cat([pred_render_images, sample_render_images], dim=-1) * 0.5 + 0.5 
        render_videos = make_grid_4d(render_videos, nrow=2)
        
        videos = []
        for frame in render_videos:
            frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
            frame = Image.fromarray(frame.astype(np.uint8))
            videos.append(frame)

        videos[0].save(fp=os.path.join(self.image_dir, f"{prefix}_video_step-{self.step}.gif"),
                        format='png',
                        append_images=videos[1:],
                        save_all=True,
                        duration=100,
                        loop=0)
                            
        self.logger.add_videos(f"visualization/{prefix}_videos", render_videos.unsqueeze(0), self.step)