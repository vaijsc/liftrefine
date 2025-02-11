# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

from pathlib import Path
from collections import namedtuple
from collections import OrderedDict
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam

from einops import reduce, rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

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

from model.metric import compute_fid
from utils import exists, cycle, to_gpu, jet_depth, feature_map_pca, count_parameters, concat_camera,\
                    get_cosine_schedule_with_warmup, get_constant_hyperparameter_schedule_with_warmup, \
                    make_grid_4d
from model.metric import Metricator

# trainer class
class Trainer1StageCond(object):
    def __init__(
        self,
        diffuser,
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
        random_drop_view=False,
        stat_path=""
    ):
        super().__init__()

        self.accelerator = accelerator
        self.inception_path = "../../pretrained/metric/inception-2015-12-05.pt"
        self.stat_path = stat_path
        
        if accelerator.is_main_process:
            with open(stat_path, "rb") as f:
                data = pickle.load(f)
                self.mean   = data["mean"]
                self.cov    = data["cov"]

        self.num_gpus = accelerator.state.num_processes
        accelerator.print(f"Number of GPUs available: {self.num_gpus}")
        self.num_samples = 30000 // self.num_gpus        # number of sample image per gpu

        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp
        self.lpips_weight  = optimization_cfg.lpips_weight
        self.recons_weight = optimization_cfg.recons_weight
        self.novel_weight  = optimization_cfg.novel_weight
        self.tv_weight     = optimization_cfg.tv_weight
        self.depth_weight  = optimization_cfg.depth_weight
        self.diffusion_weight   = optimization_cfg.diffusion_weight
        self.use_rays           = optimization_cfg.use_rays
        

        self.diffusion_model = diffuser
        self.recons_model = reconstructor
        self.random_drop_view = random_drop_view
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
        self.diffusion_dataloader = cycle(val_dataloader)
        self.recons_dataloader = val_dataloader

        # set up for diffusion model
        params = [p for n, p in self.diffusion_model.named_parameters() if p.requires_grad]
        names = [n for n, p in self.diffusion_model.named_parameters() if p.requires_grad]

        self.diffusion_opt = Adam(params, lr=train_lr, betas=adam_betas)
        diffusion_lr_scheduler = get_cosine_schedule_with_warmup(
            self.diffusion_opt, warmup_period, train_num_steps, constant=optimization_cfg.constant_lr
        )

        if self.accelerator.is_main_process:
            num_params = count_parameters(self.diffusion_model)
            # print(f"****************Optimization layer****************\n", names)
            print(f"Total number of trainable parameters in diffusion model: {num_params // 1e6}M")


        self.diffusion_ema = EMA(self.diffusion_model, beta=ema_decay, update_every=ema_update_every, include_online_model=False)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion_model, self.diffusion_opt, self.diffusion_lr_scheduler = self.accelerator.prepare(
            self.diffusion_model, self.diffusion_opt, diffusion_lr_scheduler
        )
        
        
        # set up for reconstructor
        params = [p for n, p in self.recons_model.named_parameters() if "perceptual" not in n]
        names = [n for n, p in self.recons_model.named_parameters() if "perceptual" not in n]

        self.recons_opt = Adam(params, lr=train_lr, betas=adam_betas)
        recons_lr_scheduler = get_cosine_schedule_with_warmup(
            self.recons_opt, warmup_period, train_num_steps, constant=optimization_cfg.constant_lr
        )

        if self.accelerator.is_main_process:
            num_params = count_parameters(self.recons_model)
            # print(f"****************Optimization layer****************\n", names)
            print(f"Total number of trainable parameters in reconstruction model: {num_params // 1e6}M")


        self.recons_ema = EMA(self.recons_model, beta=ema_decay, update_every=ema_update_every, include_online_model=False)

        # prepare model, dataloader, optimizer with accelerator
        self.recons_model, self.recons_opt, self.recons_lr_scheduler = self.accelerator.prepare(
            self.recons_model, self.recons_opt, recons_lr_scheduler
        )

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
            "recons_model": self.accelerator.get_state_dict(self.recons_model),
            "recons_opt": self.recons_opt.state_dict(),
            "recons_ema": self.recons_ema.state_dict(),
            "recons_lr_scheduler": self.recons_lr_scheduler.state_dict(),
       
            "diffusion_model": self.accelerator.get_state_dict(self.diffusion_model),
            "diffusion_opt": self.diffusion_opt.state_dict(),
            "diffusion_ema": self.diffusion_ema.state_dict(),
            "diffusion_lr_scheduler": self.diffusion_lr_scheduler.state_dict(),

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

        recons_model = self.accelerator.unwrap_model(self.recons_model)
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        recons_model.load_state_dict(data["recons_model"], strict=is_resume)
   
        diffusion_model = self.accelerator.unwrap_model(self.diffusion_model)
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        diffusion_model.load_state_dict(data["diffusion_model"], strict=is_resume)

        if is_resume:
            self.step = data["step"]
            if "recons_lr_scheduler" in data:
                self.recons_lr_scheduler.load_state_dict(data["recons_lr_scheduler"])

            if "recons_ema" in data:
                self.recons_ema.load_state_dict(data["recons_ema"], strict=True)
           
            if "recons_opt" in data:
                self.recons_opt.load_state_dict(data["recons_opt"])
         
            if "diffusion_lr_scheduler" in data:
                self.diffusion_lr_scheduler.load_state_dict(data["diffusion_lr_scheduler"])

            if "diffusion_ema" in data:
                self.diffusion_ema.load_state_dict(data["diffusion_ema"], strict=True)
           
            if "diffusion_opt" in data:
                self.diffusion_opt.load_state_dict(data["diffusion_opt"])

            if "version" in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data["scaler"]):
                self.accelerator.scaler.load_state_dict(data["scaler"])

                
    def get_losses(self, losses, prefix='train'):
        loss_dict = {}

        novel_view_rgb_loss     = losses["novel_view_rgb_loss"]
        novel_view_lpips_loss   = losses["novel_view_lpips_loss"]
        recons_rgb_loss         = losses["recons_rgb_loss"]
        recons_lpips_loss       = losses["recons_lpips_loss"]
    
        loss_dict[f"{prefix}/novel_view_rgb_loss"]    = novel_view_rgb_loss.item()
        loss_dict[f"{prefix}/novel_view_lpips_loss"]  = novel_view_lpips_loss.item()
        loss_dict[f"{prefix}/recons_rgb_loss"]        = recons_rgb_loss.item()
        loss_dict[f"{prefix}/recons_lpips_loss"]      = recons_lpips_loss.item()

        novel_view_loss = novel_view_rgb_loss + novel_view_lpips_loss * self.lpips_weight
        recons_loss     = recons_rgb_loss + recons_lpips_loss * self.lpips_weight

        loss = recons_loss * self.recons_weight + novel_view_loss * self.novel_weight 
        
        if self.tv_weight > 0:
            tv_loss                         = losses["tv_loss"]
            loss_dict[f"{prefix}/tv_loss"]  = tv_loss.item()
            loss                            += tv_loss * self.tv_weight 
        
        if self.depth_weight > 0:
            depth_loss                          = losses["depth_loss"]
            loss_dict[f"{prefix}/depth_loss"]   = depth_loss.item()
            loss                                += depth_loss * self.depth_weight 
        
        loss /= self.gradient_accumulate_every
        return loss, loss_dict   


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        best_fid_score = float("inf")
        best_psnr = 0.0

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss_dict                                 = {}
                total_loss_dict["train/novel_view_rgb_loss"]    = 0.0
                total_loss_dict["train/novel_view_lpips_loss"]  = 0.0
                total_loss_dict["train/recons_rgb_loss"]        = 0.0
                total_loss_dict["train/recons_lpips_loss"]      = 0.0
                total_loss_dict["train/diffusion_loss"]         = 0.0
                total_loss_dict["train/total_loss"]             = 0.0
                
                if self.tv_weight > 0:
                    total_loss_dict["train/tv_loss"] = 0.0
                if self.depth_weight > 0:
                    total_loss_dict["train/depth_loss"] = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dataloader)  # .to(device)
                    data = to_gpu(data, device)

                    with self.accelerator.autocast():
                        input_images    = data["input_images"]

                        if self.use_rays:
                            input_rays      = data["input_rays"]
                            input_images    = torch.cat([input_images, input_rays], dim=2)
                            
                        input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                        
                        if self.random_drop_view:
                            nviews = input_images.size(1)
                            num_views = torch.randint(nviews, (1,)) + 1
                            input_images = input_images[:, :num_views]
                            input_cameras["R"] = input_cameras["R"][:, :num_views]
                            input_cameras["T"] = input_cameras["T"][:, :num_views]
                            input_cameras["focal_lengths"] = input_cameras["focal_lengths"][:, :num_views]
                            input_cameras["principal_points"] = input_cameras["principal_points"][:, :num_views]

                        target_images   = data["target_images"]
                        target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                        losses, misc = self.recons_model(input_images, input_cameras, target_images, target_cameras)
                        latent  = misc["latent"]
                        volume  = misc["volume"]

                        diffusion_loss, misc_diffusion = self.diffusion_model(latent, volume)
                        rendering_loss, loss_dict = self.get_losses(losses, "train")
                        loss = (diffusion_loss/self.gradient_accumulate_every) * self.diffusion_weight + rendering_loss

                        loss_dict["train/diffusion_loss"] = diffusion_loss.item()
                        loss_dict["train/total_loss"] = loss.item()
                        for key in loss_dict:
                            total_loss_dict[key] += loss_dict[key]

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.recons_model.parameters(), 1.0)
                total_loss = total_loss_dict["train/total_loss"] + total_loss_dict["train/diffusion_loss"]
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.recons_opt.step()
                self.recons_opt.zero_grad()
           
                self.diffusion_opt.step()
                self.diffusion_opt.zero_grad()

                accelerator.wait_for_everyone()
                self.recons_ema.update()
                self.diffusion_ema.update()

                if accelerator.is_main_process and self.step % self.logging_every == 0:
                    total_loss_dict["recons_lr"]            = self.recons_lr_scheduler.get_last_lr()[0]
                    total_loss_dict["diffusion_lr"]         = self.diffusion_lr_scheduler.get_last_lr()[0]
                    total_loss_dict["sanity/feature_max"]   = misc["features"].max().item()
                    total_loss_dict["sanity/feature_min"]   = misc["features"].min().item()
                    total_loss_dict["sanity/feature_mean"]  = misc["features"].mean().item()
                    total_loss_dict["sanity/feature_std"]   = misc["features"].std().item()
                    total_loss_dict["norm_factor"]          = misc_diffusion["norm_factor"].item()
                
                    self.logger.add_scalars(total_loss_dict, global_step=self.step)
                    with torch.inference_mode():
                        losses, misc = self.recons_ema(input_images, input_cameras, target_images, target_cameras)

                    loss, loss_dict = self.get_losses(losses, "ema")
                    self.logger.add_scalars(loss_dict, global_step=self.step)


                if accelerator.is_main_process:
                    if self.step % self.summary_every == 0:
                        with torch.inference_mode():
                            _, misc_recons      = self.recons_ema(input_images, input_cameras, target_images, target_cameras)
                            _, misc_diffusion   = self.diffusion_ema(misc_recons["latent"], misc_recons["volume"])

                        render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                            "focal_lengths": data["render_focal_lengths"], \
                                            "principal_points": data["render_principal_points"]}
                        misc_recons["render_cameras"]  = render_cameras       # for video rendering
                        misc_recons["input_cameras"]   = input_cameras
                        
                        misc_diffusion["input"]          = misc_recons["input"]
                        misc_diffusion["target_cameras"] = render_cameras
                        misc_diffusion["volume"]         = misc_recons["volume"]
                        self.logger_summary_recons(misc_recons)
                        self.logger_summary_diffusion(misc_diffusion)
                                           

                    if self.step != 0 and self.step % self.save_every == 0:
                        self.save("last")


                if self.step % self.eval_every == 0 and self.step >= self.eval_every * 1:
                    
                    diffusion_model = self.diffusion_ema.ema_model    
                    recons_model    = self.recons_ema.ema_model
                    with torch.inference_mode():
                        num_samples = 0
                        sample_images = []
                        
                        test_pbar = tqdm(total=self.num_samples)
                        while num_samples < self.num_samples:
                            print(f"num_samples: {num_samples}")
                            data = next(self.diffusion_dataloader)  # .to(device)

                            with self.accelerator.autocast():
                                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                    "focal_lengths": data["input_focal_lengths"], \
                                                    "principal_points": data["input_principal_points"]}

                                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                    "focal_lengths": data["target_focal_lengths"], \
                                                    "principal_points": data["target_principal_points"]}
                              
                                sampling_cameras  = {"R" : data["sampling_camera_Rs"], "T" : data["sampling_camera_Ts"], \
                                                    "focal_lengths": data["sampling_focal_lengths"], \
                                                    "principal_points": data["sampling_principal_points"]}

                                cameras                  = concat_camera([input_cameras, target_cameras, sampling_cameras])
                                _, volume_features       = self.recons_ema.ema_model.encode(input_images, input_cameras)
                                sample_latent            = diffusion_model.ddim_sample(latent.shape, volume_features, classifier_free_guidance_w=2.0)
                                sample_pred_images, _, _ = recons_model.decode(sample_latent, cameras)
                                
                                sample_pred_images = sample_pred_images.flatten(0, 1)
                                size = sample_pred_images.size(0)
                                num_samples += size
                                test_pbar.update(size)
                                sample_images.append(sample_pred_images.cpu())

                        all_features = []
                        sample_images = torch.cat(sample_images, dim=0)
                        sample_images = (sample_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        inception_network = torch.jit.load(self.inception_path).eval().to(device)

                        list_sample_images = torch.split(sample_images, 32, dim=0)
                        for sample_images in list_sample_images:
                            sample_images = sample_images.to(device)
                            features = inception_network(sample_images, return_features=True)
                            features = accelerator.gather(features)
                            all_features.append(features)

                        all_features = torch.cat(all_features, dim=0)

                        if accelerator.is_main_process:
                            all_features = all_features.cpu().numpy()
                            mean = np.mean(all_features, axis=0)
                            cov = np.cov(all_features, rowvar=False)
    
                            fid_score = compute_fid(mu_real=self.mean, sigma_real=self.cov, \
                                                    mu_gen=mean , sigma_gen=cov)


                            self.logger.add_scalars({"FID" : fid_score}, global_step=self.step)
                            if fid_score < best_fid_score:
                                print(f"Saving best at step {self.step} - {fid_score} < {best_fid_score}\n")
                                self.save("best_fid")
                                best_fid_score = fid_score
                            else:
                                print(f"Skipping at {self.step} - {best_fid_score} < {fid_score}\n")


                    metricator = Metricator(accelerator.device)
                    with torch.inference_mode():
                        all_psnr = []
                        all_ssim = []
                        all_lpips = []

                        for data in tqdm(self.recons_dataloader, disable=not accelerator.is_main_process):
                            with self.accelerator.autocast():
                                input_images    = data["input_images"]
                                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                    "focal_lengths": data["input_focal_lengths"], \
                                                    "principal_points": data["input_principal_points"]}

                                if self.use_rays:
                                    input_rays      = data["input_rays"]
                                    input_images    = torch.cat([input_images, input_rays], dim=2)

                                target_images   = data["target_images"]
                                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                    "focal_lengths": data["target_focal_lengths"], \
                                                    "principal_points": data["target_principal_points"]}

                                pred_images         = self.recons_ema.ema_model.inference(input_images, input_cameras, target_cameras)
                                pred_images         = pred_images.flatten(0, 1)       
                                target_images       = target_images.flatten(0, 1)       
                                psnr, ssim, lpips   = metricator.compute_metrics(pred_images, target_images)

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

                            self.logger.add_scalars({"PSNR" : all_psnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM" : all_ssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS" : all_lpips}, global_step=self.step)

                            if best_psnr < all_psnr:
                                print(f"Saving best at step {self.step} - {best_psnr} < {all_psnr}\n")
                                best_psnr = all_psnr
                                self.save("best_psnr")
                            else:
                                print(f"Skipping at {self.step} - {best_psnr} > {all_psnr}\n")
                            
                            render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                "focal_lengths": data["render_focal_lengths"], \
                                                "principal_points": data["render_principal_points"]}

                            input_images    = data["input_images"]
                            input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                "focal_lengths": data["input_focal_lengths"], \
                                                "principal_points": data["input_principal_points"]}

                            if self.use_rays:
                                input_rays      = data["input_rays"]
                                input_images    = torch.cat([input_images, input_rays], dim=2)

                            target_images   = data["target_images"]
                            target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                "focal_lengths": data["target_focal_lengths"], \
                                                "principal_points": data["target_principal_points"]}

                            _, misc        = self.recons_ema(input_images, input_cameras, target_images, target_cameras)
                            misc["render_cameras"]  = render_cameras       # for video rendering
                            misc["input_cameras"]   = input_cameras
                            self.logger_summary_recons(misc, "test")

                    metricator = None

                self.step += 1

                self.recons_lr_scheduler.step()
                self.diffusion_lr_scheduler.step()
                pbar.update(1)

        accelerator.print("training complete")

                 
    def logger_summary_recons(self, misc, prefix="train"):
        print("logger summary reconstruction")
        model = self.recons_ema.ema_model

        with torch.inference_mode():
            input_images    = misc["input"]
            render_cameras  = misc["render_cameras"]
            input_cameras   = misc["input_cameras"]
            render_images   = model.inference(input_images, input_cameras, render_cameras).cpu().detach() * 0.5 + 0.5

        if self.use_rays:
            input_images = input_images[:, :, :3].cpu().detach()
        else:
            input_images = input_images.cpu().detach()
        
        target_images     = misc["target"].cpu().detach()
        novel_view_images = misc["novel_view_rgbs"].cpu().detach()
        novel_view_depths = misc["novel_view_depths"].squeeze(2).cpu().detach()
      
        recons_images = misc["recons_rgbs"].cpu().detach()
        recons_depths = misc["recons_depths"].squeeze(2).cpu().detach()

        bs              = input_images.size(0)
        num_cond        = input_images.size(1)
        num_render      = target_images.size(1)

        input_images        = rearrange(input_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        target_images       = rearrange(target_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        novel_view_images   = rearrange(novel_view_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5 
        novel_view_depths   = rearrange(novel_view_depths, "b t h w -> (b t) h w") 
        recons_images       = rearrange(recons_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        recons_depths       = rearrange(recons_depths, "b t h w -> (b t) h w")

        novel_view_depths   = torch.from_numpy(jet_depth(novel_view_depths)).permute(0, 3, 1, 2)
        recons_depths       = torch.from_numpy(jet_depth(recons_depths)).permute(0, 3, 1, 2)

        target_images   = torch.cat([target_images, novel_view_depths, \
                                                    novel_view_images], dim=-1)

        input_images   = torch.cat([input_images, recons_depths, \
                                                    recons_images], dim=-1)

        render_videos   = make_grid_4d(render_images, nrow=num_render)
        input_images    = make_grid(input_images, nrow=num_cond, padding=4, pad_value=0.5)
        target_images   = make_grid(target_images, nrow=num_render, padding=4)
        
        image_dict = {
            f"{prefix}_visualization/input": input_images,
            f"{prefix}_visualization/target": target_images,
        }
        saved_images = torch.cat([input_images, target_images], dim=-1)
        saved_images = np.clip(saved_images.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{prefix}_image_step-{self.step}.png"))

        videos = []
        for frame in render_videos:
            frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
            frame = Image.fromarray(frame.astype(np.uint8))
            videos.append(frame)

        videos[0].save(fp=os.path.join(self.image_dir, f"{prefix}_recons_video_step-{self.step}.gif"),
                        format='png',
                        append_images=videos[1:],
                        save_all=True,
                        duration=100,
                        loop=0)
                        
        self.logger.add_images(image_dict, self.step)
        self.logger.add_videos(f"{prefix}_visualization/render_videos", render_videos.unsqueeze(0), self.step)
        self.visualize_features(misc["features"], "train", "input")
        
    
    def logger_summary_diffusion(self, misc):
        print("logger summary diffusion")
        diffusion_model = self.diffusion_ema.ema_model
        recons_model    = self.recons_ema.ema_model

        with torch.inference_mode():
            input_images    = misc["input"].cpu().detach()
            if self.use_rays:
                input_images = input_images[:, :, :3]

            target_cameras  = misc["target_cameras"]

            cleaned_latent  = misc["clean"]
            noisy_latent    = misc["noise"]
            denoised_latent = misc["denoise"]
            torch.cuda.empty_cache()
       
            cleaned_pred_images, cleaned_pred_depths, cleaned_triplane_features = recons_model.decode(cleaned_latent, target_cameras)
            cleaned_video_rgbs, cleaned_video_depths = self.visualize(cleaned_pred_images, \
                                                                cleaned_pred_depths, cleaned_triplane_features, postfix="clean")
            
            noisy_pred_images, noisy_pred_depths, noisy_triplane_features = recons_model.decode(noisy_latent, target_cameras)
            noisy_video_rgbs, noisy_video_depths = self.visualize(noisy_pred_images, \
                                                                noisy_pred_depths, noisy_triplane_features, postfix="noisy")
          
            denoised_pred_images, denoised_pred_depths, denoised_triplane_features = recons_model.decode(denoised_latent, target_cameras)
            denoised_video_rgbs, denoised_video_depths = self.visualize(denoised_pred_images, \
                                                                denoised_pred_depths, denoised_triplane_features, postfix="denoised")

            sample_latent        = diffusion_model.ddim_sample(cleaned_latent.shape, misc["volume"], classifier_free_guidance_w=2.0)
            sample_pred_images, sample_pred_depths, sample_triplane_features = recons_model.decode(sample_latent, target_cameras)
            sample_video_rgbs, sample_video_depths = self.visualize(sample_pred_images, \
                                                                sample_pred_depths, sample_triplane_features, postfix="sample")
       
            bs              = input_images.size(0)
            num_cond        = input_images.size(1)
            num_render      = cleaned_video_rgbs.size(1)

            target_videos   = torch.cat([cleaned_video_rgbs, noisy_video_rgbs, \
                                         denoised_video_rgbs, sample_video_rgbs], dim=-1)

            input_images    = rearrange(input_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
            input_images    = make_grid(input_images, nrow=num_cond)

            target_videos   = make_grid_4d(target_videos, nrow=num_render)
            self.logger.add_videos("visualization/target_video", target_videos.unsqueeze(0), self.step)

            input_videos = input_images.repeat(num_render, 1, 1, 1)
            saved_videos = torch.cat([input_videos, target_videos], dim=-1)
            videos = []
            for frame in saved_videos:
                frame = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255.0
                frame = Image.fromarray(frame.astype(np.uint8))
                videos.append(frame)

            videos[0].save(fp=os.path.join(self.image_dir, f"diffusion_video_step-{self.step}.gif"),
                            format='png',
                            append_images=videos[1:],
                            save_all=True,
                            duration=100,
                            loop=0)


    def visualize(self, rgbs, depths, features, prefix="train", postfix="clean"):
        video_rgbs  = rgbs.cpu() * 0.5 + 0.5
        depths      = depths.squeeze(2).cpu()
        features    = features.cpu()
        self.visualize_features(features, prefix, postfix)

        num_render      = rgbs.size(1)
        depths          = rearrange(depths, "b t h w -> (b t) h w")
        colorized_depth = torch.from_numpy(jet_depth(depths)).permute(0, 3, 1, 2)
        video_depths    = rearrange(colorized_depth, "(b t) c h w -> b t c h w", t=num_render)
        torch.cuda.empty_cache()

        return video_rgbs, video_depths


    def visualize_features(self, features, prefix="train", postfix="clean"):
        features    = features.cpu().detach()
        if features.size(1) == 3:
            # should be triplane
            hw_features      = features[:, 0]
            dw_features      = features[:, 1]
            dh_features      = features[:, 2]
        else:
            hw_features = rearrange(features, "bs c d h w -> bs (c d) h w")
            dw_features = rearrange(features, "bs c d h w -> bs (c h) d w")
            dh_features = rearrange(features, "bs c d h w -> bs (c w) d h")

        hw_images = feature_map_pca(hw_features).unsqueeze(1)
        dw_images = feature_map_pca(dw_features).unsqueeze(1)
        dh_images = feature_map_pca(dh_features).unsqueeze(1)

        features_image = torch.cat([hw_images, dw_images, dh_images], dim=1)
        features_image = rearrange(features_image, "bs n c h w -> (bs n) c h w")
        features_image   = make_grid(features_image, nrow=3)
        pca_image_dict = {
            f"{prefix}_visualization/{postfix}_feature": features_image,
        }

        saved_images = np.clip(features_image.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{prefix}_{postfix}_feature_step-{self.step}.png"))

        self.logger.add_images(pca_image_dict, self.step)