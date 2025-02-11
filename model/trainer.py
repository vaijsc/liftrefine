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

from utils import exists, cycle, to_gpu, jet_depth, feature_map_pca, count_parameters, make_grid_4d,\
                    get_cosine_schedule_with_warmup, get_constant_hyperparameter_schedule_with_warmup, split_view
from model.metric import Metricator

# trainer class
class Trainer(object):
    def __init__(
        self,
        reconstruction_model,
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
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp
        self.latent_gain    = optimization_cfg.latent_gain
        self.lpips_weight   = optimization_cfg.lpips_weight
        self.recons_weight  = optimization_cfg.recons_weight
        self.novel_weight   = optimization_cfg.novel_weight


        self.model = reconstruction_model
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
        self.val_dataloader = val_dataloader

        # optimizer
        params = [p for n, p in self.model.named_parameters() if "perceptual" not in n]
        perceptual_params = [
            n for n, p in self.model.named_parameters() if "perceptual" in n
        ]
        if self.accelerator.is_main_process:
            print(f"perceptual params: {perceptual_params}")
        self.opt = Adam(params, lr=train_lr, betas=adam_betas)
        lr_scheduler = get_cosine_schedule_with_warmup(
            self.opt, warmup_period, train_num_steps
        )

        if self.accelerator.is_main_process:
            num_params = count_parameters(self.model)
            print(f"Total number of trainable parameters: {num_params // 1e6}M")


        # if self.accelerator.is_main_process:
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every, include_online_model=False)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.opt, lr_scheduler
        )

        if checkpoint_path is not None:
            self.load(checkpoint_path, is_resume)
            # self.load_from_david_checkpoint(checkpoint_path)
            # self.load_from_external_checkpoint(checkpoint_path)

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
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
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

        model = self.accelerator.unwrap_model(self.model)
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        model.load_state_dict(data["model"], strict=is_resume)

        if is_resume:
            self.step = data["step"]
            if "lr_scheduler" in data:
                self.lr_scheduler.load_state_dict(data["lr_scheduler"])

            if self.accelerator.is_main_process:
                self.ema.load_state_dict(data["ema"], strict=False)

            if "version" in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data["scaler"]):
                self.accelerator.scaler.load_state_dict(data["scaler"])
    

    def get_losses(self, losses, prefix='train'):
        loss_dict = {}

        novel_rgb_loss    = losses["novel_rgb_loss"]
        novel_lpips_loss  = losses["novel_lpips_loss"]
        recons_rgb_loss   = losses["recons_rgb_loss"]
        recons_lpips_loss = losses["recons_lpips_loss"]
       
        novel_loss = novel_rgb_loss + novel_lpips_loss * self.lpips_weight
        recons_loss = recons_rgb_loss + recons_lpips_loss * self.lpips_weight

        loss = recons_loss * self.recons_weight + novel_loss * self.novel_weight
        
        loss /= self.gradient_accumulate_every
        loss_dict[f"{prefix}/total_loss"]  = loss.item()
        loss_dict[f"{prefix}/novel_loss"]  = novel_loss.item()
        loss_dict[f"{prefix}/recons_loss"] = recons_loss.item()

        return loss, loss_dict


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        best_psnr = 0.0

        # torch.autograd.set_detect_anomaly(True)
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss_dict                      = {}
                total_loss_dict["train/novel_loss"]  = 0.0
                total_loss_dict["train/recons_loss"] = 0.0
                total_loss_dict["train/total_loss"]  = 0.0
                
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dataloader)  # .to(device)
                    # data = to_gpu(data, device)

                    with self.accelerator.autocast():
                        input_images    = data["input_images"]     
                        input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                            "focal_lengths": data["input_focal_lengths"], \
                                            "principal_points": data["input_principal_points"]}
                    
                        target_images   = data["target_images"]
                        target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                            "focal_lengths": data["target_focal_lengths"], \
                                            "principal_points": data["target_principal_points"]}

                        losses, misc = self.model(input_images, input_cameras, \
                                                    target_images, target_cameras)
                        loss, loss_dict = self.get_losses(losses, "train")
                        
                        for key in loss_dict:
                            total_loss_dict[key] += loss_dict[key]

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                total_loss = total_loss_dict["train/total_loss"]
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process and self.step % self.logging_every == 0:
                    total_loss_dict["lr"] = self.lr_scheduler.get_last_lr()[0]
                    self.logger.add_scalars(total_loss_dict, global_step=self.step)


                self.ema.update()
                all_images = None
                all_videos_list = None
                if accelerator.is_main_process:
                    if self.step % self.summary_every == 0:
                        with torch.inference_mode():
                            _, misc = self.ema(input_images, input_cameras, \
                                                        target_images, target_cameras)
                            
                        render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                            "focal_lengths": data["render_focal_lengths"], \
                                            "principal_points": data["render_principal_points"]}

                        misc["render_cameras"]  = render_cameras       # for video rendering
                        misc["input_cameras"]   = input_cameras
                        self.logger_summary(misc, prefix="train")                                           

                    if self.step != 0 and self.step % self.save_every == 0:
                        self.save("last")
                    
                if self.step % self.eval_every == 0 and self.step >= self.eval_every * 1:
                    model = self.ema.ema_model
                    metricator = Metricator(accelerator.device)
                    with torch.inference_mode():
                        all_psnr = []
                        all_ssim = []
                        all_lpips = []
             
                        all_a_psnr = []
                        all_a_ssim = []
                        all_a_lpips = []

                        for data in tqdm(self.val_dataloader, disable=not accelerator.is_main_process):
                            with self.accelerator.autocast():
                                input_images    = data["input_images"]
                                input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                    "focal_lengths": data["input_focal_lengths"], \
                                                    "principal_points": data["input_principal_points"]}

                                target_images   = data["target_images"]
                                target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                    "focal_lengths": data["target_focal_lengths"], \
                                                    "principal_points": data["target_principal_points"]}

                                first_images, first_cameras, last_images, last_cameras = split_view(input_images, input_cameras)
                                _, first_volumes = model.encode(first_images, first_cameras)
                                _, last_volumes = model.encode(last_images, last_cameras)

                                aggregate_volumes = torch.cat([first_volumes, last_volumes], dim=1)
                                pred_aggregate_images = model.decode_volumes(aggregate_volumes, target_cameras)

                                pred_images             = model.inference(input_images, input_cameras, target_cameras)
                                pred_images             = pred_images.flatten(0, 1)       
                                pred_aggregate_images   = pred_aggregate_images.flatten(0, 1)       
                                target_images           = target_images.flatten(0, 1)       
                                psnr, ssim, lpips       = metricator.compute_metrics(pred_images, target_images)
                                a_psnr, a_ssim, a_lpips = metricator.compute_metrics(pred_aggregate_images, target_images)
                    
                                psnr                = accelerator.gather(psnr).cpu()
                                ssim                = accelerator.gather(ssim).cpu()
                                lpips               = accelerator.gather(lpips).cpu()
                              
                                a_psnr                = accelerator.gather(a_psnr).cpu()
                                a_ssim                = accelerator.gather(a_ssim).cpu()
                                a_lpips               = accelerator.gather(a_lpips).cpu()

                                if accelerator.is_main_process:
                                    all_psnr.append(psnr)
                                    all_ssim.append(ssim)
                                    all_lpips.append(lpips)
                                 
                                    all_a_psnr.append(a_psnr)
                                    all_a_ssim.append(a_ssim)
                                    all_a_lpips.append(a_lpips)


                        if accelerator.is_main_process:
                            all_psnr = torch.cat(all_psnr, dim=0).mean().item()
                            all_ssim = torch.cat(all_ssim, dim=0).mean().item()
                            all_lpips = torch.cat(all_lpips, dim=0).mean().item()
                        
                            all_a_psnr = torch.cat(all_a_psnr, dim=0).mean().item()
                            all_a_ssim = torch.cat(all_a_ssim, dim=0).mean().item()
                            all_a_lpips = torch.cat(all_a_lpips, dim=0).mean().item()

                            self.logger.add_scalars({"PSNR" : all_psnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM" : all_ssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS" : all_lpips}, global_step=self.step)

                            self.logger.add_scalars({"PSNR/aggregate" : all_a_psnr}, global_step=self.step)
                            self.logger.add_scalars({"SSIM/aggregate" : all_a_ssim}, global_step=self.step)
                            self.logger.add_scalars({"LPIPS/aggregate" : all_a_lpips}, global_step=self.step)

                            if best_psnr < all_psnr:
                                print(f"Saving best at step {self.step} - {best_psnr} < {all_psnr}\n")
                                best_psnr = all_psnr
                                self.save("best")
                            else:
                                print(f"Skipping at {self.step} - {best_psnr} > {all_psnr}\n")
                            
                            render_cameras  = {"R" : data["render_camera_Rs"], "T" : data["render_camera_Ts"], \
                                                "focal_lengths": data["render_focal_lengths"], \
                                                "principal_points": data["render_principal_points"]}

                            input_images    = data["input_images"]
                            input_cameras   = {"R" : data["input_camera_Rs"], "T" : data["input_camera_Ts"],
                                                "focal_lengths": data["input_focal_lengths"], \
                                                "principal_points": data["input_principal_points"]}

                            target_images   = data["target_images"]
                            target_cameras  = {"R" : data["target_camera_Rs"], "T" : data["target_camera_Ts"], \
                                                "focal_lengths": data["target_focal_lengths"], \
                                                "principal_points": data["target_principal_points"]}

                            _, misc                 = self.ema(input_images, input_cameras, \
                                                                target_images, target_cameras)
                            misc["render_cameras"]  = render_cameras       # for video rendering
                            misc["input_cameras"]   = input_cameras
                            self.logger_summary(misc, "test")
                    
                self.step += 1
                self.lr_scheduler.step()
                pbar.update(1)

        accelerator.print("training complete")

    def logger_summary(self, misc, prefix="train"):
        print("logger summary")
        model = self.ema.ema_model
        self.visualize(model, misc, prefix)

    def visualize(self, model, misc, preprefix="train"):
        with torch.inference_mode():
            render_cameras  = misc["render_cameras"]
            render_images   = model.decode(misc["latent"], render_cameras).cpu().detach() * 0.5 + 0.5
        
        input_images = misc["input"].cpu().detach()
        recons_gt_images = misc["recons"].cpu().detach()
        target_images = misc["target"].cpu().detach()
        
        novel_view_images = misc["novel_rgbs"].cpu().detach()
        novel_view_depths = misc["novel_depths"].squeeze(2).cpu().detach()
      
        recons_images = misc["recons_rgbs"].cpu().detach()
        recons_depths = misc["recons_depths"].squeeze(2).cpu().detach()

        bs              = input_images.size(0)
        num_cond        = input_images.size(1)
        num_render      = target_images.size(1)
        num_recons      = recons_gt_images.size(1)

        input_images        = rearrange(input_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        recons_gt_images    = rearrange(recons_gt_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        target_images       = rearrange(target_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        novel_view_images   = rearrange(novel_view_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5 
        novel_view_depths   = rearrange(novel_view_depths, "b t h w -> (b t) h w") 
        recons_images       = rearrange(recons_images, "b t c h w -> (b t) c h w") * 0.5 + 0.5
        recons_depths       = rearrange(recons_depths, "b t h w -> (b t) h w")

        novel_view_depths   = torch.from_numpy(jet_depth(novel_view_depths)).permute(0, 3, 1, 2)
        recons_depths       = torch.from_numpy(jet_depth(recons_depths)).permute(0, 3, 1, 2)

        target_images   = torch.cat([target_images, novel_view_depths, \
                                                    novel_view_images], dim=-1)

        recons_images   = torch.cat([recons_gt_images, recons_depths, \
                                                    recons_images], dim=-1)


        render_videos   = make_grid_4d(render_images, nrow=num_render)
        input_images    = make_grid(input_images, nrow=num_cond, padding=0)
        target_images   = make_grid(target_images, nrow=num_render, padding=0)
        recons_images   = make_grid(recons_images, nrow=num_recons, padding=0)

        input_images    = torch.nn.functional.pad(input_images, (4, 4, 4, 4), mode='constant', value=0.3)
        recons_images   = torch.nn.functional.pad(recons_images, (4, 4, 4, 4), mode='constant', value=0.5)
        target_images   = torch.nn.functional.pad(target_images, (4, 4, 4, 4), mode='constant', value=0.8)
        
        image_dict = {
            f"input/{preprefix}_visualization": input_images,
            f"recons/{preprefix}_visualization": recons_images,
            f"target/{preprefix}_visualization": target_images,
        }
        saved_images = torch.cat([input_images, recons_images, target_images], dim=-1)
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
        self.logger.add_videos(f"render_videos/{preprefix}_visualization", render_videos.unsqueeze(0), self.step)
        
        features    = misc["novel_features"].cpu().detach()
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
        features_image = make_grid(features_image, nrow=3)
        pca_image_dict = {
            f"feature/{preprefix}_visualization": features_image,
        }

        saved_images = np.clip(features_image.permute(1, 2, 0).numpy(), 0, 1) * 255.0
        img = Image.fromarray(saved_images.astype(np.uint8))
        img.save(os.path.join(self.image_dir, f"{preprefix}_feature_step-{self.step}.png"))

        self.logger.add_images(pca_image_dict, self.step)