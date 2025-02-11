from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, reduce, extract
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import identity

from omegaconf import OmegaConf
import torch.nn.functional as F
import torch
import numpy as np

from collections import namedtuple
from functools import partial
from tqdm import tqdm
from utils import split_camera, drop_view
from einops import rearrange
from utility.load_model import load_vae


# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class Diffuser(GaussianDiffusion):
    def __init__(
        self,
        model=None,
        image_size=None,
        cfg=None,
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__(model,
                        image_size=image_size,
                        timesteps = cfg.diffuser.timesteps,
                        sampling_timesteps = cfg.diffuser.sampling_timesteps,
                        objective = cfg.diffuser.objective,
                        beta_schedule = cfg.diffuser.beta_schedule,
                        ddim_sampling_eta = ddim_sampling_eta,
                        offset_noise_strength = offset_noise_strength,
                        min_snr_loss_weight = min_snr_loss_weight, # https://arxiv.org/abs/2303.09556
                        min_snr_gamma = min_snr_gamma)
        self.cond_drop_prob = cfg.diffuser.cond_drop_prob
        vae_ckpt = "/home/ubuntu/kt/khatran/code/sparsefusion/checkpoints/sd/sd-v1-3-vae.ckpt"
        self.vae = load_vae(vae_ckpt, verbose=True)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.z_scale_factor = 0.18215


    def get_cond(self, input_images, input_cameras):
        return self.model.get_cond(input_images, input_cameras)


    def p_losses(self, x_start, t, cond, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        misc = {}
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        misc["x_noise"] = x
        # predict and take gradient step
        model_out = self.model(x, t, cond, self.cond_drop_prob)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
            misc["pred_x0"] = model_out

        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(), misc

    @property
    def rgb_loss_fn(self):
        return F.mse_loss


    def forward(self, input_images, input_cameras, target_images, target_cameras, return_misc=False):
        reconstructor_loss, misc    = self.model.forward_reconstructor(input_images, input_cameras, target_images, target_cameras)
        target_features = misc["feat"].flatten(0, 1)

        with torch.no_grad():
            target_images = target_images.flatten(0, 1)
            target_images = torch.nn.functional.interpolate(target_images, scale_factor=2.0)
            latent        = self.vae.encode(target_images).mode() * self.z_scale_factor

<<<<<<< HEAD
        ret = img
        return self.model.inv_act(ret)
    
    
    @torch.inference_mode()
    def inference(self, cameras, disable=False):
        size  = [cameras["R"].size(0)] + self.image_size
        sample_latent   = self.ddim_sample(size, disable)
        
        all_sample_pred_images = []
        for camera in split_camera(cameras, 1):
            sample_pred_images, _, _ = self.model.decode(sample_latent, camera)
            all_sample_pred_images.append(sample_pred_images.cpu())

        all_sample_pred_images  = torch.cat(all_sample_pred_images, dim=1)
        output                  = {}
        output["rgb"]           = all_sample_pred_images.reshape(-1, *all_sample_pred_images.shape[2:])
        output["size"]          = output["rgb"].size(0)

        return output


class SingleStageDiffuser(GaussianDiffusion):
    def __init__(
        self,
        model=None,
        image_size=None,
        cfg=None,
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__(model,
                        image_size=image_size,
                        timesteps = cfg.diffuser.timesteps,
                        sampling_timesteps = cfg.diffuser.sampling_timesteps,
                        objective = cfg.diffuser.objective,
                        beta_schedule = cfg.diffuser.beta_schedule,
                        schedule_fn_kwargs = schedule_fn_kwargs,
                        ddim_sampling_eta = ddim_sampling_eta,
                        auto_normalize = auto_normalize,
                        offset_noise_strength = offset_noise_strength,
                        min_snr_loss_weight = min_snr_loss_weight, # https://arxiv.org/abs/2303.09556
                        min_snr_gamma = min_snr_gamma)
        self.ucg_rate = cfg.diffuser.ucg_rate
        self.norm_factor = torch.ones(1, dtype=torch.float)
        self.momentum = 0.001

    def forward(self, latent):
        latent = self.model.act(latent)
        norm_factor = latent.detach().square().mean()
        
        self.norm_factor = self.norm_factor.to(norm_factor.device)
        self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
                                + self.momentum * norm_factor

        assert latent.size(1) == self.image_size[0] \
                and latent.size(2) == self.image_size[1] \
                and latent.size(3) == self.image_size[2], f'height and width of image must be {self.image_size}'
=======
        assert latent.size(1) == self.channels \
                and latent.size(2) == self.image_size \
                and latent.size(3) == self.image_size, f'height and width of image must be {self.image_size}'
>>>>>>> dev_aggregate_zero123

        bs, c, h, w  = latent.shape
        device  = latent.device       
        t = torch.randint(0, self.num_timesteps, (bs,), device=device).long()
        diffusion_loss, loss_misc = self.p_losses(latent, t, target_features)

<<<<<<< HEAD
        misc                    = {}
        misc["norm_factor"]     = 1/self.norm_factor
        misc["clean"]           = self.model.inv_act(latent)
        misc["noise"]           = self.model.inv_act(model_noise)
        misc["denoise"]         = self.model.inv_act(model_out)
=======
        if return_misc:
            with torch.no_grad():
                if "pred_x0" in loss_misc.keys():
                    misc["denoise"] = self.vae.decode(1.0 / self.z_scale_factor * loss_misc["pred_x0"])
>>>>>>> dev_aggregate_zero123

                misc["clean"]   = self.vae.decode(1.0 / self.z_scale_factor * latent)
                misc["noise"]   = self.vae.decode(1.0 / self.z_scale_factor * loss_misc["x_noise"])
                misc["cond"]    = target_features

        return reconstructor_loss, diffusion_loss, misc


    @torch.inference_mode()
    def decode_latent(self, latent):
        return self.vae.decode(1.0 / self.z_scale_factor * latent)

    @torch.inference_mode()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        return img


    @torch.inference_mode()
    def ddim_sample(self, cond, shape, cond_scale = 2., rescaled_phi = 0.7, clip_denoised = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img


    @torch.inference_mode()
    def sample(self, cond, cond_scale = 6., rescaled_phi = 0.7):
        batch_size, image_size, channels = cond.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        output = sample_fn(cond, (batch_size, channels, image_size, image_size), cond_scale, rescaled_phi)
        return output
    
    
    @torch.inference_mode()
    def decode(self, latent, cameras):
        pred_images = []
        for camera in split_camera(cameras, 1):
            pred_image, _, _ = self.model.decode(latent, camera)
            pred_images.append(pred_image)

        pred_images  = torch.cat(pred_images, dim=1)
        return pred_images


    @torch.inference_mode()
    def inference(self, cond_images, cond_cameras, cameras, cond_scale = 6., rescaled_phi = 0.7):
        cond            = self.get_cond(cond_images, cond_cameras)
        sample_latent   = self.sample(cond, cond_scale, rescaled_phi)
        aggregate_latent   = torch.cat([cond.unsqueeze(1), sample_latent.unsqueeze(1)], dim=1)

        all_sample_pred_images      = self.decode(sample_latent, cameras)
        all_aggregate_pred_images   = self.decode(aggregate_latent, cameras)
        output                  = {}
        output["sample_rgb"]    = all_sample_pred_images
        output["aggregate_rgb"] = all_aggregate_pred_images
        output["cond"]          = cond
        output["size"]          = output["sample_rgb"].size(0) * output["sample_rgb"].size(1)

        return output

    @torch.inference_mode()
    def get_sample(self, cond_images, cond_cameras, cond_scale = 6., rescaled_phi = 0.7):
        cond_latent     = self.get_cond(cond_images, cond_cameras)
        sample_latent   = self.sample(cond_latent, cond_scale, rescaled_phi)

<<<<<<< HEAD
        return output


class SingleStageDiffuserCond(GaussianDiffusion):
    def __init__(
        self,
        model=None,
        image_size=None,
        cfg=None,
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        cond_size=16,
    ):
        super().__init__(model,
                        image_size=image_size,
                        timesteps = cfg.diffuser.timesteps,
                        sampling_timesteps = cfg.diffuser.sampling_timesteps,
                        objective = cfg.diffuser.objective,
                        beta_schedule = cfg.diffuser.beta_schedule,
                        schedule_fn_kwargs = schedule_fn_kwargs,
                        ddim_sampling_eta = ddim_sampling_eta,
                        auto_normalize = auto_normalize,
                        offset_noise_strength = offset_noise_strength,
                        min_snr_loss_weight = min_snr_loss_weight, # https://arxiv.org/abs/2303.09556
                        min_snr_gamma = min_snr_gamma)
        self.ucg_rate = cfg.diffuser.ucg_rate
        self.norm_factor = torch.ones(1, dtype=torch.float)
        self.momentum = 0.001
        self.cond_size = cond_size

    def forward(self, latent, cond):
        latent = self.model.act(latent)
        norm_factor = latent.detach().square().mean()
        
        self.norm_factor = self.norm_factor.to(norm_factor.device)
        self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
                                + self.momentum * norm_factor

        assert      cond.size(-1) == self.cond_size \
                and cond.size(-2) == self.cond_size\
                and cond.size(-3) == self.cond_size, f'height and width of cond must be {self.cond_size}'

        assert latent.size(1) == self.image_size[0] \
                and latent.size(2) == self.image_size[1] \
                and latent.size(3) == self.image_size[2], f'height and width of image must be {self.image_size}'

        bs, c, h, w  = latent.shape
        device  = latent.device       
        t = torch.randint(0, self.num_timesteps, (bs,), device=device).long()
        loss, model_out, model_noise = self.p_losses(latent, t, cond)
        loss = loss / self.norm_factor

        misc                    = {}
        misc["norm_factor"]     = 1/self.norm_factor
        misc["clean"]           = self.model.inv_act(latent)
        misc["noise"]           = self.model.inv_act(model_noise)
        misc["denoise"]         = self.model.inv_act(model_out)

        return loss, misc


    def p_losses(self, x_start, t, cond, noise = None):   
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step

        # drop condition with cond_drop_prob = 0.5 
        drop_idx = torch.bernoulli(0.5 * torch.ones(cond.shape[0], device=cond.device)).type(cond.dtype)
        drop_idx = drop_idx.reshape(-1, 1, 1, 1, 1)
        cond     = drop_idx * cond

        model_out = self.model(x, t, cond)

        if self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(), model_out, x


    def model_predictions(self, x, t, cond):
        model_output = self.model(x, t, cond)
        if self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.inference_mode()
    def ddim_sample(self, shape, cond, classifier_free_guidance_w=2.0, disable=False):
        batch, device,                      = shape[0], self.device
        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        eta, objective                      = self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start     = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=disable):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond)
            uncond_pred_noise, uncond_x_start, *_ = self.model_predictions(img, time_cond, torch.zeros_like(cond))
            
            overall_pred_noise = (1+classifier_free_guidance_w) * pred_noise - \
                                         classifier_free_guidance_w * uncond_pred_noise
            x_start = self.predict_start_from_noise(img, time_cond, overall_pred_noise)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img
        return self.model.inv_act(ret)
    
    
    @torch.inference_mode()
    def inference(self, cameras, cond, classifier_free_guidance_w=2.0, disable=False):
        size  = [cameras["R"].size(0)] + self.image_size
        sample_latent   = self.ddim_sample(size, cond, cfg, disable)
        
        all_sample_pred_images = []
        for camera in split_camera(cameras, 1):
            sample_pred_images, _, _ = self.model.decode(sample_latent, camera)
            all_sample_pred_images.append(sample_pred_images.cpu())

        all_sample_pred_images  = torch.cat(all_sample_pred_images, dim=1)
        output                  = {}
        output["rgb"]           = all_sample_pred_images.reshape(-1, *all_sample_pred_images.shape[2:])
        output["size"]          = output["rgb"].size(0)

        return output
=======
        return sample_latent, cond_latent
>>>>>>> dev_aggregate_zero123
