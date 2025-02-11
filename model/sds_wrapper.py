import torch
from torch import nn
import torch.nn.functional as F


from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


class SDSWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
        self.teacher = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")


        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer", use_fast=False)
        # import correct text encoder classes
        text_encoder_cls = import_model_class_from_model_name_or_path("stabilityai/stable-diffusion-2-1-base")
        self.text_encoder = text_encoder_cls.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.teacher.requires_grad_(False)

        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod


    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.text_encoder.device))[0]

        return embeddings

    def forward(self, image):
        image = torch.nn.functional.interpolate(image, scale_factor=4)
        bs = image.size(0)
        pred_original_samples = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor

        # Sample noise that we'll add to the predicted original samples
        noise = torch.randn_like(pred_original_samples)

        # Sample a random timestep for each image
        timesteps_range = torch.tensor([0.02, 0.981]) * self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(*timesteps_range.long(), (bs,), device=pred_original_samples.device).long()
        noisy_samples = self.noise_scheduler.add_noise(pred_original_samples, noise, timesteps)

        prompt_embeds = self.get_text_embeds("a teddybear").repeat(bs, 1, 1)
        prompt_null_embeds = self.get_text_embeds("").repeat(bs, 1, 1)

        with torch.no_grad():
            pred_cond = self.teacher(noisy_samples, timesteps, prompt_embeds).sample
            pred_uncond = self.teacher(noisy_samples, timesteps, prompt_null_embeds).sample

            # Apply classifier-free guidance to the teacher prediction
            teacher_pred = pred_uncond + 4 * (pred_cond - pred_uncond)

        self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sigma_t = ((1 - self.alphas_cumprod[timesteps])**0.5).view(-1, 1, 1, 1)
        score_gradient = torch.nan_to_num(sigma_t**2 * (teacher_pred - noise))

        target = (pred_original_samples - score_gradient).detach()
        loss_sds = 0.5 * F.mse_loss(pred_original_samples.float(), target.float(), reduction="mean")

        return loss_sds


       
        
      