import torch
from torch import nn
import torch.nn.functional as F
import lpips
from utils import split_camera, drop_view, split_view, concat_camera
from model.loss.tv_loss import TVLoss
import numpy as np
from einops import rearrange

class ModelWrapper(nn.Module):
    def __init__(
        self, model, cfg=None):
        super().__init__()
        self.model = model
        self.loss_type = cfg.optimization.loss
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()

        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

    @property
    def rgb_loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


    def decode(self, latent, target_cameras):
        list_target_cameras = split_camera(target_cameras, 1)
        pred_images = []
        pred_depths = []

        for cameras in list_target_cameras:
            images, depths, features = self.model.decode(latent, cameras)
            pred_images.append(images.cpu())
            pred_depths.append(depths.cpu())

        pred_images = torch.cat(pred_images, dim=1)
        pred_depths = torch.cat(pred_depths, dim=1)
        return pred_images, pred_depths, features.cpu()
   
   
    def encode(self, input_images, input_cameras):
        latent, volume_features = self.model.encode(input_images, input_cameras)
        return latent, volume_features


    def encode(self, input_images, input_cameras):
        return self.model.encode(input_images, input_cameras)


    def forward(self, input_images, input_cameras, target_images, target_cameras, rendered_images, rendered_cameras):
        cond_images, cond_cameras = drop_view(input_images, input_cameras)

        latent, volume_features = self.encode(cond_images, cond_cameras)
        novel_loss, novel_misc = self.calc_losses(latent, target_images, target_cameras, prefix="novel")

        all_losses  = {}
        all_misc    = {}
        all_misc["input"]   = cond_images
        all_misc["target"]  = target_images
        all_misc["latent"]  = latent

        pred_rendered_images, rendered_features, rendered_depth = self.model.render_volumes(volume_features, rendered_cameras)

        nv = rendered_images.size(1)
        rendered_images = torch.nn.functional.interpolate(rendered_images.flatten(0, 1), scale_factor=0.125, mode="bilinear")
        rendered_images = rearrange(rendered_images, "(bs nv) c h w -> bs nv c h w", nv=nv)
        rendered_loss_dict = self.loss_fnc(pred_rendered_images, rendered_images, "rendered")

        all_misc[f"rendered_rgbs"]      = pred_rendered_images                                     
        all_misc[f"gt_rendered_rgbs"]   = rendered_images                                     
        all_misc[f"rendered_depths"]    = rendered_depth                                     
        all_misc[f"rendered_features"]  = rendered_features
            
        all_losses.update(novel_loss)
        all_losses.update(rendered_loss_dict)
        all_misc.update(novel_misc)

        return all_losses, all_misc

    def loss_fnc(self, pred_images, gt_images, prefix="novel_view"):
        losses = {}
        losses[f"{prefix}_rgb_loss"]            = self.rgb_loss_fn(pred_images, gt_images)
        losses[f"{prefix}_lpips_loss"]          = self.perceptual_loss(pred_images.reshape(-1, *pred_images.shape[2:]), \
                                                             gt_images.reshape(-1, *gt_images.shape[2:])).mean()

        return losses

    
    def calc_losses(self, pred_latent, gt_images, gt_cameras, prefix="novel_view"):
        losses, misc = {}, {}
        pred_images, pred_depth, features  = self.model.decode(pred_latent, gt_cameras)

        loss_dict = self.loss_fnc(pred_images, gt_images, prefix)
        losses.update(loss_dict)
     
        misc[f"{prefix}_rgbs"]      = pred_images                                     
        misc[f"{prefix}_depths"]    = pred_depth                                     
        misc[f"{prefix}_features"]  = features                                     
        
        return losses, misc

    def decode(self, pred_latent, cameras):
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images
    
    
    def decode_volumes(self, pred_volues, cameras):
        pred_latent = self.model.split_planes(pred_volues)
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images


    def inference(self, input_images, input_cameras, target_cameras):
        list_target_cameras = split_camera(target_cameras, 4)
        pred_images = []

        for cameras in list_target_cameras:
            images, _, _ = self.model(input_images, input_cameras, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images


class SDSWrapper(nn.Module):
    def __init__(
        self, model, cfg=None):
        super().__init__()
        self.model = model
        self.loss_type = cfg.optimization.loss
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()

        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.reconstructor.plane_ups.parameters():
            param.requires_grad = True
     
        for param in self.model.reconstructor.final_plane_res_block.parameters():
            param.requires_grad = True

    def decode(self, latent, target_cameras):
        list_target_cameras = split_camera(target_cameras, 1)
        pred_images = []
        pred_depths = []

        for cameras in list_target_cameras:
            images, depths, features = self.model.decode(latent, cameras)
            pred_images.append(images.cpu())
            pred_depths.append(depths.cpu())

        pred_images = torch.cat(pred_images, dim=1)
        pred_depths = torch.cat(pred_depths, dim=1)
        return pred_images, pred_depths, features.cpu()


    def encode(self, input_images, input_cameras):
        return self.model.encode(input_images, input_cameras)


    def forward(self, input_images, input_cameras, target_images, target_cameras):
        cond_images, cond_cameras = drop_view(input_images, input_cameras)

        latent, volume_features = self.encode(cond_images, cond_cameras)
        pred_images, pred_depth, features  = self.model.decode(latent, target_cameras)
        rendered_features = self.model.render_volumes(volume_features, target_cameras)[1]

        all_misc    = {}
        all_misc["input"]       = cond_images
        all_misc["target"]      = target_images
        all_misc["latent"]      = latent
        all_misc["features"]    = features
        all_misc["feat"]        = rendered_features.detach()
        all_misc["pred"]        = pred_images
        all_misc["depth"]       = pred_depth

        return all_misc

    
    def calc_losses(self, pred_latent, gt_images, gt_cameras, prefix="novel_view"):
        losses, misc = {}, {}
        pred_images, pred_depth, features  = self.model.decode(pred_latent, gt_cameras)

        losses[f"{prefix}_rgb_loss"]            = self.rgb_loss_fn(pred_images, gt_images)
        losses[f"{prefix}_lpips_loss"]          = self.perceptual_loss(pred_images.reshape(-1, *pred_images.shape[2:]), \
                                                             gt_images.reshape(-1, *gt_images.shape[2:])).mean()

        misc[f"{prefix}_rgbs"]      = pred_images                                     
        misc[f"{prefix}_depths"]    = pred_depth                                     
        misc[f"{prefix}_features"]  = features                                     
        
        return losses, misc

    def decode(self, pred_latent, cameras):
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images
    
    
    def decode_volumes(self, pred_volues, cameras):
        pred_latent = self.model.split_planes(pred_volues)
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images


    def inference(self, input_images, input_cameras, target_cameras):
        list_target_cameras = split_camera(target_cameras, 4)
        pred_images = []

        for cameras in list_target_cameras:
            images, _, _ = self.model(input_images, input_cameras, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images


class DeterministicWrapper(nn.Module):
    def __init__(
        self, model, cfg=None):
        super().__init__()
        self.model = model
        self.loss_type = cfg.optimization.loss
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()

        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

    @property
    def rgb_loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


    def decode(self, latent, target_cameras):
        list_target_cameras = split_camera(target_cameras, 1)
        pred_images = []
        pred_depths = []

        for cameras in list_target_cameras:
            images, depths, features = self.model.decode(latent, cameras)
            pred_images.append(images.cpu())
            pred_depths.append(depths.cpu())

        pred_images = torch.cat(pred_images, dim=1)
        pred_depths = torch.cat(pred_depths, dim=1)
        return pred_images, pred_depths, features.cpu()


    def encode(self, input_images, input_cameras):
        return self.model.encode(input_images, input_cameras)


    def forward(self, input_images, input_cameras, target_images, target_cameras, rendered_images, rendered_cameras):
        cond_images, cond_cameras = drop_view(input_images, input_cameras)

        latent, volume_features = self.encode(cond_images, cond_cameras)
        novel_loss, novel_misc = self.calc_losses(latent, target_images, target_cameras, prefix="novel")

        all_losses  = {}
        all_misc    = {}
        all_misc["input"]   = cond_images
        all_misc["target"]  = target_images
        all_misc["latent"]  = latent

        if rendered_images is not None:
            pred_rendered_images, rendered_depth = self.model.render_volumes(volume_features, rendered_cameras)
            nv = rendered_images.size(1)
            rendered_images = torch.nn.functional.interpolate(rendered_images.flatten(0, 1), \
                                            size=pred_rendered_images.shape[-2:], mode="bilinear")
            rendered_images = rearrange(rendered_images, "(bs nv) c h w -> bs nv c h w", nv=nv)
            rendered_loss_dict = self.loss_fnc(pred_rendered_images, rendered_images, "rendered")
            all_misc[f"rendered_rgbs"]      = pred_rendered_images                                     
            all_misc[f"gt_rendered_rgbs"]   = rendered_images                                     
            all_misc[f"rendered_depths"]    = rendered_depth                                     
            all_losses.update(rendered_loss_dict)

        all_losses.update(novel_loss)
        all_misc.update(novel_misc)

        return all_losses, all_misc

    def loss_fnc(self, pred_images, gt_images, prefix="novel_view"):
        losses = {}
        losses[f"{prefix}_rgb_loss"]            = self.rgb_loss_fn(pred_images, gt_images)
        losses[f"{prefix}_lpips_loss"]          = self.perceptual_loss(pred_images.reshape(-1, *pred_images.shape[2:]), \
                                                             gt_images.reshape(-1, *gt_images.shape[2:])).mean()

        return losses

    
    def calc_losses(self, pred_latent, gt_images, gt_cameras, prefix="novel_view"):
        losses, misc = {}, {}
        pred_images, pred_depth, features  = self.model.decode(pred_latent, gt_cameras)

        loss_dict = self.loss_fnc(pred_images, gt_images, prefix)
        losses.update(loss_dict)
     
        misc[f"{prefix}_rgbs"]      = pred_images                                     
        misc[f"{prefix}_depths"]    = pred_depth                                     
        misc[f"{prefix}_features"]  = features                                     
        
        return losses, misc

    def decode(self, pred_latent, cameras):
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images
    
    
    def decode_volumes(self, pred_volues, cameras):
        pred_latent = self.model.split_planes(pred_volues)
        list_cameras = split_camera(cameras, 1)
        pred_images = []

        for cameras in list_cameras:
            images, _, _ = self.model.decode(pred_latent, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images


    def inference(self, input_images, input_cameras, target_cameras):
        list_target_cameras = split_camera(target_cameras, 4)
        pred_images = []

        for cameras in list_target_cameras:
            images, _, _ = self.model(input_images, input_cameras, cameras)
            pred_images.append(images)

        pred_images = torch.cat(pred_images, dim=1)
        return pred_images
