import functools
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
from utils import plucker_embedding

from .unet_parts import ResnetBlock2D


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class ResnetFeatureExtractor2D(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        cfg,
        backbone="resnet34",
        pretrained=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param pretrained Whether to use model weights pretrained on ImageNet
        """
        super().__init__()
        self.cfg = cfg
        dim = cfg.model.unet.input_dim

        if norm_type != "batch":
            assert not pretrained

        norm_layer = get_norm_layer(norm_type)

        print("Using torchvision", backbone, "encoder")
        if backbone == "resnet34":
            pretrained_weight = torchvision.models.ResNet34_Weights.DEFAULT
            input_outconv_dim = 1024
        elif backbone == "resnet50":
            pretrained_weight = torchvision.models.ResNet50_Weights.DEFAULT
            input_outconv_dim = 3904
        elif backbone == "resnet101":
            pretrained_weight = torchvision.models.ResNet101_Weights.DEFAULT
            input_outconv_dim = 3904
        elif backbone == "resnet152":
            pretrained_weight = torchvision.models.ResNet152_Weights.DEFAULT
            input_outconv_dim = 2048
        else:
            raise Exception("Not implemented")
        
        if not pretrained:
            pretrained_weight = None
        
        self.model = getattr(torchvision.models, backbone)(
            weights=pretrained_weight, norm_layer=norm_layer
        )

        del self.model.fc
        del self.model.avgpool

        self.out_conv = ResnetBlock2D(input_outconv_dim, dim)

    def forward(self, viewset2d):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        B, Cond, C, H, W = viewset2d.shape
        viewset2d = viewset2d.reshape(B*Cond, C, H, W)

        x = self.model.conv1(viewset2d)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        latents.append(x)

        feat_size = x.shape[-2:]
        x = self.model.layer2(x)
        latents.append(x)

        x = self.model.layer3(x)
        latents.append(x)

        x = self.model.layer4(x)
        latents.append(x)

        for idx, feat in enumerate(latents):
            latents[idx] = torch.nn.functional.interpolate(feat, feat_size, mode='bilinear', antialias=True)

        feature_maps = torch.cat(latents, dim=1)

        feature_maps = self.out_conv(feature_maps)
        feature_maps = feature_maps.reshape(B, Cond, *feature_maps.shape[1:])

        return feature_maps
    



