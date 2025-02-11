import torch
from model.ssdnerf.modules import MultiHeadAttentionMod, DenoisingResBlockMod, \
                                    DenoisingDownsampleMod, DenoisingUpsampleMod
from model.ssdnerf.denoising import DenoisingUnetMod
from mmcv.cnn.bricks.conv_module import ConvModule

layer = MultiHeadAttentionMod(in_channels=96, num_heads=6, groups=1)

input_rand = torch.rand(4, 96, 128, 128)
emb_rand = torch.rand(4, 64)
time_rand = torch.rand(4)

down_layer = DenoisingDownsampleMod(in_channels=96, groups=1)
up_layer = DenoisingUpsampleMod(in_channels=96, groups=1)
resnet_layer = DenoisingResBlockMod(
                                    in_channels=96,
                                    embedding_channels=64,
                                    use_scale_shift_norm=True,
                                    dropout=0.1,
                                    groups=1,
                                    out_channels=32,
                                    shortcut_kernel_size=3)

print(down_layer(input_rand).shape)
print(up_layer(input_rand).shape)
print(resnet_layer(input_rand, emb_rand).shape)


unet_model = DenoisingUnetMod(image_size=128,  # size of triplanes (not images)
                                in_channels=96,
                                base_channels=128,
                                channels_cfg=[1, 2, 2, 4, 4], # 128 64 32 16 8 
                                resblocks_per_downsample=2,
                                dropout=0.0,
                                use_scale_shift_norm=True,
                                downsample_conv=True,
                                upsample_conv=True,
                                num_heads=4,
                                attention_res=[32, 16, 8])

print(unet_model(input_rand, time_rand).shape)