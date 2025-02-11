import torch
import numpy as np
from PIL import Image
import os
from einops import rearrange
from torchvision.utils import make_grid
import sys
sys.path.append("/lustre/scratch/client/vinai/users/khatpn/tung/code/NVS")
from utils import jet_depth, jet_depth_scale, feature_map_pca

image_dir = "/lustre/scratch/client/vinai/users/khatpn/tung/code/NVS/visualize/images"
os.makedirs(image_dir, exist_ok=True)

reconstructor_path = "/lustre/scratch/client/vinai/users/khatpn/tung/code/NVS/reconstructor_visualize_data.pt"
diffusion_path = "/lustre/scratch/client/vinai/users/khatpn/tung/code/NVS/diffusion_visualize_data.pt"
feature_path = "/lustre/scratch/client/vinai/users/khatpn/tung/code/NVS/intermediate_data.pt"

feature_visualize_data = torch.load(feature_path)
hw_data = feature_visualize_data["hw"]
dw_data = feature_visualize_data["dw"]
dh_data = feature_visualize_data["dh"]

for level, features in enumerate(hw_data):
    features = feature_map_pca(features.detach())
    for batch_idx, image in enumerate(features):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_feature-hw-{level}.png"))

for level, features in enumerate(dw_data):
    features = feature_map_pca(features.detach())
    for batch_idx, image in enumerate(features):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_feature-dw-{level}.png"))

for level, features in enumerate(dh_data):
    features = feature_map_pca(features.detach())
    for batch_idx, image in enumerate(features):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_feature-dh-{level}.png"))


reconstructor_visualize_data = torch.load(reconstructor_path, map_location='cpu')
diffusion_visualize_data = torch.load(diffusion_path, map_location='cpu')

x_start = diffusion_visualize_data["x_start"].detach()
x_noisy = diffusion_visualize_data["x_noisy"].detach()
model_out = diffusion_visualize_data["model_out"].detach()

x_start = feature_map_pca(x_start)
x_start = torch.nn.functional.interpolate(x_start, scale_factor=4.0, mode='bilinear')
for batch_idx, image in enumerate(x_start):
    image = image.permute(1,2,0).cpu().numpy()
    image = image * 255.0
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(image_dir, f"{batch_idx}_x-start.png"))

x_noisy = feature_map_pca(x_noisy)
x_noisy = torch.nn.functional.interpolate(x_noisy, scale_factor=4.0, mode='bilinear')
for batch_idx, image in enumerate(x_noisy):
    image = image.permute(1,2,0).cpu().numpy()
    image = image * 255.0
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(image_dir, f"{batch_idx}_x-noisy.png"))

model_out = feature_map_pca(model_out)
model_out = torch.nn.functional.interpolate(model_out, scale_factor=4.0, mode='bilinear')
for batch_idx, image in enumerate(model_out):
    image = image.permute(1,2,0).cpu().numpy()
    image = image * 255.0
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(image_dir, f"{batch_idx}_model_out.png"))

cond_images = reconstructor_visualize_data["cond_images"].detach()
cond_cameras = reconstructor_visualize_data["cond_cameras"]
volume_features = reconstructor_visualize_data["volume_features"].detach()
pred_rendered_images = reconstructor_visualize_data["pred_rendered_images"].detach()
rendered_features = reconstructor_visualize_data["rendered_features"].detach()
rendered_depth = reconstructor_visualize_data["rendered_depth"].detach()
pred_images = reconstructor_visualize_data["pred_images"].detach()
pred_depths = reconstructor_visualize_data["pred_depths"].detach()
latent = reconstructor_visualize_data["latent"].detach()


def visualize_features(features):
    features    = features.cpu().detach()
    if features.size(1) == 3:
        hw_features      = features[:, 0]
        dw_features      = features[:, 1]
        dh_features      = features[:, 2]
    else:
        hw_features = rearrange(features, "bs c d h w -> bs (c d) h w")
        dw_features = rearrange(features, "bs c d h w -> bs (c h) d w")
        dh_features = rearrange(features, "bs c d h w -> bs (c w) d h")

    hw_images = feature_map_pca(hw_features)
    dw_images = feature_map_pca(dw_features)
    dh_images = feature_map_pca(dh_features)

    features_image = torch.cat([hw_images, dw_images, dh_images], dim=-1)
    return features_image

volume_features_pca = visualize_features(volume_features)
volume_features_pca = torch.nn.functional.interpolate(volume_features_pca, scale_factor=8.0, mode='bilinear')

for batch_idx, image in enumerate(volume_features_pca):
    image = image.permute(1,2,0).cpu().numpy()
    image = image * 255.0
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(image_dir, f"{batch_idx}_volume-feature.png"))

latent = rearrange(latent, "bs (n c) h w -> bs n c h w", n=3)
latent = visualize_features(latent)
for batch_idx, image in enumerate(latent):
    image = image.permute(1,2,0).cpu().numpy()
    image = image * 255.0
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(image_dir, f"{batch_idx}_latent.png"))

for batch_idx, images in enumerate(rendered_depth):
    images = torch.nn.functional.interpolate(images, scale_factor=4.0, mode='bilinear')
    images   = torch.from_numpy(jet_depth_scale(images.squeeze(0).cpu(), max=2.5)).permute(0, 3, 1, 2)
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_rendered-depths.png"))

for batch_idx, images in enumerate(pred_rendered_images):
    images = torch.nn.functional.interpolate(images, scale_factor=4.0, mode='bilinear')
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy() * 0.5 + 0.5
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_rendered-images.png"))

for batch_idx, images in enumerate(rendered_features):
    images = feature_map_pca(images)
    images = torch.nn.functional.interpolate(images, scale_factor=4.0, mode='bilinear')
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_rendered-features.png"))


for batch_idx, images in enumerate(pred_depths):
    images   = torch.from_numpy(jet_depth_scale(images.squeeze(0).cpu(), max=2.0)).permute(0, 3, 1, 2)
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy()
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_pred-depths.png"))

for batch_idx, images in enumerate(cond_images):
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy() * 0.5 + 0.5
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_cond-images.png"))


for batch_idx, images in enumerate(pred_images):
    for idx, image in enumerate(images):
        image = image.permute(1,2,0).cpu().numpy() * 0.5 + 0.5
        image = image * 255.0
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(image_dir, f"{batch_idx}_{idx}_pred-images.png"))
        
