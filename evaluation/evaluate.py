"""
Compute metrics on rendered images (after eval.py).
First computes per-object metric then reduces them. If --multicat is used
then also summarized per-categority metrics. Use --reduce_only to skip the
per-object computation step.

Note eval.py already outputs PSNR/SSIM.
This also computes LPIPS and is useful for double-checking metric is correct.
"""

import cv2
import os
import os.path as osp
import argparse
import skimage.measure
from tqdm import tqdm
import warnings
import lpips as lpips_lib
import numpy as np
import torch
import imageio.v3 as iio
import json
from loss_utils import ssim as ssim_fn
import imageio.v3 as iio


class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target): # image in range [0, 1]
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips

parser = argparse.ArgumentParser(description="Calculate PSNR for rendered images.")
parser.add_argument(
    "--datadir",
    "-D",
    type=str,
    default="/home/group/chairs_test",
    help="Dataset directory; note: different from usual, directly use the thing please",
)
parser.add_argument(
    "--output",
    "-O",
    type=str,
    default="eval",
    help="Root path of rendered output (our format, from eval.py)",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU id. Only single GPU supported for this script.",
)


args = parser.parse_args()

cuda = "cuda:" + str(args.gpu_id)
metricator = Metricator(device=cuda)
data_root = args.datadir
render_root = args.output

assert len(os.listdir(render_root)) == len(os.listdir(data_root))

all_cpsnr = []
all_cssim = []
all_clpips = []

all_spsnr = []
all_sssim = []
all_slpips = []

all_dpsnr = []
all_dssim = []
all_dlpips = []


for pred_name in tqdm(os.listdir(render_root)):
    pred_folder = os.path.join(render_root, pred_name)
    gt_folder = os.path.join(data_root, pred_name)

    gt_image_path = os.path.join(gt_folder, "rgb", "000001.png")
    cond_image_path = os.path.join(pred_folder, "cond_000001.png")
    sample_image_path = os.path.join(pred_folder, "sample_000001.png")
    diffusion_image_path = os.path.join(pred_folder, "diffusion_000001.png")
    assert os.path.isfile(cond_image_path)
    assert os.path.isfile(sample_image_path)
    assert os.path.isfile(diffusion_image_path)
    assert os.path.isfile(gt_image_path)

    cond_image = iio.imread(cond_image_path).astype(np.float32)[..., :3] / 255.0
    sample_image = iio.imread(sample_image_path).astype(np.float32) / 255.0
    diffusion_image = iio.imread(diffusion_image_path).astype(np.float32) / 255.0
    gt_image = iio.imread(gt_image_path).astype(np.float32) / 255.0
    
    cond_image = torch.from_numpy(cond_image).permute(2, 0, 1).to(cuda)
    sample_image = torch.from_numpy(sample_image).permute(2, 0, 1).to(cuda)
    diffusion_image = torch.from_numpy(diffusion_image).permute(2, 0, 1).to(cuda)
    gt_image = torch.from_numpy(gt_image).permute(2, 0, 1).to(cuda)
    
    cpsnr, cssim, clpips = metricator.compute_metrics(cond_image, gt_image)
    spsnr, sssim, slpips = metricator.compute_metrics(sample_image, gt_image)
    dpsnr, dssim, dlpips = metricator.compute_metrics(diffusion_image, gt_image)

    all_cpsnr.append(cpsnr)
    all_cssim.append(cssim)
    all_clpips.append(clpips)
   
    all_spsnr.append(spsnr)
    all_sssim.append(sssim)
    all_slpips.append(slpips)
  
    all_dpsnr.append(dpsnr)
    all_dssim.append(dssim)
    all_dlpips.append(dlpips)
    
    error_cond_image = ((cond_image - gt_image)**2).mean(0, keepdim=True).repeat(3, 1, 1)
    error_sample_image = ((sample_image - gt_image)**2).mean(0, keepdim=True).repeat(3, 1, 1)
    error_diffusion_image = ((diffusion_image - gt_image)**2).mean(0, keepdim=True).repeat(3, 1, 1)

    top_row = torch.cat([cond_image, sample_image, diffusion_image, gt_image], dim=-1)
    bot_row = torch.cat([error_cond_image, error_sample_image, error_diffusion_image, torch.zeros_like(gt_image)], dim=-1)

    saved_image = torch.cat([top_row, bot_row], dim=-2).permute(1, 2, 0).cpu().numpy() * 255.0
    saved_image = np.uint8(saved_image)
    error_image_path = os.path.join(pred_folder, "error.png")
    iio.imwrite(error_image_path, saved_image)
    
    result_str = f"cond psnr: {cpsnr} cond ssim: {cssim} cond lpips: {clpips} \n"
    result_str += f"sample psnr: {spsnr} sample ssim: {sssim} sample lpips: {slpips} \n"
    result_str += f"diffusion psnr: {dpsnr} diffusion ssim: {dssim} diffusion lpips: {dlpips} "

    rfile = open(os.path.join(pred_folder, "result.txt"), "+w")
    rfile.write(result_str)
    rfile.close()

total_cpsnr = sum(all_cpsnr)/len(all_cpsnr)
total_cssim = sum(all_cssim)/len(all_cssim)
total_clpips = sum(all_clpips)/len(all_clpips)

total_spsnr = sum(all_spsnr)/len(all_spsnr)
total_sssim = sum(all_sssim)/len(all_sssim)
total_slpips = sum(all_slpips)/len(all_slpips)

total_dpsnr = sum(all_dpsnr)/len(all_dpsnr)
total_dssim = sum(all_dssim)/len(all_dssim)
total_dlpips = sum(all_dlpips)/len(all_dlpips)

print(f"Cond PSNR={total_cpsnr}  - Cond SSIM={total_cssim}  - Cond LPIPS={total_clpips}")
print(f"Sample PSNR={total_spsnr}  - Sample SSIM={total_sssim}  - Sample LPIPS={total_slpips}")
print(f"Diffusion PSNR={total_dpsnr}  - Diffusion SSIM={total_dssim}  - Diffusion LPIPS={total_dlpips}")

rfile = open(os.path.join(render_root, "all_result.txt"), "+w")
result_str = f"cond psnr: {total_cpsnr} cond ssim: {total_cssim} cond lpips: {total_clpips} \n"
result_str += f"sample psnr: {total_spsnr} sample ssim: {total_sssim} sample lpips: {total_slpips} \n"
result_str += f"diffusion psnr: {total_dpsnr} diffusion ssim: {total_dssim} diffusion lpips: {total_dlpips} "
rfile.write(result_str)
rfile.close()
