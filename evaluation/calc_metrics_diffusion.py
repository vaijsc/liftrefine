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

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
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
    "--dataset_format",
    "-F",
    type=str,
    default="dvr",
    help="Dataset format, nerf | srn | dvr",
)
parser.add_argument(
    "--list_name", type=str, default="softras_test", help="Filter list prefix for DVR",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU id. Only single GPU supported for this script.",
)
parser.add_argument(
    "--overwrite", action="store_true", help="overwriting existing metrics.txt",
)
parser.add_argument(
    "--exclude_dtu_bad", action="store_true", help="exclude hardcoded DTU bad views",
)
parser.add_argument(
    "--multicat",
    action="store_true",
    help="Prepend category id to object id. Specify if model fits multiple categories.",
)

parser.add_argument(
    "--viewlist",
    "-L",
    type=str,
    default="",
    help="Path to source view list e.g. src_dvr.txt; if specified, excludes the source view from evaluation",
)
parser.add_argument(
    "--eval_view_list", type=str, default=None, help="Path to eval view list"
)
parser.add_argument(
    "--primary", "-P", type=str, default="", help="List of views to exclude"
)
parser.add_argument(
    "--lpips_batch_size", type=int, default=32, help="Batch size for LPIPS",
)

parser.add_argument(
    "--reduce_only",
    "-R",
    action="store_true",
    help="skip the map (per-obj metric computation)",
)
parser.add_argument(
    "--metadata",
    type=str,
    default="metadata.yaml",
    help="Path to dataset metadata under datadir, used for getting category names if --multicat",
)
parser.add_argument(
    "--dtu_sort", action="store_true", help="Sort using DTU scene order instead of lex"
)
args = parser.parse_args()


if args.dataset_format == "dvr":
    list_name = args.list_name + ".lst"
    img_dir_name = "image"
elif args.dataset_format == "srn":
    list_name = ""
    img_dir_name = "rgb"
elif args.dataset_format == "nerf":
    warnings.warn("test split not implemented for NeRF synthetic data format")
    list_name = ""
    img_dir_name = ""
else:
    raise NotImplementedError("Not supported data format " + args.dataset_format)


data_root = args.datadir
render_root = args.output


def run_map():
    if args.multicat:
        cats = os.listdir(data_root)

        def fmt_obj_name(c, x):
            return c + "_" + x

    else:
        cats = ["."]

        def fmt_obj_name(c, x):
            return x

    use_exclude_lut = len(args.viewlist) > 0
    if use_exclude_lut:
        print("Excluding views from list", args.viewlist)
        with open(args.viewlist, "r") as f:
            tmp = [x.strip().split() for x in f.readlines()]
        exclude_lut = {
            x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
            for x in tmp
        }
    base_exclude_views = list(map(int, args.primary.split()))
    if args.exclude_dtu_bad:
        base_exclude_views.extend(
            [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        )

    if args.eval_view_list is not None:
        with open(args.eval_view_list, "r") as f:
            eval_views = list(map(int, f.readline().split()))
            print("Only using views", eval_views)
    else:
        eval_views = None

    all_objs = []

    print("CATEGORICAL SUMMARY")
    total_objs = 0

    for cat in cats:
        cat_root = osp.join(data_root, cat)
        if not osp.isdir(cat_root):
            continue

        objs = sorted([x for x in os.listdir(cat_root)])

        if len(list_name) > 0:
            list_path = osp.join(cat_root, list_name)
            with open(list_path, "r") as f:
                split = set([x.strip() for x in f.readlines()])
            objs = [x for x in objs if x in split]

        objs_rend = [osp.join(render_root, fmt_obj_name(cat, x)) for x in objs]

        objs = [osp.join(cat_root, x) for x in objs]
        objs = [x for x in objs if osp.isdir(x)]

        objs = list(zip(objs, objs_rend))
        objs_avail = [x for x in objs if osp.exists(x[1])]
        print(cat, "TOTAL", len(objs), "AVAILABLE", len(objs_avail))
        #  assert len(objs) == len(objs_avail)
        total_objs += len(objs)
        all_objs.extend(objs_avail)
    print(">>> USING", len(all_objs), "OF", total_objs, "OBJECTS")

    cuda = "cuda:" + str(args.gpu_id)
    metricator = Metricator(device=cuda)

    def get_metrics(rgb, gt):
        ssim = skimage.metrics.structural_similarity(rgb, gt, channel_axis=2,data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(rgb, gt, data_range=1)
        return psnr, ssim

    def isimage(path):
        ext = osp.splitext(path)[1]
        return ext == ".jpg" or ext == ".png"

    def process_obj(path, rend_path, metricator, prefix="cond"):
        if len(img_dir_name) > 0:
            im_root = osp.join(path, img_dir_name)
        else:
            im_root = path
        out_path = osp.join(rend_path, f"{prefix}_metrics.txt")
        if osp.exists(out_path) and not args.overwrite:
            return
        ims = [x for x in sorted(os.listdir(im_root)) if isimage(x)]
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        gts = []
        preds = []
        num_ims = 0
        if use_exclude_lut:
            lut_key = osp.basename(rend_path).replace("_", "/")
            exclude_views = exclude_lut[lut_key]
        else:
            exclude_views = []
        exclude_views.extend(base_exclude_views)

        for im_name in ims:
            im_path = osp.join(im_root, im_name)
            im_name_id = int(osp.splitext(im_name)[0])
            im_name_out = "{}_{:06}.png".format(prefix, im_name_id)
            im_rend_path = osp.join(rend_path, im_name_out)
            if osp.exists(im_rend_path) and im_name_id not in exclude_views:
                if eval_views is not None and im_name_id not in eval_views:
                    continue
                gt = iio.imread(im_path).astype(np.float32)[..., :3] / 255.0
                pred = iio.imread(im_rend_path).astype(np.float32) / 255.0

                if gt.shape != pred.shape:
                    W, H = pred.shape[0:2]
                    gt = cv2.resize(gt, (W, H))
                    
                gt = torch.from_numpy(gt).permute(2, 0, 1).to(cuda)
                pred = torch.from_numpy(pred).permute(2, 0, 1).to(cuda)
                psnr, ssim, lpips = metricator.compute_metrics(pred, gt)
                psnr_total  += psnr
                ssim_total  += ssim
                lpips_total += lpips
                num_ims += 1

        psnr_avg = psnr_total/num_ims
        ssim_avg = ssim_total/num_ims
        lpips_avg = lpips_total/num_ims
        out_txt = "psnr {}\nssim {}\nlpips {}".format(psnr_avg, ssim_avg, lpips_avg)
        with open(out_path, "w") as f:
            f.write(out_txt)

        return psnr_total, ssim_total, lpips_total, num_ims
        
    all_sample_psnr = 0.0
    all_sample_ssim = 0.0
    all_sample_lpips = 0.0
    num_sample_ims = 0.0
    
    all_cond_psnr = 0.0
    all_cond_ssim = 0.0
    all_cond_lpips = 0.0
    num_cond_ims = 0.0
    for obj_path, obj_rend_path in tqdm(all_objs):
        cond_psnr, cond_ssim, cond_lpips, num_cond_im = process_obj(obj_path, obj_rend_path, metricator, "cond")
        sample_psnr, sample_ssim, sample_lpips, num_sample_im = process_obj(obj_path, obj_rend_path, metricator, "sample")

        all_sample_psnr += sample_psnr
        all_sample_ssim += sample_ssim
        all_sample_lpips += sample_lpips
        num_sample_ims += num_sample_im
    
        all_cond_psnr += cond_psnr
        all_cond_ssim += cond_ssim
        all_cond_lpips += cond_lpips
        num_cond_ims += num_cond_im

    print(f"Cond PSNR: {all_cond_psnr/num_cond_ims}")
    print(f"Cond SSIM: {all_cond_ssim/num_cond_ims}")
    print(f"Cond LPIPS: {all_cond_lpips/num_cond_ims}")
 
    print(f"Sample PSNR: {all_sample_psnr/num_sample_ims}")
    print(f"Sample SSIM: {all_sample_ssim/num_sample_ims}")
    print(f"Sample LPIPS: {all_sample_lpips/num_sample_ims}")

def run_reduce(prefix="cond"):
    if args.multicat:
        meta = json.load(open(osp.join(args.datadir, args.metadata), "r"))
        cats = sorted(list(meta.keys()))
        cat_description = {cat: meta[cat]["name"].split(",")[0] for cat in cats}

    all_objs = []
    objs = [x for x in os.listdir(render_root)]
    objs = [osp.join(render_root, x) for x in objs if x[0] != "_"]
    objs = [x for x in objs if osp.isdir(x)]
    if args.dtu_sort:
        objs = sorted(objs, key=lambda x: int(x[x.rindex("/") + 5 :]))
    else:
        objs = sorted(objs)
    all_objs.extend(objs)

    print(">>> PROCESSING", len(all_objs), "OBJECTS")

    METRIC_NAMES = ["psnr", "ssim", "lpips"]

    out_metrics_path = osp.join(render_root, f"{prefix}_all_metrics.txt")

    if args.multicat:
        cat_sz = {}
        for cat in cats:
            cat_sz[cat] = 0

    all_metrics = {}
    for name in METRIC_NAMES:
        if args.multicat:
            for cat in cats:
                all_metrics[cat + "." + name] = 0.0
        all_metrics[name] = 0.0

    should_print_all_objs = len(all_objs) < 100

    for obj_root in tqdm(all_objs):
        metrics_path = osp.join(obj_root, f"{prefix}_metrics.txt")
        with open(metrics_path, "r") as f:
            metrics = [line.split() for line in f.readlines()]
        if args.multicat:
            cat_name = osp.basename(obj_root).split("_")[0]
            cat_sz[cat_name] += 1
            for metric, val in metrics:
                all_metrics[cat_name + "." + metric] += float(val)

        for metric, val in metrics:
            all_metrics[metric] += float(val)
        if should_print_all_objs:
            print(obj_root, end=" ")
            for metric, val in metrics:
                print(val, end=" ")
            print()

    for name in METRIC_NAMES:
        if args.multicat:
            for cat in cats:
                if cat_sz[cat] > 0:
                    all_metrics[cat + "." + name] /= cat_sz[cat]
        all_metrics[name] /= len(all_objs)
        print(name, all_metrics[name])

    metrics_txt = []
    if args.multicat:
        for cat in cats:
            if cat_sz[cat] > 0:
                cat_txt = "{:12s}".format(cat_description[cat])
                for name in METRIC_NAMES:
                    cat_txt += " {}: {:.6f}".format(name, all_metrics[cat + "." + name])
                cat_txt += " n_inst: {}".format(cat_sz[cat])
                metrics_txt.append(cat_txt)

        total_txt = "---\n{:12s}".format("total")
    else:
        total_txt = ""
    for name in METRIC_NAMES:
        total_txt += " {}: {:.6f}".format(name, all_metrics[name])
    metrics_txt.append(total_txt)

    metrics_txt = "\n".join(metrics_txt)
    with open(out_metrics_path, "w") as f:
        f.write(metrics_txt)
    print("WROTE", out_metrics_path)
    print(metrics_txt)


if __name__ == "__main__":
    # if not args.reduce_only:
    #     print(">>> Compute")
    run_map()
    print(">>> Reduce")
    run_reduce("cond")
    run_reduce("sample")