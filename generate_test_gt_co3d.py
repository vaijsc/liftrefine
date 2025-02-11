import sys
import os
import hydra
from omegaconf import DictConfig

import lpips 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import imageio
import numpy as np
from numpy import random
from PIL import Image 
from einops import rearrange


from data_manager import get_data_manager
from model.reconstructor import Reconstructor
from model.wrapper import ModelWrapper
from model.evaluator import Evaluator

from accelerate import Accelerator
from accelerate.utils import set_seed



@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def evaluate(cfg: DictConfig):
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", 
    )
    set_seed(cfg.random_seed)

    # dataset
    test_batch_size = cfg.batch_size 
    print('Loading data manager')
    dataset = get_data_manager(cfg, cfg.data.category[0], split="test")
    print('Finish loading data manager')

    dl = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    if cfg.data.white_background:
        save_dir = os.path.join(dataset.data_path, f"test_object_white_bg")
    else:
        save_dir = os.path.join(dataset.data_path, f"test_object_black_bg")

    for data in dl:
        target_images = data["target_images"] * 0.5 + 0.5
        object_names = data["object_names"]

        for batch_idx, (name, images) in enumerate(zip(object_names, target_images)):
            object_path = os.path.join(save_dir, str(name.item()), "rgb")
            os.makedirs(object_path, exist_ok=True)
            for image in images:
                saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img         = Image.fromarray(saved_image)
                path        = os.path.join(object_path, "{:06}.png".format(1))
                img.save(path)


if __name__ == "__main__":
    evaluate()