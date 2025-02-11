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
from tqdm import tqdm

from data_manager import get_data_manager
from model.reconstructor import Reconstructor
from model.wrapper import ModelWrapper
from model.evaluator import Evaluator

from accelerate import Accelerator
from accelerate.utils import set_seed

import pickle


@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def evaluate(cfg: DictConfig):
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", 
    )
    set_seed(cfg.random_seed)

    # dataset
    test_batch_size = cfg.batch_size 
    print('Loading data manager')
    dataset = get_data_manager(cfg, split="test")
    print('Finish loading data manager')

    dl = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    inception_path = "../../pretrained/metric/inception-2015-12-05.pt"
    inception_network = torch.jit.load(inception_path).eval().cuda()
    all_features = []

    with torch.inference_mode():
        for data in tqdm(dl):
            images = torch.cat([data["input_images"], data["target_images"]], dim=1)
            images = images.reshape(-1, *images.shape[2:])
            images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).cuda()
            list_images = torch.split(images, 64, dim=0)
            for images in list_images:
                images = images.cuda()
                features = inception_network(images, return_features=True).cpu()
                all_features.append(features)
    
    all_features = torch.cat(all_features, dim=0)
    all_features = np.array(all_features)

    with open(dataset.stat_path, 'wb') as f:
        pickle.dump(
            {
                'mean': np.mean(all_features, axis=0),
                'cov': np.cov(all_features, rowvar=False),
            }, f)

if __name__ == "__main__":
    evaluate()