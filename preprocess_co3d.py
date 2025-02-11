import sys
import os
import hydra
from omegaconf import DictConfig

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
    print('Loading data manager')
    dataset = get_data_manager(cfg, cfg.data.category[0], split=cfg.dataset_name)
    print('Finish loading data manager')

if __name__ == "__main__":
    evaluate()