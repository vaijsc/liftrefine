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
    train_dataset = get_data_manager(cfg, cfg.data.category[0], split="train")
    dataset = get_data_manager(cfg, cfg.data.category[0], split="test")
    print('Finish loading data manager')

    dl = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    reconstructor   = Reconstructor(cfg, num_latent=len(train_dataset)).cuda()
    modelwrapper    = ModelWrapper(reconstructor, cfg).cuda()

    evaluator = Evaluator(
                        reconstruction_model=modelwrapper,
                        accelerator=accelerator,
                        dataloader=dl,
                        test_ema=cfg.test_ema,
                        num_test_views=cfg.num_test_views,
                        optimization_cfg=cfg.optimization,
                        test_batch_size=test_batch_size,
                        checkpoint_path=cfg.checkpoint_path,
                        amp=False,
                        fp16=False,
                        split_batches=True,
                        evaluation_dir=cfg.test_dir,
                        run_name=cfg.name,
                    )   

    evaluator.evaluate_co3d()

    

if __name__ == "__main__":
    evaluate()