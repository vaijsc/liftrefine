import sys
import os
import datetime
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import platform
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_manager import get_data_manager
from model.reconstructor import Reconstructor
from model.wrapper import DiffusionWrapper
from model.trainer_diffusion import TrainerDiffusion
from model.diffuser import Diffuser
from model.unet_diffusion import TriplaneDiffusion
from model.sgm.unet import get_default_unet
from model.diffusion_evaluator import DiffusionEvaluator



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
    print('Finish loading data manager')

    dl = DataLoader(
        train_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    logdir          = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    reconstructor   = Reconstructor(cfg, num_latent=len(train_dataset)).cuda()

    if cfg.unet_diffusion.type == "triplane":
        unet_diffusion  = TriplaneDiffusion(cfg).cuda()
        # unet_diffusion  = get_default_unet().cuda()
    
    diffusionwrapper    = DiffusionWrapper(reconstructor, unet_diffusion, cfg.pretrained_reconstructor).cuda()
    diffuser            = Diffuser(model=diffusionwrapper, image_size=reconstructor.image_size, cfg=cfg).cuda()

    evaluator = DiffusionEvaluator(
                        diffuser=diffuser,
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
                        stat_path=train_dataset.stat_path,
                        run_name=cfg.name,
                    )   
    evaluator.evaluate_fid_latent_co3d()

    

if __name__ == "__main__":
    evaluate()