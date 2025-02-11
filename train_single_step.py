import sys
import os
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import platform
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator

from data_manager import get_data_manager
from model.reconstructor import Reconstructor
from model.wrapper import DiffusionWrapper, SDSWrapper
from model.trainer_single_step import TrainerSingleStep
from model.diffuser import Diffuser
import torch
from ldm.util import instantiate_from_config


@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def train(cfg: DictConfig):
    # dataset
    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )
    print('Loading data manager')
    opt_base = cfg.optimization.diffusion_cfg
    configs = OmegaConf.load(opt_base) 
    diffusion_model = instantiate_from_config(configs.model).cuda()

    train_dataset = get_data_manager(cfg, cfg.data.category[0], "train")   
    val_dataset = get_data_manager(cfg, cfg.data.category[0], "test")
    print('Finish loading data manager')

    train_batch_size    = cfg.batch_size 
    train_dataloader    = DataLoader(
                                        train_dataset,
                                        batch_size=train_batch_size,
                                        shuffle=True,
                                        pin_memory=False,
                                        num_workers=cfg.num_workers ,
                                    )
    
    val_dataloader      = DataLoader(
                                        val_dataset,
                                        batch_size=train_batch_size,
                                        shuffle=True,
                                        pin_memory=False,
                                        num_workers=cfg.num_workers,
                                    )

    logdir          = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    reconstructor   = Reconstructor(cfg).cuda()
    reconstructor   = SDSWrapper(reconstructor, cfg=cfg).cuda()
    
    if cfg.unet_diffusion.type == "triplane":
        from model.unet_diffusion import TriplaneDiffusion
        unet_diffusion  = TriplaneDiffusion(cfg).cuda()
    elif cfg.unet_diffusion.type == "plane_aware":
        from model.unet_diffusion import PlaneAwareDiffusion
        unet_diffusion  = PlaneAwareDiffusion(cfg).cuda()
    elif cfg.unet_diffusion.type == "volume":
        from model.unet_diffusion import VolumeDiffusion
        unet_diffusion  = VolumeDiffusion(cfg).cuda()
    elif cfg.unet_diffusion.type == "volume_crossattn":
        from model.unet_diffusion import VolumeDiffusionCrossAttn
        unet_diffusion  = VolumeDiffusionCrossAttn(cfg).cuda()
    elif cfg.unet_diffusion.type == "sgm":
        from model.sgm.unet import get_default_unet
        unet_diffusion  = get_default_unet(cfg).cuda()
    
    print(f"using lr= {cfg.lr}")
    print(f"logging at {logdir}")

    trainer = TrainerSingleStep(
                        unet_model=unet_diffusion,
                        reconstructor=reconstructor,
                        diffusion_model=diffusion_model,
                        accelerator=accelerator,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        train_batch_size=train_batch_size,
                        train_lr= cfg.lr,
                        train_num_steps=cfg.num_steps,  # total training steps
                        test_num_steps=cfg.test_steps,  # total test steps per evaluation
                        gradient_accumulate_every=1,  # gradient accumulation steps
                        optimization_cfg=cfg.optimization,
                        ema_update_every=cfg.optimization.ema.update_every,
                        ema_decay=cfg.optimization.ema.decay,
                        adam_betas=cfg.optimization.betas,
                        eval_every=cfg.eval_every,
                        logging_every=cfg.logging_every,
                        summary_every=cfg.summary_every,
                        save_every=cfg.save_every,
                        warmup_period=1000,
                        pretrained_reconstructor=cfg.pretrained_reconstructor,
                        checkpoint_path=cfg.checkpoint_path,
                        amp=False,
                        fp16=False,
                        split_batches=True,
                        is_resume=cfg.resume,
                        logdir=logdir,
                        run_name=cfg.name,
                        stat_path=train_dataset.stat_path
                    )   

    trainer.train()


if __name__ == "__main__":
    train()

