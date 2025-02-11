import sys
import os
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator

from data_manager import get_data_manager
from model.reconstructor import Reconstructor
from model.wrapper import ModelWrapper
from ldm.util import instantiate_from_config
from accelerate.utils import set_seed

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
    dataset = get_data_manager(cfg, cfg.data.category[0], split="test")
    print('Finish loading data manager')

    dl = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    opt_base = cfg.optimization.diffusion_cfg
    configs = OmegaConf.load(opt_base) 
    diffusion_model = instantiate_from_config(configs.model).cuda()

    logdir          = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    reconstructor   = Reconstructor(cfg).cuda()
    reconstructor   = ModelWrapper(reconstructor, cfg=cfg).cuda()
    
    evaluator = DiffusionEvaluator(
                        reconstructor=reconstructor,
                        diffusion_model=diffusion_model,
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
    evaluator.evaluate_co3d_qualitative_autoregressive()
    # evaluator.evaluate_co3d_qualitative()


if __name__ == "__main__":
    evaluate()