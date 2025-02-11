from . import QuadrupleDataset
from . import SRNDataset
from . import CO3DDataset

def get_data_manager(cfg, category, split='train',
                     **kwargs):

    assert split in ['train', 'val', 'test'], "Invalid split"

    if cfg.data.dataset_type == "skins":
        if split == 'train':
            dataset_name = "training"
        elif split == 'val':
            dataset_name = "ID_skin_OOD_pose_OOD_cond"
        elif split == 'test':
            dataset_name = "testing"
        dataset = QuadrupleDataset(cfg, dataset_name=dataset_name, **kwargs)
    elif cfg.data.dataset_type == "co3d":
        dataset = CO3DDataset(cfg, category, dataset_name=split, **kwargs)
    elif cfg.data.dataset_type == "srn":
        dataset = SRNDataset(cfg, category, dataset_name=split, **kwargs)
        
    return dataset