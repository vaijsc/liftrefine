import os
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

class Logger:
    def __init__(self, logdir):
        self.writer = SummaryWriter(os.path.join(logdir, "logger"))

    def add_scalars(self, log_dict, global_step):
        for key in log_dict.keys():
            self.writer.add_scalar(key, log_dict[key], global_step=global_step)
    
    def add_scalar(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step=global_step)
    
    def add_images(self, log_dict, global_step):
        for key in log_dict.keys():
            self.writer.add_image(key, log_dict[key], global_step, dataformats="CHW")

    def add_videos(self, tag, batch_videos, global_step):
        self.writer.add_video(tag, batch_videos, global_step=global_step, fps=8)


