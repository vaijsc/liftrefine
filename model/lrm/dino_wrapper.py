# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel


class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        if freeze:
            self._freeze()

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [-1,1] scale and properly sized
        image = image * 0.5 + 0.5
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False, do_resize=True).to(self.model.device)
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(**inputs, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            processor = ViTImageProcessor.from_pretrained(model_name)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err


from pytorch3d.renderer import (
                                    FoVPerspectiveCameras,
                                    PerspectiveCameras,
                                )
import torch

class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.
    
    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        R_matrix = x.R
        T_matrix = x.T
        if isinstance(x, FoVPerspectiveCameras):
            focal_length       = torch.zeros((R_matrix.size(0), 2)).to(R_matrix.device)
            principal_point    = torch.zeros((R_matrix.size(0), 2)).to(R_matrix.device)     
        elif isinstance(x, PerspectiveCameras):
            focal_length = x.focal_length
            principal_point = x.principal_point
            
        camera_emb = torch.cat([R_matrix.flatten(-2, -1), T_matrix, focal_length, principal_point], dim=-1)
        return self.mlp(camera_emb)