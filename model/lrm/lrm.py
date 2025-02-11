from model.lrm.dino_wrapper import DinoWrapper, CameraEmbedder
from model.lrm.transformer import TriplaneTransformer
from torch import nn

lrm_small_model_kwagrs ={'camera_embed_dim': 768, \
                        'transformer_dim': 768, \
                        'transformer_layers': 12, \
                        'transformer_heads': 16, \
                        'triplane_low_res': 32, \
                        'triplane_high_res': 64, \
                        'triplane_dim': 32, 
                        'encoder_freeze': False}

class LRMGenerator(nn.Module):
    """
    Full model of the large reconstruction model.
    """
    def __init__(self, camera_embed_dim: int,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                 encoder_freeze: bool = True, encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768):
        super().__init__()
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim

        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12+4, embed_dim=camera_embed_dim,
        )
        self.transformer = TriplaneTransformer(
            inner_dim=transformer_dim, num_layers=transformer_layers, num_heads=transformer_heads,
            image_feat_dim=encoder_feat_dim,
            camera_embed_dim=camera_embed_dim,
            triplane_low_res=triplane_low_res, triplane_high_res=triplane_high_res, triplane_dim=triplane_dim,
        )


    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        N = image.shape[0]

        # encode image
        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating planes
        planes = self.transformer(image_feats, camera_embeddings)
        return planes

    def forward(self, image, source_camera):
        # image: [N, N_images, C_img, H_img, W_img]
        # source_camera: [N, D_cam_raw]
        image = image.flatten(0, 1)
        assert image.shape[0] == len(source_camera), "Batch size mismatch for image and source_camera"

        planes = self.forward_planes(image, source_camera)

        return planes