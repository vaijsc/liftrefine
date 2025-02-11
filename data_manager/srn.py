import glob
import imageio
import os
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms

from . import augment_cameras
from pytorch3d.renderer.cameras import look_at_view_transform


class SRNDataset():
    def __init__(self, cfg, category, 
                 dataset_name="train",
                 source_img_idxs=[64],
                 ):
        self.cfg = cfg
        assert self.cfg.data.white_background # This flag should be false for shapenet

        self.category = category
        self.data_root = cfg.data.path 
        self.dataset_name = dataset_name
        self.no_imgs_per_example = cfg.data.no_imgs_per_example

        self.original_size = 128
        if self.cfg.data.input_size[0] != self.original_size:
            # we will resize the images, adjust the focal length later in dataloading
            # we do not need to adjust the world size if focal length is adjusted
            self.resize_transform = transforms.Resize((self.cfg.data.input_size[0], 
                                                       self.cfg.data.input_size[1]))

        self.source_img_idxs = source_img_idxs
        self.base_path = os.path.join(self.data_root, "{}/{}_{}".format(self.category,
                                                                        self.category,
                                                                        dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        self.image_to_tensor = transforms.ToTensor()

        # SRN dataset is in convention x right, y down, z away from camera
        # Pytorch3D is in convention x left, y up, z away from the camera
        self._coord_trans = torch.diag(
            torch.tensor([-1, -1, 1, 1], dtype=torch.float32)
        )

        # focal field of view remains unchanged
        fov_focal = cfg.render.fov * 2 * np.pi / 360
        # focal length in pixels is adjusted with the data input size
        # Pytorch3D cameras created are FOV cameras with default
        # principal point at 0 so we do not need to adjust principal point
        self.focal = cfg.data.input_size[0] / (2 * np.tan(fov_focal / 2))

        # For video rendering
        path    = os.path.join(self.data_root, self.category)
        lap_R   = torch.load(os.path.join(path, "render_camera_Rs.pt"))[::5]
        lap_T   = torch.load(os.path.join(path, "render_camera_Ts.pt"))[::5]

        self.render_camera_Rs         = lap_R
        self.render_camera_Ts         = lap_T
        self.render_principal_points  = torch.zeros([lap_R.size(0), 2]) # dummy value
        self.render_focal_lengths     = torch.ones([lap_R.size(0), 2]) # dummy value
        ##########################################

    def pose_to_target_Rs_and_Ts(self, pose):
        # pose is the camera to world matrix in column major order
        # Pytorch3D expects target R and T in row major order
        target_T = - pose[:3, :3].T @ pose[:3, 3]
        target_R = pose[:3, :3] # transpose for inverse and for
        # changing the major axis swap
        return target_R, target_T


    def __len__(self):
        return len(self.intrins)


    def get_attributes_selected_sequence_and_frames(self, example_id, frame_idxs,
                                                    for_testing = False):
        all_images = []
        camera_Rs = []
        camera_Ts = []
    
        for frame_idx in frame_idxs:
            img = self.all_rgbs[example_id][frame_idx].unsqueeze(0)
            pose = self.all_poses[example_id][frame_idx]
            pose = pose @ self._coord_trans
            target_R, target_T = self.pose_to_target_Rs_and_Ts(pose)

            all_images.append(img)
            camera_Rs.append(target_R.unsqueeze(0))
            camera_Ts.append(target_T.unsqueeze(0))

        all_images = torch.cat(all_images, dim=0)
        camera_Rs = torch.cat(camera_Rs, dim=0)
        camera_Ts = torch.cat(camera_Ts, dim=0)

        return all_images, camera_Rs, camera_Ts


    def load_example_id(self, example_id, intrin_path,
                        idxs_to_load=None):
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        assert len(rgb_paths) == len(pose_paths)

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_poses = {}
            self.all_above_0_z_ind = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_poses[example_id] = []
            self.all_above_0_z_ind[example_id] = []

            if idxs_to_load is not None:
                rgb_paths_load = [rgb_paths[i] for i in idxs_to_load]
                pose_paths_load = [pose_paths[i] for i in idxs_to_load]
            else:
                rgb_paths_load = rgb_paths
                pose_paths_load = pose_paths

            for path_idx, (rgb_path, pose_path) in enumerate(zip(rgb_paths_load, pose_paths_load)):
                rgb = imageio.imread(rgb_path)[..., :3]
                rgb = self.image_to_tensor(rgb)
                pose = torch.from_numpy(
                    np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
                )
                self.all_rgbs[example_id].append(rgb)
                self.all_poses[example_id].append(pose)
                if pose[2, 3] > 0:
                    self.all_above_0_z_ind[example_id].append(path_idx)

            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
            self.all_poses[example_id] = torch.stack(self.all_poses[example_id])


    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id


    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        if self.dataset_name == "train" or self.dataset_name == "val":
            # Loads all frames in an example
            self.load_example_id(example_id, intrin_path)
            frame_idxs = [self.all_above_0_z_ind[example_id][i] 
                          for i in torch.randperm(len(self.all_above_0_z_ind[example_id]))[:self.no_imgs_per_example]]
        else:
            self.load_example_id(example_id, intrin_path)
            frame_idxs = [self.all_above_0_z_ind[example_id][i] 
                                for i in range(len(self.all_above_0_z_ind[example_id]))]
            
            if len(frame_idxs) != 251:
                raise Exception(f"Wrong number of images at {example_id}")
        
        attributes              = self.get_attributes_selected_sequence_and_frames(example_id, frame_idxs)
        result                  = self.my_rearange(attributes) 
        result["object_names"]  = example_id
        return result


    def my_rearange(self, attributes):
        result                     = {}
        rgbs, camera_Rs, camera_Ts = attributes
        rgbs = rgbs * 2 - 1
        rendered_data = {}

        if self.dataset_name == "train" or self.dataset_name == "val":
            input_images    = rgbs[:-1]
            target_images   = rgbs[-1:]
            
            input_camera_Rs     = camera_Rs[:-1]
            input_camera_Ts     = camera_Ts[:-1]
            target_camera_Rs    = camera_Rs[-1:]
            target_camera_Ts    = camera_Ts[-1:]

            rendered_data["rendered_images"] = torch.nn.functional.interpolate(target_images, scale_factor=2.0, mode='bilinear')
            rendered_data["clip_images"] = torch.nn.functional.interpolate(input_images[0:1], scale_factor=2.0, mode='bilinear')
            rendered_data["rendered_camera_Rs"] = target_camera_Rs
            rendered_data["rendered_camera_Ts"] = target_camera_Ts
            rendered_data["rendered_focal_lengths"] = torch.zeros(target_camera_Ts.size(0))
            rendered_data["rendered_principal_points"] = torch.zeros(target_camera_Ts.size(0))

        else:
            input_idx       = self.source_img_idxs
            target_idx      = [i for i in range(rgbs.size(0)) if i not in input_idx]

            first_half_idx  = [i for i in range(input_idx[0]-1, input_idx[0]-5, -1)]
            # second_half_idx = [i for i in range(input_idx[0]+1, rgbs.size(0), 30)]

            interpolated_idx = [0] + [rgbs.size(0) -1]

            input_images    = rgbs[input_idx]
            input_camera_Rs = camera_Rs[input_idx]
            input_camera_Ts = camera_Ts[input_idx]

            target_images       = rgbs[target_idx]
            target_camera_Rs    = camera_Rs[target_idx]
            target_camera_Ts    = camera_Ts[target_idx]
       
            interpolated_camera_Rs = camera_Rs[interpolated_idx]
            interpolated_camera_Ts = camera_Ts[interpolated_idx]

            rendered_data["clip_images"]        = torch.nn.functional.interpolate(input_images, scale_factor=2.0, mode='bilinear')

            result["interpolated_camera_Rs"]          = interpolated_camera_Rs
            result["interpolated_camera_Ts"]          = interpolated_camera_Ts
            result["interpolated_principal_points"]   = torch.zeros(interpolated_camera_Ts.size(0))
            result["interpolated_focal_lengths"]      = torch.zeros(interpolated_camera_Ts.size(0))

        result["input_images"]              = input_images
        result["input_camera_Rs"]           = input_camera_Rs
        result["input_camera_Ts"]           = input_camera_Ts
        result["input_principal_points"]    = torch.zeros(input_camera_Ts.size(0))     # dummy value
        result["input_focal_lengths"]       = torch.zeros(input_camera_Ts.size(0))

        result["target_images"]             = target_images
        result["target_camera_Rs"]          = target_camera_Rs
        result["target_camera_Ts"]          = target_camera_Ts
        result["target_principal_points"]   = torch.zeros(target_camera_Ts.size(0))
        result["target_focal_lengths"]      = torch.zeros(target_camera_Ts.size(0))
      

        if not (self.dataset_name == "train" or self.dataset_name == "val"):
            result["target_idx"]    = torch.tensor(target_idx)

        result["render_camera_Rs"]         = self.render_camera_Rs
        result["render_camera_Ts"]         = self.render_camera_Ts
        result["render_principal_points"]  = self.render_principal_points
        result["render_focal_lengths"]     = self.render_focal_lengths

        result.update(rendered_data)
        return result