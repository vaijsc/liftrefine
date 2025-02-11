import torch
from torchvision import transforms
import numpy as np

import hydra
from omegaconf import DictConfig

import os
from tqdm import tqdm

from . import (
    EXCLUDE_SEQUENCE, 
    LOW_QUALITY_SEQUENCE, 
    CAMERAS_CLOSE_SEQUENCE, 
    CAMERAS_FAR_AWAY_SEQUENCE
)

from . import augment_cameras, normalize_sequence
# from .co3d_utils.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import look_at_view_transform

class CO3DDataset():
    def __init__(self, cfg, category,
                 dataset_name="train"):

        self.cfg = cfg
        self.category = category
        self.data_root = cfg.data.path 
        self.dataset_name = dataset_name
        self.split_idx = cfg.data.split_idx
        # =============== Dataset parameters ===============
        self.no_imgs_per_example = cfg.data.no_imgs_per_example
        self.novel_view_weight = cfg.data.novel_view_weight

        # =============== Dataset loading ===============
        self.data_path = os.path.join(self.data_root, 
                                 self.category)
        self.stat_path = os.path.join(self.data_path, \
                        f"co3d_{self.category}_{self.cfg.data.white_background}_test_stat.pkl")
        try:
            self.read_dataset(self.data_path, dataset_name)
        except:
            if not os.path.isdir(self.data_path):
                os.mkdir(self.data_path)
            print("building dataset from scratch at {}".format(self.data_path))
            self.create_dataset(cfg, self.data_path, dataset_name)
            return 0

        # self.preprocess_pose_embeddings()
        self.sequence_starts_from = torch.tensor(self.sequence_starts_from)

        # =============== Dataset order for evaluation ===============
        if dataset_name == "test" or dataset_name == "val":
            self.fixed_frame_idxs = torch.from_numpy(np.load(os.path.join(self.data_path,
                                                                          "valid_fixed_frame_idxs_{}.npy".format(dataset_name))))


    def create_dataset(self, cfg, data_out_path, dataset_name):
        # run dataset creation 
        # check flagged sequences
        # copy over old validation split order

        # implement foreground augmentation
        # change training, validation and testing to have white backgrounds

        subset_name = "fewview_dev"

        expand_args_fields(JsonIndexDatasetMapProviderV2)
        dataset_map = JsonIndexDatasetMapProviderV2(
            category=self.category,
            subset_name=subset_name,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_root=self.data_root,
            dataset_JsonIndexDataset_args=DictConfig(
                {"remove_empty_masks": False, "load_point_clouds": True}#,\
                # "center_crop": True, "load_depth_masks": False, "load_depths": False, "load_masks": True }
            ),
        ).get_dataset_map()

        self.created_dataset = dataset_map[dataset_name]

        # Exclude bad and low quality sequences
        if self.category in EXCLUDE_SEQUENCE.keys():
            valid_sequence_names = [k for k in self.created_dataset.seq_annots.keys() if k not in EXCLUDE_SEQUENCE[self.category]]
        else:
            valid_sequence_names = list(self.created_dataset.seq_annots.keys())
        if self.category in LOW_QUALITY_SEQUENCE.keys():
            valid_sequence_names = [k for k in valid_sequence_names if k not in LOW_QUALITY_SEQUENCE[self.category]]

        self.images_all_sequences = {}
        self.focal_lengths_all_sequences = {}
        self.principal_points_all_sequences = {}
        self.camera_Rs_all_sequences = {}
        self.camera_Ts_all_sequences = {}
        min_overall_distance = 100000
        max_overall_distance = 0
        sequences_that_need_checking = []
        for sequence_name in tqdm(valid_sequence_names):
            frame_idx_gen = self.created_dataset.sequence_indices_in_order(sequence_name)
            frame_idxs = []
            images_this_sequence = []
            focal_lengths_this_sequence = []
            principal_points_this_sequence = []

            while True:
                try:
                    frame_idx = next(frame_idx_gen)
                    frame_idxs.append(frame_idx)
                except StopIteration:
                    break

            for frame_idx in frame_idxs:
                frame = self.created_dataset[frame_idx]
                rgb = torch.cat([frame.image_rgb, frame.fg_probability], dim=0)
                assert frame.image_rgb.shape[1] == frame.image_rgb.shape[2], "Expected square images"
                assert rgb.shape[0] == 4, "Expected RGBA images, got {}".format(rgb.shape[0])
                # resizing_factor = self.cfg.data.input_size[0] / frame.image_rgb.shape[1]
                rgb = transforms.functional.resize(rgb,
                                                self.cfg.data.input_size[0],
                                                interpolation=transforms.InterpolationMode.BILINEAR)
                # cameras are in NDC convention so when resizing the image we do not need to change
                # the focal length or principal point
                focal_lengths_this_sequence.append(frame.camera.focal_length)
                principal_points_this_sequence.append(frame.camera.principal_point)                
                images_this_sequence.append(rgb.unsqueeze(0))
                
            self.images_all_sequences[sequence_name] = torch.cat(images_this_sequence,
                                                                 dim=0)
            self.focal_lengths_all_sequences[sequence_name] = torch.cat(focal_lengths_this_sequence,
                                                         dim=0)
            self.principal_points_all_sequences[sequence_name] = torch.cat(principal_points_this_sequence,
                                                                           dim=0)
            
            normalized_cameras, min_dist, max_dist, _, needs_checking = normalize_sequence(self.created_dataset, sequence_name,
                                                                                           self.cfg.render.volume_extent_world)
            if needs_checking:
                sequences_that_need_checking.append(str(sequence_name) + "\n")
            self.camera_Rs_all_sequences[sequence_name] = normalized_cameras.R
            self.camera_Ts_all_sequences[sequence_name] = normalized_cameras.T

            if min_dist < min_overall_distance:
                min_overall_distance = min_dist
            if max_dist > max_overall_distance:
                max_overall_distance = max_dist

        print("Min distance: ", min_overall_distance)
        print("Max distance: ", max_overall_distance)
        with open(os.path.join(data_out_path, "sequences_to_check_{}.txt".format(dataset_name)), "w+") as f:
            f.writelines(sequences_that_need_checking)
        # get the sequence names - this is what we will sample from
        self.sequence_names = [k for k in self.images_all_sequences.keys()]
        self.sequence_starts_from = [0]
        for i in range(1, len(self.sequence_names)+1):
            self.sequence_starts_from.append(self.sequence_starts_from[-1] + len(self.images_all_sequences[self.sequence_names[i-1]]))

        # convert the data to numpy archives and save
        for dict_to_save, dict_name in zip([self.images_all_sequences,
                                            self.focal_lengths_all_sequences,
                                            self.principal_points_all_sequences,
                                            self.camera_Rs_all_sequences,
                                            self.camera_Ts_all_sequences],
                                           ["images",
                                            "focal_lengths",
                                            "principal_points",
                                            "camera_Rs",
                                            "camera_Ts"]):
            np.savez(os.path.join(data_out_path, dict_name+"_{}.npz".format(dataset_name)),
                                  **{k: v.detach().cpu().numpy() for k, v in dict_to_save.items()})
        
        # If the dataset is being made for evaluation we need to fix the frame indices that are
        # passed in as one batch. Each batch should have 4 images - this will support 
        # 3-image testing, 2-image testing and 1-image testing. The images for each batch should
        # be from the same sequence. The frames should be selected randomly. The batches should
        # include as many images from every sequence as possible, sampling randomly without 
        # replacement.
        if dataset_name == "test" or dataset_name == "val":
            self.fixed_frame_idxs = []
            for sequence_name in self.sequence_names:
                sequence_length = len(self.images_all_sequences[sequence_name])
                # randomly permute the frame indices within the sequence and then split into
                # batches of 4
                frame_idxs = torch.randperm(sequence_length)
                frame_idxs = frame_idxs[:len(frame_idxs) // 4 * 4]
                frame_idxs = frame_idxs.view(-1, 4)

                valid_frame_idxs = []
                for frame_idx in frame_idxs:
                    rgbs = self.images_all_sequences[sequence_name][frame_idx]
                    masks = rgbs[:, 3:, ...]
                    non_zero_values = masks.mean(dim=[1,2,3])
                    if torch.all(non_zero_values > 0.1):
                        valid_frame_idxs.append(frame_idx)

                valid_frame_idxs = torch.stack(valid_frame_idxs, dim=0)
                self.fixed_frame_idxs.append(valid_frame_idxs + 
                                                self.sequence_starts_from[self.sequence_names.index(sequence_name)])

            np.save(os.path.join(data_out_path, "valid_fixed_frame_idxs_{}.npy".format(dataset_name)),
                    torch.cat(self.fixed_frame_idxs, dim=0).detach().cpu().numpy())

        return None

    def read_dataset(self, data_path, dataset_name):
    
        join_excluded_sequences = []
        for excluded_category_dict in [EXCLUDE_SEQUENCE, 
                                       LOW_QUALITY_SEQUENCE, 
                                       CAMERAS_FAR_AWAY_SEQUENCE, 
                                       CAMERAS_CLOSE_SEQUENCE]:
            if self.category in excluded_category_dict.keys():
                join_excluded_sequences = join_excluded_sequences + excluded_category_dict[self.category]
        # read the data from the npz archives
        self.images_all_sequences = {k: torch.from_numpy(v) for k, v in 
                                     tqdm(np.load(os.path.join(data_path, "images_{}.npz".format(dataset_name))).items())
                                     if k not in join_excluded_sequences}
        self.focal_lengths_all_sequences = {k: torch.from_numpy(v) for k, v in
                                            np.load(os.path.join(data_path, "focal_lengths_{}.npz".format(dataset_name))).items()
                                            if k not in join_excluded_sequences}
        self.principal_points_all_sequences = {k: torch.from_numpy(v) for k, v in
                                               np.load(os.path.join(data_path, "principal_points_{}.npz".format(dataset_name))).items()
                                               if k not in join_excluded_sequences}
        self.camera_Rs_all_sequences = {k: torch.from_numpy(v) for k, v in
                                        np.load(os.path.join(data_path, "camera_Rs_{}.npz".format(dataset_name))).items()
                                        if k not in join_excluded_sequences}
        self.camera_Ts_all_sequences = {k: torch.from_numpy(v) for k, v in
                                        np.load(os.path.join(data_path, "camera_Ts_{}.npz".format(dataset_name))).items()
                                        if k not in join_excluded_sequences}

        min_overall_distance = 1000000
        max_overall_distance = 0

        for seq_name, camera_Ts in self.camera_Ts_all_sequences.items():
            camera_dists = torch.norm(camera_Ts, dim=1)
            if camera_dists.min() < min_overall_distance:
                min_overall_distance = camera_dists.min()
                min_dist_seq = seq_name
            if camera_dists.max() > max_overall_distance:
                max_overall_distance = camera_dists.max()
                max_dist_seq = seq_name

        print("Min distance: ", min_overall_distance)
        print("Min distance seq: ", min_dist_seq)
        print("Max distance: ", max_overall_distance)
        print("Max distance seq: ", max_dist_seq)

        # For video rendering

        elev_range      = np.linspace(10, 50, num=50)
        azim_range      = np.linspace(0, 360, num=50)
        lap_R, lap_T    = look_at_view_transform(dist=2.5, elev=elev_range, azim=azim_range, degrees=True)

        self.render_camera_Rs         = lap_R
        self.render_camera_Ts         = lap_T
        self.render_principal_points  = torch.zeros([lap_R.size(0), 2])
        self.render_focal_lengths     = torch.ones([lap_R.size(0), 2]) * 3.5
        ##########################################
        
        self.sequence_names = [k for k in self.images_all_sequences.keys()]
        self.sequence_starts_from = [0]
        for i in range(1, len(self.sequence_names)+1):
            self.sequence_starts_from.append(self.sequence_starts_from[-1] + len(self.images_all_sequences[self.sequence_names[i-1]]))

    def get_camera_screen_unprojected(self):
        # Step 1. and 2 to encode ray direction
        # 1. generate a grid of x, y coordinates of every point in screen coordinates
        # 2. convert the grid to x, y coordinates in NDC coordinates
        # NDC coordinates are positive up and left, the image pixel matrix indexes increase down and right
        # so to go to NDC we need to invert the direction
        Y, X = torch.meshgrid(-(torch.linspace(0.5, self.cfg.data.input_size[1]-0.5, 
                                                    self.cfg.data.input_size[1]) * 2 / self.cfg.data.input_size[1] - 1),
                              -(torch.linspace(0.5, self.cfg.data.input_size[0]-0.5,
                                                    self.cfg.data.input_size[0]) * 2 / self.cfg.data.input_size[0] - 1),
                                indexing='ij')
        Z = torch.ones_like(X)
        return X, Y, Z

    def get_raydir_embedding(self, camera_R, principal_points, focal_lengths, X, Y, Z):
        # Steps 3 and 4 to encode ray direction
        # 3. add the depth dimension and scale focal lengths
        raydirs_cam = torch.stack(((X - principal_points[0]) / focal_lengths[0],
                                   (Y - principal_points[1]) / focal_lengths[1],
                                    Z))
        # 4. convert from camera coordinates to world coordinates
        raydirs_cam = raydirs_cam / torch.norm(raydirs_cam, dim=0, keepdim=True) # 3 x H x W
        raydirs_cam = raydirs_cam.permute(1, 2, 0).reshape(-1, 3)
        # camera to world rotation matrix is camera_R.T. It assumes row-major order, i.e. that the
        # position vectors are row vectors so we post-multiply row vectors by the rotation matrix
        # camera position gets ignored because we want ray directions, not their end-points.
        raydirs_world = torch.matmul(raydirs_cam, camera_R.T)
        raydirs_world = raydirs_world.reshape(self.cfg.data.input_size[0], 
                                              self.cfg.data.input_size[1], 3).permute(2, 0, 1).float().unsqueeze(0)
        return raydirs_world

    def preprocess_pose_embeddings(self):
        self.pose_orig_embed_all_sequences = {}
        self.pose_dir_embed_all_sequences = {}
        X, Y, Z = self.get_camera_screen_unprojected()

        for sequence_name in self.sequence_names:
            H, W = self.images_all_sequences[sequence_name].shape[2:]
            
            pose_orig_embed, pose_dir_embed = self.pose_embeddings_camera_sequence(
                self.camera_Rs_all_sequences[sequence_name],
                self.camera_Ts_all_sequences[sequence_name],
                H, W,
                self.principal_points_all_sequences[sequence_name],
                self.focal_lengths_all_sequences[sequence_name],
                X, Y, Z
            )

            self.pose_orig_embed_all_sequences[sequence_name] = pose_orig_embed
            self.pose_dir_embed_all_sequences[sequence_name] = pose_dir_embed

    def pose_embeddings_camera_sequence(self, camera_Rs, camera_Ts, H, W,
                                        principal_points, focal_lengths, X, Y, Z):
        pose_orig_embeds = []
        pose_dir_embeds = []
        for camera_idx in range(len(camera_Rs)):
            camera_R = camera_Rs[camera_idx]
            camera_T = camera_Ts[camera_idx]
            T_embedded = - torch.matmul(camera_R, camera_T.clone().detach())
            pose_orig_embeds.append(T_embedded[..., None, None].repeat(1, 1, H, W).float())
            # encode camera direction with the z-vector of the camera (away from the image)
            # z-vector in world coordinates is the third column of the rotation matrix
            assert self.cfg.data.encode_rays
            raydirs_world = self.get_raydir_embedding(camera_R, 
                                                      principal_points[camera_idx],
                                                      focal_lengths[camera_idx],
                                                      X, Y, Z)
            pose_dir_embeds.append(raydirs_world)

        pose_orig_embeds = torch.cat(pose_orig_embeds, dim=0)
        pose_dir_embeds = torch.cat(pose_dir_embeds, dim=0)

        return pose_orig_embeds, pose_dir_embeds

    def __len__(self):
        if hasattr(self, "fixed_frame_idxs"):
            return len(self.fixed_frame_idxs)
        else:
            return len(self.sequence_names)
        
    def __getitem__(self, idx):
        sampling_cameras = {}
        if hasattr(self, "fixed_frame_idxs"):
            frame_idxs = self.fixed_frame_idxs[idx]
            # for the sequence name need to find the sequence that the frame indices belong to
            # this is done by finding in which interval in self.sequence_starts_from the frame index falls
            sequence_name = self.sequence_names[torch.searchsorted(self.sequence_starts_from, frame_idxs[0], right=True)-1]
            # the first N-1 frames are conditioning, the last one is the target
            sampling_cameras = self.get_all_poses_from_sequence(sequence_name)
            frame_idxs = frame_idxs - self.sequence_starts_from[self.sequence_names.index(sequence_name)]
            rendered_idxs = frame_idxs[-1:]
        else:
            sequence_name = self.sequence_names[idx]
            frame_idxs = torch.randint(self.sequence_starts_from[idx],
                                       self.sequence_starts_from[idx+1],
                                       (self.no_imgs_per_example,)) - self.sequence_starts_from[idx]
            rendered_idxs = torch.randint(self.sequence_starts_from[idx],
                                       self.sequence_starts_from[idx+1],
                                       (1,)) - self.sequence_starts_from[idx]
        
        attributes = self.get_attributes_selected_sequence_and_frames(sequence_name, frame_idxs, rendered_idxs)
        result = self.my_rearange(attributes)
        result["object_names"] = idx
        result.update(sampling_cameras)

        return result

    def get_all_poses_from_sequence(self, sequence_name):
        camera_Rs   = self.camera_Rs_all_sequences[sequence_name].clone()
        camera_Ts   = self.camera_Ts_all_sequences[sequence_name].clone()

        principal_points    = self.principal_points_all_sequences[sequence_name]
        focal_lengths       = self.focal_lengths_all_sequences[sequence_name]

        rand_idx            = np.random.randint(0, camera_Rs.size(0), size=100)
        camera_Rs           = camera_Rs[rand_idx]
        camera_Ts           = camera_Ts[rand_idx]
        principal_points    = principal_points[rand_idx]
        focal_lengths       = focal_lengths[rand_idx]

        result = {}
        result["sampling_camera_Rs"]         = camera_Rs
        result["sampling_camera_Ts"]         = camera_Ts
        result["sampling_principal_points"]  = principal_points
        result["sampling_focal_lengths"]     = focal_lengths

        return result

    def get_attributes_selected_sequence_and_frames(self, sequence_name, frame_idxs, rendered_idxs):
        rgbs = self.images_all_sequences[sequence_name][frame_idxs].clone()
        clip_images = rgbs[0:1] 

        rgbs = torch.nn.functional.interpolate(rgbs, scale_factor=0.5, mode='bilinear', antialias=True)
        rendered_rgbs = self.images_all_sequences[sequence_name][rendered_idxs].clone() 

        if rgbs.shape[1] == 4:
            # get rid of the background
            if self.cfg.data.white_background:
                bkgd = 1.0
            else:
                bkgd = 0.0
            if self.cfg.data.background:
                rgbs = rgbs[:, :3, ...]
                rendered_rgbs = rendered_rgbs[:, :3, ...]
                clip_images = clip_images[:, :3, ...]
            else:
                rgbs = rgbs[:, :3, ...] * rgbs[:, 3:, ...] + bkgd * (1-rgbs[:, 3:, ...])
                rendered_rgbs = rendered_rgbs[:, :3, ...] * rendered_rgbs[:, 3:, ...] + bkgd * (1-rendered_rgbs[:, 3:, ...])
                clip_images = clip_images[:, :3, ...] * clip_images[:, 3:, ...] + bkgd * (1-clip_images[:, 3:, ...])

        rgbs            = torch.clamp(rgbs, 0, 1) * 2 - 1
        rendered_rgbs   = torch.clamp(rendered_rgbs, 0, 1) * 2 - 1
        clip_images     = torch.clamp(clip_images, 0, 1) * 2 - 1

        camera_Rs           = self.camera_Rs_all_sequences[sequence_name][frame_idxs].clone()
        camera_Ts           = self.camera_Ts_all_sequences[sequence_name][frame_idxs].clone()
        principal_points    = self.principal_points_all_sequences[sequence_name][frame_idxs]
        focal_lengths       = self.focal_lengths_all_sequences[sequence_name][frame_idxs]
      
        rendered_data = {}
        rendered_data["rendered_images"] = rendered_rgbs
        rendered_data["clip_images"] = clip_images
        rendered_data["rendered_camera_Rs"] = self.camera_Rs_all_sequences[sequence_name][rendered_idxs].clone()
        rendered_data["rendered_camera_Ts"] = self.camera_Ts_all_sequences[sequence_name][rendered_idxs].clone()
        rendered_data["rendered_focal_lengths"] = self.focal_lengths_all_sequences[sequence_name][rendered_idxs]
        rendered_data["rendered_principal_points"] = self.principal_points_all_sequences[sequence_name][rendered_idxs]

        return rgbs, principal_points, focal_lengths, camera_Rs, camera_Ts, rendered_data


    def my_rearange(self, attributes):
        rgbs, principal_points, focal_lengths, camera_Rs, camera_Ts, rendered_data = attributes

        input_images            = rgbs[:self.split_idx]
        input_camera_Rs         = camera_Rs[:self.split_idx]
        input_camera_Ts         = camera_Ts[:self.split_idx]
        input_principal_points  = principal_points[:self.split_idx]
        input_focal_lengths     = focal_lengths[:self.split_idx]

        if np.random.rand() < (1 - self.novel_view_weight) and self.dataset_name != "test":
            # using reconstruction image
            target_images           = rgbs[0:1]
            target_camera_Rs        = camera_Rs[0:1]
            target_camera_Ts        = camera_Ts[0:1]
            target_principal_points = principal_points[0:1]
            target_focal_lengths    = focal_lengths[0:1]
        else:
            # using novel view image
            target_images           = rgbs[self.split_idx:]
            target_camera_Rs        = camera_Rs[self.split_idx:]
            target_camera_Ts        = camera_Ts[self.split_idx:]
            target_principal_points = principal_points[self.split_idx:]
            target_focal_lengths    = focal_lengths[self.split_idx:]

        result                              = {}
        result["input_images"]              = input_images
        result["input_camera_Rs"]           = input_camera_Rs
        result["input_camera_Ts"]           = input_camera_Ts
        result["input_principal_points"]    = input_principal_points
        result["input_focal_lengths"]       = input_focal_lengths

        result["target_images"]             = target_images
        result["target_camera_Rs"]          = target_camera_Rs
        result["target_camera_Ts"]          = target_camera_Ts
        result["target_principal_points"]   = target_principal_points
        result["target_focal_lengths"]      = target_focal_lengths

        result["render_camera_Rs"]         = self.render_camera_Rs
        result["render_camera_Ts"]         = self.render_camera_Ts
        result["render_principal_points"]  = self.render_principal_points
        result["render_focal_lengths"]     = self.render_focal_lengths

        result.update(rendered_data)
        return result
