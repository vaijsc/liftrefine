import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np
import cv2
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


def interpolate_camera(initial, final, t):
    # Get the relative rotation.
    r_initial = initial["R"]
    r_final = final["R"]
    r_relative = r_final @ r_initial.transpose(-1, -2)


    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=r_initial.dtype, device=r_initial.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial["T"]
    t_final = final["T"]
    t_interpolated = t_initial + (t_final - t_initial) * t
    
    focal_lengths = initial["focal_lengths"] + (final["focal_lengths"] - initial["focal_lengths"]) * t
    principal_points = initial["principal_points"] + (final["principal_points"] - initial["principal_points"]) * t

    interpolated_cameras = {
                "R" : r_interpolated.unsqueeze(1),
                "T" : t_interpolated.unsqueeze(1),
                "focal_lengths" : focal_lengths.unsqueeze(1),
                "principal_points" : principal_points.unsqueeze(1),
    }
    return interpolated_cameras

def plucker_embedding(rays):
    cam_origin = rays[:, :3]
    cam_direction = rays[:, 3:6]
    cross = torch.cross(cam_origin, cam_direction, dim=1)
    plucker = torch.cat((cam_direction, cross), dim=1)

    return plucker
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize(a):
    return (a - a.min()) / (a.max() - a.min())


def jet_depth(depth):
    # depth (B, H, W)
    # normalize depth to [0,1]
    depth = normalize(depth)
    depth = plt.cm.jet(depth.numpy())[..., :3]  # (B, H, W, 3)
    return depth


def jet_depth_scale(depth, max=2.0):
    # depth (B, H, W)
    # normalize depth to [0,1]
    depth = np.clip(depth.numpy(), 0, max)
    depth = normalize(depth)
    depth = plt.cm.jet(depth)[..., :3]  # (B, H, W, 3)
    return depth


def put_optical_flow_arrows_on_image(
    image, optical_flow_image, threshold=1.0, skip_amount=30
):
    # Don't affect original image
    # image = image.copy()

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(
            range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])
        ),
        2,
    )
    flow_end = (
        optical_flow_image[flow_start[:, :, 1], flow_start[:, :, 0], :1] * 3
        + flow_start
    ).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(
            image,
            pt1=tuple(flow_start[y, x]),
            pt2=tuple(flow_end[y, x]),
            color=(0, 255, 0),
            thickness=1,
            tipLength=0.2,
        )
    return image


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(-radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    # c2w[2, -1] += radius
    return c2w


def render_spherical(model_input, model, resolution, n):
    radius = (1.2 + 4.0) * 0.5

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            torch.einsum(
                "ij, jk -> ik",
                model_input["ctxt_c2w"][0].cpu(),
                pose_spherical(angle, -0.0, radius).cpu(),
            )
            # pose_spherical(angle, -10., radius)
            for angle in np.linspace(-180, 180, n + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

    # torch.set_printoptions(precision=2)
    frames = []

    # print(model_input.keys())
    for k in ["x_pix", "intrinsics", "ctxt_rgb", "ctxt_c2w", "idx", "z_near", "z_far"]:
        if k in model_input:
            model_input[k] = model_input[k][:1]

    for i in range(n):
        model_input["trgt_c2w"] = render_poses[i : i + 1].cuda()

        with torch.no_grad():
            rgb_pred, depth_pred, _ = model(model_input)

        rgb_pred = (
            rgb_pred.cpu().view(*(1, resolution[1], resolution[2], 3)).detach().numpy()
        )
        frames.append(rgb_pred)
    return frames


def exists(x):
    return x is not None


def get_constant_hyperparameter_schedule_with_warmup(
    num_warmup_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return lr_lambda


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    constant=False,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    def constant_lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        return 1
    
    if constant:
        return LambdaLR(optimizer, constant_lr_lambda, last_epoch)
    else:
        return LambdaLR(optimizer, lr_lambda, last_epoch)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def to_gpu(ob, device):
    if isinstance(ob, dict):
        return {k: to_gpu(v, device) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(to_gpu(k, device) for k in ob)
    elif isinstance(ob, list):
        return [to_gpu(k, device) for k in ob]
    else:
        try:
            return ob.to(device)
        except Exception:
            return ob


from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


@torch.no_grad()
def interpolate_pose(
    initial: Float[Tensor, "4 4"], final: Float[Tensor, "4 4"], t: float,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T
    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    t_interpolated = t_initial + (t_final - t_initial) * t

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


def add_wobble(t, radius=0.2):
    angle = 2 * np.pi * t
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    # torch make array [x,y,0]
    return torch.tensor([x, y, 0.0], device=radius.device, dtype=radius.dtype)


@torch.no_grad()
def interpolate_pose_wobble(
    initial: Float[Tensor, "4 4"],
    final: Float[Tensor, "4 4"],
    t: float,
    wobble: bool = True,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T
    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    dir = t_final - t_initial
    t_interpolated = t_initial + (dir) * t

    if wobble:
        radius = torch.norm(dir) * 0.05
        t_wobble = add_wobble(t, radius)
        t_interpolated += t_wobble

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


@torch.no_grad()
def interpolate_intrinsics(
    initial: Float[Tensor, "3 3"], final: Float[Tensor, "3 3"], t: float,
) -> Float[Tensor, "3 3"]:
    return initial + (final - initial) * t


import functools
import torch.nn as nn


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

def split_camera(cameras, batch_size):
    list_R = torch.split(cameras["R"], batch_size, dim=1)
    list_T = torch.split(cameras["T"], batch_size, dim=1)
    list_focal_lengths = torch.split(cameras["focal_lengths"], batch_size, dim=1)
    list_principal_points = torch.split(cameras["principal_points"], batch_size, dim=1)
    list_cameras = []

    for R, T, focal_lengths, principal_points in zip(list_R, list_T, list_focal_lengths, list_principal_points):
        camera = {"R": R, "T": T, \
                  "focal_lengths" : focal_lengths, "principal_points" : principal_points}
        list_cameras.append(camera)

    return list_cameras


def concat_camera(list_cameras):
    dict_key = ["R", "T", "focal_lengths", "principal_points"]
    all_cameras = {}

    for key in dict_key:
        data = []
        for camera in list_cameras:
            data.append(camera[key])
        data = torch.cat(data, dim=1)
        all_cameras[key] = data

    return all_cameras

    list_R = torch.split(cameras["R"], batch_size, dim=1)
    list_T = torch.split(cameras["T"], batch_size, dim=1)
    list_focal_lengths = torch.split(cameras["focal_lengths"], batch_size, dim=1)
    list_principal_points = torch.split(cameras["principal_points"], batch_size, dim=1)
    list_cameras = []

    for R, T, focal_lengths, principal_points in zip(list_R, list_T, list_focal_lengths, list_principal_points):
        camera = {"R": R, "T": T, \
                  "focal_lengths" : focal_lengths, "principal_points" : principal_points}
        list_cameras.append(camera)

    return list_cameras

def feature_map_pca(feature_maps):
    batch_feature_map_pixels = []
    for feature_map in feature_maps:
        _, h, w = feature_map.shape
        feature_map = feature_map.reshape(feature_map.shape[0], -1)
        feature_map_pixels = feature_map.permute(1, 0).cpu().numpy()


        feature_map_pixels = PCA(n_components=3).fit_transform(feature_map_pixels)
        feature_map_pixels = feature_map_pixels.transpose(1, 0).reshape(3, h, w)

        feature_map_pixels = (feature_map_pixels - feature_map_pixels.min()) / (feature_map_pixels.max() - feature_map_pixels.min()) 
        feature_map_pixels = torch.tensor(feature_map_pixels, dtype=torch.float32, device=feature_maps.device)

        batch_feature_map_pixels.append(feature_map_pixels.unsqueeze(0))

    features_map = torch.cat(batch_feature_map_pixels)

    return features_map


def make_grid_4d(videos, nrow):
    new_videos = []
    for frame in videos.transpose(0, 1):
        frame = make_grid(frame, nrow=1)
        new_videos.append(frame)

    return torch.stack(new_videos, dim=0)


def drop_view(images, cameras, num_views=None):
    if num_views is None:
        nviews = images.size(1)
        num_views = torch.randint(nviews, (1,)) + 1
    elif num_views > 1:
        num_views = torch.randint(num_views, (1,)) + 1
        num_views = 2
    
    new_cameras = {}
    new_images  = images[:, :num_views]
    new_cameras["R"] = cameras["R"][:, :num_views]
    new_cameras["T"] = cameras["T"][:, :num_views]
    new_cameras["focal_lengths"] = cameras["focal_lengths"][:, :num_views]
    new_cameras["principal_points"] = cameras["principal_points"][:, :num_views]

    return new_images, new_cameras


def split_view(images, cameras):
    nviews = images.size(1)
    num_views = torch.randint(nviews-1, (1,)) + 1
    
    new_cameras_1 = {}
    new_images_1  = images[:, :num_views]
    new_cameras_1["R"] = cameras["R"][:, :num_views]
    new_cameras_1["T"] = cameras["T"][:, :num_views]
    new_cameras_1["focal_lengths"] = cameras["focal_lengths"][:, :num_views]
    new_cameras_1["principal_points"] = cameras["principal_points"][:, :num_views]
    
    new_cameras_2 = {}
    new_images_2  = images[:, num_views:]
    new_cameras_2["R"] = cameras["R"][:, num_views:]
    new_cameras_2["T"] = cameras["T"][:, num_views:]
    new_cameras_2["focal_lengths"] = cameras["focal_lengths"][:, num_views:]
    new_cameras_2["principal_points"] = cameras["principal_points"][:, num_views:]

    return new_images_1, new_cameras_1, new_images_2, new_cameras_2