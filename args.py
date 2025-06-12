# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import tyro
import os


@dataclass
class DataConfig:
    """Data and image loading options"""
    source_path: Path
    """Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)"""
    images_dir: Path = Path("images")
    """source_path/images_dir is the path to the images (with extensions jpg, png or jpeg)"""
    masks_dir: Path = Path("")
    """If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png)"""
    num_loader_threads: int = 4
    """Number of workers to load and prepare input images"""
    downsampling: float = -1.0
    """Downsampling ratio for input images"""
    pyr_levels: int = 2
    """Number of pyramid levels. Each level l will downsample the image 2^l times in width and height"""
    min_displacement: float = 0.03
    """Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width"""
    start_at: int = 0
    """Number of frames to skip from the dataset"""
    sh_degree: int = 3
    """Spherical harmonics degree"""


@dataclass
class ColmapConfig:
    """COLMAP-related options"""
    eval_poses: bool = False
    """Compare poses to COLMAP"""
    use_colmap_poses: bool = False
    """Load COLMAP data for pose and intrinsics initialization"""


@dataclass
class LearningRateConfig:
    """Learning rate parameters"""
    lr_poses: float = 1e-4
    """Pose learning rate"""
    lr_exposure: float = 5e-4
    """Exposure compensation learning rate"""
    lr_depth_scale_offset: float = 1e-4
    """Depth scale offset learning rate"""
    position_lr_init: float = 0.00005
    """Initial position learning rate"""
    position_lr_decay: float = 1 - 2e-5
    """Multiplicative decay factor for position learning rate"""
    feature_lr: float = 0.005
    """Feature learning rate"""
    opacity_lr: float = 0.1
    """Opacity learning rate"""
    scaling_lr: float = 0.01
    """Scaling learning rate"""
    rotation_lr: float = 0.002
    """Rotation learning rate"""


@dataclass
class TrainingConfig:
    """Training schedule and loss parameters"""
    lambda_dssim: float = 0.2
    """Weight for DSSIM loss"""
    num_iterations: int = 30
    """Number of training iterations per keyframe"""
    depth_loss_weight_init: float = 1e-2
    """Initial depth loss weight"""
    depth_loss_weight_decay: float = 0.9
    """Weight decay for depth loss, multiply depth loss weight by this factor every iterations"""
    save_at_finetune_epoch: List[int] = field(default_factory=list)
    """Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning"""
    use_last_frame_proba: float = 0.2
    """Probability of using the last registered frame for each training iteration"""


@dataclass
class MatchingConfig:
    """Keypoint matching parameters"""
    num_kpts: int = int(4096 * 1.5)
    """Number of keypoints to extract from each image"""
    match_max_error: float = 2e-3
    """Maximum reprojection error for matching keypoints, proportion of the image width"""
    fundmat_samples: int = 2000
    """Maximum number of set of matches used to estimate the fundamental matrix for outlier removal"""
    min_num_inliers: int = 100
    """The keyframe will be added only if the number of inliers is greater than this value"""


@dataclass
class MiniBAConfig:
    """Mini bundle adjustment parameters"""
    num_keyframes_miniba_bootstrap: int = 8
    """Number of first keyframes accumulated for pose and focal estimation before optimization"""
    num_pts_miniba_bootstrap: int = 2000
    """Number of keypoints considered for initial mini bundle adjustment"""
    iters_miniba_bootstrap: int = 200
    """Iterations for initial mini bundle adjustment"""
    num_prev_keyframes_miniba_incr: int = 6
    """Number of previous keyframes for incremental pose initialization"""
    num_prev_keyframes_check: int = 20
    """Number of previous keyframes to check for matches with new keyframe"""
    pnpransac_samples: int = 2000
    """Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal"""
    num_pts_miniba_incr: int = 2000
    """Number of keypoints considered for initial mini bundle adjustment"""
    iters_miniba_incr: int = 20
    """Iterations for incremental mini bundle adjustment"""


@dataclass
class FocalConfig:
    """Focal length estimation parameters"""
    fix_focal: bool = False
    """If set, will use init_focal or init_fov without reoptimizing focal"""
    init_focal: float = -1.0
    """Initial focal length in pixels"""
    init_fov: float = -1.0
    """Initial horizontal FoV in degrees"""


@dataclass
class GaussianConfig:
    """Gaussian initialization parameters"""
    init_proba_scaler: float = 2
    """Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive"""


@dataclass
class AnchorConfig:
    """Anchor management parameters"""
    anchor_overlap: float = 0.3
    """Size of the overlapping regions when blending between anchors"""


@dataclass
class KeyframeConfig:
    """Keyframe management parameters"""
    max_active_keyframes: int = 200
    """Maximum number of keyframes to keep in GPU memory"""


@dataclass
class EvaluationConfig:
    """Evaluation parameters"""
    test_hold: int = -1
    """Holdout for test set, will exclude every test_hold image from the Gaussian optimization"""
    test_frequency: int = -1
    """Test and get metrics every test_frequency keyframes"""
    display_runtimes: bool = False
    """Display runtimes for each step in the tqdm bar"""


@dataclass
class CheckpointConfig:
    """Checkpoint and output parameters"""
    model_path: Path = Path("")
    """Directory to store the renders from test view and checkpoints after training"""
    save_every: int = -1
    """Frequency of exporting renders w.r.t input frames"""


@dataclass
class ViewerConfig:
    """Viewer configuration"""
    viewer_mode: Literal["local", "server", "web", "none"] = "none"
    """Viewer mode"""
    ip: str = "0.0.0.0"
    """IP address of the viewer client, if using server viewer_mode"""
    port: int = 6009
    """Port of the viewer client, if using server viewer_mode"""


@dataclass
class Config:
    """Main configuration for data loading and training"""
    data: DataConfig
    colmap: ColmapConfig
    learning_rates: LearningRateConfig
    training: TrainingConfig
    matching: MatchingConfig
    miniba: MiniBAConfig
    focal: FocalConfig
    gaussian: GaussianConfig
    anchor: AnchorConfig
    keyframes: KeyframeConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    viewer: ViewerConfig
    enable_reboot: bool = False
    """Enable reboot"""


def get_args() -> Config:
    tyro.extras.set_accent_color("red")
    args = tyro.cli(Config)

    # # Set the output directory if not specified
    # if str(args.checkpoint.model_path) == "":
    #     i = 0
    #     while os.path.exists(f"results/{i:06d}"):
    #         i += 1
    #     args.checkpoint.model_path = Path(f"results/{i:06d}")

    return args


if __name__ == "__main__":
    args = get_args()