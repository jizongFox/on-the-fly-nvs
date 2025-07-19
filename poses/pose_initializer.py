#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import time
from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Int, Float
from loguru import logger
from torch import Tensor

from args import Config
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.mini_ba import MiniBA
from poses.ransac import RANSACEstimator, EstimatorType
from poses.triangulator import Triangulator
from scene.keyframe import Keyframe
from utils import fov2focal, depth2points, sixD2mtx


@dataclass
class mask_given_index:
    indices: Int[Tensor, "n"] = field(repr=False)
    n_matches: int

    def __post_init__(self):
        self.__mask = torch.zeros(self.n_matches, dtype=torch.bool, device="cuda")
        self.__mask[self.indices] = True

    def __call__(self, new_indices: Int[Tensor, "n"]) -> Bool[Tensor, "n"]:
        return self.__mask[new_indices]


@dataclass
class index_to_local_index:
    from_indices: Int[Tensor, "n1"]
    to_indices: Int[Tensor, "n1"]
    matching_num_kpts: int

    def __post_init__(self):
        self.__hidden_table = torch.empty(
            self.matching_num_kpts, device="cuda", dtype=torch.long
        ).fill_(-1)
        self.__hidden_table[self.from_indices] = self.to_indices

    def __call__(self, new_indices: Int[Tensor, "n2"]) -> Int[Tensor, "n2"]:
        return self.__hidden_table[new_indices]


class PoseInitializer:
    """Fast pose initializer using MiniBA and the previous frames."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        triangulator: Triangulator,
        matcher: Matcher,
        max_pnp_error: float,
        args: Config,
    ):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")
        self.num_pts_miniba_bootstrap = args.miniba.num_pts_miniba_bootstrap
        self.matching_num_kpts = args.matching.num_kpts

        self.num_pts_pnpransac = 2 * args.miniba.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.miniba.num_pts_miniba_incr
        self.min_num_inliers = args.matching.min_num_inliers

        # Initialize the focal length
        if args.focal.init_focal > 0:
            self.f_init = args.focal.init_focal
        elif args.focal.init_fov > 0:
            self.f_init = fov2focal(args.focal.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1,
            args.miniba.num_keyframes_miniba_bootstrap,
            0,
            args.miniba.num_pts_miniba_bootstrap,
            not args.focal.fix_focal,
            True,
            make_cuda_graph=True,
            iters=args.miniba.iters_miniba_bootstrap,
        )
        self.miniba_rebooting = MiniBA(
            1,
            args.miniba.num_keyframes_miniba_bootstrap,
            0,
            args.miniba.num_pts_miniba_bootstrap,
            False,
            True,
            make_cuda_graph=True,
            iters=args.miniba.iters_miniba_bootstrap,
        )
        self.miniBA_incr = MiniBA(
            1,
            1,
            0,
            args.miniba.num_pts_miniba_incr,
            optimize_focal=False,
            optimize_3Dpts=False,
            make_cuda_graph=True,
            iters=args.miniba.iters_miniba_incr,
        )

        self.PnPRANSAC = RANSACEstimator(
            args.miniba.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P
        )

    def build_problem(
        self,
        desc_kpts_list: list[DescribedKeypoints],
        npts: int,
        n_cams: int,
        n_primary_cam: int,
        min_n_matches: int,
        kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras.

        ┌───────────────────────────────────┐
        │         Function Inputs           │
        │ desc_kpts_list, npts, n_cams,     │
        │ n_primary_cam, min_n_matches, etc.│
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │       Initialize Data Arrays      │
        │ uvs[npts, n_cams, 2] = -1         │
        │ xyz_indices[npts, n_cams] = -1    │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │    Loop over Primary Cameras      │
        │        (k = 0...n_primary_cam)    │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │  Count Matches for Each Keypoint  │
        │     in Current Primary Camera     │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ Randomly Sample Keypoints That    │
        │      Have Matches (using          │
        │        torch.multinomial)         │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │  Create Mapping from Global to    │
        │   Local Indices for Selected      │
        │           Keypoints              │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │   Get Current Slice of Output     │
        │        Tensors (uvs_k)           │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ Loop Over All Cameras (l=0..n_cams)│
        └─────────────────┬─────────────────┘
                          ▼
                ┌─────────┴─────────┐
                ▼                   ▼
        ┌───────────────┐   ┌───────────────────┐
        │  Is l == k?   │   │ Find Matches from │
        │ (Same Camera) │   │  Camera k to l    │
        └───────┬───────┘   └─────────┬─────────┘
                │                     │
                ▼                     ▼
        ┌───────────────┐   ┌───────────────────┐
        │ Direct Copy   │   │  Filter Matches   │
        │ of Keypoints  │   │  to Only Selected │
        └───────┬───────┘   │     Keypoints     │
                │           └─────────┬─────────┘
                │                     │
                │                     ▼
                │           ┌───────────────────┐
                │           │ Store Coordinates │
                │           │   and Indices     │
                │           └─────────┬─────────┘
                │                     │
                │                     ▼
                │           ┌───────────────────────┐
                │           │ Setup for Transitive  │
                │           │ Matching Propagation  │
                │           └─────────┬─────────────┘
                │                     │
                │                     ▼
                │           ┌───────────────────────┐
                │           │ Loop Over Remaining   │
                │           │ Cameras (m > l)       │
                │           └─────────┬─────────────┘
                │                     │
                │                     ▼
                │           ┌───────────────────────┐
                │           │ Find and Propagate    │
                │           │ Matches from l to m   │
                │           └─────────┬─────────────┘
                │                     │
                ▼                     ▼
                └─────────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │  Count Valid Observations for     │
        │         Each Point                │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │  Filter Out Points with Fewer     │
        │  Than min_n_matches Valid Views   │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │        Return uvs and             │
        │         xyz_indices               │
        └───────────────────────────────────┘
        """

        # Calculate number of points to allocate per primary camera
        npts_per_primary_cam = npts // n_primary_cam

        # Initialize tensors for storing 2D coordinates (uvs) and keypoint indices (xyz_indices)
        # Shape: [npts, n_cams, 2] for uvs and [npts, n_cams] for xyz_indices
        # Initialize with -1 to indicate missing data points
        uvs = torch.empty(npts, n_cams, 2, device="cuda").fill_(-1)
        xyz_indices = torch.empty(
            npts,
            n_cams,
            dtype=torch.int64,
            device="cuda",
        ).fill_(-1)

        # Track which keypoints have been used to avoid duplicates
        unused_kpts_mask = torch.ones(
            (n_cams, desc_kpts_list[0].n), device="cuda", dtype=torch.bool
        )

        # Process each primary camera
        for k in range(n_primary_cam):
            # Count occurrences of each keypoint in matches for camera k
            idx_occurrences = torch.zeros(
                self.matching_num_kpts, device="cuda", dtype=torch.int
            )
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1

            # Only consider keypoints that haven't been used yet
            idx_occurrences *= unused_kpts_mask[k]

            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue

            # Convert to binary mask: 1 for keypoints with at least one match
            idx_occurrences = idx_occurrences > 0

            # Randomly sample keypoints that have matches (without replacement)
            torch.manual_seed(0)
            selected_indices = torch.multinomial(
                idx_occurrences.float(), npts_per_primary_cam, replacement=False
            )
            print(selected_indices[:5])

            # Create a mask for the selected keypoints
            selected_mask = torch.zeros(
                self.matching_num_kpts, device="cuda", dtype=torch.bool
            )
            selected_mask[selected_indices] = True
            # this selected_mask is for selecting *npts_per_primary_cam* per-camera
            # and is for current camera desc_kpts.

            # Create mapping from global keypoint indices to local array indices (0 to npts_per_primary_cam-1)
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(
                self.matching_num_kpts, device="cuda", dtype=aligned_ids.dtype
            )
            all_aligned_ids[selected_indices] = aligned_ids

            # Get the slice of uvs and xyz_indices for the current primary camera
            uvs_k = uvs[k * npts_per_primary_cam : (k + 1) * npts_per_primary_cam, :, :]
            # shape: [npts_per_primary_cam, n_cams, 2]
            xyz_indices_k = xyz_indices[
                k * npts_per_primary_cam : (k + 1) * npts_per_primary_cam
            ]
            # shape: [npts_per_primary_cam, n_cams]

            # with fixed camera k, Process each camera to establish correspondences
            for l in range(n_cams):
                if l == k:
                    # For the primary camera itself, directly assign the keypoint coordinates and indices
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    # For other cameras, find correspondences based on matches
                    lId = desc_kpts_list[l].frame_id
                    if lId in desc_kpts_list[k].matches:
                        # Get indices of matches between camera k and l
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        # Filter matches to only include selected keypoints
                        mask = selected_mask[idxk]
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        # Map to local array indices and store keypoint coordinates and indices
                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False  # Mark as used
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        # Prepare for propagating matches to other cameras (transitive matching)
                        # This allows finding correspondences across multiple views
                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(
                            self.matching_num_kpts, device="cuda", dtype=torch.bool
                        )
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(
                            self.matching_num_kpts,
                            device="cuda",
                            dtype=aligned_ids.dtype,
                        )
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        # Propagate matches from camera l to remaining cameras
                        for m in range(l + 1, n_cams):
                            # mId = kfId_list[m]
                            mId = desc_kpts_list[m].frame_id

                            if mId in desc_kpts_list[l].matches:
                                # Get indices of matches between camera l and m
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                # Filter to only include selected keypoints from camera l
                                mask = selected_mask_l[idxl]
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                # Find the corresponding local array indices and populate if not already set
                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = (
                                    uvs_k[set_idx, m, 0] == -1
                                )  # Only update unset points
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[
                                    idxm[set_mask]
                                ]

        # Count valid observations for each point (number of cameras where it's visible)
        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)

        # Filter out points that don't have enough valid observations
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1

        return uvs, xyz_indices

    def build_problem_simplified(
        self,
        desc_kpts_list: list[DescribedKeypoints],
        npts: int,
        n_cams: int,
        n_primary_cam: int,
        min_n_matches: int,
        *arsg,
        **kwargs,
    ):
        # Calculate number of points to allocate per primary camera
        npts_per_primary_cam = npts // n_primary_cam

        # Initialize tensors for storing 2D coordinates (uvs) and keypoint indices (xyz_indices)
        # Shape: [npts, n_cams, 2] for uvs and [npts, n_cams] for xyz_indices
        # Initialize with -1 to indicate missing data points
        uvs = torch.empty(npts, n_cams, 2, device="cuda").fill_(-1)
        xyz_indices = torch.empty(
            npts,
            n_cams,
            dtype=torch.int64,
            device="cuda",
        ).fill_(-1)
        assert (
            self.matching_num_kpts == desc_kpts_list[0].n
        ), "Number of keypoints mismatch"

        # Track which keypoints have been used to avoid duplicates
        unused_kpts_mask: Bool[Tensor, "n_cams n_kpts"] = torch.ones(
            (n_cams, self.matching_num_kpts), device="cuda", dtype=torch.bool
        )  # (8, 6144)

        def _process_primary_cam(
            camera_id: int,
            uvs_k: Float[Tensor, "npts_per_primary_cam n_cams 2"],
            xyz_indices_k: Int[Tensor, "npts_per_primary_cam n_cams"],
        ):
            # Count occurrences of each keypoint in matches for camera k
            _idx_occurrences = torch.zeros(
                self.matching_num_kpts, device="cuda", dtype=torch.int
            )
            for _match in desc_kpts_list[camera_id].matches.values():
                _idx_occurrences[_match.idx] += 1
            # sample from _idx_occurrences
            # Only consider keypoints that haven't been used yet
            _idx_occurrences *= unused_kpts_mask[camera_id]

            if _idx_occurrences.sum() == 0:
                logger.warning("No keypoints found in camera {}".format(camera_id))
                return
            # Randomly sample keypoints that have matches (without replacement)
            torch.manual_seed(0)
            _idx_occurrences = _idx_occurrences > 0

            _selected_indices: Int[Tensor, "npts_per_primary_cam"] = torch.multinomial(
                _idx_occurrences.float(), npts_per_primary_cam, replacement=False
            )

            selected_index_to_local = index_to_local_index(
                _selected_indices,
                torch.arange(len(_selected_indices), device="cuda"),
                self.matching_num_kpts,
            )

            select_mask_cls = mask_given_index(
                indices=_selected_indices, n_matches=self.matching_num_kpts
            )

            # with fixed camera k, Process each camera to establish correspondences
            for other_camera_id in range(n_cams):
                if other_camera_id == camera_id:
                    # For the primary camera itself, directly assign the keypoint coordinates and indices
                    uvs_k[:, camera_id, :] = desc_kpts_list[other_camera_id].kpts[
                        _selected_indices
                    ]
                    xyz_indices_k[:, camera_id] = _selected_indices
                else:
                    # For other cameras, find correspondences based on matches
                    lId = desc_kpts_list[other_camera_id].frame_id
                    if lId in desc_kpts_list[camera_id].matches:
                        # Get indices of matches between camera k and l
                        idx_primary = desc_kpts_list[camera_id].matches[lId].idx
                        idx_other = desc_kpts_list[camera_id].matches[lId].idx_other

                        # Filter matches to only include selected keypoints
                        mask = select_mask_cls(idx_primary)
                        idx_primary = idx_primary[mask]
                        idx_other = idx_other[mask]
                        # del mask

                        # Map to local array indices and store keypoint coordinates and indices
                        set_idx_copy = set_idx = selected_index_to_local(idx_primary)
                        unused_kpts_mask[other_camera_id, idx_other] = (
                            False  # Mark as used
                        )
                        uvs_k[set_idx, other_camera_id, :] = desc_kpts_list[
                            other_camera_id
                        ].kpts[idx_other]
                        xyz_indices_k[set_idx, other_camera_id] = idx_other

                        # Prepare for propagating matches to other cameras (transitive matching)
                        # This allows finding correspondences across multiple views
                        selected_indices_l = idx_other.clone()
                        _select_mask_l_cls = mask_given_index(
                            indices=selected_indices_l, n_matches=self.matching_num_kpts
                        )

                        # selected_mask_l = torch.zeros(
                        #     self.matching_num_kpts, device="cuda", dtype=torch.bool
                        # )
                        # selected_mask_l[selected_indices_l] = True

                        # Propagate matches from camera l to remaining cameras
                        for m in range(other_camera_id + 1, n_cams):
                            mId = desc_kpts_list[m].frame_id
                            if mId in desc_kpts_list[other_camera_id].matches:
                                # Get indices of matches between camera l and m
                                idx_other = (
                                    desc_kpts_list[other_camera_id].matches[mId].idx
                                )
                                idx_next = (
                                    desc_kpts_list[other_camera_id]
                                    .matches[mId]
                                    .idx_other
                                )

                                # Filter to only include selected keypoints from camera l
                                mask = _select_mask_l_cls(idx_other)
                                idx_other = idx_other[mask]
                                idx_next = idx_next[mask]

                                # Find the corresponding local array indices and populate if not already set
                                set_idx = index_to_local_index(
                                    selected_indices_l,
                                    set_idx_copy,
                                    self.matching_num_kpts,
                                )(idx_other)

                                set_mask = (
                                    uvs_k[set_idx, m, 0] == -1
                                )  # Only update unset points
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[
                                    idx_next[set_mask]
                                ]

        for i in range(n_primary_cam):
            _process_primary_cam(
                i,
                uvs_k=uvs[
                    i * npts_per_primary_cam : (i + 1) * npts_per_primary_cam, :, :
                ],
                xyz_indices_k=xyz_indices[
                    i * npts_per_primary_cam : (i + 1) * npts_per_primary_cam, :
                ],
            )
        # Count valid observations for each point (number of cameras where it's visible)
        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)

        # Filter out points that don't have enough valid observations
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(
        self, desc_kpts_list: list[DescribedKeypoints], rebooting=False
    ):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list.
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(
                    desc_kpts_list[i],
                    desc_kpts_list[j],
                    remove_outliers=True,
                    update_kpts_flag="inliers",
                )

        uvs, xyz_indices = self.build_problem_simplified(
            desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams))
        )

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        f_init = torch.tensor([self.f_init], device="cuda")
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(
                Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1)
            )
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(
                Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1)
            )
        final_residual = (r * mask).abs().sum() / mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
    ):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """

        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                # kID=index,
                # kID_other=keyframe.index,
            )

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(
                confs, self.num_pts_miniba_incr, replacement=False
            )
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        Rt, inliers = self.PnPRANSAC(
            uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs
        )

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz[..., 0]),
                self.num_pts_miniba_incr,
                dim=0,
                largest=False,
            )[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat(
                [
                    xyz,
                    torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda"),
                ],
                dim=0,
            )
            uvs_ba = torch.cat(
                [
                    uvs,
                    -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda"),
                ],
                dim=0,
            )

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(
            Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1)
        )
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            print("Too few inliers for pose initialization")
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None
