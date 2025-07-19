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

import typing
import typing as t

import torch
from jaxtyping import Float, Int
from torch import Tensor

from poses.ransac import EstimatorType, RANSACEstimator

if typing.TYPE_CHECKING:
    from poses.feature_detector import DescribedKeypoints

from dataclasses import dataclass, KW_ONLY
import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class Matches:
    """
    Stores matched keypoints and their indices between two images.

    Shapes
    -------
    kpts, kpts_other : (N, 2) float32/float64
        Pixel coordinates of the *selected* key-points in the reference and the other image.
    idx, idx_other : (N,) int64
        Indices that refer back to the *full* key-point array of each image.
    """

    kpts: Float[Tensor, "n 2"]
    kpts_other: Float[Tensor, "n 2"]
    idx: Int[Tensor, "n"]
    idx_other: Int[Tensor, "n"]
    _: KW_ONLY
    frame_id1: int
    frame_id2: int

    # -----------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------
    def __post_init__(self):
        """Light-weight sanity checks executed without grad."""
        n = self.kpts.shape[0]
        assert (
            n
            == self.kpts_other.shape[0]
            == self.idx.shape[0]
            == self.idx_other.shape[0]
        ), (
            "All fields in `Matches` must reference the same number of correspondences "
            f"(got {n}, {self.kpts_other.shape[0]}, {self.idx.shape[0]}, {self.idx_other.shape[0]})."
        )
        # basic shape check for key-points
        assert (
            self.kpts.shape[1] == 2 and self.kpts_other.shape[1] == 2
        ), "Key-points must be 2-D coordinates."

    @property
    def n(self) -> int:
        """Return the number of correspondences."""
        return int(self.kpts.shape[0])

    @property
    def device(self):
        """Torch device of the stored tensors (assumes all tensors on same device)."""
        return self.kpts.device

    # -----------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------
    def to(self, device: torch.device):
        """In-place move of all tensors to *device* and return self for chaining."""
        self.kpts = self.kpts.to(device)
        self.kpts_other = self.kpts_other.to(device)
        self.idx = self.idx.to(device)
        self.idx_other = self.idx_other.to(device)
        return self

    def cpu(self):
        """Shorthand for ``self.to('cpu')``."""
        return self.to(torch.device("cpu"))

    def cuda(self):
        """Shorthand for ``self.to('cuda')`` (only if available)."""
        return self.to(torch.device("cuda"))

    def swapped(self) -> "Matches":
        """Return a *new* ``Matches`` instance with *reference* and *other* swapped.

        Helpful when you already have a set of matches computed from *image A → image B*
        and you need the reverse *image B → image A* mapping.
        """
        return Matches(
            self.kpts_other,
            self.kpts,
            self.idx_other,
            self.idx,
            frame_id1=self.frame_id2,
            frame_id2=self.frame_id1,
        )

    def __repr__(self):
        return (
            f"Matches(frame_id1={self.frame_id1}, frame_id2={self.frame_id2}, n={self.n}, "
            f"kpts={tuple(self.kpts.shape)}, idx={tuple(self.idx.shape)}, "
            f"idx_other={tuple(self.idx_other.shape)}, kpts_other={tuple(self.kpts_other.shape)})"
        )


def plot_matches(
    img1: Float[Tensor, "3 h w"],
    img2: Float[Tensor, "3 h w"],
    matches: Matches,
    *,
    orientation: t.Literal["horizontal", "vertical"] = "vertical",
    figsize=(6, 12),
    linewidth: float = 0.5,
    kp_size: int = 20,
) -> plt.Figure:
    """Visualize keypoint correspondences.

    Two images are shown side-by-side with lines connecting matched
    keypoints. Green lines indicate correspondences, while red / cyan dots
    highlight the positions in each view.

    Args:
        img1: First image as a tensor in **CHW** format with values in [0,1].
        img2: Second image in the same format / range as *img1*.
        matches: A :class:`~ontheflynvs_dc.protocols.Matches` instance
            produced by :class:`~ontheflynvs_dc.matcher.Matcher`.
        orientation: Orientation of the concatenated images.
        figsize: Size of the matplotlib figure.
        linewidth: Width of the correspondence lines.
        kp_size: Scatter size for keypoints.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure so callers can further tweak or save it.
    """
    # Ensure everything lives on CPU and is NumPy.
    img1_np = img1.detach().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.detach().cpu().permute(1, 2, 0).numpy()

    # Prepare concatenated canvas according to orientation
    if orientation == "horizontal":
        canvas = np.concatenate([img1_np, img2_np], axis=1)
        h1, w1, _ = img1_np.shape
        offset_x, offset_y = w1, 0
    else:
        canvas = np.concatenate([img1_np, img2_np], axis=0)
        h1, w1, _ = img1_np.shape
        offset_x, offset_y = 0, h1

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas)
    ax.axis("off")

    kpts1 = matches.kpts.detach().cpu().numpy()
    kpts2 = matches.kpts_other.detach().cpu().numpy()

    # Draw correspondences
    for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
        ax.plot(
            [x1 + offset_x * 0, x2 + offset_x],
            [y1 + offset_y * 0, y2 + offset_y],
            color="lime",
            linewidth=linewidth,
        )

    ax.scatter(kpts1[:, 0], kpts1[:, 1] + 0, s=kp_size, c="red")
    ax.scatter(kpts2[:, 0] + offset_x, kpts2[:, 1] + offset_y, s=kp_size, c="cyan")

    return fig


# Adapted from https://github.com/verlab/accelerated_features
def match(feats1, feats2, min_cossim=0.82):
    cossim = feats1 @ feats2.t()

    bestcossim, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mask = match21[match12] == idx0

    if min_cossim > 0:
        mask *= bestcossim > min_cossim

    return idx0, match12, mask


class Matcher:

    def __str__(self):
        return f"Matcher(max_error={self.max_error}, fundmat_samples={self.fundmat_estimator.__class__.__name__}, samples={self.fundmat_samples})"

    def __repr__(self):
        return self.__str__()

    @torch.no_grad()
    def __init__(self, fundmat_samples: int, max_error: float):
        """
        Initialize the Matcher.
        Args:
            fundmat_samples (int): Number of RANSAC etimations when estimating inliers with fundamental matrix estimation.
            max_error (float): Maximum error for RANSAC inlier threshold.
        """
        self.max_error = max_error
        self.fundmat_samples = fundmat_samples
        self.fundmat_estimator = RANSACEstimator(
            fundmat_samples, max_error, EstimatorType.FUNDAMENTAL_8PTS
        )

    def evaluate_match(
        self, desc_kpts: "DescribedKeypoints", desc_kpts_other: "DescribedKeypoints"
    ):
        """
        Get the number of matches between two sets of described keypoints.
        """
        _, _, mask = match(desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda())
        return mask.sum()

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: "DescribedKeypoints",
        desc_kpts_other: "DescribedKeypoints",
        *,
        remove_outliers: bool = False,
        update_kpts_flag: t.Literal["all", "inliers", ""] = "",
    ) -> Matches:
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        idx, idx_other, mask = match(
            desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda()
        )
        idx = idx[mask]
        idx_other = idx_other[mask]
        kpts = desc_kpts.kpts[idx]
        kpts_other = desc_kpts_other.kpts[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other

        if remove_outliers:
            F, mask = self.fundmat_estimator(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]

        assert update_kpts_flag in ["all", "inliers", ""]
        if update_kpts_flag == "all":
            _match = Matches(
                kpts_all,
                kpts_other_all,
                idx_all,
                idx_other_all,
                frame_id1=desc_kpts.frame_id,
                frame_id2=desc_kpts_other.frame_id,
            )
            desc_kpts.update_matches(desc_kpts_other.frame_id, _match)
            desc_kpts_other.update_matches(desc_kpts.frame_id, _match.swapped())
        elif update_kpts_flag == "inliers":
            _match = Matches(
                kpts,
                kpts_other,
                idx,
                idx_other,
                frame_id1=desc_kpts.frame_id,
                frame_id2=desc_kpts_other.frame_id,
            )
            desc_kpts.update_matches(desc_kpts_other.frame_id, _match)
            desc_kpts_other.update_matches(desc_kpts.frame_id, _match.swapped())
        else:
            pass  # do nothing.

        return Matches(
            kpts,
            kpts_other,
            idx,
            idx_other,
            frame_id1=desc_kpts.frame_id,
            frame_id2=desc_kpts_other.frame_id,
        )
