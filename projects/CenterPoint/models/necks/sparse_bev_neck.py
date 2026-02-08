"""Sparse BEV Neck — channel projection + sparse-to-dense BEV conversion.

Projects the high-dimensional up_cast features from the Concerto backbone
to the target channel count, then scatters them into a dense BEV tensor
with optional 2D conv refinement.

See docs/design/concerto_centerpoint_adaptation.md §Step B.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class SparseBEVNeck(nn.Module):
    """Convert sparse 3D point features into a dense BEV feature map.

    Performs the following steps in a single efficient pass:

    1. **Channel projection** — Linear + BN + ReLU maps the backbone's
       high-dimensional features (e.g. 1088) to the target dimension
       (e.g. 384).
    2. **BEV quantization** — Maps real-world XY coordinates into
       the target BEV grid using ``floor((xy - range_min) / cell_size)``.
       This implicitly collapses the Z dimension because all points
       sharing the same (batch, bev_i, bev_j) are pooled together.
    3. **Scatter-max densification** — Scatters projected features into a
       dense ``(B, C, H, W)`` tensor via ``scatter_reduce(amax)``.
    4. **2D conv refinement** — A small stack of Conv2d + BN + ReLU
       provides horizontal spatial context and fills gaps.

    Args:
        in_channels (int): Input feature dimension from the backbone.
        out_channels (int): Output dense BEV channel dimension.
        bev_h (int): Height of the target BEV grid.
        bev_w (int): Width of the target BEV grid.
        point_cloud_range (Sequence[float]): ``[x_min, y_min, z_min,
            x_max, y_max, z_max]``.
        target_cell_size (float): BEV cell size in metres
            (= ``voxel_size[0] * out_size_factor``).
        num_convs (int): Number of 2D refinement conv layers (0 to skip).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 384,
        bev_h: int = 510,
        bev_w: int = 510,
        point_cloud_range: Sequence[float] = (
            -122.4,
            -122.4,
            -3.0,
            122.4,
            122.4,
            5.0,
        ),
        target_cell_size: float = 0.48,
        num_convs: int = 3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.target_cell_size = target_cell_size
        self.register_buffer(
            "range_min",
            torch.tensor(
                [point_cloud_range[0], point_cloud_range[1]],
                dtype=torch.float32,
            ),
        )

        # Channel projection: in_channels → out_channels
        self.channel_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Optional 2D conv refinement on the dense BEV map
        convs: List[nn.Module] = []
        for _ in range(num_convs):
            convs.extend(
                [
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.refine = nn.Sequential(*convs) if convs else nn.Identity()

    # ------------------------------------------------------------------

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Tensor,
        batch_size: Optional[int] = None,
    ) -> List[Tensor]:
        """Convert sparse point features to a dense BEV feature map.

        Args:
            feat: ``(N, C_in)`` float — point features from backbone.
            coord: ``(N, 3)`` float — real-world XYZ coordinates.
            batch: ``(N,)`` int — per-point batch index.
            batch_size: Explicit batch size. Inferred if ``None``.

        Returns:
            Single-element list ``[(B, C_out, H, W)]`` for
            ``CenterHead.forward(feats)``.
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        # 1. Channel projection  (N, in_ch) → (N, out_ch)
        proj = self.channel_proj(feat)

        # 2. Quantize XY → target BEV grid indices
        xy = coord[:, :2]  # (N, 2)
        range_min: Tensor = self.range_min  # type: ignore[assignment]
        bev_ij = ((xy - range_min) / self.target_cell_size).long()
        bev_ij[:, 0].clamp_(0, self.bev_h - 1)
        bev_ij[:, 1].clamp_(0, self.bev_w - 1)

        # 3. Scatter-max into dense (B*H*W, C) — implicitly pools Z
        linear_idx = (
            batch.long() * (self.bev_h * self.bev_w)
            + bev_ij[:, 0] * self.bev_w
            + bev_ij[:, 1]
        )  # (N,)

        total_cells = batch_size * self.bev_h * self.bev_w
        dense_flat = torch.zeros(
            total_cells,
            self.out_channels,
            dtype=proj.dtype,
            device=proj.device,
        )
        idx_expand = linear_idx.unsqueeze(1).expand_as(proj)
        dense_flat.scatter_reduce_(
            0,
            idx_expand,
            proj,
            reduce="amax",
            include_self=False,
        )

        # Reshape → (B, C, H, W)
        dense_bev = (
            dense_flat.view(batch_size, self.bev_h, self.bev_w, self.out_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # 4. 2D conv refinement
        dense_bev = self.refine(dense_bev)

        return [dense_bev]
