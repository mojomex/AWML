"""ConcertoCenterPoint — CenterPoint with ConcertoLargeOutdoorXYZ backbone.

Replaces the traditional PillarFeatureNet → PointPillarsScatter → SECOND →
SECONDFPN pipeline with:

    PTv3DetDataPreprocessor → ConcertoLargeOutdoorXYZ (up_cast) → SparseBEVNeck → CenterHead

See docs/design/concerto_centerpoint_adaptation.md for the full design.
"""

from typing import Dict, List, Optional, Tuple

import torch
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from torch import Tensor

from projects.Concerto.concerto.structure import Point


@MODELS.register_module()
class ConcertoCenterPoint(Base3DDetector):
    """CenterPoint detector using a Concerto (PTv3) backbone.

    Unlike the standard :class:`CenterPoint` which inherits
    ``MVXTwoStageDetector`` and relies on ``pts_voxel_encoder``,
    ``pts_middle_encoder``, ``pts_backbone``, and ``pts_neck``, this
    detector wires the Concerto encoder-only backbone (with ``up_cast``)
    directly into a :class:`SparseBEVNeck` and the standard
    :class:`CenterHead`.

    The data preprocessor (:class:`PTv3DetDataPreprocessor`) produces
    the PTv3 offset-batched dict consumed by ``Point(data_dict)`` inside
    the Concerto backbone.

    Args:
        data_preprocessor (dict): Config for :class:`PTv3DetDataPreprocessor`.
        backbone (dict): Config for the Concerto backbone
            (``ConcertoLargeOutdoorXYZ``).
        neck (dict): Config for :class:`SparseBEVNeck`.
        bbox_head (dict): Config for :class:`CenterHead`.
        train_cfg (dict, optional): Training config forwarded to the head.
        test_cfg (dict, optional): Testing config forwarded to the head.
        freeze_backbone (bool): If ``True``, backbone parameters are
            frozen. Defaults to ``True``.
    """

    def __init__(
        self,
        data_preprocessor: dict,
        backbone: dict,
        neck: dict,
        bbox_head: dict,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)

        # ---------- build sub-modules via MODELS registry ----------
        self.backbone = self._build_backbone(backbone)
        self.neck = MODELS.build(neck)

        # Inject train_cfg / test_cfg into the head config
        if train_cfg is not None:
            bbox_head["train_cfg"] = train_cfg.get("pts", train_cfg)
        if test_cfg is not None:
            bbox_head["test_cfg"] = test_cfg.get("pts", test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Backbone construction (from Concerto's own registry)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_backbone(cfg: dict) -> torch.nn.Module:
        """Build the Concerto backbone.

        The Concerto model instances are registered in Concerto's *local*
        ``Registry``, not in mmdet3d's ``MODELS``.  We import and
        instantiate them directly.
        """
        from projects.Concerto.concerto.model_instances import (
            ConcertoLargeOutdoorXYZ,
        )

        # Remove the ``type`` key so we can pass the rest as kwargs.
        cfg = dict(cfg)  # shallow copy
        model_type = cfg.pop("type", "ConcertoLargeOutdoorXYZ")
        assert model_type == "ConcertoLargeOutdoorXYZ", (
            f"Only ConcertoLargeOutdoorXYZ is supported, got {model_type}"
        )
        return ConcertoLargeOutdoorXYZ(**cfg)

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_feat(
        self,
        batch_inputs: Dict,
    ) -> List[Tensor]:
        """Run the Concerto backbone + SparseBEVNeck.

        Args:
            batch_inputs: Output of :meth:`data_preprocessor.forward`.
                Must contain ``"ptv3"`` key with the offset-batched dict.

        Returns:
            ``List[Tensor]`` — BEV feature maps for the CenterHead.
        """
        ptv3_dict = batch_inputs["ptv3"]

        # Run Concerto backbone
        if self.freeze_backbone:
            with torch.no_grad():
                point: Point = self.backbone(ptv3_dict)
        else:
            point: Point = self.backbone(ptv3_dict)

        # Extract the offset-batched tensors from the Point dict
        feat = point.feat  # (N, C) e.g. (N, 1536)
        coord = point.coord  # (N, 3)

        # Compute per-point batch index from offset
        offset = point.offset  # (B,) cumulative
        batch_idx = self._offset_to_batch(offset)  # (N,)
        batch_size = offset.shape[0]

        # SparseBEVNeck: sparse 3D → dense 2D BEV
        bev_feats = self.neck(
            feat=feat,
            coord=coord,
            batch=batch_idx,
            batch_size=batch_size,
        )
        return bev_feats

    @staticmethod
    def _offset_to_batch(offset: Tensor) -> Tensor:
        """Convert cumulative offset to per-point batch index.

        Args:
            offset: ``(B,)`` long — cumulative point counts,
                e.g. ``[N0, N0+N1, N0+N1+N2]``.

        Returns:
            ``(N_total,)`` int — batch index per point.
        """
        counts = torch.diff(
            offset, prepend=torch.zeros(1, dtype=offset.dtype, device=offset.device)
        )
        return torch.repeat_interleave(
            torch.arange(counts.shape[0], device=offset.device),
            counts,
        )

    # ------------------------------------------------------------------
    # train / val / test entry points
    # ------------------------------------------------------------------

    # forward() is inherited from Base3DDetector and dispatches to
    # loss / predict / _forward / aug_test based on ``mode``.

    def loss(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Compute training losses.

        Args:
            batch_inputs_dict: Pre-processed batch inputs (from
                ``data_preprocessor``).
            batch_data_samples: Ground-truth annotations.

        Returns:
            Dict of loss tensors.
        """
        bev_feats = self.extract_feat(batch_inputs_dict)
        losses = self.bbox_head.loss(bev_feats, batch_data_samples, **kwargs)
        return losses

    def predict(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        """Run inference and return predictions.

        Args:
            batch_inputs_dict: Pre-processed batch inputs.
            batch_data_samples: Meta-info / annotations.

        Returns:
            List of ``Det3DDataSample`` with predicted bboxes.
        """
        bev_feats = self.extract_feat(batch_inputs_dict)
        results_list_3d = self.bbox_head.predict(
            bev_feats, batch_data_samples, **kwargs
        )
        return self.add_pred_to_datasample(batch_data_samples, results_list_3d)

    def _forward(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: Optional[List[Det3DDataSample]] = None,
        **kwargs,
    ) -> Tuple[List[Tensor]]:
        """Raw feature extraction (tensor mode)."""
        bev_feats = self.extract_feat(batch_inputs_dict)
        return (bev_feats,)

    def aug_test(
        self,
        batch_inputs_dict: List[Dict],
        batch_data_samples: List[List[Det3DDataSample]],
        **kwargs,
    ) -> List[Det3DDataSample]:
        """Test-time augmentation — not implemented.

        Raises:
            NotImplementedError: Always, TTA is not supported.
        """
        raise NotImplementedError(
            "Test-time augmentation is not supported for ConcertoCenterPoint."
        )

    # ------------------------------------------------------------------
    # Backbone train/eval mode management
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
