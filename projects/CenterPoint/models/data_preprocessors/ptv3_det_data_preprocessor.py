"""PTv3-compatible data preprocessor for 3D detection.

Replaces mmdet3d's pillar-based hard voxelization with PTv3-style grid hash
sampling, producing the ``coord`` / ``grid_coord`` / ``feat`` / ``offset``
dict that the PTv3/Concerto ``Point`` structure expects.

See docs/design/centerpoint_ptv3_data_pipeline_adaptation.md for details.
"""

from typing import Dict, List

import torch
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS
from torch import Tensor

from projects.Concerto.concerto.transform import GridSample, SphereCrop


@MODELS.register_module()
class PTv3DetDataPreprocessor(Det3DDataPreprocessor):
    """Data preprocessor that converts mmdet3d point clouds into PTv3 format.

    Instead of the standard pillar/voxel encoding used by CenterPoint, this
    preprocessor performs PTv3-style grid-hash downsampling and produces the
    offset-batched dict consumed by ``Point(data_dict)`` in PTv3/Concerto.

    Grid sampling and point capping are delegated to Concerto's
    :class:`~projects.Concerto.concerto.transform.GridSample` and
    :class:`~projects.Concerto.concerto.transform.SphereCrop`, keeping
    the hashing logic in a single place.

    The output ``batch_inputs`` dict contains:

    - ``"points"``: original per-sample point tensors (kept for downstream use)
    - ``"ptv3"``: dict with keys ``coord``, ``grid_coord``, ``feat``,
      ``offset`` ready for ``Point()``.

    Args:
        grid_size (float): Isotropic voxel size for grid sampling.
            One random point is kept per voxel cell. Defaults to 0.05.
        max_points (int): Maximum number of points to keep per sample
            after grid sampling (SphereCrop). If a sample exceeds this
            limit, a random subset is drawn. Defaults to 128000.
        **kwargs: Forwarded to ``Det3DDataPreprocessor.__init__`` with
            ``voxel=False`` forced.
    """

    def __init__(
        self,
        grid_size: float = 0.05,
        max_points: int = 128_000,
        **kwargs,
    ) -> None:
        # Disable the parent's built-in hard voxelization – we do our own.
        kwargs.pop("voxel", None)
        kwargs.pop("voxel_layer", None)
        super().__init__(voxel=False, **kwargs)
        self.grid_size = grid_size
        self.max_points = max_points

        # Reuse Concerto's GridSample (numpy-based, FNV64-1A hash).
        self._grid_sample = GridSample(
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        )
        self._sphere_crop = SphereCrop(
            point_max=max_points,
            mode="random",
        )

    # ------------------------------------------------------------------
    # simple_process – override parent to build PTv3 dict instead of
    #                  hard-voxelizing
    # ------------------------------------------------------------------

    def simple_process(
        self,
        data: Dict,
        training: bool = False,
    ) -> Dict:
        """Convert collated mmdet3d data into PTv3 offset-batched format.

        This overrides
        :meth:`Det3DDataPreprocessor.simple_process` and replaces the
        pillar/voxel encoding path with PTv3-style grid-hash downsampling.

        Pipeline:

        1. Collate and move data to device (parent helper).
        2. For each sample's point tensor ``(N_i, D)``:
           a. Extract xyz coordinates as a numpy dict.
           b. Delegate to Concerto's ``GridSample`` → subsampled
              ``coord`` + ``grid_coord``.
           c. Delegate to Concerto's ``SphereCrop`` → cap at
              ``max_points``.
           d. Convert back to torch and build ``feat = coord``.
        3. Concatenate across samples into a single offset-batched dict.

        Args:
            data: Collated batch dict from the dataloader with keys
                ``"inputs"`` and ``"data_samples"``.
            training: Whether in training mode. Defaults to False.

        Returns:
            Dict with ``"inputs"`` and ``"data_samples"`` keys.
            ``inputs["ptv3"]`` contains the PTv3-format dict.
        """
        data = self.collate_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]

        points_list: List[Tensor] = inputs["points"]  # List[Tensor(Ni, D)]

        coords: List[Tensor] = []
        grid_coords: List[Tensor] = []
        feats: List[Tensor] = []
        offsets: List[int] = []

        for pts in points_list:
            device = pts.device
            coord_np = pts[:, :3].cpu().numpy()  # (N, 3) float32

            # Delegate to Concerto's GridSample (numpy in, numpy out).
            # Set index_valid_keys so index_operator only subsamples "coord".
            sample_dict = {
                "coord": coord_np,
                "index_valid_keys": ["coord"],
            }
            sample_dict = self._grid_sample(sample_dict)

            # Delegate to Concerto's SphereCrop
            sample_dict = self._sphere_crop(sample_dict)

            coord = torch.from_numpy(sample_dict["coord"]).to(device)
            grid_coord = torch.from_numpy(sample_dict["grid_coord"]).int().to(device)
            feat = coord.clone()  # 3-channel features (x, y, z)

            coords.append(coord)
            grid_coords.append(grid_coord)
            feats.append(feat)
            offsets.append(coord.shape[0])

        # Build offset-batched PTv3 dict
        ptv3_dict: Dict[str, Tensor] = {
            "coord": torch.cat(coords, dim=0),  # (N_total, 3)
            "grid_coord": torch.cat(grid_coords, dim=0),  # (N_total, 3)
            "feat": torch.cat(feats, dim=0),  # (N_total, 3)
            "offset": torch.cumsum(
                torch.tensor(offsets, dtype=torch.long, device=coords[0].device),
                dim=0,
            ),  # (B,)
        }

        batch_inputs: Dict = {
            "points": points_list,  # keep originals for any downstream use
            "ptv3": ptv3_dict,
        }
        return {"inputs": batch_inputs, "data_samples": data_samples}
