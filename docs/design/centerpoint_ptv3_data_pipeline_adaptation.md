# CenterPoint → PTv3: Data Pipeline Adaptation

## 1. CenterPoint Pipeline Outputs

**Source:** `projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_amp.py`

### Per-sample pipeline output (after `Pack3DDetInputs`)

`Pack3DDetInputs` produces a dict with two keys:

```python
{
    "inputs": {
        "points": Tensor  # shape (N, 5), dtype float32
                           # columns: x, y, z, intensity, ring_id
    },
    "data_samples": Det3DDataSample  # contains gt_bboxes_3d, gt_labels_3d, metainfo
}
```

### After dataloader collation + `Det3DDataPreprocessor`

The `Det3DDataPreprocessor(voxel=True, voxel_type='hard')` runs **after** collation. It takes the list of per-sample point tensors and produces:

```python
batch_inputs = {
    "points": [Tensor(N1, 5), Tensor(N2, 5), ...],  # list of B tensors, variable N per sample
    "voxels": {
        "voxels":         Tensor(M, 32, 5),  # M total voxels across batch, max 32 pts/voxel, 5 features
        "coors":          Tensor(M, 4),       # (batch_idx, z_idx, y_idx, x_idx) per voxel
        "num_points":     Tensor(M,),         # actual point count per voxel
        "voxel_centers":  Tensor(M, 3),       # (x, y, z) center of each voxel
    }
}
```

**Voxel parameters:**
- `voxel_size = [0.24, 0.24, 8.0]` (pillar-style: flat in z)
- `point_cloud_range = [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]`
- `max_num_points = 32` per voxel, `max_voxels = 96000`
- Grid shape: `1020 × 1020 × 1`

### Train pipeline augmentation steps (in order)

| Step | Transform | Effect |
|------|-----------|--------|
| 1 | `LoadPointsFromFile` | Loads raw `.bin` → `LiDARPoints(N, 5)` |
| 2 | `LoadPointsFromMultiSweeps` | Appends sweep points (sweeps_num=1 → no extra sweeps) |
| 3 | `LoadAnnotations3D` | Loads `gt_bboxes_3d`, `gt_labels_3d` |
| 4 | `RandomFlip3D` | Flips x/y coords + bboxes (p=0.5 horiz, p=0.5 vert) |
| 5 | `GlobalRotScaleTrans` | Rotate [-π/2, π/2], scale [0.8, 1.2], translate σ=[1,1,0.2] — applied to points + bboxes |
| 6 | `PointsRangeFilter` | **Drops** points outside `point_cloud_range` |
| 7 | `ObjectRangeFilter` | Drops bboxes with center outside range |
| 8 | `ObjectNameFilter` | Keeps only target classes |
| 9 | `ObjectMinPointsFilter` | Drops bboxes with <5 interior points |
| 10 | `PointShuffle` | Random permutation of point order |
| 11 | `Pack3DDetInputs` | Packs into `{"inputs": {"points": Tensor(N,5)}, "data_samples": ...}` |

---

## 2. PTv3 Pipeline Outputs

**Source:** `projects/PTv3/configs/semseg-pt-v3m1-0-t4dataset.py`

### Per-sample pipeline output (after `Collect`)

The `Collect` transform produces a **flat dict** (no nesting):

```python
{
    "coord":      Tensor(N', 3),   # float32, x/y/z after GridSample downsampling
    "grid_coord": Tensor(N', 3),   # int32, quantized grid coordinates (floor(coord / grid_size) - min)
    "segment":    Tensor(N',),     # int64, per-point class label (semseg; not used for detection)
    "feat":       Tensor(N', 4),   # float32, cat(coord, strength) = [x, y, z, intensity]
    "offset":     Tensor(1,),      # int64, value = N' (point count for this sample)
}
```

**Key:** `feat` is constructed by the `Collect` kwarg `feat_keys=("coord", "strength")` which does `torch.cat([coord, strength], dim=1)`.

### After dataloader collation (`point_collate_fn`)

The custom `collate_fn` **concatenates** all tensor keys along dim 0 and converts `offset` to **cumulative** counts:

```python
{
    "coord":      Tensor(N_total, 3),   # all samples concatenated
    "grid_coord": Tensor(N_total, 3),   # all samples concatenated
    "segment":    Tensor(N_total,),     # all samples concatenated
    "feat":       Tensor(N_total, 4),   # all samples concatenated
    "offset":     Tensor(B,),           # cumulative: [N'_1, N'_1+N'_2, ..., sum(N'_i)]
}
```

There is **no data_preprocessor** step. This dict goes directly into the model's `forward(data_dict)`.

### Train pipeline augmentation steps (in order)

| Step | Transform | Effect |
|------|-----------|--------|
| 1 | `T4Dataset.get_data()` | Loads `.bin` → numpy `coord(N,3)` + `strength(N,1)` (intensity/255) + `segment(N,)` |
| 2 | `RandomRotate` | Rotate z-axis, angle ∈ [-π, π], p=0.5 — **only coords** (no bboxes) |
| 3 | `RandomScale` | Scale coords by [0.9, 1.1] |
| 4 | `PointClip` | **Clamps** (not drops) coords to `point_cloud_range` |
| 5 | `RandomFlip` | Negates x then y independently (p=0.5 each) |
| 6 | `RandomJitter` | Adds Gaussian noise σ=0.005, clipped ±0.02 |
| 7 | `GridSample` | Hash-based voxel downsampling at `grid_size=0.1`. Keeps one random point per voxel. Outputs `coord`, `strength`, `segment` (subsampled) + `grid_coord` |
| 8 | `SphereCrop` | Caps at 128,000 points (random crop if exceeded) |
| 9 | `ToTensor` | numpy → torch tensors |
| 10 | `Collect` | Selects `coord`, `grid_coord`, `segment`; builds `feat = cat(coord, strength)`; sets `offset = [N']` |

---

## 3. Adaptation: CenterPoint outputs → PTv3-compatible inputs

The CenterPoint pipeline produces mmdet3d-format data. To feed it into a PTv3 backbone, the following transformations are needed. They can be implemented as a **custom `Det3DDataPreprocessor`** replacement (runs after collation, before the model).

### 3a. Replace the data pipeline augmentations

Replace/adapt 3D augmentations so they also produce `grid_coord`. Two options:

```python
train_pipeline = [
    # --- existing CenterPoint steps 1-6 (load, augment, range filter) stay the same ---
    dict(type="LoadPointsFromFile", ...),
    dict(type="LoadPointsFromMultiSweeps", ...),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="RandomFlip3D", ...),
    dict(type="GlobalRotScaleTrans", ...),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=...),
    dict(type="ObjectMinPointsFilter", min_num_points=5),
    # --- NO PointShuffle (GridSample handles randomization) ---
    # --- pack with detection labels as before ---
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
```

Grid sampling and PTv3-format conversion happen in the preprocessor (see 3b).

### 3b. Custom data preprocessor (pseudocode)

Replace `Det3DDataPreprocessor` with a custom one that converts the collated batch into PTv3 `Point` format:

```python
class PTv3DetDataPreprocessor(Det3DDataPreprocessor):
    """Replaces pillar voxelization with PTv3-style grid sampling."""

    def __init__(self, grid_size=0.1, max_points=128000, **kwargs):
        # Do NOT pass voxel=True to parent; we handle voxelization ourselves
        super().__init__(voxel=False, **kwargs)
        self.grid_size = grid_size
        self.max_points = max_points

    def forward(self, data, training=False):
        data = self.collate_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]

        points_list = inputs["points"]  # List[Tensor(Ni, 5)]
        coords, grid_coords, feats, offsets = [], [], [], []

        for pts in points_list:
            coord = pts[:, :3]                # (Ni, 3) x,y,z (discarding intensity/strength)

            # --- GridSample: hash-based voxel downsampling ---
            scaled = coord / self.grid_size
            grid_coord = torch.floor(scaled).int()
            grid_coord -= grid_coord.min(dim=0).values
            # hash → unique → keep one random point per voxel
            idx_unique = grid_hash_subsample(grid_coord)  # returns (N',) indices

            coord = coord[idx_unique]
            grid_coord = grid_coord[idx_unique]

            # --- SphereCrop ---
            if coord.shape[0] > self.max_points:
                perm = torch.randperm(coord.shape[0])[:self.max_points]
                coord, grid_coord = coord[perm], grid_coord[perm]

            # --- Feat only contains x,y,z (= coord)
            feat = coord.clone()

            coords.append(coord)
            grid_coords.append(grid_coord)
            feats.append(feat)
            offsets.append(coord.shape[0])

        # --- Collate into single batch (PTv3 offset-batching) ---
        ptv3_dict = {
            "coord":      torch.cat(coords, dim=0),       # (N_total, 3)
            "grid_coord": torch.cat(grid_coords, dim=0),  # (N_total, 3)
            "feat":       torch.cat(feats, dim=0),         # (N_total, 4)
            "offset":     torch.cumsum(torch.tensor(offsets), dim=0),  # (B,)
        }

        batch_inputs = {
            "points": points_list,  # keep original for any downstream use
            "ptv3":   ptv3_dict,    # new PTv3-format data
        }
        return {"inputs": batch_inputs, "data_samples": data_samples}
```

### 3c. Key differences summary

| Aspect | CenterPoint output | PTv3 expected | What to do |
|--------|-------------------|---------------|------------|
| **Point format** | `Tensor(N, 5)` as `LiDARPoints` | Flat dict with `coord`, `feat`, `grid_coord`, `offset` | Decompose in preprocessor |
| **Intensity normalization** | Raw (0–255 range) | Divided by 255 → [0, 1] | `strength = pts[:, 3:4] / 255` |
| **Features** | All 5 dims passed to `PillarFeatureNet` | `feat = cat(coord, strength)` → 4 channels | Construct explicitly |
| **Voxelization** | Hard pillar voxel `[0.24, 0.24, 8.0]` with up to 32 pts/voxel → dense BEV grid | `GridSample` at `grid_size=0.1` (isotropic 3D), keeps 1 point/voxel → sparse point set | Replace hard voxelization with grid hash sampling |
| **Range handling** | `PointsRangeFilter`: drops out-of-range points | `PointClip`: clamps to range (keeps all points) | Either approach works; CenterPoint's drop is fine |
| **Batching** | List of tensors (one per sample), batch dim via `coors[:, 0]` | Single concatenated tensor + `offset` array (cumulative counts) | Build `offset` from per-sample point counts |
| **grid_coord** | Not produced (pillar grid is implicit in `coors`) | `floor(coord / grid_size) - min_grid` as int tensor | Compute in preprocessor |
| **Point cap** | None (all points kept) | `SphereCrop` at 128k points | Add random subsampling in preprocessor |
