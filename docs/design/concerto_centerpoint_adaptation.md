# ConcertoLargeOutdoorXYZ → CenterHead Adaptation Plan

## Overview

Replace the traditional **PillarFeatureNet → PointPillarsScatter → SECOND → SECONDFPN** pipeline in CenterPoint with the **ConcertoLargeOutdoorXYZ** (encoder-only PTv3) backbone using its built-in `up_cast` mechanism for multi-scale feature fusion, plus lightweight sparse-to-dense BEV conversion modules.

The `up_cast` operation (implemented in `PointTransformerV3`) traverses the `pooling_parent`/`pooling_inverse` chain to cast coarse encoder features back to finer resolutions via concatenation. With `up_cast_level=2`, the backbone directly outputs a single `Point` at enc2 resolution (0.20 m) with 1088-dimensional features fusing information from stages 2, 3, and 4 — **eliminating the need for any external multi-scale extraction or fusion logic**.

---

## 1. Interface Analysis

### 1.1 ConcertoLargeOutdoorXYZ Output (with `up_cast_level=2`)

The backbone is an **encoder-only** PTv3 (`enc_mode=True`) with 5 encoder stages:

| Stage | Channels | Stride (cumulative) | Effective grid size¹ |
|-------|----------|---------------------|----------------------|
| enc0  | 48       | 1×                  | 0.05 m               |
| enc1  | 96       | 2×                  | 0.10 m               |
| enc2  | 192      | 4×                  | 0.20 m               |
| enc3  | 384      | 8×                  | 0.40 m               |
| enc4  | 512      | 16×                 | 0.80 m               |

¹ Assuming `grid_size = 0.05` in `PTv3DetDataPreprocessor`.

With `up_cast_level=2`, the `up_cast` operation broadcasts enc4 features back through enc3 and enc2 via concatenation:

| Iteration | Start point    | Parent       | Result `feat` dim        | Result resolution |
|-----------|---------------|--------------|--------------------------|-------------------|
| 0         | enc4 (512-d)  | enc3 (384-d) | `cat(384, 512) = 896`   | enc3 grid (0.40 m)|
| 1         | merged (896-d)| enc2 (192-d) | `cat(192, 896) = 1088`  | enc2 grid (0.20 m)|

The final output is a `Point` dict (addict `Dict`) with:

| Field              | Shape              | Description                                          |
|--------------------|--------------------|------------------------------------------------------|
| `feat`             | `(N_enc2, 1088)`   | Multi-scale fused features (enc2 + enc3 + enc4)      |
| `coord`            | `(N_enc2, 3)`      | XYZ coordinates at enc2 resolution                   |
| `grid_coord`       | `(N_enc2, 3)`      | Integer 3D grid coordinates at enc2 resolution       |
| `batch`            | `(N_enc2,)`        | Batch index per point                                |
| `offset`           | `(B,)`             | Cumulative point counts per sample                   |
| `sparse_conv_feat` | `SparseConvTensor`  | Sparse 3D tensor wrapping `feat`                    |
| `grid_size`        | scalar             | Effective grid cell size = `0.05 × 2² = 0.20 m`     |

The output is **sparse, unordered, 3D point data** at 0.20 m resolution — already multi-scale fused, not a dense BEV grid.

### 1.2 CenterHead Input

`CenterHead.forward(feats: List[Tensor])` calls `forward_single(x)` per feature level:

| Field | Shape             | Description              |
|-------|-------------------|--------------------------|
| `x`   | `(B, C, H, W)`   | Dense 2D BEV feature map  |

In the current config: `C = 384` (sum of `[128, 128, 128]`), spatial size `(H, W) = (510, 510)` = `grid_size / out_size_factor = 1020 / 2`.

The head applies a `shared_conv` (3×3 Conv2d), then per-task `SeparateHead` branches that output heatmaps, regression targets, etc. — all as dense 2D convolutions.

### 1.3 The Gap

|                 | ConcertoLargeOutdoorXYZ output (up_cast=2) | CenterHead input                 |
|-----------------|---------------------------------------------|----------------------------------|
| **Format**      | Sparse point set `(N, C)` + batch index     | Dense BEV tensor `(B, C, H, W)` |
| **Spatial**     | 3D, arbitrary positions                     | 2D BEV grid, fixed resolution   |
| **Channels**    | 1088                                        | 384 (configurable)               |
| **Resolution**  | ~0.20 m effective spacing                   | 0.48 m BEV cell (0.24 m voxel × out_size_factor 2) |

The gap is now purely **format conversion** (sparse 3D → dense 2D BEV) and **channel projection** (1088 → target). Multi-scale fusion is already handled inside the backbone by `up_cast`.

---

## 2. Adaptation Steps

### Step A — Sparse 3D → Sparse BEV Pillar Pooling

**Module**: `SparseBEVScatter` (new)  
**Purpose**: Flatten Z dimension by pooling along the vertical axis per BEV cell, producing a sparse 2D BEV representation.

For the backbone's single-scale sparse `Point` output:

1. Compute 2D BEV grid coordinates: `bev_coord = grid_coord[:, :2]` (drop Z).
2. Combine with batch index to get unique BEV cells: `key = bev_coord | (batch << 48)`.
3. Pool features within each BEV cell using **max-pool** along Z via `torch_scatter.scatter_max`.

**Interface**:
- **Input**: `Point` with 3D `grid_coord (N, 3)`, `feat (N, 1088)`, `batch (N,)`
- **Output**: Sparse BEV: `feat (N_bev, 1088)`, `bev_coord (N_bev, 2)`, `batch (N_bev,)`

This is efficient because the outdoor point cloud is very sparse — most BEV cells are empty.

### Step B — Channel Projection + Dense BEV Conversion

**Module**: `SparseBEVNeck` (new)  
**Purpose**: Project channels and convert sparse BEV features into a single dense BEV feature map.

**Sub-steps**:

1. **Channel projection** — A single linear layer maps the 1088-dim up_cast features to the target channel dimension (e.g., 384):
   - `1088 → 384`

2. **Quantize to the target BEV grid** — The target grid has cell size 0.48 m and shape `(510, 510)`. Map each point's real-world `coord[:, :2]` into the target grid via:
   ```
   target_bev_ij = floor((coord[:, :2] - range_min) / target_cell_size)
   ```

3. **Scatter into dense tensor** — Allocate a `(B, 384, 510, 510)` tensor (zeros). `scatter_max` the 384-ch features into the tensor at the computed `target_bev_ij`.

4. **2D conv refinement** (optional but recommended) — 2–3 Conv2d layers with BN + ReLU on the dense BEV map. This provides horizontal spatial context that the 3D encoder lacks and fills gaps between sparse cells.

**Interface**:
- **Input**: Sparse BEV: `(feat, bev_coord, batch)` — single scale
- **Output**: `List[Tensor]` — `[(B, 384, 510, 510)]` — single-level dense BEV feature map, wrapped in a list for `CenterHead.forward(feats)`

---

## 3. Full Pipeline Diagram

```
Raw Points (N, 3)
       │
       ▼
┌──────────────────────────────┐
│  PTv3DetDataPreprocessor      │  Grid-hash downsample → PTv3 dict
│  (already implemented)        │  Out: Point{coord, grid_coord, feat, offset}
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  ConcertoLargeOutdoorXYZ     │  Encoder-only PTv3 + up_cast(level=2)
│  (up_cast_level=2)            │  Multi-scale fusion built-in
│                               │  Out: Point at enc2 resolution
│                               │  Channels: 1088 (=192+384+512)
│                               │  Grid res: 0.20 m
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  SparseBEVScatter             │  Step A: Z-axis max-pool
│  (new module)                 │  torch_scatter along Z
│                               │  Out: sparse 2D BEV (N_bev, 1088)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  SparseBEVNeck                │  Step B: Project + densify
│  (new module)                 │  1. Linear proj 1088 → 384
│                               │  2. Quantize to target BEV grid
│                               │  3. scatter_max into dense (B,384,510,510)
│                               │  4. 2–3 Conv2d refinement layers
│                               │  Out: List[Tensor(B, 384, 510, 510)]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  CenterHead                   │  Unchanged
│  in_channels=384              │  shared_conv → task_heads
│  heatmap: (B, 5, 510, 510)   │
└──────────────────────────────┘
```

---

## 4. Config Changes

The following CenterPoint config keys are **removed**:

- `pts_voxel_encoder` (was `PillarFeatureNet`)
- `pts_middle_encoder` (was `PointPillarsScatter`)
- `pts_backbone` (was `SECOND`)
- `pts_neck` (was `SECONDFPN`)

They are replaced by the Concerto backbone (with `up_cast_level=2`) + the two new lightweight modules. This also requires overriding `extract_pts_feat` in the `CenterPoint` detector class to route data through the new pipeline instead of the standard voxel → scatter → backbone → neck path.

The backbone config must set:
- `enc_mode=True` — encoder-only, no decoder
- `up_cast_level=2` — fuse enc4 → enc3 → enc2 via concatenation

---

## 5. Efficiency Notes

1. **Stay sparse as long as possible.** The full 510×510 BEV grid has 260K cells, but outdoor LiDAR typically occupies only 10–30% of BEV cells. All operations before the final densification use `torch_scatter` on occupied cells only.

2. **Z-pooling is trivial.** The point cloud range is only 8 m in Z (`[-3, 5]`), so each BEV column has very few points. A max-pool via `torch_scatter.scatter_max` is O(N) and negligible cost.

3. **`up_cast` replaces external multi-scale extraction entirely.** The backbone's built-in `up_cast` fuses enc4/enc3/enc2 features with zero learnable parameters (pure index + concatenation), using the `pooling_parent`/`pooling_inverse` chain already recorded during encoding. No need to modify the encoder forward pass or tap intermediate stages manually.

4. **Densification only once, at the target resolution.** The single sparse output is projected and scattered into `(B, 384, 510, 510)` in one step.

5. **The 2D conv refinement block** (2–3 lightweight Conv2d layers with BN + ReLU) is important because scattered features lack local spatial context in the BEV plane. The Concerto encoder operates in 3D — the 2D conv block provides the first opportunity for purely horizontal feature diffusion, filling in gaps between sparse cells.

6. **`up_cast_level=2` is the sweet spot.** It produces 0.20 m resolution (close to the 0.24 m of the original pillar-based pipeline) with rich 1088-dim multi-scale features. The point count at enc2 (~6K points for a typical outdoor scene) is very manageable. Going to level 3 (enc1, 0.10 m, 1136-dim) would quadruple point count for marginal benefit.
