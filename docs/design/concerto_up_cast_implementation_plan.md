# Concerto `up_cast` Implementation Plan

## Background

The pre-training Concerto model ([`concerto_v1m1_base.py`](https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/concerto/concerto_v1m1_base.py)) has an `up_cast` operation that casts coarse features from later encoder stages back to the finer resolution of earlier stages via concatenation. This produces multi-scale fused features at a higher spatial resolution than the final encoder output alone. The current Concerto `PointTransformerV3` in `projects/Concerto/concerto/model.py` lacks this mechanism.

---

## How `up_cast` Works

### The `pooling_parent` / `pooling_inverse` Linked List

During encoding, each `GridPooling` layer with `traceable=True` stores two fields on its output `Point`:

- **`pooling_parent`** → the **input** `Point` to that pooling layer (the finer-resolution point cloud, including its own features)
- **`pooling_inverse`** → a `(N_fine,)` index tensor mapping each fine point to its coarse cluster: `coarse_feat[pooling_inverse[i]]` gives the coarse feature for fine point `i`

This creates a linked list from the deepest stage back through each resolution:

```
enc4.Point
  └─ pooling_parent → enc3.Point
                         └─ pooling_parent → enc2.Point
                                               └─ pooling_parent → enc1.Point
                                                                     └─ pooling_parent → enc0.Point
```

### The `up_cast` Method

```python
def up_cast(self, point, upcast_level=None):
    if upcast_level is None:
        upcast_level = self.up_cast_level   # default: 2
    for _ in range(upcast_level):
        parent = point.pop("pooling_parent")    # finer-resolution Point
        inverse = point.pop("pooling_inverse")  # coarse→fine mapping
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point
```

Each iteration:

1. Retrieves the parent (finer-resolution) point cloud.
2. **Broadcasts** coarse features to every fine point via indexing: `point.feat[inverse]` has shape `(N_fine, C_coarse)`.
3. **Concatenates** the parent's own features with the upcast coarse features: `cat([parent.feat, point.feat[inverse]], dim=-1)`.
4. Moves to the parent for the next iteration.

### Concrete Example

With `up_cast_level=2` (default in pre-training) and `enc_channels = (48, 96, 192, 384, 512)`:

| Iteration | Start point    | Parent     | Result `feat` dim            | Result resolution |
|-----------|---------------|------------|------------------------------|-------------------|
| 0         | enc4 (512-d)  | enc3 (384-d) | `cat(384, 512) = 896`      | enc3 grid (8×)    |
| 1         | merged (896-d)| enc2 (192-d) | `cat(192, 896) = 1088`     | enc2 grid (4×)    |

Final output: a point cloud at **enc2 resolution** with **1088-dimensional** features encoding information from stages 2, 3, and 4.

### Inputs & Outputs

|          | Input                                                                 | Output                                                |
|----------|-----------------------------------------------------------------------|-------------------------------------------------------|
| **Data** | `Point` from final encoder stage (enc4), with `pooling_parent`/`pooling_inverse` chain intact | `Point` at finer resolution (enc stage = num_stages - 1 - up_cast_level) |
| **feat** | `(N_enc4, 512)`                                                       | `(N_enc2, 1088)` for level=2                          |
| **Spatial** | Coarsest grid                                                      | Finer grid (4× finer for level=2)                     |
| **Side effect** | Consumes (pops) the pooling chain entries as it traverses them |                                                       |

### `up_cast` vs. `GridUnpooling` (Decoder)

| | `up_cast` (pre-training) | `GridUnpooling` (decoder) |
|---|---|---|
| **Fusion** | Concatenation (`torch.cat`) | Addition after learned projection |
| **Parameters** | Zero (index + cat) | Learned linear projections |
| **Output dim** | Grows with each level | Fixed (`out_channels`) |
| **Purpose** | Multi-scale feature aggregation for heads | U-Net decoder reconstruction |

---

## Implementation Steps

### Step 1: Verify `traceable=True` Is Active in `GridPooling`

**File**: `projects/Concerto/concerto/model.py`

`GridPooling` defaults to `traceable=True` (line 403), and nothing in `PointTransformerV3.__init__` overrides this. **No change needed** — the pooling chain is already recorded during `forward()`.

### Step 2: Add `up_cast_level` Parameter to `PointTransformerV3.__init__`

**File**: `projects/Concerto/concerto/model.py`

- Add `up_cast_level=0` to the constructor signature (default 0 = no up-casting, fully backward-compatible).
- Store `self.up_cast_level = up_cast_level`.
- Store `self._enc_channels = tuple(enc_channels)` for output dimension computation.

### Step 3: Add `up_cast()` Method to `PointTransformerV3`

**File**: `projects/Concerto/concerto/model.py`

```python
def up_cast(self, point, up_cast_level=None):
    """Cast coarse encoder features back to finer resolutions by
    traversing the pooling_parent chain and concatenating features.

    Args:
        point: Point dict from the final encoder stage, with
               pooling_parent/pooling_inverse chain intact.
        up_cast_level: Number of levels to go back. If None, uses
                       self.up_cast_level (defaults to 0 = no-op).

    Returns:
        Point at finer resolution with concatenated multi-scale features.
        feat dimension = sum of channels from target level through final level.
    """
    if up_cast_level is None:
        up_cast_level = self.up_cast_level
    for _ in range(up_cast_level):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point
```

### Step 4: Add `output_channels` Property

**File**: `projects/Concerto/concerto/model.py`

```python
@property
def output_channels(self):
    """Feature dimension of the output after up_cast."""
    if self.up_cast_level == 0:
        return self._enc_channels[-1]
    start = len(self._enc_channels) - 1 - self.up_cast_level
    return sum(self._enc_channels[start:])
```

### Step 5: Call `up_cast` in `forward()`

**File**: `projects/Concerto/concerto/model.py`

Current:
```python
def forward(self, data_dict):
    ...
    point = self.enc(point)
    if not self.enc_mode:
        point = self.dec(point)
    return point
```

Change to:
```python
def forward(self, data_dict):
    ...
    point = self.enc(point)
    if not self.enc_mode:
        point = self.dec(point)
    if self.up_cast_level > 0:
        point = self.up_cast(point)
    return point
```

### Step 6: Update `model_instances.py`

**File**: `projects/Concerto/concerto/model_instances.py`

Pass `up_cast_level` through `kwargs` when constructing `ConcertoLargeOutdoor`. The caller / config specifies the desired level (e.g., `up_cast_level=2`).

### Step 7: Update Downstream Consumers (CenterPoint Configs)

The main consumer is the CenterPoint adaptation pipeline. With `up_cast`, the output changes:

| `up_cast_level` | Output resolution | Output channels | Output point count |
|-----------------|------------------|-----------------|--------------------|
| 0 (current)     | enc4 (0.80 m)    | 512             | ~N/16⁴             |
| 2 (recommended) | enc2 (0.20 m)    | 1088            | ~N/4²              |
| 3               | enc1 (0.10 m)    | 1136            | ~N/2               |

For CenterPoint detection, `up_cast_level=2` is the recommended value — it gives 0.20 m resolution (close to the 0.24 m of the original pillar-based pipeline) with rich multi-scale features. Downstream modules (`SparseBEVScatter`, `SparseBEVNeck`, or any head) must adjust their `in_channels` to match the new output dimension (1088 instead of 512).

---

## Summary of Files to Modify

| File | Change |
|------|--------|
| `projects/Concerto/concerto/model.py` | Add `up_cast_level` param to `__init__`, store `_enc_channels`, add `up_cast()` method, add `output_channels` property, call in `forward()` |
| `projects/Concerto/concerto/model_instances.py` | Pass `up_cast_level` through to backbone config |
| Downstream CenterPoint configs | Adjust `in_channels` to match new output dimension |

---

## Risks & Considerations

1. **Memory**: `up_cast_level=2` means the output has ~16× more points than level 0 (enc2 vs enc4). For a typical outdoor scene with ~100K input points, enc2 might have ~6K points — still very manageable.

2. **The `pooling_parent` chain is consumed by `up_cast`** (via `pop`). If anything else needs to traverse the chain afterward (e.g., `pool_corr` for correspondence, or a decoder), it can't. For encoder-only detection this is fine. If both `up_cast` and another consumer are needed, the chain would need to be deep-copied or traversed non-destructively.

3. **Decoder interaction**: When `enc_mode=False`, the decoder's `GridUnpooling` also traverses `pooling_parent`/`pooling_inverse`. Running both `up_cast` and the decoder on the same output would break. The implementation guards this by only applying `up_cast` after the decoder (or when in encoder-only mode). Safest practice: only enable `up_cast` when `enc_mode=True`.

4. **Pre-trained weight compatibility**: The `up_cast` operation introduces **zero new parameters** — it's purely index + concatenation. Pre-trained Concerto encoder weights load unchanged. The only weight implications are downstream: whatever head consumes the output must expect the larger feature dimension.
