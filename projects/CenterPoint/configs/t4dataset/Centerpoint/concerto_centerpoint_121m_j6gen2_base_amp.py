_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_v6.py",
    "../../default/concerto_centerpoint_base.py",
]
custom_imports = dict(
    imports=["projects.CenterPoint.models", "projects.Concerto.concerto"],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]
custom_imports["imports"] += ["autoware_ml.hooks"]
custom_imports["imports"] += ["autoware_ml.samplers"]

# --- Point cloud / BEV grid parameters ---
point_cloud_range = [-122.40, -122.40, -3.0, 122.40, 122.40, 5.0]
voxel_size = [0.24, 0.24, 8.0]
grid_size = [1020, 1020, 1]  # (122.40 / 0.24 == 510, 510 * 2 == 1020)
sweeps_num = 1
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)
out_size_factor = 2

backend_args = None
point_load_dim = 5  # x, y, z, intensity, ring_id
point_use_dim = 3  # x, y, z
lidar_sweep_dims = [0, 1, 2, 3, 4]

# eval parameter
eval_class_range = {
    "car": 121,
    "truck": 121,
    "bus": 121,
    "bicycle": 121,
    "pedestrian": 121,
}

# user setting
data_root = "/mnt/qnapdata/internal/t4datasets/"
info_directory_path = "info/kokseang_2_5/"
train_gpu_size = 1
train_batch_size = 16
test_batch_size = 2
num_workers = 16
val_interval = 1
max_epochs = 10

experiment_group_name = "centerpoint/j6gen2_v6/" + _base_.dataset_type
experiment_name = "concerto_centerpoint_121m_j6gen2_base_amp"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type="RandomFlip3D",
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5,
    # ),
    # dict(
    #     type="GlobalRotScaleTrans",
    #     rot_range=[-1.571, 1.571],
    #     scale_ratio_range=[0.80, 1.20],
    #     translation_std=[1.0, 1.0, 0.2],
    # ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes={{_base_.class_names}}),
    dict(type="ObjectMinPointsFilter", min_num_points=5),
    # dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "timestamp",
            "lidar2img",
            "depth2img",
            "cam2img",
            "box_type_3d",
            "sample_idx",
            "sample_token",
            "lidar_path",
            "ori_cam2img",
            "cam2global",
            "lidar2cam",
            "ego2global",
            "city",
            "vehicle_type",
        ),
    ),
]

eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "timestamp",
            "lidar2img",
            "depth2img",
            "cam2img",
            "box_type_3d",
            "sample_idx",
            "sample_token",
            "lidar_path",
            "ori_cam2img",
            "cam2global",
            "lidar2cam",
            "ego2global",
            "city",
            "vehicle_type",
        ),
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        pipeline=train_pipeline,
        modality=input_modality,
        backend_args=backend_args,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
    ),
)

val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    metric="bbox",
    backend_args=backend_args,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    eval_class_range=eval_class_range,
    filter_attributes=_base_.filter_attributes,
)

test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    metric="bbox",
    backend_args=backend_args,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    eval_class_range=eval_class_range,
    filter_attributes=_base_.filter_attributes,
    save_csv=True,
)

# --- Model overrides ---
model = dict(
    neck=dict(
        point_cloud_range=point_cloud_range,
        target_cell_size=voxel_size[0] * out_size_factor,  # 0.24 × 2 = 0.48
        bev_h=grid_size[0] // out_size_factor,  # 510
        bev_w=grid_size[1] // out_size_factor,  # 510
    ),
    bbox_head=dict(
        tasks=[
            dict(
                num_class=5,
                class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
            ),
        ],
        bbox_coder=dict(
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            out_size_factor=out_size_factor,
            # Override base score_threshold=0.1 which is incompatible with
            # init_bias=-4.595 (sigmoid ≈ 0.01).  With the aggressive AMP bias
            # the heatmap needs many iterations before any cell exceeds 0.1,
            # producing zero predictions in the meantime.  Disabling the hard
            # threshold lets top-k + NMS handle box selection instead.
            score_threshold=None,
        ),
        separate_head=dict(type="CustomSeparateHead", init_bias=-4.595, final_kernel=1),
        loss_cls=dict(
            type="mmdet.AmpGaussianFocalLoss", reduction="none", loss_weight=1.0
        ),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
        ),
    ),
    test_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            post_center_limit_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
        ),
    ),
)

randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)

lr = 3e-4
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=22,
        eta_min=lr * 1e-4,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=22,
        eta_min=1,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[(max_epochs - 5, 1)],
)
val_cfg = dict()
test_cfg = dict()

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
clip_grad = dict(max_norm=15, norm_type=2)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=optimizer,
    clip_grad=clip_grad,
    loss_scale={
        "init_scale": 2.0**12,
        "growth_interval": 600,
    },
)

auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

if train_gpu_size > 1:
    sync_bn = "torch"

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

logger_interval = 50
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=logger_interval),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=10,
        save_best="NuScenes metric/T4Metric/mAP",
    ),
)

custom_hooks = [
    dict(type="MomentumInfoHook"),
    dict(type="LossScaleInfoHook"),
]

load_from = None
