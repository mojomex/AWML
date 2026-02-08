# Concerto-CenterPoint base model config
#
# Replaces the traditional PillarFeatureNet → PointPillarsScatter → SECOND →
# SECONDFPN pipeline with ConcertoLargeOutdoorXYZ (encoder-only PTv3 with
# up_cast) + SparseBEVNeck.
#
# See docs/design/concerto_centerpoint_adaptation.md for the full design.

# The ConcertoLargeOutdoorXYZ backbone with up_cast_level=2 produces:
#   - 1088-dim features (192 + 384 + 512, from enc2 + enc3 + enc4)
#   - 0.20 m effective resolution (enc2 grid)
# SparseBEVNeck projects 1088 → 384 and scatters into (B, 384, 510, 510).

out_size_factor = 2

model = dict(
    type="ConcertoCenterPoint",
    data_preprocessor=dict(
        type="PTv3DetDataPreprocessor",
        grid_size=0.05,
        max_points=128_000,
    ),
    backbone=dict(
        type="ConcertoLargeOutdoorXYZ",
        enc_mode=True,
        up_cast_level=2,
    ),
    neck=dict(
        type="SparseBEVNeck",
        in_channels=1088,  # 192 + 384 + 512 from up_cast_level=2
        out_channels=384,
        bev_h=510,
        bev_w=510,
        point_cloud_range=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
        target_cell_size=0.48,  # voxel_size (0.24) × out_size_factor (2)
        num_convs=3,
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=384,
        # (output_channel_size, num_conv_layers)
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2),
        ),
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            max_num=500,
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            code_size=9,
        ),
        share_conv_channel=64,
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=1),
        loss_cls=dict(
            type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0
        ),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        pts=dict(
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        )
    ),
    test_cfg=dict(
        pts=dict(
            nms_type="circle",
            min_radius=[1.0],
            post_max_size=100,
        )
    ),
    freeze_backbone=True,
)
