_base_ = ["projects/configs/_base_/default_runtime.py"]

# Minimal configuration for inference-only
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Model parameters (same as original)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
canvas_size = (bev_h_, bev_w_)
queue_length = 3

# Model configuration (minimal for inference)
model = dict(
    type="UniAD",
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=10,
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,
    img_backbone=dict(
        type="ResNet", depth=101, num_stages=4, out_indices=(1, 2, 3),
        frozen_stages=4, norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True, style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN", in_channels=[512, 1024, 2048], out_channels=_dim_,
        start_level=0, add_extra_convs="on_output", num_outs=4, relu_before_extra_convs=True,
    ),
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
)

# Minimal dataset configuration - try to bypass complex loading
dataset_type = "CustomDataset"
data_root = "data/"
file_client_args = dict(backend="disk")

# Simple test pipeline
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img"])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
    )
)