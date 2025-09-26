# ITRI Inference Configuration for UniAD
# Simplified config optimized for inference with ITRI data

_base_ = ["stage2_e2e/base_e2e.py"]

# Override for inference mode
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Point cloud range for ITRI setup
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# ITRI specific camera setup (4 cameras)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# Input modality for ITRI
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=True,
    use_external=True
)

# BEV parameters
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
canvas_size = (bev_h_, bev_w_)
queue_length = 3

# Prediction parameters
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

# Occupancy parameters
occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

# Planning parameters
planning_steps = 6
use_col_optim = True
planning_evaluation_strategy = "uniad"

# Occupancy grid configuration
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Class names for ITRI (matching NuScenes classes)
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
group_id_list = [[0,1,2,3,4], [6,7], [8], [5,9]]

# Model configuration - keep most from base but optimize for inference
model = dict(
    type="UniAD",
    gt_iou_threshold=0.3,
    queue_length=queue_length,
    use_grid_mask=False,  # Disable for inference
    video_test_mode=True,
    num_query=900,
    num_classes=10,
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,

    # Freeze components for inference
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,

    score_thresh=0.4,
    filter_score_thresh=0.35,
)

# Dataset configuration for ITRI data
dataset_type = "NuScenesE2EDataset"  # Use base dataset for now
data_root = "data/itri/hct_train/"
file_client_args = dict(backend="disk")

# Simplified test pipeline for inference
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, file_client_args=file_client_args),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="CustomCollect3D", keys=["img"])
        ],
    ),
]

# Data configuration for inference
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,  # Reduced for inference
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
    ),
)