_base_ = ["stage2_e2e/base_e2e.py"]

# ITRI-specific overrides for motion+occ+planning head training
# This config masks out image inputs and uses pre-computed data

# Plugin configuration - fix the import path
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Import class_names and other variables from base config
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorbike",
    "bicycle",
    "people",
    "traffic_cone",
]

# Other required variables from base config
queue_length = 3
predict_steps = 12
past_steps = 4
fut_steps = 4
use_nonlinear_optimizer = True
occ_n_future_max = 6
planning_evaluation_strategy = "uniad"

# Update data paths for ITRI dataset
data_root = "data/training/"
info_root = "data/training/"

# Override input modality - no camera, no BEV features, use external data only
input_modality = dict(
    use_lidar=False,
    use_camera=False,  # Masked out
    use_radar=False,
    use_map=True,
    use_external=True,  # Use CAN bus, embeddings, queries, etc.
    use_bev_features=False  # No BEV features input
)

# Enable virtual BEV mode and freeze components for ITRI training
model = dict(
    # Enable virtual BEV mode
    use_virtual_bev=True,
    virtual_bev_config=dict(
        embed_dim=256,
        bev_h=200,
        bev_w=200,
        use_learnable_bev=True,
        use_spatial_encoding=True
    ),

    # Only these heads will be trainable
    motion_head=dict(
        type='MotionHead',
        # Keep existing motion_head config from base
    ),
    occ_head=dict(
        type='OccHead',
        # Keep existing occ_head config from base
    ),
    planning_head=dict(
        type='PlanningHeadSingleMode',
        # Keep existing planning_head config from base
    ),
)

# Custom ITRI dataset configuration
dataset_type = "ITRIDataset"  # Will need to implement this
ann_file_train = info_root + "train_annotations.pkl"  # Will need to create
ann_file_val = info_root + "val_annotations.pkl"      # Will need to create

# Modified pipeline for ITRI data without images or BEV features
train_pipeline = [
    # Load CAN bus data
    dict(type="LoadCANBusData",
         canbus_root=data_root + "canbus/"),

    # Load SDC embeddings
    dict(type="LoadSDCEmbeddings",
         sdc_embeddings_root=data_root + "sdc_embeddings/"),

    # Load track queries
    dict(type="LoadTrackQueries",
         track_query_root=data_root + "track_query/"),

    # Load map queries
    dict(type="LoadMapQueries",
         map_query_root=data_root + "map_query/"),

    # Load ground truth future trajectories
    dict(type="LoadGTFutureTraj",
         gt_fut_traj_root=data_root + "gt_fut_traj/"),

    # Load SDC planning data
    dict(type="LoadSDCPlanningData",
         sdc_planning_root=data_root + "sdc_planning/"),

    # Format and collect for training
    dict(type="FormatITRIData", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "img",                   # Dummy image for compatibility
            "img_metas",             # Image metadata
            "canbus_data",           # CAN bus data
            "sdc_embeddings",        # SDC embeddings
            "track_queries",         # Track queries
            "track_query_matched_idxes",  # Track matched indices
            "track_bbox_results",    # Real track bbox results (if available)
            "sdc_track_bbox_results",# Real SDC track bbox (if available)
            "map_queries",           # Map queries
            "gt_fut_traj",           # GT future trajectories
            "sdc_planning",          # SDC planning data
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            # Motion prediction GT
            "gt_fut_traj_mask",
            "gt_past_traj",
            "gt_past_traj_mask",
            # Occupancy GT (will need to generate/load)
            "gt_segmentation",
            "gt_instance",
            "gt_centerness",
            "gt_offset",
            "gt_flow",
            "gt_backward_flow",
            "gt_occ_has_invalid_frame",
            "gt_occ_img_is_valid",
            # Planning GT
            "sdc_planning_mask",
            "command",
        ],
    ),
]

test_pipeline = [
    # Similar to train but without data augmentation
    dict(type="LoadCANBusData",
         canbus_root=data_root + "canbus/"),
    dict(type="LoadSDCEmbeddings",
         sdc_embeddings_root=data_root + "sdc_embeddings/"),
    dict(type="LoadTrackQueries",
         track_query_root=data_root + "track_query/"),
    dict(type="LoadMapQueries",
         map_query_root=data_root + "map_query/"),
    dict(type="LoadSDCPlanningData",
         sdc_planning_root=data_root + "sdc_planning/"),
    dict(type="FormatITRIData", class_names=class_names, with_label=False),
    dict(
        type="CustomCollect3D",
        keys=[
            "img",                   # Dummy image for compatibility
            "img_metas",             # Image metadata
            "canbus_data",
            "sdc_embeddings",
            "track_queries",
            "track_query_matched_idxes",
            "track_bbox_results",
            "sdc_track_bbox_results",
            "map_queries",
            "sdc_planning",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "sdc_planning_mask",
            "command",
        ]
    ),
]

# Override data config for ITRI
data = dict(
    samples_per_gpu=2,  # Can potentially increase since no image processing
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # ITRI-specific parameters
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['motion', 'occ', 'planning'],  # Only eval these modules
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_val,  # Use val for testing
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        eval_mod=['motion', 'occ', 'planning'],
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# Optimizer - focus on motion/occ/planning parameters only
optimizer = dict(
    type="AdamW",
    lr=1e-4,  # Lower learning rate for fine-tuning
    paramwise_cfg=dict(
        custom_keys={
            # Freeze all other components
            "img_backbone": dict(lr_mult=0.0),
            "img_neck": dict(lr_mult=0.0),
            "pts_bbox_head": dict(lr_mult=0.0),
            "seg_head": dict(lr_mult=0.0),
            # Only train these heads
            "motion_head": dict(lr_mult=1.0),
            "occ_head": dict(lr_mult=1.0),
            "planning_head": dict(lr_mult=1.0),
        }
    ),
    weight_decay=0.01,
)

# Adjust loss weights to focus on motion/occ/planning
loss_weights = dict(
    motion_loss_weight=2.0,    # Increase motion loss weight
    occ_loss_weight=2.0,       # Increase occ loss weight
    planning_loss_weight=2.0,  # Increase planning loss weight
    detection_loss_weight=0.0, # Disable detection loss
    tracking_loss_weight=0.0,  # Disable tracking loss
    seg_loss_weight=0.0,       # Disable segmentation loss
)

# Training schedule - shorter since only training subset of model
total_epochs = 10
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-4,
)

evaluation = dict(
    interval=2,  # More frequent evaluation
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# Load from pre-trained UniAD checkpoint
load_from = "ckpts/uniad_base_track_map.pth"

# Only save motion/occ/planning head weights
checkpoint_config = dict(
    interval=1,
    save_optimizer=False,  # Save space
    create_symlink=False,
)

# Custom work directory for this experiment
work_dir = './work_dirs/itri_motion_occ_planning'

# Enable gradient checkpointing to save memory
gradient_checkpoint = True
find_unused_parameters = False  # Set to False for efficiency