"""
UniAD Configuration for ITRI 6-Camera Setup (Padded)

This configuration adapts the standard UniAD setup for the ITRI 4-camera system 
extended to 6 cameras with dummy cameras for UniAD compatibility:
- Real cameras: front_100deg, front_left_100deg, front_right_100deg, back_60deg
- Dummy cameras: left_side, right_side
- Integration with semantic map lane queries
- Custom PC range and BEV grid dimensions
"""

# Base configuration adapted from base_track_map.py
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# ITRI-specific point cloud range (extended to cover semantic map data)
# Optimized based on coordinate alignment analysis
point_cloud_range = [-100.0, -200.0, -15.0, 400.0, 200.0, 5.0]
voxel_size = [0.5, 0.5, 20.0]  # Adjusted for larger range
patch_size = [500.0, 400.0]    # Adjusted for ITRI coordinate system

# Image normalization (standard ImageNet values for better compatibility)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# Object classes (reduced set for ITRI scenario)
class_names = [
    "car",
    "truck", 
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
]

# Input modality for ITRI setup
input_modality = dict(
    use_lidar=False, 
    use_camera=True, 
    use_radar=False, 
    use_map=True,      # Enable map input for semantic map integration
    use_external=True   # Enable CAN bus data
)

# Model dimensions
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4

# BEV grid configuration (matching UniAD checkpoint dimensions)
bev_h_ = 200  # Standard UniAD BEV height (40,000 total elements = 200x200)
bev_w_ = 200  # Standard UniAD BEV width
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

# Sequence settings
queue_length = 3  # Reduced for ITRI data availability

# Trajectory prediction
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

# Occupancy flow settings
occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

# Planning
planning_steps = 6
use_col_optim = True
planning_evaluation_strategy = "uniad"

# Occupancy grid configuration (adjusted for ITRI range)
occflow_grid_conf = {
    'xbound': [-100.0, 400.0, 1.0],   # Larger X range with 1m resolution
    'ybound': [-200.0, 200.0, 1.0],   # Larger Y range with 1m resolution  
    'zbound': [-15.0, 5.0, 20.0],     # Extended Z range
}

# Training settings
train_gt_iou_threshold = 0.3

# Camera configuration for ITRI 4-camera setup
# 6-camera setup: 4 real ITRI cameras + 2 dummy cameras for UniAD compatibility
camera_names = ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg', 'left_side', 'right_side']
num_cams = len(camera_names)

model = dict(
    type="UniAD",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=len(class_names),  # Adjusted for ITRI class count
    pc_range=point_cloud_range,
    
    # Image backbone (ResNet-101 with DCN)
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,  # Unfreeze for ITRI adaptation
        norm_cfg=dict(type="BN2d", requires_grad=True),
        norm_eval=False,
        style="pytorch",  # Changed from caffe for better compatibility
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),
        stage_with_dcn=(False, False, True, True),
    ),
    
    # FPN neck
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    
    # Training settings (unfroze for ITRI adaptation)
    freeze_img_backbone=False,
    freeze_img_neck=False,
    freeze_bn=False,
    
    score_thresh=0.4,
    filter_score_thresh=0.35,
    
    # Query interaction module
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),
    
    # Memory bank
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    
    # Loss configuration
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=len(class_names),
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
    ),
    
    # BEVFormer tracking head
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=len(class_names),
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        
        # Transformer configuration
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            num_cams=num_cams,  # Set to 6 for ITRI setup with dummy cameras
            
            # Encoder (BEVFormer)
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", 
                            embed_dims=_dim_, 
                            num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn", "norm", "cross_attn", "norm", "ffn", "norm",
                    ),
                ),
            ),
            
            # Decoder
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn", "norm", "cross_attn", "norm", "ffn", "norm",
                    ),
                ),
            ),
        ),
        
        # Bbox coder
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-110.0, -210.0, -20.0, 410.0, 210.0, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names),
        ),
        
        # Positional encoding
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        
        # Losses
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    
    # Segmentation head for semantic map integration
    seg_head=dict(
        type='PansegformerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        num_query=300,  # Match semantic map lane queries
        num_classes=4,   # Road surface classes
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        
        # Transformer for segmentation
        transformer=dict(
            type='SegDeformableTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                    ),
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            ),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                ),
            ),
        ),
        
        # Positional encoding
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5
        ),
        
        # Losses
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        
        # Mask heads
        thing_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=4
        ),
        stuff_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True
        ),
        
        # Training configuration
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0),
            ),
            sampler=dict(type='PseudoSampler'),
            sampler_with_mask=dict(type='PseudoSampler_segformer'),
        ),
    ),
    
    # Training configuration
    train_cfg=dict(
        pts=dict(
            grid_size=[800, 600, 1],  # Adjusted for larger PC range
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# Data configuration would be replaced with ITRI dataset loader
# This is a placeholder showing the expected structure

# Training optimization
optimizer = dict(
    type="AdamW",
    lr=1e-4,  # Reduced learning rate for fine-tuning
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# Training epochs
total_epochs = 12  # Increased for ITRI adaptation

# Evaluation
evaluation = dict(
    interval=2,
    planning_evaluation_strategy=planning_evaluation_strategy,
)

# Runner
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# Logging
log_config = dict(
    interval=10, 
    hooks=[
        dict(type="TextLoggerHook"), 
        dict(type="TensorboardLoggerHook")
    ]
)

# Checkpointing
checkpoint_config = dict(interval=1)

# Load pretrained BEVFormer weights
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"

# Find unused parameters
find_unused_parameters = True

# ITRI-specific configuration
itri_config = dict(
    # Camera setup
    camera_names=camera_names,
    num_cameras=num_cams,
    
    # Data paths
    data_root="/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic",
    semantic_map_dir="/home/bryan/Desktop/Allen/UniAD/semantic-map",
    
    # Coordinate system
    pc_range=point_cloud_range,
    bev_size=(bev_h_, bev_w_),
    
    # Integration settings
    use_semantic_map_queries=True,
    semantic_map_query_dim=300,
    lane_query_embed_dim=_dim_,
)