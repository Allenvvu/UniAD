_base_ = ["./base_e2e.py"]

# Override the test dataset configuration to use only 500 samples
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type='NuScenesE2EDataset',
        file_client_args=dict(backend="disk"),
        data_root='data/nuscenes/',
        test_mode=True,
        ann_file='data/infos/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
                    file_client_args=dict(backend="disk"), img_root='data/nuscenes/'),
            dict(type="NormalizeMultiviewImage", **dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(type='LoadAnnotations3D_E2E', 
                 with_bbox_3d=False,
                 with_label_3d=False, 
                 with_attr_label=False,
                 with_future_anns=True,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=True,
                 ),
            dict(type='GenerateOccFlowLabels', grid_conf={
                    'xbound': [-50.0, 50.0, 0.5],
                    'ybound': [-50.0, 50.0, 0.5],
                    'zbound': [-10.0, 10.0, 20.0],
                }, ignore_index=255, only_vehicle=True, 
                                                   filter_invisible=False),
            dict(
                type="MultiScaleFlipAug3D",
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type="DefaultFormatBundle3D", class_names=[
                            "car", "truck", "construction_vehicle", "bus", "trailer", 
                            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
                        ], with_label=False
                    ),
                    dict(
                        type="CustomCollect3D", keys=[
                                                    "img",
                                                    "timestamp",
                                                    "l2g_r_mat",
                                                    "l2g_t",
                                                    "gt_lane_labels",
                                                    "gt_lane_bboxes",
                                                    "gt_lane_masks",
                                                    "gt_segmentation",
                                                    "gt_instance", 
                                                    "gt_centerness", 
                                                    "gt_offset", 
                                                    "gt_flow",
                                                    "gt_backward_flow",
                                                    "gt_occ_has_invalid_frame",	
                                                    "gt_occ_img_is_valid",	
                                                    "sdc_planning",	
                                                    "sdc_planning_mask",	
                                                    "command",
                                                ]
                    ),
                ],
            ),
        ],
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        predict_steps=12,
        past_steps=4,
        fut_steps=4,
        occ_n_future=6,
        use_nonlinear_optimizer=True,
        classes=[
            "car", "truck", "construction_vehicle", "bus", "trailer", 
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
        ],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        eval_mod=['det', 'map', 'track','motion']
    ),
)

# Use the end-to-end checkpoint
load_from = 'ckpts/uniad_base_e2e.pth'