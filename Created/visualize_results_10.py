#!/usr/bin/env python3

import sys
import os
import pickle
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

from tools.analysis_tools.visualize.run import Visualizer

def visualize_results_10(output_folder='output/results_10_viz'):
    """
    Visualize the results_10.pkl file
    """
    
    render_cfg = dict(
        with_occ_map=False,
        with_map=False,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=False,
        show_lidar=False,
        show_command=True,
        show_hd_map=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )

    viser = Visualizer(
        version='v1.0-trainval', 
        predroot='output/results_10_converted.pkl',  # Use converted results
        dataroot='data/nuscenes', 
        **render_cfg
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Visualizing results from output/results_10.pkl...")
    print(f"Number of samples with predictions: {len(viser.token_set)}")
    
    img_count = 0
    for i, sample_token in enumerate(viser.token_set):
        if img_count >= 10:  # Limit to 10 samples
            break
            
        try:
            # Get scene information
            sample = viser.nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            scene = viser.nusc.get('scene', scene_token)
            scene_name = scene['name']
            
            output_path = os.path.join(output_folder, f"sample_{img_count:03d}_{scene_name}")
            
            print(f"Processing sample {img_count}: {scene_name} (token: {sample_token[:16]}...)")
            
            # Generate both BEV and camera views, then combine them
            viser.visualize_bev(sample_token, output_path)
            viser.visualize_cam(sample_token, output_path)
            viser.combine(output_path)
            
            img_count += 1
            
        except Exception as e:
            print(f"Error processing sample {sample_token}: {e}")
            continue
    
    print(f"Generated {img_count} visualization images in {output_folder}")
    return img_count

if __name__ == '__main__':
    visualize_results_10()