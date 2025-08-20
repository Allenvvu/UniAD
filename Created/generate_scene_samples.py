#!/usr/bin/env python3

import sys
import os
import pickle
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

from tools.analysis_tools.visualize.run import Visualizer
from nuscenes.nuscenes import NuScenes

def generate_scene_samples(output_folder):
    """
    Generate one sample image for each scene that has predictions
    """
    
    # Load prediction results
    with open('output/results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    bbox_results = results['bbox_results']
    sample_tokens = [result['token'] for result in bbox_results]
    
    # Load NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    
    # Group samples by scene and get first sample for each scene
    scene_first_samples = {}
    
    for token in sample_tokens:
        try:
            sample = nusc.get('sample', token)
            scene_token = sample['scene_token']
            scene = nusc.get('scene', scene_token)
            scene_name = scene['name']
            
            # Keep only the first sample for each scene
            if scene_name not in scene_first_samples:
                scene_first_samples[scene_name] = token
        except:
            continue
    
    print(f"Found {len(scene_first_samples)} scenes with predictions")
    
    # Set up visualizer
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
        predroot='output/results.pkl', 
        dataroot='data/nuscenes', 
        **render_cfg
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Generating sample images for {len(scene_first_samples)} scenes...")
    
    generated_files = []
    for i, (scene_name, sample_token) in enumerate(sorted(scene_first_samples.items())):
        output_path = os.path.join(output_folder, f"{scene_name}_sample")
        
        print(f"Processing {i+1}/{len(scene_first_samples)}: {scene_name}")
        
        # Generate both BEV and camera views, then combine them
        viser.visualize_bev(sample_token, output_path)
        viser.visualize_cam(sample_token, output_path)
        viser.combine(output_path)
        
        generated_files.append(f"{scene_name}_sample.jpg")
    
    print(f"\nGenerated {len(generated_files)} sample images in {output_folder}")
    print("\nGenerated files:")
    for filename in generated_files:
        print(f"  {filename}")
    
    return generated_files

if __name__ == '__main__':
    output_dir = 'output/scene_samples'
    generate_scene_samples(output_dir)