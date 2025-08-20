#!/usr/bin/env python3

import sys
import os
import glob
import numpy as np
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

from tools.analysis_tools.visualize.run import Visualizer

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not found. Video generation will be disabled.")
    print("Install OpenCV with: pip install opencv-python")

def visualize_specific_scenes(scene_names, output_folder):
    """
    Visualize specific scenes from mini dataset
    
    Args:
        scene_names: list of scene names (e.g., ['scene-0061', 'scene-0103'])
        output_folder: where to save visualization images
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
        predroot='output/results.pkl', 
        dataroot='data/nuscenes', 
        **render_cfg
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create mapping from scene token to name
    scene_token_to_name = {}
    scene_name_to_token = {}
    for scene in viser.nusc.scene:
        scene_token_to_name[scene['token']] = scene['name']
        scene_name_to_token[scene['name']] = scene['token']

    # Filter samples by specified scenes
    target_scene_tokens = [scene_name_to_token[name] for name in scene_names if name in scene_name_to_token]
    
    print(f"Visualizing scenes: {scene_names}")
    print(f"Found {len(target_scene_tokens)} matching scenes")
    
    img_count = 0
    for i, sample in enumerate(viser.nusc.sample):
        sample_token = sample['token']
        scene_token = sample['scene_token']
        
        # Only process samples from target scenes
        if scene_token not in target_scene_tokens:
            continue
            
        # Only process samples that have predictions
        if sample_token not in viser.token_set:
            print(f"Sample {sample_token} not in prediction pkl, skipping...")
            continue
            
        scene_name = scene_token_to_name[scene_token]
        output_path = os.path.join(output_folder, f"{scene_name}_{str(img_count).zfill(3)}")
        
        print(f"Processing sample {img_count}: {scene_name}")
        # Generate both BEV and camera views, then combine them
        viser.visualize_bev(sample_token, output_path)
        viser.visualize_cam(sample_token, output_path)
        viser.combine(output_path)
        img_count += 1
    
    print(f"Generated {img_count} visualization images in {output_folder}")
    
    # Create video from generated images
    if img_count > 0 and HAS_OPENCV:
        video_filename = create_video_from_images(output_folder, scene_names)
        if video_filename:
            print(f"Video created: {video_filename}")
    elif img_count > 0:
        print("Video generation skipped - OpenCV not available")
    
    return img_count

def create_video_from_images(output_folder, scene_names, fps=10):
    """
    Create AVI video from generated visualization images
    
    Args:
        output_folder: folder containing the images
        scene_names: list of scene names for video filename
        fps: frames per second for the video
    """
    if not HAS_OPENCV:
        print("OpenCV not available for video creation")
        return None
    # Get all image files in the output folder (try both PNG and JPG)
    png_pattern = os.path.join(output_folder, "*.png")
    jpg_pattern = os.path.join(output_folder, "*.jpg")
    image_files = sorted(glob.glob(png_pattern) + glob.glob(jpg_pattern))
    
    if not image_files:
        print("No PNG or JPG images found for video creation")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read first image: {image_files[0]}")
        return
        
    height, width, layers = first_image.shape
    
    # Create video filename
    scene_str = "_".join(scene_names[:3])  # Use first 3 scene names
    if len(scene_names) > 3:
        scene_str += "_and_more"
    video_filename = os.path.join(output_folder, f"{scene_str}_visualization.avi")
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    print(f"Creating video: {video_filename}")
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    print(f"Processing {len(image_files)} images...")
    
    # Add each image to video
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is not None:
            video_writer.write(img)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release video writer
    video_writer.release()
    print(f"Video saved successfully: {video_filename}")
    
    # Verify video file was created and has size > 0
    if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
        print(f"Video file size: {os.path.getsize(video_filename)} bytes")
        return video_filename
    else:
        print("Error: Video file was not created properly")
        return None

def create_combined_video_for_scenes(scene_list, output_folder_base="output/combined_scenes", fps=10):
    """
    Generate visualization images for multiple scenes and combine into one video
    
    Args:
        scene_list: list of scene names to visualize
        output_folder_base: base folder for outputs
        fps: frames per second for the video
    """
    if not HAS_OPENCV:
        print("OpenCV not available for video creation")
        return None
    
    print(f"Generating combined video for {len(scene_list)} scenes: {scene_list}")
    
    # Create output folder
    if not os.path.exists(output_folder_base):
        os.makedirs(output_folder_base)
    
    all_images = []
    total_frames = 0
    
    # Generate images for each scene
    for scene_name in scene_list:
        scene_folder = os.path.join(output_folder_base, scene_name)
        print(f"\nProcessing scene: {scene_name}")
        
        # Generate images for this scene
        img_count = visualize_specific_scenes([scene_name], scene_folder)
        
        if img_count > 0:
            # Get all images from this scene
            scene_images = sorted(glob.glob(os.path.join(scene_folder, "*.jpg")))
            all_images.extend(scene_images)
            total_frames += len(scene_images)
            print(f"Added {len(scene_images)} frames from {scene_name}")
    
    if not all_images:
        print("No images generated for any scene")
        return None
    
    # Create combined video
    print(f"\nCreating combined video from {total_frames} total frames...")
    
    # Read first image to get dimensions
    first_image = cv2.imread(all_images[0])
    if first_image is None:
        print(f"Could not read first image: {all_images[0]}")
        return None
        
    height, width, layers = first_image.shape
    
    # Create video filename
    video_filename = os.path.join(output_folder_base, f"combined_{len(scene_list)}_scenes.avi")
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Add each image to video
    for i, image_file in enumerate(all_images):
        img = cv2.imread(image_file)
        if img is not None:
            video_writer.write(img)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(all_images)} frames")
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release video writer
    video_writer.release()
    print(f"Combined video saved: {video_filename}")
    
    # Verify video file was created and has size > 0
    if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
        print(f"Video file size: {os.path.getsize(video_filename)} bytes")
        return video_filename
    else:
        print("Error: Video file was not created properly")
        return None

if __name__ == '__main__':
    # Example usage - modify these scene names as needed
    scenes_to_visualize = ['scene-0093', 'scene-0094', 'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098']
    
    if len(sys.argv) > 1:
        scenes_to_visualize = sys.argv[1].split(',')
    
    # Create combined video for all scenes
    create_combined_video_for_scenes(scenes_to_visualize)