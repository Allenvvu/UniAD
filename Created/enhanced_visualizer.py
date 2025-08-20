#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
import cv2
import sys

# Add NuScenes to path
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

class EnhancedVisualization:
    def __init__(self, results_path, dataroot='data/nuscenes', output_dir='output/enhanced_viz'):
        self.results_path = results_path
        self.dataroot = dataroot
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Loaded {len(self.results)} inference results")
        
        # Load NuScenes dataset
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
            self.has_nusc = True
            print("NuScenes dataset loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load NuScenes: {e}")
            self.has_nusc = False
            
        # Camera order for layout
        self.camera_order = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
        ]
        
    def get_camera_images(self, sample_token):
        """Get 6 camera images for a sample token"""
        if not self.has_nusc:
            return {}
            
        try:
            sample = self.nusc.get('sample', sample_token)
            camera_images = {}
            
            for cam_name in self.camera_order:
                if cam_name in sample['data']:
                    cam_token = sample['data'][cam_name]
                    cam_data = self.nusc.get('sample_data', cam_token)
                    
                    # Load image
                    img_path = os.path.join(self.dataroot, cam_data['filename'])
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            camera_images[cam_name] = img
                            
            return camera_images
        except Exception as e:
            print(f"Error loading camera images for token {sample_token}: {e}")
            return {}
    
    def create_bev_plot(self, result):
        """Create BEV plot as matplotlib figure"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up BEV coordinate system
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Bird\'s Eye View - UniAD Results', fontsize=14, fontweight='bold')
        
        # Draw ego vehicle
        ego_rect = Rectangle((-1, -2), 2, 4, linewidth=3, edgecolor='red', facecolor='red', alpha=0.8)
        ax.add_patch(ego_rect)
        ax.text(0, -5, 'EGO', ha='center', fontsize=12, fontweight='bold', color='red')
        
        # Extract and draw planning trajectory
        if 'planning' in result and 'result_planning' in result['planning']:
            planning_result = result['planning']['result_planning']
            if 'sdc_traj' in planning_result:
                traj = planning_result['sdc_traj']
                if hasattr(traj, 'cpu'):
                    traj_np = traj.cpu().numpy().squeeze()
                else:
                    traj_np = traj
                
                if len(traj_np.shape) >= 2 and traj_np.shape[1] >= 2:
                    x_coords = traj_np[:, 0]
                    y_coords = traj_np[:, 1]
                    
                    # Draw trajectory line
                    ax.plot(x_coords, y_coords, 'b-', linewidth=4, label='Planned Path', marker='o', markersize=6)
                    ax.scatter(x_coords[0], y_coords[0], c='green', s=120, marker='o', label='Start', zorder=5, edgecolor='darkgreen', linewidth=2)
                    ax.scatter(x_coords[-1], y_coords[-1], c='blue', s=120, marker='s', label='Goal', zorder=5, edgecolor='darkblue', linewidth=2)
                    
        # Extract and draw detected objects
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            scores = result.get('scores_3d', None)
            labels = result.get('labels_3d', None)
            
            if hasattr(boxes, 'tensor'):
                box_tensor = boxes.tensor
                if hasattr(box_tensor, 'cpu'):
                    boxes_np = box_tensor.cpu().numpy()
                else:
                    boxes_np = box_tensor
                    
                # NuScenes class names and colors
                class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
                colors = ['purple', 'brown', 'orange', 'yellow', 'pink', 
                         'gray', 'cyan', 'magenta', 'lightgreen', 'lightblue']
                
                for i, box in enumerate(boxes_np):
                    x, y, z, w, l, h, yaw = box[:7]
                    
                    # Create rotated rectangle for BEV
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)
                    
                    corners = np.array([
                        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
                    ])
                    
                    rotated_corners = []
                    for corner in corners:
                        rotated_x = corner[0] * cos_yaw - corner[1] * sin_yaw + x
                        rotated_y = corner[0] * sin_yaw + corner[1] * cos_yaw + y
                        rotated_corners.append([rotated_x, rotated_y])
                    
                    rotated_corners = np.array(rotated_corners)
                    
                    # Get label and color
                    label_idx = labels[i].item() if labels is not None else 0
                    label_name = class_names[label_idx] if label_idx < len(class_names) else 'unknown'
                    color = colors[label_idx] if label_idx < len(colors) else 'black'
                    
                    # Draw bounding box
                    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 
                           color=color, linewidth=2.5, alpha=0.9)
                    ax.fill(rotated_corners[:-1, 0], rotated_corners[:-1, 1], 
                           color=color, alpha=0.4)
                    
                    # Add label with confidence
                    score_text = f"{scores[i]:.2f}" if scores is not None else ""
                    ax.text(x, y, f'{label_name}\n{score_text}', ha='center', va='center', 
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.9, edgecolor=color))
                    
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        
        return fig
    
    def create_combined_visualization(self, sample_idx=0):
        """Create combined visualization with 6 cameras + BEV"""
        if sample_idx >= len(self.results):
            print(f"Sample {sample_idx} out of range")
            return None
            
        result = self.results[sample_idx]
        sample_token = result.get('token', '')
        
        print(f"Creating visualization for sample {sample_idx}, token: {sample_token}")
        
        # Get camera images
        camera_images = self.get_camera_images(sample_token)
        if not camera_images:
            print(f"No camera images found for sample {sample_idx}")
            return None
            
        # Create combined figure: 2x3 cameras on left, BEV on right
        fig = plt.figure(figsize=(20, 12))
        
        # Define layout: 2 rows x 4 columns (3 for cameras, 1 for BEV)
        # Top row cameras
        ax1 = plt.subplot2grid((2, 4), (0, 0))  # CAM_FRONT_LEFT
        ax2 = plt.subplot2grid((2, 4), (0, 1))  # CAM_FRONT
        ax3 = plt.subplot2grid((2, 4), (0, 2))  # CAM_FRONT_RIGHT
        
        # Bottom row cameras  
        ax4 = plt.subplot2grid((2, 4), (1, 0))  # CAM_BACK_LEFT
        ax5 = plt.subplot2grid((2, 4), (1, 1))  # CAM_BACK
        ax6 = plt.subplot2grid((2, 4), (1, 2))  # CAM_BACK_RIGHT
        
        # BEV plot spans both rows on the right
        ax_bev = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
        
        camera_axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        
        # Display camera images
        for i, (ax, cam_name) in enumerate(zip(camera_axes, self.camera_order)):
            if cam_name in camera_images:
                img = camera_images[cam_name]
                # Resize for display
                img_resized = cv2.resize(img, (640, 360))
                ax.imshow(img_resized)
                ax.set_title(f'{cam_name}', fontsize=11, fontweight='bold')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'{cam_name}\nNot Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.axis('off')
        
        # Create BEV plot
        ax_bev.set_xlim(-50, 50)
        ax_bev.set_ylim(-50, 50)
        ax_bev.set_aspect('equal')
        ax_bev.grid(True, alpha=0.3)
        ax_bev.set_xlabel('X (m)', fontsize=12)
        ax_bev.set_ylabel('Y (m)', fontsize=12)
        ax_bev.set_title('Bird\'s Eye View\nUniAD Results', fontsize=12, fontweight='bold')
        
        # Draw ego vehicle on BEV
        ego_rect = Rectangle((-1, -2), 2, 4, linewidth=3, edgecolor='red', facecolor='red', alpha=0.8)
        ax_bev.add_patch(ego_rect)
        ax_bev.text(0, -5, 'EGO', ha='center', fontsize=10, fontweight='bold', color='red')
        
        # Add planning trajectory to BEV
        if 'planning' in result and 'result_planning' in result['planning']:
            planning_result = result['planning']['result_planning']
            if 'sdc_traj' in planning_result:
                traj = planning_result['sdc_traj']
                if hasattr(traj, 'cpu'):
                    traj_np = traj.cpu().numpy().squeeze()
                else:
                    traj_np = traj
                
                if len(traj_np.shape) >= 2 and traj_np.shape[1] >= 2:
                    x_coords = traj_np[:, 0]
                    y_coords = traj_np[:, 1]
                    
                    ax_bev.plot(x_coords, y_coords, 'b-', linewidth=3, label='Planned Path', marker='o', markersize=5)
                    ax_bev.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
                    ax_bev.scatter(x_coords[-1], y_coords[-1], c='blue', s=100, marker='s', label='Goal', zorder=5)
        
        # Add detected objects to BEV
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            scores = result.get('scores_3d', None)
            labels = result.get('labels_3d', None)
            
            if hasattr(boxes, 'tensor'):
                box_tensor = boxes.tensor
                if hasattr(box_tensor, 'cpu'):
                    boxes_np = box_tensor.cpu().numpy()
                else:
                    boxes_np = box_tensor
                    
                class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
                colors = ['purple', 'brown', 'orange', 'yellow', 'pink', 
                         'gray', 'cyan', 'magenta', 'lightgreen', 'lightblue']
                
                for i, box in enumerate(boxes_np):
                    x, y, z, w, l, h, yaw = box[:7]
                    
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)
                    
                    corners = np.array([
                        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
                    ])
                    
                    rotated_corners = []
                    for corner in corners:
                        rotated_x = corner[0] * cos_yaw - corner[1] * sin_yaw + x
                        rotated_y = corner[0] * sin_yaw + corner[1] * cos_yaw + y
                        rotated_corners.append([rotated_x, rotated_y])
                    
                    rotated_corners = np.array(rotated_corners)
                    
                    label_idx = labels[i].item() if labels is not None else 0
                    label_name = class_names[label_idx] if label_idx < len(class_names) else 'unknown'
                    color = colors[label_idx] if label_idx < len(colors) else 'black'
                    
                    ax_bev.plot(rotated_corners[:, 0], rotated_corners[:, 1], 
                               color=color, linewidth=2, alpha=0.9)
                    ax_bev.fill(rotated_corners[:-1, 0], rotated_corners[:-1, 1], 
                               color=color, alpha=0.4)
                    
                    score_text = f"{scores[i]:.2f}" if scores is not None else ""
                    ax_bev.text(x, y, f'{label_name}\n{score_text}', ha='center', va='center', 
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
        
        # Add legend to BEV
        ax_bev.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Add overall title
        fig.suptitle(f'UniAD Inference Results - Sample {sample_idx}\nToken: {sample_token[:16]}...', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'enhanced_sample_{sample_idx:03d}.png')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved enhanced visualization: {output_path}")
        return output_path
        
    def visualize_all_samples(self):
        """Create enhanced visualizations for all samples"""
        output_files = []
        for i in range(len(self.results)):
            try:
                output_file = self.create_combined_visualization(i)
                if output_file:
                    output_files.append(output_file)
            except Exception as e:
                print(f"Error creating visualization for sample {i}: {e}")
                continue
                
        print(f"\nGenerated {len(output_files)} enhanced visualizations in {self.output_dir}")
        return output_files

def main():
    # Create enhanced visualizations
    visualizer = EnhancedVisualization('output/results_10.pkl')
    output_files = visualizer.visualize_all_samples()
    
    if output_files:
        print(f"\nEnhanced visualization complete! Generated {len(output_files)} images:")
        for f in output_files[:5]:
            print(f"  - {f}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more")

if __name__ == '__main__':
    main()