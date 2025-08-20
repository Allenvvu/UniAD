#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

class SimpleVisualization:
    def __init__(self, results_path, output_dir='output/simple_viz'):
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Loaded {len(self.results)} inference results")
        
    def visualize_bev_with_planning(self, sample_idx=0):
        """Create a simple BEV visualization with planning trajectory and detected objects"""
        if sample_idx >= len(self.results):
            print(f"Sample {sample_idx} out of range")
            return None
            
        result = self.results[sample_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set up BEV coordinate system (100m x 100m centered on ego)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'UniAD BEV Visualization - Sample {sample_idx}')
        
        # Draw ego vehicle at center (0, 0)
        ego_rect = Rectangle((-1, -2), 2, 4, linewidth=3, edgecolor='red', facecolor='red', alpha=0.8)
        ax.add_patch(ego_rect)
        ax.text(0, -5, 'EGO', ha='center', fontsize=12, fontweight='bold', color='red')
        
        # Extract and draw planning trajectory if available
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
                    ax.plot(x_coords, y_coords, 'b-', linewidth=3, label='Planned Path', marker='o', markersize=5)
                    
                    # Highlight start and end points
                    ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
                    ax.scatter(x_coords[-1], y_coords[-1], c='blue', s=100, marker='s', label='Goal', zorder=5)
                    
                    print(f"Planning trajectory: {len(x_coords)} points")
                    
        # Extract and draw detected 3D bounding boxes
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
                    
                # NuScenes class names
                class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
                colors = ['purple', 'brown', 'orange', 'yellow', 'pink', 
                         'gray', 'cyan', 'magenta', 'lightgreen', 'lightblue']
                
                for i, box in enumerate(boxes_np):
                    # Box format: [x, y, z, w, l, h, yaw, vx, vy]
                    x, y, z, w, l, h, yaw = box[:7]
                    
                    # Create rotated rectangle for BEV
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)
                    
                    # Box corners in local coordinates
                    corners = np.array([
                        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
                    ])
                    
                    # Rotate and translate
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
                           color=color, linewidth=2, alpha=0.8)
                    ax.fill(rotated_corners[:-1, 0], rotated_corners[:-1, 1], 
                           color=color, alpha=0.3)
                    
                    # Add label with confidence score
                    score_text = f"{scores[i]:.2f}" if scores is not None else ""
                    ax.text(x, y, f'{label_name}\n{score_text}', ha='center', va='center', 
                           fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                    
                print(f"Drew {len(boxes_np)} detected objects")
                
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'sample_{sample_idx:03d}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {output_path}")
        return output_path
        
    def visualize_all_samples(self):
        """Visualize all samples"""
        output_files = []
        for i in range(len(self.results)):
            try:
                output_file = self.visualize_bev_with_planning(i)
                if output_file:
                    output_files.append(output_file)
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")
                continue
                
        print(f"\nGenerated {len(output_files)} visualizations in {self.output_dir}")
        return output_files

def main():
    # Visualize the 10 inference results
    visualizer = SimpleVisualization('output/results_10.pkl')
    output_files = visualizer.visualize_all_samples()
    
    if output_files:
        print(f"\nVisualization complete! Generated {len(output_files)} images:")
        for f in output_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more")

if __name__ == '__main__':
    main()