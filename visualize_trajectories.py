#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_trajectories(data, num_samples=10, save_plots=True):
    """Visualize planning trajectories from l2g_results"""

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot individual trajectories
    for i in range(min(6, len(axes))):
        ax = axes[i]

        # Plot multiple samples
        for sample_idx in range(min(num_samples, len(data['bbox_results']))):
            entry = data['bbox_results'][sample_idx]
            traj = entry['planning_traj'][0].cpu().numpy()  # Shape: [6, 2]

            # Extract x, y coordinates
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]

            # Plot trajectory with different colors for different samples
            color = plt.cm.tab10(sample_idx % 10)
            ax.plot(x_coords, y_coords, 'o-', color=color, alpha=0.7,
                   linewidth=2, markersize=4, label=f'Sample {sample_idx}')

            # Mark start and end points
            ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, alpha=0.8)  # Start
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, alpha=0.8)  # End

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Planning Trajectories - View {i+1}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_plots:
        plt.savefig('planning_trajectories_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Create a detailed single plot
    plt.figure(figsize=(12, 8))

    for sample_idx in range(min(num_samples, len(data['bbox_results']))):
        entry = data['bbox_results'][sample_idx]
        traj = entry['planning_traj'][0].cpu().numpy()

        x_coords = traj[:, 0]
        y_coords = traj[:, 1]

        color = plt.cm.viridis(sample_idx / num_samples)
        plt.plot(x_coords, y_coords, 'o-', color=color, alpha=0.8,
                linewidth=2, markersize=6, label=f'Sample {sample_idx}')

        # Add arrows to show direction
        for j in range(len(x_coords)-1):
            dx = x_coords[j+1] - x_coords[j]
            dy = y_coords[j+1] - y_coords[j]
            plt.arrow(x_coords[j], y_coords[j], dx*0.8, dy*0.8,
                     head_width=0.2, head_length=0.2, fc=color, ec=color, alpha=0.6)

    # Mark origin
    plt.plot(0, 0, 'ks', markersize=10, label='Origin (SDC start)')

    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title(f'Planning Trajectories - First {num_samples} Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')

    if save_plots:
        plt.savefig('planning_trajectories_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Statistics plot
    plt.figure(figsize=(12, 4))

    # Collect all trajectories
    all_trajectories = []
    for i in range(len(data['bbox_results'])):
        traj = data['bbox_results'][i]['planning_traj'][0].cpu().numpy()
        all_trajectories.append(traj)

    all_trajectories = np.array(all_trajectories)  # Shape: [N, 6, 2]

    # Plot 1: X displacement over time
    plt.subplot(1, 3, 1)
    mean_x = np.mean(all_trajectories[:, :, 0], axis=0)
    std_x = np.std(all_trajectories[:, :, 0], axis=0)
    timesteps = range(1, 7)

    plt.plot(timesteps, mean_x, 'b-o', linewidth=2, markersize=6)
    plt.fill_between(timesteps, mean_x - std_x, mean_x + std_x, alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('X displacement (m)')
    plt.title('X Displacement Statistics')
    plt.grid(True, alpha=0.3)

    # Plot 2: Y displacement over time
    plt.subplot(1, 3, 2)
    mean_y = np.mean(all_trajectories[:, :, 1], axis=0)
    std_y = np.std(all_trajectories[:, :, 1], axis=0)

    plt.plot(timesteps, mean_y, 'r-o', linewidth=2, markersize=6)
    plt.fill_between(timesteps, mean_y - std_y, mean_y + std_y, alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Y displacement (m)')
    plt.title('Y Displacement Statistics')
    plt.grid(True, alpha=0.3)

    # Plot 3: Distance from origin
    plt.subplot(1, 3, 3)
    distances = np.sqrt(all_trajectories[:, :, 0]**2 + all_trajectories[:, :, 1]**2)
    mean_dist = np.mean(distances, axis=0)
    std_dist = np.std(distances, axis=0)

    plt.plot(timesteps, mean_dist, 'g-o', linewidth=2, markersize=6)
    plt.fill_between(timesteps, mean_dist - std_dist, mean_dist + std_dist, alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Distance from origin (m)')
    plt.title('Distance Statistics')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig('trajectory_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Visualized trajectories for {len(data['bbox_results'])} samples")
    print(f"Mean final distance: {mean_dist[-1]:.2f} ± {std_dist[-1]:.2f} meters")
    print(f"Mean final X: {mean_x[-1]:.2f} ± {std_x[-1]:.2f} meters")
    print(f"Mean final Y: {mean_y[-1]:.2f} ± {std_y[-1]:.2f} meters")

if __name__ == "__main__":
    # Load the data
    with open('output/l2g_results.pkl', 'rb') as f:
        data = pickle.load(f)

    print("Creating trajectory visualizations...")
    visualize_trajectories(data, num_samples=20, save_plots=True)