import pickle
import matplotlib.pyplot as plt
import torch

# Load the data
with open('output/l2g_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Get sample 1 (index 1)
entry = data['bbox_results'][1]

# Extract trajectory data (shape: [1, 6, 12, 5])
traj = entry['traj'][0]  # Remove batch dimension -> [6, 12, 5]
scores = entry['traj_scores'][0]  # Remove batch dimension -> [6]

# Convert to probabilities
probs = torch.softmax(scores, dim=0)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot each trajectory mode
for i in range(6):
    x_coords = traj[i, :, 0].numpy()  # X coordinates over 12 timesteps
    y_coords = traj[i, :, 1].numpy()  # Y coordinates over 12 timesteps

    # Plot trajectory with transparency based on probability
    alpha = float(probs[i])
    plt.plot(x_coords, y_coords, 'o-', alpha=alpha*2, linewidth=2,
             label=f'Mode {i} (p={probs[i]:.3f})')

    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 's', markersize=8, alpha=alpha*2)  # Start
    plt.plot(x_coords[-1], y_coords[-1], '^', markersize=8, alpha=alpha*2)  # End

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Sample 1: Multi-modal Trajectory Predictions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# Save the plot
plt.savefig('sample1_trajectories.png', dpi=150, bbox_inches='tight')
print("Trajectory visualization saved as 'sample1_trajectories.png'")
