

  canbus 18-index format:
  - 0-2: position_xyz (x, y, z coordinates)
  - 3-6: quaternion_wxyz (orientation as quaternion w,x,y,z)
  - 7-9: linear_accel (acceleration in x,y,z)
  - 10-12: angular_velocity (rotation rates)
  - 13-15: velocity_xyz (velocity in x,y,z)
  - 16: yaw (heading angle)
  - 17: reserved (unused)

useful functions for canbus.pkl extraction:
def load_canbus_data(canbus_file):
    """Load CAN bus data from pickle file"""
    try:
        with open(canbus_file, 'rb') as f:
            data = pickle.load(f)
        return data['can_bus'], data['timestamps']
    except Exception as e:
        print(f"Error loading CAN bus data from {canbus_file}: {e}")
        return None, None

def find_canbus_file_for_bag(bag_name, canbus_dir):
    """Find corresponding CAN bus file for a bag file"""
    # Remove .bag extension and add _can_bus.pkl
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    canbus_file = os.path.join(canbus_dir, f"{base_name}_can_bus.pkl")

    if os.path.exists(canbus_file):
        return canbus_file
    else:
        print(f"Warning: CAN bus file not found: {canbus_file}")
        return None

def extract_car_state_from_canbus(canbus_dir, global_timestamps, bag_name):
    """Extract car state data from preprocessed CAN bus files using global timestamps"""

    # Find corresponding CAN bus file
    canbus_file = find_canbus_file_for_bag(bag_name, canbus_dir)
    if canbus_file is None:
        return None, None

    print(f"Loading CAN bus data from {canbus_file}...")

    # Load CAN bus data
    can_bus_data, canbus_timestamps = load_canbus_data(canbus_file)
    if can_bus_data is None:
        return None, None

    print(f"Loaded {len(can_bus_data)} CAN bus samples")

    # Extract position and velocity data aligned to global timestamps
    positions = []
    velocities = []
    valid_timestamps = []

    # CAN bus format: [0-2]: position (x,y,z), [13-15]: velocity (vel_x, vel_y, vel_z)
    for target_time in global_timestamps:
        # Find closest CAN bus timestamp within tolerance
        closest_time, closest_idx = find_closest_timestamp(float(target_time), canbus_timestamps, tolerance=0.5)

        if closest_time is not None:
            # Extract position [x, y] (ignoring z for 2D trajectory)
            pos_x = can_bus_data[closest_idx, 0]  # x position
            pos_y = can_bus_data[closest_idx, 1]  # y position
            positions.append([pos_x, pos_y])

            # Extract velocity [vel_x, vel_y]
            vel_x = can_bus_data[closest_idx, 13]  # velocity x
            vel_y = can_bus_data[closest_idx, 14]  # velocity y
            velocities.append([vel_x, vel_y])

            valid_timestamps.append(target_time)

    if len(positions) == 0:
        print("Warning: No valid position data found")
        return None, None

    print(f"Extracted {len(positions)} synchronized position/velocity samples")
    return np.array(positions), np.array(valid_timestamps)
