#!/bin/bash

# Example script to convert L2G data to UniAD format

# Configuration
CAMERA_CONFIG="sample_camera_config.py"
CANBUS_DIR="canbus/"

# Create output directory
mkdir -p data/

echo "Processing all CAN bus files in: $CANBUS_DIR"
shopt -s nullglob

# Iterate through all CAN bus files and extract scenes
for CANBUS_PKL in "$CANBUS_DIR"/*_can_bus.pkl; do
  if [ ! -f "$CANBUS_PKL" ]; then
    echo "No CAN bus files found in $CANBUS_DIR"
    exit 1
  fi

  # Extract base name from CAN bus filename
  CANBUS_FILENAME=$(basename "$CANBUS_PKL")
  BASE_NAME="${CANBUS_FILENAME%_can_bus.pkl}"  # Remove _can_bus.pkl suffix
  BAG_FILE="${BASE_NAME}.bag"  # Reconstruct bag filename

  echo "Processing CAN bus file: $CANBUS_FILENAME"
  echo "Extracted base name: $BASE_NAME"
  echo "Corresponding bag file: $BAG_FILE"

  echo "Extracting ALL timestamps from $CANBUS_PKL ..."
  TIMESTAMPS=$(python3 - "$CANBUS_PKL" <<'PY'
import sys, pickle
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
# Use ALL timestamps from this CAN bus file for complete scene
timestamps = data['timestamps']
print(f"Using all {len(timestamps)} frames for scene", file=sys.stderr)
print(' '.join(str(t) for t in timestamps))
PY
)

  OUTPUT_FILE="data/${BASE_NAME}_l2g_converted_infos.pkl"
  SCENE_TOKEN="scene_${BASE_NAME}"

  echo "Converting scene: $SCENE_TOKEN -> $OUTPUT_FILE"
  python3 convert_l2g_to_uniad.py \
    --camera_config "$CAMERA_CONFIG" \
    --canbus_dir "$CANBUS_DIR" \
    --output_file "$OUTPUT_FILE" \
    --bag_name "$BAG_FILE" \
    --timestamps $TIMESTAMPS \
    --scene_token "$SCENE_TOKEN"

  echo "Verifying output: $OUTPUT_FILE"
  python3 - "$OUTPUT_FILE" <<'PY'
import sys, pickle
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print(f'Scene: {data["metadata"]["scene_token"]}')
print(f'Samples: {len(data.get("infos", []))}')
print(f'Sequences: {len(data.get("temporal_sequences", []))}')
PY
  echo "---"
done

shopt -u nullglob

echo "All scenes processed successfully!"