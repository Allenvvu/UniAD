#!/usr/bin/env python3
"""
run_nuscenes_model.py

Usage:
    python run_nuscenes_model.py \
      --data-root data/itri/hct_train \
      --checkpoint /path/to/checkpoint.pth \
      --out-dir outputs \
      --device cuda:0 \
      --num-temporal 3

Notes:
- Replace build_model() with your model builder (returning an nn.Module).
- Camera config parsing expects a simple key:value or YAML-like format in camera_config.md.
- CAN-bus reading tries rosbag first; if not available, expects per-timestamp npz/csv or a single poses.npz file.
"""

import argparse
import os
import glob
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ---------- Configurable defaults ----------
CAMERA_NAMES_DEFAULT = [
    "lucid_cameras_x00.gige_100_f_hdr.h265",    # front
    "lucid_cameras_x00.gige_60_b_hdr.h265",     # back  
    "lucid_cameras_x01.gige_100_fl_hdr.h265",   # front_left
    "lucid_cameras_x01.gige_100_fr_hdr.h265"    # front_right
]
MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR order, pixel scale 0-255
STD_BGR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
PAD_DIVISOR = 32
# -------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="root: data/itri/hct_train")
    p.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path (.pth)")
    p.add_argument("--out-dir", required=True, help="where inference outputs are written")
    p.add_argument("--device", default="cuda:0", help="torch device")
    p.add_argument("--num-temporal", type=int, default=3, help="temporal window size (including current). e.g. 3 => t-2,t-1,t")
    p.add_argument("--model-config", default=None, help="optional model config file (if needed by build_model)")
    p.add_argument("--camera-config", default="image/camera_config.md", help="path relative to data-root")
    p.add_argument("--timestamps", default="timestamps/continuous_timestamps.pkl", help="path relative to data-root")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ---------------- IO helpers ----------------
def load_timestamps(pkl_path: str) -> List[float]:
    with open(pkl_path, "rb") as f:
        ts = pickle.load(f)
    # Expect list-like of timestamps (float or int); convert to str-friendly form
    return list(ts)

def find_image_files_for_timestamp(root: Path, timestamp, cam_names=CAMERA_NAMES_DEFAULT) -> Dict[str, Path]:
    """
    Tries several common filename layouts and returns mapping cam_name -> full path.
    Candidate patterns attempted (in order):
      1) <root>/image/<cam_name>/<timestamp>.*  (per-camera subfolders) - PRIMARY PATTERN
      2) <root>/image/<timestamp>_<cam_name>.*   (timestamp prefix)
      3) <root>/image/<cam_name>_<timestamp>.*   (camera prefix)
      4) <root>/image/<timestamp>.*  (if exactly 6 files present, try to match by camera names contained)
    """
    img_root = root / "image"
    found = {}
    ts_str = str(timestamp)
    
    # pattern 1: <root>/image/<cam_name>/<timestamp>.* (per-camera subfolders)
    # This is the primary pattern for the ITRI dataset
    for cam in cam_names:
        # try any common extension
        for ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
            p1 = img_root / cam / f"{ts_str}.{ext}"
            if p1.exists():
                found[cam] = p1
                break
        if cam in found:
            continue
    
    # If we found all cameras with pattern 1, return immediately
    if len(found) == len(cam_names):
        return found

    # pattern 2 & 3: fallback patterns for other datasets
    for cam in cam_names:
        if cam in found: continue
        # timestamp_cam.*
        for ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
            p2 = img_root / f"{ts_str}_{cam}.{ext}"
            p3 = img_root / f"{cam}_{ts_str}.{ext}"
            if p2.exists():
                found[cam] = p2
                break
            if p3.exists():
                found[cam] = p3
                break

    # pattern 4: if still missing, try to find files with timestamp anywhere in name and attempt to assign
    if len(found) < len(cam_names):
        candidates = list(img_root.glob(f"*{ts_str}*"))
        # if there are exactly 6 candidates, try to map by camera substrings
        if len(candidates) >= len(cam_names):
            for cam in cam_names:
                if cam in found: continue
                for c in candidates:
                    name = c.name.lower()
                    if cam.replace("_", "") in name.replace("_","") or cam.split("_")[0] in name:
                        found[cam] = c
                        break

    # final check: any missing remains -> leave missing
    return found

def parse_camera_config(camera_config_path: str) -> Dict[str, Dict]:
    """
    Parse a simple camera_config.md with lines like:
      camera: front
      intrinsic: [fx, 0, cx; 0, fy, cy; 0,0,1]  # or comma-separated row-major
      extrinsic: [4x4 row-major flatten]
    Returns mapping: camera_name -> {"intrinsic": np.array(3x3), "extrinsic": np.array(4x4)}
    If parsing fails, returns empty dict.
    """
    config = {}
    p = Path(camera_config_path)
    if not p.exists():
        return config

    # Simple parser: split into sections by camera name
    text = p.read_text()
    # attempt YAML-like parse
    try:
        import yaml
        parsed = yaml.safe_load(text)
        # Expect parsed to be dict of cameras -> dicts
        if isinstance(parsed, dict):
            for k,v in parsed.items():
                if not isinstance(v, dict):
                    continue
                intr = v.get("intrinsic")
                ext = v.get("extrinsic")
                if intr is not None:
                    intr = np.array(intr).reshape((3,3))
                if ext is not None:
                    ext = np.array(ext).reshape((4,4))
                config[k] = {"intrinsic": intr, "extrinsic": ext}
            return config
    except Exception:
        pass

    # Fallback: crude line scanning
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cur_cam = None
    for ln in lines:
        if ln.lower().startswith("camera") or ln.lower().startswith("name:"):
            # camera name line
            parts = ln.split(":")
            if len(parts)>=2:
                cur_cam = parts[1].strip()
                config[cur_cam] = {}
        elif "intrin" in ln.lower() and cur_cam:
            # extract numbers
            nums = [float(s) for s in ln.replace("["," ").replace("]"," ").replace(";",",").replace(","," ").split() if _is_number(s)]
            if len(nums) >= 9:
                config[cur_cam]["intrinsic"] = np.array(nums[:9]).reshape(3,3)
        elif "extrin" in ln.lower() and cur_cam:
            nums = [float(s) for s in ln.replace("["," ").replace("]"," ").replace(";",",").replace(","," ").split() if _is_number(s)]
            if len(nums) >= 16:
                config[cur_cam]["extrinsic"] = np.array(nums[:16]).reshape(4,4)
    return config

def _is_number(s):
    try:
        float(s)
        return True
    except:
        return False

# ---------------- image preprocessing ----------------
def pad_to_divisor(img: np.ndarray, divisor: int = PAD_DIVISOR) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Pad H,W to next multiple of divisor. Return padded image and padding (pad_h, pad_w)."""
    h, w = img.shape[:2]
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    pad_h = new_h - h
    pad_w = new_w - w
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
    return padded, (pad_h, pad_w)

def preprocess_image_bgr(img_bgr: np.ndarray, mean_bgr=MEAN_BGR, std_bgr=STD_BGR, to_tensor=True):
    """
    Input: BGR uint8 HxWx3
    Output: torch.FloatTensor CHW, normalized by mean/std, padded to divisible size.
    """
    img = img_bgr.astype(np.float32)
    img = (img - mean_bgr) / std_bgr
    padded, pad = pad_to_divisor(img, PAD_DIVISOR)
    img_chw = padded.transpose(2,0,1)  # C,H,W
    if to_tensor:
        return torch.from_numpy(img_chw).float(), pad
    else:
        return img_chw, pad

# ---------------- canbus / ego pose reading ----------------
def load_ego_poses_from_rosbag(canbus_folder: Path, timestamps: List[float]) -> Dict[float, np.ndarray]:
    """
    Try to read ego pose (4x4) per timestamp from rosbag .bag files in canbus_folder.
    This function attempts to import rosbag (ROS python). If unavailable, it'll raise ImportError.
    Return mapping timestamp -> 4x4 homogeneous transform (world <- ego) or similar (consistent).
    """
    try:
        import rosbag  # type: ignore
    except Exception as e:
        raise ImportError("rosbag not available in this environment.") from e

    # naive loader: scan .bag files and read /tf or custom topics to extract pose; this will be model/repo-specific.
    # For now, we provide a placeholder implementation; users should edit to match their bag topics.
    poses = {}
    bag_files = list(canbus_folder.glob("*.bag"))
    if not bag_files:
        return poses

    for bf in bag_files:
        bag = rosbag.Bag(str(bf))
        # attempt topics
        for topic, msg, t in bag.read_messages():
            # Replace these checks with the actual message/topic types in your bags.
            if topic.endswith("/vehicle/pose") or topic.endswith("/tf") or topic.endswith("/odometry"):
                # user must adapt parsing here...
                pass
        bag.close()
    return poses

def load_ego_poses_fallback(canbus_folder: Path) -> Dict[float, np.ndarray]:
    """
    Fallback: look for a poses.npz or per-timestamp npz/csv files exported from the bag.
    Expected shape for each entry: 4x4 matrix (world <- ego) and keyed by timestamp.
    """
    poses = {}
    # try poses.npz
    pnp = canbus_folder / "poses.npz"
    if pnp.exists():
        data = np.load(str(pnp), allow_pickle=True)
        for k in data.files:
            try:
                ts = float(k)
                poses[ts] = np.array(data[k])
            except:
                continue
        return poses

    # try any .npz files
    for f in canbus_folder.glob("*.npz"):
        try:
            data = np.load(str(f), allow_pickle=True)
            for k in data.files:
                ts = float(k)
                poses[ts] = np.array(data[k])
        except Exception:
            continue

    # try csv with columns: timestamp, x,y,z, qx,qy,qz,qw  (make 4x4)
    for f in canbus_folder.glob("*.csv"):
        try:
            arr = np.loadtxt(str(f), delimiter=",")
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            for row in arr:
                ts = float(row[0])
                x,y,z,qx,qy,qz,qw = row[1:8]
                R = quat_to_rot_matrix([qw, qx, qy, qz])  # convert
                T = np.eye(4, dtype=np.float32)
                T[:3,:3] = R
                T[:3,3] = [x,y,z]
                poses[ts] = T
        except Exception:
            continue
    return poses

def quat_to_rot_matrix(q):
    # q as [w, x, y, z] or [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    # normalize
    n = np.linalg.norm([qw,qx,qy,qz])
    if n == 0:
        return np.eye(3)
    qw,qx,qy,qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy- qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx + qy*qy)]
    ], dtype=np.float32)
    return R

# ---------------- model building/loading (USER: replace this) ----------------
def build_model(model_config=None):
    """
    Placeholder. **REPLACE** this with your project's model constructor.
    Example for a typical repo:
      from mmdet3d.models import build_detector
      cfg = mmcv.Config.fromfile(model_config)
      model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    Ensure model.eval() after loading weights.
    """
    raise NotImplementedError("Replace build_model() with your model constructor.")

def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ck, dict) and "state_dict" in ck:
        st = ck["state_dict"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        st = ck["model_state_dict"]
    else:
        # maybe checkpoint is a raw state_dict
        st = ck
    # attempt to adapt keys (strip 'module.' if present)
    new_st = {}
    for k,v in st.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_st[nk] = v
    model.load_state_dict(new_st, strict=False)
    return model

# ---------------- assemble inputs & run ----------------
def compose_input_for_timestamp(root: Path, ts_idx: int, timestamps: List, num_temporal: int,
                                cam_names: List[str], cam_config: Dict,
                                poses_map: Dict[float, np.ndarray], device: str, verbose=False):
    """
    Compose a dict of tensors expected by the model:
      - images: Tensor [B=1, num_cams, C, H, W]
      - intrinsics: list or tensor of per-camera intrinsics (num_cams x 3x3)
      - extrinsics: list or tensor of per-camera extrinsics (num_cams x 4x4)
      - ego_trans: list/array of relative transforms for temporal frames (num_temporal x 4x4)
    """
    # temporal indexes: take last num_temporal timestamps ending at current index
    start_idx = max(0, ts_idx - (num_temporal - 1))
    chosen_idxs = list(range(start_idx, ts_idx+1))
    if len(chosen_idxs) < num_temporal:
        # pad by repeating earliest available
        chosen_idxs = [chosen_idxs[0]]*(num_temporal - len(chosen_idxs)) + chosen_idxs

    imgs_all = []
    intrinsics = []
    extrinsics = []
    H,W = None, None
    # For each camera, load current timestamp image and preprocess (we assume same resolution)
    # We'll use the current (last) timestamp images for spatial inputs; for temporal BEV alignment we pass ego transforms.
    cur_ts = timestamps[ts_idx]
    img_paths = find_image_files_for_timestamp(root, cur_ts, cam_names)
    if verbose:
        print(f"[INFO] Found image mapping for ts {cur_ts}: {img_paths}")
    for cam in cam_names:
        p = img_paths.get(cam)
        if p is None or not p.exists():
            raise FileNotFoundError(f"Missing image for camera '{cam}' at timestamp {cur_ts}. Tried: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2 failed to read {p}")
        if H is None:
            H,W = img.shape[:2]
        # Preprocess (BGR assumed)
        timg, pad = preprocess_image_bgr(img)
        imgs_all.append(timg)  # C,H',W' tensors

        # camera params from config if present else identity placeholders
        cam_cfg = cam_config.get(cam, {})
        intr = cam_cfg.get("intrinsic", np.eye(3, dtype=np.float32))
        ext = cam_cfg.get("extrinsic", np.eye(4, dtype=np.float32))
        intrinsics.append(intr.astype(np.float32))
        extrinsics.append(ext.astype(np.float32))

    # stack per camera into tensor shape (num_cams, C, H', W')
    imgs_stack = torch.stack(imgs_all, dim=0).unsqueeze(0)  # [1, num_cams, C, H, W]
    imgs_stack = imgs_stack.to(device)

    # intrinsics / extrinsics -> tensors
    intr_t = torch.from_numpy(np.stack(intrinsics)).float().to(device)   # (num_cams,3,3)
    ext_t = torch.from_numpy(np.stack(extrinsics)).float().to(device)   # (num_cams,4,4)

    # Compose ego transforms for chosen temporal frames
    ego_trans_list = []
    # We want transforms mapping each temporal pose into the current frame (relative transforms)
    # Define T_world_from_ego(t) = poses_map[t] (4x4)
    cur_pose = poses_map.get(float(cur_ts), None)
    if cur_pose is None:
        raise KeyError(f"Current timestamp pose {cur_ts} not found in poses_map.")
    for idx in chosen_idxs:
        t = timestamps[idx]
        p = poses_map.get(float(t), None)
        if p is None:
            # fallback: use current pose (no motion)
            p = cur_pose
        # compute relative transform from that frame to current: T_rel = T_cur * inv(T_past)
        T_past_inv = np.linalg.inv(p)
        T_rel = cur_pose @ T_past_inv
        ego_trans_list.append(T_rel.astype(np.float32))
    ego_trans_np = np.stack(ego_trans_list, axis=0)  # (num_temporal,4,4)
    ego_trans_t = torch.from_numpy(ego_trans_np).float().to(device)

    inp = {
        "images": imgs_stack,            # [1, num_cams, C, H, W]
        "intrinsics": intr_t,           # [num_cams, 3, 3]
        "extrinsics": ext_t,            # [num_cams, 4, 4]
        "ego_transforms": ego_trans_t,  # [num_temporal,4,4]
        "timestamp": cur_ts,
    }
    return inp

def run_inference_for_all(args):
    root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # read timestamps
    ts_path = Path(args.data_root) / args.timestamps
    if not ts_path.exists():
        raise FileNotFoundError(f"Timestamps file not found: {ts_path}")
    timestamps = load_timestamps(str(ts_path))

    # camera config
    cam_cfg_path = Path(args.data_root) / args.camera_config
    cam_config = parse_camera_config(str(cam_cfg_path))
    if not cam_config:
        print("[WARN] camera config could not be parsed or is empty. Using identity intrinsics/extrinsics placeholders.")
    # load ego poses
    canbus_folder = Path(args.data_root) / "canbus"
    poses = {}
    try:
        poses = load_ego_poses_from_rosbag(canbus_folder, timestamps)
        if not poses:
            raise RuntimeError("rosbag loader returned empty poses.")
    except Exception as e:
        if args.verbose:
            print("[INFO] rosbag loader failed or not available:", e)
        poses = load_ego_poses_fallback(canbus_folder)
        if not poses:
            raise RuntimeError("No ego poses found. Please export poses into poses.npz or csv in canbus folder (keys = timestamps).")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # build model - user should replace build_model
    model = build_model(args.model_config)
    model.to(device)
    model.eval()
    # load checkpoint
    load_checkpoint_to_model(model, args.checkpoint, map_location=device)

    # loop timestamps
    for idx, ts in enumerate(timestamps):
        try:
            inp = compose_input_for_timestamp(Path(args.data_root), idx, timestamps, args.num_temporal,
                                              CAMERA_NAMES_DEFAULT, cam_config, poses, device, verbose=args.verbose)
        except Exception as e:
            print(f"[ERROR] Skipping timestamp {ts} due to: {e}")
            continue

        # model forward - exact call depends on model API. Common patterns:
        # out = model(**inp)  OR  out = model(inp["images"], intrinsics=..., extrinsics=..., ego=...)
        # We attempt several patterns with fallbacks; replace if your model uses different API.
        with torch.no_grad():
            output = None
            try:
                output = model(inp)  # if your model accepts a dict
            except Exception:
                try:
                    output = model(inp["images"], intrinsics=inp["intrinsics"], extrinsics=inp["extrinsics"],
                                   ego_transforms=inp["ego_transforms"])
                except Exception:
                    # last-ditch: try images only
                    try:
                        output = model(inp["images"])
                    except Exception as e:
                        print(f"[ERROR] Model forward failed for ts {ts}: {e}")
                        output = None

        # Save outputs
        save_path = out_dir / f"pred_{str(ts)}.npz"
        try:
            if torch.is_tensor(output):
                out_np = output.cpu().numpy()
                np.savez_compressed(str(save_path), output=out_np)
            elif isinstance(output, dict):
                # convert tensors to numpy where possible
                out_save = {}
                for k,v in output.items():
                    if torch.is_tensor(v):
                        out_save[k] = v.detach().cpu().numpy()
                    else:
                        out_save[k] = v
                np.savez_compressed(str(save_path), **out_save)
            elif output is None:
                # blank marker
                np.savez_compressed(str(save_path), output="MODEL_FORWARD_FAILED")
            else:
                # unknown type -> try np.save
                np.savez_compressed(str(save_path), output=output)
        except Exception as e:
            print(f"[WARN] Could not save output for {ts}: {e}")

        if args.verbose:
            print(f"[INFO] Inference done for ts {ts} -> {save_path}")

    print("[DONE] All timestamps processed.")

# -------------- main --------------
def main():
    args = parse_args()
    run_inference_for_all(args)

if __name__ == "__main__":
    main()
