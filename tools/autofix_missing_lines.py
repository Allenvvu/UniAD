#!/usr/bin/env python3

import json
import math
import os
import shutil
import sys
from typing import Dict, List, Tuple, Optional

"""
Autofix missing line references for traffic_light records in nuScenes expansion maps.
Strategy:
- Build indices of polygon, line, node, and traffic_light.
- For each traffic_light with line_token missing from line layer:
  - If traffic_light has pose {tx, ty}: find 2 nearest distinct nodes by (x, y).
    - Create a new line record with that token and node_tokens = [nodeA, nodeB].
  - Else: disable the traffic_light by setting a private flag `_disabled`: true (or remove it if --remove is set).
- Writes a backup of the original JSON before modification.
- Preserves formatting minimally (default json.dump with indent=2).

Note: This creates minimal 2-node polylines; adjust if your consumers expect longer geometry.
"""


def distance2(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by)


def find_two_nearest_nodes(nodes: List[Dict], x: float, y: float) -> Optional[Tuple[str, str]]:
    if not nodes:
        return None
    # Find two nearest distinct nodes by squared distance
    best1 = (None, float('inf'))  # (token, dist2)
    best2 = (None, float('inf'))
    for n in nodes:
        try:
            nx = float(n['x'])
            ny = float(n['y'])
        except Exception:
            continue
        d2 = distance2(nx, ny, x, y)
        if d2 < best1[1]:
            best2 = best1
            best1 = (n['token'], d2)
        elif d2 < best2[1] and n['token'] != best1[0]:
            best2 = (n['token'], d2)
    if best1[0] is None or best2[0] is None:
        return None
    return best1[0], best2[0]


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, obj: Dict) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def ensure_backup(path: str) -> str:
    backup = path + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    return backup


def autofix(json_path: str, remove_if_no_fix: bool = False) -> Dict:
    data = load_json(json_path)

    line_by_token = {rec['token']: rec for rec in data.get('line', [])}
    nodes = data.get('node', [])
    node_by_token = {n['token']: n for n in nodes}

    traffic_lights = data.get('traffic_light', [])

    created = 0
    disabled = 0
    removed = 0

    for tl in traffic_lights[:]:  # copy for safe removal
        lt = tl.get('line_token')
        if not lt:
            continue
        if lt in line_by_token:
            continue
        pose = tl.get('pose') or {}
        tx = pose.get('tx')
        ty = pose.get('ty')
        if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
            nn = find_two_nearest_nodes(nodes, float(tx), float(ty))
            if nn is not None:
                node_a, node_b = nn
                # Create new line record
                new_line = {
                    'token': lt,
                    'node_tokens': [node_a, node_b]
                }
                data.setdefault('line', []).append(new_line)
                line_by_token[lt] = new_line
                created += 1
                continue
        # Could not create geometry; disable or remove
        if remove_if_no_fix:
            traffic_lights.remove(tl)
            removed += 1
        else:
            tl['_disabled'] = True
            disabled += 1

    # Save back
    ensure_backup(json_path)
    save_json(json_path, data)

    return {
        'created_lines': created,
        'disabled_lights': disabled,
        'removed_lights': removed,
        'total_traffic_lights': len(data.get('traffic_light', []))
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Autofix missing traffic_light line references in nuScenes expansion JSON')
    parser.add_argument('json_path', help='Path to expansion JSON')
    parser.add_argument('--remove-unfixable', action='store_true', help='Remove traffic_lights that cannot be auto-fixed (default: disable with _disabled=true)')
    args = parser.parse_args()

    stats = autofix(args.json_path, remove_if_no_fix=args.remove_unfixable)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
