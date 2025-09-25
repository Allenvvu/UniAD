#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, List, Set, Tuple

"""
Validator for nuScenes expansion map JSON cross-layer references.
Checks that all tokens referenced by non-geometric layers exist in their geometric layers.
- polygon.exterior_node_tokens -> node.token
- non_geometric_polygon_layers polygon_token -> polygon.token
- non_geometric_line_layers line_token -> line.token
- lane.{left_lane_divider_segments,right_lane_divider_segments}.node_token -> node.token
- stop_line: resolve cues for types and ensure referenced records exist
- traffic_light.items.to_road_block_tokens -> road_block.token (best-effort)

This script does not validate geometry or coordinates; only token cross-references.
"""

NON_GEOMETRIC_POLYGON_LAYERS = [
    'drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
    'walkway', 'stop_line', 'carpark_area'
]
NON_GEOMETRIC_LINE_LAYERS = [
    'road_divider', 'lane_divider', 'traffic_light'
]
GEOMETRIC_LAYERS = ['polygon', 'line', 'node']


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def index_tokens(records: List[Dict]) -> Set[str]:
    return {rec['token'] for rec in records}


def get_layer(json_obj: Dict, name: str) -> List[Dict]:
    layer = json_obj.get(name)
    if layer is None:
        return []
    return layer


def report_missing(layer: str, key: str, token: str, ctx: Dict, problems: List[str]) -> None:
    problems.append(f"Missing {key} '{token}' referenced from {layer} token={ctx.get('token','<no-token>')}.")


def validate(json_obj: Dict) -> List[str]:
    problems: List[str] = []

    # Build indices
    polygon = get_layer(json_obj, 'polygon')
    line = get_layer(json_obj, 'line')
    node = get_layer(json_obj, 'node')

    polygon_tokens = index_tokens(polygon)
    line_tokens = index_tokens(line)
    node_tokens = index_tokens(node)

    # 1) polygon.exterior_node_tokens -> node
    for poly in polygon:
        for nt in poly.get('exterior_node_tokens', []):
            if nt not in node_tokens:
                report_missing('polygon', 'node', nt, poly, problems)
        for hole in poly.get('holes', []):
            for nt in hole:
                if nt not in node_tokens:
                    report_missing('polygon.hole', 'node', nt, poly, problems)

    # 2) non-geometric polygon-layers' polygon_token -> polygon
    for layer in NON_GEOMETRIC_POLYGON_LAYERS:
        for rec in get_layer(json_obj, layer):
            pt = rec.get('polygon_token')
            if pt and pt not in polygon_tokens:
                report_missing(layer, 'polygon', pt, rec, problems)

    # 3) non-geometric line-layers' line_token -> line
    for layer in NON_GEOMETRIC_LINE_LAYERS:
        for rec in get_layer(json_obj, layer):
            lt = rec.get('line_token')
            if lt and lt not in line_tokens:
                report_missing(layer, 'line', lt, rec, problems)

    # 4) lane divider segments' node_token -> node
    for lane_rec in get_layer(json_obj, 'lane'):
        for seg in lane_rec.get('left_lane_divider_segments', []):
            nt = seg.get('node_token')
            if nt and nt not in node_tokens:
                report_missing('lane.left_lane_divider_segments', 'node', nt, lane_rec, problems)
        for seg in lane_rec.get('right_lane_divider_segments', []):
            nt = seg.get('node_token')
            if nt and nt not in node_tokens:
                report_missing('lane.right_lane_divider_segments', 'node', nt, lane_rec, problems)

    # 5) stop_line cue resolution best-effort
    for sl in get_layer(json_obj, 'stop_line'):
        sl_type = sl.get('stop_line_type')
        if sl_type in ['PED_CROSSING', 'TURN_STOP']:
            for tok in sl.get('ped_crossing_tokens', []):
                # ped_crossing is a non-geometric polygon-layer
                # Ensure record exists by token
                found = any(rec.get('token') == tok for rec in get_layer(json_obj, 'ped_crossing'))
                if not found:
                    report_missing('stop_line.ped_crossing_tokens', 'ped_crossing', tok, sl, problems)
        elif sl_type in ['STOP_SIGN', 'YIELD']:
            pass
        elif sl_type == 'TRAFFIC_LIGHT':
            for tok in sl.get('traffic_light_tokens', []):
                found = any(rec.get('token') == tok for rec in get_layer(json_obj, 'traffic_light'))
                if not found:
                    report_missing('stop_line.traffic_light_tokens', 'traffic_light', tok, sl, problems)

    # 6) traffic_light.items.to_road_block_tokens -> road_block
    road_block_tokens = index_tokens(get_layer(json_obj, 'road_block'))
    for tl in get_layer(json_obj, 'traffic_light'):
        for item in tl.get('items', []):
            for tok in item.get('to_road_block_tokens', []):
                if tok not in road_block_tokens:
                    report_missing('traffic_light.items.to_road_block_tokens', 'road_block', tok, tl, problems)

    return problems


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/validate_map_json.py <path-to-map-json>')
        sys.exit(2)

    json_path = sys.argv[1]
    if not os.path.isfile(json_path):
        print(f'File not found: {json_path}')
        sys.exit(2)

    data = load_json(json_path)
    problems = validate(data)

    if not problems:
        print('OK: No missing cross-layer tokens found.')
        sys.exit(0)

    print('Found missing references:')
    for p in problems:
        print(f'- {p}')
    sys.exit(1)


if __name__ == '__main__':
    main()
