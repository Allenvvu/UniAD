#!/usr/bin/env python3
"""
Semantic map → nuScenes Expansion JSON converter (skeleton).

This script ingests HCT/semantic_map JSON files and emits a single
nuScenes expansion-style map JSON containing tokenized geometry tables
(`polygon`, `line`, `node`) and selected semantic layers
(`drivable_area`, `ped_crossing`, `road_divider`, `stop_line`,
`traffic_light`, `carpark_area`).

Notes
- This is a skeleton: several mappings are heuristic or left as TODOs.
- An affine transform aligns semantic_map coordinates into the target
  2D map frame. Provide parameters via CLI.

Example
  python tools/semantic_map_to_nuscenes_expansion.py \
    --semantic-dir data/nuscenes/semantic_map \
    --output data/nuscenes/itri_map/itri_expansion.json \
    --scale 1.0 --theta-deg 0.0 --tx 0.0 --ty 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


def generate_token() -> str:
    return str(uuid.uuid4())


@dataclass
class Affine2D:
    """Simple 2D affine transform: p' = R(s, theta) @ p + t.

    Parameters
    - scale: uniform scale factor
    - theta_rad: rotation angle in radians (counter-clockwise)
    - tx, ty: translation components
    - override_matrix: optional 2x2 matrix [[a11, a12],[a21, a22]] replacing rotation*scale
    """
    scale: float = 1.0
    theta_rad: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    override_matrix: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        if self.override_matrix is not None:
            a11, a12 = self.override_matrix[0]
            a21, a22 = self.override_matrix[1]
            x_t = a11 * x + a12 * y + self.tx
            y_t = a21 * x + a22 * y + self.ty
            return x_t, y_t
        s = self.scale
        c = math.cos(self.theta_rad)
        s_ = math.sin(self.theta_rad)
        xr = s * (c * x - s_ * y)
        yr = s * (s_ * x + c * y)
        return xr + self.tx, yr + self.ty


@dataclass
class NodeRegistry:
    tolerance: float = 1e-6
    node_token_by_xy: Dict[Tuple[int, int], str] = field(default_factory=dict)
    nodes: List[Dict[str, Any]] = field(default_factory=list)

    def _quantize(self, x: float, y: float) -> Tuple[int, int]:
        # Quantize for deduplication; adjust if needed
        qx = int(round(x / self.tolerance))
        qy = int(round(y / self.tolerance))
        return qx, qy

    def get_or_create(self, x: float, y: float) -> str:
        key = self._quantize(x, y)
        if key in self.node_token_by_xy:
            return self.node_token_by_xy[key]
        token = generate_token()
        self.node_token_by_xy[key] = token
        self.nodes.append({"token": token, "x": x, "y": y})
        return token


@dataclass
class LineRegistry:
    lines: List[Dict[str, Any]] = field(default_factory=list)

    def create(self, node_tokens: List[str]) -> str:
        token = generate_token()
        self.lines.append({"token": token, "node_tokens": node_tokens})
        return token


@dataclass
class PolygonRegistry:
    polygons: List[Dict[str, Any]] = field(default_factory=list)

    def create(self, exterior_node_tokens: List[str], holes: Optional[List[List[str]]] = None) -> str:
        token = generate_token()
        self.polygons.append({
            "token": token,
            "exterior_node_tokens": exterior_node_tokens,
            "holes": holes or []
        })
        return token


@dataclass
class ExpansionMap:
    """Container for nuScenes expansion map output."""
    version: str = "1.3"
    polygon: List[Dict[str, Any]] = field(default_factory=list)
    line: List[Dict[str, Any]] = field(default_factory=list)
    node: List[Dict[str, Any]] = field(default_factory=list)

    # Layers (populate selectively)
    drivable_area: List[Dict[str, Any]] = field(default_factory=list)
    road_segment: List[Dict[str, Any]] = field(default_factory=list)
    lane: List[Dict[str, Any]] = field(default_factory=list)
    lane_connector: List[Dict[str, Any]] = field(default_factory=list)
    ped_crossing: List[Dict[str, Any]] = field(default_factory=list)
    walkway: List[Dict[str, Any]] = field(default_factory=list)
    stop_line: List[Dict[str, Any]] = field(default_factory=list)
    road_divider: List[Dict[str, Any]] = field(default_factory=list)
    carpark_area: List[Dict[str, Any]] = field(default_factory=list)
    traffic_light: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "polygon": self.polygon,
            "line": self.line,
            "node": self.node,
            "drivable_area": self.drivable_area,
            "road_segment": self.road_segment,
            "lane": self.lane,
            "lane_connector": self.lane_connector,
            "ped_crossing": self.ped_crossing,
            "walkway": self.walkway,
            "stop_line": self.stop_line,
            "road_divider": self.road_divider,
            "carpark_area": self.carpark_area,
            "traffic_light": self.traffic_light,
        }


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def map_driving_area(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, polys: PolygonRegistry, out: ExpansionMap) -> None:
    data = sem.get("driving_area")
    if not isinstance(data, list):
        return

    polygon_tokens: List[str] = []
    for region in data:
        pts = region.get("points") or []
        if not pts:
            continue
        node_tokens: List[str] = []
        for p in pts:
            x, y = float(p["x"]), float(p["y"])
            xt, yt = tr.apply(x, y)
            node_tokens.append(nodes.get_or_create(xt, yt))
        polygon_token = polys.create(node_tokens)
        polygon_tokens.append(polygon_token)

    if polygon_tokens:
        out.drivable_area.append({
            "token": generate_token(),
            "polygon_tokens": polygon_tokens,
        })


def map_ped_crossing(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, polys: PolygonRegistry, out: ExpansionMap) -> None:
    data = sem.get("pedestrian_crossing")
    if not isinstance(data, list):
        return
    for cross in data:
        pts = cross.get("points") or []
        if not pts:
            continue
        node_tokens = []
        for p in pts:
            x, y = float(p["x"]), float(p["y"])
            xt, yt = tr.apply(x, y)
            node_tokens.append(nodes.get_or_create(xt, yt))
        polygon_token = polys.create(node_tokens)
        out.ped_crossing.append({
            "token": generate_token(),
            "polygon_token": polygon_token,
            "road_segment_token": "",
        })


def map_road_lines(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, lines: LineRegistry, out: ExpansionMap) -> None:
    """Map roadlines to either road_divider or stop_line based on type.

    This uses a placeholder mapping; adapt to your dataset's type codes.
    """
    data = sem.get("roadlines")
    if not isinstance(data, list):
        return

    # TODO: Provide a proper mapping from semantic types to expansion layers
    # Example heuristic: type 6 => road_divider; type 25 => stop_line
    type_to_layer: Dict[int, str] = {
        6: "road_divider",
        25: "stop_line",
    }

    for line in data:
        pts = line.get("points") or []
        if not pts:
            continue
        node_tokens: List[str] = []
        for p in pts:
            x, y = float(p["x"]), float(p["y"])
            xt, yt = tr.apply(x, y)
            node_tokens.append(nodes.get_or_create(xt, yt))
        line_token = lines.create(node_tokens)

        # Decide layer by first point type if present
        p0 = pts[0]
        ptype = int(p0.get("type")) if "type" in p0 else None
        layer = type_to_layer.get(ptype, "road_divider")
        if layer == "stop_line":
            out.stop_line.append({"token": generate_token(), "line_token": line_token})
        else:
            out.road_divider.append({"token": generate_token(), "line_token": line_token})


def map_curbs_as_dividers(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, lines: LineRegistry, out: ExpansionMap) -> None:
    data = sem.get("curbs")
    if not isinstance(data, list):
        return
    for curb in data:
        pts = curb.get("points") or []
        if not pts:
            continue
        node_tokens: List[str] = []
        for p in pts:
            x, y = float(p["x"]), float(p["y"])
            xt, yt = tr.apply(x, y)
            node_tokens.append(nodes.get_or_create(xt, yt))
        line_token = lines.create(node_tokens)
        out.road_divider.append({"token": generate_token(), "line_token": line_token})


def map_traffic_lights(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, lines: LineRegistry, out: ExpansionMap) -> None:
    data = sem.get("traffic_light")
    if not isinstance(data, list):
        return
    for tl in data:
        pts = tl.get("points") or []
        if not pts:
            continue
        # Create a tiny vertical line to represent the signal post/face
        p0 = pts[0]
        x, y, z = float(p0["x"]), float(p0["y"]), float(p0.get("z", 0.0))
        xt, yt = tr.apply(x, y)
        n1 = nodes.get_or_create(xt, yt)
        n2 = nodes.get_or_create(xt, yt + 0.01)  # small offset; purely symbolic
        line_token = lines.create([n1, n2])

        items = [
            {
                "color": "RED",
                "shape": "CIRCLE",
                "rel_pos": {"tx": 0.0, "ty": 0.0, "tz": 0.762, "rx": 0.0, "ry": 0.0, "rz": 0.0},
                "to_road_block_tokens": [],
            },
            {
                "color": "YELLOW",
                "shape": "CIRCLE",
                "rel_pos": {"tx": 0.0, "ty": 0.0, "tz": 0.4572, "rx": 0.0, "ry": 0.0, "rz": 0.0},
                "to_road_block_tokens": [],
            },
            {
                "color": "GREEN",
                "shape": "CIRCLE",
                "rel_pos": {"tx": 0.0, "ty": 0.0, "tz": 0.1524, "rx": 0.0, "ry": 0.0, "rz": 0.0},
                "to_road_block_tokens": [],
            },
        ]
        out.traffic_light.append({
            "token": generate_token(),
            "line_token": line_token,
            "traffic_light_type": "VERTICAL",
            "from_road_block_token": "",
            "items": items,
            "pose": {"tx": xt, "ty": yt, "tz": float(z), "rx": 0.0, "ry": 0.0, "rz": float(tl.get("heading", 0.0))},
        })


def map_carpark_area(sem: Dict[str, Any], tr: Affine2D, nodes: NodeRegistry, polys: PolygonRegistry, out: ExpansionMap) -> None:
    """Skeleton for carpark areas.

    The provided parkinglots.json references nav roads/points rather than explicit area polygons.
    If you have explicit lot boundary points, adapt this to create polygons. Left empty by default.
    """
    # Example: if semantic_map had "parkinglots_boundary": [{"points": [{x,y}, ...]}]
    data = sem.get("parkinglots_boundary")
    if not isinstance(data, list):
        return
    for lot in data:
        pts = lot.get("points") or []
        if not pts:
            continue
        node_tokens = []
        for p in pts:
            x, y = float(p["x"]), float(p["y"])
            xt, yt = tr.apply(x, y)
            node_tokens.append(nodes.get_or_create(xt, yt))
        polygon_token = polys.create(node_tokens)
        out.carpark_area.append({"token": generate_token(), "polygon_token": polygon_token})


def build_output_map(expansion: ExpansionMap, nodes: NodeRegistry, lines: LineRegistry, polys: PolygonRegistry) -> Dict[str, Any]:
    d = expansion.to_dict()
    d["polygon"] = polys.polygons
    d["line"] = lines.lines
    d["node"] = nodes.nodes
    return d


def read_semantic_map_dir(semantic_dir: str) -> Dict[str, Any]:
    files = [
        "driving_area.json",
        "pedestrian_crossing.json",
        "roadlines.json",
        "curbs.json",
        "traffic_light.json",
        # Optional/unused by this skeleton but available for future use:
        # "lanes_info.json", "waypoints.json", "navgroads.json",
        # "parkinglots.json", "parkingspaces.json",
    ]
    merged: Dict[str, Any] = {}
    for name in files:
        path = os.path.join(semantic_dir, name)
        data = load_json(path)
        if data is None:
            continue
        # Merge top-level keys (each file has one top-level array key)
        for k, v in data.items():
            merged[k] = v
    return merged


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert semantic_map JSONs to nuScenes expansion JSON (skeleton)")
    parser.add_argument("--semantic-dir", type=str, required=True, help="Path to semantic_map directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor")
    parser.add_argument("--theta-deg", type=float, default=0.0, help="Rotation angle in degrees (CCW)")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation x")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation y")
    parser.add_argument("--mat", type=float, nargs=4, default=None, metavar=("a11","a12","a21","a22"),
                        help="Optional override 2x2 matrix for the linear part (replaces scale/rotation)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    override_matrix = None
    if args.mat is not None:
        a11, a12, a21, a22 = args.mat
        override_matrix = ((float(a11), float(a12)), (float(a21), float(a22)))

    tr = Affine2D(
        scale=float(args.scale),
        theta_rad=math.radians(float(args.theta_deg)),
        tx=float(args.tx),
        ty=float(args.ty),
        override_matrix=override_matrix,
    )

    semantic = read_semantic_map_dir(args.semantic_dir)

    nodes = NodeRegistry(tolerance=1e-5)
    lines = LineRegistry()
    polys = PolygonRegistry()
    expansion = ExpansionMap()

    # Map layers (skeleton)
    map_driving_area(semantic, tr, nodes, polys, expansion)
    map_ped_crossing(semantic, tr, nodes, polys, expansion)
    map_road_lines(semantic, tr, nodes, lines, expansion)
    map_curbs_as_dividers(semantic, tr, nodes, lines, expansion)
    map_traffic_lights(semantic, tr, nodes, lines, expansion)
    map_carpark_area(semantic, tr, nodes, polys, expansion)

    output = build_output_map(expansion, nodes, lines, polys)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote expansion map to: {args.output}")
    print(f"Summary: nodes={len(nodes.nodes)}, lines={len(lines.lines)}, polygons={len(polys.polygons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


