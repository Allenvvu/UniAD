#---------------------------------------------------------------------------------#
# Virtual BEV Module for ITRI Data Training
# Creates BEV-like representation from available embeddings without image input
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.parallel import DataContainer


class VirtualBEVModule(BaseModule):
    """
    Virtual BEV Module that creates BEV-like features from available embeddings.

    This module takes sdc_embeddings, track_queries, and map_queries to create
    a spatial representation that can replace BEV features for training.
    """

    def __init__(self,
                 embed_dim=256,
                 bev_h=200,
                 bev_w=200,
                 use_learnable_bev=True,
                 use_spatial_encoding=True,
                 init_cfg=None):
        super(VirtualBEVModule, self).__init__(init_cfg)

        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_learnable_bev = use_learnable_bev
        self.use_spatial_encoding = use_spatial_encoding

        if use_learnable_bev:
            # Learnable base BEV features
            self.learnable_bev = nn.Parameter(
                torch.randn(bev_h * bev_w, embed_dim) * 0.1
            )

        if use_spatial_encoding:
            # Spatial position encoding for BEV grid
            self.pos_embed_x = nn.Parameter(torch.randn(bev_w, embed_dim // 2) * 0.1)
            self.pos_embed_y = nn.Parameter(torch.randn(bev_h, embed_dim // 2) * 0.1)

        # Projection layers for different embedding types
        self.sdc_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.track_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.map_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def create_spatial_encoding(self, batch_size, device):
        """Create spatial position encoding for BEV grid."""
        if not self.use_spatial_encoding:
            return torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device)

        # Create meshgrid for spatial positions
        y_pos = torch.arange(self.bev_h, device=device).float()
        x_pos = torch.arange(self.bev_w, device=device).float()

        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')

        # Get position embeddings
        y_embed = self.pos_embed_y[y_grid.long()]  # [H, W, D//2]
        x_embed = self.pos_embed_x[x_grid.long()]  # [H, W, D//2]

        # Concatenate x and y embeddings
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)  # [H, W, D]

        # Reshape to [H*W, D] and expand for batch
        pos_embed = pos_embed.view(self.bev_h * self.bev_w, self.embed_dim)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, batch_size, -1)

        return pos_embed

    def forward(self, sdc_embeddings=None, track_queries=None, map_queries=None, batch_size=1):
        """
        Forward pass to create virtual BEV features.

        Args:
            sdc_embeddings: Self-driving car embeddings [B, D] or [D]
            track_queries: Track query embeddings [B, N_track, D] or [N_track, D]
            map_queries: Map query embeddings [B, N_map, D] or [N_map, D]
            batch_size: Batch size for output

        Returns:
            virtual_bev: [H*W, B, D] format compatible with UniAD heads
        """
        device = next(self.parameters()).device

        def _unwrap(obj):
            return obj.data if isinstance(obj, DataContainer) else obj

        def _to_tensor_or_none(obj, prefer_keys=None):
            obj = _unwrap(obj)
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, dict):
                keys = prefer_keys or []
                for k in keys:
                    v = obj.get(k, None)
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                # fallback: first tensor value in dict
                for v in obj.values():
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
            return None

        # Initialize base BEV features
        if self.use_learnable_bev:
            base_bev = self.learnable_bev.unsqueeze(1).expand(-1, batch_size, -1).to(device)
        else:
            base_bev = torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device)

        # Add spatial encoding
        spatial_encoding = self.create_spatial_encoding(batch_size, device)
        virtual_bev = base_bev + spatial_encoding

        # Process and integrate SDC embeddings (global context)
        if sdc_embeddings is not None:
            sdc_embeddings = _to_tensor_or_none(sdc_embeddings)
            # Ensure sdc_embeddings is on the correct device
            if sdc_embeddings is None:
                sdc_embeddings = torch.zeros((batch_size, self.embed_dim), device=device)
            sdc_embeddings = sdc_embeddings.to(device)
            if sdc_embeddings.dim() == 1:
                sdc_embeddings = sdc_embeddings.unsqueeze(0)  # [1, D]
            if sdc_embeddings.size(0) != batch_size:
                sdc_embeddings = sdc_embeddings.expand(batch_size, -1)  # [B, D]

            sdc_features = self.sdc_projector(sdc_embeddings)  # [B, D]
            sdc_global = sdc_features.unsqueeze(0).expand(self.bev_h * self.bev_w, -1, -1)  # [H*W, B, D]
            virtual_bev = virtual_bev + 0.1 * sdc_global  # Add as global context

        # Process track queries (object-level information)
        if track_queries is not None:
            track_queries = _to_tensor_or_none(track_queries, prefer_keys=['queries', 'track_queries', 'lane_queries', 'lane_query'])
            # Ensure track_queries is on the correct device
            if track_queries is None:
                track_queries = torch.zeros((batch_size, 1, self.embed_dim), device=device)
            track_queries = track_queries.to(device)
            if track_queries.dim() == 2:
                track_queries = track_queries.unsqueeze(0)  # [1, N, D]
            if track_queries.size(0) != batch_size:
                track_queries = track_queries.expand(batch_size, -1, -1)  # [B, N, D]

            track_features = self.track_projector(track_queries)  # [B, N, D]
            track_pooled = track_features.mean(dim=1)  # [B, D] - pool across tracks
            track_spatial = track_pooled.unsqueeze(0).expand(self.bev_h * self.bev_w, -1, -1)  # [H*W, B, D]
            virtual_bev = virtual_bev + 0.1 * track_spatial

        # Process map queries (lane/map information)
        if map_queries is not None:
            map_queries = _to_tensor_or_none(map_queries, prefer_keys=['lane_queries', 'lane_query', 'queries', 'track_queries'])
            # Ensure map_queries is on the correct device
            if map_queries is None:
                map_queries = torch.zeros((batch_size, 1, self.embed_dim), device=device)
            map_queries = map_queries.to(device)
            if map_queries.dim() == 2:
                map_queries = map_queries.unsqueeze(0)  # [1, N, D]
            if map_queries.size(0) != batch_size:
                map_queries = map_queries.expand(batch_size, -1, -1)  # [B, N, D]

            map_features = self.map_projector(map_queries)  # [B, N, D]
            map_pooled = map_features.mean(dim=1)  # [B, D] - pool across map elements
            map_spatial = map_pooled.unsqueeze(0).expand(self.bev_h * self.bev_w, -1, -1)  # [H*W, B, D]
            virtual_bev = virtual_bev + 0.1 * map_spatial

        return virtual_bev

    def create_dummy_outs_track(self, sdc_embeddings, track_queries, batch_size=1, device=None):
        """Create dummy outs_track structure for compatibility."""
        if device is None:
            device = next(self.parameters()).device

        def _unwrap(obj):
            return obj.data if isinstance(obj, DataContainer) else obj

        def _get_tensor_from(obj, prefer_keys=None):
            """Robustly extract a tensor from possibly wrapped structures.

            Supports tensors, dicts (preferred keys then any tensor), and lists/tuples.
            """
            obj = _unwrap(obj)
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in reversed(obj):
                    t = _get_tensor_from(item, prefer_keys=prefer_keys)
                    if torch.is_tensor(t):
                        return t
                return None
            if isinstance(obj, dict):
                keys = prefer_keys or []
                for k in keys:
                    v = obj.get(k, None)
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                    if isinstance(v, (list, tuple)):
                        for item in reversed(v):
                            item = _unwrap(item)
                            if torch.is_tensor(item):
                                return item
                for v in obj.values():
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                    if isinstance(v, (list, tuple)):
                        for item in reversed(v):
                            item = _unwrap(item)
                            if torch.is_tensor(item):
                                return item
            return None

        # Handle sdc_embeddings
        sdc_embeddings = _get_tensor_from(sdc_embeddings) if sdc_embeddings is not None else None
        if sdc_embeddings is None:
            sdc_embeddings = torch.zeros(batch_size, self.embed_dim).to(device)
        else:
            sdc_embeddings = sdc_embeddings.to(device)
            if sdc_embeddings.dim() == 1:
                sdc_embeddings = sdc_embeddings.unsqueeze(0)  # [1, D]
            if sdc_embeddings.size(0) != batch_size:
                sdc_embeddings = sdc_embeddings[:1].expand(batch_size, -1)

        # Handle track_queries
        track_queries = _get_tensor_from(track_queries, prefer_keys=['queries', 'track_queries', 'track_query_embeddings', 'lane_queries', 'lane_query']) if track_queries is not None else None
        if track_queries is None:
            print("No track queries provided, creating dummy track queries")
            num_queries = 1
            track_queries = torch.zeros(batch_size, num_queries, self.embed_dim).to(device)
        else:
            track_queries = track_queries.to(device)
            if track_queries.dim() == 2:
                track_queries = track_queries.unsqueeze(0)  # [1, N, D]
            if track_queries.size(0) != batch_size:
                track_queries = track_queries[:1].expand(batch_size, -1, -1)
            num_queries = track_queries.size(1)

        # Create dummy bbox results structure
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        print("Creating dummy tracking bbox results structure")
        dummy_boxes = torch.zeros(num_queries, 7).to(device)  # [N, 7] for LiDAR boxes (x,y,z,w,l,h,yaw)
        dummy_scores = torch.ones(num_queries).to(device) * 0.5
        dummy_labels = torch.zeros(num_queries, dtype=torch.long).to(device)
        dummy_track_ids = torch.arange(num_queries).to(device)

        lidar_boxes = LiDARInstance3DBoxes(dummy_boxes, box_dim=7)

        outs_track = {
            'bev_embed': self.forward(sdc_embeddings, track_queries, None, batch_size),
            'bev_pos': torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device),
            'track_query_embeddings': track_queries.squeeze(0),  # [N, D]
            'track_query_matched_idxes': torch.arange(num_queries).to(device),
            'sdc_embedding': sdc_embeddings.squeeze(0),  # [D]
            'track_bbox_results': [[lidar_boxes, dummy_scores, dummy_labels, dummy_track_ids]],
            'sdc_track_bbox_results': [[LiDARInstance3DBoxes(torch.zeros(1, 7).to(device), box_dim=7),
                                      torch.ones(1).to(device) * 0.5,
                                      torch.zeros(1, dtype=torch.long).to(device),
                                      torch.tensor([999]).to(device)]]
        }

        return outs_track

    def create_outs_track(self,
                          sdc_embeddings,
                          track_queries,
                          track_query_matched_idxes=None,
                          track_bbox_results=None,
                          sdc_track_bbox_results=None,
                          batch_size=1,
                          device=None):
        """Create outs_track structure using provided real inputs when available.

        Falls back to dummy components only for missing pieces.
        """
        if device is None:
            device = next(self.parameters()).device

        def _unwrap(obj):
            return obj.data if isinstance(obj, DataContainer) else obj

        def _get_tensor_from(obj, prefer_keys=None):
            obj = _unwrap(obj)
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in reversed(obj):
                    t = _get_tensor_from(item, prefer_keys=prefer_keys)
                    if torch.is_tensor(t):
                        return t
                return None
            if isinstance(obj, dict):
                keys = prefer_keys or []
                for k in keys:
                    v = obj.get(k, None)
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                    if isinstance(v, (list, tuple)):
                        for item in reversed(v):
                            item = _unwrap(item)
                            if torch.is_tensor(item):
                                return item
                for v in obj.values():
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                    if isinstance(v, (list, tuple)):
                        for item in reversed(v):
                            item = _unwrap(item)
                            if torch.is_tensor(item):
                                return item
            return None

        # SDC embedding tensor
        sdc_embeddings_t = _get_tensor_from(sdc_embeddings)
        if sdc_embeddings_t is None:
            sdc_embeddings_t = torch.zeros(batch_size, self.embed_dim).to(device)
        else:
            sdc_embeddings_t = sdc_embeddings_t.to(device)
            if sdc_embeddings_t.dim() == 1:
                sdc_embeddings_t = sdc_embeddings_t.unsqueeze(0)
            if sdc_embeddings_t.size(0) != batch_size:
                sdc_embeddings_t = sdc_embeddings_t[:1].expand(batch_size, -1)

        # Track queries tensor
        track_queries_t = _get_tensor_from(track_queries, prefer_keys=['queries', 'track_queries', 'track_query_embeddings', 'lane_queries', 'lane_query'])
        used_dummy_track_queries = False
        if track_queries_t is None:
            used_dummy_track_queries = True
            num_queries = 1
            track_queries_t = torch.zeros(batch_size, num_queries, self.embed_dim).to(device)
        else:
            track_queries_t = track_queries_t.to(device)
            if track_queries_t.dim() == 2:
                track_queries_t = track_queries_t.unsqueeze(0)
            if track_queries_t.size(0) != batch_size:
                track_queries_t = track_queries_t[:1].expand(batch_size, -1, -1)
            num_queries = track_queries_t.size(1)

        # Matched indices
        matched_idxes = _unwrap(track_query_matched_idxes)
        if isinstance(matched_idxes, list):
            matched_idxes = matched_idxes[0]
        if not torch.is_tensor(matched_idxes):
            matched_idxes = torch.arange(num_queries, device=device)

        # BBox results
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        real_bbox = _unwrap(track_bbox_results)
        real_sdc_bbox = _unwrap(sdc_track_bbox_results)
        used_dummy_bbox = False
        if not isinstance(real_bbox, list) or not real_bbox:
            used_dummy_bbox = True
            dummy_boxes = torch.zeros(num_queries, 7).to(device)
            dummy_scores = torch.ones(num_queries).to(device) * 0.5
            dummy_labels = torch.zeros(num_queries, dtype=torch.long).to(device)
            dummy_track_ids = torch.arange(num_queries).to(device)
            lidar_boxes = LiDARInstance3DBoxes(dummy_boxes, box_dim=7)
            real_bbox = [[lidar_boxes, dummy_scores, dummy_labels, dummy_track_ids]]

        if not isinstance(real_sdc_bbox, list) or not real_sdc_bbox:
            used_dummy_bbox = True
            real_sdc_bbox = [[LiDARInstance3DBoxes(torch.zeros(1, 7).to(device), box_dim=7),
                              torch.ones(1).to(device) * 0.5,
                              torch.zeros(1, dtype=torch.long).to(device),
                              torch.tensor([999]).to(device)]]

        outs_track = {
            'bev_embed': self.forward(sdc_embeddings_t, track_queries_t, None, batch_size),
            'bev_pos': torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device),
            'track_query_embeddings': track_queries_t.squeeze(0),
            'track_query_matched_idxes': matched_idxes,
            'sdc_embedding': sdc_embeddings_t.squeeze(0),
            'track_bbox_results': real_bbox,
            'sdc_track_bbox_results': real_sdc_bbox,
        }

        if used_dummy_track_queries or used_dummy_bbox:
            print("Creating dummy tracking bbox results structure")

        return outs_track

    def create_dummy_outs_seg(self, map_queries, batch_size=1, device=None):
        """Create dummy outs_seg structure for compatibility."""
        if device is None:
            device = next(self.parameters()).device

        def _unwrap(obj):
            return obj.data if isinstance(obj, DataContainer) else obj

        def _get_tensor_from(obj, prefer_keys=None):
            obj = _unwrap(obj)
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, dict):
                keys = prefer_keys or []
                for k in keys:
                    v = obj.get(k, None)
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
                for v in obj.values():
                    v = _unwrap(v)
                    if torch.is_tensor(v):
                        return v
            return None

        # Handle map_queries
        map_queries = _get_tensor_from(map_queries, prefer_keys=['lane_queries', 'lane_query', 'queries', 'track_queries']) if map_queries is not None else None
        if map_queries is None:
            print("No map queries provided, creating dummy map queries")
            num_queries = 100  # Default number of lane queries
            map_queries = torch.zeros(batch_size, num_queries, self.embed_dim).to(device)
        else:
            map_queries = map_queries.to(device)
            if map_queries.dim() == 2:
                map_queries = map_queries.unsqueeze(0)  # [1, N, D]
            if map_queries.size(0) != batch_size:
                map_queries = map_queries[:1].expand(batch_size, -1, -1)
            num_queries = map_queries.size(1)

        # Create dummy segmentation outputs
        memory = torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device)
        memory_mask = torch.zeros(batch_size, self.bev_h * self.bev_w, dtype=torch.bool).to(device)
        memory_pos = torch.zeros(self.bev_h * self.bev_w, batch_size, self.embed_dim).to(device)
        lane_query = map_queries.squeeze(0)  # [N, D]
        lane_query_pos = torch.zeros_like(lane_query).to(device)
        hw_lvl = [(self.bev_h, self.bev_w)]

        args_tuple = (memory, memory_mask, memory_pos, lane_query, None, lane_query_pos, hw_lvl)

        outs_seg = {
            'args_tuple': args_tuple
        }

        return outs_seg