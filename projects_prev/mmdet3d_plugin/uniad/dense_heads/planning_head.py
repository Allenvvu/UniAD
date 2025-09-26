#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np
import copy

@HEADS.register_module()
class PlanningHeadSingleMode(nn.Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(3, embed_dims)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        
        #### planning head
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        self.pos_embed = nn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)
        
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)
           
    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        occ_mask = None
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning

    def forward(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding.
            occ_mask (torch.Tensor): Instance mask for occupancy.
            bev_pos (torch.Tensor): BEV position.
            sdc_traj_query (torch.Tensor): SDC trajectory query. Expected: [3, 1, 6, 256]
            sdc_track_query (torch.Tensor): SDC track query. Expected: [B, C]
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        print(f"DEBUG PlanningHead - Input tensor shapes:")
        print(f"  sdc_traj_query: {sdc_traj_query.shape}")
        print(f"  sdc_track_query: {sdc_track_query.shape}")
        print(f"  bev_embed: {bev_embed.shape}")
        print(f"  command: {command}")
        
        sdc_track_query = sdc_track_query.detach()
        print(f"DEBUG PlanningHead - After detach: {sdc_track_query.shape}")
        
        sdc_traj_query = sdc_traj_query[-1]  # [3, 1, 6, 256] -> [1, 6, 256]
        print(f"DEBUG PlanningHead - sdc_traj_query[-1]: {sdc_traj_query.shape}")
        
        P = sdc_traj_query.shape[1]  # Should be 6 (planning steps)
        print(f"DEBUG PlanningHead - P (planning steps): {P}")
        
        # Handle potential 4D sdc_track_query by squeezing temporal dimensions
        while sdc_track_query.ndim > 2:
            print(f"DEBUG PlanningHead - Squeezing extra dimensions from {sdc_track_query.shape}")
            sdc_track_query = sdc_track_query.squeeze(1)
        
        print(f"DEBUG PlanningHead - sdc_track_query before expansion: {sdc_track_query.shape}")
        sdc_track_query_expanded = sdc_track_query[:, None]  # [B, C] -> [B, 1, C]
        print(f"DEBUG PlanningHead - After [:, None]: {sdc_track_query_expanded.shape}")
        
        sdc_track_query = sdc_track_query_expanded.expand(-1, P, -1)  # [B, 1, C] -> [B, P, C]
        print(f"DEBUG PlanningHead - After expand: {sdc_track_query.shape}")
        
        
        print(f"DEBUG PlanningHead - command: {command}")
        print(f"DEBUG PlanningHead - navi_embed.weight shape: {self.navi_embed.weight.shape}")
        
        # Handle None command by using default command (0)
        if command is None:
            command = 0
            print(f"DEBUG PlanningHead - Using default command: {command}")
        
        navi_embed = self.navi_embed.weight[command]  # Expected: [256]
        print(f"DEBUG PlanningHead - navi_embed after weight selection: {navi_embed.shape}")
        
        navi_embed = navi_embed[None].unsqueeze(1)  # [256] -> [1, 256] -> [1, 1, 256]
        print(f"DEBUG PlanningHead - navi_embed after [None].unsqueeze(1): {navi_embed.shape}")
        
        navi_embed = navi_embed.expand(-1, P, -1)  # [1, 1, 256] -> [1, P, 256] 
        print(f"DEBUG PlanningHead - navi_embed after expand: {navi_embed.shape}")
        
        print(f"DEBUG PlanningHead - About to concatenate:")
        print(f"  sdc_traj_query: {sdc_traj_query.shape}")
        print(f"  sdc_track_query: {sdc_track_query.shape}")
        print(f"  navi_embed: {navi_embed.shape}")
        
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
        print(f"DEBUG PlanningHead - plan_query after cat: {plan_query.shape}")
        print(f"DEBUG PlanningHead - plan_query device: {plan_query.device}")
        print(f"DEBUG PlanningHead - plan_query memory usage: {plan_query.numel() * 4 / 1024**2:.2f} MB")
        
        print(f"DEBUG PlanningHead - About to call mlp_fuser...")
        try:
            plan_query_fused = self.mlp_fuser(plan_query)
            print(f"DEBUG PlanningHead - mlp_fuser output: {plan_query_fused.shape}")
        except Exception as e:
            print(f"ERROR in mlp_fuser: {e}")
            print(f"DEBUG PlanningHead - mlp_fuser input shape: {plan_query.shape}")
            raise
        
        plan_query = plan_query_fused.max(1, keepdim=True)[0]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        print(f"DEBUG PlanningHead - plan_query after max: {plan_query.shape}")
        
        plan_query = rearrange(plan_query, 'b p c -> p b c')
        print(f"DEBUG PlanningHead - plan_query after rearrange: {plan_query.shape}")
        
        print(f"DEBUG PlanningHead - About to rearrange bev_pos: {bev_pos.shape}")
        bev_pos = rearrange(bev_pos, 'b c h w -> b (h w) c')  # Fixed: should be b (h w) c not (h w) b c
        print(f"DEBUG PlanningHead - bev_pos after rearrange: {bev_pos.shape}")
        
        print(f"DEBUG PlanningHead - About to add bev_embed + bev_pos")
        print(f"  bev_embed: {bev_embed.shape}")  
        print(f"  bev_pos: {bev_pos.shape}")
        bev_feat = bev_embed +  bev_pos
        print(f"DEBUG PlanningHead - bev_feat after addition: {bev_feat.shape}")
        
        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)  # Fixed to match input format
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, 'b c h w -> b (h w) c')  # Fixed to match output format
        ##########################
      
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
        print(f"DEBUG PlanningHead - plan_query after pos_embed: {plan_query.shape}")
        
        # Convert bev_feat to expected format for attention module
        print(f"DEBUG PlanningHead - bev_feat before final rearrange: {bev_feat.shape}")
        bev_feat = rearrange(bev_feat, 'b (h w) c -> (h w) b c', h=self.bev_h, w=self.bev_w)
        print(f"DEBUG PlanningHead - bev_feat after final rearrange: {bev_feat.shape}")
        
        # plan_query: [1, 1, 256]  
        # bev_feat: [40000, 1, 256]
        print(f"DEBUG PlanningHead - About to call attention module")
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        print(f"DEBUG PlanningHead - plan_query after attention: {plan_query.shape}")
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
        
        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor.
            occ_mask (torch.Tensor): Occupancy flow instance mask. 
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """
        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)
        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict