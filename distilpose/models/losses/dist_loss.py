# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmpose.models.builder import LOSSES
from mmpose.models.losses import JointsMSELoss, L1Loss
import cv2

@LOSSES.register_module()
class TokenDistilLoss(nn.Module):
    """Tokens Distillation loss."""

    def __init__(self, dist_type='L2', loss_weight=1.):
        super().__init__()
        if dist_type == 'L2':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            # TO BE IMPLEMENTED
            self.criterion = None
        self.loss_weight = loss_weight

    def forward(self, token_s, token_t):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K
            - embedding dimension: D
        Args:
            token_s (torch.Tensor[N, K, D]): tokens of student.
            token_t (torch.Tensor[N, K, D]): tokens of teacher.
        """
        loss = self.criterion(token_s, token_t)
        
        return loss * self.loss_weight

@LOSSES.register_module()
class ScoreLoss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, teacher_output, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - teacher heatmap shape : [H, W]
        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            output : dict
                <used in this function>
                - output.pred (torch.Tensor[N, K, 2]): Output regression
                - output.score (torch.Tensor[N, K, 1]): Confidence scores prediction
            teacher_output (torch.Tensor[N, K, H, W]): Heatmap predicted by teacher
            target_weight (torch.Tensor[N, K, 2]): Weights across different joint types.
        """
        output_reg = output.pred
        output_score = output.score
        
        student_score = torch.zeros_like(output_score)
        N, num_joints, _ = output_reg.shape
        H, W = teacher_output.shape[-2:]
        student_index_h = (output_reg[...,1].clamp(min=0, max=1)*(H-1)).round().unsqueeze(-1).long()
        student_index_w = (output_reg[...,0].clamp(min=0, max=1)*(W-1)).round().unsqueeze(-1).long()

        student_score = teacher_output[torch.arange(N)[:,None,None], 
                                       torch.arange(num_joints)[None,:,None], 
                                       student_index_h, student_index_w]
        # deal with regression output > 1 or < 0
        mask = (output_reg >= 0) * (output_reg <= 1)
        mask = (mask[...,0] * mask[...,1]).unsqueeze(-1)
        student_score = student_score*mask
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output_score * target_weight,
                                  student_score * target_weight)
        else:
            loss = self.criterion(output_score, student_score)

        return loss * self.loss_weight


@LOSSES.register_module()
class Reg2HMLoss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1., heatmap_size=[48, 64],
                 encoding='MSRA', unbiased_encoding=False, image_size=[192, 256],
                 gaussian_mode='s2'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.encoding = encoding
        self.unbiased_encoding = unbiased_encoding
        self.gaussian_mode = gaussian_mode
        self.criterion = JointsMSELoss(use_target_weight=use_target_weight, loss_weight=loss_weight)

    def forward(self, output, teacher_hm, target_weight=None):
        output_kpt = output.pred
        sigma = output.sigma
        output_reg = torch.cat([output_kpt, sigma], dim=-1)

        if self.encoding == 'MSRA':
            reg_heatmaps, target_weight_sigma = self.generate_msra_heatmap(output_reg, target_weight)
        elif self.encoding == 'UDP':
            reg_heatmaps, target_weight_sigma = self.generate_udp_heatmap(output_reg, target_weight)
        else:
            raise ValueError('encoding should be either '
                             "'MSRA' or 'UDP'")

        loss = self.criterion(reg_heatmaps, teacher_hm, target_weight_sigma.unsqueeze(-1).unsqueeze(-1))

        return loss

    def generate_msra_heatmap(self, joints, joints_3d_visible):
        N, num_joints, _ = joints.shape
        W, H = self.heatmap_size[0], self.heatmap_size[1]
        if self.unbiased_encoding:
            mu_x = joints[..., 0] * W
            mu_y = joints[..., 1] * H
            if self.gaussian_mode == 's2':
                sigma_x = joints[..., 2]
                sigma_y = joints[..., 3]
            elif self.gaussian_mode == 's1':
                sigma_x = joints[..., 2]
                sigma_y = sigma_x
            else:
                raise ValueError('gaussian_mode should be either '
                                 "'s2' or 's1'")

            tmp_size_x = sigma_x * 3
            tmp_size_y = sigma_y * 3
            ul0 = mu_x - tmp_size_x
            ul1 = mu_y - tmp_size_y
            br0 = mu_x + tmp_size_x + 1
            br1 = mu_y + tmp_size_y + 1
            mask = (torch.lt(ul0, W) * torch.lt(ul1, H) * torch.ge(br0, 0) * torch.ge(br1, 0)).long()
            target_weight = joints_3d_visible[...,0] * mask

            mu_x = mu_x.reshape(N, num_joints, 1, 1)
            mu_y = mu_y.reshape(N, num_joints, 1, 1)
            sigma_x = sigma_x.reshape(N, num_joints, 1, 1)
            sigma_y = sigma_y.reshape(N, num_joints, 1, 1)
            
            x = torch.arange(0, W, 1, dtype=torch.float32).to(sigma_x.device).reshape(1, 1, 1, W)
            y = torch.arange(0, H, 1, dtype=torch.float32).to(sigma_x.device).reshape(1, 1, H, 1)

            g = torch.exp(-(1 / 2 ) * 
                          (((x - mu_x)**2 / (sigma_x**2+1e-9)) +
                           ((y - mu_y)**2 / (sigma_y**2+1e-9))))
            target = g * mask.reshape(N, num_joints, 1, 1)
        else:
            target = torch.zeros((N, num_joints, H, W), dtype=torch.float32).to(joints.device)
            target_weight = joints_3d_visible[...,0]
            for idx in range(N):
                for joint_id in range(self.num_joints):
                    if self.gaussian_mode == 's2':
                        sigma_x, sigma_y = joints[idx, joint_id, 2:4]
                    elif self.gaussian_mode == 's1':
                        sigma_x = joints[idx, joint_id, 2]
                        sigma_y = sigma_x
                    else:
                        raise ValueError('gaussian_mode should be either '
                                         "'s2r', 's2' or 's1'")

                    tmp_size_x = sigma_x * 3
                    tmp_size_y = sigma_y * 3
                    tmp_size_x = torch.floor(tmp_size_x).detach().cpu().numpy().tolist()
                    tmp_size_y = torch.floor(tmp_size_y).detach().cpu().numpy().tolist()
                    mu_x = torch.floor(joints[idx, joint_id, 0] * W + 0.5).detach().cpu().numpy().tolist()
                    mu_y = torch.floor(joints[idx, joint_id, 1] * H + 0.5).detach().cpu().numpy().tolist()
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size_x), int(mu_y - tmp_size_y)]
                    br = [int(mu_x + tmp_size_x + 1), int(mu_y + tmp_size_y + 1)]
                    if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                        target_weight[idx, joint_id] = 0

                    if target_weight[idx, joint_id] > 0.5:
                        size_x = 2 * tmp_size_x + 1
                        size_y = 2 * tmp_size_y + 1

                        x = torch.arange(0, size_x, 1, dtype=torch.float32).to(sigma_x.device)
                        y = torch.arange(0, size_y, 1, dtype=torch.float32).to(sigma_x.device)
                        y = y[:, None]

                        x0 = size_x // 2
                        y0 = size_y // 2
                        # The gaussian is normalized,
                        g = torch.exp(-(1 / 2) * 
                                      (((x - mu_x)**2 / (sigma_x**2+1e-9)) +
                                       ((y - mu_y)**2 / (sigma_y**2+1e-9))))

                        # Usable gaussian range
                        g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                        g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                        # Image range
                        img_x = max(0, ul[0]), min(br[0], W)
                        img_y = max(0, ul[1]), min(br[1], H)

                        target[idx, joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def generate_udp_heatmap(self, joints, joints_3d_visible):
        N, num_joints, _ = joints.shape
        target_weight = joints_3d_visible[...,0]
        W, H = self.heatmap_size[0], self.heatmap_size[1]
        target = torch.zeros((N, num_joints, H, W), dtype=torch.float32).to(joints.device)
        for idx in range(N):
            for joint_id in range(num_joints):
                if self.gaussian_mode == 's2':
                    sigma_x, sigma_y = joints[idx, joint_id, 2:4]
                elif self.gaussian_mode == 's1':
                    sigma_x = joints[idx, joint_id, 2]
                    sigma_y = sigma_x
                else:
                    raise ValueError('gaussian_mode should be either '
                                     "'s2r', 's2' or 's1'")

                tmp_size_x = sigma_x * 3
                tmp_size_y = sigma_y * 3
                tmp_size_x = tmp_size_x.detach().cpu().numpy().tolist()
                tmp_size_y = tmp_size_y.detach().cpu().numpy().tolist()

                size_x = int(2 * tmp_size_x + 1)
                size_y = int(2 * tmp_size_y + 1)
                x = torch.arange(0, size_x, 1, dtype=torch.float32).to(sigma_x.device)
                y = torch.arange(0, size_y, 1, dtype=torch.float32).to(sigma_x.device)
                y = y[:, None]

                feat_stride_x = (self.image_size[0] - 1.0) / (W - 1.0)
                feat_stride_y = (self.image_size[1] - 1.0) / (H - 1.0)
                mu_x = torch.floor(joints[idx, joint_id, 0] / feat_stride_x + 0.5).detach().cpu().numpy().tolist()
                mu_y = torch.floor(joints[idx, joint_id, 1] / feat_stride_y + 0.5).detach().cpu().numpy().tolist()
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size_x), int(mu_y - tmp_size_y)]
                br = [int(mu_x + tmp_size_x + 1), int(mu_y + tmp_size_y + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[idx, joint_id] = 0

                if target_weight[idx, joint_id] == 0:
                    continue

                # # Generate gaussian
                mu_x_ac = joints[idx, joint_id, 0] / feat_stride_x
                mu_y_ac = joints[idx, joint_id, 1] / feat_stride_y
                x0 = size_x // 2
                y0 = size_y // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = torch.exp(-(1 / 2 ) * 
                              (((x - mu_x)**2 / (sigma_x**2+1e-9)) +
                               ((y - mu_y)**2 / (sigma_y**2+1e-9))))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], W)
                img_y = max(0, ul[1]), min(br[1], H)

                v = target_weight[idx, joint_id]
                if v > 0.5:
                    target[idx, joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
