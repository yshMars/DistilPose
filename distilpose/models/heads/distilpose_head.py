# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import HEADS, build_loss
from easydict import EasyDict
from .utils.tokenbase import DistilPose_base

BN_MOMENTUM = 0.1

@HEADS.register_module()
class DistilPoseHead(nn.Module):
    """DistilPose head with TDE module and Simulated Heatmaps.

    "DistilPose: Tokenized Pose Regression with Heatmap Distillation".

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        loss_vis_token (dict): Config for visual-token loss. Default: None.
        loss_kpt_token (dict): Config for keypoint-token loss. Default: None.
        loss_score (dict): Config for keypoint loss. Default: None.
        tde_cfg (dict): Config for TDE.
        out_mode (str): Mode of head output.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 loss_vis_token_dist=None,
                 loss_kpt_token_dist=None,
                 loss_score=None,
                 loss_reg2hm=None,
                 tde_cfg=None,
                 out_mode='all',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.kpt_loss = build_loss(loss_keypoint)
        if loss_vis_token_dist is not None:
            self.vis_token_dist_loss = build_loss(loss_vis_token_dist)
        else:
            self.vis_token_dist_loss = None
        if loss_kpt_token_dist is not None:
            self.kpt_token_dist_loss = build_loss(loss_kpt_token_dist)
        else:
            self.kpt_token_dist_loss = None
        if loss_reg2hm is not None:
            self.loss_reg2hm = build_loss(loss_reg2hm)
        else:
            self.loss_reg2hm = None
        if loss_score is not None:
            self.score_loss = build_loss(loss_score)
        else:
            self.score_loss = None

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.tde_cfg = {} if tde_cfg is None else tde_cfg

        self.tokenhead = DistilPose_base(feature_size=tde_cfg.feature_size, 
                                         patch_size=tde_cfg.patch_size, 
                                         num_keypoints=self.num_joints, 
                                         dim=tde_cfg.dim, 
                                         depth=tde_cfg.depth, 
                                         heads=tde_cfg.heads,
                                         mlp_ratio=tde_cfg.mlp_ratio, 
                                         hidden_dim=tde_cfg.hidden_dim,
                                         channels=in_channels,
                                         pos_embedding_type=tde_cfg.pos_embedding_type,
                                         apply_init=tde_cfg.apply_init,
                                         out_mode = out_mode)

    def forward(self, x):
        """Forward function."""
        x = self.tokenhead(x)
        return x
    
    def get_loss(self, output, teacher_output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - teacher heatmap shape : [H, W]
            - embedding dimension: D
            - Patches Number : P

        Args:
            output (dict): student output 
                < used in this function >
                - output.pred (torch.Tensor[N, K, 2]): Output regression
                - output.score (torch.Tensor[N, K, 1]): Confidence scores prediction
                - output.vis_token (torch.Tensor[N, P, D]): student visual tokens
                - output.kpt_token (torch.Tensor[N, K, D]): teacher keypoint tokens
            teacher_output (dict): teacher output
                < used in this function >
                - teacher_output.pred (torch.Tensor[N, K, H, W]):
                - teacher_output.vis_token (torch.Tensor[N, P, D]): teacher visual tokens
                - teacher_output.kpt_token (torch.Tensor[N, K, D]): teacher keypoint tokens
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        student_vis_token = output.vis_token
        student_kpt_token = output.kpt_token
        student_reg = output.pred
        teacher_vis_token = teacher_output.vis_token
        teacher_kpt_token = teacher_output.kpt_token
        teacher_hm = teacher_output.pred
        losses = dict()
        assert not isinstance(self.kpt_loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        losses['reg_loss'] = self.kpt_loss(student_reg, target, target_weight)
        if self.vis_token_dist_loss is not None:
            losses['vis_dist_loss'] = self.vis_token_dist_loss(student_vis_token, teacher_vis_token)
        if self.kpt_token_dist_loss is not None:
            losses['kpt_dist_loss'] = self.kpt_token_dist_loss(student_kpt_token, teacher_kpt_token)
        if self.score_loss is not None:
            losses['score_loss'] = self.score_loss(output, teacher_hm, target_weight)
        if self.loss_reg2hm is not None:
            losses['reg2hm_loss'] = self.loss_reg2hm(output, teacher_hm, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output : dict
                < used in this function >
                - output.pred (torch.Tensor[N, K, 2]): Output regression
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()

        output = output.pred
        N = output.shape[0]
        
        if isinstance(target, list):
            target_r, target_h = target[0], target[1]
            target_weight_r, target_weight_h = target_weight[0], target_weight[1]
        else:
            target_r = target
            target_weight_r = target_weight

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target_r.detach().cpu().numpy(),
            target_weight_r[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        output_r = output.pred
        output_score = output.score

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output_r.detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output_r.detach().cpu().numpy()
        output = EasyDict(
            pred_jts=output_regression,
            score=output_score
        )
        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output : dict
                < used in this function >
                - output.pred_jts (torch.Tensor[N, K, 2]): Output regression
                - output.score (torch.Tensor[N, K, 1]): Confidence scores prediction
            teacher_output (torch.Tensor[N, K, H, W]): Heatmap predicted by teacher
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
        """
        batch_size = len(img_metas)
        output_pts = output.pred_jts
        output_score = output.score
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output_pts, c, s,
                                                   kwargs['img_size'])

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = output_score.detach().cpu().numpy()
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score 

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

