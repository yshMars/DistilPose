# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

from mmpose.models.detectors.top_down import TopDown
from distilpose.models import build_posenet
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
@POSENETS.register_module()
class TopDownDistil(TopDown):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 teacher_config,
                 output_feature=False,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__(backbone, neck, keypoint_head, train_cfg, test_cfg, pretrained, loss_pose)
        self.eval_teacher = eval_teacher
        self.output_feature = output_feature
        # build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_posenet(teacher_config['model'])
        
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        
        # used for FPN
        if isinstance(output, list):
            output = output[0]

        if self.with_keypoint:
            output = self.keypoint_head(output)

        # get teacher output
        with torch.no_grad():
            teacher_output = self.teacher_model.backbone(img)
            if self.teacher_model.with_neck:
                teacher_output = self.teacher_model.neck(teacher_output)
            if self.teacher_model.with_keypoint:
                teacher_output = self.teacher_model.keypoint_head(teacher_output)

        losses = dict()
        if self.with_keypoint and self.teacher_model.with_keypoint:
            distillation_loss = self.keypoint_head.get_loss(
                output, teacher_output, target, target_weight
            )
            losses.update(distillation_loss)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)

        # used for FPN
        if isinstance(features, list):
            features = features[0]

        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)

            # used for FPN
            if isinstance(features_flipped, list):
                features_flipped = features_flipped[0]

            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                if isinstance(output_heatmap, dict):
                    for k in output_heatmap.keys():
                        output_heatmap[k] =  (output_heatmap[k] + output_flipped_heatmap[k]) * 0.5
                else:
                    output_heatmap = (output_heatmap +
                                      output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result


    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
