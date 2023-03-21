# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.heads import *
from .tokenpose_head import TokenPoseHead
from .distilpose_head import DistilPoseHead

__all__ = [
    'TopdownHeatmapSimpleHead', 'TopdownHeatmapMultiStageHead',
    'TopdownHeatmapMSMUHead', 'TopdownHeatmapBaseHead',
    'AEHigherResolutionHead', 'AESimpleHead', 'AEMultiStageHead',
    'DeepposeRegressionHead', 'TemporalRegressionHead', 'Interhand3DHead',
    'HMRMeshHead', 'DeconvHead', 'ViPNASHeatmapSimpleHead', 'CuboidCenterHead',
    'CuboidPoseHead', 'TokenPoseHead', 'DistilPoseHead'
]
