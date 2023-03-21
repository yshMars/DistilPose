# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.losses import *

from .dist_loss import (TokenDistilLoss, ScoreLoss, Reg2HMLoss)
__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 
    'TokenDistilLoss', 'ScoreLoss',  'Reg2HMLoss'
]
