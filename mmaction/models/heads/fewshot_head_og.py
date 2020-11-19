import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class FewShotHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass


    def forward(self, x, num_segs, n_way = None, k_shot = None):

        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        x = self.consensus(x)
        x = x.squeeze(1)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)

        support, query = x[:x.shape[0]-1], x[x.shape[0]-1:]
        support_split = torch.split(support,k_shot,dim=0)

        dist = []
        for split in support_split:
            # dist.append(F.pairwise_distance(split, query, p=2).mean(0))
            dist.append(torch.cosine_similarity(split, query, dim=1).mean(0))

        cls_score = torch.stack(dist,dim=0).unsqueeze(0)

        return cls_score