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

        # goby
        self.gt = GT(in_channels)

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

        x = self.gt(x)

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

class GT(nn.Module):
    def __init__(self,channel):
        super(GT, self).__init__()

        self.n_length = 8

        # self.maxpool = nn.MaxPool3d((3,3,3),1,1)
        self.maxpool = nn.MaxPool3d((3,1,1),1,(1,0,0))
        self.conv1 = nn.Conv2d(channel//4, channel//4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channel//4, channel//4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(channel//4, channel//4, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        t_split = torch.split(x, x.size()[1]//4, 1)
        n, c, h, w = t_split[0].size()

        t_split_1 = self.conv1(t_split[1]).reshape(-1, self.n_length,c,h,w).transpose(1, 2)
        t_split_1 = self.maxpool(t_split_1).transpose(1, 2).reshape(-1, c, w, h)

        t_split_2 = self.conv2(t_split[2]).reshape(-1, self.n_length,c,h,w).transpose(1, 2)
        t_split_2 = self.maxpool(t_split_2).transpose(1, 2).reshape(-1, c, w, h)

        t_split_3 = self.conv3(t_split[3]).reshape(-1, self.n_length,c,h,w).transpose(1, 2)
        t_split_3 = self.maxpool(t_split_3).transpose(1, 2).reshape(-1, c, w, h)

        t_concat = torch.cat((t_split[0],t_split_1,t_split_2,t_split_3),1)
        t_concat = self.relu(t_concat)

        return t_concat