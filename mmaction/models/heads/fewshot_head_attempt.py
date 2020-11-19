import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class FewShotHead(BaseHead):
    """Class head for TSN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

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
        # self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        # self.lstm = nn.LSTM(2048, 1024, 1, batch_first=True, bidirectional=True)
        # self.scaler = nn.Parameter(torch.tensor(5.0))

        # self.linear = nn.Linear(int(1024*2) if True else 1024, 1024)


    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.fc_cls, std=self.init_std)
        pass

    # def forward(self, x, num_segs):
    #     """Defines the computation performed at every call.

    #     Args:
    #         x (torch.Tensor): The input data.
    #         num_segs (int): Number of segments into which a video
    #             is divided.
    #     Returns:
    #         torch.Tensor: The classification scores for input samples.
    #     """
    #     # [N * num_segs, in_channels, 7, 7]
    #     if self.avg_pool is not None:
    #         x = self.avg_pool(x)
    #         # [N * num_segs, in_channels, 1, 1]
    #     x = x.reshape((-1, num_segs) + x.shape[1:])
    #     # [N, num_segs, in_channels, 1, 1]
    #     x = self.consensus(x)
    #     # [N, 1, in_channels, 1, 1]
    #     x = x.squeeze(1)
    #     # [N, in_channels, 1, 1]
    #     if self.dropout is not None:
    #         x = self.dropout(x)
    #         # [N, in_channels, 1, 1]
    #     x = x.view(x.size(0), -1)

    #     cls_score = self.fc_cls(x)

    #     return cls_score


    # version1
    # def forward(self, x, num_segs):
    #     """Defines the computation performed at every call.

    #     Args:
    #         x (torch.Tensor): The input data.
    #         num_segs (int): Number of segments into which a video
    #             is divided.
    #     Returns:
    #         torch.Tensor: The classification scores for input samples.
    #     """
    #     # [N * num_segs, in_channels, 7, 7]
    #     if self.avg_pool is not None:
    #         x = self.avg_pool(x)
    #         # [N * num_segs, in_channels, 1, 1]
    #     x = x.reshape((-1, num_segs) + x.shape[1:])
    #     # [N, num_segs, in_channels, 1, 1]
    #     x = self.consensus(x)
    #     # [N, 1, in_channels, 1, 1]
    #     x = x.squeeze(1)
    #     # [N, in_channels, 1, 1]
    #     if self.dropout is not None:
    #         x = self.dropout(x)
    #         # [N, in_channels, 1, 1]
    #     x = x.view(x.size(0), -1)

    #     support_feature, query_feature = x[:x.size(0)-1],x[x.size(0)-1:]

    #     # support_feature = torch.nn.functional.normalize(support_feature, p=2, dim=1)
    #     # support_feature = support_feature.cpu().detach().numpy()
    #     # # support_feature = np.mean(support_feature, axis=1)
    #     # support_feature = np.mean(support_feature, axis=0)
    #     # support_feature = np.array(support_feature).reshape(1,-1)

    #     support_feature_split = torch.split(support_feature,1,dim=0)
    #     support_norm = []
    #     for split in support_feature_split:
    #         split = torch.nn.functional.normalize(split, p=2, dim=1)
    #         split = split.cpu().detach().numpy()
    #         split = np.mean(split, axis=0)
    #         support_norm.append(split)
    #     support_norm = np.array(support_norm)

    #     query_norm = []
    #     query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
    #     query_feature = query_feature.cpu().detach().numpy()
    #     query_feature = np.mean(query_feature, axis=0)
    #     query_norm.append(query_feature)
    #     query_norm = np.array(query_norm)

    #     distance_cosine = cosine_similarity(query_norm,support_norm)
    #     predicted_y = np.argsort(-distance_cosine)
    #     predicted_y = predicted_y[:,0]

    #     return predicted_y


    # def forward(self, x, num_segs):
    #     """Defines the computation performed at every call.

    #     Args:
    #         x (torch.Tensor): The input data.
    #         num_segs (int): Number of segments into which a video
    #             is divided.
    #     Returns:
    #         torch.Tensor: The classification scores for input samples.
    #     """
    #     # [N * num_segs, in_channels, 7, 7]
    #     if self.avg_pool is not None:
    #         x = self.avg_pool(x)

    #     x = x.reshape((-1, num_segs, x.shape[1]))

    #     x = (self.lstm(x)[0]).mean(1)
    #     x = self.linear(x)

    #     shot, query = x[:x.size(0)-1], x[x.size(0)-1:]

    #     shot = shot.reshape(2, 4, -1).mean(dim=0)

    #     # cosine similarity
    #     shot = F.normalize(shot, dim=-1)
    #     query = F.normalize(query, dim=-1)        
    #     logits = torch.mm(query, shot.t())

    #     cls_score = logits * self.scaler

    #     return cls_score


    # version3

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
            dist.append(F.pairwise_distance(split, query, p=2).mean(0))
            # dist.append(torch.cosine_similarity(split, query, dim=1).mean(0))

        cls_score = torch.stack(dist,dim=0).unsqueeze(0)

        return cls_score