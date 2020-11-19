import torch
import numpy as np
from ..registry import RECOGNIZERS
from .base import BaseRecognizer

@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1

        # cls_score = self.cls_head(x, num_segs)
        # gt_labels = labels.squeeze()

        # goby
        cls_score = self.cls_head(x, num_segs=num_segs//(self.n_way * self.k_shot + 1), n_way=self.n_way, k_shot=self.k_shot)   # 9,174    1,4
        gt_labels = self.fewshot_label(labels.squeeze())         

        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        test_crops = self.test_cfg.get('test_crops', None)
        twice_sample = self.test_cfg.get('twice_sample', False)

        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        # cls_score = self.cls_head(x, num_segs)
        cls_score = self.cls_head(x, num_segs=num_segs//(self.n_way * self.k_shot + 1), n_way=self.n_way, k_shot=self.k_shot)   # 9,174    1,4

        if test_crops is not None:
            if twice_sample:
                test_crops = test_crops * 2
            cls_score = self.average_clip(cls_score, test_crops)

        return cls_score.cpu().numpy()
    
    # goby
    def fewshot_label(self,gt_labels = None):
        query_gt = gt_labels[-1]
        ture_or_false = gt_labels[:-1] == query_gt
        gen_label = torch.from_numpy(np.repeat(range(self.n_way), self.k_shot))
        gt_labels = gen_label[ture_or_false][0].to(gt_labels.device)
        # gt_labels = gen_label[ture_or_false][0].cuda()

        return gt_labels

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x, num_segs), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        test_crops = self.test_cfg.get('test_crops', None)
        twice_sample = self.test_cfg.get('twice_sample', False)

        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        cls_score = self.cls_head(x, num_segs)
        if test_crops is not None:
            if twice_sample:
                test_crops = test_crops * 2
            cls_score = self.average_clip(cls_score, test_crops)

        return cls_score
