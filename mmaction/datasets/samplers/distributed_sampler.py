import torch
import random
import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class DistributedPowerSampler(_DistributedSampler):
    """DistributedPowerSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with the probability that is proportional to the power
    of label frequency (freq ^ power). The sampler only applies to single class
    recognition dataset.

    The default value of power is 1, which is equivalent to bootstrap sampling
    from the entire dataset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, power=1):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.power = power

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        video_infos_by_class = self.dataset.video_infos_by_class
        num_classes = self.dataset.num_classes
        # For simplicity, discontinuous labels are not permitted
        assert set(video_infos_by_class) == set(range(num_classes))
        counts = [len(video_infos_by_class[i]) for i in range(num_classes)]
        counts = [cnt**self.power for cnt in counts]

        indices = torch.multinomial(
            torch.Tensor(counts),
            self.total_size,
            replacement=True,
            generator=g)
        indices = indices.data.numpy().tolist()
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

class DistributedGobySampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None,shuffle=True,):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    # def __iter__(self):

    #     video_infos_by_class = self.dataset.goby_video_infos_by_class
    #     self.n_way = self.dataset.n_way
    #     self.k_shot = self.dataset.k_shot
    #     self.train_episode = self.dataset.train_episode
    #     self.val_episode = self.dataset.val_episode
    #     self.dataset_class = self.dataset.dataset_class

    #     # assert len(set(video_infos_by_class)) == len(set(range(self.dataset_class)))

    #     episode = self.train_episode
    #     if len(set(video_infos_by_class)) == 12:
    #         episode = self.val_episode
       
    #     class_id =[item for item in video_infos_by_class if video_infos_by_class[item]]

    #     indices = []
    #     for _ in range(episode):
    #         support_class = random.sample(class_id,self.n_way)
    #         query_class = random.sample(support_class,1)

    #         batch = []
    #         for c in support_class:
    #             if c == query_class[0]:
    #                 query_support = random.sample(range(len(video_infos_by_class[c])),self.k_shot+1)
    #                 temp =[]
    #                 for info in query_support:
    #                     item = {}
    #                     item["cls"] = c
    #                     item["index"] = info
    #                     temp.append(item)
    #                 query_indices = temp[:1]
    #                 batch.extend(temp[1:])
    #             else:
    #                 support = random.sample(range(len(video_infos_by_class[c])),self.k_shot)
    #                 for info in support:
    #                     item = {}
    #                     item["cls"] = c
    #                     item["index"] = info
    #                     batch.append(item)
    #         batch.extend(query_indices)
    #         indices.append(batch)

    #     # assert len(indices) == self.total_size

    #     # for val and test
    #     self.dataset.val_or_test = indices

    #     indices = indices[self.rank:self.total_size:self.num_replicas]

    #     return iter(indices)

    def __iter__(self):

        video_infos_by_class = self.dataset.goby_video_infos_by_class
        self.n_way = self.dataset.n_way
        self.k_shot = self.dataset.k_shot

        class_id =np.array([item for item in video_infos_by_class if video_infos_by_class[item]])

        if len(set(video_infos_by_class)) == self.dataset.test_dataset_class_num:
            indices = []
            self.total_size = self.dataset.val_episode

            for val_cls in class_id:
                val_cls_array = np.arange(len(video_infos_by_class[val_cls]))
                for query_per_cls in val_cls_array:
                    batch = []
                    query_indices = dict(cls = val_cls, index = query_per_cls)

                    query_support = random.sample(list(np.delete(val_cls_array, query_per_cls)),self.k_shot)
                    for info in query_support:
                        batch.append(dict(cls = val_cls, index = info))

                    support_class = random.sample(list(np.delete(class_id, val_cls)),self.n_way-1)
                    for support_item in support_class:
                        support = random.sample(range(len(video_infos_by_class[support_item])),self.k_shot)
                        for info in support:
                            batch.append(dict(cls = support_item, index = info))
                    random.shuffle(batch)
                    batch.append(query_indices)
                    indices.append(batch)

            random.shuffle(indices)    
        else:
            assert set(video_infos_by_class) == set(range(self.dataset.training_dataset_class_num))
            self.total_size = self.dataset.train_episode
            indices = []
            class_id = list(class_id)
            for _ in range(self.dataset.train_episode):
                support_class = random.sample(class_id,self.n_way)
                query_class = random.sample(support_class,1)

                batch = []
                for c in support_class:
                    if c == query_class[0]:
                        query_support = random.sample(range(len(video_infos_by_class[c])),self.k_shot+1)
                        temp =[]
                        for info in query_support:
                            temp.append(dict(cls = c, index = info))
                        query_indices = temp[:1]
                        batch.extend(temp[1:])
                    else:
                        support = random.sample(range(len(video_infos_by_class[c])),self.k_shot)
                        for info in support:
                            batch.append(dict(cls = c, index = info))
                batch.extend(query_indices)
                indices.append(batch)

        # for val and test
        self.dataset.val_or_test = indices
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        if len(set(self.dataset.goby_video_infos_by_class)) == 24:
            return self.dataset.val_episode
        else:
            return self.dataset.train_episode


    # def __iter__(self):

    #     video_infos_by_class = self.dataset.goby_video_infos_by_class
    #     self.n_way = self.dataset.n_way
    #     self.k_shot = self.dataset.k_shot
    #     self.episode = self.dataset.episode
    #     self.dataset_class = self.dataset.dataset_class


    #     assert set(video_infos_by_class) == set(range(self.dataset_class))
    #     counts = [len(video_infos_by_class[i]) for i in range(self.dataset_class)]

    #     indices = []
    #     for  i in range(self.episode):
    #         support_class = random.sample(range(self.dataset_class),self.n_way)
    #         query_class = random.sample(support_class,1)
    #         batch = []
    #         for c in support_class:
    #             if c == query_class[0]:
    #                 query_support = random.sample(range(counts[c]),self.k_shot+1)
    #                 temp =[]
    #                 for _ in query_support:
    #                     item = {}
    #                     item["cls"] = c
    #                     item["index"] = _
    #                     temp.append(item)
    #                 query_indices = temp[:1]
    #                 batch.extend(temp[1:])
    #             else:
    #                 support = random.sample(range(counts[c]),self.k_shot)
    #                 for _ in support:
    #                     item = {}
    #                     item["cls"] = c
    #                     item["index"] = _
    #                     batch.append(item)
    #         batch.extend(query_indices)
    #         indices.append(batch)

    #     # assert len(indices) == self.total_size

    #     # for val and test
    #     self.dataset.val_or_test = indices

    #     indices = indices[self.rank:self.total_size:self.num_replicas]

    #     return iter(indices)
