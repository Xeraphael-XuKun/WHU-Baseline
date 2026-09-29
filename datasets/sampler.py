from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import os.path as osp
import random
import re
import numpy as np

class PKMSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances for each modality,
    so the batch size is P*K*M.
    data_source: list of (img_path, pid, camid, modality).
    """

    def __init__(self, data_source, batch_size, num_instances, modalities, sync_frames=False):
        if sync_frames:
            raise ValueError('A baseline requires SYNC_FRAMES=False')
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(lambda : defaultdict(list))
        self.modality_ls = list(modalities)
        self.sync_frames = sync_frames
        if self.data_source.keys():
            for m in self.modality_ls:
                for (index, (_, pid, _, _)) in enumerate(self.data_source[m]):
                    self.index_dic[pid][m].append(index)
            self.pids = sorted(list(self.index_dic.keys()))
            self.length = 0
            self.pid_max_count = dict()
            for pid in self.pids:
                max_num = 0
                for m in self.modality_ls:
                    cnt = len(self.index_dic[pid][m])
                    cnt = max(cnt, self.num_instances)
                    max_num = max(max_num, cnt)
                max_num -= max_num % self.num_instances
                self.pid_max_count[pid] = max_num
                self.length += max_num

    def __iter__(self):
        batch_idxs_dict = defaultdict(lambda : defaultdict(list))
        for pid in self.pids:
            need = self.pid_max_count[pid]
            for m in self.modality_ls:
                idxs = copy.deepcopy(self.index_dic[pid][m])
                cur = len(idxs)
                if cur >= need:
                    idxs = np.random.choice(idxs, size=need, replace=False)
                else:
                    (k, r) = divmod(need, cur)
                    extended_idxs = list(idxs) * k
                    if r > 0:
                        extended_idxs += random.sample(list(idxs), r)
                    idxs = extended_idxs
                random.shuffle(idxs)
                groups = [idxs[i:i + self.num_instances] for i in range(0, len(idxs), self.num_instances)]
                batch_idxs_dict[pid][m] = groups
        final_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                temp_idxs = []
                for m in self.modality_ls:
                    group = batch_idxs_dict[pid][m].pop(0)
                    temp_idxs.append(group)
                final_idxs.extend(zip(*temp_idxs))
                if len(batch_idxs_dict[pid][self.modality_ls[0]]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
