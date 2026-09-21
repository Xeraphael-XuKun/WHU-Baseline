from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import os.path as osp
import random
import re

import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class PKMSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances for each modality,
    so the batch size is P*K*M.
    data_source: list of (img_path, pid, camid, modality).
    """
    
    def __init__(self, data_source, batch_size, num_instances, modalities,
                 sync_frames=False):

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(lambda: defaultdict(list))
        self.modality_ls = list(modalities)
        self.sync_frames = sync_frames

        if sync_frames:
            self._build_sync_index()
        elif self.data_source.keys():
            for m in self.modality_ls:
                for index, (_, pid, _, _) in enumerate(self.data_source[m]):
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

    def _build_sync_index(self):
        """Group the three modalities by the capture they came from.

        Filenames are ``{pid}_c{cam}_m{modality}_f{frame}.jpg`` and the frame
        number is shared by the sensors that recorded the same instant, so a
        capture is identified by (pid, cam, frame). The default path shuffles
        each modality independently, which makes a batch triplet the same person
        at three unrelated moments; here the three images are the same instant,
        same pose, same viewpoint, and differ only in spectrum.

        Captures missing a modality are dropped -- on WHU-MARS train that is
        none of them (30,711 images per modality, 30,711 complete triplets).
        """
        pattern = re.compile(r'^(\d+)_c(\d+)_m(\d+)_f(\d+)\.jpg$')
        captures = defaultdict(dict)          # (pid, cam, frame) -> {modality: index}
        unparsed = 0
        for m in self.modality_ls:
            for index, (img_path, pid, _, _) in enumerate(self.data_source.get(m, [])):
                matched = pattern.match(osp.basename(img_path))
                if matched is None:
                    unparsed += 1
                    continue
                captures[(pid, matched.group(2), matched.group(4))][m] = index

        self.sync_index = defaultdict(list)
        incomplete = 0
        for (pid, _, _), per_modality in captures.items():
            if len(per_modality) == len(self.modality_ls):
                self.sync_index[pid].append(per_modality)
            else:
                incomplete += 1

        self.pids = sorted(self.sync_index.keys())
        self.length = 0
        self.pid_max_count = dict()
        for pid in self.pids:
            count = max(len(self.sync_index[pid]), self.num_instances)
            count -= count % self.num_instances
            self.pid_max_count[pid] = count
            self.length += count
        total = sum(len(v) for v in self.sync_index.values())
        print('PKM sampler: frame-synchronised, {} complete captures over {} identities'
              ' ({} incomplete dropped, {} filenames unparsed)'
              .format(total, len(self.pids), incomplete, unparsed))
        if not self.pids:
            raise RuntimeError('DATALOADER.SYNC_FRAMES is on but no capture has all '
                               'modalities; check the filename convention')

    def _iter_sync(self):
        """Same grouping logic as __iter__, but the unit is a capture, not an index."""
        batch_groups = defaultdict(list)
        for pid in self.pids:
            need = self.pid_max_count[pid]
            caps = copy.deepcopy(self.sync_index[pid])
            cur = len(caps)
            if cur >= need:
                caps = random.sample(caps, need)
            else:
                quotient, remainder = divmod(need, cur)
                caps = caps * quotient
                if remainder > 0:
                    caps += random.sample(self.sync_index[pid], remainder)
            random.shuffle(caps)
            batch_groups[pid] = [caps[i:i + self.num_instances]
                                 for i in range(0, len(caps), self.num_instances)]

        final_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                group = batch_groups[pid].pop(0)
                final_idxs.extend(tuple(cap[m] for m in self.modality_ls) for cap in group)
                if len(batch_groups[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __iter__(self):
        if self.sync_frames:
            return self._iter_sync()

        batch_idxs_dict = defaultdict(lambda: defaultdict(list))

        for pid in self.pids:
            need = self.pid_max_count[pid]
            for m in self.modality_ls:
                idxs = copy.deepcopy(self.index_dic[pid][m])
                cur = len(idxs)
                if cur >= need:
                    idxs = np.random.choice(idxs, size=need, replace=False)
                else:
                    k, r = divmod(need, cur)
                    extended_idxs = list(idxs) * k
                    if r > 0:
                        extended_idxs += random.sample(list(idxs), r)
                    idxs = extended_idxs

                random.shuffle(idxs)
                groups = [idxs[i:i+self.num_instances] 
                          for i in range(0, len(idxs), self.num_instances)]
                batch_idxs_dict[pid][m] = groups

        final_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                temp_idxs=[]
                for m in self.modality_ls:
                    group = batch_idxs_dict[pid][m].pop(0)
                    temp_idxs.append(group)
                final_idxs.extend(zip(*temp_idxs))
                if len(batch_idxs_dict[pid][self.modality_ls[0]]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class StratifiedPKMViewSampler(Sampler):
    """WHU-MARS identity x modality sampler with conditional view balance.

    A logical sample is a tuple containing one index per modality. A batch with
    ``P`` identities and ``K`` logical samples per identity therefore forwards
    ``P * K * M`` images. Cross-view identities contribute ``K/2`` Ground and
    ``K/2`` Aerial images per modality; Ground-only identities contribute ``K``
    Ground images per modality.

    The sampler is intentionally single-process. The formal candidate runs use
    one A800 and ``WORLD_SIZE=1``.
    """

    def __init__(self, data_source, batch_size, num_instances, modalities,
                 num_cross_view_pids=8, num_ground_only_pids=8,
                 aerial_cam_ids=(5, 6), seed=1234):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.modality_ls = list(modalities)
        self.num_cross_view_pids = int(num_cross_view_pids)
        self.num_ground_only_pids = int(num_ground_only_pids)
        self.aerial_cam_ids = {int(camid) for camid in aerial_cam_ids}
        self.rng = random.Random(int(seed))

        if self.num_instances <= 0 or self.num_instances % 2 != 0:
            raise ValueError('StratifiedPKMViewSampler requires an even K, got {}'
                             .format(self.num_instances))
        expected = ((self.num_cross_view_pids + self.num_ground_only_pids)
                    * self.num_instances)
        if self.batch_size != expected:
            raise ValueError(
                'Stratified PKM logical batch must be (P_AG + P_G) * K = {}, got {}'
                .format(expected, self.batch_size))

        self.pid_mod_view = defaultdict(
            lambda: defaultdict(lambda: {'Ground': [], 'Aerial': []}))
        all_pids = set()
        for modality in self.modality_ls:
            for index, (_, pid, camid, _) in enumerate(self.data_source[modality]):
                view = 'Aerial' if int(camid) in self.aerial_cam_ids else 'Ground'
                self.pid_mod_view[pid][modality][view].append(index)
                all_pids.add(pid)

        self.cross_view_pids = []
        self.ground_only_pids = []
        for pid in sorted(all_pids):
            by_modality = self.pid_mod_view[pid]
            has_ground = all(by_modality[m]['Ground'] for m in self.modality_ls)
            has_aerial = all(by_modality[m]['Aerial'] for m in self.modality_ls)
            any_aerial = any(by_modality[m]['Aerial'] for m in self.modality_ls)
            if has_ground and has_aerial:
                self.cross_view_pids.append(pid)
            elif has_ground and not any_aerial:
                self.ground_only_pids.append(pid)

        if len(self.cross_view_pids) < self.num_cross_view_pids:
            raise ValueError('Need {} cross-view PIDs, found {}'
                             .format(self.num_cross_view_pids,
                                     len(self.cross_view_pids)))
        if len(self.ground_only_pids) < self.num_ground_only_pids:
            raise ValueError('Need {} Ground-only PIDs, found {}'
                             .format(self.num_ground_only_pids,
                                     len(self.ground_only_pids)))

        # Match the teacher PKM epoch exposure before rounding to a whole batch.
        source_length = 0
        for pid in sorted(all_pids):
            max_count = max(
                max(len(self.pid_mod_view[pid][m]['Ground'])
                    + len(self.pid_mod_view[pid][m]['Aerial']),
                    self.num_instances)
                for m in self.modality_ls
            )
            source_length += max_count - max_count % self.num_instances
        self.source_length = source_length
        self.num_batches = ((self.source_length + self.batch_size - 1)
                            // self.batch_size)
        if self.num_batches == 0:
            raise ValueError('Stratified PKM produced no complete batches')
        # Keep every emitted batch faithful to the requested P_AG/P_G/K
        # contract.  Rounding up changes exposure by at most one logical batch
        # while preserving the same scheduler denominator as ceil(source/batch).
        self.length = self.num_batches * self.batch_size

        print(
            'Stratified PKM sampler: {} cross-view PIDs, {} Ground-only PIDs; '
            'P_AG={}, P_G={}, K={}, logical batch={}, actual image batch={}, '
            '{} batches/epoch'.format(
                len(self.cross_view_pids), len(self.ground_only_pids),
                self.num_cross_view_pids, self.num_ground_only_pids,
                self.num_instances, self.batch_size,
                self.batch_size * len(self.modality_ls), self.num_batches))

    def _sample(self, indices, count):
        if len(indices) >= count:
            return self.rng.sample(indices, count)
        return self.rng.choices(indices, k=count)

    def _pid_samples(self, pid, cross_view):
        per_modality = []
        half = self.num_instances // 2
        for modality in self.modality_ls:
            pools = self.pid_mod_view[pid][modality]
            if cross_view:
                indices = (self._sample(pools['Ground'], half)
                           + self._sample(pools['Aerial'], half))
            else:
                indices = self._sample(pools['Ground'], self.num_instances)
            self.rng.shuffle(indices)
            per_modality.append(indices)
        return list(zip(*per_modality))

    def __iter__(self):
        final_indices = []
        for _ in range(self.num_batches):
            cross_pids = self.rng.sample(
                self.cross_view_pids, self.num_cross_view_pids)
            ground_pids = self.rng.sample(
                self.ground_only_pids, self.num_ground_only_pids)

            logical_batch = []
            for pid in cross_pids:
                logical_batch.extend(self._pid_samples(pid, cross_view=True))
            for pid in ground_pids:
                logical_batch.extend(self._pid_samples(pid, cross_view=False))
            self.rng.shuffle(logical_batch)
            final_indices.extend(logical_batch)
        return iter(final_indices)

    def __len__(self):
        return self.length
