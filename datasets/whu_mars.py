# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
class WHU_MARS(BaseImageDataset):
    """
    WHU-MARS
    Reference:
    Zhao et al. WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification. CVPR 2026.
    URL: https://github.com/msm8976/WHU-MARS

    Dataset statistics:
    # identities: 1,000/2,337
    # images: 185,922/434,620
    """
    # Which split on disk.  DATASETS.SUBDIR overrides it; the two that exist are
    # 'WHU-MARS' (1,000 identities: 500 train + 500 test) and 'WHU-MARS-2337'
    # (1,000 train + 1,337 test).  Note the hyphen -- an older comment here read
    # `WHU-MARS_2337` with an underscore, which is not what is on disk.
    #
    # Read from config rather than switched by editing this line, because a
    # commented-out switch leaves no trace: two runs would produce two logs that
    # look identical and are not comparable.  The split is printed at load.
    dataset_dir = 'WHU-MARS'

    # WHU-MARS-1000-GD is not a third dataset.  Table 3 of the paper: "WHU-MARS-
    # 1000-GD evaluates the WHU-MARS-1000-trained model using ground and daytime
    # queries and galleries only."  So it is an evaluation protocol over an
    # already-trained model -- the filter touches query and gallery and never
    # train, and no retraining is involved.
    #
    # camid <= 5 is the ground subset (c1-c5 ground, c6-c7 the two UAVs), which
    # matches the paper's "ground".  The identity window is the "daytime" half;
    # it is what the original repo's commented-out filter used, and on our
    # WHU-MARS-1000 test split it keeps 112 of 500 identities -- a smaller and
    # easier subset, consistent with every published GD column sitting above its
    # WHU-MARS-1000 counterpart.
    GD_PID_RANGE = (1000, 2000)
    GD_MAX_GROUND_CAM = 5

    def __init__(self, root='', verbose=True, pid_begin = 0, modalities=None,
                 protocol='ALL', subdir='', **kwargs):
        super(WHU_MARS, self).__init__()
        if modalities is None:
            raise ValueError("DATASETS.MODALITIES are not be set.")
        self.modality_ls = list(modalities)
        self.protocol = str(protocol).upper()
        if self.protocol not in ('ALL', 'GD'):
            raise ValueError(
                "DATASETS.PROTOCOL for WHU-MARS must be ALL or GD, got {!r}. "
                "(CARGO's AA/GG/AG do not apply here.)".format(protocol))
        self.dataset_dir = osp.join(root, subdir or self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        self.pid_begin = pid_begin
        # train is never filtered: under GD the model is the one trained on the
        # full split, and only the evaluation is restricted.
        train = self._process_dir(self.train_dir, relabel=True)
        gd = self.protocol == 'GD'
        query = self._process_dir(self.query_dir, relabel=False, gd=gd)
        gallery = self._process_dir(self.gallery_dir, relabel=False, gd=gd)

        if gd:
            kept = {m: len(v) for m, v in query.items()}
            if not any(kept.values()):
                # A filter that silently empties the query set would report a
                # meaningless mAP rather than fail, so refuse it here.
                raise RuntimeError(
                    'PROTOCOL GD left no query images in {}. The identity window '
                    '{} and ground camera limit c<={} do not match this split.'
                    .format(self.query_dir, self.GD_PID_RANGE, self.GD_MAX_GROUND_CAM))

        if verbose:
            print("=> WHU-MARS loaded  (split: {}, protocol: {})".format(
                osp.basename(self.dataset_dir.rstrip('/\\')), self.protocol))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def _keep(self, pid, camid, gd):
        """camid is one-based here, as it appears in the filename."""
        if not gd:
            return True
        lo, hi = self.GD_PID_RANGE
        return lo <= pid < hi and camid <= self.GD_MAX_GROUND_CAM

    def _process_dir(self, dir_path, relabel=False, gd=False):
        dataset = dict()
        pid_container = set()
        pattern = re.compile(r'(\d+)_c(\d+)')

        for modality_name in self.modality_ls:
            modality_dir = osp.join(dir_path, modality_name)
            img_paths = glob.glob(osp.join(modality_dir, '*.jpg'))
            img_paths = sorted(img_paths)
            for img_path in img_paths:
                basename = osp.basename(img_path)
                pid, camid = map(int, pattern.search(basename).groups())
                if not self._keep(pid, camid, gd):
                    continue
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        for mindex, modality_name in enumerate(self.modality_ls, 1):
            modality_dir = osp.join(dir_path, modality_name)
            img_paths = glob.glob(osp.join(modality_dir, '*.jpg'))
            img_paths = sorted(img_paths)
            
            for img_path in img_paths:
                basename = osp.basename(img_path)
                pid, camid = map(int, pattern.search(basename).groups())
                if not self._keep(pid, camid, gd):
                    continue
                camid -= 1
                if relabel:
                    pid = pid2label[pid]
                dataset.setdefault(modality_name, []).append((img_path, pid, camid, mindex))
        return dataset
