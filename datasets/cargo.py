# encoding: utf-8

import glob
import os.path as osp

from .bases import BaseImageDataset


class CARGO(BaseImageDataset):
    """CARGO: a synthetic aerial-ground person ReID benchmark.

    Reference:
        Zhang et al. View-decoupled Transformer for Person Re-identification
        under Aerial-ground Camera Network. CVPR 2024.
    URL: https://github.com/LinlyAC/VDT-AGPReID

    13 cameras (5 aerial, 8 ground), 5,000 identities, 108,563 images.

    Layout::

        CARGO/
          train/    Cam1/ ... Cam13/    *.jpg
          query/    Cam1/ ... Cam13/
          gallery/  Cam1/ ... Cam13/

    Filenames are ``Cam{X}_..._{pid}_....jpg``.  The parsing below is copied
    from the official loader (fastreid/data/datasets/cargo.py) rather than
    inferred, because an off-by-one in the underscore index would silently
    relabel the whole dataset::

        pid    = int(name.split('_')[2])
        camid  = int(name.split('_')[0][3:])      # 'Cam12' -> 12
        viewid = 'Aerial' if camid <= 5 else 'Ground'

    Four protocols, all sharing one training set except where noted:

    ==========  ===============================================================
    ``ALL``     everything; the headline number
    ``AA``      aerial only, in train and in test
    ``GG``      ground only, in train and in test
    ``AG``      full training set, but camera ids collapse to aerial/ground so
                the evaluator's same-camera exclusion becomes a same-*view*
                exclusion -- that is what makes it a cross-view protocol
    ==========  ===============================================================

    The dataset is single-modality, but the loaders here are built around a
    modality list.  Rather than fork the pipeline, CARGO reports itself as one
    pseudo-modality: ImageDataset then yields a 1-element list, the model's
    ``torch.cat`` over modalities becomes a no-op, and PKMSampler degenerates
    to a standard PK sampler.
    """

    dataset_dir = 'CARGO'
    num_cameras = 13
    aerial_max_cam = 5          # one-based: Cam1..Cam5 are the drones

    def __init__(self, root='', verbose=True, pid_begin=0, modalities=None,
                 protocol='ALL', **kwargs):
        super(CARGO, self).__init__()
        protocol = str(protocol).upper()
        if protocol not in ('ALL', 'AA', 'GG', 'AG'):
            raise ValueError("DATASETS.PROTOCOL must be ALL, AA, GG or AG, got {}".format(protocol))
        if modalities is None:
            raise ValueError("DATASETS.MODALITIES are not be set.")
        if len(modalities) != 1:
            raise ValueError(
                "CARGO is single-modality; DATASETS.MODALITIES must hold exactly one "
                "placeholder name, got {}".format(list(modalities)))
        self.protocol = protocol
        self.modality_ls = list(modalities)

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CARGO loaded (protocol {})".format(protocol))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for path in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    @staticmethod
    def parse(img_path):
        """-> (pid, camid_one_based, is_aerial).  Raises on anything unexpected."""
        name = osp.basename(img_path)
        parts = name.split('_')
        if len(parts) < 3 or not parts[0].lower().startswith('cam'):
            raise ValueError(
                "unexpected CARGO filename {!r}: expected Cam{{X}}_..._{{pid}}_...".format(name))
        try:
            pid = int(parts[2])
            camid = int(parts[0][3:])
        except ValueError:
            raise ValueError(
                "cannot read pid/camid out of CARGO filename {!r} "
                "(pid is the 3rd underscore field, camid the digits after 'Cam')".format(name))
        return pid, camid, camid <= CARGO.aerial_max_cam

    def _keep(self, is_aerial):
        if self.protocol == 'AA':
            return is_aerial
        if self.protocol == 'GG':
            return not is_aerial
        return True

    def _camid(self, camid, is_aerial):
        # AG collapses the 13 cameras onto the two views.  Everything
        # downstream that excludes "the query's own camera" then excludes the
        # query's own *view*, which is exactly the cross-view protocol.
        if self.protocol == 'AG':
            return 0 if is_aerial else 1
        return camid - 1        # one-based on disk, zero-based in the pipeline

    def _process_dir(self, dir_path, relabel=False):
        img_paths = []
        for cam in range(1, self.num_cameras + 1):
            img_paths += glob.glob(osp.join(dir_path, 'Cam{}'.format(cam), '*.jpg'))
        img_paths = sorted(img_paths)
        if not img_paths:
            raise RuntimeError(
                "no images under {}/Cam1..Cam{}; check the CARGO layout".format(
                    dir_path, self.num_cameras))

        kept = []
        for img_path in img_paths:
            pid, camid, is_aerial = self.parse(img_path)
            if not self._keep(is_aerial):
                continue
            kept.append((img_path, pid, self._camid(camid, is_aerial), 1 if is_aerial else 2))

        if relabel:
            pid2label = {pid: i for i, pid in enumerate(sorted({p for _, p, _, _ in kept}))}
            kept = [(path, pid2label[pid] + self.pid_begin, camid, view)
                    for path, pid, camid, view in kept]

        # Single pseudo-modality: see the class docstring.
        return {self.modality_ls[0]: kept}
