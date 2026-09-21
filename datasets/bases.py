import random

import torch
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        
        pids, cams, tracks = [], [], []
        num_imgs = 0

        for k in data.keys():
            num_imgs+=len(data[k])
            for _, pid, camid, trackid in data[k]:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

def split_erasing(transform):
    """(head, tail) with the TRAILING erasing ops moved into `tail`.

    Matched on the class name rather than by importing a type: the pipeline
    uses timm's RandomErasing, torchvision ships one of its own under the same
    name, and either would have to be split the same way.  Only a trailing run
    of them is moved -- an erasing op in the middle of the pipeline would mean
    the ordering assumption here is wrong, and silently splitting around it
    would change what the augmentation does.
    """
    ops = getattr(transform, 'transforms', None)
    if not ops:
        return transform, None
    k = len(ops)
    while k > 0 and type(ops[k - 1]).__name__ == 'RandomErasing':
        k -= 1
    if k == len(ops):
        return transform, None
    import torchvision.transforms as _T
    return _T.Compose(ops[:k]), _T.Compose(ops[k:])


class ImageDataset(Dataset):
    def __init__(self, dataset, modalities, transform=None, tie_augmentation=False,
                 tie_erasing=True):
        # dataset: dict(modality -> list of (img_path, pid, camid, modality_id))
        self.dataset = dataset
        self.modalities = list(modalities)
        self.transform = transform
        # Draw the augmentation ONCE per capture and replay it for every
        # modality, instead of once per image.
        #
        # The default pipeline is Resize -> RandomHorizontalFlip(p=0.5) ->
        # Pad -> RandomCrop -> RandomErasing(p=0.5), and `transform` is called
        # inside the modality loop below, so each spectrum draws its own.  With
        # DATALOADER.SYNC_FRAMES the three images are the same instant and are
        # supposed to differ only in spectrum -- but at p=0.5 for the flip,
        # HALF of those triplets come out mirrored relative to each other, plus
        # a different crop offset and a different erased patch each.  Measured,
        # not inferred: applying the shipped train transform twice to one image
        # gave a different tensor 400 times out of 400.
        #
        # A ranking loss tolerates that noise; a per-sample regression onto the
        # reference-modality feature does not, and a per-modality constant
        # cannot undo a mirror at all.  Off by default so every run recorded so
        # far keeps its exact input distribution.
        self.tie_augmentation = tie_augmentation
        # Tying the crop and the flip is the point; tying the ERASING is a side
        # effect of replaying one seed over the whole pipeline, and it fills the
        # same rectangle with the same constant in both images of a twin pair --
        # inserting identical content into exactly the comparison the twin loss
        # is trying to measure.  With tie_erasing False the pipeline is split
        # and only the head is replayed.
        self.tie_erasing = tie_erasing
        self._head, self._erase = (transform, None)
        if transform is not None and tie_augmentation and not tie_erasing:
            self._head, self._erase = split_erasing(transform)
            if self._erase is None:
                # Silence here would mean the flag looked set and did nothing.
                raise ValueError(
                    'DATALOADER.TIE_ERASING False needs a trailing RandomErasing '
                    'in the transform; this pipeline ends with {!r}'.format(
                        type(getattr(transform, 'transforms', [transform])[-1]).__name__))

    def __len__(self):
        return sum(len(self.dataset[m]) for m in self.modalities)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            imgs = []
            camids = []
            pid = None
            # One seed per capture, re-seeded before each modality: torchvision
            # transforms read `random`, timm's RandomErasing reads torch's
            # generator, so both have to be set.  Each DataLoader worker owns
            # its own RNG state, and __getitem__ leaves it advanced either way,
            # so this changes what the augmentation *is*, never whether the
            # epoch is reproducible.
            seed = random.getrandbits(31) if self.tie_augmentation else None
            # Drawn BEFORE the loop on purpose.  Inside it `random.seed(seed)`
            # has already run, so anything drawn there would come out the same
            # for all three spectra -- which is the opposite of untying.
            erase_seeds = ([random.getrandbits(31) for _ in self.modalities]
                           if seed is not None and self._erase is not None else None)
            for m_idx, modality in enumerate(self.modalities):
                img_path, cur_pid, camid, _ = self.dataset[modality][index[m_idx]]
                if pid is None:
                    pid = cur_pid
                img = read_image(img_path)
                if self.transform is not None:
                    if seed is not None:
                        random.seed(seed)
                        torch.manual_seed(seed)
                    img = self._head(img)
                    if self._erase is not None:
                        # timm's RandomErasing reads torch's generator, so a
                        # fresh torch seed is what makes this draw independent.
                        torch.manual_seed(erase_seeds[m_idx])
                        img = self._erase(img)
                imgs.append(img)
                camids.append(camid)
            return imgs, pid, camids

class ImageDatasetTest(Dataset):
    def __init__(self, dataset, modality, transform=None):
        # dataset: list of (img_path, pid, camid, modality)
        self.dataset = dataset
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.dataset[self.modality])
    
    def __getitem__(self, index):
        if isinstance(index, int):
            img_path, pid, camid, modality = self.dataset[self.modality][index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            filename = osp.basename(img_path)
            return img, pid, camid, modality, filename
