"""End-to-end check that CARGO survives a pipeline written for three modalities.

CPU-only, no real data: builds a small CARGO tree of real (tiny) JPEGs and runs
it through the actual sampler, dataset and collate functions.
Run directly::

    python tests/test_cargo_pipeline.py

The claim being tested is the one that saved writing a second pipeline: a
single-modality dataset needs no fork, because every stage is already
parameterised by a modality list and degenerates correctly at length one --

    PKMSampler   P x M x K  ->  P x 1 x K, i.e. plain PK, yielding 1-tuples
    ImageDataset returns a 1-element list of images
    collate      stacks into a 1-element list of [B, 3, H, W]
    the model    does torch.cat(list(x)) over modalities, a no-op at length 1

If any of those is wrong the run does not crash -- it trains on a malformed
batch -- so each link is asserted rather than argued.
"""

import os
import shutil
import sys
import tempfile
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# make_dataloader imports timm only for RandomErasing, which nothing here
# exercises.  Stub it when absent so these tests run outside the training env;
# where timm is installed (the cluster) the real module wins.
try:
    import timm.data.random_erasing  # noqa: F401
except ImportError:
    _timm = types.ModuleType('timm')
    _data = types.ModuleType('timm.data')
    _re = types.ModuleType('timm.data.random_erasing')
    _re.RandomErasing = object
    _data.random_erasing = _re
    _timm.data = _data
    sys.modules.update({'timm': _timm, 'timm.data': _data,
                        'timm.data.random_erasing': _re})

import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.bases import ImageDataset, ImageDatasetTest  # noqa: E402
from datasets.cargo import CARGO  # noqa: E402
from datasets.make_dataloader import train_collate_fn, val_collate_fn  # noqa: E402
from datasets.sampler import PKMSampler  # noqa: E402

NUM_PIDS = 8
PER_CAM = 2
SIZE = (256, 128)


def build_tree(root):
    """Real 8x16 JPEGs -- PIL has to be able to open them."""
    base = os.path.join(root, 'CARGO')
    img = Image.new('RGB', (8, 16), (127, 127, 127))
    for split in ('train', 'query', 'gallery'):
        for cam in range(1, 14):
            d = os.path.join(base, split, 'Cam{}'.format(cam))
            os.makedirs(d)
            for pid in range(NUM_PIDS):
                for n in range(PER_CAM):
                    img.save(os.path.join(d, 'Cam{}_0001_{}_{}.jpg'.format(cam, pid, n)))
    return root


class Tree(object):
    def __enter__(self):
        self.root = tempfile.mkdtemp()
        build_tree(self.root)
        return self.root

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def transforms():
    return T.Compose([T.Resize(SIZE, interpolation=3), T.ToTensor()])


def load(root):
    return CARGO(root=root, verbose=False, modalities=['RGB'], protocol='ALL')


# --------------------------------------------------------------------------

def test_sampler_yields_one_tuples():
    """zip(*temp_idxs) over a single modality must give 1-tuples, not bare ints."""
    with Tree() as root:
        ds = load(root)
        sampler = PKMSampler(ds.train, batch_size=8, num_instances=4, modalities=['RGB'])
        items = list(sampler)
        assert items, 'sampler produced nothing'
        for it in items:
            assert isinstance(it, tuple) and len(it) == 1, it
        # ImageDataset dispatches on isinstance(index, tuple); a bare int would
        # fall through its only branch and return None.
        assert ImageDataset(ds.train, ['RGB'], transforms())[items[0]] is not None


def test_sampler_keeps_the_pk_structure():
    """P identities x K instances per batch, which is what the triplet loss needs."""
    with Tree() as root:
        ds = load(root)
        num_instances, batch_size = 4, 8
        sampler = PKMSampler(ds.train, batch_size=batch_size,
                             num_instances=num_instances, modalities=['RGB'])
        rows = ds.train['RGB']
        idxs = [i[0] for i in sampler]
        # `len(sampler)` deliberately overcounts -- see
        # test_epoch_length_is_stable_at_realistic_scale.  What must hold here
        # is that the iterator emits whole batches and nothing beyond the
        # declared budget; asserting equality would be asserting a bug.
        assert 0 < len(idxs) <= len(sampler)
        assert len(idxs) % batch_size == 0, len(idxs)

        for start in range(0, len(idxs) - batch_size + 1, batch_size):
            pids = [rows[i][1] for i in idxs[start:start + batch_size]]
            counts = {p: pids.count(p) for p in set(pids)}
            assert len(counts) == batch_size // num_instances, counts
            assert set(counts.values()) == {num_instances}, counts


def test_train_collate_gives_a_one_element_modality_list():
    with Tree() as root:
        ds = load(root)
        train_set = ImageDataset(ds.train, ['RGB'], transforms())
        sampler = PKMSampler(ds.train, batch_size=8, num_instances=4, modalities=['RGB'])
        batch = [train_set[i] for i in list(sampler)[:8]]
        imgs, pids, camids = train_collate_fn(batch)

        assert isinstance(imgs, list) and len(imgs) == 1
        assert imgs[0].shape == (8, 3, SIZE[0], SIZE[1]), imgs[0].shape
        assert isinstance(camids, list) and len(camids) == 1
        assert camids[0].shape == (8,)
        assert pids.shape == (8,) and pids.dtype == torch.int64


def test_model_side_cat_over_modalities_is_a_no_op():
    """build_transformer.forward does torch.cat(list(x)) -- at length 1 that must
    hand the batch through untouched, and target.repeat(1) must not duplicate."""
    with Tree() as root:
        ds = load(root)
        train_set = ImageDataset(ds.train, ['RGB'], transforms())
        sampler = PKMSampler(ds.train, batch_size=8, num_instances=4, modalities=['RGB'])
        imgs, pids, _ = train_collate_fn([train_set[i] for i in list(sampler)[:8]])

        stacked = torch.cat(list(imgs), dim=0)
        assert stacked.shape == imgs[0].shape
        assert torch.equal(stacked, imgs[0])
        assert torch.equal(pids.repeat(len(imgs)), pids)


def test_val_path_carries_pid_camid_and_view():
    with Tree() as root:
        ds = load(root)
        val_data = {'RGB': ds.query['RGB'] + ds.gallery['RGB']}
        val_set = ImageDatasetTest(val_data, 'RGB', transforms())
        batch = [val_set[i] for i in range(4)]
        img, pids, camids, camidt, modids, paths = val_collate_fn(batch)

        assert img.shape == (4, 3, SIZE[0], SIZE[1])
        assert camidt.shape == (4,) and camidt.dtype == torch.int64
        assert all(m in (1, 2) for m in modids), modids     # 1 aerial, 2 ground
        assert all(p.endswith('.jpg') for p in paths)
        # The evaluator splits query from gallery by position, so the
        # concatenation order above has to be query-first.
        assert len(val_set) == len(ds.query['RGB']) + len(ds.gallery['RGB'])
        assert pids[0] == ds.query['RGB'][0][1]


def test_epoch_length_is_whole_batches_and_within_budget():
    """`len(sampler)` overcounts the iterator, by an RNG-dependent batch or two.

    PKMSampler's `while len(avai_pids) >= num_pids_per_batch` loop stops the
    moment it cannot fill a whole batch, so whichever identities still have
    groups left are discarded -- while `__len__` counted them.  How many get
    stranded depends on the shuffle, so the epoch length is *not* deterministic
    even at realistic scale: over 30 shuffles, 2400 identities gave both 47936
    and 48000, and 120 gave 2816 and 2752.

    That is pre-existing behaviour, shared with WHU-MARS, and harmless: the
    drift is at most a couple of batches in ~1850, DataLoader tolerates a short
    iterator, and the global step counter that drives the LR schedule stays
    monotonic.  Changing it would move every number this repo has recorded.

    So what gets pinned is what is structurally guaranteed, not a value that
    happened to repeat in a handful of samples.
    """
    # Fabricated rows: the sampler never opens a file, so this needs no images
    # and can run at a scale that actually matches CARGO.
    for n_pids, per_pid, batch_size in ((120, 26, 64), (2400, 20, 64)):
        rows = {'RGB': [('p{}_{}.jpg'.format(p, i), p, i % 13, 1)
                        for p in range(n_pids) for i in range(per_pid)]}
        sampler = PKMSampler(rows, batch_size=batch_size, num_instances=4, modalities=['RGB'])
        for _ in range(10):
            n = len(list(sampler))
            assert n % batch_size == 0, (n_pids, n)          # only whole batches
            assert n <= len(sampler), (n_pids, n)            # never over budget
            # And never so short that an epoch quietly loses real data.
            assert n >= len(sampler) - 4 * batch_size, (n_pids, n, len(sampler))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL  {}: {}: {}'.format(fn.__name__, type(exc).__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
