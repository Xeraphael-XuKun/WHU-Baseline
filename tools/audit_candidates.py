import argparse
import itertools
import os
import random
from collections import Counter, defaultdict

import numpy as np

from config import cfg as default_cfg
from datasets.sampler import PKMSampler, StratifiedPKMViewSampler
from datasets.whu_mars import WHU_MARS


CANDIDATES = (
    'configs/A_baseline_candidate.yml',
    'configs/B_baseline_candidate.yml',
    'configs/C_fused_baseline_candidate.yml',
)


def load_config(path, data_root):
    config = default_cfg.clone()
    config.merge_from_file(path)
    config.DATASETS.ROOT_DIR = data_root
    return config


def item_info(dataset, modalities, logical_index):
    pids = []
    views = []
    for modality, index in zip(modalities, logical_index):
        _, pid, camid, _ = dataset.train[modality][index]
        pids.append(pid)
        views.append('Aerial' if camid in (5, 6) else 'Ground')
    if len(set(pids)) != 1:
        raise AssertionError('A logical sample mixes PIDs: {}'.format(pids))
    return pids[0], views


def audit_batch(name, config, dataset):
    modalities = list(config.DATASETS.MODALITIES)
    random.seed(config.SOLVER.SEED)
    np.random.seed(config.SOLVER.SEED)

    if config.DATALOADER.SAMPLER.upper() == 'PKM':
        sampler = PKMSampler(
            dataset.train, config.SOLVER.IMS_PER_BATCH,
            config.DATALOADER.NUM_INSTANCE, modalities=modalities,
            sync_frames=config.DATALOADER.SYNC_FRAMES)
    else:
        sampler = StratifiedPKMViewSampler(
            dataset.train, config.SOLVER.IMS_PER_BATCH,
            config.DATALOADER.NUM_INSTANCE, modalities=modalities,
            num_cross_view_pids=config.DATALOADER.PKM_VIEW.NUM_CROSS_VIEW_PIDS,
            num_ground_only_pids=config.DATALOADER.PKM_VIEW.NUM_GROUND_ONLY_PIDS,
            aerial_cam_ids=config.DATALOADER.PKM_VIEW.AERIAL_CAM_IDS,
            seed=config.SOLVER.SEED)

    all_indices = list(iter(sampler))
    if (config.DATALOADER.SAMPLER.upper() == 'PKM_VIEW'
            and len(all_indices) % config.SOLVER.IMS_PER_BATCH != 0):
        raise AssertionError(
            '{} stratified sampler ended with a partial batch'.format(name))
    logical_batch = list(itertools.islice(
        all_indices, config.SOLVER.IMS_PER_BATCH))
    if len(logical_batch) != config.SOLVER.IMS_PER_BATCH:
        raise AssertionError('{} returned an incomplete first batch'.format(name))

    pid_counts = Counter()
    pid_mod_views = defaultdict(lambda: defaultdict(Counter))
    for logical_index in logical_batch:
        if len(logical_index) != len(modalities):
            raise AssertionError('Expected {} modality indices, got {}'
                                 .format(len(modalities), logical_index))
        pid, views = item_info(dataset, modalities, logical_index)
        pid_counts[pid] += 1
        for modality, view in zip(modalities, views):
            pid_mod_views[pid][modality][view] += 1

    expected_pids = config.SOLVER.IMS_PER_BATCH // config.DATALOADER.NUM_INSTANCE
    if len(pid_counts) != expected_pids:
        raise AssertionError('{} expected {} PIDs, got {}'
                             .format(name, expected_pids, len(pid_counts)))
    if set(pid_counts.values()) != {config.DATALOADER.NUM_INSTANCE}:
        raise AssertionError('{} PID multiplicities are {}'.format(name, pid_counts))

    if config.DATALOADER.SAMPLER.upper() == 'PKM_VIEW':
        cross_count = 0
        ground_count = 0
        half = config.DATALOADER.NUM_INSTANCE // 2
        for pid, by_modality in pid_mod_views.items():
            aerial_total = sum(counts['Aerial'] for counts in by_modality.values())
            if aerial_total:
                cross_count += 1
                for modality in modalities:
                    counts = by_modality[modality]
                    if counts['Ground'] != half or counts['Aerial'] != half:
                        raise AssertionError(
                            '{} PID {} modality {} has view counts {}'
                            .format(name, pid, modality, counts))
            else:
                ground_count += 1
                for modality in modalities:
                    counts = by_modality[modality]
                    if counts['Ground'] != config.DATALOADER.NUM_INSTANCE:
                        raise AssertionError(
                            '{} PID {} modality {} has view counts {}'
                            .format(name, pid, modality, counts))
        expected_cross = config.DATALOADER.PKM_VIEW.NUM_CROSS_VIEW_PIDS
        expected_ground = config.DATALOADER.PKM_VIEW.NUM_GROUND_ONLY_PIDS
        if (cross_count, ground_count) != (expected_cross, expected_ground):
            raise AssertionError(
                '{} expected cross/ground {}/{}, got {}/{}'
                .format(name, expected_cross, expected_ground,
                        cross_count, ground_count))

    actual_batch = config.SOLVER.IMS_PER_BATCH * len(modalities)
    nominal_steps = ((len(sampler) + config.SOLVER.IMS_PER_BATCH - 1)
                     // config.SOLVER.IMS_PER_BATCH)
    actual_steps = ((len(all_indices) + config.SOLVER.IMS_PER_BATCH - 1)
                    // config.SOLVER.IMS_PER_BATCH)
    print('{}: sampler={} K={} logical_batch={} actual_batch={} PIDs={} '
          'updates={}/{} (actual/scheduler denominator)'.format(
              name, config.DATALOADER.SAMPLER,
              config.DATALOADER.NUM_INSTANCE,
              config.SOLVER.IMS_PER_BATCH, actual_batch,
              len(pid_counts), actual_steps, nominal_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Parent directory containing WHU-MARS')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = [load_config(os.path.join(project_root, path), args.data_root)
               for path in CANDIDATES]

    for config in configs:
        if config.SOLVER.LOSS_TYPE != 'base':
            raise AssertionError('All candidates must use LOSS_TYPE=base')
        if config.TEST.METRIC != 'sysu' or config.TEST.FEAT_NORM != 'yes':
            raise AssertionError('All candidates must use sysu + FEAT_NORM=yes')
        if config.MODEL.MOD_DELTA or config.MODEL.TEXT_ALIGN:
            raise AssertionError('Non-baseline model modules are enabled')

    first = configs[0]
    dataset = WHU_MARS(
        root=args.data_root, verbose=False,
        modalities=list(first.DATASETS.MODALITIES),
        protocol=first.DATASETS.PROTOCOL,
        subdir=first.DATASETS.SUBDIR)
    for path, config in zip(CANDIDATES, configs):
        audit_batch(os.path.basename(path), config, dataset)
    print('CANDIDATE_AUDIT_OK')


if __name__ == '__main__':
    main()
