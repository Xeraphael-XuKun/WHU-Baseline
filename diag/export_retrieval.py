#!/usr/bin/env python3
"""Export real, auditable WHU-MARS retrieval cases without training.

CPU is the default. Features use the repository's validation transform and
model, the complete gallery, float32, pre-BN descriptors and L2 normalization.
Run --help or see the accompanying Chinese instructions.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

import numpy as np

REPO_DEFAULT = '/mnt/cache/wanghanzhi/HSY/HiHR'
DATA_DEFAULT = '/mnt/cache/wanghanzhi/Datasets'
RUNS = {
    'baseline': ('whu_recipe_clip',
                 '00b4b6b69dfa0f97a405c941d4ac5a6a8645a71d8f31caf3ad6737500a295a2d'),
    'mvpr': ('whu_pemod_mtext_clip',
             'bac76c6d6c983d29d8b507dd3b0b58c086581492d19fb47f1421d1e6e85dd19f'),
}
MODALITIES = ('RGB', 'IR', 'Thermal')
DISPLAY_MODS = {1: 'RGB', 2: 'NIR', 3: 'TIR'}
FEATURE_VERSION = 1


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().strip().split()
        if quota != 'max':
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return count


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(str(tmp), str(path))


def image_record(row, index, side):
    path, pid, cam, mid = row
    hit = re.search(r'_f(\d+)', Path(path).name)
    mod = DISPLAY_MODS[int(mid)]
    return dict(index=int(index), side=side, path=str(Path(path).resolve()),
                query_id=mod + '/' + Path(path).name if side == 'query' else None,
                pid=int(pid), camera_zero_based=int(cam), camera='c{}'.format(int(cam) + 1),
                modality_index=int(mid), modality=mod,
                view='aerial' if int(cam) in (5, 6) else 'ground',
                frame=int(hit.group(1)) if hit else -1)


def eligible_queries(q, g):
    cameras = {}
    for row in g:
        cameras.setdefault(row['pid'], set()).add(row['camera_zero_based'])
    return [r['index'] for r in q
            if cameras.get(r['pid'], set()) - {r['camera_zero_based']}]


def choose_queries(q, g, args):
    valid = set(eligible_queries(q, g))
    pool = [r for r in q if r['index'] in valid
            and (args.query_view == 'all' or r['view'] == args.query_view)
            and (args.query_modality == 'all' or r['modality'] == args.query_modality)]
    explicit = list(args.query_id or [])
    pids = set(args.pid or [])
    if explicit or pids:
        selected = set()
        for name in explicit:
            hits = [r for r in pool if name in (r['query_id'], Path(r['path']).name,
                                               str(r['index']))]
            if not hits:
                raise ValueError('Query not found or has no valid positive after filtering: ' + name)
            if len(hits) > 1:
                raise ValueError('Ambiguous query filename; use MODALITY/filename: ' + name)
            selected.add(hits[0]['index'])
        for pid in pids:
            hits = [r['index'] for r in pool if r['pid'] == pid]
            if not hits:
                raise ValueError('No eligible query for PID {}'.format(pid))
            selected.update(hits)
        rule = 'User-specified query IDs / PIDs; not a representative sample.'
        ids = sorted(selected)
    else:
        # Select before computing either model's features or retrieval results.
        # One image per identity within each modality/view stratum avoids
        # presenting consecutive frames of the same identity as separate cases.
        rng = np.random.RandomState(args.selection_seed)
        ids = []
        for mod in ('RGB', 'NIR', 'TIR'):
            for view in ('aerial', 'ground'):
                group = [r for r in pool if r['modality'] == mod and r['view'] == view]
                seen = set()
                picked = 0
                for pos in rng.permutation(len(group)):
                    row = group[int(pos)]
                    if row['pid'] in seen:
                        continue
                    ids.append(row['index'])
                    seen.add(row['pid'])
                    picked += 1
                    if picked == args.per_stratum:
                        break
        rule = ('Before-model stratified random selection over RGB/NIR/TIR x aerial/ground; '
                'at most one image per identity within each stratum. All selected successes '
                'and failures are retained. This qualitative sample is not dataset mAP.')
    if not ids:
        raise ValueError('No eligible query remains. Check protocol and query filters.')
    return ids, dict(rule=rule, seed=args.selection_seed,
                    requested_per_stratum=args.per_stratum,
                    explicit_query_ids=explicit, explicit_pids=sorted(pids),
                    query_modality_filter=args.query_modality, query_view_filter=args.query_view,
                    total_queries=len(q), valid_queries=len(valid),
                    pool_after_filters=len(pool), selected_indices=ids,
                    invalid_query_indices=[r['index'] for r in q if r['index'] not in valid])


def ranking_metrics(order, qrow, g, topk=10, distances=None):
    # Exactly the WHU-MARS SYSU camera rule: exclude the WHOLE query camera,
    # including different identities from that camera.
    kept = [int(i) for i in order
            if g[int(i)]['camera_zero_based'] != qrow['camera_zero_based']]
    hits = np.asarray([g[i]['pid'] == qrow['pid'] for i in kept], dtype=bool)
    positives = np.flatnonzero(hits)
    if not len(positives):
        return None
    precision = np.arange(1, len(positives) + 1, dtype=np.float64) / (positives + 1)
    top = []
    for rank, idx in enumerate(kept[:topk], 1):
        top.append(dict(rank=rank, gallery_index=idx, pid=g[idx]['pid'],
                        camera=g[idx]['camera'], camera_zero_based=g[idx]['camera_zero_based'],
                        modality=g[idx]['modality'], view=g[idx]['view'], path=g[idx]['path'],
                        correct=bool(g[idx]['pid'] == qrow['pid']),
                        squared_l2=float(distances[idx]) if distances is not None else None))
    return dict(AP=float(precision.mean()), INP=float(len(positives) / (positives[-1] + 1)),
                rank1=bool(positives[0] < 1), rank5=bool(positives[0] < 5),
                rank10=bool(positives[0] < 10), first_positive_rank=int(positives[0] + 1),
                last_positive_rank=int(positives[-1] + 1), num_positives=int(len(positives)),
                valid_gallery_size=len(kept), top10=top)


class FeatureCache:
    def __init__(self, directory, signature, rows):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = self.directory / 'metadata.json'
        if meta.exists():
            data = json.loads(meta.read_text())
            if data['signature'] != signature or data['rows'] != rows:
                raise ValueError('Cache signature mismatch: {}'.format(self.directory))
        else:
            write_json(meta, dict(signature=signature, rows=rows, width=768,
                                  dtype='float32', complete=False))
        fp, dp = self.directory / 'features.npy', self.directory / 'done.npy'
        self.feat = (np.load(str(fp), mmap_mode='r+') if fp.exists() else
                     np.lib.format.open_memmap(str(fp), mode='w+', dtype=np.float32,
                                               shape=(rows, 768)))
        self.done = (np.load(str(dp), mmap_mode='r+') if dp.exists() else
                     np.lib.format.open_memmap(str(dp), mode='w+', dtype=np.bool_, shape=(rows,)))
        if self.feat.shape != (rows, 768) or self.done.shape != (rows,):
            raise ValueError('Invalid cache shape')

    def commit(self, ids, values):
        if values.shape != (len(ids), 768) or not np.isfinite(values).all():
            raise ValueError('Invalid extracted features')
        self.feat[ids] = values
        self.feat.flush()       # Persist features BEFORE recording completion.
        self.done[ids] = True
        self.done.flush()


def build_cfg(repo, data, tag, batch, workers):
    from config import cfg as base_cfg
    run, _ = RUNS[tag]
    config_file = repo / 'configs' / ('hihr_' + run + '.yml')
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(str(config_file))
    cfg.DATASETS.ROOT_DIR = str(data)
    cfg.DATASETS.PROTOCOL = 'ALL'   # GD is an evaluation filter, not retraining.
    cfg.DATASETS.SUBDIR = 'WHU-MARS'
    cfg.DATASETS.AERIAL_CAMS = [5, 6]
    cfg.MODEL.PRETRAIN_PATH = str(data / 'ViT-B-16.pt')
    if cfg.MODEL.TEXT_ALIGN:
        cfg.MODEL.TEXT_CLIP_PATH = str(data / 'ViT-B-16.pt')
    cfg.MODEL.DIST_TRAIN = False
    cfg.TEST.NECK_FEAT = 'before'
    cfg.TEST.FEAT_NORM = 'yes'
    cfg.TEST.RE_RANKING = False
    cfg.TEST.TOP_K_EVAL = 0
    cfg.TEST.IMS_PER_BATCH = batch
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.freeze()
    return cfg, config_file


def worker_init(_):
    import torch
    torch.set_num_threads(1)


def extract_records(model, cfg, records, ids, args, on_batch):
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from datasets.bases import ImageDatasetTest
    from datasets.make_dataloader import val_collate_fn
    transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                           T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])
    start = time.monotonic()
    completed = 0
    last_report = 0.0
    for mode, mod in enumerate(MODALITIES, 1):
        current = [int(i) for i in ids if records[int(i)]['modality_index'] == mode]
        if not current:
            continue
        tuples = [(records[i]['path'], records[i]['pid'],
                   records[i]['camera_zero_based'], mode) for i in current]
        ds = ImageDatasetTest({mod: tuples}, mod, transform)
        kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                      collate_fn=val_collate_fn, pin_memory=args.device == 'cuda',
                      worker_init_fn=worker_init)
        if args.workers:
            kwargs['prefetch_factor'] = 2
        loader = DataLoader(ds, **kwargs)
        cursor = 0
        with torch.no_grad():
            for images, pids, cams, cam_tensor, mids, names in loader:
                batch_ids = current[cursor:cursor + len(images)]
                cursor += len(images)
                for j, idx in enumerate(batch_ids):
                    row = records[idx]
                    if (int(pids[j]), int(cams[j]), int(mids[j]), str(names[j])) != (
                            row['pid'], row['camera_zero_based'], mode, Path(row['path']).name):
                        raise RuntimeError('Image/metadata order changed during extraction')
                feat = model(images.to(args.device), mode=mode,
                             camids=cam_tensor.to(args.device))
                on_batch(batch_ids, feat.detach().float().cpu().numpy())
                completed += len(batch_ids)
                elapsed = time.monotonic() - start
                if elapsed - last_report >= 20 or completed == len(ids):
                    print('  extracted {}/{} images; {:.2f} images/s observed; {:.1f} min elapsed'
                          .format(completed, len(ids), completed / max(elapsed, .001), elapsed / 60),
                          flush=True)
                    last_report = elapsed


def legacy_candidates(repo, run, explicit):
    candidates = [Path(explicit)] if explicit else [repo / ('feats_' + run + '.npz'),
                                                    Path('/tmp') / ('feats_' + run + '.npz')]
    return [p for p in candidates if p.is_file()]


def import_legacy(path, model, cfg, cache, q, g, records, weight, args):
    """Validate legacy pathless caches using order metadata AND fresh sentinels.

    Old dumps did not embed checkpoint hashes. Do not trust their filename
    alone: compare nine freshly extracted images with the cached descriptors.
    Any mismatch rejects reuse and triggers normal full-gallery extraction.
    """
    with np.load(str(path), allow_pickle=False) as z:
        required = ['qf', 'gf', 'q_pids', 'g_pids', 'q_camids', 'g_camids',
                    'q_mids', 'g_mids', 'q_frames', 'g_frames', 'weight', 'neck_feat']
        if not all(k in z for k in required):
            raise ValueError('legacy cache lacks required order/provenance fields')
        saved_weight = Path(str(z['weight'].item()))
        if not saved_weight.is_absolute():
            saved_weight = Path(args.repo) / saved_weight
        if saved_weight.resolve() != weight.resolve() or str(z['neck_feat'].item()) != 'before':
            raise ValueError('legacy cache describes another checkpoint or feature')
        for prefix, rows in [('q', q), ('g', g)]:
            for suffix, key in [('pids', 'pid'), ('camids', 'camera_zero_based'),
                                ('mids', 'modality_index'), ('frames', 'frame')]:
                if not np.array_equal(z[prefix + '_' + suffix], np.asarray([r[key] for r in rows])):
                    raise ValueError('legacy cache is sampled, filtered, or differently ordered')
        features = np.concatenate([z['qf'], z['gf']], axis=0).astype(np.float32, copy=False)
        if features.shape != (len(records), 768) or not np.isfinite(features).all():
            raise ValueError('invalid legacy feature shape/value')
    sentinel = []
    for mode in (1, 2, 3):
        qi = [r['index'] for r in q if r['modality_index'] == mode]
        gi = [len(q) + r['index'] for r in g if r['modality_index'] == mode]
        sentinel.extend([qi[len(qi) // 2], gi[0], gi[-1]])
    measured = {}
    extract_records(model, cfg, records, sentinel, args,
                    lambda ids, values: measured.update({i: values[j] for j, i in enumerate(ids)}))
    for idx in sentinel:
        if not np.allclose(measured[idx], features[idx], rtol=1e-4, atol=1e-5):
            raise ValueError('legacy sentinel feature differs from the verified checkpoint')
    cache.commit(np.arange(len(records)), features)
    return dict(source=str(path.resolve()), sha256=sha256(path),
                validation='full ordered PID/camera/modality/frame arrays and nine fresh descriptors',
                sentinel_indices=sentinel, embedded_checkpoint_hash=False)


def ensure_features(tag, q, g, selected, args, dataset, source_hashes, gallery_indices=None):
    import torch
    from model import make_model
    repo, data = Path(args.repo), Path(args.data_root)
    run, expected = RUNS[tag]
    weight = repo / ('out_hihr_' + run) / 'transformer_60.pth'
    if not weight.is_file():
        raise FileNotFoundError('Required existing checkpoint: ' + str(weight))
    print('Verifying paper checkpoint:', weight, flush=True)
    actual = sha256(weight)
    if actual != expected:
        raise ValueError('{} checkpoint SHA-256 does not match the recorded paper checkpoint: {}'
                         .format(tag, actual))
    cfg, config_file = build_cfg(repo, data, tag, args.batch_size, args.workers)
    records = q + g
    # Batch size/workers/device do not alter the FP32 descriptor definition.
    # Include all model/data implementation hashes; never reuse a cache after
    # silently changing the implementation or image ordering.
    signature = dict(version=FEATURE_VERSION, checkpoint_sha256=actual,
                     config_sha256=sha256(config_file), source_sha256=source_hashes,
                     records_sha256=digest_json(records), protocol='ALL',
                     feature='pre-BN 768 float32; raw, before L2 normalization')
    key = digest_json(signature)[:20]
    cache = FeatureCache(Path(args.cache_dir) / (tag + '_' + key), signature, len(records))
    gallery_indices = range(len(g)) if gallery_indices is None else gallery_indices
    needed = np.asarray(list(selected) + [len(q) + int(i) for i in gallery_indices], dtype=np.int64)
    missing = needed[~np.asarray(cache.done[needed])]
    provenance = dict(run=run, checkpoint=str(weight), checkpoint_sha256=actual,
                      config=str(config_file), config_sha256=sha256(config_file),
                      cache=str(cache.directory), cache_signature=signature,
                      missing_before_run=int(len(missing)))
    if len(missing):
        model = make_model(cfg, num_class=dataset.num_train_pids,
                           camera_num=dataset.num_train_cams, view_num=dataset.num_train_vids)
        # Repository load_param uses torch.load(..., map_location='cpu').
        # Keep construction on CPU; only the explicitly selected device is used.
        model.load_param(str(weight))
        model.to(args.device).eval()
        explicit = getattr(args, tag + '_cache')
        for candidate in legacy_candidates(repo, run, explicit):
            print('Checking legacy feature cache:', candidate, flush=True)
            try:
                provenance['legacy_import'] = import_legacy(candidate, model, cfg, cache,
                                                            q, g, records, weight, args)
                break
            except (ValueError, KeyError, OSError) as exc:
                print('Legacy cache rejected; extracting verified features:', str(exc), flush=True)
        missing = needed[~np.asarray(cache.done[needed])]
        if len(missing):
            print('{}: {} uncached images; full gallery is required.'.format(tag, len(missing)),
                  flush=True)
            extract_records(model, cfg, records, missing.tolist(), args, cache.commit)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if not np.asarray(cache.done[needed]).all():
        raise RuntimeError('Incomplete features; rerun the same command to resume.')
    return cache, provenance


def evaluate_selected(cache, q, g, selected, args):
    import torch
    from utils.metrics import iter_euclidean_distance_chunks
    qf = torch.from_numpy(np.array(cache.feat[selected], dtype=np.float32, copy=True))
    gf = torch.from_numpy(np.array(cache.feat[len(q):], dtype=np.float32, copy=True))
    qf = torch.nn.functional.normalize(qf, p=2, dim=1)
    gf = torch.nn.functional.normalize(gf, p=2, dim=1)
    result = {}
    cursor = 0
    for distances in iter_euclidean_distance_chunks(qf, gf, q_chunk_size=args.rank_batch):
        # Same full argsort as the repository evaluator; no gallery sampling,
        # approximate nearest neighbours, mean centering, or top-k AP estimate.
        orders = torch.argsort(distances, dim=1).numpy()
        d = distances.numpy()
        for i, order in enumerate(orders):
            qi = selected[cursor + i]
            result[qi] = ranking_metrics(order, q[qi], g, 10, d[i])
            if result[qi] is None:
                raise RuntimeError('Selected query has no valid positive: ' + q[qi]['query_id'])
        cursor += len(orders)
    return result


def make_contact_sheet(case, out, assets):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    w, h, gap, margin = 128, 256, 10, 14
    row_h = h + 66
    canvas = Image.new('RGB', (margin * 2 + 11 * (w + gap) - gap, 2 * row_h + 70), 'white')
    draw = ImageDraw.Draw(canvas)
    font = None
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
              '/System/Library/Fonts/Supplemental/Arial.ttf']:
        if Path(p).is_file():
            font = ImageFont.truetype(p, 13)
            break
    font = font or ImageFont.load_default()
    q = case['query']
    title = '{} | PID {} | {} | {} | green: correct identity; red: wrong identity'.format(
        q['query_id'], q['pid'], q['camera'], q['view'])
    draw.text((margin, 10), title, fill='black', font=font)
    for row, tag in enumerate(('baseline', 'mvpr')):
        y = 43 + row * row_h
        r = case[tag]
        draw.text((margin, y), '{} | AP {:.2f}% | first positive rank {}'.format(
            'Baseline' if tag == 'baseline' else 'MVPR', r['AP'] * 100,
            r['first_positive_rank']), fill='black', font=font)
        cells = [(q, None)] + [(x, x['correct']) for x in r['top10']]
        for col, (item, correct) in enumerate(cells):
            x, iy = margin + col * (w + gap), y + 24
            with Image.open(assets[item['path']]) as source:
                source = source.convert('RGB')
                thumb = ImageOps.contain(source, (w - 8, h - 8),
                                        method=getattr(Image, 'Resampling', Image).BICUBIC)
            canvas.paste(thumb, (x + (w - thumb.width) // 2, iy + (h - thumb.height) // 2))
            color = '#59636e' if correct is None else ('#16833e' if correct else '#c82b32')
            draw.rectangle((x, iy, x + w - 1, iy + h - 1), outline=color, width=4)
            text = ('Query' if col == 0 else '#{} {} PID {}'.format(
                col, 'OK' if correct else 'WRONG', item['pid']))
            draw.text((x + 1, iy + h + 3), text, fill=color, font=font)
            draw.text((x + 1, iy + h + 19), '{} {}'.format(item['modality'], item['camera']),
                      fill='black', font=font)
    path = out / 'figures' / ('query_{:06d}.png'.format(q['index']))
    canvas.save(str(path))
    return str(path.relative_to(out))


def export_results(out, q, g, selected, results, selection, provenance, args):
    out.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'figures'):
        (out / name).mkdir(exist_ok=True)
    assets, asset_info = {}, []
    cases = []
    for qi in selected:
        case = dict(query=dict(q[qi]), baseline=results['baseline'][qi], mvpr=results['mvpr'][qi])
        case['AP_change'] = case['mvpr']['AP'] - case['baseline']['AP']
        case['outcome'] = ('both_correct' if case['baseline']['rank1'] and case['mvpr']['rank1'] else
                           'mvpr_improved_rank1' if case['mvpr']['rank1'] else
                           'mvpr_worse_rank1' if case['baseline']['rank1'] else 'both_fail_rank1')
        for item in [case['query']] + case['baseline']['top10'] + case['mvpr']['top10']:
            src = item['path']
            if src not in assets:
                name = hashlib.sha256(src.encode('utf-8')).hexdigest()[:12] + '_' + Path(src).name
                dst = out / 'images' / name
                shutil.copy2(src, str(dst))
                assets[src] = dst
                asset_info.append(dict(source=src, export=str(dst.relative_to(out)), sha256=sha256(dst)))
            item['image_export'] = str(assets[src].relative_to(out))
        case['figure'] = make_contact_sheet(case, out, assets)
        cases.append(case)
    from collections import Counter
    counts = dict(Counter(c['outcome'] for c in cases))
    manifest = dict(schema_version=1, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    protocol=dict(dataset='WHU-MARS-1000', training='existing full-split epoch-60 checkpoints',
                                  evaluation=args.protocol, query_count=len(q), gallery_count=len(g),
                                  camera_exclusion='all gallery images from query camera',
                                  invalid_queries='excluded before selection',
                                  distance='squared Euclidean on L2-normalized float32 768D pre-BN features',
                                  AP='computed from the complete filtered ranking, not top-10',
                                  reranking=False, gallery_sampling=False),
                    execution=dict(device=args.device, threads=args.threads, workers=args.workers,
                                   batch_size=args.batch_size, rank_batch=args.rank_batch,
                                   torch_version=getattr(args, 'torch_version', 'not loaded'),
                                   numpy_version=np.__version__, feature_dtype='float32',
                                   note='CPU/GPU and batch-size changes can alter floating-point ties; protocol is unchanged.'),
                    selection=selection, checkpoints=provenance, outcome_counts=counts,
                    exporter_sha256=sha256(Path(__file__)),
                    caution='Qualitative cases selected by the recorded rule; do not present sample averages as benchmark mAP.',
                    cases=cases, assets=asset_info,
                    selection_record_sha256=sha256(out / 'selection_before_inference.json'))
    if not any(not c['mvpr']['rank1'] for c in cases):
        manifest['selection_note'] = ('No MVPR Rank-1 failure occurred in this preselected sample. '
                                      'No artificial failure or result-based replacement was added. '
                                      'Increase --per-stratum to inspect more queries.')
    write_json(out / 'manifest.json', manifest)
    with (out / 'query_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['query_id', 'pid', 'modality', 'view', 'camera', 'baseline_AP', 'mvpr_AP',
                  'AP_change', 'baseline_first_positive_rank', 'mvpr_first_positive_rank', 'outcome']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            row = {k: c['query'][k] for k in fields[:5]}
            row.update(baseline_AP=c['baseline']['AP'], mvpr_AP=c['mvpr']['AP'],
                       AP_change=c['AP_change'],
                       baseline_first_positive_rank=c['baseline']['first_positive_rank'],
                       mvpr_first_positive_rank=c['mvpr']['first_positive_rank'], outcome=c['outcome'])
            writer.writerow(row)
    content = ['<!doctype html><meta charset="utf-8"><title>WHU-MARS retrieval cases</title>',
               '<style>body{font:16px sans-serif;margin:24px;max-width:1600px}img{width:100%;height:auto}'
               'section{margin:30px 0;border-top:1px solid #bbb}code{white-space:pre-wrap}</style>',
               '<h1>Baseline / MVPR: real top-10 retrieval</h1>',
               '<p>All gallery images from the query camera are excluded. AP uses the full ranking. '
               'Green = correct identity; red = wrong identity. Every selected case is shown.</p>',
               '<p>' + html.escape(selection['rule']) + '</p>',
               '<p>Outcome counts: <code>' + html.escape(str(counts)) + '</code></p>',
               '<p><a href="manifest.json">Audit manifest</a> | '
               '<a href="query_metrics.csv">Per-query metrics</a></p>']
    for c in cases:
        content.append('<section><h2>{}</h2><p>{} | AP: {:.2f}% → {:.2f}%</p>'
                       '<img src="{}" loading="lazy"></section>'.format(
                           html.escape(c['query']['query_id']), html.escape(c['outcome']),
                           100 * c['baseline']['AP'], 100 * c['mvpr']['AP'], c['figure']))
    (out / 'index.html').write_text('\n'.join(content), encoding='utf-8')
    archive = out.parent / (out.name + '.tar.gz')
    with tarfile.open(str(archive), 'w:gz') as tar:
        # Enumerate only this run's assets; caches/checkpoints never enter the archive.
        paths = [out / 'manifest.json', out / 'query_metrics.csv', out / 'index.html',
                 out / 'selection_before_inference.json']
        paths += [out / c['figure'] for c in cases]
        paths += list(assets.values())
        for path in paths:
            tar.add(str(path), arcname=str(Path(out.name) / path.relative_to(out)), recursive=False)
    print('Download this result package (no checkpoints or feature caches):', archive, flush=True)
    print('Selected outcomes:', counts, flush=True)


def self_test():
    import tempfile
    from types import SimpleNamespace
    q = [dict(index=0, pid=7, camera_zero_based=0, modality='RGB', view='aerial',
              query_id='RGB/q.jpg', path='/tmp/q.jpg')]
    g = [dict(index=i, pid=pid, camera_zero_based=cam, camera='c{}'.format(cam + 1),
              modality='RGB', view='ground', path='/tmp/{}.jpg'.format(i))
         for i, (pid, cam) in enumerate([(7, 0), (8, 0), (8, 1), (7, 1), (7, 2)])]
    r = ranking_metrics(np.arange(5), q[0], g, topk=2)
    assert [x['gallery_index'] for x in r['top10']] == [2, 3]
    assert abs(r['AP'] - 7 / 12) < 1e-12 and r['first_positive_rank'] == 2
    assert abs(r['INP'] - 2 / 3) < 1e-12
    assert ranking_metrics(np.arange(5), dict(q[0], pid=99), g) is None
    assert eligible_queries(q, g) == [0]
    sq, sg = [], []
    for mi, mod in enumerate(('RGB', 'NIR', 'TIR')):
        for vi, view in enumerate(('aerial', 'ground')):
            for identity in range(3):
                pid = mi * 100 + vi * 10 + identity
                sg.append(dict(pid=pid, camera_zero_based=1))
                for frame in range(2):
                    i = len(sq)
                    sq.append(dict(index=i, pid=pid, camera_zero_based=5 if vi == 0 else 0,
                                   modality=mod, view=view, path='/tmp/{}.jpg'.format(i),
                                   query_id=mod + '/{}.jpg'.format(i)))
    opts = SimpleNamespace(query_view='all', query_modality='all', query_id=None, pid=None,
                           selection_seed=1234, per_stratum=2)
    picked, _ = choose_queries(sq, sg, opts)
    assert picked == choose_queries(sq, sg, opts)[0] and len(picked) == 12
    for mod in ('RGB', 'NIR', 'TIR'):
        for view in ('aerial', 'ground'):
            group = [sq[i] for i in picked if sq[i]['modality'] == mod and sq[i]['view'] == view]
            assert len(group) == 2 and len({r['pid'] for r in group}) == 2
    with tempfile.TemporaryDirectory() as tmp:
        c = FeatureCache(tmp, {'test': 1}, 3)
        c.commit([1], np.ones((1, 768), dtype=np.float32))
        reopened = FeatureCache(tmp, {'test': 1}, 3)
        assert reopened.done.tolist() == [False, True, False]
        assert np.array_equal(reopened.feat[1], np.ones(768, dtype=np.float32))
    print('Self-test passed: whole-camera exclusion, full-ranking AP/INP, invalid queries, '
          'deterministic stratified selection and cache resume.')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=REPO_DEFAULT)
    ap.add_argument('--data-root', default=DATA_DEFAULT)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=min(8, available_cpus()))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--rank-batch', type=int, default=8)
    ap.add_argument('--per-stratum', type=int, default=4)
    ap.add_argument('--selection-seed', type=int, default=1234)
    ap.add_argument('--query-id', action='append', help='MODALITY/filename, unique filename, or global query index')
    ap.add_argument('--pid', type=int, action='append', help='Select all eligible query images of this identity')
    ap.add_argument('--query-view', choices=['all', 'aerial', 'ground'], default='all')
    ap.add_argument('--query-modality', choices=['all', 'RGB', 'NIR', 'TIR'], default='all')
    ap.add_argument('--protocol', choices=['ALL', 'GD'], default='ALL')
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--baseline-cache', default=None, help='Optional legacy .npz; validated before reuse')
    ap.add_argument('--mvpr-cache', default=None, help='Optional legacy .npz; validated before reuse')
    ap.add_argument('--list-queries', action='store_true', help='List eligible query IDs and exit, without loading models')
    ap.add_argument('--self-test', action='store_true', help='Run lightweight synthetic metric/cache checks; no training or images')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if min(args.threads, args.batch_size, args.rank_batch, args.per_stratum) < 1 or args.workers < 0:
        raise ValueError('Thread/batch/sample counts must be positive; workers may be zero.')
    budget = available_cpus()
    requested = (args.threads, args.workers)
    args.workers = min(args.workers, max(0, budget - 1))
    args.threads = min(args.threads, max(1, budget - args.workers))
    if requested != (args.threads, args.workers):
        print('CPU allocation limit {}: using {} inference threads + {} loader workers.'
              .format(budget, args.threads, args.workers), flush=True)
    args.repo = str(Path(args.repo).resolve())
    args.data_root = str(Path(args.data_root).resolve())
    args.out = args.out or str(Path(args.repo) / ('mvpr_retrieval_' + args.protocol.lower()))
    args.cache_dir = args.cache_dir or str(Path(args.repo) / 'mvpr_retrieval_feature_cache')
    repo, data = Path(args.repo), Path(args.data_root)
    if not (repo / 'model/make_model.py').is_file():
        raise FileNotFoundError('Repository is not available: ' + str(repo))
    sys.path.insert(0, str(repo))
    # Cap libraries before importing torch; worker processes separately use one thread.
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    import torch
    args.torch_version = torch.__version__
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not visible; use --device cpu.')
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from datasets.whu_mars import WHU_MARS
    dataset = WHU_MARS(root=str(data), modalities=list(MODALITIES), protocol='ALL', subdir='WHU-MARS')
    qrows = [r for m in MODALITIES for r in dataset.query[m]]
    grows = [r for m in MODALITIES for r in dataset.gallery[m]]
    q = [image_record(r, i, 'query') for i, r in enumerate(qrows)]
    g = [image_record(r, i, 'gallery') for i, r in enumerate(grows)]
    if (len(q), len(g), dataset.num_train_pids) != (6405, 93609, 500):
        raise ValueError('Unexpected WHU-MARS-1000 split counts: {} / {} / {}'.format(
            len(q), len(g), dataset.num_train_pids))
    # Keep cache ordering invariant across ALL/GD. Apply protocol filters only
    # to selection/ranking metadata, exactly as the benchmark GD loader does.
    q_all, g_all = q, g
    if args.protocol == 'GD':
        keep = lambda r: 1000 <= r['pid'] < 2000 and r['camera_zero_based'] <= 4
        q = [dict(r, index=i, full_query_index=r['index'])
             for i, r in enumerate([x for x in q_all if keep(x)])]
        g = [dict(r, index=i, full_gallery_index=r['index'])
             for i, r in enumerate([x for x in g_all if keep(x)])]
        if (len(q), len(g)) != (1197, 20508):
            raise ValueError('GD count mismatch: {} / {}'.format(len(q), len(g)))
    selected, selection = choose_queries(q, g, args)
    if args.list_queries:
        for i in eligible_queries(q, g):
            r = q[i]
            if args.query_view != 'all' and args.query_view != r['view']:
                continue
            if args.query_modality != 'all' and args.query_modality != r['modality']:
                continue
            print('{}\t{}\tPID={}\t{}\t{}'.format(i, r['query_id'], r['pid'], r['camera'], r['view']))
        return 0
    if not (data / 'ViT-B-16.pt').is_file():
        raise FileNotFoundError('Raw CLIP initialization file required by model construction: '
                                + str(data / 'ViT-B-16.pt'))
    out = Path(args.out).resolve()
    source_files = ['model/make_model.py', 'model/backbones/vit_pytorch.py',
                    'datasets/whu_mars.py', 'datasets/bases.py', 'datasets/make_dataloader.py',
                    'utils/metrics.py']
    source_hashes = {name: sha256(repo / name) for name in source_files}
    # Save the selection BEFORE inference, so selection is independently auditable.
    write_json(out / 'selection_before_inference.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        selection=selection, queries=[q[i] for i in selected]))
    full_selected = [q[i].get('full_query_index', i) for i in selected]
    print('Selected {} queries before inference; device={}, threads={}, workers={}, batch={}'.format(
        len(selected), args.device, args.threads, args.workers, args.batch_size), flush=True)
    results, provenance = {}, {}
    for tag in ('baseline', 'mvpr'):
        gallery_indices = [r.get('full_gallery_index', r['index']) for r in g]
        cache, prov = ensure_features(tag, q_all, g_all, full_selected, args, dataset,
                                      source_hashes, gallery_indices=gallery_indices)
        if args.protocol == 'ALL':
            results[tag] = evaluate_selected(cache, q, g, selected, args)
        else:
            # A small adapter exposes only GD rows without mutating the full cache.
            class FilteredCache:
                pass
            filtered = FilteredCache()
            filtered.feat = np.concatenate([
                np.asarray(cache.feat[[r['full_query_index'] for r in q]]),
                np.asarray(cache.feat[[len(q_all) + r['full_gallery_index'] for r in g]])], axis=0)
            results[tag] = evaluate_selected(filtered, q, g, selected, args)
        provenance[tag] = prov
        del cache
    export_results(out, q, g, selected, results, selection, provenance, args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. Saved feature batches remain in the cache; rerun the same command to resume.',
              file=sys.stderr)
        sys.exit(130)
