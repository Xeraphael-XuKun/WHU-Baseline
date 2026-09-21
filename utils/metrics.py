import torch
import numpy as np
from utils.reranking import re_ranking
import torch.distributed as dist
from collections import defaultdict

DISTMAT_Q_CHUNK = 4000

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu()

def iter_euclidean_distance_chunks(qf, gf, q_chunk_size=DISTMAT_Q_CHUNK):
    num_q = qf.shape[0]
    gf_t = gf.t().contiguous()
    gf_sq = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()

    for start in range(0, num_q, q_chunk_size):
        end = min(start + q_chunk_size, num_q)
        q_chunk = qf[start:end]
        dist_chunk = torch.pow(q_chunk, 2).sum(dim=1, keepdim=True) + gf_sq
        dist_chunk.addmm_(q_chunk, gf_t, beta=1, alpha=-2)
        yield dist_chunk.cpu()

def compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK):
    indices_chunks = []
    for dist_chunk in iter_euclidean_distance_chunks(qf, gf, q_chunk_size=q_chunk_size):
        if top_k:
            k = min(top_k, dist_chunk.shape[1])
            idx_chunk = torch.topk(dist_chunk, k=k, dim=1, largest=False).indices
        else:
            idx_chunk = torch.argsort(dist_chunk, dim=1)
        indices_chunks.append(idx_chunk.numpy().astype(np.int32))
        del dist_chunk

    if not indices_chunks:
        width = min(top_k, gf.shape[0]) if top_k else gf.shape[0]
        return np.empty((0, width), dtype=np.int32)
    return np.concatenate(indices_chunks, axis=0)

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def top_k_indices(distmat, top_k):
    num_q, num_g = distmat.shape
    k = min(top_k, num_g)
    if k <= 0:
        return np.empty((num_q, 0), dtype=np.int32)
    top_k_idx = np.argpartition(distmat, kth=k - 1, axis=1)[:, :k]
    top_k_dist = np.take_along_axis(distmat, top_k_idx, axis=1)
    sorted_order = np.argsort(top_k_dist, axis=1)
    indices = np.take_along_axis(top_k_idx, sorted_order, axis=1)
    return indices.astype(np.int32)

def junk_mask(g_pids_ordered, g_camids_ordered, q_pid, q_camid, metric):
    """Which ranked gallery entries are discarded before scoring this query.

    The two conventions in use disagree, and the disagreement is worth several
    points of mAP, so it is a setting rather than a constant:

    ``market``
        Market-1501, and what fast-reid uses for CARGO.  Only the query's own
        identity seen from the query's own camera is junk.  Other identities
        from that camera stay in the gallery as legitimate distractors.
    ``sysu``
        SYSU-MM01, and what this repo has always used for WHU-MARS.  The
        query's entire camera is dropped, distractors included, which makes
        the retrieval problem strictly easier.

    Note that the set of *relevant* gallery items is the same either way --
    same identity, different camera -- so only this mask changes.
    """
    if metric == 'market':
        return (g_pids_ordered == q_pid) & (g_camids_ordered == q_camid)
    if metric == 'sysu':
        return g_camids_ordered == q_camid
    raise ValueError("TEST.METRIC must be 'sysu' or 'market', got {!r}".format(metric))


def eval_func_top_k(indices, q_pids, g_pids, q_camids, g_camids, q_mids, g_mids, max_rank=50,
                    metric='sysu'):
    """Evaluation with market1501 metric, optimized for large datasets by limiting to top_k gallery samples per query.

    Returns mINP as NaN, on purpose.  mINP is decided by the position of the
    HARDEST true match, which is exactly the one a top-k cut is most likely to
    have thrown away; anything computed from the survivors would be an
    optimistic number wearing the right name.  NaN says "this mode cannot answer
    that", which is the honest reading.  TEST.TOP_K_EVAL is 0 in every config we
    run, so the real path is eval_func above.
    """
    num_q, num_g = indices.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_mid = q_mids[q_idx]
        
        valid_gallery = (g_pids == q_pid) & (g_camids != q_camid)
        num_rel = np.sum(valid_gallery)
        if num_rel == 0:
            # this condition is true when query identity does not appear in gallery
            continue
        
        order = indices[q_idx]  # select one row
        remove = junk_mask(g_pids[order], g_camids[order], q_pid, q_camid, metric)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if orig_cmc.size == 0:
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if cmc.shape[0] < max_rank:
            pad_width = max_rank - cmc.shape[0]
            if cmc.shape[0] > 0:
                cmc = np.pad(cmc, (0, pad_width), mode='edge')
            else:
                cmc = np.pad(cmc, (0, pad_width), mode='constant')
        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP, float('nan')

def eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50, metric='sysu'):
    """CMC / mAP / mINP over a full ranking.  See junk_mask for `metric`.

    mINP (Ye et al., TPAMI 2021, the AGW paper) scores how deep you have to read
    before the LAST true match appears: INP = num_rel / rank_of_hardest_match,
    averaged over queries.  mAP is dominated by the easy matches near the top, so
    a method can hold its mAP while leaving one view or one spectrum of an
    identity stranded far down the list; mINP is the number that notices.  WHU-
    MARS reports it for the VI protocol and for the 3x3 modality table.

    It needs the WHOLE ranking -- see eval_func_top_k for why it cannot be had
    from a truncated one.
    """
    num_q, num_g = indices.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]  # select one row
        remove = junk_mask(g_pids[order], g_camids[order], q_pid, q_camid, metric)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # mINP, read off the running total BEFORE it is clipped: at the position
        # of the last true match the cumsum equals the number of true matches, so
        # cmc[last] / (last + 1) is exactly num_rel / rank_of_hardest_match.
        # Computing it here rather than after the clip is not a style choice --
        # once cmc is clipped to 0/1 the count is gone.
        last_hit = int(np.max(np.where(orig_cmc == 1)))
        all_INP.append(cmc[last_hit] / (last_hit + 1.0))

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

_gloo_pg = None
def get_gloo_group():
    global _gloo_pg
    if _gloo_pg is None and dist.is_initialized() and dist.get_world_size() > 1:
        _gloo_pg = dist.new_group(backend="gloo")
    return _gloo_pg

def cmc_at(cmc, rank):
    """CMC at 1-based `rank`, or the last available entry if the curve is shorter.

    A thin subset -- one modality pair inside one view, say -- can leave fewer
    than ten gallery items, and eval_func trims max_rank to match.  Indexing
    past the end would raise in the middle of an evaluation that is otherwise
    fine, so the curve saturates instead: past its end it is flat by
    construction anyway.
    """
    if cmc is None or len(cmc) == 0:
        return float('nan')
    return float(cmc[min(rank, len(cmc)) - 1])


def as_bool(value, name='flag'):
    """Coerce a yes/no config string -- or a real bool -- to a bool.

    yacs stores TEST.FEAT_NORM as the string 'yes' or 'no', and this class used
    to test it with a bare `if self.feat_norm:`.  Every non-empty string is
    True, so 'no' meant yes: the switch had never done anything.  Measured
    2026-08-13 -- `bash reeval.sh whu_recipe_hihr TEST.FEAT_NORM no` returned
    10.23 / 26.65, the normalised number to the digit, and nothing in the log
    said the setting had been ignored.

    That is the third time on this project a key has been accepted by yacs and
    then silently dropped downstream (MODEL.ID_LOSS_TYPE is read by nobody;
    TEXT_MODALITY was ignored by the mtext runs).  The coercion happens once,
    here at the boundary, and anything unrecognised RAISES -- defaulting to
    either answer is exactly what made this invisible for months.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('yes', 'true', 'on', '1'):
            return True
        if v in ('no', 'false', 'off', '0', ''):
            return False
    raise ValueError(
        '{} expects yes/no (or a bool), got {!r}'.format(name, value))


class R1_mAP_eval:
    def __init__(self, max_rank=50, feat_norm=True, reranking=False,
                 top_k=50000, logger=None, metric='sysu', aerial_cams=()):
        self.max_rank   = max_rank
        self.feat_norm  = as_bool(feat_norm, 'TEST.FEAT_NORM')
        self.reranking  = reranking
        self.top_k      = top_k
        self.logger     = logger
        # 'sysu' keeps every WHU-MARS number in this repo comparable; CARGO
        # and the rest of the AG-ReID benchmarks score with 'market'.
        self.metric     = metric
        # Zero-based camera ids shot from the air.  Empty disables the view
        # matrix; it is purely an extra readout and changes no reported number.
        self.aerial_cams = tuple(int(c) for c in aerial_cams)
        self.reset()

    def reset(self):
        self.feats   = defaultdict(list)
        self.pids    = defaultdict(list)
        self.camids  = defaultdict(list)
        self.modids  = defaultdict(list)
        self.query_nums = {}

        self.qf, self.gf = [], []
        self.q_pids, self.q_camids, self.q_mids = [], [], []
        self.g_pids, self.g_camids, self.g_mids = [], [], []

    def update(self, output, mode: int):
        feat, pid, camid, modid = output
        self.feats[mode].append(feat.cpu())
        self.pids[mode].extend(np.asarray(pid))
        self.camids[mode].extend(np.asarray(camid))
        self.modids[mode].extend(np.asarray(modid))

    def set_query_num(self, mode: int, num_query: int):
        self.query_nums[mode] = num_query

    def split_all(self):
        for mode, num_query in self.query_nums.items():
            if not self.feats[mode]:
                continue
            feats_all = torch.cat(self.feats[mode], dim=0)
            qf_cur, gf_cur = feats_all[:num_query], feats_all[num_query:]

            self.qf.append(qf_cur)
            self.gf.append(gf_cur)

            self.q_pids.extend(self.pids[mode][:num_query])
            self.q_camids.extend(self.camids[mode][:num_query])
            self.q_mids.extend(self.modids[mode][:num_query])

            self.g_pids.extend(self.pids[mode][num_query:])
            self.g_camids.extend(self.camids[mode][num_query:])
            self.g_mids.extend(self.modids[mode][num_query:])

            self.feats[mode].clear()
            self.pids[mode].clear()
            self.camids[mode].clear()
            self.modids[mode].clear()

            self.logger and self.logger.info(f"=> Split mode {mode}: "
                                             f"query {num_query}, total query {len(self.q_pids)}")

    def _gather_to_rank0(self):
        multi_proc = dist.is_initialized() and dist.get_world_size() > 1
        if not multi_proc:
            # Single process (single GPU / DataParallel) path.
            return True, None

        # -------- DDP (world_size > 1) --------
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        gloo_pg    = get_gloo_group()

        local_pack = None
        if self.qf:
            local_pack = dict(
                qf=torch.cat(self.qf, dim=0).cpu(),
                gf=torch.cat(self.gf, dim=0).cpu(),
                qpid=np.asarray(self.q_pids, dtype=np.int64),
                qcam=np.asarray(self.q_camids, dtype=np.int64),
                qmid=np.asarray(self.q_mids, dtype=np.int64),
                gpid=np.asarray(self.g_pids, dtype=np.int64),
                gcam=np.asarray(self.g_camids, dtype=np.int64),
                gmid=np.asarray(self.g_mids, dtype=np.int64)
            )

        if rank == 0:
            gather_list = [None] * world_size
            dist.gather_object(local_pack, gather_list, dst=0, group=gloo_pg)
            return True, gather_list
        else:
            dist.gather_object(local_pack, dst=0, group=gloo_pg)
            return False, None

    def compute(self):
        is_rank0, packs = self._gather_to_rank0()
        if not is_rank0:
            return (None,)*8

        if packs is None:
            qf = torch.cat(self.qf, dim=0)
            gf = torch.cat(self.gf, dim=0)
            q_pids, q_camids, q_mids = map(np.asarray,
                                           (self.q_pids, self.q_camids, self.q_mids))
            g_pids, g_camids, g_mids = map(np.asarray,
                                           (self.g_pids, self.g_camids, self.g_mids))
        else:
            self.qf, self.gf = [], []
            self.q_pids, self.q_camids, self.q_mids = [], [], []
            self.g_pids, self.g_camids, self.g_mids = [], [], []

            for pack in packs:
                if pack is None:
                    continue
                self.qf.append(pack["qf"])
                self.gf.append(pack["gf"])
                qpid_pack = pack["qpid"]
                qcam_pack = pack["qcam"]
                qmid_pack = pack["qmid"]
                gpid_pack = pack["gpid"]
                gcam_pack = pack["gcam"]
                gmid_pack = pack["gmid"]

                self.q_pids.extend(qpid_pack.tolist() if isinstance(qpid_pack, np.ndarray) else qpid_pack)
                self.q_camids.extend(qcam_pack.tolist() if isinstance(qcam_pack, np.ndarray) else qcam_pack)
                self.q_mids.extend(qmid_pack.tolist() if isinstance(qmid_pack, np.ndarray) else qmid_pack)
                self.g_pids.extend(gpid_pack.tolist() if isinstance(gpid_pack, np.ndarray) else gpid_pack)
                self.g_camids.extend(gcam_pack.tolist() if isinstance(gcam_pack, np.ndarray) else gcam_pack)
                self.g_mids.extend(gmid_pack.tolist() if isinstance(gmid_pack, np.ndarray) else gmid_pack)

            qf = torch.cat(self.qf, dim=0)
            gf = torch.cat(self.gf, dim=0)
            q_pids, q_camids, q_mids = map(np.asarray,
                                           (self.q_pids, self.q_camids, self.q_mids))
            g_pids, g_camids, g_mids = map(np.asarray,
                                           (self.g_pids, self.g_camids, self.g_mids))

        # Said either way round.  Silence used to mean "normalised" and also
        # "the switch was ignored", which is how the broken flag survived.
        if self.feat_norm:
            self.logger and self.logger.info("The test feature is normalized")
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)
        else:
            self.logger and self.logger.info(
                "The test feature is NOT normalized (TEST.FEAT_NORM off)")

        distmat = None
        if self.reranking:
            self.logger and self.logger.info('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            self.logger and self.logger.info(f'=> Computing DistMat in q-chunks (chunk={DISTMAT_Q_CHUNK})')

        if self.top_k:
            self.logger and self.logger.info('=> Computing top_k_indices')
            if self.reranking:
                indices = top_k_indices(distmat, self.top_k)
            else:
                indices = compute_indices_chunked(qf, gf, top_k=self.top_k, q_chunk_size=DISTMAT_Q_CHUNK)
            cmc, mAP, mINP = eval_func_top_k(indices, q_pids, g_pids,
                                             q_camids, g_camids, q_mids, g_mids,
                                             metric=self.metric)
        else:
            if self.reranking:
                indices = np.argsort(distmat, axis=1).astype(np.int32)
            else:
                indices = compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK)
            cmc, mAP, mINP = eval_func(indices, q_pids, g_pids,
                                       q_camids, g_camids, metric=self.metric)


        def pair_eval(q_sel, g_sel):
            """mAP / CMC restricted to a (query subset, gallery subset) pair.

            The junk rule still sees the real camera ids, so the usual
            same-camera exclusion applies inside the subset -- restricting the
            gallery does not smuggle the query's own camera back in.
            """
            if q_sel.size == 0 or g_sel.size == 0:
                return None, None, None
            if self.reranking:
                idx = np.argsort(distmat[q_sel][:, g_sel], axis=1).astype(np.int32)
            else:
                idx = compute_indices_chunked(
                    qf.index_select(0, torch.from_numpy(q_sel).long()),
                    gf.index_select(0, torch.from_numpy(g_sel).long()),
                    top_k=0, q_chunk_size=DISTMAT_Q_CHUNK)
            try:
                return eval_func(idx, q_pids[q_sel], g_pids[g_sel],
                                 q_camids[q_sel], g_camids[g_sel],
                                 max_rank=self.max_rank, metric=self.metric)
            except AssertionError:
                # "all query identities do not appear in gallery" -- possible in
                # a thin subset, and not a reason to lose the whole evaluation.
                return None, None, None

        # ---- view pairs ----------------------------------------------------
        # The modality matrix below answers "how well does RGB match thermal".
        # It cannot answer "how well does an aerial query match a ground
        # gallery", which is the question the positional correction exists for.
        # Views come from the camera ids, which the junk rule already requires,
        # so this costs nothing extra to compute.
        view_pair_results = []
        if len(self.aerial_cams) and q_camids.size > 0:
            aerial = np.asarray(sorted(self.aerial_cams))
            q_view = np.isin(q_camids, aerial)          # True = aerial
            g_view = np.isin(g_camids, aerial)
            self.logger and self.logger.info(
                'Starting view-pair evaluation (aerial cams %s): query %d aerial / '
                '%d ground, gallery %d / %d' % (aerial.tolist(), int(q_view.sum()),
                                                int((~q_view).sum()), int(g_view.sum()),
                                                int((~g_view).sum())))
            names = {True: 'A', False: 'G'}
            for qa in (True, False):
                for ga in (True, False):
                    c, m, i = pair_eval(np.where(q_view == qa)[0],
                                        np.where(g_view == ga)[0])
                    if m is None:
                        self.logger and self.logger.info(
                            '  View (Q:%s, G:%s) -> empty or no valid query'
                            % (names[qa], names[ga]))
                        continue
                    view_pair_results.append((names[qa], names[ga], c, m, i))
                    # R-5 / R-10 / mINP were in the CMC array all along and were
                    # simply never printed.  The paper's AG table is mAP / R-1 /
                    # R-10, so a run that logged only R-1 could not be compared
                    # to it without re-evaluating.
                    self.logger and self.logger.info(
                        '  View (Q:%s, G:%s) -> mAP: %.2f%%, Rank-1: %.2f%%, '
                        'Rank-5: %.2f%%, Rank-10: %.2f%%, mINP: %.2f%%'
                        % (names[qa], names[ga], m * 100, c[0] * 100,
                           cmc_at(c, 5) * 100, cmc_at(c, 10) * 100, i * 100))

        modality_pair_results = []
        if q_mids.size > 0 and g_mids.size > 0: # Check if modality information is available
            unique_q_mod_ids = sorted(np.unique(q_mids))
            unique_g_mod_ids = sorted(np.unique(g_mids))
            self.logger and self.logger.info(f"Starting modality-pair specific evaluation. Query mods: {unique_q_mod_ids}, Gallery mods: {unique_g_mod_ids}")

            for q_mod_target in unique_q_mod_ids:
                for g_mod_target in unique_g_mod_ids:
                    query_indices_for_pair = np.where(q_mids == q_mod_target)[0]
                    gallery_indices_for_pair = np.where(g_mids == g_mod_target)[0]

                    q_pids_pair = q_pids[query_indices_for_pair]
                    q_camids_pair = q_camids[query_indices_for_pair]
                    g_pids_pair = g_pids[gallery_indices_for_pair]
                    g_camids_pair = g_camids[gallery_indices_for_pair]

                    # Calculate indices for this pair (full argsort, int32)
                    if self.reranking:
                        distmat_pair = distmat[query_indices_for_pair][:, gallery_indices_for_pair]
                        indices_pair = np.argsort(distmat_pair, axis=1).astype(np.int32)
                    else:
                        query_indices_tensor = torch.from_numpy(query_indices_for_pair).long()
                        gallery_indices_tensor = torch.from_numpy(gallery_indices_for_pair).long()
                        qf_pair = qf.index_select(0, query_indices_tensor)
                        gf_pair = gf.index_select(0, gallery_indices_tensor)
                        indices_pair = compute_indices_chunked(
                            qf_pair, gf_pair, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK
                        )
                    
                    # Use the same global eval_func for pairs. Ensure it can handle potentially smaller inputs.
                    pair_cmc, pair_mAP, pair_mINP = eval_func(
                        indices_pair, q_pids_pair, g_pids_pair,
                        q_camids_pair, g_camids_pair,
                        max_rank=self.max_rank, metric=self.metric)
                    
                    modality_pair_results.append(
                        (q_mod_target, g_mod_target, pair_cmc, pair_mAP, pair_mINP))
                    self.logger and self.logger.info(
                        f"  Pair (Q:{q_mod_target}, G:{g_mod_target}) -> "
                        f"mAP: {pair_mAP:.2%}, Rank-1: {pair_cmc[0]:.2%}, "
                        f"Rank-5: {cmc_at(pair_cmc, 5):.2%}, "
                        f"Rank-10: {cmc_at(pair_cmc, 10):.2%}, mINP: {pair_mINP:.2%}")
        else:
            self.logger and self.logger.info("Modality IDs (q_mids or g_mids) are empty or not available. Skipping pair-specific evaluation.")

        self.reset()
        return cmc, mAP, modality_pair_results, q_pids, q_camids, qf, gf, view_pair_results

