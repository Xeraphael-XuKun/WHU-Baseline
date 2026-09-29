import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F

def do_train(cfg, model, center_criterion, train_loader, val_loaders, optimizer, optimizer_center, scheduler, loss_fn, num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger('transreid.train')
    logger.info('start training')
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(2)]
    evaluator = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING, top_k=cfg.TEST.TOP_K_EVAL, logger=logger, metric=cfg.TEST.METRIC, aerial_cams=cfg.DATASETS.AERIAL_CAMS)
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    print(feat_dim, num_classes)
    group_size = cfg.DATALOADER.NUM_INSTANCE
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        n_iter = 0
        for (n_iter, (imgs, vid, camids)) in enumerate(train_loader):
            scheduler.step_update((epoch - 1) * len(train_loader) + n_iter)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)
            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)
            with amp.autocast(enabled=True):
                out = model(imgs, target, camids)
                (cls_score, global_feat, feat) = (out[0], out[1], out[2])
                (loss, il, tl) = loss_fn(cls_score, global_feat, target_rep)
                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                lrs = [g['lr'] for g in optimizer.param_groups]
                lr_str = '{:.2e}'.format(max(lrs)) if min(lrs) == max(lrs) else '{:.2e} (pretrained {:.2e})'.format(max(lrs), min(lrs))
                logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {}'.format(epoch, n_iter + 1, len(train_loader), loss_meter.avg, acc_meter.avg, lr_str))
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                logger.info('Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(epoch, time_per_batch, end_time - start_time))
        else:
            logger.info('Epoch {} done. Optimizer updates: {}/{} (actual/scheduler denominator). Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(epoch, n_iter + 1, len(train_loader), time_per_batch, train_loader.batch_size / time_per_batch))
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        if epoch % eval_period == 0:
            model.eval()
            dist_on = cfg.MODEL.DIST_TRAIN and dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)
            rank = dist.get_rank() if dist_on else 0
            world_sz = dist.get_world_size() if dist_on else 1
            for (mode, (val_loader, num_query)) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)
                if dist_on and (mode - 1) % world_sz != rank:
                    continue
                for (img, vid, camid, camidt, modid, _) in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img, mode=mode, camids=camidt)
                        evaluator.update((feat, vid, camid, modid), mode)
            evaluator.split_all()
            (cmc, mAP, *_) = evaluator.compute()
            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r - 1]:.2%}')
            torch.cuda.empty_cache()

def do_inference(cfg, model, val_loaders, num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')
    evaluator = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING, top_k=cfg.TEST.TOP_K_EVAL, logger=logger, metric=cfg.TEST.METRIC, aerial_cams=cfg.DATASETS.AERIAL_CAMS)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    model.eval()
    img_path_list = []
    for (mode, (val_loader, num_query)) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)
        for (img, vid, camid, camidt, modid, imgpath) in val_loader:
            with torch.no_grad():
                img = img.to(device)
                feat = model(img, mode=mode, camids=camidt)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)
    evaluator.split_all()
    (cmc, mAP, *_) = evaluator.compute()
    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return (cmc[0], cmc[4])
