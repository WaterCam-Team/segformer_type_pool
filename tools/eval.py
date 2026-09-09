"""Evaluate a SegFormer + DenseCLIP type-pool checkpoint on the water subsets.

Reports mIoU / mAcc / aAcc and per-class IoU for every subset in the config's
`eval_sets`. Numerically identical to CustomDataset.evaluate(metric='mIoU') --
the same intersect_and_union with the same ignore_index / label_map /
reduce_zero_label -- with two deliberate differences:

  * Counts are accumulated per image, so only four length-num_classes vectors
    are held instead of every full-resolution prediction and GT mask. The stock
    in-memory path gets OOM-killed on the 1348-image subset.

  * total_intersect_and_union() silently `continue`s past any image whose
    prediction and GT shapes disagree. Some GT masks in these datasets are
    saved as 3-channel RGB (the label replicated across channels), so that
    check drops them from the metric without a word -- 74% of normal_flood.
    Here such images are counted and reported, and --squeeze-rgb-gt takes
    channel 0 so every image contributes.

Usage:
    PYTHONPATH=. python tools/eval.py \
        --config configs/type_pool_b0_eval.py \
        --checkpoint weights/type_pool_b0_40k.pth \
        --squeeze-rgb-gt

Add --sets NAME [NAME ...] to evaluate only some of the subsets; the ones you
leave out are never loaded, so their data_root does not have to be valid.
"""
import argparse
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from terminaltables import AsciiTable

from mmseg.core.evaluation.metrics import intersect_and_union
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True, help='eval config file')
    p.add_argument('--checkpoint', required=True, help='.pth to evaluate')
    p.add_argument(
        '--squeeze-rgb-gt',
        action='store_true',
        help='take channel 0 of multi-channel GT masks instead of letting '
             'them be dropped by the pred/GT shape check')
    p.add_argument(
        '--sets',
        nargs='+',
        metavar='NAME',
        help="evaluate only these subsets instead of all of them, e.g. "
             "--sets normal_flood. Names work with or without the 'test_' "
             "prefix. Subsets you do not name are never loaded, so their "
             "data_root in the config can be left as-is.")
    p.add_argument('--gpu-id', type=int, default=0)
    return p.parse_args()


def build_model(cfg, checkpoint, gpu_id):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    meta = load_checkpoint(model, checkpoint, map_location='cpu')['meta']
    model.CLASSES = meta['CLASSES']
    model.PALETTE = meta['PALETTE']
    print(f'\ncheckpoint : {checkpoint}')
    print(f'iter       : {meta.get("iter")}   saved: {meta.get("time")}')
    print(f'exp_name   : {meta.get("exp_name")}')
    print(f'CLASSES    : {model.CLASSES}\n', flush=True)
    model = MMDataParallel(model.cuda(gpu_id), device_ids=[gpu_id])
    model.eval()
    return model


def evaluate_subset(model, dataset_cfg, workers, squeeze_rgb_gt):
    """Return (metrics dict, n_images, n_shape_mismatch)."""
    ds = build_dataset(dataset_cfg, dict(test_mode=True))
    loader = build_dataloader(
        ds, samples_per_gpu=1, workers_per_gpu=workers, dist=False,
        shuffle=False)
    n_cls = len(ds.CLASSES)
    intersect, union, pred_area, label_area = (np.zeros(n_cls) for _ in range(4))
    mismatch = 0

    bar = mmcv.ProgressBar(len(ds))
    for i, data in enumerate(loader):
        with torch.no_grad():
            pred = model(return_loss=False, **data)[0]

        gt = mmcv.imread(
            osp.join(ds.ann_dir, ds.img_infos[i]['ann']['seg_map']),
            flag='unchanged', backend='pillow')
        if squeeze_rgb_gt and gt.ndim == 3:
            gt = gt[..., 0]

        if pred.shape != gt.shape:
            # same rule as total_intersect_and_union, but counted
            mismatch += 1
            bar.update()
            continue

        areas = intersect_and_union(
            pred, gt, n_cls, ds.ignore_index, ds.label_map,
            ds.reduce_zero_label)
        for total, area in zip((intersect, union, pred_area, label_area), areas):
            total += area
        bar.update()
    print()

    iou = intersect / union
    acc = intersect / label_area
    metrics = dict(
        mIoU=np.nanmean(iou) * 100,
        mAcc=np.nanmean(acc) * 100,
        aAcc=intersect.sum() / label_area.sum() * 100,
        **{f'IoU.{ds.CLASSES[c]}': iou[c] * 100 for c in range(n_cls)})
    return metrics, len(ds), mismatch


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    eval_sets = cfg.get('eval_sets') or [
        k for k in cfg.data if k.startswith('test_')]
    missing = [k for k in eval_sets if k not in cfg.data]
    if missing:
        raise KeyError(f'eval_sets entries missing from cfg.data: {missing}')

    if args.sets:
        available = [k for k in cfg.data if k.startswith('test_')]
        chosen = []
        for name in args.sets:
            key = name if name in cfg.data else f'test_{name}'
            if key not in available:
                raise SystemExit(
                    f"unknown subset {name!r}. Available: "
                    + ', '.join(k[len('test_'):] for k in available))
            if key not in chosen:
                chosen.append(key)
        eval_sets = chosen

    model = build_model(cfg, args.checkpoint, args.gpu_id)

    summary = {}
    for key in eval_sets:
        name = key[len('test_'):] if key.startswith('test_') else key
        root = cfg.data[key]['data_root']
        print('=' * 72)
        print(f'[{name}]  <-  {root}')
        print('=' * 72, flush=True)

        metrics, n_imgs, mismatch = evaluate_subset(
            model, cfg.data[key], cfg.data.workers_per_gpu, args.squeeze_rgb_gt)
        if mismatch:
            print(f'  WARNING: {mismatch}/{n_imgs} images excluded '
                  f'(prediction/GT shape mismatch)')
            if not args.squeeze_rgb_gt:
                print('  -> re-run with --squeeze-rgb-gt to include them')
        summary[name] = dict(metrics, images=n_imgs - mismatch)

    cols = ['images', 'mIoU', 'mAcc', 'aAcc'] + [
        k for k in next(iter(summary.values())) if k.startswith('IoU.')]
    rows = [['subset'] + cols]
    for name, m in summary.items():
        rows.append([name] + [
            m['images'] if c == 'images' else round(m[c], 2) for c in cols])
    print('\n' + '=' * 72)
    print('ALL SUBSETS')
    print('=' * 72)
    print(AsciiTable(rows).table)


if __name__ == '__main__':
    main()
