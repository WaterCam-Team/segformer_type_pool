"""Strip a training checkpoint down to what inference actually needs.

The 761 MB training checkpoint is mostly dead weight at inference:

    clip_model.*      149,626,666 params   571 MB
    optimizer state    32,981,440 entries  126 MB
    everything else    16,890,983 params    64 MB

The CLIP text tower exists only to turn prompt_ctx into a (C, 512) table of
text features, and that table is the same for every image. Compute it once,
store it as a buffer, and the tower can go -- along with the optimizer state,
which inference never reads.

    PYTHONPATH=. python tools/export_deploy.py \
        --config configs/type_pool_b0_eval.py \
        --checkpoint weights/type_pool_b0_40k.pth \
        --out weights/type_pool_b0_40k.deploy.pth

Evaluate the result by adding --deploy to tools/eval.py. Outputs are
identical: the arithmetic is the same, performed at export time instead of at
inference.
"""
import argparse
import os

import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='training checkpoint')
    p.add_argument('--out', required=True, help='where to write the deploy checkpoint')
    return p.parse_args()


def mb(path):
    return os.path.getsize(path) / 1024 ** 2


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.deploy = False          # need the real tower to build the table

    print('building model (this allocates the CLIP tower)...', flush=True)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    learner = model.half_prompt_learner
    print(f'computing text table: {learner.C} prompts through the CLIP text '
          f'transformer...', flush=True)
    with torch.no_grad():
        table = learner._build_text_table().detach().cpu()
    print(f'  text_table {tuple(table.shape)}  '
          f'mean {table.mean():.6f}  std {table.std():.6f}')

    state = model.state_dict()
    kept, dropped = {}, 0
    for k, v in state.items():
        if k.startswith('clip_model.'):
            dropped += v.numel()
            continue
        kept[k] = v.cpu()
    kept['half_prompt_learner.text_table'] = table

    out = {'meta': ckpt.get('meta', {}), 'state_dict': kept}
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    torch.save(out, args.out)

    print(f'\ndropped clip_model.*   {dropped:>14,} params')
    print(f'dropped optimizer state (not carried over)')
    print(f'kept                   {sum(v.numel() for v in kept.values()):>14,} params'
          f'  in {len(kept)} tensors')
    print(f'\n{args.checkpoint}  {mb(args.checkpoint):8.0f} MB')
    print(f'{args.out}  {mb(args.out):8.0f} MB'
          f'   ({mb(args.checkpoint) / mb(args.out):.1f}x smaller)')
    print('\nEvaluate it with:  tools/eval.py --deploy --checkpoint ' + args.out)


if __name__ == '__main__':
    main()
