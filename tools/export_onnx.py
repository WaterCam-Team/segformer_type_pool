"""Export the deploy model to ONNX and check it against torch.

Only possible because the text table is now a constant buffer: with CLIP on the
inference path the graph contained a transformer over learned prompts, run per
image. In deploy mode the prompt learner is an einsum, an argmax and a gather.
"""
import argparse, time
import mmcv, numpy as np, torch, torch.nn as nn
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor


class Wrap(nn.Module):
    """img -> seg logits. Skips mmseg's DataContainer plumbing."""
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, img):
        # img_metas is unused by BaseDecodeHead.forward_test
        return self.seg.encode_decode(img, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--no-fold', dest='fold', action='store_false',
                    help='disable constant folding')
    ap.add_argument('--dynamic', action='store_true',
                    help='declare dynamic H/W axes (may break rank inference)')
    args = ap.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.deploy = True
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cpu().eval()

    wrap = Wrap(model).eval()
    dummy = torch.randn(1, 3, args.height, args.width)

    print(f'tracing at {args.height}x{args.width}, opset {args.opset} ...', flush=True)
    with torch.no_grad():
        t0 = time.time()
        ref = wrap(dummy)
        print(f'  torch forward ok: {tuple(ref.shape)}  ({time.time()-t0:.1f}s)')
        torch.onnx.export(
            wrap, dummy, args.out,
            input_names=['input'], output_names=['logits'],
            opset_version=args.opset, do_constant_folding=args.fold,
            dynamic_axes=({'input': {0: 'n', 2: 'h', 3: 'w'},
                           'logits': {0: 'n', 2: 'h', 3: 'w'}}
                          if args.dynamic else None))
    print(f'exported -> {args.out}')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    t0 = time.time()
    got = sess.run(None, {'input': dummy.numpy()})[0]
    print(f'  onnxruntime forward ok: {got.shape}  ({time.time()-t0:.1f}s)')

    d = np.abs(got - ref.numpy())
    agree = (got.argmax(1) == ref.numpy().argmax(1)).mean()
    print(f'\nmax abs logit diff : {d.max():.3e}')
    print(f'mean abs logit diff: {d.mean():.3e}')
    print(f'argmax agreement   : {agree*100:.4f}%')


if __name__ == '__main__':
    main()
