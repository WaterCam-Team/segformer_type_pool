"""Export the deploy model to ONNX and check it against torch.

Only possible because the text table is now a constant buffer: with CLIP on the
inference path the graph contained a transformer over learned prompts, run per
image. In deploy mode the prompt learner is an einsum, an argmax and a gather.
"""
import argparse, time
import mmcv, numpy as np, torch, torch.nn as nn
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor


def patch_resize_for_export():
    """Make the decode head's resizes scale-based, for the trace only.

    segformer_head_denseclip resizes _c4/_c3/_c2 to size=c1.size()[2:], and
    mmseg.ops.resize turns a torch.Size into Python ints -- pinning the graph
    to the traced resolution. The three are always exact power-of-two ratios
    (strides 32/16/8 against c1's 4), so scale_factor is equivalent *provided
    the input dimensions are divisible by 32*; otherwise ceil division makes
    them differ by a pixel, which is why the head itself is left alone.
    """
    from mmseg.models.decode_heads import segformer_head_denseclip as head
    import torch.nn.functional as F
    orig_resize = head.resize

    def resize(input, size=None, scale_factor=None, mode='nearest',
               align_corners=None, warning=True):
        if size is not None and scale_factor is None:
            ih, iw = int(input.shape[2]), int(input.shape[3])
            oh, ow = int(size[0]), int(size[1])
            if ih and iw and oh % ih == 0 and ow % iw == 0 and oh // ih == ow // iw:
                return F.interpolate(input, scale_factor=float(oh // ih),
                                     mode=mode, align_corners=align_corners)
        return orig_resize(input, size, scale_factor, mode, align_corners, warning)

    head.resize = resize
    return lambda: setattr(head, 'resize', orig_resize)


class Wrap(nn.Module):
    """img -> seg logits. Skips mmseg's DataContainer plumbing."""
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, img):
        # Mirrors encode_decode, except the final upsample uses scale_factor
        # rather than size=img.shape[2:]. Passing a size turns into Python ints
        # inside mmseg.ops.resize and pins the output resolution; the decode
        # head runs at stride 4, so a fixed factor is equivalent whenever the
        # input dimensions are divisible by 4 -- which SIZE_DIVISIBILITY=32
        # already guarantees. img_metas is unused by forward_test.
        x = self.seg.extract_feat(img)
        x, _score = self.seg.after_extract_feat(x, [], [])
        out = self.seg._decode_head_forward_test(x, None)
        return torch.nn.functional.interpolate(
            out, scale_factor=4, mode='bilinear',
            align_corners=self.seg.align_corners)


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

    unpatch = patch_resize_for_export() if args.dynamic else (lambda: None)
    print(f'tracing at {args.height}x{args.width}, opset {args.opset} ...', flush=True)
    with torch.no_grad():
        t0 = time.time()
        ref = wrap(dummy)
        print(f'  torch forward ok: {tuple(ref.shape)}  ({time.time()-t0:.1f}s)')
        torch.onnx.export(
            wrap, dummy, args.out,
            input_names=['input'], output_names=['logits'],
            opset_version=args.opset, do_constant_folding=args.fold,
            dynamic_axes=({'input': {2: 'h', 3: 'w'},
                           'logits': {2: 'h', 3: 'w'}}
                          if args.dynamic else None))
    unpatch()
    print(f'exported -> {args.out}')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    t0 = time.time()
    got = sess.run(None, {'input': dummy.numpy()})[0]
    print(f'  onnxruntime forward ok: {got.shape}  ({time.time()-t0:.1f}s)')

    d = np.abs(got - ref.numpy())
    agree = (got.argmax(1) == ref.numpy().argmax(1)).mean()
    print(f'\nat traced size {args.height}x{args.width}:')
    print(f'  max abs logit diff : {d.max():.3e}')
    print(f'  argmax agreement   : {agree*100:.4f}%')

    if args.dynamic:
        # The point of dynamic axes: a size never traced.
        h2, w2 = 384, 640
        print(f'\nat UNTRACED size {h2}x{w2}:')
        d2 = torch.randn(1, 3, h2, w2)
        with torch.no_grad():
            ref2 = wrap(d2).numpy()
        got2 = sess.run(None, {'input': d2.numpy()})[0]
        if got2.shape != ref2.shape:
            print(f'  SHAPE MISMATCH onnx={got2.shape} torch={ref2.shape}')
        else:
            a2 = np.abs(got2 - ref2)
            ag2 = (got2.argmax(1) == ref2.argmax(1)).mean()
            print(f'  output shape       : {got2.shape}')
            print(f'  max abs logit diff : {a2.max():.3e}')
            print(f'  argmax agreement   : {ag2*100:.4f}%')


if __name__ == '__main__':
    main()
