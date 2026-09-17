# Running on a Raspberry Pi 4B (CPU)

Branch `pi-cpu-deploy`. Upstream `main` needs a CUDA GPU, downloads CLIP
ViT-B/16 through `open_clip` on first run, and loads a 761 MB checkpoint.
None of that is possible on the deployment Pi. This branch makes the
evaluation run there without changing a single output value.

**Every number below was measured on the target hardware** — a Raspberry Pi 4B
(Cortex-A72, 4 cores, 3.7 GiB) under the `5band` conda environment
(Python 3.9, torch 1.7.1, mmcv 1.3.0) — on the 13-image `challenging_flood`
sample that was available locally.

## Why the obvious approaches don't work

| Obstacle | Why it blocks |
|---|---|
| `torch 1.7.1` in the target env | Too old for `open_clip` 2.x, which `requirements.txt` pins |
| Upgrading torch | Needs mmcv-full rebuilt against it; hours of compilation on a Pi |
| Newer torch wheels | PyPI's aarch64 build is the NVIDIA Grace/CUDA one and `SIGILL`s on Armv8.0 |
| Installing `open_clip` | See row 1 |

The way out was noticing that `open_clip` is not actually needed.

## What changed, and why

### `7622e88` — CPU execution

`build_clip()` and the prompt learner's tokenizer call hardcoded `.cuda()`, and
`tools/eval.py` wrapped the model in `MMDataParallel` on a fixed device id, so
the model could not even be *constructed* without a GPU. Device is now selected
from availability, with an explicit `--device {cuda,cpu}` override.

`rasterio` sat at module scope in `pipelines/loading.py` but is absent from
`requirements.txt`, so `import mmseg.datasets` failed outright on any machine
without it — for a 5-band loader nothing in this config uses. Now imported
lazily.

### `8608e23` — vendored CLIP instead of `open_clip`

`build_clip()` pulled ViT-B/16 through `open_clip`: a ~600 MB download on first
run. But **every tensor it produced was then overwritten** by the checkpoint's
own `clip_model.*` weights, and the tokenizer's outputs land in registered
buffers the checkpoint also overwrites. The download existed only to allocate
tensors at the right shapes.

The repo already vendors that architecture in `mmseg/models/clip/`, so the
tower is now built there with the ViT-B/16 hyperparameters. Verified against
the checkpoint: **302 of its 303 `clip_model.*` keys match, with none missing.**
The one extra is `attn_mask`, `open_clip`'s buffer for the causal text mask,
which this implementation builds internally (`model.py:282`) — and which
accounts exactly for the 5,929 = 77² parameter-count difference.

This is what removes the torch 1.7.1 blocker: no `open_clip`, no download, no
network.

### `de503e8` — lazy optional imports

Importing anything under `mmseg.models` runs the package `__init__`, which
pulls in every backbone and head — so a top-level `import clip` in
`coop_my.py` made the whole package unimportable without the OpenAI CLIP
package. The vendored `clip.py` had the same problem one level down, importing
`tqdm`, `PIL` and `torchvision` at module scope though only `_download()` and
`_transform()` use them, neither of which `tokenize()` touches.

### `17af54b` — compute the prompt table once

`Half_PromptLearner_type_tool.forward` ran the CLIP text transformer `C` times
per image to build a `(C, 512)` table, then used the image **only to index into
it**. The table depends on `prompt_ctx`, the registered buffers and frozen CLIP
weights — never on `pixel_feat`. Every image after the first repeated identical
arithmetic.

Measured: this was **62% of total runtime** (~54 s of 88 s per image).
Training still recomputes each step, since `prompt_ctx` is being learned.

### `08ba3b3` — release the tower after caching

Once the table is cached, the 149.6 M-parameter tower is never read again, yet
stayed resident. Dropping both references saves **394 MB** of steady-state RSS
(831 MB vs 1225 MB, measured against a control run with the release stubbed
out). Less than the ~571 MB those weights occupy, because glibc retains some
freed arenas and inference activations reuse the rest.

### `4cfa816` — deploy export

`tools/export_deploy.py` computes the table once at export time, stores it as a
buffer, and drops both the tower and the optimizer state. `--deploy` on
`tools/eval.py` pairs with the result: `build_clip()` is never called, so unlike
the runtime release above the 571 MB is **never allocated at all**.

```
clip_model.*      149,626,666 params   571 MB   dropped
optimizer state    32,981,440 entries  126 MB   dropped
everything else    16,890,983 params    64 MB   kept
```

## Results

### Correctness

mIoU on `challenging_flood` was **89.19 across every configuration** —
original, vendored CLIP, cached table, released tower, and stripped
checkpoint. Identical to the decimal on all five metrics. That is what makes
these optimisations rather than approximations.

The deploy checkpoint loads with **0 missing and 0 unexpected keys**.

### Cumulative effect

| | upstream | this branch |
|---|---|---|
| CPU inference | unsupported | works |
| `open_clip` / 600 MB download | required | not needed |
| torch 1.7.1 environment | incompatible | works |
| checkpoint | 761 MB | **65 MB** (11.8×) |
| peak RSS | 1221 MB | **775 MB** |
| per image | 88 s | **33 s** |
| 13 images | — | 429 s |
| mIoU | 89.19 | **89.19** |

### Resolution tradeoff

Lowering `img_scale` in the test pipeline. Not free — quote the setting
alongside any number:

| `img_scale` | mIoU | 13 images | per image | peak RSS |
|---|---|---|---|---|
| (1024, 512) | 89.19 | 429 s | 33 s | 775 MB |
| (512, 384) | 87.42 | 172 s | 13.2 s | 478 MB |

2.5× faster and 38% less memory, for −1.77 mIoU.

### Dynamic int8 quantization

91% of this model's parameters are `nn.Linear` (15,045,632 of 16,490,852;
Conv2d is only 8.4%), which is exactly what `torch.quantization.quantize_dynamic`
covers — no calibration data, no retraining. All 74 Linear layers convert.

| | mIoU | 13 images | per image | peak RSS |
|---|---|---|---|---|
| fp32 | 89.19 | 429 s | 33.0 s | 775 MB |
| int8 dynamic (`qnnpack`) | 89.44 | 353 s | 27.2 s | 718 MB |

**1.22× faster, accuracy unchanged** within the noise of a 13-image sample.
Enable with `--quantize`.

The gain is real but smaller than the 91% weight coverage suggests, because
runtime is not distributed like parameters: most of those Linear weights are in
`context_decoder`, which runs over very few tokens, while the backbone does
spatial work at high resolution with only 3.3 M parameters. Dynamic
quantization also pays per-op activation quantize/dequantize overhead, which
eats into the gain on narrow matrices.

fp16 is not an option on this board: the A72 lacks `asimdhp`, so half precision
is emulated and would be slower.

### ONNX Runtime — the largest win by far

Steady state at 512x1024, warmup discarded, 3 timed iterations, same input:

| backend | per image | vs fp32 |
|---|---|---|
| torch fp32 | 50.44 s ± 0.11 | — |
| torch int8 dynamic | 45.01 s ± 2.21 | 1.12× |
| **onnxruntime 1.19.2** | **2.69 s ± 0.06** | **18.73×** |

**End to end through `tools/eval.py --onnx`, the full 13-image subset:**

| backend | mIoU | 13 images | per image |
|---|---|---|---|
| torch fp32 | 89.19 | 429 s | 33.0 s |
| torch int8 dynamic | 89.44 | 353 s | 27.2 s |
| **ONNX Runtime** | **89.51** | **31 s** | **2.4 s** |

13.8x faster than torch on the real pipeline, including JPEG decode, resize,
normalise, pad, crop and metric accumulation — none of which ONNX accelerates.
The +0.32 mIoU comes from the pad to a multiple of 32. That was first written
off as sampling noise; it is not. Measured directly afterwards, on identical
input ONNX agrees with torch **100.0000%** of the time, while torch agrees with
*itself* only **99.21%** between padded and unpadded input, the water fraction
moving about 0.2 points. So the shift is real and caused by the pad, not by the
conversion — it is simply small on this model. See the caveat below.

Numerically equivalent, not an approximation: max absolute logit difference
5.5e-05 (fp32 rounding) and **100.0000% argmax agreement** — every pixel's
predicted class identical to torch's.

The gap is this large because the environment's torch is 1.7.1, built for
aarch64 in December 2020 without modern ARM GEMM kernels, while ONNX Runtime
1.19.2 has tuned ARM64 kernels and fuses operators across the graph. Fusion
attacks the memory-bandwidth bound directly, which is why it succeeds where
threading, multiprocessing and quantization all gave little.

Export it with `tools/export_onnx.py --dynamic`. Three separate things pinned
the graph to its traced resolution, each fixed differently:

| cause | fix |
|---|---|
| `outputs[0].unsqueeze(0)` constant-folds to a rank-3 `(1, 1, 512)` initializer under torch 1.7, breaking a `Concat` | `outputs[0:1]` — same value, keeps rank 2 |
| `pixel_feat.reshape(B, C, H*W)` bakes the traced spatial size | `pixel_feat.flatten(2)` |
| the decode head resizes `_c4/_c3/_c2` to `size=c1.size()[2:]`, which `mmseg.ops.resize` turns into Python ints | `scale_factor`, **applied during tracing only** |

The first two are edits to the model; both are value-identical and the eval
still returns 89.19. The third is *not* — for input dimensions not divisible
by 32, ceil division in the strided convolutions makes `scale_factor` and
`size` differ by a pixel, so changing the head would alter torch's own output.
`patch_resize_for_export()` therefore confines that substitution to the
exported graph, and only where the ratio is exactly integral.

Verified at a resolution never traced: exported at 512x1024, run at 384x640,
output shape `(1, 2, 384, 640)`, max logit difference 6.7e-05, **100.0000%
argmax agreement**.

> **The ONNX model requires input dimensions divisible by 32**, which is what
> makes the `scale_factor` substitution valid. `SIZE_DIVISIBILITY: 32` exists
> in the config for this reason, but the test pipeline resizes with
> `keep_ratio` and no pad step, so `OnnxRunner` pads to reach a valid size.

#### Padding is not free

MiT's spatial-reduction attention pools globally, so a padded strip shifts
predictions across the **whole image**, not just at the border. Measured by
running torch against torch, with no ONNX involved:

| model | sample | padded vs unpadded | water fraction shift |
|---|---|---|---|
| type_pool, 40k iters | 3 images | 99.21% of pixels agree | ~0.2 points |
| segformer_5band, 100 iters | **64 images** | mean 96.73% (range 91.37–98.79%) | mean +0.89 pts (range −3.85 to +4.88); 39 of 64 move > 1 pt |

The direction is **not** systematic — it moves both ways depending on content,
and no padding mode avoids it; replicate and reflect simply bias differently.
The magnitude tracks how confident the model is, which is why a converged model
tolerates it and an unconverged one does not.

Over those same 64 real captures the conversion itself is confirmed exact:
ONNX against torch on identical input agrees **99.9999%** on average, worst case
99.9994%, at **19.8×** the speed (29.7 s → 1.50 s per image).

**Feeding dimensions already divisible by 32 avoids this entirely**, and is the
right fix for a deployment pipeline: change the resize target rather than pad.
`OnnxRunner` warns once per run when padding engages.

This is only exportable at all because the text table is a constant buffer. With
CLIP on the inference path the graph contained a transformer over learned
prompts run per image.

### Parallelism: a negative result

Running several eval processes over disjoint shards was expected to raise
throughput, on the reasoning that intra-op threading scales poorly here. It
does not. Every configuration is **slower** than a single process:

| configuration | wall (13 images) | combined peak RSS |
|---|---|---|
| **1 process × 4 threads** | **429 s** | 775 MB |
| 2 processes × 2 threads | 441 s | 2196 MB |
| 4 processes × 1 thread | 451 s | 4392 MB |
| 4 processes × 4 threads | 462 s | 3581 MB |

Thread oversubscription is not the explanation: giving each process a single
thread changed almost nothing. The workload is bound by memory bandwidth, and
more processes do not add bandwidth. The Pi 4's LPDDR4 saturates before its
four A72 cores do — the same ceiling reached from the other direction by the
2-to-4 thread scaling measured on this board (1.34×, not 2×).

**So on this hardware, throughput is fixed.** The only levers are doing less
work (the caching in `17af54b`) or feeding smaller inputs (resolution above).
Neither threads nor processes help.

This corrects a claim made earlier in this branch's development — that
process-level parallelism was the form that would scale here. It was asserted
before it was measured, and measurement refutes it. The memory work in
`08ba3b3` and `4cfa816` is still worthwhile: a smaller resident footprint
leaves room for the capture pipeline and other workloads to coexist on the
device, and it removes the OOM risk a 1221 MB peak carried on a 3.7 GiB board.
It just does not buy throughput.

## Limits

- **Not 5-band.** `mix_transformer.py:213` passes a literal `in_chans=3` and
  ignores its own parameter; the checkpoint's `patch_embed1.proj.weight` is
  `(32, 3, 7, 7)`; the config normalises with three channel means. The
  `dataset_5band` / `Load_5band_ImageFromFile` plumbing is vestigial and
  dead-ends at the backbone. Real 5-band support needs a widened patch
  embedding and retraining.
- **13 images, not 120.** `challenging_flood` is a 120-image subset; only 13
  were available on the Pi. Treat the mIoU as evidence the pipeline is
  numerically sound, not as a result to publish.
- **A deploy checkpoint cannot be trained or fine-tuned**, and its vocabulary
  is frozen: changing class names, the nine water types, or the prompt wording
  means recomputing the table, which needs the tower back.
- **Peak RSS is dominated by activations**, not weights, so it falls by much
  less than the 571 MB dropped. Resolution is the lever there, not the
  checkpoint.
- **Per-shard mIoU ranged from 84.04 to 91.73** across 3-4 image shards during
  the parallelism test, against 89.19 for all 13. A useful reminder of how
  little a 13-image sample constrains the figure.

## Reproducing

```bash
# one-off: strip the training checkpoint
PYTHONPATH=. python tools/export_deploy.py \
    --config configs/type_pool_b0_eval.py \
    --checkpoint weights/type_pool_b0_40k.pth \
    --out weights/type_pool_b0_40k.deploy.pth

# evaluate
PYTHONPATH=. python tools/eval.py \
    --config configs/type_pool_b0_eval.py \
    --checkpoint weights/type_pool_b0_40k.deploy.pth \
    --squeeze-rgb-gt --device cpu --deploy --sets challenging_flood
```

`ftfy` and `regex` must be installed — both are already pinned in
`requirements.txt` and are the vendored tokenizer's only dependencies.
`open_clip` is not required.
