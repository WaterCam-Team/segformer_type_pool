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
