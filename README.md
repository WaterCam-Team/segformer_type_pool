# Water / Flood Segmentation — SegFormer Type Pool

Binary water segmentation (`background`, `water`) on SegFormer-B0 (MiT-B0) with a
DenseCLIP-style text branch. Learned text prompts — one per water *appearance
type* — are matched against the pixel features and the resulting score map is
fed to the decode head.

**Evaluation only.** No training code, no schedules, no train pipeline. Everything
here is what is needed to load a checkpoint and score it on the labelled test sets.

```
configs/type_pool_b0_eval.py   eval config (model + the three test subsets)
tools/eval.py                  the evaluation script
weights/type_pool_b0_40k.pth   checkpoint -- NOT in this repo, download it, see section 1
weights/*.train_config.py      the config this checkpoint was trained with (reference)
mmseg/                         the project's mmsegmentation fork, copied as-is
requirements.txt
```

Python 3.9, versions pinned in `requirements.txt`. Needs a CUDA GPU (there is no
CPU path) and, on the first run, network access: building the model downloads
CLIP ViT-B/16 (~600 MB) through `open_clip`. Run everything from this directory
with `PYTHONPATH=.` — `mmseg/` is used in place, not installed.

## 1. Download the checkpoint

The checkpoint is 798 MB, too big for a git repo, so it is **not** in here.
Download it and drop it in `weights/`:

<https://drive.google.com/file/d/1YnucBRAtyPJLqOSPSOSF3hDWCT8m0W4o/view?usp=sharing>

**The file on Drive is named `segformer_setting2_3ctx_7cls_latest.pth`** — that
is the original training-run name. Rename it on the way in, because every command
below expects `type_pool_b0_40k.pth`:

```bash
mv segformer_setting2_3ctx_7cls_latest.pth weights/type_pool_b0_40k.pth
```

## 2. Point the config at your data

> **You have to edit these paths — nothing works until you do.** The DATA ROOTS
> block at the top of `configs/type_pool_b0_eval.py` holds absolute paths from
> the machine this was developed on, and they will not exist on yours. Point
> them at your own copies of the datasets. This is the only file you need to
> touch — and you only need to fix the roots for the subsets you actually intend
> to run (see `--sets` in section 3).

```python
# configs/type_pool_b0_eval.py, lines 7-9 -- replace with your own paths
data_root_normal_flood      = '/your/path/to/my_normal_flood_huantao_w_urban_flood/'
data_root_challenging_flood = '/your/path/to/my_challenging_flood_huantao_w_urban_flood_new/'
data_root_challenging_water = '/your/path/to/my_waterdataset_val_edge_setting2/'
```

Each root must contain an `img_dir/val` tree of `.jpg` images and an `ann_dir/val`
tree of `.png` masks with the same basenames — those suffixes are hardcoded in
`WaterDataset`, not config options. The datasets are not in this repo either; ask
me for them separately.

| subset | images | what it is |
|---|---|---|
| `normal_flood` | 1348 | normal flood scenes |
| `challenging_flood` | 120 | hard flood scenes |
| `challenging_water` | 50 | hard non-flood water (edge cases) |

## 3. Run

```bash
PYTHONPATH=. python tools/eval.py \
    --config configs/type_pool_b0_eval.py \
    --checkpoint weights/type_pool_b0_40k.pth \
    --squeeze-rgb-gt
```

Prints mIoU / mAcc / aAcc and per-class IoU per subset, then one summary table.
`--gpu-id N` picks the GPU.

### Evaluating only one dataset

Pass `--sets`:

```bash
PYTHONPATH=. python tools/eval.py \
    --config configs/type_pool_b0_eval.py \
    --checkpoint weights/type_pool_b0_40k.pth \
    --squeeze-rgb-gt \
    --sets normal_flood
```

The names are `normal_flood`, `challenging_flood` and `challenging_water` (the
`test_` prefix is optional). Pass more than one to run just those, in that order:
`--sets challenging_flood challenging_water`.

Subsets you do not name are never loaded, so **if you only have one of the three
datasets, only that one `data_root` has to be correct** — leave the other two
lines alone. A subset's numbers do not depend on which other subsets ran
alongside it, so a single-subset run is directly comparable to the full one. An
unrecognised name fails immediately, before the checkpoint is loaded, and lists
the valid ones.
