import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class flood_dataset_6cls(CustomDataset):
    """HRF dataset.

    In segmentation map annotation for WaterDataset, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    # CLASSES = ('plants', 'water','building','people','sky','vehicle')
    CLASSES = ('building','water','vehicle', 'people','plants','sky',)

    # PALETTE = [[102,255,102], [51,221,255],[250,50,83],[92,179,162], [61,61,245],[255,96,55]]
    # PALETTE = [[102,255,102], [255,221,51],[83,50,250],[162,179,92], [245,61,61],[55,96,255]]
    PALETTE = [[250,50,83],[51,221,255],[255,96,55],[92,179,162],[102,255,102], [61,61,245],]

    def __init__(self, **kwargs):
        super(flood_dataset_6cls, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
