import os.path as osp

from .builder import DATASETS
from .custom_5band import CustomDataset_5band


@DATASETS.register_module()
class dataset_5band(CustomDataset_5band):
    """HRF dataset.

    In segmentation map annotation for WaterDataset, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'water')

    PALETTE = [[120, 120, 120], [204, 0, 102]]

    def __init__(self, **kwargs):
        super(dataset_5band, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
