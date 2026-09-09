import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        # if data['img_info']['filename'] == 'image_0006.jpg' or data['img_info']['filename'] == 'flood_0003.jpg'\
        #     or data['img_info']['filename'] == 'clear_water5.jpg':
        #     input()
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        # import torch
        # l = torch.unique(data['gt_semantic_seg'].data)
        # # if len(l)!=3 and not torch.eq(l ,torch.tensor([0,1])).all() :
        # if (12 in l) or(13 in l):
        #     print(data['img_metas'].data['filename'])
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
